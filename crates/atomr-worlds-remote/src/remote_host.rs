//! Client-side [`WorldHost`] that speaks bincode over `atomr-remote`.

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, ActorSystem, Context, Props};
use atomr_remote::{RemoteSettings, RemoteSystem};
use atomr_worlds_host::{HostError, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::wire::{subscribe_sub_id, WireReply, WireRequest, REPLY_INBOX_ACTOR_NAME};

#[derive(Clone, Debug)]
pub struct RemoteHostConfig {
    /// Full atomr actor path of the remote gateway, e.g.
    /// `atomr://server@127.0.0.1:7800/user/world-gateway`.
    pub server_path: String,
    /// Address this RemoteHost binds for its own inbound (replies + events).
    /// `127.0.0.1:0` lets the OS pick a port.
    pub bind: SocketAddr,
    /// Logical system name for the local `ActorSystem`. Distinct from the
    /// server's name. Defaults to `"atomr-worlds-client"`.
    pub system_name: String,
    /// Per-request timeout, mirrors `LocalHostConfig::request_timeout`.
    pub request_timeout: Duration,
    /// Bound for per-subscription mpsc channels.
    pub subscriber_capacity: usize,
}

impl Default for RemoteHostConfig {
    fn default() -> Self {
        Self {
            server_path: String::new(),
            bind: "127.0.0.1:0".parse().unwrap(),
            system_name: "atomr-worlds-client".into(),
            request_timeout: Duration::from_secs(10),
            subscriber_capacity: 256,
        }
    }
}

type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Envelope<WorldEvent>>>>>;
type SubsMap = Arc<Mutex<HashMap<u64, mpsc::Sender<Envelope<WorldEvent>>>>>;

pub struct RemoteHost {
    sys: ActorSystem,
    remote: Arc<RemoteSystem>,
    server_path: String,
    reply_path: String,
    pending: PendingMap,
    subs: SubsMap,
    request_timeout: Duration,
    subscriber_capacity: usize,
    /// Cached typed handle to the server gateway. `actor_selection` allocates
    /// a fresh serializer on each call, so we keep one around per RemoteHost.
    server_ref: ActorRef<WireRequest>,
}

impl fmt::Debug for RemoteHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteHost")
            .field("server_path", &self.server_path)
            .field("reply_path", &self.reply_path)
            .finish_non_exhaustive()
    }
}

impl RemoteHost {
    /// Build the RemoteHost: spins up a local `ActorSystem`, joins it to a
    /// `RemoteSystem` on `cfg.bind`, registers wire codecs, exposes a
    /// `ReplyInbox` actor, and resolves the server gateway.
    pub async fn new(cfg: RemoteHostConfig) -> Result<Self, HostError> {
        if cfg.server_path.is_empty() {
            return Err(HostError::Sys("server_path is required".into()));
        }
        let sys = ActorSystem::create(cfg.system_name.clone(), atomr_config::Config::reference())
            .await
            .map_err(|e| HostError::Sys(format!("{e}")))?;
        let remote = Arc::new(
            RemoteSystem::start(sys.clone(), cfg.bind, RemoteSettings::default())
                .await
                .map_err(|e| HostError::Sys(format!("{e}")))?,
        );
        remote.register_bincode::<WireRequest>();
        remote.register_bincode::<WireReply>();

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let subs: SubsMap = Arc::new(Mutex::new(HashMap::new()));

        let pending_for_actor = pending.clone();
        let subs_for_actor = subs.clone();
        let inbox_ref = sys
            .actor_of(
                Props::create(move || ReplyInbox {
                    pending: pending_for_actor.clone(),
                    subs: subs_for_actor.clone(),
                }),
                REPLY_INBOX_ACTOR_NAME,
            )
            .map_err(|e| HostError::Sys(format!("spawn reply inbox: {e:?}")))?;
        remote.expose_actor(inbox_ref);

        let reply_path = format!("{}/user/{}", remote.local_address, REPLY_INBOX_ACTOR_NAME);

        let server_ref = remote
            .actor_selection::<WireRequest>(&cfg.server_path)
            .ok_or_else(|| HostError::Sys(format!("bad server_path: {}", cfg.server_path)))?;

        Ok(Self {
            sys,
            remote,
            server_path: cfg.server_path,
            reply_path,
            pending,
            subs,
            request_timeout: cfg.request_timeout,
            subscriber_capacity: cfg.subscriber_capacity,
            server_ref,
        })
    }

    /// Local actor path (for tests / diagnostics).
    pub fn reply_path(&self) -> &str {
        &self.reply_path
    }
}

#[async_trait]
impl WorldHost for RemoteHost {
    async fn request(
        &self,
        env: Envelope<WorldRequest>,
    ) -> Result<Envelope<WorldEvent>, HostError> {
        let corr = env.corr_id;
        let (tx, rx) = oneshot::channel();
        {
            let mut guard = self.pending.lock().await;
            guard.insert(corr, tx);
        }
        self.server_ref
            .tell(WireRequest { reply_path: self.reply_path.clone(), env });
        match tokio::time::timeout(self.request_timeout, rx).await {
            Ok(Ok(env)) => Ok(env),
            Ok(Err(_)) => {
                self.pending.lock().await.remove(&corr);
                Err(HostError::Ask("reply channel dropped".into()))
            }
            Err(_) => {
                self.pending.lock().await.remove(&corr);
                Err(HostError::Ask("request timeout".into()))
            }
        }
    }

    async fn subscribe(
        &self,
        env: Envelope<WorldRequest>,
    ) -> Result<mpsc::Receiver<Envelope<WorldEvent>>, HostError> {
        let sub_id = subscribe_sub_id(&env.body)
            .ok_or_else(|| HostError::Sys("subscribe requires Subscribe/SubscribeMetric".into()))?;
        let (tx, rx) = mpsc::channel(self.subscriber_capacity);
        self.subs.lock().await.insert(sub_id, tx);
        self.server_ref
            .tell(WireRequest { reply_path: self.reply_path.clone(), env });
        Ok(rx)
    }

    async fn shutdown(&self) -> Result<(), HostError> {
        self.remote.shutdown().await;
        self.sys.clone().terminate().await;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReplyInbox actor — routes server-sent WireReplies to the right
// oneshot / mpsc on the client side.
// ─────────────────────────────────────────────────────────────────────────────

struct ReplyInbox {
    pending: PendingMap,
    subs: SubsMap,
}

#[async_trait]
impl Actor for ReplyInbox {
    type Msg = WireReply;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: WireReply) {
        match msg {
            WireReply::Reply { env } => {
                let corr = env.corr_id;
                let tx = self.pending.lock().await.remove(&corr);
                if let Some(tx) = tx {
                    let _ = tx.send(env);
                } else {
                    tracing::debug!(corr_id = corr, "reply inbox: no pending request");
                }
            }
            WireReply::Event { sub_id, env } => {
                let sender = self.subs.lock().await.get(&sub_id).cloned();
                if let Some(tx) = sender {
                    if tx.send(env).await.is_err() {
                        // Subscriber dropped — drop the route to stop
                        // accumulating dead state.
                        self.subs.lock().await.remove(&sub_id);
                    }
                } else {
                    tracing::debug!(sub_id, "reply inbox: no subscription route");
                }
            }
        }
    }
}
