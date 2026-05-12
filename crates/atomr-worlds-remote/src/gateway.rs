//! Server-side gateway actor. Listens for [`WireRequest`]s from remote
//! clients and dispatches them to a local [`WorldHost`].

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use atomr_remote::RemoteSystem;
use atomr_worlds_host::WorldHost;

use crate::wire::{subscribe_sub_id, WireReply, WireRequest};

/// Construction-time configuration for a [`WorldGateway`].
#[derive(Clone)]
pub struct WorldGatewayConfig {
    pub bind: std::net::SocketAddr,
    pub system_name: String,
}

impl fmt::Debug for WorldGatewayConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldGatewayConfig")
            .field("bind", &self.bind)
            .field("system_name", &self.system_name)
            .finish()
    }
}

/// Server-side actor. One per process; wraps an `Arc<dyn WorldHost>`.
///
/// On a non-subscription request, forwards to `host.request()` and sends a
/// single [`WireReply::Reply`] back to the requester's `reply_path`.
/// On a subscription, opens a stream on `host.subscribe()` and forwards
/// each event as [`WireReply::Event`] until the client disconnects (the
/// remote `tell` becomes a no-op when the endpoint dies, which lets the
/// forwarder task exit on the next event).
pub struct WorldGateway {
    pub(crate) host: Arc<dyn WorldHost>,
    pub(crate) remote: Arc<RemoteSystem>,
}

impl fmt::Debug for WorldGateway {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldGateway").finish_non_exhaustive()
    }
}

impl WorldGateway {
    pub fn new(host: Arc<dyn WorldHost>, remote: Arc<RemoteSystem>) -> Self {
        Self { host, remote }
    }
}

#[async_trait]
impl Actor for WorldGateway {
    type Msg = WireRequest;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: WireRequest) {
        let host = self.host.clone();
        let remote = self.remote.clone();
        let reply_path = msg.reply_path;
        let env = msg.env;

        if let Some(sub_id) = subscribe_sub_id(&env.body) {
            // Subscription: open stream + spawn forwarder.
            tokio::spawn(async move {
                let mut rx = match host.subscribe(env).await {
                    Ok(rx) => rx,
                    Err(e) => {
                        tracing::warn!(error = %e, "gateway: subscribe failed");
                        return;
                    }
                };
                let reply_ref = match remote.actor_selection::<WireReply>(&reply_path) {
                    Some(r) => r,
                    None => {
                        tracing::warn!(reply_path = %reply_path, "gateway: bad reply_path");
                        return;
                    }
                };
                while let Some(event_env) = rx.recv().await {
                    reply_ref.tell(WireReply::Event { sub_id, env: event_env });
                }
                tracing::debug!(sub_id, "gateway: subscription drained");
            });
        } else {
            // One-shot request.
            tokio::spawn(async move {
                match host.request(env).await {
                    Ok(reply_env) => {
                        if let Some(reply_ref) = remote.actor_selection::<WireReply>(&reply_path) {
                            reply_ref.tell(WireReply::Reply { env: reply_env });
                        } else {
                            tracing::warn!(reply_path = %reply_path, "gateway: bad reply_path");
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "gateway: request failed");
                    }
                }
            });
        }
    }
}
