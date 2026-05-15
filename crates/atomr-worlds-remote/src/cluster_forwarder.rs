//! Cross-node bridge for [`ClusterHost`].
//!
//! [`ShardRegion`] takes a `RemoteForwarder<M>` closure that ships
//! messages destined for shards owned by other regions to those regions.
//! This module wires that closure to `atomr-remote`: the closure wraps
//! `Envelope<WorldRequest>` in a [`WireRequest`] addressed to the
//! destination node's [`WorldGateway`](crate::WorldGateway), and a
//! local `ClusterReplyInbox` actor routes the resulting
//! [`WireReply::Reply`]s back into [`ClusterHost::pending_map`] and
//! [`WireReply::Event`]s back into [`ClusterHost::subs_map`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use atomr_remote::RemoteSystem;
use atomr_worlds_host::{ClusterHost, ClusterSubs, HostError};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::wire::{WireReply, WireRequest};

type Forwarder = Arc<dyn Fn(&str, Envelope<WorldRequest>) + Send + Sync>;

/// Conventional name of the cluster reply inbox actor. Each node's
/// forwarder targets the *other* nodes' [`crate::WorldGateway`]; replies
/// land at this name on the requester node.
pub const CLUSTER_REPLY_INBOX_NAME: &str = "world-cluster-reply-inbox";

/// Install an atomr-remote-backed cross-node forwarder on `cluster`.
///
/// `members` maps `region_id → gateway actor path` for every peer node
/// (excluding `self`). Returns the path of this node's reply inbox so
/// callers can publish it elsewhere if needed.
///
/// Calls [`install_cluster_remote_forwarder_with_auth`] with no token —
/// peers must run with `expected_auth_token = None` on their gateways.
pub fn install_cluster_remote_forwarder(
    cluster: &ClusterHost,
    remote: Arc<RemoteSystem>,
    members: HashMap<String, String>,
) -> Result<String, HostError> {
    install_cluster_remote_forwarder_with_auth(cluster, remote, members, None)
}

/// Same as [`install_cluster_remote_forwarder`] plus an outbound bearer
/// token attached to every forwarded `WireRequest`. Pair with
/// `WorldGateway::with_auth_token(token)` on every peer; mismatched
/// tokens are silently dropped on the receiver side.
pub fn install_cluster_remote_forwarder_with_auth(
    cluster: &ClusterHost,
    remote: Arc<RemoteSystem>,
    members: HashMap<String, String>,
    auth_token: Option<String>,
) -> Result<String, HostError> {
    // Codecs are idempotent — atomr-remote keys by TypeId.
    remote.register_bincode::<WireRequest>();
    remote.register_bincode::<WireReply>();

    let pending = cluster.pending_map().clone();
    let subs = cluster.subs_map().clone();
    let sys = cluster.actor_system();
    let pending_for_actor = pending.clone();
    let subs_for_actor = subs.clone();
    let inbox_ref = sys
        .actor_of(
            Props::create(move || ClusterReplyInbox {
                pending: pending_for_actor.clone(),
                subs: subs_for_actor.clone(),
            }),
            CLUSTER_REPLY_INBOX_NAME,
        )
        .map_err(|e| HostError::Sys(format!("spawn cluster reply inbox: {e:?}")))?;
    remote.expose_actor(inbox_ref);

    let reply_path = format!("{}/user/{}", remote.local_address, CLUSTER_REPLY_INBOX_NAME);

    let members = Arc::new(members);
    let remote_for_fwd = remote.clone();
    let reply_path_for_fwd = reply_path.clone();
    let auth_token = Arc::new(auth_token);
    let forwarder: Forwarder = Arc::new(
        move |owner: &str, env: Envelope<WorldRequest>| {
            let Some(target_path) = members.get(owner) else {
                tracing::warn!(owner = %owner, "cluster forwarder: no member entry");
                return;
            };
            let Some(target) = remote_for_fwd.actor_selection::<WireRequest>(target_path) else {
                tracing::warn!(target = %target_path, "cluster forwarder: bad actor_selection");
                return;
            };
            let mut wire = WireRequest::new(reply_path_for_fwd.clone(), env);
            if let Some(tok) = auth_token.as_ref().as_deref() {
                wire.auth_token = Some(tok.to_string());
            }
            target.tell(wire);
        },
    );
    cluster.region().set_remote_forwarder(forwarder);
    Ok(reply_path)
}

struct ClusterReplyInbox {
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Envelope<WorldEvent>>>>>,
    subs: ClusterSubs,
}

#[async_trait]
impl Actor for ClusterReplyInbox {
    type Msg = WireReply;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: WireReply) {
        match msg {
            WireReply::Reply { env } => {
                let corr = env.corr_id;
                let tx = self.pending.lock().await.remove(&corr);
                if let Some(tx) = tx {
                    let _ = tx.send(env);
                } else {
                    tracing::debug!(corr_id = corr, "cluster inbox: no pending entry");
                }
            }
            WireReply::Event { sub_id, env } => {
                // Phase 15 follow-up: cross-node subscription routing.
                // ClusterHost::subscribe registered an mpsc sender for
                // this sub_id when it forwarded the Subscribe envelope
                // to the owning peer. The peer's gateway streams events
                // back as WireReply::Event keyed by that sub_id, and we
                // forward them through the registered sender.
                let sender: Option<mpsc::Sender<Envelope<WorldEvent>>> =
                    self.subs.lock().await.get(&sub_id).cloned();
                if let Some(tx) = sender {
                    if tx.send(env).await.is_err() {
                        // Receiver dropped — reap the route so we stop
                        // accumulating dead state.
                        self.subs.lock().await.remove(&sub_id);
                    }
                } else {
                    tracing::debug!(sub_id, "cluster inbox: no subscription route for event");
                }
            }
        }
    }
}
