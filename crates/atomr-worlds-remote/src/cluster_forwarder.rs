//! Cross-node bridge for [`ClusterHost`].
//!
//! [`ShardRegion`] takes a `RemoteForwarder<M>` closure that ships
//! messages destined for shards owned by other regions to those regions.
//! This module wires that closure to `atomr-remote`: the closure wraps
//! `Envelope<WorldRequest>` in a [`WireRequest`] addressed to the
//! destination node's [`WorldGateway`](crate::WorldGateway), and a
//! local `ClusterReplyInbox` actor routes the resulting
//! [`WireReply::Reply`]s back into [`ClusterHost::pending_map`].
//!
//! Cross-node *subscription* routing is out of scope today —
//! [`ClusterHost::subscribe`] still passes through to the underlying
//! `LocalHost`, so subscribers only see events emitted on the node that
//! received the subscribe (`ClusterHost` rustdoc documents this).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use atomr_remote::RemoteSystem;
use atomr_worlds_host::{ClusterHost, HostError};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use tokio::sync::{oneshot, Mutex};

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
pub fn install_cluster_remote_forwarder(
    cluster: &ClusterHost,
    remote: Arc<RemoteSystem>,
    members: HashMap<String, String>,
) -> Result<String, HostError> {
    // Codecs are idempotent — atomr-remote keys by TypeId.
    remote.register_bincode::<WireRequest>();
    remote.register_bincode::<WireReply>();

    let pending = cluster.pending_map().clone();
    let sys = cluster.actor_system();
    let pending_for_actor = pending.clone();
    let inbox_ref = sys
        .actor_of(
            Props::create(move || ClusterReplyInbox { pending: pending_for_actor.clone() }),
            CLUSTER_REPLY_INBOX_NAME,
        )
        .map_err(|e| HostError::Sys(format!("spawn cluster reply inbox: {e:?}")))?;
    remote.expose_actor(inbox_ref);

    let reply_path = format!("{}/user/{}", remote.local_address, CLUSTER_REPLY_INBOX_NAME);

    let members = Arc::new(members);
    let remote_for_fwd = remote.clone();
    let reply_path_for_fwd = reply_path.clone();
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
            target.tell(WireRequest {
                reply_path: reply_path_for_fwd.clone(),
                env,
            });
        },
    );
    cluster.region().set_remote_forwarder(forwarder);
    Ok(reply_path)
}

struct ClusterReplyInbox {
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Envelope<WorldEvent>>>>>,
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
            WireReply::Event { .. } => {
                // Subscription cross-node routing is deferred. The
                // local-passthrough path in ClusterHost::subscribe still
                // covers same-node subscribers.
                tracing::debug!("cluster inbox: dropped streaming event (not yet supported)");
            }
        }
    }
}
