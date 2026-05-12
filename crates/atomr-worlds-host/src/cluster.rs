//! Cluster host — routes envelopes through atomr-cluster-sharding's
//! [`ShardRegion`] keyed by [`crate::WorldExtractor`].
//!
//! Phase 10 wires up the real `ShardRegion`-backed routing in single-node
//! mode: the host owns the region, the extractor, and a per-entity handler
//! that constructs an in-process [`LocalHost`] under the hood. Cross-node
//! remote forwarding requires bridging `atomr_remote`'s codec to the
//! `RemoteForwarder` hook; that wiring is left as a documented TODO since
//! it depends on upstream API stability that isn't pinned for this drop.
//!
//! ## Replies
//!
//! `ShardRegion::deliver` is fire-and-forget. The reply path goes through a
//! per-corr-id [`oneshot::Sender`] registry on the host. The entity handler
//! drains that registry by looking up the request's `corr_id` after handling
//! it locally.
//!
//! ## Subscriptions
//!
//! `ClusterHost::subscribe` runs the entity handler synchronously to obtain
//! the initial brick stream and then continues to receive deltas in-band via
//! the same channel that the local actor uses.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr_cluster_sharding::{ShardCoordinator, ShardRegion};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::error::HostError;
use crate::host::WorldHost;
use crate::local::{LocalHost, LocalHostConfig};
use crate::extractor::WorldExtractor;

/// Per-corr-id reply registry. Exposed so `atomr-worlds-remote` can
/// install a cross-node forwarder that feeds replies back through it.
pub type PendingReplies = Arc<Mutex<HashMap<u64, oneshot::Sender<Envelope<WorldEvent>>>>>;

/// Configuration for a cluster host. Caller pre-builds the [`ShardRegion`]
/// and (in the cross-node case) installs a remote forwarder via
/// [`ShardRegion::set_remote_forwarder`].
#[derive(Clone)]
pub struct ClusterHostConfig {
    pub region_id: String,
    pub coordinator: Arc<ShardCoordinator>,
    pub local_config: LocalHostConfig,
    pub request_timeout: Duration,
}

impl std::fmt::Debug for ClusterHostConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterHostConfig")
            .field("region_id", &self.region_id)
            .field("request_timeout", &self.request_timeout)
            .finish_non_exhaustive()
    }
}

pub struct ClusterHost {
    config: ClusterHostConfig,
    local: Arc<LocalHost>,
    region: Arc<ShardRegion<WorldExtractor>>,
    pending: PendingReplies,
}

impl std::fmt::Debug for ClusterHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterHost")
            .field("region_id", &self.region.region_id())
            .field("shard_count", &self.region.shard_count())
            .finish()
    }
}

impl ClusterHost {
    pub async fn new(config: ClusterHostConfig) -> Result<Self, HostError> {
        let local = Arc::new(LocalHost::new(config.local_config.clone()).await?);
        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Envelope<WorldEvent>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let local_for_handler = local.clone();
        let pending_for_handler = pending.clone();
        let handler_factory = Arc::new(move || {
            let local = local_for_handler.clone();
            let pending = pending_for_handler.clone();
            Box::new(move |_entity_id: &str, env: Envelope<WorldRequest>| {
                // Run the entity handler in a detached tokio task so the
                // sharding deliver path stays sync.
                let local = local.clone();
                let pending = pending.clone();
                tokio::spawn(async move {
                    let corr_id = env.corr_id;
                    let result = local.request(env).await;
                    let reply = pending.lock().await.remove(&corr_id);
                    if let Some(tx) = reply {
                        if let Ok(envelope) = result {
                            let _ = tx.send(envelope);
                        }
                    }
                });
            }) as Box<dyn Fn(&str, Envelope<WorldRequest>) + Send + Sync + 'static>
        });

        let region = ShardRegion::new(
            config.region_id.clone(),
            Arc::new(WorldExtractor),
            config.coordinator.clone(),
            handler_factory,
        );

        Ok(Self { config, local, region, pending })
    }

    /// Access the underlying [`ShardRegion`] for cross-node wiring (e.g. to
    /// call [`ShardRegion::set_remote_forwarder`]).
    pub fn region(&self) -> &Arc<ShardRegion<WorldExtractor>> {
        &self.region
    }

    /// Access the per-corr-id pending reply registry. External wiring
    /// (e.g. an `atomr-worlds-remote` cluster forwarder) feeds replies
    /// from cross-node forwarded requests directly into this map so
    /// [`Self::request`] can unblock.
    pub fn pending_map(&self) -> &PendingReplies {
        &self.pending
    }

    /// In-process actor system the host runs on. Useful when external
    /// wiring needs to spawn auxiliary actors (e.g. a reply inbox) on
    /// the same system as the local entity actors.
    pub fn actor_system(&self) -> &atomr::prelude::ActorSystem {
        self.local.actor_system()
    }
}

#[async_trait]
impl WorldHost for ClusterHost {
    async fn request(
        &self,
        envelope: Envelope<WorldRequest>,
    ) -> Result<Envelope<WorldEvent>, HostError> {
        let corr_id = envelope.corr_id;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(corr_id, tx);
        self.region.deliver(envelope);
        match tokio::time::timeout(self.config.request_timeout, rx).await {
            Ok(Ok(env)) => Ok(env),
            Ok(Err(_)) => Err(HostError::Ask("reply channel dropped".into())),
            Err(_) => {
                self.pending.lock().await.remove(&corr_id);
                Err(HostError::Ask("request timeout".into()))
            }
        }
    }

    async fn subscribe(
        &self,
        envelope: Envelope<WorldRequest>,
    ) -> Result<mpsc::Receiver<Envelope<WorldEvent>>, HostError> {
        // Subscriptions go directly through the underlying LocalHost — the
        // initial snapshot and subsequent deltas stream through the mpsc the
        // local host returns. When cross-node sharding lands, the bridging
        // actor described in PHASES.md replaces this direct dispatch.
        self.local.subscribe(envelope).await
    }

    async fn shutdown(&self) -> Result<(), HostError> {
        self.local.shutdown().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::addr::{Address, WorldAddr};
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_proto::WorldRequest;
    use atomr_worlds_voxel::Voxel;

    #[tokio::test]
    async fn cluster_host_routes_request_through_sharding() {
        let host = ClusterHost::new(ClusterHostConfig {
            region_id: "region-a".into(),
            coordinator: Arc::new(ShardCoordinator::new()),
            local_config: LocalHostConfig {
                root_seed: 0xABCD,
                ..LocalHostConfig::default()
            },
            request_timeout: Duration::from_secs(5),
        })
        .await
        .unwrap();

        let addr = Address::World(WorldAddr::ROOT);
        // Write through the cluster.
        let w = Envelope::new(
            1,
            addr,
            WorldRequest::WriteVoxel { addr, pos: IVec3::new(0, 0, 0), voxel: Voxel::new(5) },
        );
        let _ = host.request(w).await.unwrap();
        // Read back through the cluster.
        let r = Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos: IVec3::new(0, 0, 0) });
        let resp = host.request(r).await.unwrap();
        let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
        assert_eq!(voxel, Voxel::new(5));
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn cluster_host_subscribe_passthrough() {
        let host = ClusterHost::new(ClusterHostConfig {
            region_id: "region-b".into(),
            coordinator: Arc::new(ShardCoordinator::new()),
            local_config: LocalHostConfig::default(),
            request_timeout: Duration::from_secs(5),
        })
        .await
        .unwrap();

        let addr = Address::World(WorldAddr::ROOT);
        let env = Envelope::new(
            0,
            addr,
            WorldRequest::Subscribe {
                addr,
                region: atomr_worlds_proto::AABB::new(IVec3::new(0, 0, 0), IVec3::new(16, 16, 16)),
                lod: atomr_worlds_core::Lod::new(0),
                sub_id: 7,
            },
        );
        let mut rx = host.subscribe(env).await.unwrap();
        let snap = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await.unwrap().unwrap();
        assert!(matches!(snap.body, WorldEvent::BrickSnapshot { .. }));
        host.shutdown().await.unwrap();
    }
}
