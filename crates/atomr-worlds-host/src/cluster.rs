//! Cluster host: routes envelopes via atomr-cluster-sharding's `ShardRegion`.
//!
//! Same actor protocol as `LocalHost`; the only difference is that the
//! per-world actor may live on a different cluster node, and routing goes
//! through a `ShardRegion` keyed by [`crate::WorldExtractor`].

use async_trait::async_trait;
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use tokio::sync::mpsc;

use crate::error::HostError;
use crate::host::WorldHost;

#[derive(Debug, Default)]
pub struct ClusterHost {
    // Placeholder; will hold an `atomr_cluster_sharding::ShardRegion<WorldExtractor>` later.
    _private: (),
}

impl ClusterHost {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl WorldHost for ClusterHost {
    async fn request(
        &self,
        _envelope: Envelope<WorldRequest>,
    ) -> Result<Envelope<WorldEvent>, HostError> {
        Err(HostError::NotYetImplemented("ClusterHost::request"))
    }

    async fn subscribe(
        &self,
        _envelope: Envelope<WorldRequest>,
    ) -> Result<mpsc::Receiver<Envelope<WorldEvent>>, HostError> {
        Err(HostError::NotYetImplemented("ClusterHost::subscribe"))
    }

    async fn shutdown(&self) -> Result<(), HostError> {
        Ok(())
    }
}
