//! In-process host suitable for single-player.
//!
//! Wraps an atomr `ActorSystem` and routes world requests to per-world actors
//! living in the same process. The actor wiring is filled in in a later phase.

use async_trait::async_trait;
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use tokio::sync::mpsc;

use crate::error::HostError;
use crate::host::WorldHost;

#[derive(Debug, Default)]
pub struct LocalHost {
    // Placeholder; will hold an `atomr_core::ActorSystem` handle in a later phase.
    _private: (),
}

impl LocalHost {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl WorldHost for LocalHost {
    async fn request(
        &self,
        _envelope: Envelope<WorldRequest>,
    ) -> Result<Envelope<WorldEvent>, HostError> {
        Err(HostError::NotYetImplemented("LocalHost::request"))
    }

    async fn subscribe(
        &self,
        _envelope: Envelope<WorldRequest>,
    ) -> Result<mpsc::Receiver<Envelope<WorldEvent>>, HostError> {
        Err(HostError::NotYetImplemented("LocalHost::subscribe"))
    }

    async fn shutdown(&self) -> Result<(), HostError> {
        Ok(())
    }
}
