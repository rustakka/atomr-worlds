use async_trait::async_trait;
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use tokio::sync::mpsc;

use crate::error::HostError;

/// Hosting backend interface.
///
/// `request` returns a single response; `subscribe` returns a stream of
/// events until the subscription is dropped or [`WorldRequest::Unsubscribe`]
/// is sent.
#[async_trait]
pub trait WorldHost: Send + Sync + 'static {
    async fn request(
        &self,
        envelope: Envelope<WorldRequest>,
    ) -> Result<Envelope<WorldEvent>, HostError>;

    async fn subscribe(
        &self,
        envelope: Envelope<WorldRequest>,
    ) -> Result<mpsc::Receiver<Envelope<WorldEvent>>, HostError>;

    async fn shutdown(&self) -> Result<(), HostError>;
}
