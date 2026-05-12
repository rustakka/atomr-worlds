use atomr_worlds_core::addr::Address;
use serde::{Deserialize, Serialize};

/// Generic envelope carrying a correlation id, a source address, and a body.
///
/// `from` widened to [`Address`] in Phase 7 so vehicle actors and the cluster
/// reply-inbox can both serve as authentic envelope sources.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub corr_id: u64,
    pub from: Address,
    pub body: T,
}

impl<T> Envelope<T> {
    #[inline]
    pub fn new(corr_id: u64, from: Address, body: T) -> Self {
        Self { corr_id, from, body }
    }
}
