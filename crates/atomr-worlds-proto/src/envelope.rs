use atomr_worlds_core::addr::WorldAddr;
use serde::{Deserialize, Serialize};

/// Generic envelope carrying a correlation id, a source address, and a body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub corr_id: u64,
    pub from: WorldAddr,
    pub body: T,
}

impl<T> Envelope<T> {
    #[inline]
    pub fn new(corr_id: u64, from: WorldAddr, body: T) -> Self {
        Self { corr_id, from, body }
    }
}
