//! World-actor persistence surface.
//!
//! **Phase 3 scaffold.** Defines the journal trait and an in-memory backend.
//! The trait shape matches `atomr_persistence::Journal` so the next phase can
//! bind to it mechanically without disturbing the host crate.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use std::collections::HashMap;

use async_trait::async_trait;
use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::Voxel;
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub struct VoxelWriteEvent {
    pub addr: WorldAddr,
    pub pos: IVec3,
    pub before: Voxel,
    pub after: Voxel,
    pub seq: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    #[error("journal is closed")]
    Closed,
}

#[async_trait]
pub trait WorldJournal: Send + Sync {
    async fn append(&self, ev: VoxelWriteEvent) -> Result<(), JournalError>;
    async fn replay(&self, addr: &WorldAddr) -> Result<Vec<VoxelWriteEvent>, JournalError>;
}

#[derive(Default, Debug)]
pub struct InMemoryJournal {
    by_world: Mutex<HashMap<WorldAddr, Vec<VoxelWriteEvent>>>,
}

impl InMemoryJournal {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl WorldJournal for InMemoryJournal {
    async fn append(&self, ev: VoxelWriteEvent) -> Result<(), JournalError> {
        let mut map = self.by_world.lock().await;
        map.entry(ev.addr).or_default().push(ev);
        Ok(())
    }
    async fn replay(&self, addr: &WorldAddr) -> Result<Vec<VoxelWriteEvent>, JournalError> {
        let map = self.by_world.lock().await;
        Ok(map.get(addr).cloned().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::addr::WorldAddr;

    #[tokio::test]
    async fn round_trip() {
        let j = InMemoryJournal::new();
        let addr = WorldAddr::ROOT;
        j.append(VoxelWriteEvent {
            addr,
            pos: IVec3::new(1, 2, 3),
            before: Voxel::EMPTY,
            after: Voxel::new(7),
            seq: 1,
        })
        .await
        .unwrap();
        let events = j.replay(&addr).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].pos, IVec3::new(1, 2, 3));
    }
}
