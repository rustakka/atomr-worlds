//! World-actor persistence binding.
//!
//! Wraps `atomr_persistence::{Journal, SnapshotStore}` with world-specific
//! encoding: voxel-write events go on the journal, periodic snapshots capture
//! the per-world write overlay. Re-exports the in-memory and (optionally) SQL
//! backends so the host crate can consume them without an extra dep.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

#[cfg(feature = "derived")]
pub mod derived;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use atomr_persistence::{Journal, JournalError, PersistentRepr, SnapshotMetadata, SnapshotStore};
use atomr_worlds_core::addr::{Address, Level, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::vehicle::{ParentAddr, VehicleAddr};
use atomr_worlds_voxel::Voxel;
use serde::{Deserialize, Serialize};

pub use atomr_persistence::{InMemoryJournal, InMemorySnapshotStore};

#[cfg(feature = "sql")]
pub use atomr_persistence_sql::{SqlConfig, SqlDialect, SqlJournal, SqlSnapshotStore};

const EVENT_MANIFEST: &str = "atomr-worlds.voxel-write.v1";
const SNAPSHOT_MANIFEST: &str = "atomr-worlds.snapshot.v1";

/// A single voxel-write event journalled by a `WorldActor`. Address is the
/// canonical [`Address`] so vehicle voxel spaces journal through the same
/// pipeline as worlds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelWriteEvent {
    pub addr: Address,
    pub pos: IVec3,
    pub before: Voxel,
    pub after: Voxel,
}

/// State captured at snapshot time. Only writes that diverge from procedural
/// generation are persisted — the brick cache itself is regenerable from seed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub writes: HashMap<IVec3, Voxel>,
}

#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error(transparent)]
    Journal(#[from] JournalError),
    #[error("encode error: {0}")]
    Encode(String),
    #[error("decode error: {0}")]
    Decode(String),
}

fn world_addr_key(addr: WorldAddr) -> String {
    let u = addr.universe;
    let g = addr.galaxy;
    let s = addr.sector;
    let sy = addr.system;
    let w = addr.world;
    format!(
        "u:{},{},{},{}|g:{},{},{},{}|s:{},{},{},{}|sy:{},{},{},{}|w:{},{},{},{}",
        u.coord.x,
        u.coord.y,
        u.coord.z,
        u.dim,
        g.coord.x,
        g.coord.y,
        g.coord.z,
        g.dim,
        s.coord.x,
        s.coord.y,
        s.coord.z,
        s.dim,
        sy.coord.x,
        sy.coord.y,
        sy.coord.z,
        sy.dim,
        w.coord.x,
        w.coord.y,
        w.coord.z,
        w.dim,
    )
}

fn parent_addr_key(p: ParentAddr) -> String {
    match p {
        ParentAddr::World(a) => {
            format!("pW|{}|lvl:{:?}", world_addr_key(a.ancestor(Level::World)), Level::World)
        }
        ParentAddr::System(a) => {
            format!("pS|{}|lvl:{:?}", world_addr_key(a.ancestor(Level::System)), Level::System)
        }
        ParentAddr::Sector(a) => {
            format!("pK|{}|lvl:{:?}", world_addr_key(a.ancestor(Level::Sector)), Level::Sector)
        }
    }
}

fn vehicle_addr_key(v: VehicleAddr) -> String {
    format!("{}|slot:{}|dim:{}", parent_addr_key(v.parent), v.slot.slot_id, v.slot.dim)
}

/// Stable string key identifying an address for journal/snapshot lookups.
///
/// Format: a single-letter discriminator (`W` for `Address::World`, `V` for
/// `Address::Vehicle`) followed by an address-specific path. Existing world
/// snapshots predating this change will be re-keyed on first write — no
/// migration is needed for in-memory backends; SQL backends would need a
/// one-time rekey (out of scope).
pub fn persistence_id_for(addr: Address) -> String {
    match addr {
        Address::World(a) => format!("W|{}", world_addr_key(a)),
        Address::Vehicle(v) => format!("V|{}", vehicle_addr_key(v)),
    }
}

/// Combined journal + snapshot binding with a configurable snapshot policy.
#[derive(Clone)]
pub struct WorldPersistence {
    journal: Arc<dyn Journal>,
    snapshots: Arc<dyn SnapshotStore>,
    writer_uuid: String,
    snapshot_every: u64,
}

impl fmt::Debug for WorldPersistence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldPersistence")
            .field("writer_uuid", &self.writer_uuid)
            .field("snapshot_every", &self.snapshot_every)
            .finish_non_exhaustive()
    }
}

/// State returned from `WorldPersistence::recover`.
#[derive(Clone, Debug, Default)]
pub struct RecoveredState {
    pub writes: HashMap<IVec3, Voxel>,
    /// Last sequence number observed in the journal. Next write is `last_seq + 1`.
    pub last_seq: u64,
}

impl WorldPersistence {
    /// In-memory backends, default snapshot policy (every 64 writes).
    pub fn in_memory() -> Self {
        let journal = InMemoryJournal::new();
        let snapshots = InMemorySnapshotStore::new();
        Self::new(journal as Arc<dyn Journal>, snapshots as Arc<dyn SnapshotStore>)
    }

    pub fn new(journal: Arc<dyn Journal>, snapshots: Arc<dyn SnapshotStore>) -> Self {
        let writer_uuid = format!("atomr-worlds-{}", std::process::id());
        Self { journal, snapshots, writer_uuid, snapshot_every: 64 }
    }

    pub fn with_writer_uuid(mut self, uuid: impl Into<String>) -> Self {
        self.writer_uuid = uuid.into();
        self
    }

    /// Set the snapshot policy. `0` disables periodic snapshots; the caller
    /// can still trigger one manually via `save_snapshot`.
    pub fn with_snapshot_every(mut self, n: u64) -> Self {
        self.snapshot_every = n;
        self
    }

    pub fn snapshot_every(&self) -> u64 {
        self.snapshot_every
    }

    pub fn writer_uuid(&self) -> &str {
        &self.writer_uuid
    }

    pub fn journal(&self) -> Arc<dyn Journal> {
        self.journal.clone()
    }

    pub fn snapshot_store(&self) -> Arc<dyn SnapshotStore> {
        self.snapshots.clone()
    }

    /// Replay the journal (after loading any snapshot) and return the
    /// reconstructed write overlay plus the last sequence number observed.
    pub async fn recover(&self, addr: Address) -> Result<RecoveredState, PersistError> {
        let pid = persistence_id_for(addr);
        let mut state = RecoveredState::default();
        let mut start_from = 1u64;
        if let Some((meta, payload)) = self.snapshots.load(&pid).await {
            let snap = decode_snapshot(&payload)?;
            state.writes = snap.writes;
            state.last_seq = meta.sequence_nr;
            start_from = meta.sequence_nr + 1;
        }
        let highest = self.journal.highest_sequence_nr(&pid, 0).await?;
        if highest >= start_from {
            let events = self.journal.replay_messages(&pid, start_from, highest, u64::MAX).await?;
            for repr in events {
                let ev = decode_event(&repr.payload)?;
                apply_event_to_overlay(&mut state.writes, &ev);
                state.last_seq = repr.sequence_nr;
            }
        }
        Ok(state)
    }

    /// Append a write event at the given sequence number.
    pub async fn append(
        &self,
        addr: Address,
        ev: &VoxelWriteEvent,
        sequence_nr: u64,
    ) -> Result<(), PersistError> {
        let pid = persistence_id_for(addr);
        let payload = encode_event(ev)?;
        let repr = PersistentRepr {
            persistence_id: pid,
            sequence_nr,
            payload,
            manifest: EVENT_MANIFEST.to_string(),
            writer_uuid: self.writer_uuid.clone(),
            deleted: false,
            tags: Vec::new(),
        };
        self.journal.write_messages(vec![repr]).await?;
        Ok(())
    }

    /// Save a snapshot at `sequence_nr`. Callers typically pair this with
    /// `delete_messages_to(sequence_nr)` for log truncation, but we keep
    /// truncation manual to leave audit trails intact by default.
    pub async fn save_snapshot(
        &self,
        addr: Address,
        snap: &WorldSnapshot,
        sequence_nr: u64,
    ) -> Result<(), PersistError> {
        let pid = persistence_id_for(addr);
        let payload = encode_snapshot(snap)?;
        let _ = SNAPSHOT_MANIFEST; // version tag reserved on the payload
        let meta = SnapshotMetadata { persistence_id: pid, sequence_nr, timestamp: now_millis() };
        self.snapshots.save(meta, payload).await;
        Ok(())
    }
}

fn apply_event_to_overlay(overlay: &mut HashMap<IVec3, Voxel>, ev: &VoxelWriteEvent) {
    if ev.after == Voxel::EMPTY {
        // A write that returns a cell to empty *only* reverts a previous user
        // write — the procedural baseline is independent. So drop the entry.
        overlay.remove(&ev.pos);
    } else {
        overlay.insert(ev.pos, ev.after);
    }
}

fn now_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn encode_event(ev: &VoxelWriteEvent) -> Result<Vec<u8>, PersistError> {
    bincode::serde::encode_to_vec(ev, bincode::config::standard())
        .map_err(|e| PersistError::Encode(e.to_string()))
}

fn decode_event(bytes: &[u8]) -> Result<VoxelWriteEvent, PersistError> {
    bincode::serde::decode_from_slice::<VoxelWriteEvent, _>(bytes, bincode::config::standard())
        .map(|(ev, _)| ev)
        .map_err(|e| PersistError::Decode(e.to_string()))
}

fn encode_snapshot(snap: &WorldSnapshot) -> Result<Vec<u8>, PersistError> {
    bincode::serde::encode_to_vec(snap, bincode::config::standard())
        .map_err(|e| PersistError::Encode(e.to_string()))
}

fn decode_snapshot(bytes: &[u8]) -> Result<WorldSnapshot, PersistError> {
    bincode::serde::decode_from_slice::<WorldSnapshot, _>(bytes, bincode::config::standard())
        .map(|(s, _)| s)
        .map_err(|e| PersistError::Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(pos: IVec3, before: u16, after: u16) -> VoxelWriteEvent {
        VoxelWriteEvent {
            addr: Address::World(WorldAddr::ROOT),
            pos,
            before: if before == 0 { Voxel::EMPTY } else { Voxel::new(before) },
            after: if after == 0 { Voxel::EMPTY } else { Voxel::new(after) },
        }
    }

    #[tokio::test]
    async fn recover_returns_empty_for_fresh_world() {
        let p = WorldPersistence::in_memory();
        let r = p.recover(Address::World(WorldAddr::ROOT)).await.unwrap();
        assert_eq!(r.last_seq, 0);
        assert!(r.writes.is_empty());
    }

    #[tokio::test]
    async fn write_then_recover_replays_overlay() {
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        p.append(addr, &ev(IVec3::new(1, 2, 3), 0, 7), 1).await.unwrap();
        p.append(addr, &ev(IVec3::new(4, 5, 6), 0, 9), 2).await.unwrap();
        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.last_seq, 2);
        assert_eq!(r.writes.get(&IVec3::new(1, 2, 3)), Some(&Voxel::new(7)));
        assert_eq!(r.writes.get(&IVec3::new(4, 5, 6)), Some(&Voxel::new(9)));
    }

    #[tokio::test]
    async fn snapshot_then_journal_tail_recovers() {
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        p.append(addr, &ev(IVec3::new(1, 0, 0), 0, 7), 1).await.unwrap();
        p.append(addr, &ev(IVec3::new(2, 0, 0), 0, 8), 2).await.unwrap();
        let mut snap = WorldSnapshot::default();
        snap.writes.insert(IVec3::new(1, 0, 0), Voxel::new(7));
        snap.writes.insert(IVec3::new(2, 0, 0), Voxel::new(8));
        p.save_snapshot(addr, &snap, 2).await.unwrap();
        // Tail: a third event, post-snapshot.
        p.append(addr, &ev(IVec3::new(3, 0, 0), 0, 9), 3).await.unwrap();

        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.last_seq, 3);
        assert_eq!(r.writes.get(&IVec3::new(1, 0, 0)), Some(&Voxel::new(7)));
        assert_eq!(r.writes.get(&IVec3::new(2, 0, 0)), Some(&Voxel::new(8)));
        assert_eq!(r.writes.get(&IVec3::new(3, 0, 0)), Some(&Voxel::new(9)));
    }

    #[tokio::test]
    async fn writing_empty_voxel_clears_overlay_entry() {
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        p.append(addr, &ev(IVec3::new(1, 0, 0), 0, 7), 1).await.unwrap();
        p.append(addr, &ev(IVec3::new(1, 0, 0), 7, 0), 2).await.unwrap();
        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.last_seq, 2);
        assert!(r.writes.is_empty());
    }

    #[test]
    fn persistence_id_is_stable_and_distinct() {
        let id_a = persistence_id_for(Address::World(WorldAddr::ROOT));
        let id_b = persistence_id_for(Address::World(WorldAddr::ROOT));
        assert_eq!(id_a, id_b);
        let mut other = WorldAddr::ROOT;
        other.world.coord = IVec3::new(1, 0, 0);
        assert_ne!(id_a, persistence_id_for(Address::World(other)));
    }

    #[test]
    fn persistence_id_world_and_vehicle_are_distinct() {
        use atomr_worlds_core::vehicle::{ParentAddr, VehicleAddr, VehicleSlot};
        let w = Address::World(WorldAddr::ROOT);
        let v =
            Address::Vehicle(VehicleAddr::new(ParentAddr::World(WorldAddr::ROOT), VehicleSlot::new(42, 0)));
        assert_ne!(persistence_id_for(w), persistence_id_for(v));
        // Discriminator prefixes.
        assert!(persistence_id_for(w).starts_with("W|"));
        assert!(persistence_id_for(v).starts_with("V|"));
    }
}
