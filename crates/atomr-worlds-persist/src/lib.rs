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
use atomr_worlds_core::lww::{LwwMap, LwwStamp, WriterId};
use atomr_worlds_core::vehicle::{ParentAddr, VehicleAddr};
use atomr_worlds_core::HlcTimestamp;
use atomr_worlds_voxel::Voxel;
use serde::{Deserialize, Serialize};

pub use atomr_persistence::{InMemoryJournal, InMemorySnapshotStore};

#[cfg(feature = "sql")]
pub use atomr_persistence_sql::{SqlConfig, SqlDialect, SqlJournal, SqlSnapshotStore};

/// Legacy (pre-Rec-4) journal event manifest — events without an HLC stamp.
/// Still decoded for backward compatibility; never written anymore.
const EVENT_MANIFEST_V1: &str = "atomr-worlds.voxel-write.v1";
/// Current journal event manifest — HLC-stamped LWW events. `append` writes
/// this; appending the stamp field is why the version bumped.
const EVENT_MANIFEST: &str = "atomr-worlds.voxel-write.v2";
const SNAPSHOT_MANIFEST: &str = "atomr-worlds.snapshot.v2";
/// Leading byte tagging a v2 (HLC-stamped) snapshot payload. A v1 payload has
/// no tag; decode falls back to the legacy shape when the tag is absent or the
/// tagged body doesn't decode cleanly.
const SNAPSHOT_VERSION_V2: u8 = 2;

/// A single voxel-write event journalled by a `WorldActor`. Address is the
/// canonical [`Address`] so vehicle voxel spaces journal through the same
/// pipeline as worlds. Carries the [`LwwStamp`] (HLC + writer) so concurrent /
/// out-of-order writes converge under last-writer-wins on replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelWriteEvent {
    pub addr: Address,
    pub pos: IVec3,
    pub before: Voxel,
    pub after: Voxel,
    pub stamp: LwwStamp,
}

/// Legacy v1 event shape (no stamp). Decode-only, for migrating old journals.
#[derive(Deserialize)]
struct VoxelWriteEventV1 {
    addr: Address,
    pos: IVec3,
    before: Voxel,
    after: Voxel,
}

/// One overlay cell at snapshot time: the LWW stamp plus its voxel value (an
/// `EMPTY` voxel is a retained tombstone, not an absence).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LwwCell {
    pub stamp: LwwStamp,
    pub voxel: Voxel,
}

/// State captured at snapshot time. Only writes that diverge from procedural
/// generation are persisted — the brick cache itself is regenerable from seed.
/// The full CRDT state (stamps + tombstones) is captured so convergence
/// survives log truncation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub writes: HashMap<IVec3, LwwCell>,
}

/// Legacy v1 snapshot shape (resolved voxels, no stamps). Decode-only.
#[derive(Deserialize, Default)]
struct WorldSnapshotV1 {
    writes: HashMap<IVec3, Voxel>,
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
    /// The recovered LWW overlay — the full CRDT state, including tombstones.
    /// `overlay.get(pos)` yields the resolved voxel (an `EMPTY` value is a
    /// retained carve, not an absence).
    pub overlay: LwwMap<IVec3, Voxel>,
    /// Last sequence number observed in the journal. Next write is `last_seq + 1`.
    pub last_seq: u64,
}

impl RecoveredState {
    /// The greatest HLC timestamp in the recovered overlay, so a respawned
    /// actor seeds its clock above all persisted history (never regresses).
    pub fn max_ts(&self) -> HlcTimestamp {
        self.overlay.iter().map(|(_, s, _)| s.ts).max().unwrap_or(HlcTimestamp::ZERO)
    }
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
            state.overlay = decode_snapshot(&payload, meta.sequence_nr)?;
            state.last_seq = meta.sequence_nr;
            start_from = meta.sequence_nr + 1;
        }
        let highest = self.journal.highest_sequence_nr(&pid, 0).await?;
        if highest >= start_from {
            let events = self.journal.replay_messages(&pid, start_from, highest, u64::MAX).await?;
            for repr in events {
                let ev = decode_event_versioned(&repr)?;
                apply_event_to_overlay(&mut state.overlay, &ev);
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

/// Apply a journalled event to the overlay under last-writer-wins. Tombstones
/// are retained (an `EMPTY` write is `put`, not removed) so an older
/// out-of-order write can't resurrect a carve and deletes survive recovery.
fn apply_event_to_overlay(overlay: &mut LwwMap<IVec3, Voxel>, ev: &VoxelWriteEvent) {
    overlay.put(ev.pos, ev.after, ev.stamp);
}

/// Synthesize an LWW stamp for a legacy (v1) entry from its journal sequence
/// number. Sits in a reserved below-all-real-time band: `wall_ns == 0` (so any
/// live write, which reads a positive clock, dominates it) and `counter == seq`
/// (so migrated entries stay mutually ordered — journal `seq` is dense and
/// strictly increasing per persistence-id). [`WriterId::LEGACY`] sorts below
/// every live writer.
fn legacy_stamp(sequence_nr: u64) -> LwwStamp {
    debug_assert!(
        sequence_nr <= u32::MAX as u64,
        "legacy journal exceeds u32::MAX events for one persistence-id"
    );
    LwwStamp::new(HlcTimestamp::new(0, sequence_nr as u32), WriterId::LEGACY)
}

fn now_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn encode_bincode<T: Serialize>(value: &T) -> Result<Vec<u8>, PersistError> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| PersistError::Encode(e.to_string()))
}

/// Decode a bincode value, requiring the whole buffer be consumed (so a
/// mis-tagged payload fails loudly rather than reading garbage).
fn decode_bincode_exact<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, PersistError> {
    let (val, read) =
        bincode::serde::decode_from_slice::<T, _>(bytes, bincode::config::standard())
            .map_err(|e| PersistError::Decode(e.to_string()))?;
    if read != bytes.len() {
        return Err(PersistError::Decode(format!(
            "trailing bytes: read {read} of {}",
            bytes.len()
        )));
    }
    Ok(val)
}

fn encode_event(ev: &VoxelWriteEvent) -> Result<Vec<u8>, PersistError> {
    encode_bincode(ev)
}

/// Decode a journalled event, dispatching on its manifest. v2 carries the LWW
/// stamp directly; v1 (legacy, stampless) gets a synthetic [`legacy_stamp`]
/// from its sequence number.
fn decode_event_versioned(repr: &PersistentRepr) -> Result<VoxelWriteEvent, PersistError> {
    match repr.manifest.as_str() {
        EVENT_MANIFEST => decode_bincode_exact::<VoxelWriteEvent>(&repr.payload),
        EVENT_MANIFEST_V1 => {
            let v1 = decode_bincode_exact::<VoxelWriteEventV1>(&repr.payload)?;
            Ok(VoxelWriteEvent {
                addr: v1.addr,
                pos: v1.pos,
                before: v1.before,
                after: v1.after,
                stamp: legacy_stamp(repr.sequence_nr),
            })
        }
        other => Err(PersistError::Decode(format!("unknown event manifest: {other}"))),
    }
}

/// Encode a v2 snapshot: a [`SNAPSHOT_VERSION_V2`] tag byte then the bincode
/// body.
fn encode_snapshot(snap: &WorldSnapshot) -> Result<Vec<u8>, PersistError> {
    let mut out = Vec::with_capacity(1 + snap.writes.len() * 24);
    out.push(SNAPSHOT_VERSION_V2);
    out.extend_from_slice(&encode_bincode(snap)?);
    Ok(out)
}

/// Decode a snapshot into an LWW overlay. A v2 payload is tag-prefixed and
/// carries stamps; an untagged (legacy) payload is the old resolved-voxel map,
/// whose cells get a [`legacy_stamp`] from the snapshot's sequence number.
fn decode_snapshot(
    bytes: &[u8],
    snapshot_seq: u64,
) -> Result<LwwMap<IVec3, Voxel>, PersistError> {
    // v2: tag byte + body that fully consumes. Guard against a v1 map whose
    // first byte happens to equal the tag by requiring a clean full decode.
    if let Some((&tag, rest)) = bytes.split_first() {
        if tag == SNAPSHOT_VERSION_V2 {
            if let Ok(snap) = decode_bincode_exact::<WorldSnapshot>(rest) {
                let entries = snap
                    .writes
                    .into_iter()
                    .map(|(pos, cell)| (pos, (cell.stamp, cell.voxel)))
                    .collect();
                return Ok(LwwMap::from_entries(entries));
            }
        }
    }
    // Legacy v1: resolved voxels, no stamps. All cells share the snapshot's
    // legacy stamp; any post-snapshot journal tail has a strictly greater one.
    let v1 = decode_bincode_exact::<WorldSnapshotV1>(bytes)?;
    let stamp = legacy_stamp(snapshot_seq);
    let entries = v1.writes.into_iter().map(|(pos, voxel)| (pos, (stamp, voxel))).collect();
    Ok(LwwMap::from_entries(entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vox(v: u16) -> Voxel {
        if v == 0 {
            Voxel::EMPTY
        } else {
            Voxel::new(v)
        }
    }

    /// Stamp a live write at HLC `wall_ns == seq` (writer 1) — mirrors a single
    /// sequential writer whose clock matches the journal order.
    fn stamp_for(seq: u64) -> LwwStamp {
        LwwStamp::new(HlcTimestamp::new(seq, 0), WriterId(1))
    }

    fn ev(pos: IVec3, before: u16, after: u16, seq: u64) -> VoxelWriteEvent {
        VoxelWriteEvent {
            addr: Address::World(WorldAddr::ROOT),
            pos,
            before: vox(before),
            after: vox(after),
            stamp: stamp_for(seq),
        }
    }

    fn cell(v: u16, seq: u64) -> LwwCell {
        LwwCell { stamp: stamp_for(seq), voxel: vox(v) }
    }

    #[tokio::test]
    async fn recover_returns_empty_for_fresh_world() {
        let p = WorldPersistence::in_memory();
        let r = p.recover(Address::World(WorldAddr::ROOT)).await.unwrap();
        assert_eq!(r.last_seq, 0);
        assert!(r.overlay.is_empty());
        assert_eq!(r.max_ts(), HlcTimestamp::ZERO);
    }

    #[tokio::test]
    async fn write_then_recover_replays_overlay() {
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        p.append(addr, &ev(IVec3::new(1, 2, 3), 0, 7, 1), 1).await.unwrap();
        p.append(addr, &ev(IVec3::new(4, 5, 6), 0, 9, 2), 2).await.unwrap();
        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.last_seq, 2);
        assert_eq!(r.overlay.get(&IVec3::new(1, 2, 3)), Some(&Voxel::new(7)));
        assert_eq!(r.overlay.get(&IVec3::new(4, 5, 6)), Some(&Voxel::new(9)));
        assert_eq!(r.max_ts(), HlcTimestamp::new(2, 0));
    }

    #[tokio::test]
    async fn snapshot_then_journal_tail_recovers() {
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        p.append(addr, &ev(IVec3::new(1, 0, 0), 0, 7, 1), 1).await.unwrap();
        p.append(addr, &ev(IVec3::new(2, 0, 0), 0, 8, 2), 2).await.unwrap();
        let mut snap = WorldSnapshot::default();
        snap.writes.insert(IVec3::new(1, 0, 0), cell(7, 1));
        snap.writes.insert(IVec3::new(2, 0, 0), cell(8, 2));
        p.save_snapshot(addr, &snap, 2).await.unwrap();
        // Tail: a third event, post-snapshot.
        p.append(addr, &ev(IVec3::new(3, 0, 0), 0, 9, 3), 3).await.unwrap();

        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.last_seq, 3);
        assert_eq!(r.overlay.get(&IVec3::new(1, 0, 0)), Some(&Voxel::new(7)));
        assert_eq!(r.overlay.get(&IVec3::new(2, 0, 0)), Some(&Voxel::new(8)));
        assert_eq!(r.overlay.get(&IVec3::new(3, 0, 0)), Some(&Voxel::new(9)));
    }

    #[tokio::test]
    async fn carve_survives_recovery_as_tombstone() {
        // Place a solid voxel, then carve it to empty. The carve must persist
        // as a retained tombstone (this is the bug the old remove-on-empty had:
        // procedural-solid terrain would reappear after recovery).
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        let pos = IVec3::new(1, 0, 0);
        p.append(addr, &ev(pos, 0, 7, 1), 1).await.unwrap();
        p.append(addr, &ev(pos, 7, 0, 2), 2).await.unwrap();
        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.last_seq, 2);
        assert_eq!(r.overlay.get(&pos), Some(&Voxel::EMPTY)); // tombstone, not absent
        assert_eq!(r.overlay.len(), 1);
    }

    #[test]
    fn out_of_order_event_replay_converges() {
        // Two writes to one cell applied in both physical orders converge,
        // because LWW resolves by stamp, not arrival.
        let pos = IVec3::new(1, 0, 0);
        let older = ev(pos, 0, 7, 5);
        let newer = ev(pos, 7, 9, 10);
        let mut a = LwwMap::new();
        apply_event_to_overlay(&mut a, &older);
        apply_event_to_overlay(&mut a, &newer);
        let mut b = LwwMap::new();
        apply_event_to_overlay(&mut b, &newer);
        apply_event_to_overlay(&mut b, &older);
        assert_eq!(a.get(&pos), Some(&Voxel::new(9)));
        assert_eq!(a, b);
    }

    #[test]
    fn v2_snapshot_round_trips_with_tombstone() {
        let mut snap = WorldSnapshot::default();
        snap.writes.insert(IVec3::new(1, 0, 0), cell(7, 1));
        snap.writes.insert(IVec3::new(2, 0, 0), cell(0, 2)); // tombstone
        let bytes = encode_snapshot(&snap).unwrap();
        assert_eq!(bytes.first(), Some(&SNAPSHOT_VERSION_V2));
        let overlay = decode_snapshot(&bytes, 2).unwrap();
        assert_eq!(overlay.get(&IVec3::new(1, 0, 0)), Some(&Voxel::new(7)));
        assert_eq!(overlay.get(&IVec3::new(2, 0, 0)), Some(&Voxel::EMPTY));
        assert_eq!(overlay.len(), 2);
    }

    #[tokio::test]
    async fn legacy_v1_journal_event_migrates() {
        // Write a raw v1-manifest event (no stamp) straight to the journal, then
        // recover: it must decode and get a legacy stamp from its sequence nr.
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        let pos = IVec3::new(9, 0, 0);
        // v1 struct bincode == tuple of its fields in order.
        let payload =
            encode_bincode(&(addr, pos, Voxel::EMPTY, Voxel::new(5))).unwrap();
        let repr = PersistentRepr {
            persistence_id: persistence_id_for(addr),
            sequence_nr: 1,
            payload,
            manifest: EVENT_MANIFEST_V1.to_string(),
            writer_uuid: "legacy".into(),
            deleted: false,
            tags: Vec::new(),
        };
        p.journal().write_messages(vec![repr]).await.unwrap();
        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.overlay.get(&pos), Some(&Voxel::new(5)));
        // Legacy stamp sits at wall_ns == 0 so any live write dominates it.
        assert_eq!(r.overlay.stamp(&pos), Some(legacy_stamp(1)));
        assert_eq!(legacy_stamp(1).ts.wall_ns, 0);
    }

    #[tokio::test]
    async fn legacy_v1_snapshot_migrates() {
        // Save a raw (untagged) v1 snapshot payload, then recover.
        let p = WorldPersistence::in_memory();
        let addr = Address::World(WorldAddr::ROOT);
        let pid = persistence_id_for(addr);
        let mut writes: HashMap<IVec3, Voxel> = HashMap::new();
        writes.insert(IVec3::new(1, 0, 0), Voxel::new(7));
        writes.insert(IVec3::new(2, 0, 0), Voxel::new(8));
        let payload = encode_bincode(&writes).unwrap(); // == WorldSnapshotV1 bincode
        let meta = SnapshotMetadata { persistence_id: pid.clone(), sequence_nr: 5, timestamp: 0 };
        p.snapshot_store().save(meta, payload).await;
        let r = p.recover(addr).await.unwrap();
        assert_eq!(r.last_seq, 5);
        assert_eq!(r.overlay.get(&IVec3::new(1, 0, 0)), Some(&Voxel::new(7)));
        assert_eq!(r.overlay.get(&IVec3::new(2, 0, 0)), Some(&Voxel::new(8)));
        assert_eq!(r.overlay.stamp(&IVec3::new(1, 0, 0)), Some(legacy_stamp(5)));
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
