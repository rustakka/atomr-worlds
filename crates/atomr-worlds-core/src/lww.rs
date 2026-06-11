//! A last-writer-wins (LWW) CRDT map keyed by [`HlcTimestamp`].
//!
//! This is the conflict-resolution substrate for **Rec 4 (multiplayer
//! destruction sync)**: the world actor's voxel-edit overlay is an
//! [`LwwMap<IVec3, Voxel>`] so concurrent/out-of-order writes from different
//! players converge — the write with the greater [`LwwStamp`] wins, ties broken
//! by writer id. The sibling `atomr-distributed-data` ships an `LWWMap`, but it
//! hard-codes a raw `u128` timestamp with no HLC and a sealed merge trait; this
//! type keeps the same merge *shape* while accepting our [`HlcTimestamp`] as the
//! comparison key.
//!
//! # Tombstones
//!
//! Deletes are **retained**, not dropped: an "empty" write is stored as a
//! `(value, stamp)` pair like any other. Dropping a deleted cell would discard
//! its timestamp, letting an older, out-of-order write resurrect it — the
//! classic CRDT tombstone hazard. Retaining it also makes deletes durable
//! across recovery/regeneration.
//!
//! # Determinism
//!
//! The resolved value at a key is the value of the entry with the maximum
//! `(ts, writer)` over all writes to that key — a pure function of the write
//! *set*, independent of arrival/iteration order. So replaying a fixed journal
//! converges to one state on every machine.

use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::hlc::HlcTimestamp;

/// Stable identity of a writer, used to break [`HlcTimestamp`] ties in the LWW
/// order. Any total order works; the value just has to be consistent across
/// nodes for the same logical writer.
#[derive(
    Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default, Debug, Serialize, Deserialize,
)]
pub struct WriterId(pub u64);

impl WriterId {
    /// Reserved id for writes synthesized during a legacy-journal migration.
    /// It sorts below every live writer, so a migrated entry never beats a
    /// real write at the same timestamp. Live writers MUST be non-zero.
    pub const LEGACY: Self = Self(0);
}

/// The LWW comparison key: an HLC timestamp, tie-broken by [`WriterId`].
///
/// `Ord` derives lexicographically from field order (`ts` then `writer`), which
/// is exactly the LWW rule: later timestamp wins; equal timestamps are decided
/// by the higher writer id.
#[derive(
    Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default, Debug, Serialize, Deserialize,
)]
pub struct LwwStamp {
    pub ts: HlcTimestamp,
    pub writer: WriterId,
}

impl LwwStamp {
    #[inline]
    pub const fn new(ts: HlcTimestamp, writer: WriterId) -> Self {
        Self { ts, writer }
    }
}

/// Outcome of an [`LwwMap::put`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PutOutcome<V> {
    /// The write won and is now the resolved value. `previous` is the value it
    /// displaced (if any).
    Applied { previous: Option<V> },
    /// The write lost to an existing entry with a greater-or-equal stamp. The
    /// current winner (and its stamp) is returned so the caller can reconcile
    /// — e.g. roll an optimistic preview back to `current`.
    Rejected { current: V, current_stamp: LwwStamp },
}

impl<V> PutOutcome<V> {
    #[inline]
    pub fn is_applied(&self) -> bool {
        matches!(self, PutOutcome::Applied { .. })
    }
}

/// A last-writer-wins CRDT map: per key, the entry with the maximum
/// [`LwwStamp`] wins. `merge`/`put` are commutative, associative, and
/// idempotent, so replicas converge regardless of delivery order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LwwMap<K: Eq + Hash, V> {
    entries: HashMap<K, (LwwStamp, V)>,
}

impl<K: Eq + Hash, V> Default for LwwMap<K, V> {
    fn default() -> Self {
        Self { entries: HashMap::new() }
    }
}

impl<K: Eq + Hash, V> PartialEq for LwwMap<K, V>
where
    V: PartialEq,
{
    /// Two maps are equal when they carry the same `(stamp, value)` per key.
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl<K: Eq + Hash + Clone, V: Clone> LwwMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a write. The greater stamp wins; an equal-or-lesser stamp is
    /// rejected (so replay/merge is idempotent — re-applying a write that
    /// already won is a no-op rather than a spurious overwrite).
    pub fn put(&mut self, key: K, value: V, stamp: LwwStamp) -> PutOutcome<V> {
        if let Some((cur, cur_v)) = self.entries.get(&key) {
            if *cur >= stamp {
                return PutOutcome::Rejected { current: cur_v.clone(), current_stamp: *cur };
            }
        }
        let previous = self.entries.insert(key, (stamp, value)).map(|(_, old)| old);
        PutOutcome::Applied { previous }
    }

    /// The resolved value at `key` (tombstones are values too — the caller
    /// decides what an "empty" value means).
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|(_, v)| v)
    }

    /// The winning stamp at `key`.
    pub fn stamp(&self, key: &K) -> Option<LwwStamp> {
        self.entries.get(key).map(|(s, _)| *s)
    }

    /// Both the winning stamp and value at `key`.
    pub fn get_entry(&self, key: &K) -> Option<(LwwStamp, &V)> {
        self.entries.get(key).map(|(s, v)| (*s, v))
    }

    /// Iterate `(key, stamp, value)` over every entry (including tombstones).
    pub fn iter(&self) -> impl Iterator<Item = (&K, LwwStamp, &V)> {
        self.entries.iter().map(|(k, (s, v))| (k, *s, v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Merge `other` into `self` via per-key LWW. Commutative, associative, and
    /// idempotent (it is a fold of `put`, and `put` keeps the per-key max over
    /// the total order on [`LwwStamp`]).
    pub fn merge(&mut self, other: &Self) {
        for (k, (s, v)) in &other.entries {
            self.put(k.clone(), v.clone(), *s);
        }
    }

    /// Borrow the raw `(stamp, value)` entries — for snapshotting the full CRDT
    /// state (tombstones and stamps included).
    pub fn entries(&self) -> &HashMap<K, (LwwStamp, V)> {
        &self.entries
    }

    pub fn into_entries(self) -> HashMap<K, (LwwStamp, V)> {
        self.entries
    }

    /// Rebuild a map from raw `(stamp, value)` entries (e.g. a recovered
    /// snapshot).
    pub fn from_entries(entries: HashMap<K, (LwwStamp, V)>) -> Self {
        Self { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(wall: u64, counter: u32, writer: u64) -> LwwStamp {
        LwwStamp::new(HlcTimestamp::new(wall, counter), WriterId(writer))
    }

    /// A small voxel-like value: 0 is the tombstone ("empty"), non-zero solid.
    type V = u16;

    #[test]
    fn put_applies_strictly_greater_and_rejects_equal_or_lesser() {
        let mut m = LwwMap::<i32, V>::new();
        assert!(m.put(1, 7, stamp(10, 0, 1)).is_applied());
        assert_eq!(m.get(&1), Some(&7));

        // Equal stamp ⇒ rejected (idempotent replay).
        match m.put(1, 99, stamp(10, 0, 1)) {
            PutOutcome::Rejected { current, current_stamp } => {
                assert_eq!(current, 7);
                assert_eq!(current_stamp, stamp(10, 0, 1));
            }
            other => panic!("expected reject, got {other:?}"),
        }
        assert_eq!(m.get(&1), Some(&7));

        // Lesser stamp ⇒ rejected.
        assert!(!m.put(1, 5, stamp(9, 9, 9)).is_applied());
        assert_eq!(m.get(&1), Some(&7));

        // Greater stamp ⇒ applied, reports displaced value.
        match m.put(1, 8, stamp(11, 0, 1)) {
            PutOutcome::Applied { previous } => assert_eq!(previous, Some(7)),
            other => panic!("expected apply, got {other:?}"),
        }
        assert_eq!(m.get(&1), Some(&8));
    }

    #[test]
    fn writer_id_breaks_timestamp_ties() {
        let mut m = LwwMap::<i32, V>::new();
        m.put(1, 100, stamp(5, 0, 1));
        // Same HLC, higher writer ⇒ wins.
        assert!(m.put(1, 200, stamp(5, 0, 2)).is_applied());
        assert_eq!(m.get(&1), Some(&200));
        // Same HLC, lower writer ⇒ loses.
        assert!(!m.put(1, 300, stamp(5, 0, 1)).is_applied());
        assert_eq!(m.get(&1), Some(&200));
    }

    #[test]
    fn tombstone_is_retained_and_blocks_older_writes() {
        let mut m = LwwMap::<i32, V>::new();
        m.put(1, 7, stamp(10, 0, 1)); // solid
        m.put(1, 0, stamp(20, 0, 1)); // delete (tombstone, value 0)
        assert_eq!(m.get(&1), Some(&0)); // retained, not removed
        assert_eq!(m.len(), 1);
        // An older, out-of-order solid write must NOT resurrect the cell.
        assert!(!m.put(1, 9, stamp(15, 0, 1)).is_applied());
        assert_eq!(m.get(&1), Some(&0));
        // A newer write still wins.
        assert!(m.put(1, 9, stamp(21, 0, 1)).is_applied());
        assert_eq!(m.get(&1), Some(&9));
    }

    /// Build a map by applying a fixed write set in the given order.
    fn build(order: &[(i32, V, LwwStamp)]) -> LwwMap<i32, V> {
        let mut m = LwwMap::new();
        for (k, v, s) in order {
            m.put(*k, *v, *s);
        }
        m
    }

    #[test]
    fn convergence_is_order_independent() {
        let writes = [
            (1, 7u16, stamp(10, 0, 1)),
            (1, 8, stamp(12, 0, 1)),
            (2, 3, stamp(11, 0, 2)),
            (1, 5, stamp(9, 0, 3)),
            (2, 4, stamp(11, 0, 1)), // tie on ts, lower writer than the (2,3) write
        ];
        let forward = build(&writes);
        let mut rev: Vec<_> = writes.to_vec();
        rev.reverse();
        let backward = build(&rev);
        let mut shuffled = vec![writes[2], writes[0], writes[4], writes[3], writes[1]];
        let weird = build(&shuffled);
        shuffled.rotate_left(2);
        let weird2 = build(&shuffled);

        assert_eq!(forward, backward);
        assert_eq!(forward, weird);
        assert_eq!(forward, weird2);
        // Resolved winners.
        assert_eq!(forward.get(&1), Some(&8)); // ts 12 wins
        assert_eq!(forward.get(&2), Some(&3)); // ts 11 tie ⇒ writer 2 beats writer 1
    }

    #[test]
    fn merge_is_idempotent_commutative_associative() {
        let a = build(&[(1, 7, stamp(10, 0, 1)), (2, 1, stamp(5, 0, 1))]);
        let b = build(&[(1, 8, stamp(12, 0, 2)), (3, 9, stamp(3, 0, 1))]);
        let c = build(&[(2, 2, stamp(6, 0, 1)), (1, 4, stamp(11, 0, 9))]);

        // Idempotent: merging a copy of self changes nothing.
        let mut idem = a.clone();
        idem.merge(&a.clone());
        assert_eq!(idem, a);

        // Commutative: a∪b == b∪a.
        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);
        assert_eq!(ab, ba);

        // Associative: (a∪b)∪c == a∪(b∪c).
        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);
        let mut bc = b.clone();
        bc.merge(&c);
        let mut right = a.clone();
        right.merge(&bc);
        assert_eq!(left, right);

        // The resolved key-1 winner is ts 12 (writer 2), beating ts 11/10.
        assert_eq!(left.get(&1), Some(&8));
    }

    #[test]
    fn entries_round_trip() {
        let m = build(&[(1, 7, stamp(10, 0, 1)), (2, 0, stamp(20, 0, 2))]);
        let rebuilt = LwwMap::from_entries(m.entries().clone());
        assert_eq!(m, rebuilt);
    }
}
