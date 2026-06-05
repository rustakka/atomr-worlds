//! Hybrid Logical Clock (HLC) timestamps.
//!
//! An HLC (Kulkarni et al., 2014) gives every event a strict, causality-
//! preserving total order across machines **without** synchronized clocks. It
//! pairs a physical-time component (nanoseconds) with a logical counter that
//! advances when physical time stands still or goes backwards, so two events
//! never collide and the order respects happens-before.
//!
//! atomr-worlds needs this for **Rec 4 (multiplayer destruction sync)**: the
//! per-cell last-writer-wins voxel overlay timestamps each write with an HLC so
//! concurrent carves by different players converge (higher timestamp wins,
//! tie-broken by node id at the map layer) without a central clock or blocking
//! arbitration. The sibling `atomr-distributed-data` CRDT library ships
//! `LWWMap` but uses a raw `u128` timestamp with no HLC; this type fills that
//! gap locally (and is a candidate to upstream).
//!
//! # Determinism / testability
//!
//! [`HlcTimestamp::tick`] and [`HlcTimestamp::recv`] are **pure functions** —
//! the caller passes the current wall-clock reading rather than the clock being
//! read inside — so they are fully deterministic and unit-testable, and the
//! overlay-merge logic that consumes them stays reproducible.

use serde::{Deserialize, Serialize};

/// A Hybrid Logical Clock timestamp. Ordered first by physical time
/// (`wall_ns`), then by the logical `counter`. `Ord` derives that ordering from
/// field order, which is exactly the HLC comparison.
#[derive(
    Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default, Debug, Serialize, Deserialize,
)]
pub struct HlcTimestamp {
    /// Physical-time component in nanoseconds (wall clock).
    pub wall_ns: u64,
    /// Logical counter, incremented when `wall_ns` does not advance.
    pub counter: u32,
}

impl HlcTimestamp {
    pub const ZERO: Self = Self { wall_ns: 0, counter: 0 };

    #[inline]
    pub const fn new(wall_ns: u64, counter: u32) -> Self {
        Self { wall_ns, counter }
    }

    /// Stamp a **local** event. `local_last` is this node's most recent HLC;
    /// `now_ns` is the current wall-clock reading (the caller reads the clock,
    /// keeping this pure). The result is strictly greater than `local_last`.
    ///
    /// If wall time advanced past `local_last`, the counter resets to `0`;
    /// otherwise (clock stalled or skewed backwards) the counter increments so
    /// the stamp still moves forward monotonically.
    #[inline]
    pub fn tick(local_last: Self, now_ns: u64) -> Self {
        let wall = now_ns.max(local_last.wall_ns);
        let counter = if wall == local_last.wall_ns { local_last.counter + 1 } else { 0 };
        Self { wall_ns: wall, counter }
    }

    /// Update the local clock on **receiving** a remote event timestamped
    /// `remote`. Returns a stamp that is strictly greater than both
    /// `local_last` and `remote`, preserving causality across the two clocks.
    #[inline]
    pub fn recv(local_last: Self, remote: Self, now_ns: u64) -> Self {
        let wall = now_ns.max(local_last.wall_ns).max(remote.wall_ns);
        let counter = if wall == local_last.wall_ns && wall == remote.wall_ns {
            local_last.counter.max(remote.counter) + 1
        } else if wall == local_last.wall_ns {
            local_last.counter + 1
        } else if wall == remote.wall_ns {
            remote.counter + 1
        } else {
            0
        };
        Self { wall_ns: wall, counter }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_is_strictly_monotonic() {
        let t0 = HlcTimestamp::ZERO;
        let t1 = HlcTimestamp::tick(t0, 100);
        let t2 = HlcTimestamp::tick(t1, 100); // same wall time
        let t3 = HlcTimestamp::tick(t2, 200); // wall advances
        assert!(t1 > t0);
        assert!(t2 > t1);
        assert!(t3 > t2);
    }

    #[test]
    fn counter_bumps_when_wall_stalls_then_resets() {
        let a = HlcTimestamp::tick(HlcTimestamp::ZERO, 50);
        assert_eq!(a, HlcTimestamp::new(50, 0));
        let b = HlcTimestamp::tick(a, 50); // stalled
        assert_eq!(b, HlcTimestamp::new(50, 1));
        let c = HlcTimestamp::tick(b, 50); // still stalled
        assert_eq!(c, HlcTimestamp::new(50, 2));
        let d = HlcTimestamp::tick(c, 60); // advanced
        assert_eq!(d, HlcTimestamp::new(60, 0));
    }

    #[test]
    fn tick_handles_backwards_clock() {
        // Wall clock jumps backwards (NTP correction): the stamp must still move
        // forward via the counter rather than regress.
        let a = HlcTimestamp::tick(HlcTimestamp::ZERO, 1000);
        let b = HlcTimestamp::tick(a, 500); // earlier wall reading
        assert!(b > a);
        assert_eq!(b.wall_ns, 1000);
        assert_eq!(b.counter, 1);
    }

    #[test]
    fn recv_dominates_both_inputs() {
        let local = HlcTimestamp::new(100, 3);
        let remote = HlcTimestamp::new(100, 5);
        let merged = HlcTimestamp::recv(local, remote, 100);
        assert!(merged > local);
        assert!(merged > remote);
        // Same wall on all three ⇒ max counter + 1.
        assert_eq!(merged, HlcTimestamp::new(100, 6));
    }

    #[test]
    fn recv_takes_the_furthest_wall_time() {
        let local = HlcTimestamp::new(100, 9);
        let remote = HlcTimestamp::new(250, 1);
        let merged = HlcTimestamp::recv(local, remote, 120);
        // Remote wall is furthest; adopt it and bump its counter.
        assert_eq!(merged, HlcTimestamp::new(250, 2));
        assert!(merged > local && merged > remote);
    }

    #[test]
    fn ordering_is_total_and_lexicographic() {
        assert!(HlcTimestamp::new(1, 9) < HlcTimestamp::new(2, 0));
        assert!(HlcTimestamp::new(2, 0) < HlcTimestamp::new(2, 1));
        assert_eq!(HlcTimestamp::new(2, 1), HlcTimestamp::new(2, 1));
    }

    #[test]
    fn pure_functions_are_deterministic() {
        let last = HlcTimestamp::new(42, 7);
        assert_eq!(HlcTimestamp::tick(last, 42), HlcTimestamp::tick(last, 42));
        let r = HlcTimestamp::new(40, 2);
        assert_eq!(
            HlcTimestamp::recv(last, r, 41),
            HlcTimestamp::recv(last, r, 41)
        );
    }

    #[test]
    fn serde_round_trip() {
        let t = HlcTimestamp::new(123_456_789, 4);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<HlcTimestamp>(&json).unwrap(), t);
    }
}
