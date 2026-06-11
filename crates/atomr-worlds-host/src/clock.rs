//! Injectable wall-clock source for HLC stamping.
//!
//! [`HlcTimestamp::tick`](atomr_worlds_core::HlcTimestamp::tick) is a pure
//! function of the current nanosecond reading, so the *only* nondeterminism in
//! the LWW overlay is where that reading comes from. Production reads the system
//! clock; determinism tests and the screenshot harness inject a monotonic
//! counter so a fixed write script produces a byte-identical journal across
//! runs. Either way the resolved voxel values (hence `GetBrick` bytes) are
//! identical — the HLC counter guarantees strict monotonicity even when the
//! reading stalls — so this seam never perturbs the golden captures.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// A clock the world actor reads when stamping a write's HLC.
#[derive(Clone, Debug)]
pub enum Clock {
    /// Real time — `SystemTime` nanoseconds since the Unix epoch.
    Wall,
    /// A deterministic counter (nanoseconds), shared so a test can `advance`
    /// it. Start it at ≥ 1: migrated legacy journal entries occupy `wall_ns
    /// == 0`, so live stamps must read a positive time to dominate them.
    Manual(Arc<AtomicU64>),
}

impl Default for Clock {
    fn default() -> Self {
        Clock::Wall
    }
}

impl Clock {
    /// A deterministic clock seeded at `start_ns` (use ≥ 1).
    pub fn manual(start_ns: u64) -> Self {
        Clock::Manual(Arc::new(AtomicU64::new(start_ns)))
    }

    /// Current reading in nanoseconds.
    pub fn now_ns(&self) -> u64 {
        match self {
            Clock::Wall => SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            Clock::Manual(a) => a.load(Ordering::SeqCst),
        }
    }

    /// Advance a [`Clock::Manual`] by `by_ns` (no-op for [`Clock::Wall`]).
    pub fn advance(&self, by_ns: u64) {
        if let Clock::Manual(a) = self {
            a.fetch_add(by_ns, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_is_stable_until_advanced() {
        let c = Clock::manual(1);
        assert_eq!(c.now_ns(), 1);
        assert_eq!(c.now_ns(), 1);
        c.advance(99);
        assert_eq!(c.now_ns(), 100);
    }

    #[test]
    fn wall_clock_is_positive() {
        assert!(Clock::Wall.now_ns() > 0);
    }
}
