//! Cross-brick feature anchor primitives.
//!
//! Path-based features (Perlin worms, ore veins, WFC dungeons, floating
//! islands, L-system trees) are anchored on a coarse column grid; every
//! brick scans the 3×3×3 neighborhood of columns and deterministically
//! traces from each anchor's seed, clipping output to the brick AABB.
//! Anchors are never recursive — a worm anchored in column C is fully
//! traced regardless of which brick triggered the trace.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use atomr_worlds_core::coord::IVec3;

/// Discriminates anchor kinds so each pipeline stage only consumes its own.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum FeatureKind {
    Worm,
    OreVein,
    Structure,
    FloatingIsland,
    BufferTerrain,
    FloraTree,
}

/// One anchor on the coarse column grid. `seed` is `child_seed(world_seed,
/// FEATURE_DIM, column)` mixed with the anchor's kind so two stages on the
/// same column never collide.
#[derive(Copy, Clone, Debug)]
pub struct FeatureAnchor {
    pub kind: FeatureKind,
    /// Column position on the anchor grid (column, not voxel, coords).
    pub column: IVec3,
    /// World-meter origin of the feature inside the column.
    pub origin_m: [f32; 3],
    pub seed: u64,
}

/// Process-wide memoization of column-anchor sets, analogous to
/// [`crate::macro_state::MacroStateCache`]. Keyed by `(world_seed,
/// column)` so cross-brick traces share one materialized anchor list.
#[derive(Debug, Default)]
pub struct FeatureAnchorCache {
    inner: Mutex<HashMap<(u64, IVec3), Vec<FeatureAnchor>>>,
}

impl FeatureAnchorCache {
    pub fn new() -> Self {
        Self { inner: Mutex::new(HashMap::new()) }
    }

    pub fn get_or_seed<F>(&self, world_seed: u64, column: IVec3, seed_fn: F) -> Vec<FeatureAnchor>
    where
        F: FnOnce() -> Vec<FeatureAnchor>,
    {
        let mut g = self.lock();
        if let Some(v) = g.get(&(world_seed, column)) {
            return v.clone();
        }
        let v = seed_fn();
        g.insert((world_seed, column), v.clone());
        v
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    pub fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<(u64, IVec3), Vec<FeatureAnchor>>> {
        self.inner.lock().expect("FeatureAnchorCache poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dedups_repeat_seeds() {
        let cache = FeatureAnchorCache::new();
        let mut calls = 0;
        for _ in 0..3 {
            let v = cache.get_or_seed(7, IVec3::new(1, 2, 3), || {
                calls += 1;
                vec![FeatureAnchor {
                    kind: FeatureKind::Worm,
                    column: IVec3::new(1, 2, 3),
                    origin_m: [0.0; 3],
                    seed: 0x1234,
                }]
            });
            assert_eq!(v.len(), 1);
        }
        assert_eq!(calls, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_separates_by_world_seed() {
        let cache = FeatureAnchorCache::new();
        cache.get_or_seed(7, IVec3::new(0, 0, 0), Vec::new);
        cache.get_or_seed(8, IVec3::new(0, 0, 0), Vec::new);
        assert_eq!(cache.len(), 2);
    }
}
