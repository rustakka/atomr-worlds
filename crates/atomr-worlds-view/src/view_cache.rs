//! Generic derived-data cache for view-mode pipelines (Phase 14c/d/e).
//!
//! [`ViewCache`] is a `HashMap`-backed, `RwLock`-protected cache from a
//! [`DerivedKey`] (typed per view mode) to an `Arc<V>` of expensive-to-build
//! derived data — a `SliceTable` (14c), a `SurfaceRaster` (14d), or a
//! `WorldSummaryPyramid` tile (14e). The cache exposes three eviction
//! primitives:
//!
//! - [`ViewCache::invalidate_intersecting`] — drop every entry whose key
//!   intersects a given AABB. Wired to host `RegionDelta` events.
//! - [`ViewCache::invalidate_world`] — drop every entry under a `WorldAddr`.
//!   Wired to whole-world unloads / regenerations.
//! - [`ViewCache::invalidate_key`] — explicit single-entry eviction.
//!
//! The AABB shape here is intentionally minimal ([`CacheAabb`]) so this module
//! does not need to depend on `atomr-worlds-proto`. Once the view crate gains
//! a proto dep in a later wave, [`CacheAabb`] can be replaced by the proto
//! AABB without callers needing to change.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use atomr_worlds_core::addr::WorldAddr;

/// Minimal AABB matching the proto-side shape; once view depends on proto
/// (Wave 1b lands in parallel), this can be replaced by `atomr_worlds_proto::AABB`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CacheAabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl CacheAabb {
    /// Construct an AABB. The caller is expected to pass `min <= max`
    /// componentwise; this is not enforced.
    #[inline]
    pub const fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Axis-overlap test. Touching faces count as intersecting (closed boxes).
    #[inline]
    pub fn intersects(self, other: CacheAabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Point-in-box test (closed).
    #[inline]
    pub fn contains(self, p: [f64; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }
}

/// Stable revision-token type that view modes can use as a coarse
/// cache-buster — e.g., Phase 13c `macro_rev` for the overview (14e) cache.
/// A change in [`Revision`] makes a cache key compare-unequal, so old entries
/// stay until explicit eviction; new builds slot in alongside them. Combine
/// with [`ViewCache::invalidate_intersecting`] to drop stale revisions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Revision(pub u64);

/// Trait implemented by per-mode cache keys. A key must be hashable (for the
/// `HashMap`), name a [`WorldAddr`] (for `invalidate_world`), and answer an
/// AABB intersection test (for `invalidate_intersecting`).
pub trait DerivedKey: Hash + Eq + Clone + std::fmt::Debug + Send + Sync + 'static {
    /// World this key's derived data lives in.
    fn world_addr(&self) -> &WorldAddr;
    /// Does this key's spatial extent intersect `aabb`?
    fn intersects(&self, aabb: CacheAabb) -> bool;
}

struct CacheEntry<V> {
    /// Per-entry revision counter. Reserved for future use (e.g.,
    /// stamp-based partial invalidation); not part of the public API yet.
    #[allow(dead_code)]
    revision: u64,
    value: Arc<V>,
}

/// Generic derived-data cache. See module-level docs.
pub struct ViewCache<K: DerivedKey, V: Send + Sync + 'static> {
    entries: RwLock<HashMap<K, CacheEntry<V>>>,
}

impl<K: DerivedKey, V: Send + Sync + 'static> std::fmt::Debug for ViewCache<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.entries.read().map(|e| e.len()).unwrap_or(0);
        f.debug_struct("ViewCache").field("len", &len).finish()
    }
}

impl<K: DerivedKey, V: Send + Sync + 'static> ViewCache<K, V> {
    pub fn new() -> Self {
        Self { entries: RwLock::new(HashMap::new()) }
    }

    /// Return the cached value if present, otherwise run `build`, store the
    /// result, and return it. The build closure runs while holding the write
    /// lock — callers should not perform unbounded I/O inside it. A second
    /// concurrent caller for the same key may end up running `build` twice
    /// only if it raced the first one's read-then-upgrade window; that is
    /// not avoided here because doing so would require a double-check
    /// (cheap) which is included.
    pub fn get_or_build(&self, key: K, build: impl FnOnce() -> V) -> Arc<V> {
        // Fast path: read lock, hit.
        if let Some(entry) = self.entries.read().expect("ViewCache lock poisoned").get(&key) {
            return entry.value.clone();
        }
        // Slow path: take write lock, double-check, build, insert.
        let mut guard = self.entries.write().expect("ViewCache lock poisoned");
        if let Some(entry) = guard.get(&key) {
            return entry.value.clone();
        }
        let value = Arc::new(build());
        let entry = CacheEntry { revision: 0, value: value.clone() };
        guard.insert(key, entry);
        value
    }

    /// Evict every entry whose key intersects `aabb`. Returns the number of
    /// entries dropped.
    pub fn invalidate_intersecting(&self, aabb: CacheAabb) -> usize {
        let mut guard = self.entries.write().expect("ViewCache lock poisoned");
        let before = guard.len();
        guard.retain(|k, _| !k.intersects(aabb));
        before - guard.len()
    }

    /// Evict every entry whose [`WorldAddr`] equals `addr`. Returns the
    /// number of entries dropped.
    pub fn invalidate_world(&self, addr: &WorldAddr) -> usize {
        let mut guard = self.entries.write().expect("ViewCache lock poisoned");
        let before = guard.len();
        guard.retain(|k, _| k.world_addr() != addr);
        before - guard.len()
    }

    /// Explicitly evict a single key. Returns `true` if an entry was
    /// present and removed.
    pub fn invalidate_key(&self, key: &K) -> bool {
        let mut guard = self.entries.write().expect("ViewCache lock poisoned");
        guard.remove(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.entries.read().expect("ViewCache lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().expect("ViewCache lock poisoned").is_empty()
    }
}

impl<K: DerivedKey, V: Send + Sync + 'static> Default for ViewCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::addr::{LevelKey, WorldAddr};
    use atomr_worlds_core::coord::IVec3;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn world(x: i64) -> WorldAddr {
        let mut w = WorldAddr::ROOT;
        w.world = LevelKey::at(IVec3::new(x, 0, 0));
        w
    }

    #[derive(Clone, Debug, Hash, Eq, PartialEq)]
    struct TestKey {
        addr: WorldAddr,
        aabb_min: [i64; 3],
        aabb_max: [i64; 3],
        rev: Revision,
    }

    impl TestKey {
        fn new(addr: WorldAddr, min: [i64; 3], max: [i64; 3]) -> Self {
            Self { addr, aabb_min: min, aabb_max: max, rev: Revision(0) }
        }
        fn aabb(&self) -> CacheAabb {
            CacheAabb::new(
                [self.aabb_min[0] as f64, self.aabb_min[1] as f64, self.aabb_min[2] as f64],
                [self.aabb_max[0] as f64, self.aabb_max[1] as f64, self.aabb_max[2] as f64],
            )
        }
    }

    impl DerivedKey for TestKey {
        fn world_addr(&self) -> &WorldAddr {
            &self.addr
        }
        fn intersects(&self, aabb: CacheAabb) -> bool {
            self.aabb().intersects(aabb)
        }
    }

    #[test]
    fn get_or_build_runs_once() {
        let cache: ViewCache<TestKey, u32> = ViewCache::new();
        let calls = AtomicUsize::new(0);
        let key = TestKey::new(world(0), [0, 0, 0], [1, 1, 1]);

        let v1 = cache.get_or_build(key.clone(), || {
            calls.fetch_add(1, Ordering::SeqCst);
            42u32
        });
        let v2 = cache.get_or_build(key.clone(), || {
            calls.fetch_add(1, Ordering::SeqCst);
            99u32
        });

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(*v1, 42);
        assert_eq!(*v2, 42);
        assert!(Arc::ptr_eq(&v1, &v2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalidate_intersecting_evicts_overlap() {
        let cache: ViewCache<TestKey, u32> = ViewCache::new();
        // Three disjoint AABBs along x.
        let k_a = TestKey::new(world(0), [0, 0, 0], [10, 10, 10]);
        let k_b = TestKey::new(world(0), [20, 0, 0], [30, 10, 10]);
        let k_c = TestKey::new(world(0), [40, 0, 0], [50, 10, 10]);
        cache.get_or_build(k_a.clone(), || 1u32);
        cache.get_or_build(k_b.clone(), || 2u32);
        cache.get_or_build(k_c.clone(), || 3u32);
        assert_eq!(cache.len(), 3);

        // Invalidate rectangle that only touches k_b.
        let evicted = cache.invalidate_intersecting(CacheAabb::new([22.0, 0.0, 0.0], [28.0, 5.0, 5.0]));
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 2);
        assert!(!cache.invalidate_key(&k_b));
        assert!(cache.invalidate_key(&k_a));
        assert!(cache.invalidate_key(&k_c));
    }

    #[test]
    fn invalidate_world_evicts_one_world() {
        let cache: ViewCache<TestKey, u32> = ViewCache::new();
        let w0 = world(0);
        let w1 = world(1);
        let k_w0_a = TestKey::new(w0, [0, 0, 0], [1, 1, 1]);
        let k_w0_b = TestKey::new(w0, [10, 0, 0], [11, 1, 1]);
        let k_w1_a = TestKey::new(w1, [0, 0, 0], [1, 1, 1]);
        let k_w1_b = TestKey::new(w1, [10, 0, 0], [11, 1, 1]);
        for k in [&k_w0_a, &k_w0_b, &k_w1_a, &k_w1_b] {
            cache.get_or_build(k.clone(), || 0u32);
        }
        assert_eq!(cache.len(), 4);

        let evicted = cache.invalidate_world(&w0);
        assert_eq!(evicted, 2);
        assert_eq!(cache.len(), 2);
        // Remaining entries are w1's.
        assert!(cache.invalidate_key(&k_w1_a));
        assert!(cache.invalidate_key(&k_w1_b));
        assert!(!cache.invalidate_key(&k_w0_a));
        assert!(!cache.invalidate_key(&k_w0_b));
    }

    #[test]
    fn invalidate_key_evicts_one() {
        let cache: ViewCache<TestKey, u32> = ViewCache::new();
        let k_a = TestKey::new(world(0), [0, 0, 0], [1, 1, 1]);
        let k_b = TestKey::new(world(0), [2, 0, 0], [3, 1, 1]);
        cache.get_or_build(k_a.clone(), || 1u32);
        cache.get_or_build(k_b.clone(), || 2u32);

        assert!(cache.invalidate_key(&k_a));
        assert!(!cache.invalidate_key(&k_a));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn revision_distinct_keys() {
        let cache: ViewCache<TestKey, u32> = ViewCache::new();
        let base = TestKey::new(world(0), [0, 0, 0], [10, 10, 10]);
        let mut k_r0 = base.clone();
        k_r0.rev = Revision(0);
        let mut k_r1 = base.clone();
        k_r1.rev = Revision(1);

        cache.get_or_build(k_r0.clone(), || 100u32);
        cache.get_or_build(k_r1.clone(), || 200u32);
        assert_eq!(cache.len(), 2);

        // Hit-test both: stays at 2, values unchanged.
        let v0 = cache.get_or_build(k_r0.clone(), || 999u32);
        let v1 = cache.get_or_build(k_r1.clone(), || 999u32);
        assert_eq!(*v0, 100);
        assert_eq!(*v1, 200);
        assert_eq!(cache.len(), 2);

        // Evict only the r0 entry via key.
        assert!(cache.invalidate_key(&k_r0));
        assert_eq!(cache.len(), 1);
        // r1 still present.
        assert!(cache.invalidate_key(&k_r1));
        assert!(cache.is_empty());
    }
}
