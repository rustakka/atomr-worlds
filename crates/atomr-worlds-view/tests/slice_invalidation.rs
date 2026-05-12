//! Phase 14c gate: the slice cache rebuilds when (and only when) a
//! `VoxelDelta`-shaped AABB overlaps the slice's horizontal footprint.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_view::{build_slice_table, CacheAabb, SliceKey, SliceTable, ViewCache, WorldQuery};
use atomr_worlds_voxel::brick::Brick;
use atomr_worlds_voxel::voxel::Voxel;

/// Counter-incrementing `WorldQuery` — every `brick` call bumps a counter
/// so we can assert that `get_or_build` only ran when expected.
struct CountingWorld {
    bricks: HashMap<IVec3, Arc<Brick>>,
    calls: Arc<AtomicUsize>,
}

impl CountingWorld {
    fn new(calls: Arc<AtomicUsize>) -> Self {
        let mut bricks = HashMap::new();
        let mut b = Brick::new();
        b.set(IVec3::new(0, 0, 0), Voxel::new(1));
        bricks.insert(IVec3::new(0, 0, 0), Arc::new(b));
        Self { bricks, calls }
    }
}

impl WorldQuery for CountingWorld {
    fn brick(&self, _addr: &WorldAddr, bc: IVec3, _lod: Lod) -> Option<Arc<Brick>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.bricks.get(&bc).cloned()
    }
    fn ground_height_m(&self, _addr: &WorldAddr, _xz: [f64; 2]) -> Option<f32> {
        None
    }
    fn subscribe_region(
        &self,
        _addr: &WorldAddr,
        _region: AABB,
        _lod: Lod,
    ) -> std::sync::mpsc::Receiver<WorldEvent> {
        let (_tx, rx) = mpsc::channel();
        rx
    }
}

#[test]
fn cache_hit_skips_rebuild() {
    let calls = Arc::new(AtomicUsize::new(0));
    let world = CountingWorld::new(calls.clone());
    let cache: ViewCache<SliceKey, SliceTable> = ViewCache::new();
    let addr = WorldAddr::ROOT;
    let key = SliceKey { addr, origin_xz: [0, 0], dims: [4, 4], z_band_top: 3, z_band_thickness: 3 };

    let _ = cache.get_or_build(key.clone(), || build_slice_table(&world, &addr, [0, 0], [4, 4], 3, 3));
    let first_calls = calls.load(Ordering::SeqCst);
    assert!(first_calls > 0);

    let _ = cache.get_or_build(key.clone(), || build_slice_table(&world, &addr, [0, 0], [4, 4], 3, 3));
    assert_eq!(calls.load(Ordering::SeqCst), first_calls, "cache hit must not re-run the builder");
}

#[test]
fn invalidation_intersecting_footprint_triggers_rebuild() {
    let calls = Arc::new(AtomicUsize::new(0));
    let world = CountingWorld::new(calls.clone());
    let cache: ViewCache<SliceKey, SliceTable> = ViewCache::new();
    let addr = WorldAddr::ROOT;
    let key = SliceKey { addr, origin_xz: [0, 0], dims: [4, 4], z_band_top: 3, z_band_thickness: 3 };

    let _ = cache.get_or_build(key.clone(), || build_slice_table(&world, &addr, [0, 0], [4, 4], 3, 3));
    let baseline = calls.load(Ordering::SeqCst);

    // VoxelDelta-shaped AABB at world (2, 1, 2) — inside the footprint.
    let evicted = cache.invalidate_intersecting(CacheAabb::new([2.0, 1.0, 2.0], [3.0, 2.0, 3.0]));
    assert_eq!(evicted, 1, "intersecting delta should evict the entry");

    let _ = cache.get_or_build(key.clone(), || build_slice_table(&world, &addr, [0, 0], [4, 4], 3, 3));
    assert!(
        calls.load(Ordering::SeqCst) > baseline,
        "after intersecting invalidation the table must be rebuilt"
    );
}

#[test]
fn invalidation_outside_footprint_keeps_cache() {
    let calls = Arc::new(AtomicUsize::new(0));
    let world = CountingWorld::new(calls.clone());
    let cache: ViewCache<SliceKey, SliceTable> = ViewCache::new();
    let addr = WorldAddr::ROOT;
    let key = SliceKey { addr, origin_xz: [0, 0], dims: [4, 4], z_band_top: 3, z_band_thickness: 3 };

    let _ = cache.get_or_build(key.clone(), || build_slice_table(&world, &addr, [0, 0], [4, 4], 3, 3));
    let baseline = calls.load(Ordering::SeqCst);

    // AABB at world (10, 1, 10) — outside the [0, 4) × [0, 4) footprint.
    let evicted = cache.invalidate_intersecting(CacheAabb::new([10.0, 1.0, 10.0], [11.0, 2.0, 11.0]));
    assert_eq!(evicted, 0, "non-overlapping delta should not evict");

    let _ = cache.get_or_build(key.clone(), || build_slice_table(&world, &addr, [0, 0], [4, 4], 3, 3));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        baseline,
        "non-overlapping invalidation must not trigger a rebuild"
    );
}

#[test]
fn vertical_only_delta_still_intersects() {
    // The slice key's AABB has full f64 vertical extent on purpose — see
    // module rustdoc. A delta well above or below the band still counts as
    // an intersection because we can't cheaply tell from the AABB alone
    // whether it would affect the column's `top_voxel`.
    let calls = Arc::new(AtomicUsize::new(0));
    let world = CountingWorld::new(calls.clone());
    let cache: ViewCache<SliceKey, SliceTable> = ViewCache::new();
    let addr = WorldAddr::ROOT;
    let key = SliceKey { addr, origin_xz: [0, 0], dims: [4, 4], z_band_top: 3, z_band_thickness: 3 };
    let _ = cache.get_or_build(key.clone(), || build_slice_table(&world, &addr, [0, 0], [4, 4], 3, 3));
    // Way above the band but horizontally inside the footprint.
    let evicted =
        cache.invalidate_intersecting(CacheAabb::new([1.0, 1_000_000.0, 1.0], [2.0, 1_000_001.0, 2.0]));
    assert_eq!(evicted, 1);
}
