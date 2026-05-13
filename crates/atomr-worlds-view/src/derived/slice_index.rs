//! Phase 14c — top-visible-voxel-per-column index for Dwarf-Fortress slice.
//!
//! # Convention: +Y is up, scanning down
//!
//! atomr-worlds uses **+Y world-up** (Phase 1 / `camera.rs` doc). The
//! Dwarf-Fortress "z-level" the user cycles through with `<`/`>` therefore
//! maps to **world Y** here — `z_level == world_y`, not world Z. The horizontal
//! tile plane is the (world-X, world-Z) plane. That's what `dims[0] × dims[1]`
//! indexes and what `origin_xz` is named after.
//!
//! For each (x, z) column on the horizontal plane the slice extracts the
//! **topmost** voxel inside a vertical band of `z_band_thickness` cells
//! starting at `z_band_top` (a world-Y level) and **scanning downward**
//! through `[z_band_top - 1 .. z_band_top - z_band_thickness]` (inclusive
//! min, inclusive max — `z_band_thickness` levels total). The first
//! non-empty voxel encountered becomes `top_voxel`; `top_z` records its
//! world-Y so renderer code can layer roofs/floors. If the entire band is
//! empty, `top_voxel = Voxel(0)` (empty), `top_z = z_band_top - z_band_thickness`
//! and `thickness_above_floor = 0`.
//!
//! `thickness_above_floor` counts how many *non-empty* voxels lie directly
//! below `top_voxel` inside the band (so a one-voxel-tall feature in a
//! 3-voxel band has `thickness_above_floor = 0`; a full pillar has
//! `thickness_above_floor = z_band_thickness - 1`). Renderer modes use it to
//! stipple "thin features" without dimming colour.
//!
//! # Caching
//!
//! [`SliceKey`] implements [`crate::DerivedKey`] so the slice table can live
//! inside a [`crate::ViewCache`]. The key's AABB covers the **full vertical
//! extent of the world for that horizontal footprint** — slice mode is
//! deliberately conservative because a `VoxelDelta` anywhere in the column
//! can change `top_voxel`, even a delta well below the configured z-band
//! (if it un-roofs the column the band's view of "what's solid" doesn't
//! change, but a delta *inside* the band absolutely does, and we keep the
//! AABB simple by not slicing it vertically). Wave 3+ can refine to the
//! actual band Y-range once we have benchmarks showing the cost.

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_voxel::brick::BRICK_EDGE;
use atomr_worlds_voxel::voxel::Voxel;

use crate::view_cache::{CacheAabb, DerivedKey};
use crate::world_query::WorldQuery;

/// One horizontal-plane column extracted by [`build_slice_table`].
///
/// See the module rustdoc for the "+Y up, scanning down" rule.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SliceColumn {
    /// The topmost non-empty voxel in the band, or [`Voxel::EMPTY`] if the
    /// band has no solid voxel in this column.
    pub top_voxel: Voxel,
    /// World-Y coordinate of `top_voxel`. If `top_voxel` is empty, this is
    /// the floor of the band — `z_band_top - z_band_thickness as i32`.
    pub top_z: i32,
    /// Count of contiguous non-empty voxels directly **below** `top_voxel`
    /// that also lie inside the band. Clamped to `z_band_thickness - 1`.
    /// `0` for a single-voxel feature, `z_band_thickness - 1` for a full
    /// pillar across the band.
    pub thickness_above_floor: u8,
}

/// A horizontal slice of a world at a fixed vertical band.
#[derive(Clone, Debug)]
pub struct SliceTable {
    /// Row-major `dims[0] × dims[1]` columns. Index for `(x, z)` is
    /// `z * dims[0] + x` where `x ∈ [0, dims[0])` and `z ∈ [0, dims[1])`.
    pub columns: Vec<SliceColumn>,
    /// Horizontal dimensions of the slice, `[width_x, width_z]` in voxels.
    pub dims: [u32; 2],
    /// World voxel coordinates `(x, z)` of column `(0, 0)`.
    pub origin_xz: [i32; 2],
    /// World-Y level at the top of the scanned band (inclusive in the
    /// "first level scanned" sense — see module rustdoc).
    pub z_band_top: i32,
    /// Number of vertical Y-levels in the band. Must be `>= 1`.
    pub z_band_thickness: u8,
    /// The [`WorldQuery`] revision the table was built against — Wave 3+
    /// will plumb this from the host; today it's an opaque caller-supplied
    /// stamp that downstream code can use to disambiguate rebuild waves.
    pub world_rev: u64,
}

impl SliceTable {
    /// Borrow the column at `(x, z)` in **table-local** coordinates, or
    /// `None` if out of range.
    #[inline]
    pub fn column(&self, x: u32, z: u32) -> Option<&SliceColumn> {
        if x >= self.dims[0] || z >= self.dims[1] {
            return None;
        }
        let idx = (z as usize) * (self.dims[0] as usize) + (x as usize);
        self.columns.get(idx)
    }

    /// AABB covering the full vertical extent of the slice's horizontal
    /// footprint. See module rustdoc for the rationale (it's conservative
    /// on purpose).
    pub fn aabb(&self) -> CacheAabb {
        let min_x = self.origin_xz[0] as f64;
        let min_z = self.origin_xz[1] as f64;
        let max_x = (self.origin_xz[0] + self.dims[0] as i32) as f64;
        let max_z = (self.origin_xz[1] + self.dims[1] as i32) as f64;
        // Vertical extent: full f64 range so VoxelDelta anywhere in column
        // counts as intersecting. `intersects` uses `<=`/`>=` so f64::MIN..MAX
        // matches anything.
        CacheAabb::new([min_x, f64::MIN, min_z], [max_x, f64::MAX, max_z])
    }
}

/// Cache key for a [`SliceTable`]. Hashable / Eq so it slots into
/// `HashMap`-based [`crate::ViewCache`].
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct SliceKey {
    pub addr: WorldAddr,
    pub origin_xz: [i32; 2],
    pub dims: [u32; 2],
    pub z_band_top: i32,
    pub z_band_thickness: u8,
}

impl SliceKey {
    /// AABB matching [`SliceTable::aabb`]. Identical computation so
    /// `invalidate_intersecting` always lines up with the rendered footprint.
    fn aabb(&self) -> CacheAabb {
        let min_x = self.origin_xz[0] as f64;
        let min_z = self.origin_xz[1] as f64;
        let max_x = (self.origin_xz[0] + self.dims[0] as i32) as f64;
        let max_z = (self.origin_xz[1] + self.dims[1] as i32) as f64;
        CacheAabb::new([min_x, f64::MIN, min_z], [max_x, f64::MAX, max_z])
    }
}

impl DerivedKey for SliceKey {
    fn world_addr(&self) -> &WorldAddr {
        &self.addr
    }
    fn intersects(&self, aabb: CacheAabb) -> bool {
        self.aabb().intersects(aabb)
    }
}

/// Build a slice table by scanning each (x, z) column downward through the
/// band. See module rustdoc for the convention.
///
/// `z_band_thickness` is clamped to `>= 1`. This entry point queries all
/// bricks at [`Lod::ROOT`]; use [`build_slice_table_with_lod_fn`] to pick
/// the LOD per-column from a `ChunkStreamer`-style distance policy.
pub fn build_slice_table(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    origin_xz: [i32; 2],
    dims: [u32; 2],
    z_band_top: i32,
    z_band_thickness: u8,
) -> SliceTable {
    build_slice_table_with_lod_fn(world, addr, origin_xz, dims, z_band_top, z_band_thickness, |_| {
        Lod::ROOT
    })
}

/// Like [`build_slice_table`], but the caller supplies a closure returning
/// the LOD to query for each `(world_x, world_z)` voxel column. Slice / RTS
/// modes pass `streamer.lod_for_meters(observer, …)` so far-away columns
/// fall back to a coarser brick tier.
///
/// The brick cache is keyed by `(brick_coord, lod.depth)` so a column at
/// `lod=1` doesn't blow away a neighbouring column's `lod=0` fetch.
pub fn build_slice_table_with_lod_fn<F>(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    origin_xz: [i32; 2],
    dims: [u32; 2],
    z_band_top: i32,
    z_band_thickness: u8,
    mut lod_for_column: F,
) -> SliceTable
where
    F: FnMut([i64; 2]) -> Lod,
{
    let thickness = z_band_thickness.max(1);
    let dx = dims[0];
    let dz = dims[1];
    let mut columns = Vec::with_capacity((dx as usize) * (dz as usize));

    // Cache the most-recently-fetched brick. Slice scans are coherent in
    // (x, z) and step down in y by one cell at a time, so the brick coord
    // changes rarely — caching avoids hammering `WorldQuery::brick` for
    // every voxel. The cache key now includes lod-depth so two adjacent
    // columns with different LOD selections don't thrash each other.
    let edge: i64 = BRICK_EDGE as i64;
    let mut cached: Option<((IVec3, u8), std::sync::Arc<atomr_worlds_voxel::brick::Brick>)> = None;

    for z_idx in 0..dz {
        for x_idx in 0..dx {
            let world_x: i64 = (origin_xz[0] as i64) + (x_idx as i64);
            let world_z: i64 = (origin_xz[1] as i64) + (z_idx as i64);

            // Per-column LOD: the streamer picks near_lod inside the
            // transition radius and far_lod beyond it.
            let lod = lod_for_column([world_x, world_z]);

            let mut top_voxel = Voxel::EMPTY;
            let mut top_y = z_band_top - thickness as i32;
            let mut thickness_above: u8 = 0;
            let mut found_top = false;

            for k in 0..thickness {
                let world_y: i64 = (z_band_top as i64) - 1 - (k as i64);
                let bc =
                    IVec3::new(world_x.div_euclid(edge), world_y.div_euclid(edge), world_z.div_euclid(edge));
                let lc =
                    IVec3::new(world_x.rem_euclid(edge), world_y.rem_euclid(edge), world_z.rem_euclid(edge));
                let cache_key = (bc, lod.depth);
                let brick_opt = match &cached {
                    Some((k_cached, b)) if *k_cached == cache_key => Some(b.clone()),
                    _ => {
                        let fetched = world.brick(addr, bc, lod);
                        if let Some(ref b) = fetched {
                            cached = Some((cache_key, b.clone()));
                        } else {
                            cached = None;
                        }
                        fetched
                    }
                };
                let v = brick_opt
                    .as_ref()
                    .map(|b| {
                        let idx = (lc.z as usize * BRICK_EDGE + lc.y as usize) * BRICK_EDGE + lc.x as usize;
                        b.voxels[idx]
                    })
                    .unwrap_or(Voxel::EMPTY);
                if !found_top {
                    if !v.is_empty() {
                        top_voxel = v;
                        top_y = world_y as i32;
                        found_top = true;
                    }
                } else if !v.is_empty() {
                    thickness_above = thickness_above.saturating_add(1);
                    if thickness_above >= thickness - 1 {
                        // Saturated at the maximum the band can hold; the
                        // remaining levels can't change the answer.
                        break;
                    }
                } else {
                    // Once we hit empty space below the top, stop counting —
                    // `thickness_above_floor` is "contiguous non-empty
                    // *directly below* the top".
                    break;
                }
            }

            columns.push(SliceColumn { top_voxel, top_z: top_y, thickness_above_floor: thickness_above });
        }
    }

    SliceTable { columns, dims, origin_xz, z_band_top, z_band_thickness: thickness, world_rev: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::sync::mpsc;
    use std::sync::Arc;

    use atomr_worlds_core::addr::WorldAddr;
    use atomr_worlds_core::lod::Lod;
    use atomr_worlds_proto::{WorldEvent, AABB};
    use atomr_worlds_voxel::brick::Brick;

    struct MapWorld {
        bricks: HashMap<IVec3, Arc<Brick>>,
    }

    impl MapWorld {
        fn new() -> Self {
            Self { bricks: HashMap::new() }
        }
        fn set_voxel(&mut self, world: IVec3, v: Voxel) {
            let edge: i64 = BRICK_EDGE as i64;
            let bc = IVec3::new(world.x.div_euclid(edge), world.y.div_euclid(edge), world.z.div_euclid(edge));
            let lc = IVec3::new(world.x.rem_euclid(edge), world.y.rem_euclid(edge), world.z.rem_euclid(edge));
            let entry = self.bricks.entry(bc).or_insert_with(|| Arc::new(Brick::new()));
            let brick = Arc::make_mut(entry);
            brick.set(lc, v);
        }
    }

    impl WorldQuery for MapWorld {
        fn brick(&self, _addr: &WorldAddr, bc: IVec3, _lod: Lod) -> Option<Arc<Brick>> {
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
    fn empty_column_marks_empty() {
        let world = MapWorld::new();
        let addr = WorldAddr::ROOT;
        let t = build_slice_table(&world, &addr, [0, 0], [1, 1], 5, 3);
        let c = t.column(0, 0).unwrap();
        assert_eq!(c.top_voxel, Voxel::EMPTY);
        assert_eq!(c.top_z, 5 - 3);
        assert_eq!(c.thickness_above_floor, 0);
    }

    #[test]
    fn finds_top_voxel_in_band() {
        // band scans world_y = 4, 3, 2 (top=5, thickness=3).
        let mut world = MapWorld::new();
        world.set_voxel(IVec3::new(0, 3, 0), Voxel::new(2));
        let t = build_slice_table(&world, &WorldAddr::ROOT, [0, 0], [1, 1], 5, 3);
        let c = t.column(0, 0).unwrap();
        assert_eq!(c.top_voxel, Voxel::new(2));
        assert_eq!(c.top_z, 3);
    }

    #[test]
    fn aabb_is_full_y_range() {
        let world = MapWorld::new();
        let t = build_slice_table(&world, &WorldAddr::ROOT, [10, 20], [4, 8], 0, 3);
        let a = t.aabb();
        assert_eq!(a.min[0], 10.0);
        assert_eq!(a.max[0], 14.0);
        assert_eq!(a.min[2], 20.0);
        assert_eq!(a.max[2], 28.0);
        // Vertical extent is effectively unbounded; any test point's y
        // should land inside.
        assert!(a.contains([12.0, 1.0e9, 22.0]));
        assert!(a.contains([12.0, -1.0e9, 22.0]));
    }

    #[test]
    fn slice_key_intersects_uses_table_aabb() {
        let key = SliceKey {
            addr: WorldAddr::ROOT,
            origin_xz: [0, 0],
            dims: [4, 4],
            z_band_top: 0,
            z_band_thickness: 3,
        };
        // A box overlapping the footprint at any Y should intersect.
        assert!(key.intersects(CacheAabb::new([1.0, 100.0, 1.0], [2.0, 200.0, 2.0])));
        // A box outside the horizontal footprint should not intersect.
        assert!(!key.intersects(CacheAabb::new([10.0, 0.0, 10.0], [12.0, 1.0, 12.0])));
    }
}
