//! Phase 14d gate: writes strictly below the column's `top_z - 1` must
//! NOT invalidate the cached [`SurfaceRaster`], but writes at or above
//! the current top voxel MUST invalidate it. This keeps the RTS-mode
//! cache cheap in the common "subsurface tunneling / mining" case.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_view::{build_surface_raster, WorldQuery};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

struct OneColumn {
    bricks: HashMap<IVec3, Arc<Brick>>,
}

impl OneColumn {
    fn new(top_vy: i32) -> Self {
        let mut bricks: HashMap<IVec3, Brick> = HashMap::new();
        let edge = BRICK_EDGE as i32;
        // Column at world voxel (4, 4) with top at `top_vy`. Add a few
        // voxels below the top so we have a multi-voxel column to test
        // strict-below writes against.
        for vy in (top_vy - 3)..=top_vy {
            let bc = IVec3::new(
                4i32.div_euclid(edge) as i64,
                vy.div_euclid(edge) as i64,
                4i32.div_euclid(edge) as i64,
            );
            let lc = IVec3::new(
                4i32.rem_euclid(edge) as i64,
                vy.rem_euclid(edge) as i64,
                4i32.rem_euclid(edge) as i64,
            );
            bricks.entry(bc).or_insert_with(Brick::new).set(lc, Voxel::new(1));
        }
        Self { bricks: bricks.into_iter().map(|(k, v)| (k, Arc::new(v))).collect() }
    }
}

impl WorldQuery for OneColumn {
    fn brick(&self, _addr: &WorldAddr, bc: IVec3, _lod: Lod) -> Option<Arc<Brick>> {
        self.bricks.get(&bc).cloned()
    }
    fn ground_height_m(&self, _addr: &WorldAddr, _xz: [f64; 2]) -> Option<f32> {
        None
    }
    fn subscribe_region(&self, _addr: &WorldAddr, _r: AABB, _lod: Lod) -> mpsc::Receiver<WorldEvent> {
        let (_tx, rx) = mpsc::channel();
        rx
    }
}

#[test]
fn sub_surface_write_does_not_invalidate() {
    let top_vy = 7;
    let world = OneColumn::new(top_vy);
    let raster = build_surface_raster(&world, &WorldAddr::ROOT, [0.0, 0.0], [8, 8], 1.0, Lod::new(0));

    // Cross-check: `top_z[col]` matches what we seeded.
    let col_idx = 4 * (raster.dims[0] as usize) + 4;
    assert_eq!(raster.top_z[col_idx], top_vy, "top_z should match the seeded top voxel");

    // Compute a voxel-Y strictly below `top_z - 1`. The plan calls for
    // `heightmap_m[x,z] - 1.0` but heightmap is at voxel-center
    // (top_vy + 0.5), so "strictly below heightmap_m - 1" means
    // voxel-Y `<= top_vy - 2`. We use `top_vy - 2`.
    let sub_vy = (top_vy - 2) as i64;
    assert!(
        !raster.is_invalidated_by_write(4, sub_vy, 4),
        "write at vy={sub_vy} is strictly below top_z-1 ({}), must not invalidate",
        top_vy - 1
    );

    // Conversely: a write at `top_vy` MUST invalidate (replacing the
    // topmost voxel can change both height and biome).
    assert!(raster.is_invalidated_by_write(4, top_vy as i64, 4), "write at the top voxel must invalidate");
}

#[test]
fn write_outside_raster_does_not_invalidate() {
    let world = OneColumn::new(5);
    let raster = build_surface_raster(&world, &WorldAddr::ROOT, [0.0, 0.0], [8, 8], 1.0, Lod::new(0));
    // Far outside the 8×8 raster footprint.
    assert!(!raster.is_invalidated_by_write(100, 5, 100));
    assert!(!raster.is_invalidated_by_write(-50, 5, 4));
    assert!(!raster.is_invalidated_by_write(4, 5, -50));
}

#[test]
fn write_above_top_invalidates() {
    let world = OneColumn::new(5);
    let raster = build_surface_raster(&world, &WorldAddr::ROOT, [0.0, 0.0], [8, 8], 1.0, Lod::new(0));
    // Adding a new voxel above the current top — the column's height
    // would change.
    assert!(raster.is_invalidated_by_write(4, 9, 4));
}
