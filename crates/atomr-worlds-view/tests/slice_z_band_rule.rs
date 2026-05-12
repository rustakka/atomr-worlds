//! Phase 14c gate: codifies the "+Y up, scanning down" z-band rule.
//!
//! Each test pins one corner of the rule from `slice_index.rs`'s module
//! rustdoc so any future refactor that breaks the convention shows up here
//! before it shows up in renderer output.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_view::{build_slice_table, WorldQuery};
use atomr_worlds_voxel::brick::{Brick, BRICK_EDGE};
use atomr_worlds_voxel::voxel::Voxel;

struct StubWorld {
    bricks: HashMap<IVec3, Arc<Brick>>,
}

impl StubWorld {
    fn new() -> Self {
        Self { bricks: HashMap::new() }
    }
    fn set(&mut self, w: IVec3, v: Voxel) {
        let edge: i64 = BRICK_EDGE as i64;
        let bc = IVec3::new(w.x.div_euclid(edge), w.y.div_euclid(edge), w.z.div_euclid(edge));
        let lc = IVec3::new(w.x.rem_euclid(edge), w.y.rem_euclid(edge), w.z.rem_euclid(edge));
        let entry = self.bricks.entry(bc).or_insert_with(|| Arc::new(Brick::new()));
        let brick = Arc::make_mut(entry);
        brick.set(lc, v);
    }
}

impl WorldQuery for StubWorld {
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
fn column_entirely_below_band_reports_empty() {
    // band scans y = 7, 6, 5 (top = 8, thickness = 3). A voxel at y = 1
    // is well below the band → column is empty, top_z snaps to the band
    // floor (`z_band_top - thickness`).
    let mut w = StubWorld::new();
    w.set(IVec3::new(0, 1, 0), Voxel::new(1));
    let t = build_slice_table(&w, &WorldAddr::ROOT, [0, 0], [1, 1], 8, 3);
    let c = t.column(0, 0).unwrap();
    assert_eq!(c.top_voxel, Voxel::EMPTY);
    assert_eq!(c.top_z, 8 - 3);
    assert_eq!(c.thickness_above_floor, 0);
}

#[test]
fn column_with_voxel_at_exact_top_of_band() {
    // The "top" of the band is `z_band_top - 1` (we scan downward starting
    // from one cell below z_band_top). So a voxel placed at world_y =
    // z_band_top - 1 is the very first cell the scan visits.
    let mut w = StubWorld::new();
    let band_top: i32 = 10;
    let band_thickness: u8 = 3;
    w.set(IVec3::new(0, band_top as i64 - 1, 0), Voxel::new(7));
    let t = build_slice_table(&w, &WorldAddr::ROOT, [0, 0], [1, 1], band_top, band_thickness);
    let c = t.column(0, 0).unwrap();
    assert_eq!(c.top_voxel, Voxel::new(7));
    assert_eq!(c.top_z, band_top - 1);
}

#[test]
fn column_with_voxel_mid_band() {
    // band scans y = 9, 8, 7 (top = 10, thickness = 3). Voxel at y = 8
    // and another at y = 7 → top_voxel is the higher one, and
    // `thickness_above_floor == 1` (one solid cell directly below the top).
    let mut w = StubWorld::new();
    let band_top: i32 = 10;
    let band_thickness: u8 = 3;
    w.set(IVec3::new(0, 8, 0), Voxel::new(5));
    w.set(IVec3::new(0, 7, 0), Voxel::new(5));
    let t = build_slice_table(&w, &WorldAddr::ROOT, [0, 0], [1, 1], band_top, band_thickness);
    let c = t.column(0, 0).unwrap();
    assert_eq!(c.top_voxel, Voxel::new(5));
    assert_eq!(c.top_z, 8);
    assert_eq!(c.thickness_above_floor, 1, "two solids in a 3-band → above-floor count = 1");
}
