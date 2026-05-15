//! `view-slice` — Phase 14c demo.
//!
//! Builds a tiny stub `WorldQuery` (a HashMap of bricks with a hand-shaped
//! "village + canyon" scene) and renders three Dwarf-Fortress-style
//! horizontal slices at consecutive z-bands. Output goes to
//! `/tmp/view-slice-NN.png` (NN = 00, 01, 02). Each frame uses the same
//! horizontal footprint but a different `z_band_top`, demonstrating how
//! cycling vertical levels reveals the canyon floor, mid-cliff, and
//! plateau in turn.
//!
//! Run with `cargo run -p view-slice`. The renderer is deterministic, so
//! re-runs produce byte-identical PNGs.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_view::scene::MaterialPalette;
use atomr_worlds_view::{build_slice_table, render_slice, SliceCamera, SliceConfig, WorldQuery};
use atomr_worlds_voxel::brick::{Brick, BRICK_EDGE};
use atomr_worlds_voxel::voxel::Voxel;

struct DemoWorld {
    bricks: HashMap<IVec3, Arc<Brick>>,
}

impl DemoWorld {
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

impl WorldQuery for DemoWorld {
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

fn build_demo_world() -> DemoWorld {
    let mut w = DemoWorld::new();
    // Plateau: solid stone from y = 0..=2 across a 16×16 footprint.
    for z in 0..16i64 {
        for x in 0..16i64 {
            for y in 0..=2 {
                w.set(IVec3::new(x, y, z), Voxel::new(1));
            }
        }
    }
    // Canyon carved through middle (z = 7, 8) down to y = 0.
    for z in 7..=8i64 {
        for x in 0..16i64 {
            for y in 1..=2 {
                w.set(IVec3::new(x, y, z), Voxel::EMPTY);
            }
        }
    }
    // Floor of canyon is dirt.
    for z in 7..=8i64 {
        for x in 0..16i64 {
            w.set(IVec3::new(x, 0, z), Voxel::new(2));
        }
    }
    // A row of "huts" (single-voxel features) at y = 3 across the plateau.
    for x in (2..14i64).step_by(3) {
        w.set(IVec3::new(x, 3, 3), Voxel::new(3));
        w.set(IVec3::new(x, 3, 12), Voxel::new(3));
    }
    w
}

fn main() {
    let world = build_demo_world();
    let addr = WorldAddr::ROOT;
    let palette = MaterialPalette::default();
    let cfg = SliceConfig {
        width: 128,
        height: 128,
        tile_px: 6,
        stipple_thin_features: true,
        roof_alpha: 0.25,
        background: [20, 20, 28, 255],
        ..SliceConfig::default()
    };

    // Cycle three z-bands. Each band is 3 voxels thick: 2 m of "open air"
    // + 1 m of "roof" by convention. `z_band_top` decreases by one each
    // frame so we see plateau → mid-canyon → canyon floor in sequence.
    let bands: [i32; 3] = [4, 3, 2];
    for (i, &z_band_top) in bands.iter().enumerate() {
        let cam = SliceCamera {
            center_xz: [8.0, 8.0],
            z_band_top,
            z_band_thickness: 3,
            half_height_m: 12.0,
            aspect: 1.0,
        };
        let table = build_slice_table(&world, &addr, [0, 0], [16, 16], z_band_top, 3);
        let fb = render_slice(&table, &cam, &palette, &cfg);
        let path = format!("/tmp/view-slice-{i:02}.png");
        fb.write_png(&path).expect("write png");
        println!("  frame {i:02}: z_band_top={z_band_top} digest={:#018x}", fb.pixels_fnv1a());
    }
    println!("view-slice: done — 3 frames written to /tmp/view-slice-*.png");
}
