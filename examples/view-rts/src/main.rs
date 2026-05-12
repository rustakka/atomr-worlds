//! `view-rts` — Phase 14d demo.
//!
//! Builds a tiny stub world (a 16×16 surface with a smooth hump in the
//! middle), derives a [`SurfaceRaster`] from it, renders the scene
//! through the oblique-orthographic camera, drops three decals (unit
//! marker, build site, selection ring), and writes a PNG to
//! `/tmp/view-rts-00.png`.
//!
//! Run with `cargo run -p view-rts`. The render fits easily inside one
//! CPU core in < 100 ms.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_view::{
    build_surface_raster, render_rts, scene::MaterialPalette, Decal, ObliqueCamera, RenderConfig, WorldQuery,
};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

/// Stub WorldQuery: holds a `HashMap<IVec3, Brick>` and answers
/// `brick()` from it. Mirrors the test stub used by the unit-tests but
/// scaled up to a 16×16 column-grid populated by a hump function.
struct StubWorld {
    bricks: HashMap<IVec3, Arc<Brick>>,
}

impl StubWorld {
    fn build() -> Self {
        let mut bricks: HashMap<IVec3, Brick> = HashMap::new();
        let edge = BRICK_EDGE as i32;
        let cx = 8.0_f32;
        let cz = 8.0_f32;
        // 16×16 columns. A radial hump peaks at column (8, 8) with
        // height 6, falls off linearly to 0 at radius 10.
        for vz in 0..16i32 {
            for vx in 0..16i32 {
                let dx = vx as f32 - cx;
                let dz = vz as f32 - cz;
                let r = (dx * dx + dz * dz).sqrt();
                let h = (6.0 - r * 0.7).round().max(0.0) as i32;
                // Place a single voxel at the top of this column.
                let mat: u16 = if h >= 4 {
                    1
                } else if h >= 2 {
                    2
                } else {
                    3
                };
                if h > 0 {
                    let vy = h;
                    let bc = IVec3::new(
                        vx.div_euclid(edge) as i64,
                        vy.div_euclid(edge) as i64,
                        vz.div_euclid(edge) as i64,
                    );
                    let lc = IVec3::new(
                        vx.rem_euclid(edge) as i64,
                        vy.rem_euclid(edge) as i64,
                        vz.rem_euclid(edge) as i64,
                    );
                    bricks.entry(bc).or_default().set(lc, Voxel::new(mat));
                }
            }
        }
        StubWorld { bricks: bricks.into_iter().map(|(k, v)| (k, Arc::new(v))).collect() }
    }
}

impl WorldQuery for StubWorld {
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

fn main() {
    let world = StubWorld::build();
    let addr = WorldAddr::ROOT;
    let raster = build_surface_raster(&world, &addr, [0.0, 0.0], [16, 16], 1.0, Lod::new(0));

    // Three decals: a unit marker, a build site, a selection ring.
    let decals = vec![
        // Unit at world (4, 4): solid yellow square.
        Decal { world_xz_m: [4.5, 4.5], size_px: [4, 4], color: [255, 220, 60, 255], sprite: None },
        // Build site at (8, 8): semi-transparent blue.
        Decal { world_xz_m: [8.5, 8.5], size_px: [6, 6], color: [80, 120, 255, 180], sprite: None },
        // Selection ring at (12, 12): translucent green.
        Decal { world_xz_m: [12.5, 12.5], size_px: [5, 5], color: [120, 255, 120, 160], sprite: None },
    ];

    let cam = ObliqueCamera {
        center_xz: [8.0, 8.0],
        rotation_deg: 30.0,
        scale_m_per_px: 0.10,
        near: 0.1,
        far: 200.0,
        aspect: 1.0,
    };
    let cfg = RenderConfig { width: 128, height: 128, ..Default::default() };
    let palette = MaterialPalette::default();
    let fb = render_rts(&raster, &decals, &cam, &palette, &cfg);
    fb.write_png("/tmp/view-rts-00.png").expect("write png");
    println!("view-rts: wrote /tmp/view-rts-00.png — digest {:#018x}", fb.pixels_fnv1a());
}
