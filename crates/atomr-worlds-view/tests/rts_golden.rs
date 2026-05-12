//! Phase 14d gate: rendering a fixed-seed RTS scene through the
//! oblique-orthographic pipeline produces a byte-identical pixel
//! buffer across runs. Mirrors `deterministic_screenshot.rs` for the
//! Phase-2 3D path — the same contract holds for 14d.

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

/// Pinned FNV-1a hash. Bump deliberately when the math here intentionally
/// drifts; an unexpected change means a Phase 14d regression.
const PINNED_HASH: u64 = 0xf766_b129_72df_a325;

struct GoldenWorld {
    bricks: HashMap<IVec3, Arc<Brick>>,
}

impl GoldenWorld {
    fn new() -> Self {
        let mut bricks: HashMap<IVec3, Brick> = HashMap::new();
        let edge = BRICK_EDGE as i32;
        // Build an 8×8 grid of single-voxel columns with deterministic
        // heights driven by `(x * 13 + z * 7) % 5`.
        for vz in 0..8i32 {
            for vx in 0..8i32 {
                let h =
                    ((vx as u32).wrapping_mul(13).wrapping_add((vz as u32).wrapping_mul(7)) % 5) as i32 + 1;
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
                let mat = ((h % 3) as u16) + 1;
                bricks.entry(bc).or_insert_with(Brick::new).set(lc, Voxel::new(mat));
            }
        }
        Self { bricks: bricks.into_iter().map(|(k, v)| (k, Arc::new(v))).collect() }
    }
}

impl WorldQuery for GoldenWorld {
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

fn render() -> u64 {
    let world = GoldenWorld::new();
    let raster = build_surface_raster(&world, &WorldAddr::ROOT, [0.0, 0.0], [8, 8], 1.0, Lod::new(0));
    let cam = ObliqueCamera {
        center_xz: [4.0, 4.0],
        rotation_deg: 25.0,
        scale_m_per_px: 0.08,
        near: 0.1,
        far: 200.0,
        aspect: 1.0,
    };
    let decals = vec![
        Decal { world_xz_m: [2.0, 2.0], size_px: [3, 3], color: [255, 200, 0, 255], sprite: None },
        Decal { world_xz_m: [6.0, 6.0], size_px: [4, 4], color: [0, 200, 255, 200], sprite: None },
    ];
    let cfg = RenderConfig { width: 64, height: 64, ..Default::default() };
    let palette = MaterialPalette::default();
    let fb = render_rts(&raster, &decals, &cam, &palette, &cfg);
    fb.pixels_fnv1a()
}

#[test]
fn rts_render_is_deterministic_across_runs() {
    let h1 = render();
    let h2 = render();
    assert_eq!(h1, h2, "RTS render must be deterministic");
}

#[test]
fn rts_render_matches_pinned_hash() {
    let h = render();
    assert_eq!(h, PINNED_HASH, "RTS render hash drifted: got {h:#018x}, expected {PINNED_HASH:#018x}");
}
