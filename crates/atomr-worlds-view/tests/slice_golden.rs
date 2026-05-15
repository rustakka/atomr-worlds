//! Phase 14c gate: rendering a known slice from a known world produces a
//! byte-identical pixel buffer.
//!
//! Pattern mirrors `deterministic_screenshot.rs` — pin a 64-bit FNV-1a hash
//! of the RGBA pixel data so an unexpected drift trips the test. Bump the
//! constant intentionally with a documented reason when the slice math or
//! the 2D rasterizer changes on purpose.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_view::scene::MaterialPalette;
use atomr_worlds_view::{
    build_slice_table, render_slice, SliceCamera, SliceConfig, SliceShading, WorldQuery,
};
use atomr_worlds_voxel::brick::{Brick, BRICK_EDGE};
use atomr_worlds_voxel::voxel::Voxel;

/// Pinned hash for the fixed (world, camera, config) tuple below, with
/// flat shading. Re-pinned when the slice raster was reoriented to match
/// the first-person view: `render_slice`'s px/py mapping now negates
/// `(world - center)` on both axes, so the rendered pixel layout shifts.
const PINNED_HASH: u64 = 0x6b01_51f0_2134_b695;

/// Pinned hash for the same fixture rendered with [`SliceShading::Hillshade`]
/// and a fixed light direction. Guards the relief-shading math.
const PINNED_HILLSHADE_HASH: u64 = 0x800e_195f_26f5_7685;

/// Fixed light direction for the hillshade golden — FROM sun INTO scene,
/// packed as `[world_x, world_z, world_y]`.
const HILLSHADE_LIGHT: [f32; 3] = [-0.5, -0.3, -0.8];

struct FixedWorld {
    bricks: HashMap<IVec3, Arc<Brick>>,
}

impl FixedWorld {
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

impl WorldQuery for FixedWorld {
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

fn build_fixture_world() -> FixedWorld {
    let mut w = FixedWorld::new();
    // A diagonal staircase across an 8×8 horizontal footprint with two
    // materials, plus a "thin feature" pillar at (5, *, 5) that is exactly
    // one voxel tall so it triggers the stipple path.
    for i in 0..8i64 {
        w.set(IVec3::new(i, 0, i), Voxel::new(1));
        w.set(IVec3::new(i, 1, i), Voxel::new(1));
        w.set(IVec3::new(i, 0, 7 - i), Voxel::new(2));
    }
    w.set(IVec3::new(5, 2, 5), Voxel::new(3));
    w
}

fn render_golden_with(shading: SliceShading) -> u64 {
    let world = build_fixture_world();
    let addr = WorldAddr::ROOT;
    let table = build_slice_table(&world, &addr, [-1, -1], [10, 10], 3, 3);
    let cam = SliceCamera {
        center_xz: [4.0, 4.0],
        z_band_top: 3,
        z_band_thickness: 3,
        half_height_m: 6.0,
        aspect: 1.0,
    };
    let cfg = SliceConfig {
        width: 64,
        height: 64,
        tile_px: 4,
        stipple_thin_features: true,
        roof_alpha: 0.25,
        background: [20, 20, 28, 255],
        shading,
        light_dir_xz_y: HILLSHADE_LIGHT,
    };
    let pal = MaterialPalette::default();
    let fb = render_slice(&table, &cam, &pal, &cfg);
    fb.pixels_fnv1a()
}

fn render_golden() -> u64 {
    render_golden_with(SliceShading::Flat)
}

fn render_hillshade_golden() -> u64 {
    render_golden_with(SliceShading::Hillshade { ambient: 0.35, relief_strength: 1.0 })
}

#[test]
fn slice_renders_deterministically_across_runs() {
    let h1 = render_golden();
    let h2 = render_golden();
    assert_eq!(h1, h2, "slice render must be deterministic");
    let g1 = render_hillshade_golden();
    let g2 = render_hillshade_golden();
    assert_eq!(g1, g2, "hillshade slice render must be deterministic");
}

#[test]
fn slice_golden_pinned_hash_matches() {
    let h = render_golden();
    assert_eq!(h, PINNED_HASH, "slice render hash drifted: got {h:#018x}, expected {PINNED_HASH:#018x}");
}

#[test]
fn slice_hillshade_golden_pinned_hash_matches() {
    let h = render_hillshade_golden();
    assert_eq!(
        h, PINNED_HILLSHADE_HASH,
        "hillshade slice render hash drifted: got {h:#018x}, expected {PINNED_HILLSHADE_HASH:#018x}"
    );
}

#[test]
fn hillshade_differs_from_flat() {
    // The fixture has a diagonal staircase, so relief shading must change
    // at least some pixels — a sanity check that the branch is wired up.
    assert_ne!(
        render_golden(),
        render_hillshade_golden(),
        "hillshade shading should change the rendered pixels vs flat fill"
    );
}
