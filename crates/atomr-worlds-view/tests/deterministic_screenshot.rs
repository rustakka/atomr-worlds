//! Phase 2 gate: rendering a known brick from a known seed produces a
//! byte-identical pixel buffer across runs.
//!
//! We compare a 64-bit FNV-1a hash of the RGBA pixel data rather than the PNG
//! file contents (PNG headers / compression can drift). If the test breaks,
//! either the renderer changed (intentional or otherwise) or the terrain
//! generator math drifted — both are interesting signals.
//!
//! Phase 13f update: the projection switched to reversed-z (near→1, far→0),
//! which moves the z-buffer compare from `<` to `>` and re-ranks fragments
//! whose original-projection z values were close — visible-surface decisions
//! at silhouettes can flip and the FNV hash necessarily changes. The pinned
//! constant below is the new reversed-z value; the run-to-run determinism
//! contract (the *real* gate) is unchanged.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{BrickGenerator, TerrainConfig, TerrainGenerator};
use atomr_worlds_view::mesh::greedy_mesh;
use atomr_worlds_view::{render_mesh, Camera, RenderConfig};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// Pinned FNV-1a hash for the known brick + camera + render config. Updated
/// in Phase 13f for the reversed-z projection, again for the
/// lighting+materials upgrade (10-entry palette), and once more for the
/// greedy-mesh winding fix: axis=1 (±Y) faces had `u × v = -Y`, so Bevy's
/// default `Cull::Back` was hiding the top + bottom of every voxel. Fix
/// (mesh.rs::meshing_axis) swapped u/v to `(2, 0)` for axis=1, which
/// changes the iteration / merge order on Y-faces too and thus the
/// rasterised RGB.
///
/// Bump this and document the reason whenever the renderer or
/// terrain-generator math intentionally changes; an unexpected drift is
/// the signal this test is supposed to catch.
const PINNED_HASH: u64 = 0xf127_b797_c699_fa45;

fn render_known_brick() -> u64 {
    let gen = TerrainGenerator::new(TerrainConfig::default());
    let brick = gen.generate_brick_legacy(SEED, IVec3::new(0, -1, 0));
    let cam = Camera::isometric_default(1.0);
    let cfg = RenderConfig { width: 128, height: 128, ..Default::default() };
    let mesh = greedy_mesh(&brick);
    let fb = render_mesh(&mesh, &cam, &cfg);
    fb.pixels_fnv1a()
}

#[test]
fn renders_are_deterministic_across_runs() {
    let h1 = render_known_brick();
    let h2 = render_known_brick();
    assert_eq!(h1, h2, "rendering the same brick twice should produce identical pixels");
}

#[test]
fn pinned_hash_matches_current_render() {
    let h = render_known_brick();
    assert_eq!(h, PINNED_HASH, "render hash drifted: got {h:#018x}, expected {PINNED_HASH:#018x}");
}

#[test]
fn nonempty_brick_produces_nonbackground_pixels() {
    let gen = TerrainGenerator::new(TerrainConfig::default());
    let brick = gen.generate_brick_legacy(SEED, IVec3::new(0, -1, 0));
    assert!(brick.nonempty_count > 0, "the chosen brick should have terrain");

    let cam = Camera::isometric_default(1.0);
    let cfg = RenderConfig { width: 64, height: 64, ..Default::default() };
    let fb = render_mesh(&greedy_mesh(&brick), &cam, &cfg);

    let bg = cfg.background;
    let non_bg = fb
        .pixels
        .chunks_exact(4)
        .filter(|p| !(p[0] == bg[0] && p[1] == bg[1] && p[2] == bg[2] && p[3] == bg[3]))
        .count();
    assert!(non_bg > 0, "rendered framebuffer should contain non-background pixels");
}
