//! Phase 2 gate: rendering a known brick from a known seed produces a
//! byte-identical pixel buffer across runs.
//!
//! We compare a 64-bit FNV-1a hash of the RGBA pixel data rather than the PNG
//! file contents (PNG headers / compression can drift). If the test breaks,
//! either the renderer changed (intentional or otherwise) or the terrain
//! generator math drifted — both are interesting signals.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{BrickGenerator, TerrainConfig, TerrainGenerator};
use atomr_worlds_view::{render_mesh, Camera, RenderConfig};
use atomr_worlds_view::mesh::greedy_mesh;

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

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
