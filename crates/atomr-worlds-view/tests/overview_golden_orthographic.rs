//! Phase 14e regression — orthographic-flat overview render must produce
//! a stable FNV-1a digest for a fixed seed / config / camera.
//!
//! The seed-driven macro state, the pyramid bake, and the per-pixel
//! sample are all deterministic; this test pins the resulting hash so
//! any silent algebra drift in the projection or palette gets caught.

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::{DefaultMacroGenerator, MacroConfig, MacroGenerator};
use atomr_worlds_view::{
    bake_world_summary, render_overview, OverviewCamera, OverviewProjection, RenderConfig,
};

const SEED: u64 = 0xA110_C001_DEAD_BEEF;

fn render() -> u64 {
    let g = DefaultMacroGenerator::new(MacroConfig { grid_level: 2, ..MacroConfig::default() });
    let s = g.generate(SEED, WorldShape::Sphere { radius_m: 6.371e6 });
    let p = bake_world_summary(WorldAddr::ROOT, &s, 3, 16);
    let cam = OverviewCamera {
        center: [0.0, 0.0],
        extent: 1.0,
        projection: OverviewProjection::OrthographicFlat,
        aspect: 1.0,
    };
    let cfg = RenderConfig { width: 64, height: 64, background: [0, 0, 0, 255], ..Default::default() };
    let fb = render_overview(&p, &cam, &cfg);
    fb.pixels_fnv1a()
}

#[test]
fn render_is_deterministic() {
    let a = render();
    let b = render();
    assert_eq!(a, b, "overview render must be byte-stable across runs");
}

#[test]
fn pinned_hash_matches() {
    // Pinned digest captured at Phase 14e implementation. Bump on
    // intentional changes to the projection, palette, or pyramid bake.
    let h = render();
    assert_eq!(h, EXPECTED, "overview-flat digest drifted: got 0x{h:016x}, expected 0x{EXPECTED:016x}");
}

// Re-pinned when the hydrology overlay landed: meso-scale relief refines
// the plate elevation and ocean / lake / river water bodies are baked into
// the world summary, both of which change the overview render.
// `render_is_deterministic` still guards against floating-point drift.
const EXPECTED: u64 = 0x0b65_4517_d4d1_946c;
