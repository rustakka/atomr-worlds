//! Phase 14e regression — orthographic-sphere (globe-as-disk) overview
//! must produce a stable FNV-1a digest for a fixed seed / config /
//! camera. Pixels outside the disc must be left as background.

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
        projection: OverviewProjection::OrthographicSphere,
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
    assert_eq!(a, b, "orthographic-sphere render must be byte-stable across runs");
}

#[test]
fn pinned_hash_matches() {
    let h = render();
    assert_eq!(
        h, EXPECTED,
        "overview-orthographic-sphere digest drifted: got 0x{h:016x}, expected 0x{EXPECTED:016x}"
    );
}

// Re-pinned when the hydrology overlay landed: meso-scale relief refines
// the plate elevation and ocean / lake / river water bodies are baked into
// the world summary, both of which change the overview render.
// `render_is_deterministic` still guards against floating-point drift.
const EXPECTED: u64 = 0x15cc_cc45_22f8_97aa;

#[test]
fn corners_remain_background() {
    let g = DefaultMacroGenerator::new(MacroConfig { grid_level: 2, ..MacroConfig::default() });
    let s = g.generate(SEED, WorldShape::Sphere { radius_m: 6.371e6 });
    let p = bake_world_summary(WorldAddr::ROOT, &s, 3, 16);
    let cam = OverviewCamera {
        center: [0.0, 0.0],
        extent: 1.0,
        projection: OverviewProjection::OrthographicSphere,
        aspect: 1.0,
    };
    let cfg = RenderConfig { width: 64, height: 64, background: [11, 22, 33, 255], ..Default::default() };
    let fb = render_overview(&p, &cam, &cfg);
    // Top-left corner is well outside the inscribed unit disc.
    let pi = (0_usize * fb.width as usize + 0) * 4;
    assert_eq!(&fb.pixels[pi..pi + 4], &[11, 22, 33, 255]);
}
