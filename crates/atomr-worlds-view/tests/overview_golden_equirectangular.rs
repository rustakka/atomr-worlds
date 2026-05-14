//! Phase 14e regression — equirectangular overview render must produce
//! a stable FNV-1a digest for a fixed seed / config / camera.

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
        projection: OverviewProjection::Equirectangular,
        aspect: 2.0,
    };
    let cfg = RenderConfig { width: 128, height: 64, background: [0, 0, 0, 255], ..Default::default() };
    let fb = render_overview(&p, &cam, &cfg);
    fb.pixels_fnv1a()
}

#[test]
fn render_is_deterministic() {
    let a = render();
    let b = render();
    assert_eq!(a, b, "equirectangular render must be byte-stable across runs");
}

#[test]
fn pinned_hash_matches() {
    let h = render();
    assert_eq!(
        h, EXPECTED,
        "overview-equirectangular digest drifted: got 0x{h:016x}, expected 0x{EXPECTED:016x}"
    );
}

// Re-pinned when the hydrology overlay landed: meso-scale relief now
// refines the previously piecewise-flat plate elevation, and ocean / lake
// / river water bodies are baked into the world summary — both change the
// equirectangular overview render. `render_is_deterministic` still guards
// against floating-point drift.
const EXPECTED: u64 = 0x8ac3_ecfc_2836_7db5;
