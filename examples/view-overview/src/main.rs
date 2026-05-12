//! `view-overview` — Phase 14e demo.
//!
//! Bakes a [`WorldSummaryPyramid`] from the Phase 13c macro state for an
//! Earth-class sphere world and renders three regional overview frames
//! to `/tmp/view-overview-{flat,equirect,sphere}.png`.
//!
//! Run with `cargo run -p view-overview`. Designed to complete in well
//! under five seconds on a CI single-core (grid level 3 macro state,
//! 4-level pyramid, 256×256 frames). Pyramid bake at grid level 4 takes
//! ~10 s on a single-core release build — too slow for CI; level 3 cuts
//! that to ~2 s with a barely-visible coarsening of the underlying biome
//! field at 256-px output resolution.

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::{DefaultMacroGenerator, MacroConfig, MacroGenerator};
use atomr_worlds_view::{
    bake_world_summary, render_overview, OverviewCamera, OverviewProjection, RenderConfig,
};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const EARTH_R: f64 = 6.371e6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let generator = DefaultMacroGenerator::new(MacroConfig { grid_level: 3, ..MacroConfig::default() });
    let shape = WorldShape::Sphere { radius_m: EARTH_R };
    let state = generator.generate(SEED, shape);

    let pyramid = bake_world_summary(WorldAddr::ROOT, &state, 4, 64);
    println!(
        "baked pyramid: levels={}, tiles={}, macro_digest=0x{:016x}",
        pyramid.levels,
        pyramid.tiles.len(),
        pyramid.macro_digest
    );

    let cfg = RenderConfig {
        width: 256,
        height: 256,
        background: [10, 14, 24, 255],
        light_dir: [0.5, 0.8, 0.3],
        ambient: 0.25,
    };

    let frames = [
        ("flat", OverviewProjection::OrthographicFlat),
        ("equirect", OverviewProjection::Equirectangular),
        ("sphere", OverviewProjection::OrthographicSphere),
    ];
    for (label, projection) in frames {
        let cam = OverviewCamera { center: [0.0, 0.0], extent: 1.0, projection, aspect: 1.0 };
        let fb = render_overview(&pyramid, &cam, &cfg);
        let path = format!("/tmp/view-overview-{label}.png");
        fb.write_png(&path)?;
        println!("wrote {path} digest=0x{:016x}", fb.pixels_fnv1a());
    }
    Ok(())
}
