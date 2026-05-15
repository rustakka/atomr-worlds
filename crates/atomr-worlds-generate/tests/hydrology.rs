//! Integration test for the hydrology overlay (ocean / lake / river).
//!
//! Exercises the full macro pre-sim at the default `grid_level` and
//! asserts the `WaterField` is well-formed, populated, and deterministic.

use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::ClimateConfig;
use atomr_worlds_generate::{water_kind, DefaultMacroGenerator, MacroConfig, MacroGenerator};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const EARTH: WorldShape = WorldShape::Sphere { radius_m: 6.371e6 };

#[test]
fn default_world_has_oceans_lakes_and_rivers() {
    let g = DefaultMacroGenerator::new(MacroConfig::default());
    let s = g.generate(SEED, EARTH);
    let n = s.grid.face_count();
    let w = &s.water;

    assert_eq!(w.water_kind.len(), n);
    assert_eq!(w.water_surface_m.len(), n);
    assert_eq!(w.flow_dir.len(), n);
    assert_eq!(w.flow_accum.len(), n);

    let count = |k: u8| w.water_kind.iter().filter(|&&v| v == k).count();
    let oceans = count(water_kind::OCEAN);
    let lakes = count(water_kind::LAKE);
    let rivers = count(water_kind::RIVER);

    assert!(oceans > 0, "default world should have ocean faces");
    assert!(lakes > 0, "default world should have at least one lake");
    assert!(rivers > 0, "default world should have at least one river");
    assert!(
        oceans + lakes + rivers < n,
        "some dry land must remain ({oceans} ocean + {lakes} lake + {rivers} river of {n})",
    );
}

#[test]
fn water_surface_and_flow_invariants_hold() {
    let g = DefaultMacroGenerator::new(MacroConfig::default());
    let s = g.generate(SEED, EARTH);
    let w = &s.water;
    for f in 0..s.grid.face_count() {
        match w.water_kind[f] {
            water_kind::NONE => {
                assert_eq!(w.water_surface_m[f], f32::NEG_INFINITY);
            }
            water_kind::OCEAN => {
                assert_eq!(w.water_surface_m[f], w.sea_level_m);
            }
            water_kind::LAKE => {
                assert!(w.water_surface_m[f].is_finite());
                // A lake sits above its own ground.
                assert!(w.water_surface_m[f] > s.elevation.elev_m[f]);
            }
            water_kind::RIVER => {
                assert!(w.water_surface_m[f].is_finite());
            }
            other => panic!("unexpected water_kind {other}"),
        }
        assert!(w.flow_accum[f] >= 0.0, "flow_accum must be non-negative");
    }
}

/// Phase 18 follow-up: hydrology now seeds humidity at lake / river
/// faces and re-runs the diffusion + biome classification. With the
/// feedback enabled (default), lake & river faces sit at full humidity;
/// with `hydrology_feedback_iters = 0` they fall back to whatever the
/// initial ocean-only diffusion produced for that face.
#[test]
fn lake_and_river_faces_are_humid_with_feedback_enabled() {
    let g = DefaultMacroGenerator::new(MacroConfig::default());
    let s = g.generate(SEED, EARTH);
    let w = &s.water;

    let mut min_freshwater_humidity = f32::INFINITY;
    let mut freshwater_count = 0usize;
    for f in 0..s.grid.face_count() {
        if w.water_kind[f] == water_kind::LAKE || w.water_kind[f] == water_kind::RIVER {
            freshwater_count += 1;
            min_freshwater_humidity = min_freshwater_humidity.min(s.climate.humidity[f]);
        }
    }
    assert!(freshwater_count > 0, "test precondition: default world has freshwater");
    assert!(
        min_freshwater_humidity >= 1.0,
        "every freshwater face should hit the feedback seed (1.0), got min {min_freshwater_humidity}"
    );
}

/// Same world with feedback disabled: at least one freshwater face must
/// fall below the seed, proving the feedback pass actually moves the
/// field rather than being a digest-only change.
#[test]
fn disabling_feedback_drops_some_freshwater_humidity() {
    let mut cfg = MacroConfig::default();
    cfg.climate = ClimateConfig { hydrology_feedback_iters: 0, ..ClimateConfig::default() };
    let g = DefaultMacroGenerator::new(cfg);
    let s = g.generate(SEED, EARTH);

    let mut min_freshwater_humidity = f32::INFINITY;
    let mut freshwater_count = 0usize;
    for f in 0..s.water.water_kind.len() {
        let k = s.water.water_kind[f];
        if k == water_kind::LAKE || k == water_kind::RIVER {
            freshwater_count += 1;
            min_freshwater_humidity = min_freshwater_humidity.min(s.climate.humidity[f]);
        }
    }
    assert!(freshwater_count > 0);
    assert!(
        min_freshwater_humidity < 1.0,
        "with feedback off some freshwater face should sit below the seed, got {min_freshwater_humidity}"
    );
}

/// Biome classification consumes the post-feedback humidity, so a face
/// adjacent to a lake / river can land in a wetter biome than the same
/// face would have absent the feedback.
#[test]
fn feedback_can_change_biomes_around_freshwater() {
    let mut cfg_off = MacroConfig::default();
    cfg_off.climate = ClimateConfig { hydrology_feedback_iters: 0, ..ClimateConfig::default() };
    let g_off = DefaultMacroGenerator::new(cfg_off);
    let s_off = g_off.generate(SEED, EARTH);

    let g_on = DefaultMacroGenerator::new(MacroConfig::default());
    let s_on = g_on.generate(SEED, EARTH);

    let mut diff_count = 0usize;
    for f in 0..s_on.biomes.biome_id.len() {
        if s_on.biomes.biome_id[f] != s_off.biomes.biome_id[f] {
            diff_count += 1;
        }
    }
    assert!(
        diff_count > 0,
        "feedback should flip at least some biome classifications relative to the no-feedback baseline"
    );
}

#[test]
fn hydrology_is_deterministic_and_seed_sensitive() {
    let g = DefaultMacroGenerator::new(MacroConfig::default());
    let a = g.generate(SEED, EARTH);
    let b = g.generate(SEED, EARTH);
    assert_eq!(a.digest, b.digest, "same seed must produce the same digest");
    assert_eq!(a.water.water_kind, b.water.water_kind);
    assert_eq!(a.water.flow_dir, b.water.flow_dir);

    let c = g.generate(SEED ^ 0xFFFF_FFFF, EARTH);
    assert_ne!(a.digest, c.digest, "a different seed must change the digest");
}
