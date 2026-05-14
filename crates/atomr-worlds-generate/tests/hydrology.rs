//! Integration test for the hydrology overlay (ocean / lake / river).
//!
//! Exercises the full macro pre-sim at the default `grid_level` and
//! asserts the `WaterField` is well-formed, populated, and deterministic.

use atomr_worlds_core::shape::WorldShape;
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
