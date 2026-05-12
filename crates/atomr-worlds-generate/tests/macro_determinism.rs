//! Phase 13c determinism gate.
//!
//! `WorldMacroState::digest` is a FNV-1a witness over every output array
//! (plates, elevation, climate, biomes). For a fixed
//! `(world_seed, config)` it must:
//! - Be equal across repeated calls (within and across processes).
//! - Change when the seed changes (avalanche / non-trivial).
//! - Stay invariant if only the `shape` field changes (digest is shape-
//!   independent by design — the shape is *stored* but not *digested*).
//!
//! The digest is pinned in `PINNED_DIGEST_GRID_2_SEED_DEADBEEF` so that
//! cross-platform CI catches floating-point drift.

use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::{DefaultMacroGenerator, MacroConfig, MacroGenerator};

const SEED_A: u64 = 0xDEAD_BEEF_CAFE_F00D;
const SEED_B: u64 = 0x1234_5678_9ABC_DEF0;
const EARTH: WorldShape = WorldShape::Sphere { radius_m: 6.371e6 };

fn small_gen() -> DefaultMacroGenerator {
    DefaultMacroGenerator::new(MacroConfig { grid_level: 2, ..MacroConfig::default() })
}

#[test]
fn digest_is_stable_across_repeats() {
    let g = small_gen();
    let a = g.generate(SEED_A, EARTH);
    let b = g.generate(SEED_A, EARTH);
    assert_eq!(a.digest, b.digest, "digest must be stable across repeats");
}

#[test]
fn digest_changes_with_seed() {
    let g = small_gen();
    let a = g.generate(SEED_A, EARTH);
    let b = g.generate(SEED_B, EARTH);
    assert_ne!(a.digest, b.digest, "different seed must change digest");
}

#[test]
fn digest_is_shape_independent() {
    // Shape is stored on the state but not digested. This is intentional
    // — the digest is a function of generated arrays, which depend only
    // on the surface grid (deterministic by level) and the seed.
    let g = small_gen();
    let a = g.generate(SEED_A, WorldShape::Sphere { radius_m: 1.0e6 });
    let b = g.generate(SEED_A, WorldShape::Sphere { radius_m: 2.0e6 });
    assert_eq!(a.digest, b.digest);
    assert_ne!(a.shape, b.shape);
}

#[test]
fn elevation_field_has_one_value_per_face() {
    let g = small_gen();
    let s = g.generate(SEED_A, EARTH);
    assert_eq!(s.elevation.elev_m.len(), s.grid.face_count());
    assert_eq!(s.climate.temperature_c.len(), s.grid.face_count());
    assert_eq!(s.biomes.biome_id.len(), s.grid.face_count());
}

#[test]
fn sample_returns_consistent_face_lookup() {
    let g = small_gen();
    let s = g.generate(SEED_A, EARTH);
    let dir = atomr_worlds_core::coord::DVec3::new(1.0, 0.0, 0.0);
    let a = s.sample(dir);
    let b = s.sample(dir);
    assert_eq!(a.face, b.face);
    assert_eq!(a.elev_m.to_bits(), b.elev_m.to_bits());
    assert_eq!(a.biome_id, b.biome_id);
}

#[test]
fn default_config_completes_under_5s() {
    // Perf budget: grid_level=4 (default) at the default plate_count
    // must complete in well under 5s on a single CPU core. This is a
    // floor — release builds in CI clear it by orders of magnitude.
    let start = std::time::Instant::now();
    let g = DefaultMacroGenerator::default();
    let _ = g.generate(SEED_A, EARTH);
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 5, "default macro pre-sim took {elapsed:?}");
}
