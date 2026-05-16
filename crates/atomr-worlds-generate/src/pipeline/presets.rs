//! Named preset constructors for [`WorldGenConfig`].
//!
//! `Vanilla` must produce bricks byte-equal to [`crate::TerrainGenerator`];
//! a regression snapshot test in `tests/vanilla_byte_equality.rs` enforces
//! this. `Advanced` and `Showcase` are populated as Steps 5–10 land the
//! per-paper strategies — until then they fall back to Vanilla.

use std::sync::Arc;

use super::config::WorldGenConfig;
use super::strategies::*;
use super::vanilla::MonolithicTerrainPass;

pub fn build_vanilla() -> WorldGenConfig {
    let monolith = Arc::new(MonolithicTerrainPass::default());
    WorldGenConfig {
        density: monolith.clone(),
        strata: monolith,
        caves: Arc::new(NoneCaves),
        ore: Arc::new(NoneOre),
        erosion: Arc::new(NoneErosion),
        fluid: Arc::new(NoneFluid),
        structures: Arc::new(NoneStructures),
        flora: Arc::new(NoneFlora),
        placement: Arc::new(NonePlacement),
        biome_matrix: Arc::new(NoneBiomeMatrix),
        biome_blend: Arc::new(NoneBiomeBlend),
        sky_light: Arc::new(NoneSkyLight),
        feature_seeder: Arc::new(EmptySeeder),
    }
}

/// `Advanced` opts into every paper algorithm at moderate cost. Until
/// Steps 5–10 ship the real strategies, this is a clone of Vanilla; the
/// `apply_worldgen_strategy_by_name` registry lets the harness DSL swap
/// individual slots in any preset.
pub fn build_advanced() -> WorldGenConfig {
    build_vanilla()
}

/// `Showcase` cranks every algorithm up for the visual demo. Same caveat
/// as `Advanced` — slot-by-slot upgrades land with subsequent steps.
pub fn build_showcase() -> WorldGenConfig {
    build_vanilla()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vanilla_uses_monolith_for_density_stage() {
        let cfg = build_vanilla();
        assert_eq!(cfg.density.id(), "MonolithicTerrainPass");
    }

    #[test]
    fn vanilla_no_op_slots_are_none() {
        let cfg = build_vanilla();
        assert!(cfg.caves.id().starts_with("none::"));
        assert!(cfg.ore.id().starts_with("none::"));
        assert!(cfg.fluid.id().starts_with("none::"));
        assert!(cfg.flora.id().starts_with("none::"));
        assert!(cfg.sky_light.id().starts_with("none::"));
    }
}
