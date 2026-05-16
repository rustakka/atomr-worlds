//! Named preset constructors for [`WorldGenConfig`].
//!
//! `Vanilla` must produce bricks byte-equal to [`crate::TerrainGenerator`];
//! a regression snapshot test in `tests/vanilla_byte_equality.rs` enforces
//! this. `Advanced` and `Showcase` are populated as Steps 5–10 land the
//! per-paper strategies — until then they fall back to Vanilla.

use std::sync::Arc;

use super::config::WorldGenConfig;
use super::erosion::{DropletHydraulic, MacroRiverOnly};
use super::fluid::{CellularAutomataFlow, LatticeBoltzmannD3Q19};
use super::ore::BiasedRandomWalk;
use super::strategies::*;
use super::vanilla::MonolithicTerrainPass;

pub fn build_vanilla() -> WorldGenConfig {
    let monolith = Arc::new(MonolithicTerrainPass::default());
    WorldGenConfig {
        density: monolith.clone(),
        strata: monolith,
        caves: Arc::new(NoneCaves),
        ore: Arc::new(NoneOre),
        // Vanilla keeps the river carve inside MonolithicTerrainPass; the
        // erosion slot stays a no-op so the byte-equality test is unaffected.
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

/// `Advanced` opts into every paper algorithm at moderate cost. Step 7
/// wires ore + erosion + fluid; subsequent steps fill out the remaining
/// slots.
pub fn build_advanced() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.ore = Arc::new(BiasedRandomWalk::default());
    cfg.erosion = Arc::new(MacroRiverOnly);
    cfg.fluid = Arc::new(CellularAutomataFlow::default());
    cfg
}

/// `Showcase` cranks every algorithm up for the visual demo. Step 7
/// wires the heavyweight ore + erosion + fluid impls; subsequent steps
/// fill out the remaining slots.
pub fn build_showcase() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.ore = Arc::new(BiasedRandomWalk::default());
    cfg.erosion = Arc::new(DropletHydraulic::default());
    cfg.fluid = Arc::new(LatticeBoltzmannD3Q19::default());
    cfg
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
