//! Named preset constructors for [`WorldGenConfig`].
//!
//! `Vanilla` must produce bricks byte-equal to [`crate::TerrainGenerator`];
//! a regression snapshot test in `tests/vanilla_byte_equality.rs` enforces
//! this. `Advanced` and `Showcase` are populated as Steps 5–10 land the
//! per-paper strategies — until then they fall back to Vanilla.

use std::sync::Arc;

use super::biome_blend::{BufferTerrainInjected, NormalizedSparseConvolution};
use super::biome_matrix::{VoronoiCells, WhittakerDirect2D};
use super::config::WorldGenConfig;
use super::density::{FloatingIslandField, Hybrid2D3D};
use super::strata::LayeredGeology;
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

/// `Advanced` opts into the paper algorithms at moderate cost. Step 5
/// wires density / strata / biome matrix / biome blend; later steps fill
/// the remaining slots (caves, ore, structures, flora, sky light).
pub fn build_advanced() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.density = Arc::new(Hybrid2D3D::default());
    cfg.strata = Arc::new(LayeredGeology::default());
    cfg.biome_matrix = Arc::new(WhittakerDirect2D::default());
    cfg.biome_blend = Arc::new(NormalizedSparseConvolution::default());
    cfg
}

/// `Showcase` cranks every algorithm up for the visual demo. Step 5 wires
/// the most distinctive density (floating islands) and biome layouts
/// (Voronoi + buffer-terrain).
pub fn build_showcase() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.density = Arc::new(FloatingIslandField::default());
    cfg.strata = Arc::new(LayeredGeology::default());
    cfg.biome_matrix = Arc::new(VoronoiCells::default());
    cfg.biome_blend = Arc::new(BufferTerrainInjected::default());
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
