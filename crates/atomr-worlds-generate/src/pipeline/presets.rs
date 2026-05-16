//! Named preset constructors for [`WorldGenConfig`].
//!
//! `Vanilla` must produce bricks byte-equal to [`crate::TerrainGenerator`];
//! a regression snapshot test in `tests/vanilla_byte_equality.rs` enforces
//! this. `Advanced` and `Showcase` are populated as Steps 5–10 land the
//! per-paper strategies — until then they fall back to Vanilla.

use std::sync::Arc;

use super::biome_blend::{BufferTerrainInjected, NormalizedSparseConvolution};
use super::biome_matrix::{VoronoiCells, WhittakerDirect2D};
use super::caves::{CellularAutomata3D, IsosurfaceIntersection};
use super::config::WorldGenConfig;
use super::density::{FloatingIslandField, Hybrid2D3D};
use super::erosion::{DropletHydraulic, MacroRiverOnly};
use super::feature_seeder::{ColumnAnchorSeeder, SeederConfig};
use super::flora::LSystemTrees;
use super::fluid::{CellularAutomataFlow, LatticeBoltzmannD3Q19};
use super::light::VerticalCastWithDiffusion;
use super::ore::BiasedRandomWalk;
use super::placement::PoissonDiskBridson;
use super::strata::LayeredGeology;
use super::strategies::*;
use super::structures::{Jigsaw, WaveFunctionCollapse};
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

/// `Advanced` opts into every paper algorithm at moderate cost. Density
/// becomes hybrid 2D/3D, strata is layered geology, biome matrix is
/// direct 2D Whittaker with sparse-convolution blend, caves use 3-D CA,
/// ore is biased random walk, erosion is macro-river only, fluid is CA
/// flow, placement is Bridson Poisson-disk, and flora is L-system
/// trees. The column-anchor seeder feeds the cross-brick stages.
pub fn build_advanced() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.density = Arc::new(Hybrid2D3D::default());
    cfg.strata = Arc::new(LayeredGeology::default());
    cfg.biome_matrix = Arc::new(WhittakerDirect2D::default());
    cfg.biome_blend = Arc::new(NormalizedSparseConvolution::default());
    cfg.caves = Arc::new(CellularAutomata3D::default());
    cfg.feature_seeder = Arc::new(ColumnAnchorSeeder::new(SeederConfig {
        worm_density: 1.0,
        ore_density: 1.0,
        structure_density: 0.25,
        flora_tree_density: 1.0,
        ..Default::default()
    }));
    cfg.ore = Arc::new(BiasedRandomWalk::default());
    cfg.erosion = Arc::new(MacroRiverOnly);
    cfg.fluid = Arc::new(CellularAutomataFlow::default());
    cfg.placement = Arc::new(PoissonDiskBridson::default());
    cfg.flora = Arc::new(LSystemTrees::default());
    cfg.structures = Arc::new(Jigsaw::default());
    cfg.sky_light = Arc::new(VerticalCastWithDiffusion::default());
    cfg
}

/// `Showcase` cranks every algorithm up for the visual demo. Floating
/// island density, Voronoi + buffer-terrain biomes, cheese/spaghetti/
/// noodle isosurface caves, dense column-anchor seeder, biased-random-
/// walk ore, droplet hydraulic erosion, D3Q19 lattice fluid, Bridson
/// Poisson-disk placement, and L-system trees.
pub fn build_showcase() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.density = Arc::new(FloatingIslandField::default());
    cfg.strata = Arc::new(LayeredGeology::default());
    cfg.biome_matrix = Arc::new(VoronoiCells::default());
    cfg.biome_blend = Arc::new(BufferTerrainInjected::default());
    cfg.caves = Arc::new(IsosurfaceIntersection::default());
    cfg.feature_seeder = Arc::new(ColumnAnchorSeeder::new(SeederConfig {
        worm_density: 2.0,
        ore_density: 2.0,
        structure_density: 0.5,
        flora_tree_density: 2.0,
        floating_island_density: 0.25,
        ..Default::default()
    }));
    cfg.ore = Arc::new(BiasedRandomWalk::default());
    cfg.erosion = Arc::new(DropletHydraulic::default());
    cfg.fluid = Arc::new(LatticeBoltzmannD3Q19::default());
    cfg.placement = Arc::new(PoissonDiskBridson::default());
    cfg.flora = Arc::new(LSystemTrees::default());
    cfg.structures = Arc::new(WaveFunctionCollapse::default());
    cfg.sky_light = Arc::new(VerticalCastWithDiffusion::default());
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
