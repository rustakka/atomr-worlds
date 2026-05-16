//! Name → strategy registry for `WorldGenConfig` slots. Mirrors
//! `apply_strategy_by_name` in the client render registry — the harness
//! DSL sets strategies by `(slot, name)` pairs without code edits.
//!
//! Step 4 wires only the no-op + Vanilla impls; subsequent steps register
//! their concrete strategies here as they land.

use std::sync::Arc;

use super::biome_blend::{BufferTerrainInjected, Hard, NormalizedSparseConvolution};
use super::biome_matrix::{PerFaceWhittaker, VoronoiCells, WhittakerDirect2D};
use super::caves::{CellularAutomata3D, IsosurfaceIntersection, PerlinWorm, WorleyThreshold};
use super::config::WorldGenConfig;
use super::density::{FloatingIslandField, HeightmapPlanar, Hybrid2D3D, Pure3DOverhang};
use super::erosion::{DropletHydraulic, MacroRiverOnly};
use super::feature_seeder::ColumnAnchorSeeder;
use super::flora::{BlueNoiseGrass, LSystemTrees};
use super::fluid::{CellularAutomataFlow, LatticeBoltzmannD3Q19, Static};
use super::light::VerticalCastWithDiffusion;
use super::ore::{BiasedRandomWalk, ThresholdNoise};
use super::placement::{MitchellBestCandidate, PoissonDiskBridson, UniformGrid, WhiteNoise};
use super::strata::{KrigingInterpolated, LayeredGeology, TopsoilLayer};
use super::strategies::*;
use super::structures::{Jigsaw, QwfcClassicalSim, WaveFunctionCollapse};
use super::vanilla::MonolithicTerrainPass;

/// Apply a strategy by `(slot, name)`. Returns `true` if the slot+name was
/// recognized; `false` if either is unknown.
pub fn apply_worldgen_strategy_by_name(
    cfg: &mut WorldGenConfig,
    slot: &str,
    name: &str,
) -> bool {
    match slot {
        "density" => match name {
            "MonolithicTerrainPass" => {
                cfg.density = Arc::new(MonolithicTerrainPass::default());
                true
            }
            "HeightmapPlanar" => {
                cfg.density = Arc::new(HeightmapPlanar::default());
                true
            }
            "Hybrid2D3D" => {
                cfg.density = Arc::new(Hybrid2D3D::default());
                true
            }
            "Pure3DOverhang" => {
                cfg.density = Arc::new(Pure3DOverhang::default());
                true
            }
            "FloatingIslandField" => {
                cfg.density = Arc::new(FloatingIslandField::default());
                true
            }
            "None" => {
                cfg.density = Arc::new(NoneDensity);
                true
            }
            _ => false,
        },
        "strata" => match name {
            "MonolithicTerrainPass" => {
                cfg.strata = Arc::new(MonolithicTerrainPass::default());
                true
            }
            "TopsoilLayer" => {
                cfg.strata = Arc::new(TopsoilLayer::default());
                true
            }
            "LayeredGeology" => {
                cfg.strata = Arc::new(LayeredGeology::default());
                true
            }
            "KrigingInterpolated" => {
                cfg.strata = Arc::new(KrigingInterpolated::default());
                true
            }
            "None" => {
                cfg.strata = Arc::new(NoneStrata);
                true
            }
            _ => false,
        },
        "caves" => match name {
            "None" => {
                cfg.caves = Arc::new(NoneCaves);
                true
            }
            "WorleyThreshold" => {
                cfg.caves = Arc::new(WorleyThreshold::default());
                true
            }
            "CellularAutomata3D" => {
                cfg.caves = Arc::new(CellularAutomata3D::default());
                true
            }
            "PerlinWorm" => {
                cfg.caves = Arc::new(PerlinWorm::default());
                true
            }
            "IsosurfaceIntersection" => {
                cfg.caves = Arc::new(IsosurfaceIntersection::default());
                true
            }
            _ => false,
        },
        "ore" => match name {
            "None" => {
                cfg.ore = Arc::new(NoneOre);
                true
            }
            "ThresholdNoise" => {
                cfg.ore = Arc::new(ThresholdNoise::default());
                true
            }
            "BiasedRandomWalk" => {
                cfg.ore = Arc::new(BiasedRandomWalk::default());
                true
            }
            _ => false,
        },
        "erosion" => match name {
            "None" => {
                cfg.erosion = Arc::new(NoneErosion);
                true
            }
            "MacroRiverOnly" => {
                cfg.erosion = Arc::new(MacroRiverOnly);
                true
            }
            "DropletHydraulic" => {
                cfg.erosion = Arc::new(DropletHydraulic::default());
                true
            }
            _ => false,
        },
        "fluid" => match name {
            "None" => {
                cfg.fluid = Arc::new(NoneFluid);
                true
            }
            "Static" => {
                cfg.fluid = Arc::new(Static::default());
                true
            }
            "CellularAutomataFlow" => {
                cfg.fluid = Arc::new(CellularAutomataFlow::default());
                true
            }
            "LatticeBoltzmannD3Q19" => {
                cfg.fluid = Arc::new(LatticeBoltzmannD3Q19::default());
                true
            }
            _ => false,
        },
        "structures" => match name {
            "None" => {
                cfg.structures = Arc::new(NoneStructures);
                true
            }
            "WaveFunctionCollapse" => {
                cfg.structures = Arc::new(WaveFunctionCollapse::default());
                true
            }
            "Jigsaw" => {
                cfg.structures = Arc::new(Jigsaw::default());
                true
            }
            "QwfcClassicalSim" => {
                cfg.structures = Arc::new(QwfcClassicalSim::default());
                true
            }
            _ => false,
        },
        "flora" => match name {
            "None" => {
                cfg.flora = Arc::new(NoneFlora);
                true
            }
            "LSystemTrees" => {
                cfg.flora = Arc::new(LSystemTrees::default());
                true
            }
            "BlueNoiseGrass" => {
                cfg.flora = Arc::new(BlueNoiseGrass::default());
                true
            }
            _ => false,
        },
        "placement" => match name {
            "None" => {
                cfg.placement = Arc::new(NonePlacement);
                true
            }
            "WhiteNoise" => {
                cfg.placement = Arc::new(WhiteNoise::default());
                true
            }
            "UniformGrid" => {
                cfg.placement = Arc::new(UniformGrid::default());
                true
            }
            "PoissonDiskBridson" => {
                cfg.placement = Arc::new(PoissonDiskBridson::default());
                true
            }
            "MitchellBestCandidate" => {
                cfg.placement = Arc::new(MitchellBestCandidate::default());
                true
            }
            _ => false,
        },
        "biome_matrix" => match name {
            "PerFaceWhittaker" => {
                cfg.biome_matrix = Arc::new(PerFaceWhittaker::default());
                true
            }
            "WhittakerDirect2D" => {
                cfg.biome_matrix = Arc::new(WhittakerDirect2D::default());
                true
            }
            "VoronoiCells" => {
                cfg.biome_matrix = Arc::new(VoronoiCells::default());
                true
            }
            "None" => {
                cfg.biome_matrix = Arc::new(NoneBiomeMatrix);
                true
            }
            _ => false,
        },
        "biome_blend" => match name {
            "Hard" => {
                cfg.biome_blend = Arc::new(Hard::default());
                true
            }
            "NormalizedSparseConvolution" => {
                cfg.biome_blend = Arc::new(NormalizedSparseConvolution::default());
                true
            }
            "BufferTerrainInjected" => {
                cfg.biome_blend = Arc::new(BufferTerrainInjected::default());
                true
            }
            "None" => {
                cfg.biome_blend = Arc::new(NoneBiomeBlend);
                true
            }
            _ => false,
        },
        "sky_light" => match name {
            "None" => {
                cfg.sky_light = Arc::new(NoneSkyLight);
                true
            }
            "VerticalCastWithDiffusion" => {
                cfg.sky_light = Arc::new(VerticalCastWithDiffusion::default());
                true
            }
            _ => false,
        },
        "feature_seeder" => match name {
            "Empty" => {
                cfg.feature_seeder = Arc::new(EmptySeeder);
                true
            }
            "ColumnAnchorSeeder" => {
                cfg.feature_seeder = Arc::new(ColumnAnchorSeeder::default());
                true
            }
            _ => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::WorldGenConfig;
    use super::*;

    #[test]
    fn applies_known_slot_name_pair() {
        let mut cfg = WorldGenConfig::default();
        assert!(apply_worldgen_strategy_by_name(&mut cfg, "caves", "None"));
    }

    #[test]
    fn rejects_unknown_slot() {
        let mut cfg = WorldGenConfig::default();
        assert!(!apply_worldgen_strategy_by_name(&mut cfg, "bogus", "None"));
    }

    #[test]
    fn rejects_unknown_name() {
        let mut cfg = WorldGenConfig::default();
        assert!(!apply_worldgen_strategy_by_name(&mut cfg, "caves", "Bogus"));
    }
}
