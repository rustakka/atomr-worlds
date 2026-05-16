//! Name → strategy registry for `WorldGenConfig` slots. Mirrors
//! `apply_strategy_by_name` in the client render registry — the harness
//! DSL sets strategies by `(slot, name)` pairs without code edits.
//!
//! Step 4 wires only the no-op + Vanilla impls; subsequent steps register
//! their concrete strategies here as they land.

use std::sync::Arc;

use super::biome_blend::{BufferTerrainInjected, Hard, NormalizedSparseConvolution};
use super::biome_matrix::{PerFaceWhittaker, VoronoiCells, WhittakerDirect2D};
use super::config::WorldGenConfig;
use super::density::{FloatingIslandField, HeightmapPlanar, Hybrid2D3D, Pure3DOverhang};
use super::strata::{KrigingInterpolated, LayeredGeology, TopsoilLayer};
use super::strategies::*;
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
            _ => false,
        },
        "ore" => match name {
            "None" => {
                cfg.ore = Arc::new(NoneOre);
                true
            }
            _ => false,
        },
        "erosion" => match name {
            "None" => {
                cfg.erosion = Arc::new(NoneErosion);
                true
            }
            _ => false,
        },
        "fluid" => match name {
            "None" => {
                cfg.fluid = Arc::new(NoneFluid);
                true
            }
            _ => false,
        },
        "structures" => match name {
            "None" => {
                cfg.structures = Arc::new(NoneStructures);
                true
            }
            _ => false,
        },
        "flora" => match name {
            "None" => {
                cfg.flora = Arc::new(NoneFlora);
                true
            }
            _ => false,
        },
        "placement" => match name {
            "None" => {
                cfg.placement = Arc::new(NonePlacement);
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
            _ => false,
        },
        "feature_seeder" => match name {
            "Empty" => {
                cfg.feature_seeder = Arc::new(EmptySeeder);
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
