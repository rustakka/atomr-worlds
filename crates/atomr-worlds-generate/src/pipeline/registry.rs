//! Name → strategy registry for `WorldGenConfig` slots. Mirrors
//! `apply_strategy_by_name` in the client render registry — the harness
//! DSL sets strategies by `(slot, name)` pairs without code edits.
//!
//! Step 4 wires only the no-op + Vanilla impls; subsequent steps register
//! their concrete strategies here as they land.

use std::sync::Arc;

use super::caves::{CellularAutomata3D, IsosurfaceIntersection, PerlinWorm, WorleyThreshold};
use super::config::WorldGenConfig;
use super::feature_seeder::ColumnAnchorSeeder;
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
            "None" => {
                cfg.biome_matrix = Arc::new(NoneBiomeMatrix);
                true
            }
            _ => false,
        },
        "biome_blend" => match name {
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
