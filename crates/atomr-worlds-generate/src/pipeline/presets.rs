//! Named preset constructors for [`WorldGenConfig`].
//!
//! `Vanilla` must produce bricks byte-equal to [`crate::TerrainGenerator`];
//! a regression snapshot test in `tests/vanilla_byte_equality.rs` enforces
//! this. `Advanced` and `Showcase` are populated as Steps 5–10 land the
//! per-paper strategies — until then they fall back to Vanilla.

use std::sync::Arc;

use super::caves::{CellularAutomata3D, IsosurfaceIntersection};
use super::config::WorldGenConfig;
use super::feature_seeder::{ColumnAnchorSeeder, SeederConfig};
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

/// `Advanced` opts into every paper algorithm at moderate cost. Step 6
/// switches caves from the legacy Worley path (still bundled inside
/// `MonolithicTerrainPass` for Vanilla) to a 3-D cellular-automata carver,
/// and enables the column-anchor seeder so worm / ore / structure / flora
/// passes have anchors to consume even when each individual stage is still
/// `None`.
pub fn build_advanced() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.caves = Arc::new(CellularAutomata3D::default());
    cfg.feature_seeder = Arc::new(ColumnAnchorSeeder::new(SeederConfig {
        worm_density: 1.0,
        ore_density: 1.0,
        structure_density: 0.25,
        flora_tree_density: 1.0,
        ..Default::default()
    }));
    cfg
}

/// `Showcase` cranks every algorithm up for the visual demo. Step 6 wires
/// the cheese/spaghetti/noodle isosurface carver plus a high-density column
/// seeder.
pub fn build_showcase() -> WorldGenConfig {
    let mut cfg = build_vanilla();
    cfg.caves = Arc::new(IsosurfaceIntersection::default());
    cfg.feature_seeder = Arc::new(ColumnAnchorSeeder::new(SeederConfig {
        worm_density: 2.0,
        ore_density: 2.0,
        structure_density: 0.5,
        flora_tree_density: 2.0,
        floating_island_density: 0.25,
        ..Default::default()
    }));
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
