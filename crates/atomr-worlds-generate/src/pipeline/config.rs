//! Strategy-slot configuration for the layered pipeline.

use std::fmt::Debug;
use std::sync::Arc;

use super::strategies::*;

/// All pluggable strategy slots for a layered generator. Mirrors
/// `RenderConfig`'s shape: trait-object slots that can be swapped at
/// runtime through [`super::registry::apply_worldgen_strategy_by_name`].
#[derive(Clone)]
pub struct WorldGenConfig {
    pub density: Arc<dyn DensityFieldStrategy>,
    pub strata: Arc<dyn StrataStrategy>,
    pub caves: Arc<dyn CaveStrategy>,
    pub ore: Arc<dyn OreVeinStrategy>,
    pub erosion: Arc<dyn ErosionStrategy>,
    pub fluid: Arc<dyn FluidStrategy>,
    pub structures: Arc<dyn StructureStrategy>,
    pub flora: Arc<dyn FloraStrategy>,
    pub placement: Arc<dyn PlacementStrategy>,
    pub biome_matrix: Arc<dyn BiomeMatrixStrategy>,
    pub biome_blend: Arc<dyn BiomeBlendStrategy>,
    pub sky_light: Arc<dyn SkyLightStrategy>,
    pub feature_seeder: Arc<dyn FeatureSeederStrategy>,
}

impl Debug for WorldGenConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldGenConfig")
            .field("density", &self.density.id())
            .field("strata", &self.strata.id())
            .field("caves", &self.caves.id())
            .field("ore", &self.ore.id())
            .field("erosion", &self.erosion.id())
            .field("fluid", &self.fluid.id())
            .field("structures", &self.structures.id())
            .field("flora", &self.flora.id())
            .field("placement", &self.placement.id())
            .field("biome_matrix", &self.biome_matrix.id())
            .field("biome_blend", &self.biome_blend.id())
            .field("sky_light", &self.sky_light.id())
            .field("feature_seeder", &self.feature_seeder.id())
            .finish()
    }
}

/// Named preset selector. `Vanilla` reproduces the existing generator
/// byte-for-byte; `Advanced` opts into every paper algorithm at moderate
/// cost; `Showcase` cranks every algorithm up for the visual demo.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum WorldGenPreset {
    #[default]
    Vanilla,
    Advanced,
    Showcase,
}

impl WorldGenConfig {
    pub fn preset(p: WorldGenPreset) -> Self {
        match p {
            WorldGenPreset::Vanilla => super::presets::build_vanilla(),
            WorldGenPreset::Advanced => super::presets::build_advanced(),
            WorldGenPreset::Showcase => super::presets::build_showcase(),
        }
    }
}

impl Default for WorldGenConfig {
    fn default() -> Self {
        Self::preset(WorldGenPreset::Vanilla)
    }
}
