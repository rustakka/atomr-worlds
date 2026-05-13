//! `terrain` strategy: re-exports the existing `TerrainGenerator`.

use std::sync::Arc;

use crate::material_selection::LayeredWithFeatures;
use crate::terrain::{TerrainConfig, TerrainGenerator};

/// Convenience constructor used by [`crate::registry::default_registry`].
/// Attaches the [`LayeredWithFeatures`] material picker so worlds spawned
/// through the registry (the FP/TP client path) show grass / glow_rock /
/// ice in addition to the original 5 materials. The CUDA byte-equality
/// path stays untouched because it constructs `TerrainGenerator::new`
/// directly without going through this helper.
pub fn default_terrain() -> TerrainGenerator {
    TerrainGenerator::new(TerrainConfig::default())
        .with_material_strategy(Arc::new(LayeredWithFeatures::default()))
}
