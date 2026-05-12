//! `terrain` strategy: re-exports the existing `TerrainGenerator`.

use crate::terrain::{TerrainConfig, TerrainGenerator};

/// Convenience constructor used by [`crate::registry::default_registry`].
pub fn default_terrain() -> TerrainGenerator {
    TerrainGenerator::new(TerrainConfig::default())
}
