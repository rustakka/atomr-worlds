//! Procedural generators for atomr-worlds.
//!
//! Per-tier `Generator` impls (universe → world) and a `BrickGenerator`
//! trait whose CPU impl `TerrainGenerator` produces fully populated
//! voxel bricks from a world seed.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod brick;
pub mod error;
pub mod terrain;
pub mod tiers;

pub use brick::BrickGenerator;
pub use error::GenerateError;
pub use terrain::{TerrainConfig, TerrainGenerator, MATERIAL_AIR, MATERIAL_CAVE, MATERIAL_DIRT, MATERIAL_STONE};
pub use tiers::{
    GalaxyGen, SectorGen, SystemGen, UniverseGen, WorldGen,
};
