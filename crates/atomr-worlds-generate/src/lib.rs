//! Procedural generators for atomr-worlds.
//!
//! Per-tier `Generator` impls (universe → world) and a `BrickGenerator`
//! trait whose CPU impl `TerrainGenerator` produces fully populated
//! voxel bricks from a world seed.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod brick;
pub mod error;
pub mod macro_state;
pub mod registry;
pub mod strategies;
pub mod terrain;
pub mod tiers;

pub use brick::{BrickGenContext, BrickGenerator};
pub use error::GenerateError;
pub use macro_state::{
    BiomeMap, ClimateConfig, ClimateField, DefaultMacroGenerator, ElevationField, FaceId,
    MacroConfig, MacroGenerator, MacroKey, MacroSample, MacroStateCache, PlateConfig, PlateMap,
    SurfaceGrid, WorldMacroState,
};
pub use registry::{
    default_registry, strategy_id, BuiltinSelector, GenerationPolicy, GeneratorRegistry,
    GeneratorRegistryBuilder, ResolveError, Resolved, StrategyId, StrategySelector, ASTEROID_BELT,
    EMPTY_PLANETOID, GAS_GIANT, TERRAIN,
};
pub use terrain::{
    TerrainConfig, TerrainGenerator, MATERIAL_AIR, MATERIAL_CAVE, MATERIAL_DIRT, MATERIAL_SAND,
    MATERIAL_SNOW, MATERIAL_STONE, MATERIAL_WATER,
};
pub use tiers::{
    GalaxyGen, SectorGen, SystemGen, UniverseGen, WorldGen,
};
