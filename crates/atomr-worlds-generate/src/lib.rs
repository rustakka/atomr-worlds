//! Procedural generators for atomr-worlds.
//!
//! Per-tier `Generator` impls (universe → world) and a `BrickGenerator`
//! trait whose CPU impl `TerrainGenerator` produces fully populated
//! voxel bricks from a world seed.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod authored;
pub mod brick;
pub mod error;
pub mod macro_state;
pub mod material_selection;
pub mod pipeline;
pub mod registry;
pub mod strategies;
pub mod terrain;
pub mod tiers;

pub use authored::{
    heightmap_from_columns, region_id, AuthoredRegion, AuthoredRegionStore, HeightmapRegion,
    LiteralRegion, RegionAabb, RegionId, VoxFileRegion, VoxelTransform,
};
pub use brick::{BrickGenContext, BrickGenerator};
pub use error::GenerateError;
pub use macro_state::{
    water_kind, BiomeMap, ClimateConfig, ClimateField, DefaultMacroGenerator, ElevationField,
    FaceId, HydrologyConfig, HydrologyGenerator, MacroConfig, MacroGenerator, MacroKey,
    MacroSample, MacroStateCache, PlateConfig, PlateMap, ReliefConfig, SurfaceGrid, WaterField,
    WorldMacroState,
};
pub use pipeline::{
    apply_worldgen_strategy_by_name, build_advanced, build_showcase, build_vanilla,
    BiomeBlendStrategy, BiomeMatrixStrategy, BrickPipeline, BrickWorkspace, CaveStrategy,
    DensityFieldStrategy, ErosionStrategy, FeatureAnchor, FeatureAnchorCache, FeatureKind,
    FeatureSeederStrategy, FloraStrategy, FluidStrategy, LayeredBrickPipeline, LayeredGenerator,
    MonolithicTerrainPass, OreVeinStrategy, PlacementStrategy, SkyLightConfig, SkyLightStrategy,
    StrataStrategy, StructureStrategy, VerticalCastWithDiffusion, WorldGenConfig, WorldGenPreset,
    SKY_LIGHT_DIM, WS_APRON_EDGE,
};
pub use registry::{
    default_registry, strategy_id, BuiltinSelector, GenerationPolicy, GeneratorRegistry,
    GeneratorRegistryBuilder, ResolveError, Resolved, StrategyId, StrategySelector, ASTEROID_BELT,
    EMPTY_PLANETOID, GAS_GIANT, TERRAIN, TERRAIN_LAYERED,
};
pub use material_selection::{
    DynMaterialStrategy, LayeredWithFeatures, LegacyBanded, MaterialContext,
    MaterialSelectionStrategy,
};
pub use terrain::{
    TerrainConfig, TerrainGenerator, MATERIAL_AIR, MATERIAL_CAVE, MATERIAL_DIRT,
    MATERIAL_GLOW_ROCK, MATERIAL_GRASS, MATERIAL_ICE, MATERIAL_LEAVES, MATERIAL_SAND,
    MATERIAL_SNOW, MATERIAL_STONE, MATERIAL_WATER, MATERIAL_WOOD,
};
pub use tiers::{
    GalaxyGen, SectorGen, SystemGen, UniverseGen, WorldGen,
};
