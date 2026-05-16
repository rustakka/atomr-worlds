//! Layered brick generation pipeline.
//!
//! Hosts a `BrickPipeline` trait whose canonical impl
//! [`LayeredBrickPipeline`] runs fixed-order stages — density → strata →
//! caves → ore → erosion → fluid → structures → flora → light — each
//! plugged via a [`WorldGenConfig`] slot. [`LayeredGenerator`] adapts the
//! pipeline to the older [`crate::BrickGenerator`] trait so it can be
//! registered under [`crate::registry::TERRAIN_LAYERED`] alongside the
//! existing monolithic [`crate::TerrainGenerator`].

pub mod anchor;
pub mod biome_blend;
pub mod biome_matrix;
pub mod caves;
pub mod config;
pub mod density;
pub mod erosion;
pub mod feature_seeder;
pub mod flora;
pub mod fluid;
pub mod layered;
pub mod ore;
pub mod placement;
pub mod presets;
pub mod registry;
pub mod strata;
pub mod strategies;
pub mod structures;
pub mod vanilla;
pub mod workspace;

pub use anchor::{FeatureAnchor, FeatureAnchorCache, FeatureKind};
pub use biome_blend::{
    BufferTerrainConfig, BufferTerrainInjected, Hard, NormalizedSparseConvolution,
    SparseBlendConfig,
};
pub use biome_matrix::{
    PerFaceWhittaker, VoronoiCells, VoronoiCellsConfig, WhittakerDirect2D, WhittakerDirect2DConfig,
};
pub use caves::{CellularAutomata3D, IsosurfaceIntersection, PerlinWorm, WorleyThreshold};
pub use config::{WorldGenConfig, WorldGenPreset};
pub use density::{
    FloatingIslandField, FloatingIslandFieldConfig, HeightmapPlanar, HeightmapPlanarConfig,
    Hybrid2D3D, Hybrid2D3DConfig, Pure3DOverhang, Pure3DOverhangConfig,
};
pub use erosion::{DropletConfig, DropletHydraulic, MacroRiverOnly, DROPLET_DIM};
pub use feature_seeder::{ColumnAnchorSeeder, SeederConfig, FEATURE_DIM};
pub use flora::{BlueNoiseGrass, LSystemGrammar, LSystemTrees, TurtleParams};
pub use fluid::{
    CaFlowConfig, CellularAutomataFlow, LatticeBoltzmannD3Q19, LbmConfig, Static, StaticConfig,
};
pub use layered::{BrickPipeline, LayeredBrickPipeline, LayeredGenerator};
pub use ore::{
    BiasedRandomWalk, BiasedRandomWalkConfig, OreVeinConfig, StrataBias, ThresholdNoise,
};
pub use placement::{
    MitchellBestCandidate, MitchellConfig, PoissonDiskBridson, PoissonDiskConfig, UniformGrid,
    UniformGridConfig, WhiteNoise, WhiteNoiseConfig, PLACEMENT_DIM,
};
pub use presets::{build_advanced, build_showcase, build_vanilla};
pub use registry::apply_worldgen_strategy_by_name;
pub use strata::{
    KrigingInterpolated, LayeredGeology, StrataConfig, StratumBand, TopsoilConfig, TopsoilLayer,
};
pub use strategies::{
    BiomeBlendStrategy, BiomeMatrixStrategy, CaveStrategy, DensityFieldStrategy, ErosionStrategy,
    FeatureSeederStrategy, FloraStrategy, FluidStrategy, OreVeinStrategy, PlacementStrategy,
    SkyLightStrategy, StrataStrategy, StructureStrategy,
};
pub use structures::{
    Jigsaw, JigsawConfig, JigsawTag, QwfcClassicalSim, TileDef, TileGeometry, TileSet,
    WaveFunctionCollapse, WfcConfig,
};
pub use vanilla::MonolithicTerrainPass;
pub use workspace::{BrickWorkspace, WS_APRON_EDGE};
