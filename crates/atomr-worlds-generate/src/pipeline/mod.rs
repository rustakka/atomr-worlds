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
pub mod config;
pub mod flora;
pub mod layered;
pub mod placement;
pub mod presets;
pub mod registry;
pub mod strategies;
pub mod vanilla;
pub mod workspace;

pub use anchor::{FeatureAnchor, FeatureAnchorCache, FeatureKind};
pub use config::{WorldGenConfig, WorldGenPreset};
pub use flora::{BlueNoiseGrass, LSystemGrammar, LSystemTrees, TurtleParams};
pub use layered::{BrickPipeline, LayeredBrickPipeline, LayeredGenerator};
pub use placement::{
    MitchellBestCandidate, MitchellConfig, PoissonDiskBridson, PoissonDiskConfig, UniformGrid,
    UniformGridConfig, WhiteNoise, WhiteNoiseConfig, PLACEMENT_DIM,
};
pub use presets::{build_advanced, build_showcase, build_vanilla};
pub use registry::apply_worldgen_strategy_by_name;
pub use strategies::{
    BiomeBlendStrategy, BiomeMatrixStrategy, CaveStrategy, DensityFieldStrategy, ErosionStrategy,
    FeatureSeederStrategy, FloraStrategy, FluidStrategy, OreVeinStrategy, PlacementStrategy,
    SkyLightStrategy, StrataStrategy, StructureStrategy,
};
pub use vanilla::MonolithicTerrainPass;
pub use workspace::{BrickWorkspace, WS_APRON_EDGE};
