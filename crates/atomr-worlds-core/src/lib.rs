//! Pure-data primitives for atomr-worlds.
//!
//! No actor-runtime, async, or networking dependencies — this crate is the
//! foundation that the rest of the workspace (and external tooling) builds on.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod addr;
pub mod coord;
pub mod dim;
pub mod error;
pub mod hierarchy;
pub mod lod;
pub mod seed;

pub use addr::{Level, LevelKey, WorldAddr};
pub use coord::{
    BrickCoord, GalaxyCoord, IVec3, SectorCoord, SystemCoord, UniverseCoord, VoxelCoord, WorldCoord,
};
pub use dim::DimensionId;
pub use error::{Result, WorldsCoreError};
pub use hierarchy::{Galaxy, Generator, Sector, System, Universe, World};
pub use lod::{Lod, MetricScale};
pub use seed::{child_seed, splitmix64};
