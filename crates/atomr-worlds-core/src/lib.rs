//! Pure-data primitives for atomr-worlds.
//!
//! No actor-runtime, async, or networking dependencies — this crate is the
//! foundation that the rest of the workspace (and external tooling) builds on.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod addr;
pub mod atmosphere;
pub mod coord;
pub mod dim;
pub mod error;
pub mod hierarchy;
pub mod interaction;
pub mod lod;
pub mod material_physics;
pub mod seed;
pub mod shape;
pub mod vehicle;

pub use addr::{AddrEither, Address, Level, LevelKey, WorldAddr};
pub use atmosphere::AtmosphereRadius;
pub use coord::{
    BrickCoord, DVec3, GalaxyCoord, IVec3, Meters, Quat, SectorCoord, SystemCoord, UniverseCoord,
    VoxelCoord, WorldCoord,
};
pub use dim::DimensionId;
pub use error::{Result, WorldsCoreError};
pub use hierarchy::{Galaxy, Generator, Sector, System, Universe, World};
pub use interaction::{AffectedSet, InteractionUnit, ToolKind};
pub use lod::{Lod, MetricScale, MetricScaleRegistry};
pub use material_physics::{
    default_palette as default_physics_palette, MaterialPhysicsPalette, MaterialPhysicsProps,
};
pub use seed::{child_seed, derive_child, splitmix64, HierarchicalIdentifier};
pub use shape::{ShapeAabb, WorldShape};
pub use vehicle::{AffineFrame, ContainingFrame, ParentAddr, VehicleAddr, VehicleSlot, VehicleSlotId};
