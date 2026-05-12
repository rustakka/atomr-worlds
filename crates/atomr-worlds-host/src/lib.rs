//! Hosting backends for atomr-worlds.
//!
//! Same [`WorldHost`] trait is implemented by:
//!
//! - [`LocalHost`]  — embedded atomr `ActorSystem`, for single-player.
//! - [`ClusterHost`] — atomr-cluster-sharding `ShardRegion`, for multi-node.
//!
//! Both delegate to the same per-world actor; the only differences are
//! where the actors live and how messages are routed there.
//!
//! This phase ships shapes only — implementations are `todo!()`.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod cluster;
pub mod error;
pub mod extractor;
pub mod host;
pub mod local;
pub mod policy;
pub mod shape;
pub mod world_query_impl;

pub use cluster::{ClusterHost, ClusterHostConfig};
pub use error::HostError;
pub use extractor::WorldExtractor;
pub use host::WorldHost;
pub use local::{LocalHost, LocalHostConfig};
pub use policy::{DefaultPolicy, GenerationPolicy, PolicyResolver, PrefixPolicy};
pub use shape::{DefaultShape, PrefixShape, ShapeResolver};
pub use world_query_impl::LocalHostQuery;

pub use atomr_worlds_generate::{
    region_id, AuthoredRegion, AuthoredRegionStore, LiteralRegion, RegionAabb, RegionId,
};

pub use atomr_worlds_persist::{
    persistence_id_for, InMemoryJournal, InMemorySnapshotStore, RecoveredState, VoxelWriteEvent,
    WorldPersistence, WorldSnapshot,
};
