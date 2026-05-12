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

pub use cluster::ClusterHost;
pub use error::HostError;
pub use extractor::WorldExtractor;
pub use host::WorldHost;
pub use local::LocalHost;
