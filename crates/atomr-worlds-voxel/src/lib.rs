//! Sparse voxel structures for atomr-worlds.
//!
//! Hybrid layout: a top-level sparse voxel octree provides empty-space
//! skipping at cosmic scales; the leaves are dense 16³ "bricks" of voxels
//! for cache-friendly local access. LOD is just which depth you query.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod brick;
pub mod error;
pub mod octree;
pub mod store;
pub mod voxel;

pub use brick::{Brick, BrickDecodeError, BRICK_EDGE, BRICK_LEN};
pub use error::VoxelError;
pub use octree::{InternalNode, NodeId, NodeKind, Octree, OCTREE_NULL};
pub use store::SparseVoxelStore;
pub use voxel::Voxel;
