//! Sparse voxel structures for atomr-worlds.
//!
//! Hybrid layout: a top-level sparse voxel octree provides empty-space
//! skipping at cosmic scales; the leaves are dense 16³ "bricks" of voxels
//! for cache-friendly local access. LOD is just which depth you query.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod brick;
pub mod codec;
pub mod dag;
pub mod error;
pub mod light;
pub mod octree;
pub mod storage;
pub mod store;
pub mod voxel;

pub use brick::{Brick, BrickDecodeError, BRICK_EDGE, BRICK_LEN};
pub use codec::{BrickCodec, CodecError, PaletteRle, RawU16, Rle, Zlib};
pub use dag::{gpu_get, DagBrick, DagGpu, DAG_GPU_EMPTY_ROOT, DAG_LEAF_FLAG};
pub use error::VoxelError;
pub use light::{LightOverlay, LIGHT_OVERLAY_BYTES};
pub use octree::{InternalNode, NodeId, NodeKind, Octree, OCTREE_NULL};
pub use storage::{BrickStorage, DenseBrick, SegmentedRowBrick, SvoBrick};
pub use store::SparseVoxelStore;
pub use voxel::Voxel;
