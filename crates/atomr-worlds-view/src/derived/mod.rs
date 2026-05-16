//! Phase 14 derived data structures.
//!
//! Each submodule defines a precomputed view of voxel data tailored to
//! one display mode's access pattern. Cached in [`crate::ViewCache`] with
//! AABB-intersection invalidation driven by host `VoxelDelta`/`RegionDelta`
//! events.

pub mod horizon_shell;
pub mod slice_index;
pub mod surface_raster;
pub mod world_summary;
