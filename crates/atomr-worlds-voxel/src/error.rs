use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::error::WorldsCoreError;

#[derive(Debug, thiserror::Error)]
pub enum VoxelError {
    #[error("requested LOD depth {requested} exceeds octree max_depth {max}")]
    LodTooDeep { requested: u8, max: u8 },
    #[error("voxel coordinate {0:?} outside octree")]
    OutOfBounds(IVec3),
    #[error(transparent)]
    Core(#[from] WorldsCoreError),
}
