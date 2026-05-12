use atomr_worlds_core::error::WorldsCoreError;
use atomr_worlds_proto::ProtoError;
use atomr_worlds_voxel::VoxelError;

#[derive(Debug, thiserror::Error)]
pub enum HostError {
    #[error(transparent)]
    Voxel(#[from] VoxelError),
    #[error(transparent)]
    Proto(#[from] ProtoError),
    #[error(transparent)]
    Core(#[from] WorldsCoreError),
    #[error("host is shutting down")]
    Shutdown,
    #[error("not yet implemented: {0}")]
    NotYetImplemented(&'static str),
}
