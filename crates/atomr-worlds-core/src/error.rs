use crate::coord::IVec3;

#[derive(Debug, thiserror::Error)]
pub enum WorldsCoreError {
    #[error("coordinate {0:?} outside level bounds")]
    OutOfBounds(IVec3),
    #[error("invalid dimension id {0}")]
    BadDimension(u32),
}

pub type Result<T> = std::result::Result<T, WorldsCoreError>;
