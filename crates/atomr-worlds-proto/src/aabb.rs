use atomr_worlds_core::coord::IVec3;
use serde::{Deserialize, Serialize};

/// Inclusive-min, exclusive-max axis-aligned bounding box in voxel coordinates.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Serialize, Deserialize)]
pub struct AABB {
    pub min: IVec3,
    pub max: IVec3,
}

impl AABB {
    #[inline]
    pub const fn new(min: IVec3, max: IVec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn contains(&self, p: IVec3) -> bool {
        p.x >= self.min.x
            && p.x < self.max.x
            && p.y >= self.min.y
            && p.y < self.max.y
            && p.z >= self.min.z
            && p.z < self.max.z
    }
}
