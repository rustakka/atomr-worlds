//! Voxel material id — a 16-bit index into a material palette.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Pod, Zeroable, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Voxel(pub u16);

impl Voxel {
    /// Sentinel for empty / "no voxel".
    pub const EMPTY: Self = Self(0);

    #[inline]
    pub const fn new(material: u16) -> Self {
        Self(material)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}
