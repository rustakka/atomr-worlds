//! Coordinate types.
//!
//! A single canonical [`IVec3`] (i64 components) underlies every level. i32
//! is insufficient because voxel coordinates at meter-resolution routinely
//! exceed 2^31 at galactic and universe scales.
//!
//! Per-level `#[repr(transparent)]` newtypes prevent mixing coordinates
//! between hierarchy levels at API boundaries with zero runtime cost.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Serialize, Deserialize)]
pub struct IVec3 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl IVec3 {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    #[inline]
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn splat(v: i64) -> Self {
        Self { x: v, y: v, z: v }
    }
}

impl From<(i64, i64, i64)> for IVec3 {
    #[inline]
    fn from((x, y, z): (i64, i64, i64)) -> Self {
        Self { x, y, z }
    }
}

impl From<[i64; 3]> for IVec3 {
    #[inline]
    fn from([x, y, z]: [i64; 3]) -> Self {
        Self { x, y, z }
    }
}

macro_rules! level_coord {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Serialize, Deserialize)]
        #[repr(transparent)]
        pub struct $name(pub IVec3);

        impl $name {
            pub const ZERO: Self = Self(IVec3::ZERO);

            #[inline]
            pub const fn new(x: i64, y: i64, z: i64) -> Self {
                Self(IVec3::new(x, y, z))
            }
        }

        impl From<IVec3> for $name {
            #[inline]
            fn from(v: IVec3) -> Self { Self(v) }
        }

        impl From<$name> for IVec3 {
            #[inline]
            fn from(v: $name) -> Self { v.0 }
        }
    };
}

level_coord!(
    /// Coordinate within the universe-level grid (typically `ZERO` for the root).
    UniverseCoord
);
level_coord!(
    /// Coordinate of a galaxy within its parent universe.
    GalaxyCoord
);
level_coord!(
    /// Coordinate of a sector within its parent galaxy.
    SectorCoord
);
level_coord!(
    /// Coordinate of a star system within its parent sector.
    SystemCoord
);
level_coord!(
    /// Coordinate of a world within its parent system.
    WorldCoord
);
level_coord!(
    /// Coordinate of a brick within a voxel octree.
    BrickCoord
);
level_coord!(
    /// Voxel coordinate inside a world (or other voxel-bearing object).
    VoxelCoord
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtypes_are_transparent() {
        assert_eq!(std::mem::size_of::<UniverseCoord>(), std::mem::size_of::<IVec3>());
        assert_eq!(std::mem::size_of::<BrickCoord>(), std::mem::size_of::<IVec3>());
    }

    #[test]
    fn round_trips_through_ivec3() {
        let g = GalaxyCoord::new(1, -2, 3);
        let v: IVec3 = g.into();
        let back: GalaxyCoord = v.into();
        assert_eq!(g, back);
    }
}
