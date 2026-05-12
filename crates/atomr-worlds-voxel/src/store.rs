//! Sparse voxel store trait — the abstract surface.

use atomr_worlds_core::coord::IVec3;

use crate::brick::Brick;
use crate::error::VoxelError;
use crate::octree::Octree;
use crate::voxel::Voxel;

pub trait SparseVoxelStore {
    fn get(&self, p: IVec3) -> Result<Voxel, VoxelError>;
    fn set(&mut self, p: IVec3, v: Voxel) -> Result<(), VoxelError>;
    fn brick(&self, brick_coord: IVec3) -> Result<Option<&Brick>, VoxelError>;
    fn root_size_m(&self) -> f64;
    fn max_depth(&self) -> u8;
}

impl SparseVoxelStore for Octree {
    #[inline]
    fn get(&self, p: IVec3) -> Result<Voxel, VoxelError> {
        Octree::get_voxel(self, p)
    }
    #[inline]
    fn set(&mut self, p: IVec3, v: Voxel) -> Result<(), VoxelError> {
        Octree::set_voxel(self, p, v)
    }
    #[inline]
    fn brick(&self, brick_coord: IVec3) -> Result<Option<&Brick>, VoxelError> {
        Octree::brick(self, brick_coord)
    }
    #[inline]
    fn root_size_m(&self) -> f64 {
        self.root_size_m
    }
    #[inline]
    fn max_depth(&self) -> u8 {
        self.max_depth
    }
}
