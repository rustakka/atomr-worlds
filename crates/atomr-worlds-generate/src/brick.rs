//! `BrickGenerator` trait — the GPU-friendly "fill a brick from `(seed, coord)`"
//! shape. Phase 1 ships one CPU impl (`TerrainGenerator`); Phase 5 will add
//! a GPU adapter.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::Brick;

pub trait BrickGenerator: Send + Sync {
    /// Produce a fully-populated brick at `brick_coord` for the world seed.
    fn generate_brick(&self, world_seed: u64, brick_coord: IVec3) -> Brick;
}
