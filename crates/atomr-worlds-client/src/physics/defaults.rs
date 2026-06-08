//! Collider strategy implementations.

use atomr_worlds_voxel::Brick;
use bevy_rapier3d::prelude::Collider;

use super::collider_gen::{compound_from_boxes, per_voxel_boxes, brick_to_collider};
use super::strategy::ColliderStrategy;

/// Default: greedy box-merge → compound of cuboids. A fully-solid brick
/// collapses to one box; terrain slabs to a handful — cheap broad-phase, small
/// memory, convex (good for resting debris).
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedyBoxCompound;

impl ColliderStrategy for GreedyBoxCompound {
    fn name(&self) -> &'static str {
        "GreedyBoxCompound"
    }

    fn build(&self, brick: &Brick, voxel_size_m: f32) -> Option<Collider> {
        brick_to_collider(brick, voxel_size_m)
    }
}

/// One cuboid per solid voxel — the un-merged form. Heavier (up to 4096 shapes
/// per brick) but a useful A/B alternative and correctness oracle for the
/// greedy merge.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerVoxelCompound;

impl ColliderStrategy for PerVoxelCompound {
    fn name(&self) -> &'static str {
        "PerVoxelCompound"
    }

    fn build(&self, brick: &Brick, voxel_size_m: f32) -> Option<Collider> {
        compound_from_boxes(&per_voxel_boxes(brick), voxel_size_m)
    }
}
