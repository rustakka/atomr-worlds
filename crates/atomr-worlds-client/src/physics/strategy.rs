//! The pluggable collider-generation strategy.
//!
//! Mirrors the render crate's `Arc<dyn Trait>` strategy pattern: a small
//! `Send + Sync + 'static` trait with named implementations, swappable at
//! runtime via [`super::registry::apply_strategy_by_name`] (the `--collider`
//! CLI flag).

use atomr_worlds_voxel::Brick;
use bevy_rapier3d::prelude::Collider;

/// Builds a rapier collider for a brick's solid voxels.
///
/// The returned collider is in **brick-local** space (origin at the brick's min
/// corner), so it composes directly with the brick entity's transform. Returns
/// `None` when the brick has no solid voxels.
pub trait ColliderStrategy: Send + Sync + 'static {
    /// Stable name used by the registry / CLI.
    fn name(&self) -> &'static str;

    /// Build the collider from `brick`'s solid voxels. `voxel_size_m` is the
    /// world edge length of one voxel (1.0 at the LOD-0 render grid).
    fn build(&self, brick: &Brick, voxel_size_m: f32) -> Option<Collider>;
}
