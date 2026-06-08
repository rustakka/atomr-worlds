//! [`PhysicsConfig`] — the physics strategy registry resource.
//!
//! Mirrors the render crate's `RenderConfig`: pluggable strategy fields behind
//! `Arc<dyn Trait>`, with a `Default` that ships the behaviour we want out of
//! the box. The `--physics` / `--collider` CLI flags set these before the app
//! starts; the harness forces `enabled = false`.

use std::sync::Arc;

use bevy::prelude::*;

use super::defaults::GreedyBoxCompound;
use super::strategy::ColliderStrategy;

#[derive(Resource, Clone)]
pub struct PhysicsConfig {
    /// Runtime master switch. When `false`, [`super::PhysicsPlugin`] adds no
    /// rapier plugin or systems at all (zero cost). Forced off under the
    /// harness so golden captures are never perturbed by collision.
    pub enabled: bool,
    /// How a brick's solid voxels become a collider.
    pub collider: Arc<dyn ColliderStrategy>,
    /// Gravity, in m/s². The render grid is 1 m / voxel, so the rapier default
    /// `(0, -9.81, 0)` is already correct.
    pub gravity: Vec3,
    /// World edge length of one LOD-0 voxel, in meters (1.0 on the render grid).
    pub voxel_size_m: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collider: Arc::new(GreedyBoxCompound),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            voxel_size_m: 1.0,
        }
    }
}
