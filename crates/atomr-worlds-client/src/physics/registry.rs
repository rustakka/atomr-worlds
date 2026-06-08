//! Name → strategy registry, mirroring the render crate's
//! `apply_strategy_by_name`. The slot name is the [`PhysicsConfig`] field name.

use std::sync::Arc;

use super::config::PhysicsConfig;
use super::defaults::{GreedyBoxCompound, PerVoxelCompound};

/// Apply a strategy by `(slot, name)`. Returns `true` on success, `false` if
/// either the slot or the name is unknown.
pub fn apply_strategy_by_name(cfg: &mut PhysicsConfig, slot: &str, name: &str) -> bool {
    match slot {
        "collider" => match name {
            "GreedyBoxCompound" | "greedy" => {
                cfg.collider = Arc::new(GreedyBoxCompound);
                true
            }
            "PerVoxelCompound" | "per-voxel" | "per_voxel" => {
                cfg.collider = Arc::new(PerVoxelCompound);
                true
            }
            _ => false,
        },
        _ => false,
    }
}
