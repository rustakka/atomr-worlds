//! `BrickGenerator` trait — fills a brick from a [`BrickGenContext`].
//!
//! Phase 13c migration: the trait now consumes a context object instead of
//! the original `(world_seed, brick_coord)` tuple. Existing two-arg
//! callers (notably the CUDA accelerator's CPU fallback and Python
//! bindings) keep working via [`BrickGenerator::generate_brick_legacy`],
//! a default method that builds a default context.

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::{Lod, MetricScale};
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_voxel::Brick;

use crate::macro_state::WorldMacroState;

/// Inputs threaded into [`BrickGenerator::generate_brick`].
///
/// `macro_state` is `None` when a generator is invoked without macro pre-
/// sim (the pre-Phase-13c path); generators must preserve their existing
/// behavior in that case.
///
/// `lod` is the rendering convention LOD (depth 0 = finest, each step
/// doubles the voxel world-edge to `2^depth` meters). The terrain
/// generator uses it to sample procedural noise at the correct world
/// metric so adjacent LOD tiers agree on surface height. Default is
/// `Lod::new(0)` to preserve byte-for-byte legacy output for callers
/// (CUDA fallback, voxel writes) that haven't been ported to multi-LOD.
#[derive(Clone, Debug)]
pub struct BrickGenContext {
    pub world_seed: u64,
    pub brick_coord: IVec3,
    pub lod: Lod,
    pub shape: WorldShape,
    pub macro_state: Option<Arc<WorldMacroState>>,
    pub scale: MetricScale,
}

impl BrickGenContext {
    /// Construct a minimal context with default shape and no macro state.
    /// Used by the two-arg legacy shim.
    pub fn legacy(world_seed: u64, brick_coord: IVec3) -> Self {
        Self {
            world_seed,
            brick_coord,
            lod: Lod::new(0),
            shape: WorldShape::default_world(),
            macro_state: None,
            scale: MetricScale::DEFAULT_WORLD,
        }
    }
}

pub trait BrickGenerator: Send + Sync {
    /// Produce a fully-populated brick. Generators that don't need macro
    /// state simply ignore `ctx.macro_state` / `ctx.shape`.
    fn generate_brick(&self, ctx: &BrickGenContext) -> Brick;

    /// Two-argument convenience: build a default context and dispatch.
    /// Preserves the pre-Phase-13c signature for legacy callers.
    fn generate_brick_legacy(&self, world_seed: u64, brick_coord: IVec3) -> Brick {
        self.generate_brick(&BrickGenContext::legacy(world_seed, brick_coord))
    }
}
