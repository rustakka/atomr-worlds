//! `empty_planetoid` strategy: intentionally-empty content.
//!
//! Differs semantically from `GenerationPolicy::Empty`:
//! - `Empty` policy short-circuits the actor and bypasses generation entirely.
//! - `EmptyPlanetoidStrategy` *generates* empty bricks via the registry, so a
//!   world can still be addressed, journalled, and overlaid on top of an
//!   intentionally-blank baseline. Use this when you want a discoverable
//!   "this body exists but is blank by design" semantics.

use atomr_worlds_voxel::Brick;

use crate::brick::{BrickGenContext, BrickGenerator};

#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyPlanetoidStrategy;

impl BrickGenerator for EmptyPlanetoidStrategy {
    fn generate_brick(&self, _ctx: &BrickGenContext) -> Brick {
        Brick::new()
    }
}
