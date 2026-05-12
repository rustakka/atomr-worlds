//! `gas_giant` strategy stub.
//!
//! Phase 7 ships a stub that returns empty bricks; the real body — gaseous
//! density bands, banded coloration — will land in a later phase. Distinct
//! from [`super::empty_planetoid::EmptyPlanetoidStrategy`] in that it claims
//! the `gas_giant` strategy id and is intended to be replaced by real content.

use atomr_worlds_voxel::Brick;

use crate::brick::{BrickGenContext, BrickGenerator};

#[derive(Debug, Default, Clone, Copy)]
pub struct GasGiantStub;

impl BrickGenerator for GasGiantStub {
    fn generate_brick(&self, _ctx: &BrickGenContext) -> Brick {
        Brick::new()
    }
}
