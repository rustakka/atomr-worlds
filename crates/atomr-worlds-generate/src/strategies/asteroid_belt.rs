//! `asteroid_belt` strategy stub.
//!
//! Phase 7 ships a stub returning empty bricks; sparse rock voxels with
//! gap-tolerant clustering will land in a future phase.

use atomr_worlds_voxel::Brick;

use crate::brick::{BrickGenContext, BrickGenerator};

#[derive(Debug, Default, Clone, Copy)]
pub struct AsteroidBeltStub;

impl BrickGenerator for AsteroidBeltStub {
    fn generate_brick(&self, _ctx: &BrickGenContext) -> Brick {
        Brick::new()
    }
}
