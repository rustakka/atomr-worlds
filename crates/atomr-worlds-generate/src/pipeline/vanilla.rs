//! Vanilla "monolithic" pipeline pass — delegates to the existing
//! [`TerrainGenerator`] so the Vanilla preset is byte-equal to today's
//! generator. Steps 5–7 will replace this with per-stage byte-equal
//! impls; until then, the Vanilla preset uses [`MonolithicTerrainPass`]
//! as a single density+strata stage and `None` for all other slots.

use std::sync::Arc;

use crate::brick::BrickGenerator;
use crate::strategies::terrain::default_terrain;
use crate::terrain::TerrainGenerator;

use super::strategies::{DensityFieldStrategy, StrataStrategy};
use super::workspace::BrickWorkspace;

/// Single-pass "everything" stage that produces a brick identical to
/// [`TerrainGenerator::generate_brick`]. Registered as both the
/// `DensityFieldStrategy` and (no-op) `StrataStrategy` slot of the
/// Vanilla preset so the trait surface stays uniform.
#[derive(Clone, Debug)]
pub struct MonolithicTerrainPass {
    inner: Arc<TerrainGenerator>,
}

impl Default for MonolithicTerrainPass {
    fn default() -> Self {
        Self { inner: Arc::new(default_terrain()) }
    }
}

impl MonolithicTerrainPass {
    pub fn new(gen: TerrainGenerator) -> Self {
        Self { inner: Arc::new(gen) }
    }
}

impl DensityFieldStrategy for MonolithicTerrainPass {
    fn id(&self) -> &'static str {
        "MonolithicTerrainPass"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        ws.brick = self.inner.generate_brick(&ws.ctx);
    }
}

impl StrataStrategy for MonolithicTerrainPass {
    fn id(&self) -> &'static str {
        "MonolithicTerrainPass"
    }
    fn run(&self, _ws: &mut BrickWorkspace) {}
}
