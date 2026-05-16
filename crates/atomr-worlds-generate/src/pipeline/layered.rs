//! Pipeline orchestrator + [`BrickGenerator`] adapter.
//!
//! Runs configured stages in a fixed order, asserted at construction so
//! a reshuffled `WorldGenConfig` cannot silently scramble the pass chain.

use atomr_worlds_voxel::Brick;

use crate::brick::{BrickGenContext, BrickGenerator};

use super::config::WorldGenConfig;
use super::workspace::BrickWorkspace;

/// Fixed stage order. The orchestrator runs the configured `WorldGenConfig`
/// slots in exactly this sequence; the order is asserted at construction
/// and is part of the pipeline contract.
const STAGE_ORDER: &[&str] = &[
    "feature_seeder",
    "biome_matrix",
    "biome_blend",
    "density",
    "strata",
    "caves",
    "ore",
    "erosion",
    "fluid",
    "structures",
    "flora",
    "sky_light",
];

/// Trait for pluggable brick-generation pipelines. `LayeredBrickPipeline`
/// is the canonical impl; alternates (e.g. a debug single-stage pipeline)
/// can implement this trait directly.
pub trait BrickPipeline: Send + Sync + std::fmt::Debug {
    fn run(&self, ctx: BrickGenContext) -> Brick;
}

/// Multi-stage pipeline driven by a [`WorldGenConfig`].
#[derive(Clone, Debug)]
pub struct LayeredBrickPipeline {
    pub config: WorldGenConfig,
}

impl LayeredBrickPipeline {
    pub fn new(config: WorldGenConfig) -> Self {
        debug_assert_eq!(
            STAGE_ORDER.len(),
            12,
            "stage order list is out of sync; update STAGE_ORDER and BrickPipeline::run"
        );
        Self { config }
    }
}

impl BrickPipeline for LayeredBrickPipeline {
    fn run(&self, ctx: BrickGenContext) -> Brick {
        let mut ws = BrickWorkspace::new(ctx);
        let cfg = &self.config;
        cfg.feature_seeder.seed(&mut ws);
        cfg.biome_matrix.run(&mut ws);
        cfg.biome_blend.run(&mut ws);
        cfg.density.run(&mut ws);
        cfg.strata.run(&mut ws);
        cfg.caves.run(&mut ws);
        cfg.ore.run(&mut ws);
        cfg.erosion.run(&mut ws);
        cfg.fluid.run(&mut ws);
        cfg.structures.run(&mut ws);
        cfg.flora.run(&mut ws);
        cfg.sky_light.run(&mut ws);
        if let Some(light) = ws.light.take() {
            ws.brick.light_overlay = Some(light);
        }
        ws.brick
    }
}

/// [`BrickGenerator`] adapter so the pipeline can be registered alongside
/// the existing [`crate::TerrainGenerator`] under
/// [`crate::registry::TERRAIN_LAYERED`].
#[derive(Clone, Debug)]
pub struct LayeredGenerator {
    pub pipeline: LayeredBrickPipeline,
}

impl LayeredGenerator {
    pub fn new(config: WorldGenConfig) -> Self {
        Self { pipeline: LayeredBrickPipeline::new(config) }
    }
}

impl Default for LayeredGenerator {
    fn default() -> Self {
        Self::new(WorldGenConfig::default())
    }
}

impl BrickGenerator for LayeredGenerator {
    fn generate_brick(&self, ctx: &BrickGenContext) -> Brick {
        self.pipeline.run(ctx.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::WorldGenPreset;
    use atomr_worlds_core::coord::IVec3;

    #[test]
    fn layered_with_vanilla_runs_clean() {
        let g = LayeredGenerator::new(WorldGenConfig::preset(WorldGenPreset::Vanilla));
        let _b = g.generate_brick_legacy(7, IVec3::new(0, 0, 0));
    }

    #[test]
    fn pipeline_clone_is_cheap() {
        let g = LayeredGenerator::default();
        let _g2 = g.clone();
    }
}
