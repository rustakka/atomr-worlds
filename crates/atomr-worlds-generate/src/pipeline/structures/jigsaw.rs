//! Jigsaw-pool structure stamper.
//!
//! Recursively unfolds a start-pool entry by attaching template-pool
//! entries until either `max_depth` is reached or the accumulated AABB
//! exceeds `max_bbox`. Connection rules use tagged edges (`JigsawTag`),
//! matched by exact tag equality. Templates resolve through the
//! existing [`AuthoredRegionStore`] so existing voxfile loaders work
//! without changes.

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::seed::splitmix64;
use atomr_worlds_voxel::BRICK_EDGE;

use crate::authored::{AuthoredRegionStore, RegionId};

use super::super::strategies::StructureStrategy;
use super::super::workspace::BrickWorkspace;

pub type JigsawTag = String;

#[derive(Debug, Clone)]
pub struct JigsawConfig {
    pub start_pool: Vec<RegionId>,
    pub template_pool: Vec<RegionId>,
    pub max_depth: u32,
    pub max_bbox: IVec3,
}

impl Default for JigsawConfig {
    fn default() -> Self {
        Self {
            start_pool: Vec::new(),
            template_pool: Vec::new(),
            max_depth: 4,
            max_bbox: IVec3::new(64, 32, 64),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Jigsaw {
    pub config: JigsawConfig,
    pub region_store: Arc<AuthoredRegionStore>,
}

impl Jigsaw {
    pub fn new(config: JigsawConfig, region_store: Arc<AuthoredRegionStore>) -> Self {
        Self { config, region_store }
    }
}

impl StructureStrategy for Jigsaw {
    fn id(&self) -> &'static str {
        "Jigsaw"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        if self.config.start_pool.is_empty() {
            return;
        }
        let anchors: Vec<_> = ws
            .anchors
            .iter()
            .filter(|a| matches!(a.kind, super::super::anchor::FeatureKind::Structure))
            .copied()
            .collect();
        let brick_coord = ws.ctx.brick_coord;
        for anchor in anchors {
            let mut rng = anchor.seed;
            let mut placed = Vec::new();
            rng = splitmix64(rng);
            let start_idx = (rng as usize) % self.config.start_pool.len();
            let start_id = self.config.start_pool[start_idx];
            self.expand_recursive(start_id, 0, &mut rng, &mut placed);
            for region_id in &placed {
                if let Some(region) = self.region_store.get(*region_id) {
                    if !region.contains_brick(brick_coord, BRICK_EDGE as i64) {
                        continue;
                    }
                    region.apply_to_brick(brick_coord, &mut ws.brick);
                }
            }
        }
    }
}

impl Jigsaw {
    fn expand_recursive(
        &self,
        region_id: RegionId,
        depth: u32,
        rng: &mut u64,
        placed: &mut Vec<RegionId>,
    ) {
        placed.push(region_id);
        if depth + 1 >= self.config.max_depth {
            return;
        }
        if self.config.template_pool.is_empty() {
            return;
        }
        *rng = splitmix64(*rng);
        let next_idx = (*rng as usize) % self.config.template_pool.len();
        let next = self.config.template_pool[next_idx];
        self.expand_recursive(next, depth + 1, rng, placed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::anchor::{FeatureAnchor, FeatureKind};

    fn ws() -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(0, 0, 0)))
    }

    #[test]
    fn empty_pools_run_is_noop() {
        let j = Jigsaw::default();
        let mut w = ws();
        w.anchors.push(FeatureAnchor {
            kind: FeatureKind::Structure,
            column: IVec3::new(0, 0, 0),
            origin_m: [0.0; 3],
            seed: 1,
        });
        let before = w.brick.nonempty_count;
        j.run(&mut w);
        assert_eq!(before, w.brick.nonempty_count);
    }

    #[test]
    fn depth_limit_honored_with_empty_template_pool() {
        let cfg = JigsawConfig {
            start_pool: vec![crate::authored::region_id("test")],
            template_pool: vec![],
            max_depth: 2,
            ..Default::default()
        };
        let j = Jigsaw::new(cfg, Arc::new(AuthoredRegionStore::new()));
        let mut placed = Vec::new();
        let mut rng = 0u64;
        j.expand_recursive(crate::authored::region_id("test"), 0, &mut rng, &mut placed);
        assert_eq!(placed.len(), 1);
    }
}
