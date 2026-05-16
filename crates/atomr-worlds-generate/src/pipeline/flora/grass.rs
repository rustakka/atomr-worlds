//! `FloraStrategy` impl: stamp grass tufts on the topmost solid voxel of
//! each column targeted by a placement strategy.

use std::sync::Arc;

use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::pipeline::placement::PoissonDiskBridson;
use crate::pipeline::strategies::{FloraStrategy, PlacementStrategy};
use crate::pipeline::workspace::BrickWorkspace;
use crate::terrain::MATERIAL_GRASS;

/// Stamps a single grass voxel directly above the topmost solid voxel of
/// every brick-local column sampled by the placement strategy. Empty
/// columns and columns whose top voxel is `BRICK_EDGE - 1` (no headroom
/// inside the brick) are skipped.
#[derive(Debug, Clone)]
pub struct BlueNoiseGrass {
    pub placement: Arc<dyn PlacementStrategy>,
}

impl BlueNoiseGrass {
    pub fn new(placement: Arc<dyn PlacementStrategy>) -> Self {
        Self { placement }
    }
}

impl Default for BlueNoiseGrass {
    fn default() -> Self {
        Self::new(Arc::new(PoissonDiskBridson::with_min_distance(0.5)))
    }
}

impl FloraStrategy for BlueNoiseGrass {
    fn id(&self) -> &'static str {
        "BlueNoiseGrass"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let points = self.placement.place(ws);
        if points.is_empty() {
            return;
        }
        let grass = Voxel::new(MATERIAL_GRASS);
        for p in points {
            let x = p[0].floor() as i32;
            let z = p[2].floor() as i32;
            if !(0..BRICK_EDGE as i32).contains(&x) || !(0..BRICK_EDGE as i32).contains(&z) {
                continue;
            }
            // Walk down from the brick top to find the highest solid voxel.
            let mut found_top: Option<i32> = None;
            for y in (0..BRICK_EDGE as i32).rev() {
                if !ws.material_at(x, y, z).is_empty() {
                    found_top = Some(y);
                    break;
                }
            }
            if let Some(top) = found_top {
                let tuft_y = top + 1;
                if tuft_y < BRICK_EDGE as i32 && ws.material_at(x, tuft_y, z).is_empty() {
                    ws.set_material(x, tuft_y, z, grass);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::placement::{UniformGrid, UniformGridConfig};
    use atomr_worlds_core::coord::IVec3;

    #[test]
    fn stamps_grass_above_solid_column() {
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::ZERO));
        // One solid column at (4, 0..=3, 4); top is y=3.
        let stone = Voxel::new(crate::terrain::MATERIAL_STONE);
        for y in 0..=3 {
            ws.set_material(4, y, 4, stone);
        }
        // Use a uniform grid with 1m spacing so (4, _, 4) is hit.
        let strat = BlueNoiseGrass::new(Arc::new(UniformGrid::new(UniformGridConfig {
            spacing_m: 1.0,
        })));
        strat.run(&mut ws);
        assert_eq!(
            ws.material_at(4, 4, 4),
            Voxel::new(MATERIAL_GRASS),
            "grass tuft should sit on top of the solid column",
        );
    }

    #[test]
    fn skips_empty_columns() {
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::ZERO));
        let strat = BlueNoiseGrass::new(Arc::new(UniformGrid::new(UniformGridConfig {
            spacing_m: 1.0,
        })));
        strat.run(&mut ws);
        // Empty workspace: no grass should appear anywhere.
        for z in 0..BRICK_EDGE as i32 {
            for y in 0..BRICK_EDGE as i32 {
                for x in 0..BRICK_EDGE as i32 {
                    assert!(ws.material_at(x, y, z).is_empty());
                }
            }
        }
    }
}
