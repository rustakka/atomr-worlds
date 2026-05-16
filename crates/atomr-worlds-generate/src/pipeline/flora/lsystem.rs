//! `FloraStrategy` impl: stamp L-system trees at `FeatureKind::FloraTree`
//! anchors.

use std::sync::Arc;

use atomr_worlds_voxel::Voxel;

use crate::pipeline::anchor::FeatureKind;
use crate::pipeline::strategies::FloraStrategy;
use crate::pipeline::workspace::BrickWorkspace;
use crate::terrain::{MATERIAL_LEAVES, MATERIAL_WOOD};

use super::{LSystemGrammar, TurtleInterp};

/// Stamps an L-system tree at every `FeatureKind::FloraTree` anchor in the
/// workspace. Trunk voxels are `MATERIAL_WOOD`; canopy tips are
/// `MATERIAL_LEAVES`. Each anchor's `seed` drives the turtle's angle jitter
/// so two anchors with the same column but different seeds branch
/// differently.
#[derive(Debug, Clone)]
pub struct LSystemTrees {
    pub grammar: Arc<LSystemGrammar>,
}

impl LSystemTrees {
    pub fn new(grammar: Arc<LSystemGrammar>) -> Self {
        Self { grammar }
    }
}

impl Default for LSystemTrees {
    fn default() -> Self {
        Self::new(Arc::new(LSystemGrammar::default_tree()))
    }
}

impl FloraStrategy for LSystemTrees {
    fn id(&self) -> &'static str {
        "LSystemTrees"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        // Snapshot anchors so the borrow checker lets us mutate `ws` while
        // iterating the list.
        let anchors: Vec<_> = ws
            .anchors
            .iter()
            .filter(|a| a.kind == FeatureKind::FloraTree)
            .copied()
            .collect();
        if anchors.is_empty() {
            return;
        }
        let program = self.grammar.derive();
        let params = self.grammar.params;
        let trunk = Voxel::new(MATERIAL_WOOD);
        let canopy = Voxel::new(MATERIAL_LEAVES);
        for a in &anchors {
            let mut interp = TurtleInterp::new(ws, params, trunk, canopy, a.seed);
            interp.run_at(a.origin_m, &program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::anchor::FeatureAnchor;
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_voxel::BRICK_EDGE;

    fn ws_with_anchor(seed: u64) -> BrickWorkspace {
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::ZERO));
        ws.anchors.push(FeatureAnchor {
            kind: FeatureKind::FloraTree,
            column: IVec3::ZERO,
            origin_m: [8.0, 1.0, 8.0],
            seed,
        });
        ws
    }

    fn stamp_count(ws: &BrickWorkspace) -> usize {
        let mut n = 0;
        for z in 0..BRICK_EDGE as i32 {
            for y in 0..BRICK_EDGE as i32 {
                for x in 0..BRICK_EDGE as i32 {
                    if !ws.material_at(x, y, z).is_empty() {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    #[test]
    fn run_stamps_voxels() {
        let mut ws = ws_with_anchor(0xABCD);
        LSystemTrees::default().run(&mut ws);
        assert!(stamp_count(&ws) > 0);
    }

    #[test]
    fn same_seed_same_voxels() {
        let mut a = ws_with_anchor(0xABCD);
        let mut b = ws_with_anchor(0xABCD);
        LSystemTrees::default().run(&mut a);
        LSystemTrees::default().run(&mut b);
        for z in 0..BRICK_EDGE as i32 {
            for y in 0..BRICK_EDGE as i32 {
                for x in 0..BRICK_EDGE as i32 {
                    assert_eq!(a.material_at(x, y, z), b.material_at(x, y, z));
                }
            }
        }
    }

    #[test]
    fn no_anchors_no_stamps() {
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::ZERO));
        LSystemTrees::default().run(&mut ws);
        assert_eq!(stamp_count(&ws), 0);
    }
}
