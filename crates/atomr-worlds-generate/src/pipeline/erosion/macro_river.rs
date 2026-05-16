//! Strategy-shaped wrapper around the existing macro river carve.
//!
//! In Step 7 the Vanilla preset still routes density + strata through
//! [`crate::pipeline::vanilla::MonolithicTerrainPass`], which itself invokes
//! the legacy `TerrainGenerator::river_carve` path. Re-running the carve
//! from this strategy would double-apply, breaking the Vanilla byte-equality
//! contract. So this strategy is intentionally a no-op for Vanilla, with
//! the river carving still owned by the monolith. A future step that
//! splits density / strata into independent strategies will move the carve
//! body here (and the byte-equality test will gate the move).

use super::super::strategies::ErosionStrategy;
use super::super::workspace::BrickWorkspace;

/// Erosion slot that defers to the macro river carve already performed by
/// [`crate::pipeline::vanilla::MonolithicTerrainPass`]. Behaves as a no-op
/// at the strategy level — kept for API parity so the Vanilla preset can
/// opt in once density / strata are split.
#[derive(Debug, Default, Copy, Clone)]
pub struct MacroRiverOnly;

impl ErosionStrategy for MacroRiverOnly {
    fn id(&self) -> &'static str {
        "MacroRiverOnly"
    }

    fn run(&self, _ws: &mut BrickWorkspace) {
        // River carving currently happens inside MonolithicTerrainPass.
        // The strategy-level body activates when density+strata are
        // decomposed.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use atomr_worlds_core::coord::IVec3;

    #[test]
    fn macro_river_only_is_no_op_today() {
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(1, IVec3::new(0, 0, 0)));
        let before = ws.materials.clone();
        MacroRiverOnly.run(&mut ws);
        assert_eq!(ws.materials, before);
    }
}
