//! `Static` and `CellularAutomataFlow` fluid impls.
//!
//! `Static` is the Vanilla baseline — fill voxels below sea level with
//! water without any flow simulation. It reads sea level from the
//! attached macro state's hydrology layer when present, falling back to
//! the strategy's own config when not.
//!
//! `CellularAutomataFlow` is a Minecraft-style ticked rule set: water
//! prefers to fall straight down, then spreads horizontally up to N
//! steps. Source / sink heuristics keep the field bounded.

use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::terrain::{MATERIAL_AIR, MATERIAL_WATER};

use super::super::strategies::FluidStrategy;
use super::super::workspace::BrickWorkspace;

/// Sea-level configuration for [`Static`]. When the brick context carries
/// macro state with hydrology, the macro `sea_level_m` wins; this is the
/// fallback so harness DSL configs without macro state still produce a
/// flat water field.
#[derive(Copy, Clone, Debug)]
pub struct StaticConfig {
    pub sea_level_voxels: f32,
}

impl Default for StaticConfig {
    fn default() -> Self {
        Self { sea_level_voxels: 0.0 }
    }
}

/// Fill voxels below sea level with water.
#[derive(Clone, Debug)]
pub struct Static {
    pub config: StaticConfig,
}

impl Default for Static {
    fn default() -> Self {
        Self { config: StaticConfig::default() }
    }
}

impl FluidStrategy for Static {
    fn id(&self) -> &'static str {
        "Static"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        // Effective sea level: macro state when present, else the
        // strategy's own config.
        let sea_voxels = ws
            .ctx
            .macro_state
            .as_ref()
            .map(|m| {
                let mpv = ws.ctx.scale.meters_per_voxel(ws.ctx.lod);
                (m.water.sea_level_m as f64 / mpv) as f32
            })
            .unwrap_or(self.config.sea_level_voxels);

        let edge = BRICK_EDGE as i32;
        let base_y = ws.ctx.brick_coord.y as i32 * edge;
        let water = Voxel::new(MATERIAL_WATER);
        for ly in 0..edge {
            let wy = (base_y + ly) as f32;
            if wy >= sea_voxels {
                continue;
            }
            for z in 0..edge {
                for x in 0..edge {
                    if ws.material_at(x, ly, z).0 == MATERIAL_AIR {
                        ws.set_material(x, ly, z, water);
                    }
                }
            }
        }
    }
}

/// Tunables for [`CellularAutomataFlow`].
#[derive(Copy, Clone, Debug)]
pub struct CaFlowConfig {
    /// Number of CA ticks to run. Default 8 — enough to settle a brick.
    pub ticks: u8,
    /// Maximum horizontal spread (in voxels) from a source per tick.
    /// Default 4.
    pub max_spread: u8,
}

impl Default for CaFlowConfig {
    fn default() -> Self {
        Self { ticks: 8, max_spread: 4 }
    }
}

/// Minecraft-style cellular automata water flow.
///
/// Reads / writes `ws.materials`. Internal flow level (the "how deep is
/// this water cell" count) lives in a side buffer keyed by the
/// workspace's apron index — analogous to a `FluidLayer` field on the
/// brick but without persisting on `Brick` yet.
#[derive(Clone, Debug)]
pub struct CellularAutomataFlow {
    pub config: CaFlowConfig,
}

impl Default for CellularAutomataFlow {
    fn default() -> Self {
        Self { config: CaFlowConfig::default() }
    }
}

impl FluidStrategy for CellularAutomataFlow {
    fn id(&self) -> &'static str {
        "CellularAutomataFlow"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let edge = BRICK_EDGE as i32;
        let n = ws.materials.len();
        // Per-cell remaining horizontal spread budget. Falls to 0 once a
        // cell has spread `max_spread` times, preventing infinite oceans.
        let mut spread = vec![self.config.max_spread; n];

        for _tick in 0..self.config.ticks {
            // Phase 1: gravity. Any water cell with air immediately below
            // moves downward.
            for z in 0..edge {
                for y in (1..edge).rev() {
                    for x in 0..edge {
                        if ws.material_at(x, y, z).0 != MATERIAL_WATER {
                            continue;
                        }
                        if ws.material_at(x, y - 1, z).0 == MATERIAL_AIR {
                            ws.set_material(x, y, z, Voxel::new(MATERIAL_AIR));
                            ws.set_material(x, y - 1, z, Voxel::new(MATERIAL_WATER));
                        }
                    }
                }
            }
            // Phase 2: horizontal spread. Walk +X/-X/+Z/-Z; if a water
            // cell has air to the side (and either no support below or
            // its spread budget remaining), copy itself into the air cell.
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        if ws.material_at(x, y, z).0 != MATERIAL_WATER {
                            continue;
                        }
                        let idx = BrickWorkspace::apron_index(x, y, z);
                        if spread[idx] == 0 {
                            continue;
                        }
                        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                            let nx = x + dx;
                            let nz = z + dz;
                            if !(0..edge).contains(&nx) || !(0..edge).contains(&nz) {
                                continue;
                            }
                            if ws.material_at(nx, y, nz).0 == MATERIAL_AIR {
                                ws.set_material(nx, y, nz, Voxel::new(MATERIAL_WATER));
                                let nidx = BrickWorkspace::apron_index(nx, y, nz);
                                spread[nidx] = spread[idx].saturating_sub(1);
                            }
                        }
                        spread[idx] = spread[idx].saturating_sub(1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use atomr_worlds_core::coord::IVec3;

    fn empty_ws() -> BrickWorkspace {
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(0x5EA, IVec3::new(0, 0, 0)));
        for z in 0..BRICK_EDGE as i32 {
            for y in 0..BRICK_EDGE as i32 {
                for x in 0..BRICK_EDGE as i32 {
                    ws.set_material(x, y, z, Voxel::new(MATERIAL_AIR));
                }
            }
        }
        ws
    }

    #[test]
    fn static_fills_below_sea_level() {
        let s = Static { config: StaticConfig { sea_level_voxels: 8.0 } };
        let mut ws = empty_ws();
        s.run(&mut ws);
        // Below 8: water. Above 8: air.
        assert_eq!(ws.material_at(0, 0, 0).0, MATERIAL_WATER);
        assert_eq!(ws.material_at(0, 7, 0).0, MATERIAL_WATER);
        assert_eq!(ws.material_at(0, 8, 0).0, MATERIAL_AIR);
        assert_eq!(ws.material_at(0, 15, 0).0, MATERIAL_AIR);
    }

    #[test]
    fn static_deterministic() {
        let s = Static::default();
        let mut a = empty_ws();
        let mut b = empty_ws();
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(a.materials, b.materials);
    }

    #[test]
    fn ca_flow_water_settles_within_n_ticks() {
        let s = CellularAutomataFlow { config: CaFlowConfig { ticks: 16, max_spread: 4 } };
        let mut ws = empty_ws();
        // Drop a single source at the top.
        ws.set_material(8, 15, 8, Voxel::new(MATERIAL_WATER));
        s.run(&mut ws);
        // After flow, expect water at the bottom of the column.
        assert_eq!(ws.material_at(8, 0, 8).0, MATERIAL_WATER);
    }

    #[test]
    fn ca_flow_deterministic() {
        let s = CellularAutomataFlow::default();
        let mut a = empty_ws();
        let mut b = empty_ws();
        a.set_material(8, 15, 8, Voxel::new(MATERIAL_WATER));
        b.set_material(8, 15, 8, Voxel::new(MATERIAL_WATER));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(a.materials, b.materials);
    }
}
