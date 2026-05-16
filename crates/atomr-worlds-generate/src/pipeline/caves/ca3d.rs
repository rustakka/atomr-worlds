//! 3-D Conway-style cellular automata caves.
//!
//! Initial fill: each voxel in the workspace apron is solid iff a hash of
//! `(seed, world voxel coord)` exceeds `fill_pct`. The apron is sampled
//! deterministically from the same hash so neighbor bricks agree on the
//! one-voxel overlap; iterations only consume the apron read window.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_noise::hash3_f01;
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::pipeline::strategies::CaveStrategy;
use crate::pipeline::workspace::{BrickWorkspace, WS_APRON_EDGE};

/// Seed salt for the CA initial-fill hash — kept distinct from other
/// noise channels so a future cave + ore + erosion stack can't alias.
const CA3D_SEED_SALT: u64 = 0xCA3D_F1A0_1A57_BE57;

#[derive(Clone, Debug)]
pub struct CellularAutomata3D {
    /// Fraction of cells alive at iteration 0; 0.45 ≈ classic Game-of-Life
    /// caves once 5 iterations close the small islands.
    pub fill_pct: f32,
    /// Number of birth/death iterations.
    pub iterations: u8,
    /// A dead cell becomes alive when its live-neighbor count is `>= birth`.
    pub birth_limit: u8,
    /// A live cell stays alive when its live-neighbor count is `>= survive`.
    pub survive_limit: u8,
}

impl Default for CellularAutomata3D {
    fn default() -> Self {
        Self { fill_pct: 0.45, iterations: 5, birth_limit: 13, survive_limit: 13 }
    }
}

impl CellularAutomata3D {
    pub fn new(fill_pct: f32, iterations: u8, birth_limit: u8, survive_limit: u8) -> Self {
        Self { fill_pct, iterations, birth_limit, survive_limit }
    }

    /// Deterministic initial-fill predicate at a world voxel coordinate.
    /// Hashing world coords (not brick-local) is what gives the apron its
    /// cross-brick agreement.
    #[inline]
    fn alive_initial(&self, seed: u64, wx: i64, wy: i64, wz: i64) -> bool {
        hash3_f01(seed.wrapping_add(CA3D_SEED_SALT), wx, wy, wz) < self.fill_pct
    }
}

#[inline]
fn flat(x: usize, y: usize, z: usize) -> usize {
    z * WS_APRON_EDGE * WS_APRON_EDGE + y * WS_APRON_EDGE + x
}

impl CaveStrategy for CellularAutomata3D {
    fn id(&self) -> &'static str {
        "CellularAutomata3D"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let edge = BRICK_EDGE as i64;
        let ox = ws.ctx.brick_coord.x * edge;
        let oy = ws.ctx.brick_coord.y * edge;
        let oz = ws.ctx.brick_coord.z * edge;
        let seed = ws.ctx.world_seed;

        let n = WS_APRON_EDGE;
        let vol = n * n * n;
        let mut cur = vec![false; vol];
        for zi in 0..n {
            for yi in 0..n {
                for xi in 0..n {
                    let wx = ox + xi as i64 - 1;
                    let wy = oy + yi as i64 - 1;
                    let wz = oz + zi as i64 - 1;
                    cur[flat(xi, yi, zi)] = self.alive_initial(seed, wx, wy, wz);
                }
            }
        }

        let mut nxt = cur.clone();
        for _ in 0..self.iterations {
            // Interior cells only — the 1-voxel apron supplies the read
            // window but is not itself iterated (its neighbors would lie
            // outside the workspace).
            for zi in 1..n - 1 {
                for yi in 1..n - 1 {
                    for xi in 1..n - 1 {
                        let mut live = 0u8;
                        for dz in -1i32..=1 {
                            for dy in -1i32..=1 {
                                for dx in -1i32..=1 {
                                    if dx == 0 && dy == 0 && dz == 0 {
                                        continue;
                                    }
                                    let nx = (xi as i32 + dx) as usize;
                                    let ny = (yi as i32 + dy) as usize;
                                    let nz = (zi as i32 + dz) as usize;
                                    if cur[flat(nx, ny, nz)] {
                                        live += 1;
                                    }
                                }
                            }
                        }
                        let here = cur[flat(xi, yi, zi)];
                        nxt[flat(xi, yi, zi)] = if here {
                            live >= self.survive_limit
                        } else {
                            live >= self.birth_limit
                        };
                    }
                }
            }
            std::mem::swap(&mut cur, &mut nxt);
        }

        for lz in 0..edge {
            for ly in 0..edge {
                for lx in 0..edge {
                    let alive = cur[flat((lx + 1) as usize, (ly + 1) as usize, (lz + 1) as usize)];
                    // "Alive" = open cave space → carve through any solid
                    // material left by earlier stages.
                    if alive {
                        ws.brick.set(IVec3::new(lx, ly, lz), Voxel::EMPTY);
                        ws.set_material(lx as i32, ly as i32, lz as i32, Voxel::EMPTY);
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

    fn fill_solid(ws: &mut BrickWorkspace) {
        for z in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for x in 0..BRICK_EDGE as i64 {
                    ws.brick.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
    }

    #[test]
    fn deterministic_for_same_coord() {
        let make = || {
            let mut w = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(2, -1, 3)));
            fill_solid(&mut w);
            CellularAutomata3D::default().run(&mut w);
            w.brick.nonempty_count
        };
        assert_eq!(make(), make());
    }

    #[test]
    fn stable_after_5_iterations() {
        // "Stable" here = repeated runs at iterations = 5 produce identical
        // bricks (the documented contract), and the population at
        // iterations = 6 doesn't drift wildly from iterations = 5 — birth
        // and death are in approximate equilibrium on the initial-fill
        // cloud.
        let ctx = BrickGenContext::legacy(123, IVec3::new(0, 0, 0));
        let mut w5 = BrickWorkspace::new(ctx.clone());
        fill_solid(&mut w5);
        CellularAutomata3D { iterations: 5, ..Default::default() }.run(&mut w5);
        let mut w5b = BrickWorkspace::new(ctx.clone());
        fill_solid(&mut w5b);
        CellularAutomata3D { iterations: 5, ..Default::default() }.run(&mut w5b);
        assert_eq!(w5.brick.nonempty_count, w5b.brick.nonempty_count);

        let mut w6 = BrickWorkspace::new(ctx);
        fill_solid(&mut w6);
        CellularAutomata3D { iterations: 6, ..Default::default() }.run(&mut w6);
        let drift = (w5.brick.nonempty_count as i64 - w6.brick.nonempty_count as i64).abs();
        // 10% drift cap — well above natural CA churn, well below the
        // ~3000-cell brick volume the default carves into a cave.
        assert!(
            drift < (w5.brick.nonempty_count as i64) / 10 + 64,
            "CA population drifted from {} to {} between iter 5 and 6",
            w5.brick.nonempty_count,
            w6.brick.nonempty_count,
        );
    }
}
