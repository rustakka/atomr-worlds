//! Wave-function-collapse structure stamper.
//!
//! Operates over a small `WfcConfig::grid_dim` grid of tile cells per
//! `FeatureKind::Structure` anchor. Each cell starts with all tiles as
//! candidates; iterative observation (lowest-entropy cell collapses to a
//! weighted-random tile) plus AC-3 propagation across faces shrinks the
//! candidate sets. On contradiction (an empty candidate set), the
//! algorithm backtracks to the last decision point and resamples;
//! `max_backtrack_depth` bounds the search.
//!
//! Selected tile geometry is stamped into `ws.materials` clipped to the
//! brick AABB. Stamping uses `module_edge` voxels per cell.

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::seed::splitmix64;
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use super::super::strategies::StructureStrategy;
use super::super::workspace::BrickWorkspace;
use super::{TileDef, TileSet};

#[derive(Debug, Clone)]
pub struct WfcConfig {
    pub grid_dim: [u32; 3],
    pub module_edge: u32,
    pub max_backtrack_depth: u32,
}

impl Default for WfcConfig {
    fn default() -> Self {
        Self {
            grid_dim: [16, 16, 16],
            module_edge: 4,
            max_backtrack_depth: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaveFunctionCollapse {
    pub tiles: Arc<TileSet>,
    pub config: WfcConfig,
}

impl Default for WaveFunctionCollapse {
    fn default() -> Self {
        Self {
            tiles: Arc::new(TileSet::test_tiles()),
            config: WfcConfig::default(),
        }
    }
}

impl WaveFunctionCollapse {
    pub fn new(tiles: Arc<TileSet>, config: WfcConfig) -> Self {
        Self { tiles, config }
    }

    fn run_for_anchor(&self, anchor_seed: u64, origin: IVec3, ws: &mut BrickWorkspace) {
        if self.tiles.is_empty() {
            return;
        }
        let solved = match wfc_solve(&self.tiles, &self.config, anchor_seed) {
            Some(s) => s,
            None => return,
        };
        let me = self.config.module_edge as i64;
        let [gx, gy, gz] = self.config.grid_dim.map(|v| v as i64);
        for cz in 0..gz {
            for cy in 0..gy {
                for cx in 0..gx {
                    let idx = (cz * gy + cy) * gx + cx;
                    let tile_idx = solved[idx as usize];
                    let tile = &self.tiles.tiles[tile_idx];
                    for (vox, mat) in &tile.geometry.voxels {
                        let local = IVec3::new(
                            origin.x + cx * me + vox.x,
                            origin.y + cy * me + vox.y,
                            origin.z + cz * me + vox.z,
                        );
                        if !in_brick(local) {
                            continue;
                        }
                        ws.set_material(local.x as i32, local.y as i32, local.z as i32, Voxel::new(*mat));
                    }
                }
            }
        }
    }
}

impl StructureStrategy for WaveFunctionCollapse {
    fn id(&self) -> &'static str {
        "WaveFunctionCollapse"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let anchors: Vec<_> = ws
            .anchors
            .iter()
            .filter(|a| matches!(a.kind, super::super::anchor::FeatureKind::Structure))
            .copied()
            .collect();
        for a in anchors {
            let origin = IVec3::new(0, 0, 0); // brick-local origin (TODO: snap from a.origin_m)
            self.run_for_anchor(a.seed, origin, ws);
        }
    }
}

fn in_brick(p: IVec3) -> bool {
    let e = BRICK_EDGE as i64;
    p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < e && p.y < e && p.z < e
}

/// Solve the WFC for a tile grid. Returns one tile index (into `tiles.tiles`)
/// per cell, or `None` if no solution found within `max_backtrack_depth`.
fn wfc_solve(tiles: &TileSet, cfg: &WfcConfig, seed: u64) -> Option<Vec<usize>> {
    let n_tiles = tiles.tiles.len();
    if n_tiles == 0 {
        return None;
    }
    let [gx, gy, gz] = cfg.grid_dim.map(|v| v as usize);
    let n_cells = gx * gy * gz;

    let mut cells: Vec<Vec<bool>> = vec![vec![true; n_tiles]; n_cells];
    let mut rng = seed;
    let mut backtrack_budget = cfg.max_backtrack_depth as usize;

    loop {
        let lowest = find_lowest_entropy_cell(&cells);
        let cell = match lowest {
            Some(c) => c,
            None => {
                let mut result = Vec::with_capacity(n_cells);
                for cell in &cells {
                    let i = cell.iter().position(|b| *b)?;
                    result.push(i);
                }
                return Some(result);
            }
        };
        rng = splitmix64(rng);
        let pick = weighted_pick(&cells[cell], &tiles.tiles, rng);
        for (i, b) in cells[cell].iter_mut().enumerate() {
            *b = i == pick;
        }
        if !propagate(&mut cells, tiles, [gx, gy, gz]) {
            if backtrack_budget == 0 {
                return None;
            }
            backtrack_budget -= 1;
            cells = vec![vec![true; n_tiles]; n_cells];
        }
    }
}

fn find_lowest_entropy_cell(cells: &[Vec<bool>]) -> Option<usize> {
    let mut best = None;
    let mut best_count = usize::MAX;
    for (i, cell) in cells.iter().enumerate() {
        let count = cell.iter().filter(|b| **b).count();
        if count > 1 && count < best_count {
            best_count = count;
            best = Some(i);
        }
    }
    best
}

fn weighted_pick(mask: &[bool], tiles: &[TileDef], rng: u64) -> usize {
    let total: f32 = mask
        .iter()
        .zip(tiles.iter())
        .map(|(m, t)| if *m { t.weight.max(0.0) } else { 0.0 })
        .sum();
    if total <= 0.0 {
        return mask.iter().position(|m| *m).unwrap_or(0);
    }
    let r = ((rng >> 11) as f32 / (1u64 << 53) as f32) * total;
    let mut acc = 0.0;
    for (i, (m, t)) in mask.iter().zip(tiles.iter()).enumerate() {
        if !*m {
            continue;
        }
        acc += t.weight.max(0.0);
        if acc >= r {
            return i;
        }
    }
    mask.iter().rposition(|m| *m).unwrap_or(0)
}

fn propagate(cells: &mut [Vec<bool>], tiles: &TileSet, dim: [usize; 3]) -> bool {
    let [gx, gy, gz] = dim;
    let mut changed = true;
    while changed {
        changed = false;
        for z in 0..gz {
            for y in 0..gy {
                for x in 0..gx {
                    let here = (z * gy + y) * gx + x;
                    for (face, dir) in NEIGHBORS.iter().enumerate() {
                        let nx = x as i64 + dir.0;
                        let ny = y as i64 + dir.1;
                        let nz = z as i64 + dir.2;
                        if nx < 0 || ny < 0 || nz < 0 {
                            continue;
                        }
                        if nx as usize >= gx || ny as usize >= gy || nz as usize >= gz {
                            continue;
                        }
                        let there = (nz as usize * gy + ny as usize) * gx + nx as usize;
                        if restrict_cell(cells, here, there, face, tiles) {
                            changed = true;
                            if cells[there].iter().all(|b| !*b) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

const NEIGHBORS: [(i64, i64, i64); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

fn restrict_cell(
    cells: &mut [Vec<bool>],
    here: usize,
    there: usize,
    face: usize,
    tiles: &TileSet,
) -> bool {
    let mut changed = false;
    let allowed_here: Vec<usize> = cells[here]
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if *b { Some(i) } else { None })
        .collect();
    let allowed_set: std::collections::HashSet<u32> = allowed_here
        .iter()
        .flat_map(|i| tiles.tiles[*i].neighbors[face].iter().copied())
        .collect();
    for (i, b) in cells[there].iter_mut().enumerate() {
        if !*b {
            continue;
        }
        if !allowed_set.contains(&tiles.tiles[i].id) {
            *b = false;
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::anchor::{FeatureAnchor, FeatureKind};

    fn ws() -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(42, IVec3::new(0, 0, 0)))
    }

    #[test]
    fn solves_test_tileset_without_contradiction() {
        let tiles = TileSet::test_tiles();
        let cfg = WfcConfig {
            grid_dim: [3, 3, 3],
            module_edge: 1,
            max_backtrack_depth: 10,
        };
        let solved = wfc_solve(&tiles, &cfg, 0xABCDEF).expect("WFC should solve test tiles");
        assert_eq!(solved.len(), 27);
    }

    #[test]
    fn determinism_same_seed_same_voxels() {
        let s = WaveFunctionCollapse::default();
        let anchor = FeatureAnchor {
            kind: FeatureKind::Structure,
            column: IVec3::new(0, 0, 0),
            origin_m: [0.0; 3],
            seed: 0xCAFE_F00D,
        };
        let mut a = ws();
        a.anchors.push(anchor);
        s.run(&mut a);
        let mut b = ws();
        b.anchors.push(anchor);
        s.run(&mut b);
        for z in 0..BRICK_EDGE as i32 {
            for y in 0..BRICK_EDGE as i32 {
                for x in 0..BRICK_EDGE as i32 {
                    assert_eq!(a.material_at(x, y, z), b.material_at(x, y, z));
                }
            }
        }
    }
}
