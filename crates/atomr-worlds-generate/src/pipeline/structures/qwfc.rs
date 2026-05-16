//! Classical-simulation QWFC (Quantum Wave Function Collapse). The
//! tile-selection PDF is sampled from `splitmix64`-derived amplitude +
//! phase rather than the WFC weighted-uniform draw. Behaviour is
//! otherwise identical to [`super::wfc::WaveFunctionCollapse`]; treat
//! this strategy as a deterministic classical analogue of the
//! amplitude-collapse picture, not a real quantum simulator.

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::seed::splitmix64;
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use super::super::strategies::StructureStrategy;
use super::super::workspace::BrickWorkspace;
use super::wfc::WfcConfig;
use super::{TileDef, TileSet};

#[derive(Debug, Clone)]
pub struct QwfcClassicalSim {
    pub tiles: Arc<TileSet>,
    pub config: WfcConfig,
}

impl Default for QwfcClassicalSim {
    fn default() -> Self {
        Self {
            tiles: Arc::new(TileSet::test_tiles()),
            config: WfcConfig::default(),
        }
    }
}

impl StructureStrategy for QwfcClassicalSim {
    fn id(&self) -> &'static str {
        "QwfcClassicalSim"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        if self.tiles.is_empty() {
            return;
        }
        let anchors: Vec<_> = ws
            .anchors
            .iter()
            .filter(|a| matches!(a.kind, super::super::anchor::FeatureKind::Structure))
            .copied()
            .collect();
        for a in anchors {
            if let Some(solved) = qwfc_solve(&self.tiles, &self.config, a.seed) {
                stamp(&self.tiles, &self.config, &solved, ws);
            }
        }
    }
}

fn qwfc_solve(tiles: &TileSet, cfg: &WfcConfig, seed: u64) -> Option<Vec<usize>> {
    let n_tiles = tiles.tiles.len();
    if n_tiles == 0 {
        return None;
    }
    let [gx, gy, gz] = cfg.grid_dim.map(|v| v as usize);
    let n_cells = gx * gy * gz;
    let mut cells: Vec<Vec<bool>> = vec![vec![true; n_tiles]; n_cells];
    let mut rng = seed;

    for cell_i in 0..n_cells {
        let candidates: Vec<usize> = cells[cell_i]
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b { Some(i) } else { None })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let pick = amplitude_pick(&candidates, &tiles.tiles, &mut rng);
        for (i, b) in cells[cell_i].iter_mut().enumerate() {
            *b = i == pick;
        }
    }
    Some(
        cells
            .iter()
            .map(|c| c.iter().position(|b| *b).unwrap_or(0))
            .collect(),
    )
}

fn amplitude_pick(candidates: &[usize], tiles: &[TileDef], rng: &mut u64) -> usize {
    let mut probs = Vec::with_capacity(candidates.len());
    let mut total = 0.0f32;
    for &c in candidates {
        *rng = splitmix64(*rng);
        let phase = (*rng as f32) * (std::f32::consts::TAU / u64::MAX as f32);
        let amp = phase.cos();
        let weight = tiles[c].weight.max(0.0);
        let p = (amp * amp) * weight;
        total += p;
        probs.push(p);
    }
    if total <= 0.0 {
        return candidates[0];
    }
    *rng = splitmix64(*rng);
    let r = ((*rng >> 11) as f32 / (1u64 << 53) as f32) * total;
    let mut acc = 0.0;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if acc >= r {
            return candidates[i];
        }
    }
    candidates[candidates.len() - 1]
}

fn stamp(tiles: &TileSet, cfg: &WfcConfig, solved: &[usize], ws: &mut BrickWorkspace) {
    let me = cfg.module_edge as i64;
    let [gx, gy, gz] = cfg.grid_dim.map(|v| v as i64);
    let e = BRICK_EDGE as i64;
    for cz in 0..gz {
        for cy in 0..gy {
            for cx in 0..gx {
                let idx = (cz * gy + cy) * gx + cx;
                let tile = &tiles.tiles[solved[idx as usize]];
                for (vox, mat) in &tile.geometry.voxels {
                    let p = IVec3::new(cx * me + vox.x, cy * me + vox.y, cz * me + vox.z);
                    if p.x < 0 || p.y < 0 || p.z < 0 || p.x >= e || p.y >= e || p.z >= e {
                        continue;
                    }
                    ws.set_material(p.x as i32, p.y as i32, p.z as i32, Voxel::new(*mat));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::anchor::{FeatureAnchor, FeatureKind};

    fn ws() -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(99, IVec3::new(0, 0, 0)))
    }

    #[test]
    fn determinism_same_seed_same_voxels() {
        let s = QwfcClassicalSim::default();
        let anchor = FeatureAnchor {
            kind: FeatureKind::Structure,
            column: IVec3::new(0, 0, 0),
            origin_m: [0.0; 3],
            seed: 0xBEEF_C0DE,
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
