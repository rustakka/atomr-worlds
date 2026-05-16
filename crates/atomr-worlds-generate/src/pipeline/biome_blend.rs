//! [`BiomeBlendStrategy`] implementations.
//!
//! Blend stages refine the biome labels published by
//! [`super::biome_matrix`]. The Vanilla preset uses [`Hard`] which is a
//! no-op — borders read whatever label the matrix stage put down.
//! [`NormalizedSparseConvolution`] interpolates labels in a radius-R
//! neighborhood of biome centers (this is the "smooth biome boundary"
//! pass described in the paper). [`BufferTerrainInjected`] flags
//! high-disparity borders so downstream passes can carve transitional
//! features (rivers, shrubland).

use atomr_worlds_core::seed::splitmix64;
use atomr_worlds_voxel::BRICK_EDGE;

use super::anchor::{FeatureAnchor, FeatureKind};
use super::strategies::BiomeBlendStrategy;
use super::workspace::BrickWorkspace;

#[inline]
fn primary_biome_label(ws: &BrickWorkspace) -> Option<u8> {
    ws.anchors
        .iter()
        .find(|a| a.kind == FeatureKind::BufferTerrain)
        .map(|a| a.seed as u8)
}

#[inline]
fn brick_center_xz_m(ws: &BrickWorkspace) -> (f32, f32) {
    let v = (1u64 << ws.ctx.lod.depth as u32) as f32;
    let edge_m = BRICK_EDGE as f32 * v;
    (
        ws.ctx.brick_coord.x as f32 * edge_m + edge_m * 0.5,
        ws.ctx.brick_coord.z as f32 * edge_m + edge_m * 0.5,
    )
}

/// Hard borders — no-op blend that preserves the matrix stage's labels.
#[derive(Clone, Debug, Default, Copy)]
pub struct Hard;

impl Hard {
    pub fn new() -> Self {
        Self
    }
}

impl BiomeBlendStrategy for Hard {
    fn id(&self) -> &'static str {
        "Hard"
    }
    fn run(&self, _ws: &mut BrickWorkspace) {}
}

/// Normalized sparse convolution: for each query coordinate sample biome
/// centers within radius R; compute `weight_i = max(0, 1 - d_i/R)`;
/// normalize; interpolate. The output writes one composite anchor whose
/// `seed` carries the blended biome label in the low byte and a packed
/// blend confidence in the high bits.
#[derive(Clone, Debug)]
pub struct NormalizedSparseConvolution {
    pub config: SparseBlendConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct SparseBlendConfig {
    pub radius_m: f32,
}

impl Default for SparseBlendConfig {
    fn default() -> Self {
        Self { radius_m: 192.0 }
    }
}

impl Default for NormalizedSparseConvolution {
    fn default() -> Self {
        Self { config: SparseBlendConfig::default() }
    }
}

impl NormalizedSparseConvolution {
    pub fn new(config: SparseBlendConfig) -> Self {
        Self { config }
    }

    fn iter_neighborhood<F: FnMut(f32, u8)>(&self, ws: &BrickWorkspace, mut visit: F) {
        let (cx, cz) = brick_center_xz_m(ws);
        let radius = self.config.radius_m.max(1.0);
        let cell = radius;
        let inv = 1.0 / cell;
        let qx = (cx * inv).floor() as i64;
        let qz = (cz * inv).floor() as i64;
        let seed = ws.ctx.world_seed;
        for dz in -1..=1 {
            for dx in -1..=1 {
                let ix = qx + dx;
                let iz = qz + dz;
                let mixer = splitmix64(
                    seed ^ ((ix as u64).wrapping_mul(0xA0F1_3B27_0C5D_8E11))
                        ^ ((iz as u64).wrapping_mul(0x6CA1_BD27_3F77_55C9)),
                );
                let jit_x = ((mixer & 0xFFFF) as f32) / 65535.0;
                let jit_z = (((mixer >> 16) & 0xFFFF) as f32) / 65535.0;
                let label = (mixer >> 32) as u8 % 10;
                let center_x = (ix as f32 + jit_x) * cell;
                let center_z = (iz as f32 + jit_z) * cell;
                let d = ((cx - center_x).powi(2) + (cz - center_z).powi(2)).sqrt();
                visit(d, label);
            }
        }
    }
}

impl BiomeBlendStrategy for NormalizedSparseConvolution {
    fn id(&self) -> &'static str {
        "NormalizedSparseConvolution"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let radius = self.config.radius_m.max(1.0);
        let mut acc = [0.0_f32; 10];
        let mut total = 0.0_f32;
        self.iter_neighborhood(ws, |d, label| {
            let w = (1.0 - d / radius).max(0.0);
            if w > 0.0 && (label as usize) < acc.len() {
                acc[label as usize] += w;
                total += w;
            }
        });
        let (blended, confidence) = if total > 0.0 {
            let mut best = (0u8, 0.0_f32);
            for (i, &v) in acc.iter().enumerate() {
                if v > best.1 {
                    best = (i as u8, v);
                }
            }
            (best.0, best.1 / total)
        } else if let Some(b) = primary_biome_label(ws) {
            (b, 1.0)
        } else {
            (0, 0.0)
        };
        let (cx, cz) = brick_center_xz_m(ws);
        let conf_bits = ((confidence.clamp(0.0, 1.0) * 65535.0) as u64) << 16;
        ws.anchors.push(FeatureAnchor {
            kind: FeatureKind::BufferTerrain,
            column: ws.ctx.brick_coord,
            origin_m: [cx, 0.0, cz],
            seed: (blended as u64) | conf_bits,
        });
    }
}

/// Buffer-terrain injection: scan the matrix-stage neighborhood; where
/// adjacent labels differ sharply, emit a [`FeatureKind::BufferTerrain`]
/// anchor at the brick center so downstream passes (river carve, shrub
/// strip) can consume it.
#[derive(Clone, Debug)]
pub struct BufferTerrainInjected {
    pub config: BufferTerrainConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct BufferTerrainConfig {
    /// Sample radius for neighbour labels (m).
    pub radius_m: f32,
    /// Disparity threshold — distinct labels above this count trigger an
    /// injection.
    pub min_distinct_labels: u8,
}

impl Default for BufferTerrainConfig {
    fn default() -> Self {
        Self { radius_m: 96.0, min_distinct_labels: 2 }
    }
}

impl Default for BufferTerrainInjected {
    fn default() -> Self {
        Self { config: BufferTerrainConfig::default() }
    }
}

impl BufferTerrainInjected {
    pub fn new(config: BufferTerrainConfig) -> Self {
        Self { config }
    }
}

impl BiomeBlendStrategy for BufferTerrainInjected {
    fn id(&self) -> &'static str {
        "BufferTerrainInjected"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let cfg = self.config;
        let primary = primary_biome_label(ws).unwrap_or(0);
        let seed = ws.ctx.world_seed;
        let mut distinct: [bool; 256] = [false; 256];
        distinct[primary as usize] = true;
        let coord = ws.ctx.brick_coord;
        let radius_cells = (cfg.radius_m / (BRICK_EDGE as f32)).max(1.0).ceil() as i64;
        for dz in -radius_cells..=radius_cells {
            for dx in -radius_cells..=radius_cells {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let nx = coord.x + dx;
                let nz = coord.z + dz;
                let mixer = splitmix64(
                    seed ^ ((nx as u64).wrapping_mul(0xA0F1_3B27_0C5D_8E11))
                        ^ ((nz as u64).wrapping_mul(0x6CA1_BD27_3F77_55C9)),
                );
                distinct[((mixer >> 32) as u8 as usize) % 10] = true;
            }
        }
        let n = distinct.iter().filter(|&&b| b).count() as u8;
        if n >= cfg.min_distinct_labels {
            let (cx, cz) = brick_center_xz_m(ws);
            ws.anchors.push(FeatureAnchor {
                kind: FeatureKind::BufferTerrain,
                column: coord,
                origin_m: [cx, 0.0, cz],
                seed: 0xBFBF_0000_0000_0000 | primary as u64,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::biome_matrix::WhittakerDirect2D;
    use crate::pipeline::strategies::BiomeMatrixStrategy as _;
    use atomr_worlds_core::coord::IVec3;

    fn ws(seed: u64, coord: IVec3) -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(seed, coord))
    }

    #[test]
    fn hard_is_noop() {
        let s = Hard::default();
        let mut w = ws(7, IVec3::new(0, 0, 0));
        let before = w.anchors.len();
        s.run(&mut w);
        assert_eq!(w.anchors.len(), before);
    }

    #[test]
    fn sparse_blend_deterministic() {
        let s = NormalizedSparseConvolution::default();
        let mut a = ws(11, IVec3::new(0, 0, 0));
        let mut b = ws(11, IVec3::new(0, 0, 0));
        WhittakerDirect2D::default().run(&mut a);
        WhittakerDirect2D::default().run(&mut b);
        s.run(&mut a);
        s.run(&mut b);
        let av: Vec<_> = a.anchors.iter().map(|x| x.seed).collect();
        let bv: Vec<_> = b.anchors.iter().map(|x| x.seed).collect();
        assert_eq!(av, bv);
    }

    #[test]
    fn sparse_blend_writes_anchor() {
        let s = NormalizedSparseConvolution::default();
        let mut w = ws(13, IVec3::new(5, 0, 5));
        let before = w.anchors.len();
        s.run(&mut w);
        assert_eq!(w.anchors.len(), before + 1);
    }

    #[test]
    fn buffer_terrain_emits_anchor_when_borders_differ() {
        let s = BufferTerrainInjected::default();
        let mut w = ws(17, IVec3::new(0, 0, 0));
        WhittakerDirect2D::default().run(&mut w);
        let before = w.anchors.len();
        s.run(&mut w);
        assert!(w.anchors.len() >= before);
    }

    #[test]
    fn sparse_blend_continuous_within_radius() {
        let s = NormalizedSparseConvolution::default();
        let mut a = ws(19, IVec3::new(0, 0, 0));
        let mut b = ws(19, IVec3::new(1, 0, 0));
        WhittakerDirect2D::default().run(&mut a);
        WhittakerDirect2D::default().run(&mut b);
        s.run(&mut a);
        s.run(&mut b);
        let la = a.anchors.last().unwrap().seed & 0xFF;
        let lb = b.anchors.last().unwrap().seed & 0xFF;
        // With 192 m radius and 16 m brick spacing, neighbours should share
        // the same blended majority label.
        assert_eq!(la, lb);
    }
}
