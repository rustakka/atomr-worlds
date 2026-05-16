//! [`BiomeMatrixStrategy`] implementations.
//!
//! Biome matrix stages publish a per-workspace biome label (currently
//! recorded as anchor-style metadata) without mutating the brick. The
//! Vanilla preset uses [`PerFaceWhittaker`] which is a no-op aside from
//! reading the macro state already produced by
//! [`crate::DefaultMacroGenerator`]; the alternative impls compute biome
//! labels directly from temperature × humidity sampling
//! ([`WhittakerDirect2D`]) or from Voronoi cell centers ([`VoronoiCells`]).
//!
//! Output convention: each strategy writes one [`FeatureKind::BufferTerrain`]
//! anchor with `seed` set to the brick's dominant biome id (cast to `u64`)
//! at `origin_m = (cx, 0, cz)` for the brick center. Downstream blend and
//! strata stages can read this label without re-running matrix sampling.

use atomr_worlds_core::seed::child_seed;
use atomr_worlds_voxel::BRICK_EDGE;

use crate::macro_state::{biome_id, MacroSample};

use super::anchor::{FeatureAnchor, FeatureKind};
use super::strategies::BiomeMatrixStrategy;
use super::workspace::BrickWorkspace;

const VORONOI_DIM: u32 = 0x5101_C0DE;

#[inline]
fn brick_origin_world(ws: &BrickWorkspace) -> (f32, f32, f32) {
    let edge = BRICK_EDGE as f32;
    let v = (1u64 << ws.ctx.lod.depth as u32) as f32;
    (
        ws.ctx.brick_coord.x as f32 * edge * v,
        ws.ctx.brick_coord.y as f32 * edge * v,
        ws.ctx.brick_coord.z as f32 * edge * v,
    )
}

#[inline]
fn brick_center_xz_m(ws: &BrickWorkspace) -> (f32, f32) {
    let (ox, _, oz) = brick_origin_world(ws);
    let half = (BRICK_EDGE as f32) * (1u64 << ws.ctx.lod.depth as u32) as f32 * 0.5;
    (ox + half, oz + half)
}

fn push_biome_label(ws: &mut BrickWorkspace, biome: u8) {
    let (cx, cz) = brick_center_xz_m(ws);
    ws.anchors.push(FeatureAnchor {
        kind: FeatureKind::BufferTerrain,
        column: ws.ctx.brick_coord,
        origin_m: [cx, 0.0, cz],
        seed: biome as u64,
    });
}

/// Per-face Whittaker: read the brick's biome from
/// [`crate::WorldMacroState`]. This is the current Vanilla behavior — a
/// no-op aside from publishing the label so downstream stages can read it
/// without re-touching the macro state.
#[derive(Clone, Debug, Default)]
pub struct PerFaceWhittaker;

impl PerFaceWhittaker {
    pub fn new() -> Self {
        Self
    }

    fn sample_macro(&self, ws: &BrickWorkspace) -> Option<MacroSample> {
        let ms = ws.ctx.macro_state.as_ref()?;
        let (cx, cz) = brick_center_xz_m(ws);
        let len2 = (cx as f64).powi(2) + (cz as f64).powi(2);
        let dir = if len2 > 1e-12 {
            let len = len2.sqrt();
            atomr_worlds_core::coord::DVec3::new(cx as f64 / len, 0.0, cz as f64 / len)
        } else {
            atomr_worlds_core::coord::DVec3::new(0.0, 1.0, 0.0)
        };
        Some(ms.sample(dir))
    }
}

impl BiomeMatrixStrategy for PerFaceWhittaker {
    fn id(&self) -> &'static str {
        "PerFaceWhittaker"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let biome = self.sample_macro(ws).map(|s| s.biome_id).unwrap_or(biome_id::GRASSLAND);
        push_biome_label(ws, biome);
    }
}

/// Direct 2D Whittaker: sample temperature and humidity from analytic
/// functions of `(x, z)` (latitude band + per-column humidity noise) and
/// classify with the same threshold table used by
/// [`crate::macro_state::biome::classify_biomes`].
#[derive(Clone, Debug)]
pub struct WhittakerDirect2D {
    pub config: WhittakerDirect2DConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct WhittakerDirect2DConfig {
    pub temperature_lapse_per_m: f32,
    pub base_temperature_c: f32,
    pub humidity_frequency: f32,
}

impl Default for WhittakerDirect2DConfig {
    fn default() -> Self {
        Self {
            temperature_lapse_per_m: 0.0065,
            base_temperature_c: 15.0,
            humidity_frequency: 1.0 / 256.0,
        }
    }
}

impl Default for WhittakerDirect2D {
    fn default() -> Self {
        Self { config: WhittakerDirect2DConfig::default() }
    }
}

impl WhittakerDirect2D {
    pub fn new(config: WhittakerDirect2DConfig) -> Self {
        Self { config }
    }

    fn classify(temp_c: f32, humidity: f32) -> u8 {
        if temp_c < -10.0 {
            biome_id::ICE
        } else if temp_c < 0.0 {
            biome_id::TUNDRA
        } else if temp_c < 5.0 {
            biome_id::TAIGA
        } else if humidity < 0.15 {
            if temp_c > 20.0 { biome_id::DESERT } else { biome_id::GRASSLAND }
        } else if humidity < 0.5 {
            if temp_c > 20.0 { biome_id::SAVANNA } else { biome_id::TEMPERATE_FOREST }
        } else if temp_c > 22.0 {
            biome_id::RAINFOREST
        } else {
            biome_id::TEMPERATE_FOREST
        }
    }
}

impl BiomeMatrixStrategy for WhittakerDirect2D {
    fn id(&self) -> &'static str {
        "WhittakerDirect2D"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let cfg = self.config;
        let seed = ws.ctx.world_seed;
        let (cx, cz) = brick_center_xz_m(ws);
        let humidity = atomr_worlds_noise::fbm_value(
            seed ^ 0x4D_8B_2F_3A_9C_DE_77_05,
            cx * cfg.humidity_frequency,
            0.0,
            cz * cfg.humidity_frequency,
            atomr_worlds_noise::FbmConfig { octaves: 3, lacunarity: 2.0, gain: 0.5, frequency: 1.0 },
        );
        let temp = cfg.base_temperature_c
            - cfg.temperature_lapse_per_m
                * ws.ctx.brick_coord.y as f32
                * BRICK_EDGE as f32
                * (1u64 << ws.ctx.lod.depth as u32) as f32;
        let biome = Self::classify(temp, humidity);
        push_biome_label(ws, biome);
    }
}

/// Voronoi cells: lay out cell centers on a coarse grid keyed by
/// `child_seed`, find the nearest center to the brick, and assign the
/// biome carried by that center.
#[derive(Clone, Debug)]
pub struct VoronoiCells {
    pub config: VoronoiCellsConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct VoronoiCellsConfig {
    pub cell_spacing_m: f32,
}

impl Default for VoronoiCellsConfig {
    fn default() -> Self {
        Self { cell_spacing_m: 512.0 }
    }
}

impl Default for VoronoiCells {
    fn default() -> Self {
        Self { config: VoronoiCellsConfig::default() }
    }
}

impl VoronoiCells {
    pub fn new(config: VoronoiCellsConfig) -> Self {
        Self { config }
    }

    fn cell_origin(world_seed: u64, cell: atomr_worlds_core::coord::IVec3) -> ([f32; 2], u8) {
        let s = child_seed(world_seed, VORONOI_DIM, cell);
        let jitter_x = ((s & 0xFFFF) as f32) / 65535.0;
        let jitter_z = (((s >> 16) & 0xFFFF) as f32) / 65535.0;
        let biome = ((s >> 32) % 9) as u8 + 1;
        let biome = biome.min(biome_id::MOUNTAIN);
        ([jitter_x, jitter_z], biome)
    }
}

impl BiomeMatrixStrategy for VoronoiCells {
    fn id(&self) -> &'static str {
        "VoronoiCells"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let cfg = self.config;
        let seed = ws.ctx.world_seed;
        let (cx, cz) = brick_center_xz_m(ws);
        let inv = 1.0 / cfg.cell_spacing_m.max(1e-3);
        let qx = (cx * inv).floor() as i64;
        let qz = (cz * inv).floor() as i64;
        let mut best = (f32::INFINITY, biome_id::GRASSLAND);
        for dz in -1..=1 {
            for dx in -1..=1 {
                let cell = atomr_worlds_core::coord::IVec3::new(qx + dx, 0, qz + dz);
                let (jit, biome) = Self::cell_origin(seed, cell);
                let center_x = (cell.x as f32 + jit[0]) * cfg.cell_spacing_m;
                let center_z = (cell.z as f32 + jit[1]) * cfg.cell_spacing_m;
                let d2 = (cx - center_x).powi(2) + (cz - center_z).powi(2);
                if d2 < best.0 {
                    best = (d2, biome);
                }
            }
        }
        push_biome_label(ws, best.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use atomr_worlds_core::coord::IVec3;

    fn ws(seed: u64, coord: IVec3) -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(seed, coord))
    }

    fn label(ws: &BrickWorkspace) -> u8 {
        ws.anchors
            .iter()
            .find(|a| a.kind == FeatureKind::BufferTerrain)
            .map(|a| a.seed as u8)
            .expect("biome label missing")
    }

    #[test]
    fn whittaker_deterministic() {
        let s = WhittakerDirect2D::default();
        let mut a = ws(7, IVec3::new(0, 0, 0));
        let mut b = ws(7, IVec3::new(0, 0, 0));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(label(&a), label(&b));
    }

    #[test]
    fn voronoi_deterministic() {
        let s = VoronoiCells::default();
        let mut a = ws(11, IVec3::new(3, 0, -1));
        let mut b = ws(11, IVec3::new(3, 0, -1));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(label(&a), label(&b));
    }

    #[test]
    fn voronoi_continuity_within_cell() {
        let s = VoronoiCells::default();
        let mut a = ws(13, IVec3::new(0, 0, 0));
        let mut b = ws(13, IVec3::new(1, 0, 0));
        s.run(&mut a);
        s.run(&mut b);
        // Neighbouring bricks (16 m apart) should fall in the same Voronoi
        // cell (spacing 512 m) and therefore receive the same biome label.
        assert_eq!(label(&a), label(&b));
    }

    #[test]
    fn per_face_whittaker_falls_back_without_macro() {
        let s = PerFaceWhittaker::default();
        let mut w = ws(17, IVec3::new(0, 0, 0));
        s.run(&mut w);
        assert_eq!(label(&w), biome_id::GRASSLAND);
    }
}
