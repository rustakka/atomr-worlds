//! [`DensityFieldStrategy`] implementations.
//!
//! Each impl fills the padded 18³ `ws.density` apron. The sign convention
//! matches [`super::strategies::DensityFieldStrategy`]: positive = solid,
//! negative = empty, zero = surface.
//!
//! The Vanilla preset keeps using [`super::vanilla::MonolithicTerrainPass`]
//! for byte-for-byte parity with [`crate::TerrainGenerator`]; the impls
//! here populate Advanced and Showcase presets and are slot-swappable via
//! [`super::registry::apply_worldgen_strategy_by_name`].

use atomr_worlds_noise::{
    fbm_gradient, fbm_value, island_density, iterated_warp, warp_point, FbmConfig,
    FloatingIslandConfig, WarpConfig,
};
use atomr_worlds_voxel::BRICK_EDGE;

use super::anchor::FeatureKind;
use super::strategies::DensityFieldStrategy;
use super::workspace::BrickWorkspace;

const APRON_MIN: i32 = -1;
const APRON_MAX: i32 = BRICK_EDGE as i32;

#[inline]
fn brick_origin_m(ws: &BrickWorkspace) -> (f32, f32, f32) {
    let edge = BRICK_EDGE as f32;
    let v = (1u64 << ws.ctx.lod.depth as u32) as f32;
    (
        ws.ctx.brick_coord.x as f32 * edge * v,
        ws.ctx.brick_coord.y as f32 * edge * v,
        ws.ctx.brick_coord.z as f32 * edge * v,
    )
}

#[inline]
fn voxel_world_pos(ws: &BrickWorkspace, x: i32, y: i32, z: i32) -> [f32; 3] {
    let (ox, oy, oz) = brick_origin_m(ws);
    let v = (1u64 << ws.ctx.lod.depth as u32) as f32;
    [
        ox + (x as f32 + 0.5) * v,
        oy + (y as f32 + 0.5) * v,
        oz + (z as f32 + 0.5) * v,
    ]
}

#[inline]
fn for_each_apron<F: FnMut(i32, i32, i32)>(mut f: F) {
    for z in APRON_MIN..=APRON_MAX {
        for y in APRON_MIN..=APRON_MAX {
            for x in APRON_MIN..=APRON_MAX {
                f(x, y, z);
            }
        }
    }
}

/// Heightmap-style density: positive below a 2-D surface, negative above.
///
/// Documented as a slot occupant for non-Vanilla presets that do not need
/// byte-equality with [`crate::TerrainGenerator`]. The Vanilla preset keeps
/// using [`super::vanilla::MonolithicTerrainPass`] (a single combined
/// density+strata pass) for snapshot parity. Splitting the legacy
/// generator's interleaved heightmap / cave / river logic into a clean
/// density step would change the order of float ops and therefore break
/// the snapshot.
#[derive(Clone, Debug)]
pub struct HeightmapPlanar {
    pub config: HeightmapPlanarConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct HeightmapPlanarConfig {
    pub base_height_m: f32,
    pub amplitude_m: f32,
    pub frequency: f32,
    pub octaves: u8,
}

impl Default for HeightmapPlanarConfig {
    fn default() -> Self {
        Self { base_height_m: 32.0, amplitude_m: 24.0, frequency: 1.0 / 96.0, octaves: 4 }
    }
}

impl Default for HeightmapPlanar {
    fn default() -> Self {
        Self { config: HeightmapPlanarConfig::default() }
    }
}

impl HeightmapPlanar {
    pub fn new(config: HeightmapPlanarConfig) -> Self {
        Self { config }
    }
}

impl DensityFieldStrategy for HeightmapPlanar {
    fn id(&self) -> &'static str {
        "HeightmapPlanar"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let seed = ws.ctx.world_seed;
        let cfg = self.config;
        let fbm = FbmConfig {
            octaves: cfg.octaves,
            lacunarity: 2.0,
            gain: 0.5,
            frequency: 1.0,
        };
        for_each_apron(|x, y, z| {
            let p = voxel_world_pos(ws, x, y, z);
            let n = fbm_value(seed, p[0] * cfg.frequency, 0.0, p[2] * cfg.frequency, fbm);
            let surface = cfg.base_height_m + cfg.amplitude_m * (n * 2.0 - 1.0);
            ws.set_density(x, y, z, surface - p[1]);
        });
    }
}

/// `density = baseHeight(x,z) - y + noise3D(x,y,z)`. Hybrid 2D heightmap
/// plus a 3D perturbation that produces natural overhangs and small
/// detached blobs.
#[derive(Clone, Debug)]
pub struct Hybrid2D3D {
    pub config: Hybrid2D3DConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct Hybrid2D3DConfig {
    pub base_height_m: f32,
    pub amplitude_m: f32,
    pub frequency_2d: f32,
    pub frequency_3d: f32,
    pub overhang_strength_m: f32,
    pub octaves: u8,
    pub warp: WarpConfig,
}

impl Default for Hybrid2D3DConfig {
    fn default() -> Self {
        Self {
            base_height_m: 48.0,
            amplitude_m: 32.0,
            frequency_2d: 1.0 / 96.0,
            frequency_3d: 1.0 / 48.0,
            overhang_strength_m: 18.0,
            octaves: 4,
            warp: WarpConfig { amplitude: 4.0, frequency: 1.0 / 96.0, ..WarpConfig::default() },
        }
    }
}

impl Default for Hybrid2D3D {
    fn default() -> Self {
        Self { config: Hybrid2D3DConfig::default() }
    }
}

impl Hybrid2D3D {
    pub fn new(config: Hybrid2D3DConfig) -> Self {
        Self { config }
    }
}

impl DensityFieldStrategy for Hybrid2D3D {
    fn id(&self) -> &'static str {
        "Hybrid2D3D"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let seed = ws.ctx.world_seed;
        let cfg = self.config;
        let fbm_2d = FbmConfig { octaves: cfg.octaves, lacunarity: 2.0, gain: 0.5, frequency: 1.0 };
        let fbm_3d = FbmConfig {
            octaves: cfg.octaves.saturating_sub(1).max(1),
            lacunarity: 2.0,
            gain: 0.5,
            frequency: 1.0,
        };
        for_each_apron(|x, y, z| {
            let p = voxel_world_pos(ws, x, y, z);
            let n2 = fbm_value(seed, p[0] * cfg.frequency_2d, 0.0, p[2] * cfg.frequency_2d, fbm_2d);
            let base = cfg.base_height_m + cfg.amplitude_m * (n2 * 2.0 - 1.0);
            let warped = warp_point(seed, p, cfg.warp);
            let n3 = fbm_gradient(
                seed,
                warped[0] * cfg.frequency_3d,
                warped[1] * cfg.frequency_3d,
                warped[2] * cfg.frequency_3d,
                fbm_3d,
            );
            ws.set_density(x, y, z, base - p[1] + n3 * cfg.overhang_strength_m);
        });
    }
}

/// Pure 3D density via fBm value noise with a vertical bias profile that
/// closes the field at the world floor and a high ceiling. Supports
/// arbitrary topology — floating arches, multi-tier overhangs, etc.
#[derive(Clone, Debug)]
pub struct Pure3DOverhang {
    pub config: Pure3DOverhangConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct Pure3DOverhangConfig {
    pub frequency: f32,
    pub octaves: u8,
    pub bias_center_m: f32,
    pub bias_falloff_m: f32,
    pub density_scale: f32,
    pub warp: WarpConfig,
}

impl Default for Pure3DOverhangConfig {
    fn default() -> Self {
        Self {
            frequency: 1.0 / 64.0,
            octaves: 4,
            bias_center_m: 32.0,
            bias_falloff_m: 96.0,
            density_scale: 1.0,
            warp: WarpConfig { amplitude: 6.0, frequency: 1.0 / 80.0, ..WarpConfig::default() },
        }
    }
}

impl Default for Pure3DOverhang {
    fn default() -> Self {
        Self { config: Pure3DOverhangConfig::default() }
    }
}

impl Pure3DOverhang {
    pub fn new(config: Pure3DOverhangConfig) -> Self {
        Self { config }
    }

    #[inline]
    fn vertical_bias(&self, y_m: f32) -> f32 {
        let cfg = self.config;
        let t = (y_m - cfg.bias_center_m) / cfg.bias_falloff_m.max(1e-3);
        -t
    }
}

impl DensityFieldStrategy for Pure3DOverhang {
    fn id(&self) -> &'static str {
        "Pure3DOverhang"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let seed = ws.ctx.world_seed;
        let cfg = self.config;
        let fbm = FbmConfig { octaves: cfg.octaves, lacunarity: 2.0, gain: 0.5, frequency: 1.0 };
        for_each_apron(|x, y, z| {
            let p = voxel_world_pos(ws, x, y, z);
            let q = iterated_warp(seed, p, cfg.warp);
            let n = fbm_gradient(
                seed,
                q[0] * cfg.frequency,
                q[1] * cfg.frequency,
                q[2] * cfg.frequency,
                fbm,
            );
            let bias = self.vertical_bias(p[1]);
            ws.set_density(x, y, z, n * cfg.density_scale + bias);
        });
    }
}

/// Floating-island density: combine
/// [`atomr_worlds_noise::island_density`] over every
/// [`FeatureKind::FloatingIsland`] anchor present in `ws.anchors`, plus a
/// secondary 3-D noise that perturbs bottom-hemisphere stalactites.
///
/// The field is the per-anchor maximum so islands don't subtract from each
/// other near boundaries; bricks with zero island anchors stay fully
/// empty (`-1.0` density everywhere).
#[derive(Clone, Debug)]
pub struct FloatingIslandField {
    pub config: FloatingIslandFieldConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct FloatingIslandFieldConfig {
    pub island: FloatingIslandConfig,
    pub background: f32,
}

impl Default for FloatingIslandFieldConfig {
    fn default() -> Self {
        Self { island: FloatingIslandConfig::default(), background: -1.0 }
    }
}

impl Default for FloatingIslandField {
    fn default() -> Self {
        Self { config: FloatingIslandFieldConfig::default() }
    }
}

impl FloatingIslandField {
    pub fn new(config: FloatingIslandFieldConfig) -> Self {
        Self { config }
    }
}

impl DensityFieldStrategy for FloatingIslandField {
    fn id(&self) -> &'static str {
        "FloatingIslandField"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let cfg = self.config;
        let islands: Vec<_> = ws
            .anchors
            .iter()
            .filter(|a| a.kind == FeatureKind::FloatingIsland)
            .copied()
            .collect();

        for_each_apron(|x, y, z| {
            let p = voxel_world_pos(ws, x, y, z);
            let mut best = cfg.background;
            for a in &islands {
                let d = island_density(a.seed, p, a.origin_m, cfg.island);
                if d > best {
                    best = d;
                }
            }
            ws.set_density(x, y, z, best);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::anchor::FeatureAnchor;
    use atomr_worlds_core::coord::IVec3;

    fn ws(seed: u64, coord: IVec3) -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(seed, coord))
    }

    fn sample_density(ws: &BrickWorkspace) -> Vec<f32> {
        let mut out = Vec::new();
        for_each_apron(|x, y, z| out.push(ws.density_at(x, y, z)));
        out
    }

    #[test]
    fn heightmap_deterministic() {
        let s = HeightmapPlanar::default();
        let mut a = ws(7, IVec3::new(0, 0, 0));
        let mut b = ws(7, IVec3::new(0, 0, 0));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(sample_density(&a), sample_density(&b));
    }

    #[test]
    fn heightmap_continuity_across_brick_edge() {
        // Apron x=BRICK_EDGE in brick A is the same world voxel as brick
        // voxel x=0 in the +X neighbour — both sample the column at world
        // position `(BRICK_EDGE + 0.5) * voxel_m`.
        let s = HeightmapPlanar::default();
        let mut a = ws(11, IVec3::new(0, 0, 0));
        let mut b = ws(11, IVec3::new(1, 0, 0));
        s.run(&mut a);
        s.run(&mut b);
        for y in 0..BRICK_EDGE as i32 {
            for z in 0..BRICK_EDGE as i32 {
                let right = a.density_at(BRICK_EDGE as i32, y, z);
                let left = b.density_at(0, y, z);
                assert!(
                    (right - left).abs() < 1e-4,
                    "edge mismatch at y={y} z={z}: {right} vs {left}"
                );
            }
        }
    }

    #[test]
    fn hybrid_deterministic() {
        let s = Hybrid2D3D::default();
        let mut a = ws(13, IVec3::new(2, 0, -1));
        let mut b = ws(13, IVec3::new(2, 0, -1));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(sample_density(&a), sample_density(&b));
    }

    #[test]
    fn hybrid_continuity_across_brick_edge() {
        let s = Hybrid2D3D::default();
        let mut a = ws(17, IVec3::new(0, 0, 0));
        let mut b = ws(17, IVec3::new(0, 0, 1));
        s.run(&mut a);
        s.run(&mut b);
        for y in 0..BRICK_EDGE as i32 {
            for x in 0..BRICK_EDGE as i32 {
                let right = a.density_at(x, y, BRICK_EDGE as i32);
                let left = b.density_at(x, y, 0);
                assert!(
                    (right - left).abs() < 1e-4,
                    "edge mismatch at x={x} y={y}: {right} vs {left}"
                );
            }
        }
    }

    #[test]
    fn pure3d_deterministic() {
        let s = Pure3DOverhang::default();
        let mut a = ws(23, IVec3::new(-3, 1, 2));
        let mut b = ws(23, IVec3::new(-3, 1, 2));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(sample_density(&a), sample_density(&b));
    }

    #[test]
    fn pure3d_continuity_across_brick_edge() {
        let s = Pure3DOverhang::default();
        let mut a = ws(29, IVec3::new(0, 0, 0));
        let mut b = ws(29, IVec3::new(0, 1, 0));
        s.run(&mut a);
        s.run(&mut b);
        for x in 0..BRICK_EDGE as i32 {
            for z in 0..BRICK_EDGE as i32 {
                let top = a.density_at(x, BRICK_EDGE as i32, z);
                let bot = b.density_at(x, 0, z);
                assert!(
                    (top - bot).abs() < 1e-4,
                    "edge mismatch at x={x} z={z}: {top} vs {bot}"
                );
            }
        }
    }

    #[test]
    fn island_empty_without_anchors() {
        let s = FloatingIslandField::default();
        let mut w = ws(31, IVec3::new(0, 0, 0));
        s.run(&mut w);
        for d in sample_density(&w) {
            assert!(d <= 0.0, "expected non-positive density without anchors, got {d}");
        }
    }

    #[test]
    fn island_solid_at_anchor_center() {
        let s = FloatingIslandField::default();
        let mut w = ws(37, IVec3::new(0, 0, 0));
        let pos = voxel_world_pos(&w, 8, 8, 8);
        w.anchors.push(FeatureAnchor {
            kind: FeatureKind::FloatingIsland,
            column: IVec3::new(0, 0, 0),
            origin_m: pos,
            seed: 0xABCD_EF01,
        });
        s.run(&mut w);
        assert!(w.density_at(8, 8, 8) > 0.0);
    }

    #[test]
    fn island_deterministic_with_anchors() {
        let s = FloatingIslandField::default();
        let make = |coord| {
            let mut w = ws(41, coord);
            w.anchors.push(FeatureAnchor {
                kind: FeatureKind::FloatingIsland,
                column: IVec3::new(0, 0, 0),
                origin_m: voxel_world_pos(&w, 4, 4, 4),
                seed: 0xFEED_FACE,
            });
            s.run(&mut w);
            sample_density(&w)
        };
        assert_eq!(make(IVec3::new(0, 0, 0)), make(IVec3::new(0, 0, 0)));
    }
}
