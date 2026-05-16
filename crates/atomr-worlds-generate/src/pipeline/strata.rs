//! [`StrataStrategy`] implementations.
//!
//! Strata stages read `ws.density` (positive = solid) and write
//! `ws.materials`. The Vanilla preset routes density+strata through
//! [`super::vanilla::MonolithicTerrainPass`] to preserve byte-equality
//! with [`crate::TerrainGenerator`]; the impls here are slot-swappable
//! and consumed by Advanced and Showcase presets.
//!
//! Material ids match the constants in [`crate::terrain`] so meshing and
//! lighting paths see the same palette.

use atomr_worlds_noise::{fbm_gradient, FbmConfig};
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::terrain::{
    MATERIAL_AIR, MATERIAL_DIRT, MATERIAL_GRASS, MATERIAL_SAND, MATERIAL_STONE,
};

use super::strategies::StrataStrategy;
use super::workspace::BrickWorkspace;

const APRON_MIN: i32 = -1;
const APRON_MAX: i32 = BRICK_EDGE as i32;
const EDGE: i32 = BRICK_EDGE as i32;

#[inline]
fn for_each_brick<F: FnMut(i32, i32, i32)>(mut f: F) {
    for z in 0..EDGE {
        for y in 0..EDGE {
            for x in 0..EDGE {
                f(x, y, z);
            }
        }
    }
}

/// Topsoil strata: thin grass cap → dirt band → stone below.
///
/// `TopsoilLayer` is a documentation-only band rule; it does not attempt
/// byte-equality with the legacy `MonolithicTerrainPass`. Presets that
/// need byte-equal output keep `MonolithicTerrainPass` for both density
/// and strata.
#[derive(Clone, Debug)]
pub struct TopsoilLayer {
    pub config: TopsoilConfig,
}

#[derive(Copy, Clone, Debug)]
pub struct TopsoilConfig {
    pub grass_thickness_voxels: u8,
    pub dirt_thickness_voxels: u8,
}

impl Default for TopsoilConfig {
    fn default() -> Self {
        Self { grass_thickness_voxels: 1, dirt_thickness_voxels: 3 }
    }
}

impl Default for TopsoilLayer {
    fn default() -> Self {
        Self { config: TopsoilConfig::default() }
    }
}

impl TopsoilLayer {
    pub fn new(config: TopsoilConfig) -> Self {
        Self { config }
    }
}

#[inline]
fn column_surface_voxel(ws: &BrickWorkspace, x: i32, z: i32) -> Option<i32> {
    for y in (APRON_MIN..=APRON_MAX).rev() {
        if ws.density_at(x, y, z) > 0.0 {
            return Some(y);
        }
    }
    None
}

impl StrataStrategy for TopsoilLayer {
    fn id(&self) -> &'static str {
        "TopsoilLayer"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let cfg = self.config;
        for_each_brick(|x, y, z| {
            if ws.density_at(x, y, z) <= 0.0 {
                ws.set_material(x, y, z, Voxel::new(MATERIAL_AIR));
                return;
            }
            let surface = column_surface_voxel(ws, x, z).unwrap_or(y);
            let depth = surface - y;
            let mat = if depth < cfg.grass_thickness_voxels as i32 {
                MATERIAL_GRASS
            } else if depth
                < (cfg.grass_thickness_voxels as i32 + cfg.dirt_thickness_voxels as i32)
            {
                MATERIAL_DIRT
            } else {
                MATERIAL_STONE
            };
            ws.set_material(x, y, z, Voxel::new(mat));
        });
        copy_materials_to_brick(ws);
    }
}

/// One geological band: material id, half-open depth range below the
/// column surface (voxels), and a per-band fBm boundary perturbation.
#[derive(Copy, Clone, Debug)]
pub struct StratumBand {
    pub material: u16,
    pub depth_top_voxels: f32,
    pub depth_bottom_voxels: f32,
    pub boundary_jitter_m: f32,
    pub boundary_frequency: f32,
}

#[derive(Clone, Debug)]
pub struct StrataConfig {
    pub bands: Vec<StratumBand>,
    pub fallback: u16,
}

impl Default for StrataConfig {
    fn default() -> Self {
        Self {
            bands: vec![
                StratumBand {
                    material: MATERIAL_GRASS,
                    depth_top_voxels: 0.0,
                    depth_bottom_voxels: 1.0,
                    boundary_jitter_m: 0.5,
                    boundary_frequency: 1.0 / 12.0,
                },
                StratumBand {
                    material: MATERIAL_DIRT,
                    depth_top_voxels: 1.0,
                    depth_bottom_voxels: 4.0,
                    boundary_jitter_m: 1.0,
                    boundary_frequency: 1.0 / 24.0,
                },
                StratumBand {
                    material: MATERIAL_SAND,
                    depth_top_voxels: 4.0,
                    depth_bottom_voxels: 6.0,
                    boundary_jitter_m: 1.0,
                    boundary_frequency: 1.0 / 18.0,
                },
                StratumBand {
                    material: MATERIAL_STONE,
                    depth_top_voxels: 6.0,
                    depth_bottom_voxels: f32::INFINITY,
                    boundary_jitter_m: 0.0,
                    boundary_frequency: 1.0,
                },
            ],
            fallback: MATERIAL_STONE,
        }
    }
}

/// Depth-banded materials with per-band fBm boundary perturbation.
#[derive(Clone, Debug)]
pub struct LayeredGeology {
    pub config: StrataConfig,
}

impl Default for LayeredGeology {
    fn default() -> Self {
        Self { config: StrataConfig::default() }
    }
}

impl LayeredGeology {
    pub fn new(config: StrataConfig) -> Self {
        Self { config }
    }

    fn pick_material(&self, seed: u64, world_p: [f32; 3], depth_voxels: f32) -> u16 {
        let fbm = FbmConfig { octaves: 2, lacunarity: 2.0, gain: 0.5, frequency: 1.0 };
        for band in &self.config.bands {
            let jitter = if band.boundary_jitter_m > 0.0 {
                let n = fbm_gradient(
                    seed ^ 0xC0DE_FEED_BEEF_F00D,
                    world_p[0] * band.boundary_frequency,
                    world_p[1] * band.boundary_frequency,
                    world_p[2] * band.boundary_frequency,
                    fbm,
                );
                n * band.boundary_jitter_m
            } else {
                0.0
            };
            let top = band.depth_top_voxels + jitter;
            let bot = band.depth_bottom_voxels + jitter;
            if depth_voxels >= top && depth_voxels < bot {
                return band.material;
            }
        }
        self.config.fallback
    }
}

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
fn voxel_world_pos(ws: &BrickWorkspace, x: i32, y: i32, z: i32) -> [f32; 3] {
    let (ox, oy, oz) = brick_origin_world(ws);
    let v = (1u64 << ws.ctx.lod.depth as u32) as f32;
    [ox + (x as f32 + 0.5) * v, oy + (y as f32 + 0.5) * v, oz + (z as f32 + 0.5) * v]
}

impl StrataStrategy for LayeredGeology {
    fn id(&self) -> &'static str {
        "LayeredGeology"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        let seed = ws.ctx.world_seed;
        for_each_brick(|x, y, z| {
            if ws.density_at(x, y, z) <= 0.0 {
                ws.set_material(x, y, z, Voxel::new(MATERIAL_AIR));
                return;
            }
            let surface = column_surface_voxel(ws, x, z).unwrap_or(y);
            let depth = (surface - y) as f32;
            let p = voxel_world_pos(ws, x, y, z);
            let mat = self.pick_material(seed, p, depth);
            ws.set_material(x, y, z, Voxel::new(mat));
        });
        copy_materials_to_brick(ws);
    }
}

/// Kriging-interpolated strata (stub).
///
/// Full Empirical Bayesian Kriging interpolates between irregularly-sampled
/// control points using a fitted variogram; the math is
/// `Z*(p) = sum_i w_i * Z(x_i)` where weights `w` solve
/// `Sigma * w = sigma(p)` under unbiasedness `sum w_i = 1`. With no
/// control-points file configured, this impl falls back to
/// [`LayeredGeology`]; the variant exists to occupy the strategy slot so
/// downstream code can opt into kriging once the loader lands.
#[derive(Clone, Debug, Default)]
pub struct KrigingInterpolated {
    fallback: LayeredGeology,
}

impl KrigingInterpolated {
    pub fn new() -> Self {
        Self::default()
    }
}

impl StrataStrategy for KrigingInterpolated {
    fn id(&self) -> &'static str {
        "KrigingInterpolated"
    }
    fn run(&self, ws: &mut BrickWorkspace) {
        self.fallback.run(ws);
    }
}

#[inline]
fn copy_materials_to_brick(ws: &mut BrickWorkspace) {
    use atomr_worlds_core::coord::IVec3;
    for_each_brick(|x, y, z| {
        let m = ws.material_at(x, y, z);
        if m.0 != MATERIAL_AIR {
            ws.brick.set(IVec3::new(x as i64, y as i64, z as i64), m);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::density::HeightmapPlanar;
    use crate::pipeline::strategies::DensityFieldStrategy as _;
    use atomr_worlds_core::coord::IVec3;

    fn run_density_only(seed: u64, coord: IVec3) -> BrickWorkspace {
        let mut w = BrickWorkspace::new(BrickGenContext::legacy(seed, coord));
        HeightmapPlanar::default().run(&mut w);
        w
    }

    fn collect_materials(ws: &BrickWorkspace) -> Vec<u16> {
        let mut out = Vec::new();
        for_each_brick(|x, y, z| out.push(ws.material_at(x, y, z).0));
        out
    }

    #[test]
    fn topsoil_deterministic() {
        let s = TopsoilLayer::default();
        let mut a = run_density_only(7, IVec3::new(0, 0, 0));
        let mut b = run_density_only(7, IVec3::new(0, 0, 0));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(collect_materials(&a), collect_materials(&b));
    }

    #[test]
    fn topsoil_emits_grass_at_surface() {
        // Heightmap base is 32 m; brick y=2 spans world y [32, 48) so it
        // straddles the surface and the strata pass must emit grass at the
        // surface band.
        let s = TopsoilLayer::default();
        let mut low = run_density_only(11, IVec3::new(0, 2, 0));
        s.run(&mut low);
        let mut saw_grass = false;
        for_each_brick(|x, y, z| {
            if low.material_at(x, y, z).0 == MATERIAL_GRASS {
                saw_grass = true;
            }
        });
        assert!(saw_grass, "expected at least one grass voxel in a ground-level brick");
    }

    #[test]
    fn layered_deterministic() {
        let s = LayeredGeology::default();
        let mut a = run_density_only(13, IVec3::new(2, 0, -1));
        let mut b = run_density_only(13, IVec3::new(2, 0, -1));
        s.run(&mut a);
        s.run(&mut b);
        assert_eq!(collect_materials(&a), collect_materials(&b));
    }

    #[test]
    fn layered_bands_descend_to_stone() {
        let s = LayeredGeology::default();
        let mut w = run_density_only(17, IVec3::new(0, 0, 0));
        s.run(&mut w);
        // Deep voxels (depth » band depths) must be stone.
        let mut saw_stone = false;
        for_each_brick(|x, y, z| {
            if w.material_at(x, y, z).0 == MATERIAL_STONE {
                saw_stone = true;
            }
        });
        assert!(saw_stone, "expected stone band somewhere in a low brick");
    }

    #[test]
    fn kriging_falls_back_to_layered_geology() {
        let kriging = KrigingInterpolated::new();
        let layered = LayeredGeology::default();
        let mut a = run_density_only(19, IVec3::new(0, 0, 0));
        let mut b = run_density_only(19, IVec3::new(0, 0, 0));
        kriging.run(&mut a);
        layered.run(&mut b);
        assert_eq!(collect_materials(&a), collect_materials(&b));
    }
}
