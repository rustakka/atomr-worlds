//! CPU terrain generator.
//!
//! Layered heightfield + cave carving driven by FBM and Worley noise. Each
//! voxel maps from world voxel coordinates → material id deterministically.
//!
//! Phase 13c: when a [`WorldMacroState`] is present in the brick context,
//! the surface height becomes `macro_elevation_at_face + local_fbm_jitter`
//! and biome-driven material selection replaces the simple dirt-on-stone
//! palette. When macro state is `None` (legacy callers via
//! [`BrickGenerator::generate_brick_legacy`] / the CUDA path), the
//! generator runs exactly as it did in Phase 12 — bit-equal output.

use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::MetricScale;
use atomr_worlds_noise::{fbm_value, worley_noise_3d, FbmConfig};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use crate::brick::{BrickGenContext, BrickGenerator};
use crate::macro_state::{biome_id, WorldMacroState};
use crate::material_selection::{
    DynMaterialStrategy, MaterialContext, MaterialSelectionStrategy,
};

pub const MATERIAL_AIR: u16 = 0;
pub const MATERIAL_STONE: u16 = 1;
pub const MATERIAL_DIRT: u16 = 2;
pub const MATERIAL_CAVE: u16 = 0; // caves carve back to air
pub const MATERIAL_SAND: u16 = 3;
pub const MATERIAL_SNOW: u16 = 4;
pub const MATERIAL_WATER: u16 = 5;
pub const MATERIAL_GRASS: u16 = 6;
pub const MATERIAL_WOOD: u16 = 7;
pub const MATERIAL_LEAVES: u16 = 8;
pub const MATERIAL_GLOW_ROCK: u16 = 9;
pub const MATERIAL_ICE: u16 = 10;

#[derive(Copy, Clone, Debug)]
pub struct TerrainConfig {
    /// Mean terrain height (in voxels).
    pub base_height: f32,
    /// Vertical scale of FBM variation (voxels).
    pub amplitude: f32,
    /// Horizontal frequency of the heightfield (voxels per cycle ~= 1/freq).
    pub frequency: f32,
    /// FBM octaves for the heightfield.
    pub octaves: u8,
    /// Cave threshold for Worley noise; 0.0–1.0 cell-distance² — smaller = fewer caves.
    pub cave_threshold: f32,
    /// Cave noise frequency (voxels per cell).
    pub cave_frequency: f32,
    /// Thickness of the dirt layer above stone, in voxels.
    pub dirt_layer: u8,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            base_height: 32.0,
            amplitude: 24.0,
            frequency: 1.0 / 96.0,
            octaves: 4,
            cave_threshold: 0.04,
            cave_frequency: 1.0 / 24.0,
            dirt_layer: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TerrainGenerator {
    pub config: TerrainConfig,
    /// Optional pluggable material picker. When `None`, the generator
    /// runs its inlined legacy logic which is byte-for-byte identical
    /// to the CUDA kernel (preserves the cross-backend determinism
    /// guarantee). When `Some`, the geometry path is unchanged but
    /// material ids for solid voxels are delegated to the strategy.
    strategy: Option<DynMaterialStrategy>,
}

impl TerrainGenerator {
    pub fn new(config: TerrainConfig) -> Self {
        Self { config, strategy: None }
    }

    /// Attach a [`MaterialSelectionStrategy`]. Builder-style.
    pub fn with_material_strategy(mut self, strategy: DynMaterialStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Borrow the currently installed strategy, if any.
    pub fn material_strategy(&self) -> Option<&dyn MaterialSelectionStrategy> {
        self.strategy.as_deref()
    }

    pub fn default_config() -> TerrainConfig {
        TerrainConfig::default()
    }

    /// Surface height at world (x, z) in voxels.
    fn surface_height(&self, seed: u64, x: i64, z: i64) -> f32 {
        self.surface_height_world(seed, x as f32, z as f32)
    }

    /// Surface height at world meters (x, z) in voxels (= meters since
    /// LOD 0 voxels are 1 m). Continuous in `(x, z)` so coarse-LOD bricks
    /// sample the same heightfield as LOD 0 and adjacent tiers agree on
    /// surface height at chunk boundaries.
    fn surface_height_world(&self, seed: u64, x: f32, z: f32) -> f32 {
        let cfg = self.config;
        let fbm_cfg = FbmConfig {
            octaves: cfg.octaves,
            lacunarity: 2.0,
            gain: 0.5,
            frequency: 1.0,
        };
        let n = fbm_value(seed, x * cfg.frequency, 0.0, z * cfg.frequency, fbm_cfg);
        cfg.base_height + cfg.amplitude * (n * 2.0 - 1.0)
    }

    /// True if `(x, y, z)` is inside a cave.
    fn is_cave(&self, seed: u64, x: i64, y: i64, z: i64) -> bool {
        self.is_cave_world(seed, x as f32, y as f32, z as f32)
    }

    /// LOD-agnostic cave test: takes world-meter coordinates so coarse
    /// LODs sample the same Worley field as LOD 0.
    fn is_cave_world(&self, seed: u64, x: f32, y: f32, z: f32) -> bool {
        let cfg = self.config;
        let d2 = worley_noise_3d(
            seed.wrapping_add(0xC0_FE_E0_C0),
            x * cfg.cave_frequency,
            y * cfg.cave_frequency,
            z * cfg.cave_frequency,
        );
        d2 < cfg.cave_threshold
    }

    /// Material at a world voxel coordinate. Legacy path — no macro state.
    pub fn material_at(&self, world_seed: u64, p: IVec3) -> u16 {
        let surface = self.surface_height(world_seed, p.x, p.z);
        let fy = p.y as f32;
        if fy >= surface {
            return MATERIAL_AIR;
        }
        if self.is_cave(world_seed, p.x, p.y, p.z) {
            return MATERIAL_CAVE;
        }
        if fy >= surface - self.config.dirt_layer as f32 {
            MATERIAL_DIRT
        } else {
            MATERIAL_STONE
        }
    }

    /// LOD-aware material picker. `world_xyz_m` are world-meter coordinates
    /// (the LOD-0 voxel grid, so LOD 0 voxel `n` maps to world meter `n`).
    /// Sample the heightfield in continuous world-meter space so adjacent
    /// LOD tiers see the same surface — fixes the visible terrain
    /// discontinuity at chunk boundaries where the coarse mesh diverged
    /// from the fine one.
    pub fn material_at_world(
        &self,
        world_seed: u64,
        world_x_m: f32,
        world_y_m: f32,
        world_z_m: f32,
    ) -> u16 {
        let surface = self.surface_height_world(world_seed, world_x_m, world_z_m);
        if world_y_m >= surface {
            return MATERIAL_AIR;
        }
        if self.is_cave_world(world_seed, world_x_m, world_y_m, world_z_m) {
            return MATERIAL_CAVE;
        }
        if world_y_m >= surface - self.config.dirt_layer as f32 {
            MATERIAL_DIRT
        } else {
            MATERIAL_STONE
        }
    }

    /// LOD-aware strategy-driven picker. Mirrors `material_at_strategy`
    /// but uses continuous world-meter sampling; the strategy's
    /// `MaterialContext` still receives integer-rounded coordinates for
    /// backward compat with existing strategies.
    fn material_at_world_strategy(
        &self,
        strategy: &dyn MaterialSelectionStrategy,
        world_seed: u64,
        world_x_m: f32,
        world_y_m: f32,
        world_z_m: f32,
    ) -> u16 {
        let surface = self.surface_height_world(world_seed, world_x_m, world_z_m);
        if world_y_m >= surface {
            return MATERIAL_AIR;
        }
        if self.is_cave_world(world_seed, world_x_m, world_y_m, world_z_m) {
            return MATERIAL_CAVE;
        }
        let ctx = MaterialContext {
            world_seed,
            p: IVec3::new(
                world_x_m.floor() as i64,
                world_y_m.floor() as i64,
                world_z_m.floor() as i64,
            ),
            depth_below_surface_voxels: surface - world_y_m,
            dirt_layer: self.config.dirt_layer,
            biome_id: None,
        };
        strategy.pick(&ctx)
    }

    /// Strategy-driven material picker (no macro state). Solid-voxel
    /// classification is delegated to `self.strategy`; air / cave checks
    /// match the legacy path.
    fn material_at_strategy(
        &self,
        strategy: &dyn MaterialSelectionStrategy,
        world_seed: u64,
        p: IVec3,
    ) -> u16 {
        let surface = self.surface_height(world_seed, p.x, p.z);
        let fy = p.y as f32;
        if fy >= surface {
            return MATERIAL_AIR;
        }
        if self.is_cave(world_seed, p.x, p.y, p.z) {
            return MATERIAL_CAVE;
        }
        let ctx = MaterialContext {
            world_seed,
            p,
            depth_below_surface_voxels: surface - fy,
            dirt_layer: self.config.dirt_layer,
            biome_id: None,
        };
        strategy.pick(&ctx)
    }

    /// Strategy-driven material picker (with macro state). Mirrors
    /// `material_at_macro` for air / cave checks but defers the solid
    /// material choice to `self.strategy`.
    fn material_at_macro_strategy(
        &self,
        strategy: &dyn MaterialSelectionStrategy,
        world_seed: u64,
        p: IVec3,
        macro_state: &WorldMacroState,
        scale: MetricScale,
    ) -> u16 {
        let (surface, sample) = self.macro_surface_and_sample(p, macro_state, scale);
        let local = self.surface_height(world_seed, p.x, p.z) - self.config.base_height;
        let surface = surface + local;
        let fy = p.y as f32;
        if fy >= surface {
            return MATERIAL_AIR;
        }
        if self.is_cave(world_seed, p.x, p.y, p.z) {
            return MATERIAL_CAVE;
        }
        let ctx = MaterialContext {
            world_seed,
            p,
            depth_below_surface_voxels: surface - fy,
            dirt_layer: self.config.dirt_layer,
            biome_id: Some(sample.biome_id),
        };
        strategy.pick(&ctx)
    }

    /// Shared helper: project a voxel onto the macro surface and return
    /// `(macro_surface_voxels, sample)`. Pulled out so the legacy and
    /// strategy macro paths share the projection.
    fn macro_surface_and_sample(
        &self,
        p: IVec3,
        macro_state: &WorldMacroState,
        scale: MetricScale,
    ) -> (f32, crate::macro_state::MacroSample) {
        let mpv = scale.meters_per_voxel(atomr_worlds_core::Lod::new(scale.max_depth));
        let cx = scale.root_size_m * 0.5;
        let cz = scale.root_size_m * 0.5;
        let wx = p.x as f64 * mpv - cx;
        let wz = p.z as f64 * mpv - cz;
        let len2 = wx * wx + wz * wz;
        let dir = if len2 > 0.0 {
            let len = len2.sqrt();
            DVec3::new(wx / len, 0.0, wz / len)
        } else {
            DVec3::new(0.0, 1.0, 0.0)
        };
        let sample = macro_state.sample(dir);
        let macro_surface_voxels = (sample.elev_m as f64 / mpv) as f32;
        (macro_surface_voxels, sample)
    }

    /// Material at a world voxel coordinate, with macro state available.
    /// The voxel column's surface height is shifted by the macro
    /// elevation at the column's surface direction, and the top-layer
    /// material is biome-driven.
    pub fn material_at_macro(
        &self,
        world_seed: u64,
        p: IVec3,
        macro_state: &WorldMacroState,
        scale: MetricScale,
    ) -> u16 {
        let mpv = scale.meters_per_voxel(atomr_worlds_core::Lod::new(scale.max_depth));
        // Convert voxel coord → world-meter coord centered on world.
        let cx = scale.root_size_m * 0.5;
        let cy = scale.root_size_m * 0.5;
        let cz = scale.root_size_m * 0.5;
        let wx = p.x as f64 * mpv - cx;
        let _wy = p.y as f64 * mpv - cy;
        let wz = p.z as f64 * mpv - cz;
        // Project (wx, _, wz) onto the sphere's surface direction —
        // ignoring wy gives a "vertical column" sampling rule.
        let len2 = wx * wx + wz * wz;
        let dir = if len2 > 0.0 {
            let len = len2.sqrt();
            DVec3::new(wx / len, 0.0, wz / len)
        } else {
            DVec3::new(0.0, 1.0, 0.0)
        };
        let sample = macro_state.sample(dir);

        let macro_surface_voxels = (sample.elev_m as f64 / mpv) as f32;
        let local = self.surface_height(world_seed, p.x, p.z) - self.config.base_height;
        let surface = macro_surface_voxels + local;
        let fy = p.y as f32;
        if fy >= surface {
            return MATERIAL_AIR;
        }
        if self.is_cave(world_seed, p.x, p.y, p.z) {
            return MATERIAL_CAVE;
        }
        // Top layer: biome controls material; deeper voxels are stone.
        if fy >= surface - self.config.dirt_layer as f32 {
            match sample.biome_id {
                v if v == biome_id::DESERT || v == biome_id::SAVANNA => MATERIAL_SAND,
                v if v == biome_id::ICE || v == biome_id::TUNDRA => MATERIAL_SNOW,
                v if v == biome_id::OCEAN => MATERIAL_WATER,
                v if v == biome_id::MOUNTAIN => MATERIAL_STONE,
                _ => MATERIAL_DIRT,
            }
        } else {
            MATERIAL_STONE
        }
    }
}

impl BrickGenerator for TerrainGenerator {
    fn generate_brick(&self, ctx: &BrickGenContext) -> Brick {
        let edge = BRICK_EDGE as i64;
        let origin = IVec3::new(
            ctx.brick_coord.x * edge,
            ctx.brick_coord.y * edge,
            ctx.brick_coord.z * edge,
        );
        let mut brick = Brick::new();
        let strategy = self.strategy.as_deref();
        // LOD 0 = 1 m voxels (legacy contract — byte-equal to the CUDA
        // kernel). Higher LOD depths double voxel edge each step, so an
        // LOD-L voxel covers `2^L` meters and we sample noise at the
        // voxel center in world-meter space. Centering (rather than
        // lower-corner sampling) keeps surface error bounded by ±voxel/2
        // and produces visually continuous terrain across tier
        // boundaries.
        let voxel_m = (1u64 << ctx.lod.depth as u32) as f32;
        let lod_aware = ctx.lod.depth > 0;
        match ctx.macro_state.as_ref() {
            None => {
                if !lod_aware {
                    // Legacy path — preserves Phase-12 byte equality when
                    // strategy is None.
                    for lz in 0..edge {
                        for ly in 0..edge {
                            for lx in 0..edge {
                                let p = IVec3::new(origin.x + lx, origin.y + ly, origin.z + lz);
                                let mat = match strategy {
                                    None => self.material_at(ctx.world_seed, p),
                                    Some(s) => self.material_at_strategy(s, ctx.world_seed, p),
                                };
                                if mat != MATERIAL_AIR {
                                    brick.set(IVec3::new(lx, ly, lz), Voxel::new(mat));
                                }
                            }
                        }
                    }
                } else {
                    for lz in 0..edge {
                        for ly in 0..edge {
                            for lx in 0..edge {
                                // Voxel center in world meters.
                                let wx = ((origin.x + lx) as f32 + 0.5) * voxel_m;
                                let wy = ((origin.y + ly) as f32 + 0.5) * voxel_m;
                                let wz = ((origin.z + lz) as f32 + 0.5) * voxel_m;
                                let mat = match strategy {
                                    None => self.material_at_world(ctx.world_seed, wx, wy, wz),
                                    Some(s) => self.material_at_world_strategy(
                                        s,
                                        ctx.world_seed,
                                        wx,
                                        wy,
                                        wz,
                                    ),
                                };
                                if mat != MATERIAL_AIR {
                                    brick.set(IVec3::new(lx, ly, lz), Voxel::new(mat));
                                }
                            }
                        }
                    }
                }
            }
            Some(ms) => {
                for lz in 0..edge {
                    for ly in 0..edge {
                        for lx in 0..edge {
                            let p = IVec3::new(origin.x + lx, origin.y + ly, origin.z + lz);
                            let mat = match strategy {
                                None => self.material_at_macro(
                                    ctx.world_seed,
                                    p,
                                    ms,
                                    ctx.scale,
                                ),
                                Some(s) => self.material_at_macro_strategy(
                                    s,
                                    ctx.world_seed,
                                    p,
                                    ms,
                                    ctx.scale,
                                ),
                            };
                            if mat != MATERIAL_AIR {
                                brick.set(IVec3::new(lx, ly, lz), Voxel::new(mat));
                            }
                        }
                    }
                }
            }
        }
        brick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_brick_generation() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let a = gen.generate_brick_legacy(42, IVec3::new(0, 0, 0));
        let b = gen.generate_brick_legacy(42, IVec3::new(0, 0, 0));
        assert_eq!(a.nonempty_count, b.nonempty_count);
        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..16i64 {
                    let p = IVec3::new(i, j, k);
                    assert_eq!(a.get(p), b.get(p));
                }
            }
        }
    }

    #[test]
    fn high_brick_is_mostly_air() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        // y = 200 brick → world y ∈ [3200, 3216), well above base_height + amplitude.
        let b = gen.generate_brick_legacy(42, IVec3::new(0, 200, 0));
        assert_eq!(b.nonempty_count, 0);
    }

    #[test]
    fn deep_brick_has_material() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        // y = -10 brick → world y ∈ [-160, -144), deep under surface.
        let b = gen.generate_brick_legacy(42, IVec3::new(0, -10, 0));
        assert!(b.nonempty_count > 0);
    }
}
