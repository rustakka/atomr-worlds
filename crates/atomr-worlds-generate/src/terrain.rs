//! CPU terrain generator.
//!
//! Layered heightfield + cave carving driven by FBM and Worley noise. Each
//! voxel maps from world voxel coordinates → material id deterministically.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_noise::{fbm_value, worley_noise_3d, FbmConfig};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use crate::brick::BrickGenerator;

pub const MATERIAL_AIR: u16 = 0;
pub const MATERIAL_STONE: u16 = 1;
pub const MATERIAL_DIRT: u16 = 2;
pub const MATERIAL_CAVE: u16 = 0; // caves carve back to air

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
}

impl TerrainGenerator {
    pub fn new(config: TerrainConfig) -> Self {
        Self { config }
    }

    pub fn default_config() -> TerrainConfig {
        TerrainConfig::default()
    }

    /// Surface height at world (x, z) in voxels.
    fn surface_height(&self, seed: u64, x: i64, z: i64) -> f32 {
        let cfg = self.config;
        let fbm_cfg = FbmConfig {
            octaves: cfg.octaves,
            lacunarity: 2.0,
            gain: 0.5,
            frequency: 1.0,
        };
        let n = fbm_value(seed, x as f32 * cfg.frequency, 0.0, z as f32 * cfg.frequency, fbm_cfg);
        cfg.base_height + cfg.amplitude * (n * 2.0 - 1.0)
    }

    /// True if `(x, y, z)` is inside a cave.
    fn is_cave(&self, seed: u64, x: i64, y: i64, z: i64) -> bool {
        let cfg = self.config;
        let d2 = worley_noise_3d(
            seed.wrapping_add(0xC0_FE_E0_C0),
            x as f32 * cfg.cave_frequency,
            y as f32 * cfg.cave_frequency,
            z as f32 * cfg.cave_frequency,
        );
        d2 < cfg.cave_threshold
    }

    /// Material at a world voxel coordinate.
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
}

impl BrickGenerator for TerrainGenerator {
    fn generate_brick(&self, world_seed: u64, brick_coord: IVec3) -> Brick {
        let edge = BRICK_EDGE as i64;
        let origin = IVec3::new(brick_coord.x * edge, brick_coord.y * edge, brick_coord.z * edge);
        let mut brick = Brick::new();
        for lz in 0..edge {
            for ly in 0..edge {
                for lx in 0..edge {
                    let p = IVec3::new(origin.x + lx, origin.y + ly, origin.z + lz);
                    let mat = self.material_at(world_seed, p);
                    if mat != MATERIAL_AIR {
                        brick.set(IVec3::new(lx, ly, lz), Voxel::new(mat));
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
        let a = gen.generate_brick(42, IVec3::new(0, 0, 0));
        let b = gen.generate_brick(42, IVec3::new(0, 0, 0));
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
        let b = gen.generate_brick(42, IVec3::new(0, 200, 0));
        assert_eq!(b.nonempty_count, 0);
    }

    #[test]
    fn deep_brick_has_material() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        // y = -10 brick → world y ∈ [-160, -144), deep under surface.
        let b = gen.generate_brick(42, IVec3::new(0, -10, 0));
        assert!(b.nonempty_count > 0);
    }
}
