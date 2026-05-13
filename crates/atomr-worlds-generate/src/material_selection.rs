//! Pluggable per-voxel material selection.
//!
//! `TerrainGenerator` computes geometry (surface height, cave occupancy)
//! and then asks a [`MaterialSelectionStrategy`] which material id to
//! write for the solid voxels. The strategy is optional — when absent,
//! the generator runs its legacy inlined logic which is byte-for-byte
//! identical to the CUDA kernel.
//!
//! Adding a new strategy: implement the trait and wire it into the
//! consumer (`default_terrain` switches the default world).

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_noise::worley_noise_3d;

use crate::terrain::{
    MATERIAL_DIRT, MATERIAL_GLOW_ROCK, MATERIAL_GRASS, MATERIAL_ICE, MATERIAL_SAND,
    MATERIAL_SNOW, MATERIAL_STONE, MATERIAL_WATER,
};

/// Context passed to a material strategy for a single solid voxel.
#[derive(Debug, Copy, Clone)]
pub struct MaterialContext {
    pub world_seed: u64,
    pub p: IVec3,
    /// `surface_y - p.y`, in voxels (positive == below surface).
    pub depth_below_surface_voxels: f32,
    pub dirt_layer: u8,
    /// Macro biome id, or `None` for the legacy non-macro path.
    pub biome_id: Option<u8>,
}

impl MaterialContext {
    pub fn is_surface(&self) -> bool {
        // The "top" voxel of a column has depth ≈ 0 (the voxel at the
        // floor of the surface). Use a half-voxel window for robustness
        // against fractional surface heights.
        self.depth_below_surface_voxels < 1.0
    }

    pub fn is_topsoil(&self) -> bool {
        self.depth_below_surface_voxels < self.dirt_layer as f32
    }
}

pub trait MaterialSelectionStrategy: std::fmt::Debug + Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Pick the material id for a *solid* voxel. Air / cave voxels are
    /// filtered out before this is called.
    fn pick(&self, ctx: &MaterialContext) -> u16;
}

/// Today's legacy behaviour, exposed as a strategy so it can be picked
/// explicitly. Identical output to `TerrainGenerator::material_at` /
/// `material_at_macro`. Used as the bit-for-bit reference path.
#[derive(Debug, Default, Clone, Copy)]
pub struct LegacyBanded;

impl MaterialSelectionStrategy for LegacyBanded {
    fn name(&self) -> &'static str {
        "LegacyBanded"
    }
    fn pick(&self, ctx: &MaterialContext) -> u16 {
        match ctx.biome_id {
            None => {
                if ctx.is_topsoil() {
                    MATERIAL_DIRT
                } else {
                    MATERIAL_STONE
                }
            }
            Some(biome) => {
                if ctx.is_topsoil() {
                    biome_legacy_topsoil(biome)
                } else {
                    MATERIAL_STONE
                }
            }
        }
    }
}

/// Step 1's richer material picker. Adds grass (replacing dirt top layer
/// where temperate/forest biomes apply), glow_rock (rare worley-noise
/// minima in deep stone), and ice (replacing snow on the ICE biome).
/// Wood/leaves require multi-voxel stamping and are reserved for a
/// later feature pass.
#[derive(Debug, Clone, Copy)]
pub struct LayeredWithFeatures {
    pub glow_rock_threshold: f32,
    pub glow_rock_frequency: f32,
    pub glow_rock_min_depth: f32,
}

impl Default for LayeredWithFeatures {
    fn default() -> Self {
        Self {
            // Worley distance-squared at the chosen frequency. Smaller =
            // rarer hits. Tuned to leave glow_rock as occasional
            // outcrops in deep stone rather than a banded layer.
            glow_rock_threshold: 0.012,
            glow_rock_frequency: 1.0 / 18.0,
            // Only show glow_rock several voxels below the surface so
            // top-layer terrain reads cleanly.
            glow_rock_min_depth: 6.0,
        }
    }
}

impl LayeredWithFeatures {
    fn is_glow_rock(&self, world_seed: u64, p: IVec3) -> bool {
        // Distinct seed offset so the glow-rock distribution doesn't
        // alias with caves.
        let d2 = worley_noise_3d(
            world_seed.wrapping_add(0x6_BEAD_FACE_0EE_F00D),
            p.x as f32 * self.glow_rock_frequency,
            p.y as f32 * self.glow_rock_frequency,
            p.z as f32 * self.glow_rock_frequency,
        );
        d2 < self.glow_rock_threshold
    }
}

impl MaterialSelectionStrategy for LayeredWithFeatures {
    fn name(&self) -> &'static str {
        "LayeredWithFeatures"
    }
    fn pick(&self, ctx: &MaterialContext) -> u16 {
        let surface = ctx.is_surface();
        let topsoil = ctx.is_topsoil();

        // Glow-rock substitution for deep stone — rare worley minima.
        let deep_glow = !topsoil
            && ctx.depth_below_surface_voxels >= self.glow_rock_min_depth
            && self.is_glow_rock(ctx.world_seed, ctx.p);

        match ctx.biome_id {
            None => {
                if surface {
                    MATERIAL_GRASS
                } else if topsoil {
                    MATERIAL_DIRT
                } else if deep_glow {
                    MATERIAL_GLOW_ROCK
                } else {
                    MATERIAL_STONE
                }
            }
            Some(biome) => {
                if topsoil {
                    biome_layered_topsoil(biome, surface)
                } else if deep_glow {
                    MATERIAL_GLOW_ROCK
                } else {
                    MATERIAL_STONE
                }
            }
        }
    }
}

fn biome_legacy_topsoil(biome: u8) -> u16 {
    use crate::macro_state::biome_id;
    match biome {
        v if v == biome_id::DESERT || v == biome_id::SAVANNA => MATERIAL_SAND,
        v if v == biome_id::ICE || v == biome_id::TUNDRA => MATERIAL_SNOW,
        v if v == biome_id::OCEAN => MATERIAL_WATER,
        v if v == biome_id::MOUNTAIN => MATERIAL_STONE,
        _ => MATERIAL_DIRT,
    }
}

fn biome_layered_topsoil(biome: u8, surface: bool) -> u16 {
    use crate::macro_state::biome_id;
    match biome {
        v if v == biome_id::DESERT || v == biome_id::SAVANNA => MATERIAL_SAND,
        // Ice biome: ice instead of snow, gives a smoother PBR look.
        v if v == biome_id::ICE => MATERIAL_ICE,
        v if v == biome_id::TUNDRA => MATERIAL_SNOW,
        v if v == biome_id::OCEAN => MATERIAL_WATER,
        v if v == biome_id::MOUNTAIN => MATERIAL_STONE,
        // Temperate / grassland / forest / rainforest: grass on top,
        // dirt just below. Surface == the topmost voxel.
        v if v == biome_id::GRASSLAND
            || v == biome_id::TEMPERATE_FOREST
            || v == biome_id::RAINFOREST
            || v == biome_id::TAIGA =>
        {
            if surface {
                MATERIAL_GRASS
            } else {
                MATERIAL_DIRT
            }
        }
        // Unknown / future biomes: dirt-with-grass-top.
        _ => {
            if surface {
                MATERIAL_GRASS
            } else {
                MATERIAL_DIRT
            }
        }
    }
}

/// Convenience: shareable trait-object alias used by `TerrainGenerator`.
pub type DynMaterialStrategy = Arc<dyn MaterialSelectionStrategy>;
