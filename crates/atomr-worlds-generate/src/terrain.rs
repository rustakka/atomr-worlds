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
use crate::macro_state::{biome_id, water_kind, WorldMacroState, NO_FLOW};
use crate::material_selection::{
    DynMaterialStrategy, MaterialContext, MaterialSelectionStrategy,
};

/// Seed salt for the river-meander FBM — keeps the channel-centerline
/// noise from aliasing with the heightfield / cave / glow-rock fields.
const RIVER_MEANDER_SALT: u64 = 0xA1DE_F10A_7CE5_2B91;
/// Seed salt for the river bank-width Worley jitter.
const RIVER_BANK_SALT: u64 = 0xB42C_8F30_DA67_1E55;

/// Result of projecting a voxel column against a river corridor.
#[derive(Copy, Clone, Debug)]
struct RiverCarve {
    /// `true` when the column lies inside the carved channel.
    in_channel: bool,
    /// Terrain surface after carving (voxels). Equals the input surface
    /// when the column is not inside a channel.
    carved_surface_voxels: f32,
    /// Water surface inside the channel (voxels); `f32::NEG_INFINITY`
    /// when the column is not inside a channel.
    water_level_voxels: f32,
}

/// Convert a macro water-surface elevation (m) into voxel-`y` units using
/// the same `mpv` the macro surface uses, so water and terrain share a
/// vertical frame. The `NO_WATER_SURFACE` sentinel passes through as
/// `NEG_INFINITY` so `fy < result` is always false on dry faces.
fn water_surface_to_voxels(water_surface_m: f32, mpv: f64) -> f32 {
    if !water_surface_m.is_finite() {
        return f32::NEG_INFINITY;
    }
    (water_surface_m as f64 / mpv) as f32
}

/// The voxel-`y` of the topmost water at a column: the river channel water
/// level when in-channel, else the ocean/lake surface, else
/// `NEG_INFINITY` (no water). River and ocean/lake are mutually exclusive
/// per face — a `RIVER`-classified face is never also `OCEAN`/`LAKE`.
fn water_top_voxels(carve: &RiverCarve, sample: &crate::macro_state::MacroSample, mpv: f64) -> f32 {
    if carve.in_channel {
        carve.water_level_voxels
    } else {
        match sample.water_kind {
            water_kind::OCEAN | water_kind::LAKE => {
                water_surface_to_voxels(sample.water_surface_m, mpv)
            }
            _ => f32::NEG_INFINITY,
        }
    }
}

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
    // --- River channel carving (macro-path only) ---
    /// Channel half-extent base width, in meters.
    pub river_base_width_m: f32,
    /// Extra channel width per unit `sqrt(flow_accum)`, in meters.
    pub river_width_per_accum: f32,
    /// Hard cap on channel width, in meters.
    pub river_max_width_m: f32,
    /// Channel carve depth base, in meters.
    pub river_base_depth_m: f32,
    /// Extra carve depth per unit `sqrt(flow_accum)`, in meters.
    pub river_depth_per_accum: f32,
    /// Hard cap on carve depth, in meters.
    pub river_max_depth_m: f32,
    /// Spatial frequency of the channel-meander FBM (per world meter).
    pub river_meander_freq: f32,
    /// Peak lateral meander offset of the channel centerline, in meters.
    pub river_meander_amp_m: f32,
    /// Spatial frequency of the bank-width Worley jitter (per world meter).
    pub river_bank_freq: f32,
    /// Peak bank-width jitter, in meters.
    pub river_bank_jitter_m: f32,
    /// How far the river water surface sits below the surrounding bank, in
    /// meters.
    pub river_inset_m: f32,
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
            river_base_width_m: 18.0,
            river_width_per_accum: 2.5,
            river_max_width_m: 140.0,
            river_base_depth_m: 3.0,
            river_depth_per_accum: 0.6,
            river_max_depth_m: 22.0,
            river_meander_freq: 1.0 / 320.0,
            river_meander_amp_m: 70.0,
            river_bank_freq: 1.0 / 40.0,
            river_bank_jitter_m: 6.0,
            river_inset_m: 1.0,
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
            // Non-macro path has no hydrology layer — never submerged.
            under_water: false,
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
            // Non-macro path has no hydrology layer — never submerged.
            under_water: false,
        };
        strategy.pick(&ctx)
    }

    /// Shared helper: project a voxel onto the macro surface and return
    /// `(macro_surface_voxels, mpv, wx, wz, sample)`. `wx`/`wz` are the
    /// column's horizontal world-meter coordinates centered on the world
    /// (the frame the macro face lookup uses); `mpv` is the meters-per-
    /// voxel conversion shared by terrain *and* water surfaces so they
    /// stay in one vertical frame.
    fn macro_surface_and_sample(
        &self,
        p: IVec3,
        macro_state: &WorldMacroState,
        scale: MetricScale,
    ) -> (f32, f64, f64, f64, crate::macro_state::MacroSample) {
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
        (macro_surface_voxels, mpv, wx, wz, sample)
    }

    /// Project a voxel column against the macro river network and, if it
    /// falls inside a corridor face's channel, return the carved riverbed
    /// surface and the channel water level.
    ///
    /// The macro layer supplies the *global* context — which faces are
    /// river corridors, the flow direction, and the flow accumulation
    /// magnitude. The *local* seed supplies the detail: a low-frequency
    /// FBM meanders the channel centerline (sampled in voxel-centered,
    /// LOD-consistent world meters), and a Worley field jitters the bank
    /// width. Channel width and carve depth scale with `sqrt(flow_accum)`.
    fn river_carve(
        &self,
        world_seed: u64,
        p: IVec3,
        voxel_m: f32,
        surface_voxels: f32,
        mpv: f64,
        wx: f64,
        wz: f64,
        sample: &crate::macro_state::MacroSample,
        macro_state: &WorldMacroState,
    ) -> RiverCarve {
        let no_carve = RiverCarve {
            in_channel: false,
            carved_surface_voxels: surface_voxels,
            water_level_voxels: f32::NEG_INFINITY,
        };
        // Only RIVER-classified faces with a valid downhill direction carve.
        if sample.water_kind != water_kind::RIVER || sample.flow_dir == NO_FLOW {
            return no_carve;
        }
        let cfg = &self.config;

        // Horizontal flow axis: face centroid → downstream centroid,
        // projected onto the X/Z plane.
        let c = macro_state.grid.face_centroid(sample.face);
        let c_down = macro_state.grid.face_centroid(sample.flow_dir);
        let mut fx = c_down.x - c.x;
        let mut fz = c_down.z - c.z;
        let flen2 = fx * fx + fz * fz;
        if flen2 < 1e-18 {
            return no_carve; // degenerate flow axis (e.g. near a pole)
        }
        let flen = flen2.sqrt();
        fx /= flen;
        fz /= flen;
        // Perpendicular (right-hand) axis in the X/Z plane.
        let perp_x = -fz;
        let perp_z = fx;

        // Anchor the channel centerline on the face centroid, taken at
        // this column's horizontal radius so it lies in the column's
        // plane. `perp_dist` is the column's signed perpendicular distance
        // from that anchor, in (macro-frame) meters.
        let cxz2 = c.x * c.x + c.z * c.z;
        if cxz2 < 1e-18 {
            return no_carve; // centroid on the polar axis — no X/Z anchor
        }
        let r = (wx * wx + wz * wz).sqrt();
        let cxz_inv = cxz2.sqrt().recip();
        let anchor_x = c.x * cxz_inv * r;
        let anchor_z = c.z * cxz_inv * r;
        let off_x = wx - anchor_x;
        let off_z = wz - anchor_z;
        let perp_dist = (off_x * perp_x + off_z * perp_z) as f32;

        // Meandering centerline — low-frequency FBM in voxel-centered,
        // LOD-consistent world meters yields a lateral offset.
        let wx_m = (p.x as f32 + 0.5) * voxel_m;
        let wz_m = (p.z as f32 + 0.5) * voxel_m;
        let meander_cfg = FbmConfig {
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            frequency: 1.0,
        };
        let meander = fbm_value(
            world_seed ^ RIVER_MEANDER_SALT,
            wx_m * cfg.river_meander_freq,
            0.0,
            wz_m * cfg.river_meander_freq,
            meander_cfg,
        );
        let lateral_offset_m = (meander * 2.0 - 1.0) * cfg.river_meander_amp_m;
        let signed_perp = perp_dist - lateral_offset_m;

        // Channel width / depth scale with sqrt(flow_accum) — keeps large
        // rivers bounded — and are clamped to config maxima.
        let sqrt_accum = sample.flow_accum.max(0.0).sqrt();
        let width_m = (cfg.river_base_width_m + cfg.river_width_per_accum * sqrt_accum)
            .min(cfg.river_max_width_m);
        let depth_m = (cfg.river_base_depth_m + cfg.river_depth_per_accum * sqrt_accum)
            .min(cfg.river_max_depth_m);

        // Bank-width jitter (Worley) so the channel walls aren't straight.
        let bank = worley_noise_3d(
            world_seed ^ RIVER_BANK_SALT,
            wx_m * cfg.river_bank_freq,
            0.0,
            wz_m * cfg.river_bank_freq,
        );
        let half_w = (width_m * 0.5 + (bank - 0.5) * cfg.river_bank_jitter_m).max(0.5);

        if signed_perp.abs() >= half_w {
            return no_carve;
        }

        // Parabolic bed: deepest at the centerline, tapering to the banks.
        let t = (signed_perp.abs() / half_w).clamp(0.0, 1.0);
        let carve_m = depth_m * (1.0 - t * t);
        let carved_surface = surface_voxels - (carve_m as f64 / mpv) as f32;

        // River water sits slightly below the surrounding bank, and never
        // above the macro corridor's water surface — so a river meeting a
        // lake or the sea shares its level.
        let bank_level = surface_voxels - (cfg.river_inset_m as f64 / mpv) as f32;
        let macro_level = water_surface_to_voxels(sample.water_surface_m, mpv);
        let water_level = if macro_level.is_finite() {
            bank_level.min(macro_level)
        } else {
            bank_level
        };

        RiverCarve {
            in_channel: true,
            carved_surface_voxels: carved_surface,
            water_level_voxels: water_level,
        }
    }

    /// Material at a world voxel coordinate, with macro state available.
    /// The column's surface height is shifted by the macro elevation;
    /// river corridors carve a channel; ocean / lake / river water columns
    /// fill the air above the terrain up to the water surface; and the
    /// top-layer material is biome-driven (sand on submerged beds).
    pub fn material_at_macro(
        &self,
        world_seed: u64,
        p: IVec3,
        macro_state: &WorldMacroState,
        scale: MetricScale,
        voxel_m: f32,
    ) -> u16 {
        let (macro_surface_voxels, mpv, wx, wz, sample) =
            self.macro_surface_and_sample(p, macro_state, scale);
        let local = self.surface_height(world_seed, p.x, p.z) - self.config.base_height;
        let surface = macro_surface_voxels + local;
        let fy = p.y as f32;

        let carve =
            self.river_carve(world_seed, p, voxel_m, surface, mpv, wx, wz, &sample, macro_state);
        let effective_surface = carve.carved_surface_voxels;
        let water_top = water_top_voxels(&carve, &sample, mpv);

        if fy >= effective_surface {
            // Above the (carved) terrain surface — water column or air.
            if fy < water_top {
                return MATERIAL_WATER;
            }
            return MATERIAL_AIR;
        }
        if self.is_cave(world_seed, p.x, p.y, p.z) {
            return MATERIAL_CAVE;
        }
        // Top layer: biome controls material; deeper voxels are stone. A
        // submerged bed reads as sand regardless of the biome above.
        if fy >= effective_surface - self.config.dirt_layer as f32 {
            if effective_surface < water_top {
                MATERIAL_SAND
            } else {
                match sample.biome_id {
                    v if v == biome_id::DESERT || v == biome_id::SAVANNA => MATERIAL_SAND,
                    v if v == biome_id::ICE || v == biome_id::TUNDRA => MATERIAL_SNOW,
                    v if v == biome_id::OCEAN => MATERIAL_SAND,
                    v if v == biome_id::MOUNTAIN => MATERIAL_STONE,
                    _ => MATERIAL_DIRT,
                }
            }
        } else {
            MATERIAL_STONE
        }
    }

    /// Strategy-driven material picker (with macro state). Mirrors
    /// `material_at_macro` for the geometry — surface shift, river carve,
    /// water fill, cave check — but defers the solid-voxel material choice
    /// to `self.strategy`.
    fn material_at_macro_strategy(
        &self,
        strategy: &dyn MaterialSelectionStrategy,
        world_seed: u64,
        p: IVec3,
        macro_state: &WorldMacroState,
        scale: MetricScale,
        voxel_m: f32,
    ) -> u16 {
        let (macro_surface_voxels, mpv, wx, wz, sample) =
            self.macro_surface_and_sample(p, macro_state, scale);
        let local = self.surface_height(world_seed, p.x, p.z) - self.config.base_height;
        let surface = macro_surface_voxels + local;
        let fy = p.y as f32;

        let carve =
            self.river_carve(world_seed, p, voxel_m, surface, mpv, wx, wz, &sample, macro_state);
        let effective_surface = carve.carved_surface_voxels;
        let water_top = water_top_voxels(&carve, &sample, mpv);

        if fy >= effective_surface {
            if fy < water_top {
                return MATERIAL_WATER;
            }
            return MATERIAL_AIR;
        }
        if self.is_cave(world_seed, p.x, p.y, p.z) {
            return MATERIAL_CAVE;
        }
        let ctx = MaterialContext {
            world_seed,
            p,
            // Depth is measured below the carved bed so topsoil banding
            // follows the riverbed.
            depth_below_surface_voxels: effective_surface - fy,
            dirt_layer: self.config.dirt_layer,
            biome_id: Some(sample.biome_id),
            under_water: effective_surface < water_top,
        };
        strategy.pick(&ctx)
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
                                    voxel_m,
                                ),
                                Some(s) => self.material_at_macro_strategy(
                                    s,
                                    ctx.world_seed,
                                    p,
                                    ms,
                                    ctx.scale,
                                    voxel_m,
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

    // --- Macro-path hydrology tests ---

    use crate::macro_state::{
        water_kind, DefaultMacroGenerator, MacroConfig, MacroGenerator, MacroSample,
        WorldMacroState,
    };
    use atomr_worlds_core::lod::Lod;
    use atomr_worlds_core::shape::WorldShape;
    use std::sync::Arc;

    const MACRO_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

    fn macro_state() -> Arc<WorldMacroState> {
        let g = DefaultMacroGenerator::new(MacroConfig {
            grid_level: 3,
            ..MacroConfig::default()
        });
        g.generate(MACRO_SEED, WorldShape::Sphere { radius_m: 6.371e6 })
    }

    /// Collect up to `limit` horizontal voxel columns `(px, pz)` whose
    /// direction projects to a face with `water_kind == kind`.
    fn find_columns(
        ws: &WorldMacroState,
        scale: MetricScale,
        kind: u8,
        limit: usize,
    ) -> Vec<(i64, i64, MacroSample)> {
        let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
        let cx = scale.root_size_m * 0.5;
        let span = (scale.root_size_m / mpv) as i64;
        let base = (cx / mpv) as i64;
        let step = (span / 200).max(1);
        let mut out = Vec::new();
        let mut px = base - span / 2;
        while px < base + span / 2 && out.len() < limit {
            let mut pz = base - span / 2;
            while pz < base + span / 2 {
                let wx = px as f64 * mpv - cx;
                let wz = pz as f64 * mpv - cx;
                let len2 = wx * wx + wz * wz;
                if len2 > 0.0 {
                    let len = len2.sqrt();
                    let dir = DVec3::new(wx / len, 0.0, wz / len);
                    let s = ws.sample(dir);
                    if s.water_kind == kind {
                        out.push((px, pz, s));
                        if out.len() >= limit {
                            break;
                        }
                    }
                }
                pz += step;
            }
            px += step;
        }
        out
    }

    /// First column whose direction projects to a `water_kind == kind` face.
    fn find_column(
        ws: &WorldMacroState,
        scale: MetricScale,
        kind: u8,
    ) -> Option<(i64, i64, MacroSample)> {
        find_columns(ws, scale, kind, 1).into_iter().next()
    }

    fn macro_ctx(ws: &Arc<WorldMacroState>, brick_coord: IVec3, lod: u8) -> BrickGenContext {
        BrickGenContext {
            world_seed: MACRO_SEED,
            brick_coord,
            lod: Lod::new(lod),
            shape: ws.shape,
            macro_state: Some(ws.clone()),
            scale: MetricScale::DEFAULT_WORLD,
        }
    }

    #[test]
    fn macro_brick_generation_is_deterministic() {
        let ws = macro_state();
        // default_terrain() installs the LayeredWithFeatures strategy, so
        // this exercises the strategy macro path too.
        let gen = crate::strategies::terrain::default_terrain();
        let ctx = macro_ctx(&ws, IVec3::new(3, -40, 5), 0);
        let a = gen.generate_brick(&ctx);
        let b = gen.generate_brick(&ctx);
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
    fn ocean_columns_have_water_above_a_sand_bed() {
        let ws = macro_state();
        let scale = MetricScale::DEFAULT_WORLD;
        let gen = TerrainGenerator::new(TerrainConfig::default());
        // Probe several ocean columns: any single column's thin topsoil
        // band can be fully carved away by caves, but not every column's.
        let columns = find_columns(&ws, scale, water_kind::OCEAN, 24);
        assert!(!columns.is_empty(), "default world has equatorial ocean");
        let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
        let mut seen_water = false;
        let mut seen_sand = false;
        for (px, pz, sample) in &columns {
            assert!(sample.elev_m < 0.0, "ocean face elevation must be negative");
            let seafloor = (sample.elev_m as f64 / mpv) as i64;
            for py in (seafloor - 40)..40 {
                match gen.material_at_macro(MACRO_SEED, IVec3::new(*px, py, *pz), &ws, scale, 1.0)
                {
                    MATERIAL_WATER => seen_water = true,
                    MATERIAL_SAND => seen_sand = true,
                    _ => {}
                }
            }
        }
        assert!(seen_water, "ocean columns must contain a water column");
        assert!(seen_sand, "ocean columns must have a sand bed");
    }

    #[test]
    fn ocean_water_surface_sits_at_sea_level() {
        let ws = macro_state();
        let scale = MetricScale::DEFAULT_WORLD;
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let (px, pz, _s) =
            find_column(&ws, scale, water_kind::OCEAN).expect("ocean column");
        // Sea level (elevation 0) → voxel y 0: the voxel below is water,
        // the voxel at/above is air.
        let at = |py| gen.material_at_macro(MACRO_SEED, IVec3::new(px, py, pz), &ws, scale, 1.0);
        assert_eq!(at(-1), MATERIAL_WATER);
        assert_eq!(at(0), MATERIAL_AIR);
        assert_eq!(at(80), MATERIAL_AIR);
    }

    #[test]
    fn ocean_water_fill_ignores_lod_voxel_size() {
        let ws = macro_state();
        let scale = MetricScale::DEFAULT_WORLD;
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let (px, pz, _s) =
            find_column(&ws, scale, water_kind::OCEAN).expect("ocean column");
        // Ocean / lake water fill is driven by the scale-only `mpv`, never
        // by the per-LOD `voxel_m` — so it is identical at every LOD.
        for py in [-400i64, -120, -1, 0, 64] {
            let p = IVec3::new(px, py, pz);
            let a = gen.material_at_macro(MACRO_SEED, p, &ws, scale, 1.0);
            let b = gen.material_at_macro(MACRO_SEED, p, &ws, scale, 8.0);
            assert_eq!(a, b, "ocean water fill must not depend on voxel_m (py={py})");
        }
    }

    #[test]
    fn river_carve_produces_a_channel() {
        use crate::macro_state::NO_FLOW;
        let ws = macro_state();
        let scale = MetricScale::DEFAULT_WORLD;
        let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
        let cx = scale.root_size_m * 0.5;
        let gen = TerrainGenerator::new(TerrainConfig::default());

        // Any river face with a valid downhill direction.
        let rf = (0..ws.grid.face_count())
            .find(|&f| {
                ws.water.water_kind[f] == water_kind::RIVER && ws.water.flow_dir[f] != NO_FLOW
            })
            .expect("default world has river corridors");
        let sample = MacroSample {
            face: rf as u32,
            elev_m: ws.elevation.elev_m[rf],
            temperature_c: ws.climate.temperature_c[rf],
            humidity: ws.climate.humidity[rf],
            biome_id: ws.biomes.biome_id[rf],
            water_kind: water_kind::RIVER,
            water_surface_m: ws.water.water_surface_m[rf],
            flow_dir: ws.water.flow_dir[rf],
            flow_accum: ws.water.flow_accum[rf].max(60.0),
        };
        let surface_voxels = (sample.elev_m as f64 / mpv) as f32;

        // Build a perpendicular-to-flow scan line through the face
        // centroid: `river_carve`'s perp distance then equals the scan
        // parameter `t`, so the line is guaranteed to cross the channel.
        let c = ws.grid.face_centroid(sample.face);
        let cd = ws.grid.face_centroid(sample.flow_dir);
        let (mut fx, mut fz) = (cd.x - c.x, cd.z - c.z);
        let fl = (fx * fx + fz * fz).sqrt();
        fx /= fl;
        fz /= fl;
        let (perp_x, perp_z) = (-fz, fx);
        let cxz_inv = (c.x * c.x + c.z * c.z).sqrt().recip();
        let r0 = scale.root_size_m * 0.35;
        let wx0 = c.x * cxz_inv * r0;
        let wz0 = c.z * cxz_inv * r0;

        let mut in_channel = 0;
        let mut deepest = (surface_voxels, f32::NEG_INFINITY);
        for ti in -250..=250 {
            let t = ti as f64;
            let wx = wx0 + t * perp_x;
            let wz = wz0 + t * perp_z;
            let p = IVec3::new(((wx + cx) / mpv) as i64, 0, ((wz + cx) / mpv) as i64);
            let rc = gen.river_carve(MACRO_SEED, p, 1.0, surface_voxels, mpv, wx, wz, &sample, &ws);
            if rc.in_channel {
                in_channel += 1;
                assert!(rc.carved_surface_voxels <= surface_voxels);
                if rc.carved_surface_voxels < deepest.0 {
                    deepest = (rc.carved_surface_voxels, rc.water_level_voxels);
                }
            }
        }
        assert!(in_channel > 0, "a perpendicular scan must cross the channel");
        assert!(deepest.0 < surface_voxels, "the channel carves below the bank");
        assert!(
            deepest.1 > deepest.0,
            "the deepest point holds water above the carved bed",
        );

        // A non-river sample never carves.
        let mut dry = sample;
        dry.water_kind = water_kind::NONE;
        let rc = gen.river_carve(
            MACRO_SEED,
            IVec3::new(10, 0, 20),
            1.0,
            surface_voxels,
            mpv,
            wx0,
            wz0,
            &dry,
            &ws,
        );
        assert!(!rc.in_channel);
        assert_eq!(rc.carved_surface_voxels, surface_voxels);
    }

    #[test]
    fn legacy_path_is_unaffected_by_hydrology() {
        // The non-macro path places no water and is byte-identical to the
        // pre-hydrology generator.
        let gen = TerrainGenerator::new(TerrainConfig::default());
        for by in [-12, -4, 0, 6] {
            let b = gen.generate_brick_legacy(7, IVec3::new(1, by, -2));
            for i in 0..16i64 {
                for j in 0..16i64 {
                    for k in 0..16i64 {
                        assert_ne!(
                            b.get(IVec3::new(i, j, k)).0,
                            MATERIAL_WATER,
                            "legacy path must never emit water",
                        );
                    }
                }
            }
        }
    }
}
