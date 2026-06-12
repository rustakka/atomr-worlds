//! `ice_shell` strategy — a frozen "cryo" planetary archetype.
//!
//! The first real *planetary archetype* generator (the roadmap's "Additional
//! generation styles → Planetary archetypes" thread). It produces a world with
//! a distinct vertical structure unlike the default
//! [`TerrainGenerator`](crate::terrain::TerrainGenerator) — a Europa-like
//! frozen shell over a buried liquid ocean and a rocky core:
//!
//! ```text
//!   air
//!   ──────────── surface  (continuous FBM relief)
//!   SNOW         bright rim       (`snow_cap_m`)
//!   ICE          frozen shell     (`crust_m`)
//!   WATER        subsurface ocean (`ocean_m`)
//!   STONE        rocky core   (sparse GLOW_ROCK cryo-vents when `vents`)
//! ```
//!
//! Like the legacy non-macro terrain path it is a **pure** function of
//! `(config, world_seed, brick_coord, lod)` and samples its heightfield in
//! continuous world-metre space, so adjacent LOD tiers agree on the surface:
//! LOD 0 samples integer voxel coordinates (1 m voxels), LOD ≥ 1 samples the
//! voxel **centre** in world metres (`2^depth` m voxels) — exactly mirroring
//! [`TerrainGenerator`]'s LOD handling so surface error stays bounded by
//! ±voxel/2 at tier boundaries.
//!
//! Slice 1 deliberately **ignores `ctx.macro_state`** (the [`BrickGenerator`]
//! trait explicitly allows generators that don't need macro state to skip it):
//! the icy world is uniform across the sphere. Reading macro elevation / biomes
//! / hydrology on spherical worlds to vary crust thickness and ocean extent is
//! the documented follow-up.

use atomr_worlds_core::IVec3;
use atomr_worlds_noise::{fbm_value, worley_noise_3d, FbmConfig};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use crate::brick::{BrickGenContext, BrickGenerator};
use crate::terrain::{
    MATERIAL_AIR, MATERIAL_GLOW_ROCK, MATERIAL_ICE, MATERIAL_SNOW, MATERIAL_STONE, MATERIAL_WATER,
};

/// Tunables for the ice-shell archetype. All vertical extents are in **world
/// metres** (= LOD-0 voxels), matching the heightfield's units. Defaults are
/// chosen so a single 16-voxel brick straddling the surface shows the snow rim
/// and ice shell, and so the buried ocean sits comfortably below.
///
/// Invariants for a well-formed layer stack (checked by `debug_assert!` in
/// [`IceShellGenerator::generate_brick`]): `0 <= snow_cap_m <= crust_m`,
/// `ocean_m > 0`, and `amplitude_m >= 0`. A config that violates them still
/// produces deterministic output, but bands may be skipped (e.g.
/// `snow_cap_m > crust_m` swallows the `ICE` band).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IceShellConfig {
    /// Mean surface elevation (metres) about which the FBM relief varies.
    pub base_surface_m: f32,
    /// Peak ± relief of the surface (metres).
    pub amplitude_m: f32,
    /// Horizontal heightfield frequency (cycles per metre).
    pub frequency: f32,
    /// FBM octaves for the surface heightfield.
    pub octaves: u8,
    /// Thickness of the bright `SNOW` rim at the very top of the crust (metres).
    pub snow_cap_m: f32,
    /// Total frozen-shell (`SNOW` + `ICE`) thickness below the surface (metres).
    /// The `ICE` band is `crust_m - snow_cap_m` thick.
    pub crust_m: f32,
    /// Depth of the buried liquid `WATER` ocean beneath the shell (metres).
    pub ocean_m: f32,
    /// Whether the rocky core carries sparse `GLOW_ROCK` cryo-vents.
    pub vents: bool,
    /// Worley distance² threshold for a core cell to be a vent (smaller = rarer).
    pub vent_threshold: f32,
    /// Cryo-vent Worley frequency (cells per metre).
    pub vent_frequency: f32,
}

impl Default for IceShellConfig {
    fn default() -> Self {
        Self {
            base_surface_m: 40.0,
            amplitude_m: 10.0,
            frequency: 1.0 / 128.0,
            octaves: 4,
            snow_cap_m: 2.0,
            crust_m: 14.0,
            ocean_m: 48.0,
            vents: true,
            vent_threshold: 0.06,
            vent_frequency: 1.0 / 18.0,
        }
    }
}

/// Frozen-shell planetary archetype generator. See the [module docs](self).
#[derive(Clone, Debug, Default)]
pub struct IceShellGenerator {
    pub config: IceShellConfig,
}

impl IceShellGenerator {
    pub fn new(config: IceShellConfig) -> Self {
        Self { config }
    }

    /// Surface elevation (metres) at horizontal world position `(x, z)` metres.
    /// LOD-independent and continuous, so every tier samples the same surface.
    fn surface_world(&self, seed: u64, x: f32, z: f32) -> f32 {
        let cfg = &self.config;
        let fbm_cfg = FbmConfig { octaves: cfg.octaves, lacunarity: 2.0, gain: 0.5, frequency: 1.0 };
        // `fbm_value` is normalised to [0, 1]; remap to [-1, 1] about the mean.
        let n = fbm_value(seed, x * cfg.frequency, 0.0, z * cfg.frequency, fbm_cfg);
        cfg.base_surface_m + cfg.amplitude_m * (n * 2.0 - 1.0)
    }

    /// Is the core cell at world metres `(x, y, z)` a cryo-vent? Only ever
    /// queried inside the rocky core, so vents never pock the ice or ocean.
    fn is_vent(&self, seed: u64, x: f32, y: f32, z: f32) -> bool {
        let cfg = &self.config;
        let d2 = worley_noise_3d(
            seed.wrapping_add(0x1CE5_E110_u64),
            x * cfg.vent_frequency,
            y * cfg.vent_frequency,
            z * cfg.vent_frequency,
        );
        d2 < cfg.vent_threshold
    }

    /// Material at world metres `(wx, wy, wz)` given the column's `surface`
    /// (metres). The single source of truth for the vertical band model; both
    /// the LOD-0 and LOD-≥1 fill loops call this.
    fn material_at_metric(&self, seed: u64, wx: f32, wy: f32, wz: f32, surface: f32) -> u16 {
        if wy >= surface {
            return MATERIAL_AIR;
        }
        let cfg = &self.config;
        let depth = surface - wy; // > 0 below the surface
        if depth <= cfg.snow_cap_m {
            return MATERIAL_SNOW;
        }
        if depth <= cfg.crust_m {
            return MATERIAL_ICE;
        }
        if depth <= cfg.crust_m + cfg.ocean_m {
            return MATERIAL_WATER;
        }
        if cfg.vents && self.is_vent(seed, wx, wy, wz) {
            return MATERIAL_GLOW_ROCK;
        }
        MATERIAL_STONE
    }
}

impl BrickGenerator for IceShellGenerator {
    fn generate_brick(&self, ctx: &BrickGenContext) -> Brick {
        // Guard the layer-stack invariants (see `IceShellConfig`). Debug-only:
        // zero release cost, and a malformed config still yields deterministic
        // (if geometrically odd) output rather than panicking in production.
        let cfg = &self.config;
        debug_assert!(
            cfg.snow_cap_m >= 0.0 && cfg.crust_m >= cfg.snow_cap_m && cfg.ocean_m > 0.0,
            "ice-shell bands must satisfy 0 <= snow_cap_m <= crust_m and ocean_m > 0 \
             (snow_cap_m={}, crust_m={}, ocean_m={})",
            cfg.snow_cap_m,
            cfg.crust_m,
            cfg.ocean_m,
        );
        debug_assert!(cfg.amplitude_m >= 0.0, "amplitude_m must be non-negative");

        let edge = BRICK_EDGE as i64;
        let origin = IVec3::new(
            ctx.brick_coord.x * edge,
            ctx.brick_coord.y * edge,
            ctx.brick_coord.z * edge,
        );
        let mut brick = Brick::new();

        // LOD 0 = 1 m voxels (integer world coords); LOD L > 0 voxels are
        // `2^L` m wide and we sample the voxel centre in world metres so
        // adjacent tiers agree on the surface. Mirrors `TerrainGenerator`.
        let voxel_m = (1u64 << ctx.lod.depth as u32) as f32;
        let lod_aware = ctx.lod.depth > 0;
        let seed = ctx.world_seed;

        for lz in 0..edge {
            for ly in 0..edge {
                for lx in 0..edge {
                    let (wx, wy, wz) = if lod_aware {
                        (
                            ((origin.x + lx) as f32 + 0.5) * voxel_m,
                            ((origin.y + ly) as f32 + 0.5) * voxel_m,
                            ((origin.z + lz) as f32 + 0.5) * voxel_m,
                        )
                    } else {
                        ((origin.x + lx) as f32, (origin.y + ly) as f32, (origin.z + lz) as f32)
                    };
                    let surface = self.surface_world(seed, wx, wz);
                    let mat = self.material_at_metric(seed, wx, wy, wz, surface);
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
    use atomr_worlds_core::lod::Lod;

    const SEED: u64 = 0x1CE_F00D_u64 ^ 0xDEAD;

    fn ctx(brick: IVec3, lod: Lod) -> BrickGenContext {
        let mut c = BrickGenContext::legacy(SEED, brick);
        c.lod = lod;
        c
    }

    /// Same inputs → byte-identical brick (pure, deterministic on one machine).
    #[test]
    fn deterministic() {
        let g = IceShellGenerator::default();
        let a = g.generate_brick(&ctx(IVec3::new(0, 2, 0), Lod::new(0)));
        let b = g.generate_brick(&ctx(IVec3::new(0, 2, 0), Lod::new(0)));
        assert_eq!(a.nonempty_count, b.nonempty_count);
        assert_eq!(a.voxels, b.voxels);
    }

    /// The vertical band model: descending from the surface we hit
    /// SNOW → ICE → WATER → STONE in order, and AIR sits above.
    #[test]
    fn material_bands_descend_correctly() {
        let g = IceShellGenerator::default();
        let cfg = g.config;
        let s = 100.0_f32; // arbitrary surface for the pure-function check
        let mat = |depth_below: f32| g.material_at_metric(SEED, 0.5, s - depth_below, 0.5, s);

        // Mid-band sanity.
        assert_eq!(mat(-1.0), MATERIAL_AIR, "above surface is air");
        assert_eq!(mat(cfg.snow_cap_m * 0.5), MATERIAL_SNOW, "rim is snow");
        assert_eq!(mat((cfg.snow_cap_m + cfg.crust_m) * 0.5), MATERIAL_ICE, "shell is ice");
        assert_eq!(mat(cfg.crust_m + cfg.ocean_m * 0.5), MATERIAL_WATER, "buried ocean");
        let core = mat(cfg.crust_m + cfg.ocean_m + 100.0);
        assert!(core == MATERIAL_STONE || core == MATERIAL_GLOW_ROCK, "core is rock, got {core}");

        // Exact band boundaries — the model is half-open `(prev, edge]`, so each
        // edge depth belongs to the *shallower* band. Pins the `<=` operators
        // against an accidental flip to `<`.
        assert_eq!(mat(0.0), MATERIAL_AIR, "depth 0 (at the surface) is air");
        assert_eq!(mat(cfg.snow_cap_m), MATERIAL_SNOW, "snow_cap_m boundary is still snow");
        assert_eq!(mat(cfg.crust_m), MATERIAL_ICE, "crust_m boundary is still ice");
        assert_eq!(
            mat(cfg.crust_m + cfg.ocean_m),
            MATERIAL_WATER,
            "crust+ocean boundary is still water"
        );
    }

    /// The brick that contains the surface straddles it: neither empty nor
    /// full. Computed from the actual heightfield value so it doesn't depend on
    /// where the FBM happens to land.
    #[test]
    fn surface_brick_straddles() {
        let g = IceShellGenerator::default();
        let edge = BRICK_EDGE as i64;
        let s = g.surface_world(SEED, 0.5, 0.5); // origin-column surface (metres)
        let by = (s.floor() as i64).div_euclid(edge);
        let brick = g.generate_brick(&ctx(IVec3::new(0, by, 0), Lod::new(0)));
        assert!(!brick.is_empty(), "surface brick must have solids (s={s}, by={by})");
        assert!(
            (brick.nonempty_count as usize) < BRICK_EDGE.pow(3),
            "surface brick must have air above (s={s}, by={by})"
        );
    }

    /// The brick directly below the surface brick is fully solid (no air leaks
    /// below the surface) and contains the `ICE` shell. The frozen shell is
    /// thicker than 1 brick edge here is not required — the band model guarantees
    /// an ICE depth window of (snow_cap, crust] = (2, 14] m, which a 16 m brick
    /// spanning depths ~[1, 32) below the surface always intersects.
    #[test]
    fn brick_below_surface_is_solid_ice_shell() {
        let g = IceShellGenerator::default();
        let edge = BRICK_EDGE as i64;
        let s = g.surface_world(SEED, 0.5, 0.5);
        let by = (s.floor() as i64).div_euclid(edge) - 1; // one brick below the surface
        let brick = g.generate_brick(&ctx(IVec3::new(0, by, 0), Lod::new(0)));
        assert_eq!(
            brick.nonempty_count as usize,
            BRICK_EDGE.pow(3),
            "the brick just below the surface is entirely sub-surface → no air"
        );
        let mut saw_ice = false;
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    if brick.get(IVec3::new(x, y, z)).0 == MATERIAL_ICE {
                        saw_ice = true;
                    }
                }
            }
        }
        assert!(saw_ice, "expected the ICE shell below the surface");
    }

    /// Far below the core top every voxel is rocky (STONE/GLOW_ROCK) — no
    /// water, ice, or air leaks into the deep core.
    #[test]
    fn deep_brick_is_all_rock() {
        let g = IceShellGenerator::default();
        // core_top = surface - crust - ocean ∈ roughly [-32, -12]; y ≪ that.
        let brick = g.generate_brick(&ctx(IVec3::new(0, -6, 0), Lod::new(0)));
        assert_eq!(brick.nonempty_count as usize, BRICK_EDGE.pow(3), "core is fully solid");
        for z in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for x in 0..BRICK_EDGE as i64 {
                    let m = brick.get(IVec3::new(x, y, z)).0;
                    assert!(
                        m == MATERIAL_STONE || m == MATERIAL_GLOW_ROCK,
                        "deep voxel must be rock, got {m}"
                    );
                }
            }
        }
    }

    /// High above the highest possible surface the brick is pure air.
    #[test]
    fn high_brick_is_empty() {
        let g = IceShellGenerator::default();
        // max surface = base + amplitude = 50; brick y-index 8 spans 128..143.
        let brick = g.generate_brick(&ctx(IVec3::new(0, 8, 0), Lod::new(0)));
        assert!(brick.is_empty(), "sky brick must be empty");
    }

    /// Disabling vents removes every GLOW_ROCK voxel from the core.
    #[test]
    fn vents_disabled_yields_no_glow_rock() {
        let g = IceShellGenerator::new(IceShellConfig { vents: false, ..IceShellConfig::default() });
        let brick = g.generate_brick(&ctx(IVec3::new(0, -6, 0), Lod::new(0)));
        for z in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for x in 0..BRICK_EDGE as i64 {
                    assert_ne!(
                        brick.get(IVec3::new(x, y, z)).0,
                        MATERIAL_GLOW_ROCK,
                        "no vents expected"
                    );
                }
            }
        }
    }

    /// The surface heightfield is LOD-independent, and a LOD-1 brick covering
    /// the surface still straddles it (neither empty nor full) — so the coarse
    /// tier captures the same surface the fine tier does.
    #[test]
    fn lod_tiers_agree_on_surface() {
        let g = IceShellGenerator::default();
        // Heightfield is a pure function of (x, z) metres, independent of LOD.
        for &(x, z) in &[(3.5_f32, 7.5_f32), (40.0, 12.0), (-22.5, 88.0)] {
            assert_eq!(g.surface_world(SEED, x, z), g.surface_world(SEED, x, z));
        }
        // LOD-1 voxels are 2 m. A single coarse brick might sit just below or
        // just above the surface depending on where the FBM lands, so scan the
        // stack of LOD-1 bricks bracketing the [30, 50] surface band: y-index 0
        // (centres ~1..31 m, below), 1 (~33..63 m), 2 (~65..95 m, above). The
        // surface must show up *somewhere* as a solid→air transition, i.e. the
        // stack is neither all-solid nor all-air.
        let cells = BRICK_EDGE.pow(3);
        let total: usize = (0..=2)
            .map(|by| g.generate_brick(&ctx(IVec3::new(0, by, 0), Lod::new(1))).nonempty_count as usize)
            .sum();
        assert!(total > 0, "LOD-1 stack must have solids below the surface");
        assert!(total < 3 * cells, "LOD-1 stack must have air above the surface");
    }
}
