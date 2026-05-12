//! Phase 14e — tile-pyramid world summary baked from Phase 13c macro state.
//!
//! The regional / world overview mode renders the *whole world* at once
//! (or a coarse slice of it). Sampling [`WorldMacroState`] directly per
//! output pixel is fine for a 256×256 frame but blows up at 2048×2048;
//! and even at 256×256 we want to support pan/zoom without recomputing
//! the macro state on every input event. The solution is a small
//! pre-baked pyramid: a fixed handful of LOD levels, each level a 2D
//! grid of tiles, each tile a flat array of `(elev, biome, plate,
//! climate)` samples.
//!
//! # Pyramid layout
//!
//! - Level 0: 1 tile, covering the entire world.
//! - Level L: `2^L × 2^L = 4^L` tiles, each covering `(1 / 4^L)` of the
//!   surface.
//! - Tile (level, xy): one [`WorldSummaryTile`] with `size_px * size_px`
//!   parallel arrays.
//!
//! For a 4-level pyramid with `tile_size_px = 64`, total cells:
//!   `(1 + 4 + 16 + 64) * 64 * 64 ≈ 348k` per channel × 4 channels ≈ 1.4 M
//! cells — a few MB, easily cacheable.
//!
//! # Projection per shape
//!
//! - **Sphere**: tile (xy, level) covers an equirectangular lat/lon
//!   patch. The full world (level 0) is the full
//!   `(longitude ∈ [-π, π], latitude ∈ [-π/2, π/2])` rectangle; a tile
//!   at `(level, [tx, ty])` covers the sub-rectangle
//!   `longitude ∈ [-π + 2π · tx / N, -π + 2π · (tx+1) / N]`,
//!   `latitude  ∈ [+π/2 - π · (ty+1) / N, +π/2 - π · ty / N]` where
//!   `N = 2^level`. Each pixel within the tile resolves to a unit
//!   direction via [`crate::projection_sphere::equirectangular_pixel_to_dir`]
//!   evaluated at the tile-local pixel coordinate scaled into the
//!   global image space.
//! - **Cube** / **Cylinder**: tiles cover the planar XZ extent. For a
//!   cube of edge `E` the world covers `XZ ∈ [-E/2, E/2]²`; for a
//!   cylinder of radius `R` the extent is `XZ ∈ [-R, R]²` (we project
//!   the cylinder onto its top-down disc and let pixels outside the
//!   circle hold the same data as the boundary cell). Each pixel
//!   converts to a 2D world position, then to a unit direction
//!   `(x, 0, z)` for the macro-state lookup. Elevation is taken
//!   directly from the underlying face; latitude doesn't apply.
//!
//! # Cache invalidation
//!
//! [`WorldSummaryKey`] intentionally returns `false` from `intersects` —
//! voxel writes never invalidate the macro-derived summary; only the
//! macro_digest (Phase 13c) changing does. Callers regenerate via
//! `bake_world_summary` after a macro-state rebuild and store under a new
//! key.

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::WorldMacroState;

use crate::projection_sphere::equirectangular_pixel_to_dir;
use crate::view_cache::{CacheAabb, DerivedKey};

/// Per-pixel climate snapshot. Two channels — temperature in °C and
/// humidity in `[0, 1]`. Precipitation isn't included; the overview mode
/// uses biome colour for that.
#[derive(Copy, Clone, Debug, Default)]
pub struct ClimateSample {
    pub temperature_c: f32,
    pub humidity: f32,
}

/// One pyramid tile. Four parallel arrays of `size_px * size_px` cells
/// (row-major, top-left origin within the tile's projected rectangle).
#[derive(Clone, Debug)]
pub struct WorldSummaryTile {
    pub level: u8,
    pub xy: [u32; 2],
    pub elevation_m: Vec<f32>,
    pub biome_id: Vec<u8>,
    pub plate_id: Vec<u16>,
    pub climate: Vec<ClimateSample>,
    pub size_px: u32,
}

/// Pre-baked LOD pyramid for one world. Flattened tile storage is
/// level-major: level 0 first (one tile), then level 1's four tiles in
/// `[(0,0), (1,0), (0,1), (1,1)]` order, then level 2's sixteen tiles,
/// etc.
#[derive(Clone, Debug)]
pub struct WorldSummaryPyramid {
    pub tiles: Vec<WorldSummaryTile>,
    pub levels: u8,
    pub tiles_per_level: Vec<u32>,
    pub shape: WorldShape,
    pub macro_digest: u64,
}

impl WorldSummaryPyramid {
    /// Look up a tile by `(level, xy)`. `None` if the level is out of
    /// range or the xy coordinate exceeds `2^level`.
    pub fn tile(&self, level: u8, xy: [u32; 2]) -> Option<&WorldSummaryTile> {
        if (level as usize) >= self.tiles_per_level.len() {
            return None;
        }
        let n = 1u32 << level;
        if xy[0] >= n || xy[1] >= n {
            return None;
        }
        let mut offset = 0usize;
        for l in 0..level {
            offset += self.tiles_per_level[l as usize] as usize;
        }
        let idx = offset + (xy[1] as usize) * (n as usize) + (xy[0] as usize);
        self.tiles.get(idx)
    }
}

/// Cache key for [`WorldSummaryPyramid`] entries. Hash distinguishes
/// `(world, macro_rev, levels)`; `intersects` always returns `false` so
/// AABB-keyed invalidations from voxel deltas never drop a summary —
/// only an explicit `invalidate_key` or `invalidate_world` does.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct WorldSummaryKey {
    pub addr: WorldAddr,
    pub macro_digest: u64,
    pub levels: u8,
}

impl DerivedKey for WorldSummaryKey {
    fn world_addr(&self) -> &WorldAddr {
        &self.addr
    }
    fn intersects(&self, _aabb: CacheAabb) -> bool {
        // Voxel-level writes don't change macro state. The pyramid is
        // only stale when the macro digest changes, which is keyed
        // explicitly above (the hash will differ).
        false
    }
}

/// Build a complete pyramid by sampling `macro_state` at every pixel of
/// every tile at every level. See module docs for the projection rules.
pub fn bake_world_summary(
    addr: WorldAddr,
    macro_state: &WorldMacroState,
    levels: u8,
    tile_size_px: u32,
) -> WorldSummaryPyramid {
    let _ = addr; // address is only used by the cache key — the pyramid
                  // itself is fully determined by macro_state.
    let levels = levels.max(1);
    let tile_size_px = tile_size_px.max(1);
    let mut tiles_per_level = Vec::with_capacity(levels as usize);
    let mut tiles = Vec::new();
    for l in 0..levels {
        let n = 1u32 << l;
        tiles_per_level.push(n * n);
        for ty in 0..n {
            for tx in 0..n {
                tiles.push(bake_tile(macro_state, l, [tx, ty], n, tile_size_px));
            }
        }
    }
    WorldSummaryPyramid {
        tiles,
        levels,
        tiles_per_level,
        shape: macro_state.shape,
        macro_digest: macro_state.digest,
    }
}

fn bake_tile(
    macro_state: &WorldMacroState,
    level: u8,
    xy: [u32; 2],
    n: u32,
    size_px: u32,
) -> WorldSummaryTile {
    let cells = (size_px as usize) * (size_px as usize);
    let mut elevation_m = Vec::with_capacity(cells);
    let mut biome_id = Vec::with_capacity(cells);
    let mut plate_id = Vec::with_capacity(cells);
    let mut climate = Vec::with_capacity(cells);

    // Full-image pixel size at this level: each tile is `size_px` wide,
    // so the whole pyramid level has `n * size_px` pixels across.
    let full_w = n * size_px;
    let full_h = n * size_px;
    let tx_off = xy[0] * size_px;
    let ty_off = xy[1] * size_px;

    for py in 0..size_px {
        for px in 0..size_px {
            let gx = tx_off + px;
            let gy = ty_off + py;
            let dir = direction_for(macro_state.shape, gx, gy, full_w, full_h);
            let s = macro_state.sample(dir);
            let plate = macro_state.plates.plate_id[s.face as usize];
            elevation_m.push(s.elev_m);
            biome_id.push(s.biome_id);
            plate_id.push(plate);
            climate.push(ClimateSample { temperature_c: s.temperature_c, humidity: s.humidity });
        }
    }

    WorldSummaryTile { level, xy, elevation_m, biome_id, plate_id, climate, size_px }
}

/// Per-shape pixel-to-direction map. Sphere uses equirectangular; cube
/// and cylinder use a planar top-down projection where the pixel column
/// is X and row is Z (mapped onto `[-radius, +radius]²`), and the
/// direction returned points outward at `y = 0`.
#[inline]
fn direction_for(shape: WorldShape, gx: u32, gy: u32, full_w: u32, full_h: u32) -> DVec3 {
    match shape {
        WorldShape::Sphere { .. } => equirectangular_pixel_to_dir(gx, gy, full_w, full_h),
        WorldShape::Cube { .. } | WorldShape::Cylinder { .. } => {
            // Map pixel into `[-1, +1]` square, take that as the (x, z)
            // direction in the XZ plane. Pixels with `r2 > 1` (cylinder
            // outside the circle) get clamped onto the circle so the
            // boundary is well-defined; the macro sampler treats any
            // unit direction as a valid face lookup.
            let w = full_w.max(1) as f64;
            let h = full_h.max(1) as f64;
            let u = (gx as f64 + 0.5) / w;
            let v = (gy as f64 + 0.5) / h;
            let x = 2.0 * u - 1.0;
            let z = 2.0 * v - 1.0;
            let len2 = x * x + z * z;
            if len2 > 0.0 {
                let len = len2.sqrt().max(1e-12);
                DVec3::new(x / len, 0.0, z / len)
            } else {
                DVec3::new(1.0, 0.0, 0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_generate::macro_state::{DefaultMacroGenerator, MacroConfig, MacroGenerator};

    fn small_state() -> std::sync::Arc<WorldMacroState> {
        let g = DefaultMacroGenerator::new(MacroConfig { grid_level: 2, ..MacroConfig::default() });
        g.generate(0xCAFE_F00D, WorldShape::Sphere { radius_m: 6.371e6 })
    }

    #[test]
    fn pyramid_levels_and_tile_counts() {
        let s = small_state();
        let p = bake_world_summary(WorldAddr::ROOT, &s, 3, 4);
        assert_eq!(p.levels, 3);
        assert_eq!(p.tiles_per_level, vec![1, 4, 16]);
        assert_eq!(p.tiles.len(), 1 + 4 + 16);
        // Each tile has the requested size_px arrays.
        for t in &p.tiles {
            assert_eq!(t.elevation_m.len(), 16);
            assert_eq!(t.biome_id.len(), 16);
            assert_eq!(t.plate_id.len(), 16);
            assert_eq!(t.climate.len(), 16);
        }
    }

    #[test]
    fn tile_lookup_in_bounds() {
        let s = small_state();
        let p = bake_world_summary(WorldAddr::ROOT, &s, 2, 4);
        assert!(p.tile(0, [0, 0]).is_some());
        assert!(p.tile(1, [1, 1]).is_some());
        assert!(p.tile(1, [2, 0]).is_none()); // out of range
        assert!(p.tile(7, [0, 0]).is_none()); // level out of range
    }

    #[test]
    fn digest_stamped_on_pyramid() {
        let s = small_state();
        let p = bake_world_summary(WorldAddr::ROOT, &s, 1, 2);
        assert_eq!(p.macro_digest, s.digest);
    }
}
