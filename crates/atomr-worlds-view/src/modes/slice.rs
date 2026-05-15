//! Phase 14c — Dwarf-Fortress-style horizontal slice renderer.
//!
//! Top-down orthographic tile renderer. Build a [`SliceTable`] from the
//! [`WorldQuery`] (cached via [`ViewCache`]), then blit one rect per
//! horizontal column using the 2D rasterizer ([`crate::raster2d`]) — no
//! triangles, no z-buffer interaction. The scale on screen is set by
//! `SliceConfig::tile_px`; the camera's `half_height_m` is computed from the
//! framebuffer height so a one-voxel column always lands as a tile_px-square
//! pixel-aligned rect.
//!
//! See [`crate::derived::slice_index`] for the "+Y up, scanning down" rule.

use crate::camera::{Camera, Projection};
use crate::derived::slice_index::{build_slice_table, SliceKey, SliceTable};
use crate::raster2d::{blend_rect, fill_rect, fill_rect_stipple, StipplePattern};
use crate::render::{material_color, Framebuffer};
use crate::scene::MaterialPalette;
use crate::view_cache::ViewCache;
use crate::world_query::WorldQuery;

use atomr_worlds_core::addr::WorldAddr;

/// Top-down orthographic camera for a slice. The view plane is the
/// (world-X, world-Z) plane at world-Y `z_band_top`; the camera "looks down"
/// −Y onto that plane.
#[derive(Copy, Clone, Debug)]
pub struct SliceCamera {
    /// Horizontal-plane center of the view, in **world voxel units**. A
    /// fractional value pans without snapping to a tile boundary.
    pub center_xz: [f32; 2],
    pub z_band_top: i32,
    pub z_band_thickness: u8,
    /// Half-height of the orthographic frustum in **world voxel units**.
    /// The renderer's pixel size is set by [`SliceConfig::tile_px`]; this
    /// field controls the on-screen world extent for the 3D [`Camera`]
    /// (used by downstream consumers that need the matrix, e.g. tests that
    /// project world points to screen).
    pub half_height_m: f32,
    pub aspect: f32,
}

impl SliceCamera {
    /// Convert to a generic [`Camera`] with [`Projection::Orthographic`].
    ///
    /// The eye sits one voxel above the band top looking straight down
    /// (`up = -Z` so the framebuffer's +y is world -Z — the same
    /// convention the renderer below uses when placing rects). `near =
    /// 0.1`, `far` is generous (`band_thickness + 1024`) so any reasonable
    /// band fits.
    pub fn to_camera(&self) -> Camera {
        let eye_y = (self.z_band_top as f32) + 1.0;
        let target_y = (self.z_band_top - self.z_band_thickness as i32) as f32;
        Camera {
            eye: [self.center_xz[0], eye_y, self.center_xz[1]],
            target: [self.center_xz[0], target_y, self.center_xz[1]],
            // World -Z is "up" on screen so framebuffer +y aligns with
            // world -Z — matches the rect placement in `render_slice` and
            // the first-person view's screen orientation.
            up: [0.0, 0.0, -1.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect: self.aspect.max(1e-6),
            near: 0.1,
            far: (self.z_band_thickness as f32) + 1024.0,
            projection: Projection::Orthographic { half_height_m: self.half_height_m.max(1e-3) },
        }
    }
}

/// How [`render_slice`] shades each column.
#[derive(Copy, Clone, Debug)]
pub enum SliceShading {
    /// Flat fill with the palette's `base_color` — the historical look.
    Flat,
    /// Hillshade relief: a per-column surface normal is derived from the
    /// neighbouring columns' `top_z` height field and lit by
    /// [`SliceConfig::light_dir_xz_y`]. `ambient` is the unlit floor
    /// (`0.0` = black shadows, `1.0` = no shading); `relief_strength`
    /// scales the height gradient before the normal is built.
    Hillshade { ambient: f32, relief_strength: f32 },
}

#[derive(Copy, Clone, Debug)]
pub struct SliceConfig {
    pub width: u32,
    pub height: u32,
    pub tile_px: u32,
    pub stipple_thin_features: bool,
    pub roof_alpha: f32,
    pub background: [u8; 4],
    /// Per-column shading mode. [`SliceShading::Flat`] preserves the
    /// historical flat-fill look.
    pub shading: SliceShading,
    /// Sun direction FROM the sun INTO the scene, reordered as
    /// `[world_x, world_z, world_y]` so it lines up with the slice's
    /// `(x, z)` tile plane. Only consulted when `shading` is
    /// [`SliceShading::Hillshade`]; need not be normalized.
    pub light_dir_xz_y: [f32; 3],
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            tile_px: 4,
            stipple_thin_features: true,
            roof_alpha: 0.25,
            background: [20, 20, 28, 255],
            shading: SliceShading::Flat,
            light_dir_xz_y: [-0.4, -0.3, -0.8],
        }
    }
}

/// Render a pre-built [`SliceTable`] to an RGBA framebuffer.
///
/// The mapping is straight pixel arithmetic: column `(x, z)` (table-local)
/// covers framebuffer rect `(px, py, tile_px, tile_px)` where
///
/// ```text
/// world_x = origin_xz[0] + x_local
/// world_z = origin_xz[1] + z_local
/// px = round((cam.center_xz[0] - world_x) * tile_px) + width/2
/// py = round((cam.center_xz[1] - world_z) * tile_px) + height/2
/// ```
///
/// Both axes negate `(world - center)` so screen-right is world `-X` and
/// screen-up is world `+Z` — the slice raster is oriented to match the
/// first-person view, whose camera faces world `+Z` with screen-right at
/// world `-X`.
///
/// When `cfg.shading` is [`SliceShading::Hillshade`] each non-empty
/// column's colour is multiplied by a relief factor derived from the
/// neighbouring columns' `top_z` and `cfg.light_dir_xz_y`.
///
/// Columns whose `top_voxel` is empty are blended with `cfg.roof_alpha` so
/// open space reads as "covered by sky" rather than the raw background.
pub fn render_slice(
    table: &SliceTable,
    cam: &SliceCamera,
    palette: &MaterialPalette,
    cfg: &SliceConfig,
) -> Framebuffer {
    let mut fb = Framebuffer {
        width: cfg.width,
        height: cfg.height,
        pixels: Vec::with_capacity((cfg.width * cfg.height * 4) as usize),
        depth: vec![0.0f32; (cfg.width * cfg.height) as usize],
    };
    for _ in 0..(cfg.width * cfg.height) {
        fb.pixels.extend_from_slice(&cfg.background);
    }

    let tile_px = cfg.tile_px.max(1);
    let half_w = (cfg.width as f32) * 0.5;
    let half_h = (cfg.height as f32) * 0.5;
    let tile_pxf = tile_px as f32;
    let thin_threshold: u8 = 2;

    // Effective height of a neighbouring column for hillshading: the
    // neighbour's `top_z`, or `fallback` (the centre column's own height)
    // when the neighbour is off the table edge or has an empty top — so a
    // table edge reads flat instead of as a cliff.
    let height_at = |xi: u32, zi: u32, fallback: f32| -> f32 {
        match table.column(xi, zi) {
            Some(c) if !c.top_voxel.is_empty() => c.top_z as f32,
            _ => fallback,
        }
    };

    for z_idx in 0..table.dims[1] {
        for x_idx in 0..table.dims[0] {
            let col = match table.column(x_idx, z_idx) {
                Some(c) => c,
                None => continue,
            };
            let world_x = table.origin_xz[0] as f32 + x_idx as f32;
            let world_z = table.origin_xz[1] as f32 + z_idx as f32;
            let px = ((cam.center_xz[0] - world_x) * tile_pxf + half_w).round() as i32;
            let py = ((cam.center_xz[1] - world_z) * tile_pxf + half_h).round() as i32;

            if col.top_voxel.is_empty() {
                // Roof / open sky overlay — a translucent wash so the
                // background reads through.
                let alpha = (cfg.roof_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                if alpha > 0 {
                    blend_rect(&mut fb, px, py, tile_px, tile_px, [255, 255, 255, alpha]);
                }
                continue;
            }

            let mut color = rgba_from_voxel(palette, col.top_voxel.0);
            if let SliceShading::Hillshade { ambient, relief_strength } = cfg.shading {
                let h_c = col.top_z as f32;
                let factor = hillshade_factor(
                    height_at(x_idx.wrapping_sub(1), z_idx, h_c),
                    height_at(x_idx + 1, z_idx, h_c),
                    height_at(x_idx, z_idx.wrapping_sub(1), h_c),
                    height_at(x_idx, z_idx + 1, h_c),
                    cfg.light_dir_xz_y,
                    ambient,
                    relief_strength,
                );
                color = shade_rgb(color, factor);
            }
            if cfg.stipple_thin_features && col.thickness_above_floor < thin_threshold {
                fill_rect_stipple(&mut fb, px, py, tile_px, tile_px, color, StipplePattern::Dense75);
            } else {
                fill_rect(&mut fb, px, py, tile_px, tile_px, color);
            }
        }
    }
    fb
}

/// One-shot: build (or fetch from cache) a [`SliceTable`] and render it.
///
/// The cache key is constructed from `addr`, `origin_xz`, `dims`, and the
/// camera's `z_band_top` / `z_band_thickness`. Eviction is the caller's
/// responsibility — wire a host `RegionDelta` listener to
/// [`ViewCache::invalidate_intersecting`].
pub fn render_slice_cached(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    cache: &ViewCache<SliceKey, SliceTable>,
    cam: &SliceCamera,
    dims: [u32; 2],
    origin_xz: [i32; 2],
    palette: &MaterialPalette,
    cfg: &SliceConfig,
) -> Framebuffer {
    let key = SliceKey {
        addr: *addr,
        origin_xz,
        dims,
        z_band_top: cam.z_band_top,
        z_band_thickness: cam.z_band_thickness,
    };
    let table = cache.get_or_build(key, || {
        build_slice_table(world, addr, origin_xz, dims, cam.z_band_top, cam.z_band_thickness)
    });
    render_slice(&table, cam, palette, cfg)
}

#[inline]
fn rgba_from_voxel(palette: &MaterialPalette, mat: u16) -> [u8; 4] {
    if let Some(e) = palette.entries.get(mat as usize) {
        let c = e.base_color;
        [linear_to_u8(c[0]), linear_to_u8(c[1]), linear_to_u8(c[2]), 255]
    } else {
        let c = material_color(mat);
        [c[0], c[1], c[2], 255]
    }
}

#[inline]
fn linear_to_u8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Per-column relief shading factor for [`SliceShading::Hillshade`].
///
/// `h_xn` / `h_xp` / `h_zn` / `h_zp` are the world-Y heights of the four
/// axis neighbours (x-1, x+1, z-1, z+1); callers pass the centre column's
/// own height for off-edge / empty neighbours so the table edge reads
/// flat. `light_dir_xz_y` is the sun direction FROM the sun INTO the
/// scene, components `[world_x, world_z, world_y]`; it need not be
/// normalized. The return value is in `[0, 1]` — `ambient` at the unlit
/// floor, up to `1.0` on a fully sun-facing slope.
#[inline]
fn hillshade_factor(
    h_xn: f32,
    h_xp: f32,
    h_zn: f32,
    h_zp: f32,
    light_dir_xz_y: [f32; 3],
    ambient: f32,
    relief_strength: f32,
) -> f32 {
    // Central-difference gradient of the height field (1-voxel spacing).
    let dz_dx = (h_xp - h_xn) * 0.5 * relief_strength;
    let dz_dz = (h_zp - h_zn) * 0.5 * relief_strength;
    // Surface normal of the height field y = h(x, z).
    let (nx, ny, nz) = (-dz_dx, 1.0, -dz_dz);
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
    // Surface-to-light is the negated incoming sun direction. The config
    // packs the direction as [world_x, world_z, world_y].
    let (lx, ly, lz) = (-light_dir_xz_y[0], -light_dir_xz_y[2], -light_dir_xz_y[1]);
    let l_len = (lx * lx + ly * ly + lz * lz).sqrt().max(1e-6);
    let lambert = ((nx * lx + ny * ly + nz * lz) / (n_len * l_len)).max(0.0);
    let ambient = ambient.clamp(0.0, 1.0);
    (ambient + (1.0 - ambient) * lambert).clamp(0.0, 1.0)
}

/// Multiply a column colour's RGB by a shading `factor`, leaving alpha
/// untouched.
#[inline]
fn shade_rgb(c: [u8; 4], factor: f32) -> [u8; 4] {
    let f = factor.clamp(0.0, 1.0);
    [
        (c[0] as f32 * f).round() as u8,
        (c[1] as f32 * f).round() as u8,
        (c[2] as f32 * f).round() as u8,
        c[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::slice_index::SliceColumn;
    use atomr_worlds_voxel::voxel::Voxel;

    fn make_table_1x1(top: Voxel, thickness_above: u8) -> SliceTable {
        SliceTable {
            columns: vec![SliceColumn { top_voxel: top, top_z: 0, thickness_above_floor: thickness_above }],
            dims: [1, 1],
            origin_xz: [0, 0],
            z_band_top: 1,
            z_band_thickness: 3,
            world_rev: 0,
        }
    }

    fn slice_cam() -> SliceCamera {
        SliceCamera {
            center_xz: [0.5, 0.5],
            z_band_top: 1,
            z_band_thickness: 3,
            half_height_m: 4.0,
            aspect: 1.0,
        }
    }

    #[test]
    fn renders_solid_column_with_palette_color() {
        let table = make_table_1x1(Voxel::new(1), 2);
        let cam = slice_cam();
        let cfg = SliceConfig { width: 16, height: 16, tile_px: 4, ..Default::default() };
        let pal = MaterialPalette::default();
        let fb = render_slice(&table, &cam, &pal, &cfg);
        // The column is at world (0, 0); cam center is (0.5, 0.5),
        // tile_px=4. With the FP-aligned mapping (screen-right = world -X)
        // px = round((0.5 - 0)*4 + 8) = 10, py = 10 → tile (10..14, 10..14).
        let pi = ((11 * 16 + 11) * 4) as usize;
        assert_eq!(&fb.pixels[pi..pi + 3], &material_color(1));
    }

    #[test]
    fn empty_column_blends_roof_alpha() {
        let table = make_table_1x1(Voxel::EMPTY, 0);
        let cam = slice_cam();
        let cfg = SliceConfig {
            width: 8,
            height: 8,
            tile_px: 4,
            roof_alpha: 0.5,
            background: [0, 0, 0, 255],
            ..Default::default()
        };
        let pal = MaterialPalette::default();
        let fb = render_slice(&table, &cam, &pal, &cfg);
        // px = round((0.5 - 0)*4 + 4) = 6, py = 6 → tile (6..10, 6..10)
        // clipped to the 8×8 fb. A pixel inside (e.g. (7, 7)) should be a
        // 50% blend of white over black ≈ 127.
        let pi = ((7 * 8 + 7) * 4) as usize;
        assert!(fb.pixels[pi] > 100 && fb.pixels[pi] < 160);
    }

    #[test]
    fn slice_camera_emits_orthographic() {
        let cam = slice_cam();
        let camera = cam.to_camera();
        match camera.projection {
            Projection::Orthographic { half_height_m } => {
                assert!((half_height_m - 4.0).abs() < 1e-6);
            }
            other => panic!("expected Orthographic, got {other:?}"),
        }
        // World -Z is "up" on screen — the FP-aligned orientation.
        assert_eq!(camera.up, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn hillshade_lit_slope_brighter_than_shadowed() {
        // A slope rising toward +x (neighbour heights h_xn=0, h_xp=2) has
        // a surface normal tilting toward -x. A sun on the -x side (light
        // travelling toward +x → positive world-x component) lights it;
        // a sun on the +x side leaves it in shadow.
        let ambient = 0.3;
        let lit_light = [1.0, 0.0, -0.5]; // sun on -x side
        let dark_light = [-1.0, 0.0, -0.5]; // sun on +x side
        let lit = hillshade_factor(0.0, 2.0, 1.0, 1.0, lit_light, ambient, 1.0);
        let dark = hillshade_factor(0.0, 2.0, 1.0, 1.0, dark_light, ambient, 1.0);
        assert!(lit > dark, "sun-facing slope must be brighter: lit={lit} dark={dark}");
        assert!(dark >= ambient - 1e-6, "shaded side must not dip below ambient");
        assert!(lit <= 1.0 && dark <= 1.0, "factor stays in [0, 1]");
        // Flat ground (no gradient) lands between ambient and full bright.
        let flat = hillshade_factor(1.0, 1.0, 1.0, 1.0, lit_light, ambient, 1.0);
        assert!(flat > ambient && flat <= 1.0, "flat ground: {flat}");
    }
}
