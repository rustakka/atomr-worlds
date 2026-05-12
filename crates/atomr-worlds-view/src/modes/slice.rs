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
    /// (`up = +Z` so the framebuffer's +y is world +Z — the same convention
    /// the renderer below uses when placing rects). `near = 0.1`, `far` is
    /// generous (`band_thickness + 1024`) so any reasonable band fits.
    pub fn to_camera(&self) -> Camera {
        let eye_y = (self.z_band_top as f32) + 1.0;
        let target_y = (self.z_band_top - self.z_band_thickness as i32) as f32;
        Camera {
            eye: [self.center_xz[0], eye_y, self.center_xz[1]],
            target: [self.center_xz[0], target_y, self.center_xz[1]],
            // World +Z is "down" on screen so framebuffer +y aligns with
            // world +Z — matches the rect placement in `render_slice`.
            up: [0.0, 0.0, 1.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect: self.aspect.max(1e-6),
            near: 0.1,
            far: (self.z_band_thickness as f32) + 1024.0,
            projection: Projection::Orthographic { half_height_m: self.half_height_m.max(1e-3) },
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SliceConfig {
    pub width: u32,
    pub height: u32,
    pub tile_px: u32,
    pub stipple_thin_features: bool,
    pub roof_alpha: f32,
    pub background: [u8; 4],
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
/// px = round((world_x - cam.center_xz[0]) * tile_px) + width/2
/// py = round((world_z - cam.center_xz[1]) * tile_px) + height/2
/// ```
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

    for z_idx in 0..table.dims[1] {
        for x_idx in 0..table.dims[0] {
            let col = match table.column(x_idx, z_idx) {
                Some(c) => c,
                None => continue,
            };
            let world_x = table.origin_xz[0] as f32 + x_idx as f32;
            let world_z = table.origin_xz[1] as f32 + z_idx as f32;
            let px = ((world_x - cam.center_xz[0]) * tile_pxf + half_w).round() as i32;
            let py = ((world_z - cam.center_xz[1]) * tile_pxf + half_h).round() as i32;

            if col.top_voxel.is_empty() {
                // Roof / open sky overlay — a translucent wash so the
                // background reads through.
                let alpha = (cfg.roof_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                if alpha > 0 {
                    blend_rect(&mut fb, px, py, tile_px, tile_px, [255, 255, 255, alpha]);
                }
                continue;
            }

            let color = rgba_from_voxel(palette, col.top_voxel.0);
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
        // Center of fb. The column is at world (0, 0); cam center is
        // (0.5, 0.5), tile_px=4, so px = round((0 - 0.5)*4 + 8) = 6,
        // py = 6. The tile is 4×4 → pixels (6..10, 6..10).
        let pi = ((8 * 16 + 8) * 4) as usize;
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
        // Center pixel should be a 50% blend of white over black ≈ 127.
        let pi = ((4 * 8 + 4) * 4) as usize;
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
    }
}
