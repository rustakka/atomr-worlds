//! Phase 14e — Regional / world overview mode.
//!
//! Given a pre-baked [`WorldSummaryPyramid`] (built once per world from
//! Phase 13c macro state — see [`crate::bake_world_summary`]), the
//! overview mode renders a 2D map at any zoom level in three
//! projections:
//!
//! - [`OverviewProjection::OrthographicFlat`] — top-down planar
//!   `(x_world, z_world) → (px, py)`. The pyramid stores its data in
//!   planar layout already (for cube/cylinder); we pick a level that
//!   matches the requested extent and blit the relevant tile rectangle.
//! - [`OverviewProjection::Equirectangular`] — full-sphere lat/lon
//!   rectangle. Per-pixel inverse of the equirectangular map; the
//!   pyramid is sampled at the global pixel coordinate.
//! - [`OverviewProjection::OrthographicSphere`] — globe-as-disk. Per
//!   pixel we run [`crate::projection_sphere::orthographic_sphere_pixel_to_dir`]
//!   to recover the direction, then look up the equirectangular pixel
//!   inside the pyramid via [`crate::projection_sphere::equirectangular_dir_to_pixel`].
//!
//! Output colour is driven by `biome_id`: a small built-in palette
//! covers the Phase 13c biome constants (see [`biome_color`]); unknown
//! ids fall back to [`crate::material_color`]. Elevation, climate, and
//! plate channels are *available* on every tile cell for downstream
//! overlays (e.g. tinting by temperature) — the default `render_overview`
//! only reads biome, but mode-specific renderers in later phases will
//! consume the other channels.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::shape::WorldShape;

use crate::derived::world_summary::{WorldSummaryPyramid, WorldSummaryTile};
use crate::projection_sphere::{equirectangular_dir_to_pixel, orthographic_sphere_pixel_to_dir};
use crate::raster2d::fill_rect;
use crate::render::{material_color, Framebuffer, RenderConfig};

/// Which of the three overview projections to use.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OverviewProjection {
    /// Top-down planar `(x_world, z_world)` orthographic. Best fit for
    /// cube/cylinder worlds; for spheres this is the "polar cap" view
    /// (north pole at image centre).
    OrthographicFlat,
    /// Globe seen from infinity along -Z. Disc-on-background; pixels
    /// outside the disc get [`RenderConfig::background`].
    OrthographicSphere,
    /// Full-sphere lat/lon rectangle.
    Equirectangular,
}

/// Camera parameters for the overview mode. `center` is the world-space
/// 2D anchor (longitude/latitude in radians for sphere projections;
/// `(x_world, z_world)` in meters for `OrthographicFlat`); `extent` is
/// the half-width of the visible window in the same units.
#[derive(Copy, Clone, Debug)]
pub struct OverviewCamera {
    pub center: [f64; 2],
    pub extent: f64,
    pub projection: OverviewProjection,
    pub aspect: f32,
}

/// Choose the pyramid level whose tile pixel-pitch best matches the
/// requested viewport pitch.
///
/// Coarse-by-default: when the viewport is small or the visible window
/// is wide, pick a low level (few large tiles). When the user zooms in
/// (small `extent`) or the viewport gets very tall, pick a finer level.
///
/// The picker is intentionally conservative — it caps at
/// `pyramid_levels - 1` so callers never sample past the baked pyramid.
pub fn pick_pyramid_level(cam: &OverviewCamera, viewport: [u32; 2], pyramid_levels: u8) -> u8 {
    if pyramid_levels <= 1 {
        return 0;
    }
    // Linear pixels per (world-extent unit). Equirectangular: extent is
    // in radians and the full image is 2π wide, so saturate at level
    // matching pixels/radian. Planar: extent is in meters; we treat the
    // world extent (`shape radius` would be needed for an exact match,
    // but we don't have shape here) as 1.0 to keep this a pure ratio.
    let v = viewport[0].max(viewport[1]) as f64;
    let extent = cam.extent.max(1e-12);
    // Heuristic: at extent == 1.0 (= "full world"), level 0 suffices for
    // viewport ≤ tile_pitch (~64 px). Each halving of extent justifies
    // one extra level. log2(1 / extent) + log2(v / 64) ≈ how much detail
    // we want; clamp to available range.
    let detail_log2 = (1.0 / extent).log2() + (v / 64.0).log2();
    let level = detail_log2.round().clamp(0.0, (pyramid_levels - 1) as f64) as u8;
    level.min(pyramid_levels - 1)
}

/// Built-in biome → RGB palette. Mirrors the Phase 13c `biome` constants
/// (ocean, ice, tundra, taiga, …). Unknown ids fall back to
/// [`material_color`] so a future expansion of the biome enum still
/// renders something rather than blank.
#[inline]
pub fn biome_color(biome: u8) -> [u8; 4] {
    let rgb = match biome {
        0 => [30, 60, 130],   // OCEAN — deep blue
        1 => [220, 230, 240], // ICE — pale blue-white
        2 => [180, 180, 165], // TUNDRA — grey-tan
        3 => [70, 95, 75],    // TAIGA — dark conifer green
        4 => [60, 130, 70],   // TEMPERATE_FOREST — forest green
        5 => [150, 175, 90],  // GRASSLAND — olive
        6 => [210, 195, 130], // DESERT — sand
        7 => [200, 175, 90],  // SAVANNA — dry yellow
        8 => [40, 110, 55],   // RAINFOREST — saturated green
        9 => [120, 105, 95],  // MOUNTAIN — rock brown
        _ => {
            let c = material_color(biome as u16);
            return [c[0], c[1], c[2], 255];
        }
    };
    [rgb[0], rgb[1], rgb[2], 255]
}

/// Render a complete overview frame.
///
/// The output is sized by `cfg.width × cfg.height` and contains pre-
/// multiplied opaque RGBA. Background pixels (outside the disc for
/// `OrthographicSphere`, and anywhere the projection is undefined) get
/// `cfg.background`.
pub fn render_overview(
    pyramid: &WorldSummaryPyramid,
    cam: &OverviewCamera,
    cfg: &RenderConfig,
) -> Framebuffer {
    let w = cfg.width;
    let h = cfg.height;
    let mut fb = Framebuffer {
        width: w,
        height: h,
        pixels: vec![0u8; (w as usize) * (h as usize) * 4],
        depth: vec![0.0f32; (w as usize) * (h as usize)],
    };
    // Background.
    fill_rect(&mut fb, 0, 0, w, h, cfg.background);

    let level = pick_pyramid_level(cam, [w, h], pyramid.levels);
    let n = 1u32 << level;
    let tile_size_px = pyramid.tile(level, [0, 0]).map(|t| t.size_px).unwrap_or(1);
    let full_w = n * tile_size_px;
    let full_h = n * tile_size_px;

    // `cam.center` is interpreted as a (yaw, pitch) offset (radians) for
    // sphere projections and as a normalised pan for planar flat. At
    // (0, 0) every branch reduces to the previous default — the golden
    // tests rely on that.
    let yaw = cam.center[0];
    let pitch = cam.center[1];
    match cam.projection {
        OverviewProjection::Equirectangular => {
            // Pan by shifting global pixel sampling: yaw maps to a
            // longitude shift (`full_w` per 2π), pitch to a latitude
            // shift (`full_h` per π).
            let dx = (yaw / (2.0 * core::f64::consts::PI) * full_w as f64) as i64;
            let dy = (pitch / core::f64::consts::PI * full_h as f64) as i64;
            for py in 0..h {
                for px in 0..w {
                    let base_x = (px as u64 * full_w as u64) / (w.max(1) as u64);
                    let base_y = (py as u64 * full_h as u64) / (h.max(1) as u64);
                    let gx = (base_x as i64 + dx).rem_euclid(full_w as i64) as u32;
                    let gy = (base_y as i64 + dy).clamp(0, full_h as i64 - 1) as u32;
                    let color = sample_global(pyramid, level, tile_size_px, gx, gy, n);
                    write_pixel(&mut fb, px, py, color);
                }
            }
        }
        OverviewProjection::OrthographicSphere => {
            // Disc on background. View axis starts at -Z (looking at
            // the +Z hemisphere from the front) and rotates by
            // (yaw, pitch) so dragging the mouse spins the globe.
            let view_axis = rotate_view_axis(DVec3::new(0.0, 0.0, -1.0), yaw, pitch);
            for py in 0..h {
                for px in 0..w {
                    let Some(dir) = orthographic_sphere_pixel_to_dir(px, py, w, h, view_axis) else {
                        continue;
                    };
                    let [gx, gy] = equirectangular_dir_to_pixel(dir, full_w, full_h);
                    let color = sample_global(pyramid, level, tile_size_px, gx, gy, n);
                    write_pixel(&mut fb, px, py, color);
                }
            }
        }
        OverviewProjection::OrthographicFlat => {
            // For sphere worlds, treat the flat projection as a
            // north-pole-centred azimuthal view (still done by sampling
            // an equirectangular direction so the pyramid's planar/
            // sphere distinction is invisible to the caller). For cube
            // and cylinder worlds the pyramid is already planar; we
            // still index it the same way.
            match pyramid.shape {
                WorldShape::Sphere { .. } => {
                    let view_axis = rotate_view_axis(DVec3::new(0.0, -1.0, 0.0), yaw, pitch);
                    for py in 0..h {
                        for px in 0..w {
                            let Some(dir) = orthographic_sphere_pixel_to_dir(px, py, w, h, view_axis) else {
                                continue;
                            };
                            let [gx, gy] = equirectangular_dir_to_pixel(dir, full_w, full_h);
                            let color = sample_global(pyramid, level, tile_size_px, gx, gy, n);
                            write_pixel(&mut fb, px, py, color);
                        }
                    }
                }
                _ => {
                    // Planar flat: yaw/pitch pan in normalised tile-space
                    // (full pyramid spans (-1, 1) in the same units used
                    // by `extent`).
                    let dx = (yaw * full_w as f64) as i64;
                    let dy = (pitch * full_h as f64) as i64;
                    for py in 0..h {
                        for px in 0..w {
                            let base_x = (px as u64 * full_w as u64) / (w.max(1) as u64);
                            let base_y = (py as u64 * full_h as u64) / (h.max(1) as u64);
                            let gx = (base_x as i64 + dx).clamp(0, full_w as i64 - 1) as u32;
                            let gy = (base_y as i64 + dy).clamp(0, full_h as i64 - 1) as u32;
                            let color = sample_global(pyramid, level, tile_size_px, gx, gy, n);
                            write_pixel(&mut fb, px, py, color);
                        }
                    }
                }
            }
        }
    }

    let _ = cam.aspect;
    fb
}

/// Rotate `axis` by (`yaw`, `pitch`): yaw rotates around world-Y, then
/// pitch tilts around the resulting right axis. At `(0, 0)` returns
/// `axis` unchanged so callers that haven't wired any input through
/// keep their previous output bit-for-bit (the overview golden tests
/// depend on this).
fn rotate_view_axis(axis: DVec3, yaw: f64, pitch: f64) -> DVec3 {
    // Yaw around +Y world axis.
    let (sin_y, cos_y) = yaw.sin_cos();
    let after_yaw = DVec3::new(
        cos_y * axis.x + sin_y * axis.z,
        axis.y,
        -sin_y * axis.x + cos_y * axis.z,
    );
    // Pitch around the right axis (= +Y × after_yaw, normalised). For
    // axes parallel to +Y the right axis is ill-defined; in that case
    // fall back to rotating around +X so a small pitch still moves the
    // pole.
    let up = DVec3::new(0.0, 1.0, 0.0);
    let rx = up.y * after_yaw.z - up.z * after_yaw.y;
    let ry = up.z * after_yaw.x - up.x * after_yaw.z;
    let rz = up.x * after_yaw.y - up.y * after_yaw.x;
    let rl = (rx * rx + ry * ry + rz * rz).sqrt();
    let right = if rl > 1e-9 {
        DVec3::new(rx / rl, ry / rl, rz / rl)
    } else {
        DVec3::new(1.0, 0.0, 0.0)
    };
    let (sin_p, cos_p) = pitch.sin_cos();
    // Rodrigues' rotation around `right`.
    let v = after_yaw;
    let dot = right.x * v.x + right.y * v.y + right.z * v.z;
    let cx = right.y * v.z - right.z * v.y;
    let cy = right.z * v.x - right.x * v.z;
    let cz = right.x * v.y - right.y * v.x;
    DVec3::new(
        v.x * cos_p + cx * sin_p + right.x * dot * (1.0 - cos_p),
        v.y * cos_p + cy * sin_p + right.y * dot * (1.0 - cos_p),
        v.z * cos_p + cz * sin_p + right.z * dot * (1.0 - cos_p),
    )
}

#[inline]
fn sample_global(
    pyramid: &WorldSummaryPyramid,
    level: u8,
    tile_size_px: u32,
    gx: u32,
    gy: u32,
    n: u32,
) -> [u8; 4] {
    let tx = (gx / tile_size_px).min(n.saturating_sub(1));
    let ty = (gy / tile_size_px).min(n.saturating_sub(1));
    let lx = gx % tile_size_px;
    let ly = gy % tile_size_px;
    let Some(tile) = pyramid.tile(level, [tx, ty]) else {
        return [0, 0, 0, 255];
    };
    biome_at(tile, lx, ly)
}

#[inline]
fn biome_at(tile: &WorldSummaryTile, lx: u32, ly: u32) -> [u8; 4] {
    let idx = (ly as usize) * (tile.size_px as usize) + (lx as usize);
    let b = *tile.biome_id.get(idx).unwrap_or(&0);
    biome_color(b)
}

#[inline]
fn write_pixel(fb: &mut Framebuffer, x: u32, y: u32, color: [u8; 4]) {
    let pi = ((y as usize) * (fb.width as usize) + (x as usize)) * 4;
    fb.pixels[pi] = color[0];
    fb.pixels[pi + 1] = color[1];
    fb.pixels[pi + 2] = color[2];
    fb.pixels[pi + 3] = color[3];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_level_small_extent_large_viewport() {
        // Zoomed in (small extent) with a big viewport → high detail.
        let cam = OverviewCamera {
            center: [0.0, 0.0],
            extent: 0.05,
            projection: OverviewProjection::Equirectangular,
            aspect: 1.0,
        };
        let l = pick_pyramid_level(&cam, [2048, 2048], 5);
        assert!(l >= 3, "expected fine detail level, got {}", l);
    }

    #[test]
    fn pick_level_full_world_small_viewport() {
        // Full world (extent = 1.0) on a small viewport → level 0.
        let cam = OverviewCamera {
            center: [0.0, 0.0],
            extent: 1.0,
            projection: OverviewProjection::Equirectangular,
            aspect: 1.0,
        };
        let l = pick_pyramid_level(&cam, [64, 64], 5);
        assert_eq!(l, 0);
    }
}
