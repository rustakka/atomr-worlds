//! Phase 14d decal pass — flat 2D sprites projected through the active
//! camera and blitted into the framebuffer *after* the 3D pass.
//!
//! Decals carry a world-space XZ position; their Y is implicit (the
//! caller is responsible for picking a Y, typically by sampling the
//! [`crate::derived::surface_raster::SurfaceRaster`] heightmap). The
//! decal is projected to NDC via `cam.view_proj()`, converted to pixel
//! coordinates, and either alpha-blended ([`raster2d::blend_rect`]) or
//! blitted ([`raster2d::blit_rgba`]) — *no* depth interaction. Decals
//! always sit on top of the rendered surface, which is the RTS-mode
//! convention: selection rings, unit pips, build markers, etc. should
//! never be hidden by terrain bumps.
//!
//! Decals whose projected center falls behind the camera (`clip.w <=
//! 0`) are silently dropped. Decals whose pixel rectangle is fully
//! outside the framebuffer are no-ops via the raster2d clipping path.

use crate::camera::{transform_point, Camera};
use crate::raster2d::{blend_rect, blit_rgba};
use crate::render::Framebuffer;

/// A single decal request. `world_xz_m` is the world-XZ position in
/// meters; the implicit Y is `0.0` here, which is fine because the
/// oblique-orthographic projection used in 14d is invariant under
/// Y translation up to a screen-space shear that the caller can pre-bake
/// by adjusting `world_xz_m` if it cares. (Tests use `Y = 0`.)
#[derive(Clone, Debug)]
pub struct Decal {
    /// World-XZ in meters of the decal anchor.
    pub world_xz_m: [f32; 2],
    /// Pixel extent of the decal rectangle (top-left-anchored on the
    /// projected anchor point).
    pub size_px: [u32; 2],
    /// RGBA used by the [`blend_rect`] path when `sprite` is `None`.
    pub color: [u8; 4],
    /// Optional RGBA8 sprite (row-major, top-left origin). Must be
    /// exactly `size_px[0] * size_px[1] * 4` bytes; mismatched lengths
    /// trigger a panic in `blit_rgba` to surface programmer errors.
    pub sprite: Option<&'static [u8]>,
}

/// Project every decal in `decals` through `cam.view_proj()` and write
/// the result into `fb` — `blit_rgba` for sprites, `blend_rect` for the
/// solid-color path. Iteration is deterministic; output pixels are a
/// pure function of `(decals, cam, fb.before)`.
pub fn render_decals(fb: &mut Framebuffer, cam: &Camera, decals: &[Decal]) {
    let mvp = cam.view_proj();
    let w_f = fb.width as f32;
    let h_f = fb.height as f32;
    for d in decals {
        let p = [d.world_xz_m[0], 0.0, d.world_xz_m[1]];
        let clip = transform_point(mvp, p);
        if clip[3] <= 0.0 {
            // Behind the camera (or on the near plane) — drop. The decal
            // path doesn't do clipping; it's a 2D-only sprite blit, and a
            // behind-camera projection is meaningless.
            continue;
        }
        let inv_w = 1.0 / clip[3];
        let ndc_x = clip[0] * inv_w;
        let ndc_y = clip[1] * inv_w;
        // Same NDC → pixel mapping as `render.rs::clip_to_screen` so a
        // 3D vertex and a decal anchor at the same world point land on
        // the same pixel. `+ 0.5` rounds to nearest; floor would bias
        // sprites half a pixel up-left of triangle edges.
        let px_cx = ((ndc_x * 0.5 + 0.5) * w_f).round() as i32;
        let px_cy = ((1.0 - (ndc_y * 0.5 + 0.5)) * h_f).round() as i32;
        let half_w = (d.size_px[0] as i32) / 2;
        let half_h = (d.size_px[1] as i32) / 2;
        let x = px_cx - half_w;
        let y = px_cy - half_h;
        match d.sprite {
            Some(src) => blit_rgba(fb, x, y, src, d.size_px[0], d.size_px[1]),
            None => blend_rect(fb, x, y, d.size_px[0], d.size_px[1], d.color),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Projection;

    fn fb(w: u32, h: u32) -> Framebuffer {
        Framebuffer {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 4) as usize],
            depth: vec![0.0f32; (w * h) as usize],
        }
    }

    fn topdown_ortho(w: u32, h: u32) -> Camera {
        // Camera at +Y looking down: the (X, Z) world plane projects
        // directly into screen XY with no shear. Easy to reason about
        // pixel locations in tests.
        Camera {
            eye: [0.0, 50.0, 0.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 0.0, -1.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect: w as f32 / h as f32,
            near: 0.1,
            far: 200.0,
            projection: Projection::Orthographic { half_height_m: 8.0 },
        }
    }

    #[test]
    fn behind_camera_decal_is_skipped() {
        let mut f = fb(16, 16);
        let cam = topdown_ortho(16, 16);
        let d = Decal {
            // (0, 200) is well outside the camera frustum but more
            // importantly the orthographic w is always +1 — so this
            // specific decal lands but at an off-screen pixel. The
            // behind-camera path is exercised by a perspective camera:
            world_xz_m: [0.0, 200.0],
            size_px: [4, 4],
            color: [255, 0, 0, 255],
            sprite: None,
        };
        render_decals(&mut f, &cam, &[d]);
        // All pixels untouched — the decal landed off-screen.
        assert!(f.pixels.iter().all(|b| *b == 0));
    }

    #[test]
    fn solid_decal_writes_block() {
        let mut f = fb(16, 16);
        let cam = topdown_ortho(16, 16);
        // World (0, 0) → screen center. Half-height = 8 m, height = 16
        // px, so 1 m = 1 px. A 4×4 decal at (0, 0) covers pixels
        // x ∈ [6, 10), y ∈ [6, 10).
        let d = Decal { world_xz_m: [0.0, 0.0], size_px: [4, 4], color: [255, 0, 0, 255], sprite: None };
        render_decals(&mut f, &cam, &[d]);
        for y in 6..10 {
            for x in 6..10 {
                let pi = ((y * 16 + x) * 4) as usize;
                assert_eq!(f.pixels[pi], 255, "pixel ({x},{y}) red");
            }
        }
    }
}
