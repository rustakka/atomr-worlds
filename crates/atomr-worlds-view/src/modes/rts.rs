//! Phase 14d — RTS oblique-orthographic display mode.
//!
//! The pipeline is:
//!
//! 1. Caller has a [`SurfaceRaster`](crate::derived::surface_raster::SurfaceRaster)
//!    of the region they want to render.
//! 2. Build a flat mesh from it via
//!    [`surface_raster_to_mesh`](crate::derived::surface_raster::surface_raster_to_mesh).
//! 3. Project that through the existing 3D rasterizer
//!    ([`render_mesh`](crate::render::render_mesh)) using
//!    [`Projection::Oblique`](crate::camera::Projection::Oblique).
//! 4. Run [`render_decals`](crate::decals::render_decals) over the result
//!    for unit / selection / annotation sprites.
//!
//! Decals are 2D-only and never participate in depth; this matches the
//! RTS-genre convention (selection rings should always be visible).
//!
//! The camera is wrapped in [`ObliqueCamera`] so callers can specify
//! the parameters in RTS-native units (center, rotation, m/px) rather
//! than the renderer's internal `eye/target/up` triple.

use crate::camera::{Camera, Projection};
use crate::decals::{render_decals, Decal};
use crate::derived::surface_raster::{surface_raster_to_mesh, SurfaceRaster};
use crate::render::{render_mesh, Framebuffer, RenderConfig};
use crate::scene::MaterialPalette;

/// RTS / Warcraft-style oblique-orthographic camera. The camera looks
/// down on the XZ plane from a high eye position; `rotation_deg`
/// controls the in-plane heading and `scale_m_per_px` controls zoom
/// (smaller value = more zoomed in).
#[derive(Copy, Clone, Debug)]
pub struct ObliqueCamera {
    /// World-XZ position the camera centers on, in meters.
    pub center_xz: [f32; 2],
    /// In-plane rotation of the camera, degrees. `0` aligns world-X
    /// with screen-right and world-Z with screen-up.
    pub rotation_deg: f32,
    /// Meters per output pixel along Y. Smaller = more zoomed in.
    pub scale_m_per_px: f32,
    /// Near plane (meters). Reversed-z depth maps `-near` to 1.0.
    pub near: f32,
    /// Far plane (meters). Reversed-z depth maps `-far` to 0.0.
    pub far: f32,
    /// Output viewport aspect ratio (width / height).
    pub aspect: f32,
}

impl Default for ObliqueCamera {
    fn default() -> Self {
        Self {
            center_xz: [0.0, 0.0],
            rotation_deg: 0.0,
            scale_m_per_px: 0.25,
            near: 0.1,
            far: 1000.0,
            aspect: 1.0,
        }
    }
}

impl ObliqueCamera {
    /// Convert to the renderer-internal [`Camera`] with
    /// [`Projection::Oblique`]. The view is set up so the camera eye
    /// sits directly above `center_xz` (high `Y`) and looks straight
    /// down; the projection's shear is what produces the oblique look.
    pub fn to_camera(&self) -> Camera {
        // High eye, look straight down at the center. The
        // shear in `Projection::Oblique` is what makes the projection
        // axonometric — the view itself stays top-down so we can
        // exploit Y-translation invariance of orthographic projection
        // for sub-surface decal placement.
        let eye_y = self.far * 0.5;
        let eye = [self.center_xz[0], eye_y, self.center_xz[1]];
        let target = [self.center_xz[0], 0.0, self.center_xz[1]];
        // For a straight-down look, +Z in world maps to "up" on screen
        // (the conventional RTS layout). We negate so screen-up is
        // -world-Z, matching the orthographic camera the tests use.
        let up = [0.0, 0.0, -1.0];
        Camera {
            eye,
            target,
            up,
            fov_y_rad: std::f32::consts::FRAC_PI_4, // unused by oblique projection
            aspect: self.aspect,
            near: self.near,
            far: self.far,
            projection: Projection::Oblique {
                rotation_deg: self.rotation_deg,
                scale_m_per_px: self.scale_m_per_px,
            },
        }
    }
}

/// Render a [`SurfaceRaster`] + decals through the oblique camera. The
/// pixel-byte output is a pure function of the inputs (the underlying
/// rasterizer is deterministic and the decal pass uses fixed iteration
/// order).
pub fn render_rts(
    raster: &SurfaceRaster,
    decals: &[Decal],
    cam: &ObliqueCamera,
    palette: &MaterialPalette,
    cfg: &RenderConfig,
) -> Framebuffer {
    let mesh = surface_raster_to_mesh(raster, palette);
    let camera = cam.to_camera();
    let mut fb = render_mesh(&mesh, &camera, cfg);
    render_decals(&mut fb, &camera, decals);
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_camera_is_oblique() {
        let oc = ObliqueCamera { rotation_deg: 30.0, scale_m_per_px: 0.5, ..Default::default() };
        let cam = oc.to_camera();
        match cam.projection {
            Projection::Oblique { rotation_deg, scale_m_per_px } => {
                assert!((rotation_deg - 30.0).abs() < 1e-5);
                assert!((scale_m_per_px - 0.5).abs() < 1e-5);
            }
            _ => panic!("expected Projection::Oblique"),
        }
    }
}
