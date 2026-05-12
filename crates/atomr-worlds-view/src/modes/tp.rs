//! Phase 14b — 3rd-person chase camera.
//!
//! A [`ChaseCamera`] orbits an `anchor` (a tracked DVec3 — usually the
//! player's position) at fixed `distance_m` and `height_m`. Per-tick we
//! advance a critical-damped smoothed copy of the anchor with the
//! closed-form expression
//!
//! ```text
//!   smoothed += (target - smoothed) * (1 - exp(-2π * f_c * dt))
//! ```
//!
//! where `f_c = smoothing_hz`. This is the exact continuous-time response
//! evaluated at `dt`, so the camera converges identically across any time
//! step (no implicit Euler drift). At `dt → 0` the second factor goes to
//! `2π * f_c * dt` (linear), at `dt → ∞` it saturates at `1` (camera snaps).
//!
//! Rendering re-uses [`build_fp_scene`](super::fp::build_fp_scene): the
//! chase camera produces a standard [`Camera`] (perspective FOV, world-up,
//! looking at the smoothed anchor), and `extra_meshes` lets the caller
//! inject the anchor decal so the player sees themself.

use std::f64::consts::PI;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::lod::Lod;

use crate::camera::{Camera, Projection};
use crate::modes::fp::build_fp_scene;
use crate::render::{render_composite, Framebuffer, RenderConfig};
use crate::scene::MeshNode;
use crate::world_query::WorldQuery;

/// Orbiting 3rd-person camera.
#[derive(Copy, Clone, Debug)]
pub struct ChaseCamera {
    /// Most-recently-pushed anchor pose (typically the player's position).
    pub anchor: DVec3,
    /// Smoothed copy of `anchor` — what the camera actually looks at and
    /// orbits around. Equals `anchor` at construction.
    pub smoothed_anchor: DVec3,
    /// Yaw / pitch around the anchor, radians.
    pub yaw: f32,
    pub pitch: f32,
    /// Distance from the smoothed anchor to the camera along the orbit
    /// vector (in meters).
    pub distance_m: f32,
    /// Extra height above the smoothed anchor (camera sits at
    /// `smoothed_anchor + up * height_m + orbit`).
    pub height_m: f32,
    pub fov_y_rad: f32,
    pub aspect: f32,
    /// First-order smoothing cutoff in Hz. The exponential time constant is
    /// `1 / (2π * smoothing_hz)`. 4 Hz feels brisk-but-not-snappy.
    pub smoothing_hz: f32,
}

impl ChaseCamera {
    pub fn new(anchor: DVec3, aspect: f32) -> Self {
        Self {
            anchor,
            smoothed_anchor: anchor,
            yaw: 0.0,
            pitch: -0.2,
            distance_m: 6.0,
            height_m: 2.0,
            fov_y_rad: std::f32::consts::FRAC_PI_3,
            aspect,
            smoothing_hz: 4.0,
        }
    }

    /// Advance one tick. Updates `anchor`, smooths it toward
    /// `smoothed_anchor`, and applies yaw/pitch deltas.
    pub fn tick(&mut self, new_anchor: DVec3, yaw_delta: f32, pitch_delta: f32, dt_s: f32) {
        self.anchor = new_anchor;
        // Closed-form first-order low-pass: `smoothed += (target - smoothed)
        // * (1 - exp(-2π * f_c * dt))`. f_c = smoothing_hz.
        let k = 1.0 - (-2.0 * PI * self.smoothing_hz as f64 * dt_s as f64).exp();
        let dx = new_anchor.x - self.smoothed_anchor.x;
        let dy = new_anchor.y - self.smoothed_anchor.y;
        let dz = new_anchor.z - self.smoothed_anchor.z;
        self.smoothed_anchor = DVec3::new(
            self.smoothed_anchor.x + dx * k,
            self.smoothed_anchor.y + dy * k,
            self.smoothed_anchor.z + dz * k,
        );
        self.yaw += yaw_delta;
        self.pitch = (self.pitch + pitch_delta).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Build the camera matrix for the current pose.
    pub fn camera(&self) -> Camera {
        // Orbit vector from the smoothed anchor. yaw=0 ⇒ camera sits at
        // -Z relative to the anchor (looking in +Z); positive pitch raises
        // the camera. Forward (from camera → anchor) is the negation of
        // that orbit vector.
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let orbit =
            [-sin_y * cos_p * self.distance_m, sin_p * self.distance_m, -cos_y * cos_p * self.distance_m];
        let anchor_f =
            [self.smoothed_anchor.x as f32, self.smoothed_anchor.y as f32, self.smoothed_anchor.z as f32];
        let eye = [anchor_f[0] + orbit[0], anchor_f[1] + orbit[1] + self.height_m, anchor_f[2] + orbit[2]];
        Camera {
            eye,
            target: anchor_f,
            up: [0.0, 1.0, 0.0],
            fov_y_rad: self.fov_y_rad,
            aspect: self.aspect,
            near: 0.1,
            far: 1024.0,
            projection: Projection::Perspective { fov_y_rad: self.fov_y_rad },
        }
    }
}

const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Render a 3rd-person frame. Delegates to [`build_fp_scene`] (the brick
/// pipeline doesn't care whether the camera is 1st- or 3rd-person — only
/// the camera matrix matters) and pipes `extra_meshes` through unchanged so
/// the caller can attach an anchor decal.
pub fn render_tp(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    cam: &Camera,
    lod: Lod,
    region_m: f32,
    extra_meshes: &[MeshNode],
    cfg: &RenderConfig,
) -> Framebuffer {
    let scene = build_fp_scene(world, addr, cam, lod, region_m, extra_meshes);
    let composite = scene.as_composite(cam, region_m);
    render_composite(&composite, cam, cfg)
}
