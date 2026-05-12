//! Camera with view+projection matrices and MetricScale-driven LOD.
//!
//! Coordinate convention: right-handed, +Y up. The view matrix maps world
//! coordinates into eye space (looking down −Z). The projection is a standard
//! perspective transform with `near`/`far` planes; the resulting clip-space
//! coordinates land in `[-1, 1]^3` with +Z forward.

use atomr_worlds_core::lod::{Lod, MetricScale};

use crate::skybox::CubeFace;

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_rad: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// 45° FOV, looking at the origin from (24, 18, 24).
    pub fn isometric_default(aspect: f32) -> Self {
        Self {
            eye: [24.0, 18.0, 24.0],
            target: [8.0, 4.0, 8.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect,
            near: 0.1,
            far: 200.0,
        }
    }

    /// Camera oriented to capture one face of a cubemap.
    ///
    /// Sets `target = eye + face.forward()`, `up = face.up()`, `fov_y_rad =
    /// π/2`, `aspect = 1.0`. The combination of 90° FOV and aspect 1.0 covers
    /// exactly one of the six axis-aligned cube faces with no overlap and no
    /// gap.
    pub fn for_cube_face(eye: [f32; 3], face: CubeFace, near: f32, far: f32) -> Camera {
        let fwd = face.forward();
        let target = [eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2]];
        Camera {
            eye,
            target,
            up: face.up(),
            fov_y_rad: std::f32::consts::FRAC_PI_2,
            aspect: 1.0,
            near,
            far,
        }
    }

    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        look_at(self.eye, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        perspective(self.fov_y_rad, self.aspect, self.near, self.far)
    }

    pub fn view_proj(&self) -> [[f32; 4]; 4] {
        mat4_mul(self.projection_matrix(), self.view_matrix())
    }

    /// Camera focal length in pixels along the y axis, for a viewport `height`
    /// pixels tall.
    pub fn focal_px_y(&self, height: u32) -> f32 {
        0.5 * (height as f32) / (0.5 * self.fov_y_rad).tan()
    }

    /// Pick an [`Lod`] suitable for rendering the world `world` (described by
    /// `scale`) into a viewport `height` pixels tall, assuming the camera
    /// sits roughly `distance_m` meters from its subject.
    pub fn pick_lod(
        &self,
        scale: MetricScale,
        distance_m: f64,
        viewport_height: u32,
        target_px_per_voxel: f64,
    ) -> Lod {
        scale.lod_for_screen(
            distance_m,
            self.focal_px_y(viewport_height) as f64,
            target_px_per_voxel,
        )
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
    [v[0] / l, v[1] / l, v[2] / l]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    // RH look_at (eye looks down −z in view space).
    let f = norm3(sub3(target, eye));
    let s = norm3(cross3(f, up));
    let u = cross3(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    // RH, **reversed-z**: near → 1.0, far → 0.0. Reversed-z spreads f32
    // precision evenly across the depth range under perspective division
    // (since `1/z` is roughly linear in depth-buffer space when the buffer is
    // flipped), eliminating z-fighting at long range — exactly what the
    // Phase 13f skybox needs to keep celestial bodies stable against
    // near-field terrain.
    //
    // Derivation: the standard RH [0, 1] projection writes
    //     clip.z = far*nf*z_view + far*near*nf  (with nf = 1/(near-far))
    //     clip.w = -z_view
    // depth = clip.z/clip.w maps `z_view = -near → 0`, `-far → 1`. Compose with
    // `z' = 1 - z`, which is equivalent to `clip.z' = clip.w - clip.z`:
    //     clip.z' = -z_view - (far*nf*z_view + far*near*nf)
    //             = -z_view*(1 + far*nf) - far*near*nf
    //             = -z_view*(near*nf)    - far*near*nf
    // i.e. [2][2] = -near*nf and [3][2] = -far*near*nf — the two rows the
    // standard form had `far*nf` and `far*near*nf` in.
    let f = 1.0 / (0.5 * fov_y).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, -near * nf, -1.0],
        [0.0, 0.0, -far * near * nf, 0.0],
    ]
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k][j] * b[i][k];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Apply a 4×4 matrix to a 3-vector in homogeneous coordinates.
pub fn transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
        m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_lod_uses_metric_scale() {
        let cam = Camera::isometric_default(1.0);
        let scale = MetricScale { root_size_m: 1024.0, max_depth: 10 };
        let near = cam.pick_lod(scale, 10.0, 256, 1.0);
        let far = cam.pick_lod(scale, 10_000.0, 256, 1.0);
        assert!(far.depth <= near.depth);
    }

    #[test]
    fn projection_maps_origin_in_front_to_positive_w() {
        let cam = Camera {
            eye: [0.0, 0.0, 5.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let p = transform_point(cam.view_proj(), [0.0, 0.0, 0.0]);
        assert!(p[3] > 0.0, "point in front of camera should have positive w");
    }
}
