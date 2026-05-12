//! Camera with view+projection matrices and MetricScale-driven LOD.
//!
//! Coordinate convention: right-handed, +Y up. The view matrix maps world
//! coordinates into eye space (looking down −Z). All projections produce
//! clip-space coordinates with **reversed-z** depth (near → 1, far → 0); see
//! [`Projection`] for the per-mode derivations.

use atomr_worlds_core::lod::{Lod, MetricScale};

use crate::skybox::CubeFace;

/// Per-mode projection parameters. `projection_matrix` on [`Camera`] dispatches
/// on this enum to build the clip-space transform.
///
/// All variants produce **reversed-z** depth: a view-space point at
/// `z_view = -near` maps to clip-space depth 1, and `z_view = -far` maps to 0.
/// This convention is shared with the existing perspective path so the
/// `new > old` depth test in `Framebuffer` keeps working unchanged.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Projection {
    /// Standard reversed-z perspective with vertical field-of-view `fov_y_rad`.
    /// Matrix derivation: see [`perspective`].
    Perspective { fov_y_rad: f32 },
    /// Reversed-z orthographic. `half_height_m` is the half-height of the
    /// view-frustum in **view-space meters** at any depth (ortho has no
    /// foreshortening); half-width is `half_height_m * aspect`. Derivation:
    /// see [`orthographic`].
    Orthographic { half_height_m: f32 },
    /// Reversed-z oblique-orthographic (RTS / axonometric look). The base is
    /// the same ortho frustum sized by `scale_m_per_px` × viewport pixels,
    /// post-multiplied by a horizontal shear that displaces `x_view` and
    /// `y_view` as a function of `z_view` (rotation around the up axis given
    /// by `rotation_deg`). Depth stays reversed-z: the shear is linear in
    /// `z_view`, so monotonicity is preserved. Derivation: see [`oblique`].
    Oblique { rotation_deg: f32, scale_m_per_px: f32 },
}

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    /// Vertical field-of-view in radians. Retained as a public field so
    /// callers that pre-date [`Projection`] (and the skybox sampler in
    /// `render.rs`, which always operates in perspective) keep working. For
    /// [`Projection::Perspective`] this field is **authoritative for downstream
    /// consumers** but `projection_matrix` itself reads the value embedded in
    /// `projection` so the two stay in sync when set via the standard
    /// constructors.
    pub fov_y_rad: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    /// Projection mode (perspective / orthographic / oblique). Built-in
    /// constructors (`isometric_default`, `for_cube_face`) populate this as
    /// [`Projection::Perspective`] with the same `fov_y_rad` as the field
    /// above, so existing call sites produce byte-identical output.
    pub projection: Projection,
}

impl Camera {
    /// 45° FOV, looking at the origin from (24, 18, 24).
    pub fn isometric_default(aspect: f32) -> Self {
        let fov_y_rad = std::f32::consts::FRAC_PI_4;
        Self {
            eye: [24.0, 18.0, 24.0],
            target: [8.0, 4.0, 8.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad,
            aspect,
            near: 0.1,
            far: 200.0,
            projection: Projection::Perspective { fov_y_rad },
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
        let fov_y_rad = std::f32::consts::FRAC_PI_2;
        Camera {
            eye,
            target,
            up: face.up(),
            fov_y_rad,
            aspect: 1.0,
            near,
            far,
            projection: Projection::Perspective { fov_y_rad },
        }
    }

    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        look_at(self.eye, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        match self.projection {
            Projection::Perspective { fov_y_rad } => perspective(fov_y_rad, self.aspect, self.near, self.far),
            Projection::Orthographic { half_height_m } => {
                orthographic(half_height_m, self.aspect, self.near, self.far)
            }
            Projection::Oblique { rotation_deg, scale_m_per_px } => {
                oblique(rotation_deg, scale_m_per_px, self.aspect, self.near, self.far)
            }
        }
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
        scale.lod_for_screen(distance_m, self.focal_px_y(viewport_height) as f64, target_px_per_voxel)
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

fn orthographic(half_height_m: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    // RH, **reversed-z** orthographic: maps `z_view = -near → 1.0`,
    // `z_view = -far → 0.0`. Same convention as the perspective path so the
    // `new > old` depth test in `Framebuffer` keeps its meaning under either
    // mode.
    //
    // Derivation: orthographic projection has no perspective divide
    // (`clip.w = 1`), so the depth mapping is a plain affine of `z_view` into
    // `clip.z`:
    //     clip.z = a * z_view + b
    // Solve for the boundary conditions
    //     z_view = -near  →  clip.z = 1
    //     z_view = -far   →  clip.z = 0
    //     ⇒  -a * near + b = 1
    //        -a * far  + b = 0
    //     Subtracting: a * (far - near) = 1  ⇒  a = 1/(far - near) = -1/(near - far)
    //     Back-substituting: b = a * far = far/(far - near) = -far/(near - far)
    //
    // Equivalently, with `nf = 1/(near - far)` (matching the perspective fn's
    // local naming):
    //     [2][2] = -nf      = 1/(far - near)
    //     [3][2] = -far*nf  = far/(far - near)
    //
    // Sanity check at the boundary:
    //     z_view = -near:  clip.z =  1/(far-near) * (-near) + far/(far-near)
    //                            = (-near + far)/(far - near) = 1   ✓
    //     z_view = -far:   clip.z =  1/(far-near) * (-far)  + far/(far-near)
    //                            = (-far  + far)/(far - near) = 0   ✓
    //
    // X / Y mapping is the standard ortho box: `clip.x = x_view / half_w`,
    // `clip.y = y_view / half_h`, with `half_w = half_height_m * aspect`.
    // Column-major (same layout as `perspective` above).
    let half_h = half_height_m.max(1e-20);
    let half_w = half_h * aspect.max(1e-20);
    let nf = 1.0 / (near - far);
    [
        [1.0 / half_w, 0.0, 0.0, 0.0],
        [0.0, 1.0 / half_h, 0.0, 0.0],
        [0.0, 0.0, -nf, 0.0],
        [0.0, 0.0, -far * nf, 1.0],
    ]
}

fn oblique(rotation_deg: f32, scale_m_per_px: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    // Oblique-orthographic ("axonometric" / RTS-style) projection.
    //
    // We start from the reversed-z ortho matrix and post-compose a shear that
    // displaces `x_view` (and, via the up-axis rotation, `y_view`) as a linear
    // function of `z_view`. Concretely we want, for a rotation `θ` around the
    // world-up axis, a non-clipping displacement
    //     x_view' = x_view + s * cos(θ) * z_view
    //     y_view' = y_view + s * sin(θ) * z_view
    //     z_view' = z_view
    // where `s = tan(α)` for some fixed shear angle `α`. Embedding `s` into a
    // 4×4 matrix gives the post-multiplier
    //     S = [ 1  0  s*cos θ  0 ]
    //         [ 0  1  s*sin θ  0 ]
    //         [ 0  0    1      0 ]
    //         [ 0  0    0      1 ]
    // Then `M = ortho * S` is the full projection. Because `S` is linear in
    // `z_view` with unit `z_view'` slope, **depth monotonicity is preserved**:
    // increasing `|z_view|` continues to map to smaller `clip.z` under the
    // outer reversed-z ortho, so the `new > old` depth test still selects the
    // nearer fragment. (Anything fancier — e.g. depth-axis tilt — would have
    // to revisit that test.)
    //
    // Sizing: we don't have a viewport here, so `half_height_m` is computed
    // assuming a nominal 1-px-tall viewport scaled by `scale_m_per_px`. The
    // caller-facing convention is "meters per output pixel"; with viewport
    // pixel height `H`, set `half_height_m = scale_m_per_px * H * 0.5`
    // upstream. We use `scale_m_per_px` here as the half-height directly so
    // the matrix is well-defined and unit-consistent for the test path; mode
    // code in `modes/rts.rs` (Phase 14d) will refine.
    let half_h = scale_m_per_px.max(1e-20);
    let half_w = half_h * aspect.max(1e-20);
    let nf = 1.0 / (near - far);
    let theta = rotation_deg.to_radians();
    // Use a fixed shear strength of `tan(30°)` ≈ 0.577 so a non-zero
    // `rotation_deg` produces a visibly non-vertical projection in the tests.
    // Mode code (Phase 14d) is free to expose this as a separate knob.
    let s = (std::f32::consts::FRAC_PI_6).tan();
    let sx = s * theta.cos();
    let sy = s * theta.sin();
    // M = ortho * S. ortho is column-major as built by `orthographic`. Hand-
    // expand the product so we only touch the columns that change (col 2,
    // the z_view-dependent column).
    //
    //     clip.x = x_view/half_w        + (sx / half_w) * z_view
    //     clip.y = y_view/half_h        + (sy / half_h) * z_view
    //     clip.z =          -nf*z_view  - far*nf
    //     clip.w = 1
    [
        [1.0 / half_w, 0.0, 0.0, 0.0],
        [0.0, 1.0 / half_h, 0.0, 0.0],
        [sx / half_w, sy / half_h, -nf, 0.0],
        [0.0, 0.0, -far * nf, 1.0],
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
            projection: Projection::Perspective { fov_y_rad: std::f32::consts::FRAC_PI_4 },
        };
        let p = transform_point(cam.view_proj(), [0.0, 0.0, 0.0]);
        assert!(p[3] > 0.0, "point in front of camera should have positive w");
    }

    /// Wave 1a regression: `Camera::isometric_default(1.0)` must produce the
    /// same projection matrix as the pre-`Projection` perspective-only code.
    /// Floats are bit-pinned to catch any unintended algebraic drift.
    #[test]
    fn perspective_matrix_unchanged() {
        let cam = Camera::isometric_default(1.0);
        let m = cam.projection_matrix();
        // Pre-refactor values from `perspective(π/4, 1.0, 0.1, 200.0)`:
        //   f  = 1/tan(π/8)            ≈ 2.4142137
        //   nf = 1/(near - far)        = 1/(0.1 - 200.0)
        //   m[0][0] = f / aspect = f   ≈ 2.4142137
        //   m[1][1] = f                ≈ 2.4142137
        //   m[2][2] = -near * nf       ≈  5.0025013e-4
        //   m[2][3] = -1.0
        //   m[3][2] = -far * near * nf ≈  0.10005003
        //   m[3][3] = 0.0
        // We rebuild the same constants here rather than copy float literals
        // so the assertion is robust to ULP drift in the host tan() — if a
        // future refactor changes the algebra, *that* triggers the failure.
        let fov = std::f32::consts::FRAC_PI_4;
        let f = 1.0 / (0.5 * fov).tan();
        let nf = 1.0 / (0.1 - 200.0_f32);
        let expected: [[f32; 4]; 4] = [
            [f, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, -0.1 * nf, -1.0],
            [0.0, 0.0, -200.0 * 0.1 * nf, 0.0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    m[i][j].to_bits(),
                    expected[i][j].to_bits(),
                    "m[{i}][{j}]: got {} expected {}",
                    m[i][j],
                    expected[i][j]
                );
            }
        }
    }

    fn ortho_test_camera(half_height_m: f32, near: f32, far: f32) -> Camera {
        Camera {
            eye: [0.0, 0.0, 0.0],
            target: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect: 1.0,
            near,
            far,
            projection: Projection::Orthographic { half_height_m },
        }
    }

    #[test]
    fn orthographic_maps_z_view_to_reversed_depth() {
        // `eye = (0,0,0)`, `target = (0,0,-1)` ⇒ world-Z and view-Z agree
        // (camera looks down −Z, view matrix is identity for points at the
        // origin's xy). A world point at z = -near becomes view-space
        // z_view = -near, so reversed-z depth should be 1.0; z = -far ⇒ 0.0.
        let cam = ortho_test_camera(10.0, 0.1, 100.0);
        let mvp = cam.view_proj();

        let near_clip = transform_point(mvp, [0.0, 0.0, -cam.near]);
        let near_depth = near_clip[2] / near_clip[3];
        assert!(
            (near_depth - 1.0).abs() < 1e-5,
            "near depth (reversed-z ortho) should be ~1.0, got {near_depth}"
        );

        let far_clip = transform_point(mvp, [0.0, 0.0, -cam.far]);
        let far_depth = far_clip[2] / far_clip[3];
        assert!(far_depth.abs() < 1e-5, "far depth (reversed-z ortho) should be ~0.0, got {far_depth}");
    }

    #[test]
    fn orthographic_no_perspective_divide() {
        // `half_height_m = 10`, `aspect = 1` ⇒ `half_width = 10`. A point at
        // view-space (5, 0, -50) should map to clip.x = 5/10 = 0.5 and
        // clip.w = 1 (ortho has no perspective divide).
        let cam = ortho_test_camera(10.0, 0.1, 100.0);
        let mvp = cam.view_proj();
        let clip = transform_point(mvp, [5.0, 0.0, -50.0]);
        assert!((clip[3] - 1.0).abs() < 1e-6, "ortho clip.w should be 1, got {}", clip[3]);
        assert!(
            (clip[0] - 0.5).abs() < 1e-6,
            "ortho clip.x at x=5 with half_width=10 should be 0.5, got {}",
            clip[0]
        );
    }

    #[test]
    fn oblique_shear_displaces_with_depth() {
        // A vertical line at world-x=0 spanning z ∈ [-near, -far] (in view
        // space, identical to world space here) should project to a
        // **non-vertical** screen line when `rotation_deg ≠ 0`: the shear
        // makes clip.x depend on z_view.
        let cam = Camera {
            eye: [0.0, 0.0, 0.0],
            target: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            projection: Projection::Oblique {
                rotation_deg: 0.0, // shear along +x view axis
                scale_m_per_px: 10.0,
            },
        };
        let mvp = cam.view_proj();
        let p_near = transform_point(mvp, [0.0, 0.0, -cam.near]);
        let p_far = transform_point(mvp, [0.0, 0.0, -cam.far]);
        let ndc_x_near = p_near[0] / p_near[3];
        let ndc_x_far = p_far[0] / p_far[3];
        assert!(
            (ndc_x_near - ndc_x_far).abs() > 1e-3,
            "oblique with rotation_deg=0 should shear x by z_view; got equal \
             ndc.x at near={ndc_x_near} and far={ndc_x_far}"
        );
        // Depth must still be reversed-z (the shear preserves z_view
        // monotonicity, so the depth derivation above continues to hold).
        let near_depth = p_near[2] / p_near[3];
        let far_depth = p_far[2] / p_far[3];
        assert!(
            (near_depth - 1.0).abs() < 1e-5 && far_depth.abs() < 1e-5,
            "oblique depth must remain reversed-z: near={near_depth}, far={far_depth}"
        );
    }
}
