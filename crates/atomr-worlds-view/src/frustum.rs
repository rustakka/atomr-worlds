//! Six-plane view-frustum derived from a [`Camera`]'s view·proj.
//!
//! We use the **Gribb-Hartmann** plane-extraction trick: the rows of the
//! `view_proj` matrix are linear combinations of the six clip-space plane
//! equations, so each frustum plane in *world space* is a sum/difference of
//! the matrix rows. Concretely, for a 4×4 `m` stored column-major and a
//! homogeneous world-space point `(x, y, z, 1)`:
//!
//! - `left  =  m_row3 + m_row0`
//! - `right =  m_row3 - m_row0`
//! - `bottom=  m_row3 + m_row1`
//! - `top   =  m_row3 - m_row1`
//! - `near  =  m_row3 + m_row2`
//! - `far   =  m_row3 - m_row2`
//!
//! Each plane is then normalised by its xyz magnitude so the signed distance
//! `plane.dot(point)` is in world meters. A point is **inside** when every
//! plane returns `>= 0`.
//!
//! **Reversed-z safety.** The Gribb-Hartmann derivation makes no assumption
//! about the depth direction — it only requires that the clip-space invariants
//! `−w ≤ x,y ≤ w` and `0 ≤ z ≤ w` (or `−w ≤ z ≤ w` for OpenGL-style depth)
//! hold. With our reversed-z convention `z_view = -near ⇒ clip.z = w`,
//! `z_view = -far ⇒ clip.z = 0`, the `near` plane becomes
//! `m_row3 + m_row2` (point inside ⇔ `m_row3·p + m_row2·p ≥ 0` ⇔ `clip.w +
//! clip.z ≥ 0` ⇔ `clip.z ≥ -clip.w` — always true) and the `far` plane
//! becomes `m_row3 - m_row2` (inside ⇔ `clip.w - clip.z ≥ 0` ⇔ `clip.z ≤
//! clip.w` — i.e. the point projects inside `[0, w]`). So we get back the
//! standard "drop everything outside the unit clip box" interpretation
//! without flipping signs.
//!
//! For **orthographic** projection, `clip.w = 1` and the planes degenerate
//! into world-space planes that are translation-only (no perspective-divide
//! warp). The same row-sum extraction still produces correct planes —
//! orthographic is a special case of the same algebra.

use crate::camera::Camera;
use crate::view_cache::CacheAabb;

/// World-space plane `n · p + d = 0`. `n` is unit-length; `d` is the signed
/// distance from origin to the plane along `-n`. A point `p` is on the
/// **positive** side (the frustum interior) when `n · p + d ≥ 0`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Plane {
    pub normal: [f32; 3],
    pub d: f32,
}

impl Plane {
    /// Signed distance from `p` to the plane. Positive on the normal side.
    #[inline]
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.d
    }
}

/// Six-plane view-frustum. Planes are ordered `[left, right, bottom, top,
/// near, far]` and all normals point **inward** — a point is inside the
/// frustum iff `plane.signed_distance(p) >= 0` for every plane.
#[derive(Copy, Clone, Debug)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Build a frustum from a [`Camera`]. Works for both
    /// [`Projection::Perspective`](crate::camera::Projection::Perspective)
    /// and [`Projection::Orthographic`](crate::camera::Projection::Orthographic)
    /// — the row-sum extraction is projection-agnostic.
    pub fn from_camera(cam: &Camera) -> Self {
        Self::from_view_proj(cam.view_proj())
    }

    /// Build a frustum directly from a 4×4 view·projection matrix (stored as
    /// column-major rows of length 4, matching the rest of `camera.rs`).
    pub fn from_view_proj(m: [[f32; 4]; 4]) -> Self {
        // `m[col][row]` because the camera matrices are column-major.
        // Row `i` of the conceptual matrix is `[m[0][i], m[1][i], m[2][i], m[3][i]]`.
        let row = |i: usize| [m[0][i], m[1][i], m[2][i], m[3][i]];
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let mk = |a: [f32; 4], b: [f32; 4], sign: f32| -> Plane {
            let p = [a[0] + sign * b[0], a[1] + sign * b[1], a[2] + sign * b[2], a[3] + sign * b[3]];
            normalize(p)
        };

        let left = mk(r3, r0, 1.0);
        let right = mk(r3, r0, -1.0);
        let bottom = mk(r3, r1, 1.0);
        let top = mk(r3, r1, -1.0);
        let near = mk(r3, r2, 1.0);
        let far = mk(r3, r2, -1.0);
        Self { planes: [left, right, bottom, top, near, far] }
    }

    /// Conservative box-vs-frustum overlap test. Returns `true` when the AABB
    /// touches or lies inside the frustum (false positives possible at the
    /// silhouette where a corner box would clip outside all planes
    /// individually — the so-called "AABB straddles the corner" case — but no
    /// false negatives, so it's sound for visibility culling).
    pub fn intersects_aabb(&self, aabb: CacheAabb) -> bool {
        for p in &self.planes {
            // Pick the AABB corner farthest along the plane normal. If even
            // that corner is on the negative side, the whole box is outside.
            let px = if p.normal[0] >= 0.0 { aabb.max[0] } else { aabb.min[0] } as f32;
            let py = if p.normal[1] >= 0.0 { aabb.max[1] } else { aabb.min[1] } as f32;
            let pz = if p.normal[2] >= 0.0 { aabb.max[2] } else { aabb.min[2] } as f32;
            if p.signed_distance([px, py, pz]) < 0.0 {
                return false;
            }
        }
        true
    }

    /// True iff every corner of `aabb` is strictly inside every plane.
    pub fn contains_aabb(&self, aabb: CacheAabb) -> bool {
        let corners = [
            [aabb.min[0], aabb.min[1], aabb.min[2]],
            [aabb.max[0], aabb.min[1], aabb.min[2]],
            [aabb.min[0], aabb.max[1], aabb.min[2]],
            [aabb.max[0], aabb.max[1], aabb.min[2]],
            [aabb.min[0], aabb.min[1], aabb.max[2]],
            [aabb.max[0], aabb.min[1], aabb.max[2]],
            [aabb.min[0], aabb.max[1], aabb.max[2]],
            [aabb.max[0], aabb.max[1], aabb.max[2]],
        ];
        for p in &self.planes {
            for c in &corners {
                if p.signed_distance([c[0] as f32, c[1] as f32, c[2] as f32]) < 0.0 {
                    return false;
                }
            }
        }
        true
    }
}

fn normalize(p: [f32; 4]) -> Plane {
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-20);
    let inv = 1.0 / len;
    Plane { normal: [p[0] * inv, p[1] * inv, p[2] * inv], d: p[3] * inv }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Projection;

    fn perspective_cam() -> Camera {
        // Eye on +Z looking at origin (camera looks down -Z). FOV=90°, aspect=1,
        // near=0.1, far=100. Frustum extends in -Z from eye.
        Camera {
            eye: [0.0, 0.0, 10.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_2,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            projection: Projection::Perspective { fov_y_rad: std::f32::consts::FRAC_PI_2 },
        }
    }

    fn orthographic_cam() -> Camera {
        Camera {
            eye: [0.0, 0.0, 10.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { half_height_m: 5.0 },
        }
    }

    #[test]
    fn perspective_inside_origin() {
        let cam = perspective_cam();
        let f = Frustum::from_camera(&cam);
        // Origin is in front of the eye at z=0, well inside the frustum.
        let aabb = CacheAabb::new([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        assert!(f.intersects_aabb(aabb));
        assert!(f.contains_aabb(aabb));
    }

    #[test]
    fn perspective_outside_behind() {
        let cam = perspective_cam();
        let f = Frustum::from_camera(&cam);
        // Far behind the camera (z > eye.z): outside the near plane.
        let aabb = CacheAabb::new([-0.5, -0.5, 50.0], [0.5, 0.5, 51.0]);
        assert!(!f.intersects_aabb(aabb));
    }

    #[test]
    fn perspective_outside_lateral() {
        let cam = perspective_cam();
        let f = Frustum::from_camera(&cam);
        // Way off to the side at the same depth.
        let aabb = CacheAabb::new([100.0, -1.0, 0.0], [101.0, 1.0, 1.0]);
        assert!(!f.intersects_aabb(aabb));
    }

    #[test]
    fn perspective_straddles_near() {
        let cam = perspective_cam();
        let f = Frustum::from_camera(&cam);
        // Box straddles the near plane (eye at z=10, near=0.1 ⇒ near plane at z≈9.9).
        let aabb = CacheAabb::new([-0.5, -0.5, 9.5], [0.5, 0.5, 11.0]);
        assert!(f.intersects_aabb(aabb));
        // Not strictly contained: it pokes behind the eye.
        assert!(!f.contains_aabb(aabb));
    }

    #[test]
    fn orthographic_inside_origin() {
        let cam = orthographic_cam();
        let f = Frustum::from_camera(&cam);
        let aabb = CacheAabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(f.intersects_aabb(aabb));
        assert!(f.contains_aabb(aabb));
    }

    #[test]
    fn orthographic_outside_lateral() {
        let cam = orthographic_cam();
        let f = Frustum::from_camera(&cam);
        // half_height_m=5, aspect=1 ⇒ half-width=5. A box at x=20 is well
        // outside the right plane.
        let aabb = CacheAabb::new([20.0, -1.0, -1.0], [21.0, 1.0, 1.0]);
        assert!(!f.intersects_aabb(aabb));
    }

    #[test]
    fn orthographic_straddles_right() {
        let cam = orthographic_cam();
        let f = Frustum::from_camera(&cam);
        // half_width=5: a box from x=4 → x=6 straddles the right plane.
        let aabb = CacheAabb::new([4.0, -1.0, -1.0], [6.0, 1.0, 1.0]);
        assert!(f.intersects_aabb(aabb));
        assert!(!f.contains_aabb(aabb));
    }
}
