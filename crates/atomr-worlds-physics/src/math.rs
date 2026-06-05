//! Minimal `f64` linear algebra for the inertia solver.
//!
//! The pure-data core ([`atomr_worlds_core::DVec3`]) only carries `add`/`sub`/
//! `length`, so this module adds the `dot`/`cross`/`scale` helpers and a small
//! symmetric-friendly [`Mat3`] (3×3 matrix) with an inverse — just enough to
//! build and invert an inertia tensor without pulling in `glam`/`nalgebra`.

use atomr_worlds_core::DVec3;
use serde::{Deserialize, Serialize};

#[inline]
pub fn dot(a: DVec3, b: DVec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

#[inline]
pub fn cross(a: DVec3, b: DVec3) -> DVec3 {
    DVec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

#[inline]
pub fn scale(a: DVec3, s: f64) -> DVec3 {
    DVec3::new(a.x * s, a.y * s, a.z * s)
}

/// A 3×3 matrix, row-major. Used for inertia tensors (which are symmetric and
/// positive-definite for any non-degenerate body).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mat3 {
    /// `m[row][col]`.
    pub m: [[f64; 3]; 3],
}

impl Mat3 {
    pub const ZERO: Self = Self { m: [[0.0; 3]; 3] };
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    #[inline]
    pub fn from_diagonal(d: DVec3) -> Self {
        Self {
            m: [[d.x, 0.0, 0.0], [0.0, d.y, 0.0], [0.0, 0.0, d.z]],
        }
    }

    /// Outer product `a ⊗ b` (a column times b row).
    #[inline]
    pub fn outer(a: DVec3, b: DVec3) -> Self {
        Self {
            m: [
                [a.x * b.x, a.x * b.y, a.x * b.z],
                [a.y * b.x, a.y * b.y, a.y * b.z],
                [a.z * b.x, a.z * b.y, a.z * b.z],
            ],
        }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        let mut out = Self::ZERO;
        for r in 0..3 {
            for c in 0..3 {
                out.m[r][c] = self.m[r][c] * s;
            }
        }
        out
    }

    #[inline]
    pub fn mul_vec(self, v: DVec3) -> DVec3 {
        DVec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }

    #[inline]
    pub fn determinant(self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Matrix inverse via the adjugate / determinant. Returns `None` when the
    /// matrix is singular (|det| below a small epsilon) — the caller is
    /// expected to regularize the tensor (clamp principal moments) first.
    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-18 {
            return None;
        }
        let inv_det = 1.0 / det;
        let m = &self.m;
        let cof = |r0: usize, r1: usize, c0: usize, c1: usize| {
            m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
        };
        // Transposed cofactor matrix (adjugate), each scaled by 1/det.
        Some(Self {
            m: [
                [
                    cof(1, 2, 1, 2) * inv_det,
                    -cof(0, 2, 1, 2) * inv_det,
                    cof(0, 1, 1, 2) * inv_det,
                ],
                [
                    -cof(1, 2, 0, 2) * inv_det,
                    cof(0, 2, 0, 2) * inv_det,
                    -cof(0, 1, 0, 2) * inv_det,
                ],
                [
                    cof(1, 2, 0, 1) * inv_det,
                    -cof(0, 2, 0, 1) * inv_det,
                    cof(0, 1, 0, 1) * inv_det,
                ],
            ],
        })
    }
}

impl core::ops::Add for Mat3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        let mut out = Self::ZERO;
        for r in 0..3 {
            for c in 0..3 {
                out.m[r][c] = self.m[r][c] + o.m[r][c];
            }
        }
        out
    }
}

impl core::ops::Sub for Mat3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        let mut out = Self::ZERO;
        for r in 0..3 {
            for c in 0..3 {
                out.m[r][c] = self.m[r][c] - o.m[r][c];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn dot_cross_basics() {
        let x = DVec3::new(1.0, 0.0, 0.0);
        let y = DVec3::new(0.0, 1.0, 0.0);
        assert!(approx(dot(x, y), 0.0));
        assert!(approx(dot(x, x), 1.0));
        let z = cross(x, y);
        assert!(approx(z.x, 0.0) && approx(z.y, 0.0) && approx(z.z, 1.0));
    }

    #[test]
    fn identity_inverse_is_identity() {
        let inv = Mat3::IDENTITY.inverse().unwrap();
        assert_eq!(inv, Mat3::IDENTITY);
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        let a = Mat3 {
            m: [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]],
        };
        let inv = a.inverse().unwrap();
        // a * inv ≈ I
        for c in 0..3 {
            let col = DVec3::new(
                if c == 0 { 1.0 } else { 0.0 },
                if c == 1 { 1.0 } else { 0.0 },
                if c == 2 { 1.0 } else { 0.0 },
            );
            let r = a.mul_vec(inv.mul_vec(col));
            assert!(approx(r.x, col.x) && approx(r.y, col.y) && approx(r.z, col.z));
        }
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        // Rank-1 (outer product) is singular.
        let s = Mat3::outer(DVec3::new(1.0, 2.0, 3.0), DVec3::new(1.0, 1.0, 1.0));
        assert!(s.inverse().is_none());
    }
}
