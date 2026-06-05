//! Mass, center of mass, and inertia tensor from a voxel set.
//!
//! A debris body is a bag of solid voxels, each with a material density. Its
//! rigid-body mass properties are an `O(N_voxels)` two-pass computation:
//!
//! 1. accumulate total mass and the mass-weighted centroid (center of mass);
//! 2. accumulate the inertia tensor about that centroid via the standard
//!    point-mass form `I += mᵢ (|rᵢ|² · 1 − rᵢ ⊗ rᵢ)`, where `rᵢ = pᵢ − com`.
//!
//! The tensor is regularized before inversion (a small multiple of identity is
//! added when it is near-singular) so thin / single-voxel-thick bodies don't
//! produce an infinite angular response.

use atomr_worlds_core::DVec3;
use serde::{Deserialize, Serialize};

use crate::math::{dot, scale, Mat3};

/// Rigid-body mass properties in a body-local frame whose origin is the center
/// of mass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MassProperties {
    /// Total mass in kilograms.
    pub mass_kg: f64,
    /// Center of mass, in the same local coordinates the input positions used.
    pub com: DVec3,
    /// Inertia tensor about the center of mass (kg·m²).
    pub inertia: Mat3,
    /// Inverse inertia tensor (regularized), ready for the solver.
    pub inertia_inv: Mat3,
}

impl MassProperties {
    /// The zero body: no mass, no inertia. Inverse inertia is zero so an empty
    /// body has no angular response.
    pub const ZERO: Self = Self {
        mass_kg: 0.0,
        com: DVec3::ZERO,
        inertia: Mat3::ZERO,
        inertia_inv: Mat3::ZERO,
    };
}

/// Compute [`MassProperties`] from an iterator of `(center_position_m, mass_kg)`
/// point samples — one per solid voxel, where `center_position_m` is the voxel
/// center in local meters and `mass_kg = density · voxel_volume`.
///
/// `min_principal` clamps the smallest principal moment before inversion (a
/// small positive value, e.g. `mass · voxel_size²` scale) so degenerate (flat /
/// 1-D) bodies stay invertible. Pass `0.0` to disable clamping.
pub fn mass_properties(
    voxels: impl IntoIterator<Item = (DVec3, f64)> + Clone,
    min_principal: f64,
) -> MassProperties {
    // Pass 1: mass + centroid.
    let mut mass = 0.0f64;
    let mut com_acc = DVec3::ZERO;
    for (p, m) in voxels.clone() {
        if m <= 0.0 {
            continue;
        }
        mass += m;
        com_acc = com_acc + scale(p, m);
    }
    if mass <= 0.0 {
        return MassProperties::ZERO;
    }
    let com = scale(com_acc, 1.0 / mass);

    // Pass 2: inertia tensor about the centroid.
    let mut inertia = Mat3::ZERO;
    for (p, m) in voxels {
        if m <= 0.0 {
            continue;
        }
        let r = p - com;
        let rr = dot(r, r);
        // mᵢ (|r|² I − r ⊗ r)
        let term = (Mat3::IDENTITY.scale(rr) - Mat3::outer(r, r)).scale(m);
        inertia = inertia + term;
    }

    let inertia_inv = regularized_inverse(inertia, min_principal);
    MassProperties { mass_kg: mass, com, inertia, inertia_inv }
}

/// Invert an inertia tensor, nudging it toward positive-definite first so thin
/// bodies (one principal moment ≈ 0) don't blow up.
fn regularized_inverse(mut inertia: Mat3, min_principal: f64) -> Mat3 {
    if min_principal > 0.0 {
        // Cheap regularization: ensure each diagonal entry is at least
        // `min_principal`. For a centroid-frame inertia tensor the diagonal
        // dominates, so this keeps a flat body invertible without distorting a
        // healthy one (whose diagonal already exceeds the floor).
        for d in 0..3 {
            if inertia.m[d][d] < min_principal {
                inertia.m[d][d] = min_principal;
            }
        }
    }
    inertia.inverse().unwrap_or(Mat3::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn single_voxel_mass_and_com() {
        let pos = DVec3::new(2.0, 5.0, -1.0);
        let mp = mass_properties([(pos, 3.0)], 0.0);
        assert!(approx(mp.mass_kg, 3.0, 1e-12));
        assert!(approx(mp.com.x, 2.0, 1e-12));
        assert!(approx(mp.com.y, 5.0, 1e-12));
        assert!(approx(mp.com.z, -1.0, 1e-12));
    }

    #[test]
    fn empty_body_is_zero() {
        let mp = mass_properties(Vec::<(DVec3, f64)>::new(), 0.0);
        assert_eq!(mp, MassProperties::ZERO);
    }

    #[test]
    fn mass_is_conserved_and_com_centered_for_symmetric_cube() {
        // 2×2×2 unit-spaced voxels centered on the origin, uniform mass.
        let mut vox = Vec::new();
        for x in [-0.5, 0.5] {
            for y in [-0.5, 0.5] {
                for z in [-0.5, 0.5] {
                    vox.push((DVec3::new(x, y, z), 1.0));
                }
            }
        }
        let mp = mass_properties(vox, 0.0);
        assert!(approx(mp.mass_kg, 8.0, 1e-12));
        assert!(approx(mp.com.x, 0.0, 1e-12));
        assert!(approx(mp.com.y, 0.0, 1e-12));
        assert!(approx(mp.com.z, 0.0, 1e-12));
        // Symmetry ⇒ diagonal tensor with equal principal moments, ~zero
        // off-diagonals.
        let i = mp.inertia;
        assert!(approx(i.m[0][0], i.m[1][1], 1e-9));
        assert!(approx(i.m[1][1], i.m[2][2], 1e-9));
        assert!(approx(i.m[0][1], 0.0, 1e-9));
        assert!(approx(i.m[0][2], 0.0, 1e-9));
        assert!(approx(i.m[1][2], 0.0, 1e-9));
        // 8 point masses at radius² = 0.75 each: Ixx = Σ m (y²+z²) = 8·0.5 = 4.
        assert!(approx(i.m[0][0], 4.0, 1e-9));
    }

    #[test]
    fn thin_body_inverse_is_finite_with_clamp() {
        // A 1-D line of voxels along x has ~zero inertia about the x axis.
        let vox: Vec<_> = (0..5).map(|i| (DVec3::new(i as f64, 0.0, 0.0), 1.0)).collect();
        let mp = mass_properties(vox, 1e-3);
        // Without the clamp the x-axis principal moment would be 0 and the
        // inverse undefined; with it, the inverse must be finite.
        for r in 0..3 {
            for c in 0..3 {
                assert!(mp.inertia_inv.m[r][c].is_finite());
            }
        }
    }

    #[test]
    fn deterministic() {
        let vox: Vec<_> = (0..10)
            .map(|i| (DVec3::new(i as f64, (i % 3) as f64, 0.0), 1.5))
            .collect();
        let a = mass_properties(vox.clone(), 1e-6);
        let b = mass_properties(vox, 1e-6);
        assert_eq!(a, b);
    }
}
