//! Coordinate types.
//!
//! A single canonical [`IVec3`] (i64 components) underlies every level. i32
//! is insufficient because voxel coordinates at meter-resolution routinely
//! exceed 2^31 at galactic and universe scales.
//!
//! Per-level `#[repr(transparent)]` newtypes prevent mixing coordinates
//! between hierarchy levels at API boundaries with zero runtime cost.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Serialize, Deserialize)]
pub struct IVec3 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl IVec3 {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    #[inline]
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn splat(v: i64) -> Self {
        Self { x: v, y: v, z: v }
    }
}

impl From<(i64, i64, i64)> for IVec3 {
    #[inline]
    fn from((x, y, z): (i64, i64, i64)) -> Self {
        Self { x, y, z }
    }
}

impl From<[i64; 3]> for IVec3 {
    #[inline]
    fn from([x, y, z]: [i64; 3]) -> Self {
        Self { x, y, z }
    }
}

macro_rules! level_coord {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Serialize, Deserialize)]
        #[repr(transparent)]
        pub struct $name(pub IVec3);

        impl $name {
            pub const ZERO: Self = Self(IVec3::ZERO);

            #[inline]
            pub const fn new(x: i64, y: i64, z: i64) -> Self {
                Self(IVec3::new(x, y, z))
            }
        }

        impl From<IVec3> for $name {
            #[inline]
            fn from(v: IVec3) -> Self { Self(v) }
        }

        impl From<$name> for IVec3 {
            #[inline]
            fn from(v: $name) -> Self { v.0 }
        }
    };
}

level_coord!(
    /// Coordinate within the universe-level grid (typically `ZERO` for the root).
    UniverseCoord
);
level_coord!(
    /// Coordinate of a galaxy within its parent universe.
    GalaxyCoord
);
level_coord!(
    /// Coordinate of a sector within its parent galaxy.
    SectorCoord
);
level_coord!(
    /// Coordinate of a star system within its parent sector.
    SystemCoord
);
level_coord!(
    /// Coordinate of a world within its parent system.
    WorldCoord
);
level_coord!(
    /// Coordinate of a brick within a voxel octree.
    BrickCoord
);
level_coord!(
    /// Voxel coordinate inside a world (or other voxel-bearing object).
    VoxelCoord
);

/// Continuous (f64) position in meters within some containing frame.
///
/// Used for vehicle frame positions, observer poses, and brush centers. Kept
/// as plain data so [`atomr_worlds_core`](crate) stays zero-dep — no `glam`
/// or similar.
#[derive(Copy, Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct DVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl DVec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }
}

impl core::ops::Sub for DVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl core::ops::Add for DVec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

/// Unit quaternion representing orientation. Stored as `(x, y, z, w)` with
/// `w` the scalar component. Identity is `(0, 0, 0, 1)`.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// Re-normalize to a unit quaternion. Returns [`Self::IDENTITY`] for a
    /// degenerate (near-zero norm) quaternion rather than producing `NaN`s.
    #[inline]
    pub fn normalize(self) -> Quat {
        let n2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if n2 <= 1e-30 {
            return Quat::IDENTITY;
        }
        let inv = 1.0 / n2.sqrt();
        Quat { x: self.x * inv, y: self.y * inv, z: self.z * inv, w: self.w * inv }
    }

    /// Advance this orientation by a world-frame angular velocity `w` (rad/s)
    /// over `dt` seconds, first-order and re-normalized to stay unit.
    ///
    /// Uses the quaternion derivative `q̇ = ½ ω_q ⊗ q` with `ω_q = (w, 0)` a
    /// pure quaternion, integrated explicitly: `q' = normalize(q + ½ dt ω_q ⊗ q)`.
    /// `self` is taken to map body→world, consistent with [`Self::mul`].
    #[inline]
    pub fn integrate(self, w: DVec3, dt: f64) -> Quat {
        let omega = Quat { x: w.x, y: w.y, z: w.z, w: 0.0 };
        let dq = omega * self;
        let h = 0.5 * dt;
        Quat {
            x: self.x + dq.x * h,
            y: self.y + dq.y * h,
            z: self.z + dq.z * h,
            w: self.w + dq.w * h,
        }
        .normalize()
    }
}

impl core::ops::Mul for Quat {
    type Output = Quat;
    /// Hamilton product `self ⊗ rhs` (composition of rotations, `self` applied
    /// after `rhs` when both map body→world).
    #[inline]
    fn mul(self, r: Quat) -> Quat {
        Quat {
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        }
    }
}

impl Default for Quat {
    #[inline]
    fn default() -> Self { Self::IDENTITY }
}

/// A scalar quantity in SI meters. Newtype over `f64` to keep the unit
/// explicit at API boundaries where distance fields might otherwise be
/// confused with voxel-count or pixel quantities.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default, Serialize, Deserialize)]
pub struct Meters(pub f64);

impl Meters {
    pub const ZERO: Self = Self(0.0);

    #[inline]
    pub const fn new(v: f64) -> Self { Self(v) }

    #[inline]
    pub fn value(self) -> f64 { self.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtypes_are_transparent() {
        assert_eq!(std::mem::size_of::<UniverseCoord>(), std::mem::size_of::<IVec3>());
        assert_eq!(std::mem::size_of::<BrickCoord>(), std::mem::size_of::<IVec3>());
    }

    #[test]
    fn round_trips_through_ivec3() {
        let g = GalaxyCoord::new(1, -2, 3);
        let v: IVec3 = g.into();
        let back: GalaxyCoord = v.into();
        assert_eq!(g, back);
    }

    fn quat_approx_eq(a: Quat, b: Quat, eps: f64) -> bool {
        // Quaternions q and -q represent the same rotation; accept either sign.
        let same = (a.x - b.x).abs() < eps
            && (a.y - b.y).abs() < eps
            && (a.z - b.z).abs() < eps
            && (a.w - b.w).abs() < eps;
        let neg = (a.x + b.x).abs() < eps
            && (a.y + b.y).abs() < eps
            && (a.z + b.z).abs() < eps
            && (a.w + b.w).abs() < eps;
        same || neg
    }

    fn quat_norm(q: Quat) -> f64 {
        (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt()
    }

    #[test]
    fn quat_mul_identity_is_noop() {
        let q = Quat::new(0.1, 0.2, 0.3, 0.9).normalize();
        assert!(quat_approx_eq(q * Quat::IDENTITY, q, 1e-12));
        assert!(quat_approx_eq(Quat::IDENTITY * q, q, 1e-12));
    }

    #[test]
    fn quat_mul_is_associative() {
        let a = Quat::new(0.1, -0.2, 0.3, 0.9).normalize();
        let b = Quat::new(-0.4, 0.1, 0.2, 0.8).normalize();
        let c = Quat::new(0.2, 0.3, -0.1, 0.7).normalize();
        assert!(quat_approx_eq((a * b) * c, a * (b * c), 1e-12));
    }

    #[test]
    fn quat_normalize_makes_unit() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0).normalize();
        assert!((quat_norm(q) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn quat_normalize_zero_is_identity() {
        assert_eq!(Quat::new(0.0, 0.0, 0.0, 0.0).normalize(), Quat::IDENTITY);
    }

    #[test]
    fn quat_integrate_zero_omega_is_noop() {
        let q = Quat::new(0.1, 0.2, 0.3, 0.9).normalize();
        assert!(quat_approx_eq(q.integrate(DVec3::ZERO, 1.0 / 30.0), q, 1e-12));
    }

    #[test]
    fn quat_integrate_stays_unit_over_many_steps() {
        // Spin about Y at 3 rad/s for 10k steps; the per-step re-normalize must
        // keep it on the unit sphere despite first-order drift.
        let mut q = Quat::IDENTITY;
        let w = DVec3::new(0.0, 3.0, 0.0);
        for _ in 0..10_000 {
            q = q.integrate(w, 1.0 / 240.0);
        }
        assert!((quat_norm(q) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn quat_integrate_quarter_turn_about_y() {
        // Integrate ω = π/2 rad/s about Y for 1 s in small steps → a 90°
        // rotation, whose scalar part is cos(π/4) ≈ 0.70710678.
        let mut q = Quat::IDENTITY;
        let w = DVec3::new(0.0, std::f64::consts::FRAC_PI_2, 0.0);
        let dt = 1.0 / 4096.0;
        for _ in 0..4096 {
            q = q.integrate(w, dt);
        }
        let expected = Quat::new(0.0, (std::f64::consts::PI / 4.0).sin(), 0.0, (std::f64::consts::PI / 4.0).cos());
        assert!(quat_approx_eq(q, expected, 1e-4), "got {q:?}");
    }
}
