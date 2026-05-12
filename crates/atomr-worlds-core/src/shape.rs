//! World shape — defines the geometric envelope of a [`World`].
//!
//! Phases 0–12 modeled every world as an unbounded Euclidean cube. With a
//! [`WorldShape`] embedded in [`World`], a world may instead be a sphere
//! (planetoid → gas giant) or a cylinder (ringworld / Discworld). Most
//! worlds in practice will be spheres; `Cube` remains the backwards-compat
//! default so existing tests and downstream callers keep their behavior.
//!
//! The shape is consumed by:
//! - [`WorldActor::ensure_brick`] — bricks fully outside the shape return
//!   empty without invoking the generator.
//! - [`StreamingPolicy::ring_for_curved`] — horizon distance bounds the
//!   max streaming radius so spherical worlds don't try to stream past
//!   their visible surface.
//! - The renderer's skybox / LOD ring planner.
//!
//! The shape's coordinate frame is centered: `contains(p)` treats `p` as
//! the offset from the world's geometric center, NOT a voxel index. The
//! host is responsible for converting brick coordinates to centered
//! meters before consulting the shape.
//!
//! Determinism: every method is a pure function of the shape's f64 fields.
//! `Hash`, `PartialEq`, and `Eq` route through `f64::to_bits()` so the
//! type is usable as a HashMap key (macro-state cache) and bit-stable
//! across platforms.

use core::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::coord::DVec3;

/// Continuous (f64-meter) axis-aligned bounding box centered on the world
/// origin. Distinct from `atomr_worlds_proto::aabb::AABB`, which is an
/// integer-voxel-coord box.
#[derive(Copy, Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct ShapeAabb {
    pub min: DVec3,
    pub max: DVec3,
}

impl ShapeAabb {
    #[inline]
    pub const fn new(min: DVec3, max: DVec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn centered(half_extent: DVec3) -> Self {
        Self {
            min: DVec3::new(-half_extent.x, -half_extent.y, -half_extent.z),
            max: half_extent,
        }
    }
}

/// Geometric envelope of a [`World`].
///
/// Stored as a small enum so the same call sites handle every variant
/// uniformly. The default is `Cube { edge_m: 1.0e7 }` to preserve
/// pre-Phase-13 behavior.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum WorldShape {
    /// Backwards-compat: unbounded-cube semantics. `contains` returns true
    /// for any point inside the cube of edge `edge_m` centered on origin.
    Cube { edge_m: f64 },
    /// Solid sphere of given radius centered on the world origin. The
    /// primary new variant — sphere worlds drive horizon math, out-of-
    /// shape brick filtering, and skybox generation.
    Sphere { radius_m: f64 },
    /// Cylinder oriented along +Y. Useful for ringworld / Discworld-style
    /// worlds; horizon math degrades to a small-angle approximation.
    Cylinder { radius_m: f64, height_m: f64 },
}

impl WorldShape {
    /// Earth-class default — preserves existing `World` behavior at the
    /// 1e7 m root cube edge.
    #[inline]
    pub const fn default_world() -> Self {
        Self::Cube { edge_m: 1.0e7 }
    }

    /// Radius (or half-extent) of the shape in meters. For a cube this is
    /// half the edge; for sphere/cylinder it is the literal radius.
    #[inline]
    pub fn radius_m(self) -> f64 {
        match self {
            Self::Cube { edge_m } => edge_m * 0.5,
            Self::Sphere { radius_m } => radius_m,
            Self::Cylinder { radius_m, .. } => radius_m,
        }
    }

    /// Continuous bounding box centered on the world origin.
    #[inline]
    pub fn bounding_aabb(self) -> ShapeAabb {
        match self {
            Self::Cube { edge_m } => {
                let h = edge_m * 0.5;
                ShapeAabb::centered(DVec3::new(h, h, h))
            }
            Self::Sphere { radius_m } => {
                ShapeAabb::centered(DVec3::new(radius_m, radius_m, radius_m))
            }
            Self::Cylinder { radius_m, height_m } => {
                ShapeAabb::centered(DVec3::new(radius_m, height_m * 0.5, radius_m))
            }
        }
    }

    /// True if the centered point `p` lies inside or on the shape boundary.
    #[inline]
    pub fn contains(self, p: DVec3) -> bool {
        match self {
            Self::Cube { edge_m } => {
                let h = edge_m * 0.5;
                p.x.abs() <= h && p.y.abs() <= h && p.z.abs() <= h
            }
            Self::Sphere { radius_m } => {
                let r2 = radius_m * radius_m;
                p.x * p.x + p.y * p.y + p.z * p.z <= r2
            }
            Self::Cylinder { radius_m, height_m } => {
                let h = height_m * 0.5;
                let r2 = radius_m * radius_m;
                (p.x * p.x + p.z * p.z) <= r2 && p.y.abs() <= h
            }
        }
    }

    /// Outward-pointing unit surface normal at (or near) `p`. For `p` at
    /// the origin a deterministic fallback `(0, 1, 0)` is returned.
    #[inline]
    pub fn surface_normal_at(self, p: DVec3) -> DVec3 {
        match self {
            Self::Sphere { .. } => {
                let len2 = p.x * p.x + p.y * p.y + p.z * p.z;
                if len2 > 0.0 {
                    let len = len2.sqrt();
                    DVec3::new(p.x / len, p.y / len, p.z / len)
                } else {
                    DVec3::new(0.0, 1.0, 0.0)
                }
            }
            Self::Cube { .. } => {
                let ax = p.x.abs();
                let ay = p.y.abs();
                let az = p.z.abs();
                if ax >= ay && ax >= az {
                    DVec3::new(p.x.signum(), 0.0, 0.0)
                } else if ay >= az {
                    DVec3::new(0.0, p.y.signum(), 0.0)
                } else {
                    DVec3::new(0.0, 0.0, p.z.signum())
                }
            }
            Self::Cylinder { .. } => {
                let r2 = p.x * p.x + p.z * p.z;
                if r2 > 0.0 {
                    let r = r2.sqrt();
                    DVec3::new(p.x / r, 0.0, p.z / r)
                } else {
                    DVec3::new(1.0, 0.0, 0.0)
                }
            }
        }
    }

    /// Distance to the geometric horizon from an observer at the given
    /// altitude above the shape's surface (along the surface normal).
    ///
    /// Spherical formula: `sqrt(2*R*h + h²)`. For a Cube the horizon is
    /// effectively unbounded (`f64::INFINITY`). For a Cylinder we use the
    /// spherical formula with the cylinder's radius as a small-angle
    /// approximation — adequate at altitudes much less than the radius.
    #[inline]
    pub fn horizon_distance_m(self, altitude_m: f64) -> f64 {
        match self {
            Self::Cube { .. } => f64::INFINITY,
            Self::Sphere { radius_m } | Self::Cylinder { radius_m, .. } => {
                let h = altitude_m.max(0.0);
                (2.0 * radius_m * h + h * h).sqrt()
            }
        }
    }

    /// Coordinate wrapping. Identity for sphere and cube (the inscribed-
    /// sphere design uses render-side local-origin rebasing rather than
    /// in-storage modulo arithmetic — see plan 13a). Cylinder wraps the
    /// angular component but leaves height unbounded; documented as
    /// non-idempotent at f64 precision (use sparingly).
    #[inline]
    pub fn wrap(self, p: DVec3) -> DVec3 {
        match self {
            Self::Sphere { .. } | Self::Cube { .. } => p,
            Self::Cylinder { .. } => {
                let r2 = p.x * p.x + p.z * p.z;
                if r2 == 0.0 {
                    return p;
                }
                let r = r2.sqrt();
                let theta = p.z.atan2(p.x);
                let wrapped = theta.rem_euclid(core::f64::consts::TAU);
                DVec3::new(r * wrapped.cos(), p.y, r * wrapped.sin())
            }
        }
    }

    /// Surface area in m². Sphere: `4πR²`. Cube: `6 * edge²`. Cylinder
    /// (closed): `2πR² + 2πR*h`.
    #[inline]
    pub fn surface_area_m2(self) -> f64 {
        match self {
            Self::Cube { edge_m } => 6.0 * edge_m * edge_m,
            Self::Sphere { radius_m } => 4.0 * core::f64::consts::PI * radius_m * radius_m,
            Self::Cylinder { radius_m, height_m } => {
                let pi = core::f64::consts::PI;
                2.0 * pi * radius_m * radius_m + 2.0 * pi * radius_m * height_m
            }
        }
    }
}

impl Default for WorldShape {
    #[inline]
    fn default() -> Self {
        Self::default_world()
    }
}

// Manual PartialEq/Eq/Hash via bit patterns. f64 can't derive Eq/Hash but
// IEEE 754 byte equality is platform-stable and good enough for cache
// keying on shape parameters. We never store NaN here — NaN parameters
// would be a configuration bug, not a normal value.
impl PartialEq for WorldShape {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cube { edge_m: a }, Self::Cube { edge_m: b }) => a.to_bits() == b.to_bits(),
            (Self::Sphere { radius_m: a }, Self::Sphere { radius_m: b }) => {
                a.to_bits() == b.to_bits()
            }
            (
                Self::Cylinder { radius_m: ar, height_m: ah },
                Self::Cylinder { radius_m: br, height_m: bh },
            ) => ar.to_bits() == br.to_bits() && ah.to_bits() == bh.to_bits(),
            _ => false,
        }
    }
}

impl Eq for WorldShape {}

impl Hash for WorldShape {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Cube { edge_m } => {
                0u8.hash(state);
                edge_m.to_bits().hash(state);
            }
            Self::Sphere { radius_m } => {
                1u8.hash(state);
                radius_m.to_bits().hash(state);
            }
            Self::Cylinder { radius_m, height_m } => {
                2u8.hash(state);
                radius_m.to_bits().hash(state);
                height_m.to_bits().hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn h(s: WorldShape) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn default_is_cube_1e7() {
        let s = WorldShape::default();
        assert!(matches!(s, WorldShape::Cube { edge_m } if edge_m == 1.0e7));
    }

    #[test]
    fn sphere_contains_origin_and_rejects_outside() {
        let s = WorldShape::Sphere { radius_m: 1.0 };
        assert!(s.contains(DVec3::ZERO));
        assert!(s.contains(DVec3::new(1.0, 0.0, 0.0))); // on boundary
        assert!(!s.contains(DVec3::new(1.1, 0.0, 0.0)));
        assert!(!s.contains(DVec3::new(0.6, 0.6, 0.6))); // |p| = sqrt(1.08) > 1
    }

    #[test]
    fn cube_contains_corner_but_not_outside() {
        let s = WorldShape::Cube { edge_m: 2.0 };
        assert!(s.contains(DVec3::new(1.0, 1.0, 1.0))); // corner
        assert!(s.contains(DVec3::ZERO));
        assert!(!s.contains(DVec3::new(1.1, 0.0, 0.0)));
    }

    #[test]
    fn cylinder_contains_along_axis_within_height() {
        let s = WorldShape::Cylinder { radius_m: 1.0, height_m: 4.0 };
        assert!(s.contains(DVec3::new(0.0, 2.0, 0.0))); // top center
        assert!(!s.contains(DVec3::new(0.0, 2.1, 0.0))); // above
        assert!(!s.contains(DVec3::new(1.1, 0.0, 0.0))); // outside radius
    }

    #[test]
    fn horizon_at_sea_level_is_zero() {
        let s = WorldShape::Sphere { radius_m: 6.371e6 };
        assert_eq!(s.horizon_distance_m(0.0), 0.0);
    }

    #[test]
    fn horizon_earth_class_at_1km() {
        // Earth radius 6371 km, observer at 1 km → horizon ≈ 112_884.897 m
        // (sqrt(2 * 6_371_000 * 1000 + 1000²) = sqrt(12_743_000_000)).
        let s = WorldShape::Sphere { radius_m: 6.371e6 };
        let d = s.horizon_distance_m(1000.0);
        assert!((d - 112_884.897).abs() < 1.0, "got horizon = {d}");
    }

    #[test]
    fn horizon_monotonic_in_altitude() {
        let s = WorldShape::Sphere { radius_m: 6.371e6 };
        let d0 = s.horizon_distance_m(0.0);
        let d1 = s.horizon_distance_m(100.0);
        let d2 = s.horizon_distance_m(10_000.0);
        assert!(d0 <= d1 && d1 <= d2);
    }

    #[test]
    fn cube_has_infinite_horizon() {
        let s = WorldShape::Cube { edge_m: 1.0e7 };
        assert_eq!(s.horizon_distance_m(100.0), f64::INFINITY);
    }

    #[test]
    fn surface_normal_sphere_is_outward_unit() {
        let s = WorldShape::Sphere { radius_m: 5.0 };
        let n = s.surface_normal_at(DVec3::new(3.0, 4.0, 0.0));
        let len = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
        assert!((len - 1.0).abs() < 1e-12);
        // pointing outward = same direction as input
        assert!(n.x > 0.0 && n.y > 0.0 && (n.z).abs() < 1e-12);
    }

    #[test]
    fn surface_normal_origin_is_deterministic_fallback() {
        let s = WorldShape::Sphere { radius_m: 1.0 };
        let n = s.surface_normal_at(DVec3::ZERO);
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 1.0);
        assert_eq!(n.z, 0.0);
    }

    #[test]
    fn bounding_aabb_sphere_is_radius_box() {
        let s = WorldShape::Sphere { radius_m: 7.0 };
        let b = s.bounding_aabb();
        assert_eq!(b.min, DVec3::new(-7.0, -7.0, -7.0));
        assert_eq!(b.max, DVec3::new(7.0, 7.0, 7.0));
    }

    #[test]
    fn bounding_aabb_cube_is_half_edge() {
        let s = WorldShape::Cube { edge_m: 10.0 };
        let b = s.bounding_aabb();
        assert_eq!(b.min, DVec3::new(-5.0, -5.0, -5.0));
        assert_eq!(b.max, DVec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn radius_m_matches_geometry() {
        assert_eq!(WorldShape::Cube { edge_m: 10.0 }.radius_m(), 5.0);
        assert_eq!(WorldShape::Sphere { radius_m: 7.0 }.radius_m(), 7.0);
        assert_eq!(
            WorldShape::Cylinder { radius_m: 3.0, height_m: 8.0 }.radius_m(),
            3.0
        );
    }

    #[test]
    fn surface_area_sphere_matches_formula() {
        // 4πR² at R=1 → 4π
        let a = WorldShape::Sphere { radius_m: 1.0 }.surface_area_m2();
        assert!((a - 4.0 * core::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn hash_eq_distinguishes_variants_and_parameters() {
        // Same variant, same params → same hash and equal.
        let a = WorldShape::Sphere { radius_m: 6.371e6 };
        let b = WorldShape::Sphere { radius_m: 6.371e6 };
        assert_eq!(a, b);
        assert_eq!(h(a), h(b));
        // Different radius → not equal, different hash (with overwhelming
        // probability — exact equality of two distinct f64 hashes is
        // statistically zero, so we treat inequality as the assertion).
        let c = WorldShape::Sphere { radius_m: 6.371e6 + 1.0 };
        assert_ne!(a, c);
        assert_ne!(h(a), h(c));
        // Different variant → not equal.
        let d = WorldShape::Cube { edge_m: 6.371e6 };
        assert_ne!(a, d);
        assert_ne!(h(a), h(d));
    }

    #[test]
    fn hash_is_bit_stable_for_fixed_parameters() {
        // The hash of a hardcoded sphere should be the same across runs.
        // We don't pin a specific u64 here (since DefaultHasher's seed is
        // not part of the stability contract — the bit-stability of
        // f64::to_bits is). What we *do* pin is that two constructions of
        // the same shape produce equal hashes in this process.
        let h1 = h(WorldShape::Sphere { radius_m: 6.371e6 });
        let h2 = h(WorldShape::Sphere { radius_m: 6.371e6 });
        assert_eq!(h1, h2);
    }
}
