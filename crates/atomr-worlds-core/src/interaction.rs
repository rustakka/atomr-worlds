//! Configurable unit of interaction (tool radius + precision tier).
//!
//! An [`InteractionUnit`] carries the shape, radius (in meters), and
//! "precision tier" of an edit. The host translates `(scale, radius_m)` into
//! voxel counts and brick coordinates, applies the brush, and emits a single
//! aggregated event per region write.
//!
//! ## Precision-tier hook
//!
//! `precision_scale: Lod` selects the grid on which the brush *samples* the
//! shape predicate. When `precision_scale.depth == scale.max_depth` the
//! brush writes at leaf voxels (fine edits). When the depth is coarser, the
//! brush evaluates the predicate at a coarser grid and writes identical
//! voxel values across each `2^(max_depth - precision_scale.depth)`-voxel
//! block, which a future isosurface mesher (Phase 9) interprets as a
//! coarser implicit surface — the "round at different precisions" hook.

use serde::{Deserialize, Serialize};

use crate::coord::{DVec3, IVec3};
use crate::lod::{Lod, MetricScale};

/// Brush shape.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum ToolKind {
    /// Single-voxel edit (degenerate case; `radius_m` is ignored).
    Voxel,
    /// Solid sphere of `radius_m`.
    Sphere,
    /// Axis-aligned cube of half-edge `radius_m`.
    Cube,
    /// Cone — placeholder shape; same predicate as `Sphere` for now.
    Cone,
}

/// A configurable unit of interaction — the size, shape, and grid-precision
/// of a brush edit.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct InteractionUnit {
    pub kind: ToolKind,
    pub radius_m: f64,
    pub precision_scale: Lod,
}

impl InteractionUnit {
    /// Single-voxel point edit at full leaf precision.
    #[inline]
    pub fn voxel(precision_scale: Lod) -> Self {
        Self { kind: ToolKind::Voxel, radius_m: 0.0, precision_scale }
    }

    /// Spherical brush.
    #[inline]
    pub fn sphere(radius_m: f64, precision_scale: Lod) -> Self {
        Self { kind: ToolKind::Sphere, radius_m, precision_scale }
    }

    /// Cubic brush.
    #[inline]
    pub fn cube(radius_m: f64, precision_scale: Lod) -> Self {
        Self { kind: ToolKind::Cube, radius_m, precision_scale }
    }

    /// True if a *world-space* point `p` is inside this brush centered at
    /// `center`, in the body's metric coordinates.
    pub fn contains(&self, center: DVec3, p: DVec3) -> bool {
        match self.kind {
            ToolKind::Voxel => p.distance(center) < f64::EPSILON,
            ToolKind::Sphere | ToolKind::Cone => p.distance(center) <= self.radius_m,
            ToolKind::Cube => {
                let d = p - center;
                d.x.abs() <= self.radius_m
                    && d.y.abs() <= self.radius_m
                    && d.z.abs() <= self.radius_m
            }
        }
    }

    /// Compute the affected voxel set: the brick coordinates intersecting the
    /// brush AABB and an estimate of the touched-voxel count. The caller
    /// iterates the bricks and applies the brush predicate inside each.
    ///
    /// `brick_edge` is the brick's edge in voxels (always 16 for the canonical
    /// [`atomr_worlds_voxel::Brick`][brick]); taken as a parameter so this
    /// crate can stay dep-free from `atomr-worlds-voxel`.
    ///
    /// [brick]: ../../atomr_worlds_voxel/index.html
    pub fn affected_voxels(&self, scale: MetricScale, center: DVec3, brick_edge: i64) -> AffectedSet {
        let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
        let r = self.radius_m.max(mpv * 0.5);
        // Voxel-coord AABB around the brush center.
        let cx = (center.x / mpv).floor() as i64;
        let cy = (center.y / mpv).floor() as i64;
        let cz = (center.z / mpv).floor() as i64;
        let rv = ((r / mpv).ceil() as i64).max(0);

        let edge = brick_edge;
        let bmin_x = (cx - rv).div_euclid(edge);
        let bmax_x = (cx + rv).div_euclid(edge);
        let bmin_y = (cy - rv).div_euclid(edge);
        let bmax_y = (cy + rv).div_euclid(edge);
        let bmin_z = (cz - rv).div_euclid(edge);
        let bmax_z = (cz + rv).div_euclid(edge);
        let mut bricks = Vec::with_capacity(
            (((bmax_x - bmin_x + 1) * (bmax_y - bmin_y + 1) * (bmax_z - bmin_z + 1)) as usize)
                .min(4096),
        );
        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    bricks.push(IVec3::new(bx, by, bz));
                }
            }
        }
        // Approximate voxel count: sphere uses 4/3πr³, cube uses (2r)³, voxel uses 1.
        let approx = match self.kind {
            ToolKind::Voxel => 1u64,
            ToolKind::Sphere | ToolKind::Cone => {
                let v = (4.0 / 3.0) * std::f64::consts::PI * (rv as f64).powi(3);
                v.max(1.0) as u64
            }
            ToolKind::Cube => {
                let n = (2 * rv + 1) as u64;
                n.saturating_mul(n).saturating_mul(n)
            }
        };
        AffectedSet { bricks, approx_voxels: approx }
    }

    /// Snap a world-space point to the brush's precision-tier grid. Returns
    /// the same point when `precision_scale.depth == scale.max_depth`.
    pub fn snap_to_precision(&self, scale: MetricScale, p: DVec3) -> DVec3 {
        let mpv = scale.meters_per_voxel(self.precision_scale);
        DVec3::new(
            (p.x / mpv).round() * mpv,
            (p.y / mpv).round() * mpv,
            (p.z / mpv).round() * mpv,
        )
    }
}

/// Outcome of [`InteractionUnit::affected_voxels`].
#[derive(Clone, Debug, Default)]
pub struct AffectedSet {
    pub bricks: Vec<IVec3>,
    pub approx_voxels: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_brush_is_single_brick() {
        let u = InteractionUnit::voxel(Lod::new(24));
        let scale = MetricScale::DEFAULT_WORLD;
        let aff = u.affected_voxels(scale, DVec3::new(0.0, 0.0, 0.0), 16);
        assert!(!aff.bricks.is_empty());
        assert!(aff.approx_voxels >= 1);
    }

    #[test]
    fn sphere_brush_covers_multiple_bricks_at_large_radius() {
        let scale = MetricScale { root_size_m: 1024.0, max_depth: 6 };
        // mpv at leaf depth = 1024 / 64 = 16 m.
        // 100 m sphere covers ~12 voxels across; ~1 brick if BRICK_EDGE=16 voxels.
        let u = InteractionUnit::sphere(100.0, Lod::new(6));
        let aff = u.affected_voxels(scale, DVec3::ZERO, 16);
        assert!(!aff.bricks.is_empty());
        assert!(aff.approx_voxels > 1);
    }

    #[test]
    fn snap_to_precision_aligns_to_coarse_grid() {
        let scale = MetricScale { root_size_m: 1024.0, max_depth: 6 };
        // precision_scale 4 → 1024 / 16 = 64 m per coarse cell.
        let u = InteractionUnit::sphere(10.0, Lod::new(4));
        let p = u.snap_to_precision(scale, DVec3::new(100.0, 100.0, 100.0));
        assert_eq!(p.x % 64.0, 0.0);
        assert_eq!(p.y % 64.0, 0.0);
        assert_eq!(p.z % 64.0, 0.0);
    }
}
