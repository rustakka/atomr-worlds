//! `.vox` / schematic-style sparse-voxel authored region.
//!
//! Phase 13e ships a format-agnostic in-memory loader: callers parse the
//! external file (MagicaVoxel `.vox`, Minecraft `.schematic`, etc.) and
//! hand a `Vec<(IVec3, u16)>` to [`VoxFileRegion::new`]. A wrapper that
//! parses the file format itself can sit on top — gating the heavy
//! dependencies (`dot_vox`, NBT crates) behind cargo features so the
//! default workspace stays lightweight.
//!
//! Determinism contract: the same `(voxels, transform)` produces
//! byte-identical brick output across runs.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use super::{region_id, AuthoredRegion, RegionAabb, RegionId};

/// Affine voxel-space transform applied to imported voxels. Currently
/// supports translation only; rotation/mirror is left for future work
/// (deterministic but requires care with non-axis-aligned bounds).
#[derive(Copy, Clone, Debug)]
pub struct VoxelTransform {
    pub translation: IVec3,
}

impl Default for VoxelTransform {
    fn default() -> Self {
        Self { translation: IVec3::ZERO }
    }
}

impl VoxelTransform {
    pub fn translation(t: IVec3) -> Self {
        Self { translation: t }
    }

    #[inline]
    pub fn apply(&self, p: IVec3) -> IVec3 {
        IVec3::new(p.x + self.translation.x, p.y + self.translation.y, p.z + self.translation.z)
    }
}

/// A region of sparse authored voxels — typically loaded from a `.vox`
/// or schematic file. Stored as a sorted Vec (by world voxel coord) so
/// iteration is deterministic across runs.
#[derive(Debug, Clone)]
pub struct VoxFileRegion {
    id: RegionId,
    name: String,
    /// Sorted by (z, y, x) for deterministic iteration.
    voxels: Vec<(IVec3, u16)>,
    bounds: RegionAabb,
}

impl VoxFileRegion {
    /// Build a region from in-memory voxel data + an optional transform.
    /// Voxels are translated by `transform.translation` before being
    /// stored in world voxel coordinates.
    pub fn new(
        name: impl Into<String>,
        voxels: impl IntoIterator<Item = (IVec3, u16)>,
        transform: VoxelTransform,
    ) -> Self {
        let mut transformed: Vec<(IVec3, u16)> = voxels
            .into_iter()
            .map(|(p, m)| (transform.apply(p), m))
            .collect();
        transformed.sort_by_key(|(p, _)| (p.z, p.y, p.x));

        let name = name.into();
        let id = region_id(&name);
        let bounds = if transformed.is_empty() {
            RegionAabb::new(IVec3::ZERO, IVec3::ZERO)
        } else {
            let mut min = transformed[0].0;
            let mut max = transformed[0].0;
            for (p, _) in &transformed {
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                min.z = min.z.min(p.z);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
                max.z = max.z.max(p.z);
            }
            // Exclusive max.
            RegionAabb::new(min, IVec3::new(max.x + 1, max.y + 1, max.z + 1))
        };

        Self { id, name, voxels: transformed, bounds }
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn voxel_count(&self) -> usize { self.voxels.len() }
}

impl AuthoredRegion for VoxFileRegion {
    fn id(&self) -> RegionId { self.id }
    fn bounds(&self) -> RegionAabb { self.bounds }

    fn apply_to_brick(&self, brick_coord: IVec3, brick: &mut Brick) -> usize {
        let edge = BRICK_EDGE as i64;
        let bo = IVec3::new(brick_coord.x * edge, brick_coord.y * edge, brick_coord.z * edge);
        let bmax = IVec3::new(bo.x + edge, bo.y + edge, bo.z + edge);
        let mut count = 0;
        // Iterate in stored order (already sorted) for determinism.
        for (p, m) in &self.voxels {
            if p.x < bo.x || p.x >= bmax.x { continue; }
            if p.y < bo.y || p.y >= bmax.y { continue; }
            if p.z < bo.z || p.z >= bmax.z { continue; }
            let local = IVec3::new(p.x - bo.x, p.y - bo.y, p.z - bo.z);
            brick.set(local, Voxel::new(*m));
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_list_writes_inside_brick() {
        let voxels = vec![
            (IVec3::new(0, 0, 0), 1),
            (IVec3::new(5, 6, 7), 2),
            (IVec3::new(15, 15, 15), 3),
        ];
        let r = VoxFileRegion::new("test", voxels, VoxelTransform::default());
        let mut b = Brick::new();
        let written = r.apply_to_brick(IVec3::new(0, 0, 0), &mut b);
        assert_eq!(written, 3);
        assert_eq!(b.get(IVec3::new(0, 0, 0)), Voxel::new(1));
        assert_eq!(b.get(IVec3::new(5, 6, 7)), Voxel::new(2));
        assert_eq!(b.get(IVec3::new(15, 15, 15)), Voxel::new(3));
    }

    #[test]
    fn transform_offsets_into_target_brick() {
        // Voxel at (0,0,0) translated by (+16, 0, 0) lands in brick (1,0,0).
        let voxels = vec![(IVec3::new(0, 0, 0), 42)];
        let t = VoxelTransform::translation(IVec3::new(16, 0, 0));
        let r = VoxFileRegion::new("test", voxels, t);
        let mut b1 = Brick::new();
        let w1 = r.apply_to_brick(IVec3::new(1, 0, 0), &mut b1);
        assert_eq!(w1, 1);
        assert_eq!(b1.get(IVec3::new(0, 0, 0)), Voxel::new(42));
        let mut b0 = Brick::new();
        let w0 = r.apply_to_brick(IVec3::new(0, 0, 0), &mut b0);
        assert_eq!(w0, 0);
    }

    #[test]
    fn iteration_is_deterministic() {
        // Insert voxels in shuffled order; storage sorts → iteration order stable.
        let voxels = vec![
            (IVec3::new(2, 0, 0), 1),
            (IVec3::new(0, 0, 0), 2),
            (IVec3::new(1, 0, 0), 3),
        ];
        let r = VoxFileRegion::new("d", voxels, VoxelTransform::default());
        let mut b = Brick::new();
        r.apply_to_brick(IVec3::new(0, 0, 0), &mut b);
        // All three landed; values reflect last-write-wins per cell.
        assert_eq!(b.get(IVec3::new(0, 0, 0)), Voxel::new(2));
        assert_eq!(b.get(IVec3::new(1, 0, 0)), Voxel::new(3));
        assert_eq!(b.get(IVec3::new(2, 0, 0)), Voxel::new(1));
    }

    #[test]
    fn empty_voxel_list_bounded() {
        let r = VoxFileRegion::new("empty", Vec::<(IVec3, u16)>::new(), VoxelTransform::default());
        assert_eq!(r.voxel_count(), 0);
    }
}
