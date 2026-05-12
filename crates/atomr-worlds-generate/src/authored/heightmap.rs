//! Heightmap-backed authored region.
//!
//! Phase 13e: imports a 2D height field (typically derived from a DEM / GIS
//! source) into the world as voxel columns. The loader is intentionally
//! format-agnostic — it takes a raw `Vec<u16>` height array. A PNG / GeoTIFF
//! wrapper can be added on top in a follow-up crate (or behind an optional
//! `image-region` feature) by:
//!
//! ```ignore
//! let img = image::open(path)?.into_luma16();
//! let heights: Vec<u16> = img.into_raw();
//! HeightmapRegion::new("dem", origin, width, height, heights, base_mat)
//! ```
//!
//! Determinism contract: same `(heights, origin, width, height, fill,
//! base_material)` → identical brick output, byte-stable across runs.

use std::collections::HashMap;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use super::{region_id, AuthoredRegion, RegionAabb, RegionId};

/// 2D heightmap projected as vertical voxel columns into a [`Brick`]
/// grid.
///
/// `heights[z * width + x]` gives the column height at `(origin.x + x,
/// origin.z + z)`. Each column is filled with `base_material` from
/// `origin.y` up to (but not including) `origin.y + heights[idx]`.
#[derive(Debug, Clone)]
pub struct HeightmapRegion {
    id: RegionId,
    name: String,
    /// `(origin.x, origin.y, origin.z)` is the column-origin in voxel
    /// coordinates. `y` is the column's base; columns extend upward.
    origin: IVec3,
    width: i64,
    depth: i64,
    /// `heights[z * width + x]` — column height in voxels. 0 leaves the
    /// column entirely empty.
    heights: Vec<u16>,
    base_material: u16,
    bounds: RegionAabb,
}

impl HeightmapRegion {
    pub fn new(
        name: impl Into<String>,
        origin: IVec3,
        width: u32,
        depth: u32,
        heights: Vec<u16>,
        base_material: u16,
    ) -> Self {
        assert_eq!(
            heights.len() as u64,
            (width as u64) * (depth as u64),
            "heights length must equal width * depth",
        );
        let name = name.into();
        let id = region_id(&name);
        let max_h = heights.iter().copied().max().unwrap_or(0) as i64;
        let bounds = RegionAabb::new(
            origin,
            IVec3::new(origin.x + width as i64, origin.y + max_h, origin.z + depth as i64),
        );
        Self {
            id,
            name,
            origin,
            width: width as i64,
            depth: depth as i64,
            heights,
            base_material,
            bounds,
        }
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn width(&self) -> i64 { self.width }
    pub fn depth(&self) -> i64 { self.depth }

    /// Column height at offset `(x, z)` from the region's origin.
    /// Returns 0 for out-of-range queries.
    #[inline]
    fn height_at(&self, x: i64, z: i64) -> u16 {
        if x < 0 || z < 0 || x >= self.width || z >= self.depth {
            return 0;
        }
        let idx = (z * self.width + x) as usize;
        self.heights[idx]
    }
}

impl AuthoredRegion for HeightmapRegion {
    fn id(&self) -> RegionId { self.id }
    fn bounds(&self) -> RegionAabb { self.bounds }

    fn apply_to_brick(&self, brick_coord: IVec3, brick: &mut Brick) -> usize {
        let edge = BRICK_EDGE as i64;
        let bo = IVec3::new(brick_coord.x * edge, brick_coord.y * edge, brick_coord.z * edge);
        // For each (x, z) column inside the brick, find the height field
        // value and fill vertically as appropriate.
        let mut count = 0;
        for lx in 0..edge {
            for lz in 0..edge {
                let wx = bo.x + lx;
                let wz = bo.z + lz;
                let rx = wx - self.origin.x;
                let rz = wz - self.origin.z;
                if rx < 0 || rz < 0 || rx >= self.width || rz >= self.depth {
                    continue;
                }
                let h = self.height_at(rx, rz) as i64;
                if h == 0 {
                    continue;
                }
                for ly in 0..edge {
                    let wy = bo.y + ly;
                    let ry = wy - self.origin.y;
                    if ry < 0 || ry >= h {
                        continue;
                    }
                    brick.set(IVec3::new(lx, ly, lz), Voxel::new(self.base_material));
                    count += 1;
                }
            }
        }
        count
    }
}

/// Convenience: build a [`HeightmapRegion`] from a list of explicit
/// columns. Useful for tests and small authored stages.
pub fn heightmap_from_columns(
    name: impl Into<String>,
    origin: IVec3,
    width: u32,
    depth: u32,
    columns: HashMap<(i64, i64), u16>,
    base_material: u16,
) -> HeightmapRegion {
    let mut heights = vec![0u16; (width as usize) * (depth as usize)];
    for ((x, z), h) in columns {
        if x < 0 || z < 0 || x >= width as i64 || z >= depth as i64 {
            continue;
        }
        let idx = (z as usize) * (width as usize) + (x as usize);
        heights[idx] = h;
    }
    HeightmapRegion::new(name, origin, width, depth, heights, base_material)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_heightmap_fills_columns() {
        // 8×8 flat-3 heightmap → 192 voxels in brick (0,0,0).
        let r = HeightmapRegion::new(
            "flat",
            IVec3::new(0, 0, 0),
            8,
            8,
            vec![3u16; 64],
            42,
        );
        let mut b = Brick::new();
        let written = r.apply_to_brick(IVec3::new(0, 0, 0), &mut b);
        assert_eq!(written, 8 * 8 * 3);
        assert_eq!(b.get(IVec3::new(0, 0, 0)), Voxel::new(42));
        assert_eq!(b.get(IVec3::new(7, 2, 7)), Voxel::new(42));
        assert_eq!(b.get(IVec3::new(7, 3, 7)), Voxel::EMPTY);
    }

    #[test]
    fn varying_heightmap_respects_columns() {
        // Column at (3, 4) has height 5; everything else is 0.
        let mut heights = vec![0u16; 64];
        heights[4 * 8 + 3] = 5;
        let r = HeightmapRegion::new("spike", IVec3::new(0, 0, 0), 8, 8, heights, 7);
        let mut b = Brick::new();
        let written = r.apply_to_brick(IVec3::new(0, 0, 0), &mut b);
        assert_eq!(written, 5);
        assert_eq!(b.get(IVec3::new(3, 0, 4)), Voxel::new(7));
        assert_eq!(b.get(IVec3::new(3, 4, 4)), Voxel::new(7));
        assert_eq!(b.get(IVec3::new(3, 5, 4)), Voxel::EMPTY);
    }

    #[test]
    fn deterministic_apply() {
        let mut heights = vec![0u16; 64];
        heights[4 * 8 + 3] = 5;
        let r = HeightmapRegion::new("d", IVec3::new(0, 0, 0), 8, 8, heights, 7);
        let mut b1 = Brick::new();
        let mut b2 = Brick::new();
        let _ = r.apply_to_brick(IVec3::new(0, 0, 0), &mut b1);
        let _ = r.apply_to_brick(IVec3::new(0, 0, 0), &mut b2);
        assert_eq!(b1.to_bytes(), b2.to_bytes());
    }
}
