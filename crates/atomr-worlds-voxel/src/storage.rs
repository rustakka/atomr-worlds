//! Pluggable in-memory brick layouts.
//!
//! Each strategy presents the same `read`/`write`/`to_brick` interface so
//! callers can pick a layout to match a brick's content profile without
//! changing the dense `Brick` default.

use crate::brick::{Brick, BRICK_EDGE, BRICK_LEN};
use crate::voxel::Voxel;

pub trait BrickStorage: Send + Sync + std::fmt::Debug {
    fn id(&self) -> &'static str;
    fn read(&self, x: u8, y: u8, z: u8) -> Voxel;
    fn write(&mut self, x: u8, y: u8, z: u8, v: Voxel);
    fn to_brick(&self) -> Brick;
    fn from_brick(brick: &Brick) -> Self
    where Self: Sized;
}

#[inline]
fn flat_index(x: u8, y: u8, z: u8) -> usize {
    debug_assert!(
        (x as usize) < BRICK_EDGE && (y as usize) < BRICK_EDGE && (z as usize) < BRICK_EDGE
    );
    ((z as usize) * BRICK_EDGE + y as usize) * BRICK_EDGE + x as usize
}

fn brick_from_array(voxels: Box<[Voxel; BRICK_LEN]>) -> Brick {
    let mut count = 0u16;
    for v in voxels.iter() {
        if !v.is_empty() {
            count = count.saturating_add(1);
        }
    }
    Brick { voxels, nonempty_count: count, light_overlay: None }
}

// ---------------------------------------------------------------------------
// DenseBrick: byte-equal Vanilla default.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DenseBrick {
    inner: Brick,
}

impl DenseBrick {
    pub fn new() -> Self {
        Self { inner: Brick::new() }
    }

    pub fn into_inner(self) -> Brick {
        self.inner
    }

    pub fn as_brick(&self) -> &Brick {
        &self.inner
    }
}

impl Default for DenseBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickStorage for DenseBrick {
    fn id(&self) -> &'static str {
        "dense"
    }

    fn read(&self, x: u8, y: u8, z: u8) -> Voxel {
        self.inner.voxels[flat_index(x, y, z)]
    }

    fn write(&mut self, x: u8, y: u8, z: u8, v: Voxel) {
        let i = flat_index(x, y, z);
        let prev = self.inner.voxels[i];
        if prev == v {
            return;
        }
        match (prev.is_empty(), v.is_empty()) {
            (true, false) => {
                self.inner.nonempty_count = self.inner.nonempty_count.saturating_add(1)
            }
            (false, true) => {
                self.inner.nonempty_count = self.inner.nonempty_count.saturating_sub(1)
            }
            _ => {}
        }
        self.inner.voxels[i] = v;
    }

    fn to_brick(&self) -> Brick {
        self.inner.clone()
    }

    fn from_brick(brick: &Brick) -> Self {
        Self { inner: brick.clone() }
    }
}

// ---------------------------------------------------------------------------
// SegmentedRowBrick: per-row uniform/indexed tag.
// ---------------------------------------------------------------------------

pub const ROW_COUNT: usize = BRICK_EDGE * BRICK_EDGE; // 256 rows of 16 voxels.
pub const ROW_LEN: usize = BRICK_EDGE;

#[derive(Debug, Clone, Copy)]
enum RowKind {
    Uniform(Voxel),
    Indexed(u32),
}

#[derive(Debug, Clone)]
pub struct SegmentedRowBrick {
    rows: Box<[RowKind; ROW_COUNT]>,
    pool: Vec<Voxel>,
}

impl SegmentedRowBrick {
    pub fn new() -> Self {
        Self {
            rows: Box::new([RowKind::Uniform(Voxel::EMPTY); ROW_COUNT]),
            pool: Vec::new(),
        }
    }

    #[inline]
    fn row_index(y: u8, z: u8) -> usize {
        (z as usize) * BRICK_EDGE + y as usize
    }

    fn materialize_row(&mut self, row: usize) -> u32 {
        if let RowKind::Indexed(offset) = self.rows[row] {
            return offset;
        }
        let value = match self.rows[row] {
            RowKind::Uniform(v) => v,
            RowKind::Indexed(_) => unreachable!(),
        };
        let offset = self.pool.len() as u32;
        self.pool.extend(std::iter::repeat(value).take(ROW_LEN));
        self.rows[row] = RowKind::Indexed(offset);
        offset
    }

    /// Number of rows currently stored as a uniform tag (for tests).
    pub fn uniform_row_count(&self) -> usize {
        self.rows.iter().filter(|r| matches!(r, RowKind::Uniform(_))).count()
    }
}

impl Default for SegmentedRowBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickStorage for SegmentedRowBrick {
    fn id(&self) -> &'static str {
        "segmented-row"
    }

    fn read(&self, x: u8, y: u8, z: u8) -> Voxel {
        let row = Self::row_index(y, z);
        match self.rows[row] {
            RowKind::Uniform(v) => v,
            RowKind::Indexed(offset) => self.pool[offset as usize + x as usize],
        }
    }

    fn write(&mut self, x: u8, y: u8, z: u8, v: Voxel) {
        let row = Self::row_index(y, z);
        match self.rows[row] {
            RowKind::Uniform(existing) if existing == v => {}
            RowKind::Uniform(_) => {
                let offset = self.materialize_row(row);
                self.pool[offset as usize + x as usize] = v;
            }
            RowKind::Indexed(offset) => {
                self.pool[offset as usize + x as usize] = v;
            }
        }
    }

    fn to_brick(&self) -> Brick {
        let mut arr: Box<[Voxel; BRICK_LEN]> = Box::new([Voxel::EMPTY; BRICK_LEN]);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                let row = z * BRICK_EDGE + y;
                let base = (z * BRICK_EDGE + y) * BRICK_EDGE;
                match self.rows[row] {
                    RowKind::Uniform(v) => {
                        for x in 0..BRICK_EDGE {
                            arr[base + x] = v;
                        }
                    }
                    RowKind::Indexed(offset) => {
                        let off = offset as usize;
                        for x in 0..BRICK_EDGE {
                            arr[base + x] = self.pool[off + x];
                        }
                    }
                }
            }
        }
        brick_from_array(arr)
    }

    fn from_brick(brick: &Brick) -> Self {
        // Fold rows into Uniform tags when possible; this is the "compaction"
        // path used after a `to_brick` round-trip on dense input.
        let mut rows: Box<[RowKind; ROW_COUNT]> =
            Box::new([RowKind::Uniform(Voxel::EMPTY); ROW_COUNT]);
        let mut pool: Vec<Voxel> = Vec::new();
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                let row = z * BRICK_EDGE + y;
                let base = (z * BRICK_EDGE + y) * BRICK_EDGE;
                let first = brick.voxels[base];
                let uniform = (1..BRICK_EDGE).all(|x| brick.voxels[base + x] == first);
                if uniform {
                    rows[row] = RowKind::Uniform(first);
                } else {
                    let offset = pool.len() as u32;
                    pool.extend_from_slice(&brick.voxels[base..base + BRICK_EDGE]);
                    rows[row] = RowKind::Indexed(offset);
                }
            }
        }
        Self { rows, pool }
    }
}

// ---------------------------------------------------------------------------
// SvoBrick: sparse voxel octree internal to a brick.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum SvoNode {
    /// Leaf at the bottom (single voxel).
    Leaf(Voxel),
    /// Internal: 8-bit child mask + dense list of `popcount(mask)` child
    /// indices into the arena. Absent octants imply `Voxel::EMPTY`.
    Internal { child_mask: u8, children: Vec<u32> },
}

#[derive(Debug, Clone)]
pub struct SvoBrick {
    /// Root is always at index 0 and covers a 16³ cube. Recursion halves the
    /// edge at each level until depth 4 reaches a single voxel leaf.
    nodes: Vec<SvoNode>,
}

const SVO_DEPTH: u8 = 4; // 16 → 8 → 4 → 2 → 1.

impl SvoBrick {
    pub fn new() -> Self {
        Self { nodes: vec![SvoNode::Internal { child_mask: 0, children: Vec::new() }] }
    }

    #[inline]
    fn octant_for(x: u8, y: u8, z: u8, depth_from_root: u8) -> u8 {
        let level = SVO_DEPTH - depth_from_root - 1; // bit position
        let bx = (x >> level) & 1;
        let by = (y >> level) & 1;
        let bz = (z >> level) & 1;
        bx | (by << 1) | (bz << 2)
    }

    fn read_recursive(&self, node: u32, x: u8, y: u8, z: u8, depth: u8) -> Voxel {
        match &self.nodes[node as usize] {
            SvoNode::Leaf(v) => *v,
            SvoNode::Internal { child_mask, children } => {
                let octant = Self::octant_for(x, y, z, depth);
                let bit = 1u8 << octant;
                if (*child_mask & bit) == 0 {
                    return Voxel::EMPTY;
                }
                let slot = (*child_mask & (bit - 1)).count_ones() as usize;
                self.read_recursive(children[slot], x, y, z, depth + 1)
            }
        }
    }

    fn write_recursive(&mut self, node: u32, x: u8, y: u8, z: u8, depth: u8, v: Voxel) {
        if depth == SVO_DEPTH {
            self.nodes[node as usize] = SvoNode::Leaf(v);
            return;
        }
        let octant = Self::octant_for(x, y, z, depth);
        let bit = 1u8 << octant;
        let (mask, slot, child_id) = {
            let SvoNode::Internal { child_mask, children } = &self.nodes[node as usize] else {
                unreachable!("write must traverse internal nodes above SVO_DEPTH");
            };
            let mask = *child_mask;
            let slot = (mask & (bit - 1)).count_ones() as usize;
            let child_id = if (mask & bit) != 0 { Some(children[slot]) } else { None };
            (mask, slot, child_id)
        };

        let child_id = if let Some(id) = child_id {
            id
        } else {
            // Skip allocation when writing EMPTY into an absent octant —
            // keeps to_brick output identical to a brick that never saw
            // the write, which matters for the "drops empty subtrees" test.
            if v.is_empty() {
                return;
            }
            let new_id = self.nodes.len() as u32;
            let new_node = if depth + 1 == SVO_DEPTH {
                SvoNode::Leaf(Voxel::EMPTY)
            } else {
                SvoNode::Internal { child_mask: 0, children: Vec::new() }
            };
            self.nodes.push(new_node);
            let SvoNode::Internal { child_mask, children } = &mut self.nodes[node as usize] else {
                unreachable!();
            };
            *child_mask = mask | bit;
            children.insert(slot, new_id);
            new_id
        };

        self.write_recursive(child_id, x, y, z, depth + 1, v);
    }
}

impl Default for SvoBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickStorage for SvoBrick {
    fn id(&self) -> &'static str {
        "svo"
    }

    fn read(&self, x: u8, y: u8, z: u8) -> Voxel {
        self.read_recursive(0, x, y, z, 0)
    }

    fn write(&mut self, x: u8, y: u8, z: u8, v: Voxel) {
        self.write_recursive(0, x, y, z, 0, v);
    }

    fn to_brick(&self) -> Brick {
        let mut arr: Box<[Voxel; BRICK_LEN]> = Box::new([Voxel::EMPTY; BRICK_LEN]);
        for z in 0..BRICK_EDGE as u8 {
            for y in 0..BRICK_EDGE as u8 {
                for x in 0..BRICK_EDGE as u8 {
                    arr[flat_index(x, y, z)] = self.read(x, y, z);
                }
            }
        }
        brick_from_array(arr)
    }

    fn from_brick(brick: &Brick) -> Self {
        let mut out = Self::new();
        for z in 0..BRICK_EDGE as u8 {
            for y in 0..BRICK_EDGE as u8 {
                for x in 0..BRICK_EDGE as u8 {
                    let v = brick.voxels[flat_index(x, y, z)];
                    if !v.is_empty() {
                        out.write(x, y, z, v);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::coord::IVec3;

    fn dense_reference() -> Brick {
        let mut b = Brick::new();
        b.set(IVec3::new(0, 0, 0), Voxel::new(1));
        b.set(IVec3::new(15, 15, 15), Voxel::new(2));
        b.set(IVec3::new(3, 5, 7), Voxel::new(42));
        b.set(IVec3::new(8, 8, 8), Voxel::new(99));
        b
    }

    fn assert_storage_matches_brick<S: BrickStorage>(s: &S, b: &Brick) {
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let expected = b.get(IVec3::new(x, y, z));
                    let actual = s.read(x as u8, y as u8, z as u8);
                    assert_eq!(actual, expected, "at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn dense_read_write_round_trip() {
        let reference = dense_reference();
        let storage = DenseBrick::from_brick(&reference);
        assert_storage_matches_brick(&storage, &reference);
        let back = storage.to_brick();
        assert_eq!(back.voxels.as_ref(), reference.voxels.as_ref());
        assert_eq!(back.nonempty_count, reference.nonempty_count);
    }

    #[test]
    fn segmented_row_compacts_uniform_rows() {
        // Sparse: only one voxel non-empty, so 255/256 rows should be Uniform.
        let mut b = Brick::new();
        b.set(IVec3::new(2, 3, 4), Voxel::new(11));
        let s = SegmentedRowBrick::from_brick(&b);
        assert_eq!(s.uniform_row_count(), ROW_COUNT - 1);
        assert_storage_matches_brick(&s, &b);
        let back = s.to_brick();
        assert_eq!(back.voxels.as_ref(), b.voxels.as_ref());
    }

    #[test]
    fn segmented_row_write_then_uniform_again() {
        let mut s = SegmentedRowBrick::new();
        s.write(0, 0, 0, Voxel::new(1));
        s.write(1, 0, 0, Voxel::new(1));
        // Row (y=0, z=0) is now Indexed; reading row (y=1, z=0) still uniform.
        assert_eq!(s.uniform_row_count(), ROW_COUNT - 1);
        assert_eq!(s.read(0, 0, 0), Voxel::new(1));
        assert_eq!(s.read(2, 0, 0), Voxel::EMPTY);
    }

    #[test]
    fn svo_read_write_round_trip() {
        let reference = dense_reference();
        let storage = SvoBrick::from_brick(&reference);
        assert_storage_matches_brick(&storage, &reference);
        let back = storage.to_brick();
        assert_eq!(back.voxels.as_ref(), reference.voxels.as_ref());
        assert_eq!(back.nonempty_count, reference.nonempty_count);
    }

    #[test]
    fn svo_drops_empty_subtrees() {
        // An empty brick must have *only* the root node (zero children).
        let s = SvoBrick::new();
        assert_eq!(s.nodes.len(), 1);
        let b = Brick::new();
        let s2 = SvoBrick::from_brick(&b);
        assert_eq!(s2.nodes.len(), 1, "empty brick should not allocate child nodes");

        // Writing then clearing must not leave a node tree larger than a
        // freshly-built single-write tree: writes that touch a value then
        // clear it materialize nodes, but writing empties never allocates.
        let mut s3 = SvoBrick::new();
        s3.write(7, 7, 7, Voxel::EMPTY);
        assert_eq!(s3.nodes.len(), 1, "empty writes must not allocate");
    }

    #[test]
    fn svo_write_then_read_at_extremes() {
        let mut s = SvoBrick::new();
        s.write(0, 0, 0, Voxel::new(1));
        s.write(15, 15, 15, Voxel::new(2));
        s.write(7, 8, 9, Voxel::new(3));
        assert_eq!(s.read(0, 0, 0), Voxel::new(1));
        assert_eq!(s.read(15, 15, 15), Voxel::new(2));
        assert_eq!(s.read(7, 8, 9), Voxel::new(3));
        assert_eq!(s.read(4, 4, 4), Voxel::EMPTY);
    }
}
