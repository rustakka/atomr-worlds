//! Arena-allocated sparse voxel octree with brick leaves.
//!
//! Internal nodes use an 8-bit child mask plus a `children_base` arena
//! offset; only present octants take space (`popcount(child_mask)` entries
//! starting at `children_base`). That keeps the internal node tiny (5 bytes
//! payload) so empty-space skipping touches less cache.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;

use crate::brick::{Brick, BRICK_EDGE};
use crate::error::VoxelError;
use crate::voxel::Voxel;

pub type NodeId = u32;
pub const OCTREE_NULL: NodeId = u32::MAX;

#[derive(Copy, Clone, Debug)]
pub struct InternalNode {
    pub child_mask: u8,
    pub children_base: u32,
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    Empty,
    Internal(InternalNode),
    Leaf(NodeId),
}

#[derive(Debug)]
pub struct Octree {
    pub root_size_m: f64,
    pub max_depth: u8,
    pub nodes: Vec<NodeKind>,
    pub bricks: Vec<Brick>,
    /// Test/instrumentation: counts node-arena reads during `get`. Not
    /// authoritative — wrap in `#[cfg(test)]` accessors if you need to
    /// assert on it.
    #[doc(hidden)]
    pub probe_count: std::cell::Cell<u64>,
}

impl Octree {
    pub fn new(root_size_m: f64, max_depth: u8) -> Self {
        assert!(max_depth > 0, "max_depth must be at least 1 (else there are no leaves)");
        Self {
            root_size_m,
            max_depth,
            nodes: vec![NodeKind::Empty],
            bricks: Vec::new(),
            probe_count: std::cell::Cell::new(0),
        }
    }

    /// Edge of the leaf-brick cube in voxels along each axis.
    #[inline]
    pub const fn brick_edge_voxels() -> i64 {
        BRICK_EDGE as i64
    }

    /// Total addressable voxel-space edge: `2^max_depth * BRICK_EDGE`.
    #[inline]
    pub fn voxel_grid_edge(&self) -> i64 {
        (1i64 << self.max_depth) * (BRICK_EDGE as i64)
    }

    fn brick_coord_of(&self, p: IVec3) -> Option<IVec3> {
        let edge = BRICK_EDGE as i64;
        let half = self.voxel_grid_edge() / 2;
        if p.x < -half || p.x >= half || p.y < -half || p.y >= half || p.z < -half || p.z >= half {
            return None;
        }
        // Floor-divide by edge with negative-handling.
        let bx = p.x.div_euclid(edge);
        let by = p.y.div_euclid(edge);
        let bz = p.z.div_euclid(edge);
        Some(IVec3::new(bx, by, bz))
    }

    fn local_in_brick(&self, p: IVec3) -> IVec3 {
        let edge = BRICK_EDGE as i64;
        IVec3::new(p.x.rem_euclid(edge), p.y.rem_euclid(edge), p.z.rem_euclid(edge))
    }

    /// Index of an octant child for the current octree depth.
    ///
    /// Translates the (already-recentred) brick coord such that bit 0..3 of
    /// the octant index packs `(x>=0, y>=0, z>=0)` for the cell at this depth.
    fn octant_index(brick: IVec3, depth_from_root: u8, max_depth: u8) -> u8 {
        let level = max_depth - depth_from_root - 1; // bit position
        let mx = ((brick.x >> level) & 1) as u8;
        let my = ((brick.y >> level) & 1) as u8;
        let mz = ((brick.z >> level) & 1) as u8;
        mx | (my << 1) | (mz << 2)
    }

    /// Reset the probe counter (test helper).
    #[doc(hidden)]
    pub fn reset_probes(&self) {
        self.probe_count.set(0);
    }

    #[doc(hidden)]
    pub fn probes(&self) -> u64 {
        self.probe_count.get()
    }

    fn bump_probe(&self) {
        self.probe_count.set(self.probe_count.get() + 1);
    }

    /// Lookup the brick id for a brick coordinate, walking the octree.
    /// Returns `None` if the path bottoms out at [`NodeKind::Empty`].
    fn lookup_brick(&self, brick: IVec3) -> Option<NodeId> {
        if self.nodes.is_empty() {
            return None;
        }
        // Recentre so brick coords are in [0, 2^max_depth).
        let half = 1i64 << (self.max_depth - 1);
        let recentred = IVec3::new(brick.x + half, brick.y + half, brick.z + half);
        if recentred.x < 0
            || recentred.y < 0
            || recentred.z < 0
            || recentred.x >= (1i64 << self.max_depth)
            || recentred.y >= (1i64 << self.max_depth)
            || recentred.z >= (1i64 << self.max_depth)
        {
            return None;
        }

        let mut cur: NodeId = 0;
        for depth in 0..self.max_depth {
            self.bump_probe();
            match self.nodes[cur as usize].clone() {
                NodeKind::Empty => return None,
                NodeKind::Leaf(_) => return None, // shouldn't happen mid-walk; defensive
                NodeKind::Internal(node) => {
                    let octant = Self::octant_index(recentred, depth, self.max_depth);
                    if (node.child_mask & (1u8 << octant)) == 0 {
                        return None;
                    }
                    let slot = (node.child_mask & ((1u8 << octant) - 1)).count_ones() as u32;
                    cur = node.children_base + slot;
                }
            }
        }
        self.bump_probe();
        match self.nodes[cur as usize] {
            NodeKind::Leaf(brick_id) => Some(brick_id),
            _ => None,
        }
    }

    /// Look up the brick at the given brick coordinate (cosmic — relative to
    /// the recentred octree origin).
    pub fn brick(&self, brick_coord: IVec3) -> Result<Option<&Brick>, VoxelError> {
        match self.lookup_brick(brick_coord) {
            Some(id) => Ok(self.bricks.get(id as usize)),
            None => Ok(None),
        }
    }

    /// Read a single voxel.
    pub fn get_voxel(&self, p: IVec3) -> Result<Voxel, VoxelError> {
        let Some(brick_coord) = self.brick_coord_of(p) else {
            return Err(VoxelError::OutOfBounds(p));
        };
        match self.lookup_brick(brick_coord) {
            Some(id) => Ok(self.bricks[id as usize].get(self.local_in_brick(p))),
            None => Ok(Voxel::EMPTY),
        }
    }

    /// Write a single voxel, allocating internal nodes and a brick as needed.
    pub fn set_voxel(&mut self, p: IVec3, v: Voxel) -> Result<(), VoxelError> {
        let Some(brick_coord) = self.brick_coord_of(p) else {
            return Err(VoxelError::OutOfBounds(p));
        };
        let half = 1i64 << (self.max_depth - 1);
        let recentred =
            IVec3::new(brick_coord.x + half, brick_coord.y + half, brick_coord.z + half);

        let local = self.local_in_brick(p);
        let brick_id = self.ensure_path(recentred);
        self.bricks[brick_id as usize].set(local, v);
        Ok(())
    }

    /// Walk to the leaf, growing internal nodes / bricks as needed, return the brick id.
    fn ensure_path(&mut self, recentred: IVec3) -> NodeId {
        let mut cur: NodeId = 0;
        for depth in 0..self.max_depth {
            let octant = Self::octant_index(recentred, depth, self.max_depth);
            let is_leaf_parent = depth + 1 == self.max_depth;
            let next_id = self.ensure_child(cur, octant, is_leaf_parent);
            cur = next_id;
        }
        // cur now points at a leaf node containing the brick id
        match self.nodes[cur as usize] {
            NodeKind::Leaf(brick_id) => brick_id,
            _ => unreachable!("ensure_path must end at a Leaf"),
        }
    }

    /// Ensure a child of `parent` for `octant` exists; return the child node id.
    /// If `child_is_leaf`, the new node is `Leaf(brick_id)`; otherwise `Internal`.
    fn ensure_child(&mut self, parent: NodeId, octant: u8, child_is_leaf: bool) -> NodeId {
        let make_empty_node = |this: &mut Self, leaf: bool| -> NodeId {
            let id = this.nodes.len() as u32;
            if leaf {
                let brick_id = this.bricks.len() as u32;
                this.bricks.push(Brick::new());
                this.nodes.push(NodeKind::Leaf(brick_id));
            } else {
                this.nodes.push(NodeKind::Internal(InternalNode { child_mask: 0, children_base: 0 }));
            }
            id
        };

        let parent_node = self.nodes[parent as usize].clone();
        let (mut child_mask, children_base) = match parent_node {
            NodeKind::Internal(n) => (n.child_mask, n.children_base),
            NodeKind::Empty => {
                // Promote empty to internal.
                self.nodes[parent as usize] =
                    NodeKind::Internal(InternalNode { child_mask: 0, children_base: 0 });
                (0u8, 0u32)
            }
            NodeKind::Leaf(_) => {
                // Shouldn't happen; parent should be internal at this depth.
                unreachable!("cannot add octant child to a leaf node")
            }
        };

        let octant_bit = 1u8 << octant;
        let already_present = (child_mask & octant_bit) != 0;

        if already_present {
            let slot = (child_mask & (octant_bit - 1)).count_ones() as u32;
            return children_base + slot;
        }

        // Need to insert. Append a new contiguous run for children, copy existing siblings + new node.
        let popcount_old = child_mask.count_ones() as u32;
        let new_base = self.nodes.len() as u32;
        // Copy existing siblings (preserving popcount order) into the new contiguous range.
        for i in 0..popcount_old {
            let src = (children_base + i) as usize;
            let cloned = self.nodes[src].clone();
            self.nodes.push(cloned);
        }
        // Insert the new child at its sorted position.
        let new_child_id = make_empty_node(self, child_is_leaf);
        // The new child is at the end; we need it ordered by octant index.
        let slot = (child_mask & (octant_bit - 1)).count_ones() as u32;
        let dest_index = new_base + slot;
        let last_index = self.nodes.len() as u32 - 1;
        // Rotate the freshly-appended node into position.
        if dest_index != last_index {
            // shift [dest_index..last_index] right by one and place new at dest_index
            let new_node = self.nodes.remove(last_index as usize);
            self.nodes.insert(dest_index as usize, new_node);
        }
        let _ = new_child_id; // id is now `dest_index` after the insert/rotate

        child_mask |= octant_bit;
        self.nodes[parent as usize] =
            NodeKind::Internal(InternalNode { child_mask, children_base: new_base });

        dest_index
    }

    /// Convenience: convert a [`Lod`] into the effective octree depth used for
    /// queries. Currently the octree always serves leaves; this hook exists so
    /// later phases can satisfy queries from mip-levels above the leaf.
    pub fn check_lod(&self, lod: Lod) -> Result<(), VoxelError> {
        if lod.depth > self.max_depth {
            Err(VoxelError::LodTooDeep { requested: lod.depth, max: self.max_depth })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_octree_returns_empty_voxels() {
        let oct = Octree::new(1024.0, 4);
        let v = oct.get_voxel(IVec3::new(0, 0, 0)).unwrap();
        assert_eq!(v, Voxel::EMPTY);
    }

    #[test]
    fn set_then_get_round_trip() {
        let mut oct = Octree::new(1024.0, 4);
        oct.set_voxel(IVec3::new(0, 0, 0), Voxel::new(42)).unwrap();
        assert_eq!(oct.get_voxel(IVec3::new(0, 0, 0)).unwrap(), Voxel::new(42));
    }

    #[test]
    fn distant_voxel_round_trip() {
        let mut oct = Octree::new(1024.0, 4);
        let p = IVec3::new(31, -17, 5);
        oct.set_voxel(p, Voxel::new(7)).unwrap();
        assert_eq!(oct.get_voxel(p).unwrap(), Voxel::new(7));
    }

    #[test]
    fn out_of_bounds_errors() {
        let oct = Octree::new(1024.0, 2); // grid edge = 4 * 16 = 64; half-extent = 32
        assert!(oct.get_voxel(IVec3::new(1_000_000, 0, 0)).is_err());
    }

    #[test]
    fn empty_space_skip_is_shallow() {
        let mut oct = Octree::new(1024.0, 4);
        oct.set_voxel(IVec3::new(0, 0, 0), Voxel::new(1)).unwrap();
        oct.reset_probes();
        // Probe many empty cells; per-probe descent should be ≤ max_depth + 1.
        for n in 1..2000 {
            let p = IVec3::new(n % 31, (n * 7) % 31, (n * 13) % 31);
            let _ = oct.get_voxel(p).unwrap();
        }
        // bounded by descent length per probe
        assert!(oct.probes() <= 2000 * (oct.max_depth as u64 + 1));
    }
}
