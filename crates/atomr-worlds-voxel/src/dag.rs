//! `DagBrick` — a deduplicated Sparse Voxel **DAG** built from a [`Brick`].
//!
//! An [`SvoBrick`](crate::storage::SvoBrick) is a sparse voxel *octree*: every
//! node has a unique path to the root, so two structurally-identical subtrees
//! (e.g. two solid-stone octants) are stored twice. A Sparse Voxel **DAG**
//! removes that redundancy by *hash-consing* — interning each distinct
//! `(child-mask, children)` / leaf so identical subtrees share a single node.
//! On the homogeneous regions procedural terrain produces (solid rock, open
//! air, flat strata) this collapses thousands of voxels into a handful of
//! nodes, which is the compression the GPU raymarcher (Rec 1 of the *Advanced
//! Voxel Architectures* plan) needs to keep far terrain resident in VRAM.
//!
//! ## Derived, non-canonical state
//!
//! A `DagBrick` is a **pure, derived** view of a `Brick`'s bytes. It is never
//! the canonical store: it does not flow into `VoxelWriteEvent` or the journal,
//! and it has no edit path (SVDAGs are static — the destruction story rebuilds
//! the affected brick's DAG rather than editing in place). The canonical
//! `Brick` is untouched, so the byte-determinism contract is unaffected.
//!
//! ## Determinism
//!
//! Construction is a pure function of the brick: subtrees are built in a fixed
//! octant order (`bx | by<<1 | bz<<2`, matching `SvoBrick`) and interned in
//! first-encounter order, so node ids — and therefore [`DagBrick::digest`] —
//! are identical on every machine regardless of the intern map's hasher (the
//! hasher only affects bucketing, never the id assignment order).
//!
//! ## Geometry / color
//!
//! Material ids live in the leaf nodes today (so two leaves of different
//! material are distinct nodes). The decoupled occupancy-DAG + parallel color
//! array the GPU layout wants is a follow-up that co-designs the flat buffer
//! encoding with the WGSL traversal shader; this module is the CPU-side builder
//! + reconstruction the raymarcher and its determinism mirror will consume.

use std::collections::HashMap;

use crate::brick::{Brick, BRICK_EDGE};
use crate::voxel::Voxel;
use atomr_worlds_core::coord::IVec3;

/// A node in the voxel DAG. Empty subtrees are represented by *absence* (a clear
/// bit in a parent's `mask`), so there is no explicit empty node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DagNode {
    /// A single solid voxel at the leaf level.
    Leaf(u16),
    /// An internal node: an 8-bit child mask plus `popcount(mask)` child ids,
    /// one per set octant in ascending octant order.
    Internal { mask: u8, children: Vec<u32> },
}

/// A deduplicated sparse-voxel DAG over one 16³ brick. Derived state — see the
/// module docs.
#[derive(Debug, Clone)]
pub struct DagBrick {
    nodes: Vec<DagNode>,
    /// Root node id, or `None` for a fully-empty brick.
    root: Option<u32>,
}

impl DagBrick {
    /// Build the minimal DAG for a brick by bottom-up hash-consing.
    pub fn from_brick(brick: &Brick) -> Self {
        let mut nodes: Vec<DagNode> = Vec::new();
        let mut interner: HashMap<DagNode, u32> = HashMap::new();
        let root = build(brick, [0, 0, 0], 0, &mut nodes, &mut interner);
        Self { nodes, root }
    }

    /// Number of distinct nodes in the DAG (the compression metric). A
    /// fully-empty brick is `0`; a fully-uniform brick is `SVO_DEPTH + 1`.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// `true` if the brick was entirely empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Read the voxel at brick-local `(x, y, z)` by descending the DAG.
    pub fn get(&self, x: u8, y: u8, z: u8) -> Voxel {
        let mut node = match self.root {
            Some(r) => r,
            None => return Voxel::EMPTY,
        };
        let mut origin = [0u8; 3];
        let mut depth = 0u8;
        loop {
            match &self.nodes[node as usize] {
                DagNode::Leaf(v) => return Voxel::new(*v),
                DagNode::Internal { mask, children } => {
                    let half = (BRICK_EDGE as u8 >> depth) >> 1;
                    let ox = ((x - origin[0]) >= half) as u8;
                    let oy = ((y - origin[1]) >= half) as u8;
                    let oz = ((z - origin[2]) >= half) as u8;
                    let octant = ox | (oy << 1) | (oz << 2);
                    let bit = 1u8 << octant;
                    if (*mask & bit) == 0 {
                        return Voxel::EMPTY;
                    }
                    let slot = (*mask & (bit - 1)).count_ones() as usize;
                    node = children[slot];
                    origin = [origin[0] + ox * half, origin[1] + oy * half, origin[2] + oz * half];
                    depth += 1;
                }
            }
        }
    }

    /// Reconstruct the dense [`Brick`] this DAG represents. Round-trips exactly:
    /// `DagBrick::from_brick(b).to_brick()` has the same voxels as `b`.
    pub fn to_brick(&self) -> Brick {
        let mut b = Brick::new();
        let edge = BRICK_EDGE as u8;
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    let v = self.get(x, y, z);
                    if !v.is_empty() {
                        b.set(IVec3::new(x as i64, y as i64, z as i64), v);
                    }
                }
            }
        }
        b
    }

    /// A stable FNV-1a digest over the node pool + root. Equal for any two DAGs
    /// with the same structure; the determinism witness.
    pub fn digest(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut mix = |byte: u8| {
            h ^= byte as u64;
            h = h.wrapping_mul(PRIME);
        };
        match self.root {
            None => mix(0),
            Some(r) => {
                mix(1);
                for b in r.to_le_bytes() {
                    mix(b);
                }
            }
        }
        for node in &self.nodes {
            match node {
                DagNode::Leaf(v) => {
                    mix(0xA1); // leaf tag
                    for b in v.to_le_bytes() {
                        mix(b);
                    }
                }
                DagNode::Internal { mask, children } => {
                    mix(0xB2); // internal tag
                    mix(*mask);
                    for c in children {
                        for b in c.to_le_bytes() {
                            mix(b);
                        }
                    }
                }
            }
        }
        h
    }
}

/// Recursively build (and intern) the subtree covering the cube whose min
/// corner is `origin` at `depth` (edge = `16 >> depth`). Returns the interned
/// node id, or `None` for a fully-empty subtree.
fn build(
    brick: &Brick,
    origin: [u8; 3],
    depth: u8,
    nodes: &mut Vec<DagNode>,
    interner: &mut HashMap<DagNode, u32>,
) -> Option<u32> {
    let edge = BRICK_EDGE as u8 >> depth;
    if edge == 1 {
        let v = brick.get(IVec3::new(origin[0] as i64, origin[1] as i64, origin[2] as i64));
        if v.is_empty() {
            return None;
        }
        return Some(intern(DagNode::Leaf(v.0), nodes, interner));
    }
    let half = edge >> 1;
    let mut mask = 0u8;
    let mut children = Vec::new();
    for octant in 0u8..8 {
        let ox = octant & 1;
        let oy = (octant >> 1) & 1;
        let oz = (octant >> 2) & 1;
        let cmin = [origin[0] + ox * half, origin[1] + oy * half, origin[2] + oz * half];
        if let Some(cid) = build(brick, cmin, depth + 1, nodes, interner) {
            mask |= 1 << octant;
            children.push(cid);
        }
    }
    if mask == 0 {
        return None;
    }
    Some(intern(DagNode::Internal { mask, children }, nodes, interner))
}

/// Intern a node: reuse an existing id for a structurally-identical node, else
/// assign the next id in first-encounter order.
fn intern(node: DagNode, nodes: &mut Vec<DagNode>, interner: &mut HashMap<DagNode, u32>) -> u32 {
    if let Some(&id) = interner.get(&node) {
        return id;
    }
    let id = nodes.len() as u32;
    nodes.push(node.clone());
    interner.insert(node, id);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    const EDGE: i64 = BRICK_EDGE as i64;

    fn assert_round_trip(b: &Brick) {
        let dag = DagBrick::from_brick(b);
        for z in 0..EDGE {
            for y in 0..EDGE {
                for x in 0..EDGE {
                    let expected = b.get(IVec3::new(x, y, z));
                    let actual = dag.get(x as u8, y as u8, z as u8);
                    assert_eq!(actual, expected, "mismatch at ({x},{y},{z})");
                }
            }
        }
        let back = dag.to_brick();
        assert_eq!(back.voxels.as_ref(), b.voxels.as_ref());
    }

    #[test]
    fn empty_brick_has_no_nodes() {
        let dag = DagBrick::from_brick(&Brick::new());
        assert_eq!(dag.node_count(), 0);
        assert!(dag.is_empty());
        assert_eq!(dag.get(0, 0, 0), Voxel::EMPTY);
    }

    #[test]
    fn uniform_solid_brick_collapses_to_one_node_per_level() {
        // Every cell the same material ⇒ 1 leaf + 1 internal node per octree
        // level (depths 0..4) = 5 nodes total, regardless of the 4096 voxels.
        let mut b = Brick::new();
        for z in 0..EDGE {
            for y in 0..EDGE {
                for x in 0..EDGE {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let dag = DagBrick::from_brick(&b);
        assert_eq!(dag.node_count(), 5);
        assert_round_trip(&b);
    }

    #[test]
    fn sparse_brick_round_trips() {
        let mut b = Brick::new();
        b.set(IVec3::new(0, 0, 0), Voxel::new(1));
        b.set(IVec3::new(15, 15, 15), Voxel::new(2));
        b.set(IVec3::new(3, 5, 7), Voxel::new(42));
        b.set(IVec3::new(8, 8, 8), Voxel::new(99));
        assert_round_trip(&b);
    }

    #[test]
    fn half_filled_brick_round_trips_and_dedups() {
        // Fill the lower half (y < 8) solid: large homogeneous region ⇒ heavy
        // dedup but still fewer nodes than the naive octree would hold.
        let mut b = Brick::new();
        for z in 0..EDGE {
            for y in 0..8 {
                for x in 0..EDGE {
                    b.set(IVec3::new(x, y, z), Voxel::new(3));
                }
            }
        }
        assert_round_trip(&b);
        let dag = DagBrick::from_brick(&b);
        // Far fewer than the 2048 solid voxels.
        assert!(dag.node_count() < 32, "node_count = {}", dag.node_count());
    }

    #[test]
    fn construction_is_deterministic() {
        let mut b = Brick::new();
        b.set(IVec3::new(1, 2, 3), Voxel::new(7));
        b.set(IVec3::new(14, 1, 9), Voxel::new(7));
        b.set(IVec3::new(2, 2, 2), Voxel::new(5));
        let a = DagBrick::from_brick(&b);
        let c = DagBrick::from_brick(&b);
        assert_eq!(a.node_count(), c.node_count());
        assert_eq!(a.digest(), c.digest());
    }

    #[test]
    fn different_content_has_different_digest() {
        let mut b1 = Brick::new();
        b1.set(IVec3::new(0, 0, 0), Voxel::new(1));
        let mut b2 = Brick::new();
        b2.set(IVec3::new(0, 0, 0), Voxel::new(2));
        assert_ne!(
            DagBrick::from_brick(&b1).digest(),
            DagBrick::from_brick(&b2).digest()
        );
    }
}
