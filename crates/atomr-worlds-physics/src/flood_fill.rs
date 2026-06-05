//! Deterministic structural-connectivity flood-fill.
//!
//! When a voxel structure is damaged, some chunks of it may lose every path to
//! an immovable anchor (bedrock / static terrain) and must fall as debris. This
//! module finds those chunks: it labels the 6-connected components of a solid
//! voxel region and reports which components are **not** reachable from an
//! anchor voxel. Those are the floating islands the physics layer turns into
//! dynamic rigid bodies (see [`crate::debris`]).
//!
//! The traversal is a pure function of its inputs — voxels are visited in a
//! fixed `(x, y, z)`-ascending order and components are numbered in
//! first-visit order — so the same region yields byte-identical labels on
//! every machine. This matters because the *same* flood-fill runs both
//! client-side (to spawn local debris) and server-side (to emit deterministic
//! fracture commands for multiplayer); both must agree exactly.

/// Region dimensions `(nx, ny, nz)` in voxels.
pub type Dims = [i32; 3];

/// Result of [`connected_components`].
#[derive(Clone, Debug)]
pub struct Components {
    dims: Dims,
    /// Per-voxel component label in linear order; `-1` for empty voxels.
    label: Vec<i32>,
    /// Per-component flag: did any of its voxels touch an anchor?
    anchored: Vec<bool>,
}

impl Components {
    /// Number of distinct solid components found.
    #[inline]
    pub fn count(&self) -> usize {
        self.anchored.len()
    }

    /// Region dimensions this result was computed over.
    #[inline]
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Component label at `(x, y, z)`, or `None` if the voxel is empty / out of
    /// bounds.
    #[inline]
    pub fn label_at(&self, x: i32, y: i32, z: i32) -> Option<i32> {
        let i = lin(self.dims, x, y, z)?;
        match self.label[i] {
            -1 => None,
            l => Some(l),
        }
    }

    /// Whether component `id` reaches an anchor (and is therefore *not* debris).
    #[inline]
    pub fn is_anchored(&self, id: i32) -> bool {
        self.anchored.get(id as usize).copied().unwrap_or(true)
    }

    /// Collect the voxel coordinates of every component that does **not** reach
    /// an anchor — the floating islands. Each inner `Vec` is one island, and
    /// the coordinates within it are in `(x, y, z)`-ascending order. Islands are
    /// returned in ascending component-id order (i.e. first-discovered first).
    pub fn unanchored_islands(&self) -> Vec<Vec<[i32; 3]>> {
        let mut islands: Vec<Vec<[i32; 3]>> = (0..self.count())
            .filter(|&c| !self.anchored[c])
            .map(|_| Vec::new())
            .collect();
        // Map component id -> compacted island index.
        let mut compact = vec![usize::MAX; self.count()];
        let mut next = 0usize;
        for (c, &is_anchored) in self.anchored.iter().enumerate() {
            if !is_anchored {
                compact[c] = next;
                next += 1;
            }
        }
        let [nx, ny, nz] = self.dims;
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let i = (x * ny * nz + y * nz + z) as usize;
                    let l = self.label[i];
                    if l >= 0 && !self.anchored[l as usize] {
                        islands[compact[l as usize]].push([x, y, z]);
                    }
                }
            }
        }
        islands
    }
}

#[inline]
fn lin(dims: Dims, x: i32, y: i32, z: i32) -> Option<usize> {
    let [nx, ny, nz] = dims;
    if x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz {
        return None;
    }
    Some((x * ny * nz + y * nz + z) as usize)
}

/// Label the 6-connected solid components of a voxel region and mark which
/// components touch an anchor.
///
/// `is_solid(x, y, z)` reports whether a cell is part of a body; `is_anchor(x,
/// y, z)` reports whether a *solid* cell is fixed to the world (e.g. its world
/// `y` is below the static-terrain threshold, or it overlaps an immovable
/// chunk). Both are queried only for in-bounds coordinates.
///
/// Uses an explicit stack (no recursion) so arbitrarily large regions cannot
/// blow the call stack.
pub fn connected_components(
    dims: Dims,
    is_solid: impl Fn(i32, i32, i32) -> bool,
    is_anchor: impl Fn(i32, i32, i32) -> bool,
) -> Components {
    let [nx, ny, nz] = dims;
    let n = (nx.max(0) * ny.max(0) * nz.max(0)) as usize;
    let mut label = vec![-1i32; n];
    let mut anchored: Vec<bool> = Vec::new();
    let mut stack: Vec<[i32; 3]> = Vec::new();

    // Deterministic seed order: ascending x, then y, then z.
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let i = (x * ny * nz + y * nz + z) as usize;
                if label[i] != -1 || !is_solid(x, y, z) {
                    continue;
                }
                let comp = anchored.len() as i32;
                anchored.push(false);
                label[i] = comp;
                stack.push([x, y, z]);
                while let Some([cx, cy, cz]) = stack.pop() {
                    if is_anchor(cx, cy, cz) {
                        anchored[comp as usize] = true;
                    }
                    // 6-neighbourhood, fixed order for determinism.
                    const NEIGH: [[i32; 3]; 6] = [
                        [-1, 0, 0],
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 1, 0],
                        [0, 0, -1],
                        [0, 0, 1],
                    ];
                    for [dx, dy, dz] in NEIGH {
                        let (nx2, ny2, nz2) = (cx + dx, cy + dy, cz + dz);
                        if let Some(j) = lin(dims, nx2, ny2, nz2) {
                            if label[j] == -1 && is_solid(nx2, ny2, nz2) {
                                label[j] = comp;
                                stack.push([nx2, ny2, nz2]);
                            }
                        }
                    }
                }
            }
        }
    }

    Components { dims, label, anchored }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a dense bool grid helper for tests.
    fn grid(solid: &[[i32; 3]]) -> impl Fn(i32, i32, i32) -> bool + '_ {
        move |x, y, z| solid.iter().any(|&[a, b, c]| a == x && b == y && c == z)
    }

    #[test]
    fn single_anchored_blob_has_no_islands() {
        let dims = [3, 3, 1];
        let solid = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]];
        let c = connected_components(dims, grid(&solid), |_, y, _| y == 0);
        assert_eq!(c.count(), 1);
        assert!(c.is_anchored(0));
        assert!(c.unanchored_islands().is_empty());
    }

    #[test]
    fn floating_blob_is_detected_as_island() {
        // y=0 row is anchored ground; a separate blob floats at y=2.
        let dims = [4, 4, 1];
        let solid = [
            [0, 0, 0],
            [1, 0, 0], // ground (anchored)
            [2, 2, 0],
            [3, 2, 0], // floating island
        ];
        let c = connected_components(dims, grid(&solid), |_, y, _| y == 0);
        assert_eq!(c.count(), 2);
        let islands = c.unanchored_islands();
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0], vec![[2, 2, 0], [3, 2, 0]]);
    }

    #[test]
    fn diagonal_voxels_are_not_connected() {
        // 6-connectivity: a diagonal touch does NOT join components.
        let dims = [2, 2, 1];
        let solid = [[0, 0, 0], [1, 1, 0]];
        // Anchor only the first; the diagonal neighbour must be its own island.
        let c = connected_components(dims, grid(&solid), |x, y, _| x == 0 && y == 0);
        assert_eq!(c.count(), 2);
        assert_eq!(c.unanchored_islands(), vec![vec![[1, 1, 0]]]);
    }

    #[test]
    fn labels_are_deterministic() {
        let dims = [5, 5, 2];
        let solid = [[0, 0, 0], [4, 4, 1], [4, 4, 0]];
        let run = || {
            let c = connected_components(dims, grid(&solid), |_, _, _| false);
            (0..dims[0])
                .flat_map(|x| {
                    (0..dims[1]).flat_map(move |y| (0..dims[2]).map(move |z| (x, y, z)))
                })
                .map(|(x, y, z)| c.label_at(x, y, z))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
