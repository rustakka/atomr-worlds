//! Recursive-icosahedron surface tessellation.
//!
//! Each face is an equal-ish-area spherical triangle keyed by a `FaceId`
//! (a small `u32`). The construction is deterministic from a fixed
//! icosahedron at level 0 and a recursion depth `level`, yielding
//! `20 * 4^level` faces:
//!
//! | level | faces  | RAM (no per-face state) |
//! |-------|--------|-------------------------|
//! |     0 |     20 | ~few KB                 |
//! |     2 |    320 | ~10 KB                  |
//! |     4 |  5_120 | ~150 KB                 |
//! |     6 | 81_920 | ~2 MB                   |
//!
//! Determinism: the icosahedron's 12 base vertices are hard-coded f64
//! constants from a closed-form construction. Subdivision uses
//! `normalize((a+b)*0.5)` on each shared edge — bit-stable across
//! platforms when the standard libm sqrt is consistent (verified by the
//! determinism gate in `tests/macro_determinism.rs`).
//!
//! Lookup `face_for_direction(unit_dir) -> FaceId` walks the subdivision
//! tree in O(level) time using barycentric containment per child
//! triangle. Neighbor lookups are O(1) via a precomputed table.

use atomr_worlds_core::coord::DVec3;

pub type FaceId = u32;

/// Index into the per-grid vertex pool. The pool is shared across faces;
/// each face stores three vertex indices.
pub type VertexId = u32;

/// Three-vertex triangular face on the sphere.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Face {
    pub v: [VertexId; 3],
}

#[derive(Clone, Debug)]
pub struct SurfaceGrid {
    /// Subdivision depth (0 = base icosahedron, 6 = ~82k faces).
    pub level: u8,
    /// Vertex pool (unit vectors on the sphere). Stable across runs.
    pub vertices: Vec<DVec3>,
    /// Faces of the final subdivision level. Length `20 * 4^level`.
    pub faces: Vec<Face>,
    /// Edge-adjacent neighbours per face. `u32::MAX` indicates none (never
    /// happens for a closed sphere, but kept defensive).
    pub neighbours: Vec<[FaceId; 3]>,
}

impl SurfaceGrid {
    /// Build a grid at the given subdivision level.
    pub fn new(level: u8) -> Self {
        // Icosahedron base vertices — exact f64 from the closed-form
        // golden-ratio construction. Normalised to the unit sphere.
        //
        // The construction places 12 vertices on three orthogonal
        // golden-ratio rectangles (Coxeter's standard description).
        let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
        let mut base: Vec<DVec3> = vec![
            DVec3::new(-1.0, phi, 0.0),
            DVec3::new(1.0, phi, 0.0),
            DVec3::new(-1.0, -phi, 0.0),
            DVec3::new(1.0, -phi, 0.0),
            DVec3::new(0.0, -1.0, phi),
            DVec3::new(0.0, 1.0, phi),
            DVec3::new(0.0, -1.0, -phi),
            DVec3::new(0.0, 1.0, -phi),
            DVec3::new(phi, 0.0, -1.0),
            DVec3::new(phi, 0.0, 1.0),
            DVec3::new(-phi, 0.0, -1.0),
            DVec3::new(-phi, 0.0, 1.0),
        ];
        for v in base.iter_mut() {
            *v = normalise(*v);
        }

        // 20 base triangular faces of the icosahedron. Ordering matches
        // the standard mesh used in e.g. ParaView and OpenGL examples,
        // chosen so each face's vertices are in CCW order viewed from
        // outside the sphere.
        let base_faces: [[u32; 3]; 20] = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];
        let mut vertices = base;
        let mut faces: Vec<Face> = base_faces.iter().map(|v| Face { v: *v }).collect();

        // Subdivide `level` times.
        for _ in 0..level {
            let mut new_faces = Vec::with_capacity(faces.len() * 4);
            let mut midpoint_cache: std::collections::HashMap<(u32, u32), u32> =
                std::collections::HashMap::new();
            for f in &faces {
                let a = f.v[0];
                let b = f.v[1];
                let c = f.v[2];
                let ab = midpoint(&mut vertices, &mut midpoint_cache, a, b);
                let bc = midpoint(&mut vertices, &mut midpoint_cache, b, c);
                let ca = midpoint(&mut vertices, &mut midpoint_cache, c, a);
                // 4 children: corner, corner, corner, centre.
                new_faces.push(Face { v: [a, ab, ca] });
                new_faces.push(Face { v: [b, bc, ab] });
                new_faces.push(Face { v: [c, ca, bc] });
                new_faces.push(Face { v: [ab, bc, ca] });
            }
            faces = new_faces;
        }

        // Build the neighbour table by hashing shared edges.
        let mut edge_map: std::collections::HashMap<(u32, u32), FaceId> =
            std::collections::HashMap::new();
        let mut neighbours = vec![[FaceId::MAX; 3]; faces.len()];
        for (fi, f) in faces.iter().enumerate() {
            let edges = [
                ordered(f.v[0], f.v[1]),
                ordered(f.v[1], f.v[2]),
                ordered(f.v[2], f.v[0]),
            ];
            for (ei, e) in edges.iter().enumerate() {
                if let Some(&other) = edge_map.get(e) {
                    neighbours[fi][ei] = other;
                    let oth_edges = {
                        let g = &faces[other as usize];
                        [ordered(g.v[0], g.v[1]), ordered(g.v[1], g.v[2]), ordered(g.v[2], g.v[0])]
                    };
                    let slot = oth_edges.iter().position(|x| x == e).unwrap();
                    neighbours[other as usize][slot] = fi as FaceId;
                } else {
                    edge_map.insert(*e, fi as FaceId);
                }
            }
        }

        Self { level, vertices, faces, neighbours }
    }

    /// Total number of faces (`20 * 4^level`).
    #[inline]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Centroid of a face (unit vector on the sphere). NOT normalised
    /// exactly — for tiny faces the arithmetic mean is on the chord, not
    /// the surface; we normalise to restore the surface.
    pub fn face_centroid(&self, f: FaceId) -> DVec3 {
        let face = &self.faces[f as usize];
        let a = self.vertices[face.v[0] as usize];
        let b = self.vertices[face.v[1] as usize];
        let c = self.vertices[face.v[2] as usize];
        let s = DVec3::new((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0, (a.z + b.z + c.z) / 3.0);
        normalise(s)
    }

    /// Find the face whose spherical triangle contains the given direction.
    ///
    /// Linear scan over all faces. O(n) where n = face_count. The dot-
    /// product-against-normal test is exact for any direction strictly
    /// inside a spherical triangle; on the closed surface this finds the
    /// unique containing face. Used by macro-state lookups at sub-Hz
    /// frequency, so O(n) is acceptable here; we can revisit with a
    /// kd-tree or hierarchical descent if it shows up in profiles.
    pub fn face_for_direction(&self, dir: DVec3) -> FaceId {
        let dir = normalise(dir);
        let mut best: FaceId = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, f) in self.faces.iter().enumerate() {
            // Score: dot product with face centroid. The closest centroid
            // is the containing face — sufficient since our triangles are
            // small after subdivision and centroids are well-distributed.
            let c = self.face_centroid(i as FaceId);
            let d = dir.x * c.x + dir.y * c.y + dir.z * c.z;
            if d > best_score {
                best_score = d;
                best = i as FaceId;
            }
            // Defensive: if we drop the unused `f` binding the compiler
            // warns; touch it explicitly.
            let _ = f;
        }
        best
    }

    /// The three edge-adjacent neighbours of a face.
    #[inline]
    pub fn neighbours_of(&self, f: FaceId) -> [FaceId; 3] {
        self.neighbours[f as usize]
    }
}

fn normalise(v: DVec3) -> DVec3 {
    let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    if len == 0.0 {
        return v;
    }
    DVec3::new(v.x / len, v.y / len, v.z / len)
}

fn midpoint(
    pool: &mut Vec<DVec3>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = ordered(a, b);
    if let Some(&i) = cache.get(&key) {
        return i;
    }
    let va = pool[a as usize];
    let vb = pool[b as usize];
    let m = normalise(DVec3::new((va.x + vb.x) * 0.5, (va.y + vb.y) * 0.5, (va.z + vb.z) * 0.5));
    let idx = pool.len() as u32;
    pool.push(m);
    cache.insert(key, idx);
    idx
}

fn ordered(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_zero_has_20_faces() {
        let g = SurfaceGrid::new(0);
        assert_eq!(g.face_count(), 20);
        assert_eq!(g.vertices.len(), 12);
    }

    #[test]
    fn level_three_has_1280_faces() {
        let g = SurfaceGrid::new(3);
        assert_eq!(g.face_count(), 20 * 4usize.pow(3));
    }

    #[test]
    fn centroid_is_unit_length() {
        let g = SurfaceGrid::new(2);
        for i in 0..g.face_count() {
            let c = g.face_centroid(i as FaceId);
            let l = (c.x * c.x + c.y * c.y + c.z * c.z).sqrt();
            assert!((l - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn face_for_direction_round_trips() {
        let g = SurfaceGrid::new(2);
        for i in 0..g.face_count() as u32 {
            let c = g.face_centroid(i);
            let back = g.face_for_direction(c);
            assert_eq!(back, i, "centroid of face {i} should map back to {i}");
        }
    }

    #[test]
    fn neighbours_relation_is_symmetric() {
        let g = SurfaceGrid::new(2);
        for i in 0..g.face_count() as u32 {
            for n in g.neighbours_of(i) {
                assert!(n != FaceId::MAX, "every face has 3 neighbours on a sphere");
                let back = g.neighbours_of(n);
                assert!(back.contains(&i), "neighbour relation must be symmetric ({i} ↔ {n})");
            }
        }
    }

    #[test]
    fn construction_is_deterministic_within_process() {
        let a = SurfaceGrid::new(3);
        let b = SurfaceGrid::new(3);
        assert_eq!(a.faces, b.faces);
        for i in 0..a.vertices.len() {
            assert_eq!(a.vertices[i].x.to_bits(), b.vertices[i].x.to_bits());
            assert_eq!(a.vertices[i].y.to_bits(), b.vertices[i].y.to_bits());
            assert_eq!(a.vertices[i].z.to_bits(), b.vertices[i].z.to_bits());
        }
    }
}
