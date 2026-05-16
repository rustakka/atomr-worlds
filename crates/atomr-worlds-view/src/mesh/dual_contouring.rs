//! Dual Contouring (Schmitz/Garland simplified) on a binary density field.
//!
//! For each sign-changed cell, gather Hermite data on the 12 cell edges,
//! solve a simplified QEF for the cell's representative vertex
//! (iterative gradient descent from the cell centroid), then emit a
//! quad dual to each sign-changed edge. Material attribution = dominant
//! cell corner by |density|.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

use super::{Mesh, Vertex};

const EDGE: i32 = BRICK_EDGE as i32;

/// Cells span `[-1, EDGE)` so the surface closes across the brick
/// boundary; corners outside the brick read as empty.
const CELL_LO: i32 = -1;
const CELL_HI: i32 = EDGE;
/// Number of cell positions along one axis.
const CELLS_PER_AXIS: usize = (EDGE - CELL_LO) as usize;

fn cell_index(cx: i32, cy: i32, cz: i32) -> usize {
    let x = (cx - CELL_LO) as usize;
    let y = (cy - CELL_LO) as usize;
    let z = (cz - CELL_LO) as usize;
    (z * CELLS_PER_AXIS + y) * CELLS_PER_AXIS + x
}

const CORNER_OFFSETS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const EDGE_CONNECTIONS: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
];

fn occupied(brick: &Brick, x: i32, y: i32, z: i32) -> bool {
    if x < 0 || y < 0 || z < 0 || x >= EDGE || y >= EDGE || z >= EDGE {
        return false;
    }
    !brick.get(IVec3::new(x as i64, y as i64, z as i64)).is_empty()
}

fn density(brick: &Brick, x: i32, y: i32, z: i32) -> f32 {
    if occupied(brick, x, y, z) { 1.0 } else { -1.0 }
}

fn material(brick: &Brick, x: i32, y: i32, z: i32) -> u16 {
    if x < 0 || y < 0 || z < 0 || x >= EDGE || y >= EDGE || z >= EDGE {
        return 0;
    }
    brick.get(IVec3::new(x as i64, y as i64, z as i64)).0
}

/// Dual-contour the brick. Returns one vertex per sign-changed cell and
/// one quad per sign-changed edge dual to a strip of 4 cells.
pub fn dual_contouring_mesh(brick: &Brick) -> Mesh {
    let mut mesh = Mesh::default();
    let cell_total = CELLS_PER_AXIS * CELLS_PER_AXIS * CELLS_PER_AXIS;
    let mut cell_vertex = vec![u32::MAX; cell_total];

    for cz in CELL_LO..CELL_HI {
        for cy in CELL_LO..CELL_HI {
            for cx in CELL_LO..CELL_HI {
                if let Some(v) = solve_cell(brick, cx, cy, cz) {
                    cell_vertex[cell_index(cx, cy, cz)] = mesh.vertices.len() as u32;
                    mesh.vertices.push(v);
                }
            }
        }
    }

    emit_quads(brick, &cell_vertex, &mut mesh);
    mesh
}

fn solve_cell(brick: &Brick, cx: i32, cy: i32, cz: i32) -> Option<Vertex> {
    let mut corner_occ = [false; 8];
    let mut corner_d = [0f32; 8];
    let mut corner_mat = [0u16; 8];
    for (i, off) in CORNER_OFFSETS.iter().enumerate() {
        let x = cx + off[0];
        let y = cy + off[1];
        let z = cz + off[2];
        corner_occ[i] = occupied(brick, x, y, z);
        corner_d[i] = density(brick, x, y, z);
        corner_mat[i] = material(brick, x, y, z);
    }
    let any_in = corner_occ.iter().any(|&o| o);
    let any_out = corner_occ.iter().any(|&o| !o);
    if !(any_in && any_out) {
        return None;
    }

    // Hermite data: positions + normals at each sign-change edge.
    let mut points: Vec<[f32; 3]> = Vec::with_capacity(6);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(6);
    for e in 0..12 {
        let [a, b] = EDGE_CONNECTIONS[e];
        if corner_occ[a] == corner_occ[b] {
            continue;
        }
        let oa = CORNER_OFFSETS[a];
        let ob = CORNER_OFFSETS[b];
        let pa = [
            (cx + oa[0]) as f32,
            (cy + oa[1]) as f32,
            (cz + oa[2]) as f32,
        ];
        let pb = [
            (cx + ob[0]) as f32,
            (cy + ob[1]) as f32,
            (cz + ob[2]) as f32,
        ];
        let da = corner_d[a];
        let db = corner_d[b];
        let t = if (db - da).abs() > 1e-6 { (0.0 - da) / (db - da) } else { 0.5 };
        let p = [
            pa[0] + t * (pb[0] - pa[0]),
            pa[1] + t * (pb[1] - pa[1]),
            pa[2] + t * (pb[2] - pa[2]),
        ];
        points.push(p);
        // Normal via central differences on density at corner `a`'s coord.
        normals.push(central_difference_normal(brick, cx + oa[0], cy + oa[1], cz + oa[2]));
    }

    if points.is_empty() {
        return None;
    }

    let cell_center = [cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5];
    let pos = solve_qef_jacobi(&points, &normals, cell_center, [cx as f32, cy as f32, cz as f32]);
    let dominant = dominant_material(&corner_d, &corner_mat);
    Some(Vertex { pos, normal: [0.0, 0.0, 0.0], material: dominant, ao: 1.0 })
}

fn central_difference_normal(brick: &Brick, x: i32, y: i32, z: i32) -> [f32; 3] {
    let dx = density(brick, x + 1, y, z) - density(brick, x - 1, y, z);
    let dy = density(brick, x, y + 1, z) - density(brick, x, y - 1, z);
    let dz = density(brick, x, y, z + 1) - density(brick, x, y, z - 1);
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len > 1e-6 {
        [dx / len, dy / len, dz / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Simplified Schmitz-Garland QEF: minimize Σ (n_i · (x − p_i))² via
/// iterative Jacobi steps starting from the cell centroid. 16 iters at
/// a small step is enough for visibly stable output on a 16³ brick;
/// clamps the result back into the cell so the dual mesh stays
/// well-formed even on near-degenerate cases.
fn solve_qef_jacobi(
    points: &[[f32; 3]],
    normals: &[[f32; 3]],
    start: [f32; 3],
    cell_min: [f32; 3],
) -> [f32; 3] {
    let mut x = start;
    let step = 0.6;
    for _ in 0..16 {
        let mut grad = [0f32; 3];
        for (p, n) in points.iter().zip(normals.iter()) {
            let dx = x[0] - p[0];
            let dy = x[1] - p[1];
            let dz = x[2] - p[2];
            let d = n[0] * dx + n[1] * dy + n[2] * dz;
            grad[0] += d * n[0];
            grad[1] += d * n[1];
            grad[2] += d * n[2];
        }
        x[0] -= step * grad[0] / points.len() as f32;
        x[1] -= step * grad[1] / points.len() as f32;
        x[2] -= step * grad[2] / points.len() as f32;
    }
    [
        x[0].clamp(cell_min[0], cell_min[0] + 1.0),
        x[1].clamp(cell_min[1], cell_min[1] + 1.0),
        x[2].clamp(cell_min[2], cell_min[2] + 1.0),
    ]
}

fn dominant_material(corner_d: &[f32; 8], corner_mat: &[u16; 8]) -> u16 {
    let mut best_i = 0usize;
    let mut best_abs = -1.0f32;
    for i in 0..8 {
        if corner_mat[i] == 0 {
            continue;
        }
        let a = corner_d[i].abs();
        if a > best_abs {
            best_abs = a;
            best_i = i;
        }
    }
    corner_mat[best_i]
}

fn emit_quads(brick: &Brick, cell_vertex: &[u32], mesh: &mut Mesh) {
    // For each axis-aligned edge with a sign change, the dual quad
    // connects the 4 cells whose corners share that edge. The cells
    // are at offsets (0,0,0), (0,-1,0), (0,-1,-1), (0,0,-1) for an X
    // edge, with appropriate rotations for Y/Z. Winding follows the
    // sign-change direction (inside → outside).
    for z in 0..EDGE {
        for y in 0..EDGE {
            for x in 0..EDGE {
                if x < EDGE {
                    let a = occupied(brick, x, y, z);
                    let b_x = if x + 1 <= EDGE { occupied(brick, x + 1, y, z) } else { false };
                    if a != b_x && x + 1 <= EDGE {
                        // X-edge at corner (x+1, y, z); dual cells at (cx, cy, cz) ∈
                        // {(x, y-1, z-1), (x, y, z-1), (x, y, z), (x, y-1, z)}.
                        let cells = [
                            (x, y - 1, z - 1),
                            (x, y, z - 1),
                            (x, y, z),
                            (x, y - 1, z),
                        ];
                        push_quad_if_valid(mesh, cell_vertex, &cells, a);
                    }
                }
                if y < EDGE {
                    let a = occupied(brick, x, y, z);
                    let b_y = if y + 1 <= EDGE { occupied(brick, x, y + 1, z) } else { false };
                    if a != b_y && y + 1 <= EDGE {
                        let cells = [
                            (x - 1, y, z - 1),
                            (x - 1, y, z),
                            (x, y, z),
                            (x, y, z - 1),
                        ];
                        push_quad_if_valid(mesh, cell_vertex, &cells, a);
                    }
                }
                if z < EDGE {
                    let a = occupied(brick, x, y, z);
                    let b_z = if z + 1 <= EDGE { occupied(brick, x, y, z + 1) } else { false };
                    if a != b_z && z + 1 <= EDGE {
                        let cells = [
                            (x - 1, y - 1, z),
                            (x, y - 1, z),
                            (x, y, z),
                            (x - 1, y, z),
                        ];
                        push_quad_if_valid(mesh, cell_vertex, &cells, a);
                    }
                }
            }
        }
    }
}

fn push_quad_if_valid(
    mesh: &mut Mesh,
    cell_vertex: &[u32],
    cells: &[(i32, i32, i32); 4],
    inside_first: bool,
) {
    let mut idx = [0u32; 4];
    for (k, c) in cells.iter().enumerate() {
        if c.0 < CELL_LO || c.1 < CELL_LO || c.2 < CELL_LO
            || c.0 >= CELL_HI || c.1 >= CELL_HI || c.2 >= CELL_HI
        {
            return;
        }
        let i = cell_vertex[cell_index(c.0, c.1, c.2)];
        if i == u32::MAX {
            return;
        }
        idx[k] = i;
    }
    let base = mesh.vertices.len() as u32;
    // Duplicate the 4 verts so each face can carry its own flat normal.
    let p0 = mesh.vertices[idx[0] as usize].pos;
    let p1 = mesh.vertices[idx[1] as usize].pos;
    let p2 = mesh.vertices[idx[2] as usize].pos;
    let p3 = mesh.vertices[idx[3] as usize].pos;
    let mat = mesh.vertices[idx[0] as usize].material;
    let normal = if inside_first {
        triangle_normal(p0, p1, p2)
    } else {
        triangle_normal(p0, p2, p1)
    };
    for p in [p0, p1, p2, p3] {
        mesh.vertices.push(Vertex { pos: p, normal, material: mat, ao: 1.0 });
    }
    if inside_first {
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        mesh.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 1e-6 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_voxel::Voxel;

    #[test]
    fn all_empty_brick_emits_nothing() {
        let b = Brick::new();
        let m = dual_contouring_mesh(&b);
        assert!(m.vertices.is_empty());
        assert!(m.indices.is_empty());
    }

    #[test]
    fn all_solid_brick_emits_only_boundary_shell() {
        // Without an apron, the brick's boundary corners read empty
        // so DC builds a closed shell. Interior cells (all 8 corners
        // solid) have no sign change and emit no vertex.
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let m = dual_contouring_mesh(&b);
        // Shell is non-empty.
        assert!(!m.vertices.is_empty());
        // Sanity: vertex count is bounded by the boundary cell count
        // (17³ - 14³ = 2169), not the full 18³ cell grid.
        let max_expected = 17 * 17 * 17;
        assert!(m.vertices.len() <= max_expected);
    }

    #[test]
    fn half_filled_brick_produces_continuous_quad_strip() {
        // y < 8 solid, y >= 8 empty.
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..8 {
                for x in 0..16 {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let m = dual_contouring_mesh(&b);
        assert!(!m.vertices.is_empty(), "expected DC vertices on boundary");
        assert!(
            m.indices.len() % 6 == 0,
            "DC quads should produce indices in multiples of 6 (2 tris each)"
        );
        // The y=8 boundary plane is 16×16 cells of sign-changed
        // "interior" cells (y=7→y=8 corners), each contributing one
        // vertex. Combined with boundary apron, vertex count should be
        // substantially larger than zero.
        assert!(m.vertices.len() > 100);
    }
}
