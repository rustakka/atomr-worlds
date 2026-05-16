//! Naive per-voxel-face meshing: emit one quad per exposed voxel face
//!
//! Baseline reference mesher for A/B comparison with the greedy /
//! marching-cubes / dual-contouring strategies. Out-of-brick neighbours
//! are treated as empty, so brick boundaries always emit faces.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

use super::{Mesh, Vertex};

const EDGE: i32 = BRICK_EDGE as i32;

const FACE_DIRS: [[i32; 3]; 6] = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
];

fn material_at(brick: &Brick, x: i32, y: i32, z: i32) -> u16 {
    if x < 0 || y < 0 || z < 0 || x >= EDGE || y >= EDGE || z >= EDGE {
        return 0;
    }
    brick.get(IVec3::new(x as i64, y as i64, z as i64)).0
}

/// One quad per visible voxel face. Material comes from the source
/// voxel; AO stays at `1.0`; normals match the face direction.
pub fn naive_mesh(brick: &Brick) -> Mesh {
    let mut mesh = Mesh::default();
    for z in 0..EDGE {
        for y in 0..EDGE {
            for x in 0..EDGE {
                let m = material_at(brick, x, y, z);
                if m == 0 {
                    continue;
                }
                for (face_idx, dir) in FACE_DIRS.iter().enumerate() {
                    let nx = x + dir[0];
                    let ny = y + dir[1];
                    let nz = z + dir[2];
                    if material_at(brick, nx, ny, nz) == 0 {
                        emit_face(&mut mesh, x, y, z, face_idx, m);
                    }
                }
            }
        }
    }
    mesh
}

fn emit_face(mesh: &mut Mesh, x: i32, y: i32, z: i32, face_idx: usize, material: u16) {
    let dir = FACE_DIRS[face_idx];
    let axis = if dir[0] != 0 {
        0
    } else if dir[1] != 0 {
        1
    } else {
        2
    };
    let positive = dir[axis] > 0;
    // Same handedness rule as greedy.rs: `u × v = +axis`.
    let (u_axis, v_axis) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let layer = if positive { axis_coord([x, y, z], axis) + 1 } else { axis_coord([x, y, z], axis) };

    let mut origin = [0f32; 3];
    origin[axis] = layer as f32;
    origin[u_axis] = axis_coord([x, y, z], u_axis) as f32;
    origin[v_axis] = axis_coord([x, y, z], v_axis) as f32;

    let mut u_vec = [0f32; 3];
    u_vec[u_axis] = 1.0;
    let mut v_vec = [0f32; 3];
    v_vec[v_axis] = 1.0;

    let mut normal = [0f32; 3];
    normal[axis] = if positive { 1.0 } else { -1.0 };

    let base = mesh.vertices.len() as u32;
    let p0 = origin;
    let p1 = [origin[0] + u_vec[0], origin[1] + u_vec[1], origin[2] + u_vec[2]];
    let p2 = [
        origin[0] + u_vec[0] + v_vec[0],
        origin[1] + u_vec[1] + v_vec[1],
        origin[2] + u_vec[2] + v_vec[2],
    ];
    let p3 = [origin[0] + v_vec[0], origin[1] + v_vec[1], origin[2] + v_vec[2]];
    for p in [p0, p1, p2, p3] {
        mesh.vertices.push(Vertex { pos: p, normal, material, ao: 1.0, sky_light: 1.0 });
    }
    if positive {
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        mesh.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

fn axis_coord(c: [i32; 3], axis: usize) -> i32 {
    c[axis]
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_voxel::Voxel;

    #[test]
    fn empty_brick_emits_nothing() {
        let b = Brick::new();
        let m = naive_mesh(&b);
        assert!(m.vertices.is_empty());
        assert!(m.indices.is_empty());
    }

    #[test]
    fn interior_single_voxel_emits_six_quads() {
        let mut b = Brick::new();
        b.set(IVec3::new(8, 8, 8), Voxel::new(1));
        let m = naive_mesh(&b);
        // Six quads → 24 verts, 36 indices, 12 triangles.
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
        assert_eq!(m.triangle_count(), 12);
    }

    #[test]
    fn solid_brick_emits_only_boundary_faces() {
        // Every voxel solid → only the 6 outer faces of the brick are
        // exposed (apron treated as empty). 16² quads per face × 6 = 1536.
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let m = naive_mesh(&b);
        let expected_quads = 16 * 16 * 6;
        assert_eq!(m.vertices.len(), expected_quads * 4);
        assert_eq!(m.indices.len(), expected_quads * 6);
    }

    #[test]
    fn material_attribution_comes_from_source_voxel() {
        let mut b = Brick::new();
        b.set(IVec3::new(4, 4, 4), Voxel::new(7));
        let m = naive_mesh(&b);
        assert!(m.vertices.iter().all(|v| v.material == 7));
    }
}
