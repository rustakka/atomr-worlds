//! Greedy meshing of a [`Brick`] into axis-aligned face quads.
//!
//! For each of the six face directions we sweep the brick layer by layer and
//! merge contiguous coplanar quads with the same material. The output is a
//! flat-shaded triangle mesh with per-vertex position + normal + material id.
//! Vertex count for a worst-case checkerboard is bounded by `3 * BRICK_LEN`
//! (one triangle per nonempty face per voxel); empty bricks produce zero
//! geometry.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub material: u16,
}

#[derive(Copy, Clone, Debug)]
pub struct Quad {
    pub origin: [f32; 3], // bottom-left corner in face's UV frame
    pub u: [f32; 3],
    pub v: [f32; 3],
    pub normal: [f32; 3],
    pub material: u16,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

const EDGE: usize = BRICK_EDGE;

/// Six face directions, indexed by axis (0=x, 1=y, 2=z) and sign (0=−, 1=+).
const FACE_DIRS: [[i32; 3]; 6] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];

fn material_at(brick: &Brick, x: i32, y: i32, z: i32) -> u16 {
    if x < 0 || y < 0 || z < 0 || x >= EDGE as i32 || y >= EDGE as i32 || z >= EDGE as i32 {
        return 0; // treat OOB as empty so brick boundaries emit faces
    }
    brick.get(IVec3::new(x as i64, y as i64, z as i64)).0
}

/// Convert a [`Brick`] to a flat-shaded triangle mesh in **local** brick
/// coordinates (each voxel occupies the unit cube `[lx, lx+1] × [ly, ly+1] ×
/// [lz, lz+1]`).
pub fn greedy_mesh(brick: &Brick) -> Mesh {
    let mut mesh = Mesh::default();
    for face in 0..6 {
        meshing_axis(brick, face, &mut mesh);
    }
    mesh
}

fn meshing_axis(brick: &Brick, face_idx: usize, mesh: &mut Mesh) {
    let dir = FACE_DIRS[face_idx];
    let axis = if dir[0] != 0 {
        0
    } else if dir[1] != 0 {
        1
    } else {
        2
    };
    let positive = dir[axis] > 0;

    // u, v are the in-plane axes (the two non-`axis` axes).
    let (u_axis, v_axis) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };

    for layer in 0..EDGE as i32 {
        // Build a (u, v) mask of the materials whose face points along `dir`
        // at this layer.
        let mut mask = vec![0u16; EDGE * EDGE];
        for vi in 0..EDGE as i32 {
            for ui in 0..EDGE as i32 {
                let mut coord = [0i32; 3];
                coord[axis] = layer;
                coord[u_axis] = ui;
                coord[v_axis] = vi;
                // `near` is the solid voxel that owns the face; `far` sits on
                // the empty side. For both signs, the face exists when `near
                // != 0 && far == 0`.
                let near = material_at(brick, coord[0], coord[1], coord[2]);
                let mut far_coord = coord;
                far_coord[axis] = layer + if positive { 1 } else { -1 };
                let far = material_at(brick, far_coord[0], far_coord[1], far_coord[2]);
                if near != 0 && far == 0 {
                    mask[(vi as usize) * EDGE + ui as usize] = near;
                }
            }
        }

        // Greedy merge of contiguous same-material runs in the mask.
        let mut vi = 0usize;
        while vi < EDGE {
            let mut ui = 0usize;
            while ui < EDGE {
                let m = mask[vi * EDGE + ui];
                if m == 0 {
                    ui += 1;
                    continue;
                }
                // Extend along u.
                let mut w = 1usize;
                while ui + w < EDGE && mask[vi * EDGE + ui + w] == m {
                    w += 1;
                }
                // Extend along v.
                let mut h = 1usize;
                'h: while vi + h < EDGE {
                    for k in 0..w {
                        if mask[(vi + h) * EDGE + ui + k] != m {
                            break 'h;
                        }
                    }
                    h += 1;
                }
                // Emit quad.
                emit_quad(mesh, axis, positive, layer, ui, vi, w, h, m, u_axis, v_axis);
                // Zero the consumed region.
                for j in 0..h {
                    for i in 0..w {
                        mask[(vi + j) * EDGE + ui + i] = 0;
                    }
                }
                ui += w;
            }
            vi += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    mesh: &mut Mesh,
    axis: usize,
    positive: bool,
    layer: i32,
    ui: usize,
    vi: usize,
    w: usize,
    h: usize,
    material: u16,
    u_axis: usize,
    v_axis: usize,
) {
    let layer_pos = if positive { (layer + 1) as f32 } else { layer as f32 };
    let mut origin = [0f32; 3];
    origin[axis] = layer_pos;
    origin[u_axis] = ui as f32;
    origin[v_axis] = vi as f32;

    let mut u_vec = [0f32; 3];
    u_vec[u_axis] = w as f32;
    let mut v_vec = [0f32; 3];
    v_vec[v_axis] = h as f32;

    let mut normal = [0f32; 3];
    normal[axis] = if positive { 1.0 } else { -1.0 };

    let base = mesh.vertices.len() as u32;
    let p0 = origin;
    let p1 = [origin[0] + u_vec[0], origin[1] + u_vec[1], origin[2] + u_vec[2]];
    let p2 =
        [origin[0] + u_vec[0] + v_vec[0], origin[1] + u_vec[1] + v_vec[1], origin[2] + u_vec[2] + v_vec[2]];
    let p3 = [origin[0] + v_vec[0], origin[1] + v_vec[1], origin[2] + v_vec[2]];
    for p in [p0, p1, p2, p3] {
        mesh.vertices.push(Vertex { pos: p, normal, material });
    }
    // Wind so the normal faces "outwards". Flip when the face points along the
    // negative axis so back-face culling stays consistent.
    if positive {
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        mesh.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_voxel::Voxel;

    #[test]
    fn empty_brick_meshes_to_nothing() {
        let b = Brick::new();
        let m = greedy_mesh(&b);
        assert!(m.vertices.is_empty());
        assert!(m.indices.is_empty());
    }

    #[test]
    fn single_voxel_has_six_quads() {
        let mut b = Brick::new();
        b.set(IVec3::new(5, 5, 5), Voxel::new(1));
        let m = greedy_mesh(&b);
        // Six faces → six quads → 24 vertices, 36 indices.
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
    }

    #[test]
    fn solid_brick_top_face_is_one_merged_quad() {
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let m = greedy_mesh(&b);
        // Six faces, each a 16×16 merged quad (4 vertices, 6 indices) → 24 verts, 36 indices.
        assert_eq!(m.vertices.len(), 24, "expected 6 merged quads, got {} verts", m.vertices.len());
        assert_eq!(m.indices.len(), 36);
    }

    #[test]
    fn checkerboard_does_not_merge() {
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    if (x + y + z) % 2 == 0 {
                        b.set(IVec3::new(x, y, z), Voxel::new(1));
                    }
                }
            }
        }
        let m = greedy_mesh(&b);
        assert!(!m.vertices.is_empty());
        // Each solid voxel exposes all 6 faces in a checkerboard pattern.
        // 16^3 / 2 = 2048 solid voxels × 6 faces × 4 verts = 49152.
        assert_eq!(m.vertices.len(), 2048 * 6 * 4);
    }
}
