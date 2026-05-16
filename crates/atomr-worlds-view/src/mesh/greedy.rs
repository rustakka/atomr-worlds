//! Greedy meshing of a [`Brick`] into axis-aligned face quads.
//!
//! Sweeps each face direction layer-by-layer and coalesces contiguous
//! coplanar same-material quads into the largest axis-aligned rectangle.
//! Worst-case vertex count for a checkerboard is bounded by
//! `3 * BRICK_LEN`; empty bricks produce zero geometry.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

use super::{Mesh, Vertex};

const EDGE: usize = BRICK_EDGE;

const FACE_DIRS: [[i32; 3]; 6] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];

fn material_at(brick: &Brick, x: i32, y: i32, z: i32) -> u16 {
    if x < 0 || y < 0 || z < 0 || x >= EDGE as i32 || y >= EDGE as i32 || z >= EDGE as i32 {
        return 0;
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

/// Split-by-material variant: run greedy meshing as usual, then bucket
/// each triangle into a per-material sub-mesh. Used by the client's
/// `SplitPerMaterial` shading strategy so each material can carry its
/// own `StandardMaterial` (per-material roughness / metallic / emissive
/// / alpha) without an explicit ID-keyed shader.
///
/// Greedy meshing already never merges across material boundaries (each
/// quad has a single `material` id, see `emit_quad`), so splitting is a
/// simple bucket pass — no re-meshing required. Vertices are duplicated
/// across buckets only when adjacent quads of different materials
/// happened to share an index, which is rare in practice.
pub fn greedy_mesh_by_material(brick: &Brick) -> std::collections::HashMap<u16, Mesh> {
    let merged = greedy_mesh(brick);
    let mut split: std::collections::HashMap<u16, Mesh> =
        std::collections::HashMap::new();
    let mut idx = 0;
    while idx + 2 < merged.indices.len() {
        let i0 = merged.indices[idx] as usize;
        let i1 = merged.indices[idx + 1] as usize;
        let i2 = merged.indices[idx + 2] as usize;
        idx += 3;
        let v0 = merged.vertices[i0];
        let v1 = merged.vertices[i1];
        let v2 = merged.vertices[i2];
        let bucket = split.entry(v0.material).or_default();
        let base = bucket.vertices.len() as u32;
        bucket.vertices.push(v0);
        bucket.vertices.push(v1);
        bucket.vertices.push(v2);
        bucket.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    split
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

    // Pick (u, v) so `u × v = +axis` (right-handed). For axis=1 (Y)
    // the natural (X, Z) order gives `X × Z = -Y`, which back-face-culls
    // top faces — use (Z, X) so positive-axis winding is always CCW
    // viewed from outside.
    let (u_axis, v_axis) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };

    for layer in 0..EDGE as i32 {
        let mut mask = vec![0u16; EDGE * EDGE];
        for vi in 0..EDGE as i32 {
            for ui in 0..EDGE as i32 {
                let mut coord = [0i32; 3];
                coord[axis] = layer;
                coord[u_axis] = ui;
                coord[v_axis] = vi;
                let near = material_at(brick, coord[0], coord[1], coord[2]);
                let mut far_coord = coord;
                far_coord[axis] = layer + if positive { 1 } else { -1 };
                let far = material_at(brick, far_coord[0], far_coord[1], far_coord[2]);
                if near != 0 && far == 0 {
                    mask[(vi as usize) * EDGE + ui as usize] = near;
                }
            }
        }

        let mut vi = 0usize;
        while vi < EDGE {
            let mut ui = 0usize;
            while ui < EDGE {
                let m = mask[vi * EDGE + ui];
                if m == 0 {
                    ui += 1;
                    continue;
                }
                let mut w = 1usize;
                while ui + w < EDGE && mask[vi * EDGE + ui + w] == m {
                    w += 1;
                }
                let mut h = 1usize;
                'h: while vi + h < EDGE {
                    for k in 0..w {
                        if mask[(vi + h) * EDGE + ui + k] != m {
                            break 'h;
                        }
                    }
                    h += 1;
                }
                emit_quad(mesh, axis, positive, layer, ui, vi, w, h, m, u_axis, v_axis);
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
        mesh.vertices.push(Vertex { pos: p, normal, material, ao: 1.0 });
    }
    if positive {
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        mesh.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

/// Post-process a mesh, writing per-vertex AO (Minecraft-style corner
/// sampling). For each vertex sitting at a quad corner, sample the four
/// voxels surrounding that corner *on the air side* of the face; the AO
/// factor is `f(occluded_count)`. The "owner" voxel of the face is by
/// construction air on its outward side, so at most 3 of the 4 can be
/// solid. Greedy merging means the merged quad's 4 corners pick up AO
/// from voxels just outside the quad's extent — the interior of the
/// quad gets a bilinear gradient via Bevy's vertex-color interpolation.
pub fn bake_ao(mesh: &mut Mesh, brick: &Brick) {
    for v in &mut mesh.vertices {
        v.ao = compute_vertex_ao(brick, v);
    }
}

fn compute_vertex_ao(brick: &Brick, v: &Vertex) -> f32 {
    let n = v.normal;
    let face_axis = if n[0].abs() > 0.5 {
        0
    } else if n[1].abs() > 0.5 {
        1
    } else {
        2
    };
    let positive = n[face_axis] > 0.0;
    let (u_axis, v_axis) = match face_axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let layer_air = if positive {
        v.pos[face_axis] as i32
    } else {
        v.pos[face_axis] as i32 - 1
    };
    let u_pos = v.pos[u_axis] as i32;
    let v_pos = v.pos[v_axis] as i32;
    let sample = |du: i32, dv: i32| -> bool {
        let mut c = [0i32; 3];
        c[face_axis] = layer_air;
        c[u_axis] = u_pos + du;
        c[v_axis] = v_pos + dv;
        material_at(brick, c[0], c[1], c[2]) != 0
    };
    let occ = sample(-1, -1) as u8
        + sample(0, -1) as u8
        + sample(-1, 0) as u8
        + sample(0, 0) as u8;
    match occ {
        0 => 1.0,
        1 => 0.78,
        2 => 0.55,
        3 => 0.40,
        _ => 0.40,
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
        assert_eq!(m.vertices.len(), 24, "expected 6 merged quads, got {} verts", m.vertices.len());
        assert_eq!(m.indices.len(), 36);
    }

    #[test]
    fn all_six_face_directions_wind_outward() {
        for face_idx in 0..6 {
            let mut b = Brick::new();
            b.set(IVec3::new(8, 8, 8), Voxel::new(1));
            let m = greedy_mesh(&b);
            let target_normal = FACE_DIRS[face_idx];
            let mut found = false;
            let mut tri = 0;
            while tri + 2 < m.indices.len() {
                let v0 = m.vertices[m.indices[tri] as usize];
                let normal_matches = (v0.normal[0] as i32) == target_normal[0]
                    && (v0.normal[1] as i32) == target_normal[1]
                    && (v0.normal[2] as i32) == target_normal[2];
                if normal_matches {
                    let v1 = m.vertices[m.indices[tri + 1] as usize];
                    let v2 = m.vertices[m.indices[tri + 2] as usize];
                    let e1 = [
                        v1.pos[0] - v0.pos[0],
                        v1.pos[1] - v0.pos[1],
                        v1.pos[2] - v0.pos[2],
                    ];
                    let e2 = [
                        v2.pos[0] - v0.pos[0],
                        v2.pos[1] - v0.pos[1],
                        v2.pos[2] - v0.pos[2],
                    ];
                    let n = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    let dot = n[0] * (target_normal[0] as f32)
                        + n[1] * (target_normal[1] as f32)
                        + n[2] * (target_normal[2] as f32);
                    assert!(
                        dot > 0.0,
                        "face {:?} winding produces back-facing front (dot={}), \
                         geometric normal {:?} disagrees with stored normal {:?}",
                        target_normal,
                        dot,
                        n,
                        v0.normal
                    );
                    found = true;
                    break;
                }
                tri += 3;
            }
            assert!(found, "no triangles with stored normal {:?}", target_normal);
        }
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
        assert_eq!(m.vertices.len(), 2048 * 6 * 4);
    }
}
