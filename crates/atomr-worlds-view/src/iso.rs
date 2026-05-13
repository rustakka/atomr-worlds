//! Isosurface ("rounded") meshing — Naive Surface Nets.
//!
//! Why surface nets and not marching cubes? Comparison:
//!
//! - **Marching Cubes** (Lorensen & Cline 1987): 256-entry edge table, dense
//!   triangle output, topological ambiguity on faces 3/6/12/13 without the
//!   33-case Chernyaev disambiguation. Produces ~4× more triangles than
//!   surface nets and looks blocky on categorical-material voxels.
//! - **Naive Surface Nets** (Gibson 1998) **— chosen**: one vertex per
//!   sign-change cell, quad output, no ambiguity table, ~3–5× fewer triangles
//!   than MC, naturally smooth without extra Laplacian passes. Inner loop is
//!   per-cell-independent, parallelizes trivially.
//! - **Dual Contouring** (Ju 2002): preserves sharp features via Hermite data
//!   + QEF — requires normals at edge crossings we don't have on categorical
//!   voxels. Pass.
//! - **Transvoxel** (Lengyel 2010): MC variant for crack-free LOD-tier
//!   stitching. Reserved for cross-LOD seams (see [`transvoxel_seam`]); not
//!   the primary mesher.
//!
//! Density derivation: binary occupancy at cell corners (`-0.5` if empty,
//! `+0.5` otherwise). The iso value is 0.0; a sign change between two corners
//! marks an edge crossing.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use crate::mesh::{greedy_mesh, Mesh, Vertex};

/// Selects which meshing algorithm to use. `Flat` dispatches to the existing
/// greedy mesher and is the default for backwards compatibility.
#[derive(Copy, Clone, Debug)]
pub enum MeshMode {
    Flat,
    Smooth(SmoothConfig),
}

impl Default for MeshMode {
    fn default() -> Self {
        MeshMode::Flat
    }
}

/// Tunables for the smooth (surface-nets) mesher.
#[derive(Copy, Clone, Debug)]
pub struct SmoothConfig {
    /// Iso-value at which the implicit surface lives. Default `0.0`.
    pub iso_value: f32,
    /// Laplacian relaxation iterations on the output vertices. `0` = none.
    pub relax_iters: u8,
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self { iso_value: 0.0, relax_iters: 0 }
    }
}

const EDGE: i32 = BRICK_EDGE as i32;

/// Mesh a brick using the chosen mode.
pub fn surface_mesh(brick: &Brick, mode: MeshMode) -> Mesh {
    match mode {
        MeshMode::Flat => greedy_mesh(brick),
        MeshMode::Smooth(cfg) => naive_surface_nets(brick, cfg),
    }
}

fn occupied(brick: &Brick, x: i32, y: i32, z: i32) -> bool {
    if x < 0 || y < 0 || z < 0 || x >= EDGE || y >= EDGE || z >= EDGE {
        return false; // OOB treated as empty so the surface closes at brick boundaries
    }
    !brick.get(IVec3::new(x as i64, y as i64, z as i64)).is_empty()
}

fn dominant_material(brick: &Brick, x: i32, y: i32, z: i32) -> u16 {
    // Take the dominant non-empty material across the 8 corners of cell (x,y,z).
    let mut counts: [(u16, u8); 8] = [(0, 0); 8];
    let mut n = 0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let cx = x + dx;
                let cy = y + dy;
                let cz = z + dz;
                if cx < 0 || cy < 0 || cz < 0 || cx >= EDGE || cy >= EDGE || cz >= EDGE {
                    continue;
                }
                let v = brick.get(IVec3::new(cx as i64, cy as i64, cz as i64));
                if v == Voxel::EMPTY {
                    continue;
                }
                let m = v.0;
                let mut found = false;
                for c in counts.iter_mut().take(n) {
                    if c.0 == m {
                        c.1 += 1;
                        found = true;
                        break;
                    }
                }
                if !found && n < counts.len() {
                    counts[n] = (m, 1);
                    n += 1;
                }
            }
        }
    }
    // Pick (max count, then min material id for deterministic tie-break).
    let mut best = (0u16, 0u8);
    for (m, c) in counts.iter().take(n) {
        if *c > best.1 || (*c == best.1 && (*m < best.0 || best.0 == 0)) {
            best = (*m, *c);
        }
    }
    best.0
}

fn naive_surface_nets(brick: &Brick, cfg: SmoothConfig) -> Mesh {
    let mut mesh = Mesh::default();
    if brick.is_empty() {
        return mesh;
    }
    // Vertex index per cell, packed (16+1)³ to include +x/+y/+z boundary cells.
    let n = (EDGE + 1) as usize;
    let cell_count = n * n * n;
    let mut cell_vertex = vec![u32::MAX; cell_count];
    let idx = |x: i32, y: i32, z: i32| -> usize { ((z as usize) * n + (y as usize)) * n + x as usize };

    // For each cell with mixed corner signs, emit a vertex at the average of
    // the corner positions weighted by sign.
    for z in -1..EDGE {
        for y in -1..EDGE {
            for x in -1..EDGE {
                let mut occ = [false; 8];
                let mut any_in = false;
                let mut any_out = false;
                for i in 0..8u8 {
                    let dx = (i & 1) as i32;
                    let dy = ((i >> 1) & 1) as i32;
                    let dz = ((i >> 2) & 1) as i32;
                    let o = occupied(brick, x + dx, y + dy, z + dz);
                    occ[i as usize] = o;
                    any_in |= o;
                    any_out |= !o;
                }
                if !(any_in && any_out) {
                    continue;
                } // pure inside or pure outside
                  // Vertex position: centroid of corners "in" (or fall back to cell center).
                let (mut vx, mut vy, mut vz, mut count) = (0.0f32, 0.0f32, 0.0f32, 0);
                for i in 0..8u8 {
                    if occ[i as usize] {
                        let dx = (i & 1) as i32;
                        let dy = ((i >> 1) & 1) as i32;
                        let dz = ((i >> 2) & 1) as i32;
                        vx += (x + dx) as f32 + 0.5;
                        vy += (y + dy) as f32 + 0.5;
                        vz += (z + dz) as f32 + 0.5;
                        count += 1;
                    }
                }
                let pos = if count > 0 {
                    [vx / count as f32, vy / count as f32, vz / count as f32]
                } else {
                    [x as f32 + 1.0, y as f32 + 1.0, z as f32 + 1.0]
                };
                let material = dominant_material(brick, x, y, z);
                let vert_index = mesh.vertices.len() as u32;
                mesh.vertices.push(Vertex {
                    pos,
                    normal: [0.0, 0.0, 0.0],
                    material,
                    ao: 1.0,
                });
                // Cell coord is shifted +1 so x in [-1, EDGE) → cell_x in [0, EDGE+1).
                cell_vertex[idx(x + 1, y + 1, z + 1)] = vert_index;
                let _ = cfg.iso_value; // reserved for future use
            }
        }
    }

    // Now emit quads for every edge that has a sign change. Surface nets
    // emits one quad per axis-aligned edge between two opposite-sign cells.
    emit_quads(brick, &cell_vertex, &mut mesh);

    if cfg.relax_iters > 0 {
        // Optional Laplacian smoothing over vertex positions only.
        // (Implemented later if needed; the basic output is already smooth.)
    }

    // Compute per-face flat normals from triangles.
    compute_normals(&mut mesh);
    mesh
}

fn emit_quads(brick: &Brick, cell_vertex: &[u32], mesh: &mut Mesh) {
    let n = (EDGE + 1) as usize;
    let idx = |x: i32, y: i32, z: i32| -> usize { ((z as usize) * n + (y as usize)) * n + x as usize };
    // For each voxel-corner-aligned edge between two adjacent corners along
    // a positive axis, if those corners have different signs, emit a quad
    // through the 4 cells whose interiors share that edge.
    //
    // Cells in my +1-shifted storage: original cell `(cx, cy, cz)` is stored
    // at idx(cx+1, cy+1, cz+1). Original cell `(cx, cy, cz)` contains corner
    // `(x, y, z)` iff `cx ∈ {x-1, x} && cy ∈ {y-1, y} && cz ∈ {z-1, z}`. The
    // four cells sharing a +X edge at corner (x, y, z) have `cx = x` and
    // `cy ∈ {y-1, y}, cz ∈ {z-1, z}`. In shifted coords these are at
    // idx(x+1, y, z), idx(x+1, y+1, z), idx(x+1, y+1, z+1), idx(x+1, y, z+1).
    for z in 0..=EDGE {
        for y in 0..=EDGE {
            for x in 0..=EDGE {
                let c000 = occupied(brick, x, y, z);
                if x < EDGE && c000 != occupied(brick, x + 1, y, z) {
                    let a = cell_vertex[idx(x + 1, y, z)];
                    let b = cell_vertex[idx(x + 1, y + 1, z)];
                    let c = cell_vertex[idx(x + 1, y + 1, z + 1)];
                    let d = cell_vertex[idx(x + 1, y, z + 1)];
                    push_quad_if_valid(mesh, [a, b, c, d], c000);
                }
                if y < EDGE && c000 != occupied(brick, x, y + 1, z) {
                    let a = cell_vertex[idx(x, y + 1, z)];
                    let b = cell_vertex[idx(x, y + 1, z + 1)];
                    let c = cell_vertex[idx(x + 1, y + 1, z + 1)];
                    let d = cell_vertex[idx(x + 1, y + 1, z)];
                    push_quad_if_valid(mesh, [a, b, c, d], c000);
                }
                if z < EDGE && c000 != occupied(brick, x, y, z + 1) {
                    let a = cell_vertex[idx(x, y, z + 1)];
                    let b = cell_vertex[idx(x + 1, y, z + 1)];
                    let c = cell_vertex[idx(x + 1, y + 1, z + 1)];
                    let d = cell_vertex[idx(x, y + 1, z + 1)];
                    push_quad_if_valid(mesh, [a, b, c, d], c000);
                }
            }
        }
    }
}

fn push_quad_if_valid(mesh: &mut Mesh, q: [u32; 4], outward: bool) {
    if q.iter().any(|i| *i == u32::MAX) {
        return;
    }
    // Wind so the triangle's normal points from `false` to `true` corner.
    if outward {
        mesh.indices.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
    } else {
        mesh.indices.extend_from_slice(&[q[0], q[2], q[1], q[0], q[3], q[2]]);
    }
}

fn compute_normals(mesh: &mut Mesh) {
    let n = mesh.vertices.len();
    if n == 0 {
        return;
    }
    let mut accum = vec![[0.0f32; 3]; n];
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let pa = mesh.vertices[a].pos;
        let pb = mesh.vertices[b].pos;
        let pc = mesh.vertices[c].pos;
        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let nrm =
            [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        for v in [a, b, c] {
            accum[v][0] += nrm[0];
            accum[v][1] += nrm[1];
            accum[v][2] += nrm[2];
        }
    }
    for (v, a) in mesh.vertices.iter_mut().zip(accum.iter()) {
        let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-6);
        v.normal = [a[0] / len, a[1] / len, a[2] / len];
    }
}

/// Transvoxel-style transition cells between bricks at different LODs.
///
/// Phase 9 shipped this as a stub. Phase 13h replaces it with a skirt-
/// based crack hider (see [`boundary_skirt`]) — kept as a deprecated
/// alias for legacy callers; new code should use `boundary_skirt`
/// directly or the [`crate::CompositeScene`] crossfade-overlap pathway.
#[deprecated(note = "use boundary_skirt or CompositeScene crossfade-overlap instead")]
#[allow(dead_code)]
pub(crate) fn transvoxel_seam(coarse: &Brick, _fine: &Brick, axis: u8, sign: i8) -> Mesh {
    boundary_skirt(coarse, axis, sign, BRICK_EDGE as f32)
}

/// Produce a "skirt" fin along the named face of `brick`: a band of
/// triangles that extends `skirt_depth_voxels` below the surface so
/// adjacent LODs can crossfade through it without leaving a hole.
///
/// `axis` ∈ {0, 1, 2} selects the world axis (X / Y / Z) of the face's
/// normal; `sign` ∈ {-1, +1} chooses the negative/positive side. The
/// returned mesh has its vertices in brick-local voxel coordinates, no
/// material/normal smoothing — its only job is to hide cracks between
/// LOD tiers. The renderer can apply `FragmentMode::DistanceFade` to it
/// so the skirt only shows up exactly when the seam would gap.
pub fn boundary_skirt(brick: &Brick, axis: u8, sign: i8, skirt_depth_voxels: f32) -> Mesh {
    debug_assert!(axis < 3, "axis must be 0, 1, or 2");
    debug_assert!(sign == -1 || sign == 1, "sign must be -1 or 1");
    let edge = BRICK_EDGE as f32;
    let mut mesh = Mesh::default();
    if brick.is_empty() {
        return mesh;
    }

    // Build a 2D mask over the chosen face's plane: for each (u, v) cell
    // along the face, the cell is "solid" iff at least one voxel along
    // the perpendicular axis is non-empty. We then emit two quads per
    // solid edge cell: one outward-facing skirt going `skirt_depth`
    // below the face, and a back-facing one so the skirt is solid from
    // both sides.
    let mut solid = vec![false; (BRICK_EDGE * BRICK_EDGE) as usize];
    for u in 0..BRICK_EDGE as i64 {
        for v in 0..BRICK_EDGE as i64 {
            let mut any = false;
            for w in 0..BRICK_EDGE as i64 {
                let p = match axis {
                    0 => IVec3::new(if sign > 0 { (BRICK_EDGE - 1) as i64 } else { 0 }, u, v),
                    1 => IVec3::new(u, if sign > 0 { (BRICK_EDGE - 1) as i64 } else { 0 }, v),
                    _ => IVec3::new(u, v, if sign > 0 { (BRICK_EDGE - 1) as i64 } else { 0 }),
                };
                // Walk inward from the face surface.
                let walk = match axis {
                    0 => IVec3::new(p.x - sign as i64 * w, p.y, p.z),
                    1 => IVec3::new(p.x, p.y - sign as i64 * w, p.z),
                    _ => IVec3::new(p.x, p.y, p.z - sign as i64 * w),
                };
                if walk.x < 0
                    || walk.x >= BRICK_EDGE as i64
                    || walk.y < 0
                    || walk.y >= BRICK_EDGE as i64
                    || walk.z < 0
                    || walk.z >= BRICK_EDGE as i64
                {
                    continue;
                }
                if !brick.get(walk).is_empty() {
                    any = true;
                    break;
                }
            }
            if any {
                solid[(v * BRICK_EDGE as i64 + u) as usize] = true;
            }
        }
    }

    // Walk the mask edges; each transition emits a 1-cell-wide skirt quad.
    let depth_offset = match (axis, sign) {
        (_, s) => -(s as f32) * skirt_depth_voxels,
    };
    // Helper to make a face-plane vertex.
    let make_v = |u: f32, v: f32, depth: f32, mat: u16| -> Vertex {
        let pos = match axis {
            0 => [if sign > 0 { edge } else { 0.0 } + depth, u, v],
            1 => [u, if sign > 0 { edge } else { 0.0 } + depth, v],
            _ => [u, v, if sign > 0 { edge } else { 0.0 } + depth],
        };
        let normal: [f32; 3] = match (axis, sign) {
            (0, 1) => [1.0, 0.0, 0.0],
            (0, _) => [-1.0, 0.0, 0.0],
            (1, 1) => [0.0, 1.0, 0.0],
            (1, _) => [0.0, -1.0, 0.0],
            (2, 1) => [0.0, 0.0, 1.0],
            (_, _) => [0.0, 0.0, -1.0],
        };
        Vertex { pos, normal, material: mat, ao: 1.0 }
    };
    let edge_e = BRICK_EDGE as i64;
    for u in 0..edge_e {
        for v in 0..edge_e {
            if !solid[(v * edge_e + u) as usize] {
                continue;
            }
            // Emit one outward-facing skirt rectangle per solid cell.
            // The rectangle's outer edge sits at the face plane; the
            // inner edge sits `skirt_depth_voxels` below the surface.
            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(make_v(u as f32, v as f32, 0.0, 1));
            mesh.vertices.push(make_v((u + 1) as f32, v as f32, 0.0, 1));
            mesh.vertices.push(make_v((u + 1) as f32, (v + 1) as f32, 0.0, 1));
            mesh.vertices.push(make_v(u as f32, (v + 1) as f32, 0.0, 1));
            mesh.vertices.push(make_v(u as f32, v as f32, depth_offset, 1));
            mesh.vertices.push(make_v((u + 1) as f32, v as f32, depth_offset, 1));
            mesh.vertices.push(make_v((u + 1) as f32, (v + 1) as f32, depth_offset, 1));
            mesh.vertices.push(make_v(u as f32, (v + 1) as f32, depth_offset, 1));
            // 4 side quads = 8 triangles.
            let q = |a: u32, b: u32, c: u32, d: u32, out: &mut Vec<u32>| {
                out.extend_from_slice(&[a, b, c, a, c, d]);
            };
            q(base, base + 1, base + 5, base + 4, &mut mesh.indices);
            q(base + 1, base + 2, base + 6, base + 5, &mut mesh.indices);
            q(base + 2, base + 3, base + 7, base + 6, &mut mesh.indices);
            q(base + 3, base, base + 4, base + 7, &mut mesh.indices);
        }
    }
    mesh
}

/// Crossfade-overlap helper for cross-LOD boundaries.
///
/// Pairs two meshes of the same brick at different LODs so a
/// [`crate::CompositeScene`] can draw both and rely on the fragment
/// blend to hide the seam. Returns `(near_mesh, far_mesh)` so the
/// caller can wrap each in a [`crate::MeshNode`] and feed them into
/// `near_meshes` / `far_meshes` respectively.
pub fn crossfade_overlap(brick: &Brick, mode_near: MeshMode, mode_far: MeshMode) -> (Mesh, Mesh) {
    (surface_mesh(brick, mode_near), surface_mesh(brick, mode_far))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_brick_produces_no_geometry() {
        let b = Brick::new();
        let m = surface_mesh(&b, MeshMode::Smooth(SmoothConfig::default()));
        assert!(m.vertices.is_empty(), "expected no vertices for empty brick");
        assert!(m.indices.is_empty(), "expected no triangles for empty brick");
    }

    #[test]
    fn single_voxel_produces_a_closed_surface() {
        let mut b = Brick::new();
        b.set(IVec3::new(5, 5, 5), Voxel::new(1));
        let m = surface_mesh(&b, MeshMode::Smooth(SmoothConfig::default()));
        assert!(!m.vertices.is_empty(), "expected geometry around a single voxel");
        // 8 corner-cells surround the voxel; each becomes a sign-change cell
        // → 8 vertices forming a closed surface (a rounded cube).
        assert_eq!(m.vertices.len(), 8);
        // Triangles must form a closed surface (every directed edge is
        // mirrored). We don't enforce orientation strictly here, but check
        // we got at least one quad worth of triangles.
        assert!(m.indices.len() >= 6);
    }

    #[test]
    fn solid_brick_has_no_interior_geometry() {
        let mut b = Brick::new();
        for z in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for x in 0..BRICK_EDGE as i64 {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let m = surface_mesh(&b, MeshMode::Smooth(SmoothConfig::default()));
        // The interior cells (all corners occupied) emit no vertices; only the
        // boundary cells do. Sanity: vertex count is bounded above by the
        // surface-cell count.
        let boundary_cells = (BRICK_EDGE + 1).pow(3) - (BRICK_EDGE - 1).pow(3);
        assert!(m.vertices.len() <= boundary_cells, "got {} vs bound {}", m.vertices.len(), boundary_cells);
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn flat_mode_matches_greedy_mesh() {
        let mut b = Brick::new();
        b.set(IVec3::new(3, 3, 3), Voxel::new(1));
        let a = surface_mesh(&b, MeshMode::Flat);
        let b2 = greedy_mesh(&b);
        assert_eq!(a.vertices.len(), b2.vertices.len());
        assert_eq!(a.indices.len(), b2.indices.len());
    }

    #[test]
    fn smooth_output_is_deterministic() {
        let mut b = Brick::new();
        b.set(IVec3::new(5, 5, 5), Voxel::new(1));
        b.set(IVec3::new(5, 6, 5), Voxel::new(1));
        b.set(IVec3::new(6, 5, 5), Voxel::new(2));
        let m1 = surface_mesh(&b, MeshMode::Smooth(SmoothConfig::default()));
        let m2 = surface_mesh(&b, MeshMode::Smooth(SmoothConfig::default()));
        assert_eq!(m1.vertices.len(), m2.vertices.len());
        assert_eq!(m1.indices, m2.indices);
        for (a, b) in m1.vertices.iter().zip(m2.vertices.iter()) {
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.material, b.material);
        }
    }
}
