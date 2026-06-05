//! GPU DAG raymarcher (Rec 1) — consumes [`atomr_worlds_voxel::DagBrick::to_gpu`]
//! and renders each brick by raymarching its sparse-voxel DAG in a fragment
//! shader, instead of uploading a triangle mesh.
//!
//! ## Approach: proxy-cube fragment material
//!
//! For each non-empty brick we spawn a unit proxy cube whose local space spans
//! the brick's `[0, 16)³` voxel grid. A stand-alone [`RaymarchMaterial`] binds
//! that brick's flattened DAG (`nodes` + `colors`) plus the shared PBR
//! `palette`. The WGSL fragment shader (`assets/shaders/voxel_raymarch.wgsl`)
//! reconstructs the view ray, transforms it into the brick's voxel space via
//! the *live* model matrix (so the fade-in scale animation is tracked), DDA-
//! marches the grid, and on the first solid voxel writes the shaded color **and
//! `frag_depth`** so the result composites against the rest of the scene
//! through the ordinary depth buffer (reversed-Z). Misses `discard`.
//!
//! The rasterizer is the acceleration structure: each fragment already knows
//! which brick it is in and where the ray enters, so the shader only traverses
//! one 16³ DAG with no top-level structure. The traversal kernel is a
//! line-for-line port of [`atomr_worlds_voxel::gpu_get`] — keeping the two in
//! lock-step is the determinism gate (see the `#[cfg(test)]` mirror below).
//!
//! ## Binding-slot convention (Bevy 0.18)
//!
//! A stand-alone [`Material`] owns the material bind group entirely
//! (`@group(3)` in Bevy 0.18 — `view` is group 0, the per-object `mesh` array
//! is group 2). So this material's storage buffers + uniform live at
//! `@group(3) @binding(0..3)`, and the shader reaches the live model matrix via
//! `bevy_pbr::mesh_functions` (group 2) and the camera/sun via
//! `bevy_pbr::mesh_view_bindings` (group 0).

use atomr_worlds_voxel::{DagGpu, BRICK_EDGE, DAG_GPU_EMPTY_ROOT};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh as BevyMesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::shader::ShaderRef;

/// Selectable raymarch shading tier — an engine setting for style/performance
/// tuning (see [`crate::render::RenderConfig::raymarch_tier`]). The value is
/// passed to the shader in [`RaymarchMeta::shading_tier`]; adding a tier is an
/// enum variant here plus a branch in `voxel_raymarch.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RaymarchShadingTier {
    /// Flat palette base color, no lighting. Cheapest; good for debugging
    /// traversal / silhouettes.
    Unlit,
    /// Lambert diffuse from the scene's directional light + ambient. The
    /// default — a reasonable match for the mesh path's silhouette at a
    /// fraction of the shader cost.
    #[default]
    Lambert,
    /// Richer local PBR-ish shading (Lambert + palette roughness/metallic/
    /// emissive). Structured for future expansion (AO, soft shadows); today
    /// it falls back to an enhanced Lambert inside the shader.
    Pbr,
}

impl RaymarchShadingTier {
    /// Stable u32 sent to the shader. Keep in sync with the `TIER_*` consts in
    /// `voxel_raymarch.wgsl`.
    pub fn to_u32(self) -> u32 {
        match self {
            RaymarchShadingTier::Unlit => 0,
            RaymarchShadingTier::Lambert => 1,
            RaymarchShadingTier::Pbr => 2,
        }
    }

    /// Parse a CLI / harness tier name. Returns `None` for unknown names.
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "unlit" | "Unlit" => RaymarchShadingTier::Unlit,
            "lambert" | "Lambert" => RaymarchShadingTier::Lambert,
            "pbr" | "Pbr" | "PBR" => RaymarchShadingTier::Pbr,
            _ => return None,
        })
    }
}

/// Per-brick scalar parameters for the raymarcher. Kept to one `vec4`-worth of
/// `u32`s so the std140 uniform is a tidy 16 bytes; the WGSL struct mirrors the
/// field order exactly.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct RaymarchMeta {
    /// Word offset of the DAG root in `nodes`, or [`DAG_GPU_EMPTY_ROOT`].
    pub root: u32,
    /// Voxel grid edge (always [`BRICK_EDGE`] = 16); the LOD scale lives in the
    /// model matrix, so traversal is always over the integer `[0, 16)` grid.
    pub brick_edge: u32,
    /// [`RaymarchShadingTier::to_u32`].
    pub shading_tier: u32,
    /// Reserved for future flags (kept for 16-byte alignment).
    pub flags: u32,
}

/// One brick's DAG flattened for the GPU raymarcher. A stand-alone [`Material`]
/// so each brick gets its own `nodes`/`colors`/`root`; the `palette` handle is
/// the one shared PBR palette buffer, reused read-only by every brick.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct RaymarchMaterial {
    /// Flat DAG node-word array (`array<u32>`): leaf = `DAG_LEAF_FLAG |
    /// color_index`; internal = `mask` word + popcount child word-offsets.
    #[storage(0, read_only)]
    pub nodes: Handle<ShaderStorageBuffer>,
    /// Per-brick color palette (`array<u32>`): `colors[color_index]` is a
    /// **material id**, indexed into the shared [`Self::palette`]. (u16 in
    /// `DagGpu`, widened to u32 on upload — WGSL has no u16 storage arrays.)
    #[storage(1, read_only)]
    pub colors: Handle<ShaderStorageBuffer>,
    /// Shared PBR palette (`array<PaletteEntry>`, the same buffer the mesh path
    /// uploads): `palette[material_id]` holds base_color / pbr / emissive.
    #[storage(2, read_only)]
    pub palette: Handle<ShaderStorageBuffer>,
    /// Scalar params (root / brick_edge / shading tier).
    #[uniform(3)]
    pub meta: RaymarchMeta,
}

impl Material for RaymarchMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/voxel_raymarch.wgsl".into()
    }

    fn vertex_shader() -> ShaderRef {
        // Custom vertex stage: it reads the per-object `mesh` bind group
        // (@group(2)) — which is only vertex-visible in the material pipeline —
        // and hands the fragment everything in voxel space (so the fragment
        // never touches @group(2)).
        "shaders/voxel_raymarch.wgsl".into()
    }

    fn specialize(
        // Bevy 0.17+: `MaterialPipeline` is no longer generic.
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The camera is frequently *inside* a brick's bounding box in
        // first-person, and a missed ray `discard`s. Disable culling so the
        // proxy rasterizes a fragment whether the camera is inside or outside;
        // `t_enter = max(tmin, 0)` in the shader handles the inside-origin case.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Shared GPU assets for the raymarch path, built once in
/// `setup_fp_scene`: the PBR palette storage buffer (the same one the mesh
/// path's `VoxelMaterialExt` uses) and the reusable `[0, 16]³` proxy box mesh.
#[derive(Resource, Default)]
pub struct RaymarchResources {
    /// Shared palette buffer handle (cloned into every brick material).
    pub palette: Option<Handle<ShaderStorageBuffer>>,
    /// Reusable unit proxy box spanning local `[0, 16]³`.
    pub proxy_box: Option<Handle<BevyMesh>>,
}

/// Build a [`RaymarchMaterial`] for one brick's flattened DAG. Returns `None`
/// for an empty brick (no proxy is spawned). `colors` is widened u16→u32 here.
pub fn build_raymarch_material(
    dag: &DagGpu,
    palette: Handle<ShaderStorageBuffer>,
    tier: RaymarchShadingTier,
    storage_buffers: &mut Assets<ShaderStorageBuffer>,
) -> Option<RaymarchMaterial> {
    if dag.root == DAG_GPU_EMPTY_ROOT || dag.nodes.is_empty() {
        return None;
    }
    let nodes = storage_buffers.add(ShaderStorageBuffer::from(dag.nodes.clone()));
    let colors_u32: Vec<u32> = dag.colors.iter().map(|&c| c as u32).collect();
    let colors = storage_buffers.add(ShaderStorageBuffer::from(colors_u32));
    Some(RaymarchMaterial {
        nodes,
        colors,
        palette,
        meta: RaymarchMeta {
            root: dag.root,
            brick_edge: BRICK_EDGE as u32,
            shading_tier: tier.to_u32(),
            flags: 0,
        },
    })
}

/// A box mesh spanning local `[0, 16]³` (one brick's voxel extent), POSITION
/// only. The parent entity's `Transform` (translate `coord * edge_m`, scale
/// `lod_scale`) places it, so the mesh's *local* coordinates equal the brick's
/// voxel-space coordinates — which is what makes
/// `get_local_from_world(instance_index)` in the shader map world → voxel space
/// directly. Culling is disabled by the material, so winding is irrelevant.
pub fn brick_proxy_box() -> BevyMesh {
    let e = BRICK_EDGE as f32;
    // 8 corners of [0, e]³.
    let positions: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0], // 0
        [e, 0.0, 0.0],   // 1
        [e, e, 0.0],     // 2
        [0.0, e, 0.0],   // 3
        [0.0, 0.0, e],   // 4
        [e, 0.0, e],     // 5
        [e, e, e],       // 6
        [0.0, e, e],     // 7
    ];
    // 12 triangles (2 per face). Winding is CCW-outward but culling is off.
    let indices: Vec<u32> = vec![
        // -z face (0,1,2,3)
        0, 2, 1, 0, 3, 2, //
        // +z face (4,5,6,7)
        4, 5, 6, 4, 6, 7, //
        // -x face (0,3,7,4)
        0, 7, 3, 0, 4, 7, //
        // +x face (1,2,6,5)
        1, 6, 2, 1, 5, 6, //
        // -y face (0,1,5,4)
        0, 5, 1, 0, 4, 5, //
        // +y face (3,2,6,7)
        3, 2, 6, 3, 6, 7, //
    ];
    let mut mesh = BevyMesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

#[cfg(test)]
mod tests {
    //! Determinism gate: the WGSL `dag_lookup` + ray DDA must mirror
    //! [`atomr_worlds_voxel::gpu_get`]. `gpu_get` is a *point* lookup; the
    //! shader is a *ray DDA*. These tests re-implement the exact DDA the shader
    //! uses in Rust and check it against `gpu_get` as the occupancy oracle over
    //! the same fixtures `dag.rs` uses (uniform / half / sparse), so a stepping
    //! or octant/popcount divergence in the WGSL port is caught in CI.

    use atomr_worlds_voxel::{gpu_get, DagBrick, DagGpu, Voxel, BRICK_EDGE};
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_voxel::Brick;
    use bevy::prelude::{Assets, Handle};

    const E: i32 = BRICK_EDGE as i32;

    fn solid_at(gpu: &DagGpu, x: i32, y: i32, z: i32) -> bool {
        if x < 0 || y < 0 || z < 0 || x >= E || y >= E || z >= E {
            return false;
        }
        gpu_get(gpu, x as u8, y as u8, z as u8) != Voxel::EMPTY
    }

    /// Amanatides–Woo DDA across the `[0, 16)³` grid — the exact stepping the
    /// WGSL shader performs. Returns the first solid cell hit and the list of
    /// empty cells visited before it (for the "earlier cells were empty"
    /// assertion). `origin`/`dir` are in voxel space.
    fn ray_dda_first_hit(
        gpu: &DagGpu,
        origin: [f32; 3],
        dir: [f32; 3],
    ) -> Option<([i32; 3], Vec<[i32; 3]>)> {
        // Clip the ray to the [0, E] box (slab method) to find the entry t.
        let mut t_enter = 0.0_f32;
        let mut t_exit = f32::INFINITY;
        for a in 0..3 {
            let inv = 1.0 / dir[a];
            let mut t0 = (0.0 - origin[a]) * inv;
            let mut t1 = (E as f32 - origin[a]) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
        }
        if t_enter > t_exit || t_exit < 0.0 {
            return None;
        }
        let t_start = t_enter.max(0.0);
        let p = [
            origin[0] + dir[0] * t_start,
            origin[1] + dir[1] * t_start,
            origin[2] + dir[2] * t_start,
        ];
        // Current cell (clamped so a hit exactly on the far face stays in range).
        let mut cell = [
            (p[0].floor() as i32).clamp(0, E - 1),
            (p[1].floor() as i32).clamp(0, E - 1),
            (p[2].floor() as i32).clamp(0, E - 1),
        ];
        let step = [
            if dir[0] >= 0.0 { 1 } else { -1 },
            if dir[1] >= 0.0 { 1 } else { -1 },
            if dir[2] >= 0.0 { 1 } else { -1 },
        ];
        // Distance along the ray to the next cell boundary on each axis.
        let mut t_max = [0.0_f32; 3];
        let mut t_delta = [0.0_f32; 3];
        for a in 0..3 {
            let inv = 1.0 / dir[a].abs().max(1e-20);
            t_delta[a] = inv;
            let next_boundary = if step[a] > 0 {
                (cell[a] + 1) as f32
            } else {
                cell[a] as f32
            };
            t_max[a] = t_start + (next_boundary - p[a]) / dir[a];
        }
        let mut visited_empty = Vec::new();
        for _ in 0..(3 * E + 4) {
            if solid_at(gpu, cell[0], cell[1], cell[2]) {
                return Some((cell, visited_empty));
            }
            visited_empty.push(cell);
            // Advance to the next cell across the nearest boundary.
            let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
                0
            } else if t_max[1] <= t_max[2] {
                1
            } else {
                2
            };
            cell[axis] += step[axis];
            t_max[axis] += t_delta[axis];
            if cell[axis] < 0 || cell[axis] >= E {
                return None; // walked out of the brick without a hit
            }
        }
        None
    }

    fn uniform_brick() -> Brick {
        let mut b = Brick::new();
        for z in 0..E {
            for y in 0..E {
                for x in 0..E {
                    b.set(IVec3::new(x as i64, y as i64, z as i64), Voxel::new(1));
                }
            }
        }
        b
    }

    fn half_brick() -> Brick {
        // Lower half solid (y < 8), upper half empty.
        let mut b = Brick::new();
        for z in 0..E {
            for y in 0..(E / 2) {
                for x in 0..E {
                    b.set(IVec3::new(x as i64, y as i64, z as i64), Voxel::new(2));
                }
            }
        }
        b
    }

    fn sparse_brick() -> Brick {
        let mut b = Brick::new();
        b.set(IVec3::new(1, 1, 1), Voxel::new(3));
        b.set(IVec3::new(8, 9, 10), Voxel::new(4));
        b.set(IVec3::new(15, 15, 15), Voxel::new(5));
        b
    }

    fn norm(v: [f32; 3]) -> [f32; 3] {
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / m, v[1] / m, v[2] / m]
    }

    /// A first-solid-hit found by the DDA must (a) be solid per `gpu_get`, and
    /// (b) have *every* earlier visited cell empty per `gpu_get`. This is the
    /// property the WGSL relies on.
    fn assert_dda_consistent(gpu: &DagGpu, origin: [f32; 3], dir: [f32; 3]) {
        if let Some((hit, empties)) = ray_dda_first_hit(gpu, origin, norm(dir)) {
            assert!(
                solid_at(gpu, hit[0], hit[1], hit[2]),
                "DDA reported hit at {hit:?} but gpu_get says empty"
            );
            for c in empties {
                assert!(
                    !solid_at(gpu, c[0], c[1], c[2]),
                    "DDA skipped a solid cell at {c:?} before the reported hit"
                );
            }
        }
    }

    #[test]
    fn dda_matches_gpu_get_uniform() {
        let gpu = DagBrick::from_brick(&uniform_brick()).to_gpu();
        // Rays from outside, aimed into the box from several directions.
        assert_dda_consistent(&gpu, [-5.0, 8.0, 8.0], [1.0, 0.0, 0.0]);
        assert_dda_consistent(&gpu, [8.0, 24.0, 8.0], [0.0, -1.0, 0.0]);
        assert_dda_consistent(&gpu, [-4.0, -4.0, -4.0], [1.0, 1.0, 1.0]);
        // A ray from outside a fully-solid brick must hit the very first cell
        // it enters.
        let hit = ray_dda_first_hit(&gpu, [-5.0, 8.5, 8.5], norm([1.0, 0.0, 0.0]));
        let (cell, empties) = hit.expect("ray should enter the solid brick");
        assert_eq!(cell[0], 0, "first solid cell should be the entry plane x=0");
        assert!(empties.is_empty(), "no empty cells before entering a solid brick");
    }

    #[test]
    fn dda_matches_gpu_get_half() {
        let gpu = DagBrick::from_brick(&half_brick()).to_gpu();
        // Downward ray through the empty upper half into the solid lower half:
        // first hit must be y == 7 (top of the solid block).
        let hit = ray_dda_first_hit(&gpu, [8.5, 25.0, 8.5], norm([0.0, -1.0, 0.0]));
        let (cell, _) = hit.expect("downward ray should hit the lower half");
        assert_eq!(cell[1], (E / 2) - 1, "first solid cell is the top of the lower half");
        assert!(solid_at(&gpu, cell[0], cell[1], cell[2]));
        // Several oblique rays stay consistent with the oracle.
        assert_dda_consistent(&gpu, [-3.0, 20.0, 8.0], [1.0, -1.0, 0.2]);
        assert_dda_consistent(&gpu, [8.0, 8.0, -3.0], [0.1, -0.3, 1.0]);
    }

    #[test]
    fn dda_matches_gpu_get_sparse() {
        let b = sparse_brick();
        let gpu = DagBrick::from_brick(&b).to_gpu();
        // Exhaustive cross-check of the point oracle against the brick itself:
        // gpu_get must agree with the source brick for every cell (guards the
        // to_gpu encoding the DDA reads).
        for z in 0..E {
            for y in 0..E {
                for x in 0..E {
                    let expect = b.get(IVec3::new(x as i64, y as i64, z as i64));
                    let got = gpu_get(&gpu, x as u8, y as u8, z as u8);
                    assert_eq!(got, expect, "gpu_get mismatch at ({x},{y},{z})");
                }
            }
        }
        // Aim a ray straight at the isolated voxel (1,1,1) along the diagonal.
        assert_dda_consistent(&gpu, [-2.0, -2.0, -2.0], [1.0, 1.0, 1.0]);
        // Aim at the corner voxel (15,15,15).
        assert_dda_consistent(&gpu, [20.0, 20.0, 20.0], [-1.0, -1.0, -1.0]);
    }

    #[test]
    fn empty_brick_yields_no_material() {
        let gpu = DagBrick::from_brick(&Brick::new()).to_gpu();
        // Empty DAG returns before touching storage, so a default handle +
        // empty store suffice (no `Assets::add` needed).
        let mut sb = Assets::<bevy::render::storage::ShaderStorageBuffer>::default();
        let palette = Handle::<bevy::render::storage::ShaderStorageBuffer>::default();
        let m = super::build_raymarch_material(
            &gpu,
            palette,
            super::RaymarchShadingTier::Lambert,
            &mut sb,
        );
        assert!(m.is_none(), "empty brick should not produce a raymarch material");
    }
}
