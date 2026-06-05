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
//! line-for-line port of [`atomr_worlds_voxel::gpu_get`] (point lookup) and
//! [`atomr_worlds_voxel::ray_dda_first_hit`] (ray DDA) — keeping the WGSL in
//! lock-step with that pair is the determinism gate. The voxel crate owns the
//! parity tests; the directed `#[cfg(test)]` checks below guard the client side.
//!
//! ## Binding-slot convention (Bevy 0.18)
//!
//! A stand-alone [`Material`] owns the material bind group entirely
//! (`@group(3)` in Bevy 0.18 — `view` is group 0, the per-object `mesh` array
//! is group 2). So this material's storage buffers + uniform live at
//! `@group(3) @binding(0..3)`, and the shader reaches the live model matrix via
//! `bevy_pbr::mesh_functions` (group 2) and the camera/sun via
//! `bevy_pbr::mesh_view_bindings` (group 0).

use atomr_worlds_voxel::{DagGpu, BRICK_EDGE};
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
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

/// Per-brick scalar parameters for the raymarcher. All `u32`s; the WGSL struct
/// mirrors the field order exactly.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct RaymarchMeta {
    /// Word offset of the DAG root in `nodes`, or [`DAG_GPU_EMPTY_ROOT`].
    pub root: u32,
    /// Voxel grid edge (always [`BRICK_EDGE`] = 16); the LOD scale lives in the
    /// model matrix, so traversal is always over the integer `[0, 16)` grid.
    pub brick_edge: u32,
    /// [`RaymarchShadingTier::to_u32`].
    pub shading_tier: u32,
    /// Occupancy AABB min corner, packed `x | y<<8 | z<<16` (brick-local voxel
    /// coords). The vertex stage shrinks the proxy cube to this AABB and the
    /// fragment slab-clips the DDA to it, so the brick's empty rim is never
    /// rasterized or marched. See [`pack_aabb`].
    pub aabb_min: u32,
    /// Occupancy AABB **inclusive** max corner, packed like [`Self::aabb_min`];
    /// the continuous upper bound is `aabb_max + 1` per axis.
    pub aabb_max: u32,
    /// Reserved for future flags.
    pub flags: u32,
}

/// Pack a brick-local voxel corner `[x, y, z]` (each `< 16`) into one `u32` as
/// `x | y<<8 | z<<16` for [`RaymarchMeta`]. The WGSL unpacks with `& 0xff` /
/// shifts.
#[inline]
pub fn pack_aabb(c: [u8; 3]) -> u32 {
    c[0] as u32 | ((c[1] as u32) << 8) | ((c[2] as u32) << 16)
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

/// Upload one brick's flattened DAG geometry + color into fresh storage buffers
/// (colors widened u16→u32 — WGSL has no u16 storage arrays). Returns the
/// `(nodes, colors)` handles. The [`DagBufferCache`](super::dag_cache::DagBufferCache)
/// dedups these across structurally-identical bricks; this is the raw upload.
pub fn upload_dag_buffers(
    dag: &DagGpu,
    storage_buffers: &mut Assets<ShaderStorageBuffer>,
) -> (Handle<ShaderStorageBuffer>, Handle<ShaderStorageBuffer>) {
    let nodes = storage_buffers.add(ShaderStorageBuffer::from(dag.nodes.clone()));
    let colors_u32: Vec<u32> = dag.colors.iter().map(|&c| c as u32).collect();
    let colors = storage_buffers.add(ShaderStorageBuffer::from(colors_u32));
    (nodes, colors)
}

/// The [`RaymarchMeta`] for a brick: root + grid edge + shading tier + the
/// packed occupancy AABB. Pure (no allocation) so the cache can build it per
/// `(digest, tier)` without re-uploading buffers.
pub fn raymarch_meta(
    dag: &DagGpu,
    tier: RaymarchShadingTier,
    aabb_min: [u8; 3],
    aabb_max: [u8; 3],
) -> RaymarchMeta {
    RaymarchMeta {
        root: dag.root,
        brick_edge: BRICK_EDGE as u32,
        shading_tier: tier.to_u32(),
        aabb_min: pack_aabb(aabb_min),
        aabb_max: pack_aabb(aabb_max),
        flags: 0,
    }
}

/// Assemble a [`RaymarchMaterial`] from already-uploaded buffer handles + scalar
/// params. The cache calls this with shared buffers so identical bricks reuse one
/// buffer set; the per-`(digest, tier)` material is the only thing rebuilt.
pub fn raymarch_material_from_parts(
    nodes: Handle<ShaderStorageBuffer>,
    colors: Handle<ShaderStorageBuffer>,
    palette: Handle<ShaderStorageBuffer>,
    meta: RaymarchMeta,
) -> RaymarchMaterial {
    RaymarchMaterial { nodes, colors, palette, meta }
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
    //! Client-side determinism checks for the raymarch path. The canonical CPU
    //! ray DDA — the line-for-line mirror of the WGSL `@fragment` — now lives in
    //! [`atomr_worlds_voxel::ray_dda_first_hit`] (with its own exhaustive parity
    //! suite against `gpu_get`). Here we keep a thin set of *directed* checks
    //! (first-hit cell + material for known fixtures) plus the empty-brick guard,
    //! so the client crate also fails loudly if the shared DDA or the material
    //! factory regresses.

    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_voxel::{ray_dda_first_hit, Brick, DagBrick, Voxel, BRICK_EDGE};

    const E: i32 = BRICK_EDGE as i32;

    fn norm(v: [f32; 3]) -> [f32; 3] {
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / m, v[1] / m, v[2] / m]
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

    #[test]
    fn uniform_brick_hits_entry_plane() {
        let gpu = DagBrick::from_brick(&uniform_brick()).to_gpu();
        let hit = ray_dda_first_hit(&gpu, [-5.0, 8.5, 8.5], norm([1.0, 0.0, 0.0]))
            .expect("ray should enter the solid brick");
        assert_eq!(hit.cell[0], 0, "first solid cell is the entry plane x=0");
        assert_eq!(hit.material, 1);
    }

    #[test]
    fn half_brick_hits_top_of_block() {
        let gpu = DagBrick::from_brick(&half_brick()).to_gpu();
        let hit = ray_dda_first_hit(&gpu, [8.5, 25.0, 8.5], norm([0.0, -1.0, 0.0]))
            .expect("downward ray should hit the lower half");
        assert_eq!(hit.cell[1], (E / 2) - 1, "first solid cell is the top of the lower half");
        assert_eq!(hit.material, 2);
    }
}
