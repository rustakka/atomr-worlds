//! Custom material assets for Step 8 (PaletteVoxelMaterial) and Step 9
//! (ProceduralDomeSky).
//!
//! - [`VoxelMaterialExt`] extends [`StandardMaterial`] with a palette
//!   storage buffer; one [`VoxelMaterial`] handle per brick covers every
//!   material id present, dropping the draw-call count from N→1 per
//!   brick.
//! - [`SkyDomeMaterial`] is a stand-alone [`Material`] applied to a
//!   sphere parented to the camera. It writes a gradient + sun-disc
//!   inside the fragment shader; depth is forced to far so terrain still
//!   occludes the dome correctly.
//!
//! WGSL lives at `assets/shaders/voxel_material.wgsl` and
//! `assets/shaders/sky_dome.wgsl`.
//!
//! ## Binding-slot convention
//!
//! `StandardMaterial` reserves slots 0–99 on its bind group. Custom
//! extensions go at >= 100; we use slot 100 for the palette storage
//! buffer. Sky-dome is a stand-alone material so its uniforms go at slot
//! 0.

use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, Face, RenderPipelineDescriptor, ShaderRef, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderStorageBuffer;

/// Palette entry packed for the GPU. Stays 48 bytes (3× vec4 = vec4
/// alignment) so the storage buffer layout matches the WGSL struct.
#[derive(Clone, Copy, Debug, ShaderType, Default)]
pub struct PaletteEntryGpu {
    pub base_color: Vec4,
    /// (perceptual_roughness, metallic, _, _)
    pub pbr: Vec4,
    /// (emissive_r, emissive_g, emissive_b, _)
    pub emissive: Vec4,
}

/// Extension to `StandardMaterial`: a palette storage buffer the
/// fragment shader indexes by `material_id` (encoded in `uv.x`).
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct VoxelMaterialExt {
    /// Palette storage buffer. `StandardMaterial` reserves slots 0–99; the
    /// convention for extensions is to start at 100.
    ///
    /// Bevy 0.16 moved `#[storage]` buffers off inline `Vec<T>` and onto a
    /// `Handle<ShaderStorageBuffer>` asset — build it with
    /// `ShaderStorageBuffer::from(Vec<PaletteEntryGpu>)` (the `Vec` is a
    /// `ShaderType` runtime array, encoded via encase) and add it to
    /// `Assets<ShaderStorageBuffer>`.
    #[storage(100, read_only)]
    pub palette: Handle<ShaderStorageBuffer>,
}

impl MaterialExtension for VoxelMaterialExt {
    fn fragment_shader() -> ShaderRef {
        "shaders/voxel_material.wgsl".into()
    }
}

/// Type alias used everywhere a [`Handle`] / [`MaterialPlugin`] /
/// [`MaterialMeshBundle`] of the combined material is needed.
pub type VoxelMaterial = ExtendedMaterial<StandardMaterial, VoxelMaterialExt>;

// ---------------------------------------------------------------------------
// Sky dome material (Step 9)
// ---------------------------------------------------------------------------

/// Inside-out sphere material. Renders a gradient sky tinted by sun
/// state, with a soft sun disc + glow. The dome is parented to the
/// camera and sized larger than the FP fog falloff so it never
/// foreground-clips.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct SkyDomeMaterial {
    /// Linear RGB at the horizon when looking horizontally.
    #[uniform(0)]
    pub horizon_color: Vec4,
    /// Linear RGB at the zenith when looking straight up.
    #[uniform(1)]
    pub zenith_color: Vec4,
    /// Linear RGB of the sun disc (used for both the disc and the
    /// surrounding bloom-friendly glow).
    #[uniform(2)]
    pub sun_color: Vec4,
    /// World-space direction FROM the sun (so it's the same convention
    /// as `DirectionalLight.transform.forward()`).
    #[uniform(3)]
    pub sun_direction: Vec4,
}

impl Material for SkyDomeMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/sky_dome.wgsl".into()
    }
    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Cull front faces so the dome is visible from inside.
        descriptor.primitive.cull_mode = Some(Face::Front);
        Ok(())
    }
}
