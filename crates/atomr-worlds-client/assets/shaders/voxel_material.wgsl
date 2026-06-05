// Custom voxel material — overrides the StandardMaterial's PBR fields
// per-fragment from a palette storage buffer. The material id is encoded
// in vertex.uv.x and the AO factor in vertex.color.r (set by the
// `PaletteVoxelMaterial` shading strategy in fp.rs).
//
// Lets one merged brick mesh fan out to N PBR materials in a single draw
// call instead of N child PbrBundles.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::pbr_functions::apply_pbr_lighting

struct PaletteEntry {
    base_color: vec4<f32>,
    pbr: vec4<f32>,
    emissive: vec4<f32>,
};

// Bevy 0.18: the material bind group is @group(3) (view = 0, mesh = 2). This
// was @group(2) before the 0.13 -> 0.18 upgrade. StandardMaterial reserves
// bindings 0–99 of the material group; the extension's palette is at 100.
@group(3) @binding(100) var<storage, read> palette: array<PaletteEntry>;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Material id is in uv.x, rounded to nearest integer.
    let mat_id = u32(in.uv.x + 0.5);
    let entry = palette[mat_id];

    // AO factor is in vertex.color.r (set by the AoStrategy bake step).
    let ao = clamp(in.color.r, 0.0, 1.0);

    pbr_input.material.base_color = vec4<f32>(
        entry.base_color.rgb * ao,
        entry.base_color.a,
    );
    pbr_input.material.perceptual_roughness = entry.pbr.x;
    pbr_input.material.metallic = entry.pbr.y;
    pbr_input.material.emissive = vec4<f32>(entry.emissive.rgb, 1.0);

    return apply_pbr_lighting(pbr_input);
}
