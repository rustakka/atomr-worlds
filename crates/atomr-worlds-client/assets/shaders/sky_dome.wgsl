// Procedural sky dome (Step 9).
//
// Renders inside an inside-out sphere parented to the camera. The
// fragment shader computes the view ray, blends horizon → zenith by
// the y component, and adds a sun disc + soft glow.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

// Bevy 0.18: the material bind group is @group(3) (view = 0, mesh = 2). This
// was @group(2) before the 0.13 -> 0.18 upgrade; group 2 is now the per-object
// mesh array, so the stale number caused a pipeline validation panic.
@group(3) @binding(0) var<uniform> horizon_color: vec4<f32>;
@group(3) @binding(1) var<uniform> zenith_color: vec4<f32>;
@group(3) @binding(2) var<uniform> sun_color: vec4<f32>;
@group(3) @binding(3) var<uniform> sun_direction: vec4<f32>;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // World-space view ray from camera through this fragment.
    let dir = normalize(in.world_position.xyz - view.world_position);

    // Vertical mix factor: 1 at zenith, 0 at horizon, < 0 below.
    let t = clamp(dir.y, 0.0, 1.0);
    // Bias toward horizon so the gradient doesn't compress at the
    // top — `pow(1 - t, 4)` from the original plan, but stay simple.
    let h = pow(1.0 - t, 4.0);
    let base = mix(zenith_color.rgb, horizon_color.rgb, h);

    // Sun: `sun_direction` points FROM sun INTO scene, so the unit
    // vector toward the sun (from observer) is `-sun_direction.xyz`.
    let to_sun = -normalize(sun_direction.xyz);
    let cos_theta = max(dot(dir, to_sun), 0.0);
    let disc = smoothstep(0.9994, 0.9998, cos_theta);
    let glow = pow(cos_theta, 96.0) * 0.6;

    let sky = base + sun_color.rgb * (disc + glow);
    return vec4<f32>(sky, 1.0);
}
