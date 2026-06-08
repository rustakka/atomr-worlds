// GPU DAG raymarcher (Rec 1).
//
// Renders one brick by raymarching its sparse-voxel DAG
// (atomr_worlds_voxel::DagBrick::to_gpu) in the fragment shader, instead of a
// triangle mesh. The proxy cube's *local* space spans the brick's [0, 16)^3
// voxel grid, so the per-object model matrix maps voxel space <-> world.
//
// We use a CUSTOM VERTEX stage (where the per-object `mesh` bind group, @group(2),
// is available) to hand the fragment everything it needs in *voxel space*:
//   - local_pos   : interpolated cube-surface point  -> ray entry in voxel space
//   - cam_local    : camera in voxel space (flat)     -> ray origin
//   - clip_from_local : view-projection * model (flat) -> hit -> reversed-Z depth
// The fragment never touches @group(2), so the mesh bind group can stay
// vertex-only in the pipeline layout. The brick transform is translate + uniform
// scale (no rotation), so local axis-normals are already world-space normals.
//
// On the first solid voxel the fragment writes the shaded color AND frag_depth,
// so the result composites against the rest of the scene through the ordinary
// reversed-Z depth buffer. Misses discard.
//
// Bevy 0.18 bind groups: view/lights = @group(0), per-object mesh = @group(2)
// (vertex stage only, via mesh_functions), this material = @group(3).

#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_pbr::mesh_functions::{get_world_from_local, get_local_from_world}

// --- DAG word encoding (mirror of atomr_worlds_voxel::dag) --------------------
const DAG_LEAF_FLAG: u32 = 0x80000000u;      // high bit marks a leaf
const DAG_GPU_EMPTY_ROOT: u32 = 0xffffffffu; // u32::MAX sentinel

// --- Shading tiers (mirror of RaymarchShadingTier::to_u32) --------------------
const TIER_UNLIT: u32 = 0u;
const TIER_LAMBERT: u32 = 1u;
const TIER_PBR: u32 = 2u;

// Half-voxel overdraw the tightened proxy box carries on every side. Adjacent
// bricks' proxies then overlap at their shared faces, so the sub-pixel
// rasterization gap between them (visible as dotted seam lines at grazing
// angles) is covered by a neighbour. The DDA still clamps cells to [0, edge),
// so the visible voxel content is unchanged — only the proxy silhouette and the
// slab entry t grow by the pad.
const PROXY_PAD: f32 = 0.5;

struct PaletteEntry {
    base_color: vec4<f32>,
    pbr: vec4<f32>,       // (perceptual_roughness, metallic, _, _)
    emissive: vec4<f32>,  // (r, g, b, _)
};

struct RaymarchMeta {
    root: u32,
    brick_edge: u32,
    shading_tier: u32,
    aabb_min: u32,   // occupancy AABB min corner, packed x | y<<8 | z<<16
    aabb_max: u32,   // inclusive max corner, packed; continuous bound = max + 1
    flags: u32,
};

// Unpack a RaymarchMeta AABB corner (x | y<<8 | z<<16) into voxel coords.
fn unpack_aabb(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu));
}

@group(3) @binding(0) var<storage, read> nodes: array<u32>;
@group(3) @binding(1) var<storage, read> colors: array<u32>;     // material ids (u16 widened)
@group(3) @binding(2) var<storage, read> palette: array<PaletteEntry>;
@group(3) @binding(3) var<uniform> dag_meta: RaymarchMeta;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct Varyings {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
    @location(1) @interpolate(flat) cam_local: vec3<f32>,
    // Columns of `clip_from_world * world_from_local` (constant per object).
    @location(2) @interpolate(flat) m0: vec4<f32>,
    @location(3) @interpolate(flat) m1: vec4<f32>,
    @location(4) @interpolate(flat) m2: vec4<f32>,
    @location(5) @interpolate(flat) m3: vec4<f32>,
};

struct RaymarchOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vertex(v: Vertex) -> Varyings {
    var out: Varyings;
    // Tighten the [0, edge]^3 proxy box to the brick's occupancy AABB so the
    // empty rim is never rasterized (and the DDA's empty prefix is skipped).
    // aabb_max is the inclusive max voxel, so the continuous upper bound is +1.
    let edge = f32(dag_meta.brick_edge);
    let pad = vec3<f32>(PROXY_PAD);
    let amin = unpack_aabb(dag_meta.aabb_min) - pad;
    let amax = unpack_aabb(dag_meta.aabb_max) + vec3<f32>(1.0) + pad;
    let vpos = amin + (v.position / edge) * (amax - amin);

    let world_from_local = get_world_from_local(v.instance_index);
    let local_from_world = get_local_from_world(v.instance_index);
    let world_pos = world_from_local * vec4<f32>(vpos, 1.0);
    out.clip_position = view.clip_from_world * world_pos;
    out.local_pos = vpos;
    out.cam_local = (local_from_world * vec4<f32>(view.world_position, 1.0)).xyz;
    let clip_from_local = view.clip_from_world * world_from_local;
    out.m0 = clip_from_local[0];
    out.m1 = clip_from_local[1];
    out.m2 = clip_from_local[2];
    out.m3 = clip_from_local[3];
    return out;
}

// Point lookup into the flat DAG — a line-for-line port of
// `atomr_worlds_voxel::gpu_get`. Returns the material id at (x, y, z), or -1 for
// empty space. The CPU mirror trio is `gpu_get` (point) + `ray_dda_first_hit`
// (ray, in atomr-worlds-voxel/src/raymarch.rs) + this shader; keeping all three
// in lock-step is the determinism gate (the voxel crate's parity tests guard the
// pair, and the view crate's raymarch_golden pins the rendered CPU output).
fn dag_lookup(x: u32, y: u32, z: u32) -> i32 {
    if (dag_meta.root == DAG_GPU_EMPTY_ROOT) {
        return -1;
    }
    var word: u32 = dag_meta.root;
    var ox0: u32 = 0u;
    var oy0: u32 = 0u;
    var oz0: u32 = 0u;
    var depth: u32 = 0u;
    // 16^3 -> at most 4 internal levels + a leaf; loop bound is a safety cap.
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let w = nodes[word];
        if ((w & DAG_LEAF_FLAG) != 0u) {
            let ci = w & 0x7fffffffu;
            return i32(colors[ci]);
        }
        let mask = w & 0xffu;
        let half = (dag_meta.brick_edge >> depth) >> 1u;
        let ox = select(0u, 1u, (x - ox0) >= half);
        let oy = select(0u, 1u, (y - oy0) >= half);
        let oz = select(0u, 1u, (z - oz0) >= half);
        let octant = ox | (oy << 1u) | (oz << 2u);
        let bit = 1u << octant;
        if ((mask & bit) == 0u) {
            return -1;
        }
        let slot = countOneBits(mask & (bit - 1u));
        word = nodes[word + 1u + slot];
        ox0 = ox0 + ox * half;
        oy0 = oy0 + oy * half;
        oz0 = oz0 + oz * half;
        depth = depth + 1u;
    }
    return -1;
}

const PI: f32 = 3.14159265359;

// Fixed ambient floor (shared by Lambert + Pbr): keeps faces turned away from
// the sun visible, and the (1 - AMBIENT) complement is the direct-light weight.
// Both tiers stay in this regime so switching tiers is not a brightness jump.
const AMBIENT: f32 = 0.30;

// Pbr AO: how strongly local DAG occupancy darkens the ambient term. 0 = off.
const AO_STRENGTH: f32 = 0.7;
// Pbr self-shadow: voxel step + max steps for the brick-local sun-occlusion
// march (32 * 0.5 = 16 = one brick edge). Half-voxel steps so axis-aligned
// overhangs are not stepped over.
const SHADOW_STEP: f32 = 0.5;
const SHADOW_MAX_STEPS: i32 = 32;

// --- PBR helpers (Cook-Torrance, single directional light) -------------------
// `a` is the GGX roughness (perceptual_roughness squared); `roughness` below is
// the perceptual value Bevy's StandardMaterial exposes.

fn distribution_ggx(ndh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn geometry_schlick_ggx(nd: f32, k: f32) -> f32 {
    return nd / max(nd * (1.0 - k) + k, 1e-7);
}

fn geometry_smith(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    // Direct-lighting remap k = (r + 1)^2 / 8.
    let r1 = roughness + 1.0;
    let k = (r1 * r1) / 8.0;
    return geometry_schlick_ggx(ndv, k) * geometry_schlick_ggx(ndl, k);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Is the brick-local cell `c` solid? Out-of-brick cells read as air: AO and
// shadows are brick-local only (cross-brick occlusion needs a top-level
// acceleration structure that does not exist yet — see the module docs).
fn solid_at(c: vec3<i32>) -> bool {
    let edge = i32(dag_meta.brick_edge);
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(edge))) {
        return false;
    }
    return dag_lookup(u32(c.x), u32(c.y), u32(c.z)) >= 0;
}

// Ambient occlusion from local DAG occupancy. `ni` is the integer outward face
// normal; we look at the 8-neighbour ring, in the face's tangent plane, around
// the air cell in front of the lit face. A surface sitting in an inside corner
// (more solid neighbours) is darkened — the raymarch analogue of the mesh
// path's baked per-vertex AO (`base_color * vertex.color.r`).
fn ao_from_occupancy(cell: vec3<i32>, ni: vec3<i32>) -> f32 {
    let air = cell + ni;
    var t1 = vec3<i32>(1, 0, 0);
    var t2 = vec3<i32>(0, 1, 0);
    if (abs(ni.x) > 0) {
        t1 = vec3<i32>(0, 1, 0);
        t2 = vec3<i32>(0, 0, 1);
    } else if (abs(ni.y) > 0) {
        t1 = vec3<i32>(1, 0, 0);
        t2 = vec3<i32>(0, 0, 1);
    }
    var occ = 0.0;
    for (var di = -1; di <= 1; di = di + 1) {
        for (var dj = -1; dj <= 1; dj = dj + 1) {
            if (di == 0 && dj == 0) {
                continue;
            }
            if (solid_at(air + t1 * di + t2 * dj)) {
                occ = occ + 1.0;
            }
        }
    }
    return 1.0 - AO_STRENGTH * (occ / 8.0);
}

// Brick-local hard self-shadow: point-march along the sun direction `l` from the
// air cell in front of the lit face. Returns 0 if a solid voxel occludes the sun
// before the ray leaves the brick, else 1. Point sampling via `dag_lookup` keeps
// this identical to the CPU twin (both reuse the mirrored point lookup).
fn sun_shadow(cell: vec3<i32>, ni: vec3<i32>, l: vec3<f32>) -> f32 {
    let edge = f32(dag_meta.brick_edge);
    var p = vec3<f32>(cell) + vec3<f32>(0.5) + vec3<f32>(ni);
    let step = l * SHADOW_STEP;
    for (var i = 0; i < SHADOW_MAX_STEPS; i = i + 1) {
        p = p + step;
        if (any(p < vec3<f32>(0.0)) || any(p >= vec3<f32>(edge))) {
            return 1.0; // left the brick without an occluder
        }
        let c = vec3<i32>(floor(p));
        if (dag_lookup(u32(c.x), u32(c.y), u32(c.z)) >= 0) {
            return 0.0; // occluded
        }
    }
    return 1.0;
}

fn shade(mat_id: u32, world_normal: vec3<f32>, cell: vec3<i32>, view_dir: vec3<f32>) -> vec3<f32> {
    let entry = palette[mat_id];
    let base = entry.base_color.rgb;

    if (dag_meta.shading_tier == TIER_UNLIT) {
        return base;
    }

    // Sun for the geometry term: use the light's DIRECTION and its HUE for tint,
    // but not its raw illuminance magnitude (which would blow out before
    // tonemapping); the fixed AMBIENT floor keeps shaded faces visible.
    var l = vec3<f32>(0.0, 1.0, 0.0);
    var sun_rgb = vec3<f32>(1.0);
    if (lights.n_directional_lights > 0u) {
        l = normalize(lights.directional_lights[0].direction_to_light);
        let c = lights.directional_lights[0].color.rgb;
        let m = max(max(c.r, c.g), max(c.b, 1e-6));
        sun_rgb = c / m;
    }
    let n = normalize(world_normal);
    let ndl = max(dot(n, l), 0.0);

    if (dag_meta.shading_tier == TIER_LAMBERT) {
        return base * (AMBIENT + (1.0 - AMBIENT) * ndl) * sun_rgb;
    }

    // --- TIER_PBR: Cook-Torrance specular + occupancy AO + self-shadow --------
    let ni = vec3<i32>(round(n));
    let v = normalize(view_dir);
    let h = normalize(l + v);
    let ndv = max(dot(n, v), 1e-4);
    let ndh = max(dot(n, h), 0.0);
    let vdh = max(dot(v, h), 0.0);

    let roughness = clamp(entry.pbr.x, 0.045, 1.0);
    let metal = clamp(entry.pbr.y, 0.0, 1.0);
    let a = roughness * roughness;

    let f0 = mix(vec3<f32>(0.04), base, metal);
    let d_term = distribution_ggx(ndh, a);
    let g_term = geometry_smith(ndv, ndl, roughness);
    let f_term = fresnel_schlick(vdh, f0);
    // The ndl in the BRDF denominator cancels the cosine term applied below.
    let spec_brdf = (d_term * g_term * f_term) / (4.0 * ndv * ndl + 1e-4);

    let ao = ao_from_occupancy(cell, ni);
    var shadow = 1.0;
    if (ndl > 0.0) {
        shadow = sun_shadow(cell, ni, l);
    }

    // Diffuse keeps the Lambert brightness regime (so PBR is not a brightness
    // jump): ambient floor is AO-modulated, direct term is shadowed and
    // metal-suppressed. Specular is the sun's highlight, also direct-shadowed.
    let diffuse = base * (1.0 - metal)
        * (AMBIENT * ao + (1.0 - AMBIENT) * ndl * shadow);
    let specular = spec_brdf * ndl * shadow * (1.0 - AMBIENT);
    return (diffuse + specular) * sun_rgb + entry.emissive.rgb;
}

@fragment
fn fragment(in: Varyings) -> RaymarchOutput {
    var out: RaymarchOutput;
    out.color = vec4<f32>(0.0);
    out.depth = 0.0;

    let cam_local = in.cam_local;
    let frag_local = in.local_pos;
    let dir = normalize(frag_local - cam_local);

    let edge_i = i32(dag_meta.brick_edge);

    // Slab-intersect the ray against the occupancy AABB [amin, amax+1] (tighter
    // than the full [0, edge]^3 cube — skips the brick's empty rim). The DDA
    // still indexes cells in [0, edge); the tight slab only moves the entry t.
    // NB: the DDA slab uses the TIGHT occupancy AABB (no PROXY_PAD). The pad
    // lives only in the vertex stage, where it grows the rasterized proxy so
    // neighbours overlap and close seam cracks. Padding the slab here too made
    // a ray entering the half-voxel rim above a brick-top-boundary voxel
    // (cell == edge-1) clamp onto that voxel and register the hit at the padded
    // entry t — rendering a half-voxel "lip" on top of the surface.
    let amin = unpack_aabb(dag_meta.aabb_min);
    let amax = unpack_aabb(dag_meta.aabb_max) + vec3<f32>(1.0);
    let inv_dir = 1.0 / dir;                       // inf for axis-parallel rays (handled by min/max)
    let ta = (amin - cam_local) * inv_dir;
    let tb = (amax - cam_local) * inv_dir;
    let tmin3 = min(ta, tb);
    let tmax3 = max(ta, tb);
    let t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter > t_exit || t_exit < 0.0) {
        discard;
        return out;
    }

    let start = max(t_enter, 0.0);
    let p = cam_local + dir * start;

    // Amanatides-Woo DDA setup.
    var cell = clamp(vec3<i32>(floor(p)), vec3<i32>(0), vec3<i32>(edge_i - 1));
    let stepf = sign(dir);
    let step = vec3<i32>(stepf);
    let small = abs(dir) < vec3<f32>(1e-12);
    let inv = 1.0 / select(dir, vec3<f32>(1.0), small);
    // Next axis-aligned boundary the ray crosses: cell+1 if moving +, else cell.
    let next_boundary = vec3<f32>(cell) + max(stepf, vec3<f32>(0.0));
    var t_max = start + select((next_boundary - p) * inv, vec3<f32>(1e30), small);
    let t_delta = select(abs(inv), vec3<f32>(1e30), small);

    var hit_mat: i32 = -1;
    var t_entry: f32 = start;   // t at which the ray enters `cell`
    var enter_axis: i32 = -1;   // axis crossed to enter `cell` (-1 = started inside)

    for (var s: u32 = 0u; s < 64u; s = s + 1u) {
        if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(edge_i))) {
            break;
        }
        let mat = dag_lookup(u32(cell.x), u32(cell.y), u32(cell.z));
        if (mat >= 0) {
            hit_mat = mat;
            break;
        }
        // Advance to the next cell across the nearest boundary.
        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            cell.x = cell.x + step.x;
            t_entry = t_max.x;
            t_max.x = t_max.x + t_delta.x;
            enter_axis = 0;
        } else if (t_max.y <= t_max.z) {
            cell.y = cell.y + step.y;
            t_entry = t_max.y;
            t_max.y = t_max.y + t_delta.y;
            enter_axis = 1;
        } else {
            cell.z = cell.z + step.z;
            t_entry = t_max.z;
            t_max.z = t_max.z + t_delta.z;
            enter_axis = 2;
        }
    }

    if (hit_mat < 0) {
        discard;
        return out;
    }

    // Hit point in voxel space -> clip space (reversed-Z depth). The brick has
    // no rotation, so axis-aligned local normals are already world normals.
    let p_hit = cam_local + dir * t_entry;
    let clip_from_local = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let clip = clip_from_local * vec4<f32>(p_hit, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);

    var n = vec3<f32>(0.0);
    if (enter_axis == 0) {
        n.x = -stepf.x;
    } else if (enter_axis == 1) {
        n.y = -stepf.y;
    } else if (enter_axis == 2) {
        n.z = -stepf.z;
    } else {
        n = -dir; // camera started inside this voxel; face it
    }

    // View direction (surface -> camera) in voxel space; the brick has no
    // rotation, so voxel-space directions are world-space directions.
    let v_dir = normalize(cam_local - p_hit);
    out.color = vec4<f32>(shade(u32(hit_mat), n, cell, v_dir), 1.0);
    return out;
}
