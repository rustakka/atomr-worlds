//! Deterministic CPU software rasterizer.
//!
//! Half-space edge-function triangle rasterizer with a single-precision
//! z-buffer. Flat-shaded: each triangle's color is `material_color(material)
//! * lambert(normal · light_dir)`. Output: an RGBA8 byte buffer with width *
//! height * 4 entries plus a same-size f32 depth buffer.
//!
//! Determinism: all math uses native `f32`; iteration order is bound by
//! `for index in ..indices.len()`, so identical inputs produce byte-identical
//! pixel buffers across runs and platforms.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use atomr_worlds_voxel::{gpu_get, ray_dda_first_hit, Brick, DagGpu, RayHit, Voxel, BRICK_EDGE};

use crate::camera::{transform_point, Camera};
use crate::mesh::{greedy_mesh, Mesh, Vertex};
use crate::scene::MeshNode;
use crate::skybox::Skybox;
use crate::ViewError;

#[derive(Copy, Clone, Debug)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub background: [u8; 4],
    pub light_dir: [f32; 3], // pointing FROM the surface TO the light source
    pub ambient: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            background: [16, 18, 24, 255],
            // sun coming from upper-back-right
            light_dir: [0.5, 0.8, 0.3],
            ambient: 0.25,
        }
    }
}

/// Output buffer pair returned by [`render_mesh`].
///
/// **Depth convention (reversed-z, Phase 13f):** `depth[i] = 1.0` is the near
/// plane and `depth[i] = 0.0` is the far plane. The buffer is initialised to
/// `0.0` (everything is "infinitely far" until written) and the rasterizer
/// keeps the largest seen z per pixel — closer fragments win because the
/// reversed-z projection maps closer points to larger depth values.
#[derive(Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA8
    pub depth: Vec<f32>, // reversed-z: 1.0 = near, 0.0 = far (already z/w)
}

impl std::fmt::Debug for Framebuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Framebuffer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pixels_len", &self.pixels.len())
            .field("depth_len", &self.depth.len())
            .finish()
    }
}

impl Framebuffer {
    pub fn write_png<P: AsRef<Path>>(&self, path: P) -> Result<(), ViewError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let mut enc = png::Encoder::new(&mut writer, self.width, self.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| ViewError::Png(e.to_string()))?;
        w.write_image_data(&self.pixels).map_err(|e| ViewError::Png(e.to_string()))?;
        Ok(())
    }

    /// 64-bit FNV-1a digest of the pixel buffer. Stable hash for screenshot
    /// regression tests that don't want to commit binary baselines.
    pub fn pixels_fnv1a(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in &self.pixels {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

/// Map a material id to an RGB color. Air (`0`) is unused (transparent).
/// Mirrors the client-side `HardcodedPalette` so the raster2d view modes
/// (slice/RTS/overview) agree with the 3D modes on what each id looks like.
pub fn material_color(material: u16) -> [u8; 3] {
    match material {
        1 => [107, 102, 97],   // stone (linear 0.42/0.40/0.38 → sRGB ~107/102/97)
        2 => [82, 56, 36],     // dirt
        3 => [199, 178, 122],  // sand
        4 => [210, 218, 228],  // snow
        5 => [25, 89, 140],    // water
        6 => [46, 115, 41],    // grass
        7 => [76, 46, 25],     // wood
        8 => [33, 92, 30],     // leaves
        9 => [255, 200, 60],   // glow_rock (bright — emissive)
        10 => [199, 224, 242], // ice
        _ => {
            // Cheap deterministic palette for unknown materials.
            let m = material as u32;
            let r = ((m * 73) & 0xFF) as u8;
            let g = ((m * 167) & 0xFF) as u8;
            let b = ((m * 251) & 0xFF) as u8;
            [r, g, b]
        }
    }
}

/// Map a material id to `(perceptual_roughness, metallic, emissive_rgb)`.
///
/// Mirrors the client `HardcodedPalette`
/// (`atomr-worlds-client/src/render/defaults.rs`) so the [`RaymarchTier::Pbr`]
/// CPU twin uses the same PBR inputs the GPU palette buffer carries. Emissive
/// uses the same ×2 upload scale as the GPU palette. Unknown ids fall back to a
/// rough dielectric. This is the PBR analogue of [`material_color`].
pub fn material_pbr(material: u16) -> (f32, f32, [f32; 3]) {
    match material {
        1 => (0.85, 0.0, [0.0; 3]),  // stone
        2 => (0.95, 0.0, [0.0; 3]),  // dirt
        3 => (0.75, 0.0, [0.0; 3]),  // sand
        4 => (0.70, 0.0, [0.0; 3]),  // snow
        5 => (0.05, 0.0, [0.0; 3]),  // water (smooth)
        6 => (0.90, 0.0, [0.0; 3]),  // grass
        7 => (0.85, 0.0, [0.0; 3]),  // wood
        8 => (0.95, 0.0, [0.0; 3]),  // leaves
        9 => (0.50, 0.0, [1.2 * 2.0, 0.8 * 2.0, 0.2 * 2.0]), // glow_rock (emissive)
        10 => (0.10, 0.0, [0.0; 3]), // ice (smooth)
        _ => (1.0, 0.0, [0.0; 3]),
    }
}

pub fn render_brick_png<P: AsRef<Path>>(
    brick: &Brick,
    camera: &Camera,
    cfg: &RenderConfig,
    path: P,
) -> Result<Framebuffer, ViewError> {
    let mesh = greedy_mesh(brick);
    let fb = render_mesh(&mesh, camera, cfg);
    fb.write_png(path)?;
    Ok(fb)
}

pub fn render_mesh(mesh: &Mesh, camera: &Camera, cfg: &RenderConfig) -> Framebuffer {
    let mut fb = Framebuffer {
        width: cfg.width,
        height: cfg.height,
        pixels: Vec::with_capacity((cfg.width * cfg.height * 4) as usize),
        // Reversed-z: clear depth to 0.0 (far) so any drawn fragment wins.
        depth: vec![0.0f32; (cfg.width * cfg.height) as usize],
    };
    let bg = cfg.background;
    for _ in 0..(cfg.width * cfg.height) {
        fb.pixels.extend_from_slice(&bg);
    }

    if mesh.indices.is_empty() {
        return fb;
    }

    let mvp = camera.view_proj();
    let light = norm3(cfg.light_dir);

    // Project all vertices once.
    let mut clip: Vec<[f32; 4]> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        clip.push(transform_point(mvp, v.pos));
    }

    let w_f = cfg.width as f32;
    let h_f = cfg.height as f32;
    let mut tri = 0usize;
    while tri < mesh.indices.len() {
        let i0 = mesh.indices[tri] as usize;
        let i1 = mesh.indices[tri + 1] as usize;
        let i2 = mesh.indices[tri + 2] as usize;
        tri += 3;

        let c0 = clip[i0];
        let c1 = clip[i1];
        let c2 = clip[i2];
        // Cull triangles entirely behind the near plane.
        if c0[3] <= 0.0 && c1[3] <= 0.0 && c2[3] <= 0.0 {
            continue;
        }
        // Cull any triangle with a vertex behind the near plane; skipping
        // clipping keeps this CPU rasterizer simple. The eventual atomr-view
        // bridge will hand this off to wgpu where the GPU handles it.
        if c0[3] <= 0.0 || c1[3] <= 0.0 || c2[3] <= 0.0 {
            continue;
        }

        let s0 = clip_to_screen(c0, w_f, h_f);
        let s1 = clip_to_screen(c1, w_f, h_f);
        let s2 = clip_to_screen(c2, w_f, h_f);

        // Backface cull in screen space.
        let area2 = edge_fn(s0, s1, s2);
        if area2 <= 0.0 {
            continue;
        }

        // Flat shading: use the first vertex's normal + material.
        let v0 = mesh.vertices[i0];
        rasterize_triangle(&mut fb, [s0, s1, s2], area2, &v0, light, cfg.ambient);
    }

    fb
}

fn clip_to_screen(c: [f32; 4], w: f32, h: f32) -> [f32; 3] {
    let inv_w = 1.0 / c[3];
    let ndc_x = c[0] * inv_w;
    let ndc_y = c[1] * inv_w;
    let ndc_z = c[2] * inv_w;
    [(ndc_x * 0.5 + 0.5) * w, (1.0 - (ndc_y * 0.5 + 0.5)) * h, ndc_z]
}

fn edge_fn(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn rasterize_triangle(
    fb: &mut Framebuffer,
    s: [[f32; 3]; 3],
    area2: f32,
    v0: &Vertex,
    light: [f32; 3],
    ambient: f32,
) {
    let w = fb.width as i32;
    let h = fb.height as i32;
    let xmin = s[0][0].min(s[1][0]).min(s[2][0]).floor().max(0.0) as i32;
    let xmax = s[0][0].max(s[1][0]).max(s[2][0]).ceil().min(w as f32 - 1.0) as i32;
    let ymin = s[0][1].min(s[1][1]).min(s[2][1]).floor().max(0.0) as i32;
    let ymax = s[0][1].max(s[1][1]).max(s[2][1]).ceil().min(h as f32 - 1.0) as i32;
    if xmin > xmax || ymin > ymax {
        return;
    }
    let inv_area = 1.0 / area2;
    let base = material_color(v0.material);
    let shade = ambient + (1.0 - ambient) * dot3(v0.normal, light).max(0.0);
    let r = (base[0] as f32 * shade).clamp(0.0, 255.0) as u8;
    let g = (base[1] as f32 * shade).clamp(0.0, 255.0) as u8;
    let b = (base[2] as f32 * shade).clamp(0.0, 255.0) as u8;

    for y in ymin..=ymax {
        for x in xmin..=xmax {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge_fn(s[1], s[2], p);
            let w1 = edge_fn(s[2], s[0], p);
            let w2 = edge_fn(s[0], s[1], p);
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let b0 = w0 * inv_area;
            let b1 = w1 * inv_area;
            let b2 = w2 * inv_area;
            let z = b0 * s[0][2] + b1 * s[1][2] + b2 * s[2][2];
            let idx = (y as u32 * fb.width + x as u32) as usize;
            // Reversed-z: closer fragments have a *larger* z (1 = near, 0 = far).
            if z > fb.depth[idx] && (0.0..=1.0).contains(&z) {
                fb.depth[idx] = z;
                let pi = idx * 4;
                fb.pixels[pi] = r;
                fb.pixels[pi + 1] = g;
                fb.pixels[pi + 2] = b;
                fb.pixels[pi + 3] = 255;
            }
        }
    }
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 13g: composite renderer (skybox + far-ring fade + near ring).
// ─────────────────────────────────────────────────────────────────────────────

/// Fragment shading mode — controls per-pixel alpha + depth interaction.
/// Used by [`render_composite`] to alpha-fade the outer edge of the far
/// ring against the skybox without a hard pop.
#[derive(Copy, Clone, Debug)]
pub enum FragmentMode {
    /// Source pixel replaces the destination unconditionally (the
    /// pre-Phase-13g behavior of `render_mesh`).
    Opaque,
    /// Alpha-fade by world-space distance from `observer`. `alpha = 1`
    /// when the fragment's distance is `<= start_m`; `alpha = 0` when
    /// `>= end_m`; linearly interpolated between. Source-over blend with
    /// the destination color; depth writes only when alpha > 0.5.
    DistanceFade { start_m: f32, end_m: f32, observer: [f32; 3] },
}

/// Composite-rendering inputs: optional skybox + two ring mesh lists.
#[derive(Debug)]
pub struct CompositeScene<'a> {
    /// Cubemap painted before any mesh; samples per-pixel by camera ray.
    pub skybox: Option<&'a Skybox>,
    /// Outer ring meshes. Rasterized with a `DistanceFade` band so the
    /// last `fade_band_frac` of `[transition_radius_m..max_radius_m]`
    /// crossfades into the skybox.
    pub far_meshes: &'a [MeshNode],
    /// Inner ring meshes. Drawn opaque.
    pub near_meshes: &'a [MeshNode],
    /// Observer position (world-space), used by the fade calculation.
    pub observer: [f32; 3],
    /// Inner / outer boundary of the far ring in world meters.
    pub transition_radius_m: f32,
    pub max_radius_m: f32,
    /// Fraction of the band `[transition..max]` over which the fade is
    /// active. Default 0.10 keeps the visible seam narrow.
    pub fade_band_frac: f32,
}

impl<'a> CompositeScene<'a> {
    pub fn new(
        skybox: Option<&'a Skybox>,
        far_meshes: &'a [MeshNode],
        near_meshes: &'a [MeshNode],
        observer: [f32; 3],
        transition_radius_m: f32,
        max_radius_m: f32,
    ) -> Self {
        Self {
            skybox,
            far_meshes,
            near_meshes,
            observer,
            transition_radius_m,
            max_radius_m,
            fade_band_frac: 0.10,
        }
    }
}

/// Composite the three layers — skybox → far meshes (faded) → near
/// meshes (opaque) — into a single framebuffer. Iteration order is
/// fixed; the rasterizer state is a single z-buffer + a single pixel
/// buffer, so output bytes are a pure function of the inputs.
pub fn render_composite(scene: &CompositeScene<'_>, camera: &Camera, cfg: &RenderConfig) -> Framebuffer {
    let mut fb = Framebuffer {
        width: cfg.width,
        height: cfg.height,
        pixels: Vec::with_capacity((cfg.width * cfg.height * 4) as usize),
        depth: vec![0.0f32; (cfg.width * cfg.height) as usize],
    };

    // Step 1: paint the background — skybox if present, otherwise solid.
    if let Some(sky) = scene.skybox {
        fb.pixels = paint_skybox_background(scene.observer, sky, camera, cfg);
    } else {
        let bg = cfg.background;
        for _ in 0..(cfg.width * cfg.height) {
            fb.pixels.extend_from_slice(&bg);
        }
    }

    // Step 2: rasterize the far ring with distance fade. The fade band is
    // the last `fade_band_frac` of `[transition..max]` — outer edge
    // smoothly dissolves into the skybox below.
    let band_width = (scene.max_radius_m - scene.transition_radius_m).max(0.0);
    let fade_start_m = scene.max_radius_m - band_width * scene.fade_band_frac.max(0.0);
    let fade_end_m = scene.max_radius_m;
    for node in scene.far_meshes {
        rasterize_node(
            &mut fb,
            node,
            camera,
            cfg,
            FragmentMode::DistanceFade { start_m: fade_start_m, end_m: fade_end_m, observer: scene.observer },
        );
    }

    // Step 3: rasterize the near ring opaque (the canonical, pre-13g path).
    for node in scene.near_meshes {
        rasterize_node(&mut fb, node, camera, cfg, FragmentMode::Opaque);
    }

    fb
}

/// Paint a skybox into a fresh pixel buffer by tracing each pixel back
/// to a world-space ray direction. The skybox lookup is the cubemap
/// sampler; no depth writes (depth stays at 0.0 / "far" so the mesh
/// passes always win).
fn paint_skybox_background(observer: [f32; 3], sky: &Skybox, camera: &Camera, cfg: &RenderConfig) -> Vec<u8> {
    let w = cfg.width as i32;
    let h = cfg.height as i32;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    let inv_w = 1.0 / w as f32;
    let inv_h = 1.0 / h as f32;
    // Build a deterministic ray basis from the camera. We don't use the
    // projection matrix — the cubemap is direction-only — we just rebuild
    // the forward/right/up basis from `eye → target` and `up`.
    let forward = norm3([
        camera.target[0] - camera.eye[0],
        camera.target[1] - camera.eye[1],
        camera.target[2] - camera.eye[2],
    ]);
    let right = norm3([
        forward[1] * camera.up[2] - forward[2] * camera.up[1],
        forward[2] * camera.up[0] - forward[0] * camera.up[2],
        forward[0] * camera.up[1] - forward[1] * camera.up[0],
    ]);
    let up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ];
    let aspect = camera.aspect.max(1e-6);
    let half_h = (camera.fov_y_rad * 0.5).tan();
    let half_w = half_h * aspect;
    let _ = observer; // skybox sample is direction-only; observer pose lives in `sky.origin`
    for y in 0..h {
        for x in 0..w {
            // NDC in [-1, 1] with +y up.
            let nx = ((x as f32 + 0.5) * inv_w) * 2.0 - 1.0;
            let ny = 1.0 - ((y as f32 + 0.5) * inv_h) * 2.0;
            let dx = right[0] * (nx * half_w) + up[0] * (ny * half_h) + forward[0];
            let dy = right[1] * (nx * half_w) + up[1] * (ny * half_h) + forward[1];
            let dz = right[2] * (nx * half_w) + up[2] * (ny * half_h) + forward[2];
            let rgba = sky.sample([dx, dy, dz]);
            out.extend_from_slice(&rgba);
        }
    }
    out
}

/// Rasterize one mesh node into the framebuffer using the given fragment
/// mode. Mirrors `render_mesh` but (a) accepts pre-allocated framebuffer,
/// (b) applies the node's transform, (c) routes per-fragment shading
/// through `mode`.
fn rasterize_node(
    fb: &mut Framebuffer,
    node: &MeshNode,
    camera: &Camera,
    cfg: &RenderConfig,
    mode: FragmentMode,
) {
    let mesh = &node.mesh;
    if mesh.indices.is_empty() {
        return;
    }
    let mvp = camera.view_proj();
    let light = norm3(cfg.light_dir);
    let t = node.transform;
    let world_pos: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| transform_point_affine(t, v.pos)).collect();
    let clip: Vec<[f32; 4]> = world_pos.iter().map(|p| transform_point(mvp, *p)).collect();

    let w_f = cfg.width as f32;
    let h_f = cfg.height as f32;
    let mut tri = 0usize;
    while tri < mesh.indices.len() {
        let i0 = mesh.indices[tri] as usize;
        let i1 = mesh.indices[tri + 1] as usize;
        let i2 = mesh.indices[tri + 2] as usize;
        tri += 3;
        let c0 = clip[i0];
        let c1 = clip[i1];
        let c2 = clip[i2];
        if c0[3] <= 0.0 || c1[3] <= 0.0 || c2[3] <= 0.0 {
            continue;
        }
        let s0 = clip_to_screen(c0, w_f, h_f);
        let s1 = clip_to_screen(c1, w_f, h_f);
        let s2 = clip_to_screen(c2, w_f, h_f);
        let area2 = edge_fn(s0, s1, s2);
        if area2 <= 0.0 {
            continue;
        }
        // Pre-compute per-vertex world distances for the fade band.
        let v0 = &mesh.vertices[i0];
        let n_world = transform_dir(t, v0.normal);
        let d0 = world_distance(
            world_pos[i0],
            match mode {
                FragmentMode::DistanceFade { observer, .. } => observer,
                FragmentMode::Opaque => [0.0; 3],
            },
        );
        let d1 = world_distance(
            world_pos[i1],
            match mode {
                FragmentMode::DistanceFade { observer, .. } => observer,
                FragmentMode::Opaque => [0.0; 3],
            },
        );
        let d2 = world_distance(
            world_pos[i2],
            match mode {
                FragmentMode::DistanceFade { observer, .. } => observer,
                FragmentMode::Opaque => [0.0; 3],
            },
        );

        rasterize_triangle_mode(
            fb,
            [s0, s1, s2],
            [d0, d1, d2],
            area2,
            v0.material,
            n_world,
            light,
            cfg.ambient,
            mode,
        );
    }
}

#[inline]
fn world_distance(p: [f32; 3], o: [f32; 3]) -> f32 {
    let dx = p[0] - o[0];
    let dy = p[1] - o[1];
    let dz = p[2] - o[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn transform_point_affine(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

#[inline]
fn transform_dir(m: [[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

#[allow(clippy::too_many_arguments)]
fn rasterize_triangle_mode(
    fb: &mut Framebuffer,
    s: [[f32; 3]; 3],
    dists: [f32; 3],
    area2: f32,
    material: u16,
    normal: [f32; 3],
    light: [f32; 3],
    ambient: f32,
    mode: FragmentMode,
) {
    let w = fb.width as i32;
    let h = fb.height as i32;
    let xmin = s[0][0].min(s[1][0]).min(s[2][0]).floor().max(0.0) as i32;
    let xmax = s[0][0].max(s[1][0]).max(s[2][0]).ceil().min(w as f32 - 1.0) as i32;
    let ymin = s[0][1].min(s[1][1]).min(s[2][1]).floor().max(0.0) as i32;
    let ymax = s[0][1].max(s[1][1]).max(s[2][1]).ceil().min(h as f32 - 1.0) as i32;
    if xmin > xmax || ymin > ymax {
        return;
    }
    let inv_area = 1.0 / area2;
    let base = material_color(material);
    let shade = ambient + (1.0 - ambient) * dot3(normal, light).max(0.0);
    let sr = (base[0] as f32 * shade).clamp(0.0, 255.0);
    let sg = (base[1] as f32 * shade).clamp(0.0, 255.0);
    let sb = (base[2] as f32 * shade).clamp(0.0, 255.0);

    for y in ymin..=ymax {
        for x in xmin..=xmax {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge_fn(s[1], s[2], p);
            let w1 = edge_fn(s[2], s[0], p);
            let w2 = edge_fn(s[0], s[1], p);
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let b0 = w0 * inv_area;
            let b1 = w1 * inv_area;
            let b2 = w2 * inv_area;
            let z = b0 * s[0][2] + b1 * s[1][2] + b2 * s[2][2];
            let idx = (y as u32 * fb.width + x as u32) as usize;
            if !(0.0..=1.0).contains(&z) {
                continue;
            }
            let alpha = match mode {
                FragmentMode::Opaque => 1.0_f32,
                FragmentMode::DistanceFade { start_m, end_m, .. } => {
                    let d = b0 * dists[0] + b1 * dists[1] + b2 * dists[2];
                    if d <= start_m {
                        1.0
                    } else if d >= end_m {
                        0.0
                    } else {
                        1.0 - (d - start_m) / (end_m - start_m).max(1e-6)
                    }
                }
            };
            if alpha <= 0.0 {
                continue;
            }
            // Z-test under reversed-z.
            if z <= fb.depth[idx] {
                continue;
            }
            let pi = idx * 4;
            let dr = fb.pixels[pi] as f32;
            let dg = fb.pixels[pi + 1] as f32;
            let db = fb.pixels[pi + 2] as f32;
            let or = dr * (1.0 - alpha) + sr * alpha;
            let og = dg * (1.0 - alpha) + sg * alpha;
            let ob = db * (1.0 - alpha) + sb * alpha;
            fb.pixels[pi] = or.clamp(0.0, 255.0) as u8;
            fb.pixels[pi + 1] = og.clamp(0.0, 255.0) as u8;
            fb.pixels[pi + 2] = ob.clamp(0.0, 255.0) as u8;
            fb.pixels[pi + 3] = 255;
            // Z-write only for visually-solid fragments — fade-out
            // fragments don't occlude near-ring pixels.
            if alpha > 0.5 {
                fb.depth[idx] = z;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU DAG raymarcher CPU twin (determinism gate).
// ─────────────────────────────────────────────────────────────────────────────

/// Shading tier for [`render_raymarch`].
///
/// CPU twin of the client's `render::raymarch::RaymarchShadingTier`, kept
/// separate to avoid inverting the crate dep graph. Each tier mirrors the WGSL
/// `shade()` branch of the same name. Shading math is hash-exempt (the golden
/// pins [`Unlit`](RaymarchTier::Unlit)); the twin exists so the lit tiers — and
/// the DAG-occupancy sampling the PBR tier adds — stay testable deterministically.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RaymarchTier {
    /// Flat material color, no lighting term. The golden pins this tier.
    Unlit,
    /// Single directional `n·l` term over a fixed ambient floor.
    Lambert,
    /// Cook-Torrance PBR: GGX specular from the material's roughness/metallic,
    /// ambient occlusion from local DAG occupancy, and a brick-local sun
    /// self-shadow — over the same ambient floor as [`Lambert`](Self::Lambert).
    Pbr,
}

/// CPU reference render of a single brick's [`DagGpu`] via per-pixel ray-DDA —
/// the deterministic mirror of the client's GPU DAG fragment raymarcher and the
/// determinism gate for that path.
///
/// **Placement:** the brick is at the ORIGIN with identity placement, so it
/// occupies world `[0, 16)³` and brick-local voxel space == world space. There
/// is no model matrix to invert; the camera eye and per-pixel ray direction are
/// already in voxel space and are handed straight to [`ray_dda_first_hit`].
///
/// **Ray construction** reuses the exact forward/right/up + fov/aspect/NDC math
/// from [`paint_skybox_background`] so the two stay consistent. Misses leave the
/// `cfg.background` clear color in place (same as `render_mesh`). The depth
/// buffer is left at the cleared `0.0` (far): the GPU path's reversed-Z depth is
/// driver-divergent and hash-exempt, so the golden compares color only.
pub fn render_raymarch(dag: &DagGpu, camera: &Camera, cfg: &RenderConfig, tier: RaymarchTier) -> Framebuffer {
    let mut fb = Framebuffer {
        width: cfg.width,
        height: cfg.height,
        pixels: Vec::with_capacity((cfg.width * cfg.height * 4) as usize),
        depth: vec![0.0f32; (cfg.width * cfg.height) as usize],
    };
    let bg = cfg.background;
    for _ in 0..(cfg.width * cfg.height) {
        fb.pixels.extend_from_slice(&bg);
    }

    // Camera ray basis — identical construction to `paint_skybox_background`.
    let forward = norm3([
        camera.target[0] - camera.eye[0],
        camera.target[1] - camera.eye[1],
        camera.target[2] - camera.eye[2],
    ]);
    let right = norm3([
        forward[1] * camera.up[2] - forward[2] * camera.up[1],
        forward[2] * camera.up[0] - forward[0] * camera.up[2],
        forward[0] * camera.up[1] - forward[1] * camera.up[0],
    ]);
    let up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ];
    let aspect = camera.aspect.max(1e-6);
    let half_h = (camera.fov_y_rad * 0.5).tan();
    let half_w = half_h * aspect;

    // Identity placement: eye is already in voxel space.
    let eye = camera.eye;
    let light = norm3(cfg.light_dir);

    let w = cfg.width as i32;
    let h = cfg.height as i32;
    let inv_w = 1.0 / w as f32;
    let inv_h = 1.0 / h as f32;
    // Fixed `for x in 0..` iteration order over a single pixel buffer ⇒ output
    // bytes are a pure function of the inputs (the determinism contract).
    for y in 0..h {
        for x in 0..w {
            // NDC in [-1, 1] with +y up (matches the skybox path).
            let nx = ((x as f32 + 0.5) * inv_w) * 2.0 - 1.0;
            let ny = 1.0 - ((y as f32 + 0.5) * inv_h) * 2.0;
            let dir = [
                right[0] * (nx * half_w) + up[0] * (ny * half_h) + forward[0],
                right[1] * (nx * half_w) + up[1] * (ny * half_h) + forward[1],
                right[2] * (nx * half_w) + up[2] * (ny * half_h) + forward[2],
            ];
            // Identity placement ⇒ eye and dir are already in voxel space.
            let Some(hit) = ray_dda_first_hit(dag, eye, dir) else {
                continue; // miss: leave the background clear color
            };
            let base = material_color(hit.material);
            let rgb = match tier {
                // Golden pins UNLIT: returns material_color only (no
                // light/ambient) so it's robust to float drift. NOTE: GPU
                // `shade()` reads LINEAR palette base_color; this uses sRGB
                // material_color — intentionally NOT byte-comparable to the GPU;
                // the GPU path is hash-exempt. Do not 'fix' this.
                RaymarchTier::Unlit => base,
                // base*(0.30 + 0.70*ndl) — mirrors the WGSL Lambert formula.
                RaymarchTier::Lambert => {
                    let ndl = dot3(hit.normal, light).max(0.0);
                    let shade = 0.30 + 0.70 * ndl;
                    [
                        (base[0] as f32 * shade).clamp(0.0, 255.0) as u8,
                        (base[1] as f32 * shade).clamp(0.0, 255.0) as u8,
                        (base[2] as f32 * shade).clamp(0.0, 255.0) as u8,
                    ]
                }
                // Cook-Torrance PBR — mirrors the WGSL TIER_PBR branch. View dir
                // is surface→camera = -normalize(ray dir). Same sRGB-base caveat
                // as Lambert: NOT byte-comparable to the GPU (which shades in
                // linear), and the golden therefore stays on Unlit.
                RaymarchTier::Pbr => {
                    let v = norm3([-dir[0], -dir[1], -dir[2]]);
                    let (roughness, metal, emissive) = material_pbr(hit.material);
                    pbr_shade(dag, &hit, light, v, base, roughness, metal, emissive)
                }
            };
            let idx = (y as u32 * fb.width + x as u32) as usize;
            let pi = idx * 4;
            fb.pixels[pi] = rgb[0];
            fb.pixels[pi + 1] = rgb[1];
            fb.pixels[pi + 2] = rgb[2];
            fb.pixels[pi + 3] = 255;
        }
    }

    fb
}

// ─────────────────────────────────────────────────────────────────────────────
// PBR shading — CPU mirror of voxel_raymarch.wgsl `shade()`'s TIER_PBR branch.
//
// Constants and formulas are a line-for-line port of the WGSL. Math runs in
// normalized [0, 1] space (base color converted from the sRGB-ish u8
// `material_color`), then scaled back to u8 — so it is NOT byte-comparable to
// the GPU (which shades in linear), exactly like the Lambert twin. The point is
// behavioural parity (AO darkens crevices, overhangs self-shadow, smoother
// materials get sharper highlights), tested deterministically below.
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed ambient floor — shared with Lambert; `1 - AMBIENT` is the direct weight.
const RM_AMBIENT: f32 = 0.30;
/// How strongly local occupancy darkens the ambient term (`voxel_raymarch.wgsl`).
const RM_AO_STRENGTH: f32 = 0.7;
/// Self-shadow march: half-voxel steps, one brick edge of reach (`32 * 0.5`).
const RM_SHADOW_STEP: f32 = 0.5;
const RM_SHADOW_MAX_STEPS: i32 = 32;
const RM_EDGE: i32 = BRICK_EDGE as i32;

#[inline]
fn rm_solid_at(dag: &DagGpu, c: [i32; 3]) -> bool {
    if c.iter().any(|&v| !(0..RM_EDGE).contains(&v)) {
        return false; // out of brick = air (brick-local AO/shadow only)
    }
    gpu_get(dag, c[0] as u8, c[1] as u8, c[2] as u8) != Voxel::EMPTY
}

/// Ambient occlusion from the 8-neighbour ring (in the lit face's tangent plane)
/// around the air cell in front of the hit. Mirror of WGSL `ao_from_occupancy`.
fn rm_ao(dag: &DagGpu, cell: [i32; 3], ni: [i32; 3]) -> f32 {
    let air = [cell[0] + ni[0], cell[1] + ni[1], cell[2] + ni[2]];
    let (t1, t2): ([i32; 3], [i32; 3]) = if ni[0].abs() > 0 {
        ([0, 1, 0], [0, 0, 1])
    } else if ni[1].abs() > 0 {
        ([1, 0, 0], [0, 0, 1])
    } else {
        ([1, 0, 0], [0, 1, 0])
    };
    let mut occ = 0.0f32;
    for di in -1..=1 {
        for dj in -1..=1 {
            if di == 0 && dj == 0 {
                continue;
            }
            let c = [
                air[0] + t1[0] * di + t2[0] * dj,
                air[1] + t1[1] * di + t2[1] * dj,
                air[2] + t1[2] * di + t2[2] * dj,
            ];
            if rm_solid_at(dag, c) {
                occ += 1.0;
            }
        }
    }
    1.0 - RM_AO_STRENGTH * (occ / 8.0)
}

/// Brick-local hard self-shadow: point-march toward the sun from the air cell in
/// front of the lit face. 0 = occluded, 1 = lit. Mirror of WGSL `sun_shadow`.
fn rm_shadow(dag: &DagGpu, cell: [i32; 3], ni: [i32; 3], l: [f32; 3]) -> f32 {
    let edge = RM_EDGE as f32;
    let mut p = [
        cell[0] as f32 + 0.5 + ni[0] as f32,
        cell[1] as f32 + 0.5 + ni[1] as f32,
        cell[2] as f32 + 0.5 + ni[2] as f32,
    ];
    for _ in 0..RM_SHADOW_MAX_STEPS {
        p[0] += l[0] * RM_SHADOW_STEP;
        p[1] += l[1] * RM_SHADOW_STEP;
        p[2] += l[2] * RM_SHADOW_STEP;
        if p.iter().any(|&v| v < 0.0 || v >= edge) {
            return 1.0; // left the brick without an occluder
        }
        let c = [p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32];
        if rm_solid_at(dag, c) {
            return 0.0;
        }
    }
    1.0
}

#[inline]
fn rm_distribution_ggx(ndh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * d * d).max(1e-7)
}

#[inline]
fn rm_geometry_schlick(nd: f32, k: f32) -> f32 {
    nd / (nd * (1.0 - k) + k).max(1e-7)
}

#[inline]
fn rm_geometry_smith(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    let r1 = roughness + 1.0;
    let k = (r1 * r1) / 8.0;
    rm_geometry_schlick(ndv, k) * rm_geometry_schlick(ndl, k)
}

#[inline]
fn rm_fresnel(cos_theta: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(0.0, 1.0).powi(5)
}

/// Cook-Torrance shade of one hit. CPU mirror of the WGSL `shade()` TIER_PBR
/// branch (see the module-level note on the deliberate sRGB-vs-linear divergence).
#[allow(clippy::too_many_arguments)]
fn pbr_shade(
    dag: &DagGpu,
    hit: &RayHit,
    l: [f32; 3],
    v: [f32; 3],
    base_u8: [u8; 3],
    roughness: f32,
    metal: f32,
    emissive: [f32; 3],
) -> [u8; 3] {
    let n = hit.normal;
    let ni = [
        n[0].round() as i32,
        n[1].round() as i32,
        n[2].round() as i32,
    ];
    let ndl = dot3(n, l).max(0.0);
    let ndv = dot3(n, v).max(1e-4);
    let h = norm3([l[0] + v[0], l[1] + v[1], l[2] + v[2]]);
    let ndh = dot3(n, h).max(0.0);
    let vdh = dot3(v, h).max(0.0);

    let roughness = roughness.clamp(0.045, 1.0);
    let metal = metal.clamp(0.0, 1.0);
    let a = roughness * roughness;
    let base01 = [
        base_u8[0] as f32 / 255.0,
        base_u8[1] as f32 / 255.0,
        base_u8[2] as f32 / 255.0,
    ];

    let d_term = rm_distribution_ggx(ndh, a);
    let g_term = rm_geometry_smith(ndv, ndl, roughness);
    let ao = rm_ao(dag, hit.cell, ni);
    let shadow = if ndl > 0.0 {
        rm_shadow(dag, hit.cell, ni, l)
    } else {
        1.0
    };

    let mut out = [0u8; 3];
    for i in 0..3 {
        let f0 = 0.04 * (1.0 - metal) + base01[i] * metal;
        let f_term = rm_fresnel(vdh, f0);
        let spec_brdf = (d_term * g_term * f_term) / (4.0 * ndv * ndl + 1e-4);
        let diffuse = base01[i] * (1.0 - metal) * (RM_AMBIENT * ao + (1.0 - RM_AMBIENT) * ndl * shadow);
        let specular = spec_brdf * ndl * shadow * (1.0 - RM_AMBIENT);
        let lit = diffuse + specular + emissive[i];
        out[i] = (lit * 255.0).clamp(0.0, 255.0) as u8;
    }
    out
}

#[cfg(test)]
mod pbr_tests {
    //! Behavioural property tests for the [`RaymarchTier::Pbr`] CPU twin (the
    //! mirror of the WGSL `shade()` TIER_PBR branch). These assert *relative*
    //! shading behaviour — AO darkens crevices, overhangs self-shadow, smoother
    //! materials get sharper highlights — not exact bytes (PBR float math makes a
    //! pinned hash fragile; the byte-determinism golden stays on `Unlit`).

    use super::*;
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_voxel::{DagBrick, RayHit};

    fn gpu(brick: &Brick) -> DagGpu {
        DagBrick::from_brick(brick).to_gpu()
    }

    fn solid(brick: &mut Brick, p: [i32; 3], mat: u16) {
        brick.set(IVec3::new(p[0] as i64, p[1] as i64, p[2] as i64), Voxel::new(mat));
    }

    /// AO darkens a hit whose surrounding air cell is walled in (inside corner)
    /// relative to a hit with an open face.
    #[test]
    fn ao_darkens_enclosed_corner() {
        let ni = [0, 1, 0]; // +y top face
        let cell = [8, 0, 8];
        let air = [8, 1, 8];

        // Exposed: only the hit voxel is solid → no occlusion.
        let mut exposed = Brick::new();
        solid(&mut exposed, cell, 1);
        let ao_open = rm_ao(&gpu(&exposed), cell, ni);

        // Enclosed: the 8-neighbour ring around the air cell (xz plane) is solid.
        let mut enclosed = exposed.clone();
        for dx in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                solid(&mut enclosed, [air[0] + dx, air[1], air[2] + dz], 1);
            }
        }
        let ao_closed = rm_ao(&gpu(&enclosed), cell, ni);

        assert!((ao_open - 1.0).abs() < 1e-6, "open face = full ambient, got {ao_open}");
        assert!(ao_closed < ao_open, "enclosed corner must be darker: {ao_closed} !< {ao_open}");
        // 8/8 ring solid → 1 - 0.7 = 0.30.
        assert!((ao_closed - 0.30).abs() < 1e-6, "fully ringed AO should be 0.30, got {ao_closed}");
    }

    /// A voxel under an overhang is shadowed from a straight-overhead sun; an
    /// open voxel is lit.
    #[test]
    fn overhang_casts_self_shadow() {
        let ni = [0, 1, 0];
        let cell = [8, 0, 8];
        let sun_up = norm3([0.0, 1.0, 0.0]);

        let mut open = Brick::new();
        solid(&mut open, cell, 1);
        assert_eq!(rm_shadow(&gpu(&open), cell, ni, sun_up), 1.0, "open voxel is lit");

        let mut roofed = open.clone();
        solid(&mut roofed, [8, 5, 8], 1); // occluder directly above
        assert_eq!(rm_shadow(&gpu(&roofed), cell, ni, sun_up), 0.0, "roofed voxel is shadowed");
    }

    /// At the mirror angle, a smoother (lower-roughness) material produces a
    /// brighter specular highlight than a rough one (same base/metal/AO/shadow).
    #[test]
    fn smoother_material_has_sharper_highlight() {
        let mut brick = Brick::new();
        solid(&mut brick, [8, 8, 8], 1);
        let dag = gpu(&brick);
        let hit = RayHit {
            cell: [8, 8, 8],
            material: 1,
            t_entry: 0.0,
            enter_axis: 1,
            normal: [0.0, 1.0, 0.0],
        };
        // l and v symmetric about the +y normal → half-vector ≈ normal (peak D).
        let l = norm3([0.3, 0.95, 0.0]);
        let v = norm3([-0.3, 0.95, 0.0]);
        let base = [128, 128, 128];

        let smooth = pbr_shade(&dag, &hit, l, v, base, 0.08, 0.0, [0.0; 3]);
        let rough = pbr_shade(&dag, &hit, l, v, base, 0.85, 0.0, [0.0; 3]);
        let lum = |c: [u8; 3]| c[0] as u32 + c[1] as u32 + c[2] as u32;
        assert!(
            lum(smooth) > lum(rough),
            "smooth highlight {smooth:?} must outshine rough {rough:?} at the mirror angle"
        );
    }

    /// The Pbr tier is no longer the Lambert stub: rendering the same brick under
    /// each tier must produce different pixels.
    #[test]
    fn pbr_differs_from_lambert() {
        let mut brick = Brick::new();
        let edge = BRICK_EDGE as i32;
        for z in 0..edge {
            for y in 0..(edge / 2) {
                for x in 0..edge {
                    solid(&mut brick, [x, y, z], 3);
                }
            }
        }
        let dag = gpu(&brick);
        let cam = Camera::isometric_default(1.0);
        let cfg = RenderConfig { width: 64, height: 64, ..Default::default() };
        let lambert = render_raymarch(&dag, &cam, &cfg, RaymarchTier::Lambert);
        let pbr = render_raymarch(&dag, &cam, &cfg, RaymarchTier::Pbr);
        assert_ne!(lambert.pixels, pbr.pixels, "Pbr must no longer equal Lambert");
    }
}
