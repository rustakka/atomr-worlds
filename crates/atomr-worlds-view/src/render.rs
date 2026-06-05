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

use atomr_worlds_voxel::{ray_dda_first_hit, Brick, DagGpu};

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
/// CPU twin of the client's `render::raymarch::RaymarchShadingTier`; Pbr folds
/// into Lambert as the shader does. Kept separate to avoid inverting the crate
/// dep graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RaymarchTier {
    /// Flat material color, no lighting term. The golden pins this tier.
    Unlit,
    /// Single directional `n·l` term over a fixed ambient floor.
    Lambert,
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
