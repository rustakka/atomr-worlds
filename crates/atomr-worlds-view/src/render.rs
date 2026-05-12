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

use atomr_worlds_voxel::Brick;

use crate::camera::{transform_point, Camera};
use crate::mesh::{greedy_mesh, Mesh, Vertex};
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
#[derive(Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA8
    pub depth: Vec<f32>, // 1.0 = far, 0.0 = near (already z/w)
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
pub fn material_color(material: u16) -> [u8; 3] {
    match material {
        1 => [120, 110, 100], // stone
        2 => [100, 140, 70],  // dirt
        3 => [200, 180, 130], // sand (reserved)
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
        depth: vec![1.0f32; (cfg.width * cfg.height) as usize],
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
            if z < fb.depth[idx] && (0.0..=1.0).contains(&z) {
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
