//! Cubemap skybox: a six-face capture of the world around a fixed observer.
//!
//! Phase 13f ships the cubemap data type plus a mesh-input rasterizer pipeline
//! that fills its six faces by rendering the same `MeshNode` slab from six
//! `Camera::for_cube_face` viewpoints. A higher-tier wrapper that pulls the
//! parent-tier brick slab from a `WorldHost` is intentionally deferred to
//! Phase 13g/13i — keeping 13f testable in isolation means the renderer takes
//! a slice of `MeshNode`s and nothing else.
//!
//! **Coordinate convention.** Right-handed, +Y up — same as the rest of the
//! renderer. The six faces follow the OpenGL / Vulkan / DirectX cube-map
//! convention: `forward` is the outward face normal (camera looks down the
//! face's outward direction), `right` and `up` span the face plane. The basis
//! is orthonormal and right-handed: `cross(right, up) == forward`.
//!
//! **Sampling.** [`Skybox::sample`] uses the standard "largest absolute
//! component picks the face" rule, then projects the remaining two components
//! into face UVs in `[0, 1]`. The sample is scale-invariant: `sample(dir) ==
//! sample(k * dir)` for any `k > 0`.
//!
//! **Determinism.** Every output byte is a pure function of the inputs: the
//! mesh data, observer position, near/far, face resolution, and background
//! color. No `Instant::now()`, no `HashMap` iteration, no parallelism.

use crate::camera::Camera;
use crate::render::{render_mesh, Framebuffer, RenderConfig};
use crate::scene::MeshNode;

/// Number of faces in a cube map.
pub const CUBE_FACE_COUNT: usize = 6;

/// One face of a cubemap, identified by its outward-pointing axis.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum CubeFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl CubeFace {
    /// All six faces in a stable, documented order. Iteration over `ALL`
    /// matches the index used by [`Skybox::faces`].
    pub const ALL: [CubeFace; 6] = [
        CubeFace::PosX,
        CubeFace::NegX,
        CubeFace::PosY,
        CubeFace::NegY,
        CubeFace::PosZ,
        CubeFace::NegZ,
    ];

    /// Outward face normal (the direction the camera looks).
    pub fn forward(self) -> [f32; 3] {
        match self {
            CubeFace::PosX => [1.0, 0.0, 0.0],
            CubeFace::NegX => [-1.0, 0.0, 0.0],
            CubeFace::PosY => [0.0, 1.0, 0.0],
            CubeFace::NegY => [0.0, -1.0, 0.0],
            CubeFace::PosZ => [0.0, 0.0, 1.0],
            CubeFace::NegZ => [0.0, 0.0, -1.0],
        }
    }

    /// "Up" axis on the face. For PosY / NegY (looking straight up or down),
    /// the up axis is chosen so the rotation matches the standard cubemap
    /// convention (PosY looks toward -Z, NegY looks toward +Z).
    pub fn up(self) -> [f32; 3] {
        match self {
            CubeFace::PosX => [0.0, 1.0, 0.0],
            CubeFace::NegX => [0.0, 1.0, 0.0],
            CubeFace::PosY => [0.0, 0.0, -1.0],
            CubeFace::NegY => [0.0, 0.0, 1.0],
            CubeFace::PosZ => [0.0, 1.0, 0.0],
            CubeFace::NegZ => [0.0, 1.0, 0.0],
        }
    }

    /// In-plane "right" axis, chosen so `cross(right, up) == forward`. The
    /// basis is then a right-handed orthonormal frame on the face.
    pub fn right(self) -> [f32; 3] {
        let f = self.forward();
        let u = self.up();
        // right = cross(up, forward) — that makes the frame right-handed
        // with cross(right, up) = forward.
        [u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0]]
    }

    /// Zero-based index used by [`Skybox::faces`] (`PosX = 0` … `NegZ = 5`).
    pub fn index(self) -> usize {
        match self {
            CubeFace::PosX => 0,
            CubeFace::NegX => 1,
            CubeFace::PosY => 2,
            CubeFace::NegY => 3,
            CubeFace::PosZ => 4,
            CubeFace::NegZ => 5,
        }
    }
}

/// A single cubemap face: square RGBA8 image with `width * height * 4` bytes.
#[derive(Clone)]
pub struct CubeFaceImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl std::fmt::Debug for CubeFaceImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeFaceImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pixels_len", &self.pixels.len())
            .finish()
    }
}

impl CubeFaceImage {
    /// Solid-color face image of the given dimensions.
    pub fn filled(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&rgba);
        }
        Self { width, height, pixels }
    }

    /// Pixel-RGBA reader; out-of-range returns the all-zero byte.
    pub fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 0];
        }
        let i = ((y * self.width + x) * 4) as usize;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2], self.pixels[i + 3]]
    }
}

/// Cubemap capture of the world around an observer at the time the skybox was
/// built.
///
/// `origin` stores the (DVec3-shaped) observer position so a downstream
/// consumer can decide when to invalidate the skybox (e.g. the observer has
/// moved more than a few percent of `inner_radius_m`). `inner_radius_m` /
/// `outer_radius_m` describe the spherical shell the cubemap represents —
/// content inside `inner_radius_m` is *not* in the skybox (it's the near-field
/// terrain rendered every frame), content past `outer_radius_m` is at "infinite
/// distance" and shows up here as background.
///
/// `digest` is an FNV-1a hash over the concatenated face pixel buffers,
/// suitable as a screenshot regression fingerprint and as a cheap "did
/// anything change?" check.
#[derive(Clone)]
pub struct Skybox {
    pub faces: [CubeFaceImage; 6],
    pub origin: [f64; 3],
    pub inner_radius_m: f64,
    pub outer_radius_m: f64,
    pub captured_seed: u64,
    pub face_resolution: u32,
    pub digest: u64,
}

impl std::fmt::Debug for Skybox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Skybox")
            .field("origin", &self.origin)
            .field("inner_radius_m", &self.inner_radius_m)
            .field("outer_radius_m", &self.outer_radius_m)
            .field("captured_seed", &self.captured_seed)
            .field("face_resolution", &self.face_resolution)
            .field("digest", &format_args!("{:#018x}", self.digest))
            .finish()
    }
}

impl Skybox {
    /// Sample the cubemap along `dir_unit` (does not need to be unit-length —
    /// sampling is scale-invariant). The largest-magnitude axis component
    /// picks the face; the remaining two project into face UVs in `[0, 1]`.
    pub fn sample(&self, dir_unit: [f32; 3]) -> [u8; 4] {
        let (face, u, v) = face_and_uv(dir_unit);
        let img = &self.faces[face.index()];
        let x =
            ((u * img.width as f32).floor().clamp(0.0, img.width as f32 - 1.0)) as u32;
        let y =
            ((v * img.height as f32).floor().clamp(0.0, img.height as f32 - 1.0)) as u32;
        img.texel(x, y)
    }

    /// FNV-1a over all six face pixel buffers, concatenated in
    /// [`CubeFace::ALL`] order. Recomputed on demand; equal to `self.digest`
    /// when the skybox was built via [`render_skybox_from_meshes`].
    pub fn compute_digest(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for face in &self.faces {
            for b in &face.pixels {
                h ^= *b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }
}

/// Knobs for [`render_skybox_from_meshes`].
#[derive(Copy, Clone, Debug)]
pub struct SkyboxConfig {
    /// Edge length of each cube face in pixels. Default 128 (so the full
    /// cubemap is 128² × 6 = 98 304 RGBA texels — small enough to memoize
    /// per-observer pose, big enough to read terrain silhouettes against).
    pub face_resolution: u32,
    /// Background color used for empty cube-face pixels (everything past the
    /// world's edge or behind the camera).
    pub background_color: [u8; 4],
    /// Soft hint: if the host can supply a parent-tier mesh slab, include it
    /// when computing the skybox. Phase 13f ignores this flag because the
    /// host-pulling wrapper is deferred to 13g/13i; downstream code that
    /// builds the mesh slab is free to honor it.
    pub include_parent_tier: bool,
}

impl Default for SkyboxConfig {
    fn default() -> Self {
        Self {
            face_resolution: 128,
            background_color: [10, 12, 22, 255],
            include_parent_tier: true,
        }
    }
}

/// Render a cubemap from a slice of [`MeshNode`]s.
///
/// Each cube face is rendered by `render_mesh` over every node in `meshes`
/// with a camera placed at `observer` looking down that face's outward axis.
/// The mesh data is consumed read-only — multi-face rendering is safe because
/// the rasterizer's only state is the framebuffer it produces.
pub fn render_skybox_from_meshes(
    meshes: &[MeshNode],
    observer: [f64; 3],
    inner_radius_m: f64,
    outer_radius_m: f64,
    captured_seed: u64,
    cfg: &SkyboxConfig,
) -> Skybox {
    let res = cfg.face_resolution.max(1);
    let eye = [observer[0] as f32, observer[1] as f32, observer[2] as f32];

    // Near/far: the inner radius bounds the smallest distance a vertex can be
    // from the observer in the parent-tier slab; the outer radius bounds the
    // largest. We pad the near a touch so degenerate triangles at the exact
    // shell boundary don't get clipped, and clamp far to a sane minimum so
    // a zero-radius shell still produces a valid projection.
    let near = (inner_radius_m as f32).max(0.01) * 0.1;
    let far = (outer_radius_m as f32).max(near * 4.0);

    let render_cfg = RenderConfig {
        width: res,
        height: res,
        background: cfg.background_color,
        ..RenderConfig::default()
    };

    // Build the six faces in a fixed order. Using `Option` lets us emit the
    // array literal at the end without any unsafe.
    let mut slots: [Option<CubeFaceImage>; 6] = [None, None, None, None, None, None];
    for face in CubeFace::ALL {
        let cam = Camera::for_cube_face(eye, face, near, far);
        let fb = render_meshes_into(meshes, &cam, &render_cfg);
        slots[face.index()] = Some(CubeFaceImage {
            width: fb.width,
            height: fb.height,
            pixels: fb.pixels,
        });
    }
    // The loop populated every slot; unwrap is infallible here. We extract
    // each Option in a deterministic order so the digest is stable.
    let faces: [CubeFaceImage; 6] = [
        slots[0].take().expect("PosX face populated"),
        slots[1].take().expect("NegX face populated"),
        slots[2].take().expect("PosY face populated"),
        slots[3].take().expect("NegY face populated"),
        slots[4].take().expect("PosZ face populated"),
        slots[5].take().expect("NegZ face populated"),
    ];

    let mut sky = Skybox {
        faces,
        origin: observer,
        inner_radius_m,
        outer_radius_m,
        captured_seed,
        face_resolution: res,
        digest: 0,
    };
    sky.digest = sky.compute_digest();
    sky
}

/// Render every node in `meshes` into a single framebuffer using the same
/// camera. Each node's `transform` is applied via vertex pre-baking; we
/// allocate a temporary translated [`crate::mesh::Mesh`] for each node so the
/// existing `render_mesh` entry point (which doesn't know about
/// transformations) keeps its small surface.
fn render_meshes_into(
    meshes: &[MeshNode],
    camera: &Camera,
    cfg: &RenderConfig,
) -> Framebuffer {
    if meshes.is_empty() {
        return render_mesh(&crate::mesh::Mesh::default(), camera, cfg);
    }

    // Start with the first mesh so we get a framebuffer of the right size,
    // then composite subsequent meshes by feeding back the depth buffer would
    // be ideal — but `render_mesh` always starts fresh. For 13f we restrict
    // to a single combined mesh built per face.
    //
    // Building one merged mesh per face preserves all rasterizer state in a
    // single call, including the z-buffer compare and back-face cull. This is
    // strictly less general than a multi-pass renderer but is the right shape
    // for a static skybox capture.
    let combined = combine_meshes(meshes);
    render_mesh(&combined, camera, cfg)
}

/// Combine a slice of [`MeshNode`]s into one [`crate::mesh::Mesh`], applying
/// each node's `transform` to its vertex positions (and the rotational part to
/// its normals). The output is suitable for a single `render_mesh` call.
fn combine_meshes(meshes: &[MeshNode]) -> crate::mesh::Mesh {
    let mut out = crate::mesh::Mesh::default();
    for node in meshes {
        let t = node.transform;
        let base = out.vertices.len() as u32;
        for v in &node.mesh.vertices {
            let p = transform_point_affine(t, v.pos);
            let n = transform_dir(t, v.normal);
            out.vertices.push(crate::mesh::Vertex { pos: p, normal: n, material: v.material });
        }
        for idx in &node.mesh.indices {
            out.indices.push(*idx + base);
        }
    }
    out
}

/// Affine transform of a 3-vector by a column-major 4×4 matrix. The matrix
/// layout here matches `look_at` in `camera.rs`: `m[col][row]`.
fn transform_point_affine(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

/// Direction transform (ignores translation, doesn't renormalise) — fine for
/// the meshing pipeline's axis-aligned normals.
fn transform_dir(m: [[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Largest-axis cubemap selection. Returns the face and `(u, v)` in `[0, 1]`.
fn face_and_uv(dir: [f32; 3]) -> (CubeFace, f32, f32) {
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();
    // Tie-breaking: x dominates if it's strictly greatest, then y, then z.
    // This matches conventional cubemap sampling and is fully deterministic.
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if dir[0] >= 0.0 {
            // PosX: u = -z, v = -y
            (CubeFace::PosX, -dir[2], -dir[1], ax)
        } else {
            // NegX: u = +z, v = -y
            (CubeFace::NegX, dir[2], -dir[1], ax)
        }
    } else if ay >= az {
        if dir[1] >= 0.0 {
            // PosY: u = +x, v = +z
            (CubeFace::PosY, dir[0], dir[2], ay)
        } else {
            // NegY: u = +x, v = -z
            (CubeFace::NegY, dir[0], -dir[2], ay)
        }
    } else if dir[2] >= 0.0 {
        // PosZ: u = +x, v = -y
        (CubeFace::PosZ, dir[0], -dir[1], az)
    } else {
        // NegZ: u = -x, v = -y
        (CubeFace::NegZ, -dir[0], -dir[1], az)
    };
    let inv_ma = if ma > 0.0 { 1.0 / ma } else { 0.0 };
    let u = 0.5 * (sc * inv_ma + 1.0);
    let v = 0.5 * (tc * inv_ma + 1.0);
    (face, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn cube_face_index_is_stable() {
        for (i, face) in CubeFace::ALL.iter().enumerate() {
            assert_eq!(face.index(), i);
        }
    }

    #[test]
    fn face_and_uv_centers_on_positive_axes() {
        let (face, u, v) = face_and_uv([1.0, 0.0, 0.0]);
        assert_eq!(face, CubeFace::PosX);
        assert!(approx_eq(u, 0.5));
        assert!(approx_eq(v, 0.5));
    }

    #[test]
    fn sample_default_skybox_is_background() {
        let cfg = SkyboxConfig::default();
        let sky = render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, 0, &cfg);
        assert_eq!(sky.sample([1.0, 0.0, 0.0]), cfg.background_color);
    }
}
