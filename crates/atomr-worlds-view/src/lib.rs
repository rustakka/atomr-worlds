//! CPU-side renderer for atomr-worlds bricks.
//!
//! Three modules:
//!
//! - [`mesh`] — greedy meshing of a [`Brick`] into a vertex/index buffer.
//! - [`camera`] — [`Camera`] with view+projection matrices and a
//!   `MetricScale::lod_for_screen` integration.
//! - [`render`] — a deterministic software rasterizer that writes RGBA pixels
//!   plus a z-buffer; [`render_brick_png`] is the convenience entry point.
//!
//! The eventual atomr-view bridge will sit on top of [`mesh::greedy_mesh`]
//! once the upstream scene API grows 3D primitives. Until then, this crate
//! provides everything Phase 2 needed for CI / screenshot tests on a
//! headless host.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod camera;
pub mod iso;
pub mod mesh;
pub mod render;
pub mod scene;
pub mod skybox;

pub use camera::Camera;
pub use iso::{surface_mesh, MeshMode, SmoothConfig};
pub use mesh::{greedy_mesh, Mesh, Quad, Vertex};
pub use render::{material_color, render_brick_png, render_mesh, Framebuffer, RenderConfig};
pub use scene::{
    scene_from_bricks, CameraNode, FrameMetadata, LightKind, LightNode, MaterialEntry,
    MaterialPalette, MeshNode, SceneDescription, SceneId,
};
pub use skybox::{
    render_skybox_from_meshes, CubeFace, CubeFaceImage, Skybox, SkyboxConfig, CUBE_FACE_COUNT,
};

#[derive(Debug, thiserror::Error)]
pub enum ViewError {
    #[error("png encode error: {0}")]
    Png(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
