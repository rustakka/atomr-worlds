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
pub mod decals;
pub mod derived;
pub mod frustum;
pub mod iso;
pub mod mesh;
pub mod modes;
pub mod observer;
pub mod projection_sphere;
pub mod raster2d;
pub mod render;
pub mod scene;
pub mod skybox;
pub mod view_cache;
pub mod world_query;

pub use camera::{Camera, Projection};
pub use decals::{render_decals, Decal};
pub use derived::slice_index::{build_slice_table, SliceColumn, SliceKey, SliceTable};
pub use derived::surface_raster::{
    build_surface_raster, surface_raster_to_mesh, SurfaceKey, SurfaceRaster,
};
pub use frustum::{Frustum, Plane};
pub use iso::{
    boundary_skirt, crossfade_overlap, face_height_profile, lod_transition_strip, surface_mesh,
    MeshMode, SmoothConfig,
};
pub use mesh::{
    bake_ao, dual_contouring_mesh, greedy_mesh, greedy_mesh_by_material, marching_cubes_mesh,
    marching_cubes_mesh_with_iso, naive_mesh, Mesh, Quad, Vertex,
};
pub use modes::fp::{build_fp_scene, render_fp, FpScene, MeshCacheKey, WalkCamera, WalkInput};
pub use modes::rts::{render_rts, ObliqueCamera};
pub use modes::slice::{render_slice, render_slice_cached, SliceCamera, SliceConfig, SliceShading};
pub use modes::tp::{render_tp, ChaseCamera};
pub use modes::view_mode::ViewMode;
pub use observer::{ObserverState, SkyboxRefreshPolicy};
pub use raster2d::{blend_rect, blit_rgba, fill_rect, fill_rect_stipple, StipplePattern};
pub use render::{
    material_color, render_brick_png, render_composite, render_mesh, CompositeScene, FragmentMode,
    Framebuffer, RenderConfig,
};
pub use scene::{
    scene_from_bricks, CameraNode, FrameMetadata, LightKind, LightNode, MaterialEntry, MaterialPalette,
    MeshNode, SceneDescription, SceneId,
};
pub use skybox::{render_skybox_from_meshes, CubeFace, CubeFaceImage, Skybox, SkyboxConfig, CUBE_FACE_COUNT};
pub use view_cache::{CacheAabb, DerivedKey, Revision, ViewCache};
pub use world_query::WorldQuery;

pub use derived::world_summary::{
    bake_world_summary, ClimateSample, WorldSummaryKey, WorldSummaryPyramid, WorldSummaryTile,
};
pub use modes::overview::{pick_pyramid_level, render_overview, OverviewCamera, OverviewProjection};

#[derive(Debug, thiserror::Error)]
pub enum ViewError {
    #[error("png encode error: {0}")]
    Png(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
