//! Rendering strategies + configuration.
//!
//! Every meaningful render decision (meshing, palette, AO, shading, sky,
//! sun curve, shadows, fog, tonemap) is a trait with at least one default
//! implementation, all bundled into a [`RenderConfig`] resource that the
//! view-mode plugins consume. Swapping a strategy is a one-line change in
//! [`RenderConfig`] (or a `set_strategy` harness event), which keeps the
//! experimentation surface wide.

pub mod config;
pub mod defaults;
pub mod materials;
pub mod offscreen;
pub mod plugin;
pub mod registry;
pub mod sky_dome;
pub mod skybox;
pub mod strategy;
pub mod sun;

pub use config::{RenderConfig, RenderPreset};
pub use materials::{PaletteEntryGpu, SkyDomeMaterial, VoxelMaterial, VoxelMaterialExt};
pub use offscreen::{
    CaptureOutcome, CaptureOutcomes, CaptureQueueHandle, OffscreenCapturePlugin,
    OffscreenTarget,
};
pub use plugin::RenderPlugin;
pub use registry::apply_strategy_by_name;
pub use sky_dome::{SkyDome, SkyDomePlugin};
pub use skybox::{
    bake_skybox, cubemap_image, lerp_brightness, placeholder_cubemap_image, sync_skybox,
    SkyboxPlugin, SkyboxRuntime, DAY_BRIGHTNESS, DEFAULT_MIN_FRAMES_BETWEEN_BAKES,
    NIGHT_BRIGHTNESS, SKYBOX_FACE_RESOLUTION,
};
pub use strategy::{
    AoStrategy, FogStrategy, LodCoveragePolicy, MeshStrategy, PaletteStrategy, ShadingMode,
    ShadingStrategy, ShadowStrategy, SkyStrategy, SunCurveStrategy, SunState, TonemapStrategy,
};
pub use sun::{advance_world_time, sync_sky_and_fog, sync_sun, WorldSunMarker, WorldTime};
