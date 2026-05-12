//! Phase 14e — macro-state pyramid overview.
//!
//! The pyramid bake is heavy (~seconds), so we build it once on first
//! entry into overview mode and cache it. Rendering each frame is cheap
//! after that.

use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::{DefaultMacroGenerator, MacroConfig, MacroGenerator};
use atomr_worlds_view::{
    bake_world_summary, render_overview, OverviewCamera, OverviewProjection, RenderConfig,
    WorldSummaryPyramid,
};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::view_mode::ViewMode;
use crate::world_runtime::ActiveWorld;

const EARTH_R_M: f64 = 6.371e6;

pub struct OverviewPlugin;

impl Plugin for OverviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OverviewCache>()
            .init_resource::<OverviewState>()
            .add_systems(Update, overview_input)
            .add_systems(Update, overview_render);
    }
}

#[derive(Resource, Default)]
struct OverviewCache {
    pyramid: Option<WorldSummaryPyramid>,
    /// Seed the cached pyramid was baked for. Re-baked if the active
    /// world changes.
    seed: Option<u64>,
}

#[derive(Resource)]
struct OverviewState {
    projection: OverviewProjection,
    extent: f64,
    center: [f64; 2],
}

impl Default for OverviewState {
    fn default() -> Self {
        Self { projection: OverviewProjection::OrthographicFlat, extent: 1.0, center: [0.0, 0.0] }
    }
}

fn overview_input(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<OverviewState>,
) {
    if *mode != ViewMode::Overview {
        return;
    }
    if keys.just_pressed(KeyCode::KeyP) {
        state.projection = match state.projection {
            OverviewProjection::OrthographicFlat => OverviewProjection::Equirectangular,
            OverviewProjection::Equirectangular => OverviewProjection::OrthographicSphere,
            OverviewProjection::OrthographicSphere => OverviewProjection::OrthographicFlat,
        };
    }
    if keys.pressed(KeyCode::Equal) {
        state.extent = (state.extent * 0.97).max(0.05);
    }
    if keys.pressed(KeyCode::Minus) {
        state.extent = (state.extent * 1.03).min(1.0);
    }
    let pan = 0.01;
    if keys.pressed(KeyCode::ArrowLeft) {
        state.center[0] -= pan;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        state.center[0] += pan;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        state.center[1] -= pan;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        state.center[1] += pan;
    }
}

fn overview_render(
    mode: Res<ViewMode>,
    active: Res<ActiveWorld>,
    state: Res<OverviewState>,
    target: Res<RasterTarget>,
    mut cache: ResMut<OverviewCache>,
    mut images: ResMut<Assets<Image>>,
) {
    if *mode != ViewMode::Overview {
        return;
    }
    if cache.seed != Some(active.seed) {
        // Bake the pyramid on first entry. ~seconds at grid_level 3.
        tracing::info!(seed = active.seed, "baking macro-state pyramid for overview");
        let generator =
            DefaultMacroGenerator::new(MacroConfig { grid_level: 3, ..MacroConfig::default() });
        let shape = WorldShape::Sphere { radius_m: EARTH_R_M };
        let macro_state = generator.generate(active.seed, shape);
        let pyramid = bake_world_summary(active.addr, &macro_state, 4, 64);
        cache.pyramid = Some(pyramid);
        cache.seed = Some(active.seed);
    }
    let Some(pyramid) = cache.pyramid.as_ref() else { return };

    let cam = OverviewCamera {
        center: state.center,
        extent: state.extent,
        projection: state.projection,
        aspect: 1.0,
    };
    let cfg = RenderConfig {
        width: RASTER_W,
        height: RASTER_H,
        background: [10, 14, 24, 255],
        light_dir: [0.5, 0.8, 0.3],
        ambient: 0.25,
    };
    let fb = render_overview(pyramid, &cam, &cfg);
    copy_framebuffer_to_image(&mut images, &target, &fb);
}
