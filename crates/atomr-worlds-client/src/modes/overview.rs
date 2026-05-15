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
use bevy::input::mouse::MouseMotion;
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
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    mut state: ResMut<OverviewState>,
) {
    if *mode != ViewMode::Overview {
        motion.clear();
        return;
    }
    if keys.just_pressed(KeyCode::KeyP) {
        state.projection = match state.projection {
            OverviewProjection::OrthographicFlat => OverviewProjection::Equirectangular,
            OverviewProjection::Equirectangular => OverviewProjection::OrthographicSphere,
            OverviewProjection::OrthographicSphere => OverviewProjection::OrthographicFlat,
        };
    }
    // Q/E zoom (matching slice/RTS); Equal/Minus stay as aliases.
    if keys.pressed(KeyCode::KeyQ) || keys.pressed(KeyCode::Equal) {
        state.extent = (state.extent * 0.97).max(0.05);
    }
    if keys.pressed(KeyCode::KeyE) || keys.pressed(KeyCode::Minus) {
        state.extent = (state.extent * 1.03).min(1.0);
    }
    // WASD rotates on both axes. Arrow keys remain as aliases. Yaw
    // (center[0]) and pitch (center[1]) both wrap freely via
    // `rem_euclid(2π)` — no ±π/2 clamp, so dragging past the south
    // pole continues into a flipped orientation rather than locking
    // up. The trig in `projection_sphere` is periodic, so any input
    // is geometrically valid.
    let pan = 0.01;
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        state.center[0] -= pan;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        state.center[0] += pan;
    }
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        state.center[1] -= pan;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        state.center[1] += pan;
    }
    // Drag-to-rotate the globe: hold left mouse and drag. We scale by
    // 1/256 so a 256-pixel sweep maps to ~1 radian, which feels like a
    // ~57° spin per full-screen drag — comparable to map-app feel.
    if mouse_buttons.pressed(MouseButton::Left) {
        let mut dx = 0.0f64;
        let mut dy = 0.0f64;
        for ev in motion.read() {
            dx += ev.delta.x as f64;
            dy += ev.delta.y as f64;
        }
        let sensitivity = 1.0 / 256.0;
        state.center[0] += dx * sensitivity;
        // Pitch: dragging down (positive dy) should tilt the globe to
        // reveal the south pole — match the trackball feel users expect.
        state.center[1] += dy * sensitivity;
    } else {
        motion.clear();
    }
    // Wrap both axes to [-π, π] so rotation never resets / locks. The
    // pitch wrap means dragging past the pole yields a flipped view
    // (correct for an unconstrained orbit); equirect/ortho-sphere
    // projections handle that via the periodic trig.
    let two_pi = core::f64::consts::TAU;
    state.center[0] = ((state.center[0] + core::f64::consts::PI).rem_euclid(two_pi))
        - core::f64::consts::PI;
    state.center[1] = ((state.center[1] + core::f64::consts::PI).rem_euclid(two_pi))
        - core::f64::consts::PI;
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
        // Bake the pyramid on first entry. ~seconds at grid_level 3 — this
        // synchronously blocks the Update schedule, so any harness capture
        // that fires on the same frame as the first overview entry sees a
        // pre-bake (background-only) image. Trace + debug logs let an
        // interactive debugger correlate "empty sky" captures with this.
        let t0 = std::time::Instant::now();
        tracing::info!(seed = active.seed, "baking macro-state pyramid for overview");
        let generator =
            DefaultMacroGenerator::new(MacroConfig { grid_level: 3, ..MacroConfig::default() });
        let shape = WorldShape::Sphere { radius_m: EARTH_R_M };
        let macro_state = generator.generate(active.seed, shape);
        let pyramid = bake_world_summary(active.addr, &macro_state, 4, 64);
        cache.pyramid = Some(pyramid);
        cache.seed = Some(active.seed);
        tracing::info!(elapsed_ms = t0.elapsed().as_millis() as u64, "overview pyramid baked");
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
