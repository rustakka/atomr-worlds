//! Phase 14d — RTS oblique. Surface raster + top-down orthographic
//! projection.
//!
//! Note: the `Projection::Oblique` path in `atomr-worlds-view` over-applies
//! its shear (the constant `tan(α) × eye_y` offset puts the surface
//! off-screen at any sane eye altitude), so we render the RTS view as a
//! plain top-down orthographic for now. Q/E rotate by spinning the
//! camera's `up` vector around world-Y, which keeps the in-plane
//! orientation control the user expects.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_view::derived::surface_raster::build_surface_raster_with_lod_fn;
use atomr_worlds_view::{
    render_mesh, scene::MaterialPalette, surface_raster_to_mesh, Camera, Projection, RenderConfig,
};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::modes::fp::FpState;
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;
use crate::world_stream::ChunkStreamer;

const RTS_FOOTPRINT_VOX: u32 = 48;

pub struct RtsPlugin;

impl Plugin for RtsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RtsState>().add_systems(Update, rts_input).add_systems(Update, rts_render);
    }
}

#[derive(Resource)]
struct RtsState {
    rotation_deg: f32,
    scale_m_per_px: f32,
}

impl Default for RtsState {
    fn default() -> Self {
        Self { rotation_deg: 30.0, scale_m_per_px: 0.20 }
    }
}

fn rts_input(mode: Res<ViewMode>, keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<RtsState>) {
    if *mode != ViewMode::Rts {
        return;
    }
    // WASD pans horizontally (handled by `world_walk_input` in fp.rs —
    // it moves `fp_state.walk`, which `rts_render` centers the raster
    // on). Q/E zoom in/out. Z/X rotate (moved off Q/E so the zoom
    // binding matches every other "Q/E zoom" mode in the client).
    // Equal/Minus stay as alternative zoom bindings for muscle memory.
    if keys.just_pressed(KeyCode::KeyQ) || keys.just_pressed(KeyCode::Equal) {
        state.scale_m_per_px = (state.scale_m_per_px * 0.9).max(0.02);
    }
    if keys.just_pressed(KeyCode::KeyE) || keys.just_pressed(KeyCode::Minus) {
        state.scale_m_per_px = (state.scale_m_per_px * 1.1).min(2.0);
    }
    if keys.pressed(KeyCode::KeyZ) {
        state.rotation_deg += 1.0;
    }
    if keys.pressed(KeyCode::KeyX) {
        state.rotation_deg -= 1.0;
    }
}

fn rts_render(
    mode: Res<ViewMode>,
    runtime: Res<WorldRuntime>,
    fp_state: Res<FpState>,
    state: Res<RtsState>,
    streamer: Res<ChunkStreamer>,
    target: Res<RasterTarget>,
    mut images: ResMut<Assets<Image>>,
) {
    if *mode != ViewMode::Rts {
        return;
    }
    let cam = fp_state.walk.camera();
    let center_x = cam.eye[0] as f64;
    let center_z = cam.eye[2] as f64;
    let half = (RTS_FOOTPRINT_VOX as f64) * 0.5;
    let origin = [center_x - half, center_z - half];
    // Per-column LOD comes from the shared streamer. Columns inside the
    // transition radius scan at `near_lod`; far columns drop to
    // `far_lod`, matching the FP/TP ring streamer's tier boundary.
    let observer = fp_state.walk.observer.position;
    let raster = build_surface_raster_with_lod_fn(
        runtime.query.as_ref(),
        &fp_state.addr,
        origin,
        [RTS_FOOTPRINT_VOX, RTS_FOOTPRINT_VOX],
        1.0,
        |[wx_m, wz_m]| {
            let p = DVec3::new(wx_m, observer.y, wz_m);
            streamer.lod_for_meters(observer, p)
        },
    );
    let palette = MaterialPalette::default();
    let mesh = surface_raster_to_mesh(&raster, &palette);
    // Top-down ortho. Eye sits well above the tallest possible surface so
    // the entire mesh is in front of the camera; the orthographic
    // projection makes eye altitude irrelevant for x/y framing.
    let center_y_m = cam.eye[1];
    let eye_y_m = center_y_m + 512.0;
    let theta = state.rotation_deg.to_radians();
    // Q/E rotate the up-vector around +Y so the world spins under the
    // camera while still pointing the eye straight down.
    let up = [theta.sin(), 0.0, theta.cos()];
    let half_height_m = state.scale_m_per_px * (RASTER_H as f32) * 0.5;
    let aspect = (RASTER_W as f32) / (RASTER_H as f32);
    let camera = Camera {
        eye: [center_x as f32, eye_y_m, center_z as f32],
        target: [center_x as f32, center_y_m, center_z as f32],
        up,
        fov_y_rad: std::f32::consts::FRAC_PI_4,
        aspect,
        near: 0.1,
        far: 2048.0,
        projection: Projection::Orthographic { half_height_m },
    };
    let cfg = RenderConfig {
        width: RASTER_W,
        height: RASTER_H,
        background: [12, 16, 20, 255],
        ..Default::default()
    };
    let fb = render_mesh(&mesh, &camera, &cfg);
    copy_framebuffer_to_image(&mut images, &target, &fb);
}
