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
    SurfaceRaster,
};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::modes::fp::FpState;
use crate::modes::raster_async::AsyncBuild;
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;
use crate::world_stream::ChunkStreamer;

const RTS_FOOTPRINT_VOX: u32 = 48;

/// Footprint key for the off-thread [`SurfaceRaster`] rebuild: the view center
/// quantized to whole voxels. Rotation / zoom don't affect the raster (only the
/// camera), so they never trigger a rebuild — only a pan does.
type RtsKey = (i32, i32);

/// Caches the most-recent [`SurfaceRaster`], rebuilt off the render thread by
/// [`rts_render`] when the pan center moves. See [`AsyncBuild`].
#[derive(Resource, Default)]
pub struct RtsRasterCache(pub AsyncBuild<SurfaceRaster, RtsKey>);

pub struct RtsPlugin;

impl Plugin for RtsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RtsState>()
            .init_resource::<RtsRasterCache>()
            .add_systems(Update, rts_input)
            .add_systems(Update, rts_render);
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

#[allow(clippy::too_many_arguments)]
fn rts_render(
    mode: Res<ViewMode>,
    runtime: Res<WorldRuntime>,
    fp_state: Res<FpState>,
    state: Res<RtsState>,
    streamer: Res<ChunkStreamer>,
    target: Res<RasterTarget>,
    perf: Res<crate::perf::Perf>,
    harness: Option<Res<crate::harness::HarnessActive>>,
    mut cache: ResMut<RtsRasterCache>,
    mut images: ResMut<Assets<Image>>,
) {
    if *mode != ViewMode::Rts {
        return;
    }
    let _scope = perf.scope(crate::perf::Phase::SliceRtsRaster);
    let cam = fp_state.walk.camera();
    let center_x = cam.eye[0] as f64;
    let center_z = cam.eye[2] as f64;
    let half = (RTS_FOOTPRINT_VOX as f64) * 0.5;
    let observer = fp_state.walk.observer.position;

    // Under the harness, build + draw the raster SYNCHRONOUSLY this frame at the
    // live center — byte-identical to the pre-change path, so golden captures
    // stay deterministic. The off-thread cache path is interactive-only.
    if harness.is_some() {
        let origin = [center_x - half, center_z - half];
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
        draw_rts(&raster, [center_x, center_z], &state, cam.eye[1], &target, &mut images);
        return;
    }

    // Interactive: build the SurfaceRaster off the render thread — its builder
    // calls the host `WorldQuery::brick` (a `block_on`) for ~48×48 columns, which
    // stalled the frame every frame. Only a *pan* (center move) rebuilds it;
    // rotation / zoom are camera-only and redraw the cached raster.
    cache.0.poll();
    perf.set_snapshot_rebuilding(cache.0.is_rebuilding());
    let key: RtsKey = (center_x.round() as i32, center_z.round() as i32);
    if cache.0.needs_rebuild(&key) {
        let query = runtime.query.clone();
        let streamer = streamer.clone();
        let addr = fp_state.addr;
        // Origin derived from the rounded key (not the live center) so the
        // raster's true center is exactly `built_for()` — the camera frames there,
        // keeping mesh and camera aligned to the voxel grid.
        let origin = [key.0 as f64 - half, key.1 as f64 - half];
        cache.0.spawn(key, move || {
            build_surface_raster_with_lod_fn(
                query.as_ref(),
                &addr,
                origin,
                [RTS_FOOTPRINT_VOX, RTS_FOOTPRINT_VOX],
                1.0,
                |[wx_m, wz_m]| {
                    let p = DVec3::new(wx_m, observer.y, wz_m);
                    streamer.lod_for_meters(observer, p)
                },
            )
        });
    }
    // Nothing built yet — keep the raster target's prior contents.
    let Some(raster) = cache.0.current() else {
        return;
    };
    // Frame the camera on the footprint the *current* raster was built for (it
    // may lag the live pan by one rebuild), so the mesh and camera stay aligned.
    let built = cache.0.built_for().copied().unwrap_or(key);
    draw_rts(raster, [built.0 as f64, built.1 as f64], &state, cam.eye[1], &target, &mut images);
}

/// Mesh + top-down-ortho render a [`SurfaceRaster`] to the shared raster target.
/// `center_xz` is the world center the raster covers (live under the harness,
/// `built_for` interactively); rotation / zoom come live from [`RtsState`].
/// Shared by the harness and interactive paths so both render identically.
fn draw_rts(
    raster: &SurfaceRaster,
    center_xz: [f64; 2],
    state: &RtsState,
    eye_center_y: f32,
    target: &RasterTarget,
    images: &mut Assets<Image>,
) {
    let palette = MaterialPalette::default();
    let mesh = surface_raster_to_mesh(raster, &palette);
    let center_x = center_xz[0] as f32;
    let center_z = center_xz[1] as f32;
    // Top-down ortho. Eye sits well above the tallest possible surface so the
    // entire mesh is in front of the camera; the orthographic projection makes
    // eye altitude irrelevant for x/y framing.
    let center_y_m = eye_center_y;
    let eye_y_m = center_y_m + 512.0;
    let theta = state.rotation_deg.to_radians();
    // Q/E rotate the up-vector around +Y so the world spins under the camera
    // while still pointing the eye straight down.
    let up = [theta.sin(), 0.0, theta.cos()];
    let half_height_m = state.scale_m_per_px * (RASTER_H as f32) * 0.5;
    let aspect = (RASTER_W as f32) / (RASTER_H as f32);
    let camera = Camera {
        eye: [center_x, eye_y_m, center_z],
        target: [center_x, center_y_m, center_z],
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
    copy_framebuffer_to_image(images, target, &fb);
}
