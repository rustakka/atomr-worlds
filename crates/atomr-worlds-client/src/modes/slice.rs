//! Phase 14c — Dwarf-Fortress slice (orthographic z-band raster).
//!
//! Builds a [`SliceTable`](atomr_worlds_view::SliceTable) from the
//! [`WorldQuery`](atomr_worlds_view::WorldQuery) and hands it to the
//! active [`SliceRenderStrategy`](crate::render::SliceRenderStrategy) (see
//! [`RenderConfig::slice`]) to rasterize, then blits the result through
//! the shared [`RasterTarget`].
//!
//! The view is oriented to match the first-person camera: world `+Z` is
//! up on screen, world `-X` is to the right. WASD pans the slice's own
//! `center_xz` in those screen directions — independent of the FP
//! camera's yaw — and the center is seeded from the FP eye each time the
//! view is entered. Q/E, Space/Ctrl, and PageUp/PageDown all shift the
//! visible z-band.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_view::derived::slice_index::build_slice_table_with_lod_fn;
use atomr_worlds_view::{SliceCamera, SliceConfig, WorldQuery};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::modes::fp::FpState;
use crate::render::{RenderConfig, SliceRenderInputs, WorldTime};
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;
use crate::world_stream::ChunkStreamer;

/// Z-band thickness in voxels. 3 ≈ DF default.
const Z_BAND_THICKNESS: u8 = 3;
/// How many voxels wide the slice samples horizontally around the center.
/// 64 voxels = 4×4 chunks (`BRICK_EDGE` = 16).
const SLICE_FOOTPRINT_VOX: u32 = 64;
/// On-screen pixels per voxel tile. Derived so the footprint fills the
/// fixed raster exactly (`64 * 4 = 256`).
const SLICE_TILE_PX: u32 = RASTER_W / SLICE_FOOTPRINT_VOX;

pub struct SlicePlugin;

impl Plugin for SlicePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SliceState>()
            .add_systems(Update, slice_input)
            .add_systems(Update, slice_render);
    }
}

#[derive(Resource)]
struct SliceState {
    /// Horizontal-plane center of the view, in world voxel units. Seeded
    /// from the FP eye on entry, then panned independently by WASD.
    center_xz: [f32; 2],
    /// Top of the visible z-band, in voxel-Y coords.
    z_band_top: i32,
}

impl Default for SliceState {
    fn default() -> Self {
        // `center_xz` is overwritten the first frame slice mode is
        // entered (see `slice_input`); the placeholder is never rendered.
        Self { center_xz: [0.0, 0.0], z_band_top: 6 }
    }
}

fn slice_input(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    fp_state: Res<FpState>,
    runtime: Res<WorldRuntime>,
    mut state: ResMut<SliceState>,
    mut prev_mode: Local<Option<ViewMode>>,
) {
    let just_entered = *mode == ViewMode::Slice && *prev_mode != Some(ViewMode::Slice);
    *prev_mode = Some(*mode);
    if *mode != ViewMode::Slice {
        return;
    }
    if just_entered {
        // Seed the pan center from the FP eye so switching into slice
        // mode keeps you over the same place. From here WASD pans the
        // slice independently — the FP position is left untouched.
        let cam = fp_state.walk.camera();
        state.center_xz = [cam.eye[0], cam.eye[2]];
        // Seed the z-band to bracket the surface near the player so the
        // view opens on terrain that corresponds to the FP scene rather
        // than blank sky or uniform underground. The band scans the two
        // voxels below `z_band_top`, so ground + 2 puts the surface in
        // view. Falls back to the FP eye height if the host can't
        // resolve a ground column.
        state.z_band_top = match runtime
            .query
            .ground_height_m(&fp_state.addr, [cam.eye[0] as f64, cam.eye[2] as f64])
        {
            Some(h) => h.round() as i32 + 2,
            None => cam.eye[1].round() as i32,
        };
    }

    // WASD pans `center_xz` in screen-aligned directions, decoupled from
    // the FP camera yaw. Screen-up is world +Z, screen-right is world -X
    // (matches the FP view + `render_slice`'s pixel mapping).
    let dt = time.delta_seconds().min(0.05);
    let speed = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        12.0
    } else {
        4.0
    };
    if keys.pressed(KeyCode::KeyW) {
        state.center_xz[1] += speed * dt;
    }
    if keys.pressed(KeyCode::KeyS) {
        state.center_xz[1] -= speed * dt;
    }
    if keys.pressed(KeyCode::KeyA) {
        state.center_xz[0] += speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        state.center_xz[0] -= speed * dt;
    }

    // Q/E shift the visible Z-band up/down. Space/Ctrl mirror the FP
    // view's vertical controls, and PageUp/PageDown stay as aliases for
    // any existing muscle memory.
    let band_up = keys.just_pressed(KeyCode::KeyQ)
        || keys.just_pressed(KeyCode::Space)
        || keys.just_pressed(KeyCode::PageUp);
    let band_down = keys.just_pressed(KeyCode::KeyE)
        || keys.just_pressed(KeyCode::ControlLeft)
        || keys.just_pressed(KeyCode::ControlRight)
        || keys.just_pressed(KeyCode::PageDown);
    if band_up {
        state.z_band_top += 1;
    }
    if band_down {
        state.z_band_top -= 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn slice_render(
    mode: Res<ViewMode>,
    runtime: Res<WorldRuntime>,
    state: Res<SliceState>,
    fp_state: Res<FpState>,
    streamer: Res<ChunkStreamer>,
    render_cfg: Res<RenderConfig>,
    world_time: Res<WorldTime>,
    target: Res<RasterTarget>,
    mut images: ResMut<Assets<Image>>,
) {
    if *mode != ViewMode::Slice {
        return;
    }
    let center_x = state.center_xz[0];
    let center_z = state.center_xz[1];
    let half = (SLICE_FOOTPRINT_VOX as f32) * 0.5;
    let min_x = (center_x - half).floor() as i32;
    let min_z = (center_z - half).floor() as i32;
    // Per-column LOD: columns within the streamer's transition radius
    // sample at near_lod; everything beyond falls back to far_lod. Voxel
    // size in slice mode is 1 m per voxel, so XZ voxel coords already line
    // up with the meter-space the streamer expects. The LOD observer is
    // the slice's own pan center (lifted to the active z-band so the LOD
    // ring sits on the visible plane), so panning the slice always keeps
    // the high-detail ring under the visible footprint instead of leaving
    // it stuck where the FP camera last stood.
    let lod_observer = DVec3::new(
        center_x as f64,
        state.z_band_top as f64,
        center_z as f64,
    );
    let table = build_slice_table_with_lod_fn(
        runtime.query.as_ref(),
        &fp_state.addr,
        [min_x, min_z],
        [SLICE_FOOTPRINT_VOX, SLICE_FOOTPRINT_VOX],
        state.z_band_top,
        Z_BAND_THICKNESS,
        |[wx, wz]| {
            let p = DVec3::new(wx as f64, lod_observer.y, wz as f64);
            streamer.lod_for_meters(lod_observer, p)
        },
    );
    let cam = SliceCamera {
        center_xz: [center_x, center_z],
        z_band_top: state.z_band_top,
        z_band_thickness: Z_BAND_THICKNESS,
        half_height_m: half,
        aspect: 1.0,
    };
    // `shading` / `light_dir_xz_y` are overridden by the strategy; the
    // rest of the config fills the fixed raster exactly.
    let base_cfg = SliceConfig {
        width: RASTER_W,
        height: RASTER_H,
        tile_px: SLICE_TILE_PX,
        stipple_thin_features: true,
        roof_alpha: 0.25,
        background: [20, 20, 28, 255],
        ..SliceConfig::default()
    };
    let palette = render_cfg.palette.palette();
    // Sun direction FROM sun INTO scene — same value the FP view's
    // directional light uses, so the slice's relief shading is consistent
    // with the 3D scene.
    let sun_dir = render_cfg.sun_curve.sun_state(world_time.0).direction;
    let inputs = SliceRenderInputs {
        table: &table,
        cam: &cam,
        palette: &palette,
        base_cfg,
        sun_dir,
    };
    let fb = render_cfg.slice.render(&inputs);
    copy_framebuffer_to_image(&mut images, &target, &fb);
}
