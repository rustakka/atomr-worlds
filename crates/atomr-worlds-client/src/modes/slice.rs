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
use atomr_worlds_view::{SliceCamera, SliceConfig, SliceTable, WorldQuery};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::modes::fp::FpState;
use crate::modes::raster_async::AsyncBuild;
use crate::render::{RenderConfig, SliceRenderInputs, WorldTime};
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;
use crate::world_stream::ChunkStreamer;

/// Footprint key that decides when the off-thread [`SliceTable`] rebuild fires:
/// the sampled-region min corner (1 voxel granularity) + the z-band top.
type SliceKey = (i32, i32, i32);

/// Caches the most-recent [`SliceTable`], rebuilt off the render thread by
/// [`slice_render`] when the footprint moves. See [`AsyncBuild`] for why.
#[derive(Resource, Default)]
pub struct SliceTableCache(pub AsyncBuild<SliceTable, SliceKey>);

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
            .init_resource::<SliceTableCache>()
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
    let dt = time.delta_secs().min(0.05);
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
    perf: Res<crate::perf::Perf>,
    harness: Option<Res<crate::harness::HarnessActive>>,
    mut cache: ResMut<SliceTableCache>,
    mut images: ResMut<Assets<Image>>,
) {
    if *mode != ViewMode::Slice {
        return;
    }
    let _scope = perf.scope(crate::perf::Phase::SliceRtsRaster);
    let center_x = state.center_xz[0];
    let center_z = state.center_xz[1];
    let half = (SLICE_FOOTPRINT_VOX as f32) * 0.5;
    let min_x = (center_x - half).floor() as i32;
    let min_z = (center_z - half).floor() as i32;

    // Under the harness, build + draw the table SYNCHRONOUSLY this frame,
    // centered on the live pan center — byte-identical to the pre-change path,
    // so golden captures stay deterministic. (The off-thread cache below would
    // make the capture frame depend on a background thread's wall-clock.) The
    // async cache path is interactive-only.
    if harness.is_some() {
        let lod_observer = DVec3::new(center_x as f64, state.z_band_top as f64, center_z as f64);
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
        draw_slice(
            &table,
            [center_x, center_z],
            state.z_band_top,
            &render_cfg,
            &world_time,
            &target,
            &mut images,
        );
        return;
    }

    // Interactive: build the SliceTable off the render thread (the builder calls
    // the host `WorldQuery::brick` = `block_on` for ~64×64 columns, which stalled
    // the frame every frame). Rebuild only when the footprint (min corner +
    // z-band) moves; otherwise redraw the cached table. The per-column LOD
    // observer is the slice's own pan center, lifted to the active z-band.
    cache.0.poll();
    perf.set_snapshot_rebuilding(cache.0.is_rebuilding());
    let key: SliceKey = (min_x, min_z, state.z_band_top);
    if cache.0.needs_rebuild(&key) {
        let query = runtime.query.clone();
        let streamer = streamer.clone();
        let addr = fp_state.addr;
        let z_band_top = state.z_band_top;
        let lod_observer = DVec3::new(center_x as f64, z_band_top as f64, center_z as f64);
        cache.0.spawn(key, move || {
            build_slice_table_with_lod_fn(
                query.as_ref(),
                &addr,
                [min_x, min_z],
                [SLICE_FOOTPRINT_VOX, SLICE_FOOTPRINT_VOX],
                z_band_top,
                Z_BAND_THICKNESS,
                |[wx, wz]| {
                    let p = DVec3::new(wx as f64, lod_observer.y, wz as f64);
                    streamer.lod_for_meters(lod_observer, p)
                },
            )
        });
    }
    // Nothing built yet (first frames after entering the view) — skip; the
    // raster target keeps its prior contents until the first table lands.
    let Some(table) = cache.0.current() else {
        return;
    };
    // Frame the camera on the footprint the *current* table was built for (which
    // may lag the live pan by one rebuild), so the camera and table stay aligned
    // — the whole view steps forward when a rebuild lands rather than the camera
    // sliding over a stale table.
    let built = cache.0.built_for().copied().unwrap_or(key);
    draw_slice(
        table,
        [built.0 as f32 + half, built.1 as f32 + half],
        built.2,
        &render_cfg,
        &world_time,
        &target,
        &mut images,
    );
}

/// Rasterize a [`SliceTable`] to the shared raster target. Shared by the harness
/// path (synchronous build, live center) and the interactive path (cached build,
/// `built_for` center) so both rasterize identically.
fn draw_slice(
    table: &SliceTable,
    center_xz: [f32; 2],
    z_band_top: i32,
    render_cfg: &RenderConfig,
    world_time: &WorldTime,
    target: &RasterTarget,
    images: &mut Assets<Image>,
) {
    let half = (SLICE_FOOTPRINT_VOX as f32) * 0.5;
    let cam = SliceCamera {
        center_xz,
        z_band_top,
        z_band_thickness: Z_BAND_THICKNESS,
        half_height_m: half,
        aspect: 1.0,
    };
    // `shading` / `light_dir_xz_y` are overridden by the strategy; the rest of
    // the config fills the fixed raster exactly.
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
    // Sun direction FROM sun INTO scene — same value the FP view's directional
    // light uses, so the slice's relief shading matches the 3D scene.
    let sun_dir = render_cfg.sun_curve.sun_state(world_time.0).direction;
    let inputs = SliceRenderInputs { table, cam: &cam, palette: &palette, base_cfg, sun_dir };
    let fb = render_cfg.slice.render(&inputs);
    copy_framebuffer_to_image(images, target, &fb);
}
