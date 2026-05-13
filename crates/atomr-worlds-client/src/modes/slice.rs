//! Phase 14c — Dwarf-Fortress slice (orthographic z-band raster).
//!
//! Re-uses the CPU `render_slice` rasterizer from `atomr-worlds-view`
//! and blits the output through the shared [`RasterTarget`].

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_view::derived::slice_index::build_slice_table_with_lod_fn;
use atomr_worlds_view::{render_slice, scene::MaterialPalette, SliceCamera, SliceConfig};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::modes::fp::FpState;
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;
use crate::world_stream::ChunkStreamer;

/// Z-band thickness in voxels. 3 ≈ DF default.
const Z_BAND_THICKNESS: u8 = 3;
/// How many voxels wide the slice samples horizontally around the camera.
const SLICE_FOOTPRINT_VOX: u32 = 32;

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
    /// Top of the visible z-band, in voxel-Y coords.
    z_band_top: i32,
}

impl Default for SliceState {
    fn default() -> Self {
        Self { z_band_top: 6 }
    }
}

fn slice_input(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<SliceState>,
) {
    if *mode != ViewMode::Slice {
        return;
    }
    // WASD pans horizontally — handled by `world_walk_input` in fp.rs
    // (it drives `fp_state.walk`, which `slice_render` centers the
    // raster on). Q/E shift the visible Z-band up/down; PageUp/Down
    // remain as alternatives so the bindings users may already have
    // muscle-memory for still work.
    if keys.just_pressed(KeyCode::KeyQ) || keys.just_pressed(KeyCode::PageUp) {
        state.z_band_top += 1;
    }
    if keys.just_pressed(KeyCode::KeyE) || keys.just_pressed(KeyCode::PageDown) {
        state.z_band_top -= 1;
    }
}

fn slice_render(
    mode: Res<ViewMode>,
    runtime: Res<WorldRuntime>,
    state: Res<SliceState>,
    fp_state: Res<FpState>,
    streamer: Res<ChunkStreamer>,
    target: Res<RasterTarget>,
    mut images: ResMut<Assets<Image>>,
) {
    if *mode != ViewMode::Slice {
        return;
    }
    let cam = fp_state.walk.camera();
    let center_x = cam.eye[0];
    let center_z = cam.eye[2];
    let half = (SLICE_FOOTPRINT_VOX as f32) * 0.5;
    let min_x = (center_x - half).floor() as i32;
    let min_z = (center_z - half).floor() as i32;
    // Per-column LOD: columns within the streamer's transition radius
    // sample at near_lod; everything beyond falls back to far_lod. Voxel
    // size in slice mode is 1 m per voxel, so XZ voxel coords already line
    // up with the meter-space the streamer expects.
    let observer = fp_state.walk.observer.position;
    let table = build_slice_table_with_lod_fn(
        runtime.query.as_ref(),
        &fp_state.addr,
        [min_x, min_z],
        [SLICE_FOOTPRINT_VOX, SLICE_FOOTPRINT_VOX],
        state.z_band_top,
        Z_BAND_THICKNESS,
        |[wx, wz]| {
            let p = DVec3::new(wx as f64, observer.y, wz as f64);
            streamer.lod_for_meters(observer, p)
        },
    );
    let cam = SliceCamera {
        center_xz: [center_x, center_z],
        z_band_top: state.z_band_top,
        z_band_thickness: Z_BAND_THICKNESS,
        half_height_m: SLICE_FOOTPRINT_VOX as f32 * 0.5,
        aspect: 1.0,
    };
    let cfg = SliceConfig {
        width: RASTER_W,
        height: RASTER_H,
        tile_px: 8,
        stipple_thin_features: true,
        roof_alpha: 0.25,
        background: [20, 20, 28, 255],
    };
    let palette = MaterialPalette::default();
    let fb = render_slice(&table, &cam, &palette, &cfg);
    copy_framebuffer_to_image(&mut images, &target, &fb);
}
