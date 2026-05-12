//! Phase 14c — Dwarf-Fortress slice (orthographic z-band raster).
//!
//! Re-uses the CPU `render_slice` rasterizer from `atomr-worlds-view`
//! and blits the output through the shared [`RasterTarget`].

use atomr_worlds_view::{
    build_slice_table, render_slice, scene::MaterialPalette, SliceCamera, SliceConfig,
};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::modes::fp::FpState;
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;

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
    // Space/Ctrl mirror the FP vertical controls so the same fingers that
    // move the player up/down in 3D shift the visible slice up/down here.
    if keys.just_pressed(KeyCode::PageUp)
        || keys.just_pressed(KeyCode::Equal)
        || keys.just_pressed(KeyCode::Space)
    {
        state.z_band_top += 1;
    }
    if keys.just_pressed(KeyCode::PageDown)
        || keys.just_pressed(KeyCode::Minus)
        || keys.just_pressed(KeyCode::ControlLeft)
        || keys.just_pressed(KeyCode::ControlRight)
    {
        state.z_band_top -= 1;
    }
}

fn slice_render(
    mode: Res<ViewMode>,
    runtime: Res<WorldRuntime>,
    state: Res<SliceState>,
    fp_state: Res<FpState>,
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
    let table = build_slice_table(
        runtime.query.as_ref(),
        &fp_state.addr,
        [min_x, min_z],
        [SLICE_FOOTPRINT_VOX, SLICE_FOOTPRINT_VOX],
        state.z_band_top,
        Z_BAND_THICKNESS,
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
