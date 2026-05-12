//! Phase 14d — RTS oblique. Surface raster + oblique-ortho projection.

use atomr_worlds_core::lod::Lod;
use atomr_worlds_view::{
    build_surface_raster, render_rts, scene::MaterialPalette, ObliqueCamera, RenderConfig,
};
use bevy::prelude::*;

use crate::modes::blit::{copy_framebuffer_to_image, RasterTarget, RASTER_H, RASTER_W};
use crate::modes::fp::FpState;
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;

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
    if keys.pressed(KeyCode::KeyQ) {
        state.rotation_deg += 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        state.rotation_deg -= 1.0;
    }
    if keys.just_pressed(KeyCode::Equal) {
        state.scale_m_per_px = (state.scale_m_per_px * 0.9).max(0.02);
    }
    if keys.just_pressed(KeyCode::Minus) {
        state.scale_m_per_px = (state.scale_m_per_px * 1.1).min(2.0);
    }
}

fn rts_render(
    mode: Res<ViewMode>,
    runtime: Res<WorldRuntime>,
    fp_state: Res<FpState>,
    state: Res<RtsState>,
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
    let raster = build_surface_raster(
        runtime.query.as_ref(),
        &fp_state.addr,
        origin,
        [RTS_FOOTPRINT_VOX, RTS_FOOTPRINT_VOX],
        1.0,
        Lod::new(0),
    );
    let oblique = ObliqueCamera {
        center_xz: [center_x as f32, center_z as f32],
        rotation_deg: state.rotation_deg,
        scale_m_per_px: state.scale_m_per_px,
        near: 0.1,
        far: 1024.0,
        aspect: 1.0,
    };
    let cfg = RenderConfig {
        width: RASTER_W,
        height: RASTER_H,
        background: [12, 16, 20, 255],
        ..Default::default()
    };
    let palette = MaterialPalette::default();
    let fb = render_rts(&raster, &[], &oblique, &palette, &cfg);
    copy_framebuffer_to_image(&mut images, &target, &fb);
}
