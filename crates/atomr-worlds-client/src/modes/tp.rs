//! Phase 14b — 3rd-person chase.
//!
//! Re-uses the brick streaming installed by [`crate::modes::fp::FpPlugin`]
//! (the same 3D scene is fine for both perspective modes — only the
//! camera matrix differs). This plugin owns a [`ChaseCamera`] that orbits
//! the player anchor with critical-damped smoothing.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_view::ChaseCamera;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use crate::modes::fp::{FpState, WorldCamera};
use crate::view_mode::ViewMode;

pub struct TpPlugin;

impl Plugin for TpPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TpState>()
            .add_systems(Update, (tp_input, tp_sync_camera).chain());
    }
}

#[derive(Resource)]
pub struct TpState {
    pub chase: ChaseCamera,
}

impl Default for TpState {
    fn default() -> Self {
        Self {
            chase: ChaseCamera::new(DVec3::new(8.0, 24.0, 8.0), 16.0 / 9.0),
        }
    }
}

fn tp_input(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion: EventReader<MouseMotion>,
    mut wheel: EventReader<MouseWheel>,
    fp_state: Res<FpState>,
    time: Res<Time>,
    mut state: ResMut<TpState>,
) {
    if *mode != ViewMode::Tp {
        motion.clear();
        wheel.clear();
        return;
    }
    let dt = time.delta_seconds().min(0.05);
    let mut yaw_delta = 0.0f32;
    let mut pitch_delta = 0.0f32;
    for ev in motion.read() {
        yaw_delta -= ev.delta.x * 0.005;
        pitch_delta -= ev.delta.y * 0.005;
    }
    let look_speed = 1.5;
    if keys.pressed(KeyCode::ArrowLeft) {
        yaw_delta += look_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        yaw_delta -= look_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        pitch_delta += look_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        pitch_delta -= look_speed * dt;
    }
    for ev in wheel.read() {
        state.chase.distance_m = (state.chase.distance_m - ev.y * 0.5).clamp(2.0, 40.0);
    }
    // Anchor follows the fp walk pose so the same WASD controls drive both
    // modes; the chase camera simply orbits the player.
    let anchor = fp_state.walk.observer.position;
    state.chase.tick(anchor, yaw_delta, pitch_delta, dt);
}

fn tp_sync_camera(
    mode: Res<ViewMode>,
    state: Res<TpState>,
    mut q: Query<&mut Transform, With<WorldCamera>>,
) {
    if *mode != ViewMode::Tp {
        return;
    }
    let cam = state.chase.camera();
    let eye = Vec3::new(cam.eye[0], cam.eye[1], cam.eye[2]);
    let target = Vec3::new(cam.target[0], cam.target[1], cam.target[2]);
    if let Ok(mut t) = q.get_single_mut() {
        t.translation = eye;
        t.look_at(target, Vec3::Y);
    }
}
