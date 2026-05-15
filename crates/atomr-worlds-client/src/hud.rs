//! Debug HUD — FPS, camera coords, active view mode.
//!
//! Native `bevy_ui` `TextBundle`s. The atomr-view `UiBridge` route is
//! documented in `docs/CLIENT_SERVER.md` as a follow-up; pulling
//! `atomr-view-backends` here today would drag in wgpu/egui/winit/uniffi
//! /pyo3 *and* trigger a `path` vs `git` collision on `atomr-core`.
//!
//! The HUD does not own its own camera. Instead, [`route_hud_target`]
//! reassigns the UI root's [`TargetCamera`] each frame to whichever of
//! `WorldCamera` (FP/TP) or `BlitCamera` (slice/RTS/overview) is the
//! active camera for the current [`ViewMode`]. Reason: in Bevy 0.13 a
//! `Camera2d` and a `Camera3d` both actively targeting the same offscreen
//! `Image` causes the 3D output to be dropped — the dedicated HudCamera
//! we used previously broke FP/TP harness capture for exactly this reason.
//! Routing the UI onto the one active camera keeps exactly one camera per
//! target and lets `bevy_ui`'s ui_pass (which is registered into both
//! `Core2d` and `Core3d`) composite the HUD above whichever main pass ran.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::modes::blit::BlitCamera;
use crate::modes::fp::{FpState, WorldCamera};
use crate::view_mode::ViewMode;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin)
            .add_systems(Startup, setup_hud)
            .add_systems(
                Update,
                (route_hud_target, update_fps, update_coords, update_mode),
            );
    }
}

/// Marker on the HUD's root UI node so [`route_hud_target`] can find it
/// and update its [`TargetCamera`] when the view mode changes.
#[derive(Component)]
pub struct HudUiRoot;

#[derive(Component)]
struct FpsText;

#[derive(Component)]
struct CoordsText;

#[derive(Component)]
struct ModeText;

fn setup_hud(mut commands: Commands) {
    let text_style = TextStyle {
        font_size: 18.0,
        color: Color::WHITE,
        ..default()
    };
    // No TargetCamera here — `route_hud_target` attaches one from frame 1
    // onward. For the very first frame, `bevy_ui` resolves the camera via
    // the `IsDefaultUiCamera` marker on `WorldCamera` (set in
    // `modes::fp::setup_fp_scene`), which is correct for the default FP
    // mode and avoids `ui_layout_system`'s "no default camera" panic.
    commands
        .spawn((
            NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(8.0),
                    left: Val::Px(8.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    padding: UiRect::all(Val::Px(6.0)),
                    ..default()
                },
                background_color: Color::rgba(0.0, 0.0, 0.0, 0.45).into(),
                ..default()
            },
            HudUiRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                TextBundle::from_section("mode: fp", text_style.clone()),
                ModeText,
            ));
            parent.spawn((
                TextBundle::from_section("fps: --", text_style.clone()),
                FpsText,
            ));
            parent.spawn((
                TextBundle::from_section("xyz: (--, --, --)", text_style.clone()),
                CoordsText,
            ));
        });
}

/// Per-frame: point the HUD UI root at whichever camera is `is_active`
/// for the current [`ViewMode`]. Bevy 0.13's UI extraction follows
/// [`TargetCamera`], so this keeps the HUD on the live camera without
/// ever spawning a dedicated UI camera that would race the 3D output on
/// the same offscreen target.
fn route_hud_target(
    mode: Res<ViewMode>,
    world_cam: Query<Entity, (With<WorldCamera>, Without<BlitCamera>)>,
    blit_cam: Query<Entity, (With<BlitCamera>, Without<WorldCamera>)>,
    mut roots: Query<(Entity, Option<&mut TargetCamera>), With<HudUiRoot>>,
    mut commands: Commands,
) {
    let raster = matches!(*mode, ViewMode::Slice | ViewMode::Rts | ViewMode::Overview);
    let camera = if raster { blit_cam.get_single() } else { world_cam.get_single() };
    let Ok(target) = camera else {
        // Cameras not spawned yet (startup-frame gap). `IsDefaultUiCamera`
        // covers this frame; the explicit assignment lands next frame.
        return;
    };
    for (root, existing) in roots.iter_mut() {
        match existing {
            Some(mut tc) if tc.0 == target => {}
            Some(mut tc) => tc.0 = target,
            None => {
                commands.entity(root).insert(TargetCamera(target));
            }
        }
    }
}

fn update_fps(diag: Res<DiagnosticsStore>, mut q: Query<&mut Text, With<FpsText>>) {
    let Some(fps) = diag
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
    else {
        return;
    };
    if let Ok(mut text) = q.get_single_mut() {
        text.sections[0].value = format!("fps: {fps:>5.1}");
    }
}

fn update_coords(fp_state: Res<FpState>, mut q: Query<&mut Text, With<CoordsText>>) {
    if !fp_state.ready {
        return;
    }
    let p = fp_state.walk.observer.position;
    if let Ok(mut text) = q.get_single_mut() {
        text.sections[0].value = format!("xyz: ({:.1}, {:.1}, {:.1})", p.x, p.y, p.z);
    }
}

fn update_mode(mode: Res<ViewMode>, mut q: Query<&mut Text, With<ModeText>>) {
    if let Ok(mut text) = q.get_single_mut() {
        let want = format!("mode: {}", mode.label());
        if text.sections[0].value != want {
            text.sections[0].value = want;
        }
    }
}
