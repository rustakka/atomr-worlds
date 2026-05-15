//! Debug HUD — FPS, camera coords, active view mode.
//!
//! Native `bevy_ui` `TextBundle`s. The atomr-view `UiBridge` route is
//! documented in `docs/CLIENT_SERVER.md` as a follow-up; pulling
//! `atomr-view-backends` here today would drag in wgpu/egui/winit/uniffi
//! /pyo3 *and* trigger a `path` vs `git` collision on `atomr-core`.
//!
//! The HUD owns a dedicated [`HudCamera`] (Bevy `Camera2d`) that runs at a
//! higher `Camera::order` than the FP world camera and the slice/RTS/
//! overview blit camera, with `ClearColorConfig::None` so it composites on
//! top without wiping the colour buffer. Marking it `IsDefaultUiCamera`
//! keeps the bevy_ui layout system happy in harness mode (where every
//! other camera targets the offscreen image) and ensures the HUD shows up
//! on top of the raster blits, which previously covered it.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::camera::{ClearColorConfig, RenderTarget};
use bevy::render::view::RenderLayers;

use crate::modes::fp::FpState;
use crate::render::OffscreenTarget;
use crate::view_mode::ViewMode;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin)
            .add_systems(Startup, setup_hud)
            .add_systems(Update, (update_fps, update_coords, update_mode));
    }
}

/// Render layer reserved for HUD UI nodes — the dedicated [`HudCamera`]
/// renders only this layer, so it never picks up world meshes or the
/// slice/rts/overview blit sprite.
pub const HUD_LAYER: u8 = 31;

/// Marker on the dedicated UI camera that renders the HUD on top of every
/// other camera (order 10 > blit's 1 > world's 0).
#[derive(Component)]
pub struct HudCamera;

#[derive(Component)]
struct FpsText;

#[derive(Component)]
struct CoordsText;

#[derive(Component)]
struct ModeText;

fn setup_hud(mut commands: Commands, offscreen: Option<Res<OffscreenTarget>>) {
    // Mirror the FP / blit camera-target plumbing: in harness mode every
    // camera renders into the same offscreen `Image` so PNG captures see
    // the composed frame; otherwise the cameras fall back to the primary
    // window. Without this, harness screenshots would miss the HUD.
    let camera_target = offscreen
        .as_deref()
        .map(|t| RenderTarget::Image(t.image.clone()))
        .unwrap_or_default();

    let hud_camera = commands
        .spawn((
            Camera2dBundle {
                camera: Camera {
                    // Higher than blit's `order = 1` and the world camera's
                    // implicit `0`. `ClearColorConfig::None` preserves the
                    // contents underneath; only the HUD pixels are written.
                    order: 10,
                    clear_color: ClearColorConfig::None,
                    target: camera_target,
                    ..default()
                },
                ..default()
            },
            RenderLayers::layer(HUD_LAYER),
            // `bevy_ui`'s default-camera resolver panics in harness mode if
            // no camera carries this marker; pinning it here keeps the
            // resolution deterministic regardless of which world camera
            // happens to spawn first.
            bevy::ui::IsDefaultUiCamera,
            HudCamera,
        ))
        .id();

    let text_style = TextStyle {
        font_size: 18.0,
        color: Color::WHITE,
        ..default()
    };
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
            // Pin every HUD UI node to the dedicated camera; otherwise
            // bevy_ui picks the `IsDefaultUiCamera` at spawn time, which is
            // fine but explicit is clearer here.
            TargetCamera(hud_camera),
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
