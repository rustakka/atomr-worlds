//! Debug HUD — FPS, camera coords, active view mode.
//!
//! Native `bevy_ui` `TextBundle`s. The atomr-view `UiBridge` route is
//! documented in `docs/CLIENT_SERVER.md` as a follow-up; pulling
//! `atomr-view-backends` here today would drag in wgpu/egui/winit/uniffi
//! /pyo3 *and* trigger a `path` vs `git` collision on `atomr-core`.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::modes::fp::FpState;
use crate::view_mode::ViewMode;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin)
            .add_systems(Startup, setup_hud)
            .add_systems(Update, (update_fps, update_coords, update_mode));
    }
}

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
    commands
        .spawn(NodeBundle {
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
        })
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
