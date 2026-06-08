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

use std::collections::VecDeque;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use atomr_worlds_core::interaction::ToolKind;

use crate::modes::blit::BlitCamera;
use crate::modes::edit::EditState;
use crate::modes::fp::{FpState, WorldCamera};
use crate::view_mode::ViewMode;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, setup_hud)
            .add_systems(
                Update,
                (
                    route_hud_target,
                    update_fps,
                    update_coords,
                    update_mode,
                    update_edit_readout,
                    update_crosshair_visibility,
                ),
            );
    }
}

// ---------------------------------------------------------------------------
// Frame-time diagnostics ring buffer
// ---------------------------------------------------------------------------

/// Capacity of the [`FrameDiagBuffer`] ring buffer. 1024 frames ≈ 17 s at
/// 60 Hz — enough to span a sprint harness scenario.
pub const FRAME_DIAG_BUFFER_LEN: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct FrameSample {
    pub frame: u64,
    pub micros: u64,
}

/// Ring buffer of per-frame µs. Updated every frame by
/// [`record_frame_diag`]; consumed by the `dump_frame_diag` harness event
/// when scenarios want a frame-time trace.
#[derive(Resource)]
pub struct FrameDiagBuffer {
    samples: VecDeque<FrameSample>,
    next_frame: u64,
}

impl Default for FrameDiagBuffer {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(FRAME_DIAG_BUFFER_LEN),
            next_frame: 0,
        }
    }
}

impl FrameDiagBuffer {
    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
    pub fn capacity(&self) -> usize { FRAME_DIAG_BUFFER_LEN }
    pub fn samples(&self) -> impl Iterator<Item = &FrameSample> + '_ { self.samples.iter() }

    pub fn push(&mut self, micros: u64) -> u64 {
        let frame = self.next_frame;
        if self.samples.len() == FRAME_DIAG_BUFFER_LEN {
            self.samples.pop_front();
        }
        self.samples.push_back(FrameSample { frame, micros });
        self.next_frame = self.next_frame.wrapping_add(1);
        frame
    }
}

pub struct FrameDiagPlugin;

impl Plugin for FrameDiagPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameDiagBuffer>()
            .add_systems(First, record_frame_diag);
    }
}

fn record_frame_diag(time: Res<Time>, mut buf: ResMut<FrameDiagBuffer>) {
    let micros = (time.delta_secs_f64() * 1.0e6).round().max(0.0) as u64;
    buf.push(micros);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_diag_buffer_caps_at_capacity_and_drops_oldest() {
        let mut buf = FrameDiagBuffer::default();
        for i in 0..(FRAME_DIAG_BUFFER_LEN as u64 + 16) {
            buf.push(i);
        }
        assert_eq!(buf.len(), FRAME_DIAG_BUFFER_LEN);
        let first = buf.samples().next().unwrap();
        assert_eq!(first.frame, 16);
        assert_eq!(first.micros, 16);
        let last = buf.samples().last().unwrap();
        assert_eq!(last.frame, FRAME_DIAG_BUFFER_LEN as u64 + 15);
    }

    #[test]
    fn frame_diag_buffer_assigns_monotonic_frame_ids() {
        let mut buf = FrameDiagBuffer::default();
        let a = buf.push(100);
        let b = buf.push(200);
        let c = buf.push(300);
        assert_eq!((a, b, c), (0, 1, 2));
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

/// The voxel-edit tool/material/radius readout line in the HUD panel.
#[derive(Component)]
struct EditText;

/// A full-screen centered crosshair container. Two of these (one per bar) form
/// the `+`. Toggled visible only in FP by [`update_crosshair_visibility`].
#[derive(Component)]
struct Crosshair;

fn setup_hud(mut commands: Commands, harness: Option<Res<crate::harness::HarnessActive>>) {
    // Bevy 0.15+ Text API: each text node is `Text` + `TextFont` + `TextColor`;
    // the root is a `Node` + `BackgroundColor` (bundles were removed in favor of
    // required components).
    let font = TextFont { font_size: 18.0, ..default() };
    let color = TextColor(Color::WHITE);
    // No UiTargetCamera here — `route_hud_target` attaches one from frame 1
    // onward. For the very first frame, `bevy_ui` resolves the camera via
    // the `IsDefaultUiCamera` marker on `WorldCamera` (set in
    // `modes::fp::setup_fp_scene`), which is correct for the default FP
    // mode and avoids `ui_layout_system`'s "no default camera" panic.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                left: Val::Px(8.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                padding: UiRect::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
            HudUiRoot,
        ))
        .with_children(|parent| {
            parent.spawn((Text::new("mode: fp"), font.clone(), color, ModeText));
            parent.spawn((Text::new("fps: --"), font.clone(), color, FpsText));
            parent.spawn((Text::new("xyz: (--, --, --)"), font.clone(), color, CoordsText));
            parent.spawn((Text::new("tool: voxel  mat: 1"), font.clone(), color, EditText));
        });

    // Crosshair: two thin centered bars forming a `+`. Each lives in its own
    // full-screen flex-centered container so the bars overlap dead-center.
    // Tagged `HudUiRoot` so `route_hud_target` points it at the live camera,
    // and `Crosshair` so `update_crosshair_visibility` shows it only in FP.
    // Skipped under the harness so FP captures don't gain crosshair pixels.
    if harness.is_none() {
        for (w, h) in [(2.0_f32, 14.0_f32), (14.0_f32, 2.0_f32)] {
            commands
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    Visibility::Hidden,
                    HudUiRoot,
                    Crosshair,
                ))
                .with_children(|p| {
                    p.spawn((
                        Node { width: Val::Px(w), height: Val::Px(h), ..default() },
                        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.85)),
                    ));
                });
        }
    }
}

/// Show the crosshair only in first-person (where editing happens); hide it in
/// every other view mode.
fn update_crosshair_visibility(mode: Res<ViewMode>, mut q: Query<&mut Visibility, With<Crosshair>>) {
    let want = if *mode == ViewMode::Fp { Visibility::Visible } else { Visibility::Hidden };
    for mut v in q.iter_mut() {
        if *v != want {
            *v = want;
        }
    }
}

/// Mirror the current edit tool / material / brush radius into the HUD readout.
fn update_edit_readout(edit: Option<Res<EditState>>, mut q: Query<&mut Text, With<EditText>>) {
    let Some(edit) = edit else { return };
    let tool = match edit.tool {
        ToolKind::Voxel => "voxel",
        ToolKind::Sphere => "sphere",
        ToolKind::Cube => "cube",
        ToolKind::Cone => "cone",
    };
    let want = if edit.tool == ToolKind::Voxel {
        format!("tool: {tool}  mat: {}", edit.selected_material)
    } else {
        format!("tool: {tool} r{:.0}  mat: {}", edit.radius_voxels, edit.selected_material)
    };
    if let Ok(mut text) = q.single_mut() {
        if text.0 != want {
            text.0 = want;
        }
    }
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
    mut roots: Query<(Entity, Option<&mut UiTargetCamera>), With<HudUiRoot>>,
    mut commands: Commands,
) {
    let raster = matches!(*mode, ViewMode::Slice | ViewMode::Rts | ViewMode::Overview);
    let camera = if raster { blit_cam.single() } else { world_cam.single() };
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
                commands.entity(root).insert(UiTargetCamera(target));
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
    if let Ok(mut text) = q.single_mut() {
        text.0 = format!("fps: {fps:>5.1}");
    }
}

fn update_coords(fp_state: Res<FpState>, mut q: Query<&mut Text, With<CoordsText>>) {
    if !fp_state.ready {
        return;
    }
    let p = fp_state.walk.observer.position;
    if let Ok(mut text) = q.single_mut() {
        text.0 = format!("xyz: ({:.1}, {:.1}, {:.1})", p.x, p.y, p.z);
    }
}

fn update_mode(mode: Res<ViewMode>, mut q: Query<&mut Text, With<ModeText>>) {
    if let Ok(mut text) = q.single_mut() {
        let want = format!("mode: {}", mode.label());
        if text.0 != want {
            text.0 = want;
        }
    }
}
