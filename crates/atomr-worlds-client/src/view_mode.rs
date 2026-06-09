//! Five Phase-14 view modes + a hotkey-driven switcher.

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Resource)]
pub enum ViewMode {
    /// 1st-person walk.
    Fp,
    /// 3rd-person chase.
    Tp,
    /// Dwarf-Fortress orthographic slice.
    Slice,
    /// RTS oblique surface raster.
    Rts,
    /// Macro-state pyramid overview.
    Overview,
}

impl ViewMode {
    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Fp => "fp",
            ViewMode::Tp => "tp",
            ViewMode::Slice => "slice",
            ViewMode::Rts => "rts",
            ViewMode::Overview => "overview",
        }
    }
}

/// Bevy system: cycle the active [`ViewMode`] from number-key input.
///
/// `1..=5` jumps directly to fp/tp/slice/rts/overview; `Tab` cycles forward.
///
/// **Suppressed while grabbed into first-person** (actively editing): there,
/// `Tab` and the number row belong to the voxel editor (brush shape / material),
/// so view switching only fires when the cursor is free — press `Esc` to release
/// it, then switch. This keeps the editor's `Tab`/digit bindings from also
/// flipping the camera to third-person.
pub fn view_mode_input_system(
    keys: Res<ButtonInput<KeyCode>>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    harness: Option<Res<crate::harness::HarnessActive>>,
    mut mode: ResMut<ViewMode>,
) {
    let cursor_grabbed = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    let editing = cursor_grabbed || crate::harness::scripted_edit_active(harness.as_deref());
    if *mode == ViewMode::Fp && editing {
        return;
    }

    let new = if keys.just_pressed(KeyCode::Digit1) {
        Some(ViewMode::Fp)
    } else if keys.just_pressed(KeyCode::Digit2) {
        Some(ViewMode::Tp)
    } else if keys.just_pressed(KeyCode::Digit3) {
        Some(ViewMode::Slice)
    } else if keys.just_pressed(KeyCode::Digit4) {
        Some(ViewMode::Rts)
    } else if keys.just_pressed(KeyCode::Digit5) {
        Some(ViewMode::Overview)
    } else if keys.just_pressed(KeyCode::Tab) {
        Some(match *mode {
            ViewMode::Fp => ViewMode::Tp,
            ViewMode::Tp => ViewMode::Slice,
            ViewMode::Slice => ViewMode::Rts,
            ViewMode::Rts => ViewMode::Overview,
            ViewMode::Overview => ViewMode::Fp,
        })
    } else {
        None
    };

    if let Some(new) = new {
        if new != *mode {
            tracing::info!(target = "view_mode", from = %mode.label(), to = %new.label(), "switch");
            *mode = new;
        }
    }
}
