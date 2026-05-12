//! Five Phase-14 view modes + a hotkey-driven switcher.

use bevy::prelude::*;

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
pub fn view_mode_input_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<ViewMode>,
) {
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
