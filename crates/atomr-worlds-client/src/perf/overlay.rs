//! The in-app profiler overlay (toggle **F3**) and its toggle/update systems.
//!
//! One `Text` node, tagged [`crate::hud::HudUiRoot`] so the existing
//! `route_hud_target` points it at the live camera for free, positioned
//! top-right so it never overlaps the top-left HUD panel. Spawned hidden;
//! `update_perf_overlay` only runs while visible (`run_if`), so the overlay is
//! a single bool check per frame when off.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::hud::{FrameDiagBuffer, HudUiRoot};

use super::{Perf, PerfStats, Phase};

/// Whether the overlay is currently shown. Toggled by F3.
#[derive(Resource, Default)]
pub struct PerfOverlayState {
    pub visible: bool,
}

/// Marker on the overlay's text node.
#[derive(Component)]
pub struct PerfText;

/// `run_if` gate so `update_perf_overlay` is skipped (and does no formatting /
/// diagnostics reads) while the overlay is hidden.
pub fn overlay_visible(state: Res<PerfOverlayState>) -> bool {
    state.visible
}

/// F3 flips the overlay on/off and mirrors the state onto its `Visibility`.
pub fn perf_overlay_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<PerfOverlayState>,
    mut q: Query<&mut Visibility, With<PerfText>>,
) {
    if !keys.just_pressed(KeyCode::F3) {
        return;
    }
    state.visible = !state.visible;
    let want = if state.visible { Visibility::Visible } else { Visibility::Hidden };
    for mut v in q.iter_mut() {
        if *v != want {
            *v = want;
        }
    }
}

/// Spawn the (hidden) overlay node. Tagged `HudUiRoot` for camera routing and
/// `PerfText` so the toggle/update systems can find it.
pub fn setup_perf_overlay(mut commands: Commands) {
    commands.spawn((
        Text::new("perf (F3)"),
        TextFont { font_size: 13.0, ..default() },
        TextColor(Color::srgb(0.62, 1.0, 0.72)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            right: Val::Px(8.0),
            padding: UiRect::all(Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        Visibility::Hidden,
        HudUiRoot,
        PerfText,
    ));
}

/// Repaint the overlay from [`PerfStats`] (the previous frame's snapshot). Only
/// runs while visible. Self-timed under [`Phase::HudOverlay`] to prove it's cheap.
pub fn update_perf_overlay(
    perf: Res<Perf>,
    stats: Res<PerfStats>,
    diag: Res<DiagnosticsStore>,
    framebuf: Res<FrameDiagBuffer>,
    mut q: Query<&mut Text, With<PerfText>>,
) {
    let _g = perf.scope(Phase::HudOverlay);
    let Ok(mut text) = q.single_mut() else {
        return;
    };
    let fps = diag
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let ph = |p: Phase| stats.ema_us[p as usize] / 1000.0;
    let s = format!(
        "fps {fps:>5.1}   frame {:>5.1}ms   other {:>4.1}\n\
         edit {:>4.1}  eref {:>4.1}  stream {:>4.1}  spawn {:>4.1}  lodvis {:>4.1}\n\
         fracD {:>4.1}  fracA {:>4.1}  raster {:>4.1}  hud {:>4.1}\n\
         q  brick {}  load {}  frac {}  fref {}  eref {}  rebuild {}\n\
         {}",
        stats.frame_us as f64 / 1000.0,
        stats.other_us as f64 / 1000.0,
        ph(Phase::EditApply),
        ph(Phase::EditRefresh),
        ph(Phase::Streaming),
        ph(Phase::BrickSpawn),
        ph(Phase::LodVisibility),
        ph(Phase::FractureDispatch),
        ph(Phase::FractureApply),
        ph(Phase::SliceRtsRaster),
        ph(Phase::HudOverlay),
        stats.brick_in_flight,
        stats.loaded_chunks,
        stats.fracture_in_flight,
        stats.fracture_refresh_in_flight,
        stats.edit_refresh_in_flight,
        if stats.snapshot_rebuilding { "Y" } else { "-" },
        sparkline(&framebuf),
    );
    if text.0 != s {
        text.0 = s;
    }
}

/// 60-frame unicode block sparkline of recent whole-frame times, scaled to the
/// window max. Reuses the [`FrameDiagBuffer`] ring (no new state).
fn sparkline(buf: &FrameDiagBuffer) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    // `samples()` yields chronological order but is only a plain `Iterator`
    // (no `DoubleEndedIterator`), so collect then take the most-recent window.
    let all: Vec<u64> = buf.samples().map(|s| s.micros).collect();
    if all.is_empty() {
        return String::new();
    }
    let start = all.len().saturating_sub(60);
    let recent = &all[start..];
    let max = recent.iter().copied().max().unwrap_or(1).max(1);
    recent
        .iter()
        .map(|&us| {
            let idx = ((us.saturating_mul(BARS.len() as u64 - 1)) / max) as usize;
            BARS[idx.min(BARS.len() - 1)]
        })
        .collect()
}
