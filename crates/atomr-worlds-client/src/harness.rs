//! Screenshot test harness — drives the Bevy client through a scripted
//! scenario and captures PNGs at chosen frames.
//!
//! Activated by the `--harness <scenario.toml>` CLI flag. When inactive,
//! adds nothing to the App.
//!
//! ## Timeline semantics
//!
//! `HarnessClock::frame` starts at `0` and is incremented in the `First`
//! schedule each frame. Scenario events fire at the frame whose value
//! matches `HarnessClock::frame` after that bump — so the first frame
//! visible to scenario events is `frame = 1`. To keep scenarios readable
//! we treat scenario-author event frames as offsets relative to the end
//! of warmup: every event's `frame` is rewritten to
//! `frame + warmup_frames` at load time. Authors then write `frame = 0`
//! for "first frame after warmup".
//!
//! ## `key_tap`
//!
//! `key_tap` is desugared at load time into a `key_press` at frame `N`
//! and a `key_release` at frame `N + 1`.
//!
//! ## Deterministic dt (limitation)
//!
//! Bevy 0.13 owns the `Time` resource via its own time-update system; we
//! deliberately do NOT try to override it for v1. Frame counts drive the
//! timeline (events fire at frame boundaries regardless of dt), but the
//! per-frame movement amount produced by `fp_input` will scale with wall
//! clock dt. The harness is therefore not pixel-deterministic across
//! machines — it is intended for visual regression review, not exact-
//! bytes golden comparisons.
//!
//! ## Seed
//!
//! For v1, `Scenario::seed` is **ignored**; the CLI `--seed` flag wins.

use std::path::{Path, PathBuf};

use bevy::app::AppExit;
use bevy::input::mouse::MouseMotion;
use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, Window};
use serde::Deserialize;

use crate::render::{
    CaptureOutcomes, CaptureQueueHandle, OffscreenTarget, RenderConfig, RenderPreset,
    WorldTime,
};
use crate::view_mode::ViewMode;

// ---------------------------------------------------------------------------
// Scenario schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct Scenario {
    #[serde(default = "default_seed")]
    pub seed: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_warmup")]
    pub warmup_frames: u64,
    #[serde(default = "default_prefix")]
    pub output_prefix: String,
    #[serde(default)]
    pub events: Vec<ScenarioEvent>,
}

fn default_seed() -> String {
    "0xDEADBEEFCAFEF00D".into()
}
fn default_mode() -> String {
    "fp".into()
}
fn default_width() -> u32 {
    1280
}
fn default_height() -> u32 {
    720
}
fn default_warmup() -> u64 {
    60
}
fn default_prefix() -> String {
    "shot".into()
}

#[derive(Debug, Deserialize, Clone)]
pub struct ScenarioEvent {
    pub frame: u64,
    /// One of: `"key_press" | "key_release" | "key_tap" | "screenshot" |
    /// "mouse_move" | "mouse_button_press" | "mouse_button_release" |
    /// "exit" | "set_time_of_day" | "set_render_preset" | "set_strategy"`.
    pub kind: String,
    #[serde(default)]
    pub key: Option<String>,
    /// Mouse button name for `mouse_button_*` events: "Left" | "Right" |
    /// "Middle".
    #[serde(default)]
    pub button: Option<String>,
    #[serde(default)]
    pub dx: Option<f32>,
    #[serde(default)]
    pub dy: Option<f32>,
    /// Hours-of-day for `set_time_of_day`.
    #[serde(default)]
    pub hours: Option<f32>,
    /// Preset name for `set_render_preset`.
    #[serde(default)]
    pub preset: Option<String>,
    /// Strategy slot name for `set_strategy` (e.g. "shading", "sky").
    #[serde(default)]
    pub slot: Option<String>,
    /// Strategy implementation name for `set_strategy` (e.g. "AcesTonemap").
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

impl Scenario {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {}", path.display(), e))?;
        let mut scenario: Scenario = toml::from_str(&text)
            .map_err(|e| format!("parsing {}: {}", path.display(), e))?;

        // Validate keys and desugar `key_tap` into press+release.
        let mut expanded: Vec<ScenarioEvent> = Vec::with_capacity(scenario.events.len());
        for (idx, ev) in scenario.events.iter().enumerate() {
            match ev.kind.as_str() {
                "key_press" | "key_release" => {
                    let key_name = ev.key.as_deref().ok_or_else(|| {
                        format!("event #{idx} ({}): missing `key`", ev.kind)
                    })?;
                    if key_from_name(key_name).is_none() {
                        return Err(format!(
                            "event #{idx} ({}): unknown key `{}`",
                            ev.kind, key_name
                        )
                        .into());
                    }
                    expanded.push(ev.clone());
                }
                "key_tap" => {
                    let key_name = ev.key.as_deref().ok_or_else(|| {
                        format!("event #{idx} (key_tap): missing `key`")
                    })?;
                    if key_from_name(key_name).is_none() {
                        return Err(format!(
                            "event #{idx} (key_tap): unknown key `{}`",
                            key_name
                        )
                        .into());
                    }
                    expanded.push(ScenarioEvent {
                        frame: ev.frame,
                        kind: "key_press".into(),
                        key: ev.key.clone(),
                        button: None,
                        dx: None,
                        dy: None,
                        hours: None,
                        preset: None,
                        slot: None,
                        strategy: None,
                        note: ev.note.clone(),
                    });
                    expanded.push(ScenarioEvent {
                        frame: ev.frame.saturating_add(1),
                        kind: "key_release".into(),
                        key: ev.key.clone(),
                        button: None,
                        dx: None,
                        dy: None,
                        hours: None,
                        preset: None,
                        slot: None,
                        strategy: None,
                        note: None,
                    });
                }
                "mouse_move" => {
                    if ev.dx.is_none() && ev.dy.is_none() {
                        return Err(format!(
                            "event #{idx} (mouse_move): at least one of `dx`/`dy` required"
                        )
                        .into());
                    }
                    expanded.push(ev.clone());
                }
                "mouse_button_press" | "mouse_button_release" => {
                    let button_name = ev.button.as_deref().ok_or_else(|| {
                        format!("event #{idx} ({}): missing `button`", ev.kind)
                    })?;
                    if mouse_button_from_name(button_name).is_none() {
                        return Err(format!(
                            "event #{idx} ({}): unknown button `{}`",
                            ev.kind, button_name
                        )
                        .into());
                    }
                    expanded.push(ev.clone());
                }
                "screenshot" | "exit" => {
                    expanded.push(ev.clone());
                }
                "set_time_of_day" => {
                    if ev.hours.is_none() {
                        return Err(format!(
                            "event #{idx} (set_time_of_day): missing `hours`"
                        )
                        .into());
                    }
                    expanded.push(ev.clone());
                }
                "set_render_preset" => {
                    let preset_name = ev.preset.as_deref().ok_or_else(|| {
                        format!("event #{idx} (set_render_preset): missing `preset`")
                    })?;
                    if RenderPreset::from_str(preset_name).is_none() {
                        return Err(format!(
                            "event #{idx} (set_render_preset): unknown preset `{}`",
                            preset_name
                        )
                        .into());
                    }
                    expanded.push(ev.clone());
                }
                "set_strategy" => {
                    if ev.slot.is_none() || ev.strategy.is_none() {
                        return Err(format!(
                            "event #{idx} (set_strategy): missing `slot` or `strategy`"
                        )
                        .into());
                    }
                    expanded.push(ev.clone());
                }
                other => {
                    return Err(format!(
                        "event #{idx}: unknown kind `{}`",
                        other
                    )
                    .into());
                }
            }
        }

        // Resolve to absolute frames (add warmup) and sort.
        for ev in expanded.iter_mut() {
            ev.frame = ev.frame.saturating_add(scenario.warmup_frames);
        }
        expanded.sort_by_key(|e| e.frame);
        scenario.events = expanded;

        Ok(scenario)
    }

    pub fn initial_view_mode(&self) -> ViewMode {
        match self.mode.as_str() {
            "fp" => ViewMode::Fp,
            "tp" => ViewMode::Tp,
            "slice" => ViewMode::Slice,
            "rts" => ViewMode::Rts,
            "overview" => ViewMode::Overview,
            _ => ViewMode::Fp,
        }
    }

    pub fn last_event_frame(&self) -> u64 {
        self.events.iter().map(|e| e.frame).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// KeyCode mapping
// ---------------------------------------------------------------------------

fn key_from_name(s: &str) -> Option<KeyCode> {
    Some(match s {
        // View mode
        "Digit1" => KeyCode::Digit1,
        "Digit2" => KeyCode::Digit2,
        "Digit3" => KeyCode::Digit3,
        "Digit4" => KeyCode::Digit4,
        "Digit5" => KeyCode::Digit5,
        "Tab" => KeyCode::Tab,
        // FP movement
        "KeyW" => KeyCode::KeyW,
        "KeyA" => KeyCode::KeyA,
        "KeyS" => KeyCode::KeyS,
        "KeyD" => KeyCode::KeyD,
        "Space" => KeyCode::Space,
        "ShiftLeft" => KeyCode::ShiftLeft,
        "ShiftRight" => KeyCode::ShiftRight,
        "ControlLeft" => KeyCode::ControlLeft,
        "ControlRight" => KeyCode::ControlRight,
        "KeyC" => KeyCode::KeyC,
        // Arrow look
        "ArrowUp" => KeyCode::ArrowUp,
        "ArrowDown" => KeyCode::ArrowDown,
        "ArrowLeft" => KeyCode::ArrowLeft,
        "ArrowRight" => KeyCode::ArrowRight,
        // FP cursor
        "Escape" => KeyCode::Escape,
        // Slice
        "PageUp" => KeyCode::PageUp,
        "PageDown" => KeyCode::PageDown,
        "Equal" => KeyCode::Equal,
        "Minus" => KeyCode::Minus,
        // RTS
        "KeyQ" => KeyCode::KeyQ,
        "KeyE" => KeyCode::KeyE,
        // Overview
        "KeyP" => KeyCode::KeyP,
        _ => return None,
    })
}

fn mouse_button_from_name(s: &str) -> Option<MouseButton> {
    Some(match s {
        "Left" => MouseButton::Left,
        "Right" => MouseButton::Right,
        "Middle" => MouseButton::Middle,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct HarnessConfig {
    pub scenario: Scenario,
    pub output_dir: PathBuf,
}

#[derive(Resource, Default)]
pub struct HarnessClock {
    pub frame: u64,
}

#[derive(Resource)]
pub struct HarnessState {
    /// Counter for screenshot filename suffix.
    pub shot_index: usize,
    pub paths: Vec<PathBuf>,
    pub exit_requested: bool,
    /// `last_event_frame + 600` — safety-net cutoff if no `exit` event fires.
    pub deadline: u64,
}

/// Marker resource: presence indicates harness mode is active. Other
/// systems (`grab_cursor`, `fp_input`) check `Option<Res<HarnessActive>>`
/// to bypass cursor grab and read `MouseMotion` unconditionally.
#[derive(Resource)]
pub struct HarnessActive;

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct HarnessPlugin;

impl Plugin for HarnessPlugin {
    fn build(&self, app: &mut App) {
        let cfg = app
            .world
            .get_resource::<HarnessConfig>()
            .expect("HarnessConfig must be inserted before HarnessPlugin is added");
        let deadline = cfg.scenario.last_event_frame() + 600;
        app.init_resource::<HarnessClock>()
            .insert_resource(HarnessState {
                shot_index: 0,
                paths: Vec::new(),
                exit_requested: false,
                deadline,
            })
            .insert_resource(HarnessActive)
            .add_systems(First, tick_clock)
            .add_systems(PreUpdate, drive_input_events)
            .add_systems(PostUpdate, drive_screenshots)
            .add_systems(Last, (drain_capture_outcomes, drive_exit).chain());
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

fn tick_clock(mut clock: ResMut<HarnessClock>) {
    clock.frame = clock.frame.wrapping_add(1);
}

#[allow(clippy::too_many_arguments)]
fn drive_input_events(
    clock: Res<HarnessClock>,
    cfg: Res<HarnessConfig>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut mouse_buttons: ResMut<ButtonInput<MouseButton>>,
    mut mouse_writer: EventWriter<MouseMotion>,
    mut world_time: Option<ResMut<WorldTime>>,
    mut render_config: Option<ResMut<RenderConfig>>,
) {
    let now = clock.frame;
    for ev in cfg.scenario.events.iter().filter(|e| e.frame == now) {
        match ev.kind.as_str() {
            "key_press" => {
                if let Some(k) = ev.key.as_deref().and_then(key_from_name) {
                    keys.press(k);
                }
            }
            "key_release" => {
                if let Some(k) = ev.key.as_deref().and_then(key_from_name) {
                    keys.release(k);
                }
            }
            "mouse_move" => {
                let dx = ev.dx.unwrap_or(0.0);
                let dy = ev.dy.unwrap_or(0.0);
                mouse_writer.send(MouseMotion {
                    delta: Vec2::new(dx, dy),
                });
            }
            "mouse_button_press" => {
                if let Some(b) = ev.button.as_deref().and_then(mouse_button_from_name) {
                    mouse_buttons.press(b);
                }
            }
            "mouse_button_release" => {
                if let Some(b) = ev.button.as_deref().and_then(mouse_button_from_name) {
                    mouse_buttons.release(b);
                }
            }
            "set_time_of_day" => {
                if let (Some(h), Some(t)) = (ev.hours, world_time.as_deref_mut()) {
                    t.set(h);
                }
            }
            "set_render_preset" => {
                if let (Some(name), Some(cfg)) =
                    (ev.preset.as_deref(), render_config.as_deref_mut())
                {
                    if let Some(preset) = RenderPreset::from_str(name) {
                        cfg.apply_preset(preset);
                    }
                }
            }
            "set_strategy" => {
                if let (Some(slot), Some(strat), Some(cfg)) = (
                    ev.slot.as_deref(),
                    ev.strategy.as_deref(),
                    render_config.as_deref_mut(),
                ) {
                    if !crate::render::apply_strategy_by_name(cfg, slot, strat) {
                        eprintln!(
                            "HARNESS_WARNING set_strategy slot={} name={} unknown",
                            slot, strat
                        );
                    }
                }
            }
            // screenshot / exit handled in their own systems
            _ => {}
        }
    }
}

/// Enqueue a capture request into the offscreen plugin's queue. The
/// PNG is written from the `RenderApp`'s image-copy system; we drain
/// outcomes in [`drain_capture_outcomes`].
fn drive_screenshots(
    clock: Res<HarnessClock>,
    cfg: Res<HarnessConfig>,
    mut state: ResMut<HarnessState>,
    queue: Option<Res<CaptureQueueHandle>>,
) {
    let now = clock.frame;
    let Some(queue) = queue else {
        return; // not in harness/offscreen mode
    };
    for ev in cfg.scenario.events.iter().filter(|e| e.frame == now) {
        if ev.kind != "screenshot" {
            continue;
        }
        let path = cfg.output_dir.join(format!(
            "{}_{:04}.png",
            cfg.scenario.output_prefix, state.shot_index
        ));
        let mut q = queue.0.lock().unwrap();
        let id = q.next_id;
        q.next_id = q.next_id.wrapping_add(1);
        q.pending.push_back((id, path.clone()));
        drop(q);
        state.paths.push(path);
        state.shot_index += 1;
    }
}

/// Pull capture outcomes off the shared bus and emit `HARNESS_SHOT` /
/// `HARNESS_ERROR` lines. Runs in `Last` so it sees the result of
/// captures issued earlier in the same frame.
fn drain_capture_outcomes(outcomes: Option<Res<CaptureOutcomes>>) {
    let Some(outcomes) = outcomes else { return };
    let mut taken = Vec::new();
    {
        let mut guard = outcomes.0.lock().unwrap();
        std::mem::swap(&mut *guard, &mut taken);
    }
    for outcome in taken {
        if outcome.ok {
            println!("HARNESS_SHOT {}", outcome.path.display());
        } else {
            eprintln!(
                "HARNESS_ERROR capture for {} failed: {}",
                outcome.path.display(),
                outcome.message.unwrap_or_else(|| "(no detail)".into())
            );
        }
    }
}

// ---------------------------------------------------------------------------
// External X11 capture (replaces Bevy's broken-on-hybrid-GPU ScreenshotManager)
// ---------------------------------------------------------------------------

/// Capture the X11 window with WM_NAME == `title` and write a PNG at `path`.
/// Invokes `xwd -name <title> -silent` and parses the XWD2 dump in-process.
fn capture_window_png(title: &str, path: &std::path::Path) -> Result<(), String> {
    let out = std::process::Command::new("xwd")
        .args(["-name", title, "-silent"])
        .output()
        .map_err(|e| format!("spawning xwd: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "xwd exited with status {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let img = parse_xwd_to_rgba(&out.stdout)
        .map_err(|e| format!("parsing xwd output ({} bytes): {}", out.stdout.len(), e))?;
    img.save(path).map_err(|e| format!("writing png {}: {}", path.display(), e))?;
    Ok(())
}

/// Parse an XWD2 (file_version=7) dump into an RGBA image. Handles the
/// common case: ZPixmap, depth 24, 32 bits/pixel, big-endian header,
/// 8-8-8 RGB masks.
fn parse_xwd_to_rgba(bytes: &[u8]) -> Result<image::RgbaImage, String> {
    if bytes.len() < 100 {
        return Err(format!("xwd buffer too short: {} bytes", bytes.len()));
    }
    let u32be = |off: usize| -> u32 {
        u32::from_be_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let header_size = u32be(0) as usize;
    let file_version = u32be(4);
    let pixmap_format = u32be(8);
    let pixmap_depth = u32be(12);
    let pixmap_width = u32be(16);
    let pixmap_height = u32be(20);
    let byte_order = u32be(28);          // 0 = LSBFirst, 1 = MSBFirst
    let bits_per_pixel = u32be(44);
    let bytes_per_line = u32be(48) as usize;
    let red_mask = u32be(56);
    let green_mask = u32be(60);
    let blue_mask = u32be(64);
    let ncolors = u32be(76) as usize;

    if file_version != 7 {
        return Err(format!("unsupported xwd file_version {}", file_version));
    }
    if pixmap_format != 2 {
        return Err(format!("unsupported pixmap_format {} (need 2=ZPixmap)", pixmap_format));
    }
    if pixmap_depth != 24 && pixmap_depth != 32 {
        return Err(format!("unsupported pixmap_depth {} (need 24 or 32)", pixmap_depth));
    }
    if bits_per_pixel != 24 && bits_per_pixel != 32 {
        return Err(format!("unsupported bits_per_pixel {} (need 24 or 32)", bits_per_pixel));
    }
    let bytes_per_pixel = (bits_per_pixel / 8) as usize;

    // Pixel data starts after the header + window name + colormap.
    let colormap_bytes = ncolors * 12;
    let pixel_start = header_size + colormap_bytes;
    if pixel_start > bytes.len() {
        return Err("xwd buffer truncated before pixel data".into());
    }
    let pixels = &bytes[pixel_start..];
    let expected = bytes_per_line * pixmap_height as usize;
    if pixels.len() < expected {
        return Err(format!(
            "xwd pixel data short: have {}, need {} ({}x{} stride {} bpp {})",
            pixels.len(), expected, pixmap_width, pixmap_height, bytes_per_line, bits_per_pixel
        ));
    }

    let (r_shift, g_shift, b_shift) = (
        mask_to_shift(red_mask)?,
        mask_to_shift(green_mask)?,
        mask_to_shift(blue_mask)?,
    );

    let mut out = image::RgbaImage::new(pixmap_width, pixmap_height);
    for y in 0..pixmap_height as usize {
        let row_start = y * bytes_per_line;
        for x in 0..pixmap_width as usize {
            let px = &pixels[row_start + x * bytes_per_pixel..row_start + x * bytes_per_pixel + bytes_per_pixel];
            // Pack the per-pixel bytes into a u32 according to byte_order,
            // padding to 32 bits if the pixel is 24 bpp.
            let value: u32 = if bytes_per_pixel == 4 {
                if byte_order == 0 {
                    u32::from_le_bytes([px[0], px[1], px[2], px[3]])
                } else {
                    u32::from_be_bytes([px[0], px[1], px[2], px[3]])
                }
            } else {
                // 24bpp: three bytes, treat MSB as 0.
                if byte_order == 0 {
                    u32::from_le_bytes([px[0], px[1], px[2], 0])
                } else {
                    u32::from_be_bytes([0, px[0], px[1], px[2]])
                }
            };
            let r = ((value & red_mask) >> r_shift) as u8;
            let g = ((value & green_mask) >> g_shift) as u8;
            let b = ((value & blue_mask) >> b_shift) as u8;
            out.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, 255]));
        }
    }
    Ok(out)
}

fn mask_to_shift(mask: u32) -> Result<u32, String> {
    if mask == 0 {
        return Err("zero colour mask".into());
    }
    Ok(mask.trailing_zeros())
}

fn drive_exit(
    clock: Res<HarnessClock>,
    cfg: Res<HarnessConfig>,
    mut state: ResMut<HarnessState>,
    mut exit_writer: EventWriter<AppExit>,
) {
    let now = clock.frame;
    for ev in cfg.scenario.events.iter().filter(|e| e.frame == now) {
        if ev.kind == "exit" {
            state.exit_requested = true;
        }
    }

    let grace_until = cfg.scenario.last_event_frame() + 5;
    if state.exit_requested && now >= grace_until {
        exit_writer.send(AppExit);
        return;
    }

    if now > state.deadline {
        eprintln!(
            "HARNESS_WARNING deadline frame {} exceeded; forcing exit",
            state.deadline
        );
        exit_writer.send(AppExit);
    }
}
