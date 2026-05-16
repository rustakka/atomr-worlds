//! Bevy-driven atomr-worlds client.
//!
//! Architecture (see `docs/CLIENT_SERVER.md` for the deep dive):
//! - A multi-threaded tokio runtime owns the chosen `WorldHost` backend
//!   (`LocalHost`, `RemoteHost`, or `ClusterHost`). It outlives the Bevy
//!   `App`.
//! - Bevy systems get a synchronous [`WorldQuery`](atomr_worlds_view::WorldQuery)
//!   bridge via [`LocalHostQuery`](atomr_worlds_host::LocalHostQuery), which
//!   uses the tokio handle's `block_on` from off-runtime threads.
//! - The five Phase-14 view modes are separate Bevy plugins, gated on the
//!   `ViewMode` resource. Press 1..=5 to switch (Tab cycles).

#![forbid(unsafe_code)]

mod brick_gen;
mod cli;
mod harness;
mod host_backend;
mod hud;
mod modes;
mod render;
mod view_mode;
mod world_runtime;
mod world_stream;

use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use bevy::prelude::*;
use bevy::window::{PresentMode, WindowResolution};
use bevy::winit::{UpdateMode, WinitSettings};
use clap::Parser;

use crate::cli::Cli;
use crate::view_mode::{view_mode_input_system, ViewMode};
use crate::world_runtime::{ActiveWorld, WorldRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Send logs to stderr so stdout stays clean for `HARNESS_SHOT` lines.
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();
    let cli = Cli::parse();

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .thread_name("atomr-worlds-client-rt")
            .build()?,
    );
    let host = runtime
        .block_on(host_backend::build_backend(&cli))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let world_runtime = WorldRuntime::new(runtime, host);
    let active = ActiveWorld {
        addr: WorldAddr::ROOT,
        seed: cli.seed,
        shape: atomr_worlds_core::shape::WorldShape::default_world(),
    };

    // Load harness scenario (if any) early so we can size the window
    // before the swapchain is created — resizing later on some drivers
    // produces an "Unrecognized present mode" warning and breaks the
    // capture path.
    let harness_bits = if let Some(path) = cli.harness.as_deref() {
        let scenario = harness::Scenario::load(path).map_err(
            |e| -> Box<dyn std::error::Error> {
                format!("loading harness scenario: {e}").into()
            },
        )?;
        let out = cli
            .harness_out
            .clone()
            .expect("clap requires --harness-out when --harness is set");
        std::fs::create_dir_all(&out)?;
        let out_abs = std::fs::canonicalize(&out)?;
        Some((scenario, out_abs))
    } else {
        None
    };

    let (initial_mode, window_resolution) = match harness_bits.as_ref() {
        Some((scenario, _)) => (
            scenario.initial_view_mode(),
            // scale_factor_override(1.0) so logical pixels == physical
            // pixels. Without this, a HiDPI display (scale 2) would
            // create a 2x larger swapchain than the capture path expects.
            WindowResolution::new(scenario.width as f32, scenario.height as f32)
                .with_scale_factor_override(1.0),
        ),
        None => (ViewMode::Fp, WindowResolution::default()),
    };

    let mut app = App::new();
    // Resolve the assets dir relative to the binary so the Step 8 + 9
    // shaders (`assets/shaders/{voxel_material,sky_dome}.wgsl`) load
    // regardless of where the binary is invoked from. We try a few
    // canonical locations (workspace root cwd, exec dir, parent of
    // exec) and fall back to Bevy's default if none exist.
    let asset_root = resolve_asset_root();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::asset::AssetPlugin { file_path: asset_root, ..default() })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: format!("atomr-worlds-client [{:?}]", cli.backend),
                    resolution: window_resolution,
                    // Force FIFO — some drivers report an exotic present mode
                    // (e.g. FIFO_LATEST_READY_EXT = 1000361000) that wgpu 0.19
                    // doesn't recognise.
                    present_mode: PresentMode::Fifo,
                    // In harness mode, request the window be created
                    // unfocused so it doesn't steal focus from whatever the
                    // user is doing. We still create a visible window (the
                    // X11/hybrid-GPU presentation path needs one to exist
                    // alongside the offscreen render target) — it just sits
                    // in the background. Interactive runs keep the default
                    // (focused) behavior.
                    focused: harness_bits.is_none(),
                    ..default()
                }),
                ..default()
            }),
    )
        .insert_resource(world_runtime)
        .insert_resource(active)
        .insert_resource(initial_mode)
        .insert_resource(ClearColor(Color::rgb(0.45, 0.65, 0.85)))
        .insert_resource({
            let mut cfg = render::RenderConfig::default();
            cfg.apply_perf_preset(match cli.perf {
                cli::PerfPreset::Balanced => render::PerfPreset::Balanced,
                cli::PerfPreset::Quality => render::PerfPreset::Quality,
            });
            cfg
        })
        .add_plugins(render::RenderPlugin)
        .add_plugins(render::HorizonShellPlugin)
        .add_plugins(world_stream::ChunkStreamerPlugin)
        .add_plugins(brick_gen::BrickGenPlugin)
        .add_plugins(modes::fp::FpPlugin)
        .add_plugins(modes::tp::TpPlugin)
        .add_plugins(modes::blit::BlitPlugin)
        .add_plugins(modes::slice::SlicePlugin)
        .add_plugins(modes::rts::RtsPlugin)
        .add_plugins(modes::overview::OverviewPlugin)
        .add_plugins(hud::HudPlugin)
        .add_plugins(hud::FrameDiagPlugin)
        .add_systems(Update, view_mode_input_system);

    if let Some((scenario, out_abs)) = harness_bits {
        // Offscreen render target + capture plugin sized to the scenario
        // so the readback PNG is exactly `width x height`. Installed
        // BEFORE the harness so the offscreen Image asset exists when
        // FpPlugin's setup_fp_scene runs and points the camera at it.
        app.add_plugins(render::OffscreenCapturePlugin {
            width: scenario.width,
            height: scenario.height,
        });
        app.insert_resource(harness::HarnessConfig {
            scenario,
            output_dir: out_abs,
        });
        app.add_plugins(harness::HarnessPlugin);
        // The window is created `focused: false` in harness mode so it
        // doesn't steal focus from whatever the user is doing — but
        // Bevy's default `WinitSettings::game()` throttles unfocused
        // windows to `ReactiveLowPower`, which starves the brick
        // streamer and produces sky-only PNGs. Force continuous updates
        // so the scenario plays out at the same cadence whether or not
        // the harness window happens to be the active one.
        app.insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: UpdateMode::Continuous,
        });
    }

    app.run();

    Ok(())
}

/// Resolve the directory the [`AssetServer`] reads from. Bevy's
/// `AssetPlugin::file_path` is resolved relative to the binary's
/// directory (`current_exe().parent()`), so we *return an absolute
/// path*. We probe a few canonical locations so the same binary works
/// from the workspace root (`cargo run`), the crate directory, or a
/// packaged install. Falls back to Bevy's default `"assets"`.
fn resolve_asset_root() -> String {
    use std::path::PathBuf;
    let abs_candidates: Vec<PathBuf> = [
        // CWD-relative — workspace root invocation
        std::env::current_dir()
            .ok()
            .map(|d| d.join("crates/atomr-worlds-client/assets")),
        std::env::current_dir().ok().map(|d| d.join("assets")),
        // Exec-dir relative — packaged install
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("assets"))),
        // CARGO_MANIFEST_DIR fallback (only set during build, but
        // still useful for the workspace-resident binary at runtime
        // because cargo writes target/release in that workspace).
        option_env!("CARGO_MANIFEST_DIR").map(|m| PathBuf::from(m).join("assets")),
    ]
    .into_iter()
    .flatten()
    .collect();
    for c in &abs_candidates {
        if c.join("shaders/voxel_material.wgsl").exists() {
            return c.to_string_lossy().into_owned();
        }
    }
    "assets".into()
}
