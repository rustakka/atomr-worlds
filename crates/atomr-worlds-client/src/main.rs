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

mod cli;
mod harness;
mod host_backend;
mod hud;
mod modes;
mod view_mode;
mod world_runtime;

use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use bevy::prelude::*;
use bevy::window::{PresentMode, WindowResolution};
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
    let active = ActiveWorld { addr: WorldAddr::ROOT, seed: cli.seed };

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
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: format!("atomr-worlds-client [{:?}]", cli.backend),
            resolution: window_resolution,
            // Force FIFO — some drivers report an exotic present mode
            // (e.g. FIFO_LATEST_READY_EXT = 1000361000) that wgpu 0.19
            // doesn't recognise.
            present_mode: PresentMode::Fifo,
            ..default()
        }),
        ..default()
    }))
        .insert_resource(world_runtime)
        .insert_resource(active)
        .insert_resource(initial_mode)
        .insert_resource(ClearColor(Color::rgb(0.45, 0.65, 0.85)))
        .add_plugins(modes::fp::FpPlugin)
        .add_plugins(modes::tp::TpPlugin)
        .add_plugins(modes::blit::BlitPlugin)
        .add_plugins(modes::slice::SlicePlugin)
        .add_plugins(modes::rts::RtsPlugin)
        .add_plugins(modes::overview::OverviewPlugin)
        .add_plugins(hud::HudPlugin)
        .add_systems(Update, view_mode_input_system);

    if let Some((scenario, out_abs)) = harness_bits {
        app.insert_resource(harness::HarnessConfig {
            scenario,
            output_dir: out_abs,
        });
        app.add_plugins(harness::HarnessPlugin);
    }

    app.run();

    Ok(())
}
