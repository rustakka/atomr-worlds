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
mod host_backend;
mod hud;
mod modes;
mod view_mode;
mod world_runtime;

use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use bevy::prelude::*;
use clap::Parser;

use crate::cli::Cli;
use crate::view_mode::{view_mode_input_system, ViewMode};
use crate::world_runtime::{ActiveWorld, WorldRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
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

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: format!("atomr-worlds-client [{:?}]", cli.backend),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(world_runtime)
        .insert_resource(active)
        .insert_resource(ViewMode::Fp)
        .insert_resource(ClearColor(Color::rgb(0.45, 0.65, 0.85)))
        .add_plugins(modes::fp::FpPlugin)
        .add_plugins(modes::tp::TpPlugin)
        .add_plugins(modes::blit::BlitPlugin)
        .add_plugins(modes::slice::SlicePlugin)
        .add_plugins(modes::rts::RtsPlugin)
        .add_plugins(modes::overview::OverviewPlugin)
        .add_plugins(hud::HudPlugin)
        .add_systems(Update, view_mode_input_system)
        .run();

    Ok(())
}
