use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum Backend {
    Local,
    Remote,
    Cluster,
}

/// One-line performance preset. `Balanced` (default) keeps every
/// motion-aware strategy active (coarsening the LOD ladder, throttling
/// spawn budget, striding visibility, widening rebuild thresholds);
/// `Quality` swaps them all to static no-ops so visual fidelity stays
/// constant whether the player is moving or not, at the cost of
/// occasional frame-time spikes during sprint.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum PerfPreset {
    Balanced,
    Quality,
}

#[derive(Debug, Parser)]
#[command(
    name = "atomr-worlds-client",
    version,
    about = "Bevy-driven interactive client for atomr-worlds"
)]
pub struct Cli {
    /// Which host backend to drive the renderer.
    #[arg(long, value_enum, default_value_t = Backend::Local)]
    pub backend: Backend,

    /// Server actor path for `--backend remote` (e.g.
    /// `atomr://server@127.0.0.1:7800/user/world-gateway`). Required when
    /// `--backend remote`.
    #[arg(long)]
    pub connect: Option<String>,

    /// Local UDP/TCP bind for the client's own remote system (only used
    /// when `--backend remote|cluster`). `0` lets the OS pick.
    #[arg(long, default_value = "127.0.0.1:0")]
    pub bind: std::net::SocketAddr,

    /// Root world seed (hex with `0x` prefix or decimal).
    #[arg(long, default_value = "0xDEADBEEFCAFEF00D", value_parser = parse_seed)]
    pub seed: u64,

    /// Path to a harness scenario file (TOML). When set, the client runs
    /// the scenario plugin and exits when it completes.
    #[arg(long)]
    pub harness: Option<std::path::PathBuf>,

    /// Output directory for harness PNG screenshots. Required when --harness is set.
    #[arg(long, requires = "harness")]
    pub harness_out: Option<std::path::PathBuf>,

    /// Performance preset. `balanced` (default) lets the motion-aware
    /// strategy layer ramp LOD / spawn budget / visibility cadence /
    /// rebuild thresholds with camera speed; `quality` disables all
    /// motion-aware behaviors so visual fidelity is identical to
    /// stand-still while moving.
    #[arg(long, value_enum, default_value_t = PerfPreset::Balanced)]
    pub perf: PerfPreset,
}

fn parse_seed(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).map_err(|e| format!("invalid hex seed: {e}"))
    } else {
        s.parse::<u64>().map_err(|e| format!("invalid decimal seed: {e}"))
    }
}
