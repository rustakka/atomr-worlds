use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum Backend {
    Local,
    Remote,
    Cluster,
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
}

fn parse_seed(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).map_err(|e| format!("invalid hex seed: {e}"))
    } else {
        s.parse::<u64>().map_err(|e| format!("invalid decimal seed: {e}"))
    }
}
