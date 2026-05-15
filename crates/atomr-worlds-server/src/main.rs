//! `atomr-worlds-server` binary entry point.

use std::collections::HashMap;
use std::net::SocketAddr;

use atomr_worlds_server::{run_cluster, run_standalone, ClusterConfig, StandaloneConfig};
use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode {
    Standalone,
    Cluster,
}

#[derive(Debug, Parser)]
#[command(name = "atomr-worlds-server", version, about = "Headless atomr-worlds server")]
struct Cli {
    #[arg(long, default_value = "0.0.0.0:7800")]
    bind: SocketAddr,

    #[arg(long, default_value = "atomr-worlds-server")]
    system_name: String,

    /// Root world seed (hex with `0x` prefix or decimal).
    #[arg(long, default_value = "0xDEADBEEFCAFEF00D", value_parser = parse_seed)]
    seed: u64,

    #[arg(long, value_enum, default_value_t = Mode::Standalone)]
    mode: Mode,

    /// Cluster-mode: this node's region id.
    #[arg(long, default_value = "alpha", requires = "mode")]
    region_id: String,

    /// Cluster-mode: peer mapping in `region_id=server_path` form,
    /// repeatable. Example:
    ///   `--peer beta=atomr://atomr-worlds-server@host:7801/user/world-gateway`
    #[arg(long, value_parser = parse_peer)]
    peer: Vec<(String, String)>,

    /// Optional pre-shared bearer token. When set, the gateway only
    /// accepts requests whose `WireRequest::auth_token` matches; the
    /// outbound cluster forwarder stamps every cross-node request with
    /// the same value. Pair with `RemoteHostConfig::auth_token` on the
    /// client side. Tokens travel in plaintext until upstream
    /// `atomr-remote` lands the TLS handshake. Phase 15 follow-up.
    #[arg(long)]
    auth_token: Option<String>,
}

fn parse_seed(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).map_err(|e| format!("invalid hex seed: {e}"))
    } else {
        s.parse::<u64>().map_err(|e| format!("invalid decimal seed: {e}"))
    }
}

fn parse_peer(s: &str) -> Result<(String, String), String> {
    let (k, v) = s.split_once('=').ok_or_else(|| {
        format!("peer must be `region_id=server_path`, got `{s}`")
    })?;
    if k.is_empty() || v.is_empty() {
        return Err(format!("peer must have non-empty region_id and path, got `{s}`"));
    }
    Ok((k.to_string(), v.to_string()))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.mode {
        Mode::Standalone => {
            let server = run_standalone(StandaloneConfig {
                bind: cli.bind,
                system_name: cli.system_name,
                root_seed: cli.seed,
                auth_token: cli.auth_token,
            })
            .await?;
            println!("server_path: {}", server.server_path);
            println!("(Ctrl-C to stop)");
            tokio::signal::ctrl_c().await?;
            server.shutdown().await?;
        }
        Mode::Cluster => {
            let peers: HashMap<String, String> = cli.peer.into_iter().collect();
            let server = run_cluster(ClusterConfig {
                bind: cli.bind,
                system_name: cli.system_name,
                root_seed: cli.seed,
                region_id: cli.region_id,
                peers,
                request_timeout: std::time::Duration::from_secs(10),
                auth_token: cli.auth_token,
            })
            .await?;
            println!("server_path: {}", server.server_path);
            println!("(Ctrl-C to stop)");
            tokio::signal::ctrl_c().await?;
            server.shutdown().await?;
        }
    }
    Ok(())
}
