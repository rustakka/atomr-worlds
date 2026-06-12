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

/// Shading path. `default` keeps the configured default; `mesh` forces the
/// legacy split-per-material mesh path (the fallback once raymarch becomes the
/// default); `palette` uses the merged palette voxel material; `raymarch` skips
/// meshing and raymarches each brick's sparse-voxel DAG on the GPU.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ShadingArg {
    Default,
    Mesh,
    Palette,
    Raymarch,
}

/// Raymarch shading tier — style / performance knob, only meaningful with
/// `--shading raymarch`.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum RaymarchTier {
    Unlit,
    Lambert,
    Pbr,
}

/// World-generation archetype. `default` keeps the host's seeded selector
/// (Earth-like terrain); `ice` forces the frozen "cryo" archetype — a SNOW/ICE
/// shell over a buried WATER ocean and a STONE core — across the whole world.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum WorldGenArg {
    Default,
    Ice,
}

/// Client-side physics master switch (Rec 2). `on` (default) enables rapier
/// terrain colliders + flood-fill fracture debris; `off` keeps the world inert.
#[cfg(feature = "physics")]
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum PhysicsToggle {
    On,
    Off,
}

/// Voxel-collider strategy. `greedy` (default) merges solid voxels into a small
/// set of boxes; `per-voxel` is the un-merged form (heavier, useful for A/B).
#[cfg(feature = "physics")]
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ColliderArg {
    Greedy,
    PerVoxel,
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

    /// Shading path. `raymarch` renders bricks by GPU-raymarching their
    /// sparse-voxel DAG instead of meshing them (Rec 1).
    #[arg(long, value_enum, default_value_t = ShadingArg::Default)]
    pub shading: ShadingArg,

    /// Raymarch shading tier (only used with `--shading raymarch`).
    #[arg(long, value_enum, default_value_t = RaymarchTier::Lambert)]
    pub raymarch_tier: RaymarchTier,

    /// World-generation archetype (local backend only). `default` is seeded
    /// terrain; `ice` forces the frozen cryo archetype for the whole world.
    #[arg(long, value_enum, default_value_t = WorldGenArg::Default)]
    pub world_gen: WorldGenArg,

    /// Client-side physics (Rec 2): rapier terrain colliders + flood-fill
    /// fracture debris. On by default; forced off in harness mode.
    #[cfg(feature = "physics")]
    #[arg(long, value_enum, default_value_t = PhysicsToggle::On)]
    pub physics: PhysicsToggle,

    /// Voxel-collider strategy (only used when physics is on).
    #[cfg(feature = "physics")]
    #[arg(long, value_enum, default_value_t = ColliderArg::Greedy)]
    pub collider: ColliderArg,
}

fn parse_seed(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).map_err(|e| format!("invalid hex seed: {e}"))
    } else {
        s.parse::<u64>().map_err(|e| format!("invalid decimal seed: {e}"))
    }
}
