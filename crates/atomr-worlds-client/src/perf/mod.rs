//! Frame-pacing profiler — per-phase timing, queue-depth gauges, a live
//! overlay (F3), and a spike logger.
//!
//! # Why this exists
//!
//! The render/window loop runs every `Update` system serially on the main
//! thread, then render-extract. Any heavy work landing there stalls the
//! window. This module measures *where* per-frame time goes so we can prove
//! the invariant we care about: **no `Update` system does unbounded work or
//! blocks on the host/physics**. It is the measurement half of the
//! "buttery-smooth no matter what" effort; the fixes are the other half.
//!
//! # Design
//!
//! - [`Perf`] is a single shared [`Resource`] of atomics. Instrumented systems
//!   take `Res<Perf>` (shared, *not* `ResMut`) and open a [`PerfScope`] RAII
//!   guard at the top — so timing adds **zero** new system-ordering edges to
//!   the existing `.chain()`. On the single main thread the `Relaxed`
//!   `fetch_add`s are uncontended (~1 ns).
//! - [`Perf::enabled`] is `false` under `--harness`, so every `PerfScope`
//!   early-returns *before* `Instant::now()` — the instrumented systems run
//!   byte-for-byte the same work with no added syscalls. Golden captures are
//!   therefore unperturbed.
//! - [`perf_reset`] (in `First`) zeroes the accumulators each frame;
//!   [`perf_snapshot`] (in `Last`) folds them into [`PerfStats`] for the
//!   overlay + spike logger to read.
//! - Queue-depth gauges (in-flight bricks, loaded chunks, fracture/edit
//!   refresh queues, raster-snapshot rebuild) are pushed by the systems that
//!   already own those resources — no extra borrows in `perf_snapshot`.
//!
//! The Tracy integration (deep per-system flamegraphs) is the separate
//! `profiling` cargo feature; see `Cargo.toml` and `main.rs`.

#![allow(dead_code)] // some gauge setters are wired in by later tiers.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use bevy::prelude::*;

pub mod overlay;

/// A timed slice of the per-frame work. Each variant is one bucket the overlay
/// and spike logger break frame time into; whatever isn't attributed to a phase
/// rolls into `other` (render-extract / present / uninstrumented systems).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(usize)]
pub enum Phase {
    /// `fp_edit_voxels`: pick + write dispatch + eager resident patch.
    EditApply = 0,
    /// `apply_edit_refreshes`: drain finished edit refreshes + swap.
    EditRefresh,
    /// `fp_stream_bricks` minus the spawn loop: plan poll, stale scan, dispatch.
    Streaming,
    /// The drain + `spawn_brick_entity` GPU-upload loop inside `fp_stream_bricks`.
    BrickSpawn,
    /// `fp_update_lod_visibility`.
    LodVisibility,
    /// `dispatch_fracture_checks`: snapshot + worker dispatch.
    FractureDispatch,
    /// `apply_fracture_results`: drain analyses, spawn islands, dispatch writeback.
    FractureApply,
    /// `slice_render` / `rts_render`: rasterize + blit (host queries are off-thread).
    SliceRtsRaster,
    /// The overlay's own update — self-measured to prove it is cheap.
    HudOverlay,
    /// `pump_debris_states` + `apply_debris_interpolation`: drain host debris
    /// snapshots and interpolate the kinematic bodies.
    DebrisStream,
}

impl Phase {
    pub const N: usize = 10;
    pub const ALL: [Phase; Self::N] = [
        Phase::EditApply,
        Phase::EditRefresh,
        Phase::Streaming,
        Phase::BrickSpawn,
        Phase::LodVisibility,
        Phase::FractureDispatch,
        Phase::FractureApply,
        Phase::SliceRtsRaster,
        Phase::HudOverlay,
        Phase::DebrisStream,
    ];

    /// Short token used in the overlay and the `PERF_SPIKE` log line.
    pub fn label(self) -> &'static str {
        match self {
            Phase::EditApply => "edit",
            Phase::EditRefresh => "eref",
            Phase::Streaming => "stream",
            Phase::BrickSpawn => "spawn",
            Phase::LodVisibility => "lodvis",
            Phase::FractureDispatch => "fracD",
            Phase::FractureApply => "fracA",
            Phase::SliceRtsRaster => "raster",
            Phase::HudOverlay => "hud",
            Phase::DebrisStream => "debrisS",
        }
    }
}

/// Per-frame accumulators + live gauges, all atomic so a [`PerfScope`] needs
/// only `Res<Perf>` (shared) — no `ResMut`, so no new scheduler edges.
#[derive(Resource)]
pub struct Perf {
    /// Nanoseconds accumulated this frame, per [`Phase`]. Reset in `First`.
    ns: [AtomicU64; Phase::N],
    /// `false` under `--harness` → all scopes / reset / snapshot early-return.
    enabled: AtomicBool,
    // --- live queue gauges (overwritten each frame by the owning systems) ---
    brick_in_flight: AtomicUsize,
    loaded_chunks: AtomicUsize,
    fracture_in_flight: AtomicUsize,
    fracture_refresh_in_flight: AtomicUsize,
    edit_refresh_in_flight: AtomicUsize,
    snapshot_rebuilding: AtomicBool,
}

impl Perf {
    pub fn new(enabled: bool) -> Self {
        Self {
            ns: std::array::from_fn(|_| AtomicU64::new(0)),
            enabled: AtomicBool::new(enabled),
            brick_in_flight: AtomicUsize::new(0),
            loaded_chunks: AtomicUsize::new(0),
            fracture_in_flight: AtomicUsize::new(0),
            fracture_refresh_in_flight: AtomicUsize::new(0),
            edit_refresh_in_flight: AtomicUsize::new(0),
            snapshot_rebuilding: AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled.load(Relaxed)
    }

    pub fn set_enabled(&self, v: bool) {
        self.enabled.store(v, Relaxed);
    }

    #[inline]
    fn add_ns(&self, p: Phase, ns: u64) {
        if self.enabled.load(Relaxed) {
            self.ns[p as usize].fetch_add(ns, Relaxed);
        }
    }

    /// Open an RAII timer for `p`; the elapsed time is added on drop. Cheap
    /// no-op when disabled (no `Instant::now()`).
    #[inline]
    pub fn scope(&self, p: Phase) -> PerfScope<'_> {
        PerfScope::new(self, p)
    }

    // --- gauge setters (called by the systems that own the source data) ---
    pub fn set_brick_in_flight(&self, n: usize) {
        self.brick_in_flight.store(n, Relaxed);
    }
    pub fn set_loaded_chunks(&self, n: usize) {
        self.loaded_chunks.store(n, Relaxed);
    }
    pub fn set_fracture_in_flight(&self, n: usize) {
        self.fracture_in_flight.store(n, Relaxed);
    }
    pub fn set_fracture_refresh_in_flight(&self, n: usize) {
        self.fracture_refresh_in_flight.store(n, Relaxed);
    }
    pub fn set_edit_refresh_in_flight(&self, n: usize) {
        self.edit_refresh_in_flight.store(n, Relaxed);
    }
    pub fn set_snapshot_rebuilding(&self, v: bool) {
        self.snapshot_rebuilding.store(v, Relaxed);
    }
}

impl Default for Perf {
    fn default() -> Self {
        Self::new(true)
    }
}

/// RAII timer for one [`Phase`]. Records `start.elapsed()` into [`Perf`] on
/// drop. When [`Perf`] is disabled it captures no `Instant`, so dropping it is
/// free — this is what keeps the harness path byte-identical.
pub struct PerfScope<'a> {
    perf: &'a Perf,
    phase: Phase,
    start: Option<Instant>,
}

impl<'a> PerfScope<'a> {
    #[inline]
    pub fn new(perf: &'a Perf, phase: Phase) -> Self {
        let start = if perf.enabled() { Some(Instant::now()) } else { None };
        Self { perf, phase, start }
    }
}

impl Drop for PerfScope<'_> {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            self.perf.add_ns(self.phase, start.elapsed().as_nanos() as u64);
        }
    }
}

/// Stable per-frame snapshot the overlay + spike logger read. Filled in `Last`
/// by [`perf_snapshot`]; the overlay shows the previous frame's values (one
/// frame of display lag, intentional).
#[derive(Resource, Default)]
pub struct PerfStats {
    /// This-frame microseconds per phase.
    pub last_us: [u64; Phase::N],
    /// EMA (α = 0.1) per phase, for a stable overlay readout.
    pub ema_us: [f64; Phase::N],
    /// Whole-frame microseconds (= `Time::delta`).
    pub frame_us: u64,
    /// `frame_us − Σ phases` — render-extract / present / uninstrumented work.
    pub other_us: u64,
    pub brick_in_flight: usize,
    pub loaded_chunks: usize,
    pub fracture_in_flight: usize,
    pub fracture_refresh_in_flight: usize,
    pub edit_refresh_in_flight: usize,
    pub snapshot_rebuilding: bool,
}

/// `First`: zero the accumulators before any instrumented `Update` system runs.
fn perf_reset(perf: Res<Perf>) {
    if !perf.enabled() {
        return;
    }
    for a in &perf.ns {
        a.store(0, Relaxed);
    }
}

/// `Last`: fold the frame's accumulators + gauges into [`PerfStats`].
fn perf_snapshot(perf: Res<Perf>, time: Res<Time>, mut stats: ResMut<PerfStats>) {
    if !perf.enabled() {
        return;
    }
    let mut sum_us = 0u64;
    for i in 0..Phase::N {
        let us = perf.ns[i].load(Relaxed) / 1000;
        stats.last_us[i] = us;
        stats.ema_us[i] = stats.ema_us[i] * 0.9 + us as f64 * 0.1;
        sum_us += us;
    }
    let frame_us = (time.delta_secs_f64() * 1.0e6).round().max(0.0) as u64;
    stats.frame_us = frame_us;
    stats.other_us = frame_us.saturating_sub(sum_us);
    stats.brick_in_flight = perf.brick_in_flight.load(Relaxed);
    stats.loaded_chunks = perf.loaded_chunks.load(Relaxed);
    stats.fracture_in_flight = perf.fracture_in_flight.load(Relaxed);
    stats.fracture_refresh_in_flight = perf.fracture_refresh_in_flight.load(Relaxed);
    stats.edit_refresh_in_flight = perf.edit_refresh_in_flight.load(Relaxed);
    stats.snapshot_rebuilding = perf.snapshot_rebuilding.load(Relaxed);
}

/// `Last` (after [`perf_snapshot`]): if the frame exceeded the budget, print one
/// `PERF_SPIKE` line to **stderr** (stdout is reserved for `HARNESS_SHOT`),
/// attributing the dominant contributor. Not added under the harness.
///
/// Threshold comes from `ATOMR_PERF_SPIKE_MS` (default 20.0), parsed once.
fn perf_spike_log(
    stats: Res<PerfStats>,
    streamer: Option<Res<crate::world_stream::ChunkStreamer>>,
    mut threshold_ms: Local<Option<f64>>,
) {
    let thr = *threshold_ms.get_or_insert_with(|| {
        std::env::var("ATOMR_PERF_SPIKE_MS")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(20.0)
    });
    let total_ms = stats.frame_us as f64 / 1000.0;
    if total_ms < thr {
        return;
    }
    // Dominant contributor: the largest phase, but only if it beats `other`
    // (so we never blame a phase when the cost is actually outside our scopes).
    let mut dom = "other";
    let mut dom_us = stats.other_us;
    for p in Phase::ALL {
        let us = stats.last_us[p as usize];
        if us > dom_us {
            dom_us = us;
            dom = p.label();
        }
    }
    let frame = streamer.map(|s| s.frame).unwrap_or(0);
    let per_phase = Phase::ALL
        .iter()
        .map(|p| format!("{}_ms={:.2}", p.label(), stats.last_us[*p as usize] as f64 / 1000.0))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "PERF_SPIKE frame={frame} total_ms={total_ms:.2} dominant={dom} dominant_ms={:.2} \
         other_ms={:.2} {per_phase} brick_q={} load_q={} frac_q={} fref_q={} eref_q={}",
        dom_us as f64 / 1000.0,
        stats.other_us as f64 / 1000.0,
        stats.brick_in_flight,
        stats.loaded_chunks,
        stats.fracture_in_flight,
        stats.fracture_refresh_in_flight,
        stats.edit_refresh_in_flight,
    );
}

/// Registers the profiler. `interactive` is `harness_bits.is_none()` — under
/// the harness only the (no-op) reset/snapshot run, so nothing is added that
/// could perturb a golden capture.
pub struct PerfPlugin {
    pub interactive: bool,
}

impl Plugin for PerfPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Perf::new(self.interactive))
            .init_resource::<PerfStats>()
            .add_systems(First, perf_reset)
            .add_systems(Last, perf_snapshot);
        if self.interactive {
            app.init_resource::<overlay::PerfOverlayState>()
                .add_systems(Startup, overlay::setup_perf_overlay)
                .add_systems(Update, overlay::perf_overlay_toggle)
                .add_systems(Update, overlay::update_perf_overlay.run_if(overlay::overlay_visible))
                .add_systems(Last, perf_spike_log.after(perf_snapshot));
        }
    }
}
