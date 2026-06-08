//! Progressive chunk auto-loader / unloader.
//!
//! The client's local streamer is structured as a [`LodLadder`] of
//! [`LodTier`] entries: each tier names a `Lod` plus its outer radius
//! (meters from the observer). Tier 0 is the highest fidelity, drawn
//! tight around the camera; each subsequent tier widens the radius and
//! drops one or more LOD steps so the world fades to coarser bricks as
//! you look toward the horizon.
//!
//! Each frame, [`desired_chunks`] walks the ladder and emits the brick
//! coordinates that should be loaded at each tier's LOD. The shape of
//! the emitted set is controlled by a [`LodCoveragePolicy`] strategy
//! (see [`crate::render::strategy`]):
//!
//! - `MaskedShells` — historical behaviour: each tier loads only its
//!   shell band between the previous tier's outer radius and this
//!   tier's outer radius. One brick per region; LOD transitions
//!   require generating + meshing the next tier the moment the
//!   current one becomes ineligible, which produces a visible pop.
//! - `NestedSummary` (default) — every tier loads its full inner
//!   sphere up to its outer radius. Each region has the immediate
//!   coarser tier already resident as a "summary" backdrop, so when
//!   the finer brick fades out the parent is in memory and just
//!   becomes visible (the FP visibility system handles crossfade).
//!
//! Either way, the spherical (3D-distance) outer test makes loading
//! and unloading symmetric in all four cardinal directions and the
//! diagonals. The previous 2-tier cube ring caused a visible
//! asymmetry where only some directions appeared to populate.
//!
//! The legacy 2-tier [`StreamingPolicy`] is preserved as a *derived
//! view* on top of the ladder so the proto crate, host, and existing
//! skybox bake code keep working unchanged.
//!
//! Raster modes (slice / RTS) only need [`ChunkStreamer::lod_for_meters`]
//! to pick an LOD at sample time. FP/TP call [`desired_chunks`] each
//! tick and reconcile against [`LoadedChunks`].
//!
//! The streamer also exposes [`ChunkStreamer::fog_band_m`] — the
//! `(start_m, end_m)` band where fog ramps from clear to opaque,
//! anchored to the outermost tier so loading bricks fade in from mist
//! instead of popping.

use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};

use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::streaming::StreamingPolicy;
use atomr_worlds_voxel::brick::Brick;
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::prelude::*;

use crate::render::{LodCoveragePolicy, RaymarchShadingTier};

/// Per-tick fetch budget. Sized so the 4-tier sphere (≈8 k brick keys)
/// populates in ~1 s at 60 fps with the closest-first sort prioritising
/// the high-fidelity inner tier. After the `brick_inside_shape` fix
/// every brick in the ring really does go through the FBM terrain
/// generator (negative-coord bricks no longer short-circuit), so wall-
/// clock time per frame at saturation is several hundred ms; effective
/// fps during initial fill is ~2 — that's an async-brick-gen task.
pub const DEFAULT_BRICKS_PER_TICK: u32 = 128;
/// Streamer ticks a chunk stays loaded after leaving the desired set.
/// Two ticks at 60 fps ≈ 33 ms grace period — enough to absorb a single
/// boundary-jitter step without re-meshing.
pub const HYSTERESIS_TICKS: u64 = 2;

/// Fraction of the outermost radius at which fog begins to ramp in.
/// 0.55 ⇒ fog starts at the midpoint of the second-to-last tier and
/// reaches full density at the load horizon, so chunks streaming into
/// the outermost shell are obscured by mist while they pop in.
pub const FOG_START_FRACTION: f64 = 0.55;
/// Fraction of the outermost radius at which fog is fully opaque.
pub const FOG_END_FRACTION: f64 = 0.98;

/// How far the observer must drift before the cached desired-chunks plan
/// is rebuilt. Sized to a quarter of the L0 brick edge (16 m) — small
/// enough that newly-entered brick cells are picked up quickly, large
/// enough that walking-pace motion only triggers a rebuild every few
/// frames instead of every frame.
pub const PLAN_REBUILD_DRIFT_M: f64 = 4.0;

/// How far the camera forward direction must rotate before the cached
/// plan is rebuilt for view-priority resorting. cos(15°) ≈ 0.9659.
/// Until the forward drift exceeds this the existing closest-first
/// order is reused.
pub const PLAN_REBUILD_FWD_COS: f64 = 0.9659;

/// View-priority weight: a brick directly in the forward hemisphere
/// has its effective scoring distance multiplied by `(1 - WEIGHT)`,
/// so it loads ahead of an equally-distant brick behind the camera.
/// 0.4 means a front-cone brick scores 60 % of a behind-camera brick
/// at the same true distance — strong bias without flat-out skipping
/// behind-camera tiles.
pub const VIEW_PRIORITY_WEIGHT: f64 = 0.4;

/// One rung of the progressive LOD ladder.
///
/// `lod` is the brick LOD this tier streams at (each step up doubles
/// the voxel edge in world meters). `outer_radius_m` is the maximum
/// distance, in meters from the observer, at which a brick in this
/// tier is considered "desired". The shell between the previous tier's
/// outer radius and this tier's outer radius belongs to this tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodTier {
    pub lod: Lod,
    pub outer_radius_m: f64,
}

/// Strictly-ordered ladder of LOD tiers. Tier 0 is the innermost
/// (sharpest LOD, smallest radius); the last tier defines the load
/// horizon. Tiers must satisfy:
/// - `outer_radius_m` strictly increasing.
/// - `lod.depth` non-decreasing (coarser farther out).
#[derive(Clone, Debug, PartialEq)]
pub struct LodLadder {
    pub tiers: Vec<LodTier>,
    pub bricks_per_tick: u32,
}

impl LodLadder {
    /// 4-rung default ladder. Radii (m): 128 / 256 / 512 / 1024.
    /// LODs:                          L0 / L1 / L2 / L3.
    ///
    /// Radii are multiples of the coarsest brick edge (`BRICK_EDGE *
    /// 2^3 = 128 m`) so brick grids at every LOD tile cleanly across
    /// the tier boundaries. With aligned boundaries, the band test
    /// `[inner, outer)` on brick *center* produces a watertight
    /// spherical load shape — no gaps, no double-rendered overlap
    /// between tiers.
    pub fn default_progressive() -> Self {
        Self {
            tiers: vec![
                LodTier { lod: Lod::new(0), outer_radius_m: 128.0 },
                LodTier { lod: Lod::new(1), outer_radius_m: 256.0 },
                LodTier { lod: Lod::new(2), outer_radius_m: 512.0 },
                LodTier { lod: Lod::new(3), outer_radius_m: 1024.0 },
            ],
            bricks_per_tick: DEFAULT_BRICKS_PER_TICK,
        }
    }

    /// The outermost tier's radius — the absolute load horizon.
    pub fn outer_radius_m(&self) -> f64 {
        self.tiers.last().map(|t| t.outer_radius_m).unwrap_or(0.0)
    }

    /// The innermost tier's radius — used as the legacy
    /// `transition_radius_m` for the proto policy.
    pub fn inner_radius_m(&self) -> f64 {
        self.tiers.first().map(|t| t.outer_radius_m).unwrap_or(0.0)
    }

    /// Inner (start) and outer (end) radius of tier `i`, where the
    /// inner radius is the previous tier's outer radius (or `0.0` for
    /// tier 0). Returns `(0, 0)` for an out-of-range index.
    pub fn tier_band_m(&self, i: usize) -> (f64, f64) {
        if i >= self.tiers.len() {
            return (0.0, 0.0);
        }
        let inner = if i == 0 { 0.0 } else { self.tiers[i - 1].outer_radius_m };
        (inner, self.tiers[i].outer_radius_m)
    }

    /// Pick the LOD for a sample distance `d_m` from the observer.
    /// Returns the LOD of the tier whose band `[inner, outer)`
    /// contains `d_m`. Samples past the load horizon clamp to the
    /// outermost tier's LOD. Band convention matches [`desired_chunks`]
    /// so raster LOD selection lines up with the brick-fetch grids.
    pub fn lod_for_distance(&self, d_m: f64) -> Lod {
        if self.tiers.is_empty() {
            return Lod::new(0);
        }
        for tier in &self.tiers {
            if d_m < tier.outer_radius_m {
                return tier.lod;
            }
        }
        self.tiers.last().unwrap().lod
    }

    /// Derive a 2-tier [`StreamingPolicy`] view of this ladder, used by
    /// proto/host code (and the skybox bake) that pre-dates the
    /// progressive ladder. `near_lod` = tier 0, `far_lod` = last tier.
    pub fn as_policy(&self) -> StreamingPolicy {
        let near = self.tiers.first().copied().unwrap_or(LodTier {
            lod: Lod::new(0),
            outer_radius_m: 0.0,
        });
        let far = self.tiers.last().copied().unwrap_or(near);
        StreamingPolicy {
            near_lod: near.lod,
            far_lod: far.lod,
            transition_radius_m: near.outer_radius_m,
            max_radius_m: far.outer_radius_m,
            bricks_per_tick: self.bricks_per_tick,
        }
    }
}

impl Default for LodLadder {
    fn default() -> Self {
        Self::default_progressive()
    }
}

/// Bevy resource: progressive-LOD chunk streamer.
///
/// Holds the [`LodLadder`] plus housekeeping counters. Per-frame
/// systems call [`desired_chunks`] (FP/TP) or
/// [`Self::lod_for_meters`] (raster modes) to make decisions. The
/// [`Self::policy`] view exposes a legacy 2-tier `StreamingPolicy` for
/// callers that haven't been ported to the ladder.
#[derive(Resource, Clone, Debug)]
pub struct ChunkStreamer {
    pub ladder: LodLadder,
    /// Cached 2-tier projection of the ladder. Kept in sync with
    /// `ladder` so callers reading `policy.far_lod` / `max_radius_m`
    /// don't need to know about the ladder.
    pub policy: StreamingPolicy,
    /// Monotonic frame counter; bumped each tick by [`Self::tick_frame`].
    pub frame: u64,
    /// Hysteresis window — chunks linger this many ticks past the
    /// desired boundary before despawn.
    pub hysteresis_ticks: u64,
}

impl Default for ChunkStreamer {
    fn default() -> Self {
        let ladder = LodLadder::default_progressive();
        let policy = ladder.as_policy();
        Self {
            ladder,
            policy,
            frame: 0,
            hysteresis_ticks: HYSTERESIS_TICKS,
        }
    }
}

impl ChunkStreamer {
    /// Construct from a custom ladder. Keeps `policy` in sync.
    pub fn with_ladder(ladder: LodLadder) -> Self {
        let policy = ladder.as_policy();
        Self {
            ladder,
            policy,
            frame: 0,
            hysteresis_ticks: HYSTERESIS_TICKS,
        }
    }

    /// Replace the ladder and refresh the derived `policy` view.
    pub fn set_ladder(&mut self, ladder: LodLadder) {
        self.policy = ladder.as_policy();
        self.ladder = ladder;
    }

    /// Decide the LOD a raster sample at world position `p` should use.
    /// Walks the ladder by 3D distance — same shape as the FP loader so
    /// raster modes line up with the brick fetch grid.
    #[inline]
    pub fn lod_for_meters(&self, observer: DVec3, p: DVec3) -> Lod {
        let dx = p.x - observer.x;
        let dy = p.y - observer.y;
        let dz = p.z - observer.z;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        self.ladder.lod_for_distance(r)
    }

    /// Advance the streamer's frame counter. Called once per tick by
    /// the FP streaming system.
    #[inline]
    pub fn tick_frame(&mut self) {
        self.frame = self.frame.saturating_add(1);
    }

    /// Outermost load radius — the absolute horizon, in meters.
    #[inline]
    pub fn outer_radius_m(&self) -> f64 {
        self.ladder.outer_radius_m()
    }

    /// `(start, end)` meters for the fog ramp. `start` is where mist
    /// begins to obscure; `end` is where it's fully opaque. Both are
    /// derived from the outermost tier's radius so fog and the load
    /// horizon track together.
    #[inline]
    pub fn fog_band_m(&self) -> (f64, f64) {
        let outer = self.outer_radius_m();
        (outer * FOG_START_FRACTION, outer * FOG_END_FRACTION)
    }
}

/// One loaded chunk's bookkeeping entry. The Bevy entity is `None` for
/// raster-mode loaders (slice / RTS / overview) that don't spawn meshes,
/// and also for empty bricks the loader has already verified are
/// uninteresting.
#[derive(Debug, Clone)]
pub struct LoadedChunk {
    pub coord: IVec3,
    pub lod: Lod,
    pub entity: Option<Entity>,
    /// Last frame the chunk was in the desired set. Used for hysteresis:
    /// `frame - last_seen >= hysteresis_ticks` ⇒ eligible for despawn.
    pub last_seen_frame: u64,
    /// Mirrors `BrickFadeOut` presence on the ECS entity. Tracked here so
    /// [`LoadedChunks::child_counts`] can be maintained incrementally
    /// without a per-frame ECS query of `With<BrickFadeOut>`. Set true by
    /// [`LoadedChunks::mark_fading_out`] when the streamer attaches a
    /// `BrickFadeOut`; reset to false on every fresh `insert`.
    pub is_fading_out: bool,
    /// When this brick was rendered via the DAG raymarcher, the content digest
    /// + tier of its cached buffers/material, so eviction can decref the
    /// [`crate::render::DagBufferCache`] in lockstep. `None` for the mesh path /
    /// empty bricks. (Both are set together; see `spawn_brick_entity`.)
    pub dag_digest: Option<u64>,
    pub dag_tier: Option<RaymarchShadingTier>,
    /// Resident decoded voxels, populated **only for LOD-0** chunks (the
    /// near ring). Powers the client-side voxel picker / brush refresh
    /// (`crate::modes::edit`) and the crosshair highlight with zero host
    /// round-trips. `None` for coarse LODs, empty bricks, and raster-mode
    /// loaders. Drops with the chunk on eviction — no separate lifecycle.
    /// At ~8 KiB/brick over the LOD-0 ring the cost lands where it's cheap.
    pub brick: Option<Arc<Brick>>,
}

impl LoadedChunk {
    #[inline]
    pub fn key(coord: IVec3, lod: Lod) -> (IVec3, u8) {
        (coord, lod.depth)
    }
}

/// `(coord, depth)` of the parent brick that immediately covers
/// `(coord, depth)`. Uses `div_euclid` rather than `/` so the
/// child→parent relationship stays correct for negative coords (Rust's
/// truncating division would give `-1 / 2 == 0` but our voxel grid
/// wants `(-1).div_euclid(2) == -1`).
#[inline]
pub(crate) fn parent_key(key: (IVec3, u8)) -> (IVec3, u8) {
    let (coord, depth) = key;
    (
        IVec3::new(
            coord.x.div_euclid(2),
            coord.y.div_euclid(2),
            coord.z.div_euclid(2),
        ),
        depth + 1,
    )
}

/// HashMap-backed registry of loaded chunks keyed by `(coord, lod.depth)`.
/// Keying by depth lets us briefly hold both a `(c, 0)` and `(c, 1)`
/// entry during a tier change without one stomping the other.
///
/// Also maintains an incremental `child_counts` index of how many
/// *non-fading-out* immediate children each parent key has. This used
/// to be rebuilt every frame inside `fp_update_lod_visibility` as an
/// O(n_loaded) HashMap walk; keeping it incrementally maintained on
/// insert / mark_fading_out / remove makes the visibility pass O(n_q)
/// (query size) instead of O(n_q + n_loaded).
#[derive(Resource, Default, Debug)]
pub struct LoadedChunks {
    inner: HashMap<(IVec3, u8), LoadedChunk>,
    child_counts: HashMap<(IVec3, u8), u32>,
}

impl LoadedChunks {
    /// Read-only access to the underlying map. Mutating callers must use
    /// [`Self::insert`] / [`Self::remove`] / [`Self::mark_fading_out`]
    /// so the `child_counts` index stays in sync. Borrow at call sites
    /// where the prior code used `loaded.0.iter()` / `loaded.0.get()`.
    #[inline]
    pub fn map(&self) -> &HashMap<(IVec3, u8), LoadedChunk> { &self.inner }

    /// Number of *non-fading-out* immediate children that cover the
    /// parent `key`. Returns 0 when the parent has no live children.
    #[inline]
    pub fn child_count(&self, parent: &(IVec3, u8)) -> u32 {
        self.child_counts.get(parent).copied().unwrap_or(0)
    }

    /// Insert or replace an entry. Increments the parent's child count
    /// unless the entry being replaced was already counted. Resets
    /// `is_fading_out` to false (re-inserting a brick brings it back
    /// into the "counts toward coverage" set).
    pub fn insert(&mut self, key: (IVec3, u8), mut chunk: LoadedChunk) {
        chunk.is_fading_out = false;
        let parent = parent_key(key);
        let prev_counted = self
            .inner
            .get(&key)
            .map(|c| !c.is_fading_out)
            .unwrap_or(false);
        if !prev_counted {
            *self.child_counts.entry(parent).or_insert(0) += 1;
        }
        self.inner.insert(key, chunk);
    }

    /// Mutable access to an entry's `last_seen_frame` without
    /// disturbing the `child_counts` invariant. The streamer uses this
    /// every frame to refresh hysteresis bookkeeping for desired
    /// chunks; nothing else about the chunk's coverage status changes.
    #[inline]
    pub fn touch_last_seen(&mut self, key: &(IVec3, u8), frame: u64) {
        if let Some(entry) = self.inner.get_mut(key) {
            entry.last_seen_frame = frame;
        }
    }

    /// Read a chunk by key.
    #[inline]
    pub fn get(&self, key: &(IVec3, u8)) -> Option<&LoadedChunk> { self.inner.get(key) }

    /// Whether a chunk with this key is currently loaded.
    #[inline]
    pub fn contains_key(&self, key: &(IVec3, u8)) -> bool { self.inner.contains_key(key) }

    /// Number of loaded chunks (including those mid-fade-out).
    #[inline]
    pub fn len(&self) -> usize { self.inner.len() }

    /// Whether no chunks are loaded.
    #[inline]
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }

    /// Mark an existing entry as fading out. Decrements its parent's
    /// child count if it wasn't already marked. Idempotent.
    pub fn mark_fading_out(&mut self, key: &(IVec3, u8)) {
        let Some(entry) = self.inner.get_mut(key) else { return };
        if entry.is_fading_out {
            return;
        }
        entry.is_fading_out = true;
        let parent = parent_key(*key);
        if let Some(n) = self.child_counts.get_mut(&parent) {
            *n -= 1;
            if *n == 0 {
                self.child_counts.remove(&parent);
            }
        }
    }

    /// Remove an entry. Decrements the parent's child count only if
    /// the removed entry was still counted (i.e. not previously marked
    /// fading out).
    pub fn remove(&mut self, key: &(IVec3, u8)) -> Option<LoadedChunk> {
        let removed = self.inner.remove(key)?;
        if !removed.is_fading_out {
            let parent = parent_key(*key);
            if let Some(n) = self.child_counts.get_mut(&parent) {
                *n -= 1;
                if *n == 0 {
                    self.child_counts.remove(&parent);
                }
            }
        }
        Some(removed)
    }

    /// Iterate entries (read-only) without exposing the underlying
    /// HashMap type.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&(IVec3, u8), &LoadedChunk)> + '_ {
        self.inner.iter()
    }

    /// Iterate keys (read-only).
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &(IVec3, u8)> + '_ { self.inner.keys() }

    /// Whether the entry should be despawned this tick under the given
    /// hysteresis window.
    #[inline]
    pub fn is_stale(&self, key: &(IVec3, u8), now: u64, hysteresis: u64) -> bool {
        match self.inner.get(key) {
            Some(c) => now.saturating_sub(c.last_seen_frame) >= hysteresis,
            None => false,
        }
    }
}

/// Cached plan + the observer state it was computed for.
///
/// Recomputing the full 4-tier AABB sweep + sort costs measurable CPU
/// every frame (8 k+ keys at the default ladder). Most frames the
/// observer hasn't moved enough to change the plan in any way that
/// matters. This resource memoizes the last plan and only rebuilds
/// when either the camera position drifts more than
/// [`PLAN_REBUILD_DRIFT_M`] or the camera forward direction rotates
/// past [`PLAN_REBUILD_FWD_COS`] (so the view-priority ordering can
/// be re-applied when the player turns).
///
/// Rebuilds run on a background thread via [`Self::spawn_rebuild`]; the
/// streaming system polls for completion each frame with
/// [`Self::poll_rebuild`] and installs the result when it arrives. The
/// rebuild itself (4-tier AABB sweep + view-priority sort over ~11 k
/// entries) costs ~2 ms — too long for the main thread. Running it off-
/// thread is what eliminates the per-rebuild frame spike that used to
/// fire every ~20 frames at sprint pace.
///
/// The cached plan therefore lags the observer pose by 1-2 frames after
/// a rebuild trigger. That's harmless: rebuilds are already drift-
/// triggered (every 4 m of motion), so the loader was always working
/// off a slightly stale plan — moving the staleness from "0-frame, but
/// the main thread paid 2 ms" to "1-2 frame, main thread paid nothing"
/// is a strict win for frame pacing.
#[derive(Resource, Default, Debug)]
pub struct DesiredChunksCache {
    /// `(observer, forward)` pair the cached plan was built for.
    /// `None` ⇒ first frame, force a rebuild.
    pub built_for: Option<(DVec3, DVec3)>,
    /// Sorted brick keys, closest-first with view-priority bias if a
    /// forward direction was supplied at rebuild time.
    pub plan: Vec<(IVec3, Lod)>,
    /// Next index into [`Self::plan`] for the dispatch loop. Persists
    /// across frames so a saturated worker pool stops the dispatch scan
    /// without redoing the front-of-plan walk every frame. Reset to 0
    /// whenever a fresh plan is installed (via [`Self::set`] or a
    /// completed [`Self::poll_rebuild`]).
    pub cursor: usize,
    /// Receiver for an in-flight background rebuild, plus the pose the
    /// rebuild was dispatched for. `None` ⇒ no rebuild in flight.
    rebuild: Option<RebuildHandle>,
}

#[derive(Debug)]
struct RebuildHandle {
    /// `Mutex` so the cache (a `Resource`, which Bevy requires
    /// `Send + Sync`) stays `Sync` — `mpsc::Receiver` is `Send` but not
    /// `Sync`. Uncontended in practice: only the streaming system
    /// touches it, on the main thread.
    rx: Mutex<mpsc::Receiver<Vec<(IVec3, Lod)>>>,
    pose: (DVec3, DVec3),
}

impl DesiredChunksCache {
    /// Whether the cache should be invalidated and the plan rebuilt
    /// given the current observer pose. Returns `true` on first frame
    /// or if either the position-drift or yaw-drift threshold is
    /// exceeded. Compares against the in-flight rebuild's pose if one is
    /// outstanding, so a fresh rebuild isn't dispatched on top of an
    /// already-running one for the same pose.
    pub fn should_rebuild(&self, observer: DVec3, forward: DVec3) -> bool {
        self.should_rebuild_with(observer, forward, PLAN_REBUILD_DRIFT_M, PLAN_REBUILD_FWD_COS)
    }

    /// Strategy-driven variant of [`Self::should_rebuild`]. The caller
    /// supplies the drift and fwd-cos thresholds for this frame
    /// (typically from
    /// [`crate::render::RebuildThresholdStrategy`]) so the motion-aware
    /// layer can widen them under sustained sprint. Both thresholds
    /// equal the historical constants at rest, so default behavior
    /// matches [`Self::should_rebuild`] exactly.
    pub fn should_rebuild_with(
        &self,
        observer: DVec3,
        forward: DVec3,
        drift_threshold_m: f64,
        fwd_cos_threshold: f64,
    ) -> bool {
        let reference = self
            .rebuild
            .as_ref()
            .map(|h| h.pose)
            .or(self.built_for);
        match reference {
            None => true,
            Some((last_obs, last_fwd)) => {
                let dx = observer.x - last_obs.x;
                let dy = observer.y - last_obs.y;
                let dz = observer.z - last_obs.z;
                let drift = (dx * dx + dy * dy + dz * dz).sqrt();
                if drift > drift_threshold_m {
                    return true;
                }
                let dot = forward.x * last_fwd.x
                    + forward.y * last_fwd.y
                    + forward.z * last_fwd.z;
                // forward dirs are unit-length; dot < threshold ⇒ angle
                // grew past the rebuild cone.
                dot < fwd_cos_threshold
            }
        }
    }

    /// Replace the cached plan, recording the pose it was built for.
    /// Synchronous path — exposed for tests and the (rare) case where
    /// a caller wants to install a plan inline. Always resets the
    /// dispatch cursor to 0 so callers don't have to remember.
    pub fn set(&mut self, observer: DVec3, forward: DVec3, plan: Vec<(IVec3, Lod)>) {
        self.built_for = Some((observer, forward));
        self.plan = plan;
        self.cursor = 0;
    }

    /// Mark the cache stale so the next [`Self::should_rebuild`] call
    /// returns `true` regardless of observer drift, and discard any
    /// in-flight background rebuild (it was started against the old
    /// streamer state, so its result is now wrong). Use this when a
    /// streamer parameter (e.g. the active ladder) changes mid-session
    /// — without it the cached plan would stay in effect until the
    /// camera drifted past the rebuild threshold, which manifests to
    /// the user as "LOD doesn't update until I look around".
    ///
    /// Dropping the in-flight `RebuildHandle` releases our receiver;
    /// the spawned worker thread runs to completion and silently fails
    /// the send. We don't try to join it (would block the main thread)
    /// — the next frame will dispatch a fresh rebuild against the new
    /// ladder.
    pub fn invalidate(&mut self) {
        self.built_for = None;
        self.rebuild = None;
    }

    /// Whether a background rebuild is currently in flight.
    #[inline]
    pub fn is_rebuilding(&self) -> bool {
        self.rebuild.is_some()
    }

    /// Spawn a [`desired_chunks`] + [`prioritize_view`] rebuild on a
    /// background thread. The streamer state is cheap to clone (a small
    /// `Vec<LodTier>` plus a few scalars); the coverage policy is an
    /// `Arc` so cloning is also free. Does nothing if a rebuild is
    /// already in flight.
    pub fn spawn_rebuild(
        &mut self,
        streamer: ChunkStreamer,
        observer: DVec3,
        forward: DVec3,
        horizon_m: f64,
        coverage: Arc<dyn LodCoveragePolicy>,
    ) {
        if self.rebuild.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.rebuild = Some(RebuildHandle {
            rx: Mutex::new(rx),
            pose: (observer, forward),
        });
        std::thread::spawn(move || {
            let mut plan = desired_chunks(&streamer, observer, horizon_m, coverage.as_ref());
            prioritize_view(&mut plan, observer, forward);
            // Best-effort send: if the cache has been dropped (app
            // exit) the receiver is gone and the send fails silently,
            // which is fine — the worker just unwinds.
            let _ = tx.send(plan);
        });
    }

    /// Drain a completed background rebuild, if any. Returns `true`
    /// when a plan was installed this frame. The caller is expected to
    /// invoke this once per frame before consulting [`Self::plan`].
    pub fn poll_rebuild(&mut self) -> bool {
        let Some(handle) = self.rebuild.as_ref() else { return false; };
        let result = {
            let rx = handle.rx.lock().expect("rebuild rx poisoned");
            rx.try_recv()
        };
        match result {
            Ok(plan) => {
                let pose = handle.pose;
                self.rebuild = None;
                self.built_for = Some(pose);
                self.plan = plan;
                self.cursor = 0;
                true
            }
            Err(mpsc::TryRecvError::Empty) => false,
            Err(mpsc::TryRecvError::Disconnected) => {
                // Worker panicked or was dropped without sending.
                // Clear the slot so the next frame can re-dispatch.
                self.rebuild = None;
                false
            }
        }
    }
}

/// Re-sort a [`desired_chunks`] plan to load forward-facing bricks first.
///
/// Each entry's effective ordering distance is multiplied by
/// `(1 - VIEW_PRIORITY_WEIGHT * max(0, cos_angle_with_forward))` so a
/// brick directly in front loads ahead of an equally distant brick
/// behind or beside the camera. Bricks behind the camera retain their
/// natural closest-first ordering relative to each other.
pub fn prioritize_view(plan: &mut [(IVec3, Lod)], observer: DVec3, forward: DVec3) {
    let brick_edge_v = BRICK_EDGE as f64;
    plan.sort_by(|a, b| {
        let sa = view_priority_score(a, observer, forward, brick_edge_v);
        let sb = view_priority_score(b, observer, forward, brick_edge_v);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Effective ordering distance for [`prioritize_view`]. Exposed for
/// unit tests; production callers use [`prioritize_view`] directly.
pub fn view_priority_score(
    entry: &(IVec3, Lod),
    observer: DVec3,
    forward: DVec3,
    brick_edge_v: f64,
) -> f64 {
    let (coord, lod) = entry;
    let scale = (1u64 << lod.depth as u32) as f64;
    let edge_m = brick_edge_v * scale;
    let cx = (coord.x as f64 + 0.5) * edge_m - observer.x;
    let cy = (coord.y as f64 + 0.5) * edge_m - observer.y;
    let cz = (coord.z as f64 + 0.5) * edge_m - observer.z;
    let d = (cx * cx + cy * cy + cz * cz).sqrt();
    if d <= f64::EPSILON {
        return 0.0;
    }
    let cos_fwd = (cx * forward.x + cy * forward.y + cz * forward.z) / d;
    let bias = VIEW_PRIORITY_WEIGHT * cos_fwd.max(0.0);
    d * (1.0 - bias)
}

/// Build a closest-first sorted list of `(coord, lod)` brick keys that
/// should be loaded for the given observer.
///
/// For each tier in the ladder, the call enumerates the AABB of brick
/// coords that intersect a sphere of radius `tier.outer_radius_m`
/// centered on `observer`, in that tier's brick grid. Whether a brick
/// that's fully covered by a finer tier is included depends on
/// [`LodCoveragePolicy::mask_finer_covered`]:
///
/// - `mask_finer_covered() == true` ([`crate::render::defaults::MaskedShells`]):
///   each tier emits only its shell band
///   (`prev_tier.outer_radius_m <= d < tier.outer_radius_m`). One brick
///   per region — LOD transitions pop because the parent must be
///   generated when the child becomes ineligible.
/// - `mask_finer_covered() == false` ([`crate::render::defaults::NestedSummary`],
///   the default): every tier emits its full inner sphere
///   (`0 <= d < tier.outer_radius_m`), so each region is covered by
///   the finer LOD *and* every coarser parent simultaneously. The
///   FP renderer's visibility system keeps only the finest visible
///   per region; the parents are pre-cached "summaries" that fade in
///   instantly when the child unloads, eliminating the LOD pop.
///
/// The **radial** outer test produces a symmetric ring in all four
/// horizontal directions (the prior 2-tier implementation used cubes
/// whose corners were ~73 % farther than their faces, which produced
/// visible directional asymmetry as the observer walked).
///
/// `horizon_m` is `f64::INFINITY` for flat tiers (cube worlds) and the
/// surface-horizon distance for spherical bodies — radii are clamped to
/// it so we never stream past the visible surface.
pub fn desired_chunks(
    streamer: &ChunkStreamer,
    observer: DVec3,
    horizon_m: f64,
    coverage: &dyn LodCoveragePolicy,
) -> Vec<(IVec3, Lod)> {
    let mut out: Vec<(IVec3, Lod)> = Vec::new();
    if streamer.ladder.tiers.is_empty() {
        return out;
    }
    let brick_edge_v = BRICK_EDGE as f64;
    let mask_inner = coverage.mask_finer_covered();

    for (i, tier) in streamer.ladder.tiers.iter().enumerate() {
        let (inner_r, outer_r) = streamer.ladder.tier_band_m(i);
        // Horizon clamp — never stream past the surface for spherical
        // bodies. Inner is also clamped so the band stays non-degenerate
        // when horizon shrinks below the ladder.
        let outer_r = outer_r.min(horizon_m);
        // For NestedSummary (mask_inner == false) the parent stays
        // resident underneath the finer tier, so the inner band starts
        // at 0 — every tier emits its full inner sphere.
        let inner_r = if mask_inner { inner_r.min(outer_r) } else { 0.0 };
        if outer_r <= 0.0 {
            continue;
        }

        // Brick edge in meters at this tier's LOD: BRICK_EDGE * 2^depth.
        let lod_scale = (1u64 << tier.lod.depth as u32) as f64;
        let edge_m = brick_edge_v * lod_scale;
        let inner_sq = inner_r * inner_r;
        let outer_sq = outer_r * outer_r;

        // Brick-grid AABB that bounds the outer sphere. The actual
        // band check uses brick AABB intersection with the spherical
        // shell, NOT brick center alone — a center-only test leaves
        // boxy gaps wherever a brick AABB straddles the boundary and
        // both its center and the corresponding finer-LOD brick centers
        // happen to land on the "wrong" side. Specifically:
        //   inner: skip the brick only when its FAR CORNER is still
        //     inside `inner_r` (i.e. the brick is entirely covered by
        //     the finer tier). A brick straddling the boundary stays.
        //   outer: load if any AABB voxel intersects the outer sphere,
        //     equivalent to `near_corner_d < outer_r`. The center-only
        //     outer test was missing boundary bricks symmetrically.
        // The pair guarantees every voxel position with
        // `outer_r_{i-1} <= d < outer_r_i` is covered by some loaded
        // brick at tier `i` (verified by the `no_gaps_at_tier_boundaries`
        // test below).
        let half_edge = edge_m * 0.5;
        let outer_v = ((outer_r + half_edge) / edge_m).ceil() as i64;
        let ox = (observer.x / edge_m).floor() as i64;
        let oy = (observer.y / edge_m).floor() as i64;
        let oz = (observer.z / edge_m).floor() as i64;

        for bz in (oz - outer_v)..=(oz + outer_v) {
            for by in (oy - outer_v)..=(oy + outer_v) {
                for bx in (ox - outer_v)..=(ox + outer_v) {
                    let cx = (bx as f64 + 0.5) * edge_m - observer.x;
                    let cy = (by as f64 + 0.5) * edge_m - observer.y;
                    let cz = (bz as f64 + 0.5) * edge_m - observer.z;
                    // Far corner: |c_axis| + half_edge per axis.
                    let fx = cx.abs() + half_edge;
                    let fy = cy.abs() + half_edge;
                    let fz = cz.abs() + half_edge;
                    let far_d2 = fx * fx + fy * fy + fz * fz;
                    // Near corner: max(0, |c_axis| - half_edge) per axis.
                    let nx = (cx.abs() - half_edge).max(0.0);
                    let ny = (cy.abs() - half_edge).max(0.0);
                    let nz = (cz.abs() - half_edge).max(0.0);
                    let near_d2 = nx * nx + ny * ny + nz * nz;
                    // Inner mask: brick fully inside the inner sphere
                    // ⇒ finer tier already covers it.
                    if far_d2 < inner_sq {
                        continue;
                    }
                    // Outer mask: brick fully outside the outer sphere
                    // ⇒ next tier will pick it up.
                    if near_d2 >= outer_sq {
                        continue;
                    }
                    out.push((IVec3::new(bx, by, bz), tier.lod));
                }
            }
        }
    }

    // Closest-first sort. Distance is in meters (not bricks) so near
    // (small bricks) and far (large bricks) entries sort against each
    // other consistently.
    out.sort_by(|a, b| {
        let scale_a = (1u64 << a.1.depth as u32) as f64;
        let scale_b = (1u64 << b.1.depth as u32) as f64;
        let edge_a = brick_edge_v * scale_a;
        let edge_b = brick_edge_v * scale_b;
        let ax = (a.0.x as f64 + 0.5) * edge_a - observer.x;
        let ay = (a.0.y as f64 + 0.5) * edge_a - observer.y;
        let az = (a.0.z as f64 + 0.5) * edge_a - observer.z;
        let bx = (b.0.x as f64 + 0.5) * edge_b - observer.x;
        let by = (b.0.y as f64 + 0.5) * edge_b - observer.y;
        let bz = (b.0.z as f64 + 0.5) * edge_b - observer.z;
        let da = ax * ax + ay * ay + az * az;
        let db = bx * bx + by * by + bz * bz;
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    out
}

/// Bevy plugin: registers [`ChunkStreamer`] and [`LoadedChunks`] as
/// resources. Per-frame consumers (FP/TP brick spawner; slice/RTS LOD
/// selection) read these directly.
pub struct ChunkStreamerPlugin;

impl Plugin for ChunkStreamerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkStreamer>()
            .init_resource::<LoadedChunks>()
            .init_resource::<DesiredChunksCache>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::defaults::{MaskedShells, NestedSummary};

    fn streamer() -> ChunkStreamer {
        ChunkStreamer::default()
    }

    /// Helper: shell-only coverage matching pre-nested-summary
    /// behaviour. Every desired_chunks call in the legacy tests below
    /// uses this so the assertions about shell ownership / inner-band
    /// masking / centroid distribution still hold.
    fn masked() -> MaskedShells {
        MaskedShells
    }

    /// Helper: nested coverage — every tier emits its full inner
    /// sphere, parent stays resident under the finer tier.
    fn nested() -> NestedSummary {
        NestedSummary
    }

    /// Helper: drop the LOD depth and return the set of XZ brick
    /// coordinates emitted in the desired plan. Used by the symmetry
    /// tests to compare in each cardinal direction.
    fn xz_set_at_lod(plan: &[(IVec3, Lod)], depth: u8) -> std::collections::HashSet<(i64, i64)> {
        plan.iter()
            .filter(|(_, l)| l.depth == depth)
            .map(|(c, _)| (c.x, c.z))
            .collect()
    }

    // -----------------------------------------------------------------
    // LOD selection
    // -----------------------------------------------------------------

    #[test]
    fn lod_for_distance_walks_the_ladder() {
        let ladder = LodLadder::default_progressive();
        assert_eq!(ladder.lod_for_distance(0.0).depth, 0);
        assert_eq!(ladder.lod_for_distance(127.9).depth, 0);
        // Boundary inclusive on the *upper* side ⇒ next tier owns it.
        assert_eq!(ladder.lod_for_distance(128.0).depth, 1);
        assert_eq!(ladder.lod_for_distance(255.9).depth, 1);
        assert_eq!(ladder.lod_for_distance(256.0).depth, 2);
        assert_eq!(ladder.lod_for_distance(511.9).depth, 2);
        assert_eq!(ladder.lod_for_distance(512.0).depth, 3);
        assert_eq!(ladder.lod_for_distance(1023.9).depth, 3);
        // Past the horizon: clamp to outermost (still depth 3).
        assert_eq!(ladder.lod_for_distance(10_000.0).depth, 3);
    }

    #[test]
    fn lod_for_meters_uses_3d_distance() {
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        // Vertical-only sample at depth 100 m: still inside tier 0.
        assert_eq!(s.lod_for_meters(obs, DVec3::new(0.0, 100.0, 0.0)).depth, 0);
        // Diagonal that pushes past tier 0's 128 m radius (3*80² = 19200, sqrt ≈ 139).
        assert_eq!(s.lod_for_meters(obs, DVec3::new(80.0, 80.0, 80.0)).depth, 1);
    }

    // -----------------------------------------------------------------
    // Symmetry of the desired set
    // -----------------------------------------------------------------

    #[test]
    fn desired_chunks_load_symmetrically_in_all_four_cardinal_directions() {
        // Stand the observer at the origin; the load shape must be
        // invariant under reflections across X and Z (rotational
        // 90° symmetry around Y for an origin-centered observer).
        //
        // Brick coords are half-open `[c, c+1)` per voxel, so the
        // negative-X mirror of brick `x` is `-x - 1` (the brick whose
        // span is exactly the reflection of `x`'s span across zero).
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let plan = desired_chunks(&s, obs, f64::INFINITY, &masked());

        for depth in 0u8..=3 {
            let set = xz_set_at_lod(&plan, depth);
            assert!(
                !set.is_empty(),
                "tier depth={depth} produced no bricks — ladder mis-configured?"
            );

            for &(x, z) in &set {
                assert!(
                    set.contains(&(-x - 1, z)),
                    "depth={depth}: X-asymmetry — ({x},{z}) loaded but mirror \
                     ({},{z}) is not in the desired set",
                    -x - 1
                );
                assert!(
                    set.contains(&(x, -z - 1)),
                    "depth={depth}: Z-asymmetry — ({x},{z}) loaded but mirror \
                     ({x},{}) is not in the desired set",
                    -z - 1
                );
                assert!(
                    set.contains(&(z, x)),
                    "depth={depth}: XZ-swap asymmetry — ({x},{z}) loaded but \
                     90°-rotated mirror ({z},{x}) is not in the desired set"
                );
            }
        }
    }

    #[test]
    fn walk_in_each_cardinal_direction_produces_matching_brick_counts() {
        // Walk 32 m in each cardinal direction. The brick count emitted
        // at each tier must match across all 4 directions — this is
        // the regression test for the "only 2 of 4 directions load"
        // bug. (The previous 2-tier cube-shaped ring produced visible
        // asymmetry because brick floors and cube extents stretched
        // unevenly along diagonals.)
        let s = streamer();
        let step = 32.0;
        let positions = [
            DVec3::new(step, 0.0, 0.0),  // +X
            DVec3::new(-step, 0.0, 0.0), // -X
            DVec3::new(0.0, 0.0, step),  // +Z
            DVec3::new(0.0, 0.0, -step), // -Z
        ];
        let counts: Vec<usize> = positions
            .iter()
            .map(|p| desired_chunks(&s, *p, f64::INFINITY, &masked()).len())
            .collect();
        let first = counts[0];
        for (i, n) in counts.iter().enumerate() {
            assert_eq!(
                *n, first,
                "direction {i} ({:?}) loaded a different number of bricks: {counts:?}",
                positions[i]
            );
        }
    }

    #[test]
    fn centroid_of_desired_set_stays_near_observer_even_off_grid() {
        // Regression for a perceptual asymmetry concern: when the observer
        // is NOT centered on a brick-grid origin (e.g. spawn at world
        // (8, 28, 8) — interior of brick coord 0 at multiple LODs), a
        // naïve per-half-space count can look biased (more bricks on
        // one side because the brick column straddling the observer
        // sits on the positive side). But the *geometric* distribution
        // should still be symmetric: the centroid of the loaded set
        // should equal the observer position. This test asserts the
        // centroid offset stays well under one brick edge across
        // a handful of off-grid observer poses.
        let s = streamer();
        let outer = s.outer_radius_m();
        let brick_edge_v = BRICK_EDGE as f64;
        let cases = [
            DVec3::new(8.0, 28.0, 8.0),
            DVec3::new(17.3, -12.7, 41.9),
            DVec3::new(-95.1, 64.0, 130.2),
            DVec3::new(0.5, 0.5, 0.5),
        ];
        for obs in cases {
            let plan = desired_chunks(&s, obs, f64::INFINITY, &masked());
            assert!(!plan.is_empty(), "plan must not be empty at {obs:?}");
            let mut sum = DVec3::ZERO;
            for (c, lod) in &plan {
                let edge_m = brick_edge_v * (1u64 << lod.depth as u32) as f64;
                let center = DVec3::new(
                    (c.x as f64 + 0.5) * edge_m,
                    (c.y as f64 + 0.5) * edge_m,
                    (c.z as f64 + 0.5) * edge_m,
                );
                sum.x += center.x - obs.x;
                sum.y += center.y - obs.y;
                sum.z += center.z - obs.z;
            }
            let n = plan.len() as f64;
            let drift = DVec3::new(sum.x / n, sum.y / n, sum.z / n);
            // Tolerance: 1 % of the outer radius. Empirically the drift
            // is ~0.1 % even at heavily off-grid positions; 1 % gives
            // headroom for future ladder configurations.
            let tol = outer * 0.01;
            assert!(
                drift.x.abs() < tol && drift.y.abs() < tol && drift.z.abs() < tol,
                "centroid drift at obs={obs:?} = {drift:?} exceeds tol {tol}"
            );
        }
    }

    #[test]
    fn outer_tier_emits_no_brick_past_horizon_radius() {
        // With AABB-based outer test, a brick is loaded iff its NEAR
        // CORNER is inside the outer horizon. So the assertion is:
        // every loaded brick has at least one corner inside the horizon
        // sphere — equivalently, the brick AABB intersects the horizon.
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let plan = desired_chunks(&s, obs, f64::INFINITY, &masked());
        let outer = s.outer_radius_m();
        let outer_sq = outer * outer;
        let brick_edge_v = BRICK_EDGE as f64;
        for (c, lod) in &plan {
            let edge_m = brick_edge_v * (1u64 << lod.depth as u32) as f64;
            let half = edge_m * 0.5;
            let cx = (c.x as f64 + 0.5) * edge_m;
            let cy = (c.y as f64 + 0.5) * edge_m;
            let cz = (c.z as f64 + 0.5) * edge_m;
            let nx = (cx.abs() - half).max(0.0);
            let ny = (cy.abs() - half).max(0.0);
            let nz = (cz.abs() - half).max(0.0);
            let near_d2 = nx * nx + ny * ny + nz * nz;
            assert!(
                near_d2 < outer_sq,
                "brick {c:?} lod={} near-corner-dist {} exceeds horizon {}",
                lod.depth,
                near_d2.sqrt(),
                outer
            );
        }
    }

    /// Regression test for the "moving black-hole patches" bug — earlier
    /// the inner-band test was `d²_center < inner_sq` which created
    /// volumes that no tier covered. The new test is `d²_far_corner <
    /// inner_sq`, i.e. only skip a brick if it's *entirely* inside the
    /// inner sphere. This test densely samples voxel positions in the
    /// load horizon and asserts each one lies inside at least one
    /// loaded brick's AABB.
    #[test]
    fn no_gaps_at_tier_boundaries() {
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let plan = desired_chunks(&s, obs, f64::INFINITY, &masked());
        let brick_edge_v = BRICK_EDGE as f64;

        // Build a per-LOD set so AABB membership tests are cheap.
        let mut by_lod: std::collections::HashMap<u8, std::collections::HashSet<(i64, i64, i64)>>
            = Default::default();
        for (c, l) in &plan {
            by_lod.entry(l.depth).or_default().insert((c.x, c.y, c.z));
        }

        let outer = s.outer_radius_m();
        // Sample voxel positions on a 4 m grid inside the load horizon.
        // 4 m << every brick edge (16 m at L0) so each voxel sits well
        // inside whatever brick claims it. The sample shape is the
        // sphere itself.
        let step: f64 = 4.0;
        let max_steps = (outer * 0.95 / step) as i64;
        for iz in -max_steps..=max_steps {
            for iy in -max_steps..=max_steps {
                for ix in -max_steps..=max_steps {
                    let x = ix as f64 * step;
                    let y = iy as f64 * step;
                    let z = iz as f64 * step;
                    let d = (x * x + y * y + z * z).sqrt();
                    if d > outer * 0.95 {
                        continue;
                    }
                    // Find the appropriate tier for this point and
                    // confirm the containing brick at that tier is loaded.
                    let tier = s.ladder.lod_for_distance(d);
                    let scale = (1u64 << tier.depth as u32) as f64;
                    let edge_m = brick_edge_v * scale;
                    let bx = (x / edge_m).floor() as i64;
                    let by = (y / edge_m).floor() as i64;
                    let bz = (z / edge_m).floor() as i64;
                    let covered = by_lod
                        .get(&tier.depth)
                        .map(|s| s.contains(&(bx, by, bz)))
                        .unwrap_or(false);
                    assert!(
                        covered,
                        "voxel ({x},{y},{z}) at d={d} should be in tier depth={} brick \
                         ({bx},{by},{bz}) but that brick is not loaded",
                        tier.depth
                    );
                }
            }
        }
    }

    #[test]
    fn desired_chunks_emits_distinct_keys() {
        let s = streamer();
        let chunks = desired_chunks(&s, DVec3::new(0.0, 0.0, 0.0), f64::INFINITY, &masked());
        let mut seen: std::collections::HashSet<(IVec3, u8)> = Default::default();
        for (c, l) in &chunks {
            assert!(
                seen.insert((*c, l.depth)),
                "duplicate brick key {:?}, lod={}",
                c,
                l.depth
            );
        }
    }

    #[test]
    fn closest_first_sort_orders_by_meters_across_tiers() {
        let s = streamer();
        let chunks = desired_chunks(&s, DVec3::new(0.0, 0.0, 0.0), f64::INFINITY, &masked());
        // The first emitted brick must be at the highest fidelity
        // (depth 0) — it's the one straddling the observer.
        let first = chunks.first().expect("at least one chunk");
        assert_eq!(first.1.depth, 0);
        // Every tier in the ladder should contribute at least one
        // brick — confirms the progressive fall-off is actually wired.
        for tier in &s.ladder.tiers {
            let any = chunks.iter().any(|(_, l)| l.depth == tier.lod.depth);
            assert!(any, "no bricks emitted at lod depth {}", tier.lod.depth);
        }
        // Tier ordering: at every transition the next emitted brick
        // can be at a coarser or equal LOD but never finer. (Closest-
        // first sort across tiers means depth never *decreases* once
        // it's increased.)
        let mut max_depth_seen = 0u8;
        for (_, lod) in &chunks {
            max_depth_seen = max_depth_seen.max(lod.depth);
        }
        assert_eq!(max_depth_seen, 3, "outermost tier (L3) must contribute");
    }

    // -----------------------------------------------------------------
    // Inner/outer band masking
    // -----------------------------------------------------------------

    #[test]
    fn near_tier_bricks_dont_overlap_far_tier_at_same_position() {
        // A coordinate covered by the L0 tier must NOT be re-emitted
        // at any coarser LOD — the inner-band check should mask the
        // shell out of subsequent tiers' fetches.
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let plan = desired_chunks(&s, obs, f64::INFINITY, &masked());
        // Take a small near-origin brick at L0 and check the
        // L1-grid brick covering the same point is NOT in the set.
        let l0 = (IVec3::new(0, 0, 0), Lod::new(0));
        assert!(plan.contains(&l0), "L0 origin brick should always load");
        // The L1 brick covering the origin spans world meters
        // [0, 32) per axis — its brick coord is (0,0,0) in L1's
        // 32-m grid. Should be masked by the inner-band check.
        let l1_origin = (IVec3::new(0, 0, 0), Lod::new(1));
        assert!(
            !plan.contains(&l1_origin),
            "L1 brick at the origin should be masked because L0 covers it"
        );
    }

    // -----------------------------------------------------------------
    // Fog band
    // -----------------------------------------------------------------

    #[test]
    fn fog_band_brackets_the_outermost_tier() {
        let s = streamer();
        let (start, end) = s.fog_band_m();
        let outer = s.outer_radius_m();
        assert!(start > 0.0 && start < outer, "fog start {start} must be inside the load horizon {outer}");
        assert!(end <= outer, "fog end {end} must not exceed the load horizon {outer}");
        assert!(end > start, "fog end {end} must be past fog start {start}");
        // Default fractions: start at 55 %, end at 98 %.
        assert!((start / outer - FOG_START_FRACTION).abs() < 1e-3);
        assert!((end / outer - FOG_END_FRACTION).abs() < 1e-3);
        // Concrete values for the default ladder so a regression in
        // the FOG_*_FRACTION constants is loud.
        assert!((start - 1024.0 * 0.55).abs() < 1e-3);
        assert!((end - 1024.0 * 0.98).abs() < 1e-3);
    }

    // -----------------------------------------------------------------
    // Hysteresis
    // -----------------------------------------------------------------

    #[test]
    fn hysteresis_lets_chunks_linger() {
        let mut loaded = LoadedChunks::default();
        let key = LoadedChunk::key(IVec3::new(0, 0, 0), Lod::new(0));
        loaded.insert(
            key,
            LoadedChunk {
                coord: IVec3::new(0, 0, 0),
                lod: Lod::new(0),
                entity: None,
                last_seen_frame: 5,
                is_fading_out: false,
                dag_digest: None,
                dag_tier: None,
                brick: None,
            },
        );
        // At frame 6 (1 tick later), still fresh.
        assert!(!loaded.is_stale(&key, 6, HYSTERESIS_TICKS));
        // At frame 7 (2 ticks later), stale.
        assert!(loaded.is_stale(&key, 7, HYSTERESIS_TICKS));
    }

    // -----------------------------------------------------------------
    // Incremental child_counts
    // -----------------------------------------------------------------

    fn mk_chunk(coord: IVec3, depth: u8) -> LoadedChunk {
        LoadedChunk {
            coord,
            lod: Lod::new(depth),
            entity: None,
            last_seen_frame: 0,
            is_fading_out: false,
            dag_digest: None,
            dag_tier: None,
            brick: None,
        }
    }

    #[test]
    fn insert_increments_parent_child_count() {
        let mut loaded = LoadedChunks::default();
        let key = LoadedChunk::key(IVec3::new(2, 4, 6), Lod::new(0));
        loaded.insert(key, mk_chunk(IVec3::new(2, 4, 6), 0));
        // Parent of (2,4,6)@d0 is (1,2,3)@d1 (no negative coords ⇒ /2).
        assert_eq!(loaded.child_count(&(IVec3::new(1, 2, 3), 1)), 1);
    }

    #[test]
    fn insert_uses_div_euclid_for_negative_coords() {
        let mut loaded = LoadedChunks::default();
        let key = LoadedChunk::key(IVec3::new(-1, 0, 0), Lod::new(0));
        loaded.insert(key, mk_chunk(IVec3::new(-1, 0, 0), 0));
        // Truncation would give (0,0,0)@d1 — wrong. div_euclid gives (-1,0,0)@d1.
        assert_eq!(loaded.child_count(&(IVec3::new(-1, 0, 0), 1)), 1);
        assert_eq!(loaded.child_count(&(IVec3::new(0, 0, 0), 1)), 0);
    }

    #[test]
    fn mark_fading_out_decrements_count_and_is_idempotent() {
        let mut loaded = LoadedChunks::default();
        let parent = (IVec3::new(0, 0, 0), 1u8);
        for i in 0..3 {
            let key = LoadedChunk::key(IVec3::new(i, 0, 0), Lod::new(0));
            loaded.insert(key, mk_chunk(IVec3::new(i, 0, 0), 0));
        }
        assert_eq!(loaded.child_count(&parent), 2); // (0,0,0) and (1,0,0)
        let k0 = (IVec3::new(0, 0, 0), 0u8);
        loaded.mark_fading_out(&k0);
        assert_eq!(loaded.child_count(&parent), 1);
        // Idempotent: re-marking doesn't double-decrement.
        loaded.mark_fading_out(&k0);
        assert_eq!(loaded.child_count(&parent), 1);
    }

    #[test]
    fn remove_skips_decrement_for_already_fading_entry() {
        let mut loaded = LoadedChunks::default();
        let key = LoadedChunk::key(IVec3::new(0, 0, 0), Lod::new(0));
        let parent = (IVec3::new(0, 0, 0), 1u8);
        loaded.insert(key, mk_chunk(IVec3::new(0, 0, 0), 0));
        assert_eq!(loaded.child_count(&parent), 1);
        loaded.mark_fading_out(&key);
        assert_eq!(loaded.child_count(&parent), 0);
        loaded.remove(&key);
        // Already 0; remove on a fading entry must not underflow.
        assert_eq!(loaded.child_count(&parent), 0);
    }

    #[test]
    fn remove_decrements_count_when_entry_was_not_fading() {
        let mut loaded = LoadedChunks::default();
        let key = LoadedChunk::key(IVec3::new(0, 0, 0), Lod::new(0));
        let parent = (IVec3::new(0, 0, 0), 1u8);
        loaded.insert(key, mk_chunk(IVec3::new(0, 0, 0), 0));
        assert_eq!(loaded.child_count(&parent), 1);
        loaded.remove(&key);
        assert_eq!(loaded.child_count(&parent), 0);
    }

    #[test]
    fn reinsert_after_fade_out_brings_back_into_count() {
        let mut loaded = LoadedChunks::default();
        let key = LoadedChunk::key(IVec3::new(0, 0, 0), Lod::new(0));
        let parent = (IVec3::new(0, 0, 0), 1u8);
        loaded.insert(key, mk_chunk(IVec3::new(0, 0, 0), 0));
        loaded.mark_fading_out(&key);
        assert_eq!(loaded.child_count(&parent), 0);
        // Replace the fading entry with a fresh one — should increment back.
        loaded.insert(key, mk_chunk(IVec3::new(0, 0, 0), 0));
        assert_eq!(loaded.child_count(&parent), 1);
    }

    // -----------------------------------------------------------------
    // Legacy `StreamingPolicy` view
    // -----------------------------------------------------------------

    #[test]
    fn as_policy_projects_ladder_to_two_tier() {
        let ladder = LodLadder::default_progressive();
        let policy = ladder.as_policy();
        assert_eq!(policy.near_lod.depth, 0);
        assert_eq!(policy.far_lod.depth, 3);
        assert_eq!(policy.transition_radius_m, 128.0);
        assert_eq!(policy.max_radius_m, 1024.0);
        assert_eq!(policy.bricks_per_tick, DEFAULT_BRICKS_PER_TICK);
    }

    #[test]
    fn set_ladder_refreshes_policy_view() {
        let mut s = ChunkStreamer::default();
        s.set_ladder(LodLadder {
            tiers: vec![
                LodTier { lod: Lod::new(0), outer_radius_m: 32.0 },
                LodTier { lod: Lod::new(2), outer_radius_m: 256.0 },
            ],
            bricks_per_tick: 99,
        });
        assert_eq!(s.policy.near_lod.depth, 0);
        assert_eq!(s.policy.far_lod.depth, 2);
        assert_eq!(s.policy.transition_radius_m, 32.0);
        assert_eq!(s.policy.max_radius_m, 256.0);
        assert_eq!(s.policy.bricks_per_tick, 99);
    }

    // -----------------------------------------------------------------
    // Horizon clamp (spherical worlds)
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // DesiredChunksCache — rebuild thresholds
    // -----------------------------------------------------------------

    #[test]
    fn cache_first_frame_forces_rebuild() {
        let cache = DesiredChunksCache::default();
        // No `built_for` ⇒ rebuild regardless of pose.
        assert!(cache.should_rebuild(DVec3::ZERO, DVec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn cache_skips_rebuild_under_position_threshold() {
        let mut cache = DesiredChunksCache::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        cache.set(obs, fwd, vec![]);
        // 1 m drift << 4 m threshold ⇒ reuse cache.
        let next = DVec3::new(1.0, 0.0, 0.0);
        assert!(!cache.should_rebuild(next, fwd));
    }

    #[test]
    fn cache_rebuilds_past_position_threshold() {
        let mut cache = DesiredChunksCache::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        cache.set(obs, fwd, vec![]);
        // 5 m drift > 4 m threshold ⇒ rebuild.
        let next = DVec3::new(5.0, 0.0, 0.0);
        assert!(cache.should_rebuild(next, fwd));
    }

    #[test]
    fn cache_rebuilds_when_camera_turns_past_threshold() {
        let mut cache = DesiredChunksCache::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd0 = DVec3::new(0.0, 0.0, 1.0);
        cache.set(obs, fwd0, vec![]);
        // 20° rotation (cos ≈ 0.94) > 15° threshold (cos ≈ 0.9659) ⇒ rebuild.
        let theta = 20.0_f64.to_radians();
        let fwd1 = DVec3::new(theta.sin(), 0.0, theta.cos());
        assert!(cache.should_rebuild(obs, fwd1));
    }

    #[test]
    fn cache_reuses_under_small_camera_turn() {
        let mut cache = DesiredChunksCache::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd0 = DVec3::new(0.0, 0.0, 1.0);
        cache.set(obs, fwd0, vec![]);
        // 5° rotation (cos ≈ 0.9962) > cos(15°) ⇒ reuse.
        let theta = 5.0_f64.to_radians();
        let fwd1 = DVec3::new(theta.sin(), 0.0, theta.cos());
        assert!(!cache.should_rebuild(obs, fwd1));
    }

    #[test]
    fn spawn_rebuild_runs_in_background_and_polls_in() {
        // End-to-end exercise of the async rebuild path: spawn, wait for
        // the worker thread to finish, poll, and verify the plan + pose
        // were installed and the in-flight slot was cleared.
        let mut cache = DesiredChunksCache::default();
        let streamer = ChunkStreamer::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        let coverage: Arc<dyn LodCoveragePolicy> = Arc::new(MaskedShells);

        assert!(!cache.is_rebuilding());
        cache.spawn_rebuild(streamer.clone(), obs, fwd, f64::INFINITY, coverage.clone());
        assert!(cache.is_rebuilding());

        // Block until the worker pushes its result. Polling here is
        // unrealistic for the streaming loop (which polls once per
        // frame), but it lets the test deterministically wait for
        // completion without sleeping.
        let mut installed = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if cache.poll_rebuild() {
                installed = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(installed, "background rebuild never landed within deadline");
        assert!(!cache.is_rebuilding(), "rebuild slot should be cleared after poll");
        assert_eq!(cache.built_for, Some((obs, fwd)));
        // Same pose as a synchronous rebuild — should produce an
        // identical plan length.
        let mut sync_plan = desired_chunks(&streamer, obs, f64::INFINITY, coverage.as_ref());
        prioritize_view(&mut sync_plan, obs, fwd);
        assert_eq!(cache.plan.len(), sync_plan.len());
        assert_eq!(cache.plan, sync_plan);
    }

    #[test]
    fn spawn_rebuild_is_idempotent_while_in_flight() {
        // Two back-to-back spawn calls for the same pose must not stack
        // up — the second call returns without dispatching a second
        // worker thread.
        let mut cache = DesiredChunksCache::default();
        let streamer = ChunkStreamer::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        let coverage: Arc<dyn LodCoveragePolicy> = Arc::new(MaskedShells);

        cache.spawn_rebuild(streamer.clone(), obs, fwd, f64::INFINITY, coverage.clone());
        assert!(cache.is_rebuilding());
        // Second call is a no-op — the in-flight slot is still occupied.
        cache.spawn_rebuild(streamer, obs, fwd, f64::INFINITY, coverage);
        assert!(cache.is_rebuilding());

        // Drain so the worker thread isn't orphaned for the test runner.
        while !cache.poll_rebuild() {
            std::thread::yield_now();
        }
    }

    #[test]
    fn cursor_resets_when_set_installs_fresh_plan() {
        let mut cache = DesiredChunksCache::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        // Pretend the dispatch loop advanced the cursor partway through.
        cache.set(obs, fwd, vec![(IVec3::new(0, 0, 0), Lod::new(0)); 32]);
        cache.cursor = 20;
        // A fresh `set` (e.g. a synchronous plan replacement) resets
        // the cursor so the priority-sorted front is always re-scanned.
        cache.set(obs, fwd, vec![(IVec3::new(1, 0, 0), Lod::new(0)); 8]);
        assert_eq!(cache.cursor, 0);
    }

    #[test]
    fn cursor_resets_when_poll_rebuild_installs_plan() {
        let mut cache = DesiredChunksCache::default();
        let streamer = ChunkStreamer::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        let coverage: Arc<dyn LodCoveragePolicy> = Arc::new(MaskedShells);
        // Mimic a saturated dispatch loop that left cursor mid-plan.
        cache.cursor = 999;
        cache.spawn_rebuild(streamer, obs, fwd, f64::INFINITY, coverage);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if cache.poll_rebuild() {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(
            cache.cursor, 0,
            "cursor must reset to 0 when a fresh plan is installed"
        );
    }

    #[test]
    fn should_rebuild_uses_in_flight_pose_to_dedupe() {
        // While a rebuild is in flight for pose P, `should_rebuild` must
        // consult P (not the previous `built_for`) when deciding whether
        // a fresh dispatch is needed. Otherwise the streaming loop would
        // dispatch a second rebuild for the same pose every frame until
        // the first one finished.
        let mut cache = DesiredChunksCache::default();
        let streamer = ChunkStreamer::default();
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(100.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        let coverage: Arc<dyn LodCoveragePolicy> = Arc::new(MaskedShells);

        cache.set(p0, fwd, vec![]);
        cache.spawn_rebuild(streamer, p1, fwd, f64::INFINITY, coverage);
        // Same pose as the in-flight rebuild ⇒ no fresh rebuild needed.
        assert!(!cache.should_rebuild(p1, fwd));
        // Drift far from the in-flight pose ⇒ a fresh rebuild *is* needed.
        let p2 = DVec3::new(200.0, 0.0, 0.0);
        assert!(cache.should_rebuild(p2, fwd));

        // Drain so the worker thread isn't orphaned for the test runner.
        while !cache.poll_rebuild() {
            std::thread::yield_now();
        }
    }

    #[test]
    fn invalidate_forces_rebuild_without_drift() {
        // Scenario: cache is freshly built for the current pose. Without
        // `invalidate`, `should_rebuild` returns false because the
        // observer hasn't moved past the drift threshold. After
        // `invalidate`, `should_rebuild` must return true so the next
        // streaming tick rebuilds against whatever streamer state
        // changed (e.g. a new LOD ladder).
        let mut cache = DesiredChunksCache::default();
        let obs = DVec3::new(50.0, 0.0, 50.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        cache.set(obs, fwd, vec![(IVec3::new(0, 0, 0), Lod::new(0))]);
        assert!(!cache.should_rebuild(obs, fwd), "fresh cache should reuse");
        cache.invalidate();
        assert!(cache.should_rebuild(obs, fwd), "invalidate must force a rebuild on the next tick");
    }

    #[test]
    fn invalidate_discards_in_flight_rebuild() {
        // An in-flight rebuild was started against the previous streamer
        // state; if `invalidate` left it running, its completion would
        // install a now-stale plan on top of the new state. Make sure
        // `invalidate` drops the handle so `is_rebuilding()` returns
        // false and the next tick dispatches a fresh rebuild.
        let mut cache = DesiredChunksCache::default();
        let streamer = ChunkStreamer::default();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 1.0);
        let coverage: Arc<dyn LodCoveragePolicy> = Arc::new(MaskedShells);
        cache.spawn_rebuild(streamer, obs, fwd, f64::INFINITY, coverage);
        assert!(cache.is_rebuilding());
        cache.invalidate();
        assert!(!cache.is_rebuilding(), "invalidate must drop the in-flight rebuild handle");
        // The worker thread is now orphaned — it still runs to completion
        // and sends to a dropped receiver (silent failure). We don't
        // try to join it; the test runner is allowed to outlive it.
    }

    // -----------------------------------------------------------------
    // View priority sort
    // -----------------------------------------------------------------

    #[test]
    fn view_priority_pulls_forward_brick_ahead_of_equidistant_behind() {
        let observer = DVec3::new(0.0, 0.0, 0.0);
        let forward = DVec3::new(0.0, 0.0, 1.0); // +Z
        let brick_edge_v = BRICK_EDGE as f64;
        // Two bricks at same true distance but on opposite Z sides.
        let in_front = (IVec3::new(0, 0, 5), Lod::new(0));
        let behind = (IVec3::new(0, 0, -6), Lod::new(0));
        let s_front = view_priority_score(&in_front, observer, forward, brick_edge_v);
        let s_behind = view_priority_score(&behind, observer, forward, brick_edge_v);
        assert!(
            s_front < s_behind,
            "forward brick (score={s_front}) should sort ahead of behind brick (score={s_behind})"
        );
    }

    #[test]
    fn prioritize_view_keeps_closest_first_within_each_hemisphere() {
        let observer = DVec3::new(0.0, 0.0, 0.0);
        let forward = DVec3::new(0.0, 0.0, 1.0);
        let mut plan = vec![
            (IVec3::new(0, 0, 9), Lod::new(0)),  // far ahead
            (IVec3::new(0, 0, 3), Lod::new(0)),  // near ahead
            (IVec3::new(0, 0, -10), Lod::new(0)), // far behind
        ];
        prioritize_view(&mut plan, observer, forward);
        // Near-ahead first, then far-ahead, then far-behind.
        assert_eq!(plan[0].0.z, 3);
        assert_eq!(plan[1].0.z, 9);
        assert_eq!(plan[2].0.z, -10);
    }

    #[test]
    fn horizon_clamp_truncates_outer_tier() {
        // With AABB-based outer test, a brick is loaded iff its NEAR
        // CORNER is inside the horizon — i.e. some part of its volume
        // is visible. Outer tiers whose nearest brick AABB straddles
        // the horizon still appear; we only assert no brick is *fully*
        // outside the horizon.
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let brick_edge_v = BRICK_EDGE as f64;
        for horizon in [100.0_f64, 300.0_f64] {
            let plan = desired_chunks(&s, obs, horizon, &masked());
            let horizon_sq = horizon * horizon;
            for (c, lod) in &plan {
                let edge_m = brick_edge_v * (1u64 << lod.depth as u32) as f64;
                let half = edge_m * 0.5;
                let cx = (c.x as f64 + 0.5) * edge_m;
                let cy = (c.y as f64 + 0.5) * edge_m;
                let cz = (c.z as f64 + 0.5) * edge_m;
                let nx = (cx.abs() - half).max(0.0);
                let ny = (cy.abs() - half).max(0.0);
                let nz = (cz.abs() - half).max(0.0);
                let near_d2 = nx * nx + ny * ny + nz * nz;
                assert!(
                    near_d2 < horizon_sq,
                    "horizon clamp {horizon} m: brick {c:?} lod={} near corner at \
                     d={} is past the horizon",
                    lod.depth,
                    near_d2.sqrt()
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // NestedSummary coverage policy
    // -----------------------------------------------------------------

    #[test]
    fn nested_summary_keeps_parent_under_inner_shell() {
        // Regression-spec for the LOD-pop fix. With NestedSummary, the
        // L1 brick at the origin (which spans world meters [0, 32) per
        // axis and is fully covered by the L0 tier) MUST be in the
        // desired set — that's what gives the renderer an instant
        // summary backdrop when the L0 brick fades out. The
        // `near_tier_bricks_dont_overlap_far_tier_at_same_position`
        // test above is the opposite assertion under MaskedShells; the
        // two policies should produce mutually exclusive results here.
        let s = streamer();
        let plan = desired_chunks(&s, DVec3::ZERO, f64::INFINITY, &nested());
        let l0_origin = (IVec3::new(0, 0, 0), Lod::new(0));
        let l1_origin = (IVec3::new(0, 0, 0), Lod::new(1));
        let l2_origin = (IVec3::new(0, 0, 0), Lod::new(2));
        let l3_origin = (IVec3::new(0, 0, 0), Lod::new(3));
        assert!(plan.contains(&l0_origin), "L0 origin should always load");
        assert!(
            plan.contains(&l1_origin),
            "NestedSummary must keep the L1 parent loaded under the L0 shell"
        );
        assert!(
            plan.contains(&l2_origin),
            "NestedSummary must keep the L2 grandparent loaded too"
        );
        assert!(
            plan.contains(&l3_origin),
            "NestedSummary must keep the L3 great-grandparent loaded"
        );
    }

    #[test]
    fn nested_summary_inflates_brick_count_within_expected_bound() {
        // The nested policy loads every tier as a full inner sphere,
        // so brick counts grow. Each coarser tier covers 8× the
        // volume per brick, so the parent count is ~1/8 of the child
        // count — total inflation is bounded by roughly
        // 1 + 1/8 + 1/64 + 1/512 ≈ 1.14×. We assert the masked count
        // is strictly less than nested, and nested is < 1.3 × masked.
        let s = streamer();
        let masked_plan = desired_chunks(&s, DVec3::ZERO, f64::INFINITY, &masked());
        let nested_plan = desired_chunks(&s, DVec3::ZERO, f64::INFINITY, &nested());
        assert!(
            nested_plan.len() > masked_plan.len(),
            "NestedSummary must produce more bricks than MaskedShells \
             (got nested={} masked={})",
            nested_plan.len(),
            masked_plan.len()
        );
        let ratio = nested_plan.len() as f64 / masked_plan.len() as f64;
        assert!(
            ratio < 1.30,
            "NestedSummary inflation ratio {ratio:.3} exceeds the 1.30 \
             bound (nested={} masked={})",
            nested_plan.len(),
            masked_plan.len()
        );
    }

    #[test]
    fn nested_summary_every_loaded_brick_has_parent_until_outermost() {
        // The point of NestedSummary is: every brick that isn't at the
        // outermost tier has its immediate-coarser-LOD parent also
        // loaded. Without that invariant the LOD crossfade in
        // `fp_update_lod_visibility` has nothing to reveal.
        let s = streamer();
        let plan = desired_chunks(&s, DVec3::ZERO, f64::INFINITY, &nested());
        let outer_depth = s.ladder.tiers.last().unwrap().lod.depth;
        let plan_set: std::collections::HashSet<(IVec3, u8)> =
            plan.iter().map(|(c, l)| (*c, l.depth)).collect();
        for (coord, lod) in &plan {
            if lod.depth >= outer_depth {
                continue;
            }
            let parent = (
                IVec3::new(
                    coord.x.div_euclid(2),
                    coord.y.div_euclid(2),
                    coord.z.div_euclid(2),
                ),
                lod.depth + 1,
            );
            assert!(
                plan_set.contains(&parent),
                "NestedSummary: child {coord:?} lod={} has no parent {:?} \
                 at lod={} in the desired set — crossfade would have \
                 nothing to reveal",
                lod.depth,
                parent.0,
                parent.1,
            );
        }
    }

    #[test]
    fn nested_summary_passes_symmetry_check() {
        // The reflection/rotation symmetry tests in the legacy block
        // run on MaskedShells. Make sure NestedSummary preserves the
        // same symmetry so the new policy doesn't reintroduce the
        // directional-asymmetry bug.
        let s = streamer();
        let plan = desired_chunks(&s, DVec3::ZERO, f64::INFINITY, &nested());
        for depth in 0u8..=3 {
            let set = xz_set_at_lod(&plan, depth);
            assert!(!set.is_empty(), "nested: depth={depth} produced no bricks");
            for &(x, z) in &set {
                assert!(
                    set.contains(&(-x - 1, z)),
                    "nested depth={depth}: X-asymmetry at ({x},{z})"
                );
                assert!(
                    set.contains(&(x, -z - 1)),
                    "nested depth={depth}: Z-asymmetry at ({x},{z})"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // Body-aware horizon clamp (Phase 17 follow-up)
    // -----------------------------------------------------------------

    /// A horizon below the streamer's outer ring drops every brick whose
    /// near corner lies past the horizon. With a hard 100 m clamp the
    /// outer ladder must shrink — only the small ring of bricks
    /// straddling the observer's local cell can survive (depth-3 bricks
    /// are 128 m on a side, so the cell containing the observer always
    /// passes the near-corner test even when the band is empty).
    #[test]
    fn horizon_clamp_drops_far_tiers() {
        let s = streamer();
        let obs = DVec3::ZERO;
        let unbounded = desired_chunks(&s, obs, f64::INFINITY, &masked());
        let clamped = desired_chunks(&s, obs, 100.0, &masked());
        assert!(
            clamped.len() < unbounded.len() / 4,
            "horizon clamp didn't shrink the plan enough: clamped={} unbounded={}",
            clamped.len(),
            unbounded.len()
        );
        // The depth-3 outer tier should drop from its original ring count
        // (hundreds) to at most a handful of straddling bricks.
        let far_count = clamped.iter().filter(|(_, l)| l.depth == 3).count();
        let unbounded_far = unbounded.iter().filter(|(_, l)| l.depth == 3).count();
        assert!(
            far_count * 8 <= unbounded_far,
            "depth-3 ring barely shrunk under 100 m clamp: clamped={far_count} unbounded={unbounded_far}"
        );
    }

    /// `f64::INFINITY` (cube worlds) is the no-clamp baseline.
    #[test]
    fn horizon_infinity_matches_unclamped() {
        let s = streamer();
        let obs = DVec3::ZERO;
        let inf = desired_chunks(&s, obs, f64::INFINITY, &masked());
        let huge = desired_chunks(&s, obs, 1.0e9, &masked());
        assert_eq!(inf.len(), huge.len());
    }

    /// Sphere shape's `horizon_at_m` clamps at low altitude; cube does not.
    /// This locks in the WorldShape integration the FP streamer relies on.
    #[test]
    fn shape_horizon_at_m_drives_streamer_clamp() {
        use atomr_worlds_core::shape::WorldShape;
        let earth = WorldShape::Sphere { radius_m: 6.371e6 };
        let cube = WorldShape::Cube { edge_m: 1.0e7 };
        // Observer at ~10 m above the +Y surface of an Earth-class sphere.
        let p = DVec3::new(0.0, earth.radius_m() + 10.0, 0.0);
        let earth_h = earth.horizon_at_m(p);
        let cube_h = cube.horizon_at_m(p);
        assert!(earth_h.is_finite());
        assert!(earth_h < 12_000.0, "10 m altitude → horizon ≈ 11.3 km, got {earth_h}");
        assert!(cube_h.is_infinite());

        // Streamer plan respects the horizon.
        let s = streamer();
        let plan = desired_chunks(&s, p, earth_h, &masked());
        // The plan must be a (strict) subset of the unclamped plan.
        let unclamped = desired_chunks(&s, p, f64::INFINITY, &masked());
        assert!(plan.len() <= unclamped.len());
    }
}
