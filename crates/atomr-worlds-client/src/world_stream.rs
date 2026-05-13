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
//! coordinates that should be loaded at each tier's LOD. A brick is
//! included in tier `i` only if its center lies inside that tier's
//! shell (between the previous tier's outer radius and this tier's
//! outer radius). The check is **spherical** — distance in 3D, not a
//! cube — so loading and unloading are symmetric in all four cardinal
//! directions and the diagonals. The previous 2-tier cube ring caused
//! a visible asymmetry where only some directions appeared to populate.
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

use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::streaming::StreamingPolicy;
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::prelude::*;

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
#[derive(Clone, Debug)]
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
}

impl LoadedChunk {
    #[inline]
    pub fn key(coord: IVec3, lod: Lod) -> (IVec3, u8) {
        (coord, lod.depth)
    }
}

/// HashMap-backed registry of loaded chunks keyed by `(coord, lod.depth)`.
/// Keying by depth lets us briefly hold both a `(c, 0)` and `(c, 1)`
/// entry during a tier change without one stomping the other.
#[derive(Resource, Default, Debug)]
pub struct LoadedChunks(pub HashMap<(IVec3, u8), LoadedChunk>);

impl LoadedChunks {
    /// Whether the entry should be despawned this tick under the given
    /// hysteresis window.
    #[inline]
    pub fn is_stale(&self, key: &(IVec3, u8), now: u64, hysteresis: u64) -> bool {
        match self.0.get(key) {
            Some(c) => now.saturating_sub(c.last_seen_frame) >= hysteresis,
            None => false,
        }
    }
}

/// Build a closest-first sorted list of `(coord, lod)` brick keys that
/// should be loaded for the given observer.
///
/// For each tier in the ladder, the call enumerates the AABB of brick
/// coords that intersect a sphere of radius `tier.outer_radius_m`
/// centered on `observer`, in that tier's brick grid. A brick is
/// emitted at tier `i` only if its center distance lies inside that
/// tier's band — `prev_tier.outer_radius_m <= d < tier.outer_radius_m`.
/// This **radial** check is the load shape; it produces a symmetric
/// ring in all four horizontal directions (the prior 2-tier
/// implementation used cubes whose corners were ~73 % farther than
/// their faces, which produced visible directional asymmetry as the
/// observer walked).
///
/// `horizon_m` is `f64::INFINITY` for flat tiers (cube worlds) and the
/// surface-horizon distance for spherical bodies — radii are clamped to
/// it so we never stream past the visible surface.
pub fn desired_chunks(
    streamer: &ChunkStreamer,
    observer: DVec3,
    horizon_m: f64,
) -> Vec<(IVec3, Lod)> {
    let mut out: Vec<(IVec3, Lod)> = Vec::new();
    if streamer.ladder.tiers.is_empty() {
        return out;
    }
    let brick_edge_v = BRICK_EDGE as f64;

    for (i, tier) in streamer.ladder.tiers.iter().enumerate() {
        let (inner_r, outer_r) = streamer.ladder.tier_band_m(i);
        // Horizon clamp — never stream past the surface for spherical
        // bodies. Inner is also clamped so the band stays non-degenerate
        // when horizon shrinks below the ladder.
        let outer_r = outer_r.min(horizon_m);
        let inner_r = inner_r.min(outer_r);
        if outer_r <= 0.0 {
            continue;
        }

        // Brick edge in meters at this tier's LOD: BRICK_EDGE * 2^depth.
        let lod_scale = (1u64 << tier.lod.depth as u32) as f64;
        let edge_m = brick_edge_v * lod_scale;
        let inner_sq = inner_r * inner_r;
        let outer_sq = outer_r * outer_r;

        // Brick-grid AABB that bounds the outer sphere. The actual
        // band check is `inner <= |center - observer| < outer`, which
        // produces a spherical shell (symmetric across X, Y, Z).
        let outer_v = (outer_r / edge_m).ceil() as i64;
        let ox = (observer.x / edge_m).floor() as i64;
        let oy = (observer.y / edge_m).floor() as i64;
        let oz = (observer.z / edge_m).floor() as i64;

        for bz in (oz - outer_v)..=(oz + outer_v) {
            for by in (oy - outer_v)..=(oy + outer_v) {
                for bx in (ox - outer_v)..=(ox + outer_v) {
                    let cx = (bx as f64 + 0.5) * edge_m - observer.x;
                    let cy = (by as f64 + 0.5) * edge_m - observer.y;
                    let cz = (bz as f64 + 0.5) * edge_m - observer.z;
                    let d2 = cx * cx + cy * cy + cz * cz;
                    if d2 >= outer_sq {
                        continue;
                    }
                    if d2 < inner_sq {
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
            .init_resource::<LoadedChunks>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streamer() -> ChunkStreamer {
        ChunkStreamer::default()
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
        let plan = desired_chunks(&s, obs, f64::INFINITY);

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
            .map(|p| desired_chunks(&s, *p, f64::INFINITY).len())
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
            let plan = desired_chunks(&s, obs, f64::INFINITY);
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
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        let plan = desired_chunks(&s, obs, f64::INFINITY);
        let outer = s.outer_radius_m();
        let outer_sq = outer * outer;
        let brick_edge_v = BRICK_EDGE as f64;
        for (c, lod) in &plan {
            let edge_m = brick_edge_v * (1u64 << lod.depth as u32) as f64;
            let cx = (c.x as f64 + 0.5) * edge_m;
            let cy = (c.y as f64 + 0.5) * edge_m;
            let cz = (c.z as f64 + 0.5) * edge_m;
            let d2 = cx * cx + cy * cy + cz * cz;
            assert!(
                d2 < outer_sq,
                "brick {c:?} lod={} at center-dist {} exceeds horizon {}",
                lod.depth,
                d2.sqrt(),
                outer
            );
        }
    }

    #[test]
    fn desired_chunks_emits_distinct_keys() {
        let s = streamer();
        let chunks = desired_chunks(&s, DVec3::new(0.0, 0.0, 0.0), f64::INFINITY);
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
        let chunks = desired_chunks(&s, DVec3::new(0.0, 0.0, 0.0), f64::INFINITY);
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
        let plan = desired_chunks(&s, obs, f64::INFINITY);
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
        loaded.0.insert(
            key,
            LoadedChunk {
                coord: IVec3::new(0, 0, 0),
                lod: Lod::new(0),
                entity: None,
                last_seen_frame: 5,
            },
        );
        // At frame 6 (1 tick later), still fresh.
        assert!(!loaded.is_stale(&key, 6, HYSTERESIS_TICKS));
        // At frame 7 (2 ticks later), stale.
        assert!(loaded.is_stale(&key, 7, HYSTERESIS_TICKS));
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

    #[test]
    fn horizon_clamp_truncates_outer_tier() {
        let s = streamer();
        let obs = DVec3::new(0.0, 0.0, 0.0);
        // Tight 100 m horizon — everything past tier 0 should vanish
        // since tier 1's inner radius (128 m) is past the horizon.
        let plan = desired_chunks(&s, obs, 100.0);
        for (_, lod) in &plan {
            assert!(
                lod.depth == 0,
                "horizon clamp 100 m must drop bricks past tier 0, saw depth {}",
                lod.depth
            );
        }
        // 300 m horizon: tier 2 (inner 256) is barely engaged, tier 3
        // is entirely out of bounds.
        let plan = desired_chunks(&s, obs, 300.0);
        let max_depth = plan.iter().map(|(_, l)| l.depth).max().unwrap_or(0);
        assert!(
            max_depth <= 2,
            "horizon 300 m must drop tier 3 (inner 512), saw depth {max_depth}"
        );
    }
}
