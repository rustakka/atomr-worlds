//! Carve → flood-fill → debris orchestration, scheduled off the main thread.
//!
//! Listens for [`VoxelEditEvent`]s, and for *carves* runs the deterministic
//! structural flood-fill over the affected region. Any voxel island that no
//! longer reaches an anchor becomes a falling rigid body.
//!
//! # Why this is split across two systems
//!
//! The analysis — read the resident voxels, flood-fill, extract islands, greedy
//! box-merge, mass solve — is pure CPU that grows with the carved volume. Doing
//! it inline on the render thread stalled the frame on big carves. So it is
//! moved to a worker (the Rec 3 "off-thread flood-fill" scheduler lever):
//!
//! 1. [`dispatch_fracture_checks`] (main thread) snapshots the resident bricks
//!    around the carve — cheap `Arc<Brick>` refcount bumps — and hands them to
//!    [`FractureWorkers::dispatch`], which runs [`analyze_snapshot`] on tokio's
//!    blocking pool. The frame never blocks on the analysis.
//! 2. [`apply_fracture_results`] (main thread) drains finished
//!    [`FractureResult`]s each frame and does the unavoidable ECS work: spawn
//!    the debris bodies, journal-remove the island voxels through the host, and
//!    refresh the touched bricks.
//!
//! The flood-fill, mass, and box-merge live in the engine-agnostic
//! [`atomr_worlds_physics`] core ([`analyze_region`]); this file is the ECS glue
//! plus the snapshot sampler and the worker scheduler.

use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Mutex};

use atomr_worlds_core::addr::Address;
use atomr_worlds_core::coord::IVec3 as VoxCoord;
use atomr_worlds_core::default_physics_palette;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::WorldHost;
use atomr_worlds_physics::{analyze_region, FractureAnalysis};
use atomr_worlds_proto::{Envelope, WorldRequest};
use atomr_worlds_voxel::brick::Brick;
use atomr_worlds_voxel::voxel::Voxel;
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::prelude::*;
use tokio::runtime::Handle;

use super::config::PhysicsConfig;
use super::debris::spawn_island;
use crate::brick_gen::{fetch_and_build, refetch_bricks, BrickReady, BrickRefreshQueue};
use crate::modes::edit::{self, EditSpawn, VoxelEditEvent};
use crate::modes::fp::{spawn_edited_brick, MaterialPool};
use crate::render::{AoStrategy, RenderConfig};
use crate::world_runtime::WorldRuntime;
use crate::world_stream::{ChunkStreamer, LoadedChunks};

/// Voxels of skirt added around the affected-brick AABB before flood-fill, so a
/// small island poking just outside the carved brick is still captured.
const REGION_SKIRT: i64 = 2;
/// Hard cap on the analyzed region volume (voxels). A brush spanning more than
/// this skips fracture analysis to keep the edit off the critical path; the
/// terrain still updates normally. (~ (3 bricks)³.)
const MAX_REGION_VOXELS: i64 = 48 * 48 * 48;
/// Cap on concurrently-analyzing carves. Carves are user-paced, so this is a
/// generous safety bound, not a throughput limit; an over-cap carve simply
/// doesn't fracture (the terrain edit still applies through the editor).
const MAX_IN_FLIGHT_FRACTURES: usize = 8;

/// Monotonic id distinguishing in-flight analysis jobs.
type CarveId = u64;

/// A finished off-thread analysis, ready to apply on the main thread.
#[derive(Debug)]
struct FractureResult {
    id: CarveId,
    /// World address the carve targeted — removals are journaled against it.
    addr: Address,
    analysis: FractureAnalysis,
}

/// Bevy resource owning the dispatcher half of the off-thread fracture
/// pipeline: the in-flight set + the result channel. Mirrors
/// [`crate::brick_gen::BrickGenWorkers`].
#[derive(Resource)]
pub struct FractureWorkers {
    handle: Handle,
    in_flight: HashSet<CarveId>,
    next_id: CarveId,
    results_tx: mpsc::Sender<FractureResult>,
    /// Receiver behind a `Mutex` so the resource is `Sync` while the main
    /// thread drains it with `try_recv`. Uncontended in practice.
    results_rx: Mutex<mpsc::Receiver<FractureResult>>,
    /// Async brick-refresh pipeline: the heavy `fetch_and_build` (mesh + AO +
    /// DAG) for a carved brick runs off-thread; the finished [`BrickReady`] is
    /// swapped in flicker-free on the main thread. The dedup + dirtied-again
    /// logic lives in the shared [`BrickRefreshQueue`].
    refresh: BrickRefreshQueue,
    /// Host / AO / addr captured at the last write-back, so `drain_refresh` can
    /// re-refetch dirtied-again bricks (a second carve landed mid-refresh)
    /// without new system params. Constant for the single active world.
    refetch_ctx: Option<(Arc<dyn WorldHost>, Arc<dyn AoStrategy>, Address)>,
}

impl FractureWorkers {
    pub fn new(handle: Handle) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            handle,
            in_flight: HashSet::new(),
            next_id: 0,
            results_tx: tx,
            results_rx: Mutex::new(rx),
            refresh: BrickRefreshQueue::new(),
            refetch_ctx: None,
        }
    }

    /// Whether the in-flight cap has been reached.
    #[inline]
    fn is_saturated(&self) -> bool {
        self.in_flight.len() >= MAX_IN_FLIGHT_FRACTURES
    }

    /// Number of analyses currently outstanding (diagnostics / tests).
    #[inline]
    #[allow(dead_code)]
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Spawn an off-thread analysis of one carved region over `snapshot`.
    /// Returns `false` if saturated (the carve simply won't fracture). The
    /// snapshot is *moved* into the task, so its `Arc<Brick>`s stay alive for
    /// the analysis regardless of later eviction.
    pub fn dispatch(
        &mut self,
        addr: Address,
        region_min: VoxCoord,
        dims: [u32; 3],
        voxel_size_m: f32,
        snapshot: HashMap<VoxCoord, Arc<Brick>>,
    ) -> bool {
        if self.is_saturated() {
            return false;
        }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.in_flight.insert(id);
        let tx = self.results_tx.clone();
        self.handle.spawn_blocking(move || {
            let analysis = analyze_snapshot(region_min, dims, voxel_size_m, &snapshot);
            let _ = tx.send(FractureResult { id, addr, analysis });
        });
        true
    }

    /// Drain every finished analysis. Removes drained ids from `in_flight`.
    fn drain(&mut self) -> Vec<FractureResult> {
        let mut out = Vec::new();
        let rx = self.results_rx.lock().expect("fracture results_rx poisoned");
        while let Ok(r) = rx.try_recv() {
            self.in_flight.remove(&r.id);
            out.push(r);
        }
        out
    }

    /// Drain finished brick refreshes. Re-dispatches a refetch for any brick the
    /// queue reports as dirtied-again (a second carve landed while its refresh
    /// was in flight) against the now-latest host state.
    fn drain_refresh(&mut self) -> Vec<BrickReady> {
        let (out, redo) = self.refresh.drain();
        if !redo.is_empty() {
            if let Some((host, ao, addr)) = &self.refetch_ctx {
                let tx = self.refresh.sender();
                self.handle.spawn(refetch_bricks(host.clone(), ao.clone(), *addr, redo, tx));
            }
        }
        out
    }

    /// Off-thread carve write-back: journal-remove the island `cells` and then
    /// refetch + remesh the touched `bricks`, sending each finished
    /// [`BrickReady`] back for a make-before-break swap (drained by
    /// [`Self::drain_refresh`]).
    ///
    /// This replaces the old per-cell `block_on(WriteVoxel)` loop that ran on
    /// the **render thread** — a 100-cell island stalled the frame for 100+
    /// serial host round-trips. Now a single tokio task does all the work off
    /// the main thread. Correctness: the task awaits **every** write before any
    /// refetch, and the host actor's single FIFO mailbox makes "write completed"
    /// a happens-before edge, so each refetched brick is guaranteed to read
    /// post-carve bytes (never a stale pre-carve snapshot). `bricks` is deduped
    /// via the shared queue; a brick whose refresh is already in flight is marked
    /// pending and re-refetched on drain (so an overlapping carve isn't lost).
    fn dispatch_carve_writeback(
        &mut self,
        host: Arc<dyn WorldHost>,
        ao: Arc<dyn AoStrategy>,
        addr: Address,
        cells: Vec<VoxCoord>,
        bricks: Vec<(VoxCoord, u8)>,
    ) {
        self.refetch_ctx = Some((host.clone(), ao.clone(), addr));
        let fresh = self.refresh.claim(bricks);
        let tx = self.refresh.sender();
        self.handle.spawn(async move {
            for w in &cells {
                let env = Envelope::new(
                    0,
                    addr,
                    WorldRequest::WriteVoxel { addr, pos: *w, voxel: Voxel::EMPTY },
                );
                let _ = host.request(env).await;
            }
            for (coord, _) in fresh {
                let ready = fetch_and_build(host.clone(), ao.clone(), addr, coord, Lod::new(0)).await;
                let _ = tx.send(ready);
            }
        });
    }

    /// Number of brick refreshes currently outstanding (profiler gauge).
    #[inline]
    pub fn refresh_in_flight_count(&self) -> usize {
        self.refresh.in_flight_count()
    }
}

/// Read the resident LOD-0 voxel at world cell `c` from a snapshot, or `EMPTY`
/// when its brick isn't in the snapshot. The off-thread twin of
/// [`crate::modes::edit::sample_cell`]; both go through
/// [`crate::modes::edit::brick_of`] + [`crate::modes::edit::local_voxel`] so the
/// two samplers can't drift.
#[inline]
fn snapshot_sample(snapshot: &HashMap<VoxCoord, Arc<Brick>>, c: VoxCoord) -> Voxel {
    match snapshot.get(&edit::brick_of(c)) {
        Some(b) => edit::local_voxel(b, c),
        None => Voxel::EMPTY,
    }
}

/// Clone the resident, non-fading LOD-0 bricks overlapping `[region_min,
/// region_min + dims)` into a snapshot map (cheap `Arc` refcount bumps). Bricks
/// that aren't resident are omitted; the sampler reads them as `EMPTY`, exactly
/// as the inline path treated not-yet-streamed space.
fn snapshot_region(
    region_min: VoxCoord,
    dims: [u32; 3],
    loaded: &LoadedChunks,
) -> HashMap<VoxCoord, Arc<Brick>> {
    let bmin = edit::brick_of(region_min);
    let bmax = edit::brick_of(VoxCoord::new(
        region_min.x + dims[0] as i64 - 1,
        region_min.y + dims[1] as i64 - 1,
        region_min.z + dims[2] as i64 - 1,
    ));
    let mut snap = HashMap::new();
    for bx in bmin.x..=bmax.x {
        for by in bmin.y..=bmax.y {
            for bz in bmin.z..=bmax.z {
                let bc = VoxCoord::new(bx, by, bz);
                if let Some(chunk) = loaded.get(&(bc, 0)) {
                    if chunk.is_fading_out {
                        continue;
                    }
                    if let Some(b) = &chunk.brick {
                        snap.insert(bc, b.clone());
                    }
                }
            }
        }
    }
    snap
}

/// Run the flood-fill + island bake over a snapshot. Wires the engine-agnostic
/// [`analyze_region`] to the snapshot sampler. Pure; shared by the worker and
/// the unit tests.
fn analyze_snapshot(
    region_min: VoxCoord,
    dims: [u32; 3],
    voxel_size_m: f32,
    snapshot: &HashMap<VoxCoord, Arc<Brick>>,
) -> FractureAnalysis {
    #[cfg(feature = "profiling")]
    let _z = tracing::info_span!(
        "fracture_analyze",
        vol = dims[0] as u64 * dims[1] as u64 * dims[2] as u64
    )
    .entered();
    let [nx, ny, nz] = [dims[0] as i32, dims[1] as i32, dims[2] as i32];
    let palette = default_physics_palette();
    let world = |x: i32, y: i32, z: i32| {
        VoxCoord::new(
            region_min.x + x as i64,
            region_min.y + y as i64,
            region_min.z + z as i64,
        )
    };
    let is_solid = |x: i32, y: i32, z: i32| snapshot_sample(snapshot, world(x, y, z)).0 != 0;
    // Anchor: a solid cell on the region's outer shell *except* the top face.
    // Anything reaching the sides / bottom is treated as still attached to the
    // surrounding world; a piece only reachable upward (an overhang whose
    // support was carved) is unanchored and falls.
    let is_anchor = |x: i32, y: i32, z: i32| {
        snapshot_sample(snapshot, world(x, y, z)).0 != 0
            && (x == 0 || x == nx - 1 || z == 0 || z == nz - 1 || y == 0)
    };
    let material_at = |x: i32, y: i32, z: i32| snapshot_sample(snapshot, world(x, y, z)).0;
    analyze_region(
        [nx, ny, nz],
        region_min,
        voxel_size_m as f64,
        &palette,
        is_solid,
        is_anchor,
        material_at,
    )
}

/// Main-thread: for each carve this frame, snapshot the affected region and
/// dispatch its analysis to a worker. Never blocks on the analysis.
pub fn dispatch_fracture_checks(
    mut edits: MessageReader<VoxelEditEvent>,
    cfg: Res<PhysicsConfig>,
    loaded: Res<LoadedChunks>,
    perf: Res<crate::perf::Perf>,
    mut workers: ResMut<FractureWorkers>,
) {
    let _scope = perf.scope(crate::perf::Phase::FractureDispatch);
    // Only carves can detach structure; placements can't.
    for job in edits.read().filter(|e| e.removed) {
        let Some((region_min, dims)) = region_for_bricks(&job.bricks) else {
            continue;
        };
        let vol = dims[0] as i64 * dims[1] as i64 * dims[2] as i64;
        if vol > MAX_REGION_VOXELS {
            continue;
        }
        let snapshot = snapshot_region(region_min, dims, &loaded);
        if snapshot.is_empty() {
            continue;
        }
        workers.dispatch(job.addr, region_min, dims, cfg.voxel_size_m, snapshot);
    }
}

/// Main-thread: drain finished analyses and apply them — spawn falling bodies,
/// journal-remove the island voxels, and refresh the touched bricks (reusing
/// the editor's make-before-break swap, so no flicker).
#[allow(clippy::too_many_arguments)]
pub fn apply_fracture_results(
    cfg: Res<PhysicsConfig>,
    runtime: Res<WorldRuntime>,
    render_cfg: Res<RenderConfig>,
    streamer: Res<ChunkStreamer>,
    material_pool: Res<MaterialPool>,
    perf: Res<crate::perf::Perf>,
    mut loaded: ResMut<LoadedChunks>,
    mut workers: ResMut<FractureWorkers>,
    mut spawn: EditSpawn,
    mut commands: Commands,
) {
    let _scope = perf.scope(crate::perf::Phase::FractureApply);
    let frame = streamer.frame;
    let shading_mode = render_cfg.shading.mode();
    let raymarch_tier = render_cfg.raymarch_tier;

    // 0) Swap in any brick refreshes that finished off-thread (from earlier
    //    frames' carves). Make-before-break: the old brick stayed visible until
    //    now, so the carved hole appears with no gap or flicker. Skip bricks the
    //    streamer has since evicted / started fading (don't resurrect them).
    for ready in workers.drain_refresh() {
        let key = (ready.coord, ready.lod.depth);
        if loaded.get(&key).map(|c| c.is_fading_out).unwrap_or(true) {
            continue;
        }
        spawn_edited_brick(
            ready,
            frame,
            shading_mode,
            &spawn.pool,
            &spawn.voxel_pool,
            &spawn.res,
            raymarch_tier,
            &mut spawn.cache,
            &mut spawn.stats,
            &mut spawn.meshes,
            &mut spawn.materials,
            &mut spawn.storage_buffers,
            &mut commands,
            &mut loaded,
        );
    }

    for result in workers.drain() {
        let FractureResult { addr, analysis, .. } = result;

        // 1) Spawn the falling rigid bodies (additive; no world mutation).
        for island in &analysis.islands {
            spawn_island(island, &cfg, &material_pool, &mut spawn.meshes, &mut commands);
        }

        if analysis.cells_to_remove.is_empty() {
            continue;
        }

        // 2+3) Journal-remove the island voxels AND refresh the touched bricks
        //      entirely off the render thread (see `dispatch_carve_writeback`).
        //      Previously this was a per-cell `block_on(WriteVoxel)` loop on the
        //      main thread — a 100-cell island stalled the frame for 100+ serial
        //      host round-trips. The single off-thread task awaits every write
        //      before any refetch, so the refetch reads post-carve bytes; the
        //      finished bricks are swapped in make-before-break by step 0 on a
        //      later frame. Resident, non-fading bricks only; others self-heal
        //      on re-stream.
        let mut set: HashSet<(VoxCoord, u8)> = HashSet::new();
        for w in &analysis.cells_to_remove {
            let key = (edit::brick_of(*w), 0u8);
            if loaded.get(&key).map(|c| !c.is_fading_out).unwrap_or(false) {
                set.insert(key);
            }
        }
        let bricks: Vec<(VoxCoord, u8)> = set.into_iter().collect();
        workers.dispatch_carve_writeback(
            runtime.host.clone(),
            render_cfg.ao.clone(),
            addr,
            analysis.cells_to_remove,
            bricks,
        );
    }

    // Profiler queue gauges.
    perf.set_fracture_in_flight(workers.in_flight_count());
    perf.set_fracture_refresh_in_flight(workers.refresh_in_flight_count());
}

/// One-shot Startup system: construct [`FractureWorkers`] once the tokio runtime
/// is available. Mirrors [`crate::brick_gen`]'s `init_brick_gen_workers`.
pub fn init_fracture_workers(mut commands: Commands, runtime: Res<WorldRuntime>) {
    commands.insert_resource(FractureWorkers::new(runtime.runtime.handle().clone()));
}

/// AABB (min corner + dims in voxels) covering the affected bricks plus a skirt.
fn region_for_bricks(bricks: &[VoxCoord]) -> Option<(VoxCoord, [u32; 3])> {
    if bricks.is_empty() {
        return None;
    }
    let e = BRICK_EDGE as i64;
    let mut lo = [i64::MAX; 3];
    let mut hi = [i64::MIN; 3];
    for b in bricks {
        let bmin = [b.x * e, b.y * e, b.z * e];
        for a in 0..3 {
            lo[a] = lo[a].min(bmin[a]);
            hi[a] = hi[a].max(bmin[a] + e); // exclusive upper bound
        }
    }
    let min = VoxCoord::new(lo[0] - REGION_SKIRT, lo[1] - REGION_SKIRT, lo[2] - REGION_SKIRT);
    let dims = [
        (hi[0] - lo[0] + 2 * REGION_SKIRT) as u32,
        (hi[1] - lo[1] + 2 * REGION_SKIRT) as u32,
        (hi[2] - lo[2] + 2 * REGION_SKIRT) as u32,
    ];
    Some((min, dims))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::edit::sample_cell;
    use crate::world_stream::{LoadedChunk, LoadedChunks};

    fn brick_with(cells: &[(VoxCoord, u16)]) -> Arc<Brick> {
        let mut b = Brick::new();
        for (c, m) in cells {
            b.set(*c, Voxel::new(*m));
        }
        Arc::new(b)
    }

    fn loaded_with(coord: VoxCoord, brick: Arc<Brick>) -> LoadedChunks {
        let mut loaded = LoadedChunks::default();
        loaded.insert(
            LoadedChunk::key(coord, Lod::new(0)),
            LoadedChunk {
                coord,
                lod: Lod::new(0),
                entity: None,
                last_seen_frame: 0,
                is_fading_out: false,
                dag_digest: None,
                dag_tier: None,
                brick: Some(brick),
            },
        );
        loaded
    }

    #[test]
    fn region_covers_one_brick_plus_skirt() {
        let e = BRICK_EDGE as i64;
        let (min, dims) = region_for_bricks(&[VoxCoord::new(0, 0, 0)]).unwrap();
        assert_eq!(min, VoxCoord::new(-REGION_SKIRT, -REGION_SKIRT, -REGION_SKIRT));
        assert_eq!(dims, [(e + 2 * REGION_SKIRT) as u32; 3]);
    }

    #[test]
    fn region_spans_adjacent_bricks() {
        let e = BRICK_EDGE as i64;
        let (min, dims) =
            region_for_bricks(&[VoxCoord::new(0, 0, 0), VoxCoord::new(1, 0, 0)]).unwrap();
        assert_eq!(min.x, -REGION_SKIRT);
        assert_eq!(dims[0], (2 * e + 2 * REGION_SKIRT) as u32);
        assert_eq!(dims[1], (e + 2 * REGION_SKIRT) as u32);
    }

    #[test]
    fn empty_brick_set_has_no_region() {
        assert!(region_for_bricks(&[]).is_none());
    }

    /// The off-thread snapshot sampler agrees with the on-thread `sample_cell`
    /// for both resident and absent cells.
    #[test]
    fn snapshot_sample_matches_sample_cell() {
        let coord = VoxCoord::new(0, 0, 0);
        let brick = brick_with(&[(VoxCoord::new(5, 5, 5), 1), (VoxCoord::new(6, 5, 5), 2)]);
        let loaded = loaded_with(coord, brick);
        let snapshot = snapshot_region(VoxCoord::new(0, 0, 0), [16, 16, 16], &loaded);

        for c in [
            VoxCoord::new(5, 5, 5),  // solid
            VoxCoord::new(6, 5, 5),  // solid (other material)
            VoxCoord::new(0, 0, 0),  // empty, resident brick
            VoxCoord::new(99, 0, 0), // absent brick → EMPTY
        ] {
            assert_eq!(
                snapshot_sample(&snapshot, c).0,
                sample_cell(&loaded, c).0,
                "mismatch at {c:?}"
            );
        }
    }

    /// `snapshot_region` clones exactly the resident bricks overlapping the
    /// region and omits the rest.
    #[test]
    fn snapshot_region_collects_resident_bricks() {
        let coord = VoxCoord::new(0, 0, 0);
        let loaded = loaded_with(coord, brick_with(&[(VoxCoord::new(1, 1, 1), 1)]));
        // Region inside brick (0,0,0) only.
        let snap = snapshot_region(VoxCoord::new(0, 0, 0), [4, 4, 4], &loaded);
        assert_eq!(snap.len(), 1);
        assert!(snap.contains_key(&VoxCoord::new(0, 0, 0)));
    }

    /// An interior solid blob with no path to the region shell is detected as a
    /// floating island, with its cells flagged for removal in world space.
    #[test]
    fn interior_blob_is_an_unanchored_island() {
        let coord = VoxCoord::new(0, 0, 0);
        // 2×1×1 blob at world (5,5,5)-(6,5,5), well inside brick 0.
        let loaded = loaded_with(
            coord,
            brick_with(&[(VoxCoord::new(5, 5, 5), 1), (VoxCoord::new(6, 5, 5), 1)]),
        );
        let (region_min, dims) = region_for_bricks(&[coord]).unwrap();
        let snapshot = snapshot_region(region_min, dims, &loaded);

        let analysis = analyze_snapshot(region_min, dims, 1.0, &snapshot);
        assert_eq!(analysis.islands.len(), 1);
        let island = &analysis.islands[0];
        assert_eq!(island.origin, VoxCoord::new(5, 5, 5));
        assert_eq!(island.dims, [2, 1, 1]);
        assert_eq!(island.boxes.len(), 1);
        assert!(island.mass.mass_kg > 0.0);
        assert_eq!(
            analysis.cells_to_remove,
            vec![VoxCoord::new(5, 5, 5), VoxCoord::new(6, 5, 5)]
        );
    }

    /// `FractureWorkers` round-trips a dispatched analysis back through `drain`.
    #[test]
    fn dispatch_then_drain_round_trips() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut workers = FractureWorkers::new(rt.handle().clone());

        let coord = VoxCoord::new(0, 0, 0);
        let loaded = loaded_with(
            coord,
            brick_with(&[(VoxCoord::new(5, 5, 5), 1), (VoxCoord::new(6, 5, 5), 1)]),
        );
        let (region_min, dims) = region_for_bricks(&[coord]).unwrap();
        let snapshot = snapshot_region(region_min, dims, &loaded);

        assert!(workers.dispatch(Address::default(), region_min, dims, 1.0, snapshot));
        assert_eq!(workers.in_flight_count(), 1);

        // Poll for the worker to finish (blocking-pool task).
        let mut results = Vec::new();
        for _ in 0..400 {
            results = workers.drain();
            if !results.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(results.len(), 1, "analysis result must come back");
        assert_eq!(results[0].analysis.islands.len(), 1);
        assert_eq!(workers.in_flight_count(), 0, "drain clears in_flight");
    }
}
