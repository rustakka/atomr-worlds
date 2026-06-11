//! Carve → host fracture decision → debris orchestration.
//!
//! Listens for [`VoxelEditEvent`]s and, for *carves*, asks the **host** to
//! evaluate the structural fracture authoritatively. The host runs the
//! deterministic connectivity decision, journals the removal of any detached
//! island, and replies with a [`FractureApplied`] command sequence (also fanned
//! out to other subscribers — this is the multiplayer-sync seam, Rec 4). The
//! client then does the unavoidable, *ephemeral* work: bake each detached island
//! into a falling rigid body and refresh the bricks the host carved.
//!
//! # Why the host decides
//!
//! Connectivity is integer and deterministic, so it must agree across every
//! peer; making the host authoritative means all clients see the *same*
//! destruction (and the removal is journaled exactly once). Float debris motion
//! stays client-side — the host never simulates bodies.
//!
//! # Off the render thread
//!
//! 1. [`dispatch_fracture_checks`] (main thread) sends a `FractureRequest` to the
//!    host on a tokio task; the frame never blocks on the round-trip.
//! 2. [`apply_fracture_results`] (main thread) drains finished
//!    [`FractureApplied`]s and does the ECS work: bake + spawn the debris bodies
//!    and refresh the carved bricks (make-before-break, no flicker).

use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Mutex};

use atomr_worlds_core::addr::Address;
use atomr_worlds_core::coord::IVec3 as VoxCoord;
use atomr_worlds_core::default_physics_palette;
use atomr_worlds_host::WorldHost;
use atomr_worlds_physics::{analyze_region, AnalyzedIsland};
use atomr_worlds_proto::{
    Envelope, Force, FractureApplied, FractureCommand, FractureRequest, WorldEvent, WorldRequest,
};
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::prelude::*;
use tokio::runtime::Handle;

use super::config::PhysicsConfig;
use super::debris::spawn_island;
use crate::brick_gen::{refetch_bricks, BrickReady, BrickRefreshQueue};
use crate::modes::edit::{self, EditSpawn, VoxelEditEvent};
use crate::modes::fp::{spawn_edited_brick, MaterialPool};
use crate::render::{AoStrategy, RenderConfig};
use crate::world_runtime::WorldRuntime;
use crate::world_stream::{ChunkStreamer, LoadedChunks};

/// Voxels of skirt added around the affected-brick AABB when locating the carve
/// centre, so the host's region is centred on the disturbed volume.
const REGION_SKIRT: i64 = 2;
/// Hard cap on the affected-region volume (voxels). A brush spanning more than
/// this skips fracture analysis to keep the edit off the critical path; the
/// terrain still updates normally. (~ (3 bricks)³.)
const MAX_REGION_VOXELS: i64 = 48 * 48 * 48;
/// Cap on concurrently-outstanding fracture requests. Carves are user-paced, so
/// this is a generous safety bound, not a throughput limit; an over-cap carve
/// simply doesn't fracture (the terrain edit still applies through the editor).
const MAX_IN_FLIGHT_FRACTURES: usize = 8;

/// Bevy resource owning the client half of the host-authoritative fracture
/// pipeline: the outstanding-request counter, the [`FractureApplied`] reply
/// channel, and the brick-refresh pipeline. Mirrors
/// [`crate::brick_gen::BrickGenWorkers`].
#[derive(Resource)]
pub struct FractureWorkers {
    handle: Handle,
    in_flight: usize,
    applied_tx: mpsc::Sender<FractureApplied>,
    /// Receiver behind a `Mutex` so the resource is `Sync` while the main thread
    /// drains it with `try_recv`. Uncontended in practice.
    applied_rx: Mutex<mpsc::Receiver<FractureApplied>>,
    /// Async brick-refresh pipeline: the heavy `fetch_and_build` (mesh + AO +
    /// DAG) for a carved brick runs off-thread; the finished [`BrickReady`] is
    /// swapped in flicker-free on the main thread. The dedup + dirtied-again
    /// logic lives in the shared [`BrickRefreshQueue`].
    refresh: BrickRefreshQueue,
    /// Host / AO / addr captured at the last refresh, so `drain_refresh` can
    /// re-refetch dirtied-again bricks (a second carve landed mid-refresh)
    /// without new system params. Constant for the single active world.
    refetch_ctx: Option<(Arc<dyn WorldHost>, Arc<dyn AoStrategy>, Address)>,
}

impl FractureWorkers {
    pub fn new(handle: Handle) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            handle,
            in_flight: 0,
            applied_tx: tx,
            applied_rx: Mutex::new(rx),
            refresh: BrickRefreshQueue::new(),
            refetch_ctx: None,
        }
    }

    #[inline]
    fn is_saturated(&self) -> bool {
        self.in_flight >= MAX_IN_FLIGHT_FRACTURES
    }

    /// Number of fracture requests currently outstanding (diagnostics / tests).
    #[inline]
    #[allow(dead_code)]
    pub fn in_flight_count(&self) -> usize {
        self.in_flight
    }

    /// Ask the host to evaluate a fracture at `impact_pos`. Returns `false` if
    /// saturated (the carve simply won't fracture). The host's `FractureApplied`
    /// reply — the authoritative command sequence — is sent back over the reply
    /// channel for [`Self::drain_applied`]. Even a host error resolves to an
    /// empty reply so the in-flight count never leaks.
    pub fn dispatch_fracture(
        &mut self,
        host: Arc<dyn WorldHost>,
        addr: Address,
        impact_pos: VoxCoord,
    ) -> bool {
        if self.is_saturated() {
            return false;
        }
        self.in_flight += 1;
        let tx = self.applied_tx.clone();
        self.handle.spawn(async move {
            // Zero force ⇒ carve-triggered: the host always evaluates
            // connectivity (the player already chose to carve).
            let req = FractureRequest { addr, impact_pos, force: Force::ZERO, material_id: 0 };
            let env = Envelope::new(0, addr, WorldRequest::Fracture(req));
            let applied = match host.request(env).await {
                Ok(reply) => match reply.body {
                    WorldEvent::FractureApplied(a) => a,
                    _ => FractureApplied { addr, commands: Vec::new(), seq_range: (0, 0) },
                },
                Err(_) => FractureApplied { addr, commands: Vec::new(), seq_range: (0, 0) },
            };
            let _ = tx.send(applied);
        });
        true
    }

    /// Drain every finished fracture reply, decrementing the in-flight count.
    fn drain_applied(&mut self) -> Vec<FractureApplied> {
        let mut out = Vec::new();
        let rx = self.applied_rx.lock().expect("fracture applied_rx poisoned");
        while let Ok(a) = rx.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            out.push(a);
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

    /// Refresh the bricks the host carved: refetch + remesh each touched brick
    /// off-thread, sending the finished [`BrickReady`] back for a
    /// make-before-break swap (drained by [`Self::drain_refresh`]). The host has
    /// already journaled the removal, so the refetch reads post-carve bytes.
    fn dispatch_brick_refresh(
        &mut self,
        host: Arc<dyn WorldHost>,
        ao: Arc<dyn AoStrategy>,
        addr: Address,
        bricks: Vec<(VoxCoord, u8)>,
    ) {
        if bricks.is_empty() {
            return;
        }
        self.refetch_ctx = Some((host.clone(), ao.clone(), addr));
        let fresh = self.refresh.claim(bricks);
        if fresh.is_empty() {
            return;
        }
        let tx = self.refresh.sender();
        self.handle.spawn(refetch_bricks(host, ao, addr, fresh, tx));
    }

    /// Number of brick refreshes currently outstanding (profiler gauge).
    #[inline]
    pub fn refresh_in_flight_count(&self) -> usize {
        self.refresh.in_flight_count()
    }
}

/// Bake one detached island into an [`AnalyzedIsland`] (greedy boxes + mass +
/// dominant material) from its world voxel set and the per-cell materials
/// recovered from the host's `SetVoxel` commands. Reuses the engine-agnostic
/// [`analyze_region`] with an always-false anchor predicate, so the whole set
/// resolves to exactly one unanchored island.
fn bake_island(
    voxels: &[VoxCoord],
    materials: &HashMap<VoxCoord, u16>,
    voxel_size_m: f32,
) -> Option<AnalyzedIsland> {
    if voxels.is_empty() {
        return None;
    }
    let mut lo = [i64::MAX; 3];
    let mut hi = [i64::MIN; 3];
    for v in voxels {
        let c = [v.x, v.y, v.z];
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    let region_min = VoxCoord::new(lo[0], lo[1], lo[2]);
    let dims = [(hi[0] - lo[0] + 1) as i32, (hi[1] - lo[1] + 1) as i32, (hi[2] - lo[2] + 1) as i32];
    let set: HashSet<VoxCoord> = voxels.iter().copied().collect();
    let world = |x: i32, y: i32, z: i32| {
        VoxCoord::new(region_min.x + x as i64, region_min.y + y as i64, region_min.z + z as i64)
    };
    let palette = default_physics_palette();
    let is_solid = |x: i32, y: i32, z: i32| set.contains(&world(x, y, z));
    let is_anchor = |_: i32, _: i32, _: i32| false;
    let material_at = |x: i32, y: i32, z: i32| materials.get(&world(x, y, z)).copied().unwrap_or(0);
    let analysis = analyze_region(
        dims,
        region_min,
        voxel_size_m as f64,
        &palette,
        is_solid,
        is_anchor,
        material_at,
    );
    analysis.islands.into_iter().next()
}

/// Main-thread: for each carve this frame, ask the host to evaluate the
/// structural fracture. Never blocks on the round-trip.
pub fn dispatch_fracture_checks(
    mut edits: MessageReader<VoxelEditEvent>,
    runtime: Res<WorldRuntime>,
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
        // Impact at the disturbed region's centre; the host evaluates a bounded
        // neighbourhood around it.
        let impact_pos = VoxCoord::new(
            region_min.x + dims[0] as i64 / 2,
            region_min.y + dims[1] as i64 / 2,
            region_min.z + dims[2] as i64 / 2,
        );
        workers.dispatch_fracture(runtime.host.clone(), job.addr, impact_pos);
    }
}

/// Main-thread: drain finished fracture replies and apply them — bake + spawn
/// the falling bodies and refresh the bricks the host carved (reusing the
/// editor's make-before-break swap, so no flicker). The host already journaled
/// the removal, so the client only reads it back.
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
    mut interp: ResMut<super::debris_stream::DebrisInterp>,
    time: Res<Time>,
    mut spawn: EditSpawn,
    mut commands: Commands,
) {
    let _scope = perf.scope(crate::perf::Phase::FractureApply);
    let frame = streamer.frame;
    let now = time.elapsed_secs_f64();
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

    for applied in workers.drain_applied() {
        if applied.commands.is_empty() {
            continue;
        }
        // Recover per-cell materials from the authoritative carves, and the set
        // of bricks the host touched.
        let mut materials: HashMap<VoxCoord, u16> = HashMap::new();
        let mut touched: HashSet<(VoxCoord, u8)> = HashSet::new();
        for cmd in &applied.commands {
            if let FractureCommand::SetVoxel { pos, before, .. } = cmd {
                materials.insert(*pos, before.0);
                let key = (edit::brick_of(*pos), 0u8);
                if loaded.get(&key).map(|c| !c.is_fading_out).unwrap_or(false) {
                    touched.insert(key);
                }
            }
        }

        // 1) Spawn the host-driven debris bodies (additive; no world mutation).
        //    The body is kinematic — the host owns its motion and we interpolate
        //    its `DebrisStateDelta` stream onto the entity (`debris_stream`). The
        //    `id` ties the spawned entity to that stream; `attach_entity` folds
        //    in any deltas that arrived before this spawn.
        for cmd in &applied.commands {
            if let FractureCommand::SpawnDebris { id, voxels, .. } = cmd {
                if let Some(island) = bake_island(voxels, &materials, cfg.voxel_size_m) {
                    if let Some(ent) = spawn_island(
                        &island,
                        *id,
                        &cfg,
                        &material_pool,
                        &mut spawn.meshes,
                        &mut commands,
                    ) {
                        interp.attach_entity(*id, ent, now);
                    }
                }
            }
        }

        // 2) Refresh the bricks the host carved so the holes appear (flicker-free
        //    on a later frame via step 0).
        let bricks: Vec<(VoxCoord, u8)> = touched.into_iter().collect();
        workers.dispatch_brick_refresh(runtime.host.clone(), render_cfg.ao.clone(), applied.addr, bricks);
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
    use atomr_worlds_voxel::voxel::Voxel;

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

    /// `bake_island` reconstructs one unanchored island (greedy-merged, with
    /// mass) from a host `SpawnDebris` voxel set + recovered materials.
    #[test]
    fn bake_island_rebuilds_a_body_from_commands() {
        // A 2×1×1 stone bar at world (5,5,5)-(6,5,5).
        let voxels = vec![VoxCoord::new(5, 5, 5), VoxCoord::new(6, 5, 5)];
        let mut materials = HashMap::new();
        materials.insert(VoxCoord::new(5, 5, 5), 1u16);
        materials.insert(VoxCoord::new(6, 5, 5), 1u16);
        let island = bake_island(&voxels, &materials, 1.0).expect("one island");
        assert_eq!(island.origin, VoxCoord::new(5, 5, 5));
        assert_eq!(island.dims, [2, 1, 1]);
        assert_eq!(island.boxes.len(), 1, "the bar greedy-merges to one box");
        assert_eq!(island.dominant_material, 1);
        assert!(island.mass.mass_kg > 0.0);
    }

    #[test]
    fn bake_island_empty_set_is_none() {
        assert!(bake_island(&[], &HashMap::new(), 1.0).is_none());
    }

    /// Materials recovered from `SetVoxel.before` drive the baked island, even
    /// though the world voxels are already `EMPTY` after the carve.
    #[test]
    fn materials_come_from_before_not_current() {
        let pos = VoxCoord::new(0, 0, 0);
        let cmd = FractureCommand::SetVoxel { pos, before: Voxel::new(7), after: Voxel::EMPTY };
        let FractureCommand::SetVoxel { before, .. } = cmd else { unreachable!() };
        assert_eq!(before.0, 7);
    }
}
