//! Async brick-fetch + meshing pipeline.
//!
//! The first-person streamer would otherwise block the main thread on
//! every [`LocalHostQuery::brick`](atomr_worlds_host::LocalHostQuery::brick)
//! call — `handle.block_on` per brick, hundreds of bricks during world
//! fill. The frame loop would stall for hundreds of milliseconds while
//! the actor ran the procedural generator and the renderer did greedy
//! meshing + AO bake.
//!
//! This module replaces that with a fire-and-forget pipeline:
//!
//! 1. `fp_stream_bricks` calls [`BrickGenWorkers::dispatch`] for every
//!    desired-but-not-loaded brick (capped by [`MAX_IN_FLIGHT`]).
//! 2. The dispatcher spawns a tokio task that `host.request().await`s
//!    the brick payload, decodes it, and `spawn_blocking`s the greedy
//!    mesh + AO bake on tokio's blocking pool.
//! 3. The completed [`BrickReady`] payload is sent down an `mpsc`
//!    channel that the streaming system drains on the main thread
//!    each frame and converts to Bevy entities.
//!
//! The main thread never blocks on the host or the mesher; it only
//! does the unavoidable mesh-asset upload + entity spawn. Per-frame
//! `spawn_budget` caps how many results are converted into entities
//! to prevent GPU-upload stalls during initial world fill.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::WorldHost;
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_view::{greedy_mesh_by_material, Mesh as ViewMesh};
use atomr_worlds_voxel::brick::Brick;
use atomr_worlds_voxel::{DagBrick, DagGpuWithDigest};
use bevy::prelude::*;
use std::sync::mpsc;
use tokio::runtime::Handle;

use crate::render::AoStrategy;

/// Cap on simultaneously in-flight brick fetches. Keeps task / memory
/// churn bounded during the initial world fill (~8 k brick keys) while
/// still giving the actor pool enough concurrent work to stay busy.
pub const MAX_IN_FLIGHT: usize = 64;

/// Cap on entities spawned per frame from the result queue. Mesh-asset
/// upload is real GPU work; converting hundreds in one frame produces
/// a visible hitch. 24/frame is roughly one brick per ms at 30 fps.
pub const DEFAULT_SPAWN_BUDGET: usize = 24;

/// Brick key — `(coord, lod.depth)` — matching the convention used by
/// [`crate::world_stream::LoadedChunk::key`].
pub type Key = (IVec3, u8);

/// Result of a successful (or empty / missing) async brick fetch +
/// mesh. `brick` and `meshes` are both empty for cells the host says
/// don't exist; callers still record the loaded entry so the streamer
/// doesn't redispatch every frame.
#[derive(Debug)]
pub struct BrickReady {
    pub coord: IVec3,
    pub lod: Lod,
    /// Decoded brick (if the host had one). Consumed by the streaming
    /// system to populate [`crate::world_stream::LoadedChunk::brick`] for
    /// LOD-0 chunks, which powers the client voxel picker / brush refresh
    /// (`crate::modes::edit`). `None` for empty / missing bricks.
    pub brick: Option<Arc<Brick>>,
    pub meshes: std::collections::HashMap<u16, ViewMesh>,
    /// Flattened DAG + content digest + occupancy AABB for the raymarch path,
    /// built on the blocking pool alongside the mesh (so `spawn_brick_entity`
    /// never builds a DAG inline). Built unconditionally — strategy-agnostic, so
    /// a mid-run swap to/from raymarch needs no main-thread rebuild. `None` for
    /// empty/missing bricks.
    pub dag: Option<DagGpuWithDigest>,
}

/// Bevy resource owning the dispatcher half of the async pipeline.
///
/// Holds the in-flight set + tx side of the result channel. Wrapped in
/// an `Arc<Mutex<...>>` so the streaming system can read/mutate it
/// without holding an exclusive lock across `await`s (the dispatch
/// itself is fire-and-forget — the lock is only held for the
/// `in_flight.insert` + `handle.spawn` pair).
#[derive(Resource)]
pub struct BrickGenWorkers {
    pub host: Arc<dyn WorldHost>,
    pub handle: Handle,
    pub ao: Arc<dyn AoStrategy>,
    in_flight: HashSet<Key>,
    results_tx: mpsc::Sender<BrickReady>,
    /// Receiver lives behind a `Mutex` so [`BrickGenWorkers`] can sit
    /// in a `Res<…>` slot (single shared resource) while the
    /// streaming system uses `try_recv` on `&self`. Uncontended in
    /// practice — only the main thread drains.
    pub results_rx: Mutex<mpsc::Receiver<BrickReady>>,
}

impl BrickGenWorkers {
    pub fn new(host: Arc<dyn WorldHost>, handle: Handle, ao: Arc<dyn AoStrategy>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            host,
            handle,
            ao,
            in_flight: HashSet::new(),
            results_tx: tx,
            results_rx: Mutex::new(rx),
        }
    }

    /// Number of brick fetches currently outstanding. Useful for HUD
    /// diagnostics / scenario assertions.
    #[inline]
    #[allow(dead_code)]
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Whether the cap on simultaneous dispatches has been reached.
    #[inline]
    pub fn is_saturated(&self) -> bool {
        self.in_flight.len() >= MAX_IN_FLIGHT
    }

    /// Best-effort dispatch. Returns:
    /// - `false` if the dispatcher is saturated, the key was already
    ///   in flight, or anything else prevented the spawn.
    /// - `true` if a new tokio task was spawned.
    pub fn dispatch(&mut self, addr: WorldAddr, coord: IVec3, lod: Lod) -> bool {
        if self.is_saturated() {
            return false;
        }
        let key: Key = (coord, lod.depth);
        if !self.in_flight.insert(key) {
            return false;
        }
        let host = self.host.clone();
        let tx = self.results_tx.clone();
        let ao = self.ao.clone();
        let address: Address = addr.into();
        self.handle.spawn(async move {
            let ready = fetch_and_build(host, ao, address, coord, lod).await;
            let _ = tx.send(ready);
        });
        true
    }

    /// Drain up to `budget` finished results. The streaming system
    /// converts them into Bevy entities. Removes drained keys from
    /// `in_flight` so the next frame can dispatch their replacements.
    pub fn drain(&mut self, budget: usize) -> Vec<BrickReady> {
        let mut out = Vec::with_capacity(budget);
        let rx = self.results_rx.lock().expect("results_rx poisoned");
        for _ in 0..budget {
            match rx.try_recv() {
                Ok(ready) => {
                    self.in_flight.remove(&(ready.coord, ready.lod.depth));
                    out.push(ready);
                }
                Err(_) => break,
            }
        }
        out
    }

    /// Whether a brick key is currently in flight. Used by the
    /// streaming system to skip re-dispatching.
    #[inline]
    pub fn contains(&self, key: &Key) -> bool {
        self.in_flight.contains(key)
    }
}

/// Shared off-thread brick-refresh queue: an in-flight dedup set, a result
/// channel, and a "dirtied-again" pending set. Used by both the edit
/// ([`crate::modes::edit_workers::EditApplyWorkers`]) and fracture
/// ([`crate::physics::fracture`]) write-back paths so the dedup + redispatch
/// logic lives in one place.
///
/// # The dirtied-again problem
///
/// A refresh = re-fetch authoritative bytes + remesh, which takes longer than a
/// frame. If a second write lands on a brick whose refresh is still in flight,
/// naively deduping it (skip — already in flight) leaves the rendered mesh stuck
/// at the *first* write: the in-flight refetch was issued before the second
/// write, so its `BrickReady` reflects stale geometry, and nothing re-meshes the
/// brick afterward. [`claim`](Self::claim) instead marks such a brick `pending`,
/// and [`drain`](Self::drain) reports it for a re-refetch against the now-latest
/// host state once the first refresh lands.
pub struct BrickRefreshQueue {
    in_flight: HashSet<Key>,
    pending: HashSet<Key>,
    tx: mpsc::Sender<BrickReady>,
    rx: Mutex<mpsc::Receiver<BrickReady>>,
}

impl Default for BrickRefreshQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickRefreshQueue {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { in_flight: HashSet::new(), pending: HashSet::new(), tx, rx: Mutex::new(rx) }
    }

    /// Claim `keys` for refresh. Returns the keys the caller should refetch
    /// (those not already in flight); keys already in flight are marked
    /// `pending` so they get re-refetched when their current refresh lands.
    pub fn claim<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> Vec<Key> {
        let mut fresh = Vec::new();
        for k in keys {
            if self.in_flight.insert(k) {
                fresh.push(k);
            } else {
                self.pending.insert(k);
            }
        }
        fresh
    }

    /// A sender for the caller to hand finished `BrickReady`s back on.
    pub fn sender(&self) -> mpsc::Sender<BrickReady> {
        self.tx.clone()
    }

    /// Drain finished refreshes. Returns `(ready, redo)` — `redo` lists bricks
    /// that were dirtied again while their refresh was in flight (kept claimed),
    /// which the owner should re-refetch against the now-latest host state.
    pub fn drain(&mut self) -> (Vec<BrickReady>, Vec<Key>) {
        let mut out = Vec::new();
        let mut redo = Vec::new();
        let rx = self.rx.lock().expect("brick refresh rx poisoned");
        while let Ok(ready) = rx.try_recv() {
            let key = (ready.coord, ready.lod.depth);
            self.in_flight.remove(&key);
            if self.pending.remove(&key) {
                // Dirtied again while meshing — keep it claimed and re-refetch.
                self.in_flight.insert(key);
                redo.push(key);
            }
            out.push(ready);
        }
        (out, redo)
    }

    /// Number of refreshes currently outstanding (profiler gauge).
    #[inline]
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }
}

/// Refetch + remesh each brick in `keys` off-thread, sending each finished
/// [`BrickReady`] down `tx`. No write is issued — the host already holds the
/// latest writes; this just rebuilds the mesh. Used to service the `redo` keys
/// [`BrickRefreshQueue::drain`] reports.
pub async fn refetch_bricks(
    host: Arc<dyn WorldHost>,
    ao: Arc<dyn AoStrategy>,
    addr: Address,
    keys: Vec<Key>,
    tx: mpsc::Sender<BrickReady>,
) {
    for (coord, _) in keys {
        let ready = fetch_and_build(host.clone(), ao.clone(), addr, coord, Lod::new(0)).await;
        let _ = tx.send(ready);
    }
}

/// Fetch one brick from the host, decode it, and build *both* render
/// representations off the calling task: the per-material greedy mesh (with
/// AO baked by `ao`) and the flattened DAG + content digest for the raymarch
/// path. Empty / missing bricks yield empty meshes and no DAG.
///
/// This is the shared core of the streaming pipeline (`dispatch` spawns it on
/// the tokio reactor) and the client edit refresh (`crate::modes::edit` runs
/// it via `block_on` to rebuild exactly the bricks an edit touched). Keeping a
/// single implementation guarantees a streamed brick and an edited brick are
/// built identically — same mesher, same AO, same DAG digest, so the
/// `DagBufferCache` dedups across both paths.
///
/// The mesh + AO + DAG build runs on tokio's blocking pool so it never stalls
/// the reactor; callers driving this from a synchronous context (the edit
/// refresh) must `block_on` it from a thread that is *not* a runtime worker.
#[cfg_attr(
    feature = "profiling",
    tracing::instrument(name = "fetch_and_build", skip_all, fields(depth = lod.depth))
)]
pub async fn fetch_and_build(
    host: Arc<dyn WorldHost>,
    ao: Arc<dyn AoStrategy>,
    addr: Address,
    coord: IVec3,
    lod: Lod,
) -> BrickReady {
    let env = Envelope::new(0, addr, WorldRequest::GetBrick { addr, brick: coord, lod });
    let brick: Option<Arc<Brick>> = match host.request(env).await {
        Ok(resp) => match resp.body {
            WorldEvent::BrickSnapshot { payload, .. } => Brick::from_bytes(&payload).ok().map(Arc::new),
            _ => None,
        },
        Err(_) => None,
    };
    let (meshes, dag): (std::collections::HashMap<u16, ViewMesh>, Option<DagGpuWithDigest>) =
        match brick.as_ref() {
            Some(b) => {
                let b = b.clone();
                let ao = ao.clone();
                tokio::task::spawn_blocking(move || {
                    #[cfg(feature = "profiling")]
                    let _z = tracing::info_span!("brick_mesh_ao_dag").entered();
                    let mut by_mat = greedy_mesh_by_material(&b);
                    for sub in by_mat.values_mut() {
                        ao.bake(sub, &b);
                    }
                    let dag = DagBrick::from_brick(&b).to_gpu_with_digest(&b);
                    (by_mat, dag)
                })
                .await
                .unwrap_or_default()
            }
            None => Default::default(),
        };
    BrickReady { coord, lod, brick, meshes, dag }
}

/// Bevy plugin: registers the [`BrickGenWorkers`] resource. The
/// concrete instance is constructed lazily once [`crate::world_runtime::WorldRuntime`]
/// is available so the host + tokio handle are real.
pub struct BrickGenPlugin;

impl Plugin for BrickGenPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_brick_gen_workers);
    }
}

fn init_brick_gen_workers(
    mut commands: Commands,
    runtime: Res<crate::world_runtime::WorldRuntime>,
    render_cfg: Res<crate::render::RenderConfig>,
) {
    let workers = BrickGenWorkers::new(
        runtime.host.clone(),
        runtime.runtime.handle().clone(),
        render_cfg.ao.clone(),
    );
    commands.insert_resource(workers);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty-test stub: just confirms the `Key` type ergonomics match
    /// `LoadedChunk::key`'s tuple shape.
    #[test]
    fn key_is_coord_and_depth() {
        let k: Key = (IVec3::new(1, 2, 3), 0);
        assert_eq!(k.0.x, 1);
        assert_eq!(k.1, 0);
    }
}
