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
    /// Decoded brick (if the host had one). Not currently consumed by
    /// the streaming system — meshes already encode the rendered
    /// surface — but kept for future call-sites that need raw voxels
    /// (e.g. neighbor stitching, physics collider build).
    #[allow(dead_code)]
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
            let env = Envelope::new(
                0,
                address,
                WorldRequest::GetBrick { addr: address, brick: coord, lod },
            );
            let brick: Option<Arc<Brick>> = match host.request(env).await {
                Ok(resp) => match resp.body {
                    WorldEvent::BrickSnapshot { payload, .. } => {
                        Brick::from_bytes(&payload).ok().map(Arc::new)
                    }
                    _ => None,
                },
                Err(_) => None,
            };
            // Mesh + AO bake + DAG build on tokio's blocking pool so the reactor
            // stays free. Both the mesh and the raymarch DAG are derived from the
            // same brick here, off the main thread. Empty brick ⇒ empty meshes +
            // no DAG.
            let (meshes, dag): (std::collections::HashMap<u16, ViewMesh>, Option<DagGpuWithDigest>) =
                match brick.as_ref() {
                    Some(b) => {
                        let b = b.clone();
                        let ao = ao.clone();
                        tokio::task::spawn_blocking(move || {
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
            let _ = tx.send(BrickReady { coord, lod, brick, meshes, dag });
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
