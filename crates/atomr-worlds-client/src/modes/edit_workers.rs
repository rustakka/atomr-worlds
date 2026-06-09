//! Off-thread voxel-edit write-back — the editor's twin of the streamer's
//! [`BrickGenWorkers`](crate::brick_gen::BrickGenWorkers) and the fracture
//! pipeline's refresh channel.
//!
//! # Why this exists
//!
//! `fp_edit_voxels` used to apply an edit on the **render thread**:
//! `block_on(WriteVoxel/WriteRegion)` then `block_on(fetch_and_build)` per
//! affected brick (FBM gen + greedy mesh + AO bake — all synchronous). A small
//! brush over cache-cold terrain could stall the frame for ~240 ms.
//!
//! [`EditApplyWorkers`] moves all of that off the main thread. The editor does
//! only the *pure* work inline (pick, build the write `Envelope`, predict the
//! affected bricks, eager-patch the resident voxels so the same-frame fracture
//! snapshot + picker see the carve) and hands the rest here. A single tokio
//! task awaits the write — which the host journals before returning `Ack` — and
//! *then* refetches + remeshes each affected brick, so the refetch is guaranteed
//! to read post-edit bytes (the host actor's FIFO mailbox makes the write a
//! happens-before edge). Finished bricks are drained on the main thread and
//! swapped in make-before-break by `apply_edit_refreshes`.

use std::sync::Arc;

use atomr_worlds_core::addr::Address;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::WorldHost;
use atomr_worlds_proto::{Envelope, WorldRequest};
use bevy::prelude::*;
use tokio::runtime::Handle;

use crate::brick_gen::{fetch_and_build, refetch_bricks, BrickReady, BrickRefreshQueue};
use crate::render::AoStrategy;

/// Bevy resource owning the off-thread edit-refresh pipeline. The dedup +
/// channel + dirtied-again logic lives in the shared [`BrickRefreshQueue`].
#[derive(Resource)]
pub struct EditApplyWorkers {
    handle: Handle,
    queue: BrickRefreshQueue,
    /// Host / AO / addr captured at the last dispatch, so [`Self::drain_refresh`]
    /// can re-refetch dirtied-again bricks without threading them through the
    /// drain system's params. They are constant for the single active world.
    ctx: Option<(Arc<dyn WorldHost>, Arc<dyn AoStrategy>, Address)>,
}

impl EditApplyWorkers {
    pub fn new(handle: Handle) -> Self {
        Self { handle, queue: BrickRefreshQueue::new(), ctx: None }
    }

    /// One off-thread task: await `write` (the host journals it before returning
    /// `Ack`), THEN refetch + remesh each brick in `keys`, sending each finished
    /// [`BrickReady`] back for a make-before-break swap. The await-before-refetch
    /// ordering + the host's FIFO mailbox guarantee the refetch reads post-edit
    /// bytes — never a stale pre-edit snapshot. `keys` is deduped via the shared
    /// queue; a brick whose refresh is already in flight is marked pending and
    /// re-refetched on drain (so a rapid second edit isn't lost).
    pub fn dispatch_edit(
        &mut self,
        host: Arc<dyn WorldHost>,
        ao: Arc<dyn AoStrategy>,
        addr: Address,
        write: Envelope<WorldRequest>,
        keys: Vec<(IVec3, u8)>,
    ) {
        self.ctx = Some((host.clone(), ao.clone(), addr));
        let fresh = self.queue.claim(keys);
        let tx = self.queue.sender();
        self.handle.spawn(async move {
            let _ = host.request(write).await;
            for (coord, _) in fresh {
                let ready = fetch_and_build(host.clone(), ao.clone(), addr, coord, Lod::new(0)).await;
                let _ = tx.send(ready);
            }
        });
    }

    /// Drain finished edit refreshes. Re-dispatches a refetch for any brick the
    /// queue reports as dirtied-again (a second edit landed while its refresh was
    /// in flight) against the now-latest host state.
    pub fn drain_refresh(&mut self) -> Vec<BrickReady> {
        let (out, redo) = self.queue.drain();
        if !redo.is_empty() {
            if let Some((host, ao, addr)) = &self.ctx {
                let tx = self.queue.sender();
                self.handle.spawn(refetch_bricks(host.clone(), ao.clone(), *addr, redo, tx));
            }
        }
        out
    }

    /// Number of brick refreshes currently outstanding (profiler gauge).
    #[inline]
    pub fn refresh_in_flight_count(&self) -> usize {
        self.queue.in_flight_count()
    }
}

/// One-shot Startup system: construct [`EditApplyWorkers`] once the tokio
/// runtime is available. Mirrors `init_brick_gen_workers` / `init_fracture_workers`.
pub fn init_edit_apply_workers(
    mut commands: Commands,
    runtime: Res<crate::world_runtime::WorldRuntime>,
) {
    commands.insert_resource(EditApplyWorkers::new(runtime.runtime.handle().clone()));
}
