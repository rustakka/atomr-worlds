//! Bridge from `atomr-worlds-host`'s tokio-flavored API to the synchronous
//! [`WorldQuery`](atomr_worlds_view::WorldQuery) the view crate expects.
//!
//! The view crate is `Send + Sync`-only and `mpsc::Receiver`-flavored
//! (std-sync) by design — it doesn't pull in a runtime. The host actor
//! exposes async-only methods (`request`, `subscribe`). We bridge them with
//! a stashed `tokio::runtime::Handle`:
//!
//! - `brick` and `ground_height_m` are one-shot: `handle.block_on(...)`.
//!   The caller must hold the [`LocalHostQuery`] from a non-async context
//!   (e.g. a render thread spawned via `std::thread::spawn`) — `block_on`
//!   inside an async task would deadlock. This matches what render-thread
//!   integrations want anyway.
//! - `subscribe_region` is streaming: spawn a tokio task that drains the
//!   host's `mpsc::Receiver<Envelope<WorldEvent>>` and forwards bodies into
//!   a `std::sync::mpsc::Sender<WorldEvent>`. The task exits when either
//!   end of the channel closes.
//!
//! Generic over [`WorldHost`] so [`LocalHost`](crate::LocalHost),
//! [`ClusterHost`](crate::ClusterHost), and `atomr-worlds-remote`'s
//! `RemoteHost` all plug in unchanged.

use std::sync::mpsc as std_mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest, AABB};
use atomr_worlds_view::WorldQuery;
use atomr_worlds_voxel::brick::Brick;
use atomr_worlds_voxel::BRICK_EDGE;
use tokio::runtime::Handle;

use crate::host::WorldHost;
use crate::local::LocalHost;

/// `WorldQuery` impl that talks to any [`WorldHost`] via a stashed tokio
/// [`Handle`]. See module docs.
pub struct LocalHostQuery {
    pub host: Arc<dyn WorldHost>,
    pub handle: Handle,
}

impl LocalHostQuery {
    /// Construct from a concrete [`LocalHost`]. Backwards-compatible with
    /// pre-Phase-15 callers (view-fp / view-tp examples).
    pub fn new(host: Arc<LocalHost>, handle: Handle) -> Self {
        Self { host: host as Arc<dyn WorldHost>, handle }
    }

    /// Construct from any [`WorldHost`] impl (LocalHost, ClusterHost,
    /// RemoteHost, …).
    pub fn from_dyn(host: Arc<dyn WorldHost>, handle: Handle) -> Self {
        Self { host, handle }
    }
}

impl std::fmt::Debug for LocalHostQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalHostQuery").finish()
    }
}

impl WorldQuery for LocalHostQuery {
    fn brick(&self, addr: &WorldAddr, brick_coord: IVec3, lod: Lod) -> Option<Arc<Brick>> {
        let address = (*addr).into();
        let env =
            Envelope::new(0, address, WorldRequest::GetBrick { addr: address, brick: brick_coord, lod });
        let resp = self.handle.block_on(self.host.request(env)).ok()?;
        match resp.body {
            WorldEvent::BrickSnapshot { payload, .. } => Brick::from_bytes(&payload).ok().map(Arc::new),
            _ => None,
        }
    }

    fn ground_height_m(&self, addr: &WorldAddr, xz: [f64; 2]) -> Option<f32> {
        // Best-effort probe. We don't have a streaming-column API yet, so
        // sample one brick at LOD 0 covering the (x, z) column and scan its
        // non-empty voxels top-down. The world's vertical extent is the
        // brick edge (16 m) — callers that need a taller column should
        // scan a stack of bricks; that's left to higher-level helpers.
        let edge = BRICK_EDGE as i64;
        let bx = (xz[0] / edge as f64).floor() as i64;
        let bz = (xz[1] / edge as f64).floor() as i64;
        // Try at brick_y = 0 first — typical flat-column world.
        let brick = self.brick(addr, IVec3::new(bx, 0, bz), Lod::new(0))?;
        let lx = ((xz[0] - (bx as f64) * edge as f64) as i64).clamp(0, edge - 1);
        let lz = ((xz[1] - (bz as f64) * edge as f64) as i64).clamp(0, edge - 1);
        for y in (0..edge).rev() {
            let v = brick.get(IVec3::new(lx, y, lz));
            if !v.is_empty() {
                return Some(y as f32);
            }
        }
        None
    }

    fn subscribe_region(&self, addr: &WorldAddr, region: AABB, lod: Lod) -> std_mpsc::Receiver<WorldEvent> {
        let (std_tx, std_rx) = std_mpsc::channel();
        let address = (*addr).into();
        let env =
            Envelope::new(0, address, WorldRequest::Subscribe { addr: address, region, lod, sub_id: 0 });
        let host = self.host.clone();
        let handle = self.handle.clone();
        self.handle.spawn(async move {
            let mut rx = match host.subscribe(env).await {
                Ok(r) => r,
                Err(_) => return,
            };
            // Forward every event body. Stop the moment either side closes.
            while let Some(envelope) = rx.recv().await {
                if std_tx.send(envelope.body).is_err() {
                    break;
                }
            }
            // Touch `handle` so the borrow checker doesn't reuse-warn —
            // we intentionally outlive `self`.
            let _ = handle;
        });
        std_rx
    }
}
