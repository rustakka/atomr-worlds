//! `WorldRuntime` — Bevy resource that owns the tokio runtime and the
//! `WorldQuery` bridge.

use std::sync::Arc;

use atomr_worlds_host::{LocalHostQuery, WorldHost};
use bevy::prelude::Resource;
use tokio::runtime::Runtime;

/// Long-lived state Bevy systems use to talk to the world host.
///
/// `runtime` keeps the tokio reactor alive (LocalHost / RemoteHost spawned
/// actors live on it). `host` is the active `WorldHost`. `query` is the
/// synchronous [`WorldQuery`](atomr_worlds_view::WorldQuery) bridge used
/// by render systems.
///
/// `runtime` and `host` look unused but are load-bearing: dropping them
/// would tear down the actor system mid-frame.
#[derive(Resource)]
pub struct WorldRuntime {
    #[allow(dead_code)]
    pub runtime: Arc<Runtime>,
    #[allow(dead_code)]
    pub host: Arc<dyn WorldHost>,
    pub query: Arc<LocalHostQuery>,
}

impl WorldRuntime {
    pub fn new(runtime: Arc<Runtime>, host: Arc<dyn WorldHost>) -> Self {
        let query = Arc::new(LocalHostQuery::from_dyn(host.clone(), runtime.handle().clone()));
        Self { runtime, host, query }
    }
}

impl std::fmt::Debug for WorldRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldRuntime").finish_non_exhaustive()
    }
}

/// World we're currently rendering. For now the client shows a single
/// `WorldAddr::ROOT` instance; multi-world is a follow-up.
///
/// `shape` lets the streamer compute a body-aware horizon (sphere worlds
/// clamp the far-ring radius to `sqrt(2*R*h + h²)`; cube worlds keep the
/// `f64::INFINITY` no-op behaviour). It defaults to
/// [`WorldShape::default_world`] (cube), preserving prior behaviour for
/// callers that don't override it — only the overview mode currently
/// hardcodes a sphere shape, and that path doesn't go through the FP
/// streamer.
#[derive(Resource, Copy, Clone, Debug)]
pub struct ActiveWorld {
    pub addr: atomr_worlds_core::addr::WorldAddr,
    #[allow(dead_code)]
    pub seed: u64,
    pub shape: atomr_worlds_core::shape::WorldShape,
}
