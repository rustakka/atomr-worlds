//! Off-main-thread builder cache for the orthographic raster views (slice / RTS).
//!
//! `build_slice_table_with_lod_fn` / `build_surface_raster_with_lod_fn` call the
//! host `WorldQuery::brick` thousands of times per frame — and the only
//! `WorldQuery` the client has is [`LocalHostQuery`](atomr_worlds_host::LocalHostQuery),
//! whose `brick` does `handle.block_on(host.request(..))`. Running that builder
//! on the render thread blocked the frame on 2–4k host round-trips *every frame*
//! the view was open.
//!
//! [`AsyncBuild`] runs the whole builder on a background `std::thread` (NOT a
//! tokio worker — `block_on` must not run on a runtime thread) and the render
//! system draws the most-recent finished result, rebuilding only when the view
//! footprint changes. Same staleness tradeoff as
//! [`DesiredChunksCache`](crate::world_stream::DesiredChunksCache): the ortho
//! view can lag the pan by a rebuild, which is invisible for a map view.

use std::sync::mpsc;
use std::sync::Mutex;

/// Generic "rebuild `T` off-thread keyed by `K`, keep the latest" cache. `T` is
/// the built artifact (a `SliceTable` / `SurfaceRaster`); `K` is the footprint
/// key that decides when a rebuild is needed.
pub struct AsyncBuild<T: Send + 'static, K: PartialEq + Clone + Send + Sync + 'static> {
    built_for: Option<K>,
    result: Option<T>,
    rebuild: Option<Mutex<mpsc::Receiver<T>>>,
}

impl<T: Send + 'static, K: PartialEq + Clone + Send + Sync + 'static> Default for AsyncBuild<T, K> {
    fn default() -> Self {
        Self { built_for: None, result: None, rebuild: None }
    }
}

impl<T: Send + 'static, K: PartialEq + Clone + Send + Sync + 'static> AsyncBuild<T, K> {
    /// Install a finished rebuild if one has arrived (non-blocking). Returns
    /// `true` when a new result was installed.
    pub fn poll(&mut self) -> bool {
        let done = self
            .rebuild
            .as_ref()
            .and_then(|rx| rx.lock().expect("raster rebuild rx poisoned").try_recv().ok());
        if let Some(result) = done {
            self.result = Some(result);
            self.rebuild = None;
            true
        } else {
            false
        }
    }

    /// Whether a rebuild is currently in flight.
    #[inline]
    pub fn is_rebuilding(&self) -> bool {
        self.rebuild.is_some()
    }

    /// The footprint key the current (or in-flight) result is for.
    #[inline]
    pub fn built_for(&self) -> Option<&K> {
        self.built_for.as_ref()
    }

    /// The most-recent finished artifact, if any.
    #[inline]
    pub fn current(&self) -> Option<&T> {
        self.result.as_ref()
    }

    /// `true` if a rebuild for `key` is warranted: nothing in flight and the
    /// last (or pending) build was for a different footprint.
    pub fn needs_rebuild(&self, key: &K) -> bool {
        !self.is_rebuilding() && self.built_for.as_ref() != Some(key)
    }

    /// Spawn a rebuild for `key` on a background thread. The caller should gate
    /// on [`Self::needs_rebuild`] first. The closure must be self-contained
    /// (own its inputs); it runs off the Bevy world entirely.
    pub fn spawn<F: FnOnce() -> T + Send + 'static>(&mut self, key: K, build: F) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            #[cfg(feature = "profiling")]
            let _z = tracing::info_span!("raster_async_build").entered();
            let _ = tx.send(build());
        });
        self.built_for = Some(key);
        self.rebuild = Some(Mutex::new(rx));
    }
}
