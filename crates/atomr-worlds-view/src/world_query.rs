//! Read-only world-access shim for the view crate.
//!
//! The view crate is mesh-input-only (Phase 13f principle): it does not
//! depend on `atomr-worlds-host`. To still let renderer modes fetch bricks,
//! probe ground height, and react to streaming deltas, we **invert** the
//! dependency: the view crate defines the [`WorldQuery`] trait here, and
//! the host implements it. The real impl lives in `atomr-worlds-host`
//! (Wave 2), a stub impl will live in `atomr-worlds-testkit`.
//!
//! This module deliberately depends only on
//!
//! - `atomr-worlds-core` for [`WorldAddr`], [`Lod`], [`IVec3`],
//! - `atomr-worlds-voxel` for [`Brick`],
//! - `atomr-worlds-proto` for [`AABB`] and [`WorldEvent`].
//!
//! and exposes a tiny `Send + Sync` trait surface so callers can hold a
//! `Box<dyn WorldQuery>` or `Arc<dyn WorldQuery>`.

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_voxel::brick::Brick;

/// Read-only world access used by the view crate's rendering modes.
///
/// Implementations must be `Send + Sync` so caches and renderer pipelines
/// can hold them across threads. All methods are infallible at the trait
/// level — they return `Option` for "no data here" rather than `Result`,
/// since renderer code generally treats a missing brick as empty rather
/// than as an error.
pub trait WorldQuery: Send + Sync {
    /// Fetch a single brick at the given (address, brick coordinate, LOD).
    ///
    /// `None` means the host has no brick at that slot — empty space, not
    /// yet generated, or out of range. Returning `Arc<Brick>` lets the
    /// host share a cached page without forcing a clone per call.
    fn brick(&self, addr: &WorldAddr, brick_coord: IVec3, lod: Lod) -> Option<Arc<Brick>>;

    /// Probe the world-space ground height at horizontal coordinate `xz`
    /// (meters, world frame). `None` if the host can't answer (e.g. the
    /// column is unloaded or the world has no defined surface there).
    ///
    /// Renderer modes use this to clamp a camera or anchor above
    /// terrain — collision is out of scope.
    fn ground_height_m(&self, addr: &WorldAddr, xz: [f64; 2]) -> Option<f32>;

    /// Subscribe to streaming events inside `region` at `lod`. The
    /// returned [`Receiver`] yields the initial `BrickSnapshot`s followed
    /// by `VoxelDelta`/`RegionDelta`/etc. as the host emits them; closing
    /// the receiver (or the host's stream ending) drops the subscription.
    fn subscribe_region(&self, addr: &WorldAddr, region: AABB, lod: Lod) -> Receiver<WorldEvent>;
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::sync::mpsc;

    use atomr_worlds_core::addr::{LevelKey, WorldAddr};
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_core::dim::PRIMARY;
    use atomr_worlds_voxel::brick::Brick;
    use atomr_worlds_voxel::voxel::Voxel;

    /// Tiny canned implementation. The view crate doesn't ship a real
    /// host — these are only here to verify the trait shape compiles
    /// and is object-safe.
    struct StubWorld {
        bricks: HashMap<(IVec3, u8), Arc<Brick>>,
        ground: HashMap<(i64, i64), f32>,
        /// Single canned event the subscription emits on connect.
        canned_event: WorldEvent,
    }

    impl StubWorld {
        fn new() -> Self {
            // Seed one non-empty brick at brick_coord = (1, 2, 3), lod = 4.
            let mut brick = Brick::new();
            brick.voxels[0] = Voxel::new(7);
            brick.nonempty_count = 1;
            let mut bricks = HashMap::new();
            bricks.insert((IVec3::new(1, 2, 3), 4u8), Arc::new(brick));

            let mut ground = HashMap::new();
            ground.insert((0, 0), 12.5_f32);

            let canned_event = WorldEvent::BrickSnapshot {
                addr: WorldAddr::ROOT.into(),
                brick: IVec3::new(1, 2, 3),
                lod: Lod::new(4),
                payload: bytes::Bytes::from_static(&[]),
            };

            Self { bricks, ground, canned_event }
        }
    }

    impl WorldQuery for StubWorld {
        fn brick(&self, _addr: &WorldAddr, brick_coord: IVec3, lod: Lod) -> Option<Arc<Brick>> {
            self.bricks.get(&(brick_coord, lod.depth)).cloned()
        }

        fn ground_height_m(&self, _addr: &WorldAddr, xz: [f64; 2]) -> Option<f32> {
            self.ground.get(&(xz[0] as i64, xz[1] as i64)).copied()
        }

        fn subscribe_region(&self, _addr: &WorldAddr, _region: AABB, _lod: Lod) -> Receiver<WorldEvent> {
            let (tx, rx) = mpsc::channel();
            // Clone the canned event so the receiver gets exactly one.
            tx.send(self.canned_event.clone()).expect("local channel send must succeed");
            rx
        }
    }

    fn root_addr() -> WorldAddr {
        WorldAddr {
            universe: LevelKey::new(IVec3::ZERO, PRIMARY),
            galaxy: LevelKey::new(IVec3::ZERO, PRIMARY),
            sector: LevelKey::new(IVec3::ZERO, PRIMARY),
            system: LevelKey::new(IVec3::ZERO, PRIMARY),
            world: LevelKey::new(IVec3::ZERO, PRIMARY),
        }
    }

    #[test]
    fn trait_object_constructs() {
        // Object-safety check: must be usable behind `dyn`.
        let _q: Box<dyn WorldQuery> = Box::new(StubWorld::new());
    }

    #[test]
    fn brick_returns_canned() {
        let q = StubWorld::new();
        let addr = root_addr();
        let got = q.brick(&addr, IVec3::new(1, 2, 3), Lod::new(4));
        assert!(got.is_some(), "stub should return brick at the seeded coordinate");
        assert_eq!(got.unwrap().nonempty_count, 1);

        // Wrong coordinate → None.
        let miss = q.brick(&addr, IVec3::new(9, 9, 9), Lod::new(4));
        assert!(miss.is_none());
    }

    #[test]
    fn ground_height_returns_canned() {
        let q = StubWorld::new();
        let addr = root_addr();
        let h = q.ground_height_m(&addr, [0.0, 0.0]);
        assert_eq!(h, Some(12.5));
        let miss = q.ground_height_m(&addr, [100.0, 100.0]);
        assert_eq!(miss, None);
    }

    #[test]
    fn subscribe_region_yields_events() {
        let q = StubWorld::new();
        let addr = root_addr();
        let region = AABB::new(IVec3::ZERO, IVec3::new(16, 16, 16));
        let rx = q.subscribe_region(&addr, region, Lod::new(4));
        let ev = rx.recv().expect("stub should emit one canned event");
        match ev {
            WorldEvent::BrickSnapshot { brick, lod, .. } => {
                assert_eq!(brick, IVec3::new(1, 2, 3));
                assert_eq!(lod, Lod::new(4));
            }
            other => panic!("expected BrickSnapshot, got {other:?}"),
        }
    }
}
