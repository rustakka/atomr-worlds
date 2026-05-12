//! Distance-driven streaming policy.
//!
//! A subscriber's [`StreamingPolicy`] controls which LOD the host streams
//! bricks at, as a function of the observer's distance from the body. Bricks
//! within `transition_radius_m` stream at `near_lod`; bricks beyond stream at
//! the coarser `far_lod`. Beyond `max_radius_m` nothing streams.

use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use serde::{Deserialize, Serialize};

use crate::aabb::AABB;

/// Per-subscription streaming budget + transition geometry.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct StreamingPolicy {
    /// LOD applied to bricks within `transition_radius_m` of the observer.
    pub near_lod: Lod,
    /// LOD applied beyond `transition_radius_m` (typically coarser —
    /// `far_lod.depth <= near_lod.depth`).
    pub far_lod: Lod,
    /// Boundary between near and far LOD tiers, in meters.
    pub transition_radius_m: f64,
    /// Hard cull radius. Bricks beyond this distance never stream.
    pub max_radius_m: f64,
    /// Throughput cap: maximum bricks emitted per `ObserverTick`.
    pub bricks_per_tick: u32,
}

impl StreamingPolicy {
    /// Conservative static-region policy: same LOD everywhere within
    /// `max_radius_m`, with a small per-tick budget. Used as the conversion
    /// target for the existing static [`crate::WorldRequest::Subscribe`].
    pub fn fixed(lod: Lod, max_radius_m: f64) -> Self {
        Self {
            near_lod: lod,
            far_lod: lod,
            transition_radius_m: max_radius_m,
            max_radius_m,
            bricks_per_tick: 64,
        }
    }

    /// Plan the brick AABBs to stream at each LOD for the given observer pos.
    pub fn ring_for(&self, observer: DVec3, brick_edge_m: f64) -> RingPlan {
        let max_v = (self.max_radius_m / brick_edge_m).ceil() as i64;
        let near_v = (self.transition_radius_m / brick_edge_m).ceil() as i64;
        let ox = (observer.x / brick_edge_m).floor() as i64;
        let oy = (observer.y / brick_edge_m).floor() as i64;
        let oz = (observer.z / brick_edge_m).floor() as i64;
        let near = AABB::new(
            IVec3::new(ox - near_v, oy - near_v, oz - near_v),
            IVec3::new(ox + near_v + 1, oy + near_v + 1, oz + near_v + 1),
        );
        let far = AABB::new(
            IVec3::new(ox - max_v, oy - max_v, oz - max_v),
            IVec3::new(ox + max_v + 1, oy + max_v + 1, oz + max_v + 1),
        );
        RingPlan { near_bricks: near, far_bricks: far, near_lod: self.near_lod, far_lod: self.far_lod }
    }

    /// As [`Self::ring_for`], but clamps both `transition_radius_m` and
    /// `max_radius_m` to `horizon_m`. Used by spherical worlds so the
    /// streamer never tries to send bricks past the visible surface.
    /// `horizon_m == f64::INFINITY` (the cube case) makes this equivalent
    /// to [`Self::ring_for`].
    pub fn ring_for_curved(
        &self,
        observer: DVec3,
        brick_edge_m: f64,
        horizon_m: f64,
    ) -> RingPlan {
        let max_r = self.max_radius_m.min(horizon_m);
        let near_r = self.transition_radius_m.min(horizon_m);
        let max_v = (max_r / brick_edge_m).ceil() as i64;
        let near_v = (near_r / brick_edge_m).ceil() as i64;
        let ox = (observer.x / brick_edge_m).floor() as i64;
        let oy = (observer.y / brick_edge_m).floor() as i64;
        let oz = (observer.z / brick_edge_m).floor() as i64;
        let near = AABB::new(
            IVec3::new(ox - near_v, oy - near_v, oz - near_v),
            IVec3::new(ox + near_v + 1, oy + near_v + 1, oz + near_v + 1),
        );
        let far = AABB::new(
            IVec3::new(ox - max_v, oy - max_v, oz - max_v),
            IVec3::new(ox + max_v + 1, oy + max_v + 1, oz + max_v + 1),
        );
        RingPlan { near_bricks: near, far_bricks: far, near_lod: self.near_lod, far_lod: self.far_lod }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct RingPlan {
    pub near_bricks: AABB,
    pub far_bricks: AABB,
    pub near_lod: Lod,
    pub far_lod: Lod,
}
