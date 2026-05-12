use atomr_worlds_core::addr::Address;
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::dim::DimensionId;
use atomr_worlds_core::interaction::InteractionUnit;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::vehicle::{AffineFrame, ContainingFrame, VehicleAddr};
use atomr_worlds_voxel::Voxel;
use serde::{Deserialize, Serialize};

use crate::aabb::AABB;
use crate::streaming::StreamingPolicy;

/// Cross-dimension or cross-world portal. `dim_change` is `Some` when the
/// portal also crosses a dimension boundary.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Portal {
    pub source: Address,
    pub dest: Address,
    pub dim_change: Option<DimensionId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorldRequest {
    GetVoxel { addr: Address, pos: IVec3 },
    GetBrick { addr: Address, brick: IVec3, lod: Lod },
    WriteVoxel { addr: Address, pos: IVec3, voxel: Voxel },
    /// Static-region subscription. The host emits one `BrickSnapshot` per
    /// brick overlapping `region`, then `VoxelDelta`s for in-region writes.
    Subscribe { addr: Address, region: AABB, lod: Lod, sub_id: u64 },
    Unsubscribe { sub_id: u64 },
    /// Read a vehicle's current pose.
    GetVehicleFrame { addr: VehicleAddr },
    /// Set a vehicle's pose. Causes `VehicleFrameDelta` to fan out to
    /// subscribers of the vehicle's address.
    SetVehicleFrame { addr: VehicleAddr, frame: AffineFrame },
    /// Atmosphere-bounded dynamic subscription. The host streams bricks at
    /// `policy.near_lod` within `transition_radius_m` and `policy.far_lod`
    /// beyond, demoting to the parent tier when the observer leaves the
    /// containing frame's atmosphere.
    SubscribeMetric {
        addr: Address,
        containing_frame: ContainingFrame,
        observer_pos: DVec3,
        policy: StreamingPolicy,
        sub_id: u64,
    },
    /// Update the observer position for an existing metric subscription.
    UpdateObserverPos { sub_id: u64, observer_pos: DVec3 },
    /// Brush-edit request: apply `voxel` to every voxel inside `unit` at
    /// `center`. Fans out as either per-voxel `VoxelDelta`s or a single
    /// `RegionDelta` depending on the host's threshold.
    WriteRegion {
        addr: Address,
        center: DVec3,
        unit: InteractionUnit,
        voxel: Voxel,
    },
    /// Traverse a registered portal from `addr`. The host returns a
    /// `PortalArrival` containing the destination address.
    TraversePortal { addr: Address, portal_id: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorldEvent {
    /// Single-voxel read response.
    Voxel { addr: Address, pos: IVec3, voxel: Voxel },
    /// Brick payload — `payload` is bincode-encoded `Brick`.
    BrickSnapshot { addr: Address, brick: IVec3, lod: Lod, payload: bytes::Bytes },
    /// Streaming update for a voxel that changed.
    VoxelDelta { addr: Address, pos: IVec3, before: Voxel, after: Voxel },
    /// Write acknowledged.
    Ack { addr: Address },
    /// Subscription ended (closed by client or actor).
    StreamEnd { sub_id: u64 },
    /// Vehicle pose, returned by `GetVehicleFrame` or on subscription begin.
    VehicleFrame { addr: VehicleAddr, frame: AffineFrame, tick: u64 },
    /// Streaming update for a vehicle pose change.
    VehicleFrameDelta { addr: VehicleAddr, frame: AffineFrame, tick: u64 },
    /// Observer crossed an atmosphere boundary; the streamed tier has changed.
    ContainingFrameChange { sub_id: u64, from: ContainingFrame, to: ContainingFrame, new_addr: Address },
    /// Tier-level instruction for clients during dynamic streaming — drop or
    /// pre-allocate brick storage at this LOD over this AABB.
    Tier { sub_id: u64, addr: Address, lod: Lod, region: AABB },
    /// Brush-write delta: the actor touched the listed bricks with `voxel`.
    /// Clients re-fetch (or recompute) the affected bricks.
    RegionDelta {
        addr: Address,
        center: DVec3,
        unit: InteractionUnit,
        voxel: Voxel,
        bricks_modified: Vec<IVec3>,
    },
    /// Result of a [`WorldRequest::TraversePortal`].
    PortalArrival { dest: Address, transform: [[f32; 4]; 4] },
}
