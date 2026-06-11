use atomr_worlds_core::addr::Address;
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::dim::DimensionId;
use atomr_worlds_core::interaction::InteractionUnit;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::lww::WriterId;
use atomr_worlds_core::vehicle::{AffineFrame, ContainingFrame, VehicleAddr};
use atomr_worlds_core::HlcTimestamp;
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
    /// HLC-stamped single-voxel write. Like [`WorldRequest::WriteVoxel`] but the
    /// caller supplies the [`HlcTimestamp`]/[`WriterId`], so the host resolves
    /// it under last-writer-wins (replying [`WorldEvent::WriteRejected`] if the
    /// write lost). Lets an optimistic client reconcile concurrent edits.
    WriteVoxelStamped {
        addr: Address,
        pos: IVec3,
        voxel: Voxel,
        ts: HlcTimestamp,
        writer: WriterId,
    },
    /// HLC-stamped brush write — the [`WorldRequest::WriteRegion`] analogue of
    /// [`WorldRequest::WriteVoxelStamped`]. Cells that lose the LWW merge are
    /// silently skipped (a brush is best-effort; no per-voxel rejection).
    WriteRegionStamped {
        addr: Address,
        center: DVec3,
        unit: InteractionUnit,
        voxel: Voxel,
        ts: HlcTimestamp,
        writer: WriterId,
    },
    /// Ask the authoritative actor to evaluate a structural fracture at the
    /// impact point. The actor runs the integer connectivity decision, journals
    /// the island removal, and replies with a [`WorldEvent::FractureApplied`]
    /// command sequence (also fanned out to region subscribers).
    Fracture(crate::fracture::FractureRequest),
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
    /// A stamped write lost a concurrent last-writer-wins merge — the writer
    /// should roll its optimistic preview back to `current`.
    WriteRejected(crate::fracture::WriteRejected),
    /// Authoritative fracture result: an ordered, replayable command sequence
    /// plus the inclusive journal sequence range its voxel writes were recorded
    /// at, so late joiners can replay deterministically.
    FractureApplied(crate::fracture::FractureApplied),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fracture::{FractureApplied, FractureCommand, FractureRequest, Force, WriteRejected};
    use crate::wire::{decode, encode};
    use atomr_worlds_core::addr::WorldAddr;

    fn root() -> Address {
        Address::World(WorldAddr::ROOT)
    }

    #[test]
    fn stamped_and_fracture_requests_round_trip() {
        let reqs = [
            WorldRequest::WriteVoxelStamped {
                addr: root(),
                pos: IVec3::new(1, 2, 3),
                voxel: Voxel::new(7),
                ts: HlcTimestamp::new(42, 3),
                writer: WriterId(9),
            },
            WorldRequest::Fracture(FractureRequest {
                addr: root(),
                impact_pos: IVec3::new(4, 5, 6),
                force: Force::from_milli_n(IVec3::new(0, -8000, 0)),
                material_id: 2,
            }),
        ];
        for r in reqs {
            let bytes = encode(&r).unwrap();
            let back: WorldRequest = decode(&bytes).unwrap();
            assert_eq!(format!("{r:?}"), format!("{back:?}"));
        }
    }

    #[test]
    fn write_rejected_and_fracture_applied_round_trip() {
        let rej = WorldEvent::WriteRejected(WriteRejected {
            addr: root(),
            pos: IVec3::new(1, 0, 0),
            current: Voxel::new(5),
        });
        let applied = WorldEvent::FractureApplied(FractureApplied {
            addr: root(),
            commands: vec![FractureCommand::SetVoxel {
                pos: IVec3::new(1, 0, 0),
                before: Voxel::new(5),
                after: Voxel::EMPTY,
            }],
            seq_range: (10, 11),
        });
        for e in [rej, applied] {
            let bytes = encode(&e).unwrap();
            let back: WorldEvent = decode(&bytes).unwrap();
            assert_eq!(format!("{e:?}"), format!("{back:?}"));
        }
    }
}
