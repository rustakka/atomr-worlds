use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_voxel::Voxel;
use serde::{Deserialize, Serialize};

use crate::aabb::AABB;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorldRequest {
    GetVoxel { addr: WorldAddr, pos: IVec3 },
    GetBrick { addr: WorldAddr, brick: IVec3, lod: Lod },
    WriteVoxel { addr: WorldAddr, pos: IVec3, voxel: Voxel },
    Subscribe { addr: WorldAddr, region: AABB, lod: Lod, sub_id: u64 },
    Unsubscribe { sub_id: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorldEvent {
    /// Single-voxel read response.
    Voxel { addr: WorldAddr, pos: IVec3, voxel: Voxel },
    /// Brick payload — `payload` is bincode-encoded `Brick`.
    BrickSnapshot { addr: WorldAddr, brick: IVec3, lod: Lod, payload: bytes::Bytes },
    /// Streaming update for a voxel that changed.
    VoxelDelta { addr: WorldAddr, pos: IVec3, before: Voxel, after: Voxel },
    /// Write acknowledged.
    Ack { addr: WorldAddr },
    /// Subscription ended (closed by client or actor).
    StreamEnd { sub_id: u64 },
}
