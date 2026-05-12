//! `MessageExtractor` for atomr-cluster-sharding.
//!
//! - `entity_id` is the full hierarchical path of the addressed entity
//!   (world or vehicle), with a discriminator prefix.
//! - `shard_id` packs `(universe.coord, universe.dim, galaxy.coord,
//!   sector.coord)` of the **owning sector** so that all entities (worlds and
//!   vehicles) within one stellar system stay co-resident on the shard owner,
//!   while sectors load-balance across the cluster. Vehicles anchored at a
//!   system or sector use the same sector path so they shard with their
//!   parent neighborhood.

use atomr_cluster_sharding::MessageExtractor;
use atomr_worlds_core::addr::{Address, Level, WorldAddr};
use atomr_worlds_core::vehicle::{ParentAddr, VehicleAddr};
use atomr_worlds_proto::{Envelope, WorldRequest};

#[derive(Debug, Default, Clone, Copy)]
pub struct WorldExtractor;

impl WorldExtractor {
    /// Encode `(universe.coord, universe.dim, galaxy.coord, sector.coord)` as a
    /// shard id string from a world address.
    fn shard_id_for_world(addr: &WorldAddr) -> String {
        let u = addr.universe;
        let g = addr.galaxy.coord;
        let s = addr.sector.coord;
        format!(
            "u:{},{},{}:{}|g:{},{},{}|s:{},{},{}",
            u.coord.x, u.coord.y, u.coord.z, u.dim, g.x, g.y, g.z, s.x, s.y, s.z
        )
    }

    /// Shard id for any [`Address`]. Vehicles use their parent's sector path
    /// so they co-locate with the system/worlds in the same sector.
    pub fn shard_id_for(addr: &Address) -> String {
        match addr {
            Address::World(w) => Self::shard_id_for_world(w),
            Address::Vehicle(v) => Self::shard_id_for_world(&Self::parent_world_addr(v)),
        }
    }

    fn parent_world_addr(v: &VehicleAddr) -> WorldAddr {
        match v.parent {
            ParentAddr::World(a) => a,
            ParentAddr::System(a) => a.ancestor(Level::System),
            ParentAddr::Sector(a) => a.ancestor(Level::Sector),
        }
    }

    fn entity_id_for_world(addr: &WorldAddr) -> String {
        let mut out = String::with_capacity(96);
        for l in Level::ALL {
            let k = addr.level_key(l);
            out.push_str(&format!("{:?}:{},{},{}:{};", l, k.coord.x, k.coord.y, k.coord.z, k.dim));
        }
        out
    }

    /// Encode the full address path as a stable entity id with a
    /// discriminator (`W:` for worlds, `V:` for vehicles).
    pub fn entity_id_for(addr: &Address) -> String {
        match addr {
            Address::World(w) => format!("W:{}", Self::entity_id_for_world(w)),
            Address::Vehicle(v) => {
                let parent = Self::entity_id_for_world(&Self::parent_world_addr(v));
                let parent_tag = match v.parent {
                    ParentAddr::World(_) => "pW",
                    ParentAddr::System(_) => "pS",
                    ParentAddr::Sector(_) => "pK",
                };
                format!(
                    "V:{}|{}|slot:{}:{}",
                    parent, parent_tag, v.slot.slot_id, v.slot.dim
                )
            }
        }
    }

    fn addr_of(message: &Envelope<WorldRequest>) -> Address {
        match &message.body {
            WorldRequest::GetVoxel { addr, .. } => *addr,
            WorldRequest::GetBrick { addr, .. } => *addr,
            WorldRequest::WriteVoxel { addr, .. } => *addr,
            WorldRequest::Subscribe { addr, .. } => *addr,
            WorldRequest::SubscribeMetric { addr, .. } => *addr,
            WorldRequest::WriteRegion { addr, .. } => *addr,
            WorldRequest::TraversePortal { addr, .. } => *addr,
            WorldRequest::GetVehicleFrame { addr } => Address::Vehicle(*addr),
            WorldRequest::SetVehicleFrame { addr, .. } => Address::Vehicle(*addr),
            // Unsubscribe / observer ticks route via the envelope's `from`.
            WorldRequest::Unsubscribe { .. } | WorldRequest::UpdateObserverPos { .. } => message.from,
        }
    }
}

impl MessageExtractor for WorldExtractor {
    type Message = Envelope<WorldRequest>;

    fn entity_id(&self, message: &Self::Message) -> String {
        Self::entity_id_for(&Self::addr_of(message))
    }

    fn shard_id(&self, message: &Self::Message) -> String {
        Self::shard_id_for(&Self::addr_of(message))
    }
}
