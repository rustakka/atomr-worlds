//! `MessageExtractor` for atomr-cluster-sharding.
//!
//! - `entity_id` is the full hierarchical address of the addressed world.
//! - `shard_id` packs `(universe.coord, universe.dim, galaxy.coord, sector.coord)`
//!   so all bricks belonging to one stellar system stay co-resident on the
//!   shard owner, while sectors load-balance across the cluster.

use atomr_cluster_sharding::MessageExtractor;
use atomr_worlds_core::addr::{Level, WorldAddr};
use atomr_worlds_proto::{Envelope, WorldRequest};

#[derive(Debug, Default, Clone, Copy)]
pub struct WorldExtractor;

impl WorldExtractor {
    /// Encode `(universe.coord, universe.dim, galaxy.coord, sector.coord)` as a
    /// shard id string.
    pub fn shard_id_for(addr: &WorldAddr) -> String {
        let u = addr.universe;
        let g = addr.galaxy.coord;
        let s = addr.sector.coord;
        format!(
            "u:{},{},{}:{}|g:{},{},{}|s:{},{},{}",
            u.coord.x, u.coord.y, u.coord.z, u.dim, g.x, g.y, g.z, s.x, s.y, s.z
        )
    }

    /// Encode the full address path as a stable entity id.
    pub fn entity_id_for(addr: &WorldAddr) -> String {
        let mut out = String::with_capacity(96);
        for l in Level::ALL {
            let k = addr.level_key(l);
            out.push_str(&format!("{:?}:{},{},{}:{};", l, k.coord.x, k.coord.y, k.coord.z, k.dim));
        }
        out
    }

    fn addr_of(message: &Envelope<WorldRequest>) -> WorldAddr {
        match &message.body {
            WorldRequest::GetVoxel { addr, .. } => *addr,
            WorldRequest::GetBrick { addr, .. } => *addr,
            WorldRequest::WriteVoxel { addr, .. } => *addr,
            WorldRequest::Subscribe { addr, .. } => *addr,
            // Unsubscribe is routed using the envelope's `from` address.
            WorldRequest::Unsubscribe { .. } => message.from,
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
