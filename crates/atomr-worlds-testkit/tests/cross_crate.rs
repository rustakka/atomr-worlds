//! Cross-crate proptest verification for phase-0 invariants.

use std::collections::HashMap;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::Voxel;
use atomr_worlds_proto::{decode, encode, AABB, Envelope, WorldEvent, WorldRequest};
use atomr_worlds_testkit::{arb_brick, arb_world_addr};
use proptest::prelude::*;

proptest! {
    #[test]
    fn world_addr_bincode_round_trip(addr in arb_world_addr()) {
        let bytes = bincode::serde::encode_to_vec(addr, bincode::config::standard()).unwrap();
        let (back, _): (WorldAddr, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        prop_assert_eq!(addr, back);
    }

    #[test]
    fn world_addr_json_round_trip(addr in arb_world_addr()) {
        let s = serde_json::to_string(&addr).unwrap();
        let back: WorldAddr = serde_json::from_str(&s).unwrap();
        prop_assert_eq!(addr, back);
    }

    #[test]
    fn seed_chain_is_deterministic(addr in arb_world_addr(), root in any::<u64>()) {
        let a = addr.seed_chain(root);
        let b = addr.seed_chain(root);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn brick_matches_hashmap_oracle(b in arb_brick()) {
        // Build a HashMap oracle from the same brick.
        let mut oracle: HashMap<(usize, usize, usize), Voxel> = HashMap::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let v = b.get(IVec3::new(x as i64, y as i64, z as i64));
                    if !v.is_empty() {
                        oracle.insert((x, y, z), v);
                    }
                }
            }
        }
        let nonempty_oracle = oracle.len();
        prop_assert_eq!(nonempty_oracle, b.nonempty_count as usize);
    }
}

#[test]
fn protocol_round_trip_get_voxel() {
    let env = Envelope::new(
        7,
        WorldAddr::ROOT,
        WorldRequest::GetVoxel { addr: WorldAddr::ROOT, pos: IVec3::new(1, 2, 3) },
    );
    let bytes = encode(&env).unwrap();
    let back: Envelope<WorldRequest> = decode(&bytes).unwrap();
    assert_eq!(back.corr_id, 7);
    match back.body {
        WorldRequest::GetVoxel { pos, .. } => assert_eq!(pos, IVec3::new(1, 2, 3)),
        _ => panic!("variant mismatch"),
    }
}

#[test]
fn protocol_round_trip_brick_snapshot() {
    use atomr_worlds_core::lod::Lod;
    let payload = bytes::Bytes::from_static(&[0u8; 32]);
    let env = Envelope::new(
        9,
        WorldAddr::ROOT,
        WorldEvent::BrickSnapshot {
            addr: WorldAddr::ROOT,
            brick: IVec3::new(0, 0, 0),
            lod: Lod::new(3),
            payload: payload.clone(),
        },
    );
    let bytes_buf = encode(&env).unwrap();
    let back: Envelope<WorldEvent> = decode(&bytes_buf).unwrap();
    match back.body {
        WorldEvent::BrickSnapshot { payload: p, lod, .. } => {
            assert_eq!(p.as_ref(), payload.as_ref());
            assert_eq!(lod.depth, 3);
        }
        _ => panic!("variant mismatch"),
    }
}

#[test]
fn subscribe_envelope_round_trip() {
    use atomr_worlds_core::lod::Lod;
    let env = Envelope::new(
        1,
        WorldAddr::ROOT,
        WorldRequest::Subscribe {
            addr: WorldAddr::ROOT,
            region: AABB::new(IVec3::new(-1, -1, -1), IVec3::new(1, 1, 1)),
            lod: Lod::new(0),
            sub_id: 42,
        },
    );
    let bytes = encode(&env).unwrap();
    let _back: Envelope<WorldRequest> = decode(&bytes).unwrap();
}
