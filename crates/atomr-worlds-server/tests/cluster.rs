//! Integration test for `run_cluster`: two nodes boot via the same
//! entry point the binary uses, share a coordinator, and a client
//! connected to alpha sees writes targeting a shard pinned to beta.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use atomr_cluster_sharding::ShardCoordinator;
use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_host::{WorldExtractor, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_remote::{RemoteHost, RemoteHostConfig};
use atomr_worlds_server::{run_cluster_with, ClusterConfig};
use atomr_worlds_voxel::Voxel;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn two_node_cluster_routes_via_owning_shard() {
    let coordinator = Arc::new(ShardCoordinator::new());

    // Bring both nodes up first — gateway paths depend on the OS-assigned
    // ports, so peers can only be wired *after* both ports are known.
    let alpha = run_cluster_with(
        ClusterConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            system_name: "cluster-test-alpha".into(),
            root_seed: 0xAAAA,
            region_id: "alpha".into(),
            peers: HashMap::new(),
            request_timeout: Duration::from_secs(5),
        },
        coordinator.clone(),
    )
    .await
    .unwrap();
    let beta = run_cluster_with(
        ClusterConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            system_name: "cluster-test-beta".into(),
            root_seed: 0xBBBB,
            region_id: "beta".into(),
            peers: HashMap::new(),
            request_timeout: Duration::from_secs(5),
        },
        coordinator.clone(),
    )
    .await
    .unwrap();

    // Re-wire forwarders now that both gateway paths exist.
    let mut peers_a = HashMap::new();
    peers_a.insert("beta".to_string(), beta.server_path.clone());
    atomr_worlds_remote::install_cluster_remote_forwarder(
        &alpha.cluster,
        alpha.remote.clone(),
        peers_a,
    )
    .unwrap();
    let mut peers_b = HashMap::new();
    peers_b.insert("alpha".to_string(), alpha.server_path.clone());
    atomr_worlds_remote::install_cluster_remote_forwarder(
        &beta.cluster,
        beta.remote.clone(),
        peers_b,
    )
    .unwrap();

    // Pin the root shard to beta.
    let shard = WorldExtractor::shard_id_for(&Address::World(WorldAddr::ROOT));
    coordinator.rebalance(&shard, "beta");

    // Client connects to alpha. Alpha's cluster forwarder ships the
    // requests on to beta.
    let client = RemoteHost::new(RemoteHostConfig {
        server_path: alpha.server_path.clone(),
        request_timeout: Duration::from_secs(5),
        ..RemoteHostConfig::default()
    })
    .await
    .unwrap();
    let addr = Address::World(WorldAddr::ROOT);
    let pos = IVec3::new(2, 3, 5);

    let resp = client
        .request(Envelope::new(
            1,
            addr,
            WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(99) },
        ))
        .await
        .unwrap();
    assert!(matches!(resp.body, WorldEvent::Ack { .. }));

    let resp = client
        .request(Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos }))
        .await
        .unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else {
        panic!("expected Voxel, got {:?}", resp.body);
    };
    assert_eq!(voxel, Voxel::new(99));

    client.shutdown().await.unwrap();
    alpha.shutdown().await.unwrap();
    beta.shutdown().await.unwrap();
}
