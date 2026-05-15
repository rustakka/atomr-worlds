//! Cross-node ClusterHost forwarding test.
//!
//! Two ClusterHosts on the loopback, each running a WorldGateway. Pin
//! the shard for `WorldAddr::ROOT` to node B's region; from node A,
//! write a voxel + read it back. The request must hop A → forwarder →
//! B's gateway → B's local host, with the reply flowing back through
//! A's cluster reply inbox.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use atomr_cluster_sharding::ShardCoordinator;
use atomr_core::actor::{ActorSystem, Props};
use atomr_remote::{RemoteSettings, RemoteSystem};
use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{ClusterHost, ClusterHostConfig, LocalHostConfig, WorldExtractor, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest, AABB};
use atomr_worlds_remote::{
    install_cluster_remote_forwarder, WireReply, WireRequest, WorldGateway, GATEWAY_ACTOR_NAME,
};
use atomr_worlds_voxel::Voxel;

struct Node {
    region_id: String,
    cluster: Arc<ClusterHost>,
    sys: ActorSystem,
    remote: Arc<RemoteSystem>,
    gateway_path: String,
}

async fn boot_node(region_id: &str, seed: u64, coordinator: Arc<ShardCoordinator>) -> Node {
    let sys = ActorSystem::create(format!("node-{region_id}"), atomr_config::Config::reference())
        .await
        .unwrap();
    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let remote = Arc::new(
        RemoteSystem::start(sys.clone(), bind, RemoteSettings::default()).await.unwrap(),
    );
    remote.register_bincode::<WireRequest>();
    remote.register_bincode::<WireReply>();

    let cluster = Arc::new(
        ClusterHost::new(ClusterHostConfig {
            region_id: region_id.into(),
            coordinator,
            local_config: LocalHostConfig { root_seed: seed, ..LocalHostConfig::default() },
            request_timeout: Duration::from_secs(5),
        })
        .await
        .unwrap(),
    );

    let cluster_for_actor: Arc<dyn WorldHost> = cluster.clone();
    let remote_for_actor = remote.clone();
    let gateway_ref = sys
        .actor_of(
            Props::create(move || {
                WorldGateway::new(cluster_for_actor.clone(), remote_for_actor.clone())
            }),
            GATEWAY_ACTOR_NAME,
        )
        .unwrap();
    remote.expose_actor(gateway_ref);

    let gateway_path = format!("{}/user/{}", remote.local_address, GATEWAY_ACTOR_NAME);
    Node { region_id: region_id.into(), cluster, sys, remote, gateway_path }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cross_node_request_routes_via_forwarder() {
    let coordinator = Arc::new(ShardCoordinator::new());

    let node_a = boot_node("alpha", 0xAAAA, coordinator.clone()).await;
    let node_b = boot_node("beta", 0xBBBB, coordinator.clone()).await;

    // Pin WorldAddr::ROOT's shard to beta so requests targeting it from
    // alpha must traverse the forwarder.
    let root_shard = WorldExtractor::shard_id_for(&Address::World(WorldAddr::ROOT));
    coordinator.rebalance(&root_shard, "beta");

    // Each node forwards to the *other* node's gateway.
    let mut members_a = HashMap::new();
    members_a.insert("beta".to_string(), node_b.gateway_path.clone());
    install_cluster_remote_forwarder(&node_a.cluster, node_a.remote.clone(), members_a).unwrap();
    let mut members_b = HashMap::new();
    members_b.insert("alpha".to_string(), node_a.gateway_path.clone());
    install_cluster_remote_forwarder(&node_b.cluster, node_b.remote.clone(), members_b).unwrap();

    let addr = Address::World(WorldAddr::ROOT);
    let pos = IVec3::new(4, 5, 6);

    // Write from alpha — should be forwarded to beta and acked.
    let write = Envelope::new(
        1,
        addr,
        WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(13) },
    );
    let resp = node_a.cluster.request(write).await.expect("cross-node write");
    assert!(matches!(resp.body, WorldEvent::Ack { .. }));

    // Read from alpha — same forwarding path, must see the voxel beta
    // wrote.
    let read = Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos });
    let resp = node_a.cluster.request(read).await.expect("cross-node read");
    let WorldEvent::Voxel { voxel, .. } = resp.body else {
        panic!("expected Voxel reply, got {:?}", resp.body);
    };
    assert_eq!(voxel, Voxel::new(13));

    // Defensive cleanup.
    let _ = node_a.cluster.shutdown().await;
    let _ = node_b.cluster.shutdown().await;
    node_a.remote.shutdown().await;
    node_b.remote.shutdown().await;
    node_a.sys.terminate().await;
    node_b.sys.terminate().await;
    let _ = node_a.region_id; // referenced for clarity
}

/// Phase 15 follow-up — cross-node subscription routing.
///
/// Two ClusterHosts on the loopback. Pin the shard for `WorldAddr::ROOT`
/// to node B; from node A, subscribe to a region on that address. The
/// Subscribe envelope must hop A → forwarder → B's gateway → B's local
/// host. B's gateway streams `BrickSnapshot` events back as
/// `WireReply::Event { sub_id, … }`, which A's cluster reply inbox
/// routes through the per-`sub_id` subs map registered by
/// `ClusterHost::subscribe`.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cross_node_subscribe_streams_events_back() {
    let coordinator = Arc::new(ShardCoordinator::new());

    let node_a = boot_node("alpha-sub", 0xABCD, coordinator.clone()).await;
    let node_b = boot_node("beta-sub", 0xCDEF, coordinator.clone()).await;

    let root_shard = WorldExtractor::shard_id_for(&Address::World(WorldAddr::ROOT));
    coordinator.rebalance(&root_shard, "beta-sub");

    let mut members_a = HashMap::new();
    members_a.insert("beta-sub".to_string(), node_b.gateway_path.clone());
    install_cluster_remote_forwarder(&node_a.cluster, node_a.remote.clone(), members_a).unwrap();
    let mut members_b = HashMap::new();
    members_b.insert("alpha-sub".to_string(), node_a.gateway_path.clone());
    install_cluster_remote_forwarder(&node_b.cluster, node_b.remote.clone(), members_b).unwrap();

    let addr = Address::World(WorldAddr::ROOT);
    let sub_id = 42u64;
    let region = AABB::new(IVec3::new(0, 0, 0), IVec3::new(16, 16, 16));
    let subscribe = Envelope::new(
        0,
        addr,
        WorldRequest::Subscribe { addr, region, lod: Lod::new(0), sub_id },
    );
    let mut rx = node_a
        .cluster
        .subscribe(subscribe)
        .await
        .expect("cross-node subscribe");

    // The first event must be a BrickSnapshot streamed back from beta
    // through the cluster reply inbox. Time-out generously to avoid
    // flakes on a busy CI box.
    let snap = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("timed out waiting for brick snapshot")
        .expect("subscription channel closed");
    assert!(
        matches!(snap.body, WorldEvent::BrickSnapshot { .. }),
        "expected BrickSnapshot, got {:?}",
        snap.body
    );

    let _ = node_a.cluster.shutdown().await;
    let _ = node_b.cluster.shutdown().await;
    node_a.remote.shutdown().await;
    node_b.remote.shutdown().await;
    node_a.sys.terminate().await;
    node_b.sys.terminate().await;
}
