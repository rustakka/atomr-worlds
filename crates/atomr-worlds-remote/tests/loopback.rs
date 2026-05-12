//! End-to-end loopback test: a real RemoteSystem on each side talks
//! to itself over `127.0.0.1` and round-trips Envelope<WorldRequest> /
//! Envelope<WorldEvent>.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use atomr_core::actor::{ActorSystem, Props};
use atomr_remote::{RemoteSettings, RemoteSystem};
use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_host::{LocalHost, LocalHostConfig, WorldHost};
use atomr_worlds_proto::{AABB, Envelope, WorldEvent, WorldRequest};
use atomr_worlds_remote::{
    RemoteHost, RemoteHostConfig, WireReply, WireRequest, WorldGateway, GATEWAY_ACTOR_NAME,
};
use atomr_worlds_voxel::Voxel;

async fn boot_server(seed: u64) -> (Arc<RemoteSystem>, ActorSystem, String) {
    let sys = ActorSystem::create("test-server", atomr_config::Config::reference())
        .await
        .expect("ActorSystem::create");
    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let remote = Arc::new(
        RemoteSystem::start(sys.clone(), bind, RemoteSettings::default())
            .await
            .expect("RemoteSystem::start"),
    );
    remote.register_bincode::<WireRequest>();
    remote.register_bincode::<WireReply>();

    let host: Arc<dyn WorldHost> = Arc::new(
        LocalHost::new(LocalHostConfig { root_seed: seed, ..LocalHostConfig::default() })
            .await
            .expect("LocalHost"),
    );
    let gateway_remote = remote.clone();
    let host_for_actor = host.clone();
    let gateway_ref = sys
        .actor_of(
            Props::create(move || {
                WorldGateway::new(host_for_actor.clone(), gateway_remote.clone())
            }),
            GATEWAY_ACTOR_NAME,
        )
        .expect("spawn gateway");
    remote.expose_actor(gateway_ref);

    let server_path = format!("{}/user/{}", remote.local_address, GATEWAY_ACTOR_NAME);
    (remote, sys, server_path)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn write_then_read_round_trip() {
    let (_server_remote, _server_sys, server_path) = boot_server(0xABCD).await;

    let client = RemoteHost::new(RemoteHostConfig {
        server_path: server_path.clone(),
        request_timeout: Duration::from_secs(5),
        ..RemoteHostConfig::default()
    })
    .await
    .expect("RemoteHost");

    let addr = Address::World(WorldAddr::ROOT);
    let pos = IVec3::new(1, 2, 3);

    // Write
    let write = Envelope::new(
        1,
        addr,
        WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(7) },
    );
    let resp = client.request(write).await.expect("write");
    assert!(matches!(resp.body, WorldEvent::Ack { .. }));

    // Read
    let read = Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos });
    let resp = client.request(read).await.expect("read");
    let WorldEvent::Voxel { voxel, .. } = resp.body else {
        panic!("expected Voxel reply, got {:?}", resp.body);
    };
    assert_eq!(voxel, Voxel::new(7));

    client.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_streams_initial_snapshots() {
    let (_server_remote, _server_sys, server_path) = boot_server(0xC0DE).await;

    let client = RemoteHost::new(RemoteHostConfig {
        server_path,
        request_timeout: Duration::from_secs(5),
        ..RemoteHostConfig::default()
    })
    .await
    .expect("RemoteHost");

    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(
        0,
        addr,
        WorldRequest::Subscribe {
            addr,
            region: AABB::new(IVec3::new(0, 0, 0), IVec3::new(16, 16, 16)),
            lod: atomr_worlds_core::Lod::new(0),
            sub_id: 42,
        },
    );
    let mut rx = client.subscribe(env).await.expect("subscribe");
    let snap = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("timeout")
        .expect("channel closed");
    assert!(matches!(snap.body, WorldEvent::BrickSnapshot { .. }));

    client.shutdown().await.expect("shutdown");
}
