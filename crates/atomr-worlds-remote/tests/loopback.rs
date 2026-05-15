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

/// Phase 15 follow-up — pre-shared bearer-token auth on the gateway.
///
/// Spin up a gateway that requires `expected_auth_token = "secret-99"`.
/// A client without the token (or with a wrong one) gets its requests
/// silently dropped — `request` times out instead of receiving a reply.
/// A client with the matching token round-trips normally.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gateway_rejects_requests_with_wrong_or_missing_token() {
    let sys = ActorSystem::create("test-server-auth", atomr_config::Config::reference())
        .await
        .unwrap();
    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let remote = Arc::new(
        RemoteSystem::start(sys.clone(), bind, RemoteSettings::default())
            .await
            .unwrap(),
    );
    remote.register_bincode::<WireRequest>();
    remote.register_bincode::<WireReply>();
    let host: Arc<dyn WorldHost> = Arc::new(LocalHost::new(LocalHostConfig::default()).await.unwrap());
    let host_for_actor = host.clone();
    let remote_for_actor = remote.clone();
    let gateway_ref = sys
        .actor_of(
            Props::create(move || {
                WorldGateway::new(host_for_actor.clone(), remote_for_actor.clone())
                    .with_auth_token("secret-99")
            }),
            GATEWAY_ACTOR_NAME,
        )
        .unwrap();
    remote.expose_actor(gateway_ref);
    let server_path = format!("{}/user/{}", remote.local_address, GATEWAY_ACTOR_NAME);

    // Client WITHOUT the token: request must time out.
    let bad_client = RemoteHost::new(RemoteHostConfig {
        server_path: server_path.clone(),
        request_timeout: Duration::from_millis(500),
        ..RemoteHostConfig::default()
    })
    .await
    .unwrap();
    let addr = Address::World(WorldAddr::ROOT);
    let req = Envelope::new(1, addr, WorldRequest::GetVoxel { addr, pos: IVec3::new(0, 0, 0) });
    let result = bad_client.request(req).await;
    assert!(result.is_err(), "request without auth should be rejected (timeout); got {result:?}");
    bad_client.shutdown().await.unwrap();

    // Client WITH the wrong token: still rejected.
    let wrong_client = RemoteHost::new(RemoteHostConfig {
        server_path: server_path.clone(),
        request_timeout: Duration::from_millis(500),
        auth_token: Some("not-the-secret".into()),
        ..RemoteHostConfig::default()
    })
    .await
    .unwrap();
    let req = Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos: IVec3::new(0, 0, 0) });
    let result = wrong_client.request(req).await;
    assert!(result.is_err(), "request with wrong token should be rejected; got {result:?}");
    wrong_client.shutdown().await.unwrap();

    // Client WITH the right token: round-trips normally.
    let good_client = RemoteHost::new(RemoteHostConfig {
        server_path: server_path.clone(),
        request_timeout: Duration::from_secs(5),
        auth_token: Some("secret-99".into()),
        ..RemoteHostConfig::default()
    })
    .await
    .unwrap();
    let req = Envelope::new(3, addr, WorldRequest::GetVoxel { addr, pos: IVec3::new(0, 0, 0) });
    let resp = good_client.request(req).await.expect("authorised request");
    assert!(matches!(resp.body, WorldEvent::Voxel { .. }));
    good_client.shutdown().await.unwrap();

    remote.shutdown().await;
    sys.terminate().await;
}
