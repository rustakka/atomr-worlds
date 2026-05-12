//! Smoke test for the standalone server: boot it on `127.0.0.1:0`,
//! connect a RemoteHost, write+read a voxel, shut down cleanly.

use std::time::Duration;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_host::WorldHost;
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_remote::{RemoteHost, RemoteHostConfig};
use atomr_worlds_server::{run_standalone, StandaloneConfig};
use atomr_worlds_voxel::Voxel;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn server_round_trips_a_write_and_read() {
    let server = run_standalone(StandaloneConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        system_name: "test-server-standalone".into(),
        root_seed: 0xABCD,
    })
    .await
    .expect("server boot");

    let client = RemoteHost::new(RemoteHostConfig {
        server_path: server.server_path.clone(),
        request_timeout: Duration::from_secs(5),
        ..RemoteHostConfig::default()
    })
    .await
    .expect("RemoteHost");

    let addr = Address::World(WorldAddr::ROOT);
    let pos = IVec3::new(5, 7, 9);
    let write = Envelope::new(
        1,
        addr,
        WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(11) },
    );
    let resp = client.request(write).await.expect("write");
    assert!(matches!(resp.body, WorldEvent::Ack { .. }));

    let read = Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos });
    let resp = client.request(read).await.expect("read");
    let WorldEvent::Voxel { voxel, .. } = resp.body else {
        panic!("expected Voxel reply, got {:?}", resp.body);
    };
    assert_eq!(voxel, Voxel::new(11));

    client.shutdown().await.expect("client shutdown");
    server.shutdown().await.expect("server shutdown");
}
