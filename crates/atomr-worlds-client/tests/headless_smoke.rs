//! Headless smoke test for the client wiring.
//!
//! Builds the same `WorldRuntime` the binary builds — just without
//! `DefaultPlugins` (so it runs in CI without a display server). Verifies
//! the host_backend builder and the `WorldQuery` bridge work for both
//! `local` and `remote` backends.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, LocalHostConfig, LocalHostQuery, WorldHost};
use atomr_worlds_remote::{RemoteHost, RemoteHostConfig};
use atomr_worlds_server::{run_standalone, StandaloneConfig};
use atomr_worlds_view::WorldQuery;
use tokio::runtime::Builder;

#[test]
fn local_backend_returns_a_brick() {
    let rt = Arc::new(Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap());
    let host: Arc<dyn WorldHost> = rt.block_on(async {
        Arc::new(
            LocalHost::new(LocalHostConfig { root_seed: 0xC0DE, ..LocalHostConfig::default() })
                .await
                .unwrap(),
        )
    });
    let query = Arc::new(LocalHostQuery::from_dyn(host.clone(), rt.handle().clone()));

    // WorldQuery.brick uses block_on internally — call from a non-runtime
    // thread to avoid the "block_on inside a worker" deadlock.
    let q = query.clone();
    let handle = thread::spawn(move || {
        let brick = q.brick(&WorldAddr::ROOT, IVec3::new(0, 0, 0), Lod::new(0));
        assert!(brick.is_some(), "default LocalHost should fill brick (0,0,0)");
    });
    handle.join().expect("brick fetch thread");

    rt.block_on(async { host.shutdown().await.unwrap() });
}

#[test]
fn remote_backend_returns_a_brick() {
    let rt = Arc::new(Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap());
    let (server, client) = rt.block_on(async {
        let server = run_standalone(StandaloneConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            system_name: "client-smoke-server".into(),
            root_seed: 0xCAFE,
        })
        .await
        .unwrap();
        let client = RemoteHost::new(RemoteHostConfig {
            server_path: server.server_path.clone(),
            request_timeout: Duration::from_secs(5),
            ..RemoteHostConfig::default()
        })
        .await
        .unwrap();
        (server, client)
    });
    let host: Arc<dyn WorldHost> = Arc::new(client);
    let query = Arc::new(LocalHostQuery::from_dyn(host.clone(), rt.handle().clone()));

    let q = query.clone();
    let handle = thread::spawn(move || {
        let brick = q.brick(&WorldAddr::ROOT, IVec3::new(0, 0, 0), Lod::new(0));
        assert!(brick.is_some(), "remote backend should round-trip a brick");
    });
    handle.join().expect("remote brick fetch");

    rt.block_on(async {
        host.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    });
}
