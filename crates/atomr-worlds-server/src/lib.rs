//! Reusable server scaffolding for atomr-worlds. The binary entry point in
//! `main.rs` is a thin wrapper around [`run_standalone`] (and, once Step 10
//! lands, [`run_cluster`]).
//!
//! Exposed as a library so integration tests can drive the same code path
//! the binary uses, picking an OS-assigned port and connecting a
//! [`atomr_worlds_remote::RemoteHost`] against it.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use atomr_cluster_sharding::ShardCoordinator;
use atomr_core::actor::{ActorRef, ActorSystem, Props};
use atomr_remote::{RemoteSettings, RemoteSystem};
use atomr_worlds_host::{
    ClusterHost, ClusterHostConfig, HostError, LocalHost, LocalHostConfig, WorldHost,
};
use atomr_worlds_remote::{
    install_cluster_remote_forwarder, WireReply, WireRequest, WorldGateway, GATEWAY_ACTOR_NAME,
};

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("host error: {0}")]
    Host(#[from] HostError),
    #[error("actor system: {0}")]
    Sys(String),
    #[error("remote: {0}")]
    Remote(String),
}

/// Construction config for a standalone server.
#[derive(Clone, Debug)]
pub struct StandaloneConfig {
    pub bind: SocketAddr,
    pub system_name: String,
    pub root_seed: u64,
}

impl Default for StandaloneConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:7800".parse().unwrap(),
            system_name: "atomr-worlds-server".into(),
            root_seed: 0xDEAD_BEEF_CAFE_F00D,
        }
    }
}

/// A running standalone server. Drop it (or call [`Self::shutdown`]) to
/// stop the gateway and tear down the remote system.
pub struct StandaloneServer {
    pub sys: ActorSystem,
    pub remote: Arc<RemoteSystem>,
    pub host: Arc<dyn WorldHost>,
    pub server_path: String,
    pub local_address: String,
    _gateway: ActorRef<WireRequest>,
}

impl std::fmt::Debug for StandaloneServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StandaloneServer")
            .field("server_path", &self.server_path)
            .field("local_address", &self.local_address)
            .finish_non_exhaustive()
    }
}

impl StandaloneServer {
    pub async fn shutdown(self) -> Result<(), ServerError> {
        self.host.shutdown().await?;
        self.remote.shutdown().await;
        self.sys.terminate().await;
        Ok(())
    }
}

/// Bring up a standalone server: ActorSystem + RemoteSystem +
/// in-process LocalHost + a WorldGateway actor exposed on
/// `/user/world-gateway`. Returns once the gateway is reachable. Caller
/// is responsible for keeping the returned [`StandaloneServer`] alive.
pub async fn run_standalone(cfg: StandaloneConfig) -> Result<StandaloneServer, ServerError> {
    let sys = ActorSystem::create(cfg.system_name.clone(), atomr_config::Config::reference())
        .await
        .map_err(|e| ServerError::Sys(format!("{e}")))?;
    let remote = Arc::new(
        RemoteSystem::start(sys.clone(), cfg.bind, RemoteSettings::default())
            .await
            .map_err(|e| ServerError::Remote(format!("{e:?}")))?,
    );
    remote.register_bincode::<WireRequest>();
    remote.register_bincode::<WireReply>();

    let host: Arc<dyn WorldHost> = Arc::new(
        LocalHost::new(LocalHostConfig { root_seed: cfg.root_seed, ..LocalHostConfig::default() })
            .await?,
    );

    let host_for_actor = host.clone();
    let remote_for_actor = remote.clone();
    let gateway_ref = sys
        .actor_of(
            Props::create(move || {
                WorldGateway::new(host_for_actor.clone(), remote_for_actor.clone())
            }),
            GATEWAY_ACTOR_NAME,
        )
        .map_err(|e| ServerError::Sys(format!("spawn gateway: {e:?}")))?;
    remote.expose_actor(gateway_ref.clone());

    let local_address = remote.local_address.to_string();
    let server_path = format!("{}/user/{}", local_address, GATEWAY_ACTOR_NAME);
    tracing::info!(server_path = %server_path, "atomr-worlds-server listening");

    Ok(StandaloneServer {
        sys,
        remote,
        host,
        server_path,
        local_address,
        _gateway: gateway_ref,
    })
}

/// Configuration for a cluster server node.
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    pub bind: SocketAddr,
    pub system_name: String,
    pub root_seed: u64,
    /// This node's region id (must be unique within the cluster).
    pub region_id: String,
    /// Other nodes' gateways. Key = remote region_id, value = full
    /// `atomr://NAME@host:port/user/world-gateway` path.
    pub peers: HashMap<String, String>,
    pub request_timeout: Duration,
}

/// A running cluster node. Drop or call [`Self::shutdown`] to stop.
pub struct ClusterServer {
    pub sys: ActorSystem,
    pub remote: Arc<RemoteSystem>,
    pub cluster: Arc<ClusterHost>,
    pub coordinator: Arc<ShardCoordinator>,
    pub server_path: String,
    pub local_address: String,
    _gateway: ActorRef<WireRequest>,
}

impl std::fmt::Debug for ClusterServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterServer")
            .field("server_path", &self.server_path)
            .finish_non_exhaustive()
    }
}

impl ClusterServer {
    pub async fn shutdown(self) -> Result<(), ServerError> {
        self.cluster.shutdown().await?;
        self.remote.shutdown().await;
        self.sys.terminate().await;
        Ok(())
    }
}

/// Bring up one cluster node.
///
/// The caller is responsible for picking the right shard coordinator —
/// see [`ClusterServerOptions::coordinator`]. The default
/// [`run_cluster`] entry constructs a fresh local
/// [`ShardCoordinator`], which is correct for tests but not for
/// production multi-node deployments (each node would otherwise have
/// disjoint allocation tables).
pub async fn run_cluster(cfg: ClusterConfig) -> Result<ClusterServer, ServerError> {
    run_cluster_with(cfg, Arc::new(ShardCoordinator::new())).await
}

pub async fn run_cluster_with(
    cfg: ClusterConfig,
    coordinator: Arc<ShardCoordinator>,
) -> Result<ClusterServer, ServerError> {
    let sys = ActorSystem::create(cfg.system_name.clone(), atomr_config::Config::reference())
        .await
        .map_err(|e| ServerError::Sys(format!("{e}")))?;
    let remote = Arc::new(
        RemoteSystem::start(sys.clone(), cfg.bind, RemoteSettings::default())
            .await
            .map_err(|e| ServerError::Remote(format!("{e:?}")))?,
    );
    remote.register_bincode::<WireRequest>();
    remote.register_bincode::<WireReply>();

    let cluster = Arc::new(
        ClusterHost::new(ClusterHostConfig {
            region_id: cfg.region_id.clone(),
            coordinator: coordinator.clone(),
            local_config: LocalHostConfig { root_seed: cfg.root_seed, ..LocalHostConfig::default() },
            request_timeout: cfg.request_timeout,
        })
        .await?,
    );

    let cluster_for_gateway: Arc<dyn WorldHost> = cluster.clone();
    let remote_for_gateway = remote.clone();
    let gateway_ref = sys
        .actor_of(
            Props::create(move || {
                WorldGateway::new(cluster_for_gateway.clone(), remote_for_gateway.clone())
            }),
            GATEWAY_ACTOR_NAME,
        )
        .map_err(|e| ServerError::Sys(format!("spawn gateway: {e:?}")))?;
    remote.expose_actor(gateway_ref.clone());

    // Skip forwarder installation when peers is empty — the caller plans
    // to wire peers later (typical multi-node test pattern where both
    // nodes must boot before either knows the other's gateway path).
    if !cfg.peers.is_empty() {
        install_cluster_remote_forwarder(&cluster, remote.clone(), cfg.peers.clone())?;
    }

    let local_address = remote.local_address.to_string();
    let server_path = format!("{}/user/{}", local_address, GATEWAY_ACTOR_NAME);
    tracing::info!(
        region_id = %cfg.region_id,
        server_path = %server_path,
        peers = ?cfg.peers,
        "atomr-worlds-server cluster node listening"
    );

    Ok(ClusterServer {
        sys,
        remote,
        cluster,
        coordinator,
        server_path,
        local_address,
        _gateway: gateway_ref,
    })
}
