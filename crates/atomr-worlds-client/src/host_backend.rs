//! Picks the right [`WorldHost`] impl based on CLI flags.

use std::sync::Arc;
use std::time::Duration;

use atomr_worlds_host::{LocalHost, LocalHostConfig, WorldHost};
use atomr_worlds_remote::{RemoteHost, RemoteHostConfig};

use crate::cli::{Backend, Cli};

/// Build the chosen backend. Lives on a tokio runtime that must outlive
/// the returned `Arc<dyn WorldHost>`.
pub async fn build_backend(cli: &Cli) -> Result<Arc<dyn WorldHost>, String> {
    match cli.backend {
        Backend::Local => {
            let host = LocalHost::new(LocalHostConfig {
                root_seed: cli.seed,
                ..LocalHostConfig::default()
            })
            .await
            .map_err(|e| format!("LocalHost: {e}"))?;
            Ok(Arc::new(host))
        }
        Backend::Remote => {
            let server_path = cli
                .connect
                .as_deref()
                .ok_or("--connect <server_path> required for --backend remote")?;
            let host = RemoteHost::new(RemoteHostConfig {
                server_path: server_path.to_string(),
                bind: cli.bind,
                system_name: "atomr-worlds-client".into(),
                request_timeout: Duration::from_secs(10),
                subscriber_capacity: 256,
            })
            .await
            .map_err(|e| format!("RemoteHost: {e}"))?;
            Ok(Arc::new(host))
        }
        Backend::Cluster => {
            // From the client's perspective, joining a cluster is the same
            // as talking to a cluster member: the receiving node's
            // ClusterHost handles cross-node forwarding internally. We
            // reuse the remote path with --connect pointing at any
            // cluster member.
            let server_path = cli
                .connect
                .as_deref()
                .ok_or("--connect <member_server_path> required for --backend cluster")?;
            let host = RemoteHost::new(RemoteHostConfig {
                server_path: server_path.to_string(),
                bind: cli.bind,
                system_name: "atomr-worlds-client".into(),
                request_timeout: Duration::from_secs(10),
                subscriber_capacity: 256,
            })
            .await
            .map_err(|e| format!("RemoteHost (cluster): {e}"))?;
            Ok(Arc::new(host))
        }
    }
}
