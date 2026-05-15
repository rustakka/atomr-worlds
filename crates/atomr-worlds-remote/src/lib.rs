//! Network transport for atomr-worlds.
//!
//! Provides two pieces:
//!
//! - [`RemoteHost`] — a [`WorldHost`](atomr_worlds_host::WorldHost) impl that
//!   speaks `Envelope<WorldRequest>` / `Envelope<WorldEvent>` over
//!   `atomr-remote`'s TCP transport. Clients construct one of these to talk
//!   to a `WorldGateway` running on another process or host.
//! - [`WorldGateway`] — a server-side actor that wraps an
//!   `Arc<dyn WorldHost>` (typically `LocalHost` or `ClusterHost`) and
//!   answers wire requests.
//!
//! The wire format is the existing bincode-serializable
//! [`atomr_worlds_proto::Envelope`] wrapped in [`wire::WireRequest`] /
//! [`wire::WireReply`] so reply routing can travel alongside the payload.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod cluster_forwarder;
pub mod gateway;
pub mod remote_host;
pub mod wire;

pub use cluster_forwarder::{
    install_cluster_remote_forwarder, install_cluster_remote_forwarder_with_auth,
    CLUSTER_REPLY_INBOX_NAME,
};
pub use gateway::{WorldGateway, WorldGatewayConfig};
pub use remote_host::{RemoteHost, RemoteHostConfig};
pub use wire::{WireReply, WireRequest, GATEWAY_ACTOR_NAME, REPLY_INBOX_ACTOR_NAME};
