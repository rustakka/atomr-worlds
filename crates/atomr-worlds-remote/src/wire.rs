//! Wire envelopes carrying reply routing alongside the existing
//! [`atomr_worlds_proto::Envelope`] payload.

use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use serde::{Deserialize, Serialize};

/// Conventional actor name the [`WorldGateway`](crate::WorldGateway)
/// registers as. Clients address it as
/// `atomr://<system>@<host>:<port>/user/<GATEWAY_ACTOR_NAME>`.
pub const GATEWAY_ACTOR_NAME: &str = "world-gateway";

/// Conventional actor name the [`RemoteHost`](crate::RemoteHost) registers
/// for inbound replies/events. Servers address it back via this path.
pub const REPLY_INBOX_ACTOR_NAME: &str = "world-reply-inbox";

/// Wire request: a [`WorldRequest`] envelope plus the actor path the
/// server should send the reply (or streaming events) back to.
///
/// `auth_token` carries a pre-shared bearer token for the simple
/// shared-secret auth path (Phase 15 follow-up). The gateway validates
/// the token against its configured expected value and silently drops
/// requests that don't match. `None` means "no token sent" — gateways
/// configured with `expected_auth_token = None` accept that, gateways
/// configured with a token reject. **Tokens travel in plaintext until
/// the upstream `atomr-remote` TLS handshake lands**; treat as a
/// defense-in-depth knob on trusted networks, not a substitute for
/// transport encryption.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireRequest {
    pub reply_path: String,
    pub env: Envelope<WorldRequest>,
    #[serde(default)]
    pub auth_token: Option<String>,
}

impl WireRequest {
    /// Construct a request without auth (legacy / open-network path).
    pub fn new(reply_path: String, env: Envelope<WorldRequest>) -> Self {
        Self { reply_path, env, auth_token: None }
    }

    /// Attach a bearer token; the gateway will reject the request if
    /// its configured `expected_auth_token` doesn't match.
    pub fn with_auth(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }
}

/// Wire reply. One-shot replies match by `env.corr_id`; streaming events
/// match by `sub_id`. Splitting them keeps the client's routing map
/// unambiguous when a subscription's per-event `Envelope` carries
/// `corr_id = 0` (see `WorldActor::handle_subscribe_begin`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WireReply {
    /// Reply to a non-subscription request. Routed by `env.corr_id`.
    Reply { env: Envelope<WorldEvent> },
    /// One streamed event of an active subscription. Routed by `sub_id`.
    Event { sub_id: u64, env: Envelope<WorldEvent> },
}

/// Extract the `sub_id` from a [`WorldRequest::Subscribe`] /
/// [`WorldRequest::SubscribeMetric`]. Returns `None` for non-subscription
/// requests.
pub fn subscribe_sub_id(req: &WorldRequest) -> Option<u64> {
    match req {
        WorldRequest::Subscribe { sub_id, .. }
        | WorldRequest::SubscribeMetric { sub_id, .. } => Some(*sub_id),
        _ => None,
    }
}
