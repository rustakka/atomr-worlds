# Client / Server Architecture (Phase 15)

Phase 14 ended with five view modes (`fp`, `tp`, `slice`, `rts`, `overview`) all
rendering through `atomr-worlds-view`'s CPU rasterizer — perfect for CI, but no
windowed binary anyone could *run*. Phase 15 closes that gap.

This document covers:

1. The three deployment topologies and how to launch each.
2. The wire protocol the client and server speak.
3. How clustering routes requests across nodes.
4. What's deferred to a future phase.

---

## Topologies

```
┌──────────────────────────────────────────────────────────────────────┐
│ atomr-worlds-client (Bevy app)                                       │
│   --backend local     in-process LocalHost                           │
│   --backend remote    RemoteHost → server gateway over atomr-remote  │
│   --backend cluster   same as remote; the receiving node forwards    │
│                       cross-shard requests internally                │
│                                                                      │
│   ViewMode dispatch: fp/tp native Bevy 3D; slice/rts/overview        │
│   render to a CPU Framebuffer that is blitted into a Bevy Image.     │
│   HUD: native bevy_ui TextBundles (FPS / coords / mode label).       │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ WorldHost trait (request / subscribe)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ atomr-worlds-server                                                  │
│   --mode standalone   one node, one ActorSystem, one LocalHost       │
│   --mode cluster      one ClusterHost; cross-node forwarding via     │
│                       atomr-remote                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Single-binary (in-process)

The Bevy client builds a multi-threaded tokio runtime, instantiates a
`LocalHost` on it, and feeds the host into a `WorldRuntime` resource. No
network. Easiest path for solo play and debugging.

```sh
cargo run -p atomr-worlds-client --release -- --backend local --seed 0xCAFE
```

### Client / standalone server

The server runs headless on top of `LocalHost` wrapped in a `WorldGateway`
actor that's exposed via `atomr-remote`. The client uses `RemoteHost`, which
spins up its own `RemoteSystem` for replies and routes one-shot requests via
correlation id, streaming events via subscription id.

```sh
# terminal A
cargo run -p atomr-worlds-server --release -- --bind 127.0.0.1:7800

# terminal B — `server_path` is the line the server prints
cargo run -p atomr-worlds-client --release -- \
    --backend remote \
    --connect 'atomr://atomr-worlds-server@127.0.0.1:7800/user/world-gateway'
```

### Clustered server (≥ 2 nodes)

Each node runs its own `ClusterHost` with a shared
`ShardCoordinator` view (today: passed in-process for tests; production
deployments need a coordinator backed by gossip or persisted ddata). Pass
`--peer <region_id>=<server_path>` for every peer.

```sh
# terminal A
cargo run -p atomr-worlds-server --release -- \
    --mode cluster --region-id alpha --bind 127.0.0.1:7800 \
    --peer beta=atomr://atomr-worlds-server@127.0.0.1:7801/user/world-gateway

# terminal B
cargo run -p atomr-worlds-server --release -- \
    --mode cluster --region-id beta --bind 127.0.0.1:7801 \
    --peer alpha=atomr://atomr-worlds-server@127.0.0.1:7800/user/world-gateway

# client connects to either member
cargo run -p atomr-worlds-client --release -- \
    --backend remote \
    --connect 'atomr://atomr-worlds-server@127.0.0.1:7800/user/world-gateway'
```

`--backend cluster` on the client is an alias for `--backend remote`: the
receiving cluster member runs its `ShardRegion` extractor on every request
and forwards to the owning shard via `atomr-remote` if that shard lives
elsewhere. The client never knows which node actually executed the request.

---

## Wire protocol

`atomr-worlds-proto::Envelope<WorldRequest>` and `Envelope<WorldEvent>` were
already bincode-serializable (Phase 7 widened `from` to `Address` precisely so
non-actor sources could fill the field). Phase 15 only adds reply routing
on top:

```rust
struct WireRequest {
    reply_path: String,                      // "atomr://A@h:p/user/<inbox>"
    env: Envelope<WorldRequest>,
}

enum WireReply {
    Reply { env: Envelope<WorldEvent> },     // matched by env.corr_id
    Event { sub_id: u64, env: Envelope<WorldEvent> },  // matched by sub_id
}
```

`Reply` matches on `corr_id` because every `Envelope<WorldRequest>` carries
one (`LocalHost::request` reuses the same field internally).

`Event` is distinct because subscription deltas are emitted with
`corr_id = 0` (`WorldActor::fan_out_delta`) — using `corr_id` to route
them would collide with one-shot replies. `sub_id` is unique per
subscription and threaded through the existing `Subscribe { sub_id }` /
`SubscribeMetric { sub_id }` requests.

Both `WireRequest` and `WireReply` are registered with the `RemoteSystem`'s
bincode codec on both ends — this is idempotent in the registry, so cluster
nodes that also host a `RemoteHost` client don't double-register.

---

## Cluster forwarding

`atomr-cluster-sharding`'s `ShardRegion::set_remote_forwarder` accepts a
closure `Fn(&str, M)` where `&str` is the *owning* region id and `M` is
the message type. For atomr-worlds, `M = Envelope<WorldRequest>`.

`atomr_worlds_remote::install_cluster_remote_forwarder`:

1. Registers the wire codecs on the local `RemoteSystem`.
2. Spawns a `ClusterReplyInbox` actor wired to `ClusterHost::pending_map()`.
   When the remote node ships back a `WireReply::Reply`, the inbox routes it
   by `corr_id` straight into the local `ClusterHost`'s oneshot waiter.
3. Builds the forwarder closure: it looks up the owner region in a
   user-supplied `HashMap<region_id, gateway_path>` and `tell`s the wrapped
   `WireRequest` to that gateway over `atomr-remote`.
4. Calls `region.set_remote_forwarder(forwarder)`.

The local `WorldGateway` is what the *peer* node's forwarder targets. When
node A forwards a `WireRequest` to node B's gateway, B's gateway calls
`B.cluster_host.request(env)`. B's cluster routes locally (B owns the
shard, so B's local entity handler runs), and B's gateway ships back the
`WireReply::Reply` to A's reply inbox. A's inbox unblocks the original
`A.cluster_host.request(env)` waiter via the corr-id-keyed pending map.

```
  client.request(env)                    A.local handler (skipped — not owner)
        │
        ▼
  A.cluster.request(env) ── corr_id = N ─▶ pending[N] = oneshot::Sender
        │
        ▼
  A.region.deliver(env)
        │
        ▼ (shard owned by B)
  forwarder → tell(WireRequest{reply_path=A.inbox, env}) to B.gateway
                                                            │
                                                            ▼
                                              B.cluster_host.request(env)
                                                            │
                                                            ▼
                                              B.region.deliver → B.local owns
                                                            │
                                                            ▼
                                              entity handler returns Envelope<WorldEvent>
                                                            │
                                                            ▼
                                              B.gateway.tell(A.inbox, WireReply::Reply{env'})
                                                            │
  oneshot.send(env') ◀── A.inbox routes by corr_id ─────────┘
        │
        ▼
  client receives env'
```

---

## Crates added

| crate                 | purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| `atomr-worlds-remote` | wire types, `RemoteHost`, `WorldGateway`, cluster forwarder builder |
| `atomr-worlds-server` | reusable `run_standalone` / `run_cluster` + the binary  |
| `atomr-worlds-client` | the Bevy app                                            |

`atomr-worlds-host` grew two pub accessors on `ClusterHost`
(`pending_map`, `actor_system`) and on `LocalHost` (`actor_system`) so the
remote crate can wire reply inboxes without owning the cluster host.

`atomr-worlds-host::LocalHostQuery` was generalised from
`Arc<LocalHost>` to `Arc<dyn WorldHost>` (the `new(Arc<LocalHost>, …)`
signature is preserved for backwards compatibility; new callers use
`from_dyn`).

---

## Verification

```sh
cargo test -p atomr-worlds-remote     # loopback + cluster forwarding
cargo test -p atomr-worlds-server     # standalone + cluster smoke
cargo test -p atomr-worlds-client     # headless WorldQuery bridge against both backends
cargo test --workspace                # full suite — no regressions
```

Manual visual check:

```sh
cargo run -p atomr-worlds-client --release -- --backend local
```

Controls in-window:

- `W A S D` move (Shift = sprint, Space / Ctrl = up/down).
- Mouse-look in fp/tp once the cursor is grabbed; `Esc` releases it.
- `1..=5` select view mode (fp / tp / slice / rts / overview).
- `Tab` cycles forward through the modes.
- Slice mode: `PgUp` / `PgDn` cycle the z-band.
- RTS mode: `Q` / `E` rotate, `+` / `-` zoom.
- Overview mode: `P` cycles projection; arrows pan; `+` / `-` zoom.

---

## Deferred to a future phase

- **Cross-node subscription routing.** `ClusterHost::subscribe` still falls
  back to `LocalHost::subscribe` on the receiving node — subscribers only
  see events emitted by *their* node's actors. `cluster.rs:140-149` and the
  rustdoc on `install_cluster_remote_forwarder` flag this. The wire format
  (`WireReply::Event { sub_id, env }`) is ready; the missing piece is a
  "subscription router" actor that pairs forwarded subscribes with a
  back-channel sub_id and replays cross-node deltas.
- **`atomr-view` UI bridge.** atomr-view's `SceneDescription` is still
  UI-only (no `Mesh` / `Camera` / `Renderer`), and its Bevy backend stubs
  `SetScene`. Pulling it as a client dep would also trigger a
  `path` vs `git` collision on `atomr-core`. The HUD is therefore rendered
  with native `bevy_ui` TextBundles. When upstream lands 3D primitives and
  a real Bevy backend, the HUD plugin in `crates/atomr-worlds-client/src/hud.rs`
  is where to swap implementations.
- **Cluster membership / discovery.** `--peer` is a static, manually
  supplied map. Production deployments need a shared `ShardCoordinator`
  (e.g. `DDataShardCoordinator`) and gossip-based membership.
- **TLS / auth.** `atomr-remote` supports both; this round leaves them
  off for LAN/dev simplicity.
