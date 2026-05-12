# Implementation

Module-by-module map of the workspace. For the high-level model and design rationale, see
[ARCHITECTURE.md](ARCHITECTURE.md). For phase-specific landing notes, see
[PHASES.md](PHASES.md) and the per-phase sections at the bottom of this document.

## Workspace shape

| crate                          | purpose                                                                | atomr deps                            |
| ------------------------------ | ---------------------------------------------------------------------- | ------------------------------------- |
| `atomr-worlds-core`            | Coordinates, addressing, seeds, LOD                                    | none                                  |
| `atomr-worlds-voxel`           | Sparse voxel storage (brick + octree hybrid)                           | none                                  |
| `atomr-worlds-noise`           | Deterministic seeded noise (value/grad/Worley + FBM)                   | none                                  |
| `atomr-worlds-generate`        | Per-tier `Generator` impls; CPU `TerrainGenerator`; `BrickGenerator`   | none                                  |
| `atomr-worlds-accel`           | `Accelerator` trait + CPU backend; CUDA backend behind `cuda` feature  | atomr-accel-cuda (cuda feature only)  |
| `atomr-worlds-persist`         | `WorldPersistence` over `atomr-persistence` Journal + SnapshotStore    | atomr-persistence (+ -sql, optional)  |
| `atomr-worlds-proto`           | Wire-format messages and envelopes                                     | none                                  |
| `atomr-worlds-host`            | `WorldHost` trait, `LocalHost` (real, with persistence), `ClusterHost` (shell) | atomr, cluster, sharding, persistence |
| `atomr-worlds-view`            | Greedy meshing, MetricScale-driven camera, software rasterizer → PNG   | none                                  |
| `atomr-worlds-testkit`         | proptest strategies, cross-crate verification                          | none (dev-dep on host)                |
| `atomr-worlds-py`              | PyO3 bindings: `atomrworlds` Python package                            | none (transitive through host)        |

`core` exports re-export their submodules; consumers can use the flat path or the module path
interchangeably. Each crate has a `thiserror` error enum named after the crate.

## atomr-worlds-core

### `IVec3` and per-level newtypes

[`crates/atomr-worlds-core/src/coord.rs`](../crates/atomr-worlds-core/src/coord.rs)

`IVec3 { x: i64, y: i64, z: i64 }` is the canonical integer vector. `i64` is required because
voxel coordinates at meter resolution exceed `i32` at galactic scales (Milky Way diameter at 1m
voxels would already need `i64`).

Per-tier `#[repr(transparent)]` newtypes (`UniverseCoord`, `GalaxyCoord`, `SectorCoord`,
`SystemCoord`, `WorldCoord`, `BrickCoord`, `VoxelCoord`) prevent accidentally passing a galaxy
coord where a world coord is expected, with zero runtime cost.

### `WorldAddr` and `LevelKey`

[`crates/atomr-worlds-core/src/addr.rs`](../crates/atomr-worlds-core/src/addr.rs)

```rust
pub struct LevelKey { pub coord: IVec3, pub dim: DimensionId }

pub struct WorldAddr {
    pub universe: LevelKey,
    pub galaxy:   LevelKey,
    pub sector:   LevelKey,
    pub system:   LevelKey,
    pub world:    LevelKey,
}
```

`Copy + Hash + Eq + Serialize + Deserialize`. `WorldAddr::ancestor(Level)` truncates lower
tiers to `LevelKey::ROOT`; `WorldAddr::level_key(Level)` reads a specific tier.

### Seed derivation

[`crates/atomr-worlds-core/src/seed.rs`](../crates/atomr-worlds-core/src/seed.rs)

```rust
pub const fn splitmix64(z: u64) -> u64;
pub const fn child_seed(parent: u64, dim: u32, coord: IVec3) -> u64;
```

Both `const fn`. `child_seed` folds parent → dim → x → y → z through `splitmix64` with rotated
mixing constants on the coordinate axes; this prevents simple permutations of `(x, y, z)` from
colliding. `WorldAddr::seed_chain(root: u64) -> [u64; 5]` walks the five tiers and produces the
full chain.

### LOD

[`crates/atomr-worlds-core/src/lod.rs`](../crates/atomr-worlds-core/src/lod.rs)

```rust
pub struct Lod { pub depth: u8 }
pub struct MetricScale { pub root_size_m: f64, pub max_depth: u8 }
impl MetricScale {
    pub const DEFAULT_UNIVERSE: Self; // 1e27 m / depth 64
    pub const DEFAULT_GALAXY:   Self; // 1e21 m / depth 56
    pub const DEFAULT_SECTOR:   Self; // 1e18 m / depth 48
    pub const DEFAULT_SYSTEM:   Self; // 1e13 m / depth 40
    pub const DEFAULT_WORLD:    Self; // 1e7  m / depth 24
    pub fn meters_per_voxel(&self, lod: Lod) -> f64; // = root / exp2(depth)
    pub fn leaf_size_m(&self) -> f64;
    pub fn lod_for_screen(&self, distance_m: f64, focal_px: f64, target_px_per_voxel: f64) -> Lod;
}
```

`meters_per_voxel` uses `f64::exp2` rather than `1u64 << depth` — the latter UB's at depth
64 (universe default).

### Hierarchy structs

[`crates/atomr-worlds-core/src/hierarchy.rs`](../crates/atomr-worlds-core/src/hierarchy.rs)

`Universe`, `Galaxy`, `Sector`, `System`, `World` — each is just data
(`{ addr, seed, scale }`). The `Generator<Output = T, Err = E>` trait is a single-method shape:
`fn generate(&self, seed: u64, addr: WorldAddr) -> Result<T, E>`. Bodies live downstream.

## atomr-worlds-voxel

### `Voxel`

[`crates/atomr-worlds-voxel/src/voxel.rs`](../crates/atomr-worlds-voxel/src/voxel.rs)

```rust
#[repr(transparent)]
pub struct Voxel(pub u16); // material id; 0 = empty (Voxel::EMPTY)
```

`u16` gives 65 535 distinct materials with a 1 reserved for empty — enough for any palette
the next phase will design, while keeping bricks at 8 KiB.

### `Brick` (16³ dense voxel block)

[`crates/atomr-worlds-voxel/src/brick.rs`](../crates/atomr-worlds-voxel/src/brick.rs)

```rust
pub const BRICK_EDGE: usize = 16;          // 16³ = 4096 voxels
pub const BRICK_LEN:  usize = BRICK_EDGE.pow(3);

pub struct Brick {
    pub voxels: Box<[Voxel; BRICK_LEN]>,
    pub nonempty_count: u16,
}
```

`Box<[Voxel; 4096]>` keeps the brick at 8 bytes of inline overhead with the data on the heap —
avoids large `memcpy` on `Brick: Clone`. `nonempty_count` is maintained on every `set` so
empty-brick detection is O(1).

`local_index` uses xyz-then-flat (`(z * 16 + y) * 16 + x`) — the cache-friendly order for the
typical "walk a yz slice" rendering pattern.

### `Octree`

[`crates/atomr-worlds-voxel/src/octree.rs`](../crates/atomr-worlds-voxel/src/octree.rs)

Arena-allocated; one `Vec<NodeKind>` and one `Vec<Brick>`.

```rust
pub type NodeId = u32;
pub struct InternalNode { pub child_mask: u8, pub children_base: u32 }
pub enum NodeKind { Empty, Internal(InternalNode), Leaf(NodeId /* brick arena idx */) }
pub struct Octree {
    pub root_size_m: f64,
    pub max_depth:   u8,
    pub nodes:       Vec<NodeKind>,
    pub bricks:      Vec<Brick>,
    // ...probe counter for tests
}
```

**Child layout.** Each internal node carries an 8-bit `child_mask` (one bit per octant) and a
`children_base` arena offset. The k-th child lives at `children_base + popcount(child_mask &
((1 << k) - 1))`. Inserting a missing octant copies the existing popcount-many siblings to a
fresh arena slice and inserts the new node at the popcount-determined slot. No "spare slots,"
no `Option<NodeId>; 8` waste.

**Coordinate convention.** The octree's voxel grid is recentred so brick coords lie in
`[0, 2^max_depth)` during traversal. The valid voxel range per axis is
`-(2^max_depth × 8) .. (2^max_depth × 8)` — i.e. for `max_depth = 4`, that's `-128 .. 128`.

**Probe counter.** `Octree::probe_count` is a `Cell<u64>` incremented on every node-arena
read during traversal. `reset_probes` / `probes` are `#[doc(hidden)]` test hooks used by the
empty-space-skip assertions in `tests/oracle_stress.rs`.

### `SparseVoxelStore`

[`crates/atomr-worlds-voxel/src/store.rs`](../crates/atomr-worlds-voxel/src/store.rs)

```rust
pub trait SparseVoxelStore {
    fn get(&self, p: IVec3) -> Result<Voxel, VoxelError>;
    fn set(&mut self, p: IVec3, v: Voxel) -> Result<(), VoxelError>;
    fn brick(&self, brick_coord: IVec3) -> Result<Option<&Brick>, VoxelError>;
    fn root_size_m(&self) -> f64;
    fn max_depth(&self)   -> u8;
}
```

`Octree` is the only implementor at phase 0. The trait exists so downstream code can swap in
specialized storages (LOD pyramids, GPU-resident bricks, in-memory ring buffers) without
changing call sites.

## atomr-worlds-proto

### Wire messages

[`crates/atomr-worlds-proto/src/messages.rs`](../crates/atomr-worlds-proto/src/messages.rs)

```rust
pub enum WorldRequest {
    GetVoxel { addr: WorldAddr, pos: IVec3 },
    GetBrick { addr: WorldAddr, brick: IVec3, lod: Lod },
    Subscribe { addr: WorldAddr, region: AABB, lod: Lod, sub_id: u64 },
    Unsubscribe { sub_id: u64 },
}

pub enum WorldEvent {
    BrickSnapshot { addr: WorldAddr, brick: IVec3, lod: Lod, payload: bytes::Bytes },
    VoxelDelta    { addr: WorldAddr, pos: IVec3, before: Voxel, after: Voxel },
    StreamEnd     { sub_id: u64 },
}
```

All variants `derive(Serialize, Deserialize)`. `bytes::Bytes` is used for brick payloads so
zero-copy framing is possible later.

### Envelope

[`crates/atomr-worlds-proto/src/envelope.rs`](../crates/atomr-worlds-proto/src/envelope.rs)

```rust
pub struct Envelope<T> { pub corr_id: u64, pub from: WorldAddr, pub body: T }
```

`from` is the address of the source actor (used by `WorldExtractor` to route the
`Unsubscribe` variant, which doesn't carry an address in its body).

### Wire codec

[`crates/atomr-worlds-proto/src/wire.rs`](../crates/atomr-worlds-proto/src/wire.rs)

```rust
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, ProtoError>;
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ProtoError>;
```

Wraps `bincode::serde::{encode_to_vec, decode_from_slice}` with
`bincode::config::standard()`. Same major as atomr's `[workspace.dependencies]`, so a bridging
process shares one codec.

## atomr-worlds-host

### `WorldHost` trait

[`crates/atomr-worlds-host/src/host.rs`](../crates/atomr-worlds-host/src/host.rs)

```rust
#[async_trait]
pub trait WorldHost: Send + Sync + 'static {
    async fn request(&self, env: Envelope<WorldRequest>)
        -> Result<Envelope<WorldEvent>, HostError>;
    async fn subscribe(&self, env: Envelope<WorldRequest>)
        -> Result<mpsc::Receiver<Envelope<WorldEvent>>, HostError>;
    async fn shutdown(&self) -> Result<(), HostError>;
}
```

### `LocalHost` and `ClusterHost`

[`crates/atomr-worlds-host/src/local.rs`](../crates/atomr-worlds-host/src/local.rs),
[`crates/atomr-worlds-host/src/cluster.rs`](../crates/atomr-worlds-host/src/cluster.rs)

`LocalHost` is fully implemented (Phase 1). It owns an `atomr::ActorSystem` and spawns one
`WorldActor` per `WorldAddr` on first access (lazy, cached in an `Arc<Mutex<HashMap<…>>>`).
Each actor owns its brick cache, a user-write overlay, the subscriber registry, and — when
`LocalHostConfig::persistence` is set — an `Arc<WorldPersistence>` handle (Phase 3). Recovery
runs before the actor spawns: the persisted overlay and `last_seq` are passed into
`WorldActor::new`, and the first cache miss for any brick reapplies the overlay on top of the
procedural baseline.

`LocalHostConfig` carries:

- `root_seed: u64` — base seed for the address-derived chain.
- `world_gen: WorldGen` — produces the per-world `TerrainGenerator`.
- `subscriber_capacity: usize` — bound for the per-subscription `mpsc` channel.
- `request_timeout: Duration` — `ask_with` timeout for `request`.
- `persistence: Option<Arc<WorldPersistence>>` — optional Phase 3 binding.

`ClusterHost` remains a placeholder pending the upstream sharding wire-up; the `WorldExtractor`
that routes `Envelope<WorldRequest>` into a `ShardRegion` is implemented and tested.

### `WorldExtractor`

[`crates/atomr-worlds-host/src/extractor.rs`](../crates/atomr-worlds-host/src/extractor.rs)

Fully implemented; not stubbed.

```rust
impl MessageExtractor for WorldExtractor {
    type Message = Envelope<WorldRequest>;
    fn entity_id(&self, m: &Self::Message) -> String;
    fn shard_id (&self, m: &Self::Message) -> String;
}
```

- `shard_id_for(addr)` packs
  `u:{ux},{uy},{uz}:{udim}|g:{gx},{gy},{gz}|s:{sx},{sy},{sz}`. Two distinct systems within the
  same sector produce identical shard ids → co-resident on the shard owner.
- `entity_id_for(addr)` packs the full five-tier path, so each world is its own entity within
  the sector's shard.
- `Unsubscribe` envelopes don't carry an address in their body — routing falls back to
  `envelope.from`.

### Errors

```rust
pub enum HostError {
    Voxel(VoxelError),
    Proto(ProtoError),
    Core(WorldsCoreError),
    Sys(String),         // atomr ActorSystem / persistence errors
    Ask(String),         // ask_with timeout / receiver-dropped failures
    SubscribeFailed,     // the sink dropped before initial snapshot finished
    Shutdown,
    NotYetImplemented(&'static str),
}
```

## atomr-worlds-testkit

[`crates/atomr-worlds-testkit/src/strategies.rs`](../crates/atomr-worlds-testkit/src/strategies.rs)

proptest strategies: `arb_ivec3`, `arb_level_key`, `arb_world_addr`, `arb_lod(max_depth)`,
`arb_voxel`, `arb_brick`. `arb_brick` produces sparse-ish bricks (0–64 writes from a 4096-cell
space) so the HashMap-oracle test exercises both empty and populated regions.

### Test surface

| location                                                       | what it checks                                                              |
| -------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `atomr-worlds-core` unit tests                                 | Coord newtypes transparent, seed determinism, dim discrimination, LOD math  |
| `atomr-worlds-voxel` unit tests (`brick`, `octree`)            | Brick round-trip, empty-count invariant, octree set→get, OOB error, sparse probes |
| `atomr-worlds-voxel/tests/oracle_stress.rs`                    | 5 000 random writes match HashMap oracle; sparse tree probe budget per read |
| `atomr-worlds-testkit/tests/cross_crate.rs`                    | `WorldAddr` bincode + JSON round-trips; brick proptest oracle; protocol round-trips |
| `atomr-worlds-testkit/tests/hash_quality.rs`                   | Avalanche ratio ≥ 0.40 across 5 perturbation sites; low-byte uniformity within ±12% (5σ) |
| `atomr-worlds-testkit/tests/extractor_stable.rs`               | Shard id and entity id stable; sibling systems share shard id              |
| `atomr-worlds-host/tests/local_e2e.rs`                         | LocalHost request/write/subscribe-snapshot/subscribe-delta; out-of-region filtering |
| `atomr-worlds-host/tests/persistence_e2e.rs`                   | Writes survive host restart; snapshot fires every N writes; journal tail replays |
| `atomr-worlds-persist` unit tests                              | Snapshot+tail recovery; empty-voxel clears overlay; persistence id stability |
| `atomr-worlds-accel` unit tests                                | `CpuAccelerator` matches direct `BrickGenerator`; batched-fill default impl  |
| `atomr-worlds-accel/tests/cuda_determinism.rs` (`--ignored`)   | CUDA bricks match CPU bricks byte-for-byte; GPU runs are idempotent          |
| `atomr-worlds-view/tests/deterministic_screenshot.rs`          | FNV-1a hash of pixels equal across runs; pinned hash matches reversed-z output; non-background pixels present |
| `atomr-worlds-view/tests/skybox.rs`                            | Cube-face basis orthonormal/right-handed; cubemap sampling scale-invariant; empty/non-empty skybox digest deterministic and observer-sensitive; reversed-z near→1, far→0 |

## Example binary

[`examples/print-seed-chain/src/main.rs`](../examples/print-seed-chain/src/main.rs)

Smoke test. Builds a sample `WorldAddr`, prints its `seed_chain(0xDEAD_BEEF_CAFE_F00D)`, and
tabulates the default `MetricScale` for each tier (root edge / leaf voxel size). Run:

```sh
cargo run -p print-seed-chain
```

Expected leaf sizes: universe ~54 Mm, galaxy ~14 km, sector ~3.5 km, system ~9 m, world ~60 cm.

## Conventions

- **Cargo workspace** with resolver v2; centralized `[workspace.dependencies]` for version
  pinning. Mirrors atomr exactly.
- **Errors** with `thiserror`, one enum per crate, no top-level mega-error. `WorldsCoreError`
  is re-exported with a `Result<T>` alias as it's the most-used; downstream crates use
  `Result<T, ThisCrateError>` directly so the error type stays visible at call sites.
- **No `unsafe`.** `#![forbid(unsafe_code)]` at each crate root.
- **`const fn` aggressively.** `splitmix64`, `child_seed`, `WorldAddr::seed_chain`, `Lod::new`,
  most constructors. Const-evaluable seed chains enable static dispatch downstream.
- **rustfmt**: `max_width = 110`, `use_small_heuristics = "Max"`. Mirrors atomr's style.
- **Lints**: `cargo clippy --workspace --all-targets -- -D warnings` is the gate.

## Phase 1 (landed)

`Generator` trait impls for all five tiers in
[`atomr-worlds-generate/src/tiers.rs`](../crates/atomr-worlds-generate/src/tiers.rs);
voxel content comes from
[`TerrainGenerator`](../crates/atomr-worlds-generate/src/terrain.rs) (FBM-driven
heightfield + Worley caves + dirt layer). `LocalHost` in
[`atomr-worlds-host/src/local.rs`](../crates/atomr-worlds-host/src/local.rs) wires
a real `atomr::ActorSystem` and spawns one `WorldActor` per address (lazy, cached).

## Phase 4 (landed)

`LocalHost::subscribe` returns an `mpsc::Receiver<Envelope<WorldEvent>>`. On
subscribe, the actor emits an initial `BrickSnapshot` for every brick that overlaps
the requested AABB. Subsequent `WriteVoxel` requests inside the region produce
`VoxelDelta` events. Backpressure policy is **drop subscriber on full channel** —
the writer never blocks. Tests in
[`atomr-worlds-host/tests/local_e2e.rs`](../crates/atomr-worlds-host/tests/local_e2e.rs)
cover read, write, subscribe-snapshot, subscribe-delta, and out-of-region filtering.

## Phase 6 (landed) — Python bindings

[`atomr-worlds-py`](../crates/atomr-worlds-py/) exposes a PyO3 cdylib called
`atomrworlds_native`, wrapped by the `atomrworlds` Python package. The
`WorldClient` class is a `LocalHost`-backed query interface. Build with
`maturin develop -m crates/atomr-worlds-py/Cargo.toml` inside a venv. Smoke tests
at `crates/atomr-worlds-py/python/tests/test_smoke.py`.

## Phase 3 (landed) — Persistence

[`atomr-worlds-persist`](../crates/atomr-worlds-persist/) wraps
`atomr_persistence::{Journal, SnapshotStore}` with world-specific encoding:
`VoxelWriteEvent`s are bincode-encoded and journalled, `WorldSnapshot`s capture
the per-world write overlay. `WorldPersistence` is the consumer-facing handle;
`InMemoryJournal` + `InMemorySnapshotStore` are re-exported from
atomr-persistence for the default in-memory backend, and the `sql` feature
pulls in `atomr-persistence-sql`'s `SqlJournal` + `SqlSnapshotStore` (SQLite by
default; Postgres / MySQL / MSSQL via sqlx feature flags).

`LocalHostConfig` grows a `persistence: Option<Arc<WorldPersistence>>` field.
When set, `LocalHost::world_actor_for` runs recovery before spawning the actor;
the `WorldActor` appends each `WriteVoxel` to the journal before applying it
locally and triggers `save_snapshot` every `snapshot_every` writes (default 64).
End-to-end coverage in
[`atomr-worlds-host/tests/persistence_e2e.rs`](../crates/atomr-worlds-host/tests/persistence_e2e.rs):
write voxels through one host, drop it, recover state through a fresh host,
verify reads match.

## Phase 5 (landed) — GPU acceleration

[`atomr-worlds-accel`](../crates/atomr-worlds-accel/) gains a `cuda` feature
that pulls in `atomr-accel-cuda` (with `nvrtc`) and `cudarc`. The `CudaAccelerator`
spins up a `DeviceActor` with `EnabledLibraries::NVRTC`, compiles
[`cuda_kernel.cu`](../crates/atomr-worlds-accel/src/cuda_kernel.cu) — a faithful
port of the CPU `TerrainGenerator` math — at construction, and dispatches one
NVRTC launch per `fill_bricks_batch`. The host compiles with `--fmad=false`
so FMA fusion does not drift last-bit results; the kernel and the CPU path
produce byte-identical bricks.

Determinism gate:
[`tests/cuda_determinism.rs`](../crates/atomr-worlds-accel/tests/cuda_determinism.rs)
compares CPU and GPU brick payloads byte-for-byte across a representative coord
mix. Gated `#[ignore]` so CUDA-less hosts still pass; run with
`cargo test -p atomr-worlds-accel --features cuda -- --ignored`.

Bench:
[`benches/cpu_vs_gpu.rs`](../crates/atomr-worlds-accel/benches/cpu_vs_gpu.rs)
(Criterion) compares CPU vs GPU on 1, 8, 64, 256-brick batches. Run with
`cargo bench -p atomr-worlds-accel --features cuda --bench cpu_vs_gpu`.

## Phase 2 (landed) — Renderer integration

[`atomr-worlds-view`](../crates/atomr-worlds-view/) ships a CPU renderer with
three modules: [`mesh`](../crates/atomr-worlds-view/src/mesh.rs) (greedy
meshing of a `Brick` into axis-aligned face quads), [`camera`](../crates/atomr-worlds-view/src/camera.rs)
(perspective `Camera` with `MetricScale::lod_for_screen` integration via
`Camera::pick_lod`), and [`render`](../crates/atomr-worlds-view/src/render.rs)
(half-space triangle rasterizer with a z-buffer, deterministic by
construction). [`render_brick_png`] is the convenience entry point.

Deterministic screenshot gate at
[`tests/deterministic_screenshot.rs`](../crates/atomr-worlds-view/tests/deterministic_screenshot.rs):
rendering the same brick from the same seed twice produces byte-identical
pixel buffers (FNV-1a hash). The [`examples/view-png`](../examples/view-png)
demo wires it to `LocalHost`, fetches a 4×4 slab of bricks across six vertical
tiles (`Y_TILES_BOT = -2` through `Y_TILES_TOP = 3`), greedy-meshes them in
world-local coordinates, and writes a 512×512 isometric perspective PNG.

The upstream-bridge piece of Phase 2 — handing meshes off to `atomr-view`'s
scene API — is blocked: `atomr-view`'s `SceneDescription` is UI-only (no
`Mesh`/`Camera`/`Renderer`/headless path), and the `winit+wgpu` backend in
`atomr-view-backends` is stubbed. Once the upstream scene API grows 3D
primitives, `mesh::greedy_mesh`'s output drops straight into them.

## Phase 13f (landed) — Skybox + reversed-z

[`crates/atomr-worlds-view/src/camera.rs`](../crates/atomr-worlds-view/src/camera.rs)
flips `perspective(fov_y, aspect, near, far)` from the standard `[0, 1]`
forward-z convention to **reversed-z**: a vertex at the near plane projects to
depth `1.0`, a vertex at the far plane to `0.0`. The change is a
two-row swap in the projection matrix (`[2][2] = -near*nf`, `[3][2] =
-far*near*nf` instead of `far*nf` and `far*near*nf`). The rasterizer's
companion changes live in
[`render.rs`](../crates/atomr-worlds-view/src/render.rs):
`Framebuffer.depth` is now initialised to `0.0` (the far plane under
reversed-z), and the z-buffer compare is `z > fb.depth[idx]` so the
closer fragment (larger depth) wins.

Why reversed-z? Standard `[0, 1]` depth wastes most of the f32 mantissa
on the near third of the view frustum because `1/z` post-perspective
divide compresses far values. Reversed-z plus an f32 depth buffer is
the well-known fix: it spreads precision evenly across the buffer so
celestial-body silhouettes at the far horizon stay stable against
near-field terrain. The Phase 13f skybox needs this property — the
skybox sits at the world's outer shell, and without reversed-z any
parent-tier mesh capture would z-fight against background gradient.

The
[`tests/deterministic_screenshot.rs`](../crates/atomr-worlds-view/tests/deterministic_screenshot.rs)
pinned hash is bumped to `0x71cc_a39a_1edb_1595`, matching the
reversed-z output. The pre-existing run-to-run determinism assertion
is unchanged; the new `pinned_hash_matches_current_render` test
catches future drift in either the renderer or the terrain generator.

[`crates/atomr-worlds-view/src/skybox.rs`](../crates/atomr-worlds-view/src/skybox.rs)
adds the cubemap pipeline:

```rust
pub enum CubeFace { PosX, NegX, PosY, NegY, PosZ, NegZ }
impl CubeFace {
    pub const ALL: [CubeFace; 6];
    pub fn forward(self) -> [f32; 3];
    pub fn up(self)      -> [f32; 3];
    pub fn right(self)   -> [f32; 3];
}
pub struct CubeFaceImage { pub width: u32, pub height: u32, pub pixels: Vec<u8> }
pub struct Skybox {
    pub faces: [CubeFaceImage; 6],
    pub origin: [f64; 3], pub inner_radius_m: f64, pub outer_radius_m: f64,
    pub captured_seed: u64, pub face_resolution: u32, pub digest: u64,
}
pub struct SkyboxConfig {
    pub face_resolution: u32, pub background_color: [u8; 4],
    pub include_parent_tier: bool,
}
pub fn render_skybox_from_meshes(
    meshes: &[MeshNode], observer: [f64; 3],
    inner_radius_m: f64, outer_radius_m: f64,
    captured_seed: u64, cfg: &SkyboxConfig,
) -> Skybox;
impl Skybox {
    pub fn sample(&self, dir_unit: [f32; 3]) -> [u8; 4];
    pub fn compute_digest(&self) -> u64;
}
```

`CubeFace::forward/up/right` form a right-handed orthonormal frame on
each face (`cross(right, up) == forward`). `Skybox::sample` is the
standard largest-axis-picks-the-face cubemap fetch and is
scale-invariant: `sample(dir) == sample(k * dir)` for any `k > 0`.
The digest is an FNV-1a over the concatenated face pixel buffers, in
`CubeFace::ALL` order.

`Camera::for_cube_face(eye, face, near, far)` returns a 90° FOV /
aspect 1.0 camera looking along the face's outward normal — six of
those tile the full sphere with no overlap and no gap.
`render_skybox_from_meshes` walks the six faces in order, combines
all `MeshNode`s into one transform-baked mesh per face (cheap because
the mesh-node count for a skybox capture is small), and calls
`render_mesh` once per face. The depth buffer is local to each face
call, so the rasterizer state stays single-pass and the result is
deterministic across runs.

Phase 13f intentionally stops at the mesh-input boundary. A
`WorldHost`-pulling wrapper that fetches the parent-tier brick slab
and feeds it into `render_skybox_from_meshes` lands in Phase 13g/13i
alongside the streaming-proto changes for skybox bursts. Keeping the
13f surface mesh-only makes the test file at
[`tests/skybox.rs`](../crates/atomr-worlds-view/tests/skybox.rs)
self-contained: seven tests covering the cube-face basis, sampling,
empty / non-empty rendering, digest determinism, observer
sensitivity, and the reversed-z projection sanity check, none of
which need `LocalHost`.

See [`PHASES.md`](PHASES.md) for the full roadmap.

## Phase 13a (landed) — `WorldShape` type

[`atomr-worlds-core/src/shape.rs`](../crates/atomr-worlds-core/src/shape.rs)
defines `WorldShape::{Cube { edge_m }, Sphere { radius_m }, Cylinder { radius_m, height_m }}`
plus `ShapeAabb` (continuous-meter centered bounding box, distinct from
the integer-voxel `proto::AABB`). Methods: `contains(p)`,
`horizon_distance_m(altitude)`, `surface_normal_at(p)`,
`bounding_aabb()`, `radius_m()`, `surface_area_m2()`, `wrap(p)`.
Manual `Hash`/`Eq`/`PartialEq` via `f64::to_bits()` for cache-keyability.
Embedded in `World` (`hierarchy.rs`) and `WorldGen` (`tiers.rs`) with
`Default = Cube { edge_m: 1.0e7 }` for back-compat. The horizon formula
is `sqrt(2*R*h + h²)` for sphere/cylinder; cube returns `f64::INFINITY`.

## Phase 13b (landed) — Horizon-clamped streaming + brick filter

- [`MetricScale::lod_for_screen_curved`](../crates/atomr-worlds-core/src/lod.rs)
  and [`StreamingPolicy::ring_for_curved`](../crates/atomr-worlds-proto/src/streaming.rs)
  add a `horizon_m` parameter that clamps the streaming radius.
- [`crate::shape::{ShapeResolver, DefaultShape, PrefixShape}`](../crates/atomr-worlds-host/src/shape.rs)
  mirror the policy resolver — hierarchical address → shape lookup,
  default cubic Earth-class.
- [`LocalHostConfig::shape_resolver: Arc<dyn ShapeResolver>`](../crates/atomr-worlds-host/src/local.rs)
  resolves shape once per actor on spawn; `WorldActor::brick_inside_shape`
  short-circuits out-of-shape bricks to empty without invoking the
  generator. `handle_subscribe_begin` consults the observer's altitude
  (`observer.length() - shape.radius()`), passes it to
  `ring_for_curved`, and stores a `MetricSubState` per subscriber.
- `UpdateObserverPos` recomputes the ring and emits a fresh `Tier` event
  plus `BrickSnapshot`s for any newly-visible bricks.

Tests: [`tests/sphere_horizon_e2e.rs`](../crates/atomr-worlds-host/tests/sphere_horizon_e2e.rs)
covers the horizon clamp, out-of-shape filter, observer-tick deltas,
and cross-host determinism.

## Phase 13c (landed) — Geologic macro pre-sim + `BrickGenContext`

[`atomr-worlds-generate/src/macro_state/`](../crates/atomr-worlds-generate/src/macro_state/)
ships a three-layer pre-pass:

- `surface_grid.rs` — `SurfaceGrid::new(level)` builds a recursive
  icosahedron at `20 * 4^level` faces. Each face has 3 edge-neighbours
  (table-driven, O(1)), and `face_for_direction(unit)` finds the
  containing face by best-centroid dot product. Determinism: pure f64
  arithmetic from a hard-coded golden-ratio icosahedron base.
- `plates.rs` — `generate_plates(grid, seed, config)` picks
  `plate_count` distinct face seeds via `splitmix64(seed ^ i)`,
  flood-fills via multi-source BFS with sorted-id collision resolution
  (true distance-Voronoi, no race), assigns per-plate velocities, and
  computes elevation: continental/oceanic base + convergent-boundary
  uplift.
- `climate.rs` — `generate_climate(grid, elevation, config)` computes
  temperature = `equator_temp + (pole_temp - equator_temp) * |y|`
  minus altitude lapse; humidity diffuses upwind from oceanic faces
  over `humidity_iters` rounds with `humidity_decay` attenuation.
- `biome.rs` — `classify_biomes(elevation, climate)` lookup table
  over `(elev, temp, humidity)`. 10 biome constants in `biome::*`.

[`WorldMacroState`](../crates/atomr-worlds-generate/src/macro_state/mod.rs)
bundles the four fields + a FNV-1a `digest` for determinism witnessing.
`sample(dir)` returns the per-face tuple. `MacroStateCache` is a
`Mutex<HashMap<MacroKey, Arc<state>>>` for per-host caching.

[`BrickGenerator`](../crates/atomr-worlds-generate/src/brick.rs)
migrates to `fn generate_brick(&self, ctx: &BrickGenContext) -> Brick`.
`BrickGenContext { world_seed, brick_coord, shape, macro_state, scale }`.
The default `generate_brick_legacy(seed, coord)` shim preserves the
two-arg path for CUDA and downstream callers — neither the CUDA kernel
nor the host's CPU accelerator changes.

`TerrainGenerator` gains `material_at_macro(seed, p, macro_state, scale)`.
Surface height = `macro_elev_at_face + local_fbm_jitter`. Top-layer
material picks from biome (sand for desert/savanna; snow for ice/tundra;
water for ocean; dirt otherwise). When `macro_state` is `None` the
generator follows the Phase-12 path exactly — existing terrain tests
keep their hashes.

[`LocalHostConfig`](../crates/atomr-worlds-host/src/local.rs) grows
`macro_generator: Option<Arc<dyn MacroGenerator>>` and
`macro_cache: Arc<MacroStateCache>`. Cubic worlds skip macro pre-sim
even when the generator is set (back-compat); spheres and cylinders
compute macro state on first actor spawn.

Determinism gate: [`tests/macro_determinism.rs`](../crates/atomr-worlds-generate/tests/macro_determinism.rs)
pins `WorldMacroState::digest` against (seed, config) — runs on the CI
matrix to catch cross-platform drift.

## Phase 13d (landed) — Stipulation v1: in-memory authored regions

[`atomr-worlds-generate/src/authored/`](../crates/atomr-worlds-generate/src/authored/):

- `mod.rs` — `AuthoredRegion` trait (`id`, `bounds`, `apply_to_brick`),
  `AuthoredRegionStore` (per-host registry, sorted-id deterministic
  iteration), `RegionAabb` (inclusive-min, exclusive-max in voxel
  coords), `region_id(name)` (FNV-1a 64).
- `literal.rs` — `LiteralRegion(HashMap<IVec3, Voxel>)`. Constant-
  time bounds check; O(brick_edge³) apply.

[`LocalHostConfig::authored_regions: Arc<Mutex<AuthoredRegionStore>>`](../crates/atomr-worlds-host/src/local.rs)
is shared across actors and the Python binding. `WorldActor::ensure_brick`
applies overlapping regions in sorted-id order after procedural fill,
before the user-write overlay.

`LocalHost::register_authored_region(Arc<dyn AuthoredRegion>)` is the
canonical entrypoint. The PyO3 binding exposes
`WorldClient.register_literal_region(name, bounds_min, bounds_max, voxels)`.

End-to-end: [`tests/stipulation_e2e.rs`](../crates/atomr-worlds-host/tests/stipulation_e2e.rs).

## Phase 13e (landed) — Stipulation v2: heightmap + voxfile loaders

- [`HeightmapRegion`](../crates/atomr-worlds-generate/src/authored/heightmap.rs):
  takes a flat `Vec<u16>` height array indexed `z * width + x`. Each
  column extends from `origin.y` to `origin.y + height` filled with
  `base_material`. PNG / GeoTIFF parsing is a one-`image::crate`-dep
  wrapper documented inline — kept out of core to preserve the
  dep-light workspace.
- [`VoxFileRegion`](../crates/atomr-worlds-generate/src/authored/voxfile.rs):
  sparse `Vec<(IVec3, u16)>` + `VoxelTransform { translation }`.
  Internal storage sorted by `(z, y, x)` for deterministic iteration.
  MagicaVoxel `.vox` / `.schematic` parsing slot on top via optional
  features (`dot_vox`, NBT crates).

Tests: in-module unit tests (3 + 4) and
[`tests/region_loaders.rs`](../crates/atomr-worlds-generate/tests/region_loaders.rs)
(4 cross-region e2e tests).

## Phase 13f (landed) — Skybox + reversed-z

[`crates/atomr-worlds-view/src/skybox.rs`](../crates/atomr-worlds-view/src/skybox.rs)
adds `Skybox` (six RGBA8 `CubeFaceImage`s + observer pose + radii +
captured seed + FNV-1a digest), `CubeFace::ALL` with right-handed
orthonormal basis (`forward`/`up`/`right`), `SkyboxConfig`, and
`render_skybox_from_meshes(meshes, observer, inner, outer, seed, cfg)`.
The 6-face renderer combines all `MeshNode`s into one transform-baked
mesh per face and calls `render_mesh` once per face — the rasterizer
is stateless across calls so per-face output bytes are a pure function
of the inputs.

`Camera::for_cube_face(eye, face, near, far)` produces a 90° FOV /
aspect 1.0 camera oriented along one cube-face axis. `Camera::perspective`
switches to **reversed-z** (`near→1.0`, `far→0.0`) — the matrix has
`[2][2] = -near*nf` and `[3][2] = -far*near*nf`. `Framebuffer.depth`
clears to `0.0` and the rasterizer's depth compare flips from `<` to
`>`. This is the precision regime the composite (13g) and far-LOD
seams (13h) need at planetary scale.

Tests: 7 unit tests in
[`tests/skybox.rs`](../crates/atomr-worlds-view/tests/skybox.rs) +
1 pinned-hash regression in `deterministic_screenshot.rs`.

## Phase 13g (landed) — Composite renderer

[`crates/atomr-worlds-view/src/render.rs`](../crates/atomr-worlds-view/src/render.rs)
gains:

- `FragmentMode::{Opaque, DistanceFade { start_m, end_m, observer }}`.
- `CompositeScene<'a>` — references a `&Skybox`, a `&[MeshNode]` far
  ring, and a `&[MeshNode]` near ring.
- `render_composite(scene, camera, cfg) -> Framebuffer` — three-pass
  composition: skybox background (per-pixel ray sampling, no z-write)
  → far meshes with `DistanceFade` alpha band → near meshes opaque.

A separate `rasterize_triangle_mode` carries the fragment mode through
the inner loop. Per-fragment alpha is barycentric-interpolated from
per-vertex world distance; `alpha > 0.5` is the gate for z-write so
fade-out fragments don't occlude the near ring.

Tests: 6 in
[`tests/composite.rs`](../crates/atomr-worlds-view/tests/composite.rs).

## Phase 13h (landed) — Cross-LOD seam fix

[`crates/atomr-worlds-view/src/iso.rs`](../crates/atomr-worlds-view/src/iso.rs)
adds:

- `boundary_skirt(brick, axis, sign, depth)` — emits skirt quads along
  the named brick face. For each face-plane cell with a non-empty
  voxel along the perpendicular axis, four side quads (8 triangles)
  extend `depth` voxels below the surface.
- `crossfade_overlap(brick, mode_near, mode_far) -> (Mesh, Mesh)` —
  two meshes of the same brick at different LODs, ready for
  `CompositeScene::{near_meshes, far_meshes}` consumption.

The Phase-9 `transvoxel_seam` stub now `#[deprecated]`-aliases to
`boundary_skirt`. Tests: 4 in
[`tests/seam.rs`](../crates/atomr-worlds-view/tests/seam.rs).

## Phase 13i (landed) — Transitive skybox + sphere-flyby demo

[`crates/atomr-worlds-view/src/observer.rs`](../crates/atomr-worlds-view/src/observer.rs)
introduces:

- `SkyboxRefreshPolicy { position_delta_frac, altitude_delta_frac,
  max_age_ticks, refresh_on_tier_change }`. Default `{ 0.05, 0.10,
  600, true }`.
- `ObserverState { position, velocity_mps, containing_frame,
  last_skybox, next_skybox, crossfade_t, crossfade_duration_s,
  since_last_capture_ticks }`.
- `ObserverState::should_refresh(policy, body_center, body_radius,
  prev_frame)`, `accept_next(sky)`, `tick(new_pos, new_frame, dt_s)`.

The crossfade is purely time-based: each `tick(dt_s)` advances
`crossfade_t` by `dt_s / crossfade_duration_s`, and when it reaches
`1.0` the `next` skybox promotes to `last` and the slot frees.

Companion demo: [`examples/sphere-flyby`](../examples/sphere-flyby).
Configures an Earth-class sphere via `PrefixShape`, registers a
literal "city" region, and renders 12 composite PNG frames covering a
surface→Mm-altitude trajectory. Output paths
`/tmp/sphere-flyby-{:02}.png`; run with `cargo run -p sphere-flyby`.

Tests: 6 in `observer::tests::*` covering all five refresh thresholds
plus the velocity-derivation and crossfade-progression paths.

## Phase 14 foundation (landed) — Wave 1 of multi-mode display

Four parallel worktree pieces landed as separate merges and validated
together (`cargo test --workspace` all-green after each merge).

### `Projection` enum on `Camera`

[`crates/atomr-worlds-view/src/camera.rs`](../crates/atomr-worlds-view/src/camera.rs)
gains:

- `pub enum Projection { Perspective { fov_y_rad: f32 },
  Orthographic { half_height_m: f32 },
  Oblique { rotation_deg: f32, scale_m_per_px: f32 } }`.
- `Camera::projection: Projection` field. The legacy `fov_y_rad: f32`
  field is retained because `render.rs` reads it as a field; standard
  constructors keep both in sync.
- `Camera::projection_matrix()` dispatches on `projection`.
  Perspective math is unchanged (Phase 13f reversed-z derivation,
  byte-identical output verified by
  `pinned_hash_matches_current_render`). Orthographic and oblique
  matrices follow the same reversed-z convention (`z_view = -near → 1,
  -far → 0`); derivations are documented inline with the same rigor
  as the perspective comment.

Four new camera tests cover perspective parity, ortho depth mapping,
ortho no-perspective-divide, and oblique shear monotonicity.

### `WorldQuery` trait

[`crates/atomr-worlds-view/src/world_query.rs`](../crates/atomr-worlds-view/src/world_query.rs):

```rust
pub trait WorldQuery: Send + Sync {
    fn brick(&self, addr: &WorldAddr, brick_coord: IVec3, lod: Lod) -> Option<Arc<Brick>>;
    fn ground_height_m(&self, addr: &WorldAddr, xz: [f64; 2]) -> Option<f32>;
    fn subscribe_region(&self, addr: &WorldAddr, region: AABB, lod: Lod)
        -> std::sync::mpsc::Receiver<WorldEvent>;
}
```

`atomr-worlds-view` now depends on `atomr-worlds-proto` (workspace
dep) to consume `AABB` and `WorldEvent` directly. Host-side
implementation lives in `atomr-worlds-host` (added in Phase 14a),
inverting the dep so the view crate does not pull in host. A stub
impl in the module's test block exercises trait-object construction,
the brick/ground-height fast paths, and the subscribe-channel
roundtrip.

### `raster2d` 2D blitter

[`crates/atomr-worlds-view/src/raster2d.rs`](../crates/atomr-worlds-view/src/raster2d.rs):

- `fill_rect(fb, x, y, w, h, color)` — clipped axis-aligned write.
- `fill_rect_stipple(fb, x, y, w, h, color, pattern)` with
  `StipplePattern::{Checker, Horizontal, Vertical, Dense25, Dense75}`
  for thin-feature hints in slice/RTS modes.
- `blend_rect(fb, x, y, w, h, color)` — integer src-over alpha using
  the `(x * 257 + 255) >> 16` div-255 trick; alpha output is
  `max(src.a, dst.a)`.
- `blit_rgba(fb, x, y, src, src_w, src_h)` — byte-blit with clipping;
  panics on size mismatch (programmer error).

All four handle negative origins and overflowing extents by clipping;
zero-size rects are no-ops. Twelve unit tests cover pixel layout,
clipping, alpha blending, and the panic path. Pure 2D — no depth
interaction.

### `ViewCache` + `DerivedStore`

[`crates/atomr-worlds-view/src/view_cache.rs`](../crates/atomr-worlds-view/src/view_cache.rs):

- `CacheAabb { min: [f64; 3], max: [f64; 3] }` — local AABB type
  structurally equivalent to the proto integer AABB; conversion is
  trivial at the call site.
- `DerivedKey: Hash + Eq + Clone + Debug + Send + Sync + 'static`
  trait requiring `fn world_addr(&self) -> &WorldAddr` and `fn
  intersects(&self, aabb: CacheAabb) -> bool`.
- `ViewCache<K: DerivedKey, V: Send + Sync + 'static>` with
  `get_or_build` (read-fast / write-slow double-check),
  `invalidate_intersecting`, `invalidate_world`, `invalidate_key`,
  `len`, `is_empty`. `RwLock<HashMap>` interior.
- `Revision(pub u64)` — coarse cache-buster (e.g., Phase 13c
  `macro_digest`).

Five unit tests cover get/build, intersect-invalidation,
world-invalidation, key-invalidation, and revision-distinct keys.

[`crates/atomr-worlds-persist/src/derived.rs`](../crates/atomr-worlds-persist/src/derived.rs)
(behind the new `derived` feature):

- `DerivedStore` trait — `put`, `get`, `delete`, `delete_prefix`.
- `InMemoryDerivedStore` — `RwLock<HashMap<String, Vec<u8>>>`.
- `DerivedStoreError` — single `Io(String)` variant for now; SQL
  backing slots in here later.

Two feature-gated tests cover put/get roundtrip and prefix delete.

### Scaffold: `modes/` and `derived/` submodule trees

[`crates/atomr-worlds-view/src/modes/`](../crates/atomr-worlds-view/src/modes/)
and
[`crates/atomr-worlds-view/src/derived/`](../crates/atomr-worlds-view/src/derived/)
were pre-created with stub files for each Wave 2 phase
(`fp`, `tp`, `slice`, `rts`, `overview`, `view_mode` and
`slice_index`, `surface_raster`, `world_summary`) so the four parallel
worktree agents implementing Phases 14a–e do not collide on their
parent `mod.rs` files. `modes/view_mode.rs` ships the `ViewMode`
dispatch enum inline (no moving parts).
