# Implementation

Module-by-module map of phase 0. For the high-level model and design rationale, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Workspace shape

| crate                          | purpose                                         | atomr deps              |
| ------------------------------ | ----------------------------------------------- | ----------------------- |
| `atomr-worlds-core`            | Coordinates, addressing, seeds, LOD             | none                    |
| `atomr-worlds-voxel`           | Sparse voxel storage (brick + octree hybrid)    | none                    |
| `atomr-worlds-proto`           | Wire-format messages and envelopes              | none                    |
| `atomr-worlds-host`            | `WorldHost` trait, local / cluster impls        | core, cluster, sharding |
| `atomr-worlds-testkit`         | proptest strategies, cross-crate verification   | none (dev-dep on host)  |

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

Both are placeholder structs in phase 0. Their `request` / `subscribe` impls return
`HostError::NotYetImplemented`; `shutdown` is a no-op `Ok(())`. The phase 1 work is to populate
`LocalHost` with an `atomr_core::ActorSystem` handle and `ClusterHost` with an
`atomr_cluster_sharding::ShardRegion<WorldExtractor>` handle, then wire the per-world actor
behind both.

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
    Shutdown,
    NotYetImplemented(&'static str),
}
```

## atomr-worlds-testkit

[`crates/atomr-worlds-testkit/src/strategies.rs`](../crates/atomr-worlds-testkit/src/strategies.rs)

proptest strategies: `arb_ivec3`, `arb_level_key`, `arb_world_addr`, `arb_lod(max_depth)`,
`arb_voxel`, `arb_brick`. `arb_brick` produces sparse-ish bricks (0–64 writes from a 4096-cell
space) so the HashMap-oracle test exercises both empty and populated regions.

### Test surface (phase 0)

| location                                                       | what it checks                                                              |
| -------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `atomr-worlds-core` unit tests                                 | Coord newtypes transparent, seed determinism, dim discrimination, LOD math  |
| `atomr-worlds-voxel` unit tests (`brick`, `octree`)            | Brick round-trip, empty-count invariant, octree set→get, OOB error, sparse probes |
| `atomr-worlds-voxel/tests/oracle_stress.rs`                    | 5 000 random writes match HashMap oracle; sparse tree probe budget per read |
| `atomr-worlds-testkit/tests/cross_crate.rs`                    | `WorldAddr` bincode + JSON round-trips; brick proptest oracle; protocol round-trips |
| `atomr-worlds-testkit/tests/hash_quality.rs`                   | Avalanche ratio ≥ 0.40 across 5 perturbation sites; low-byte uniformity within ±12% (5σ) |
| `atomr-worlds-testkit/tests/extractor_stable.rs`               | Shard id and entity id stable; sibling systems share shard id              |

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

## What changes in phase 1

The shapes in phase 0 are intentionally minimal; phase 1 adds bodies:

- `Generator` implementations per tier (noise, terrain, system layouts).
- `LocalHost` / `ClusterHost` bodies — instantiate atomr actor system, route envelopes to a
  per-world actor, propagate events back through `mpsc::Receiver`.
- A persistence binding (atomr-persistence) that snapshots generated bricks so generation is
  pay-once.
- Streaming behavior for `Subscribe` envelopes: AABB-bounded brick rollup, delta diffing,
  backpressure.

Existing types stay stable; nothing in phase 0 is expected to break in phase 1.
