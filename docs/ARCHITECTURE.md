# Architecture

The high-level model behind `atomr-worlds`. For implementation specifics
(types, file paths, exact algorithms), see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Why this exists

[atomr](https://github.com/rustakka/atomr) is a Rust port of Akka: an actor runtime with cluster
membership, cluster-sharding, remote messaging, persistence, and streams. It is a **distributed
compute substrate**; it has no spatial, voxel, or procedural-generation primitives.

`atomr-worlds` builds those primitives, and uses atomr as the hosting runtime. A single-player
game is one node; a multiplayer galaxy is a cluster of nodes routing the same actor protocol.

## The hierarchy

```
Universe (root seed; cosmic scale ~10²⁷ m)
└── Galaxy   (~10²¹ m — Milky-Way-class)
    └── Sector  (~10¹⁸ m — configurable; ≈ 30 ly)
        └── System  (~10¹³ m — ~100 AU)
            └── World   (~10⁷ m — Earth-class)
```

Five tiers, closed. Each tier is a sparse 3-D coordinate grid keyed by an integer `IVec3`. The
hierarchy is fixed at phase 0; a variable-depth model can wrap this later without breaking call
sites that address the five known tiers.

**Sectors** are a mandatory tier between galaxy and system. They serve two purposes:

1. **Seed-chain regularity.** Every tier participates in the deterministic seed derivation, so
   downstream generators can rely on consistent address shape.
2. **Cluster sharding granularity.** Sectors are the unit at which work load-balances across
   atomr cluster nodes — all systems and worlds beneath a sector co-locate on one shard owner.

A galaxy that wants "no sectors" pins `sector.coord = (0, 0, 0)` (one sector per galaxy).

## Dimensions

Every tier carries a `DimensionId: u32`, mixed into the seed hash. The default `0` is the primary
plane; non-zero values are orthogonal planes that share the same coordinate grid but produce
independent content:

```
Universe(seed = U)
├── Dim(0)  ← primary
│   └── Galaxy → System → World
│                          ├── Dim(0)  ← overworld
│                          └── Dim(1)  ← Nether-style alt plane
└── Dim(1)  ← alt-physics universe
```

This means dimensions aren't a separate top-level multiverse layer — they're a free axis at every
tier. The cost is `5 × u32 = 20` bytes per `WorldAddr`; in practice most slots stay `0`.

## Seed derivation

The core rule:

```
child_seed = hash(parent_seed, dim_id, child_coord)
```

`hash` is SplitMix64's finalizer applied incrementally. Walking from the root yields a
`[u64; 5]` deterministic seed chain — one seed per tier — from a single root.

The principle: every subdivision of space is reproducible from the root seed alone, without
having to generate or store parents. A client asking for World *(0,0,0)* of System *(7,7,7)* of
Sector *(0,1,0)* of Galaxy *(3,-2,1)* of a root universe gets the same seed chain regardless of
node, version, or platform.

SplitMix64 was chosen because it is small (12 lines), well-studied, `const`, has no float
dependencies, and avalanches well — a 1-bit perturbation of any input flips ~50% of output bits.
Phase 0's test suite asserts this floor at ≥ 40% across all input axes (parent, dim, x, y, z).

## Sparse voxel storage

The unit of voxel content is a **Brick + Octree hybrid**:

```
Sparse Voxel Octree (top — empty-space skipping at cosmic scales)
    └── ... internal nodes (8-bit child mask, popcount-indexed children) ...
        └── Brick  ← dense 16³ leaf (4096 voxels × 2 bytes = 8 KiB; L1-friendly)
            [Voxel; 4096]
```

- **Why bricks at the leaves.** Voxel access locality is dominated by tight inner loops over
  small regions; a dense leaf saves the per-voxel pointer chase. 16³ = 8 KiB matches GPU
  voxel-cone-tracing and OpenVDB tile conventions.
- **Why an octree above.** Cosmic worlds are mostly empty space; a flat brick hashmap would
  burn cache walking neighborhoods of empty cells. The octree skips empty subtrees in O(depth).
- **Why 8-bit child mask + popcount-indexed arena.** A naive `[Option<NodeId>; 8]` is 32 bytes
  per internal node; a child mask plus `(base, popcount)` indexing is 5 bytes. Half the cache
  footprint of traversal — the dominant cost of empty-space skipping. One popcount per descent
  (~1 ns) buys the savings.

LOD is just which depth in the pyramid you query; `meters_per_voxel = root_size_m / 2^depth`.

## Metric LOD

Each tier has a `MetricScale { root_size_m, max_depth }`. Default tile sizes (rounded):

| tier     | root edge (m) | max depth | leaf voxel size |
| -------- | ------------- | --------- | --------------- |
| universe | 10²⁷          | 64        | ~54 Mm          |
| galaxy   | 10²¹          | 56        | ~14 km          |
| sector   | 10¹⁸          | 48        | ~3.5 km         |
| system   | 10¹³          | 40        | ~9 m            |
| world    | 10⁷           | 24        | ~60 cm          |

Universe leaves at 54 Mm are intentional: at universe scale you address galaxies, not individual
voxels. The hierarchy means leaf-scale work always happens in the appropriate tier (worlds for
terrain, systems for orbital mechanics, etc.).

`MetricScale::lod_for_screen(distance_m, focal_px, target_px_per_voxel)` picks the coarsest LOD
whose voxel projects to at most `target_px_per_voxel` pixels at the given camera distance — the
basis for streaming far chunks at lower fidelity than near ones.

## Hosting

A single trait, [`WorldHost`](../crates/atomr-worlds-host/src/host.rs), with two implementations
that share the per-world actor protocol:

- **`LocalHost`** — embedded atomr `ActorSystem` in the same process. Suitable for single-player
  or in-engine tooling. Zero network hops; same control flow as cluster.
- **`ClusterHost`** — atomr-cluster-sharding `ShardRegion` routes envelopes to the world actor
  on whatever cluster node owns its shard. Suitable for multiplayer or large worlds that exceed
  one machine's memory.

Both speak the same `Envelope<WorldRequest>` / `Envelope<WorldEvent>` protocol over bincode 2 —
the same codec atomr-remote already uses, so a process that bridges atomr and atomr-worlds
stays on a single serializer.

The clustering routing function uses an `atomr_cluster_sharding::MessageExtractor` implemented
in [`WorldExtractor`](../crates/atomr-worlds-host/src/extractor.rs):

- `shard_id` packs `(universe coord + dim, galaxy coord, sector coord)`. All systems and worlds
  under one sector live on the same shard owner — a stellar system's bricks stay co-resident.
- `entity_id` packs the full five-tier address. Each world is an entity; the shard region routes
  by `shard_id` and addresses individual worlds by `entity_id`.

This means: load-balance across the cluster at sector granularity; cache-friendly access within
a sector; deterministic routing (same address always lands on the same shard, regardless of
which client made the request).

## In scope today

Phases 0–6 have all landed. What's wired up:

- **Generator bodies.** Per-tier `Generator` impls plus the CPU `TerrainGenerator` (FBM
  heightfield + Worley caves + dirt layer). CUDA acceleration via `atomr-accel-cuda` NVRTC
  produces byte-identical bricks.
- **Streaming.** `LocalHost::subscribe` returns an `mpsc::Receiver<Envelope<WorldEvent>>`;
  initial `BrickSnapshot`s for each brick overlapping the AABB, then `VoxelDelta`s as writes
  land. Drop-subscriber backpressure.
- **Persistence.** `atomr-worlds-persist` wraps `atomr-persistence`'s `Journal` +
  `SnapshotStore`. Writes are journalled before applying, snapshots fire every N writes, state
  recovers on host restart. SQLite/Postgres/MySQL/MSSQL via the `sql` feature.
- **Renderer.** CPU greedy meshing, perspective camera with `MetricScale::lod_for_screen`
  integration, and a deterministic software rasterizer. Bridging to `atomr-view`'s scene API
  remains blocked on upstream 3D primitives.
- **Python bindings.** `atomrworlds` PyO3 + maturin extension exposes the determinism
  primitives and a `LocalHost`-backed `WorldClient`.

## Out of scope (today)

Still deferred:

- **Multi-dimension routing policy.** Dimensions are addressable, but cross-dimension portals or
  passivation rules are not modeled.
- **Variable-depth hierarchies.** The five-tier closed shape is fixed; nothing wraps it yet.
- **Multi-galaxy / multi-shard load-balancing policy** beyond the sector-granularity sharding
  function. Cluster routing through `WorldExtractor` is wired; the `ClusterHost` body that
  owns a `ShardRegion` is still a placeholder.
- **PyPI release tooling**, async / streaming Python API, and zero-copy NumPy brick views (the
  first cut copies into a NumPy `uint16` array).
- **atomr-view bridge.** Mesh output is ready; the upstream scene API has to grow 3D
  primitives or a headless wgpu path before it can be wired up.

## Phase 7+ additions (landed)

### `Address` and `HierarchicalIdentifier`

The canonical addressable thing is now [`Address::World(WorldAddr) |
Vehicle(VehicleAddr)`](../crates/atomr-worlds-core/src/addr.rs). Single-world
callers migrate via `impl From<WorldAddr> for Address`. The trait
[`HierarchicalIdentifier`](../crates/atomr-worlds-core/src/seed.rs) +
helper `derive_child` document the parent-id-plus-identifier hash invariant:
every parent → child seed transition routes through one function over
`(dim: u32, coord: IVec3)`. `LevelKey` and `VehicleSlot` are the two
identifier types today; future sub-world tiers add their own without
introducing new hash bodies.

### Vehicles and entity space

[`VehicleAddr`](../crates/atomr-worlds-core/src/vehicle.rs) = parent frame
(world / system / sector) + slot id. Vehicles own their own voxel space — a
write to a vehicle address does *not* leak into its parent world. The
[`AffineFrame`] type carries the vehicle's pose in its parent's coordinates;
[`ContainingFrame`] is the rendering-side answer to "what is the observer
currently inside?". `WorldRequest::{GetVehicleFrame, SetVehicleFrame}` and
`WorldEvent::{VehicleFrame, VehicleFrameDelta}` complete the protocol.

### Generation policy and strategies

[`GenerationPolicy::{Seeded, Empty, Custom(StrategyId)}`](../crates/atomr-worlds-generate/src/registry.rs)
+ a [`PolicyResolver`](../crates/atomr-worlds-host/src/policy.rs) attached to
`LocalHostConfig.policy`. `PrefixPolicy` walks world → system → sector →
galaxy → universe and applies the most-specific match. The
[`GeneratorRegistry`](../crates/atomr-worlds-generate/src/registry.rs)
holds [`BrickGenerator`] impls keyed by [`StrategyId`] (const FNV-1a
hashes of names) and a [`StrategySelector`] that picks one
deterministically from the world seed.

### Metric scale registry + atmosphere

[`MetricScaleRegistry`](../crates/atomr-worlds-core/src/lod.rs) lets
per-body overrides shift the standard per-tier defaults. Atmosphere
boundary (`AtmosphereRadius`, default 1.25 × body radius) demarcates the
distance within which the host streams a body's voxel grid; the
`SubscribeMetric` protocol + `ContainingFrameChange` event support the
atmospheric tier-demotion when an observer leaves.

### Interaction unit + region writes

[`InteractionUnit { kind, radius_m, precision_scale }`](../crates/atomr-worlds-core/src/interaction.rs)
parametrizes a brush. `WorldRequest::WriteRegion` applies it; the host
journals per-voxel changes for replay correctness and emits one
aggregated `WorldEvent::RegionDelta` per affected subscriber. The
`precision_scale: Lod` field is the hook for the isosurface mesher to
treat coarse-precision block-writes as a coarser implicit surface.

### Isosurface meshing (Naive Surface Nets)

[`MeshMode::{Flat, Smooth(SmoothConfig)}`](../crates/atomr-worlds-view/src/iso.rs)
+ `surface_mesh()`. `Flat` dispatches to greedy meshing (existing
behavior); `Smooth` uses Naive Surface Nets — one vertex per
sign-change cell, deterministic, naturally smooth. Algorithm choice
documented inline (vs marching cubes, dual contouring, transvoxel).

### Scene description, portals, variable-depth

[`SceneDescription`](../crates/atomr-worlds-view/src/scene.rs) is the
engine-shaped neutral exchange format for the future atomr-view bridge.
[`Portal`](../crates/atomr-worlds-proto/src/messages.rs) +
`TraversePortal` / `PortalArrival` envelope the cross-dim / cross-world
hop. [`AddrEither::Closed(Address) | Open(Vec<LevelKey>)`](../crates/atomr-worlds-core/src/addr.rs)
is the variable-depth wrapper — same `derive_child` invariant applies,
length-prefixed so different depths can't collide.

## Design principles

1. **Pure-data core.** `atomr-worlds-core` has zero dependencies on atomr or async runtimes.
   Anything that can be a plain `Copy` type and a `const fn` is.
2. **Determinism is non-negotiable.** Seed derivation, hash distribution, and routing must give
   identical results across runs, platforms, and process restarts. Test suite asserts this.
3. **Closed hierarchy at phase 0.** Five tiers, fixed shape. A `Vec<LevelKey>` would be more
   flexible but harder to make `Copy`, hash cheaply, and shard on. We can wrap later if
   variable depth becomes necessary.
4. **One codec, one runtime.** Bincode 2 for the wire (same as atomr-remote), tokio + atomr
   actors for hosting (same as everything else in the rustakka ecosystem). No parallel stacks.
