# Phases roadmap

Detailed plan for phases 1–6 plus the Python interface. Phase 0 is the
substrate; everything below is built on top of it.

This document is **descriptive of the design**, not a per-commit log. As phases
land, [IMPLEMENTATION.md](IMPLEMENTATION.md) gets updated with concrete
file/line pointers; this document stays focused on the intended end-state.

**Status (2026-05-11)**: Phases 0–6 have all landed. The only remaining
deliverable on this roadmap is the upstream-bridge piece of Phase 2 — handing
meshes from `atomr-worlds-view` off to `atomr-view`'s scene API — and it is
blocked on the upstream growing 3D primitives / a headless wgpu path. The
CPU renderer plus deterministic-screenshot gate cover everything Phase 2
needed in the interim.

## Phase 1 — Generators + `LocalHost` *(landed)*

**Goal**: a single-player vertical slice — `LocalHost::request` returns a
fully populated `BrickSnapshot` for any address.

### New crates

- **`atomr-worlds-noise`** — deterministic value, gradient, and Worley noise;
  FBM combinator. Seeded from `u64` values produced by `child_seed`.
- **`atomr-worlds-generate`** — `Generator` impls per tier. The
  `WorldGenerator` is the only tier that emits voxel content in phase 1
  (terrain + caves); the higher tiers emit metadata (galaxy density,
  system layout) that the level below consumes.

### Touched crates

- **`atomr-worlds-host`** gains:
  - a per-world `WorldActor` (`Actor` impl on top of `atomr_core::actor`)
  - a real `LocalHost` body (`ActorSystem::create` + `Props::create` +
    `actor_of`, then `ask_with` for request/response)
  - actor message type wrapping `WorldRequest` + a `oneshot::Sender<WorldEvent>`

### Determinism boundary

Every generator takes `(seed: u64, addr: WorldAddr) -> Result<Output, Err>`.
The seed comes from `WorldAddr::seed_chain(root_seed)[tier_index]`; bricks
are derived purely from `(world_seed, brick_coord)`. No global mutable state,
no time, no I/O.

### Gates

- proptest: same `(addr, brick_coord)` produces byte-identical bricks across
  100+ trials and across process restarts.
- end-to-end: `LocalHost::request(GetBrick)` returns a `BrickSnapshot` whose
  payload decodes back into the same `Brick`.
- example: `print-brick` dumps a YZ slice of a generated world as ASCII.

## Phase 2 — Renderer integration *(landed in-repo; upstream bridge blocked)*

**Phase 2 in-repo (landed)**: `atomr-worlds-view` crate with three modules
([`mesh`](../crates/atomr-worlds-view/src/mesh.rs),
[`camera`](../crates/atomr-worlds-view/src/camera.rs),
[`render`](../crates/atomr-worlds-view/src/render.rs)): greedy meshing of
bricks into face quads, perspective `Camera` with
`MetricScale::lod_for_screen` integration, a deterministic half-space
triangle rasterizer with z-buffer. Deterministic-screenshot gate in
[`tests/deterministic_screenshot.rs`](../crates/atomr-worlds-view/tests/deterministic_screenshot.rs).
The [`examples/view-png`](../examples/view-png) demo fetches a 4×4×6 brick
slab from `LocalHost`, greedy-meshes, and writes an isometric 512×512 PNG.

**Phase 2 upstream bridge (blocked)**: handing `mesh::greedy_mesh`'s output
off to `atomr-view`'s scene API. Blocked: `atomr-view`'s `SceneDescription`
is UI-only (no `Mesh`/`Camera`/`Renderer`/headless path), and the
`winit+wgpu` backend in `atomr-view-backends` is stubbed. Once the upstream
scene API grows 3D primitives, the mesh output drops straight into them.

### Dependencies for the upstream bridge

- An EGL/Wayland/X display, or `wgpu` headless surface, in CI.
- atomr-view-backends's wgpu backend to be stable.

## Phase 3 — Persistence *(landed)*

[`atomr-worlds-persist`](../crates/atomr-worlds-persist/) wraps
`atomr_persistence::{Journal, SnapshotStore}` with world-specific encoding:
`VoxelWriteEvent`s are bincode-encoded onto the journal and `WorldSnapshot`s
capture the per-world write overlay. `WorldPersistence` is the consumer-facing
handle; the in-memory `InMemoryJournal` + `InMemorySnapshotStore` are
re-exported from `atomr-persistence` for the default backend, and the `sql`
feature pulls in `atomr-persistence-sql`'s `SqlJournal` + `SqlSnapshotStore`
(SQLite by default; Postgres / MySQL / MSSQL via sqlx feature flags).

`LocalHostConfig::persistence: Option<Arc<WorldPersistence>>` wires it in.
When set, `LocalHost::world_actor_for` runs `recover` before spawning the
`WorldActor`, the actor appends each `WriteVoxel` to the journal before
applying it locally, and `save_snapshot` fires every `snapshot_every` writes
(default 64). The overlay survives host restarts and is re-applied to brick
caches on first miss.

End-to-end coverage in
[`atomr-worlds-host/tests/persistence_e2e.rs`](../crates/atomr-worlds-host/tests/persistence_e2e.rs):
write through one host, drop it, recover state through a fresh host, verify
reads match. Snapshot-then-tail recovery is asserted independently.

### Production deployment

- A running SQL instance for production use (`--features sql`); the SQLite
  default makes integration testing painless without one.

## Phase 4 — Streaming subscriptions *(landed)*

`Subscribe` envelope handling, per-subscription bounded `mpsc` channels,
AABB → brick set reduction, `VoxelDelta` emission on writes. `WorldActor`
keeps a `HashMap<u64, Subscriber>` keyed by `sub_id`; backpressure policy
is "drop subscriber on full channel" so the writer never blocks. The first
event after `SubscribeBegin` is one `BrickSnapshot` per brick overlapping
the AABB; subsequent in-region writes produce `VoxelDelta`s. `StreamEnd`
fires on unsubscribe or actor stop.

### Gates

- Subscribe, write voxel, receive matching `VoxelDelta`.
- Subscriber's receiver dropped → `WorldActor`'s send fails on next emit →
  subscription is reaped.
- Stress: 1000 writes/sec to one world, 10 subscribers each with 64-deep
  channel, none of the subscribers backpressures the writer.

## Phase 5 — GPU acceleration *(landed)*

[`atomr-worlds-accel`](../crates/atomr-worlds-accel/) exports an
`Accelerator` trait with `fill_brick(world_seed, brick_coord)` and a
batched `fill_bricks_batch(world_seed, &[IVec3])`. `CpuAccelerator` defers
to any `BrickGenerator`; `CudaAccelerator` (behind the `cuda` feature) spins
up an `atomr_accel_cuda::DeviceActor` with
`EnabledLibraries::NVRTC | BLAS`, compiles
[`cuda_kernel.cu`](../crates/atomr-worlds-accel/src/cuda_kernel.cu) — a
faithful port of the CPU `TerrainGenerator` math — at startup, and
dispatches one launch per `fill_bricks_batch`. The kernel is built with
`--fmad=false` so FMA fusion does not drift last-bit results.

Determinism gate (CPU vs GPU byte equality) in
[`tests/cuda_determinism.rs`](../crates/atomr-worlds-accel/tests/cuda_determinism.rs)
and criterion bench in
[`benches/cpu_vs_gpu.rs`](../crates/atomr-worlds-accel/benches/cpu_vs_gpu.rs).
Both `#[cfg(feature = "cuda")]` and `#[ignore]`d so CUDA-less hosts still
pass `cargo test`.

### Dependencies

- `nvcc` toolchain on the build host (NVRTC compiles `cuda_kernel.cu` at
  startup, but `cudarc` still needs the CUDA driver).
- `atomr-accel` sibling checkout (path dep at `../atomr-accel`).

## Phase 6 — Python interface *(landed)*

A `pyo3 + maturin` extension module exposing:

- `WorldAddr`, `LevelKey`, `Lod`, `MetricScale`, `Voxel`, `Brick`, the seed
  helpers (`splitmix64`, `child_seed`, `WorldAddr.seed_chain`), and a
  `LocalHost`-backed `WorldClient` with `get_voxel`, `get_brick`,
  `subscribe`.

- Module structure mirrors atomr's `py-bindings/pycore` pattern: a Rust
  cdylib crate `atomr-worlds-py` builds a `_native` extension; a thin
  Python package `atomrworlds/` re-exports it with niceties (NumPy
  interop, repr).

- Determinism: round-trip a seed chain from Python, compare to Rust output
  via `cargo test` → identical bytes.

### Build flow

```sh
maturin develop -m crates/atomr-worlds-py/Cargo.toml  # builds + installs
python -c "import atomrworlds as aw; print(aw.world_addr_root().seed_chain(0xDEAD_BEEF))"
```

### Out of scope for phase 6 (yet)

- Streaming subscriptions in Python (sync vs async story to design).
- Numpy zero-copy brick views; first cut copies into a Numpy `uint16` array.
- PyPI release tooling.

## Dependency graph between phases

```
Phase 0 (substrate)
    │
    ├─► Phase 1 (generators + LocalHost) ─────► Phase 2 (greedy mesh + CPU rasterizer)
    │                  │
    │                  ├──► Phase 3 (atomr-persistence Journal + SnapshotStore binding)
    │                  ├──► Phase 4 (streaming subscriptions)
    │                  └──► Phase 5 (Accelerator trait; CPU + CUDA backends)
    │
    └─► Phase 6 (Python bindings — depends on phase 1 for the host surface)
```

Phase 1 is the keystone. Phases 2–6 attach on top, all landed. The only
remaining roadmap item is the upstream-bridge piece of Phase 2 once
`atomr-view` exposes a 3D scene API or a headless wgpu path.

## Determinism contract (cross-phase invariant)

For any `WorldAddr` and root seed, calling `LocalHost::request(GetBrick)`
must produce a byte-identical `BrickSnapshot` payload across:

- repeated calls within one process,
- process restarts,
- platforms (x86_64 Linux + ARM64 macOS),
- single-player vs cluster hosting,
- CPU vs GPU generation,
- Rust callers vs Python callers via the bindings.

Phase 0's hash distribution + avalanche tests are the floor; every phase
adds determinism assertions at its layer.

## Phase 7 *(landed)* — Address enum + vehicles + policy + strategy registry

`HierarchicalIdentifier` trait (`crates/atomr-worlds-core/src/seed.rs`)
promotes the parent-id + identifier hash rule to a documented invariant.
[`Address`] (`crates/atomr-worlds-core/src/addr.rs`) is the new canonical
addressable thing wrapping `WorldAddr` or [`VehicleAddr`]
(`crates/atomr-worlds-core/src/vehicle.rs`). A unified [`WorldActor`]
handles both worlds and vehicle voxel spaces, dispatching on the address
variant. [`GenerationPolicy`] +
[`PolicyResolver`] (`crates/atomr-worlds-host/src/policy.rs`) let any
address opt out of generation via `Empty` or pin a specific strategy via
`Custom(StrategyId)`. [`GeneratorRegistry`] +
[`BuiltinSelector`] (`crates/atomr-worlds-generate/src/registry.rs`)
register multiple [`BrickGenerator`] strategies (`terrain` real;
`gas_giant`, `asteroid_belt`, `empty_planetoid` stubs) and pick one
deterministically from the world seed. Persistence keys widen to
`Address` with `W:`/`V:` discriminator prefixes.

End-to-end: `crates/atomr-worlds-host/tests/policy_e2e.rs` — vehicle
voxel space isolation, sector-level Empty policy, vehicle frame
round-trip, parent-world policy inheritance.

## Phase 8 *(landed)* — Atmosphere + metric LOD + interaction unit

`MetricScaleRegistry` + `tier_for_distance` in
`crates/atomr-worlds-core/src/lod.rs`. `AtmosphereRadius` in
`crates/atomr-worlds-core/src/atmosphere.rs` (default 1.25 × body
radius, per-body override). `SubscribeMetric` + `ContainingFrameChange`
+ `Tier` proto variants drive atmosphere-bounded subscriptions.
[`StreamingPolicy`] + `RingPlan` in
`crates/atomr-worlds-proto/src/streaming.rs` plan near/far ring
streaming. [`InteractionUnit`] (sphere/cube/cone/voxel brush, precision
tier hook) + `WriteRegion` + `RegionDelta` in
`crates/atomr-worlds-core/src/interaction.rs` and the proto layer give
a configurable unit-of-interaction; `Brick::set_region` applies a
predicate-driven brush. Tests:
`crates/atomr-worlds-host/tests/region_write_e2e.rs`.

## Phase 9 *(landed)* — Isosurface (Naive Surface Nets) meshing

`crates/atomr-worlds-view/src/iso.rs` ships Naive Surface Nets
(Gibson 1998) as `MeshMode::Smooth(SmoothConfig)` alongside the
existing greedy `MeshMode::Flat`. Density derived from binary
occupancy at cell corners; vertex per sign-change cell at the
centroid of "in" corners; per-face flat normals computed from
triangles. The algorithm choice is justified inline (vs marching
cubes / dual contouring / transvoxel). `transvoxel_seam` is a stub
for the LOD-tier seam case (full body deferred). `scene.rs` exposes
`scene_from_bricks` consuming either mode.

## Phase 10 *(landed)* — `ClusterHost` real body

`ClusterHost` (`crates/atomr-worlds-host/src/cluster.rs`) wraps a
real `atomr_cluster_sharding::ShardRegion<WorldExtractor>` with a
per-entity handler that delegates to an in-process `LocalHost`. The
reply path uses an out-of-band `pending: Mutex<HashMap<corr_id,
oneshot::Sender>>` registry since `ShardRegion::deliver` is
fire-and-forget. Cross-node remote forwarding is exposed via
`ClusterHost::region()` returning the `Arc<ShardRegion>` for
`set_remote_forwarder`; the full `atomr-remote`-backed bridging
actor remains a follow-up that depends on upstream codec
verification. In-tree two-node test pending.

## Phase 11 *(landed)* — Python release + zero-copy-ish accessor

`.github/workflows/release-py.yml` builds wheels on push of `py-v*`
tags across linux x86_64/aarch64, macos x86_64/arm64, windows
x86_64, python 3.10–3.13 via `PyO3/maturin-action@v1`, and
publishes via `maturin publish` on the `pypi` environment.
`PyBrick.buffer_bytes()` returns the brick's voxels as a single-copy
`bytes` object suitable for `numpy.frombuffer(...)`. Full
buffer-protocol zero-copy + `pyo3-async-runtimes`-backed
`subscribe_async` need a separate worktree to land cleanly
against the current PyO3 version pin; deferred as documented
follow-ups.

## Phase 12 *(landed)* — Scene description + portals + variable-depth

`crates/atomr-worlds-view/src/scene.rs` exposes a generic-engine
`SceneDescription` (meshes / cameras / lights / material palette /
frame metadata) that `scene_from_bricks` builds from a brick slab in
either mesh mode. The future atomr-view bridge is an ~80-LOC
adapter on top. Portals enter the wire: `Portal`,
`WorldRequest::TraversePortal`, `WorldEvent::PortalArrival`. The
host returns a trivial identity-transform arrival pending a real
per-actor portal registry. Variable-depth addressing:
`AddrEither::Closed(Address) | Open(Vec<LevelKey>)` in
`crates/atomr-worlds-core/src/addr.rs`, with a length-prefixed
seed-chain method that walks each level key through
`derive_child` — proves the variable-depth contract without
forcing the host / persist layers to migrate.
