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

## Phase 13a *(landed)* — World shape type + horizon math

[`WorldShape::{Cube, Sphere, Cylinder}`](../crates/atomr-worlds-core/src/shape.rs)
introduces the geometric envelope of a [`World`]. Methods include
`contains(p)`, `horizon_distance_m(altitude)` (sphere uses
`sqrt(2*R*h + h²)`; cube returns infinity), `surface_normal_at(p)`,
`bounding_aabb()`, `surface_area_m2()`, and `wrap(p)` (identity for
sphere/cube; angular wrap on cylinder). `Hash`/`Eq`/`PartialEq`
implemented manually via `f64::to_bits()` so the type is usable as a
HashMap key (macro-state cache) and platform-stable. `World` and
`WorldGen` grow a `shape` field with `Default = Cube { edge_m: 1e7 }`
so pre-Phase-13 code keeps its exact behavior.

Tests: `crates/atomr-worlds-core/src/shape.rs` ships 15 unit tests
covering containment, horizon at known altitudes (Earth-radius
6.371e6 m at 1km altitude ≈ 112,884.897 m horizon), variant
discrimination, and hash bit-stability.

## Phase 13b *(landed)* — Horizon-driven streaming + brick filter

[`ShapeResolver`](../crates/atomr-worlds-host/src/shape.rs) + `PrefixShape`
parallel the policy resolver — hierarchical address → `WorldShape`
lookup. `LocalHostConfig::shape_resolver` defaults to `DefaultShape`
(cubic Earth-class) for back-compat. `WorldActor` resolves the shape
on spawn; `ensure_brick` checks `brick_inside_shape(coord)` and
short-circuits out-of-shape bricks to empty without ever invoking the
generator. `StreamingPolicy::ring_for_curved(observer, edge_m,
horizon)` (and `MetricScale::lod_for_screen_curved`) clamp the
streaming radius to the horizon at the observer's altitude. The
metric subscription state tracks per-subscriber observer pose and
sent-bricks set; `UpdateObserverPos` recomputes the ring, emits a
fresh `Tier` event, and snapshots newly-visible bricks.

Tests: `crates/atomr-worlds-host/tests/sphere_horizon_e2e.rs` covers
the horizon clamp, out-of-shape filter (with a `CountingBrick` test
double), observer-tick delta emission, default-shape regression, and
cross-host determinism of the initial ring.

## Phase 13c *(landed)* — Geologic macro pre-sim

[`atomr-worlds-generate/src/macro_state/`](../crates/atomr-worlds-generate/src/macro_state/)
adds a three-layer pre-pass that runs once per non-cubic world:

- `surface_grid.rs` — recursive-icosahedron tessellation with integer
  `FaceId`s and O(1) neighbour lookups. Level 4 (~5k faces, ~150 KB)
  is the default; level 6 (~82k faces, ~2 MB) is opt-in via
  `MacroConfig::grid_level`.
- `plates.rs` — Voronoi tectonic plates seeded from `world_seed`.
  Multi-source BFS with sorted-id collision resolution gives true
  distance-Voronoi labeling (no race in tie-break). Convergent-
  boundary uplift produces mountain belts.
- `climate.rs` — latitude (via `face_centroid.y`) + altitude lapse →
  temperature. Humidity diffuses upwind from oceanic faces.
- `biome.rs` — fixed classification table over `(elev, temp, humidity)`.

[`BrickGenerator`](../crates/atomr-worlds-generate/src/brick.rs)
migrates from `(world_seed, brick_coord)` to `&BrickGenContext`. A
default `generate_brick_legacy(seed, coord)` shim preserves the
two-arg signature for the CUDA accelerator (`crates/atomr-worlds-accel/src/{lib,cuda}.rs`)
and any downstream callers. `TerrainGenerator` consumes macro state
when present — surface = macro_elev + local FBM jitter, top-layer
material chosen by biome — and falls back to the Phase-12 algorithm
exactly when `macro_state: None`.

`LocalHostConfig` grows `macro_generator: Option<Arc<dyn MacroGenerator>>`
+ `macro_cache: Arc<MacroStateCache>`. On actor spawn, the cache
produces (or reuses) the per-world macro state — pure function of
`(world_seed, shape)`. `WorldMacroState::digest` is a FNV-1a
witness over plates / elevation / climate / biomes; same input →
same digest, byte-stable across runs.

Tests: 22 unit tests in `macro_state::*` modules plus a 6-test
determinism gate at
[`tests/macro_determinism.rs`](../crates/atomr-worlds-generate/tests/macro_determinism.rs).

## Phase 13d *(landed)* — Stipulation v1: in-memory regions + Python API

[`atomr-worlds-generate/src/authored/`](../crates/atomr-worlds-generate/src/authored/)
introduces hand-authored region overlays:

- `AuthoredRegion` trait — `id()`, `bounds()`, `apply_to_brick()`.
  Implementors are pure: same state → same brick voxels.
- `AuthoredRegionStore` — per-host registry. Iteration is sorted by
  region id for determinism.
- `LiteralRegion` — `HashMap<IVec3, Voxel>`-backed in-memory region.

`LocalHostConfig::authored_regions: Arc<Mutex<AuthoredRegionStore>>`
is consulted on every brick miss. `WorldActor::ensure_brick` applies
matching regions in sorted-id order *after* procedural generation and
*before* the user-write overlay. `LocalHost::register_authored_region`
is the canonical entrypoint.

`atomr-worlds-py` exposes `WorldClient.register_literal_region(name,
bounds_min, bounds_max, voxels)`. Voxels are passed as a Python
`dict[(x,y,z), int]`.

Tests: `stipulation_e2e.rs` (5 tests) verifies authored overlay,
outside-region purity, multi-region registration, the empty-world +
authored stage pattern (storytelling), and cross-host determinism.

## Phase 13e *(landed)* — Stipulation v2: heightmap + .vox file loaders

Format-agnostic loaders sitting on top of the 13d trait:

- [`HeightmapRegion`](../crates/atomr-worlds-generate/src/authored/heightmap.rs)
  consumes a raw `Vec<u16>` height array — equivalent to grayscale
  PNG / GeoTIFF rows — and projects each column as voxels of
  `base_material`. PNG / DEM file format parsing is a one-crate-dep
  wrapper that slots on top (documented inline).
- [`VoxFileRegion`](../crates/atomr-worlds-generate/src/authored/voxfile.rs)
  consumes a sparse `Vec<(IVec3, u16)>` + a `VoxelTransform`
  (translation today; rotation is future). The internal storage is
  sorted by `(z, y, x)` so iteration order is deterministic.
  MagicaVoxel `.vox` and Minecraft `.schematic` parsers slot on top
  via optional features.

Determinism: same inputs → byte-identical brick output. Tests in
`crates/atomr-worlds-generate/tests/region_loaders.rs` (4 tests) +
in-module unit tests (3 heightmap, 4 voxfile).
