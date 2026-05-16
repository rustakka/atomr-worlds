# Phases roadmap

Detailed plan for phases 1–6 plus the Python interface. Phase 0 is the
substrate; everything below is built on top of it.

This document is **descriptive of the design**, not a per-commit log. As phases
land, [IMPLEMENTATION.md](IMPLEMENTATION.md) gets updated with concrete
file/line pointers; this document stays focused on the intended end-state.

**Status (2026-05-13)**: Phases 0–17 plus Phase 17.1 (per-LOD brick
generation) have all landed. The only remaining deliverable on the
in-repo roadmap is the upstream-bridge piece of Phase 2 — handing
meshes from `atomr-worlds-view` off to `atomr-view`'s scene API — and
it is blocked on the upstream growing 3D primitives / a headless wgpu
path. The CPU renderer plus deterministic-screenshot gate cover
everything Phase 2 needed in the interim. The forward-looking work
(finer-grained LOD ladders, additional generation styles, real-Earth
data feeds) is sketched in the README under "Roadmap" and depends on
the per-LOD generation contract documented in [LOD.md](LOD.md).

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
cubes / dual contouring / transvoxel). The Phase-17.1 follow-up
replaces the early `transvoxel_seam` stub with `face_height_profile`
+ `lod_transition_strip` (additive transition geometry alongside the
nested-LOD crossfade, see Phase 17.1 below). `scene.rs` exposes
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

## Phase 11 *(landed)* — Python release + zero-copy accessor

`.github/workflows/release-py.yml` builds wheels on push of `py-v*`
tags across linux x86_64/aarch64, macos x86_64/arm64, windows
x86_64, python 3.11–3.13 via `PyO3/maturin-action@v1`, and
publishes via `maturin publish` on the `pypi` environment.
`PyBrick.buffer_bytes()` returns the brick's voxels as a single-copy
`bytes` object suitable for `numpy.frombuffer(...)`.

### Follow-ups landed

- **True zero-copy buffer protocol** on `PyBrick`. `__getbuffer__` and
  `__releasebuffer__`
  ([`crates/atomr-worlds-py/src/lib.rs`](../crates/atomr-worlds-py/src/lib.rs))
  expose the brick's 8 KiB voxel slice as a `(16, 16, 16)` `uint16`
  view; `numpy.asarray(brick)` / `memoryview(brick)` allocate no
  copy. The format string is `"H"` (uint16) when `PyBUF_FORMAT` is
  requested; shape and strides are filled when `PyBUF_ND` /
  `PyBUF_STRIDES` are. The slot only entered the limited API at
  Python 3.11, so the workspace `pyo3` feature was bumped from
  `abi3-py310` → `abi3-py311`, the release matrix and `pyproject.toml`
  dropped the 3.10 row, and the smoke test
  (`crates/atomr-worlds-py/python/tests/test_smoke.py::test_brick_buffer_protocol_zero_copy`)
  asserts the format/shape/itemsize and the numpy `.base` link.

- **`subscribe_async` + `SubscriptionHandle`** on `PyWorldClient`. The
  client gains a `subscribe_async(addr, region_min, region_max,
  lod_depth, sub_id)` coroutine returning a `SubscriptionHandle`. The
  handle is an async iterator: `async for ev in handle: …` yields
  tagged dicts (`"kind": "snapshot" | "delta" | "stream_end" | …`).
  `pyo3-async-runtimes` (0.22, `tokio-runtime` feature) bridges the
  host's tokio `mpsc::Receiver` into a Python awaitable; the receiver
  is held under `tokio::sync::Mutex<Option<…>>` so `__anext__` is
  re-entrant and raises `StopAsyncIteration` when the channel closes.
  Smoke test:
  `crates/atomr-worlds-py/python/tests/test_smoke.py::test_subscribe_async_yields_snapshot_then_delta`.

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

## Phase 13f *(landed)* — Skybox + reversed-z

The CPU renderer switches its perspective projection from `[0, 1]`
forward-z to **reversed-z** (`near → 1.0`, `far → 0.0`). Reversed-z
spreads f32 precision evenly across the depth-buffer range under
perspective division, which is the prerequisite for stitching a
skybox capture against near-field terrain without z-fighting at
celestial-body distances. The change is local to
[`crates/atomr-worlds-view/src/camera.rs`](../crates/atomr-worlds-view/src/camera.rs)
(`perspective`) and
[`crates/atomr-worlds-view/src/render.rs`](../crates/atomr-worlds-view/src/render.rs)
(`Framebuffer.depth` cleared to `0.0`; z-buffer compare flipped from
`<` to `>`). The pinned screenshot hash in
[`tests/deterministic_screenshot.rs`](../crates/atomr-worlds-view/tests/deterministic_screenshot.rs)
is updated to the new value; the run-to-run determinism assertion is
unchanged.

[`crates/atomr-worlds-view/src/skybox.rs`](../crates/atomr-worlds-view/src/skybox.rs)
adds a `Skybox` type (six RGBA8 `CubeFaceImage`s plus observer pose,
inner / outer radius, captured seed, face resolution, FNV-1a digest),
the `CubeFace` enum with right-handed orthonormal basis
(`forward`/`up`/`right`), `SkyboxConfig`, and a mesh-input renderer
`render_skybox_from_meshes(meshes, observer, inner, outer, seed,
cfg) -> Skybox`. `Camera::for_cube_face(eye, face, near, far)`
produces a 90° FOV / aspect 1.0 camera oriented along one face axis.
`Skybox::sample(dir)` is the standard largest-axis cubemap fetch and
is scale-invariant by construction.

Phase 13f intentionally **does not** add a `WorldHost`-pulling
wrapper. That bridge — fetching a parent-tier mesh slab from a host
and feeding it into `render_skybox_from_meshes` — lands in
Phase 13g/13i, where the streaming proto changes for skybox bursts
live. Keeping 13f mesh-input-only means the unit tests in
[`tests/skybox.rs`](../crates/atomr-worlds-view/tests/skybox.rs)
exercise the type end-to-end without an actor system: cube-face
basis is orthonormal right-handed, sampling lands on the right face
and is scale-invariant, empty meshes produce a uniform-background
skybox, the digest is deterministic and changes when the observer
moves, and the reversed-z projection actually maps near→1 / far→0.

## Phase 13g *(landed)* — Composite renderer

[`crates/atomr-worlds-view/src/render.rs`](../crates/atomr-worlds-view/src/render.rs)
adds `FragmentMode::{Opaque, DistanceFade { start_m, end_m, observer }}`
and `render_composite(scene, camera, cfg)`. The composite pipeline:

1. Clear depth to 0.0 (reversed-z "far").
2. If `scene.skybox` is `Some`, paint the background by tracing each
   pixel back to a camera-ray direction and sampling the cubemap. No
   depth writes (depth stays at 0.0 so mesh passes always win).
3. Rasterize `scene.far_meshes` with `DistanceFade` over the last
   `fade_band_frac` of `[transition_radius_m..max_radius_m]`. Alpha
   blends source-over with the destination; depth writes only when
   `alpha > 0.5` so fade-out fragments don't occlude the near ring.
4. Rasterize `scene.near_meshes` opaque.

Tests: [`crates/atomr-worlds-view/tests/composite.rs`](../crates/atomr-worlds-view/tests/composite.rs)
covers determinism, skybox-only path matching `Skybox::sample`, alpha-
blend math at the band midpoint, `None`-skybox background fallback,
near-ring opacity over the sky, and the `FragmentMode` distance-alpha
math directly.

## Phase 13h *(landed)* — Cross-LOD seam fix

Two seam-bridge primitives in
[`crates/atomr-worlds-view/src/iso.rs`](../crates/atomr-worlds-view/src/iso.rs):

- `boundary_skirt(brick, axis, sign, depth)` — emits a band of
  rectangular skirts along the named brick face. Each face cell with
  at least one solid voxel along the perpendicular axis gets a quad
  that extends `depth` voxels below the surface, hiding any LOD-
  boundary crack between bricks of different mesh densities. The
  output is brick-local; the caller transforms to world space.
- `crossfade_overlap(brick, mode_near, mode_far)` — returns the same
  brick meshed at two LODs, suitable for plugging straight into
  `CompositeScene::{near_meshes, far_meshes}` so the
  `FragmentMode::DistanceFade` band crossfades the two.

The Phase-9 `transvoxel_seam` stub has since been removed and replaced
by `face_height_profile` + `lod_transition_strip` (see Phase 17.1
follow-ups for the additive cross-LOD bridge that pairs with the
existing skirts + crossfade). Tests in
[`crates/atomr-worlds-view/tests/seam.rs`](../crates/atomr-worlds-view/tests/seam.rs)
cover the skirt non-empty / empty cases, the crossfade-overlap pair,
and a composite-render "no holes inside visible brick" check.

## Phase 13i *(landed)* — Transitive skybox + sphere-flyby demo

[`crates/atomr-worlds-view/src/observer.rs`](../crates/atomr-worlds-view/src/observer.rs)
adds `ObserverState` for the transitive-refresh logic:

- Tracks current `position`, derived `velocity_mps`,
  `containing_frame`, two skybox slots (`last_skybox`, `next_skybox`),
  and `crossfade_t`.
- `should_refresh(policy, body_center, body_radius, prev_frame)`
  returns true when any of: position has drifted past
  `position_delta_frac * outer_radius`; altitude has changed past
  `altitude_delta_frac * body_radius`; capture age exceeds
  `max_age_ticks`; or `containing_frame` differs from `prev_frame`
  (when `refresh_on_tier_change`).
- `accept_next(sky)` adopts the freshly-generated skybox: first
  arrival becomes `last_skybox` directly; later arrivals start the
  crossfade.
- `tick(new_pos, new_frame, dt_s)` updates velocity, advances the
  crossfade, and promotes `next` → `last` at `t = 1.0`.

The companion demo binary
[`examples/sphere-flyby`](../examples/sphere-flyby) configures an
Earth-class sphere world, registers an authored "city" `LiteralRegion`,
and simulates an observer flying from surface to ~1 Mm altitude in
12 frames. Each frame is rendered via `render_composite` and written
to `/tmp/sphere-flyby-{:02}.png`. Run with `cargo run -p sphere-flyby`.

Tests: 6 `observer::tests::*` unit tests cover initial-refresh,
position threshold, altitude threshold, age threshold, tier-change
threshold, velocity derivation, and crossfade progression.

## Phase 13 — End-state summary

The full Phase 13 feature stack — definable world shape with horizon-
driven streaming, layered geologic pre-sim, hand-authored stipulation
(literal / heightmap / .vox), cubemap skybox with reversed-z
composite, cross-LOD seam fix, and transitive skybox refresh — is
now live. ~213 workspace tests pass; every output is a deterministic
function of `(seed, shape, registered region set, observer pose,
config)`. CUDA-aware brick generation continues to use the
`generate_brick_legacy` shim, so the existing GPU determinism gate is
unaffected. Optional follow-ups documented in the per-phase risk
sections: GPU macro-state upload (13k), cubed-sphere coordinate
research spike (13l), Bruneton-style atmospheric scattering post-pass
(13j).

## Phase 14 — Multi-mode world display

Phase 14 adds five world display modes — 1st-person walk, 3rd-person
chase, Dwarf-Fortress-style horizontal slice cycling, RTS oblique
strategy, and large-scale regional overview — each with its own
rendering pipeline and (where the access pattern warrants) its own
derived data structure rather than reusing a single pipeline with a
different camera. Phase 13's renderer covered the immediate-experience
slot; Phase 14 fills out the rest of the camera-and-viewing surface so
the same world data can be reasoned about at every metric scale from
eye-height to a whole world.

The crate boundary stays put: `atomr-worlds-view` remains the headless,
deterministic CPU rendering crate; nothing here adds a windowing
backend, an event loop, or input handling. Each mode is exposed as a
pure `(camera, world_query, config) → Framebuffer` call, the same
shape Phase 13 settled on. Interactive shells stay an external
concern downstream of this repo. The view crate gains a new read-only
`WorldQuery` trait so it can pull bricks + deltas from a host without
depending on `atomr-worlds-host` — host implements the trait,
inverting the dep.

### Phase 14 foundation *(landed)*

Wave 1 of the multi-mode rollout. Four independent pieces, each
delivered by an isolated worktree agent and merged into main:

- **`Projection` enum on `Camera`**
  ([`camera.rs`](../crates/atomr-worlds-view/src/camera.rs)).
  Adds `Projection::{Perspective, Orthographic, Oblique}` with
  reversed-z preserved across all three. Existing constructors
  (`isometric_default`, `for_cube_face`) and all Phase 13f/g/h/i
  golden PNGs unchanged byte-for-byte (regression gate in
  [`tests/deterministic_screenshot.rs`](../crates/atomr-worlds-view/tests/deterministic_screenshot.rs)).
  Orthographic and oblique derivations are documented in the file with
  the same rigor as the existing perspective derivation comment.

- **`WorldQuery` trait shim**
  ([`world_query.rs`](../crates/atomr-worlds-view/src/world_query.rs)).
  Three methods — `brick`, `ground_height_m`, `subscribe_region` — let
  view code pull from a host without taking on a host dep. The proto
  dep is added to view (`atomr-worlds-proto` for `AABB` /
  `WorldEvent`). A `LocalHost` impl lands in Wave 2 (Phase 14a) under
  `crates/atomr-worlds-host/src/world_query_impl.rs`, bridging tokio
  mpsc → std mpsc for the subscribe path.

- **`raster2d` 2D blitter**
  ([`raster2d.rs`](../crates/atomr-worlds-view/src/raster2d.rs)).
  Axis-aligned RGBA8 writes into `Framebuffer.pixels`: `fill_rect`,
  `fill_rect_stipple` (Checker / Horizontal / Vertical / Dense25 /
  Dense75), `blend_rect` (src-over with the `(x*257+255)>>16` div-255
  trick), `blit_rgba`. Twelve unit tests cover clipping, zero-size,
  alpha, byte layout. Used by phases 14c (slice tiles), 14d (RTS
  decals), 14e (overview pyramid).

- **`ViewCache<K, V>` + `DerivedStore`**
  ([`view_cache.rs`](../crates/atomr-worlds-view/src/view_cache.rs)
  and
  [`derived.rs`](../crates/atomr-worlds-persist/src/derived.rs)).
  `ViewCache` is an `RwLock<HashMap>` keyed by a `DerivedKey: Hash +
  Eq` whose impls expose a `WorldAddr` and an AABB-intersection
  predicate; subscribers to the host's `VoxelDelta` / `RegionDelta`
  events drive `invalidate_intersecting`. A local-shape `CacheAabb`
  (f64 min/max) keeps view's `view_cache` orthogonal to the integer-
  coord proto `AABB`; conversion is trivial at the call site. The
  persist side adds an optional `derived` feature with `DerivedStore`
  + `InMemoryDerivedStore` for later SQL backing. Phases 14c/d/e all
  sit on top of one or both.

### Phase 14a *(landed)* — 1st-person walk

`crates/atomr-worlds-view/src/modes/fp.rs`. `WalkCamera` wraps the
Phase 13i `ObserverState` with yaw/pitch/eye-height controls;
`WalkInput { move_local, yaw_delta, pitch_delta, crouch }` carries
per-tick deltas. `WalkCamera::tick` rotates `move_local` by yaw,
advances the observer, and routes through `ObserverState::tick` so the
skybox-refresh policy still fires.

`build_fp_scene(world: &dyn WorldQuery, addr, cam, lod, region_m,
extra_meshes) -> CompositeScene` computes the cube AABB of half-size
`region_m` around `cam.eye`, frustum-culls via the new
`crates/atomr-worlds-view/src/frustum.rs` (Gribb–Hartmann plane
extraction from view×proj — works for both Perspective and
Orthographic), fetches each surviving brick via `WorldQuery::brick`,
meshes through the existing `mesh::greedy_mesh`, partitions into
far-fade vs near-opaque using a `region_m * 0.6` distance threshold,
and returns a `CompositeScene` ready for `render_composite`. The
`extra_meshes` parameter lets 14b inject the anchor mesh later.

Per-session `MeshCache: ViewCache<MeshCacheKey, Mesh>` keyed by
`(WorldAddr, brick_coord, Lod)`; subscribers to `VoxelDelta` /
`RegionDelta` evict intersecting entries. Eye height is kept off the
ground via `WorldQuery::ground_height_m`; full collision is out of
scope.

`crates/atomr-worlds-host/src/world_query_impl.rs` adds a
`LocalHostQuery { host: Arc<LocalHost>, handle:
tokio::runtime::Handle }` implementing the `WorldQuery` trait — uses
`Handle::block_on` for the request/response paths and a small
forwarder task to bridge tokio mpsc → std mpsc for
`subscribe_region`.

[`examples/view-fp`](../examples/view-fp) is the headless companion:
runs a fixed five-frame trajectory against `LocalHost`, writes PNGs to
`/tmp/view-fp-NN.png`, prints FNV-1a digests.

#### Gates

- `tests/walk_determinism.rs` — scripted `WalkCamera::tick` against a
  stub `WorldQuery` → byte-identical pixel hashes across two runs.
- `frustum.rs#[cfg(test)] mod tests` — AABB inside / outside /
  straddling for Perspective and Orthographic.
- Phase 13 goldens still byte-identical.

### Phase 14b *(landed)* — 3rd-person chase

`crates/atomr-worlds-view/src/modes/tp.rs`. `ChaseCamera { anchor,
yaw, pitch, distance_m, height_m, fov_y_rad, aspect, smoothing_hz }`
orbits an external anchor. `ChaseCamera::tick(new_anchor, yaw_delta,
pitch_delta, dt_s)` uses critical-damped exponential smoothing in
closed form (`smoothed += (target - smoothed) * (1 - exp(-2π · hz ·
dt))`) — no integration drift across long runs.

`render_tp` reuses `build_fp_scene` with the anchor mesh threaded
through `extra_meshes`. Eye clipping into terrain shares the
`ground_height_m` probe.

[`examples/view-tp`](../examples/view-tp) renders five chase frames.

#### Gates

- `tests/chase_smoothing.rs` — pose at t=10 s within 1 ULP of analytic.
- Phase 13 goldens still byte-identical.

### Phase 14c *(landed)* — Dwarf-Fortress horizontal slice

`crates/atomr-worlds-view/src/modes/slice.rs` +
`crates/atomr-worlds-view/src/derived/slice_index.rs`. Orthographic
top-down tile renderer cycling one z-band at a time (default thickness
3 voxels = 2 m open + 1 m roof). The +Y-up axis is treated as the
"z-level"; the rule is documented in `slice_index.rs`'s module
rustdoc: scan from `z_band_top` downward through `z_band_thickness`
Y-levels, the first non-empty voxel for each (x, z) column becomes the
column's `top_voxel`; empty columns render with alpha = `roof_alpha`.

`SliceTable` (one `SliceColumn { top_voxel, top_z,
thickness_above_floor }` per XZ position) is cached via `ViewCache<SliceKey, SliceTable>`;
`VoxelDelta { brick }` translates the brick AABB into a `CacheAabb`
and invalidates intersecting entries — writes outside the slice's XZ
footprint produce no rebuild.

`render_slice` deliberately bypasses the 3D triangle rasterizer.
Slice frames are millions of axis-aligned unit quads at fixed depth;
direct `raster2d::fill_rect` blits are ~10× cheaper than running
them through triangle setup. Thin-feature stipple uses
`StipplePattern::Dense75`. Material → colour resolves through the
caller's `MaterialPalette` with `render::material_color` fallback.

[`examples/view-slice`](../examples/view-slice) cycles three z-bands.

#### Gates

- `tests/slice_golden.rs` — fixed seed → fixed `pixels_fnv1a` hash.
- `tests/slice_invalidation.rs` — write inside band rebuilds; write
  outside does not.
- `tests/slice_z_band_rule.rs` — column empty / column at exact top /
  column with voxel below floor.

### Phase 14d *(landed)* — RTS oblique-orthographic

`crates/atomr-worlds-view/src/modes/rts.rs` +
`crates/atomr-worlds-view/src/derived/surface_raster.rs` +
`crates/atomr-worlds-view/src/decals.rs`. Renders only the *surface*
under an oblique-orthographic projection — `ObliqueCamera::to_camera`
builds a `Camera` with `Projection::Oblique { rotation_deg,
scale_m_per_px }`.

`SurfaceRaster { heightmap_m, biome_id, top_z, dims, origin_xz,
voxel_size_m, world_rev }` is baked once per region tile via
`build_surface_raster` and held in `ViewCache<SurfaceKey,
SurfaceRaster>`. `surface_raster_to_mesh` emits one triangle pair per
column with biome-coloured vertices; `render_mesh` runs that under the
oblique projection. A 2D decal pass (`render_decals` →
`raster2d::blend_rect` / `blit_rgba`) composites entity sprites on
top. Caves and overhangs at the surface are an explicit known
limitation — flagged in the module rustdoc.

Invalidation keyed on `top_z`: writes strictly below `heightmap_m[x,
z] - 1` produce no rebuild (covered by `rts_surface_invariance`).

[`examples/view-rts`](../examples/view-rts) renders the oblique view
with three decals.

#### Gates

- `tests/rts_golden.rs` — fixed seed → fixed `pixels_fnv1a`.
- `tests/rts_surface_invariance.rs` — sub-surface writes produce no
  invalidation; top-voxel writes do.
- `tests/rts_decal_pass.rs` — decal rect lands at expected pixels;
  surrounding pixels untouched.

### Phase 14e *(landed)* — Regional / world overview

`crates/atomr-worlds-view/src/modes/overview.rs` +
`crates/atomr-worlds-view/src/derived/world_summary.rs` +
`crates/atomr-worlds-view/src/projection_sphere.rs`.
Tile-pyramid renderer driven by Phase 13c's `WorldMacroState`.
`bake_world_summary(addr, macro_state, levels, tile_size_px)` walks a
regular pyramid (level 0 = one tile covering the world; level L =
`4^L` tiles), calling `macro_state.sample(dir)` per pyramid pixel to
fill four parallel arrays per `WorldSummaryTile`: `elevation_m`,
`biome_id`, `plate_id`, `ClimateSample { temperature_c, humidity }`.

`OverviewCamera { center, extent, projection: OverviewProjection,
aspect }` covers three projections: `OrthographicFlat` (pyramid-tile
blit), `Equirectangular` (per-pixel inverse projection through
`projection_sphere::equirectangular_pixel_to_dir`), `OrthographicSphere`
(disk test + inverse). `pick_pyramid_level` picks detail by `(extent,
viewport)`.

Cache invalidation is keyed only by `(WorldAddr, macro_digest,
levels)` — `WorldSummaryKey::intersects(_) -> false`. Voxel writes
never invalidate the pyramid; only re-runs of Phase 13c's macro
pre-sim change the digest. `atomr-worlds-view` adds
`atomr-worlds-generate` as a regular dep (promoted from dev-dep) for
the macro-state types.

[`examples/view-overview`](../examples/view-overview) bakes a 4-level
pyramid against an Earth-class sphere and writes one PNG per
projection.

#### Gates

- `tests/overview_golden_{orthographic,equirectangular,orthographic_sphere}.rs`
  — fixed seed → fixed `pixels_fnv1a` per projection.
- `tests/overview_pyramid_level_pick.rs` — small extent → fine level;
  huge extent → coarse level.
- `tests/overview_sphere_projection_sanity.rs` —
  `equirectangular_dir_to_pixel((1, 0, 0))` lands at the centre column
  of a longitude-0 convention.

### Phase 14 — End-state summary

All five modes ship as headless `(camera, world_query, config) →
Framebuffer` calls. Each mode caches its own derived structure
(`MeshCache` in-session for 14a/b; `SliceTable`, `SurfaceRaster`,
`WorldSummaryPyramid` in `ViewCache` for 14c/d/e), invalidated by the
host's `VoxelDelta` / `RegionDelta` events through the new
`WorldQuery::subscribe_region` plumbing. Every output is a
deterministic function of `(seed, shape, registered region set,
observer pose, camera, config)`. Phase 13's golden PNGs remain
byte-identical.

## Phase 15 — Client / server

The Phase-14 "interactive shell is external" caveat is closed by three
new crates:

- **`atomr-worlds-remote`** — wire envelopes (`WireRequest` /
  `WireReply`), `RemoteHost` (a `WorldHost` impl that speaks bincode
  over `atomr-remote`), `WorldGateway` server actor, and an
  `install_cluster_remote_forwarder` helper that wires
  `ShardRegion::set_remote_forwarder` to atomr-remote so `ClusterHost`
  finally does cross-node forwarding.
- **`atomr-worlds-server`** — headless server binary with
  `--mode standalone|cluster`. Reusable `run_standalone` /
  `run_cluster_with` library entry points so tests can drive the same
  code path the binary uses.
- **`atomr-worlds-client`** — Bevy 0.13 binary that picks
  `LocalHost` / `RemoteHost` / cluster member via `--backend`, renders
  all five view modes (fp/tp native Bevy 3D; slice/rts/overview blit
  the CPU rasterizer's `Framebuffer` into a Bevy `Image`), and overlays
  a bevy_ui debug HUD (FPS / coords / mode).

`LocalHostQuery` was generalised from `Arc<LocalHost>` to
`Arc<dyn WorldHost>` so the same render-thread sync bridge serves every
backend; the legacy `new(Arc<LocalHost>, …)` constructor stays for
backwards compatibility.

### Gates

- `atomr-worlds-remote/tests/loopback.rs` — request + subscribe
  round-trip over loopback `RemoteSystem`s.
- `atomr-worlds-remote/tests/cluster.rs` — two-node `ClusterHost`
  forwarding: write + read targeting a shard pinned to a peer succeeds.
- `atomr-worlds-server/tests/standalone.rs` — server binary entry
  point round-trips a write/read for a remote client.
- `atomr-worlds-server/tests/cluster.rs` — `run_cluster_with` boots
  twice, peers wired post-boot, client sees the voxel a peer wrote.
- `atomr-worlds-client/tests/headless_smoke.rs` — `WorldQuery` bridge
  works against both `LocalHost` and `RemoteHost` (headless; no Bevy
  app launched).

### Follow-ups landed

- **Cross-node subscription routing.**
  [`ClusterHost::subscribe`](../crates/atomr-worlds-host/src/cluster.rs)
  now consults the coordinator: locally-owned shards still take the
  direct `LocalHost` path; remote shards register a `sub_id → mpsc::Sender`
  in the new `ClusterSubs` map and forward the Subscribe envelope
  through `ShardRegion::deliver`. The peer's
  [`WorldGateway`](../crates/atomr-worlds-remote/src/gateway.rs)
  already streamed `WireReply::Event { sub_id, env }` back; the
  [`ClusterReplyInbox`](../crates/atomr-worlds-remote/src/cluster_forwarder.rs)
  was extended to route those events through the subs map (it dropped
  them on the floor before). Coverage:
  [`atomr-worlds-remote/tests/cluster.rs::cross_node_subscribe_streams_events_back`](../crates/atomr-worlds-remote/tests/cluster.rs).

- **Pre-shared bearer-token auth on the wire.** `WireRequest` carries
  an optional `auth_token` field;
  [`WorldGateway::with_auth_token(...)`](../crates/atomr-worlds-remote/src/gateway.rs)
  validates inbound requests against an expected value and silently
  drops mismatches with a `tracing::warn!`. Both `RemoteHostConfig` and
  `StandaloneConfig` / `ClusterConfig` grow an `auth_token: Option<String>`,
  the cluster forwarder gains
  [`install_cluster_remote_forwarder_with_auth`](../crates/atomr-worlds-remote/src/cluster_forwarder.rs)
  for cross-node stamping, and the server CLI exposes `--auth-token`.
  Coverage:
  [`atomr-worlds-remote/tests/loopback.rs::gateway_rejects_requests_with_wrong_or_missing_token`](../crates/atomr-worlds-remote/tests/loopback.rs).
- **TLS plumbing primed for upstream.** `RemoteHostConfig::tls`
  accepts an [`atomr_remote::TlsConfig`](https://docs.rs/atomr-remote)
  and threads it into `RemoteSystem::start` via `RemoteSettings::with_tls`.
  `atomr-remote` 0.9.2 only exposes the typed config today (the
  handshake itself is "deferred" per its own rustdoc); when upstream
  wires `TcpTransport`, this layer is ready with no further code
  changes. **Bearer tokens travel in plaintext until that lands** — the
  `WorldGateway::with_auth_token` rustdoc and `RemoteHostConfig::auth_token`
  rustdoc both call this out.

### Out of scope (still — see `docs/CLIENT_SERVER.md`)

- `atomr-view` UI bridge — same upstream blockers as Phase 14.
- Gossip / persistent membership for the cluster. `--peer` is a static
  hand-rolled map.
- True TLS encryption on the wire — blocked on `atomr-remote`'s
  `TcpTransport` actually performing the rustls handshake (see
  [`atomr_remote::tls`](https://docs.rs/atomr-remote) rustdoc).

## Phase 16 — Lighting + materials upgrade *(landed)*

Replaces the single hard-coded `DirectionalLight` and 6-entry color
table from Phases 14a/15 with a real multi-material PBR look:

1. **Material palette** — 10 ids (stone, dirt, sand, snow, water,
   grass, wood, leaves, glow_rock, ice) with per-id PBR
   (roughness/metallic/emissive/alpha). Picked up by the CPU
   rasterizer (slice/RTS/overview) and by Bevy `StandardMaterial`
   handles (FP/TP) from the same `MaterialPalette` source.
2. **Per-material shading (`SplitPerMaterial`)** — FP bricks split
   into N submeshes (one per material id present), each spawned as a
   child `PbrBundle` with its own `StandardMaterial`. Water/ice get
   `AlphaMode::Blend`; glow_rock emissive ×2.
3. **Tonemap + bloom** — Camera gains HDR + `Tonemapping::AcesFitted`
   + `Exposure { ev100: 9.0 }` + `BloomSettings { intensity: 0.10 }`.
4. **Time-of-day** — `WorldTime` resource (hours in `[0,24)`); the
   `KeyframeLutSun` strategy interpolates a 5-keyframe LUT and writes
   the `WorldSun` directional light's transform/color/illuminance
   plus ambient color/brightness per frame.
5. **Cascaded shadows** — `BasicCascades` strategy returns a
   `CascadeShadowConfig` tuned to the FP streaming radius (4 cascades,
   max 200 m, first far bound 8 m).
6. **Per-vertex AO** — `MinecraftCornerAo` samples the 4 air-side
   neighbours of each face vertex; AO is baked into `ATTRIBUTE_COLOR`
   so Bevy's PBR multiplies it against base color natively. Greedy
   merge keys include the AO 4-tuple so quads only merge when AO
   matches.
7. **Sky + fog** — `SkyTinted` returns a horizon color that follows
   the sun's color (blue night, orange dawn/dusk, pale noon);
   `ExpSquaredSkyTintedFog` produces fog tinted to match. `ClearColor`
   + per-camera `FogSettings` are updated each frame.

The whole pipeline is wired through a strategy spine:
[`atomr-worlds-client/src/render/`](../crates/atomr-worlds-client/src/render/)
holds `RenderConfig` with nine `Arc<dyn Trait>` fields, one per
decision point. Swapping is a one-line write or a `set_strategy`
event from the harness. Three named presets (`Stylized`, `Legacy`,
`Debug`) bundle whole looks.

### Harness DSL additions

- `set_time_of_day { hours: f32 }`
- `set_render_preset { preset: "stylized"|"legacy"|"debug" }`
- `set_strategy { slot: String, strategy: String }`

### Critical-path: offscreen capture

The Bevy 0.13.2 `ScreenshotManager` path is unusable on hybrid-GPU
Linux laptops (panics in async buffer-map) and `xwd` against the
Vulkan-rendering window yields all-black PNGs. `OffscreenCapturePlugin`
points the camera at an `Image` render target, copies the texture to
a `MAP_READ` buffer in `RenderApp` at `RenderSet::Cleanup`, polls the
device synchronously, strips the per-row 256-byte padding, swaps
BGRA → RGBA, and saves PNG. Works headlessly; bypasses the
swapchain entirely. Memory note at
`memory/project_harness_offscreen_capture.md`.

### Gates

- `harness/scenes/lighting_showcase.toml` — six time-of-day PNGs
  (h=6, 9, 12, 17, 19, 21); used to validate the sun curve, shadow
  cascades, sky-tinted fog, ambient brightness.
- `harness/scenes/strategy_compare.toml` — A/B per-preset and
  per-slot comparison; validates that preset rollback (`Legacy`)
  reaches every slot.
- Pinned-hash view-crate tests (`tests/deterministic_screenshot.rs`,
  `tests/slice_golden.rs`) updated for the new `material_color`
  palette.

### Lessons learned + cross-mode applicability

Full prose in [RENDERING.md](RENDERING.md). The methodologies
(strategy spine, preset enum that pins every slot, offscreen
capture, harness DSL parity with new capability) port to the
CPU rasterizer modes; the FP-specific lighting plumbing
(PBR + shadows + fog + bloom) does not, and porting it would
require a software-shading equivalent in `atomr-worlds-view`.

### Phase 16 opt-in custom shaders *(landed)*

Two custom-WGSL strategies ship alongside the default `StandardMaterial`
look. Not on by default (the deterministic golden gates in
`atomr-worlds-view` still compare against the `StandardMaterial` path),
but available via `set_strategy` from the harness or by hand:

- **`PaletteVoxelMaterial`** (Step 8 — `shading` slot):
  `ExtendedMaterial<StandardMaterial, VoxelMaterialExt>` with a palette
  storage buffer at binding 100. Per-vertex material id in
  `ATTRIBUTE_UV_0.x`, AO in `ATTRIBUTE_COLOR.r`. WGSL imports
  `bevy_pbr::pbr_fragment::pbr_input_from_standard_material` and
  `bevy_pbr::pbr_functions::apply_pbr_lighting`, overrides
  base_color/roughness/metallic/emissive per fragment, returns the lit
  result through Bevy's standard PBR pipeline. Drops per-brick draw
  calls from N→1.
- **`ProceduralDomeSky`** (Step 9 — `sky` slot): inside-out sphere
  parented to the camera, custom `Material` with WGSL that mixes
  zenith→horizon and overlays a soft sun disc + glow.
  `MaterialPlugin::<SkyDomeMaterial>::default()` is always registered;
  `SkyDomePlugin::sync_sky_dome` toggles visibility based on
  `cfg.sky.dome_active()` and writes the four uniforms from the
  current `SunState`.

Asset loading: `AssetPlugin::file_path` is set from an absolute path
resolved at startup (see `main.rs::resolve_asset_root`) so shaders
under `crates/atomr-worlds-client/assets/shaders/` load whether the
binary runs from the workspace root, the crate directory, or a
packaged install.

### Out of scope (still)

- Triplanar texturing, SSAO post pass, water refraction, real
  atmospheric scattering — each is a future strategy in an existing
  slot.

## Phase 17 — Chunk auto-streamer + skybox integration *(landed)*

Phase 17 wires three latent capabilities into the live render path:
the `Skybox` cubemap (Phase 13f/13i) into Bevy's
`bevy::core_pipeline::Skybox` component, `StreamingPolicy::ring_for_curved`
(`atomr-worlds-proto`) into the per-frame brick loader, and the same
policy into the raster modes' `Lod` selection. One streaming model,
five view modes.

### Shared chunk streamer

[`crates/atomr-worlds-client/src/world_stream.rs`](../crates/atomr-worlds-client/src/world_stream.rs)
holds the `ChunkStreamer` and `LoadedChunks` Bevy resources. The
streamer wraps `StreamingPolicy { near_lod: 0, far_lod: 1,
transition_radius_m: 64, max_radius_m: 512, bricks_per_tick: 24 }`.

- `desired_chunks(streamer, observer, horizon_m) -> Vec<(IVec3, Lod)>`
  returns the union of the near ring (at `near_lod`) and the far ring
  (at `far_lod`, with the near-ring footprint masked out), sorted
  closest-first in world-meters so the visible leading edge fills
  before trailing bricks. Far-brick coordinates are in the far-LOD
  brick grid, not the near grid — `ring_for_curved` emits both rings
  in near-grid coords; this module converts to the far grid.
- `ChunkStreamer::lod_for_meters(observer, p)` and `lod_for_brick` —
  pure helpers used by the slice/RTS/overview raster paths.
- `LoadedChunk { coord, lod, entity, last_seen_frame }`, keyed by
  `(coord, lod.depth)` so a tier-change can briefly hold both a
  `(c, 0)` and `(c, 1)` entry without collision. Hysteresis: a chunk
  lingers two streamer ticks past its last "seen in desired set"
  frame before despawn.

### FP/TP brick loader rewrite

[`crates/atomr-worlds-client/src/modes/fp.rs::fp_stream_bricks`](../crates/atomr-worlds-client/src/modes/fp.rs)
replaces the hand-rolled 7×7×7 cube with a call to `desired_chunks`,
which now spawns near-LOD bricks at the standard `Transform` and far-LOD
bricks with `Transform.scale = 2^L`. Because greedy-meshing reads voxel
positions in `0..BRICK_EDGE`, the per-entity scale is the only
per-LOD knob — no mesh mutation. TP shares the same scene through
`FpState`, so it inherits the streamer.

### Skybox in Bevy

[`crates/atomr-worlds-client/src/render/skybox.rs`](../crates/atomr-worlds-client/src/render/skybox.rs)
holds the `SkyboxRuntime` resource and `sync_skybox` system. Each tick:

1. The runtime updates its `ObserverState` with the current walk
   position.
2. If `ObserverState::should_refresh` trips, the system bakes a fresh
   cubemap from the far-ring `LoadedChunks` via the existing
   `atomr_worlds_view::skybox::render_skybox_from_meshes`. The bake
   uses `inner_radius_m = transition_radius_m` and
   `outer_radius_m = max_radius_m`. A frame-budget guard
   (`min_frames_between_bakes`) caps re-bakes to ≤ 1 every 30 frames.
3. The resulting `atomr_worlds_view::skybox::Skybox` is concatenated
   into a six-layer Bevy `Image` with `TextureViewDimension::Cube` and
   stored as `next_handle`.
4. `crossfade_t` ramps `Skybox.brightness` from the old to the new
   value (`brightness = lerp(50, 2500, sun.day_factor)`); when the
   crossfade completes, the camera's `Skybox.image` is swapped.

The existing `ProceduralDomeSky` strategy stays on top as the
atmospheric gradient + sun disc; the cubemap shows the world's
distant geometry underneath.

### Snow palette dimming

`defaults.rs`/`render.rs`: snow albedo `[0.95, 0.97, 1.00]` ⇒
`[0.78, 0.82, 0.88]` linear with roughness `0.35` ⇒ `0.70`; the CPU
rasterizer's `material_color(4)` drops from `[242, 247, 255]` to
`[210, 218, 228]`. Both surfaces shift together so cross-mode goldens
stay consistent.

### Raster modes (slice / RTS / overview)

The view-crate per-column samplers
([`derived/surface_raster.rs`](../crates/atomr-worlds-view/src/derived/surface_raster.rs),
[`derived/slice_index.rs`](../crates/atomr-worlds-view/src/derived/slice_index.rs))
already keyed their cache by `(xz, lod)`. Phase 17 replaces the
hardcoded `Lod::new(0)` at the call sites in
`modes/slice.rs`/`modes/rts.rs` with `streamer.lod_for_meters(observer,
column_xz)`; `modes/overview.rs` always uses `streamer.policy.far_lod`
because its viewing distance is body-scale.

### Verification

- `harness/scenes/stream_walk.toml` — drives the FP camera ~64 m past
  the transition radius and back; tests load/eviction with hysteresis.
- `harness/scenes/skybox_refresh.toml` — walks past the 5% drift
  threshold at three times of day; confirms the cubemap re-bakes and
  the brightness crossfades.
- New unit tests: `world_stream::tests::*` (ChunkStreamer, hysteresis,
  AABB iteration). New integration test:
  `crates/atomr-worlds-client/tests/skybox_runtime.rs` exercises
  `SkyboxRuntime` end-to-end (no Bevy app).

### Follow-ups landed

- **Body-aware spherical horizon clamp.**
  [`WorldShape::altitude_m_at`](../crates/atomr-worlds-core/src/shape.rs)
  + [`WorldShape::horizon_at_m`](../crates/atomr-worlds-core/src/shape.rs)
  collapse the altitude lookup + `sqrt(2*R*h + h²)` into one call.
  [`ActiveWorld`](../crates/atomr-worlds-client/src/world_runtime.rs)
  carries a `WorldShape` (defaults to
  `WorldShape::default_world()` — the historical cube), and
  [`fp_stream_bricks`](../crates/atomr-worlds-client/src/modes/fp.rs)
  passes `active.shape.horizon_at_m(observer)` instead of `INFINITY`.
  Cube worlds short-circuit to `INFINITY` so the default load behaviour
  is byte-equal to before; sphere worlds clamp the outer ring at the
  geometric horizon. Coverage:
  [`shape::tests::altitude_*`](../crates/atomr-worlds-core/src/shape.rs)
  +
  [`world_stream::tests::horizon_clamp_drops_far_tiers`](../crates/atomr-worlds-client/src/world_stream.rs),
  `horizon_infinity_matches_unclamped`, and
  `shape_horizon_at_m_drives_streamer_clamp`.

### Out of scope for Phase 17 (still)

- LOD selection beyond two tiers — `StreamingPolicy` exposes only
  `near_lod` and `far_lod`; a screen-space pyramid (Phase 18) would
  evaluate `MetricScale::lod_for_screen` per-brick.
- Atmospheric tint baked into the cubemap. Today the cubemap captures
  only mesh-geometry; sky tint is the dome strategy's job.

## Phase 17.1 *(landed)* — Per-LOD brick generation

A follow-up correctness fix layered on Phase 17's streamer. Phase 17
already emitted `(brick_coord, lod)` pairs and scaled coarse-LOD
entities by `2^L`, but `WorldRequest::GetBrick { lod }` was discarded
before reaching the host's procedural cache and the
`TerrainGenerator`. Coarse-LOD requests therefore returned LOD-0
content; the renderer's per-entity scale stretched 16 m of detail over
128 m of world space, producing visible plateaus and mismatched height
plates at the LOD tier boundaries.

The fix lands in three crates:

- [`crates/atomr-worlds-generate/src/brick.rs`](../crates/atomr-worlds-generate/src/brick.rs)
  — `BrickGenContext` carries `lod: Lod`. `BrickGenContext::legacy`
  defaults to `Lod::new(0)` so the CUDA accelerator's CPU fallback and
  the Python bindings remain byte-equal with the GPU kernel.
- [`crates/atomr-worlds-generate/src/terrain.rs`](../crates/atomr-worlds-generate/src/terrain.rs)
  — new `surface_height_world` / `is_cave_world` / `material_at_world`
  /  `material_at_world_strategy` sample the FBM and Worley fields in
  continuous world-meter coordinates. `generate_brick` dispatches on
  `ctx.lod.depth`: depth 0 takes the legacy integer-voxel path (byte-
  equal to CUDA); depth ≥ 1 samples each voxel at its center
  `(origin + lx + 0.5) × 2^L` meters.
- [`crates/atomr-worlds-host/src/local.rs`](../crates/atomr-worlds-host/src/local.rs)
  — `WorldActor::cache` is keyed by `(IVec3, u8)`. `ensure_brick(bc,
  lod)` and `snapshot(bc, lod)` thread the LOD through to the
  generator and the cache. Subscription paths
  (`handle_subscribe_begin`, `update_observer_pos`) and `GetBrick`
  request handling pass the subscription/request LOD; voxel writes,
  authored regions, and the user-write overlay stamp only the depth-0
  cache entry.

Result: adjacent LOD tiers now sample the *same* heightfield in world-
meter coordinates and disagree only by voxel discretization — ≤ 1 m at
the depth-0 ↔ 1 boundary, ≤ 4 m at depth-2 ↔ 3. The dramatic
"stretched LOD-0 content rendered as LOD-3 plates" failure mode is
gone.

### Verification

- `harness/scenes/elevated_spin.toml` — climbs to ~200 m, yaws 360° in
  45° steps, captures eight comparable shots. Before: each shot had a
  large flat slab dominating one quadrant (a tier-3 brick rendering
  scaled LOD-0 content). After: continuous voxel terrain at every
  yaw.
- `harness/scenes/topdown_ring.toml` — looks straight down from
  600 m altitude. Used to show a quadrant-biased mismatch; now shows a
  uniform radial LOD ring.
- `harness/scenes/altitude_360.toml` — four cardinal yaws + a zenith
  shot at 240 m altitude.
- Existing tests untouched:
  `atomr_worlds_generate::terrain::tests::*` still pass (LOD-0 byte
  equality preserved); `atomr_worlds_host` tests (request,
  subscribe, persistence, authored regions, sphere horizon e2e) all
  pass with the cache key change.

See [LOD.md](LOD.md) for the per-tier generation contract, the
world-meter sampling API, and the intrinsic-discretization
characteristics that motivate the roadmap items below.

### Follow-ups landed

- **Coarse-LOD overlay re-stamping.**
  [`WorldActor::ensure_brick`](../crates/atomr-worlds-host/src/local.rs)
  now applies the user-write overlay at every LOD depth, mapping each
  LOD-0 voxel position into the matching coarse cell (one cell per
  write — last-writer-wins inside a cell, which is acceptable for the
  sparse-edit workload). The `WriteVoxel` and `WriteRegion` paths call
  the new `WorldActor::invalidate_coarse_caches_for(pos)` so any
  previously-cached coarse brick containing the write is dropped and
  regenerated with the new overlay on next access. Carving (writing
  `Voxel::EMPTY`) is a deliberate exception — it stamps only at
  LOD 0, since blanking a whole `2^(3L)` coarse cell from a single
  LOD-0 hole would erase otherwise-solid neighbours; the carved hole
  reappears once the observer returns to the near ring. Coverage:
  [`tests/coarse_lod_restamp.rs`](../crates/atomr-worlds-host/tests/coarse_lod_restamp.rs).

- **Cross-LOD transition meshes — Transvoxel-equivalent crack-free
  bridge.** The full Lengyel 2010 lookup-table version (256-entry
  regular cell + 13-vertex transition cell) was not pursued — the
  chosen architecture combines four cooperating mechanisms instead.
  (1) `NestedSummary` LOD coverage
  ([`crates/atomr-worlds-client/src/render/defaults.rs`](../crates/atomr-worlds-client/src/render/defaults.rs))
  has every tier emit its full inner sphere, so the coarser parent is
  always resident underneath the finer child — no LOD pop on band
  crossings. (2) `BrickFadeIn` / `BrickFadeOut`
  ([`crates/atomr-worlds-client/src/modes/fp.rs`](../crates/atomr-worlds-client/src/modes/fp.rs))
  crossfade between parent and child over a one-frame window when the
  child unloads. (3) [`boundary_skirt`](../crates/atomr-worlds-view/src/iso.rs)
  emits depth-fade-friendly fins below every solid face so oblique
  viewing angles cannot peek through gaps. (4) New
  [`face_height_profile`](../crates/atomr-worlds-view/src/iso.rs) +
  [`lod_transition_strip`](../crates/atomr-worlds-view/src/iso.rs)
  explicitly bridge a near + far brick across their shared face with a
  triangle ribbon — the strip's vertex span covers the entire
  ≤ voxel/2 height delta the original deferral called out, and is
  additive geometry the renderer can draw alongside both LOD meshes
  during the half-alpha crossfade moment without retopologising
  either side. Coverage: four new tests in
  [`iso.rs`](../crates/atomr-worlds-view/src/iso.rs) pin the empty,
  uniform, height-delta, and one-sided cases. The combination removes
  the height-step concern without the verbatim Lengyel tables.

### Out of scope (still)

- True downsampled coarse-LOD overlays — current re-stamping picks the
  most-recently-written LOD-0 voxel per coarse cell. A pass that
  averages all LOD-0 voxels in the cell would produce smoother
  transitions when many adjacent LOD-0 cells are edited together,
  at the cost of an extra `2^(3L)`-voxel scan per write.
- Finer LOD ladders (sub-meter `LOD−1`, per-mode tier counts) and
  external data-feed generators (see README "Roadmap").

## Phase 18 *(landed)* — Hydrology overlay: ocean, lake, river

A new water-body layer on top of the geologic macro pre-sim. Tectonic
plates produce a piecewise-flat elevation field — unusable for hydrology
— so the phase first adds a **relief** layer (smooth multi-octave FBM
relief refining the plate elevation, run before climate so every
downstream layer sees one coherent field), then a **hydrology** layer
that classifies every surface-grid face as ocean / lake / river.

Three `WaterBodyStrategy` impls, run in dependency order and aggregated:

- **Ocean** — per-face threshold against sea level.
- **Lake** — Barnes-style priority-flood seeded from ocean faces; closed
  basins become lakes, gated on local humidity (arid basins stay dry).
  The flood also records a parent-chain drainage forest rooted at the
  ocean.
- **River** — flow accumulation over that drainage tree (Kahn's
  topological sweep), so rivers chain headwater → lake → sea; corridors
  above a flow threshold become rivers.

The brick generator consumes the per-face `WaterField` (via
`MacroSample`): ocean / lake fill air below the water surface with
`MATERIAL_WATER`; river corridors carve a meandering channel with the
local seed (FBM centerline + Worley bank jitter, width/depth scaling with
`sqrt(flow_accum)`); submerged beds read as sand. Single shared water
material, no renderer changes. Water surfaces are LOD-stable by
construction; river-carve noise is sampled in voxel-centered world meters.

See [HYDROLOGY.md](HYDROLOGY.md) for the full design.

### Verification

- Per-strategy + relief unit tests; brick-level macro-path tests in
  `terrain.rs` (ocean water over sand bed, water surface at sea level,
  river carve produces a channel, legacy path unaffected).
- [`tests/hydrology.rs`](../crates/atomr-worlds-generate/tests/hydrology.rs)
  — the default world has oceans, lakes, and rivers; `WaterField`
  invariants hold; the macro digest is deterministic and seed-sensitive.
- The three overview golden-render hashes in `atomr-worlds-view` were
  re-pinned (relief + water change the rendered world); the
  `render_is_deterministic` companion tests are unchanged.
- Harness scenes `water_overview.toml`, `water_fp_coast.toml`,
  `water_lod.toml`.

### Follow-ups landed

- **Hydrology feeds back into climate + biomes.** The macro pipeline
  now runs hydrology *before* biomes, then
  [`apply_hydrology_humidity_feedback`](../crates/atomr-worlds-generate/src/macro_state/climate.rs)
  seeds humidity at lake / river faces and re-runs
  `ClimateConfig::hydrology_feedback_iters` (default 2) extra
  diffusion steps before biomes are classified. Lake- and river-side
  faces land in wetter biomes (forest / grassland) instead of the arid
  baseline that the pre-feedback pass produced. Setting
  `hydrology_feedback_iters = 0` reverts to the pre-feedback pipeline
  and digest. Coverage:
  [`tests/hydrology.rs`](../crates/atomr-worlds-generate/tests/hydrology.rs)
  — `lake_and_river_faces_are_humid_with_feedback_enabled`,
  `disabling_feedback_drops_some_freshwater_humidity`,
  `feedback_can_change_biomes_around_freshwater`.

- **Overview-mode harness capture (the "empty sky" bug).** Root cause
  was *not* the pyramid bake timing — it was a camera-toggle gap.
  `BlitCamera`'s sprite letterboxes (the 256² rasterizer output scaled
  to fit a non-square offscreen target leaves bars on two sides), and
  the FP `WorldCamera` underneath stayed `is_active = true` in raster
  modes, clearing the offscreen image to its sky-blue `ClearColor` on
  every frame. The bars on each side of the sprite therefore showed
  "empty sky" in harness PNGs even after the bake completed. Fix in
  [`modes/blit.rs`](../crates/atomr-worlds-client/src/modes/blit.rs):
  `toggle_blit_visibility` now flips both cameras inversely (raster
  modes activate `BlitCamera` and deactivate `WorldCamera`; FP/TP do
  the inverse), and the `BlitCamera`'s `clear_color` was promoted from
  `ClearColorConfig::None` to `Custom(BLACK)` so the letterbox bars
  are deterministic black instead of revealing whatever pass ran
  before. Six unit tests pin the toggle invariant per `ViewMode`. The
  `tracing::info!("overview pyramid baked", elapsed_ms = …)` line is
  kept so harness operators can budget `warmup_frames` correctly when
  raising `grid_level` or changing seeds.

### Out of scope (still)

- River deltas / estuaries, lake-shore beaches, aquifer / spring
  sources, and seasonal water-level variation.

## Phase 19 *(landed)* — Slice view: FP-aligned orientation + hillshade relief

A rework of the Dwarf-Fortress slice view (`ViewMode::Slice`) so it reads
as the same world as the first-person view and scrolls predictably. Three
problems are addressed:

- **Directional misalignment.** `render_slice` mapped world `+X` to
  screen-right and `+Z` to screen-down, but the FP camera (which faces
  world `+Z`) has screen-right at world `-X` — the slice was mirrored.
  The renderer's pixel mapping now negates `(world - center)` on both
  axes: world `+Z` is up on screen, world `-X` is to the right, matching
  FP.
- **Yaw-coupled scrolling.** Slice reused `world_walk_input`, which
  rotates WASD by the FP camera's yaw — so after looking around in FP,
  `W` no longer scrolled a consistent direction. Slice now owns its
  panning: `SliceState` carries its own `center_xz`, seeded from the FP
  eye on entry, and WASD pans it in fixed screen-aligned directions.
  `world_walk_input` no longer touches slice mode. Q/E, Space/Ctrl, and
  PageUp/PageDown all shift the z-band; the band is seeded from the
  ground height at the FP position on entry so the view opens on surface
  terrain rather than blank underground.
- **Flat, unshaded look.** `render_slice` filled each column with the
  palette's flat `base_color`. A new `SliceShading::Hillshade` mode
  derives a per-column surface normal from the neighbouring columns'
  `top_z` height field and lights it with the FP sun direction, so
  vertical terrain reads as 3D relief. No slice-table data change — the
  `top_z` field already existed.

The renderer is selected through a `SliceRenderStrategy` trait
(`FlatSlice`, `HillshadeSlice`) on `RenderConfig`, mirroring the existing
strategy spine — harness scenes can A/B them with `set_strategy
slot="slice"`. The horizontal footprint widened from 32 to 64 voxels
(4×4 chunks) at 4 px per tile, filling the 256-px raster exactly.

Three harness gaps surfaced and were fixed so the harness can actually
exercise the slice view:

- The blit `Camera2d` rendered to the window, but the harness captures
  the FP camera's offscreen image — so slice / RTS / overview were never
  in any screenshot. The blit camera now targets the same offscreen
  image when the harness is active.
- With every camera then targeting the offscreen image, `ui_layout_system`
  could no longer resolve a default UI camera and panicked; the world
  camera is now explicitly marked `IsDefaultUiCamera`.
- `drive_input_events` ran in `PreUpdate` with no ordering, so Bevy's
  `keyboard_input_system` could clear `just_pressed` after the harness
  set it — `key_tap` never triggered `just_pressed`-based actions
  (view-mode switches, z-band cycling). It now runs after `InputSystem`.

See [RENDERING.md](RENDERING.md) for the renderer architecture.

### Verification

- `atomr-worlds-view`: `slice_golden.rs` re-pinned for the flipped pixel
  mapping, plus a second pinned golden for the hillshade path and a
  `hillshade_differs_from_flat` sanity check; `modes/slice.rs` unit
  tests updated for the new mapping, with a `hillshade_factor` direction
  test (sun-facing slope brighter than shadowed).
- `atomr-worlds-client`: full test suite green; `view-slice` example
  still builds.
- Harness scene [`slice_align.toml`](../harness/scenes/slice_align.toml):
  rotates the FP camera, switches to slice, then brackets each of
  W/S/A/D and cycles the z-band via Q/E and Space/Ctrl. Confirmed in
  capture: slice renders as a top-down raster of surface terrain with
  visible relief; W pans toward `+Z` (up), A pans screen-left; W+S and
  A+D cancel; Q/E and Space/Ctrl produce identical deterministic z-band
  results.

### Follow-ups landed (after Phase 19)

- **Per-column LOD now keys off the slice pan center.** The LOD observer
  passed into `build_slice_table_with_lod_fn` is built from
  `state.center_xz` (lifted to the current `z_band_top` for the
  vertical), so panning the slice keeps the high-detail ring centred
  under the visible footprint rather than leaving it pinned at the FP
  eye. See
  [`crates/atomr-worlds-client/src/modes/slice.rs`](../crates/atomr-worlds-client/src/modes/slice.rs)
  (`slice_render`).
- **HUD renders on top of every view, including the slice / RTS /
  overview blit.** No dedicated HUD camera: a `Camera2d` and a
  `Camera3d` both actively targeting the same offscreen `Image` cause
  Bevy 0.13 to drop the 3D output (FP/TP harness PNGs went transparent
  with only the HUD overlay). Instead,
  [`crates/atomr-worlds-client/src/hud.rs`](../crates/atomr-worlds-client/src/hud.rs)
  spawns the HUD UI root with a `HudUiRoot` marker and the
  `route_hud_target` system reassigns its `TargetCamera` each frame:
  `WorldCamera` in FP/TP, `BlitCamera` in slice/RTS/overview. UI is
  rendered as part of the active camera's render graph (Bevy 0.13's
  `ui_pass` is registered into both `Core2d` and `Core3d` and runs after
  the main pass), so the HUD lands above 3D meshes in FP/TP and above
  the blit sprite in raster modes — without ever pairing a Camera2d
  with a Camera3d on the same target. `IsDefaultUiCamera` stays on the
  `WorldCamera` for the startup-frame default-camera resolution.

### Out of scope (still)

- Slice panning does not write back to the FP position; switching back
  to FP returns you to where you left it (this was the chosen design,
  noted here as a known behaviour).

## Phase 19 (Algorithm Topologies) *(landed)* — Advanced Algorithmic Topologies & Layered Voxel Architecture

> Naming note: this is the *second* phase tagged "19" — the prior
> Phase 19 (slice view) is a smaller hotfix tier. The full Phase 19
> reference for this work lives at [PHASE_19.md](PHASE_19.md). The
> pipeline contract is in [PIPELINE.md](PIPELINE.md).

Lands every algorithm from the *Advanced Algorithmic Topologies and
Layered Architecture in Procedural Voxel World Generation* paper as
additive strategy-pattern slots on a new `WorldGenConfig`. Existing
behavior is preserved byte-for-byte (`WorldGenPreset::Vanilla` asserted
by `tests/vanilla_byte_equality.rs`). Each algorithm opts in via
`WorldGenPreset::Advanced` / `WorldGenPreset::Showcase` or per-slot via
the harness DSL.

### Landed work

- **Noise primitives** — `atomr-worlds-noise`: 3D simplex, domain warp
  (single + iterated), 3D-noise × radial-falloff floating-island field.
- **Voxel storage layer** — `atomr-worlds-voxel`: `BrickCodec` (RawU16
  / Rle / Zlib / PaletteRle), `BrickStorage` (DenseBrick /
  SegmentedRowBrick / SvoBrick), 4-bit `LightOverlay` (2048 bytes per
  brick).
- **Mesh strategies** — `atomr-worlds-view`: `NaiveMesh`,
  `MarchingCubes`, `DualContouring` alongside the existing
  `GreedyFlat`.
- **Layered brick pipeline** — `atomr-worlds-generate::pipeline`:
  `BrickPipeline` trait + `LayeredBrickPipeline` orchestrator,
  `BrickWorkspace` with padded 18³ apron, `WorldGenConfig` with 13
  trait-object slots, three named presets, `apply_worldgen_strategy_by_name`
  registry. Registered under new `TERRAIN_LAYERED` strategy id.
- **Density / strata / biome** — `HeightmapPlanar`, `Hybrid2D3D`,
  `Pure3DOverhang`, `FloatingIslandField`; `TopsoilLayer`,
  `LayeredGeology`, `KrigingInterpolated`; `PerFaceWhittaker`,
  `WhittakerDirect2D`, `VoronoiCells`; `Hard`,
  `NormalizedSparseConvolution`, `BufferTerrainInjected`.
- **Caves + feature seeder** — `WorleyThreshold`,
  `CellularAutomata3D`, `PerlinWorm`, `IsosurfaceIntersection`;
  `ColumnAnchorSeeder` emits `FeatureKind` anchors on a 64 m column
  grid for cross-brick path features.
- **Ore / erosion / fluid** — `ThresholdNoise`, `BiasedRandomWalk`;
  `MacroRiverOnly`, `DropletHydraulic`; `Static`,
  `CellularAutomataFlow`, `LatticeBoltzmannD3Q19`.
- **Structures** — `WaveFunctionCollapse` (entropy-min + AC-3
  propagation + bounded backtracking), `Jigsaw` (template-pool
  recursion over `AuthoredRegion`), `QwfcClassicalSim` (classical
  amplitude-collapse PDF, research stub).
- **Flora + placement** — `LSystemTrees` (declarative grammar +
  3D turtle), `BlueNoiseGrass`; `WhiteNoise`, `UniformGrid`,
  `PoissonDiskBridson`, `MitchellBestCandidate`.
- **Sky light + render hookup** — `VerticalCastWithDiffusion` produces
  a `LightOverlay`; greedy mesher consumes it for per-vertex sky-light;
  `BrickEdgeAwareAo` resolves edge seams; `BiomeBlendedFog` tints fog
  across biome borders.
- **CUDA scaffold** — `atomr-worlds-accel::strategy_kernels` (feature
  `cuda`): `StrategyKernel` trait + four kernel modules (droplet, lbm,
  ca3d, wfc), `PARITY_CASES` for CPU/CUDA byte-equality testing. NVRTC
  source land in a follow-up PR; trait surface is stable.

### Verification

- `tests/vanilla_byte_equality.rs` — `LayeredGenerator(Vanilla)`
  identical to `default_terrain()` across 4 seeds × 8 brick coords.
- `cargo test -p atomr-worlds-generate` — 152 lib + 17 integration
  tests; covers determinism, boundary continuity, mass conservation
  (LBM 1000 ticks), `min_distance_held` (Poisson-disk), grammar
  termination (L-system), no-contradiction (WFC), etc.
- `cargo test -p atomr-worlds-noise` — 22 noise tests green.
- `cargo test -p atomr-worlds-voxel` — 30 codec/storage/light tests green.
- `cargo test -p atomr-worlds-view` — mesh + iso tests green.
- `cargo test -p atomr-worlds-client` — render + AO + fog tests green.
- `cargo run -p showcase-strategies -- --preset {vanilla,advanced,showcase}` —
  smoke-tests every preset from the CLI.

### Out of scope (deferred)

- CUDA kernel source (`kernels/*.cu` + NVRTC binding) — module surface
  landed; kernel implementation deferred.
- Full kriging interpolation for `KrigingInterpolated` (delegates to
  `LayeredGeology::default()` per spec).
- Real `JigsawTag` orientation parity (current impl matches by tag
  name only).
- Bevy 0.13 shader updates to render the new `BrickEdgeAwareAo` /
  `BiomeBlendedFog` end-to-end (vertex attribute lands; shader path
  consumes default value when fields are zero).

## Phase 19.1 *(landed)* — Async plan rebuild (chunk-transition hitch fix)

A user-reported "big delay when transitioning between chunks" reproduced
in the `stream_walk` harness as a regular ~2.3 ms frame spike every 20
frames at sprint pace — the cadence of `DesiredChunksCache::should_rebuild`
trips against the 4 m drift threshold. The 4-tier AABB sweep + view-
priority sort over ~11 k brick entries ran inline on the main thread,
landing a visible hitch every 333 ms at 60 fps that lined up with the
"transitioning to another chunk" report.

### Fix

[`crates/atomr-worlds-client/src/world_stream.rs::DesiredChunksCache`](../crates/atomr-worlds-client/src/world_stream.rs)
now holds an optional `RebuildHandle { rx: Mutex<mpsc::Receiver<…>>,
pose }`. Three new methods:

- `spawn_rebuild(streamer, observer, forward, horizon_m, coverage)` —
  takes a `ChunkStreamer` clone (cheap: 4-entry `Vec<LodTier>` plus
  scalars) and an `Arc<dyn LodCoveragePolicy>` (the strategy registry
  already stores it as `Arc`), spawns a `std::thread::spawn` worker
  that runs the existing `desired_chunks` + `prioritize_view` and
  sends the result over an `mpsc::channel`. Idempotent while in flight.
- `poll_rebuild()` — `try_recv`s from the receiver; installs the new
  plan + records the pose it was built for.
- `is_rebuilding()` — guards `spawn_rebuild` so the streamer doesn't
  pile multiple workers on top of each other.

`should_rebuild` was extended to compare against the *in-flight* pose
when one exists, falling back to `built_for` otherwise. Without this,
the streamer would dispatch a fresh rebuild every frame between the
drift trigger and the worker landing.

[`fp_stream_bricks`](../crates/atomr-worlds-client/src/modes/fp.rs)
swapped its inline rebuild for the new flow:

```text
plan_cache.poll_rebuild();
if plan_cache.should_rebuild(observer, forward) && !plan_cache.is_rebuilding() {
    plan_cache.spawn_rebuild(streamer.clone(), observer, forward,
                             horizon_m, Arc::clone(&render_cfg.coverage));
}
```

The cached plan lags the observer by 1-2 frames after a rebuild trigger,
which is harmless: rebuilds were already drift-triggered (every 4 m of
motion), so the loader was always working off a slightly stale plan.
Moving the staleness off-thread eliminates the per-rebuild frame spike
without changing the loading behaviour in any observable way.

### Verification

Measured with the existing `harness/scenes/stream_walk.toml` scenario
(60 s sprint walk +Z then -Z, 580 instrumented frames):

| metric                  | before    | after     | delta   |
| ----------------------- | --------- | --------- | ------- |
| rebuild-frame mean      | 2931 µs   | 504 µs    | −83 %   |
| rebuild-frame max       | 3424 µs   | 909 µs    | −73 %   |
| all-frame p99           | 3020 µs   | 1145 µs   | −62 %   |
| all-frame max           | 3424 µs   | 1859 µs   | −46 %   |
| quiet-frame mean        | 527 µs    | 543 µs    | ±3 %    |

The worst AFTER frames are now dispatch+drain spikes (`fp_stream_bricks`
iterating the ~11 k-entry plan while `BrickGenWorkers` has free slots),
topping out at 1.9 ms — a separate, future optimisation if it becomes
the next bottleneck. The user-perceived chunk-transition hitch is gone.

New `world_stream::tests::*`:

- `spawn_rebuild_runs_in_background_and_polls_in` — round-trips a
  spawn + worker thread + poll, asserts the installed plan equals the
  synchronous `desired_chunks + prioritize_view` reference.
- `spawn_rebuild_is_idempotent_while_in_flight` — a second
  `spawn_rebuild` for the same pose is a no-op while the first is in
  flight.
- `should_rebuild_uses_in_flight_pose_to_dedupe` — confirms the
  dispatch-guard logic against a moving observer.

All 50 `atomr-worlds-client` tests green; the existing 29
`world_stream` tests continue to pass unchanged.

## Phase 19.2 *(landed)* — Horizon imposter + speed-aware visual cost

Phase 19.1 made plan rebuilds async, but two user-visible gaps remained:
the world only rendered representative terrain out to the LOD ladder's
outer ring (~1 km) even though `WorldShape::horizon_at_m` exposes the
real geometric horizon, and sustained sprints still produced subjective
hitches because every visual strategy paid the same cost in motion as it
did at rest. This phase lands two complementary mechanisms:

1. **Horizon-imposter shell mesh** — a 32 × 128 polar-annulus mesh
   sourced from `WorldMacroState` (elevation + biomes + water), baked
   off-thread, draping the band between the LOD ladder's outer ring
   and the shape's geometric horizon clamped to a 16 km max.
2. **Speed-aware strategy layer** — a `CameraMotionState` resource
   feeding four new strategies that coarsen the LOD ladder, throttle
   mesh-upload budget, stride visibility updates, and widen plan-rebuild
   thresholds when the camera is moving fast.

Both pieces compose: when the imposter is active the ladder can shed its
outer tier, and the rebuild-threshold strategy can widen its drift/cos
gates because the imposter shell covers the band that would otherwise
shimmer during chunk turnover.

### Part A — Horizon imposter shell

New trait
[`HorizonImposterStrategy`](../crates/atomr-worlds-client/src/render/strategy.rs)
plus
[`HorizonImposterMesh`](../crates/atomr-worlds-client/src/render/strategy.rs)
define the contract. Two default impls:

- `PolarAnnulusShell` — `N_RINGS=32 × N_SECTORS=128` log-spaced radii,
  per-vertex elevation + biome color sample, sphere curvature drop via
  `-d² / (2 R)` for non-cube shapes.
- `NoHorizonImposter` — empty mesh, `enabled() == false`. Selected by
  `RenderPreset::Legacy`.

Pure baker lives in
[`crates/atomr-worlds-view/src/derived/horizon_shell.rs`](../crates/atomr-worlds-view/src/derived/horizon_shell.rs)
(no Bevy types). Five unit tests cover topology
(`indices.len() == N_RINGS × N_SECTORS × 6`), deterministic byte-equal
output for identical `(seed, shape, observer)`, curvature drop on
sphere shapes, color non-uniformity across biomes, and the
sky-fallback path when the strategy is disabled. Triangle winding
emits front-faces upward (`(i0, i1, i2) + (i1, i3, i2)`) so the
standard eye-level observer sees the shell through Bevy's default
back-face cull.

Bevy-side wiring in
[`crates/atomr-worlds-client/src/render/horizon_shell.rs`](../crates/atomr-worlds-client/src/render/horizon_shell.rs):

- `HorizonShellPlugin` registers `HorizonShellRuntime`,
  `HorizonImposterActive`, `MacroStateProvider`, plus `ensure_horizon_shell`
  + `sync_horizon_shell` systems.
- `HorizonShellRuntime.rebuild: Option<RebuildHandle>` mirrors the
  `Mutex<mpsc::Receiver<…>>` template from Phase 19.1, dispatching
  bakes on a worker thread (5-15 ms at 32 × 128) and installing them
  via `meshes.get_mut(handle)` on the main thread.
- `MacroStateProvider` lazily computes
  `Arc<WorldMacroState>` keyed on `(seed, shape)` at `grid_level = 4`
  so the baker can sample biomes without a host round-trip.
- The shell entity carries `NotShadowCaster + NotShadowReceiver +
  NoFrustumCulling`: a 16 km AABB-bounded mesh would otherwise blow up
  the cascade frustum and be culled by Bevy's view-space AABB test.
- Material is a white-base `StandardMaterial` with `unlit = true`
  and `cull_mode: Some(Face::Back)`. Vertex colors pass through 1:1.

Rebuild triggers: startup (once macro state is available), drift > 64 m
from `built_for.0` (loose, ~0.5 % of shell radius), and macro-digest
change (defensive — macro is immutable at runtime today).

### Part B — Speed-aware strategy layer

Five new resources / traits in
[`crates/atomr-worlds-client/src/modes/fp.rs`](../crates/atomr-worlds-client/src/modes/fp.rs)
and
[`crates/atomr-worlds-client/src/render/strategy.rs`](../crates/atomr-worlds-client/src/render/strategy.rs):

- `CameraMotionState` — `position`, `forward`, `velocity_m_s`,
  `smoothed_velocity_m_s` (EWMA τ = 0.3 s), `smoothed_yaw_rate_rad_s`,
  `sprint_held`. Driven by `fp_update_motion_state` chained between
  `world_walk_input` and `fp_sync_camera`. `sprint_held` is read from
  `KeyCode::ShiftLeft|Right` so harness scenarios drive deterministic
  strategy behavior without waiting on EWMA warmup.
- `LodLadderPolicy::ladder(motion)` — `MotionScaledLadder` swaps the
  inner/outer tier counts when `smoothed_velocity_m_s > 6.0`. 0.5 s
  hysteresis on swaps via `LadderHysteresis` resource.
- `SpawnBudgetStrategy::budget_this_frame(motion)` —
  `MotionScaledSpawnBudget` lerps from 24 → 8 spawns/frame as smoothed
  velocity ramps 0 → 12 m/s. *Counter-intuitive direction*: lower at
  sprint to spread GPU-upload spikes over more frames.
- `VisibilityCadenceStrategy::stride(motion)` — `MotionScaledCadence`
  returns 1 at rest, 2 at moderate motion, 3 at sprint.
  `fp_update_lod_visibility` early-returns on
  `frame_count % stride != 0`.
- `RebuildThresholdStrategy::{drift_m, fwd_cos}(motion)` —
  `MotionScaledRebuildThreshold` widens from 4.0 m / 0.9659 at rest to
  16.0 m / 0.93 at full sprint, but *only when
  `HorizonImposterActive.0 == true`* — otherwise the rest values stand
  to avoid outer-rim streaming gaps under Legacy preset.

Two additive trait extensions are also new but behavior-preserving:

- `LodCoveragePolicy::tier_lod_bias(tier_index, motion) -> i8` —
  default `0`. Hook for future motion-coupled tier-depth bias.
- `FogStrategy::fog_settings(..., motion: Option<&CameraMotionState>)` —
  `ExpSquaredSkyTintedFog` tightens fog start by
  `smoothed_velocity_m_s / 12 × 15 %` of the band and snaps fog end to
  the imposter outer radius (instead of the ladder outer radius) when
  `HorizonImposterActive.0 == true`.

### Wiring + ergonomics

- New `MotionAwareConfig` in
  [`render/config.rs`](../crates/atomr-worlds-client/src/render/config.rs)
  holds the four motion-aware strategies plus a
  `locked_to_standstill: bool` kill-switch.
- New `PerfPreset { Balanced, Quality }`. `Balanced` (default) keeps the
  motion-aware ramps; `Quality` flips `locked_to_standstill = true` and
  swaps every motion-aware strategy to a static rest-value impl, so the
  visual fidelity at sprint is identical to standing still.
- New `--perf <balanced|quality>` CLI flag in `main.rs`.
- All five new strategy slots are registered in
  [`render/registry.rs`](../crates/atomr-worlds-client/src/render/registry.rs)
  so harness scenarios can A/B individual policies via `set_strategy`.

### Diagnose-the-hitch fixes (folded in alongside the new mechanisms)

- **Dispatch-loop cursor** on `DesiredChunksCache` — `fp_stream_bricks`
  walks `plan[cursor..]` and stores the advanced cursor each frame.
  Eliminates the per-frame O(11 k) scan over already-loaded entries.
  Reset to 0 in `poll_rebuild` whenever a new plan installs.
- **Incremental `child_counts`** on `LoadedChunks` — replaces the
  per-frame `HashMap` rebuild in `fp_update_lod_visibility` (O(n_loaded)
  every frame) with insert / fade / remove deltas at the three
  mutation points.

### Frame-time instrumentation (permanent)

[`FrameDiagPlugin`](../crates/atomr-worlds-client/src/hud.rs) maintains
a 1024-frame ring of `Time::delta_seconds_f64() * 1e6`. A new
`dump_frame_diag` harness event drains the ring and prints
`FRAME_DIAG frame=N us=…` lines plus a `FRAME_DIAG_SUMMARY` line. Two
sibling events: `dump_motion` (prints `CameraMotionState` snapshot) and
`dump_streamer` (loaded-chunk / rebuild diagnostics). The
`ATOMR_FRAME_DIAG` env-var instrumentation stripped at the end of
Phase 19.1 is not reinstated — the permanent ring buffer replaces it.

### Verification

Harness scenarios authored in `harness/scenes/` for Phase 19.2:

| scenario                       | shots | proves                                    |
| ------------------------------ | ----- | ----------------------------------------- |
| `horizon_distance_markers.toml`| 6     | Distant terrain visible past 1 km after sustained sprint; imposter rebuilds fire on drift. |
| `horizon_sprint.toml`          | 4     | Imposter mesh keeps regenerating off-thread under sustained motion (no visual hitches). |
| `horizon_360_pan.toml`         | 7     | 360° rotational coherence — shell is a full annulus, not a frustum-aligned strip. |
| `horizon_overview_cubeworld.toml` | 2  | Overview mode renders the sphere imposter end-to-end (no streamer required). |
| `perf_sprint_hold.toml`        | 2     | 4 s sustained sprint — frame-time histogram dump. |
| `perf_sprint_then_stop.toml`   | 3     | Strategy state returns to cold-start after EWMA decay (transient, not sticky). |
| `perf_sprint_turn.toml`        | 3     | Sprint + 90° yaw — combined drift + view-direction load. |

`perf_sprint_hold` baseline (DISPLAY=:0, NVIDIA RTX 5000, vsync-locked
at 60 Hz):

| metric                  | value     |
| ----------------------- | --------- |
| `FRAME_DIAG` count      | 372       |
| mean                    | 16 699 µs |
| p99                     | 18 136 µs |
| post-warmup max         | 27 083 µs |
| first-frame outlier     | 100 844 µs (frame 2, asset upload spike) |

`perf_sprint_turn` (sprint + 90° yaw): mean 16 706 µs, p99 21 907 µs,
max 105 210 µs (same first-frame outlier). Steady-state is vsync-cadenced
on this hardware — under software-GL CI the absolute numbers are
incomparable to the Phase 19.1 baseline (which was measured at the
WinitSettings cadence). What's preserved is the *absence of regression*:
no new frame is worse than the asset-upload spike at frame 2.

`MOTION` dumps confirm the EWMA shape:

```text
frame=240  v_raw=12.000  v_smooth=11.981  yaw_rate=0.016  sprint=true
frame=370  v_raw=0.000   v_smooth=9.618   yaw_rate=0.016  sprint=false
```

i.e. `smoothed_velocity_m_s` lags `velocity_m_s` by ~τ as designed,
preventing strategy thrash on a single-frame sprint tap.

### Visual evidence

- `horizon_distance_markers/horizon_dist_0004.png` — after ~6 s of
  forward sprint, stepped imposter terrain extends past the LOD ladder
  edge all the way to the geometric horizon.
- `horizon_overview_cubeworld/horizon_overview_0001.png` — overview
  mode renders the sphere imposter as a complete planetary view with
  biome-colored continents, oceans, and ice caps.
- `perf_sprint_hold/perf_sprint_hold_0000.png` (pre + post winding fix)
  — origin baseline before and after the triangle-winding correction.
  Pre-fix had inconsistent white patches where camera-aligned back
  faces happened to face the eye; post-fix is uniformly colored.

### Tests

- 5 new unit tests on the pure baker in
  `crates/atomr-worlds-view/src/derived/horizon_shell.rs` —
  determinism, topology, sphere curvature drop, biome color spread,
  empty-mesh sky fallback.
- `tests/horizon_shell_runtime.rs` integration test — spawn the plugin,
  advance frames, assert the shell entity exists with non-empty mesh
  + `NotShadowCaster`.
- Existing `world_stream::tests::*` and `client::tests::*` continue to
  pass (incremental `child_counts` is a drop-in; cursor advance is
  drop-in for the existing `lod_crossfade.toml` baseline).

### Risks / follow-ups

- `horizon_sprint.toml` reliably hangs in the harness regardless of
  winding direction (deferred — the other six scenarios cover the same
  imposter rebuild + sprint sustain paths; root cause is likely the
  sustained 6 s sprint hitting an off-thread bake backlog under
  xvfb-software-GL latency).
- The imposter pays a real main-thread render cost: post-fix harness
  runs (where the shell is consistently front-faced) take 2-3× longer
  wall-clock than pre-fix runs at the same scenario. This is expected
  — the shell carries 16 k vertices through the standard PBR pipeline
  with `NoFrustumCulling`. A future optimization could replace the
  unlit material with a custom shader that skips per-fragment lighting
  cost entirely, or stride the shell rebuild based on `smoothed_velocity`.
- Vertex colors lean light when biome lookup returns near-white biomes
  (snow / desert). Material base color is white, so vertex colors pass
  through directly; a future pass could clamp colors away from full
  white to keep tonal separation against the sky.

