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

The Phase-9 `transvoxel_seam` stub is `#[deprecated]`-aliased to
`boundary_skirt` for legacy callers. Tests in
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

### Phase 14a *(landing)* — 1st-person walk

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

### Phase 14b *(landing)* — 3rd-person chase

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

### Phase 14c *(landing)* — Dwarf-Fortress horizontal slice

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

### Phase 14d *(landing)* — RTS oblique-orthographic

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

### Phase 14e *(landing)* — Regional / world overview

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
byte-identical. The interactive shell — winit + input + chosen mode
dispatch — stays an external concern downstream of this repo.
