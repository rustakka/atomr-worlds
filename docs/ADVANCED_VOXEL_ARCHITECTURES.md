# Advanced Voxel Architectures — roadmap

The strategic roadmap derived from the research analysis
`docs/Voxel Engine Improvement Analysis.pdf` ("Advanced Voxel Architectures:
Integrating High-Fidelity Rendering, Rigid Body Physics, and Actor-Model
Concurrency for Procedural Worlds"). That paper surveys state-of-the-art voxel
systems (Teardown, Douglas Dwyer's Octo / rigid_pixels / micropool, SVDAG
research, Aokana) and closes with **four strategic recommendations** for
atomr-worlds. This document is the living roadmap for implementing them: the
plan, the decisions, and the progress.

## The four recommendations

1. **SVDAG compression + GPU compute raymarching** — replace mesh re-upload with
   a deduplicated Sparse Voxel DAG rendered by a GPU raymarcher, so destruction
   is instantaneous and far terrain stays resident in VRAM.
2. **TGS physics + hybrid rounded-edge collisions** — a Temporal Gauss-Seidel
   solver with warm-starting, rounded voxel collision primitives, flood-fill
   structural-connectivity fracture into debris rigid bodies, and a per-material
   physics palette (density / friction / restitution / yield).
3. **Lock-free, scope-pinned, low-latency scheduler** — isolate frame-critical
   work (physics, rendering) from background generation.
4. **Actor-CRDT hybrid destruction sync** — deterministic fracture events plus
   continuous interpolated debris state for multiplayer.

## Grounding (where the paper is idealized vs. the real codebase)

The paper is somewhat aspirational; the plan is anchored in the actual code:

| Paper claims / implies | Reality |
| --- | --- |
| "migrate from Rayon to micropool" | The engine does **not** use Rayon (only a transitive `criterion` dev-dep). Concurrency = tokio + `std::thread` workers + Bevy task pools + CUDA. |
| atomr has "native CRDT" | **True** — `../atomr`'s `atomr-distributed-data` ships `LWWMap`/`LwwRegister`/`ORMap`/`OrSet`/`CrdtMerge`/`Replicator`. But `CrdtMerge` is *sealed*, consistency levels are *unwired* (only `Local`), and there is *no HLC* and *no cluster gossip yet*. |
| SVDAG / GPU raymarching exist | Storage is a Brick+Octree hybrid (`SvoBrick` — the SVDAG precursor); rendering is mesh-based (Bevy PBR + headless CPU rasterizer). |
| Physics / destruction exist | Greenfield — no physics crate; only an eye-height `ground_height_m` probe. |

**Determinism is non-negotiable:** `GetBrick` output is byte-identical across
runs / platforms / CPU-vs-GPU / Rust-vs-Python. Float physics diverges across
hardware, so **physics is a client-side, non-deterministic, ephemeral
subsystem** that never flows into `GetBrick` / the `Journal`. Fracture
*decisions* are kept integer/fixed-point so geometry replays identically; debris
*motion* is synced as interpolated snapshots, never replayed.

## Decisions

- **Reuse `rapier3d`** (`bevy_rapier3d`, TGS-Soft solver) for Rec 2 rather than a
  bespoke solver; add a thin voxel adapter + flood-fill fracture. The paper's
  bespoke per-voxel narrow-phase (Bonten Corner/Edge/Face) is a deferred v2.
- **Bevy upgrade as Phase 0** — the GPU raymarcher (Rec 1) and rapier native
  voxel colliders (Rec 2) both need a modern Bevy, so upgrade first.
- **Foundations first** — land the shared substrate (material-physics palette,
  fracture protocol, flood-fill, `atomr-worlds-physics`, HLC) before any single
  recommendation's vertical slice, so Recs 2 & 4 share one foundation.

## Phase plan & status

| Phase | Scope | Status |
| ----- | ----- | ------ |
| **Prereq** | atomr path-dep pin 0.9.2 → 0.10.1 (workspace wouldn't resolve) | ✅ landed (PR #5) |
| **Phase 0** | Bevy 0.13 → 0.18 upgrade (4 majors; 0.15 skipped) | ✅ landed (PRs #6–#10) — see [PHASES.md](PHASES.md) "Phase 0 (Advanced Voxel Architectures)" |
| **Phase 1** | Shared foundations | ✅ landed (PRs #1–#4) — see [PHYSICS.md](PHYSICS.md), PHASES.md "Phase 20/20.1/20.2" |
| **Rec 1** | SVDAG + GPU raymarcher + voxel editing | ✅ finished — GPU DAG raymarcher is now the **default** render path (proxy-cube fragment raymarcher + off-thread build + cross-brick buffer dedup + occupancy-AABB proxy + CPU render golden); first-person **voxel editing** landed (single-voxel + sphere/cube brushes, host-authoritative, live refresh in both paths); mesh path stays via `--shading mesh` / `RenderPreset::Legacy` |
| **Rec 2** | rapier physics + fracture | 🟢 Phase A landed (PHASES.md "Phase 20.3") — `bevy_rapier3d` client integration: static leaf-LOD terrain colliders + carve→flood-fill→falling debris. Char-controller / Tier-1 debris / off-thread fracture deferred |
| **Rec 4** | Actor-CRDT destruction sync | 🟡 `HlcTimestamp` landed; actor/proto/CRDT wiring remains |
| **Rec 3** | physics-island scheduler | ⬜ deferred — use Bevy `ComputeTaskPool`; micropool only if profiling warrants |

### Landed so far

- **Phase 1 foundations** (`atomr-worlds-core`, `atomr-worlds-physics`,
  `atomr-worlds-proto`):
  - `MaterialPhysicsProps` palette (`core::material_physics`) — density /
    friction / restitution / yield, keyed by the render material id.
  - New **`atomr-worlds-physics`** crate (Bevy-free, rapier-free): deterministic
    `flood_fill` structural connectivity, `inertia` (mass / center-of-mass /
    inertia tensor from per-voxel density), `DebrisBody`, minimal `Mat3`.
  - Fracture-event protocol types (`proto::fracture`): `FractureCommand`,
    `FractureRequest`, `FractureApplied`, `DebrisStateDelta`, `WriteRejected`,
    fixed-point `Force` (defined + serde-tested; not yet wired into the actor).
  - `HlcTimestamp` (`core::hlc`) — Hybrid Logical Clock for the Rec 4 LWW overlay
    (fills the missing-HLC gap in `atomr-distributed-data`).
- **Rec 1:** `DagBrick` (`atomr-worlds-voxel::dag`) — a deduplicated Sparse
  Voxel DAG hash-consed from a `Brick` (uniform 16³ brick → 5 nodes);
  `DagBrick::to_gpu()` flat GPU buffers with occupancy/color decoupled, plus
  `gpu_get()` + `ray_dda_first_hit()` (`atomr-worlds-voxel::raymarch`) — the CPU
  point + ray traversals that mirror the WGSL DDA shader (the raymarcher's
  determinism gate). **GPU render path (now the default):** `RaymarchDagShading`
  shading strategy draws each brick by raymarching its DAG in a fragment shader
  (`voxel_raymarch.wgsl`) with pluggable shading tiers; the DAG is built off the
  main thread (`DagGpuWithDigest` on `BrickReady`), a refcounted `DagBufferCache`
  dedups GPU buffers + materials across identical bricks (freed in lockstep with
  eviction), and the proxy/DDA are clipped to each brick's occupancy AABB to cut
  overdraw. A deterministic CPU render golden
  (`atomr-worlds-view/tests/raymarch_golden.rs`) pins the path; a debug A/B
  showed ~19× lower GPU memory vs meshing. The greedy-mesh path stays reachable
  via `--shading mesh` / `RenderPreset::Legacy`. All derived, non-canonical state.
  **Voxel editing:** `world_ray_first_solid` (`atomr-worlds-voxel::world_dda`, a
  pure world-space Amanatides–Woo picker, *not* WGSL-mirrored) +
  `crate::modes::edit` drive first-person carve/place (single-voxel `WriteVoxel`
  and sphere/cube `WriteRegion` brushes). The host remains the sole mutator; the
  client predicts the touched bricks (`InteractionUnit::affected_voxels`) and
  re-fetches authoritative bytes (`fetch_and_build`), swapping them in
  flicker-free (`spawn_edited_brick`) — so edits show in both render paths.
- **Phase 0:** Bevy upgraded to the latest stable (0.18); `cargo test
  --workspace` green.

### Next (now unblocked)

- **Rec 1** — ✅ done: the GPU DAG raymarcher is the default render path and
  first-person voxel editing landed (above). The release-build A/B
  (`harness/scenes/perf_raymarch_ab.toml`, `--shading mesh` vs `raymarch`) is
  recorded as data, not a gate — the flip was committed regardless, keeping mesh
  one flag away. The named follow-up lever, if the recorded p50 regression is
  large: (2) if proxy-cube overdraw is the bottleneck (the fragment writes
  `frag_depth`, disabling early-Z), move to a **single-pass / render-graph compute
  node** with a top-level brick acceleration structure that composites against the
  PBR pass via reversed-Z depth — the route to truly "pure" raymarching. Editing
  currently covers the LOD-0 near ring; coarse-tier edits self-heal on re-stream,
  and a harness-driven edit hook (for automated edit captures) is a small
  follow-up.
- **Rec 2** — 🟢 Phase A landed (`bevy_rapier3d`, client-side only, behind the
  client's `physics` feature; never touches the determinism path): static
  leaf-LOD voxel colliders from resident bricks (greedy box-merge → rapier
  compound) + carve→flood-fill→falling `DebrisBody` debris that lands on the
  terrain, rendered via the existing `MaterialPool`. Pluggable `ColliderStrategy`
  (`--collider greedy|per-voxel`) mirrors the render strategy spine. Deferred to
  later slices: a collidable first-person character controller (camera is still
  free-fly), raymarched/Tier-1 debris + rounded narrow-phase (v2), and off-thread
  flood-fill (the Rec 3 lever) for large brushes.
- **Rec 4** — wire `FractureRequest`/`FractureApplied` into `WorldRequest`/
  `WorldEvent` + the `WorldActor`; promote the write overlay to an
  HLC-timestamped LWW map; deterministic geometry (reliable channel) + debris
  interpolation (unreliable channel) + per-cell CRDT merge.
- **Rec 3** — parallel physics-island solve via Bevy `ComputeTaskPool::scope`.

## Upstream feature requests (to `../atomr` / ecosystem) — only what's needed

| Crate | Request | For |
| ----- | ------- | --- |
| `atomr-distributed-data` | An HLC timestamp primitive; wire `WriteConsistency`/`ReadConsistency` in `Replicator` (only `Local` today) | Rec 4 |
| `atomr-cluster` | Delta-CRDT gossip for `LWWMap` deltas across nodes | Rec 4 |
| `atomr-remote` | An unreliable (UDP) channel alongside TCP, for debris snapshots | Rec 4 (debris) |
| `atomr-cluster-sharding` | Passivation/handoff hooks to checkpoint debris state on shard migration | Rec 4 |
| *(optional)* `../atomr-accel` | The sibling repo is absent — `--features cuda` won't build; only needed for GPU-accelerated DAG building / strategy kernels | none of the 4 recs require it |

## Correctness invariants

- Physics state and `DagBrick` buffers are **derived / ephemeral** and must never
  flow into `GetBrick` or the `Journal`.
- Every fracture/debris voxel mutation routes **through `WorldActor`** as a
  journaled `VoxelWriteEvent(Batch)` — physics never mutates the brick cache.
- Fracture trigger forces are **fixed-point integers** so the fracture decision
  replays byte-identically across machines.
- Persisted structs (`WorldSnapshot`, `VoxelWriteEvent`) need **versioned
  migration** under bincode 2 (no `serde(default)` fallback for added fields);
  appending enum variants is safe.
