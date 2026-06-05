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
| **Rec 1** | SVDAG + GPU raymarcher | 🟡 selectable render path landed (proxy-cube fragment raymarcher + off-thread build + cross-brick buffer dedup + occupancy-AABB proxy + CPU render golden); default-flip is data-gated on a release-build frame-time measurement |
| **Rec 2** | rapier physics + fracture | ⬜ unblocked (foundations + Bevy ready) |
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
  determinism gate). **GPU render path:** `RaymarchDagShading` shading strategy
  (`--shading raymarch`) draws each brick by raymarching its DAG in a fragment
  shader (`voxel_raymarch.wgsl`) with pluggable shading tiers; the DAG is built
  off the main thread (`DagGpuWithDigest` on `BrickReady`), a refcounted
  `DagBufferCache` dedups GPU buffers + materials across identical bricks
  (freed in lockstep with eviction), and the proxy/DDA are clipped to each
  brick's occupancy AABB to cut overdraw. A deterministic CPU render golden
  (`atomr-worlds-view/tests/raymarch_golden.rs`) pins the path; a debug A/B
  showed ~19× lower GPU memory vs meshing. All derived, non-canonical state.
- **Phase 0:** Bevy upgraded to the latest stable (0.18); `cargo test
  --workspace` green.

### Next (now unblocked)

- **Rec 1** — the proxy-cube fragment raymarcher is landed and selectable
  (above). Remaining, in priority order: (1) **run the release-build perf gate**
  (`harness/scenes/perf_raymarch_ab.toml`, `--shading mesh` vs `raymarch`) and
  flip the default to `RaymarchDagShading` if it clears the bar — the one-line
  change is `RenderConfig::default().shading`; (2) if proxy-cube overdraw is the
  bottleneck (the fragment writes `frag_depth`, disabling early-Z), move to a
  **single-pass / render-graph compute node** with a top-level brick acceleration
  structure that composites against the PBR pass via reversed-Z depth — the route
  to truly "pure" raymarching; (3) **client voxel editing**
  (input → `WriteRegion` → `RegionDelta` → per-brick DAG rebuild via the dedup
  cache) to unlock raymarching the editable near zone (SVDAGs are static, so an
  edit rebuilds just that brick's DAG).
- **Rec 2** — `bevy_rapier3d` integration with native voxel colliders
  (leaf-LOD-only), mass injected from `InertiaSolver`, flood-fill fracture →
  `DebrisBody` spawning. Client-side only; never touches the determinism path.
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
