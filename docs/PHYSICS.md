# Voxel physics — foundations

The physics subsystem implements **Recommendation 2** of the *Advanced Voxel
Architectures* plan (`~/.claude/plans/take-a-look-at-groovy-sedgewick.md`):
destructible structures that fracture into independently-simulated debris. This
document covers the **Phase-1 foundations** (the pure, deterministic,
engine-agnostic core) and the **Phase-A Bevy/rapier integration** that builds on
them — static voxel colliders and carve→flood-fill→falling debris, all
client-side (see "Phase A — client integration" below, and PHASES.md
"Phase 20.3").

## Why a separate, dependency-free crate

`atomr-worlds-physics` carries **no** Bevy, rapier, or async-runtime
dependency. It is pure logic over voxel data:

- structural connectivity (flood-fill) → which chunks have detached;
- mass / center-of-mass / inertia tensor from per-voxel material density;
- a `DebrisBody` that bundles a detached island's local voxel grid with its
  rigid-body state.

The solver itself (contact resolution, the frame tick, collider generation,
debris entity lifecycle) lives in the client crate and reuses **rapier3d**'s
TGS-Soft solver — see the plan's Rec 2. Keeping the math here, free of engine
types, means it is trivially unit-testable, reusable by both the client (local
destruction) and the host (authoritative multiplayer fracture), and immune to
the Bevy version churn the rendering side faces.

## The determinism boundary (non-negotiable)

The engine has a hard contract: `GetBrick` output is byte-identical across
runs, platforms, CPU-vs-GPU, and Rust-vs-Python. Float physics is **not**
cross-platform reproducible, so physics is a **client-side, non-deterministic,
ephemeral** subsystem:

- Physics only ever *reads* voxels and *derives* quantities; it never mutates
  the brick cache directly.
- Detaching an island's voxels happens through a **journaled write on the world
  actor** (a `FractureCommand` sequence), not by poking the cache — so the
  canonical world stays seed-derived + overlay only, and the determinism gates
  (golden screenshots, CUDA parity, Python parity) remain valid.
- The *fracture decision* is made reproducible by keeping it integer-only:
  connectivity is a discrete flood-fill, and the trigger force crosses the wire
  as fixed-point milli-newtons ([`Force`]), not `f32`. Two clients re-applying
  the same `FractureApplied` command list reach identical geometry. The
  *debris motion* that follows is float and diverges, so it is synced as
  interpolated snapshots ([`DebrisStateDelta`]) rather than replayed.

## What landed (Phase 1)

### Material physics palette — `atomr-worlds-core::material_physics`

`MaterialPhysicsProps { density_kg_m3, friction, restitution, yield_strength_pa }`
indexed by the same `u16` material id as the render palette, so material id `1`
is "stone" for both its look and its mass. `default_palette()` ships plausible
values for the 11 stock materials and is a pure function (identical output every
call). Densities follow physical sense (stone > wood > snow; ice < water).
`yield_strength_pa` feeds the fracture-yield check; the rest feed the solver and
the inertia computation.

### `atomr-worlds-physics` crate

- **`flood_fill`** — `connected_components(dims, is_solid, is_anchor)` labels the
  6-connected components of a voxel region (explicit stack, fixed visit order →
  deterministic) and marks which reach an anchor. `unanchored_islands()` returns
  the floating chunks to spawn as debris. Diagonal contact does **not** connect
  (true 6-connectivity), matching the collision model.
- **`inertia`** — `mass_properties(samples, min_principal)` computes mass, the
  mass-weighted center of mass, and the inertia tensor about it (point-mass form
  `I += mᵢ(|rᵢ|²·1 − rᵢ⊗rᵢ)`), then a **regularized** inverse so a flat or
  one-voxel-thick body stays invertible instead of producing infinite angular
  acceleration.
- **`debris`** — `DebrisBody::from_voxels(...)` copies an island into a local
  dense grid, computes its `MassProperties` from the physics palette, and seeds
  the rigid-body state (pose + linear/angular velocity). Ephemeral; never flows
  back into `GetBrick`.
- **`math`** — the small `f64` `Mat3` (with a regularization-friendly inverse)
  and `dot`/`cross`/`scale` helpers the inertia solver needs, so the crate
  avoids a `glam`/`nalgebra` dependency.

### Fracture protocol types — `atomr-worlds-proto::fracture`

`FractureCommand` (`SetVoxel` / `SpawnDebris` / `DisconnectJoint`),
`FractureRequest`, `FractureApplied` (with a journal `seq_range` for
deterministic late-join replay), `DebrisStateDelta`, `WriteRejected`, and the
fixed-point `Force`. These are **defined and serde-tested but not yet wired**
into the `WorldRequest`/`WorldEvent` enums — that wiring + the actor-side
handling lands with the Rec 2 / Rec 4 phases. Appending them as enum variants
later is bincode-safe; adding fields to existing persisted structs is not (see
the plan's schema-evolution note).

## Phase A — client integration (landed)

The first engine-integration slice lives in **`atomr-worlds-client`** behind a
`physics` feature (default on); `bevy_rapier3d 0.34` (TGS-Soft solver) is confined
to that crate so the determinism-tested crates stay rapier-free. It is **purely
additive and client-side** — the canonical voxel changes still go through the
host's journaled `WriteVoxel`/`WriteRegion`, and debris never flows into
`GetBrick`.

- **Greedy box-merge** (`atomr-worlds-physics::box_merge`) — the one new piece of
  engine-agnostic core: `greedy_boxes(dims, is_solid)` coalesces a brick's solid
  voxels into a small set of axis-aligned boxes (a fully-solid brick → one box).
  Pure and deterministic, mirroring `flood_fill`'s closure API. It feeds both the
  collider and the debris render mesh.
- **Static terrain colliders** — `attach_brick_colliders` turns a LOD-0 brick's
  resident `Arc<Brick>` into a rapier compound collider (one cuboid per merged
  box) and attaches it (`RigidBody::Fixed`) to the brick entity. Leaf-LOD only;
  the collider strategy is pluggable (`ColliderStrategy`: `GreedyBoxCompound`
  default, `PerVoxelCompound` for A/B) and mirrors the render `RenderConfig` spine.
- **Fracture → debris (off-thread, Rec 3)** — the carve pipeline listens for the
  editor's `VoxelEditEvent` and is split across two systems so a big carve never
  stalls the frame (PHASES.md "Phase 20.5"). `dispatch_fracture_checks` snapshots
  the resident `Arc<Brick>`s around the carve and runs the analysis on a worker:
  `atomr-worlds-physics::analyze_region` flood-fills (anchor = solid on the region
  shell except its top face) and, for each unanchored island, bakes the material
  grid, the greedy boxes, and the `DebrisBody` mass. `apply_fracture_results`
  drains finished analyses, spawns the `RigidBody::Dynamic` debris (rendered with
  the per-material `MaterialPool`), removes the island's canonical voxels through
  the host, and dispatches the touched-brick refresh (`fetch_and_build`) through
  the async streaming pool — swapped in flicker-free (make-before-break). Debris
  is reaped on sleep / kill-plane / lifetime cap. The render grid is 1 m/voxel, so
  debris uses `voxel_size_m = 1.0`.

### Still deferred

Tier-1 raymarched debris and rounded/per-voxel narrow-phase (v2); and the Rec 4
multiplayer wiring (`FractureRequest`/`FractureApplied` into the actor + the
HLC-timestamped LWW overlay) — single-client debris needs none of the fracture
protocol, since it rides the already-journaled carve. (Phase B/C added the
collidable FP character controller + true crouch; Phase 20.5 moved the fracture
analysis + brick refresh off the render thread.)

## Tests

`cargo test -p atomr-worlds-core -p atomr-worlds-physics -p atomr-worlds-proto`
covers: palette lookups + density ordering + determinism; flood-fill island
detection (anchored vs floating, 6-connectivity, deterministic labels); the
`box_merge` greedy decomposition (exact-cover + disjointness + determinism); mass
conservation, centroid centering, symmetric-cube inertia, and thin-body inverse
stability; `Mat3` inverse correctness; `DebrisBody` mass-from-palette; and serde
round-trips for the fracture types. `cargo test -p atomr-worlds-client` covers the
client integration: collider generation, `spawn_island` (a dynamic body with a
merged child mesh), `attach_brick_colliders` (LOD-0 gets a `Fixed` collider, coarse
LODs are skipped), and the fracture region math. The rapier dependency is proven
isolated via `cargo build -p atomr-worlds-client --no-default-features` and
`cargo tree -p atomr-worlds-host -i bevy_rapier3d`.
