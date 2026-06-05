# Voxel physics — foundations

The physics subsystem implements **Recommendation 2** of the *Advanced Voxel
Architectures* plan (`~/.claude/plans/take-a-look-at-groovy-sedgewick.md`):
destructible structures that fracture into independently-simulated debris. This
document covers the **Phase-1 foundations** that have landed — the pure,
deterministic, engine-agnostic core — and how the later Bevy/rapier integration
builds on them.

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

## What's next (not yet implemented)

Per the plan, the remaining Rec 2 work is the engine integration, which is
gated on the **Bevy 0.13 → 0.18 upgrade** (Phase 0): `bevy_rapier3d` collider
generation from bricks (leaf-LOD only), the TGS-Soft solver tick with
voxel-coord warm-start caching, flood-fill-driven `DebrisSpawnEvent`s on a
background thread, and the multiplayer destruction sync (Rec 4) that reuses the
fracture protocol here over `atomr`'s CRDT + actor layers.

## Tests

`cargo test -p atomr-worlds-core -p atomr-worlds-physics -p atomr-worlds-proto`
covers: palette lookups + density ordering + determinism; flood-fill island
detection (anchored vs floating, 6-connectivity, deterministic labels); mass
conservation, centroid centering, symmetric-cube inertia, and thin-body inverse
stability; `Mat3` inverse correctness; `DebrisBody` mass-from-palette; and serde
round-trips for the fracture types.
