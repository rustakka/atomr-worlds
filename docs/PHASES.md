# Phases roadmap

Detailed plan for phases 1–6 plus the Python interface. Phase 0 is the
substrate; everything below is built on top of it.

This document is **descriptive of the design**, not a per-commit log. As phases
land, [IMPLEMENTATION.md](IMPLEMENTATION.md) gets updated with concrete
file/line pointers; this document stays focused on the intended end-state.

## Phase 1 — Generators + `LocalHost`

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

## Phase 2 — Renderer integration

**Phase 2 scaffold (this session)**: `view-png` example that pulls a brick
from a `LocalHost`, projects a YZ slice through it, and writes a PNG. No
wgpu/Bevy dependency yet; trivially headless; useful as a CI smoke test.

**Phase 2 full (next session)**: `atomr-worlds-view` crate bridging
`atomr-view`'s scene API. Greedy meshing of bricks; camera driven by
`MetricScale::lod_for_screen`; deterministic screenshot test for a known
seed.

### Dependencies for the full phase

- An EGL/Wayland/X display, or `wgpu` headless surface, in CI.
- atomr-view-backends's wgpu backend to be stable.

## Phase 3 — Persistence

**Phase 3 scaffold (this session)**: `WorldActor` becomes able to wrap an
in-memory `WriteJournal` (a `Vec<VoxelDelta>`) plus a `BrickCache`. Writes
go through the journal so a future cluster recovery can replay. The journal
trait surface matches the shape of `atomr_persistence::Journal` so a real
binding is mechanical.

**Phase 3 full (next session)**: bind `atomr_persistence::PersistentActor`
to `WorldActor`. Backends: `atomr-persistence`'s in-memory journal +
snapshot store first; `atomr-persistence-sql` next. Recovery on actor start;
periodic `save_snapshot` every N writes.

### Dependencies for the full phase

- A running SQL or Redis instance for production-backend integration tests.
- A protobuf or bincode-based event encoding for `Journal::write_messages`.

## Phase 4 — Streaming subscriptions

**This session — full**: `Subscribe` envelope handling, per-subscription
bounded `mpsc` channels, AABB → brick set reduction, `VoxelDelta` emission
on writes. `WorldActor` keeps a `HashMap<u64, SubscriberState>` keyed by
`sub_id`; backpressure policy is "drop oldest delta" so a slow consumer
never blocks the write path. `StreamEnd` on unsubscribe or actor stop.

### Gates

- Subscribe, write voxel, receive matching `VoxelDelta`.
- Subscriber's receiver dropped → `WorldActor`'s send fails on next emit →
  subscription is reaped.
- Stress: 1000 writes/sec to one world, 10 subscribers each with 64-deep
  channel, none of the subscribers backpressures the writer.

## Phase 5 — GPU acceleration

**Phase 5 scaffold (this session)**: `atomr-worlds-accel` crate exporting
a `BrickGenerator` trait (one method: fill a `&mut [Voxel; 4096]` for a
given `(world_seed, brick_coord)`). One CPU impl that wraps the
phase-1 noise generators; the trait is GPU-ready because the kernel-friendly
signature is "given (seed, coord) → fill buffer".

**Phase 5 full (next session)**: a CUDA impl via `atomr-accel`. Same trait,
different kernel. Bench: criterion comparing CPU vs GPU on a representative
mix of worlds; gate on byte-identical output (determinism is non-negotiable).

### Dependencies for the full phase

- `nvcc` toolchain on the build host.
- `atomr-accel` and its CUDA backend's API stable enough to consume.

## Phase 6 — Python interface

**This session — full**: a `pyo3 + maturin` extension module exposing:

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
Phase 0 (substrate, done)
    │
    ├─► Phase 1 (generators + LocalHost) ─────► Phase 2 scaffold (PNG slice)
    │                  │
    │                  ├──► Phase 3 scaffold (in-memory journal)
    │                  ├──► Phase 4 (streaming subscriptions)
    │                  └──► Phase 5 scaffold (BrickGenerator trait, CPU impl)
    │
    └─► Phase 6 (Python bindings — depends on phase 1 for the host surface)
```

Phase 1 is the keystone. Phases 4 and 6 attach cleanly on top. Phases 2, 3,
5 each have a "scaffold" deliverable here (gets the seams in place) and a
"full" deliverable that needs external infrastructure (display server,
database, CUDA toolchain).

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
