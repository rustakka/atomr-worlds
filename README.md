# atomr-worlds

A procedural-universe substrate for [atomr](https://github.com/rustakka/atomr): hierarchical seeded
generation across **Universe → Galaxy → Sector → System → World**, with per-node dimensions, sparse
voxel storage, metric levels of detail, and a hosting model that runs either embedded in-process
(single-player) or sharded across an atomr cluster (multiplayer) — same actor protocol either way.

## Status

**Phases 0–15 landed.** Phase 0 (primitives), Phase 1 (procedural generators + real
`LocalHost` on atomr's actor system), Phase 2 (CPU renderer: greedy meshing + software
rasterizer to PNG), Phase 3 (persistence: `atomr-persistence` Journal/SnapshotStore binding,
in-memory + optional SQL backends, recovery on host restart), Phase 4 (streaming
subscriptions), Phase 5 (GPU acceleration: CUDA backend via `atomr-accel-cuda` NVRTC, gated on
byte-for-byte determinism vs the CPU path), Phase 6 (Python bindings), Phases 7–12 (vehicles +
policy + strategy registry, atmosphere + metric LOD, isosurface meshing, `ClusterHost`, Python
release, persistence + observability hardening), Phase 13 (world shape + horizon streaming +
geologic macro pre-sim + authored-region stipulation + skybox cubemap + composite renderer +
cross-LOD seam fix + transitive skybox), Phase 14 (five world display modes — 1st-person walk,
3rd-person chase, Dwarf-Fortress horizontal slice, RTS oblique strategy, and large-scale
regional overview — each with its own rendering pipeline and derived data structure on top of
the new `Projection` enum, `WorldQuery` trait, `raster2d` blitter, and `ViewCache` foundation),
and Phase 15 (client/server: Bevy-driven interactive client, headless `atomr-worlds-server`
binary, `atomr-remote`-based `RemoteHost`, and wire-up of `ClusterHost`'s cross-node
forwarder) are all implemented and tested end-to-end.

The upstream bridge from `atomr-worlds-view`'s mesh output into `atomr-view`'s scene API is
still blocked on the latter growing 3D primitives / a headless wgpu path; the Phase-15 Bevy
client uses native `bevy_pbr` for 3D and native `bevy_ui` for the HUD as a working
substitute. See [docs/CLIENT_SERVER.md](docs/CLIENT_SERVER.md) for the topology.

See [docs/PHASES.md](docs/PHASES.md) for the roadmap, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the model, and [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for module-by-module specifics.

## Workspace layout

```
atomr-worlds/
├── crates/
│   ├── atomr-worlds-core      ─ coordinates, addressing, seed derivation, LOD
│   ├── atomr-worlds-voxel     ─ Brick (16³), arena Octree, SparseVoxelStore trait
│   ├── atomr-worlds-noise     ─ value/gradient/Worley noise + FBM, seeded
│   ├── atomr-worlds-generate  ─ per-tier Generators; CPU TerrainGenerator
│   ├── atomr-worlds-accel     ─ Accelerator trait, CPU backend, CUDA backend (feature = "cuda")
│   ├── atomr-worlds-persist   ─ WorldPersistence on top of atomr-persistence Journal/SnapshotStore
│   │                            (in-memory by default; SqlJournal/SqlSnapshotStore via `sql`)
│   ├── atomr-worlds-proto     ─ WorldRequest/WorldEvent/Envelope, bincode 2 wire format
│   ├── atomr-worlds-host      ─ WorldHost trait, LocalHost (with optional persistence), ClusterHost shell
│   ├── atomr-worlds-view      ─ greedy meshing, MetricScale-driven camera, software rasterizer → PNG
│   ├── atomr-worlds-remote    ─ RemoteHost (client) + WorldGateway (server) + cluster forwarder over atomr-remote
│   ├── atomr-worlds-server    ─ headless server binary: --mode standalone | cluster
│   ├── atomr-worlds-client    ─ Bevy-driven interactive client; all five Phase-14 view modes
│   ├── atomr-worlds-testkit   ─ proptest strategies, cross-crate verification
│   └── atomr-worlds-py        ─ Python bindings via PyO3 + maturin
├── examples/
│   ├── print-seed-chain       ─ prints derived seeds + metric scales
│   ├── print-brick            ─ ASCII slice of a generated world brick
│   └── view-png               ─ isometric perspective PNG of a 4×4×6 brick slab (headless, no GPU)
└── docs/
    ├── PHASES.md              ─ roadmap for phases 1–6 + Python
    ├── ARCHITECTURE.md
    └── IMPLEMENTATION.md
```

Dependency direction (leaf-first):
`core → voxel → {noise, generate, view, accel} → proto → persist → host`; `testkit` depends on
`core + voxel + proto` (and `host` as a dev-dep). `core`, `voxel`, `view`, `accel` (default
features), and `persist` (default features) have zero atomr dependencies so tools and CLIs can
use the primitives without dragging in the actor runtime. The CUDA backend (`accel/cuda`) and
the host pull in atomr.

## Quick start

The workspace expects atomr (and, for the GPU backend, atomr-accel) to be sibling checkouts:

```
~/source/
├── atomr           # https://github.com/rustakka/atomr
├── atomr-accel     # CUDA / NVRTC compute (only needed for `--features cuda`)
└── atomr-worlds    # this repo
```

Then from the repo root:

```sh
cargo check --workspace
cargo test  --workspace
cargo run   -p print-seed-chain   # seed chain + metric scales
cargo run   -p print-brick        # ASCII YZ-slice of generated terrain
cargo run   -p view-png           # writes view-png-output.png (no display needed)
```

### Run the interactive client

```sh
# in-process server, single binary — needs an X11 display
cargo run -p atomr-worlds-client --release -- --backend local

# headless server (one terminal) + remote client (another)
cargo run -p atomr-worlds-server --release -- --bind 127.0.0.1:7800
cargo run -p atomr-worlds-client --release -- \
    --backend remote \
    --connect 'atomr://atomr-worlds-server@127.0.0.1:7800/user/world-gateway'
```

Controls: `WASD` to move, mouse-look once the cursor is grabbed (`Esc` releases),
`1..=5` to pick a view mode (`fp` / `tp` / `slice` / `rts` / `overview`), `Tab`
cycles. Slice/RTS/overview have per-mode hotkeys — see
[docs/CLIENT_SERVER.md](docs/CLIENT_SERVER.md).

For the Python bindings:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop -m crates/atomr-worlds-py/Cargo.toml
python crates/atomr-worlds-py/python/tests/test_smoke.py
```

For the optional SQL persistence backend (SQLite by default; Postgres / MySQL / MSSQL via
`atomr-persistence-sql`'s sqlx feature flags):

```sh
cargo check -p atomr-worlds-host --features sql
```

For the CUDA accelerator (requires `nvcc` and a CUDA-capable host; the determinism test and
bench are `#[ignore]` so a CUDA-less host still passes `cargo test`):

```sh
cargo test  -p atomr-worlds-accel --features cuda -- --ignored
cargo bench -p atomr-worlds-accel --features cuda --bench cpu_vs_gpu
```

## Verification gates

All gates ship green:

| gate                                                                  | status                       |
| --------------------------------------------------------------------- | ---------------------------- |
| `cargo check --workspace`                                             | clean                        |
| `cargo test --workspace`                                              | all Rust tests pass (Phase-15 added loopback / cluster / smoke tests) |
| `cargo clippy --workspace --all-targets -- -D warnings`               | clean                        |
| `cargo run -p print-seed-chain` / `print-brick` / `view-png`          | all run                      |
| `python crates/atomr-worlds-py/python/tests/test_smoke.py`            | 7 tests pass                 |
| `cargo test -p atomr-worlds-accel --features cuda -- --ignored`       | CPU/GPU bricks byte-identical (CUDA hosts only) |

The test suite covers seed determinism, hash avalanche (≥ 40% bit flip on 1-bit input perturbation),
low-byte distribution uniformity, brick / octree round-trips against a `HashMap` oracle, octree
empty-space-skip probe-bound assertions, `WorldAddr` serde round-trips (bincode + JSON), protocol
envelope round-trips, LOD math, `MessageExtractor` stability + sibling-system co-location,
`LocalHost` request / write / subscribe-snapshot / subscribe-delta / out-of-region filtering,
persistence recovery across host restarts (writes replay; snapshot fires every N writes and the
journal tail still replays on top), greedy meshing + deterministic-screenshot rendering (FNV-1a
hash equal across runs), and (under `--features cuda`) CUDA-vs-CPU brick byte equality.

## What this is, what it isn't

This is the **foundation layer** for a procedural universe. It provides the address space, the
hash-based hierarchy of seeds, the data structures for sparse voxel content at multiple scales,
the wire/host shape downstream code routes through, CPU + CUDA brick generation, a streaming
host with durable write replay, a deterministic CPU renderer, and Python bindings.

It is **not** (yet) a game. The pieces it deliberately leaves out: a renderer-side `atomr-view`
scene bridge (blocked on upstream 3D primitives — the Bevy client uses native `bevy_pbr` in
the meantime), variable-depth hierarchies, cross-dimension portals / passivation rules,
multi-galaxy load-balancing policy, cluster subscription routing (one-shot requests forward
cross-node; subscriptions stay node-local), gossip-based cluster membership, transport TLS,
and a PyPI release. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design principles
and [docs/CLIENT_SERVER.md](docs/CLIENT_SERVER.md) for the Phase-15 topology and known gaps.

## License

Apache-2.0. See [LICENSE](LICENSE).
