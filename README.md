# atomr-worlds

A procedural-universe substrate for [atomr](https://github.com/rustakka/atomr): hierarchical seeded
generation across **Universe → Galaxy → Sector → System → World**, with per-node dimensions, sparse
voxel storage, metric levels of detail, and a hosting model that runs either embedded in-process
(single-player) or sharded across an atomr cluster (multiplayer) — same actor protocol either way.

## Status

**Phases 0 + 1 + 4 + 6 landed.** Phase 0 (primitives), Phase 1 (procedural generators + real
`LocalHost` on atomr's actor system), Phase 4 (streaming subscriptions), and Phase 6 (Python
bindings) are implemented and tested end-to-end. Phases 2 (rendering), 3 (persistence), and 5
(GPU acceleration) ship as scaffolds — trait surfaces + minimal-viable backends ready for the
next session.

See [docs/PHASES.md](docs/PHASES.md) for the full roadmap, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the model, and [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for module-by-module specifics.

## Workspace layout

```
atomr-worlds/
├── crates/
│   ├── atomr-worlds-core      ─ coordinates, addressing, seed derivation, LOD
│   ├── atomr-worlds-voxel     ─ Brick (16³), arena Octree, SparseVoxelStore trait
│   ├── atomr-worlds-noise     ─ value/gradient/Worley noise + FBM, seeded
│   ├── atomr-worlds-generate  ─ per-tier Generators; CPU TerrainGenerator
│   ├── atomr-worlds-accel     ─ Accelerator trait + CPU backend (Phase 5 scaffold)
│   ├── atomr-worlds-persist   ─ WorldJournal trait + in-memory backend (Phase 3 scaffold)
│   ├── atomr-worlds-proto     ─ WorldRequest/WorldEvent/Envelope, bincode 2 wire format
│   ├── atomr-worlds-host      ─ WorldHost trait, real LocalHost, ClusterHost shell
│   ├── atomr-worlds-testkit   ─ proptest strategies, cross-crate verification
│   └── atomr-worlds-py        ─ Python bindings via PyO3 + maturin
├── examples/
│   ├── print-seed-chain       ─ prints derived seeds + metric scales
│   ├── print-brick            ─ ASCII slice of a generated world brick
│   └── view-png               ─ top-down PNG of the surface (headless, no GPU)
└── docs/
    ├── PHASES.md              ─ roadmap for phases 1–6 + Python
    ├── ARCHITECTURE.md
    └── IMPLEMENTATION.md
```

Dependency direction (leaf-first):
`core → voxel → proto → host`; `testkit` depends on `core + voxel + proto` (and `host` as a dev-dep).
`core` has zero atomr dependencies so tools and CLIs can use the primitives without dragging in the
actor runtime.

## Quick start

The workspace expects atomr to be a sibling checkout at `../atomr`:

```
~/source/
├── atomr           # https://github.com/rustakka/atomr
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

For the Python bindings:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop -m crates/atomr-worlds-py/Cargo.toml
python crates/atomr-worlds-py/python/tests/test_smoke.py
```

## Verification gates

All gates ship green:

| gate                                                 | status              |
| ---------------------------------------------------- | ------------------- |
| `cargo check --workspace`                            | clean               |
| `cargo test --workspace`                             | 65 Rust tests pass  |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean             |
| `cargo run -p print-seed-chain` / `print-brick` / `view-png` | all run        |
| `python crates/atomr-worlds-py/python/tests/test_smoke.py` | 7 tests pass   |

The test suite covers seed determinism, hash avalanche (≥ 40% bit flip on 1-bit input perturbation),
low-byte distribution uniformity, brick / octree round-trips against a `HashMap` oracle, octree
empty-space-skip probe-bound assertions, `WorldAddr` serde round-trips (bincode + JSON), protocol
envelope round-trips, LOD math, and `MessageExtractor` stability + sibling-system co-location.

## What this is, what it isn't

This is the **foundation layer** for a procedural universe. It provides the address space, the
hash-based hierarchy of seeds, the data structures for sparse voxel content at multiple scales, and
the wire/host shape that downstream code will route through.

It is **not** (yet) a generator, a renderer, a persistence layer, or a game. Those layer on top —
the next phases fill in `Generator` bodies, persistence binding, streaming, and view integration.
See the "Out of scope" section in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full list of
deferred work.

## License

Apache-2.0. See [LICENSE](LICENSE).
