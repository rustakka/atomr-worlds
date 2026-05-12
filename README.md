# atomr-worlds

A procedural-universe substrate for [atomr](https://github.com/rustakka/atomr): hierarchical seeded
generation across **Universe → Galaxy → Sector → System → World**, with per-node dimensions, sparse
voxel storage, metric levels of detail, and a hosting model that runs either embedded in-process
(single-player) or sharded across an atomr cluster (multiplayer) — same actor protocol either way.

## Status

**Phase 0 — primitives and structures.** This phase ships the type system, data structures, wire
format, and host trait skeletons. Generation, persistence, rendering, and GPU acceleration are
explicitly deferred to subsequent phases. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
overall model and [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for module-by-module specifics.

## Workspace layout

```
atomr-worlds/
├── crates/
│   ├── atomr-worlds-core      ─ coordinates, addressing, seed derivation, LOD (no atomr deps)
│   ├── atomr-worlds-voxel     ─ Brick (16³), arena Octree, SparseVoxelStore trait
│   ├── atomr-worlds-proto     ─ WorldRequest/WorldEvent/Envelope, bincode 2 wire format
│   ├── atomr-worlds-host      ─ WorldHost trait, LocalHost/ClusterHost, MessageExtractor
│   └── atomr-worlds-testkit   ─ proptest strategies, cross-crate verification
├── examples/
│   └── print-seed-chain       ─ smoke binary; prints seed chain + metric scales
└── docs/
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
cargo run   -p print-seed-chain
```

`print-seed-chain` derives the five-level seed chain for a sample `WorldAddr` and prints the leaf
voxel size at each tier (universe → world) for the default `MetricScale`s.

## Verification gates

Phase 0 ships green:

| gate                                                 | status              |
| ---------------------------------------------------- | ------------------- |
| `cargo check --workspace`                            | clean               |
| `cargo test --workspace`                             | 38 tests pass       |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean             |
| `cargo run -p print-seed-chain`                      | runs and prints     |

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
