# LOD streaming and brick generation

How the engine fits a 1 km-radius brick ring around a moving observer
without spending fine-grained sample budget on the horizon. For the
higher-level scene model see [ARCHITECTURE.md](ARCHITECTURE.md); for the
renderer plumbing see [RENDERING.md](RENDERING.md).

## Two LOD conventions (and which one to use where)

There are two compatible conventions in the codebase. Both describe the
same hierarchy from opposite ends; mixing them silently is the source of
the bugs this document explains how to avoid.

| convention                                                  | depth 0 means        | depth `L` voxel edge         | used by                                                  |
| ----------------------------------------------------------- | -------------------- | ---------------------------- | -------------------------------------------------------- |
| **Renderer/streamer/generator** ("relative LOD")            | finest, 1 m / voxel  | `2^L` meters per voxel       | `world_stream.rs`, `terrain.rs`, FP brick scaling        |
| **`MetricScale::meters_per_voxel`** ("absolute scale tree") | the root cube itself | `root_size_m / 2^L` meters   | macro-state path, sphere-shape horizon clamp             |

This document — and everything below the `BrickGenContext.lod` boundary
— uses the **renderer/streamer** convention. `Lod::new(0)` is the finest
representation a brick can take, with voxels measuring one world meter
on a side. Tier `L` voxels are `2^L` meters on a side; an `L=3` brick
therefore covers a 128 m cube.

The `MetricScale` convention is reserved for the macro-pre-sim path and
the horizon clamp on spherical worlds, where the world's overall metric
scale (Earth, asteroid, gas giant) matters more than the relative tier
depth. Code that lives outside those two paths must not call
`MetricScale::meters_per_voxel` on a `Lod` produced by the streamer.

## The LOD ladder

[`LodLadder`](../crates/atomr-worlds-client/src/world_stream.rs) is a
strictly-ordered list of `LodTier { lod, outer_radius_m }` shells around
the observer. The default progressive ladder is:

| tier | LOD depth | voxel edge | outer radius |
| ---- | --------- | ---------- | ------------ |
| 0    | 0         | 1 m        | 128 m        |
| 1    | 1         | 2 m        | 256 m        |
| 2    | 2         | 4 m        | 512 m        |
| 3    | 3         | 8 m        | 1024 m       |

The shells are bands `[inner, outer)` measured against the
**3D distance** between brick AABB and observer (the previous
cube-shaped ring caused a visible directional asymmetry; see the
symmetry tests in `world_stream.rs`). Radii are multiples of the
coarsest brick edge (`BRICK_EDGE × 2^3 = 128 m`).

The band test is AABB-based, not center-based:

- A brick is masked out of tier `i` (covered by finer tier) only when
  its **far corner** distance < `outer_r_{i-1}` — i.e. the brick is
  *entirely* inside the previous tier's shell.
- A brick is masked out of tier `i` (past the load horizon) only when
  its **near corner** distance ≥ `outer_r_i` — i.e. the brick is
  *entirely* outside this tier's shell.

A center-only test would leave brick-shaped holes where an
inter-tier brick AABB straddles a band boundary: the brick at the
coarser tier gets skipped (center inside finer tier) and the fine
sub-bricks covering the AABB's protruding 3D corner also get skipped
(centers past finer tier's outer). Both sides empty ⇒ visible gap.
The AABB rule guarantees every voxel position in `[outer_r_{i-1},
outer_r_i)` is covered by at least one loaded brick at tier `i`; the
`no_gaps_at_tier_boundaries` test in `world_stream.rs` densely
samples voxel positions across the full ladder and asserts coverage.

`desired_chunks(streamer, observer, horizon_m)` walks the ladder, emits
`(brick_coord, lod)` pairs that pass the AABB band test, and sorts the
result closest-first in meters so the high-fidelity inner shell fills
before trailing far bricks. `horizon_m` is `f64::INFINITY` for flat
cube worlds and the surface-horizon distance for spheres — radii clamp
to it so we never stream past the visible surface. A small number of
bricks straddling each tier boundary load at both adjacent tiers
(volumes inside the inner sphere are also rendered by the finer tier),
which the depth buffer resolves cleanly because the finer-tier
surface is always nearer or equal to the coarser one.

## The (coord, lod_depth) cache key

Each LOD tier requests bricks at its own voxel scale. The host's
[`WorldActor::cache`](../crates/atomr-worlds-host/src/local.rs) is keyed
by `(IVec3 brick_coord, u8 lod_depth)`, not by `brick_coord` alone.

This is the critical invariant. Before 2026-05, the host received
`WorldRequest::GetBrick { lod }` but discarded `lod` before reaching the
cache and the procedural generator, and the cache keyed on
`brick_coord` only. A coarse-LOD request for `(0, 0, 0)` therefore hit
the LOD-0 cache entry and got 1 m / voxel content back. The FP loader
then scaled that brick by `2^L` via `Transform::with_scale`, stretching
sixteen 1-m voxels over a 128 m span. Visually this produced enormous
flat plates and stair-step plateaus at the LOD ring boundaries — the
"same heightfield rendered at the wrong metric" failure mode.

After the fix:

- [`BrickGenContext`](../crates/atomr-worlds-generate/src/brick.rs)
  carries `lod: Lod`. `BrickGenContext::legacy(seed, coord)` defaults to
  `Lod::new(0)` so callers that still rely on the LOD-0 byte-equality
  contract (CUDA accelerator, Python bindings, voxel writes) are
  unchanged.
- The host's `ensure_brick(brick_coord, lod)` and `snapshot(brick_coord,
  lod)` thread `lod` through to the generator and into the cache key.
- `WorldRequest::GetBrick { addr, brick, lod }` flows
  end-to-end. Subscription paths (`handle_subscribe_begin`,
  `update_observer_pos`) emit `BrickSnapshot { lod, … }` events using
  the subscription's tier LOD.
- Voxel writes, authored regions, and the user-write overlay stamp only
  on the depth-0 cache entry. They are LOD-0 operations by construction;
  stamping them on a coarse-LOD brick would either silently misalign
  (a 1-m write inside an 8-m voxel) or over-stamp (a 16-voxel literal
  region painting an entire 128-m brick). The coarse-LOD bricks stay
  purely procedural; coordinated multi-LOD writes will arrive in a
  follow-up.

The cache holds at most one entry per `(coord, depth)`, so memory cost
scales with the *number of bricks visible at each tier* — not with
2^L × shell volume. For the default 4-tier ladder this is ≈ 8 000
entries, dominated by the outermost tier-3 shell.

## World-meter sampling in the procedural generator

The procedural surface comes from `surface_height_world(seed, x_m,
z_m)` (FBM value noise) and `is_cave_world(seed, x_m, y_m, z_m)`
(Worley field). Both take continuous **world-meter** coordinates rather
than integer voxel indices, so an LOD-3 voxel at world center
`(132, 4, 4)` and an LOD-0 voxel at world meter `(132, 4, 4)` evaluate
the same noise field at the same point. Adjacent tiers therefore agree
on `H(x, z)` and `cave(x, y, z)` in expectation; what differs is the
voxel-scale at which they discretize the answer.

The dispatcher in
[`TerrainGenerator::generate_brick`](../crates/atomr-worlds-generate/src/terrain.rs)
splits on `ctx.lod.depth`:

- **`depth == 0`** runs the legacy integer-voxel path
  (`material_at(world_seed, p)` with `p = origin + (lx, ly, lz)`),
  byte-equal to the CUDA kernel in
  `crates/atomr-worlds-accel/src/cuda_kernel.cu`. The kernel implements
  lower-corner sampling (solid iff voxel index `Y < H`); we keep that
  contract.

- **`depth >= 1`** samples each voxel at its **center** in world
  meters:

  ```
  voxel_m = (1 << depth) as f32      // 2, 4, 8 for L = 1, 2, 3
  wx = (origin.x + lx + 0.5) * voxel_m
  wy = (origin.y + ly + 0.5) * voxel_m
  wz = (origin.z + lz + 0.5) * voxel_m
  ```

  and calls the LOD-agnostic `material_at_world` /
  `material_at_world_strategy`. Center sampling keeps the surface-
  reconstruction error bounded by ±voxel/2 in either direction
  (lower-corner sampling biases the coarse surface monotonically high
  by up to a full voxel, which is visually worse).

## Discretization characteristics

Because adjacent LODs round the same continuous surface `H(x, z)` to
different voxel grids, the visible surface heights differ by up to
`max(voxel_m_inner, voxel_m_outer) / 2` at tier boundaries:

| LOD-tier boundary    | worst-case vertical step |
| -------------------- | ------------------------ |
| 128 m: depth 0 ↔ 1   | 1 m                      |
| 256 m: depth 1 ↔ 2   | 2 m                      |
| 512 m: depth 2 ↔ 3   | 4 m                      |

These steps are intrinsic to multi-LOD voxel terrain without transition
meshes — they are *not* the same kind of failure as the stretched-LOD-0
bug. Eliminating them requires either:

- transition meshes (Transvoxel / dual contouring boundary stitching),
  or
- a tighter LOD ladder that pushes the depth-`L` shell so far out that
  `voxel_m_L / 2` is sub-pixel from the camera (the "finer LOD"
  roadmap item in the README).

`docs/PHASES.md` Phase 17 documents the streamer foundation and
`docs/RENDERING.md` describes the meshing path that consumes these
bricks.

## What writes look like

Writes are intentionally narrow:

- `WorldRequest::WriteVoxel { pos, voxel }` and the brush path both
  operate at LOD-0 (1 m voxels, integer world coords). They populate
  the LOD-0 cache entry and the overlay map, journal to persistence,
  and fan out per-voxel deltas to subscribers.
- Coarse-LOD cache entries are *not* invalidated when LOD-0 changes.
  This is correct for the procedural-fill case (a 1-m write underneath
  an 8-m voxel is sub-resolution; the depth-3 brick still shows the
  procedural mean) but means edited regions appear "smoothed out" once
  the observer crosses the depth-3 ring. A future phase can add a
  "writes-aware coarse fill" pass that downsamples the overlay into
  each coarse brick.

## Where this fits in the roadmap

The fix landed as a follow-up to Phase 17 (chunk streamer) and is a
prerequisite for the roadmap items called out in the README:

- **Finer-grained LOD ladders** — adding sub-meter LOD−1 (½-m voxels)
  for vehicle-detail rings, or a separate `LodLadder` per view-mode
  (e.g. RTS pays for a coarser inner ring than FP), requires the
  per-LOD generation contract documented above.
- **Multi-style generators** — different procedural strategies
  (cities, biome packs, planetary archetypes) plug into the same
  `BrickGenerator` trait; each receives `ctx.lod` and is responsible
  for sampling its source field at the right metric. The strategy-
  driven branch (`material_at_world_strategy`) already threads the
  `MaterialContext` through with floor-rounded world coords for
  backward compat with strategies that key on integer voxel positions.
- **Real-world data feeds** — ingesting Earth elevation tiles, OSM
  vector data, or live satellite layers means populating
  `surface_height_world` (and friends) from a tiled raster source
  rather than FBM. The world-meter sampling API is the right
  integration point: an Earth-DEM `BrickGenerator` just queries its
  tile pyramid at the right level given `ctx.lod` and writes voxels.

See [PHASES.md](PHASES.md) for the current phase log and
[ARCHITECTURE.md](ARCHITECTURE.md) for the broader streaming model.
