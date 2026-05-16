# Phase 19 — Advanced Algorithmic Topologies & Layered Voxel Architecture

> Landed in 2026-Q2. Adds the full algorithm catalogue from the paper
> *Advanced Algorithmic Topologies and Layered Architecture in Procedural
> Voxel World Generation* as additive strategy-pattern slots layered on
> top of the existing generator. Vanilla behaviour is preserved
> byte-for-byte; new algorithms opt-in via `WorldGenPreset::Advanced`
> and `WorldGenPreset::Showcase`, or per-slot via the harness DSL.

## Goals

1. **Land every paper algorithm** as a swappable trait-object strategy.
2. **Preserve byte-equality** with the prior `TerrainGenerator` output
   for the `Vanilla` preset, asserted by a CI snapshot test.
3. **Mirror `RenderConfig`'s shape** — `Arc<dyn Trait>` slots, named
   presets, name-keyed registry, harness DSL — so A/B testing any
   algorithm is one assignment.
4. **CPU-first reference**; CUDA kernels gated behind `feature = "cuda"`
   with paired byte-equality tests for each accelerated strategy.

## Architecture

A new orchestrating layer sits **alongside** (not replacing)
`TerrainGenerator`:

- `LayeredGenerator` — `BrickGenerator` impl registered under
  `registry::TERRAIN_LAYERED`. The legacy `TerrainGenerator` is still
  registered under `TERRAIN` and is unchanged.
- `LayeredBrickPipeline` — implements `BrickPipeline`. Runs stages in a
  fixed order asserted at construction so a reshuffled `WorldGenConfig`
  cannot silently scramble the pass chain.
- `BrickWorkspace` — per-brick scratch carrying a padded 18³ density
  grid and material buffer (so neighbour-aware passes sample one voxel
  of overlap), the in-progress `Brick`, an `anchors` list, and an
  optional `LightOverlay`.
- `WorldGenConfig` — thirteen `Arc<dyn Trait>` slots and three named
  presets (`Vanilla`, `Advanced`, `Showcase`).
- `FeatureSeederStrategy` — solves cross-brick determinism for
  path-based features. Anchors live on a coarse column grid (default
  64 m), seeded via `child_seed(world_seed, FEATURE_DIM, column_coord)`.
  Every brick scans its 3×3×3 column neighbourhood; features
  deterministically trace from anchor seeds, clipping output to the
  brick AABB. Memoized in a `FeatureAnchorCache` (mirror of
  `MacroStateCache`).

Macro pre-sim (plates → relief → climate → biomes → hydrology) is
extended, not rewritten: new `BiomeMatrixStrategy` and
`BiomeBlendStrategy` slots re-skin biome assignment and inter-cell
smoothing; the existing `DefaultMacroGenerator` is left intact for the
`Vanilla` preset.

## Stage order

`LayeredBrickPipeline` runs these stages, in this order. The order is
asserted at construction and is part of the pipeline contract:

```
feature_seeder
biome_matrix
biome_blend
density
strata
caves
ore
erosion
fluid
structures
flora
sky_light
```

Each is one `Arc<dyn FooStrategy>` slot on `WorldGenConfig`. `None`
impls early-return at zero cost when a preset doesn't need a given
pass.

## Strategy map

| Slot | Trait | Vanilla default | Other impls |
|---|---|---|---|
| Pipeline orchestrator | `BrickPipeline` | `LayeredBrickPipeline` (when opted in) | — |
| Density field | `DensityFieldStrategy` | `HeightmapPlanar` (via `MonolithicTerrainPass`) | `Hybrid2D3D`, `Pure3DOverhang`, `FloatingIslandField` |
| Strata | `StrataStrategy` | `TopsoilLayer` (via `MonolithicTerrainPass`) | `LayeredGeology`, `KrigingInterpolated` |
| Caves | `CaveStrategy` | `NoneCaves` (Worley still runs inside the monolith) | `WorleyThreshold`, `CellularAutomata3D`, `PerlinWorm`, `IsosurfaceIntersection` |
| Ore veins | `OreVeinStrategy` | `NoneOre` | `ThresholdNoise`, `BiasedRandomWalk` |
| Erosion | `ErosionStrategy` | `NoneErosion` (river carve inside the monolith) | `MacroRiverOnly`, `DropletHydraulic` (CPU + CUDA) |
| Fluid | `FluidStrategy` | `NoneFluid` (sea level inside the monolith) | `Static`, `CellularAutomataFlow`, `LatticeBoltzmannD3Q19` (CPU + CUDA) |
| Structures | `StructureStrategy` | `NoneStructures` | `WaveFunctionCollapse`, `Jigsaw`, `QwfcClassicalSim` |
| Flora | `FloraStrategy` | `NoneFlora` | `LSystemTrees`, `BlueNoiseGrass` |
| Placement | `PlacementStrategy` | `NonePlacement` | `WhiteNoise`, `UniformGrid`, `PoissonDiskBridson`, `MitchellBestCandidate` |
| Biome matrix | `BiomeMatrixStrategy` | `NoneBiomeMatrix` (per-face Whittaker stays in macro pre-sim) | `PerFaceWhittaker`, `WhittakerDirect2D`, `VoronoiCells` |
| Biome blend | `BiomeBlendStrategy` | `NoneBiomeBlend` (hard borders) | `Hard`, `NormalizedSparseConvolution`, `BufferTerrainInjected` |
| Sky light | `SkyLightStrategy` | `NoneSkyLight` | `VerticalCastWithDiffusion` |
| Feature seeder | `FeatureSeederStrategy` | `EmptySeeder` | `ColumnAnchorSeeder` |

Separately, the voxel and view crates gained additional pluggable layers:

| Slot | Trait | Vanilla default | Other impls |
|---|---|---|---|
| Brick codec | `BrickCodec` | `RawU16` | `Rle`, `Zlib`, `PaletteRle` |
| Brick storage | `BrickStorage` | `DenseBrick` | `SegmentedRowBrick`, `SvoBrick` |
| Mesher | `MeshStrategy` | `GreedyFlat` | `NaiveMesh`, `MarchingCubes`, `DualContouring` |
| AO | `AmbientOcclusion` | `MinecraftCornerAo` | `BrickEdgeAwareAo` |
| Fog | `FogStrategy` | (existing) | `BiomeBlendedFog` |

## Paper-section → implementation map

### §1 Foundational data structures
- **Raw byte indexing / palette / 1D layout** — already in
  `atomr-worlds-voxel`.
- **zlib + RLE + Palette+RLE** — `crates/atomr-worlds-voxel/src/codec.rs`:
  `BrickCodec` trait + `RawU16`, `Rle`, `Zlib`, `PaletteRle` impls.
- **Sparse Voxel Octree (per-brick)** — `BrickStorage::SvoBrick` in
  `crates/atomr-worlds-voxel/src/storage.rs`. 8-bit child mask +
  popcount-indexed children, mirroring the existing world-level octree.
- **Segmented row allocation** — `BrickStorage::SegmentedRowBrick`:
  per-Y-row tag `RowKind { Uniform(Voxel) | Indexed(u32) }`; uniform
  rows skip the 4096-slot allocation.

### §1 (cont.) Meshing
Implemented in `crates/atomr-worlds-view/src/mesh/`:
- `naive.rs` — `NaiveMesh`, one quad per visible voxel face.
- `mod.rs` — `GreedyFlat` (current default), retained.
- `marching_cubes.rs` — `MarchingCubes`, reads the workspace's padded
  18³ density grid; standard 256-case edge table.
- `dual_contouring.rs` — `DualContouring`, consumes Hermite data
  (intersection point + normal per edge) from the density gradient;
  solves a QEF per cell (Schmitz–Garland classic).

### §2 Continental layer
- **`DensityFieldStrategy`** — `crates/atomr-worlds-generate/src/pipeline/density.rs`:
  - `HeightmapPlanar` — current heightmap projection.
  - `Hybrid2D3D` — `density = baseHeight(x,z) - y + noise3D(x,y,z)`.
  - `Pure3DOverhang` — pure 3D `fbm_value(seed, x,y,z, cfg)` with a
    vertical bias profile.
  - `FloatingIslandField` — `island_density()` from
    `atomr-worlds-noise`, anchored by `FeatureKind::FloatingIsland`.
- **Domain warp** — `crates/atomr-worlds-noise/src/domain_warp.rs`:
  `WarpConfig { octaves, amp_m, lacunarity }` + `warp_point()` +
  `iterated_warp()`.
- **Geological strata** — `crates/atomr-worlds-generate/src/pipeline/strata.rs`:
  `TopsoilLayer`, `LayeredGeology`, `KrigingInterpolated`.

### §3 Biomes
- **`BiomeMatrixStrategy`** — `pipeline/biome_matrix.rs`:
  `PerFaceWhittaker`, `WhittakerDirect2D`, `VoronoiCells`.
- **`BiomeBlendStrategy`** — `pipeline/biome_blend.rs`: `Hard`,
  `NormalizedSparseConvolution`, `BufferTerrainInjected`.

### §4 Caves
- **`CaveStrategy`** — `pipeline/caves/`:
  - `worley.rs` — `WorleyThreshold`, current Worley behaviour
    refactored into the slot.
  - `ca3d.rs` — `CellularAutomata3D`, 3D Conway-style birth/death over
    a randomised initial density.
  - `worm.rs` — `PerlinWorm`, anchor-driven directed walk.
  - `isosurface.rs` — `IsosurfaceIntersection` (Cheese ∪ Spaghetti ∪
    Noodle), ε_y modulated by `y² - 1` parabola.

### §5 Hydraulic erosion & fluid
- **`ErosionStrategy`** — `pipeline/erosion/`:
  - `macro_river.rs` — `MacroRiverOnly`, factored-out current river
    carve.
  - `droplet.rs` — `DropletHydraulic`, CPU reference impl.
  - `droplet_cuda.rs` — CUDA NVRTC kernel (feature = "cuda"), sorted
    atomic deposits for byte-equality with CPU.
- **`FluidStrategy`** — `pipeline/fluid/`:
  - `Static` — sea level + lake surfaces only.
  - `ca_flow.rs` — `CellularAutomataFlow`, Minecraft-style ticked
    rules.
  - `lbm.rs` — `LatticeBoltzmannD3Q19`, 19-velocity LBM with BGK
    collision.
  - `lbm_cuda.rs` — CUDA NVRTC kernel (feature = "cuda").

### §6 Micro-scale generation
- **`FloraStrategy`** — `pipeline/flora/`:
  - `lsystem.rs` — `LSystemTrees`, declarative grammar + 3D turtle
    interpreter.
  - `grass.rs` — `BlueNoiseGrass`, surface decor via
    `PlacementStrategy`.
- **`PlacementStrategy`** — `pipeline/placement.rs`: `WhiteNoise`,
  `UniformGrid`, `PoissonDiskBridson`, `MitchellBestCandidate`.
- **`OreVeinStrategy`** — `pipeline/ore.rs`: `ThresholdNoise`,
  `BiasedRandomWalk`.

### §7 Macro structures
- **`StructureStrategy`** — `pipeline/structures/`:
  - `wfc.rs` — `WaveFunctionCollapse`, entropy-min observation +
    AC-3 propagation + backtracking.
  - `wfc_cuda.rs` — CUDA propagation kernel (feature = "cuda").
  - `jigsaw.rs` — `Jigsaw`, start-pool → template-pool recursive
    depth-bounded fill. Reuses existing `AuthoredRegion`
    infrastructure.
  - `qwfc.rs` — `QwfcClassicalSim`, weighted-pick variant; documented
    research stub.

### §8 Pipeline & multi-pass determinism
- **Procedural determinism** — enforced via `child_seed` chain rooted
  at the world seed.
- **Multi-pass generation** — implemented by `LayeredBrickPipeline`.
  Stage order asserted at construction.
- **Cross-brick continuity** — `FeatureSeederStrategy::ColumnAnchorSeeder`
  emits anchors on a coarse column grid (default 64 m). Each brick
  scans the 3×3×3 column neighbourhood; features clip output to the
  brick AABB. All anchor-to-anchor processing is pure (anchor seed in,
  output voxels out), never invokes the neighbour brick's pipeline.
- **Stage ordering safety** — `STAGE_ORDER` const +
  `debug_assert_eq!` in `LayeredBrickPipeline::new` prevents
  config-shuffle scrambling.

### §9 Rendering / lighting
- **`SkyLightStrategy`** — `pipeline/light.rs`:
  `VerticalCastWithDiffusion`. Casts rays down Y from brick top until
  hitting a solid voxel; 6 iterations of lateral/downward decrement.
  Output packed into a 4-bit-per-voxel `LightOverlay` attached to
  `Brick`.
- **Voxel AO** — `MinecraftCornerAo` retained as Vanilla;
  `BrickEdgeAwareAo` reads neighbour apron to remove edge seams.
- **Fog & atmospherics** — `BiomeBlendedFog` interpolates fog tint
  across biome boundaries.

### §10 CUDA acceleration
Each compute-heavy strategy ships paired CPU + CUDA impls behind
`feature = "cuda"` on `atomr-worlds-generate` (kernels themselves live
in `atomr-worlds-accel`):

- `droplet_cuda.rs` — one thread per droplet; sediment writes via
  sorted atomic-deposit indices.
- `lbm_cuda.rs` — one thread per lattice node; double-buffer streaming
  + collision.
- `ca3d_cuda.rs` — one thread per voxel; double-buffer iteration grid;
  apron in shared memory.
- `wfc_cuda.rs` — propagation queue parallelised via warp-level scan;
  tile selection stays on CPU (sequential entropy-min).

Each kernel ships with `#[cfg(test)] fn cpu_cuda_byte_equality()` that
runs both implementations on a fixed seed and asserts identical output.
CUDA is opt-in; CPU paths are always present.

## Vanilla byte-equality contract

`WorldGenPreset::Vanilla` constructs a `WorldGenConfig` where the
`density` and `strata` slots both point at `MonolithicTerrainPass`,
which delegates to the legacy `TerrainGenerator`. Every other slot is a
`None*` no-op. The codec is `RawU16` and the storage is `DenseBrick`.

A regression test
[`crates/atomr-worlds-generate/tests/vanilla_byte_equality.rs`](../crates/atomr-worlds-generate/tests/vanilla_byte_equality.rs)
iterates 4 seeds × 8 brick coordinates and asserts that
`LayeredGenerator(Vanilla)` produces bricks byte-equal to
`default_terrain()` for every voxel. The test is wired into the default
CI run and must remain green.

The default `GeneratorRegistry` still resolves `TERRAIN` to
`TerrainGenerator` (unchanged). `TERRAIN_LAYERED` is opt-in via
`BuiltinSelector` weights or `GenerationPolicy::Custom`.

## Trying it out

```rust
use atomr_worlds_generate::{
    LayeredGenerator, WorldGenConfig, WorldGenPreset,
};

// Advanced preset: every paper algorithm at moderate cost.
let g = LayeredGenerator::new(WorldGenConfig::preset(WorldGenPreset::Advanced));

// Showcase preset: every algorithm cranked up.
let g = LayeredGenerator::new(WorldGenConfig::preset(WorldGenPreset::Showcase));
```

Harness DSL (per-slot swap):

```toml
[event."10s"]
worldgen.set_strategy = { slot = "caves", name = "CellularAutomata3D" }
```

See [`examples/showcase-strategies/`](../examples/showcase-strategies)
for an interactive demo that rotates through every preset.

## Related docs

- [PIPELINE.md](PIPELINE.md) — pipeline orchestration + cross-brick
  determinism contract.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the broader engine layout.
- [RENDERING.md](RENDERING.md) — render-side strategy slots.
- [HYDROLOGY.md](HYDROLOGY.md) — macro pre-sim that produces biomes
  + water; the `BiomeMatrix`/`BiomeBlend`/`Erosion` strategies all
  layer on top of this.
- [LOD.md](LOD.md) — per-LOD generation contract; `LayeredGenerator`
  honours the same per-LOD apron rules.
