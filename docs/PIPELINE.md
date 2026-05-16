# Brick generation pipeline

> The layered brick pipeline introduced in [Phase 19](PHASE_19.md). This
> document is the **contract** — what guarantees the pipeline provides
> and what each strategy slot is allowed to do. For the algorithm
> catalogue see PHASE_19.md.

## Layers, top-down

```
                       ┌────────────────────────────────┐
WorldGen tier          │  WorldGen (per WorldAddr)      │
                       └───────────────┬────────────────┘
                                       │ child_seed(world)
                       ┌───────────────▼────────────────┐
Macro pre-sim          │  DefaultMacroGenerator         │
(plates → relief →     │   produces WorldMacroState,    │
 climate → biomes →    │   cached in MacroStateCache    │
 hydrology)            └───────────────┬────────────────┘
                                       │ ws.ctx.macro_state
                       ┌───────────────▼────────────────┐
Cross-brick anchors    │  FeatureSeederStrategy         │
(64m column grid)      │   ColumnAnchorSeeder           │
                       │   cached in FeatureAnchorCache │
                       └───────────────┬────────────────┘
                                       │ ws.anchors
                       ┌───────────────▼────────────────┐
Per-brick pipeline     │  LayeredBrickPipeline          │
(fixed stage order)    │   biome_matrix → biome_blend → │
                       │   density → strata →           │
                       │   caves → ore → erosion →      │
                       │   fluid → structures →         │
                       │   flora → sky_light            │
                       └───────────────┬────────────────┘
                                       │ Brick
                       ┌───────────────▼────────────────┐
Codec + storage        │  BrickCodec → BrickStorage     │
                       └───────────────┬────────────────┘
                                       │
                       ┌───────────────▼────────────────┐
Mesher                 │  MeshStrategy (greedy / naive /│
                       │   marching-cubes / dual-       │
                       │   contouring)                  │
                       └───────────────┬────────────────┘
                                       │
                       ┌───────────────▼────────────────┐
Render                 │  AO + Fog + sky-light bake     │
                       └────────────────────────────────┘
```

## BrickWorkspace

Every pipeline stage receives `&mut BrickWorkspace`. The workspace
carries:

- `ctx: BrickGenContext` — immutable input: `world_seed`, `brick_coord`,
  `lod`, `shape: WorldShape`, `macro_state: Arc<WorldMacroState>`,
  `scale: MetricScale`.
- `density: Vec<f32>` — padded `WS_APRON_EDGE³ = 18³ = 5832` scalar
  field. Sign convention: **positive = solid, negative = empty,
  zero = surface**. Apron occupies `-1` and `BRICK_EDGE` on each axis.
- `materials: Vec<Voxel>` — same shape as `density`; written by
  `StrataStrategy` from the density field. Apron mirrors neighbour
  data when relevant.
- `anchors: Vec<FeatureAnchor>` — union of feature anchors visible from
  this brick's 3×3×3 column neighbourhood. Populated by the seeder
  stage before any per-brick stage runs.
- `brick: Brick` — the in-progress output. Stages write into this for
  final voxels; the pipeline returns `ws.brick` at the end.
- `light: Option<Box<LightOverlay>>` — populated by the sky-light stage;
  moved into `brick.light_overlay` at pipeline exit.

Apron access uses signed coordinates (`-1..=BRICK_EDGE`):

```rust
let d = ws.density_at(-1, 0, 0);     // apron sample
ws.set_density(15, 7, 3, 1.0);       // brick-local sample
let v = ws.material_at(0, BRICK_EDGE as i32, 0);
```

`brick_index(usize, usize, usize)` returns a flat index from
brick-local coordinates (`0..BRICK_EDGE`); `apron_index(i32, i32, i32)`
accepts the full apron range.

## Stage contract

The stages run in **fixed order**:

```
feature_seeder → biome_matrix → biome_blend →
density → strata → caves → ore → erosion → fluid →
structures → flora → sky_light
```

The order is asserted at `LayeredBrickPipeline` construction time
(`debug_assert_eq!` on a static `STAGE_ORDER` array). Reshuffling slots
on a `WorldGenConfig` does not change the order; only swapping
implementations does.

Each stage is exposed as one of these traits in
`crates/atomr-worlds-generate/src/pipeline/strategies.rs`:

| Trait | Signature | Reads | Writes |
|---|---|---|---|
| `FeatureSeederStrategy` | `seed(&self, &mut ws)` | `ws.ctx` | `ws.anchors` |
| `BiomeMatrixStrategy` | `run(&self, &mut ws)` | `ws.ctx.macro_state` | per-ws biome table |
| `BiomeBlendStrategy` | `run(&self, &mut ws)` | biome table | biome table |
| `DensityFieldStrategy` | `run(&self, &mut ws)` | `ws.ctx` | `ws.density` |
| `StrataStrategy` | `run(&self, &mut ws)` | `ws.density` | `ws.materials` |
| `CaveStrategy` | `run(&self, &mut ws)` | `ws.density` | `ws.materials` |
| `OreVeinStrategy` | `run(&self, &mut ws)` | `ws.anchors`, `ws.materials` | `ws.materials` |
| `ErosionStrategy` | `run(&self, &mut ws)` | `ws.density`, `ws.materials` | `ws.materials`, `ws.density` |
| `FluidStrategy` | `run(&self, &mut ws)` | `ws.materials`, hydrology | `ws.materials` |
| `StructureStrategy` | `run(&self, &mut ws)` | `ws.anchors` | `ws.materials` |
| `FloraStrategy` | `run(&self, &mut ws)` | `ws.anchors`, `ws.materials` | `ws.materials` |
| `PlacementStrategy` | `place(&self, &ws) -> Vec<[f32;3]>` | `ws.materials` | (returns) |
| `SkyLightStrategy` | `run(&self, &mut ws)` | `ws.materials` | `ws.light` |

After every stage completes the pipeline calls `ws.brick.write_voxel`
for each non-empty `ws.materials` entry, and then moves `ws.light`
into `ws.brick.light_overlay`.

`None*` impls (`NoneDensity`, `NoneCaves`, ...) early-return at zero
cost so a preset with no caves pays nothing for the cave slot.

## Determinism contract

Every output of every stage is a pure function of `BrickGenContext` and
the strategy configuration. **No global state, no time, no I/O.**
Re-running the pipeline with the same input yields the same `Brick`.

The seed chain:

```
world_seed
   ├── child_seed(world_seed, MACRO_DIM, ()) ─── macro pre-sim
   ├── child_seed(world_seed, FEATURE_DIM, column) ─── per-anchor
   │      └── splitmix64(column_seed ^ kind_disc) ─── per-feature kind
   ├── child_seed(world_seed, DROPLET_DIM, brick_coord) ─── per-droplet
   └── child_seed(world_seed, PLACEMENT_DIM, brick_coord) ─── placement
```

Per-dimension constants (`FEATURE_DIM`, `DROPLET_DIM`, etc.) live next
to the strategy that consumes them. Re-using a dim across strategies
is forbidden — different stages must start from different sub-chains
so a tweak to one strategy can't shift another's RNG state.

## Cross-brick continuity

Path-based features (Perlin worms, biased-walk ore veins, WFC dungeons,
Jigsaw villages, L-system trees, floating-island falloff) are
**column-anchored**, not brick-anchored:

1. The world is partitioned into coarse `column_size_m`-sided columns
   (default 64 m).
2. `FeatureSeederStrategy::ColumnAnchorSeeder` seeds the union of
   anchors visible from the brick's 3×3×3 column neighbourhood into
   `ws.anchors`.
3. Each subsequent stage filters `ws.anchors` by `FeatureKind` and
   traces from each anchor's seed deterministically.
4. The trace **does not invoke the neighbour brick's pipeline**.
   Output voxels are clipped to the brick AABB and discarded outside
   it. The same trace executed from any neighbour brick produces the
   same voxels in the overlap.

The full anchor computation is memoized in `FeatureAnchorCache`
(analogous to `MacroStateCache`) keyed by `(world_seed, column_coord)`.

This is the key invariant that lets the pipeline scale to streaming
generation: a brick can be generated independently and asynchronously
without any neighbour brick's prior state, and yet features crossing
the boundary are seamless.

## Vanilla byte-equality

`WorldGenPreset::Vanilla` configures the pipeline so that
`density` and `strata` both point at `MonolithicTerrainPass`, which
delegates to the legacy `TerrainGenerator`. Every other slot is a
`None*` no-op. This preserves byte-equality with the prior generator
output, asserted by `tests/vanilla_byte_equality.rs`.

The default `GeneratorRegistry` still resolves `TERRAIN` to
`TerrainGenerator`. The new pipeline lives under
`registry::TERRAIN_LAYERED` and is opt-in via `BuiltinSelector` weights
or `GenerationPolicy::Custom`.

## Apron sourcing

Stages that need neighbour data (sky light, droplet erosion, CA caves,
LBM fluid) read from the **workspace apron** — the outer ring of cells
in `ws.density` / `ws.materials`. The apron is filled with
deterministic samples of the same density/material function as the
brick interior, so stages get the right neighbour values without ever
materialising the neighbour brick.

This is cheaper than recursing into the neighbour pipeline (which
would either re-run all stages or require an LRU of finished neighbour
bricks) and still byte-exact, because the apron source values come
from the same pure function.

## Extending the pipeline

To add a new strategy:

1. Implement the relevant trait
   (`crates/atomr-worlds-generate/src/pipeline/strategies.rs`).
2. Add it to the relevant `match` arm in
   `pipeline/registry.rs::apply_worldgen_strategy_by_name`. The
   strategy's `id()` should be the PascalCase struct name.
3. (Optional) Wire it into `build_advanced()` or `build_showcase()`
   in `pipeline/presets.rs`.
4. Add a deterministic unit test
   (`same (seed, coord) → same Brick`) and, if the strategy spans
   bricks, a boundary-continuity test.

Vanilla byte-equality must remain green for every PR. Run:

```sh
cargo test -p atomr-worlds-generate --test vanilla_byte_equality
```

## Related code

- `crates/atomr-worlds-generate/src/pipeline/` — pipeline module.
- `crates/atomr-worlds-generate/src/macro_state/` — macro pre-sim
  feeding `ws.ctx.macro_state`.
- `crates/atomr-worlds-generate/src/strategies/terrain.rs` —
  `MonolithicTerrainPass` and the legacy `TerrainGenerator`.
- `crates/atomr-worlds-voxel/src/codec.rs` — `BrickCodec`.
- `crates/atomr-worlds-voxel/src/storage.rs` — `BrickStorage`.
- `crates/atomr-worlds-voxel/src/light.rs` — `LightOverlay`.
- `crates/atomr-worlds-noise/` — primitives (Simplex, fBm, domain
  warp, island).
- `crates/atomr-worlds-view/src/mesh/` — mesh strategies.
