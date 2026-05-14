# Hydrology — ocean, lake, and river water bodies

How `atomr-worlds` places water. The hydrology overlay is the final layer
of the geologic macro pre-simulation: it consumes the elevation and
climate fields, classifies every surface-grid face as ocean / lake /
river, and records a water surface elevation plus a drainage network. The
brick generator then consults this per-face data to place real water
columns and carve river channels.

For the macro pre-sim it builds on, see the *Geologic macro pre-sim*
section of [IMPLEMENTATION.md](IMPLEMENTATION.md). For the LOD-consistency
contract the brick-level consumer must honour, see [LOD.md](LOD.md).

## The problem it solves

Before this layer, the only "water" was cosmetic: the `OCEAN` biome's
topsoil voxel was painted with `MATERIAL_WATER`. Basins were not filled to
sea level, and lakes and rivers did not exist. The hydrology overlay
replaces that with genuine water bodies derived from the global
geological context.

## Pipeline

`DefaultMacroGenerator::generate`
([`macro_state/mod.rs`](../crates/atomr-worlds-generate/src/macro_state/mod.rs))
runs, in order:

```
plates → relief → climate → biomes → hydrology
```

### Relief — `macro_state/relief.rs`

Tectonic plates produce a *piecewise-flat* elevation field: every face of
a plate sits at exactly the plate base elevation, with uplift only along
convergent boundaries. That is unusable for hydrology — on a perfectly
flat plate every interior face is a drainage pit, so no rivers form and
there are no closed basins for lakes.

`apply_relief` adds a smooth, deterministic multi-octave FBM relief on top
of the plate elevation (land takes the full amplitude; the ocean floor a
gentler amount). It runs *before* climate, so climate, biomes, the
hydrology overlay, and brick-level terrain all consume one coherent
elevation field with real drainage gradients and basins. Tunables:
[`ReliefConfig`](../crates/atomr-worlds-generate/src/macro_state/relief.rs).

### Hydrology — `macro_state/hydrology/`

A [`WaterBodyStrategy`](../crates/atomr-worlds-generate/src/macro_state/hydrology/mod.rs)
trait with three implementations, run in dependency order by
[`HydrologyGenerator`](../crates/atomr-worlds-generate/src/macro_state/hydrology/mod.rs)
and aggregated into a `WaterField`:

| strategy | file | algorithm |
| --- | --- | --- |
| `OceanStrategy` | `hydrology/ocean.rs` | per-face threshold: `elev_m < sea_level_m` → ocean, surface = sea level |
| `LakeStrategy`  | `hydrology/lake.rs`  | Barnes-style priority-flood seeded from ocean faces; closed basins become lakes, climate-gated |
| `RiverStrategy` | `hydrology/river.rs` | flow accumulation over the flood drainage tree; corridors above `river_threshold` become rivers |

Each strategy's `compute` is a whole-grid pass (lake fill is a global
priority-flood, river accumulation a global topological sweep — neither is
expressible per-face). Later strategies see earlier ones via
`HydrologyInput::prior`, so the lake flood can seed from the ocean and the
river accumulation can route into both.

#### Ocean

Faces with `elev_m < sea_level_m` (default `0.0` — the same test
`biome.rs` uses). The water surface is pinned to `sea_level_m`.

#### Lake — priority-flood basin fill

The surface grid is a closed sphere with no boundary, so the flood is
seeded from the **ocean faces** (the global drainage base level). A
min-heap pops faces in increasing flood level; each unprocessed neighbour
takes `max(its own elevation, the level it was reached at)`. A non-ocean
face whose flood level sits more than `min_lake_depth_m` above its own
ground is a closed basin — and becomes a lake only if local humidity
clears `lake_aridity_threshold` (arid basins stay dry salt flats).

The flood also records, per face, the neighbour it was flooded *from*
(`parent`). Those parent chains form a spanning forest rooted at the
ocean — a complete drainage network that routes correctly *through*
filled basins, out their spill point. It is published as the lake layer's
`flow_dir`.

#### River — flow accumulation over the drainage tree

`RiverStrategy` takes the lake layer's flood drainage tree as given, gives
every land face a local flow contribution (`base_flow_per_face` plus
`precipitation_mm × precip_to_flow_scale`), and accumulates flow
downstream in topological order (Kahn's algorithm — every face is summed
before its downstream target). Because the drainage tree routes through
filled lake basins, a river chains headwater → stream → lake → stream →
sea. Land faces (non-ocean, non-lake) whose accumulated flow clears
`river_threshold` are river corridors.

#### Aggregation

`HydrologyGenerator::generate` runs the three strategies and aggregates
with priority **ocean > lake > river** for `water_kind` / `water_surface_m`.
`flow_dir` / `flow_accum` come unconditionally from the river layer and
are retained for *every* face — the brick generator carves channels using
them.

### `WaterField` and `MacroSample`

[`WaterField`](../crates/atomr-worlds-generate/src/macro_state/hydrology/mod.rs)
is struct-of-arrays, one entry per surface-grid face: `water_kind`
(`NONE`/`OCEAN`/`LAKE`/`RIVER`), `water_surface_m`, `flow_dir`,
`flow_accum`, plus the scalar `sea_level_m`. It is stored on
`WorldMacroState` and folded into the macro-state digest.

`WorldMacroState::sample(dir)` surfaces the per-face hydrology data on
`MacroSample` as `water_kind`, `water_surface_m`, `flow_dir`, `flow_accum`
— this is the interface the brick generator consumes.

## Brick-level consumer — `terrain.rs`

`TerrainGenerator::material_at_macro` /
[`material_at_macro_strategy`](../crates/atomr-worlds-generate/src/terrain.rs)
project each voxel column onto the macro surface, then:

1. **River carve** — `river_carve` runs first for `RIVER`-classified
   faces. The macro layer supplies the corridor (which faces, `flow_dir`,
   `flow_accum`); the local seed supplies the detail. A low-frequency FBM
   meanders the channel centerline (anchored on the face centroid), a
   Worley field jitters the bank width, and the channel is carved with a
   parabolic bed. Width and depth scale with `sqrt(flow_accum)`.
2. **Water fill** — above the (carved) terrain surface, an air voxel below
   the column's water surface becomes `MATERIAL_WATER`. Ocean and lake use
   `water_surface_m`; rivers use the carved channel water level (inset
   slightly below the bank, never above the macro corridor surface so a
   river meeting a lake/sea shares its level).
3. **Submerged bed** — a solid topsoil voxel under a body of water reads
   as `MATERIAL_SAND` regardless of the biome above it.

Single shared water material: `MATERIAL_WATER` (palette id 5, alpha 0.6) —
no new material ids, no renderer changes.

### LOD consistency

Water surfaces are naturally LOD-stable: `water_surface_m` is a per-face
flat scalar converted to voxels with a scale-only `mpv`, independent of
the per-LOD voxel size. River-carve noise is sampled in voxel-centered,
LOD-consistent world meters (`(p + 0.5) × voxel_m`), the same discipline
`surface_height_world` uses — see [LOD.md](LOD.md).

### Non-macro path

The legacy path (`macro_state: None` — cube worlds, the CUDA fallback)
places no water and carves no rivers. It is byte-for-byte unchanged.

## Configuration

- [`ReliefConfig`](../crates/atomr-worlds-generate/src/macro_state/relief.rs)
  — relief amplitude over land / ocean, frequency, octaves.
- [`HydrologyConfig`](../crates/atomr-worlds-generate/src/macro_state/hydrology/mod.rs)
  — `sea_level_m`, `min_lake_depth_m`, `lake_aridity_threshold`,
  `river_threshold`, `base_flow_per_face`, `precip_to_flow_scale`.
- River channel geometry tunables live on
  [`TerrainConfig`](../crates/atomr-worlds-generate/src/terrain.rs)
  (`river_*` fields).

Both `ReliefConfig` and `HydrologyConfig` hang off `MacroConfig`.

## Determinism

Every strategy is a pure function of its inputs. Float ordering uses
`f32::total_cmp` (never `to_bits()` ordering — elevations go negative and
`to_bits` is only monotonic for non-negative floats) with a `FaceId`
tie-break; no `HashMap` iteration influences output. The `WaterField`
arrays fold into the macro-state digest exactly like the upstream layers.
Gate: [`tests/macro_determinism.rs`](../crates/atomr-worlds-generate/tests/macro_determinism.rs)
and [`tests/hydrology.rs`](../crates/atomr-worlds-generate/tests/hydrology.rs).

## Verification

- Unit tests: per-strategy determinism + behaviour
  (`macro_state/hydrology/{ocean,lake,river}.rs`), relief
  (`macro_state/relief.rs`), and brick-level water fill / river carve
  (`terrain.rs` macro-path tests).
- Integration: [`tests/hydrology.rs`](../crates/atomr-worlds-generate/tests/hydrology.rs)
  asserts the default world has oceans, lakes, and rivers, that the
  `WaterField` invariants hold, and that the macro digest is deterministic
  and seed-sensitive.
- Visual: harness scenarios `water_overview.toml`, `water_fp_coast.toml`,
  and `water_lod.toml` under `harness/scenes/`.
