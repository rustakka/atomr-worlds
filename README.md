# atomr-worlds

**A general-purpose substrate for 3D reality modeling.** A deterministic,
hierarchically-addressed, streamable sparse-voxel world that scales from a
single brick on a laptop to a sharded multi-node cluster — and a hosting
model that does both behind the same actor protocol.

Games are one obvious application — the engine ships an interactive Bevy
client with five view modes (first-person walk, third-person chase,
horizontal slice, RTS oblique-orthographic, and a regional overview). They
are not the only application. The same primitives — addressable voxel
content, metric LOD, streamed brick subscriptions, world shapes from cubic
to spherical, and a pluggable per-brick `Generator` trait — were chosen to
make the substrate usable for:

- **Simulation and analysis** — terrain processing, hydrological /
  geological / atmospheric modeling, robotics environments, agent-based
  simulations on continuous landscapes.
- **Digital twins / earth-scale environments** — ingesting real DEM,
  satellite, and OSM-style vector layers into the same brick grid that
  procedural content uses. The roadmap below is explicit about this.
- **Interactive visualization** — a working Bevy renderer with five
  view modes, PBR lighting with time-of-day sun and soft cascaded
  shadows, ambient occlusion, sky-tinted exponential fog tied to the
  streaming horizon, a re-baked cubemap skybox of the distant world,
  greedy per-material meshing, custom WGSL palette-voxel and
  sky-dome shaders, and a 4-tier progressive LOD streamer that fills
  ≈ 1 km of terrain around the camera. Every render decision is one
  of nine pluggable `RenderConfig` strategy slots — change the sky,
  the tonemap, the shadow cascades, or the entire shading mode by
  swapping a single trait object. See [Rendering](#rendering) below
  and [docs/RENDERING.md](docs/RENDERING.md) for the strategy spine.
- **Procedural-content R&D** — deterministic seed derivation across the
  Universe → Galaxy → Sector → System → World hierarchy; every brick is
  a pure function of `(world_seed, brick_coord, lod)`. Reproducible
  experiments are the default, not an afterthought.

Everything below `atomr-worlds-host` is GPU-/runtime-agnostic; the CUDA
accelerator and the Bevy client are integrations on top of the substrate
rather than the substrate itself.

## Status

**Phases 0–19 landed, plus the *Advanced Voxel Architectures* foundation
(Bevy 0.18 engine upgrade + physics / SVDAG groundwork).** Phase 0
(primitives), Phase 1 (procedural generators + real `LocalHost` on
atomr's actor system), Phase 2 (CPU renderer: greedy meshing + software
rasterizer to PNG), Phase 3 (persistence: `atomr-persistence` Journal/
SnapshotStore binding, in-memory + optional SQL backends, recovery on
host restart), Phase 4 (streaming subscriptions), Phase 5 (GPU
acceleration: CUDA backend via `atomr-accel-cuda` NVRTC, gated on
byte-for-byte determinism vs the CPU path), Phase 6 (Python bindings),
Phases 7–12 (vehicles + policy + strategy registry, atmosphere + metric
LOD, isosurface meshing, `ClusterHost`, Python release, persistence +
observability hardening), Phase 13 (world shape + horizon streaming +
geologic macro pre-sim + authored-region stipulation + skybox cubemap +
composite renderer + cross-LOD seam fix + transitive skybox), Phase 14
(five world display modes — 1st-person walk, 3rd-person chase,
Dwarf-Fortress horizontal slice, RTS oblique strategy, and large-scale
regional overview — each with its own rendering pipeline and derived
data structure on top of the new `Projection` enum, `WorldQuery` trait,
`raster2d` blitter, and `ViewCache` foundation), Phase 15 (client/
server: Bevy-driven interactive client, headless `atomr-worlds-server`
binary, `atomr-remote`-based `RemoteHost`, and wire-up of
`ClusterHost`'s cross-node forwarder), Phase 16 (PBR lighting + material
upgrade; nine pluggable render-strategy slots), Phase 17 (progressive
4-tier LOD streamer + skybox integration), Phase 17.1 (per-LOD
procedural-brick generation, threading `Lod` end-to-end so coarse-LOD
bricks discretize the same heightfield in world meters instead of
re-using LOD-0 content), and Phase 18 (hydrology overlay — meso-scale
elevation relief plus ocean / lake / river water bodies layered on the
geologic macro pre-sim, with priority-flood basins, drainage-tree river
networks, and local-seed river-channel carving) are all implemented and
tested end-to-end.

**Phase 19** reworked the Dwarf-Fortress slice view (FP-aligned orientation +
hillshade relief) and, separately, landed every algorithm from the *Advanced
Algorithmic Topologies* paper as additive strategy slots on a new
`WorldGenConfig` (3D simplex / domain-warp noise, pluggable brick storage +
codecs, marching-cubes / dual-contouring meshers, a 13-slot layered brick
pipeline, sky-light overlay). **Phase 19.1 / 19.2** moved chunk-plan rebuilds
off-thread and added a horizon-imposter shell + speed-aware visual budgeting.

On top of that, the ***Advanced Voxel Architectures*** roadmap (see
[docs/ADVANCED_VOXEL_ARCHITECTURES.md](docs/ADVANCED_VOXEL_ARCHITECTURES.md))
has landed its **Phase 0** (the Bevy **0.13 → 0.18** engine upgrade) and
**Phase 1 foundations** — a `MaterialPhysicsProps` palette, a new engine-agnostic
`atomr-worlds-physics` crate (flood-fill structural connectivity, mass/inertia,
debris bodies), a deterministic fracture-event protocol, an `HlcTimestamp`, and a
`DagBrick` SVDAG builder with GPU-buffer encoding. These unblock the four
strategic recommendations (GPU raymarching, rigid-body physics, multiplayer
destruction sync, low-latency scheduling) now in progress.

**Rec 1 (GPU DAG raymarching) is finished and now the default render path.**
Each non-empty brick is drawn by GPU-raymarching its sparse-voxel DAG in a
fragment shader instead of uploading a triangle mesh. The DAG is built off the
main thread alongside meshing; a content-digest **buffer cache dedups GPU buffers
+ materials across structurally-identical bricks** (so a mostly-uniform world
collapses to a handful of buffer sets), and the proxy is tightened to each
brick's occupancy AABB to cut overdraw. The WGSL traversal is a line-for-line
mirror of the CPU `gpu_get` / `ray_dda_first_hit` reference, pinned by a
deterministic view-crate render golden. In a debug A/B the raymarcher used
**~19× less GPU memory** than the mesh path. The greedy-mesh path stays fully
supported and one flag away (`--shading mesh` / `RenderPreset::Legacy`); the
release-build A/B (`harness/scenes/perf_raymarch_ab.toml`) is recorded as data,
not a gate.

**First-person voxel editing also landed with Rec 1.** Aim with the camera and
**left-click to carve / right-click to place** — single voxels plus sphere and
cube brushes, with a live tool/material/radius HUD readout. Every edit routes
through the authoritative `WorldActor` (the host stays the only mutator — the
client predicts which bricks changed and re-fetches the authoritative bytes),
and the touched bricks refresh live and flicker-free in **both** render paths.

### Documentation

| Doc | What it covers |
| --- | --- |
| [docs/PHASES.md](docs/PHASES.md) | Per-phase history (0–19, plus the Advanced Voxel Architectures Phase 0 / 20.x) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The system model: hierarchy, seed derivation, sparse storage, metric LOD, hosting |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Module-by-module file/line specifics |
| [docs/LOD.md](docs/LOD.md) | Per-tier streaming + the per-LOD generation contract |
| [docs/PIPELINE.md](docs/PIPELINE.md) | The layered brick-generation pipeline contract (Phase 19) |
| [docs/PHASE_19.md](docs/PHASE_19.md) | Advanced algorithmic topologies reference (Phase 19) |
| [docs/RENDERING.md](docs/RENDERING.md) | The render strategy spine, custom WGSL, offscreen capture |
| [docs/HYDROLOGY.md](docs/HYDROLOGY.md) | Ocean / lake / river water-body overlay (Phase 18) |
| [docs/CLIENT_SERVER.md](docs/CLIENT_SERVER.md) | Client / server / cluster topology |
| [docs/PHYSICS.md](docs/PHYSICS.md) | Voxel-physics foundations + the physics determinism boundary |
| [docs/ADVANCED_VOXEL_ARCHITECTURES.md](docs/ADVANCED_VOXEL_ARCHITECTURES.md) | The 4-recommendation roadmap (SVDAG raymarching, physics, CRDT sync, scheduler) + status |

### Phase progress (summary)

| Phase(s) | Theme | Status |
| -------- | ----- | ------ |
| 0–6 | Substrate: primitives, generators + `LocalHost`, CPU renderer, persistence, streaming, CUDA accel, Python | ✅ |
| 7–12 | Vehicles + policy + strategy registry, atmosphere + metric LOD, isosurface meshing, `ClusterHost`, Python release | ✅ |
| 13 | World shape + horizon streaming + geologic macro pre-sim + authored regions + skybox/composite | ✅ |
| 14 | Five view modes (fp / tp / slice / rts / overview) | ✅ |
| 15 | Client / server / cluster (`RemoteHost`, headless server, Bevy client) | ✅ |
| 16 | PBR lighting + materials; 10-slot render strategy spine | ✅ |
| 17 / 17.1 | Progressive 4-tier LOD streamer + skybox; per-LOD brick generation | ✅ |
| 18 | Hydrology overlay (relief + ocean / lake / river) | ✅ |
| 19 | Slice rework + algorithm-topologies layered pipeline; async plan rebuild + horizon imposter (19.1/19.2) | ✅ |
| AVA Phase 0 | Bevy 0.13 → 0.18 engine upgrade | ✅ |
| AVA Phase 1 | Physics palette + `atomr-worlds-physics` + fracture proto + HLC + `DagBrick` SVDAG | ✅ |
| AVA Rec 1 | GPU DAG raymarcher (**now the default render path**) + off-thread build + cross-brick buffer dedup + occupancy-AABB proxy + CPU determinism golden + **first-person voxel editing** (single-voxel + sphere/cube brushes) | ✅ raymarch default; mesh via `--shading mesh` |
| AVA Rec 2–4 | rigid-body physics · CRDT destruction sync · scheduler | 🟡 foundations landed |

(AVA = *Advanced Voxel Architectures* — see the roadmap doc above.)

## Rendering

The Bevy-based client (`crates/atomr-worlds-client`) is a full
interactive renderer, not a stub. It implements:

### View modes (Phase 14)

| mode       | hotkey | pipeline                                                                                                |
| ---------- | ------ | ------------------------------------------------------------------------------------------------------- |
| `fp`       | `1`    | First-person walk. WASD + mouse-look, sprint, crouch, jump, click-to-grab; double-tap `Space` for collidable creative flight. |
| `tp`       | `2`    | Third-person chase. Orbiting camera anchored to the FP walk position.                                   |
| `slice`    | `3`    | Dwarf-Fortress horizontal slice. Per-column raster from a derived 2D index.                             |
| `rts`      | `4`    | RTS oblique-orthographic. Sub-screen raster with strategic readability.                                 |
| `overview` | `5`    | Regional / body-scale overview. Sphere projection + drag-to-rotate globe.                               |

`Tab` cycles modes; each mode shares the same `WalkCamera` anchor so
moving in one mode persists into the others. The three raster modes
(slice / RTS / overview) reuse the `atomr-worlds-view::derived` 2D
column/slice samplers cached on `(xz, lod)`; the streamer's
`lod_for_meters` picks the LOD per sample, so raster and FP/TP see
the same brick-fetch grid.

### Strategy spine (Phase 16)

Ten `Arc<dyn Trait>` slots on a single `RenderConfig` resource,
each with a default and at least one alternative:

| slot         | trait                | default                  | other impls today                            |
| ------------ | -------------------- | ------------------------ | -------------------------------------------- |
| `mesher`     | `MeshStrategy`       | `GreedyFlat`             | —                                            |
| `palette`    | `PaletteStrategy`    | `HardcodedPalette`       | —                                            |
| `ao`         | `AoStrategy`         | `MinecraftCornerAo`      | `NoAo`                                       |
| `shading`    | `ShadingStrategy`    | `RaymarchDagShading` (GPU DAG raymarch) | `LegacyVertexColor` (mesh; `--shading mesh` / `Legacy` preset), `PaletteVoxelMaterial` (custom WGSL mesh) |
| `sky`        | `SkyStrategy`        | `ProceduralDomeSky` (WGSL) | `ConstantSky`, `SkyTinted`                 |
| `sun_curve`  | `SunCurveStrategy`   | `KeyframeLutSun`         | `StaticSun`                                  |
| `shadow`     | `ShadowStrategy`     | `BasicCascades`          | `NoShadows`                                  |
| `fog`        | `FogStrategy`        | `ExpSquaredSkyTintedFog` | `NoFog`                                      |
| `tonemap`    | `TonemapStrategy`    | `AcesTonemap`            | `DefaultTonemap`                             |
| `coverage`   | `LodCoveragePolicy`  | `NestedSummary`          | `MaskedShells`                               |

`RenderPreset` bundles named looks (`Stylized` / `Legacy` / `Debug`).
The harness DSL exposes `set_time_of_day`, `set_render_preset`, and
`set_strategy` events (the latter takes a `slot` + `strategy` name from
the registry) so scenarios can capture A/B comparisons deterministically.

### GPU DAG raymarcher (AVA Rec 1)

The `RaymarchDagShading` shading mode (**the default**; the mesh path is
`--shading mesh`) draws each brick by GPU-raymarching its sparse-voxel DAG
(`atomr_worlds_voxel::DagBrick::to_gpu`) in a fragment shader
(`assets/shaders/voxel_raymarch.wgsl`) instead of uploading a triangle mesh —
*displacing meshing where it's performant*. Raymarching is now the first-class
default path across LOD tiers, with meshing as the one-flag fallback. What's in
place:

- **Off-thread build** — the DAG (`DagGpuWithDigest`: flat buffers + content
  digest + occupancy AABB) is built on the blocking pool alongside greedy mesh,
  so `spawn_brick_entity` never builds a DAG inline.
- **Cross-brick dedup** — a refcounted `DagBufferCache` keys GPU buffers by the
  DAG content digest and materials by `(digest, tier)`, so structurally-identical
  bricks share one buffer set + material; freed in lockstep with brick eviction.
- **Tight proxy** — the proxy cube and the in-shader DDA are clipped to the
  brick's occupancy AABB so the empty rim is never rasterized or marched (the
  one overdraw mitigation that helps while the shader writes `frag_depth`).
- **Pluggable shading tiers** — `--raymarch-tier unlit|lambert|pbr`, an engine
  setting orthogonal to the strategy slot. `unlit` = flat palette color;
  `lambert` = directional `n·l` over a fixed ambient floor (default); `pbr` =
  Cook-Torrance GGX specular from the palette roughness/metallic, ambient
  occlusion from local DAG occupancy, and a brick-local sun self-shadow, in the
  same normalized-hue / fixed-ambient exposure regime as `lambert`.
- **Determinism gate** — the WGSL traversal mirrors the CPU
  `atomr_worlds_voxel::{gpu_get, ray_dda_first_hit}` reference line-for-line,
  pinned by a deterministic CPU render golden in `atomr-worlds-view`
  (`tests/raymarch_golden.rs`); the GPU float output stays hash-exempt.
- **Perf A/B** — `harness/scenes/perf_raymarch_ab.toml` run twice
  (`--shading mesh` vs `--shading raymarch`) emits `FRAME_DIAG_SUMMARY`
  (p50/p99/max) + `BRICK_MEM` (dedup hit-rate, resident VRAM, acquire cost) for
  the mesh-vs-DAG comparison. Recorded as data; the default flip is committed
  regardless (mesh stays reachable via `--shading mesh`).

### First-person voxel editing (AVA Rec 1)

Aim with the FP camera; **left-click removes** the targeted voxel, **right-click
places** the selected material against the hit face. Single voxels plus sphere /
cube brushes (`Tab` cycles the tool, `[` / `]` size the brush, digit keys pick
the material), with a crosshair, a selection highlight, and a tool/material HUD
readout. Key pieces:

- **World-space picker** — `atomr_worlds_voxel::world_ray_first_solid`, a pure
  Amanatides–Woo DDA over the unbounded 1 m/voxel grid (kept separate from the
  WGSL-mirrored brick DDA — it has no determinism-gate obligation). Samples the
  resident LOD-0 bricks (`LoadedChunk::brick`) with zero host round-trips.
- **Host stays authoritative** — edits send `WriteVoxel` (integer pos) or
  `WriteRegion` (brush) to the `WorldActor`, the only mutator. The client merely
  *predicts* which bricks changed (the same `InteractionUnit::affected_voxels`
  the host uses) and *re-fetches* the authoritative bytes via `GetBrick`; nothing
  render- or DAG-derived flows back into the world state.
- **Flicker-free live refresh** — a make-before-break swap (`spawn_edited_brick`)
  rebuilds the touched bricks via the shared `fetch_and_build` path, so edits
  update instantly in **both** render paths; the `DagBufferCache` dedups an edit
  toward an already-resident shape to zero new buffers.

### Lighting and atmosphere (Phase 16)

- **Time-of-day clock** — `WorldTime` in `[0, 24)` hours feeds a
  keyframe LUT producing a `SunState { direction, color,
  illuminance, day_factor }`. The Bevy `DirectionalLight`'s
  transform, color, and illuminance are updated each tick, along
  with `AmbientLight` color and brightness, the `Skybox`
  brightness, the per-camera `FogSettings`, and the clear color.
- **Soft shadows** — `BasicCascades` `ShadowStrategy` configures
  Bevy's `CascadeShadowConfig` with tuned depth/normal biases. Sun
  pose drives shadow direction so shadows follow the sun across
  the day cycle.
- **Sky-tinted fog** — `ExpSquaredSkyTintedFog` reads the current
  horizon color from the sky strategy and the load-horizon band
  `(start_m, end_m)` from `ChunkStreamer::fog_band_m()`. Far chunks
  streaming into the outermost tier fade in from mist instead of
  popping.
- **PBR materials** — `MaterialPool` produces one
  `StandardMaterial` per palette entry with per-material
  roughness / metallic / emissive / alpha; the alternative
  `PaletteVoxelMaterial` shading mode packs all materials into a
  storage buffer indexed by a per-vertex material id (one draw
  call per brick).
- **AO** — `MinecraftCornerAo` bakes per-vertex ambient occlusion
  factors from the four air-side neighbors of each face corner,
  written into vertex color (Bevy's `ATTRIBUTE_COLOR` interpolates
  bilinearly across the quad).
- **HDR + ACES** — `Tonemapping::AcesFitted` and `Exposure` on the
  camera; HDR is enabled so bloom has headroom.

### Streaming + skybox (Phase 17 + 17.1)

- **Progressive LOD ladder** — `ChunkStreamer` walks the default
  4-tier `LodLadder` (1 / 2 / 4 / 8 m voxels at 128 / 256 / 512 /
  1024 m radii) and emits a closest-first sorted list of
  `(brick_coord, lod)` keys each frame. The load shape is purely
  radial, so the ring is symmetric in all four cardinal directions
  (regression-tested).
- **Per-LOD generation** — `BrickGenContext.lod` and the host's
  `(IVec3, u8)` cache key let each tier discretize the same
  heightfield at its own voxel scale. Adjacent tiers agree on
  surface height in expectation; the only remaining vertical step
  at a tier boundary is voxel/2 discretization. See
  [docs/LOD.md](docs/LOD.md).
- **Async brick-gen pipeline** — `BrickGenWorkers` (`brick_gen.rs`)
  fire-and-forget tokio dispatches per desired brick; each task
  fetches the brick from the host, then `spawn_blocking`s greedy
  mesh + AO bake on the blocking pool. The main thread only drains
  a capped batch of finished payloads each frame to convert into
  Bevy entities, so neither generation nor meshing ever stalls the
  frame loop. `MAX_IN_FLIGHT` and `DEFAULT_SPAWN_BUDGET` cap memory
  / GPU-upload pressure during initial world fill.
- **View-priority sort** — the desired-set is re-sorted so forward-
  facing bricks dispatch first; cached + invalidated by an observer-
  drift / yaw-cone threshold so the per-frame cost stays near zero
  on quiet motion.
- **Cubemap skybox** — `SkyboxRuntime` bakes a six-face cubemap
  from the far-ring meshes and re-bakes when the observer drifts
  past a 5 % threshold; `crossfade_t` ramps `Skybox.brightness`
  between the old and new bakes so the swap is invisible.
- **Hysteresis** — chunks linger two streamer ticks past their
  desired-set boundary before despawn so single-step jitter
  doesn't re-mesh.
- **Nested-summary LOD coverage** — `LodCoveragePolicy` strategy
  decides whether each tier is loaded only as its shell band
  (`MaskedShells`, historical) or as the full inner sphere
  (`NestedSummary`, the default). The default keeps every coarser
  parent brick resident underneath the finer LOD so the FP
  visibility system (`fp_update_lod_visibility`) can crossfade
  between tiers when the camera moves across a boundary —
  `BrickFadeOut` shrinks the child while `BrickFadeIn` blooms the
  parent, eliminating the LOD pop. Memory inflation is bounded:
  the 4-tier ladder grows by ≲ 15 % bricks
  (`harness/scenes/lod_crossfade*.toml` is the visual A/B).

### CPU + headless path (Phase 2 / Phase 13g)

Independent of the Bevy client, `atomr-worlds-view` ships a
deterministic CPU rasterizer (`render.rs`) and an
isometric-perspective composite renderer (`iso.rs`) that produce
PNGs from `LocalHost` bricks with zero GPU dependency. The
`examples/view-png` demo writes an isometric 512×512 PNG of a 4×4×6
brick slab on a headless host; the deterministic-screenshot test
asserts an FNV-1a hash equal across runs. This is what powers the
CI screenshot gate and what enables documentation / batch
visualization without an X display.

### Screenshot harness

`crates/atomr-worlds-client/src/harness.rs` drives the live Bevy
renderer through a TOML scenario (key presses, mouse motion, time-
of-day, render-preset switches) and captures PNGs at named frames.
Output is via an offscreen `Image` target + wgpu readback (sidesteps
a Bevy 0.13.2 `ScreenshotManager` bug on hybrid-GPU Linux). See
[`harness/README.md`](harness/README.md).

## Roadmap

The substrate is the foundation; the next wave of work is about
expanding *what kind of reality* you can put on top of it. Three
threads are explicitly planned:

### Finer-grained LOD

The current ladder is four shells deep — 1 m / 2 m / 4 m / 8 m voxels
at 128 / 256 / 512 / 1024 m radii — which is enough to look right at
ground level but does not scale up or down. Planned:

- **Sub-meter LOD−1** for vehicle-, character-, and interior-detail
  rings: 0.5 m / 0.25 m voxels. Needed for legible UI / cockpit /
  interior scenes inside a larger world.
- **Deeper coarse tiers** for body- and system-scale rendering:
  16 m / 32 m / 64 m voxels at multi-kilometer radii feeding the
  overview mode and the regional skybox bake.
- **Per-view-mode tier counts** — the FP/TP path can afford a tight
  inner shell; an RTS or strategic-overview mode benefits from a much
  flatter ladder. `LodLadder` is already constructable per `Resource`
  but the modes still share one default.
- **Transition meshes** (Transvoxel / boundary stitching) to remove
  the intrinsic ≤ voxel/2 height step where adjacent LOD tiers
  discretize the same continuous surface at different metrics. See
  [docs/LOD.md](docs/LOD.md) for why this step exists and the
  bound on its magnitude.

### Additional generation styles

`BrickGenerator` is a single-method trait that already powers the
default `TerrainGenerator` and the placeholder `EmptyPlanetoid`,
`AsteroidBelt`, and `GasGiant` strategies. Each receives a
`BrickGenContext { world_seed, brick_coord, lod, shape, macro_state,
scale }` and returns a `Brick`. The roadmap adds:

- **Urban / structural generators** — procedural cities, road
  networks, building footprints, road-aware terrain conforming.
- **Biome packs** — biome-driven material / vegetation / hydrology
  layered on top of the existing macro-state biome map (currently
  used by the `LayeredWithFeatures` material strategy for surface
  topsoil; the geometry side is still procedural FBM).
- **Planetary archetypes** — alien-rock, ice-shell, water-world,
  desert. The macro-state path (Phase 13c) already produces the
  geologic / climate / biome surface grid; archetype generators
  read it and emit bricks accordingly.
- **Authored region overlays** — the Phase-13d/13e `AuthoredRegion`
  store (`LiteralRegion`, `HeightmapRegion`, `VoxFileRegion`) is the
  manual-stipulation API. Expanding the loaders to glTF, USD, and
  CityGML is on this thread.
- **Composable strategies** — multi-stage pipelines wired via a small
  DSL or registry change rather than hand-coding `BrickGenContext`
  consumers. The macro pre-sim already chains plates → relief →
  climate → biomes → hydrology (Phase 18); the roadmap item is making
  the *brick-level* stages (terrain → vegetation → structures)
  composable the same way.

### Real-world data feeds

The world-meter sampling API added in Phase 17.1 is the right
integration surface for ingesting external 3D / 2D layers. Planned
data sources, each as its own `BrickGenerator` implementation:

- **Elevation** — SRTM / ASTER / Copernicus DEM tile pyramids; LIDAR
  / DSM where available. The generator's `ctx.lod` selects the right
  pyramid level, so a 30-m DEM serves the depth-3 ring while a 1-m
  LIDAR mosaic populates the depth-0 inner shell.
- **Land cover and vegetation** — ESA WorldCover, NLCD, Sentinel-2
  derived classification feeding the material-selection strategy
  alongside the procedural biome path.
- **Vector overlays** — OSM (roads, buildings, water, landuse),
  Microsoft Building Footprints, USGS hydrography. Vector features
  are rasterized into the brick grid via the same authored-region
  pathway used by `HeightmapRegion` today.
- **Live / time-varying layers** — weather (NEXRAD, GFS) and
  satellite imagery (Sentinel-2 cloud-free composites, Landsat).
  These flow through the streaming subscription protocol the same
  way procedural updates do; the host has no opinion about whether
  a `BrickSnapshot` came from FBM, a DEM tile, or a live feed.
- **Coordinate-system bridge** — a small `geo` adapter mapping
  WGS84 / Web-Mercator / UTM into the engine's metric brick grid,
  so a `LocalHost` can be parameterized by a real-world bounding
  box rather than only a synthetic seed.

The roadmap is intentionally about *plugging into the existing per-LOD
generation contract*, not about adding a parallel pipeline. The
contract (covered in [docs/LOD.md](docs/LOD.md)) is the right shape
for any source of 3D content that can answer "what material is at
world meter `(x, y, z)` at metric scale `2^L`?".

## Workspace layout

```
atomr-worlds/
├── crates/
│   ├── atomr-worlds-core      ─ coordinates, addressing, seed derivation, LOD
│   ├── atomr-worlds-voxel     ─ Brick (16³), arena Octree, SparseVoxelStore trait
│   ├── atomr-worlds-noise     ─ value/gradient/Worley noise + FBM, seeded
│   ├── atomr-worlds-generate  ─ per-tier Generators; CPU TerrainGenerator
│   ├── atomr-worlds-accel     ─ Accelerator trait, CPU backend, CUDA backend (feature = "cuda")
│   ├── atomr-worlds-physics   ─ engine-agnostic voxel physics: flood-fill connectivity, mass/inertia, debris bodies
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
`Space` to jump, `Shift` to sprint, `C` to crouch. **Double-tap `Space` toggles
creative flight** (collision still enforced — you won't clip through terrain):
while flying, `Space` ascends, `Left Ctrl` descends, `Shift` flies faster;
double-tap `Space` again to drop out. `1..=5` picks a view mode (`fp` / `tp` /
`slice` / `rts` / `overview`), `Tab` cycles. Slice/RTS/overview have per-mode
hotkeys — see [docs/CLIENT_SERVER.md](docs/CLIENT_SERVER.md).

### Screenshot test harness

The client ships with a TOML-driven scenario harness for capturing PNGs of
the live Bevy renderer (`crates/atomr-worlds-client/src/harness.rs`):

```sh
./scripts/run-harness.sh harness/scenes/fp_lookup.toml /tmp/shots/
# or
./target/release/atomr-worlds-client \
    --harness harness/scenes/fp_lookup.toml \
    --harness-out /tmp/shots/
```

The scenario can synthesise key presses, mouse motion, and screenshots at
named frames; the binary prints `HARNESS_SHOT <path>` to stdout for each
captured frame and exits cleanly. The capture path shells out to `xwd`
(from `x11-apps`) and parses XWD in-process — it sidesteps a Bevy 0.13.2
ScreenshotManager bug on hybrid-GPU Linux. See [`harness/README.md`](harness/README.md)
for the schema and authoring guide.

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

This is the **foundation layer** for a 3D reality model — synthetic or
grounded in real data. It provides the address space, the hash-based
hierarchy of seeds, the data structures for sparse voxel content at
multiple scales, the wire/host protocol downstream code routes through,
CPU + CUDA brick generation, a streaming host with durable write replay,
a deterministic CPU rasterizer, a Bevy-based interactive client with
five view modes and a nine-slot pluggable render-strategy spine, and
Python bindings.

It is **not yet** a finished application of any kind — game, GIS
viewer, digital twin, simulator. The Bevy client is a complete and
working visualization of the substrate, but the application-shaped
work above it (mission design, persistence policy, UI for end users)
is still your job.

Pieces deliberately left out at the substrate level: variable-depth
hierarchies, cross-dimension portals / passivation rules, multi-galaxy
load-balancing policy, cluster subscription routing (one-shot requests
forward cross-node; subscriptions stay node-local), gossip-based
cluster membership, transport TLS, the real-Earth data-feed ingestion
described in the Roadmap above, transition meshes between LOD tiers
(adjacent tiers currently disagree by ≤ voxel/2 — see
[docs/LOD.md](docs/LOD.md)), and a PyPI release. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design principles
and [docs/CLIENT_SERVER.md](docs/CLIENT_SERVER.md) for the topology
and known gaps.

## License

Apache-2.0. See [LICENSE](LICENSE).
