# Rendering

How `atomr-worlds-client` turns bricks into pixels. For the broader system
model see [ARCHITECTURE.md](ARCHITECTURE.md); for module/file pointers see
[IMPLEMENTATION.md](IMPLEMENTATION.md).

## Why this exists separately

Phase 16 added a real PBR look (multiple materials, time-of-day sun, soft
shadows, AO, sky-tinted fog) to the FP/TP modes. Every meaningful decision
in that pipeline is a trait with at least one default impl, gathered into a
single resource. This document records:

1. The strategy spine and the nine pluggable slots.
2. Lessons learned validating the upgrade on FP, and how they apply (or
   don't) to TP, slice, RTS, overview.
3. Methodologies the work produced — patterns that should be reused
   anywhere a future render decision needs to be experiment-friendly.

## Strategy spine

The center of gravity is
[`crates/atomr-worlds-client/src/render/`](../crates/atomr-worlds-client/src/render/).
[`RenderConfig`](../crates/atomr-worlds-client/src/render/config.rs) is a
Bevy resource with nine `Arc<dyn Trait>` fields, one per decision:

| slot       | trait                | default               | other impls today                                       |
| ---------- | -------------------- | --------------------- | ------------------------------------------------------- |
| `mesher`   | `MeshStrategy`       | `GreedyFlat`          | —                                                       |
| `palette`  | `PaletteStrategy`    | `HardcodedPalette`    | —                                                       |
| `ao`       | `AoStrategy`         | `MinecraftCornerAo`   | `NoAo`                                                  |
| `shading`  | `ShadingStrategy`    | `RaymarchDagShading` (GPU DAG raymarch — AVA Rec 1) | `LegacyVertexColor` (mesh; `--shading mesh` / `Legacy` preset), `PaletteVoxelMaterial` (custom WGSL — Step 8) |
| `sky`      | `SkyStrategy`        | `SkyTinted`           | `ConstantSky`, `ProceduralDomeSky` (dome shader — Step 9) |
| `sun_curve`| `SunCurveStrategy`   | `KeyframeLutSun`      | `StaticSun`                                             |
| `shadow`   | `ShadowStrategy`     | `BasicCascades`       | `NoShadows`                                             |
| `fog`      | `FogStrategy`        | `ExpSquaredSkyTintedFog` | `NoFog`                                              |
| `tonemap`  | `TonemapStrategy`    | `AcesTonemap`         | `DefaultTonemap`                                        |

Each trait is one or two methods, `Send + Sync + 'static`, and always
returns plain data (never touches `World` directly). The plugin reads
`RenderConfig` once at startup and re-reads it every frame for the
time-driven slots (sun, sky, fog).

### Presets

[`RenderPreset`](../crates/atomr-worlds-client/src/render/config.rs)
bundles a whole look behind one name. The `apply_preset` method assigns
all nine slots explicitly so rolling back is total:

- `Stylized` — shipped defaults (the table above).
- `Legacy` — pre-Phase-16 baseline: `GreedyFlat`, `NoAo`,
  `LegacyVertexColor`, `ConstantSky`, `StaticSun`, `NoShadows`, `NoFog`,
  `DefaultTonemap`. Used to A/B-compare with the upgrade.
- `Debug` — flat-shading mode for inspecting raw geometry: `NoAo`,
  `StaticSun`, `NoShadows`, `NoFog`, `ConstantSky`, `DefaultTonemap`.

### Time-of-day clock

[`WorldTime(pub f32)`](../crates/atomr-worlds-client/src/render/sun.rs)
is hours-of-day in `[0, 24)`. Two systems run in `Update`, chained:

1. `advance_world_time` — optional auto-advance, gated on
   `RenderConfig::time_advances_automatically` (default off; the harness
   sets the clock directly).
2. `sync_sun` — reads `cfg.sun_curve.sun_state(world_time.0)`, writes
   the `DirectionalLight` carrying [`WorldSunMarker`] transform / color /
   illuminance, plus `AmbientLight` color and brightness.
3. `sync_sky_and_fog` — reads the same `SunState` plus the sky strategy,
   writes `ClearColor` and per-camera `FogSettings.color`/`falloff`.

The strategy returns plain `SunState { direction, color, illuminance,
day_factor }`; the systems do the Bevy wiring. Tests can construct any
combo by hand without booting Bevy.

### Harness DSL

`harness.rs` exposes three new `ScenarioEvent` kinds:

- `set_time_of_day { hours: f32 }` — writes `WorldTime`.
- `set_render_preset { preset: "stylized"|"legacy"|"debug" }` — applies
  a preset.
- `set_strategy { slot: "...", strategy: "..." }` — switches one slot
  at runtime via the
  [`registry::apply_strategy_by_name`](../crates/atomr-worlds-client/src/render/registry.rs)
  table. A scenario can A/B compare strategies without recompiling.

Scenarios that exercise the pipeline live at
`harness/scenes/lighting_showcase.toml` (six time-of-day shots) and
`harness/scenes/strategy_compare.toml` (preset and per-slot A/B).

## What's enabled per mode

| mode       | path                                        | sees lighting upgrade? |
| ---------- | ------------------------------------------- | ---------------------- |
| FP         | Bevy 3D, `PbrBundle` per material per brick | Full                   |
| TP         | Bevy 3D, shares FP scene                    | Full (inherited)       |
| Slice      | CPU rasterizer → `Framebuffer` → Bevy Image | Palette colors only    |
| RTS        | CPU rasterizer → `Framebuffer` → Bevy Image | Palette colors only    |
| Overview   | CPU rasterizer → `Framebuffer` → Bevy Image | Palette colors only    |

FP and TP both go through Bevy's PBR pipeline and consume every slot.
Slice/RTS/overview render through `atomr-worlds-view`'s software
rasterizer and only pick up the palette through
[`material_color()`](../crates/atomr-worlds-view/src/render.rs) — they
get the 10-material color set automatically but do not see the sun,
shadows, fog, AO, tonemap, or bloom. Bringing the new lighting into the
software path is a future-work item; see [Cross-mode lessons](#cross-mode-lessons).

## FP-mode lessons learned

The FP path was the first to consume the strategy spine end-to-end. The
points below are recorded because each one cost time to diagnose and each
will recur if not flagged.

### 1. Bevy 0.13 `AmbientLight.brightness` is on a 0..100 scale

The default is 80.0; we had it at 1.2 (a leftover from an early
"brightness fix") and the scene was ~67× too dim. Symptom: shadowed faces
looked black even at noon. Fix: read the Bevy source for the default,
match its scale. The sun-curve strategy returns a normalized `[0, 0.5]`
ambient value that `sync_sun` multiplies by 200 before writing
`AmbientLight.brightness`.

**Generalises to:** any Bevy engine value where the units aren't named on
the type. When in doubt, grep `bevy_pbr` / `bevy_render` for the default
literal — that's authoritative.

### 2. X11 + hybrid-GPU swapchain readback is unreliable

`xwd` against a Vulkan-rendering window on a hybrid-GPU (NVIDIA + Intel)
laptop yields an all-black PNG. Bevy 0.13.2's `ScreenshotManager` panics
on the async buffer-map path in this configuration. WGPU_BACKEND=gl,
LIBGL_ALWAYS_SOFTWARE, and `__NV_PRIME_RENDER_OFFLOAD` all reproduced
black output.

Fix:
[`OffscreenCapturePlugin`](../crates/atomr-worlds-client/src/render/offscreen.rs)
points the camera at an `Image` render target, copies the GPU texture
into a `MAP_READ` buffer with `copy_texture_to_buffer` inside
`RenderSet::Cleanup`, polls the device synchronously, strips the per-row
256-byte padding, swaps BGRA → RGBA, and saves PNG. The result completely
bypasses the swapchain.

**Generalises to:** any time CI or hardware exposes a presentation-path
quirk. Render to an offscreen image, read back through wgpu directly,
and present is now optional — the headless examples already used this
pattern (see [`examples/view-png`](../examples/view-png)). Recording
this approach as the project's default for visual regression captures
saves the next person the same week of debugging.

### 3. Strategy `Default::default()` and preset rollback can disagree

The original `Legacy` preset called `RenderConfig::default()` and then
overrode fields, but `RenderConfig::default()` had quietly upgraded as
new defaults landed — so `Legacy` no longer matched the pre-Phase-16
baseline. Symptom: A/B compare against `legacy` showed a half-upgraded
scene.

Fix: presets assign every relevant slot explicitly rather than relying on
`Default`. The preset enum is the source of truth for "what does this
look like"; `RenderConfig::default()` is just for booting.

**Generalises to:** any time `Default` is a moving target. If a config
value participates in regression testing (A/B comparisons, golden
images), pin every field explicitly at the call site, not via `Default`.

### 4. Vertex-attribute shape changes propagate everywhere

Adding `ao: f32` to `atomr-worlds-view::Vertex` broke five test files and
three rasterizer call sites (`iso.rs`, `derived/surface_raster.rs`,
`skybox.rs`, and two test fixtures). Each required `, ao: 1.0` literals
threaded through.

**Generalises to:** any struct field added to a type that's constructed
by literal across multiple crates. Two ways to make this less painful:

- Add a constructor (`Vertex::new(pos, normal, material)` that defaults
  `ao = 1.0`) before adding new fields. Future fields go through the
  constructor and don't break call sites.
- Or, for one-off ergonomics, `impl Default for Vertex` plus the
  `..Default::default()` literal form.

Phase 16 didn't take either route (the field was already widely
constructed and migrating in one pass was cheap), but the next vertex
attribute should land behind a constructor.

### 5. The harness must lead the client

The plan-stated rule from `feedback_harness_on_client_changes.md`
applied here: every new visual capability landed alongside a harness
event (`set_time_of_day`, `set_render_preset`, `set_strategy`) and a
scenario that exercises it. The win is concrete — `lighting_showcase`
caught the AmbientLight scale bug in one screenshot, and
`strategy_compare` confirmed that switching presets actually reaches
every slot.

**Generalises to:** the rendering harness should land alongside any
future visual capability — including non-client modes. The CPU
rasterizer's existing golden-PNG tests (`tests/{slice,rts,overview}_golden.rs`)
already do this for the software path; the same pattern should extend
when those modes pick up new state (e.g. when slice gets a directional
light cue from the world clock).

## Cross-mode lessons

Each lesson above maps onto the other view modes with adjustments.

### TP (third-person chase)

TP shares the FP scene and `WorldSunMarker` light; everything in
`RenderConfig` is inherited. The only thing TP-specific is the orbit
camera. **What carries forward:**

- The harness DSL already drives FP and TP from the same TOML; just set
  `mode = "tp"`. Same scenarios apply.
- The offscreen capture works identically (it's keyed on the active
  camera, not the mode).

**What's still missing:** there is no `tp` variant of
`lighting_showcase.toml` today. Cheap to add — same events, different
header.

### Slice, RTS, Overview (software path)

These render through
[`atomr-worlds-view::render*`](../crates/atomr-worlds-view/src/render.rs)
into a CPU `Framebuffer` and blit into a Bevy `Image`. They consume the
material palette (because `material_color()` was updated alongside the
new ids) but do not see lighting, shadows, fog, AO, or tonemap.

Three methodologies port over:

1. **Strategy spine.** The same pattern (a `RenderConfig`-style
   resource with `Arc<dyn Trait>` slots) is the right shape for the
   software path's decisions. Likely slots: `LambertTermStrategy` (flat
   vs. directional vs. ramp shading), `AmbientStrategy` (flat tint vs.
   sky-direction-driven), `SkyColorStrategy` (constant vs. time-of-day
   LUT), `EdgeStrategy` (none vs. screen-space outline vs. depth-based
   cavity). Each can reuse the same `SunState` from the sun-curve trait;
   they just consume it from a CPU-side function instead of the GPU.

2. **A/B compare via the harness.** The CPU rasterizer's golden tests
   already capture a pixel hash; the harness DSL can drive the same
   slice/RTS/overview scenarios with `set_strategy` once those slots
   exist. The mechanism is identical to FP — only the strategy library
   changes.

3. **Offscreen capture.** Slice/RTS/overview don't need it (the
   `Framebuffer` is already CPU-side; saving a PNG from there is two
   lines of `image` crate). But the methodology is the same: render to
   a deterministic buffer, save from CPU, do not depend on the
   swapchain.

The non-portable bits are the lighting that depends on Bevy PBR
(shadow maps, fog blending in screen space, bloom in HDR). Those would
need a software analogue (e.g. raymarched shadow term across the
heightfield for RTS, or a top-down LUT for slice). Treat them as
future strategies — the slot exists either way.

### Overview-specific (cosmic scale)

The overview mode operates at galaxy/sector/system scale where the
distances are too large for a single `f32` view matrix. The Phase-16
lighting work doesn't reach this mode at all; the strategy that
matters here is `WorldMacroState` (Phase 13c), not the sun curve. The
methodology that *does* port over is the registry pattern: macro-state
sources should be selectable by name from the harness, the same way
strategy slots are.

## Methodologies (reusable patterns)

The work produced four patterns worth applying anywhere a future
render decision needs to stay experiment-friendly.

### M1. Strategy resource with `Arc<dyn Trait>` slots

A single resource bundles all decisions; each is a small trait
(one or two methods); each has a `name()` method so the harness can
log which is active. Test code constructs concrete impls directly and
never touches the resource. Adding a new strategy is one type + one
table entry in `registry.rs` + one default in `defaults.rs`. Adding a
new decision point is one trait + one field on the resource.

### M2. Preset enum that assigns every slot

Presets exist for atomic look-swaps (`Stylized`, `Legacy`, `Debug`).
The `apply_preset` method writes every slot explicitly — never builds
on `Default::default()`. This is what makes preset-based A/B regression
testing actually predictable.

### M3. Offscreen-image rendering for visual capture

Render to an `Image` asset (`TextureUsages::COPY_SRC |
RENDER_ATTACHMENT`), copy to a `MAP_READ` buffer in a `RenderApp`
system at `RenderSet::Cleanup`, poll the device synchronously,
strip per-row 256-byte padding, swap BGRA → RGBA, save PNG. Works on
hybrid-GPU laptops, on CI without an X server, and produces
byte-deterministic output. See
[`render/offscreen.rs`](../crates/atomr-worlds-client/src/render/offscreen.rs)
and the memory note
[`memory/project_harness_offscreen_capture.md`](../../.claude/projects/-home-mattbragaw-source-atomr-worlds/memory/project_harness_offscreen_capture.md).

### M4. Harness DSL parity with new capability

Each new capability in the client lands with a corresponding
`ScenarioEvent` kind and at least one TOML scenario that exercises it.
The harness picks up regressions visually; the scenario doubles as a
"how do I see this feature?" demo. The cost is one event variant plus
one scenario file per capability; the win is fast visual feedback in
the iteration loop. See
[`feedback_harness_on_client_changes.md`](../../.claude/projects/-home-mattbragaw-source-atomr-worlds/memory/feedback_harness_on_client_changes.md).

## Custom shader strategies (Steps 8 + 9)

Two opt-in shading + sky impls landed alongside the spine. Both ship as
strategies — opt in by writing the slot from a harness `set_strategy`
event or by hand. (Note: the **default `shading` slot is now
`RaymarchDagShading`**, the GPU DAG raymarcher — see "GPU DAG raymarcher
(AVA Rec 1)" in the README; the `StandardMaterial` mesh path
`LegacyVertexColor` is reachable via `--shading mesh` / the `Legacy`
preset, and `PaletteVoxelMaterial` below is the custom-WGSL mesh path.)
The deterministic-PNG gates live in the **view crate's** CPU renderer,
which is independent of the client shading slot, so the default flip
doesn't move any golden.

### Step 8 — `PaletteVoxelMaterial`

[`render/materials.rs`](../crates/atomr-worlds-client/src/render/materials.rs)
+ [`assets/shaders/voxel_material.wgsl`](../crates/atomr-worlds-client/assets/shaders/voxel_material.wgsl).

- **Type**: `VoxelMaterial = ExtendedMaterial<StandardMaterial, VoxelMaterialExt>`.
  `VoxelMaterialExt` carries a `Vec<PaletteEntryGpu>` storage buffer at
  binding `100` (the convention is to start custom extension bindings
  at >= 100 — slots 0–99 are reserved by `StandardMaterial`).
- **Vertex encoding**: per-vertex material id goes in `ATTRIBUTE_UV_0.x`,
  AO in `ATTRIBUTE_COLOR.r`. Both attributes are already in Bevy's
  default mesh vertex layout, so no `specialize()` is needed for the
  attribute layout.
- **WGSL**: imports `bevy_pbr::pbr_fragment::pbr_input_from_standard_material`
  and `bevy_pbr::pbr_functions::apply_pbr_lighting`, looks up
  `palette[mat_id]`, overrides `base_color` / `perceptual_roughness` /
  `metallic` / `emissive` on the `PbrInput`, multiplies AO into base
  color, calls `apply_pbr_lighting`. Shadows / fog / tonemap / bloom
  remain free because the standard PBR pipeline runs on the modified
  input.
- **Mesh wiring**: `fp_stream_bricks` checks
  `RenderConfig::shading.mode()` and branches between
  `ShadingMode::SplitPerMaterial` (N child `PbrBundle`s) and
  `ShadingMode::PaletteVoxelMaterial` (one `MaterialMeshBundle<VoxelMaterial>`).
  In the latter, all per-material submeshes from
  `greedy_mesh_by_material` are merged through `merge_by_material`
  (deterministic — submeshes sorted by id before merging) and the
  voxel material handle is shared across every brick.
- **Effect**: fewer draw calls per brick. Visually equivalent to the
  split path when the same palette feeds both — `voxel_material.toml`
  scenario captures both at the same camera pose for A/B confirmation.

### Step 9 — `ProceduralDomeSky`

[`render/sky_dome.rs`](../crates/atomr-worlds-client/src/render/sky_dome.rs)
+ [`render/materials.rs::SkyDomeMaterial`](../crates/atomr-worlds-client/src/render/materials.rs)
+ [`assets/shaders/sky_dome.wgsl`](../crates/atomr-worlds-client/assets/shaders/sky_dome.wgsl).

- **Geometry**: an inside-out sphere (radius 800 m, 32×16 UV sphere)
  parented to the camera so the dome tracks the observer. The custom
  `Material` impl's `specialize()` flips `descriptor.primitive.cull_mode`
  to `Some(Face::Front)` so the back faces are what the camera sees
  from inside. `NotShadowCaster` + `NotShadowReceiver` +
  `NoFrustumCulling` keep the sphere out of the shadow path and stop
  it being culled when the camera looks away from the origin.
- **WGSL**: `view.world_position - in.world_position` gives the
  world-space ray direction; mix `zenith_color` → `horizon_color`
  weighted by `pow(1 - dir.y, 4)`; add `sun_color * (sun_disc + glow)`
  where `sun_disc = smoothstep(0.9994, 0.9998, cos_theta)` and
  `glow = pow(cos_theta, 96.0) * 0.6` (the same cone shape Bevy's
  builtin Skybox uses).
- **Activation**: `SkyStrategy::dome_active()` returns true for
  `ProceduralDomeSky`. `sync_sky_dome` reads that each frame, flips
  the dome entity's `Visibility`, and writes the four material
  uniforms (horizon/zenith/sun colors and sun direction) from the
  current `SunState`. The dome material is always registered via
  `MaterialPlugin::<SkyDomeMaterial>::default()` so toggling is a
  no-restart change.
- **Effect**: a real gradient sky + soft sun disc visible behind
  terrain. `ClearColor` + `FogSettings` continue to follow the
  horizon color so the dome's edges blend smoothly into the existing
  fog look.

### 6. Verify winding handedness across *all six* face directions

The greedy mesher in
[`atomr-worlds-view::mesh`](../crates/atomr-worlds-view/src/mesh.rs)
originally produced ±Y faces whose CCW winding gave a geometric normal
of **−Y**, opposite the stored normal **+Y**. Under Bevy's default
`Cull::Back`, every top + bottom face of every voxel was a back face
and got culled. The bug was invisible at dawn/dusk (camera sees long
side faces) and presented as "the noon scene is just dark." It went
undetected through the whole Phase-16 work because the failing test
condition — *front face geometric normal matches stored normal* —
was never asserted.

**Cause**: `(u_axis, v_axis)` was `(0, 2)` for axis=1, giving
`X × Z = −Y`. The X and Z faces used `(1, 2)` and `(0, 1)`, whose
crosses do match their respective stored normals — those rendered fine.

**Fix** (one line): change axis=1's `(u_axis, v_axis)` to `(2, 0)` so
`Z × X = +Y`.

**Regression test**:
[`mesh::tests::all_six_face_directions_wind_outward`](../crates/atomr-worlds-view/src/mesh.rs)
constructs a single-voxel brick, finds the first triangle for each of
the 6 stored normals, computes its CCW geometric normal, and asserts
`dot(geometric, stored) > 0`. Catches handedness drift on *any* axis
in one assertion.

**Generalises to**: any axis-aligned mesh emitter (greedy, marching
cubes, surface nets, even a hand-built skybox). Always test the
handedness of every face direction the emitter can produce — back-face
culling makes wrong-handedness completely invisible until a camera
looks at the affected face from "outside". A `dot(cross, stored) > 0`
test takes one minute to write and saves weeks of "why is my noon
scene dark".

### Cross-cutting lessons from Steps 8 + 9

The two shader steps added one more cluster of lessons worth recording:

1. **Bevy `AssetPlugin::file_path` is relative to the binary's
   `current_exe().parent()`, not the working directory.** Putting
   shaders under `crates/<crate>/assets/shaders/` and passing a
   `"crates/<crate>/assets"` relative path silently looks under
   `target/release/crates/...` and 404s. Fix: resolve the asset
   directory to an absolute path at startup (see
   [`main.rs::resolve_asset_root`](../crates/atomr-worlds-client/src/main.rs)).
   The same hazard hits any future asset (textures, additional
   shaders); the resolver covers them all.

2. **In Bevy 0.13, `Material::cull_mode` is not a trait method.**
   The convention is to flip `descriptor.primitive.cull_mode` inside
   `Material::specialize`. For `ExtendedMaterial` the same applies
   via `MaterialExtension::specialize`.

3. **`AsBindGroup` requires `WriteInto + ShaderSize` on contained
   structs.** A storage-buffer entry needs `#[derive(ShaderType)]`
   (from `bevy::render::render_resource::ShaderType`) — not just
   `Reflect`. Reflecting an extension that contains a non-Reflect
   `ShaderType` struct fails the derive; the fix is to drop `Reflect`
   from the extension (it's optional for `MaterialExtension`).

4. **Debug-magenta is the fastest diagnostic when a shader silently
   produces "no visible difference".** Replacing the fragment output
   with a hard-coded sentinel color confirms in one rebuild whether
   the dome is actually rendering vs. producing colors that happen to
   match the fog blend. Generalise: when adding a new render path
   that should be visually obvious, write a one-line debug variant
   first and keep it as a commented fallback.

## Mesh optimization opportunities

Each item below is an inspection finding — nothing critical, all
optional. Listed in roughly descending payoff-per-effort order.

### O1. Vertex deduplication in `greedy_mesh_by_material`

[`mesh.rs:80-101`](../crates/atomr-worlds-view/src/mesh.rs) — the
bucket pass pushes 3 vertices per triangle. Each greedy quad (2
triangles, 4 unique verts) currently lands in its bucket as 6 vertex
entries with 3 redundant duplicates.

**Fix**: maintain a small per-quad index map inside the loop (or
key on `(material, position_quantized)`) and reuse vertex indices for
the second triangle of every quad. Per brick saves ~33% of vertex
storage in the `SplitPerMaterial` path.

**Effort**: ~30 lines in `greedy_mesh_by_material`. No API change.

**Payoff**: smaller GPU vertex buffers, fewer cache misses in the
vertex shader. Estimate ~30% draw-call time reduction on
vertex-shader-bound bricks (rare at our brick sizes, but free win).

### O2. Inline AO bake inside `emit_quad`

[`mesh.rs::bake_ao`](../crates/atomr-worlds-view/src/mesh.rs) runs as
a separate post-pass that iterates every vertex and re-derives the
face axis from the stored normal. Folding the AO sampling into
`emit_quad` (where axis/positive/u_axis/v_axis are already known)
removes one full mesh traversal per brick + the floating-point sign
test for each vertex.

**Effort**: move the 4-neighbor sample logic from
`compute_vertex_ao` into `emit_quad`, gated on a `&dyn AoStrategy`
passed in. The `NoAo` strategy stays a no-op (`enabled() → false`
skips the sample).

**Payoff**: noticeable at mass-streaming time when 343 bricks
(7³) are meshed during the FP warmup. Probably 5–10% reduction in
total meshing time.

### O3. Cross-brick AO sampling (correctness, not perf)

Today [`material_at`](../crates/atomr-worlds-view/src/mesh.rs)
returns `0` for OOB coordinates, so vertices on the −X / −Y / −Z faces
of a brick get false darkening from "nonexistent neighbors". The
plan flagged this with a TODO; the user notices it as visible seams
between bricks at brick-aligned voxels.

**Fix**: extend the meshing API to take an optional
`neighbors: &[Option<&Brick>; 6]` (one per face direction) and have
`material_at` route OOB reads to the appropriate neighbor brick.
The client already has the neighbor bricks loaded — it just doesn't
pass them in.

**Effort**: ~80 lines split between `mesh.rs` and `fp.rs`.

**Payoff**: visible quality improvement at every brick boundary.
Not a perf win.

### O4. Greedy merge keyed by `(material, ao_4tuple)`

Today greedy merging only keys on `material`, so an adjacent pair of
voxels with different AO 4-tuples on the same face still merge into
one quad whose AO is then bilinearly interpolated across the merged
extent. The interpolation produces gradients where you'd want
stepped AO at the merge boundary.

**Fix**: compute AO per-cell *before* the greedy merge sweep, then
key the mask on `(material, ao_4tuple_packed_into_u32)`. Effectively
makes AO part of the "is this cell mergeable" predicate.

**Effort**: ~50 lines in `meshing_axis`. Disables some merging in
heavily-AO'd areas; vertex count rises slightly.

**Payoff**: sharper-looking corners at concave geometry. The
existing bilinear AO is already passable, so this is polish.

### O5. Switch indices from `u32` → `u16` where it fits

A brick at most produces `BRICK_EDGE³ × 6 faces × 4 verts = 98 304`
worst-case verts, which would overflow `u16` (65 535). But a typical
brick mesh ships < 4000 verts, fitting easily.

**Fix**: try `Indices::U16` first; fall back to `U32` if the vertex
count overflows. The bevy `Mesh` API supports both.

**Effort**: ~20 lines + a length check.

**Payoff**: 50% cut on index-buffer GPU memory. Usually
negligible in absolute terms but free.

### O6. Step-8 vertex attribute encoding

The `PaletteVoxelMaterial` path stores the material id in
`ATTRIBUTE_UV_0.x` as `f32`, then `u32(uv.x + 0.5)` recovers it in
WGSL. Two non-issues today (palettes have < 256 entries, `f32` is
exact through 16-million), but a custom `MeshVertexAttribute::new(...
VertexFormat::Uint32)` slot would be more honest and shaves 4 bytes
per vertex (UV_0 is `Float32x2`, the custom slot would be `Uint32`).

**Effort**: requires `MaterialExtension::specialize` to extend the
vertex layout, plus matching `@location(n)` slot in
`voxel_material.wgsl`. ~40 lines.

**Payoff**: small. Mainly cleanliness — the encoding is unprincipled
today.

### O7. Mesh-asset deduplication across identical bricks

Bricks that are entirely uniform (all stone, all air outside the
streaming radius) produce identical meshes — currently each gets its
own `Mesh` asset. A content-hash cache would dedupe these.

**Effort**: medium. Requires keying meshes by FNV-1a of vertex+index
data, with eviction policy.

**Payoff**: low at typical content density. Worth doing if/when
`STREAM_RADIUS_BRICKS` grows past 4–5.

### Out of scope

- Compressed mesh storage on disk — future work.

## LOD streaming + skybox integration (Phase 17)

Phase 17 wires three existing-but-unused capabilities
(`atomr-worlds-proto::StreamingPolicy`, the cubemap `Skybox`,
`ObserverState`) into the Bevy client and the raster modes. Three
resources land:

- **`ChunkStreamer`** (`crates/atomr-worlds-client/src/world_stream.rs`)
  — owns a [`LodLadder`](#progressive-lod-ladder) plus a cached
  2-tier `StreamingPolicy` projection for legacy callers (proto,
  host, skybox bake). Consumers call
  `desired_chunks(streamer, observer, horizon_m)` to get a
  closest-first list of `(IVec3, Lod)` keys — one entry per tier per
  brick inside its radial shell.
- **`LoadedChunks`** — `HashMap<(IVec3, u8), LoadedChunk>`. FP/TP
  store the spawned `Entity`; raster modes (slice/RTS/overview) use
  the streamer only as a *LOD oracle* via
  `streamer.lod_for_meters(observer, p)` and don't populate
  `LoadedChunks`. Eviction is hysteresis-gated (2 streamer ticks past
  "last seen in the desired set") so a single boundary-jitter step
  doesn't re-mesh.
- **`SkyboxRuntime`** (`crates/atomr-worlds-client/src/render/skybox.rs`)
  — wraps `ObserverState` plus the cubemap `Image` handle. Each tick:
  tick the observer with the walk position, check `should_refresh`,
  bake from the outermost-tier ring of `LoadedChunks`, hot-swap the
  `Skybox.image` through a brightness crossfade.

### Progressive LOD ladder

The streamer expresses its load shape as a `LodLadder`: an ordered
list of `LodTier { lod, outer_radius_m }` rungs. Each rung owns the
spherical shell between the previous tier's outer radius and its own.
A brick is emitted at tier `i` iff its **center** falls inside that
shell — the test is purely radial (3D distance), so the load shape
is symmetric across X / Y / Z and rotationally invariant around the
observer's vertical axis. (The previous 2-tier *cube* ring stretched
~73 % further along its corners than its faces; walking diagonally
felt like only two of the four cardinal directions actually loaded
new terrain. The radial check is what fixes that.)

The default ladder:

| tier | LOD | voxel edge | shell band (m)  |
|------|-----|------------|-----------------|
| 0    | L0  | 1 m        | `[0, 128)`      |
| 1    | L1  | 2 m        | `[128, 256)`    |
| 2    | L2  | 4 m        | `[256, 512)`    |
| 3    | L3  | 8 m        | `[512, 1024)`   |

Radii are aligned to multiples of the coarsest brick edge
(`BRICK_EDGE × 2^3 = 128 m`), which makes the brick grids at every LOD
tile cleanly across tier boundaries — no gaps, no double-rendered
overlap. The closest-first sort across all tiers means the sharp inner
ring fills first, then each subsequent shell fades in toward the
horizon. `bricks_per_tick = 128` sizes the budget so the inner sphere
populates in ≈ 1 second at 60 fps, with the outer tiers continuing to
back-fill while the user is already exploring.

`WorldQuery::brick(addr, coord, lod)` returns a 16³ brick at *that*
depth — each voxel is `BRICK_EDGE * 2^lod.depth` meters wide. The
mesh stays in `0..BRICK_EDGE` local coords; for far-LOD bricks the
client's `SpatialBundle` carries `scale = 2^L`. No vertex mutation,
no separate mesher.

The raster modes consult the same streamer when picking the LOD they
pass to `WorldQuery::brick`: per-column for slice/RTS, always the
outermost-tier LOD for overview (because its per-pixel viewing
distance is body-scale). `lod_for_meters` walks the ladder so raster
LOD selection lines up bit-for-bit with the FP/TP brick-fetch grids.

### Horizon fog (atmospheric perspective across every tier)

`ChunkStreamer::fog_band_m()` returns `(start, end)` derived from the
outermost tier's radius (defaults: 55 % and 98 % of the load horizon
⇒ 563 m / 1003 m for the default ladder). `sync_sky_and_fog` reads the
band each frame and passes it to the `FogStrategy`. The default
`ExpSquaredSkyTintedFog` uses **exponential-squared** falloff with a
density auto-tuned so transmittance reaches ≈ 5 % exactly at the load
horizon (`density = sqrt(-ln 0.05) / band.end`). Because exp² is
smooth from zero, every closer LOD tier picks up atmospheric
perspective: near voxels stay sharp, mid-distance LOD-1 / LOD-2 bricks
gain a soft horizon tint, and the far LOD-3 ring dissolves into the
cubemap horizon color. Without a band the strategy falls back to its
constant density (matched to the auto-tune at outer=1024 m so headless
callers still get usable distance fade).

A previous version used `FogFalloff::Linear { start, end }` whenever
the streamer band was supplied — that left 0–563 m of terrain entirely
unfogged, so near and mid bricks read as hard silhouettes against the
sky. The exp² rewrite blends every LOD into the horizon continuously.

The fog color tracks the current sky horizon (sun-curve-driven), so
the mist matches whatever atmosphere the sky strategy is rendering at
that time of day.

### Cubemap shape

Each face is 256² RGBA8 (`SkyboxConfig.face_resolution = 256`); the
six faces are concatenated in `CubeFace::ALL` order
(PosX, NegX, PosY, NegY, PosZ, NegZ). The Bevy 0.13 cubemap is
constructed with `TextureViewDimension::Cube` and attached to the
FP camera via `bevy::core_pipeline::Skybox { image, brightness }`.
`ProceduralDomeSky` keeps its custom sphere on top — the cubemap
shows the world's far horizon geometry; the dome shades the
atmospheric gradient and the sun disc.

### Refresh policy

`SkyboxRefreshPolicy { position_delta_frac: 0.05, altitude_delta_frac:
0.10, max_age_ticks: 600, refresh_on_tier_change: true }` (defaults).
A bake is also frame-budget-gated by
`SkyboxRuntime.min_frames_between_bakes` (default 30) so a fast
observer can't issue more than one bake every half-second at 60 fps.

### Future strategies (still out of scope)

These would slot in identically to Step 8 / Step 9:

- Triplanar texturing — new `ShadingStrategy` variant; the
  `PaletteVoxelMaterial` storage buffer grows a per-id texture index;
  the WGSL samples three projections and blends by normal.
- SSAO post pass — new `AoStrategy` variant; consumes Bevy's GTAO
  pipeline instead of baking corner AO into vertex colors.
- Water refraction / foam — extend `MaterialEntry` with refraction
  index + flow field, add a translucent fan-out to
  `PaletteVoxelMaterial` that uses Bevy's transmission path.
- Real atmospheric scattering — a coupled `FogStrategy` +
  `SkyStrategy` pair that share an atmosphere model; the strategy
  spine treats them as independent slots today, but the spine itself
  doesn't preclude cross-slot coordination through a shared resource.

None of these block the success criterion ("clean render, nice
lighting, multiple materials"); each ships as a strategy impl when
needed.
