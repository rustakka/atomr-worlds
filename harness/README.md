# harness/

Scenario files for the `atomr-worlds-client` screenshot harness. Each file
is a TOML scenario that the client replays under `--harness <path>` to
capture PNG screenshots at chosen frames.

## Usage

```sh
# build + run (uses xvfb-run if installed, otherwise current $DISPLAY)
./scripts/run-harness.sh harness/scenes/fp_lookup.toml /tmp/shots/

# or invoke the binary directly
DISPLAY=:1 ./target/release/atomr-worlds-client \
    --harness harness/scenes/fp_lookup.toml \
    --harness-out /tmp/shots/
```

Stdout prints one `HARNESS_SHOT <abs-path>` line per captured frame. Stderr
carries Bevy/wgpu logs.

The harness still creates a real OS window (the X11/hybrid-GPU presentation
path needs one alongside the offscreen render target), but it spawns
unfocused — it sits in the background instead of stealing focus from
whatever you were doing. `WinitSettings` is pinned to `Continuous` updates
for both the focused and unfocused states so the scenario plays out at the
same cadence regardless of which window is active. Interactive runs (no
`--harness`) keep Bevy's default focus behavior.

## Scenario schema

| field           | required | notes                                                                  |
| --------------- | -------- | ---------------------------------------------------------------------- |
| `mode`          | no       | `"fp"` / `"tp"` / `"slice"` / `"rts"` / `"overview"`. Default `"fp"`.  |
| `width`/`height`| no       | logical pixels (scale\_factor\_override is forced to 1.0).             |
| `warmup_frames` | no       | frames before event 0 fires. Default 60. **Use ~180 for FP.**          |
| `output_prefix` | no       | filename stem; PNGs are `<prefix>_NNNN.png`. Default `"shot"`.         |
| `seed`          | no       | currently ignored — use `--seed` on the CLI.                           |
| `events`        | yes      | array of event tables.                                                 |

Each event entry has a `frame` (offset from end of warmup) and a `kind`:

- `key_press` / `key_release` — needs `key = "<KeyCode-variant-name>"`.
- `key_tap` — desugars at load into a press at frame N and a release at N+1.
- `mouse_move` — needs `dx` and/or `dy`. Cursor grab is bypassed in
  harness mode so `fp_input` consumes the motion unconditionally.
- `screenshot` — captures the window via `xwd` and writes a PNG.
- `exit` — schedules `AppExit` once the last event frame + 5 has passed.

- `mouse_button_press` / `mouse_button_release` — needs `button = "Left" |
  "Right" | "Middle"`. Left-click carves / right-click places in FP (requires the
  scripted-edit hook below).
- `dump_frame_diag` / `dump_motion` — print the recent per-frame timing buffer /
  the camera motion state to stderr (`FRAME_DIAG …` / motion lines), for perf and
  movement checks.

Unknown keys or kinds cause the scenario load to fail loudly at startup.

## Physics + scripted-edit hooks (opt-in)

The harness forces **physics off** and **editing inert** by default, so golden
captures are never perturbed. Two environment variables opt a run back in (both
unset → byte-identical to before):

| env var | effect |
| ------- | ------ |
| `ATOMR_HARNESS_PHYSICS=1` | run client-side physics under the harness (needs `--physics on`) — colliders, the character controller, and fracture/debris all activate |
| `ATOMR_HARNESS_EDIT=1` | enable **scripted carving/placement** — a scene's `mouse_button_press` fires a real edit (the cursor-grab gate is bypassed) |

Together they make the carve → fracture → debris pipeline (Rec 2 / Rec 3)
capturable. Example — `harness/scenes/fracture_carve.toml` aims at the terrain,
left-clicks to carve, and screenshots the (flicker-free, off-thread-refreshed)
hole:

```sh
ATOMR_HARNESS_PHYSICS=1 ATOMR_HARNESS_EDIT=1 \
  ./target/debug/atomr-worlds-client.exe --physics on \
    --harness harness/scenes/fracture_carve.toml --harness-out /tmp/shots/
```

`ATOMR_HARNESS_EDIT=1` also renders the **targeting highlight** (normally hidden
under the harness), which matches the active tool + brush radius — a unit cube for
the Voxel tool, a sphere of radius `radius_voxels` for Sphere/Cone, a cube of
half-edge `radius_voxels` for Cube. `harness/scenes/edit_highlight.toml` cycles
the tool (`Tab`) and grows the brush (`]` = `BracketRight`) to show each shape.

## Implementation notes

The capture path does **not** use Bevy 0.13.2's `ScreenshotManager` — it
panics on async buffer map on hybrid-GPU Linux setups (NVIDIA discrete +
Intel iGPU, Vulkan or GL). Instead, the harness shells out to
`xwd -name "<window title>" -silent`, parses the XWD2 dump in-process, and
writes a PNG via the `image` crate. The binary depends on the X11 `xwd`
tool being on PATH (Ubuntu: `x11-apps`).

See `crates/atomr-worlds-client/src/harness.rs` for the full schema, key
mapping, and capture implementation, and
`~/.claude/skills/atomr-worlds-harness/SKILL.md` for usage guidance.
