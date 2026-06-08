//! First-person voxel editing — aim with the camera, click to carve / place.
//!
//! The player aims with the FP camera; **left-click removes** the targeted
//! voxel and **right-click places** the selected material against the hit face.
//! Single-voxel and sphere/cube brushes are supported. Edits route through the
//! authoritative [`WorldHost`] and the edited bricks update live in *both*
//! render paths (mesh + raymarch).
//!
//! # Determinism contract
//!
//! The host is the **only** mutator: a [`WorldRequest::WriteVoxel`] /
//! [`WorldRequest::WriteRegion`] updates the host's cache + overlay + per-voxel
//! journal. The client never derives authoritative state — it only *predicts
//! which bricks changed* (a safe superset, via the same
//! [`InteractionUnit::affected_voxels`] the host uses) and *re-fetches* the
//! authoritative bytes through [`fetch_and_build`]. Nothing render- or
//! DAG-derived flows back into `GetBrick` or the journal.
//!
//! # The dual coordinate grid (the main correctness trap)
//!
//! Two grids are in play and they are **not** the same scale:
//!
//! - **Render / voxel-index grid = 1 m per voxel.** Brick `bc` is drawn at
//!   `bc * BRICK_EDGE` meters; the camera, the picker, and the host's integer
//!   `overlay`/`WriteVoxel` `pos` all live here.
//! - **Host brush metric space = `mpv` per voxel.** `MetricScale::DEFAULT_WORLD`
//!   has `mpv = root_size_m / 2^max_depth ≈ 0.596 m`, **not 1.0**. `WriteRegion`
//!   tests its brush predicate against metric voxel centers.
//!
//! Single-voxel edits use [`WorldRequest::WriteVoxel`] with the **integer**
//! `pos` (no conversion — immune to the trap). Brushes use [`WorldRequest::WriteRegion`]
//! and convert through [`voxel_center_metric`] (`(cell + 0.5) * mpv`, mirroring
//! the host's `apply_region`) so the brush lands exactly where the crosshair points.

use atomr_worlds_core::addr::Address;
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::interaction::{InteractionUnit, ToolKind};
use atomr_worlds_core::lod::{Lod, MetricScale};
use atomr_worlds_proto::{Envelope, WorldRequest};
use atomr_worlds_voxel::voxel::Voxel;
use atomr_worlds_voxel::{world_ray_first_solid, WorldRayHit, BRICK_EDGE};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::brick_gen::{fetch_and_build, BrickGenWorkers};
use crate::modes::fp::{
    release_chunk_dag, spawn_edited_brick, FpState, MaterialPool, VoxelMaterialPool,
};
use crate::render::{
    BrickGpuStats, DagBufferCache, RaymarchMaterial, RaymarchResources, RenderConfig,
};
use crate::view_mode::ViewMode;
use crate::world_runtime::WorldRuntime;
use crate::world_stream::{ChunkStreamer, LoadedChunks};

/// Above this many affected bricks an edit refreshes asynchronously (drop the
/// entries + let the normal streamer re-fetch) instead of synchronously on the
/// main thread. Single-voxel and small brushes stay under it, so a discrete
/// click is a sub-ms-to-few-ms hitch; a fat brush over cache-cold terrain
/// doesn't stall the frame.
pub const MAX_SYNC_REFRESH_BRICKS: usize = 16;

/// Editing state: the selected material + tool, the brush radius, the reach,
/// and the most recent picker hit (refreshed every frame for the crosshair /
/// highlight box). A Bevy [`Resource`]; initialised by
/// [`crate::modes::fp::FpPlugin`].
#[derive(Resource, Debug, Clone)]
pub struct EditState {
    /// Material id placed on right-click. `1..=palette_max`; never 0 (air).
    pub selected_material: u16,
    /// Active brush. `Voxel` is the single-voxel path; `Sphere`/`Cube` are
    /// radius brushes. (`Cone` is treated as `Sphere` by the host predicate.)
    pub tool: ToolKind,
    /// Brush radius in **voxels** (render-grid units). Ignored for `Voxel`.
    pub radius_voxels: f64,
    /// Maximum pick reach in meters (render grid ⇒ voxels).
    pub reach_m: f64,
    /// Most recent picker result, or `None` when the crosshair points at
    /// nothing resident within reach. Drives the HUD crosshair / highlight.
    pub last_hit: Option<WorldRayHit>,
    /// Cursor lock state at the end of the previous frame. An edit fires only
    /// when the cursor was *already* locked — so the left-click that grabs the
    /// cursor (see `grab_cursor`) does not also carve.
    pub prev_cursor_locked: bool,
}

impl Default for EditState {
    fn default() -> Self {
        Self {
            selected_material: 1,
            tool: ToolKind::Voxel,
            radius_voxels: 2.0,
            reach_m: 6.0,
            last_hit: None,
            prev_cursor_locked: false,
        }
    }
}

/// Broadcast whenever a first-person edit is applied, so other subsystems can
/// react without the editor depending on them. `bricks` is the affected-brick
/// superset the editor already computed; `removed` distinguishes carves (which
/// can detach structure into debris) from placements.
///
/// Client-side physics (the `physics` feature) consumes this to run flood-fill
/// fracture; it is registered unconditionally in [`crate::modes::fp::FpPlugin`]
/// so the editor can emit it whether or not a reader is present.
#[derive(Message, Clone, Debug)]
pub struct VoxelEditEvent {
    pub addr: Address,
    pub removed: bool,
    pub bricks: Vec<IVec3>,
}

/// The host brush metric scale — must match the host's `brush_scale`
/// (`local.rs`), which uses [`MetricScale::DEFAULT_WORLD`] for world addresses.
#[inline]
pub(crate) fn brush_metric_scale() -> MetricScale {
    MetricScale::DEFAULT_WORLD
}

/// Metric-space center of integer voxel `cell` — `(cell + 0.5) * mpv` per axis.
/// Mirrors the host's `apply_region` voxel-center math (`local.rs`), so a
/// `WriteRegion` centered here lands the brush on exactly `cell`.
#[inline]
pub(crate) fn voxel_center_metric(cell: IVec3) -> DVec3 {
    let mpv = brush_metric_scale().meters_per_voxel(Lod::new(brush_metric_scale().max_depth));
    DVec3::new(
        (cell.x as f64 + 0.5) * mpv,
        (cell.y as f64 + 0.5) * mpv,
        (cell.z as f64 + 0.5) * mpv,
    )
}

/// Brick coordinate containing integer voxel `cell` (render grid; `div_euclid`
/// so negative coords floor — matching the host's `brick_of_voxel`).
#[inline]
pub(crate) fn brick_of(cell: IVec3) -> IVec3 {
    let e = BRICK_EDGE as i64;
    IVec3::new(cell.x.div_euclid(e), cell.y.div_euclid(e), cell.z.div_euclid(e))
}

/// Sample the resident LOD-0 voxel at integer world cell `c`, or `EMPTY` when
/// the brick isn't resident (the picker treats unloaded space as air, so a ray
/// through a not-yet-streamed region simply finds no target — correct).
#[inline]
pub(crate) fn sample_cell(loaded: &LoadedChunks, c: IVec3) -> Voxel {
    let e = BRICK_EDGE as i64;
    match loaded.get(&(brick_of(c), 0)) {
        Some(chunk) => match &chunk.brick {
            Some(b) => b.get(IVec3::new(c.x.rem_euclid(e), c.y.rem_euclid(e), c.z.rem_euclid(e))),
            None => Voxel::EMPTY,
        },
        None => Voxel::EMPTY,
    }
}

/// Filter an affected-brick set to the keys that should be refreshed in place:
/// currently loaded, **not** fading out, at **LOD 0**. Anything else either
/// isn't editable (coarse tiers self-heal on re-stream) or isn't resident.
pub(crate) fn keys_to_refresh(affected: &[IVec3], loaded: &LoadedChunks) -> Vec<(IVec3, u8)> {
    affected
        .iter()
        .filter_map(|bc| {
            let key = (*bc, 0u8);
            match loaded.get(&key) {
                Some(chunk) if !chunk.is_fading_out => Some(key),
                _ => None,
            }
        })
        .collect()
}

/// Digit keys pick the place material; `Tab` cycles the tool; `[` / `]` adjust
/// the brush radius. FP-only.
pub fn edit_select_tool_material(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    pool: Res<MaterialPool>,
    mut edit: ResMut<EditState>,
) {
    if *mode != ViewMode::Fp {
        return;
    }
    // Palette ids are dense `1..=max_id`; `handles[id]` is indexed by id, so
    // `len - 1` is the max id (index 0 is air).
    let max_id = pool.handles.len().saturating_sub(1).max(1) as u16;
    const DIGITS: [(KeyCode, u16); 9] = [
        (KeyCode::Digit1, 1),
        (KeyCode::Digit2, 2),
        (KeyCode::Digit3, 3),
        (KeyCode::Digit4, 4),
        (KeyCode::Digit5, 5),
        (KeyCode::Digit6, 6),
        (KeyCode::Digit7, 7),
        (KeyCode::Digit8, 8),
        (KeyCode::Digit9, 9),
    ];
    for (k, id) in DIGITS {
        if keys.just_pressed(k) {
            edit.selected_material = id.clamp(1, max_id);
        }
    }
    if keys.just_pressed(KeyCode::Tab) {
        edit.tool = match edit.tool {
            ToolKind::Voxel => ToolKind::Sphere,
            ToolKind::Sphere => ToolKind::Cube,
            _ => ToolKind::Voxel,
        };
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        edit.radius_voxels = (edit.radius_voxels + 1.0).min(16.0);
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        edit.radius_voxels = (edit.radius_voxels - 1.0).max(1.0);
    }
}

/// Grouped spawn-path resources, bundled as one [`SystemParam`] so
/// [`fp_edit_voxels`] stays well under Bevy's per-system param limit. Mirrors
/// the streamer's `RaymarchSpawn` but adds the mesh-path pools so an edit can
/// refresh under any [`crate::render::ShadingMode`].
#[derive(SystemParam)]
pub struct EditSpawn<'w> {
    pub pool: Res<'w, MaterialPool>,
    pub voxel_pool: Res<'w, VoxelMaterialPool>,
    pub res: Res<'w, RaymarchResources>,
    pub cache: ResMut<'w, DagBufferCache>,
    pub stats: ResMut<'w, BrickGpuStats>,
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub materials: ResMut<'w, Assets<RaymarchMaterial>>,
    pub storage_buffers: ResMut<'w, Assets<ShaderStorageBuffer>>,
}

/// Per-frame: cast the crosshair ray, store the hit for the HUD, and — on a
/// left/right click while the cursor is grabbed — apply the edit and refresh
/// the touched bricks. FP-only; no-op in harness mode (the cursor is never
/// locked there — harness-driven editing is a documented follow-up).
#[allow(clippy::too_many_arguments)]
pub fn fp_edit_voxels(
    state: Res<FpState>,
    mode: Res<ViewMode>,
    mouse: Res<ButtonInput<MouseButton>>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    harness: Option<Res<crate::harness::HarnessActive>>,
    runtime: Res<WorldRuntime>,
    render_cfg: Res<RenderConfig>,
    streamer: Res<ChunkStreamer>,
    mut edit: ResMut<EditState>,
    mut loaded: ResMut<LoadedChunks>,
    mut workers: ResMut<BrickGenWorkers>,
    mut spawn: EditSpawn,
    mut edit_tx: MessageWriter<VoxelEditEvent>,
    mut commands: Commands,
) {
    if harness.is_some() || *mode != ViewMode::Fp || !state.ready {
        edit.last_hit = None;
        return;
    }

    // Ray from the rendered camera pose (eye/target — *not* the smoothed motion
    // forward) so the crosshair and the pick agree pixel-for-pixel.
    let cam = state.walk.camera();
    let origin = [cam.eye[0] as f64, cam.eye[1] as f64, cam.eye[2] as f64];
    let dir = [
        (cam.target[0] - cam.eye[0]) as f64,
        (cam.target[1] - cam.eye[1]) as f64,
        (cam.target[2] - cam.eye[2]) as f64,
    ];

    // Pick (immutable borrow of `loaded` ends with the call).
    let hit = {
        let loaded_ref: &LoadedChunks = &loaded;
        world_ray_first_solid(origin, dir, edit.reach_m, |c| sample_cell(loaded_ref, c))
    };
    edit.last_hit = hit;

    // Only edit when the cursor was already grabbed at frame start — the
    // click that grabs the cursor must not also carve.
    let locked_now = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    let edits_enabled = locked_now && edit.prev_cursor_locked;
    edit.prev_cursor_locked = locked_now;

    let remove = mouse.just_pressed(MouseButton::Left);
    let place = mouse.just_pressed(MouseButton::Right);
    if !edits_enabled || (!remove && !place) {
        return;
    }
    let Some(hit) = hit else { return };

    // Resolve the target cell + the voxel to write.
    let (target_cell, voxel) = if remove {
        (hit.cell, Voxel::EMPTY)
    } else {
        // Place requires an exposed face (a normal); origin-inside-solid has none.
        if hit.normal == IVec3::ZERO {
            return;
        }
        (hit.place_cell, Voxel::new(edit.selected_material))
    };

    let address: Address = state.addr.into();
    let frame = streamer.frame;
    let shading_mode = render_cfg.shading.mode();
    let raymarch_tier = render_cfg.raymarch_tier;

    // The bricks an edit can touch (a superset the client refreshes).
    let affected: Vec<IVec3> = match edit.tool {
        ToolKind::Voxel => {
            // Single voxel: integer `pos` (no metric conversion), exactly one brick.
            let env = Envelope::new(0, address, WorldRequest::WriteVoxel {
                addr: address,
                pos: target_cell,
                voxel,
            });
            let _ = runtime.runtime.handle().block_on(runtime.host.request(env));
            vec![brick_of(target_cell)]
        }
        kind => {
            // Brush: convert the integer target to the host's metric space.
            let scale = brush_metric_scale();
            let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
            let center = voxel_center_metric(target_cell);
            let radius_m = edit.radius_voxels * mpv;
            let unit = match kind {
                ToolKind::Cube => InteractionUnit::cube(radius_m, Lod::new(scale.max_depth)),
                _ => InteractionUnit::sphere(radius_m, Lod::new(scale.max_depth)),
            };
            let env = Envelope::new(0, address, WorldRequest::WriteRegion {
                addr: address,
                center,
                unit,
                voxel,
            });
            let _ = runtime.runtime.handle().block_on(runtime.host.request(env));
            // Predict the touched bricks with the *same* call the host uses.
            unit.affected_voxels(scale, center, BRICK_EDGE as i64).bricks
        }
    };

    let keys = keys_to_refresh(&affected, &loaded);
    if keys.len() <= MAX_SYNC_REFRESH_BRICKS {
        // Sync: re-fetch authoritative bytes and swap each brick in place.
        for key in keys {
            let ready = runtime.runtime.handle().block_on(fetch_and_build(
                runtime.host.clone(),
                render_cfg.ao.clone(),
                address,
                key.0,
                Lod::new(0),
            ));
            spawn_edited_brick(
                ready,
                frame,
                shading_mode,
                &spawn.pool,
                &spawn.voxel_pool,
                &spawn.res,
                raymarch_tier,
                &mut spawn.cache,
                &mut spawn.stats,
                &mut spawn.meshes,
                &mut spawn.materials,
                &mut spawn.storage_buffers,
                &mut commands,
                &mut loaded,
            );
        }
    } else {
        // Large brush: drop the affected entries (despawn + decref) and let the
        // normal async pipeline re-stream them, so the frame doesn't stall.
        for key in keys {
            if let Some(ent) = loaded.get(&key).and_then(|c| c.entity) {
                commands.entity(ent).despawn();
            }
            release_chunk_dag(&loaded, &key, &mut spawn.cache);
            loaded.remove(&key);
            workers.forget(&key);
        }
    }

    // Broadcast the edit so client-side physics (and any future listener) can
    // react. `affected` is the brick superset the host touched.
    edit_tx.write(VoxelEditEvent { addr: address, removed: remove, bricks: affected });
}

/// Marker on the reusable selection-highlight cube repositioned each frame from
/// [`EditState::last_hit`].
#[derive(Component)]
pub struct EditHighlight;

/// Spawn the single reusable selection-highlight cube (hidden until a pick
/// lands). Skipped under the harness so FP captures don't gain the overlay.
pub fn setup_edit_highlight(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    harness: Option<Res<crate::harness::HarnessActive>>,
) {
    if harness.is_some() {
        return;
    }
    // Slightly larger than a unit voxel so it reads as an outline hugging the
    // targeted cell. Unlit + translucent so it tints rather than occludes.
    let mesh = meshes.add(Cuboid::new(1.04, 1.04, 1.04));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.95, 0.25, 0.22),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::default(),
        Visibility::Hidden,
        EditHighlight,
    ));
}

/// Reposition the highlight cube onto the currently-targeted voxel (centered at
/// `cell + 0.5`, since the render grid is 1 m/voxel), or hide it when there's no
/// pick / we're not in first-person.
pub fn fp_edit_highlight(
    mode: Res<ViewMode>,
    edit: Res<EditState>,
    mut q: Query<(&mut Transform, &mut Visibility), With<EditHighlight>>,
) {
    let Ok((mut tf, mut vis)) = q.single_mut() else { return };
    if *mode == ViewMode::Fp {
        if let Some(hit) = edit.last_hit {
            tf.translation = Vec3::new(
                hit.cell.x as f32 + 0.5,
                hit.cell.y as f32 + 0.5,
                hit.cell.z as f32 + 0.5,
            );
            if *vis != Visibility::Visible {
                *vis = Visibility::Visible;
            }
            return;
        }
    }
    if *vis != Visibility::Hidden {
        *vis = Visibility::Hidden;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::addr::WorldAddr;
    use atomr_worlds_host::{LocalHost, WorldHost};
    use atomr_worlds_proto::WorldEvent;
    use atomr_worlds_voxel::brick::Brick;
    use crate::world_stream::LoadedChunk;

    const SEED: u64 = 0x0A70_3D17_1234_5678;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
    }

    async fn get_brick(host: &LocalHost, addr: Address, bc: IVec3) -> Brick {
        let env = Envelope::new(0, addr, WorldRequest::GetBrick { addr, brick: bc, lod: Lod::new(0) });
        let resp = host.request(env).await.expect("get brick");
        let WorldEvent::BrickSnapshot { payload, .. } = resp.body else { panic!("variant") };
        Brick::from_bytes(&payload).expect("decode")
    }

    fn mk_chunk(coord: IVec3, depth: u8) -> LoadedChunk {
        LoadedChunk {
            coord,
            lod: Lod::new(depth),
            entity: None,
            last_seen_frame: 0,
            is_fading_out: false,
            dag_digest: None,
            dag_tier: None,
            brick: None,
        }
    }

    #[test]
    fn voxel_center_metric_matches_host_brush_space() {
        let scale = brush_metric_scale();
        let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
        let c = voxel_center_metric(IVec3::new(2, 2, 2));
        assert!((c.x - 2.5 * mpv).abs() < 1e-9);
        // floor(center / mpv) recovers the integer cell — exactly the host's
        // `affected_voxels` mapping, so a brush centered here targets cell 2.
        assert_eq!((c.x / mpv).floor() as i64, 2);
        assert_eq!((c.y / mpv).floor() as i64, 2);
    }

    #[test]
    fn keys_to_refresh_returns_only_loaded_nonfading_lod0() {
        let mut loaded = LoadedChunks::default();
        loaded.insert((IVec3::new(0, 0, 0), 0), mk_chunk(IVec3::new(0, 0, 0), 0));
        loaded.insert((IVec3::new(1, 0, 0), 0), mk_chunk(IVec3::new(1, 0, 0), 0));
        loaded.mark_fading_out(&(IVec3::new(1, 0, 0), 0));
        // A coarse (LOD-1) entry for a different coord — not keyed at depth 0.
        loaded.insert((IVec3::new(2, 0, 0), 1), mk_chunk(IVec3::new(2, 0, 0), 1));

        let affected = vec![
            IVec3::new(0, 0, 0), // loaded, LOD0, live  -> kept
            IVec3::new(1, 0, 0), // loaded, LOD0, fading -> dropped
            IVec3::new(2, 0, 0), // only loaded at LOD1  -> dropped
            IVec3::new(3, 0, 0), // not loaded           -> dropped
        ];
        assert_eq!(keys_to_refresh(&affected, &loaded), vec![(IVec3::new(0, 0, 0), 0)]);
    }

    #[test]
    fn write_voxel_changes_exactly_one_voxel() {
        let rt = rt();
        rt.block_on(async {
            let host = LocalHost::with_seed(SEED).await.expect("host");
            let addr = Address::World(WorldAddr::ROOT);

            // A cell high in the air — guaranteed empty terrain.
            let cell = IVec3::new(3, 2000, 5);
            let bc = brick_of(cell);
            let e = BRICK_EDGE as i64;
            let lc = IVec3::new(cell.x.rem_euclid(e), cell.y.rem_euclid(e), cell.z.rem_euclid(e));

            let before = get_brick(&host, addr, bc).await;
            assert!(before.get(lc).is_empty(), "test cell must start empty");

            let new_voxel = Voxel::new(9);
            let env = Envelope::new(0, addr, WorldRequest::WriteVoxel { addr, pos: cell, voxel: new_voxel });
            host.request(env).await.expect("write");

            let after = get_brick(&host, addr, bc).await;
            assert_eq!(after.get(lc), new_voxel, "the targeted voxel changed");

            // Exactly one voxel in the brick differs.
            let mut diffs = 0u32;
            for z in 0..e {
                for y in 0..e {
                    for x in 0..e {
                        let p = IVec3::new(x, y, z);
                        if before.get(p) != after.get(p) {
                            diffs += 1;
                            assert_eq!(p, lc, "only the targeted local cell changed");
                        }
                    }
                }
            }
            assert_eq!(diffs, 1, "WriteVoxel touches exactly one voxel");
            host.shutdown().await.expect("shutdown");
        });
    }

    #[test]
    fn brush_center_lands_on_target_and_prediction_covers_it() {
        let rt = rt();
        rt.block_on(async {
            let host = LocalHost::with_seed(SEED).await.expect("host");
            let addr = Address::World(WorldAddr::ROOT);

            let target = IVec3::new(3, 2000, 5);
            let bc = brick_of(target);
            let e = BRICK_EDGE as i64;
            let lc = IVec3::new(target.x.rem_euclid(e), target.y.rem_euclid(e), target.z.rem_euclid(e));

            let scale = brush_metric_scale();
            let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
            let center = voxel_center_metric(target);
            let unit = InteractionUnit::sphere(1.0 * mpv, Lod::new(scale.max_depth));

            // The client's prediction (same fn the host uses) covers the center brick.
            let predicted = unit.affected_voxels(scale, center, BRICK_EDGE as i64);
            assert!(predicted.bricks.contains(&bc), "predicted set must cover the center brick");

            // Applying the brush lands the center voxel where the crosshair points.
            let env = Envelope::new(0, addr, WorldRequest::WriteRegion {
                addr,
                center,
                unit,
                voxel: Voxel::new(7),
            });
            host.request(env).await.expect("write region");

            let brick = get_brick(&host, addr, bc).await;
            assert_eq!(brick.get(lc), Voxel::new(7), "brush center hit the intended voxel");
            host.shutdown().await.expect("shutdown");
        });
    }
}
