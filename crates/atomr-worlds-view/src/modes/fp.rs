//! Phase 14a — 1st-person walk.
//!
//! A [`WalkCamera`] wraps an [`ObserverState`] and tracks yaw/pitch + eye
//! height. Each `tick` rotates the caller's local-frame input by yaw, adds
//! it to the observer position, and forwards the new pose to the underlying
//! observer (which keeps velocity + skybox-refresh bookkeeping for free).
//!
//! [`build_fp_scene`] is the rendering pipeline: enumerate the bricks whose
//! AABB intersects a region cube around the camera, frustum-cull against the
//! camera's view, fetch each survivor through a [`WorldQuery`], greedy-mesh
//! the result, and partition into a near (opaque) ring and a far
//! (distance-faded) ring before handing the lot to
//! [`render_composite`](crate::render::render_composite).

use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_voxel::BRICK_EDGE;

use crate::camera::{Camera, Projection};
use crate::frustum::Frustum;
use crate::mesh::greedy_mesh;
use crate::observer::ObserverState;
use crate::render::{render_composite, CompositeScene, Framebuffer, RenderConfig};
use crate::scene::{MaterialPalette, MeshNode};
use crate::view_cache::{CacheAabb, DerivedKey};
use crate::world_query::WorldQuery;

/// Per-tick walk input. `move_local` is meters in the camera's local frame
/// (`+x = right`, `+y = up`, `+z = forward`); the [`WalkCamera`] rotates it
/// by yaw before applying it to the observer position.
#[derive(Copy, Clone, Debug, Default)]
pub struct WalkInput {
    pub move_local: [f32; 3],
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub crouch: bool,
}

/// 1st-person walking camera. Yaw is rotation around world-up (+Y); pitch
/// is rotation around the right axis with `[-π/2, +π/2]` clamping so the
/// look-vector never inverts. Eye height adds to the observer's Y so the
/// camera sits at standing height by default; `crouch` halves it for the
/// frame.
#[derive(Debug)]
pub struct WalkCamera {
    pub observer: ObserverState,
    pub yaw: f32,
    pub pitch: f32,
    pub eye_height_m: f32,
    pub fov_y_rad: f32,
    pub aspect: f32,
    /// Last `crouch` flag — read by [`Self::camera`] to halve eye height.
    crouched: bool,
}

impl WalkCamera {
    pub fn new(position: DVec3, containing_frame: ContainingFrame, aspect: f32) -> Self {
        Self {
            observer: ObserverState::new(position, containing_frame),
            yaw: 0.0,
            pitch: 0.0,
            eye_height_m: 1.7,
            fov_y_rad: std::f32::consts::FRAC_PI_3, // 60° vertical FOV
            aspect,
            crouched: false,
        }
    }

    /// Advance one tick.
    pub fn tick(&mut self, input: WalkInput, dt_s: f32) {
        self.yaw += input.yaw_delta;
        self.pitch = (self.pitch + input.pitch_delta).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.crouched = input.crouch;
        // Rotate `move_local` by yaw into world space. `+z_local = forward`,
        // `+x_local = right`. yaw=0 ⇒ forward = -Z world (camera looks down
        // -Z in atomr's RH convention), but for a walk camera we treat
        // forward as +Z world at yaw=0 so callers' input feels natural: a
        // forward push moves the observer in the direction the camera
        // currently faces.
        let world = self.rotate_local_to_world(input.move_local);
        let new_pos = DVec3::new(
            self.observer.position.x + world[0] as f64,
            self.observer.position.y + world[1] as f64,
            self.observer.position.z + world[2] as f64,
        );
        self.observer.tick(new_pos, None, dt_s);
    }

    /// Rotate a local-frame displacement (`+x = right`, `+y = up`,
    /// `+z = forward`) into world space by the current yaw. Pulled out of
    /// [`Self::tick`] so the free-fly path and an external character
    /// controller share one definition of the heading convention (forward is
    /// `+Z` world at yaw=0). Up stays world-up; pitch never tilts movement.
    pub fn rotate_local_to_world(&self, local: [f32; 3]) -> [f32; 3] {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let [mx, my, mz] = local;
        // Right is rotated x; forward is rotated z.
        [cos_y * mx + sin_y * mz, my, -sin_y * mx + cos_y * mz]
    }

    /// Set the crouch flag directly. Used when an external driver (e.g. a
    /// physics character controller) owns position but still wants
    /// [`Self::camera`] to lower the eye height for the frame, without routing
    /// a full [`WalkInput`] through [`Self::tick`].
    pub fn set_crouch(&mut self, crouch: bool) {
        self.crouched = crouch;
    }

    /// Build the [`Camera`] for the current pose. Eye sits at observer +
    /// `up * eye_height_m` (halved if crouched).
    pub fn camera(&self) -> Camera {
        let eye_h = if self.crouched { self.eye_height_m * 0.5 } else { self.eye_height_m };
        let pos = self.observer.position;
        let eye = [pos.x as f32, pos.y as f32 + eye_h, pos.z as f32];
        // Forward vector from yaw/pitch. Yaw rotates around +Y; pitch around
        // the resulting right axis. yaw=pitch=0 ⇒ forward = +Z world.
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let fwd = [sin_y * cos_p, sin_p, cos_y * cos_p];
        let target = [eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2]];
        Camera {
            eye,
            target,
            up: [0.0, 1.0, 0.0],
            fov_y_rad: self.fov_y_rad,
            aspect: self.aspect,
            near: 0.1,
            far: 1024.0,
            projection: Projection::Perspective { fov_y_rad: self.fov_y_rad },
        }
    }
}

const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// View-cache key for a meshed brick. Keyed on `(addr, brick_coord, lod)` so
/// per-LOD pyramids coexist without colliding; `intersects` checks the brick
/// AABB against the cache AABB so `ViewCache::invalidate_intersecting` evicts
/// only the bricks that the host's `RegionDelta` actually touched.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct MeshCacheKey {
    pub addr: WorldAddr,
    pub brick_coord: IVec3,
    pub lod: Lod,
}

impl DerivedKey for MeshCacheKey {
    fn world_addr(&self) -> &WorldAddr {
        &self.addr
    }
    fn intersects(&self, aabb: CacheAabb) -> bool {
        let edge = BRICK_EDGE as f64;
        let lo = [
            self.brick_coord.x as f64 * edge,
            self.brick_coord.y as f64 * edge,
            self.brick_coord.z as f64 * edge,
        ];
        let hi = [lo[0] + edge, lo[1] + edge, lo[2] + edge];
        CacheAabb::new(lo, hi).intersects(aabb)
    }
}

/// Build a [`CompositeScene`] for the current `cam`/`addr`. Iterates the
/// brick coordinates whose AABB intersects the region cube of half-size
/// `region_m` around the camera eye, frustum-culls each brick, fetches
/// surviving bricks through `world`, meshes them, and partitions into a far
/// (distance-faded) and near (opaque) ring at the `region_m * 0.6` threshold.
/// `extra_meshes` are appended to the near ring without filtering — this is
/// how Phase 14b (chase camera) injects an anchor decal at the player's
/// position.
///
/// The returned `CompositeScene` borrows from `cam` (for the observer
/// position) and from the freshly-allocated mesh `Vec`s — callers must keep
/// the scene alive until [`render_composite`] returns.
#[derive(Debug)]
pub struct FpScene {
    pub near: Vec<MeshNode>,
    pub far: Vec<MeshNode>,
}

impl FpScene {
    /// Build a [`CompositeScene`] referencing this `FpScene`'s ring vectors.
    pub fn as_composite<'a>(&'a self, cam: &Camera, region_m: f32) -> CompositeScene<'a> {
        CompositeScene::new(None, &self.far, &self.near, cam.eye, region_m * 0.6, region_m)
    }
}

/// Build the FP scene rings. Returned as an `FpScene` so the caller controls
/// the lifetime — `render_composite` borrows the slices.
pub fn build_fp_scene(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    cam: &Camera,
    lod: Lod,
    region_m: f32,
    extra_meshes: &[MeshNode],
) -> FpScene {
    let eye = cam.eye;
    let near_radius_m = region_m * 0.6;
    let edge = BRICK_EDGE as f32;
    let edge_f64 = BRICK_EDGE as f64;

    // Region cube — the AABB of bricks we're willing to consider this frame.
    let cube_min = [(eye[0] - region_m) as f64, (eye[1] - region_m) as f64, (eye[2] - region_m) as f64];
    let cube_max = [(eye[0] + region_m) as f64, (eye[1] + region_m) as f64, (eye[2] + region_m) as f64];

    // Brick-coord range that the region cube covers.
    let bmin_x = (cube_min[0] / edge_f64).floor() as i64;
    let bmax_x = (cube_max[0] / edge_f64).ceil() as i64;
    let bmin_y = (cube_min[1] / edge_f64).floor() as i64;
    let bmax_y = (cube_max[1] / edge_f64).ceil() as i64;
    let bmin_z = (cube_min[2] / edge_f64).floor() as i64;
    let bmax_z = (cube_max[2] / edge_f64).ceil() as i64;

    let frustum = Frustum::from_camera(cam);

    let mut near = Vec::new();
    let mut far = Vec::new();
    let mut next_id: u64 = 1;
    for bz in bmin_z..=bmax_z {
        for by in bmin_y..=bmax_y {
            for bx in bmin_x..=bmax_x {
                let lo = [bx as f64 * edge_f64, by as f64 * edge_f64, bz as f64 * edge_f64];
                let hi = [lo[0] + edge_f64, lo[1] + edge_f64, lo[2] + edge_f64];
                let bb = CacheAabb::new(lo, hi);
                if !frustum.intersects_aabb(bb) {
                    continue;
                }
                let brick = match world.brick(addr, IVec3::new(bx, by, bz), lod) {
                    Some(b) => b,
                    None => continue,
                };
                if brick.is_empty() {
                    continue;
                }
                let mesh = greedy_mesh(&brick);
                if mesh.vertices.is_empty() {
                    continue;
                }
                let tx = bx as f32 * edge;
                let ty = by as f32 * edge;
                let tz = bz as f32 * edge;
                let transform =
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [tx, ty, tz, 1.0]];
                // Distance from the brick center to the eye decides the
                // ring. Center is `lo + edge/2`.
                let cx = tx + edge * 0.5;
                let cy = ty + edge * 0.5;
                let cz = tz + edge * 0.5;
                let dx = cx - eye[0];
                let dy = cy - eye[1];
                let dz = cz - eye[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();

                let node = MeshNode {
                    id: next_id,
                    mesh: Arc::new(mesh),
                    transform,
                    material_palette: Arc::new(MaterialPalette::default()),
                    lod_hint: Some(lod),
                };
                next_id += 1;
                if d <= near_radius_m {
                    near.push(node);
                } else {
                    far.push(node);
                }
            }
        }
    }
    for m in extra_meshes {
        near.push(m.clone());
    }
    FpScene { near, far }
}

/// Render an FP frame. Convenience wrapper around [`build_fp_scene`] + the
/// composite renderer. Use this when you just need pixels; if you need to
/// inspect the scene (e.g. for tests) call [`build_fp_scene`] directly.
pub fn render_fp(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    cam: &Camera,
    lod: Lod,
    region_m: f32,
    extra_meshes: &[MeshNode],
    cfg: &RenderConfig,
) -> Framebuffer {
    let scene = build_fp_scene(world, addr, cam, lod, region_m, extra_meshes);
    let composite = scene.as_composite(cam, region_m);
    render_composite(&composite, cam, cfg)
}
