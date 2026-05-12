//! In-process host on top of atomr's `ActorSystem`.
//!
//! One `WorldActor` is spawned per [`Address`] (lazily, on first request).
//! The actor owns a brick cache, the subscriber registry, an optional vehicle
//! pose, and — when configured — an `atomr-persistence` binding for durable
//! write replay across restarts. Worlds and vehicles share the same actor
//! type; the actor branches on its `Address` variant.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr::prelude::*;
use atomr_worlds_core::addr::{Address, Level, WorldAddr};
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::interaction::InteractionUnit;
use atomr_worlds_core::lod::MetricScale;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_core::vehicle::{AffineFrame, ParentAddr, VehicleAddr};
use atomr_worlds_generate::{
    default_registry, AuthoredRegionStore, BrickGenContext, DefaultMacroGenerator,
    GeneratorRegistry, MacroGenerator, MacroStateCache, Resolved, WorldGen, WorldMacroState,
};
use atomr_worlds_persist::{VoxelWriteEvent, WorldPersistence, WorldSnapshot};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest, AABB};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::error::HostError;
use crate::host::WorldHost;
use crate::policy::{DefaultPolicy, PolicyResolver};
use crate::shape::{DefaultShape, ShapeResolver};

#[derive(Clone, Debug)]
pub struct LocalHostConfig {
    pub root_seed: u64,
    /// Hierarchical generation strategy registry. Replaces the old
    /// single-`WorldGen` field; existing callers can migrate via
    /// `generators: WorldGen::default().into()`.
    pub generators: GeneratorRegistry,
    /// Per-address generation policy resolver. Default is [`DefaultPolicy`]
    /// (every address → `Seeded`).
    pub policy: Arc<dyn PolicyResolver>,
    /// Per-address world-shape resolver. Default is [`DefaultShape`] (every
    /// address → cubic Earth-class world), which preserves pre-Phase-13
    /// streaming behavior. Configure spherical/cylindrical worlds via
    /// [`PrefixShape`](crate::shape::PrefixShape).
    pub shape_resolver: Arc<dyn ShapeResolver>,
    /// Macro-state generator. Runs once per world (cached) before any
    /// brick generation. Default is [`DefaultMacroGenerator`] at
    /// `grid_level = 4` (~5k faces). Set to `None` to disable macro
    /// pre-sim entirely — brick generators receive `macro_state: None`
    /// and fall back to their legacy code paths.
    pub macro_generator: Option<Arc<dyn MacroGenerator>>,
    /// Per-host macro-state cache. Shared `Arc` so cluster shells (or
    /// tests with multiple `LocalHost`s in the same process) reuse one
    /// cache.
    pub macro_cache: Arc<MacroStateCache>,
    /// Registry of hand-authored regions (Phase 13d). Each registered
    /// region is overlaid on every brick fetch that intersects its
    /// bounds. Shared via `Arc` so cluster shells and Python bindings
    /// can manipulate one store. Defaults to empty.
    pub authored_regions: Arc<std::sync::Mutex<AuthoredRegionStore>>,
    /// Default bound for per-subscriber mpsc channels.
    pub subscriber_capacity: usize,
    /// Timeout for `WorldHost::request`'s `ask` call.
    pub request_timeout: Duration,
    /// Optional persistence backend. When set, voxel writes journal here and
    /// the actor recovers state on spawn.
    pub persistence: Option<Arc<WorldPersistence>>,
}

impl LocalHostConfig {
    /// Convenience: build a config from a [`WorldGen`] for the migration path.
    pub fn from_world_gen(root_seed: u64, world_gen: WorldGen) -> Self {
        Self { root_seed, generators: world_gen.into(), ..Self::default() }
    }
}

impl Default for LocalHostConfig {
    fn default() -> Self {
        Self {
            root_seed: 0xDEAD_BEEF_CAFE_F00D,
            generators: default_registry(),
            policy: Arc::new(DefaultPolicy),
            shape_resolver: Arc::new(DefaultShape),
            // Macro pre-sim is off by default — turns on the moment a
            // caller configures a sphere world via `PrefixShape`. Worlds
            // with the default cubic shape never invoke macro state, so
            // legacy bricks remain bit-identical.
            macro_generator: None,
            macro_cache: Arc::new(MacroStateCache::new()),
            authored_regions: Arc::new(std::sync::Mutex::new(AuthoredRegionStore::new())),
            subscriber_capacity: 256,
            request_timeout: Duration::from_secs(10),
            persistence: None,
        }
    }
}

impl LocalHostConfig {
    /// Convenience: turn on macro pre-sim with the default generator.
    pub fn with_default_macro(mut self) -> Self {
        self.macro_generator = Some(Arc::new(DefaultMacroGenerator::default()));
        self
    }
}

pub struct LocalHost {
    sys: ActorSystem,
    actors: Arc<Mutex<HashMap<Address, ActorRef<WorldMsg>>>>,
    config: LocalHostConfig,
}

impl std::fmt::Debug for LocalHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalHost").field("config", &self.config).finish()
    }
}

impl LocalHost {
    pub async fn new(config: LocalHostConfig) -> Result<Self, HostError> {
        let sys = ActorSystem::create("atomr-worlds", Config::empty())
            .await
            .map_err(|e| HostError::Sys(format!("{e}")))?;
        Ok(Self { sys, actors: Arc::new(Mutex::new(HashMap::new())), config })
    }

    /// Convenience constructor with default config and a chosen root seed.
    pub async fn with_seed(seed: u64) -> Result<Self, HostError> {
        Self::new(LocalHostConfig { root_seed: seed, ..LocalHostConfig::default() }).await
    }

    /// Register an authored region (Phase 13d). The region's voxels
    /// overlay procedural fill on every brick fetch that intersects its
    /// bounds. Note: actors that have already cached the affected bricks
    /// will not pick up the new region until the brick is evicted or the
    /// actor is respawned — for now, register before subscribing.
    pub fn register_authored_region(
        &self,
        region: Arc<dyn atomr_worlds_generate::AuthoredRegion>,
    ) {
        let mut store = self.config.authored_regions.lock().unwrap();
        store.register(region);
    }

    /// Access the shared authored region store directly (e.g. for tests
    /// or persistence rehydration).
    pub fn authored_region_store(
        &self,
    ) -> Arc<std::sync::Mutex<AuthoredRegionStore>> {
        self.config.authored_regions.clone()
    }

    /// Convenience constructor with persistence pre-wired.
    pub async fn with_persistence(
        seed: u64,
        persistence: Arc<WorldPersistence>,
    ) -> Result<Self, HostError> {
        Self::new(LocalHostConfig {
            root_seed: seed,
            persistence: Some(persistence),
            ..LocalHostConfig::default()
        })
        .await
    }

    async fn actor_for(&self, addr: Address) -> Result<ActorRef<WorldMsg>, HostError> {
        let mut map = self.actors.lock().await;
        if let Some(a) = map.get(&addr) {
            return Ok(a.clone());
        }
        let seed = addr.seed(self.config.root_seed);

        // Resolve generation policy → registry choice once, on spawn. The
        // actor caches the outcome for its lifetime.
        let policy = self.config.policy.resolve(&addr);
        let resolved = self
            .config
            .generators
            .resolve(&addr, seed, policy)
            .map_err(|e| HostError::Sys(format!("{e}")))?;

        // Resolve world shape once, on spawn. Same determinism contract as
        // policy: same address → same shape for the actor's lifetime.
        let shape = self.config.shape_resolver.resolve(&addr);

        // Compute macro state (geologic / climate / biome pre-sim) for
        // this world. Cached across actor spawns and host instances that
        // share the same cache. Pure function of `(addr, seed, shape)`.
        // Only computed for non-cube shapes — cubes preserve legacy
        // behavior exactly.
        let macro_state: Option<Arc<WorldMacroState>> = match (
            self.config.macro_generator.as_ref(),
            shape,
        ) {
            (Some(gen), atomr_worlds_core::shape::WorldShape::Cube { .. }) => {
                let _ = gen;
                None
            }
            (Some(gen), _) => Some(self.config.macro_cache.get_or_compute(
                addr,
                seed,
                shape,
                gen.as_ref(),
            )),
            (None, _) => None,
        };

        // Recover overlay before spawning so the actor starts coherent.
        let (writes, last_seq) = if let Some(p) = &self.config.persistence {
            let r = p.recover(addr).await.map_err(|e| HostError::Sys(format!("{e}")))?;
            (r.writes, r.last_seq)
        } else {
            (HashMap::new(), 0)
        };

        let name = format!("entity-{:x}-{}", seed, map.len());
        let persistence = self.config.persistence.clone();
        let authored_regions = self.config.authored_regions.clone();
        let initial_frame = match addr {
            Address::Vehicle(v) => Some(AffineFrame::at_origin(v.parent)),
            Address::World(_) => None,
        };
        let actor = self
            .sys
            .actor_of(
                Props::create(move || {
                    WorldActor::new(
                        addr,
                        seed,
                        resolved.clone(),
                        writes.clone(),
                        last_seq,
                        persistence.clone(),
                        initial_frame,
                        shape,
                        macro_state.clone(),
                        authored_regions.clone(),
                    )
                }),
                &name,
            )
            .map_err(|e| HostError::Sys(format!("{e}")))?;
        map.insert(addr, actor.clone());
        Ok(actor)
    }
}

#[async_trait]
impl WorldHost for LocalHost {
    async fn request(
        &self,
        envelope: Envelope<WorldRequest>,
    ) -> Result<Envelope<WorldEvent>, HostError> {
        let addr = env_target_addr(&envelope);
        let actor = self.actor_for(addr).await?;
        let timeout = self.config.request_timeout;
        let res = actor
            .ask_with(|reply| WorldMsg::Request { env: envelope, reply }, timeout)
            .await
            .map_err(|e| HostError::Ask(format!("{e}")))?;
        res
    }

    async fn subscribe(
        &self,
        envelope: Envelope<WorldRequest>,
    ) -> Result<mpsc::Receiver<Envelope<WorldEvent>>, HostError> {
        let addr = env_target_addr(&envelope);
        let actor = self.actor_for(addr).await?;
        let (sink, rx) = mpsc::channel(self.config.subscriber_capacity);
        let (ready_tx, ready_rx) = oneshot::channel();
        actor.tell(WorldMsg::SubscribeBegin { env: envelope, sink, ready: ready_tx });
        ready_rx.await.map_err(|_| HostError::SubscribeFailed)??;
        Ok(rx)
    }

    async fn shutdown(&self) -> Result<(), HostError> {
        self.sys.clone().terminate().await;
        Ok(())
    }
}

fn env_target_addr(env: &Envelope<WorldRequest>) -> Address {
    match &env.body {
        WorldRequest::GetVoxel { addr, .. }
        | WorldRequest::GetBrick { addr, .. }
        | WorldRequest::WriteVoxel { addr, .. }
        | WorldRequest::Subscribe { addr, .. }
        | WorldRequest::SubscribeMetric { addr, .. }
        | WorldRequest::WriteRegion { addr, .. }
        | WorldRequest::TraversePortal { addr, .. } => *addr,
        WorldRequest::GetVehicleFrame { addr } | WorldRequest::SetVehicleFrame { addr, .. } => {
            Address::Vehicle(*addr)
        }
        WorldRequest::UpdateObserverPos { .. } | WorldRequest::Unsubscribe { .. } => env.from,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-entity actor (handles both worlds and vehicles).
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) enum WorldMsg {
    Request {
        env: Envelope<WorldRequest>,
        reply: oneshot::Sender<Result<Envelope<WorldEvent>, HostError>>,
    },
    SubscribeBegin {
        env: Envelope<WorldRequest>,
        sink: mpsc::Sender<Envelope<WorldEvent>>,
        ready: oneshot::Sender<Result<(), HostError>>,
    },
}

struct Subscriber {
    region: AABB,
    sink: mpsc::Sender<Envelope<WorldEvent>>,
    /// Optional metric-subscription bookkeeping. When `Some`, the actor
    /// recomputes the streaming ring on `UpdateObserverPos` and emits new
    /// bricks; the static `Subscribe` variant leaves this `None`.
    metric: Option<MetricSubState>,
}

#[derive(Clone)]
struct MetricSubState {
    policy: atomr_worlds_proto::StreamingPolicy,
    last_observer: DVec3,
    /// Bricks the subscriber has already received a `BrickSnapshot` for.
    /// Used to compute the additive delta when the ring moves.
    sent: std::collections::HashSet<IVec3>,
}

/// Subscribers interested in vehicle frame deltas (no spatial region — they
/// receive every pose change on the addressed vehicle).
struct FrameSubscriber {
    sink: mpsc::Sender<Envelope<WorldEvent>>,
}

pub(crate) struct WorldActor {
    addr: Address,
    seed: u64,
    /// Resolved policy + generator (or `Empty` short-circuit). Cached for the
    /// actor's lifetime.
    resolved: Resolved,
    /// World shape resolved on spawn. Drives out-of-shape brick filtering and
    /// horizon-distance computation for spherical worlds.
    shape: WorldShape,
    /// Pre-computed geologic / climate / biome state. `None` for cubic
    /// worlds (legacy behavior) or when the host config disabled macro
    /// pre-sim.
    macro_state: Option<Arc<WorldMacroState>>,
    /// Shared per-host registry of authored regions (Phase 13d). Each
    /// brick miss consults this store and overlays any regions whose
    /// bounds intersect the brick.
    authored_regions: Arc<std::sync::Mutex<AuthoredRegionStore>>,
    cache: HashMap<IVec3, Brick>,
    /// Voxel-position → user-written voxel. Mirrors what's in the journal so
    /// brick cache misses can be repopulated correctly post-recovery.
    overlay: HashMap<IVec3, Voxel>,
    subscribers: HashMap<u64, Subscriber>,
    frame_subscribers: HashMap<u64, FrameSubscriber>,
    persistence: Option<Arc<WorldPersistence>>,
    next_seq: u64,
    writes_since_snapshot: u64,
    /// Present only when `addr` is a vehicle.
    frame: Option<AffineFrame>,
    frame_tick: u64,
}

impl WorldActor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        addr: Address,
        seed: u64,
        resolved: Resolved,
        overlay: HashMap<IVec3, Voxel>,
        last_seq: u64,
        persistence: Option<Arc<WorldPersistence>>,
        frame: Option<AffineFrame>,
        shape: WorldShape,
        macro_state: Option<Arc<WorldMacroState>>,
        authored_regions: Arc<std::sync::Mutex<AuthoredRegionStore>>,
    ) -> Self {
        Self {
            addr,
            seed,
            resolved,
            shape,
            macro_state,
            authored_regions,
            cache: HashMap::new(),
            overlay,
            subscribers: HashMap::new(),
            frame_subscribers: HashMap::new(),
            persistence,
            next_seq: last_seq + 1,
            writes_since_snapshot: 0,
            frame,
            frame_tick: 0,
        }
    }

    /// True if any voxel of the given brick could lie inside the shape.
    /// Cheap rejection: if the brick AABB (in world meters, centered) is
    /// entirely outside the shape's bounding AABB AND its closest corner
    /// to the origin still fails `contains`, the brick is empty.
    fn brick_inside_shape(&self, brick_coord: IVec3) -> bool {
        let edge = BRICK_EDGE as i64;
        let scale = self.brush_scale();
        let mpv = scale.meters_per_voxel(atomr_worlds_core::Lod::new(scale.max_depth));
        let bx = brick_coord.x as f64 * edge as f64 * mpv;
        let by = brick_coord.y as f64 * edge as f64 * mpv;
        let bz = brick_coord.z as f64 * edge as f64 * mpv;
        let bxe = bx + edge as f64 * mpv;
        let bye = by + edge as f64 * mpv;
        let bze = bz + edge as f64 * mpv;
        // The world's geometric center sits at scale.root_size_m / 2 in
        // world voxel-meters (matches the existing `apply_region` math).
        let cx = scale.root_size_m * 0.5;
        let cy = scale.root_size_m * 0.5;
        let cz = scale.root_size_m * 0.5;
        // Bounding-AABB reject: pick the corner closest to the world center
        // along each axis. If that nearest point is outside the shape, the
        // whole brick is outside.
        let nearest_x = (cx).clamp(bx, bxe);
        let nearest_y = (cy).clamp(by, bye);
        let nearest_z = (cz).clamp(bz, bze);
        let rel = atomr_worlds_core::DVec3::new(nearest_x - cx, nearest_y - cy, nearest_z - cz);
        // Conservative: if the nearest point is inside, brick is in. If the
        // *furthest* corner is outside, brick is definitely outside. Between
        // those, we conservatively say "inside" (cheap reject only).
        if self.shape.contains(rel) {
            return true;
        }
        // Furthest corner from center along each axis.
        let far_x = if (bx - cx).abs() > (bxe - cx).abs() { bx } else { bxe };
        let far_y = if (by - cy).abs() > (bye - cy).abs() { by } else { bye };
        let far_z = if (bz - cz).abs() > (bze - cz).abs() { bz } else { bze };
        let far = atomr_worlds_core::DVec3::new(far_x - cx, far_y - cy, far_z - cz);
        self.shape.contains(far)
    }

    fn ensure_brick(&mut self, brick_coord: IVec3) -> &mut Brick {
        if !self.cache.contains_key(&brick_coord) {
            let mut b = if !self.brick_inside_shape(brick_coord) {
                // Brick is entirely outside the world's shape — skip the
                // generator entirely. Empty brick fills the cache so we
                // don't repeat the check on subsequent reads.
                Brick::new()
            } else {
                let ctx = BrickGenContext {
                    world_seed: self.seed,
                    brick_coord,
                    shape: self.shape,
                    macro_state: self.macro_state.clone(),
                    scale: self.brush_scale(),
                };
                match &self.resolved {
                    Resolved::Generate { gen, .. } => gen.generate_brick(&ctx),
                    Resolved::Empty => Brick::new(),
                }
            };
            // Apply authored regions (Phase 13d). Each registered region
            // whose AABB intersects this brick overlays its voxels on
            // top of the procedural fill. Iteration order is sorted by
            // region id — deterministic across runs.
            {
                let store = self.authored_regions.lock().unwrap();
                if !store.is_empty() {
                    let _ = store.apply_all(brick_coord, BRICK_EDGE as i64, &mut b);
                }
            }
            // Apply any user-write overlay falling inside this brick.
            let edge = BRICK_EDGE as i64;
            let origin = IVec3::new(brick_coord.x * edge, brick_coord.y * edge, brick_coord.z * edge);
            for (pos, voxel) in &self.overlay {
                if pos.x >= origin.x
                    && pos.x < origin.x + edge
                    && pos.y >= origin.y
                    && pos.y < origin.y + edge
                    && pos.z >= origin.z
                    && pos.z < origin.z + edge
                {
                    let lc = IVec3::new(pos.x - origin.x, pos.y - origin.y, pos.z - origin.z);
                    b.set(lc, *voxel);
                }
            }
            self.cache.insert(brick_coord, b);
        }
        self.cache.get_mut(&brick_coord).unwrap()
    }

    fn brick_of_voxel(p: IVec3) -> (IVec3, IVec3) {
        let edge = BRICK_EDGE as i64;
        let bc = IVec3::new(p.x.div_euclid(edge), p.y.div_euclid(edge), p.z.div_euclid(edge));
        let lc = IVec3::new(p.x.rem_euclid(edge), p.y.rem_euclid(edge), p.z.rem_euclid(edge));
        (bc, lc)
    }

    fn snapshot(&mut self, brick_coord: IVec3) -> bytes::Bytes {
        let b = self.ensure_brick(brick_coord);
        bytes::Bytes::from(b.to_bytes())
    }

    async fn handle_request(
        &mut self,
        env: Envelope<WorldRequest>,
    ) -> Result<Envelope<WorldEvent>, HostError> {
        let corr = env.corr_id;
        let from = env.from;
        match env.body {
            WorldRequest::GetVoxel { addr, pos } => {
                let (bc, lc) = Self::brick_of_voxel(pos);
                let voxel = self.ensure_brick(bc).get(lc);
                Ok(Envelope::new(corr, from, WorldEvent::Voxel { addr, pos, voxel }))
            }
            WorldRequest::GetBrick { addr, brick, lod } => {
                let payload = self.snapshot(brick);
                Ok(Envelope::new(corr, from, WorldEvent::BrickSnapshot { addr, brick, lod, payload }))
            }
            WorldRequest::WriteVoxel { addr, pos, voxel } => {
                let (bc, lc) = Self::brick_of_voxel(pos);
                let before = self.ensure_brick(bc).get(lc);
                if let Some(p) = &self.persistence {
                    let ev = VoxelWriteEvent { addr, pos, before, after: voxel };
                    p.append(addr, &ev, self.next_seq)
                        .await
                        .map_err(|e| HostError::Sys(format!("{e}")))?;
                    self.next_seq += 1;
                    self.writes_since_snapshot += 1;
                }
                {
                    let b = self.ensure_brick(bc);
                    b.set(lc, voxel);
                }
                if voxel == Voxel::EMPTY {
                    self.overlay.remove(&pos);
                } else {
                    self.overlay.insert(pos, voxel);
                }
                self.fan_out_delta(addr, pos, before, voxel);
                self.maybe_save_snapshot().await?;
                Ok(Envelope::new(corr, from, WorldEvent::Ack { addr }))
            }
            WorldRequest::Subscribe { .. } => {
                Err(HostError::NotYetImplemented("use WorldHost::subscribe for Subscribe envelopes"))
            }
            WorldRequest::Unsubscribe { sub_id } => {
                self.subscribers.remove(&sub_id);
                self.frame_subscribers.remove(&sub_id);
                Ok(Envelope::new(corr, from, WorldEvent::StreamEnd { sub_id }))
            }
            WorldRequest::GetVehicleFrame { addr } => {
                let frame = self.frame.unwrap_or_else(|| AffineFrame::at_origin(addr.parent));
                Ok(Envelope::new(
                    corr,
                    from,
                    WorldEvent::VehicleFrame { addr, frame, tick: self.frame_tick },
                ))
            }
            WorldRequest::SetVehicleFrame { addr, frame } => {
                self.frame = Some(frame);
                self.frame_tick = self.frame_tick.wrapping_add(1);
                self.fan_out_frame_delta(addr, frame, self.frame_tick);
                Ok(Envelope::new(corr, from, WorldEvent::Ack { addr: Address::Vehicle(addr) }))
            }
            WorldRequest::WriteRegion { addr, center, unit, voxel } => {
                let bricks_modified = self.apply_region(addr, center, unit, voxel).await?;
                // Fan out an aggregated RegionDelta to subscribers whose region
                // overlaps any of the touched bricks.
                self.fan_out_region_delta(addr, center, unit, voxel, &bricks_modified);
                Ok(Envelope::new(corr, from, WorldEvent::Ack { addr }))
            }
            WorldRequest::SubscribeMetric { .. } => {
                Err(HostError::NotYetImplemented("use WorldHost::subscribe for SubscribeMetric envelopes"))
            }
            WorldRequest::UpdateObserverPos { sub_id, observer_pos } => {
                // Recompute the metric subscription's ring from the new
                // observer position and emit `BrickSnapshot`s for any newly
                // visible bricks. Out-of-shape bricks short-circuit to
                // empty inside `ensure_brick`. Bricks that left the ring
                // are not actively removed — clients track their own
                // working set; future phases may emit Tier(drop) events.
                self.update_observer_pos(sub_id, observer_pos)?;
                Ok(Envelope::new(corr, from, WorldEvent::Ack { addr: self.addr }))
            }
            WorldRequest::TraversePortal { addr: _, portal_id: _ } => {
                // Portals are author-registered; without a portal registry the
                // actor returns the current address unchanged. A future
                // `WorldActor::portals` map (Phase 12.3 follow-up) replaces
                // this trivial echo.
                Ok(Envelope::new(corr, from, WorldEvent::PortalArrival {
                    dest: self.addr,
                    transform: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }))
            }
        }
    }

    fn brush_scale(&self) -> MetricScale {
        // Conservative default — use the address's tier scale. For per-body
        // overrides callers attach an override via `MetricScaleRegistry`.
        match self.addr {
            Address::World(_) => MetricScale::DEFAULT_WORLD,
            Address::Vehicle(_) => MetricScale::DEFAULT_WORLD, // vehicles use world scale by default
        }
    }

    /// Apply a brush write at world-space `center`. Returns the list of bricks
    /// that received any voxel changes.
    async fn apply_region(
        &mut self,
        addr: Address,
        center: DVec3,
        unit: InteractionUnit,
        voxel: Voxel,
    ) -> Result<Vec<IVec3>, HostError> {
        let scale = self.brush_scale();
        let edge = BRICK_EDGE as i64;
        let affected = unit.affected_voxels(scale, center, edge);
        let mpv = scale.meters_per_voxel(atomr_worlds_core::Lod::new(scale.max_depth));

        let mut touched = Vec::new();
        for bc in affected.bricks.iter().copied() {
            // Compute the local predicate for this brick.
            let origin_x = bc.x * edge;
            let origin_y = bc.y * edge;
            let origin_z = bc.z * edge;
            let before_count = self.ensure_brick(bc).nonempty_count;
            let mut journaled = Vec::new();
            // Iterate voxels in-brick that the brush touches, journal each
            // change individually so persistence replay reconstructs state
            // (single-voxel events have always been the unit of replay; for
            // very large brushes this is still correct, just bandwidth-heavy
            // — Phase 8.6 verification covers this).
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let local = IVec3::new(x, y, z);
                        let voxel_world_x = (origin_x + x) as f64 * mpv + mpv * 0.5;
                        let voxel_world_y = (origin_y + y) as f64 * mpv + mpv * 0.5;
                        let voxel_world_z = (origin_z + z) as f64 * mpv + mpv * 0.5;
                        let wp = DVec3::new(voxel_world_x, voxel_world_y, voxel_world_z);
                        if unit.contains(center, wp) {
                            let b = self.cache.get_mut(&bc).unwrap();
                            let before = b.get(local);
                            if b.set(local, voxel) {
                                let pos = IVec3::new(origin_x + x, origin_y + y, origin_z + z);
                                if voxel == Voxel::EMPTY {
                                    self.overlay.remove(&pos);
                                } else {
                                    self.overlay.insert(pos, voxel);
                                }
                                journaled.push((pos, before));
                            }
                        }
                    }
                }
            }
            let after_count = self.cache.get(&bc).unwrap().nonempty_count;
            if before_count != after_count || !journaled.is_empty() {
                touched.push(bc);
            }
            // Journal each per-voxel change for replay correctness.
            if let Some(p) = &self.persistence {
                for (pos, before) in journaled {
                    let ev = VoxelWriteEvent { addr, pos, before, after: voxel };
                    p.append(addr, &ev, self.next_seq)
                        .await
                        .map_err(|e| HostError::Sys(format!("{e}")))?;
                    self.next_seq += 1;
                    self.writes_since_snapshot += 1;
                }
            }
        }
        self.maybe_save_snapshot().await?;
        Ok(touched)
    }

    fn fan_out_region_delta(
        &mut self,
        addr: Address,
        center: DVec3,
        unit: InteractionUnit,
        voxel: Voxel,
        bricks: &[IVec3],
    ) {
        if bricks.is_empty() {
            return;
        }
        let mut dead = Vec::new();
        for (sub_id, sub) in &self.subscribers {
            // Cheap intersection: does the subscriber's region overlap any
            // touched brick's AABB? We send the full bricks_modified list to
            // every overlapping subscriber and let the client filter.
            let edge = BRICK_EDGE as i64;
            let overlaps = bricks.iter().any(|bc| {
                let lo = IVec3::new(bc.x * edge, bc.y * edge, bc.z * edge);
                let hi = IVec3::new(lo.x + edge, lo.y + edge, lo.z + edge);
                sub.region.min.x < hi.x
                    && sub.region.max.x > lo.x
                    && sub.region.min.y < hi.y
                    && sub.region.max.y > lo.y
                    && sub.region.min.z < hi.z
                    && sub.region.max.z > lo.z
            });
            if !overlaps {
                continue;
            }
            let ev = WorldEvent::RegionDelta {
                addr,
                center,
                unit,
                voxel,
                bricks_modified: bricks.to_vec(),
            };
            let env = Envelope::new(0, addr, ev);
            if sub.sink.try_send(env).is_err() {
                dead.push(*sub_id);
            }
        }
        for sub_id in dead {
            self.subscribers.remove(&sub_id);
        }
    }

    async fn maybe_save_snapshot(&mut self) -> Result<(), HostError> {
        let Some(p) = self.persistence.clone() else { return Ok(()) };
        let every = p.snapshot_every();
        if every == 0 || self.writes_since_snapshot < every {
            return Ok(());
        }
        let snap = WorldSnapshot { writes: self.overlay.clone() };
        let seq = self.next_seq.saturating_sub(1);
        p.save_snapshot(self.addr, &snap, seq)
            .await
            .map_err(|e| HostError::Sys(format!("{e}")))?;
        self.writes_since_snapshot = 0;
        Ok(())
    }

    /// World-space center of the body, in world voxel-meters. Used by the
    /// brick filter and horizon math to convert observer/brick coordinates
    /// into the centered frame the shape consumes.
    fn world_center(&self) -> DVec3 {
        let scale = self.brush_scale();
        DVec3::new(scale.root_size_m * 0.5, scale.root_size_m * 0.5, scale.root_size_m * 0.5)
    }

    /// Observer altitude above the shape's surface. For a sphere this is
    /// `|observer - center| - radius`. For a cube this is the perpendicular
    /// distance from the cube's surface (positive when outside, zero or
    /// negative inside). For a cylinder this approximates altitude as
    /// distance from the cylindrical axis minus radius.
    fn observer_altitude(&self, observer: DVec3) -> f64 {
        let c = self.world_center();
        let rel = observer - c;
        match self.shape {
            WorldShape::Sphere { radius_m } => rel.length() - radius_m,
            WorldShape::Cylinder { radius_m, .. } => {
                (rel.x * rel.x + rel.z * rel.z).sqrt() - radius_m
            }
            WorldShape::Cube { edge_m } => {
                let h = edge_m * 0.5;
                let dx = rel.x.abs() - h;
                let dy = rel.y.abs() - h;
                let dz = rel.z.abs() - h;
                dx.max(dy).max(dz)
            }
        }
    }

    /// Iterate brick AABB → vector of brick coords. Pulled out so subscribe
    /// and observer-tick paths share one body.
    fn bricks_in_region(region: AABB) -> Vec<IVec3> {
        let edge = BRICK_EDGE as i64;
        let bmin_x = region.min.x.div_euclid(edge);
        let bmax_x = (region.max.x - 1).div_euclid(edge);
        let bmin_y = region.min.y.div_euclid(edge);
        let bmax_y = (region.max.y - 1).div_euclid(edge);
        let bmin_z = region.min.z.div_euclid(edge);
        let bmax_z = (region.max.z - 1).div_euclid(edge);
        let mut out = Vec::new();
        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    out.push(IVec3::new(bx, by, bz));
                }
            }
        }
        out
    }

    fn handle_subscribe_begin(
        &mut self,
        env: Envelope<WorldRequest>,
        sink: mpsc::Sender<Envelope<WorldEvent>>,
    ) -> Result<(), HostError> {
        // Resolve static vs metric subscribe into a shared
        // (addr, region, lod, sub_id, metric) tuple so the snapshot-loop
        // below stays single-bodied.
        let (addr, region, lod, sub_id, metric) = match env.body {
            WorldRequest::Subscribe { addr, region, lod, sub_id } => {
                (addr, region, lod, sub_id, None)
            }
            WorldRequest::SubscribeMetric { addr, observer_pos, policy, sub_id, .. } => {
                let scale = self.brush_scale();
                let edge_m = scale.meters_per_voxel(atomr_worlds_core::Lod::new(scale.max_depth))
                    * BRICK_EDGE as f64;
                // Horizon-clamp the ring against the shape's horizon
                // distance at the observer's current altitude. Cubes
                // produce `f64::INFINITY` (unclamped); spheres truncate
                // the ring at the visible surface.
                let altitude = self.observer_altitude(observer_pos);
                let horizon = self.shape.horizon_distance_m(altitude.max(0.0));
                let plan = policy.ring_for_curved(observer_pos, edge_m, horizon);
                (
                    addr,
                    plan.near_bricks,
                    plan.near_lod,
                    sub_id,
                    Some(MetricSubState {
                        policy,
                        last_observer: observer_pos,
                        sent: std::collections::HashSet::new(),
                    }),
                )
            }
            _ => return Err(HostError::NotYetImplemented("SubscribeBegin requires Subscribe variants")),
        };

        let corr = env.corr_id;
        let from = env.from;
        let bricks = Self::bricks_in_region(region);
        // Tier event up-front for metric subscriptions — tells clients the
        // LOD + AABB they're about to receive bricks for. (No-op for static.)
        if metric.is_some() {
            let ev = WorldEvent::Tier { sub_id, addr, lod, region };
            let env_out = Envelope::new(corr, from, ev);
            if sink.try_send(env_out).is_err() {
                return Err(HostError::SubscribeFailed);
            }
        }

        let mut sent_set = std::collections::HashSet::new();
        for bc in &bricks {
            let payload = self.snapshot(*bc);
            let ev = WorldEvent::BrickSnapshot { addr, brick: *bc, lod, payload };
            let env_out = Envelope::new(corr, from, ev);
            if sink.try_send(env_out).is_err() {
                return Err(HostError::SubscribeFailed);
            }
            sent_set.insert(*bc);
        }
        // For vehicle addresses, also seed the subscriber with the current
        // frame so clients see the initial pose alongside the voxel snapshot.
        if let Address::Vehicle(va) = addr {
            let frame = self.frame.unwrap_or_else(|| AffineFrame::at_origin(va.parent));
            let ev = WorldEvent::VehicleFrame { addr: va, frame, tick: self.frame_tick };
            let env_out = Envelope::new(corr, from, ev);
            if sink.try_send(env_out).is_err() {
                return Err(HostError::SubscribeFailed);
            }
            self.frame_subscribers.insert(sub_id, FrameSubscriber { sink: sink.clone() });
        }
        let metric_with_sent = metric.map(|m| MetricSubState { sent: sent_set, ..m });
        self.subscribers
            .insert(sub_id, Subscriber { region, sink, metric: metric_with_sent });
        Ok(())
    }

    fn update_observer_pos(&mut self, sub_id: u64, observer_pos: DVec3) -> Result<(), HostError> {
        // Snapshot the metric state we need and drop the borrow before we
        // mutate the actor state below (snapshot/sent updates).
        let (policy, region, lod, addr, mut sent, sink) = {
            let Some(sub) = self.subscribers.get(&sub_id) else { return Ok(()) };
            let Some(metric) = sub.metric.as_ref() else { return Ok(()) };
            let scale = self.brush_scale();
            let edge_m = scale.meters_per_voxel(atomr_worlds_core::Lod::new(scale.max_depth))
                * BRICK_EDGE as f64;
            let altitude = self.observer_altitude(observer_pos);
            let horizon = self.shape.horizon_distance_m(altitude.max(0.0));
            let plan = metric.policy.ring_for_curved(observer_pos, edge_m, horizon);
            (
                metric.policy,
                plan.near_bricks,
                plan.near_lod,
                self.addr,
                metric.sent.clone(),
                sub.sink.clone(),
            )
        };

        // Update the subscriber's region so writes inside the new ring fan
        // out correctly.
        if let Some(sub) = self.subscribers.get_mut(&sub_id) {
            sub.region = region;
        }

        // Emit a Tier event so clients can re-plan their working set.
        let env_tier = Envelope::new(0, addr, WorldEvent::Tier { sub_id, addr, lod, region });
        if sink.try_send(env_tier).is_err() {
            self.subscribers.remove(&sub_id);
            return Ok(());
        }

        let bricks = Self::bricks_in_region(region);
        for bc in bricks {
            if sent.contains(&bc) {
                continue;
            }
            let payload = self.snapshot(bc);
            let ev = WorldEvent::BrickSnapshot { addr, brick: bc, lod, payload };
            let env_out = Envelope::new(0, addr, ev);
            if sink.try_send(env_out).is_err() {
                self.subscribers.remove(&sub_id);
                return Ok(());
            }
            sent.insert(bc);
        }

        // Write back updated state.
        if let Some(sub) = self.subscribers.get_mut(&sub_id) {
            if let Some(m) = sub.metric.as_mut() {
                m.last_observer = observer_pos;
                m.sent = sent;
                m.policy = policy;
            }
        }
        Ok(())
    }

    fn fan_out_delta(&mut self, addr: Address, pos: IVec3, before: Voxel, after: Voxel) {
        let mut dead = Vec::new();
        for (sub_id, sub) in &self.subscribers {
            if sub.region.contains(pos) {
                let ev = WorldEvent::VoxelDelta { addr, pos, before, after };
                let env = Envelope::new(0, addr, ev);
                if sub.sink.try_send(env).is_err() {
                    dead.push(*sub_id);
                }
            }
        }
        for sub_id in dead {
            self.subscribers.remove(&sub_id);
        }
    }

    fn fan_out_frame_delta(&mut self, addr: VehicleAddr, frame: AffineFrame, tick: u64) {
        let mut dead = Vec::new();
        for (sub_id, sub) in &self.frame_subscribers {
            let ev = WorldEvent::VehicleFrameDelta { addr, frame, tick };
            let env = Envelope::new(0, Address::Vehicle(addr), ev);
            if sub.sink.try_send(env).is_err() {
                dead.push(*sub_id);
            }
        }
        for sub_id in dead {
            self.frame_subscribers.remove(&sub_id);
        }
    }
}

// Suppress unused-import lint for symbols that exist primarily for downstream
// crates to reach in via the re-exports surface area.
#[allow(dead_code)]
fn _unused_addr_imports(_: WorldAddr, _: Level, _: ParentAddr) {}

#[async_trait]
impl Actor for WorldActor {
    type Msg = WorldMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: WorldMsg) {
        match msg {
            WorldMsg::Request { env, reply } => {
                let result = self.handle_request(env).await;
                let _ = reply.send(result);
            }
            WorldMsg::SubscribeBegin { env, sink, ready } => {
                let result = self.handle_subscribe_begin(env, sink);
                let _ = ready.send(result);
            }
        }
    }
}
