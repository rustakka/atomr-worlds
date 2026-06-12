//! In-process host on top of atomr's `ActorSystem`.
//!
//! One `WorldActor` is spawned per [`Address`] (lazily, on first request).
//! The actor owns a brick cache, the subscriber registry, an optional vehicle
//! pose, and — when configured — an `atomr-persistence` binding for durable
//! write replay across restarts. Worlds and vehicles share the same actor
//! type; the actor branches on its `Address` variant.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr::core::actor::scheduler::SchedulerHandle;
use atomr::prelude::*;
use atomr_worlds_core::addr::{Address, Level, WorldAddr};
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::interaction::InteractionUnit;
use atomr_worlds_core::lod::{Lod, MetricScale};
use atomr_worlds_core::lww::{LwwMap, LwwStamp, WriterId};
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_core::vehicle::{AffineFrame, ParentAddr, VehicleAddr};
use atomr_worlds_core::HlcTimestamp;
use atomr_worlds_generate::{
    default_registry, AuthoredRegionStore, BrickGenContext, DefaultMacroGenerator,
    GeneratorRegistry, MacroGenerator, MacroStateCache, Resolved, WorldGen, WorldMacroState,
};
use atomr_worlds_persist::{LwwCell, VoxelWriteEvent, WorldPersistence, WorldSnapshot};
use atomr_worlds_physics::debris_sim::{step_body, SimParams, SimState};
use atomr_worlds_physics::{bake_island_grid, connected_components, DebrisBody};
use atomr_worlds_proto::{
    DebrisStateDelta, Envelope, Force, FractureApplied, FractureCommand, FractureRequest,
    WorldEvent, WorldRequest, WriteRejected, AABB,
};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use crate::clock::Clock;
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
    /// Wall-clock source the actor reads when stamping a write's HLC. Defaults
    /// to [`Clock::Wall`]; determinism tests and the screenshot harness inject
    /// [`Clock::manual`](crate::clock::Clock::manual) for reproducible journals.
    pub clock: crate::clock::Clock,
    /// Whether the actor runs the host-authoritative debris simulation (Rec 4
    /// Slice 2): on fracture it builds rigid bodies, steps them on a periodic
    /// self-tick, and fans `WorldEvent::DebrisStates` out to subscribers.
    /// Defaults to `true`. Debris is derived/ephemeral and never feeds
    /// `GetBrick`, so this never affects determinism goldens — but golden
    /// harnesses can set it `false` to skip the registry + timer entirely.
    pub debris_sim_enabled: bool,
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
            clock: crate::clock::Clock::default(),
            debris_sim_enabled: true,
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

    /// In-process actor system the host runs on. Exposed so cluster
    /// wiring can spawn auxiliary actors (e.g. reply inboxes) on the
    /// same system as the world entity actors.
    pub fn actor_system(&self) -> &ActorSystem {
        &self.sys
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

        // Recover overlay before spawning so the actor starts coherent. Seed
        // the actor's HLC above all persisted history so it never regresses.
        let (overlay, last_seq, last_hlc) = if let Some(p) = &self.config.persistence {
            let r = p.recover(addr).await.map_err(|e| HostError::Sys(format!("{e}")))?;
            let ts = r.max_ts();
            (r.overlay, r.last_seq, ts)
        } else {
            (LwwMap::new(), 0, HlcTimestamp::ZERO)
        };

        let name = format!("entity-{:x}-{}", seed, map.len());
        let persistence = self.config.persistence.clone();
        let authored_regions = self.config.authored_regions.clone();
        let clock = self.config.clock.clone();
        let debris_sim_enabled = self.config.debris_sim_enabled;
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
                        overlay.clone(),
                        last_seq,
                        last_hlc,
                        clock.clone(),
                        persistence.clone(),
                        initial_frame,
                        shape,
                        macro_state.clone(),
                        authored_regions.clone(),
                        debris_sim_enabled,
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
        | WorldRequest::WriteVoxelStamped { addr, .. }
        | WorldRequest::WriteRegionStamped { addr, .. }
        | WorldRequest::Subscribe { addr, .. }
        | WorldRequest::SubscribeMetric { addr, .. }
        | WorldRequest::WriteRegion { addr, .. }
        | WorldRequest::TraversePortal { addr, .. } => *addr,
        WorldRequest::Fracture(req) => req.addr,
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
    /// Self-scheduled debris simulation tick (Rec 4 Slice 2). Crate-private and
    /// never serialized — adding it doesn't touch the wire protocol.
    Tick,
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
    /// Procedural-brick cache keyed by `(brick_coord, lod_depth)`. Each
    /// LOD tier hits the generator independently — coarse-LOD bricks
    /// downsample procedural noise to their voxel scale, so caching at
    /// a single key would silently re-use LOD-0 content stretched to
    /// fit, which was the original "stair-step" terrain bug at the
    /// streaming horizon. Write paths use `Lod::new(0)`.
    cache: HashMap<(IVec3, u8), Brick>,
    /// Voxel-position → last-writer-wins voxel cell. Mirrors the journal so
    /// brick cache misses repopulate correctly post-recovery. Each cell carries
    /// its HLC stamp, so concurrent/out-of-order writes converge and an `EMPTY`
    /// (carve) is a retained tombstone rather than an absence.
    overlay: LwwMap<IVec3, Voxel>,
    /// This actor's stable writer id (non-zero), used to break HLC ties in the
    /// LWW order. Derived deterministically from the world seed.
    writer_id: WriterId,
    /// Most recent HLC this actor stamped or received — its logical clock.
    last_hlc: HlcTimestamp,
    /// Wall-clock source for HLC stamping (real time, or a deterministic
    /// counter under test/harness).
    clock: Clock,
    subscribers: HashMap<u64, Subscriber>,
    frame_subscribers: HashMap<u64, FrameSubscriber>,
    persistence: Option<Arc<WorldPersistence>>,
    next_seq: u64,
    writes_since_snapshot: u64,
    /// Present only when `addr` is a vehicle.
    frame: Option<AffineFrame>,
    frame_tick: u64,
    // ── Host-authoritative debris (Rec 4 Slice 2) ──────────────────────────
    /// Whether this actor runs the debris simulation at all (config-gated).
    debris_sim_enabled: bool,
    /// Active debris bodies keyed by their stable id (`debris_id(anchor)`,
    /// shared with the `FractureCommand::SpawnDebris` id the client baked from).
    debris: HashMap<u32, DebrisEntry>,
    /// Monotonic debris tick counter; stamps each `DebrisStateDelta`.
    debris_tick: u64,
    /// Whether a self-tick is currently armed (so we don't double-arm).
    debris_ticking: bool,
    /// Set by `handle_fracture` when it adds debris; `Actor::handle` arms the
    /// tick afterward (it has the `Context` the scheduler needs).
    want_arm: bool,
    /// Integrator tunables (shared across this actor's bodies).
    sim_params: SimParams,
    /// Handle to the pending self-tick timer, cancelled on `post_stop`.
    sim_handle: Option<SchedulerHandle>,
}

/// One active debris body plus its integrator + retirement bookkeeping.
struct DebrisEntry {
    body: DebrisBody,
    state: SimState,
    /// Consecutive ticks the body has been asleep — retired past the grace.
    retire_ticks: u32,
}

/// Debris simulation cadence (≈30 Hz). One source for both the timer `Duration`
/// and the integrator `dt`, so they stay locked.
const DEBRIS_TICK_MS: u64 = 33;
const DEBRIS_TICK_S: f64 = 1.0 / 30.0;
/// Voxel edge for debris bodies: the render/index grid is 1 m/voxel (the same
/// integer grid `WriteVoxel`/fracture positions use — distinct from the host
/// brush `mpv`), so debris poses stream in metres that equal voxel coordinates.
const DEBRIS_VOXEL_SIZE_M: f64 = 1.0;
/// Ticks a sleeping body lingers before retirement (~2 s at 30 Hz).
const RETIRE_GRACE_TICKS: u32 = 60;
/// World-metre floor below which debris is retired (fell out of the world).
const MIN_DEBRIS_Y: f64 = -512.0;
/// Voxel padding around each body when snapshotting terrain solidity.
const DEBRIS_SOLIDITY_PAD: i64 = 4;

impl WorldActor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        addr: Address,
        seed: u64,
        resolved: Resolved,
        overlay: LwwMap<IVec3, Voxel>,
        last_seq: u64,
        last_hlc: HlcTimestamp,
        clock: Clock,
        persistence: Option<Arc<WorldPersistence>>,
        frame: Option<AffineFrame>,
        shape: WorldShape,
        macro_state: Option<Arc<WorldMacroState>>,
        authored_regions: Arc<std::sync::Mutex<AuthoredRegionStore>>,
        debris_sim_enabled: bool,
    ) -> Self {
        // Deterministic, stable, non-zero writer id so a local write always
        // beats a migrated legacy entry (`WriterId::LEGACY == 0`) at the same
        // timestamp, and two distinct worlds don't share an id.
        let writer_id = WriterId(atomr_worlds_core::seed::splitmix64(seed) | 1);
        Self {
            addr,
            seed,
            resolved,
            shape,
            macro_state,
            authored_regions,
            cache: HashMap::new(),
            overlay,
            writer_id,
            last_hlc,
            clock,
            subscribers: HashMap::new(),
            frame_subscribers: HashMap::new(),
            persistence,
            next_seq: last_seq + 1,
            writes_since_snapshot: 0,
            frame,
            frame_tick: 0,
            debris_sim_enabled,
            debris: HashMap::new(),
            debris_tick: 0,
            debris_ticking: false,
            want_arm: false,
            sim_params: SimParams::default(),
            sim_handle: None,
        }
    }

    /// Stamp the next locally-originated write: tick the HLC off the injected
    /// clock and pair it with this actor's writer id. Strictly monotonic, so a
    /// sequential single writer always wins LWW (i.e. byte-identical to the old
    /// unconditional overwrite).
    fn next_stamp(&mut self) -> LwwStamp {
        self.last_hlc = HlcTimestamp::tick(self.last_hlc, self.clock.now_ns());
        LwwStamp::new(self.last_hlc, self.writer_id)
    }

    /// Apply a stamped single-voxel write under last-writer-wins. Returns
    /// `Ok(None)` when applied (journalled + overlaid + fanned out), or
    /// `Ok(Some(current))` when the write lost to a greater-or-equal stamp (the
    /// caller replies [`WorldEvent::WriteRejected`]). Preserves the
    /// append-before-mutate-cache ordering of the original write path.
    async fn apply_stamped_write(
        &mut self,
        addr: Address,
        pos: IVec3,
        voxel: Voxel,
        stamp: LwwStamp,
    ) -> Result<Option<Voxel>, HostError> {
        // Non-committing LWW peek: bail before journalling if we'd lose.
        if let Some(cur) = self.overlay.get_entry(&pos).map(|(s, v)| (s, *v)) {
            if cur.0 >= stamp {
                return Ok(Some(cur.1));
            }
        }
        let (bc, lc) = Self::brick_of_voxel(pos);
        let before = self.ensure_brick(bc, Lod::new(0)).get(lc);
        if let Some(p) = &self.persistence {
            let ev = VoxelWriteEvent { addr, pos, before, after: voxel, stamp };
            p.append(addr, &ev, self.next_seq)
                .await
                .map_err(|e| HostError::Sys(format!("{e}")))?;
            self.next_seq += 1;
            self.writes_since_snapshot += 1;
        }
        self.overlay.put(pos, voxel, stamp);
        {
            let b = self.ensure_brick(bc, Lod::new(0));
            b.set(lc, voxel);
        }
        self.invalidate_coarse_caches_for(pos);
        self.fan_out_delta(addr, pos, before, voxel);
        self.maybe_save_snapshot().await?;
        Ok(None)
    }

    /// True if any voxel of the given brick could lie inside the shape.
    /// Cheap rejection: if the brick AABB (in centered world meters) is
    /// entirely outside the shape's bounding AABB AND its closest corner
    /// to the origin still fails `contains`, the brick is empty.
    ///
    /// Coordinate convention: brick coords are signed integers centered
    /// on the world origin (i.e. brick (0,0,0) straddles world origin,
    /// brick (-1,…,…) is the brick to the -X of origin). `WorldShape`
    /// itself is centered on the origin (see `shape.rs::contains`), so
    /// the brick AABB maps directly to "centered point" without any
    /// further offset. Previously this routine offset by `root_size_m/2`,
    /// which classified any brick at a negative world coord as outside
    /// the shape — that caused the asymmetric "only +X+Z renders" bug
    /// when the FP camera streamed bricks symmetrically around its
    /// origin-anchored observer.
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
        // Nearest point on the brick AABB to the world origin.
        let nearest_x = 0.0_f64.clamp(bx, bxe);
        let nearest_y = 0.0_f64.clamp(by, bye);
        let nearest_z = 0.0_f64.clamp(bz, bze);
        let near = atomr_worlds_core::DVec3::new(nearest_x, nearest_y, nearest_z);
        if self.shape.contains(near) {
            return true;
        }
        // Furthest brick corner from origin along each axis.
        let far_x = if bx.abs() > bxe.abs() { bx } else { bxe };
        let far_y = if by.abs() > bye.abs() { by } else { bye };
        let far_z = if bz.abs() > bze.abs() { bz } else { bze };
        let far = atomr_worlds_core::DVec3::new(far_x, far_y, far_z);
        self.shape.contains(far)
    }

    fn ensure_brick(&mut self, brick_coord: IVec3, lod: Lod) -> &mut Brick {
        let key = (brick_coord, lod.depth);
        if !self.cache.contains_key(&key) {
            let mut b = if !self.brick_inside_shape(brick_coord) {
                // Brick is entirely outside the world's shape — skip the
                // generator entirely. Empty brick fills the cache so we
                // don't repeat the check on subsequent reads.
                Brick::new()
            } else {
                let ctx = BrickGenContext {
                    world_seed: self.seed,
                    brick_coord,
                    lod,
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
            // region id — deterministic across runs. Authored regions
            // are LOD-0 only; skip the overlay at coarser LODs so they
            // aren't stamped at the wrong scale.
            if lod.depth == 0 {
                let store = self.authored_regions.lock().unwrap();
                if !store.is_empty() {
                    let _ = store.apply_all(brick_coord, BRICK_EDGE as i64, &mut b);
                }
            }
            // Apply user-write overlay falling inside this brick. At
            // LOD 0 the overlay is voxel-resolution: writes (including
            // EMPTY) overwrite cells one-for-one. At coarser LODs each
            // brick cell represents a `2^L` cube of LOD-0 voxels, so a
            // single user write maps to one cell — we re-stamp non-empty
            // writes (this Phase 17.1 follow-up) so a built voxel still
            // shows up after the observer crosses past the LOD transition
            // radius. EMPTY writes (carving) are left to the LOD-0 path:
            // a single LOD-0 hole shouldn't blank the whole coarse cell
            // when 2^(3L)-1 other LOD-0 voxels in the cell are still
            // procedurally solid; the carved hole reappears when the
            // observer returns to the near ring.
            let lod_scale = 1i64 << lod.depth;
            let edge = BRICK_EDGE as i64;
            let edge_world = edge * lod_scale;
            let origin = IVec3::new(
                brick_coord.x * edge_world,
                brick_coord.y * edge_world,
                brick_coord.z * edge_world,
            );
            for (pos, _stamp, voxel) in self.overlay.iter() {
                if pos.x < origin.x
                    || pos.x >= origin.x + edge_world
                    || pos.y < origin.y
                    || pos.y >= origin.y + edge_world
                    || pos.z < origin.z
                    || pos.z >= origin.z + edge_world
                {
                    continue;
                }
                // At LOD 0 an `EMPTY` tombstone carves the procedural cell (the
                // carve-durability fix); at coarse LODs a single hole shouldn't
                // blank the whole downsampled cell, so skip it there.
                if lod.depth > 0 && *voxel == Voxel::EMPTY {
                    continue;
                }
                let lc = IVec3::new(
                    (pos.x - origin.x).div_euclid(lod_scale),
                    (pos.y - origin.y).div_euclid(lod_scale),
                    (pos.z - origin.z).div_euclid(lod_scale),
                );
                b.set(lc, *voxel);
            }
            self.cache.insert(key, b);
        }
        self.cache.get_mut(&key).unwrap()
    }

    fn brick_of_voxel(p: IVec3) -> (IVec3, IVec3) {
        let edge = BRICK_EDGE as i64;
        let bc = IVec3::new(p.x.div_euclid(edge), p.y.div_euclid(edge), p.z.div_euclid(edge));
        let lc = IVec3::new(p.x.rem_euclid(edge), p.y.rem_euclid(edge), p.z.rem_euclid(edge));
        (bc, lc)
    }

    /// Drop every cached coarse-LOD brick whose world-voxel footprint
    /// contains `pos`. The next access to that brick re-runs the
    /// generator and re-applies the (now-updated) overlay through
    /// `ensure_brick`, so a write made inside the near ring is reflected
    /// the moment the observer crosses past the LOD transition radius.
    /// LOD-0 entries are left alone — the in-place `set` on the LOD-0
    /// brick keeps them coherent without a refetch. Phase 17.1 follow-up.
    fn invalidate_coarse_caches_for(&mut self, pos: IVec3) {
        let edge = BRICK_EDGE as i64;
        self.cache.retain(|(bc, depth), _| {
            if *depth == 0 {
                return true;
            }
            let lod_scale = 1i64 << *depth;
            let edge_world = edge * lod_scale;
            let lo_x = bc.x * edge_world;
            let lo_y = bc.y * edge_world;
            let lo_z = bc.z * edge_world;
            let inside = pos.x >= lo_x
                && pos.x < lo_x + edge_world
                && pos.y >= lo_y
                && pos.y < lo_y + edge_world
                && pos.z >= lo_z
                && pos.z < lo_z + edge_world;
            !inside
        });
    }

    fn snapshot(&mut self, brick_coord: IVec3, lod: Lod) -> bytes::Bytes {
        let b = self.ensure_brick(brick_coord, lod);
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
                let voxel = self.ensure_brick(bc, Lod::new(0)).get(lc);
                Ok(Envelope::new(corr, from, WorldEvent::Voxel { addr, pos, voxel }))
            }
            WorldRequest::GetBrick { addr, brick, lod } => {
                let payload = self.snapshot(brick, lod);
                Ok(Envelope::new(corr, from, WorldEvent::BrickSnapshot { addr, brick, lod, payload }))
            }
            WorldRequest::WriteVoxel { addr, pos, voxel } => {
                // Host-authoritative write: stamp with a fresh, strictly
                // monotonic HLC tick — so this write always wins LWW (a single
                // sequential writer is byte-identical to the old overwrite).
                let stamp = self.next_stamp();
                match self.apply_stamped_write(addr, pos, voxel, stamp).await? {
                    None => Ok(Envelope::new(corr, from, WorldEvent::Ack { addr })),
                    Some(current) => Ok(Envelope::new(
                        corr,
                        from,
                        WorldEvent::WriteRejected(WriteRejected { addr, pos, current }),
                    )),
                }
            }
            WorldRequest::WriteVoxelStamped { addr, pos, voxel, ts, writer } => {
                // Client-stamped write: advance the actor's clock past the
                // remote stamp (so later local writes dominate), but key the
                // LWW merge on the *client's* stamp.
                self.last_hlc = HlcTimestamp::recv(self.last_hlc, ts, self.clock.now_ns());
                let stamp = LwwStamp::new(ts, writer);
                match self.apply_stamped_write(addr, pos, voxel, stamp).await? {
                    None => Ok(Envelope::new(corr, from, WorldEvent::Ack { addr })),
                    Some(current) => Ok(Envelope::new(
                        corr,
                        from,
                        WorldEvent::WriteRejected(WriteRejected { addr, pos, current }),
                    )),
                }
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
                let stamp = self.next_stamp();
                let bricks_modified =
                    self.apply_region(addr, center, unit, voxel, stamp).await?;
                // Fan out an aggregated RegionDelta to subscribers whose region
                // overlaps any of the touched bricks.
                self.fan_out_region_delta(addr, center, unit, voxel, &bricks_modified);
                Ok(Envelope::new(corr, from, WorldEvent::Ack { addr }))
            }
            WorldRequest::WriteRegionStamped { addr, center, unit, voxel, ts, writer } => {
                self.last_hlc = HlcTimestamp::recv(self.last_hlc, ts, self.clock.now_ns());
                let stamp = LwwStamp::new(ts, writer);
                let bricks_modified =
                    self.apply_region(addr, center, unit, voxel, stamp).await?;
                self.fan_out_region_delta(addr, center, unit, voxel, &bricks_modified);
                Ok(Envelope::new(corr, from, WorldEvent::Ack { addr }))
            }
            WorldRequest::Fracture(req) => {
                let applied = self.handle_fracture(req).await?;
                self.fan_out_fracture_applied(&applied);
                Ok(Envelope::new(corr, from, WorldEvent::FractureApplied(applied)))
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
        stamp: LwwStamp,
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
            let before_count = self.ensure_brick(bc, Lod::new(0)).nonempty_count;
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
                            let pos = IVec3::new(origin_x + x, origin_y + y, origin_z + z);
                            // LWW gate: skip cells already held by a
                            // greater-or-equal stamp (only possible for a
                            // client-stamped brush; a fresh host stamp always
                            // wins). Brush rejections are silent — best-effort.
                            if self.overlay.stamp(&pos).is_some_and(|cur| cur >= stamp) {
                                continue;
                            }
                            let b = self.cache.get_mut(&(bc, 0u8)).unwrap();
                            let before = b.get(local);
                            if b.set(local, voxel) {
                                // Retain `EMPTY` as a tombstone (carve durability
                                // + LWW correctness), unlike the old remove.
                                self.overlay.put(pos, voxel, stamp);
                                journaled.push((pos, before));
                                // Phase 17.1 follow-up: invalidate coarse-LOD
                                // entries containing `pos` so they regenerate
                                // with the new overlay on next access.
                                self.invalidate_coarse_caches_for(pos);
                            }
                        }
                    }
                }
            }
            let after_count = self.cache.get(&(bc, 0u8)).unwrap().nonempty_count;
            if before_count != after_count || !journaled.is_empty() {
                touched.push(bc);
            }
            // Journal each per-voxel change for replay correctness. Every cell
            // of one brush shares the brush's stamp (distinct positions ⇒ no
            // tie), so the batch resolves atomically under LWW.
            if let Some(p) = &self.persistence {
                for (pos, before) in journaled {
                    let ev = VoxelWriteEvent { addr, pos, before, after: voxel, stamp };
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

    /// Authoritatively evaluate a structural fracture: run the integer
    /// connectivity decision over the impact region against the *authoritative*
    /// bricks, journal the removal of any detached island through the LWW
    /// overlay, and return the deterministic command sequence. Float debris
    /// motion stays the client's job — the host only decides geometry.
    async fn handle_fracture(
        &mut self,
        req: FractureRequest,
    ) -> Result<FractureApplied, HostError> {
        /// Half-extent (voxels) of the analysis region around the impact.
        const R: i64 = 8;
        let addr = req.addr;
        let first_seq = self.next_seq;
        let empty = FractureApplied {
            addr,
            commands: Vec::new(),
            seq_range: (first_seq, first_seq.saturating_sub(1)),
        };

        // Yield gate: a *non-zero* impact force below the material's yield does
        // not fracture; a zero force (carve-triggered) always evaluates
        // connectivity. The comparison is integer ⇒ deterministic.
        if req.force != Force::ZERO {
            let yield_pa = atomr_worlds_core::default_physics_palette()
                .get(req.material_id)
                .yield_strength_pa;
            if !force_meets_yield(req.force, yield_pa) {
                return Ok(empty);
            }
        }

        // Bounded region around the impact, read from authoritative bricks.
        let min = IVec3::new(req.impact_pos.x - R, req.impact_pos.y - R, req.impact_pos.z - R);
        let span = (2 * R + 1) as i32;
        let dims: [i32; 3] = [span, span, span];
        let (nx, ny, nz) = (span as i64, span as i64, span as i64);
        let lin = |x: i64, y: i64, z: i64| (x * ny * nz + y * nz + z) as usize;
        let mut grid = vec![Voxel::EMPTY; (nx * ny * nz) as usize];
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let wp = IVec3::new(min.x + x, min.y + y, min.z + z);
                    let (bc, lc) = Self::brick_of_voxel(wp);
                    grid[lin(x, y, z)] = self.ensure_brick(bc, Lod::new(0)).get(lc);
                }
            }
        }

        // Connectivity: solid = non-empty; anchor = solid on the region shell
        // *except* the top (+Y) face — so ceiling-hung structure can fall while
        // ground/side-rooted structure holds. Pure integer ⇒ deterministic.
        let idx = |x: i32, y: i32, z: i32| (x as i64 * ny * nz + y as i64 * nz + z as i64) as usize;
        let islands = {
            let solid = |x: i32, y: i32, z: i32| grid[idx(x, y, z)] != Voxel::EMPTY;
            let is_anchor = |x: i32, y: i32, z: i32| {
                grid[idx(x, y, z)] != Voxel::EMPTY
                    && (x == 0 || x == dims[0] - 1 || z == 0 || z == dims[2] - 1 || y == 0)
            };
            connected_components(dims, solid, is_anchor).unanchored_islands()
        };
        if islands.is_empty() {
            return Ok(empty);
        }

        // One HLC tick stamps the whole fracture (distinct positions ⇒ no tie).
        let stamp = self.next_stamp();
        let persistence = self.persistence.clone();
        let mut commands = Vec::new();
        for island in &islands {
            let mut voxels = Vec::with_capacity(island.len());
            let mut anchor = IVec3::new(i64::MAX, i64::MAX, i64::MAX);
            for &[lx, ly, lz] in island {
                let wp = IVec3::new(min.x + lx as i64, min.y + ly as i64, min.z + lz as i64);
                let before = grid[lin(lx as i64, ly as i64, lz as i64)];
                if let Some(p) = &persistence {
                    let ev = VoxelWriteEvent { addr, pos: wp, before, after: Voxel::EMPTY, stamp };
                    p.append(addr, &ev, self.next_seq)
                        .await
                        .map_err(|e| HostError::Sys(format!("{e}")))?;
                    self.next_seq += 1;
                    self.writes_since_snapshot += 1;
                }
                self.overlay.put(wp, Voxel::EMPTY, stamp);
                let (bc, lc) = Self::brick_of_voxel(wp);
                self.ensure_brick(bc, Lod::new(0)).set(lc, Voxel::EMPTY);
                self.invalidate_coarse_caches_for(wp);
                commands.push(FractureCommand::SetVoxel { pos: wp, before, after: Voxel::EMPTY });
                voxels.push(wp);
                anchor = IVec3::new(anchor.x.min(wp.x), anchor.y.min(wp.y), anchor.z.min(wp.z));
            }
            // Host-authoritative debris: build a rigid body for this island so
            // the actor's self-tick integrates and broadcasts its motion. The
            // body grid is baked from the *pre-carve* materials still held in
            // `grid` (the overlay/brick edits above don't touch `grid`), reusing
            // the same `bake_island_grid` the client's render path
            // (`analyze_region`) uses — so geometry and physics agree. The id is
            // shared with the `SpawnDebris` command the client bakes from.
            let id = debris_id(anchor);
            if self.debris_sim_enabled {
                let (b_origin, b_dims, b_material) = bake_island_grid(island, min, |x, y, z| {
                    grid[lin(x as i64, y as i64, z as i64)].0
                });
                let vs = DEBRIS_VOXEL_SIZE_M;
                let world_origin_m =
                    DVec3::new(b_origin.x as f64 * vs, b_origin.y as f64 * vs, b_origin.z as f64 * vs);
                let palette = atomr_worlds_core::default_physics_palette();
                let mut body =
                    DebrisBody::from_voxels(b_origin, b_dims, b_material, vs, world_origin_m, &palette);
                // Seed velocity from the deterministic fixed-point impact force;
                // a zero-force carve starts at rest and simply falls under
                // gravity. `step_body` clamps to `max_speed` on the first tick.
                let f = req.force.to_newtons();
                let j = DVec3::new(f[0] as f64, f[1] as f64, f[2] as f64); // impulse (N·s)
                let inv_m = if body.mass.mass_kg > 0.0 { 1.0 / body.mass.mass_kg } else { 0.0 };
                body.linear_velocity = DVec3::new(j.x * inv_m, j.y * inv_m, j.z * inv_m);
                // Seed spin from the *off-center* impulse: the impact applies `j`
                // at `impact_pos` rather than at the COM, so it imparts angular
                // momentum `L = r × J` (r = arm from COM). `ω = I⁻¹ L`. At spawn
                // `orientation == IDENTITY`, so body frame == world frame and the
                // body-frame `inertia_inv` applies directly. A zero-force carve
                // (`j == 0`) yields zero spin → the island just falls.
                let impact_world_m = DVec3::new(
                    (req.impact_pos.x as f64 + 0.5) * vs,
                    (req.impact_pos.y as f64 + 0.5) * vs,
                    (req.impact_pos.z as f64 + 0.5) * vs,
                );
                let r = impact_world_m - body.position;
                let l = atomr_worlds_physics::math::cross(r, j);
                body.angular_velocity = body.mass.inertia_inv.mul_vec(l);
                self.debris
                    .insert(id, DebrisEntry { body, state: SimState::default(), retire_ticks: 0 });
            }
            commands.push(FractureCommand::SpawnDebris { id, voxels, anchor });
        }
        self.maybe_save_snapshot().await?;
        // Arm the self-tick if we added bodies and aren't already ticking
        // (`Actor::handle` does the arming — it holds the `Context`).
        if self.debris_sim_enabled && !self.debris.is_empty() && !self.debris_ticking {
            self.want_arm = true;
        }
        let last_seq = self.next_seq.saturating_sub(1);
        Ok(FractureApplied { addr, commands, seq_range: (first_seq, last_seq) })
    }

    /// Fan an authoritative [`FractureApplied`] out to subscribers whose region
    /// overlaps any touched brick — mirrors [`Self::fan_out_region_delta`].
    fn fan_out_fracture_applied(&mut self, applied: &FractureApplied) {
        let edge = BRICK_EDGE as i64;
        let mut bricks: Vec<IVec3> = Vec::new();
        for cmd in &applied.commands {
            if let FractureCommand::SetVoxel { pos, .. } = cmd {
                let (bc, _) = Self::brick_of_voxel(*pos);
                if !bricks.contains(&bc) {
                    bricks.push(bc);
                }
            }
        }
        if bricks.is_empty() {
            return;
        }
        let addr = applied.addr;
        let mut dead = Vec::new();
        for (sub_id, sub) in &self.subscribers {
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
            let env = Envelope::new(0, addr, WorldEvent::FractureApplied(applied.clone()));
            if sub.sink.try_send(env).is_err() {
                dead.push(*sub_id);
            }
        }
        for sub_id in dead {
            self.subscribers.remove(&sub_id);
        }
    }

    // ── Host-authoritative debris (Rec 4 Slice 2) ─────────────────────────

    /// Arm (or re-arm) the periodic debris self-tick. The scheduled closure runs
    /// detached and can't touch `&mut self`, so it only `tell`s the actor a
    /// `WorldMsg::Tick`; the mailbox-delivered handler does the stepping and
    /// re-arms. Called from `Actor::handle`, which holds the `Context`.
    fn arm_tick(&mut self, ctx: &Context<WorldActor>) {
        let Some(sched) = ctx.system_handle().scheduler() else {
            // No scheduler (system torn down) — go idle rather than spin.
            self.debris_ticking = false;
            return;
        };
        let me = ctx.self_ref().clone();
        let handle = sched.schedule_once(
            Duration::from_millis(DEBRIS_TICK_MS),
            Box::pin(async move {
                me.tell(WorldMsg::Tick);
            }),
        );
        self.sim_handle = Some(handle);
        self.debris_ticking = true;
    }

    /// Step every active debris body one tick, retire settled / out-of-bounds
    /// bodies, and return the `DebrisStateDelta`s to broadcast. Free of any
    /// scheduler/`Context` concern. Terrain is read into an owned snapshot first
    /// so the `ensure_brick` `&mut self` borrow is released before the bodies
    /// are stepped.
    fn step_debris(&mut self) -> Vec<DebrisStateDelta> {
        self.debris_tick = self.debris_tick.wrapping_add(1);
        if self.debris.is_empty() {
            return Vec::new();
        }
        let solidity = self.snapshot_debris_solidity();
        let is_solid = |p: IVec3| solidity.contains(&p);

        let params = self.sim_params;
        let tick = self.debris_tick;
        let mut deltas = Vec::new();
        let mut retire: Vec<u32> = Vec::new();
        for (id, entry) in self.debris.iter_mut() {
            let was_sleeping = entry.state.sleeping;
            step_body(&mut entry.body, &mut entry.state, &params, DEBRIS_TICK_S, &is_solid);
            if entry.state.sleeping {
                entry.retire_ticks = entry.retire_ticks.saturating_add(1);
            }
            let out_of_bounds = entry.body.position.y < MIN_DEBRIS_Y;
            let retiring = out_of_bounds || entry.retire_ticks >= RETIRE_GRACE_TICKS;
            // Emit a delta for active bodies, the sleep transition (so clients
            // latch the settled pose), and the terminal retiring frame.
            if !entry.state.sleeping || !was_sleeping || retiring {
                deltas.push(sample_debris_delta(*id, tick, entry));
            }
            if retiring {
                retire.push(*id);
            }
        }
        for id in retire {
            self.debris.remove(&id);
        }
        deltas
    }

    /// Read terrain solidity (world voxels) in a padded neighborhood around each
    /// active body into an owned set, so the integrator's `is_solid` closure
    /// doesn't borrow `self`. Per-body ranges keep the scan bounded even when
    /// bodies drift far apart.
    fn snapshot_debris_solidity(&mut self) -> HashSet<IVec3> {
        let mut ranges: Vec<(IVec3, IVec3)> = Vec::new();
        for entry in self.debris.values() {
            if entry.state.sleeping {
                continue;
            }
            let b = &entry.body;
            let vs = b.voxel_size_m;
            let corner = b.position - b.mass.com;
            let cx = (corner.x / vs).floor() as i64;
            let cy = (corner.y / vs).floor() as i64;
            let cz = (corner.z / vs).floor() as i64;
            let lo = IVec3::new(
                cx - DEBRIS_SOLIDITY_PAD,
                cy - DEBRIS_SOLIDITY_PAD,
                cz - DEBRIS_SOLIDITY_PAD,
            );
            let hi = IVec3::new(
                cx + b.dims[0] as i64 + DEBRIS_SOLIDITY_PAD,
                cy + b.dims[1] as i64 + DEBRIS_SOLIDITY_PAD,
                cz + b.dims[2] as i64 + DEBRIS_SOLIDITY_PAD,
            );
            ranges.push((lo, hi));
        }
        let mut set = HashSet::new();
        for (lo, hi) in ranges {
            for vx in lo.x..=hi.x {
                for vy in lo.y..=hi.y {
                    for vz in lo.z..=hi.z {
                        let p = IVec3::new(vx, vy, vz);
                        if set.contains(&p) {
                            continue;
                        }
                        let (bc, lc) = Self::brick_of_voxel(p);
                        if !self.ensure_brick(bc, Lod::new(0)).get(lc).is_empty() {
                            set.insert(p);
                        }
                    }
                }
            }
        }
        set
    }

    /// Fan a batch of `DebrisStateDelta`s to subscribers whose region overlaps
    /// any active body — mirrors [`Self::fan_out_fracture_applied`].
    fn fan_out_debris_states(&mut self, deltas: Vec<DebrisStateDelta>) {
        if deltas.is_empty() {
            return;
        }
        let edge = BRICK_EDGE as i64;
        // Bricks the debris currently occupy (approx by the streamed corner; a
        // debris body is small, so one brick per body is a fine overlap key).
        let mut bricks: Vec<IVec3> = Vec::new();
        for d in &deltas {
            let p = IVec3::new(
                d.pos[0].floor() as i64,
                d.pos[1].floor() as i64,
                d.pos[2].floor() as i64,
            );
            let (bc, _) = Self::brick_of_voxel(p);
            if !bricks.contains(&bc) {
                bricks.push(bc);
            }
        }
        let addr = self.addr;
        let mut dead = Vec::new();
        for (sub_id, sub) in &self.subscribers {
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
            let env =
                Envelope::new(0, addr, WorldEvent::DebrisStates { addr, deltas: deltas.clone() });
            if sub.sink.try_send(env).is_err() {
                dead.push(*sub_id);
            }
        }
        for sub_id in dead {
            self.subscribers.remove(&sub_id);
        }
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
        // Snapshot the full CRDT state — stamps and tombstones included — so
        // last-writer-wins convergence survives log truncation.
        let writes = self
            .overlay
            .entries()
            .iter()
            .map(|(pos, (stamp, voxel))| (*pos, LwwCell { stamp: *stamp, voxel: *voxel }))
            .collect();
        let snap = WorldSnapshot { writes };
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
            let payload = self.snapshot(*bc, lod);
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
            let payload = self.snapshot(bc, lod);
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

/// Sample a debris body into a wire delta. Slice 1 streams the body's local
/// `(0,0,0)` corner in world metres (`position - com`, with identity
/// orientation), which matches the client's render frame
/// (`island.origin * voxel_size_m`) so the client drives the entity transform
/// directly. (COM + orientation is the rotation-era representation — a
/// follow-up once the integrator tumbles.)
fn sample_debris_delta(id: u32, tick: u64, entry: &DebrisEntry) -> DebrisStateDelta {
    let b = &entry.body;
    let corner = b.position - b.mass.com;
    DebrisStateDelta {
        id,
        tick,
        pos: [corner.x as f32, corner.y as f32, corner.z as f32],
        vel: [
            b.linear_velocity.x as f32,
            b.linear_velocity.y as f32,
            b.linear_velocity.z as f32,
        ],
        orient: [
            b.orientation.x as f32,
            b.orientation.y as f32,
            b.orientation.z as f32,
            b.orientation.w as f32,
        ],
        ang_vel: [
            b.angular_velocity.x as f32,
            b.angular_velocity.y as f32,
            b.angular_velocity.z as f32,
        ],
        sleeping: entry.state.sleeping,
    }
}

/// Deterministic debris id from an island's anchor. The host assigns it and
/// broadcasts the same `FractureApplied` to every client, so this only needs to
/// be stable and reasonably distinct per island location.
fn debris_id(anchor: IVec3) -> u32 {
    use atomr_worlds_core::seed::splitmix64;
    let mut h = splitmix64(anchor.x as u64);
    h = splitmix64(h ^ (anchor.y as u64).rotate_left(21));
    h = splitmix64(h ^ (anchor.z as u64).rotate_left(42));
    h as u32
}

/// Whether a fixed-point impact `force` meets a material's yield. The force
/// magnitude is integer (milli-newtons); the threshold is derived once from the
/// pascal yield (a deterministic `f64`→`i128`), and the comparison is integer —
/// so the fracture *decision* replays byte-identically. (Simplified: yield is
/// treated as a milli-newton threshold over a unit voxel face.)
fn force_meets_yield(force: Force, yield_pa: f32) -> bool {
    let mag_sq: i128 = (force.milli_n.x as i128).pow(2)
        + (force.milli_n.y as i128).pow(2)
        + (force.milli_n.z as i128).pow(2);
    let threshold_mn = (yield_pa as f64 * Force::SCALE) as i128;
    mag_sq >= threshold_mn * threshold_mn
}

// Suppress unused-import lint for symbols that exist primarily for downstream
// crates to reach in via the re-exports surface area.
#[allow(dead_code)]
fn _unused_addr_imports(_: WorldAddr, _: Level, _: ParentAddr) {}

#[async_trait]
impl Actor for WorldActor {
    type Msg = WorldMsg;
    async fn handle(&mut self, ctx: &mut Context<Self>, msg: WorldMsg) {
        match msg {
            WorldMsg::Request { env, reply } => {
                let result = self.handle_request(env).await;
                let _ = reply.send(result);
                // A fracture may have queued debris; arm the self-tick now that
                // we hold the `Context` (the scheduler needs it).
                if self.want_arm {
                    self.want_arm = false;
                    if !self.debris_ticking {
                        self.arm_tick(ctx);
                    }
                }
            }
            WorldMsg::SubscribeBegin { env, sink, ready } => {
                let result = self.handle_subscribe_begin(env, sink);
                let _ = ready.send(result);
            }
            WorldMsg::Tick => {
                let deltas = self.step_debris();
                self.fan_out_debris_states(deltas);
                // Re-arm while debris remains; otherwise go idle (no timer load).
                if !self.debris.is_empty() {
                    self.arm_tick(ctx);
                } else {
                    self.debris_ticking = false;
                }
            }
        }
    }

    async fn post_stop(&mut self, _ctx: &mut Context<Self>) {
        // Cancel any pending self-tick so it doesn't fire into a dead mailbox.
        if let Some(h) = self.sim_handle.take() {
            h.cancel();
        }
    }
}
