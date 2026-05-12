//! In-process host on top of atomr's `ActorSystem`.
//!
//! One `WorldActor` is spawned per `WorldAddr` (lazily, on first
//! request). The actor owns a brick cache and the subscriber registry.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr::prelude::*;
use atomr_worlds_core::addr::{Level, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{BrickGenerator, TerrainGenerator, WorldGen};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest, AABB};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::error::HostError;
use crate::host::WorldHost;

#[derive(Clone, Debug)]
pub struct LocalHostConfig {
    pub root_seed: u64,
    pub world_gen: WorldGen,
    /// Default bound for per-subscriber mpsc channels.
    pub subscriber_capacity: usize,
    /// Timeout for `WorldHost::request`'s `ask` call.
    pub request_timeout: Duration,
}

impl Default for LocalHostConfig {
    fn default() -> Self {
        Self {
            root_seed: 0xDEAD_BEEF_CAFE_F00D,
            world_gen: WorldGen::default(),
            subscriber_capacity: 256,
            request_timeout: Duration::from_secs(10),
        }
    }
}

pub struct LocalHost {
    sys: ActorSystem,
    worlds: Arc<Mutex<HashMap<WorldAddr, ActorRef<WorldMsg>>>>,
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
        Ok(Self { sys, worlds: Arc::new(Mutex::new(HashMap::new())), config })
    }

    /// Convenience constructor with default config and a chosen root seed.
    pub async fn with_seed(seed: u64) -> Result<Self, HostError> {
        Self::new(LocalHostConfig { root_seed: seed, ..LocalHostConfig::default() }).await
    }

    async fn world_actor_for(&self, addr: WorldAddr) -> Result<ActorRef<WorldMsg>, HostError> {
        let mut map = self.worlds.lock().await;
        if let Some(a) = map.get(&addr) {
            return Ok(a.clone());
        }
        let seed = addr.seed_at(self.config.root_seed, Level::World);
        let brick_gen = self.config.world_gen.brick_gen();
        let name = format!("world-{:x}-{}", seed, map.len());
        let factory_addr = addr;
        let factory_gen = brick_gen;
        let actor = self
            .sys
            .actor_of(
                Props::create(move || WorldActor::new(factory_addr, seed, factory_gen.clone())),
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
        let actor = self.world_actor_for(addr).await?;
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
        let actor = self.world_actor_for(addr).await?;
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

fn env_target_addr(env: &Envelope<WorldRequest>) -> WorldAddr {
    match &env.body {
        WorldRequest::GetVoxel { addr, .. }
        | WorldRequest::GetBrick { addr, .. }
        | WorldRequest::WriteVoxel { addr, .. }
        | WorldRequest::Subscribe { addr, .. } => *addr,
        WorldRequest::Unsubscribe { .. } => env.from,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-world actor.
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
}

pub(crate) struct WorldActor {
    _addr: WorldAddr,
    seed: u64,
    brick_gen: TerrainGenerator,
    cache: HashMap<IVec3, Brick>,
    subscribers: HashMap<u64, Subscriber>,
}

impl WorldActor {
    fn new(addr: WorldAddr, seed: u64, brick_gen: TerrainGenerator) -> Self {
        Self { _addr: addr, seed, brick_gen, cache: HashMap::new(), subscribers: HashMap::new() }
    }

    fn ensure_brick(&mut self, brick_coord: IVec3) -> &mut Brick {
        let seed = self.seed;
        let gen = &self.brick_gen;
        self.cache.entry(brick_coord).or_insert_with(|| gen.generate_brick(seed, brick_coord))
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

    fn handle_request(
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
                let before = {
                    let b = self.ensure_brick(bc);
                    let prev = b.get(lc);
                    b.set(lc, voxel);
                    prev
                };
                self.fan_out_delta(addr, pos, before, voxel);
                Ok(Envelope::new(corr, from, WorldEvent::Ack { addr }))
            }
            WorldRequest::Subscribe { .. } => {
                Err(HostError::NotYetImplemented("use WorldHost::subscribe for Subscribe envelopes"))
            }
            WorldRequest::Unsubscribe { sub_id } => {
                self.subscribers.remove(&sub_id);
                Ok(Envelope::new(corr, from, WorldEvent::StreamEnd { sub_id }))
            }
        }
    }

    fn handle_subscribe_begin(
        &mut self,
        env: Envelope<WorldRequest>,
        sink: mpsc::Sender<Envelope<WorldEvent>>,
    ) -> Result<(), HostError> {
        let WorldRequest::Subscribe { addr, region, lod, sub_id } = env.body else {
            return Err(HostError::NotYetImplemented("SubscribeBegin requires WorldRequest::Subscribe"));
        };

        let corr = env.corr_id;
        let from = env.from;
        let edge = BRICK_EDGE as i64;
        let bmin_x = region.min.x.div_euclid(edge);
        let bmax_x = (region.max.x - 1).div_euclid(edge);
        let bmin_y = region.min.y.div_euclid(edge);
        let bmax_y = (region.max.y - 1).div_euclid(edge);
        let bmin_z = region.min.z.div_euclid(edge);
        let bmax_z = (region.max.z - 1).div_euclid(edge);

        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    let bc = IVec3::new(bx, by, bz);
                    let payload = self.snapshot(bc);
                    let ev = WorldEvent::BrickSnapshot { addr, brick: bc, lod, payload };
                    let env_out = Envelope::new(corr, from, ev);
                    if sink.try_send(env_out).is_err() {
                        // Channel closed before initial snapshot finished.
                        return Err(HostError::SubscribeFailed);
                    }
                }
            }
        }
        self.subscribers.insert(sub_id, Subscriber { region, sink });
        Ok(())
    }

    fn fan_out_delta(&mut self, addr: WorldAddr, pos: IVec3, before: Voxel, after: Voxel) {
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
}

#[async_trait]
impl Actor for WorldActor {
    type Msg = WorldMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: WorldMsg) {
        match msg {
            WorldMsg::Request { env, reply } => {
                let result = self.handle_request(env);
                let _ = reply.send(result);
            }
            WorldMsg::SubscribeBegin { env, sink, ready } => {
                let result = self.handle_subscribe_begin(env, sink);
                let _ = ready.send(result);
            }
        }
    }
}
