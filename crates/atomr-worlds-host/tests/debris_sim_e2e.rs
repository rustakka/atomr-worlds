//! End-to-end tests for the Rec 4 Slice 2 host-authoritative debris simulation:
//! a `Fracture` populates the actor's debris registry, a periodic self-tick
//! integrates the bodies and fans `WorldEvent::DebrisStates` out to subscribers,
//! and bodies are retired once they settle or fall out of bounds. The debris
//! tick is derived/ephemeral and must never perturb `GetBrick`.

use std::time::Duration;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, LocalHostConfig, WorldHost};
use atomr_worlds_proto::{
    DebrisStateDelta, Envelope, Force, FractureCommand, FractureRequest, WorldEvent, WorldRequest,
    AABB,
};
use atomr_worlds_voxel::Voxel;
use tokio::sync::mpsc::Receiver;

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
/// High-altitude origin that is procedural air, so a placed blob is the only
/// solid in the fracture region and detaches cleanly.
const BASE: IVec3 = IVec3::new(0, 400, 0);

async fn host() -> LocalHost {
    LocalHost::new(LocalHostConfig { root_seed: TEST_SEED, ..Default::default() })
        .await
        .expect("host")
}

async fn get_voxel(host: &LocalHost, addr: Address, pos: IVec3) -> Voxel {
    let resp =
        host.request(Envelope::new(0, addr, WorldRequest::GetVoxel { addr, pos })).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    voxel
}

async fn get_brick_bytes(host: &LocalHost, addr: Address, brick: IVec3) -> bytes::Bytes {
    let resp = host
        .request(Envelope::new(0, addr, WorldRequest::GetBrick { addr, brick, lod: Lod::new(0) }))
        .await
        .unwrap();
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else { panic!("variant") };
    payload
}

async fn write_solid(host: &LocalHost, addr: Address, pos: IVec3) {
    host.request(Envelope::new(1, addr, WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(3) }))
        .await
        .unwrap();
}

/// Place a 2×2×2 floating solid blob at `BASE`.
async fn place_blob(host: &LocalHost, addr: Address) {
    assert_eq!(
        get_voxel(host, addr, IVec3::new(BASE.x + 5, BASE.y, BASE.z + 5)).await,
        Voxel::EMPTY,
        "precondition: {BASE:?} region must be air"
    );
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                write_solid(host, addr, IVec3::new(BASE.x + dx, BASE.y + dy, BASE.z + dz)).await;
            }
        }
    }
}

/// Anchored 5×1×5 slab at the region's bottom shell (`y = BASE.y - 8`), under
/// the blob's footprint, so a detached blob lands on it instead of falling away.
async fn place_floor(host: &LocalHost, addr: Address) {
    for dz in -2..=2 {
        for dx in -2..=2 {
            write_solid(host, addr, IVec3::new(BASE.x + dx, BASE.y - 8, BASE.z + dz)).await;
        }
    }
}

async fn fracture(host: &LocalHost, addr: Address) -> atomr_worlds_proto::FractureApplied {
    fracture_with(host, addr, BASE, Force::ZERO).await
}

/// Fracture at an explicit impact point with an explicit force, so a test can
/// drive an *off-center* impulse (the source of seeded spin).
async fn fracture_with(
    host: &LocalHost,
    addr: Address,
    impact_pos: IVec3,
    force: Force,
) -> atomr_worlds_proto::FractureApplied {
    let req = FractureRequest { addr, impact_pos, force, material_id: 0 };
    let resp = host.request(Envelope::new(2, addr, WorldRequest::Fracture(req))).await.unwrap();
    let WorldEvent::FractureApplied(applied) = resp.body else { panic!("variant") };
    applied
}

fn spawn_id(applied: &atomr_worlds_proto::FractureApplied) -> u32 {
    applied
        .commands
        .iter()
        .find_map(|c| match c {
            FractureCommand::SpawnDebris { id, .. } => Some(*id),
            _ => None,
        })
        .expect("a SpawnDebris command")
}

async fn subscribe(host: &LocalHost, addr: Address) -> Receiver<Envelope<WorldEvent>> {
    let region = AABB::new(
        IVec3::new(BASE.x - 16, BASE.y - 16, BASE.z - 16),
        IVec3::new(BASE.x + 16, BASE.y + 16, BASE.z + 16),
    );
    let env =
        Envelope::new(0, addr, WorldRequest::Subscribe { addr, region, lod: Lod::new(0), sub_id: 9 });
    let mut rx = host.subscribe(env).await.expect("subscribe");
    // Drain the initial brick snapshots so the channel starts clean.
    while let Ok(Some(_)) = tokio::time::timeout(Duration::from_millis(60), rx.recv()).await {}
    rx
}

/// Collect every `DebrisStateDelta` that arrives within `budget`.
async fn collect_debris(
    rx: &mut Receiver<Envelope<WorldEvent>>,
    budget: Duration,
) -> Vec<DebrisStateDelta> {
    let mut out = Vec::new();
    let deadline = tokio::time::Instant::now() + budget;
    while let Ok(Some(env)) = tokio::time::timeout_at(deadline, rx.recv()).await {
        if let WorldEvent::DebrisStates { deltas, .. } = env.body {
            out.extend(deltas);
        }
    }
    out
}

#[tokio::test]
async fn fracture_spawns_streaming_falling_debris() {
    let addr = Address::World(WorldAddr::ROOT);
    let host = host().await;
    let mut rx = subscribe(&host, addr).await;
    place_blob(&host, addr).await;
    let applied = fracture(&host, addr).await;
    let id = spawn_id(&applied);

    // Watch the body fall through the subscribed region for ~half a second.
    let deltas = collect_debris(&mut rx, Duration::from_millis(500)).await;
    let mine: Vec<&DebrisStateDelta> = deltas.iter().filter(|d| d.id == id).collect();
    assert!(mine.len() >= 2, "expected streamed debris deltas, got {}", mine.len());

    // Ticks are strictly increasing and the body descends under gravity.
    for w in mine.windows(2) {
        assert!(w[1].tick > w[0].tick, "ticks must be monotonic");
        assert!(w[1].pos[1] < w[0].pos[1] + 1e-3, "debris should be falling");
    }
    assert!(mine.last().unwrap().pos[1] < mine[0].pos[1], "net descent");
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn idle_world_streams_no_debris() {
    let addr = Address::World(WorldAddr::ROOT);
    let host = host().await;
    let mut rx = subscribe(&host, addr).await;

    // No fracture → no registry → no self-tick → no debris broadcasts.
    let _ = get_voxel(&host, addr, BASE).await;
    let deltas = collect_debris(&mut rx, Duration::from_millis(300)).await;
    assert!(deltas.is_empty(), "an un-fractured world emits no DebrisStates");
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn debris_tick_preserves_unrelated_getbrick_bytes() {
    let addr = Address::World(WorldAddr::ROOT);
    let host = host().await;
    // A brick far from the fracture site (near the ground), with real terrain.
    let far = IVec3::new(0, 0, 0);
    let before = get_brick_bytes(&host, addr, far).await;

    let mut rx = subscribe(&host, addr).await;
    place_blob(&host, addr).await;
    let _ = fracture(&host, addr).await;
    // Let several debris ticks run.
    let _ = collect_debris(&mut rx, Duration::from_millis(200)).await;

    let after = get_brick_bytes(&host, addr, far).await;
    assert_eq!(before, after, "debris simulation must not perturb GetBrick bytes");
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn debris_lands_sleeps_and_stops_streaming() {
    let addr = Address::World(WorldAddr::ROOT);
    let host = host().await;
    let mut rx = subscribe(&host, addr).await;
    place_floor(&host, addr).await;
    place_blob(&host, addr).await;
    let applied = fracture(&host, addr).await;
    let id = spawn_id(&applied);

    let mut saw_sleeping_at_y: Option<f32> = None;
    let mut stopped = false;
    // Bounded: land (~1.3 s) + sleep (~1 s) + a few quiet chunks. Breaks early.
    for _ in 0..28 {
        let batch = collect_debris(&mut rx, Duration::from_millis(250)).await;
        let mine: Vec<&DebrisStateDelta> = batch.iter().filter(|d| d.id == id).collect();
        if let Some(d) = mine.iter().find(|d| d.sleeping) {
            saw_sleeping_at_y = Some(d.pos[1]);
        }
        // Once asleep, the host stops streaming the (settled) body.
        if saw_sleeping_at_y.is_some() && mine.is_empty() {
            stopped = true;
            break;
        }
    }
    let y = saw_sleeping_at_y.expect("debris should land on host terrain and sleep");
    assert!(y > 380.0, "debris should rest on the floor (y≈393), not fall away: y={y}");
    assert!(stopped, "host should stop streaming a settled/retired body");
    host.shutdown().await.unwrap();
}

fn ang_speed(d: &DebrisStateDelta) -> f32 {
    (d.ang_vel[0] * d.ang_vel[0] + d.ang_vel[1] * d.ang_vel[1] + d.ang_vel[2] * d.ang_vel[2]).sqrt()
}

/// How far the streamed orientation has rotated away from identity. The scalar
/// part `w` of a unit quaternion is `cos(θ/2)`, so `1 - |w|` grows from 0 as the
/// body turns.
fn orient_divergence(d: &DebrisStateDelta) -> f32 {
    1.0 - d.orient[3].abs()
}

#[tokio::test]
async fn off_center_impact_accumulates_orientation() {
    let addr = Address::World(WorldAddr::ROOT);
    let host = host().await;
    // A brick far from the fracture, to re-check GetBrick byte-identity after
    // the body has tumbled.
    let far = IVec3::new(0, 0, 0);
    let before_far = get_brick_bytes(&host, addr, far).await;

    let mut rx = subscribe(&host, addr).await;
    place_blob(&host, addr).await;
    // A strong impulse applied at the blob's `(0,0,0)` corner — off the body's
    // center of mass — so `L = r × J` is non-zero and the body spins as it falls.
    // Downward so it stays roughly under its footprint (in the subscribed region)
    // rather than launching sideways out of view.
    let force = Force::from_newtons([0.0, -50_000.0, 0.0]);
    let applied = fracture_with(&host, addr, BASE, force).await;
    let id = spawn_id(&applied);

    let deltas = collect_debris(&mut rx, Duration::from_millis(600)).await;
    let mine: Vec<&DebrisStateDelta> = deltas.iter().filter(|d| d.id == id).collect();
    assert!(mine.len() >= 2, "expected streamed deltas, got {}", mine.len());

    // The body carries a non-zero angular velocity from the off-center impulse…
    assert!(
        mine.iter().any(|d| ang_speed(d) > 1e-3),
        "off-center impact should seed angular velocity"
    );
    // …and its orientation visibly diverges from identity across the stream.
    assert!(
        mine.iter().any(|d| orient_divergence(d) > 1e-3),
        "orientation should accumulate (max divergence {})",
        mine.iter().map(|d| orient_divergence(d)).fold(0.0_f32, f32::max)
    );

    // The debris tumble is ephemeral physics: it must never perturb stored voxels.
    let after_far = get_brick_bytes(&host, addr, far).await;
    assert_eq!(before_far, after_far, "debris tumble must not perturb GetBrick bytes");
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn zero_force_carve_yields_no_spin() {
    let addr = Address::World(WorldAddr::ROOT);
    let host = host().await;
    let mut rx = subscribe(&host, addr).await;
    place_blob(&host, addr).await;
    // The common hand-carve path: `Force::ZERO` → no impulse → no spin, the
    // island simply falls with identity orientation throughout.
    let applied = fracture(&host, addr).await;
    let id = spawn_id(&applied);

    let deltas = collect_debris(&mut rx, Duration::from_millis(400)).await;
    let mine: Vec<&DebrisStateDelta> = deltas.iter().filter(|d| d.id == id).collect();
    assert!(!mine.is_empty(), "the body should still fall and stream");
    for d in &mine {
        assert!(ang_speed(d) == 0.0, "zero-force carve must not spin: {:?}", d.ang_vel);
        assert_eq!(d.orient, [0.0, 0.0, 0.0, 1.0], "orientation must stay identity");
    }
    host.shutdown().await.unwrap();
}
