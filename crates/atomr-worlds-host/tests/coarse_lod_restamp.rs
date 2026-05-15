//! Phase 17.1 follow-up: coarse-LOD overlay re-stamping.
//!
//! Voxel writes used to stamp only the LOD-0 cache entry. Once an observer
//! crossed past the LOD transition radius the coarse brick (depth ≥ 1)
//! showed the *procedural* baseline instead of the user's edit, so newly-
//! built blocks vanished as you walked away. The fix: ensure_brick now
//! applies the per-position overlay at every LOD depth, mapping each
//! LOD-0 voxel-position into the matching coarse cell, and the WriteVoxel
//! / WriteRegion paths drop any cached coarse brick containing the
//! affected position so the next read regenerates with the new overlay.
//!
//! These tests exercise both halves: re-stamping at fetch time, and
//! invalidation of a previously-cached coarse brick.
//!
//! Material `2` is "stone" in the default palette; "0" is empty.

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const STONE: Voxel = Voxel(2);

async fn fresh_host() -> LocalHost {
    LocalHost::with_seed(TEST_SEED).await.expect("host")
}

async fn fetch_brick(host: &LocalHost, brick: IVec3, lod: Lod) -> Brick {
    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(0, addr, WorldRequest::GetBrick { addr, brick, lod });
    let resp = host.request(env).await.expect("request");
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
        panic!("expected BrickSnapshot")
    };
    Brick::from_bytes(&payload).expect("decode")
}

async fn write_voxel(host: &LocalHost, pos: IVec3, voxel: Voxel) {
    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(0, addr, WorldRequest::WriteVoxel { addr, pos, voxel });
    host.request(env).await.expect("write");
}

/// Pick a write position that's high enough above the surface that the
/// procedural baseline at every LOD is empty — that way "the cell holds
/// `STONE`" cleanly attributes to the overlay rather than terrain noise.
const HIGH_AIR: i64 = 3_000;

/// Pick a write position whose LOD-0 brick coord is something other than
/// `(0, *, 0)` so the test isn't accidentally picking up an out-of-shape
/// sentinel brick.
fn write_pos() -> IVec3 {
    IVec3::new(5, HIGH_AIR, 5)
}

fn lod0_brick_of(pos: IVec3) -> IVec3 {
    let edge = BRICK_EDGE as i64;
    IVec3::new(pos.x.div_euclid(edge), pos.y.div_euclid(edge), pos.z.div_euclid(edge))
}

fn lod0_local_of(pos: IVec3) -> IVec3 {
    let edge = BRICK_EDGE as i64;
    IVec3::new(pos.x.rem_euclid(edge), pos.y.rem_euclid(edge), pos.z.rem_euclid(edge))
}

fn coarse_brick_and_local(pos: IVec3, lod: Lod) -> (IVec3, IVec3) {
    let edge = BRICK_EDGE as i64;
    let scale = 1i64 << lod.depth as u32;
    let edge_world = edge * scale;
    let bc = IVec3::new(
        pos.x.div_euclid(edge_world),
        pos.y.div_euclid(edge_world),
        pos.z.div_euclid(edge_world),
    );
    let in_brick = IVec3::new(
        pos.x.rem_euclid(edge_world),
        pos.y.rem_euclid(edge_world),
        pos.z.rem_euclid(edge_world),
    );
    let lc = IVec3::new(
        in_brick.x.div_euclid(scale),
        in_brick.y.div_euclid(scale),
        in_brick.z.div_euclid(scale),
    );
    (bc, lc)
}

/// Sanity: the LOD-0 path stamps the write at the exact cell.
#[tokio::test]
async fn write_then_lod0_brick_reflects_overlay() {
    let host = fresh_host().await;
    let pos = write_pos();
    write_voxel(&host, pos, STONE).await;
    let bc = lod0_brick_of(pos);
    let lc = lod0_local_of(pos);
    let b = fetch_brick(&host, bc, Lod::new(0)).await;
    assert_eq!(b.get(lc), STONE);
    host.shutdown().await.unwrap();
}

/// Coarse-LOD fetch *after* the write picks up the overlay through the
/// re-stamp path inside `ensure_brick`.
#[tokio::test]
async fn write_then_lod1_brick_reflects_overlay() {
    let host = fresh_host().await;
    let pos = write_pos();
    write_voxel(&host, pos, STONE).await;
    let (bc, lc) = coarse_brick_and_local(pos, Lod::new(1));
    let b = fetch_brick(&host, bc, Lod::new(1)).await;
    assert_eq!(
        b.get(lc),
        STONE,
        "coarse-LOD cell containing the write was not stamped"
    );
    host.shutdown().await.unwrap();
}

/// The same is true at LOD 2 (4 m voxels) — the cell footprint widens but
/// the lookup still lands on the right cell.
#[tokio::test]
async fn write_then_lod2_brick_reflects_overlay() {
    let host = fresh_host().await;
    let pos = write_pos();
    write_voxel(&host, pos, STONE).await;
    let (bc, lc) = coarse_brick_and_local(pos, Lod::new(2));
    let b = fetch_brick(&host, bc, Lod::new(2)).await;
    assert_eq!(b.get(lc), STONE);
    host.shutdown().await.unwrap();
}

/// Cache-invalidation half: pre-warm the coarse-LOD cache *before* the
/// write so the WriteVoxel handler must drop the stale entry. Without
/// invalidation, the second fetch would still return the empty
/// procedural baseline.
#[tokio::test]
async fn coarse_brick_cached_then_invalidated_on_write() {
    let host = fresh_host().await;
    let pos = write_pos();
    let (bc, lc) = coarse_brick_and_local(pos, Lod::new(1));

    // Prime the cache: empty procedural air at this altitude.
    let before = fetch_brick(&host, bc, Lod::new(1)).await;
    assert_eq!(before.get(lc), Voxel::EMPTY, "test precondition: high air is empty");

    // Write through the LOD-0 path. The handler must drop the cached
    // coarse entry so the next fetch regenerates with the overlay
    // applied.
    write_voxel(&host, pos, STONE).await;
    let after = fetch_brick(&host, bc, Lod::new(1)).await;
    assert_eq!(
        after.get(lc),
        STONE,
        "coarse cache wasn't invalidated — stale empty cell survived the write"
    );
    host.shutdown().await.unwrap();
}

/// Carving a hole (writing `Voxel::EMPTY`) is *not* re-stamped at coarse
/// LODs — a single carved LOD-0 voxel out of `2^(3L)` would otherwise
/// blank the whole coarse cell. Documented behaviour; this test pins it.
#[tokio::test]
async fn empty_writes_do_not_blank_coarse_cell() {
    let host = fresh_host().await;
    // Pick a position deep in the procedural ground so the LOD-0
    // procedural fill is non-empty before our carve. Surface for the
    // default seed sits well above y=-100.
    let pos = IVec3::new(0, -100, 0);
    let (bc, lc) = coarse_brick_and_local(pos, Lod::new(1));
    let before = fetch_brick(&host, bc, Lod::new(1)).await;
    let baseline = before.get(lc);
    assert_ne!(baseline, Voxel::EMPTY, "test precondition: deep ground non-empty");

    write_voxel(&host, pos, Voxel::EMPTY).await;
    let after = fetch_brick(&host, bc, Lod::new(1)).await;
    assert_eq!(
        after.get(lc),
        baseline,
        "coarse cell was blanked by an empty write — should require a full re-bake"
    );
    host.shutdown().await.unwrap();
}
