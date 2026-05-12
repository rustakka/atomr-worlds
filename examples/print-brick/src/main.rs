//! Smoke binary: ask `LocalHost` for a brick and ASCII-dump a YZ slice.

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = LocalHost::with_seed(0xDEAD_BEEF_CAFE_F00D).await?;
    let addr = Address::World(WorldAddr::ROOT);

    // Pick a brick that straddles the terrain surface (y around base_height/16 ≈ 2).
    let brick_coord = IVec3::new(0, 1, 0);
    let req = WorldRequest::GetBrick { addr, brick: brick_coord, lod: Lod::new(0) };
    let env = Envelope::new(1, addr, req);

    let resp = host.request(env).await?;
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
        return Err("unexpected response variant".into());
    };
    let brick = Brick::from_bytes(&payload)?;

    println!("brick {:?}  nonempty: {}", brick_coord, brick.nonempty_count);
    println!("YZ slice at x=8 (`#` = filled, `.` = empty):");
    for y in (0..BRICK_EDGE as i64).rev() {
        let row: String = (0..BRICK_EDGE as i64)
            .map(|z| if brick.get(IVec3::new(8, y, z)).is_empty() { '.' } else { '#' })
            .collect();
        println!("  y={:>2}  {}", y, row);
    }

    host.shutdown().await?;
    Ok(())
}
