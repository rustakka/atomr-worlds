//! Per-brick scratch workspace passed through every pipeline stage.

use atomr_worlds_voxel::{light::LightOverlay, Brick, Voxel, BRICK_EDGE};

use crate::brick::BrickGenContext;

use super::anchor::FeatureAnchor;

/// Edge of the workspace's padded grids: `BRICK_EDGE + 2`. One voxel of
/// apron on each side gives neighbor-aware passes a one-voxel read window
/// without recursing into the neighbor's pipeline.
pub const WS_APRON_EDGE: usize = BRICK_EDGE + 2;

const APRON_VOLUME: usize = WS_APRON_EDGE * WS_APRON_EDGE * WS_APRON_EDGE;

/// Mutable per-brick state threaded through every pipeline stage. Stages
/// read the immutable `ctx`, read/write `density`, `materials`, and
/// `anchors`, and ultimately stamp results into `brick`. `light` is
/// populated by the optional sky-light stage.
#[derive(Debug)]
pub struct BrickWorkspace {
    pub ctx: BrickGenContext,
    pub density: Vec<f32>,
    pub materials: Vec<Voxel>,
    pub anchors: Vec<FeatureAnchor>,
    pub brick: Brick,
    pub light: Option<Box<LightOverlay>>,
}

impl BrickWorkspace {
    pub fn new(ctx: BrickGenContext) -> Self {
        Self {
            ctx,
            density: vec![0.0; APRON_VOLUME],
            materials: vec![Voxel::EMPTY; APRON_VOLUME],
            anchors: Vec::new(),
            brick: Brick::new(),
            light: None,
        }
    }

    /// Flat index into the padded `WS_APRON_EDGE`³ buffers. `(x,y,z)` are
    /// brick-local in `0..BRICK_EDGE`; the apron occupies `-1` and
    /// `BRICK_EDGE` and is reached by passing in `x = -1` etc. via the
    /// signed wrapper [`Self::apron_index`].
    #[inline]
    pub fn brick_index(x: usize, y: usize, z: usize) -> usize {
        let xi = x + 1;
        let yi = y + 1;
        let zi = z + 1;
        zi * WS_APRON_EDGE * WS_APRON_EDGE + yi * WS_APRON_EDGE + xi
    }

    /// Flat index that accepts the apron coordinates directly
    /// (`-1..=BRICK_EDGE`). Out-of-range coordinates panic in debug.
    #[inline]
    pub fn apron_index(x: i32, y: i32, z: i32) -> usize {
        debug_assert!((-1..=BRICK_EDGE as i32).contains(&x));
        debug_assert!((-1..=BRICK_EDGE as i32).contains(&y));
        debug_assert!((-1..=BRICK_EDGE as i32).contains(&z));
        let xi = (x + 1) as usize;
        let yi = (y + 1) as usize;
        let zi = (z + 1) as usize;
        zi * WS_APRON_EDGE * WS_APRON_EDGE + yi * WS_APRON_EDGE + xi
    }

    #[inline]
    pub fn density_at(&self, x: i32, y: i32, z: i32) -> f32 {
        self.density[Self::apron_index(x, y, z)]
    }

    #[inline]
    pub fn set_density(&mut self, x: i32, y: i32, z: i32, d: f32) {
        let i = Self::apron_index(x, y, z);
        self.density[i] = d;
    }

    #[inline]
    pub fn material_at(&self, x: i32, y: i32, z: i32) -> Voxel {
        self.materials[Self::apron_index(x, y, z)]
    }

    #[inline]
    pub fn set_material(&mut self, x: i32, y: i32, z: i32, v: Voxel) {
        let i = Self::apron_index(x, y, z);
        self.materials[i] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::coord::IVec3;

    fn ctx() -> BrickGenContext {
        BrickGenContext::legacy(7, IVec3::new(0, 0, 0))
    }

    #[test]
    fn apron_edge_is_brick_plus_2() {
        assert_eq!(WS_APRON_EDGE, BRICK_EDGE + 2);
    }

    #[test]
    fn workspace_round_trips_density() {
        let mut ws = BrickWorkspace::new(ctx());
        ws.set_density(0, 0, 0, 1.5);
        ws.set_density(-1, BRICK_EDGE as i32, 7, -0.25);
        assert_eq!(ws.density_at(0, 0, 0), 1.5);
        assert_eq!(ws.density_at(-1, BRICK_EDGE as i32, 7), -0.25);
    }

    #[test]
    fn workspace_round_trips_materials() {
        let mut ws = BrickWorkspace::new(ctx());
        let v = Voxel::new(42);
        ws.set_material(5, 5, 5, v);
        assert_eq!(ws.material_at(5, 5, 5), v);
    }
}
