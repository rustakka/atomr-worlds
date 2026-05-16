//! Mesh primitives + meshing algorithm modules.
//!
//! Shared types ([`Vertex`], [`Quad`], [`Mesh`]) live here; each algorithm
//! is a sibling module ([`greedy`], [`naive`], [`marching_cubes`],
//! [`dual_contouring`]) that returns a [`Mesh`] in brick-local coordinates.

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub material: u16,
    /// Per-vertex ambient-occlusion factor in `[0, 1]`. `1.0` means
    /// unobstructed (no corner occlusion); `< 1.0` means the vertex is
    /// in a concave corner. Set by AO strategies (Minecraft-style corner
    /// sampling); defaults to `1.0` so meshes without AO render unaffected.
    pub ao: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Quad {
    pub origin: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
    pub normal: [f32; 3],
    pub material: u16,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

pub mod dual_contouring;
pub mod greedy;
pub mod marching_cubes;
pub mod naive;

pub use dual_contouring::dual_contouring_mesh;
pub use greedy::{bake_ao, greedy_mesh, greedy_mesh_by_material};
pub use marching_cubes::{marching_cubes_mesh, marching_cubes_mesh_with_iso};
pub use naive::naive_mesh;
