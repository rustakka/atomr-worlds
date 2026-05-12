//! Generic-engine-shaped scene description.
//!
//! Produced by [`scene_from_bricks`]; consumed by a future
//! [`atomr-view`][av] bridge or a `wgpu` headless backend. Keeps the
//! upstream-scene-API decision out of this crate so when atomr-view's 3D
//! primitives land, the adapter is ~80 LOC instead of a refactor.
//!
//! [av]: ../../atomr_view/index.html

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_voxel::Brick;

use crate::camera::Camera;
use crate::iso::MeshMode;
use crate::mesh::Mesh;
use crate::{greedy_mesh, surface_mesh};

pub type SceneId = u64;

#[derive(Debug, Clone)]
pub struct SceneDescription {
    pub meshes: Vec<MeshNode>,
    pub cameras: Vec<CameraNode>,
    pub lights: Vec<LightNode>,
    pub frame_metadata: FrameMetadata,
}

#[derive(Debug, Clone)]
pub struct MeshNode {
    pub id: SceneId,
    pub mesh: Arc<Mesh>,
    pub transform: [[f32; 4]; 4],
    pub material_palette: Arc<MaterialPalette>,
    pub lod_hint: Option<Lod>,
}

#[derive(Debug, Clone)]
pub struct CameraNode {
    pub id: SceneId,
    pub camera: Camera,
    pub viewport: (u32, u32),
}

#[derive(Debug, Clone)]
pub struct LightNode {
    pub kind: LightKind,
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone)]
pub enum LightKind {
    Directional { dir: [f32; 3] },
    Point { pos: [f32; 3] },
    Ambient,
}

#[derive(Debug, Clone, Default)]
pub struct MaterialPalette {
    pub entries: Vec<MaterialEntry>,
}

#[derive(Debug, Clone)]
pub struct MaterialEntry {
    pub id: u16,
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
}

#[derive(Debug, Clone, Default)]
pub struct FrameMetadata {
    pub tick: u64,
    pub seed: u64,
}

/// Convert a slab of brick-coord/brick pairs into a scene description. Each
/// brick becomes one [`MeshNode`] translated to its world-local origin.
pub fn scene_from_bricks(
    bricks: &[(IVec3, Arc<Brick>)],
    camera: Camera,
    mode: MeshMode,
    palette: Arc<MaterialPalette>,
) -> SceneDescription {
    let edge = atomr_worlds_voxel::BRICK_EDGE as f32;
    let mut meshes = Vec::with_capacity(bricks.len());
    for (idx, (bc, brick)) in bricks.iter().enumerate() {
        let mesh = match mode {
            MeshMode::Flat => greedy_mesh(brick),
            MeshMode::Smooth(_) => surface_mesh(brick, mode),
        };
        if mesh.vertices.is_empty() {
            continue;
        }
        let tx = bc.x as f32 * edge;
        let ty = bc.y as f32 * edge;
        let tz = bc.z as f32 * edge;
        let transform = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [tx, ty, tz, 1.0],
        ];
        meshes.push(MeshNode {
            id: idx as SceneId,
            mesh: Arc::new(mesh),
            transform,
            material_palette: palette.clone(),
            lod_hint: None,
        });
    }
    SceneDescription {
        meshes,
        cameras: vec![CameraNode { id: 0, camera, viewport: (512, 512) }],
        lights: vec![LightNode {
            kind: LightKind::Directional { dir: [0.4, -0.8, -0.3] },
            color: [1.0, 1.0, 0.95],
            intensity: 1.0,
        }],
        frame_metadata: FrameMetadata::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_voxel::Voxel;

    #[test]
    fn single_voxel_produces_one_mesh_node() {
        let mut b = Brick::new();
        b.set(IVec3::new(3, 3, 3), Voxel::new(1));
        let cam = Camera::isometric_default(1.0);
        let palette = Arc::new(MaterialPalette::default());
        let scene = scene_from_bricks(
            &[(IVec3::new(0, 0, 0), Arc::new(b))],
            cam,
            MeshMode::Flat,
            palette,
        );
        assert_eq!(scene.meshes.len(), 1);
        assert!(!scene.meshes[0].mesh.vertices.is_empty());
    }

    #[test]
    fn empty_brick_is_culled() {
        let b = Brick::new();
        let scene = scene_from_bricks(
            &[(IVec3::new(0, 0, 0), Arc::new(b))],
            Camera::isometric_default(1.0),
            MeshMode::Flat,
            Arc::new(MaterialPalette::default()),
        );
        assert!(scene.meshes.is_empty());
    }
}
