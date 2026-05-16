//! Macro-structure strategies: WFC dungeons, Jigsaw villages, classical
//! QWFC stub. Each impl consumes `FeatureKind::Structure` anchors from
//! [`super::workspace::BrickWorkspace::anchors`] and stamps voxels into
//! `ws.materials` clipped to the brick AABB.

pub mod jigsaw;
pub mod qwfc;
pub mod wfc;

pub use jigsaw::{Jigsaw, JigsawConfig, JigsawTag};
pub use qwfc::QwfcClassicalSim;
pub use wfc::{WaveFunctionCollapse, WfcConfig};

use atomr_worlds_core::coord::IVec3;

/// One tile in a WFC / QWFC tileset. `neighbors` is six lists, indexed by
/// face: `-X, +X, -Y, +Y, -Z, +Z`, naming tile ids that can sit on the
/// matching side.
#[derive(Debug, Clone)]
pub struct TileDef {
    pub id: u32,
    pub geometry: TileGeometry,
    pub neighbors: [Vec<u32>; 6],
    pub weight: f32,
}

/// Voxel geometry stamped by a tile. Coordinates are tile-local
/// `0..module_edge` on each axis.
#[derive(Debug, Clone, Default)]
pub struct TileGeometry {
    pub voxels: Vec<(IVec3, u16)>,
}

/// Bundled set of compatible tiles. `Default` produces an empty set;
/// `TileSet::test_tiles()` produces a tiny all-compatible set used by
/// unit tests and the registry's no-arg constructors.
#[derive(Debug, Clone, Default)]
pub struct TileSet {
    pub tiles: Vec<TileDef>,
}

impl TileSet {
    pub fn test_tiles() -> Self {
        let all_two: [Vec<u32>; 6] = [
            vec![1, 2],
            vec![1, 2],
            vec![1, 2],
            vec![1, 2],
            vec![1, 2],
            vec![1, 2],
        ];
        Self {
            tiles: vec![
                TileDef {
                    id: 1,
                    geometry: TileGeometry {
                        voxels: vec![(IVec3::new(0, 0, 0), 100)],
                    },
                    neighbors: all_two.clone(),
                    weight: 1.0,
                },
                TileDef {
                    id: 2,
                    geometry: TileGeometry {
                        voxels: vec![(IVec3::new(0, 0, 0), 101)],
                    },
                    neighbors: all_two,
                    weight: 1.0,
                },
            ],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    fn index_of(&self, id: u32) -> Option<usize> {
        self.tiles.iter().position(|t| t.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiles_has_two_compatible_entries() {
        let ts = TileSet::test_tiles();
        assert_eq!(ts.len(), 2);
        for tile in &ts.tiles {
            for face in &tile.neighbors {
                assert_eq!(face.len(), 2);
            }
        }
    }
}
