//! Horizon-imposter shell baker.
//!
//! Produces a polar-annulus terrain mesh that fills the band between
//! the streamer's outer LOD ring and the geometric horizon. Vertices
//! are observer-relative meters and colored by a macro-sampled
//! elevation + biome lookup, so the shell reads as representative
//! terrain rather than a painted skybox — and stays parallax-correct
//! because every vertex moves with the camera as a single rigid body
//! (no per-vertex parallax projection needed).
//!
//! Pure function in / out — no Bevy types, no ECS state. The Bevy
//! `HorizonShellPlugin` (in `atomr-worlds-client`) calls this from a
//! background thread via the same `Mutex<mpsc::Receiver<_>>` pattern
//! the desired-chunks plan rebuild uses.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::WorldMacroState;
use atomr_worlds_generate::water_kind;

/// Output of [`bake_polar_annulus`]. Vertices are observer-relative
/// meters; the consumer is expected to wrap this into a parent transform
/// that follows the camera. `r_inner_m` / `r_outer_m` are echoed back
/// for the caller's bookkeeping (LOD-handoff fog band, debug HUD).
#[derive(Debug, Clone)]
pub struct HorizonShellMesh {
    pub vertices: Vec<[f32; 3]>,
    pub colors:   Vec<[f32; 4]>,
    pub indices:  Vec<u32>,
    pub r_inner_m: f32,
    pub r_outer_m: f32,
}

/// Hard cap on vertex count. 32 rings × 128 sectors = 4096, well below
/// this. The cap exists so misconfiguration (e.g. a richer downstream
/// strategy passes 256 × 256) can't allocate a hundred MB of
/// observer-relative geometry.
pub const MAX_SHELL_VERTS: usize = 16_384;

/// Build a polar-annulus mesh sampling [`WorldMacroState`] for elevation
/// and biome color. Triangles wind CCW when viewed from above so the
/// renderer's standard back-face culling keeps them visible from the
/// observer's eye level.
///
/// `n_rings` and `n_sectors` are clamped so `n_rings * (n_sectors + 1)`
/// stays under [`MAX_SHELL_VERTS`].
///
/// Returns an empty mesh when `outer_radius_m <= inner_radius_m` — the
/// caller is expected to skip drawing in that case.
pub fn bake_polar_annulus(
    macro_state: &WorldMacroState,
    shape: WorldShape,
    observer: DVec3,
    inner_radius_m: f64,
    outer_radius_m: f64,
    n_rings: u32,
    n_sectors: u32,
) -> HorizonShellMesh {
    if outer_radius_m <= inner_radius_m || n_rings == 0 || n_sectors == 0 {
        return HorizonShellMesh {
            vertices: Vec::new(),
            colors:   Vec::new(),
            indices:  Vec::new(),
            r_inner_m: inner_radius_m as f32,
            r_outer_m: outer_radius_m as f32,
        };
    }

    // Cap rings/sectors so the vertex count stays under MAX_SHELL_VERTS.
    let mut rings = n_rings;
    let mut sectors = n_sectors;
    while (rings as usize) * (sectors as usize + 1) > MAX_SHELL_VERTS && rings > 1 {
        rings /= 2;
    }
    while (rings as usize) * (sectors as usize + 1) > MAX_SHELL_VERTS && sectors > 8 {
        sectors /= 2;
    }

    let v_count = (rings as usize) * (sectors as usize + 1);
    let mut vertices = Vec::with_capacity(v_count);
    let mut colors = Vec::with_capacity(v_count);

    // Log-spaced radii — denser sampling near the inner ring (where
    // perspective compresses more detail per pixel) and sparser as we
    // approach the horizon. `t ∈ [0, 1]` interpolates between
    // `inner` and `outer` on a log scale.
    let log_inner = inner_radius_m.ln();
    let log_outer = outer_radius_m.ln();

    let radius_for = |ring_idx: u32| -> f64 {
        let t = ring_idx as f64 / (rings as f64 - 1.0).max(1.0);
        let lr = log_inner + t * (log_outer - log_inner);
        lr.exp()
    };

    let world_radius = shape.radius_m();

    for ring in 0..rings {
        let r = radius_for(ring);
        for sector in 0..=sectors {
            // Sectors include both endpoints (no seam at θ = 0 / θ =
            // 2π) so the texture / vertex-color interpolation is
            // identical across the wraparound; the index buffer below
            // only references `sector ∈ [0, sectors)` for triangle
            // construction so the duplicated last column doesn't add
            // degenerate triangles.
            let theta = (sector as f64 / sectors as f64) * std::f64::consts::TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            // Observer-relative XZ position on the local tangent plane.
            let dx = r * cos_t;
            let dz = r * sin_t;
            // Step 8: sample macro state straight along the local
            // tangent plane (cube-world default). Step 9 adds the
            // sphere-curvature drop `-d²/(2R)` so the shell wraps
            // visibly past the horizon on a planetary radius.
            let world_x = observer.x + dx;
            let world_z = observer.z + dz;
            // For macro sampling we need a unit direction from world
            // center. On cube worlds the world center sits at origin
            // and the macro state is sampled by direction; we build
            // a direction vector through the sample point at the
            // observer's altitude band.
            let dir = DVec3::new(world_x, observer.y, world_z);
            let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
            let dir_norm = if len > 1e-3 {
                DVec3::new(dir.x / len, dir.y / len, dir.z / len)
            } else {
                DVec3::new(0.0, 1.0, 0.0)
            };
            let sample = macro_state.sample(dir_norm);

            // Curvature drop (sphere only — cube returns radius_m()
            // for `radius_m` but we keep flat there for v1; Step 9
            // gates this on shape type). Drop = -d²/(2R) so a vertex
            // at the horizon distance dips below the tangent plane by
            // roughly its own altitude, which is exactly the
            // geometric horizon offset.
            let curvature_drop = match shape {
                WorldShape::Sphere { .. } => -(r * r) / (2.0 * world_radius.max(1.0)),
                _ => 0.0,
            };

            let elev = sample.elev_m as f64;
            let y = elev + curvature_drop;

            let color = vertex_color(sample.biome_id, sample.water_kind, sample.elev_m);
            vertices.push([dx as f32, y as f32, dz as f32]);
            colors.push(color);
        }
    }

    // Index buffer — quads as two triangles per (ring, sector) cell.
    let mut indices = Vec::with_capacity((rings as usize - 1) * sectors as usize * 6);
    let stride = sectors + 1;
    for ring in 0..(rings - 1) {
        for sector in 0..sectors {
            let i0 = ring * stride + sector;
            let i1 = ring * stride + sector + 1;
            let i2 = (ring + 1) * stride + sector;
            let i3 = (ring + 1) * stride + sector + 1;
            // Front face = CCW projected on screen for the standard
            // eye-level observer (camera near tangent plane, looking
            // outward). Bevy uses right-handed Y-up; with these vertex
            // positions on the XZ plane and Y as elevation, the winding
            // (i0, i1, i2) + (i1, i3, i2) makes the *top* face the front
            // face, so a viewer above the plane sees it.
            indices.push(i0);
            indices.push(i1);
            indices.push(i2);
            indices.push(i1);
            indices.push(i3);
            indices.push(i2);
        }
    }

    HorizonShellMesh {
        vertices,
        colors,
        indices,
        r_inner_m: inner_radius_m as f32,
        r_outer_m: outer_radius_m as f32,
    }
}

/// Crude biome / water → linear RGB lookup. Picked to read as plausible
/// "from the horizon" terrain at a glance — desaturated greens for
/// vegetation biomes, browns for arid, white for snow, blue for water
/// surfaces. Tuned for Phase 19.2 baseline; later phases can swap in a
/// dedicated `BiomePalette` once one lands in
/// [`atomr_worlds_generate::macro_state::biome`].
fn vertex_color(biome_id: u8, water_kind: u8, elev_m: f32) -> [f32; 4] {
    // Water surfaces dominate biome color so lakes / oceans don't read
    // as green at the horizon.
    if water_kind != water_kind::NONE {
        // Slight darker for deeper water — encoded by elevation being
        // below sea level on the elev field for ocean tiles.
        let depth_factor = ((-elev_m).max(0.0) / 200.0).min(1.0);
        let r = 0.08 * (1.0 - depth_factor) + 0.04 * depth_factor;
        let g = 0.20 * (1.0 - depth_factor) + 0.10 * depth_factor;
        let b = 0.50 * (1.0 - depth_factor) + 0.30 * depth_factor;
        return [r, g, b, 1.0];
    }
    // Biome-driven base color. The exact `biome_id` enum is in
    // `atomr-worlds-generate/src/macro_state/biome.rs`; rather than
    // hardcode every variant we use a coarse switch that covers the
    // common bands. Anything else falls through to a desaturated
    // green-grey so the shell never reads as black.
    let base: [f32; 3] = match biome_id {
        // Ocean / temperate water-coast biomes (water shouldn't reach
        // here, but elev underwater still gets the water branch above).
        0 => [0.08, 0.20, 0.50],
        // Tundra / cold deserts.
        1 | 2 => [0.55, 0.55, 0.50],
        // Grass / steppes.
        3 | 4 => [0.32, 0.45, 0.20],
        // Boreal / temperate forest.
        5 | 6 => [0.20, 0.35, 0.18],
        // Tropical / rainforest.
        7 => [0.12, 0.30, 0.12],
        // Hot desert.
        8 | 9 => [0.65, 0.55, 0.35],
        // Snow / ice.
        10 => [0.85, 0.88, 0.92],
        _ => [0.30, 0.35, 0.25],
    };
    // Subtle altitude attenuation — higher reads slightly lighter
    // (snow caps); below sea-level reads slightly darker.
    let alt = (elev_m / 2000.0).clamp(-1.0, 1.0);
    let lift = 0.10 * alt.max(0.0);
    let dim = 0.10 * (-alt).max(0.0);
    [
        (base[0] + lift - dim).clamp(0.0, 1.0),
        (base[1] + lift - dim).clamp(0.0, 1.0),
        (base[2] + lift - dim).clamp(0.0, 1.0),
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_generate::macro_state::{DefaultMacroGenerator, MacroConfig, MacroGenerator};
    use std::sync::Arc;

    fn small_macro() -> Arc<WorldMacroState> {
        let gen = DefaultMacroGenerator::new(MacroConfig {
            grid_level: 1,
            ..MacroConfig::default()
        });
        gen.generate(42, WorldShape::Cube { edge_m: 1.0e7 })
    }

    #[test]
    fn empty_when_outer_le_inner() {
        let m = small_macro();
        let shape = WorldShape::Cube { edge_m: 1.0e7 };
        let out = bake_polar_annulus(&*m, shape, DVec3::new(0.0, 0.0, 0.0), 1000.0, 1000.0, 8, 16);
        assert!(out.vertices.is_empty());
        assert!(out.indices.is_empty());
    }

    #[test]
    fn topology_indices_match_grid() {
        let m = small_macro();
        let shape = WorldShape::Cube { edge_m: 1.0e7 };
        let out = bake_polar_annulus(&*m, shape, DVec3::new(0.0, 0.0, 0.0), 1000.0, 4000.0, 8, 16);
        // (rings-1) * sectors * 6 indices for 8 rings, 16 sectors.
        assert_eq!(out.indices.len(), 7 * 16 * 6);
        // Every index in bounds.
        let max = out.vertices.len() as u32;
        for &i in &out.indices {
            assert!(i < max, "index {i} >= verts {max}");
        }
    }

    #[test]
    fn determinism_same_inputs_same_mesh() {
        let m = small_macro();
        let shape = WorldShape::Cube { edge_m: 1.0e7 };
        let a = bake_polar_annulus(&*m, shape, DVec3::new(100.0, 0.0, 200.0), 800.0, 3000.0, 16, 32);
        let b = bake_polar_annulus(&*m, shape, DVec3::new(100.0, 0.0, 200.0), 800.0, 3000.0, 16, 32);
        assert_eq!(a.vertices, b.vertices);
        assert_eq!(a.colors, b.colors);
        assert_eq!(a.indices, b.indices);
    }

    #[test]
    fn vertex_count_caps_at_max() {
        let m = small_macro();
        let shape = WorldShape::Cube { edge_m: 1.0e7 };
        let out = bake_polar_annulus(&*m, shape, DVec3::new(0.0, 0.0, 0.0), 800.0, 4000.0, 1024, 1024);
        assert!(out.vertices.len() <= MAX_SHELL_VERTS, "verts={}", out.vertices.len());
    }

    #[test]
    fn sphere_curvature_drops_outer_vertex_below_tangent() {
        let m = small_macro();
        let shape = WorldShape::Sphere { radius_m: 6_371_000.0 };
        // Use an outer radius small enough that the curvature drop is
        // measurable but the macro sample for elevation is consistent.
        let out = bake_polar_annulus(&*m, shape, DVec3::new(0.0, 0.0, 0.0), 200.0, 16_000.0, 8, 4);
        // First ring sits at the inner radius; last ring at outer.
        // Compute expected drop magnitudes and confirm the outer ring's
        // average Y is lower than the inner ring's average Y by at
        // least the curvature delta (modulo elev-field bias).
        let stride = 5; // sectors + 1
        let inner_y: f32 = out.vertices[0..stride].iter().map(|v| v[1]).sum::<f32>() / stride as f32;
        let last_ring_start = out.vertices.len() - stride;
        let outer_y: f32 = out.vertices[last_ring_start..]
            .iter().map(|v| v[1]).sum::<f32>() / stride as f32;
        let curvature_inner = -(200.0 * 200.0) / (2.0 * 6_371_000.0);
        let curvature_outer = -(16_000.0_f32 * 16_000.0_f32) / (2.0 * 6_371_000.0);
        let curvature_delta = curvature_outer - curvature_inner as f32;
        // Allow generous slack for elevation variation across the
        // macro sample, but the directional sign must match.
        assert!(curvature_delta < 0.0);
        assert!(outer_y - inner_y < 0.0,
            "outer_y={outer_y}, inner_y={inner_y}, expected outer below inner by ~{curvature_delta}");
    }
}
