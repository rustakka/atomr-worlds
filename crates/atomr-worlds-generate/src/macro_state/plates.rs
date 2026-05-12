//! Tectonic plate Voronoi + plate motion → elevation deltas.
//!
//! Deterministic from `(world_seed, surface_grid, plate_count)`:
//! - Plate seeds: pick `plate_count` distinct face IDs via `splitmix64`
//!   draws against `world_seed`. Each plate gets a random unit velocity
//!   on the sphere's tangent plane at its seed.
//! - Assignment: BFS flood-fill from all seed faces simultaneously over
//!   the face neighbour graph. Each face's `plate_id` is the seed that
//!   reached it first. Ties broken by seed id (lowest wins).
//! - Mountain belts: for each face on a plate boundary, if the dot
//!   product of the plates' velocity vectors at that boundary is
//!   negative ("convergent"), add an orogenic uplift proportional to the
//!   convergence magnitude.
//!
//! All arithmetic is deterministic; no time, no I/O. Floating-point
//! determinism relies on the same `sqrt`/`splitmix64` bit-equality the
//! existing seed chain depends on.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::splitmix64;

use super::surface_grid::{FaceId, SurfaceGrid};

#[derive(Clone, Debug)]
pub struct PlateMap {
    /// Plate id per face. `len == grid.face_count()`.
    pub plate_id: Vec<u16>,
    /// Per-plate seed face (centre of the plate).
    pub seeds: Vec<FaceId>,
    /// Per-plate unit velocity in 3D (tangent to the seed face).
    pub velocity: Vec<DVec3>,
}

#[derive(Clone, Debug)]
pub struct ElevationField {
    /// Sea-level-relative elevation per face, in meters.
    pub elev_m: Vec<f32>,
}

#[derive(Copy, Clone, Debug)]
pub struct PlateConfig {
    pub plate_count: u16,
    /// Base elevation (m) for continental plates.
    pub continent_base_m: f32,
    /// Base elevation (m) for oceanic plates (negative).
    pub ocean_base_m: f32,
    /// Probability (in `[0, 65535]`) that a plate is oceanic vs continental.
    pub ocean_prob_q16: u16,
    /// Uplift per unit convergence at a plate boundary, in meters.
    pub uplift_per_convergence_m: f32,
}

impl Default for PlateConfig {
    fn default() -> Self {
        Self {
            plate_count: 24,
            continent_base_m: 400.0,
            ocean_base_m: -3500.0,
            ocean_prob_q16: 0x9999, // ≈60% oceanic, matches Earth
            uplift_per_convergence_m: 4500.0,
        }
    }
}

pub fn generate_plates(
    grid: &SurfaceGrid,
    world_seed: u64,
    cfg: PlateConfig,
) -> (PlateMap, ElevationField) {
    let n_faces = grid.face_count();
    let n_plates = (cfg.plate_count as usize).min(n_faces);
    let mut seeds: Vec<FaceId> = Vec::with_capacity(n_plates);
    let mut taken = vec![false; n_faces];

    // Pick plate seeds — distinct faces drawn from splitmix64(seed ^ i).
    let mut i: u64 = 0;
    while seeds.len() < n_plates {
        let pick = splitmix64(world_seed ^ (i.wrapping_mul(0x9E37_79B9_7F4A_7C15))) % n_faces as u64;
        let pick = pick as FaceId;
        if !taken[pick as usize] {
            taken[pick as usize] = true;
            seeds.push(pick);
        }
        i += 1;
        // Defensive cap to avoid an infinite loop if user passes huge
        // plate_count + tiny grid; in practice plate_count << face_count.
        if i > (n_faces as u64 * 8) {
            break;
        }
    }

    // Multi-source BFS from all seeds simultaneously, processing one BFS
    // depth per round. At each round, every unassigned neighbour of the
    // current frontier gets a candidate assignment; collisions resolve to
    // the lowest plate id. This gives a true distance-Voronoi labeling
    // with deterministic tie-break, regardless of seed insertion order.
    let mut plate_id = vec![u16::MAX; n_faces];
    let mut current_round: Vec<(FaceId, u16)> = Vec::new();
    for (pid, &seed) in seeds.iter().enumerate() {
        plate_id[seed as usize] = pid as u16;
        current_round.push((seed, pid as u16));
    }
    while !current_round.is_empty() {
        let mut candidates: std::collections::HashMap<FaceId, u16> =
            std::collections::HashMap::new();
        for &(f, pid) in &current_round {
            for n in grid.neighbours_of(f) {
                if n == FaceId::MAX {
                    continue;
                }
                if plate_id[n as usize] != u16::MAX {
                    continue;
                }
                candidates
                    .entry(n)
                    .and_modify(|p| {
                        if pid < *p {
                            *p = pid;
                        }
                    })
                    .or_insert(pid);
            }
        }
        let mut next_round: Vec<(FaceId, u16)> = candidates.into_iter().collect();
        // Sort to ensure deterministic frontier ordering across runs.
        next_round.sort_by_key(|(f, _)| *f);
        for &(n, pid) in &next_round {
            plate_id[n as usize] = pid;
        }
        current_round = next_round;
    }

    // Assign per-plate velocity: pick a random unit vector via two
    // splitmix64 draws → spherical coordinates → tangent at the seed.
    // We compute the 3D vector directly and tangent-project at use sites.
    let mut velocity = Vec::with_capacity(n_plates);
    let mut plate_is_ocean = vec![false; n_plates];
    for (pid, &seed) in seeds.iter().enumerate() {
        let h1 = splitmix64(world_seed.wrapping_add(0xA1B2_C3D4_E5F6_0789).wrapping_add(pid as u64));
        let h2 = splitmix64(world_seed.wrapping_add(0xBE_EF_CA_FE_C0_FF_EE_42).wrapping_add(pid as u64));
        let h3 = splitmix64(world_seed.wrapping_add(0xFEED_FACE_DEAD_BEEF).wrapping_add(pid as u64));
        // x,y,z each in [-1, 1] via u64 → f64.
        let x = ((h1 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let y = ((h2 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let z = ((h3 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let len = (x * x + y * y + z * z).sqrt().max(1e-12);
        let v_world = DVec3::new(x / len, y / len, z / len);
        // Tangent-project at the seed face's centroid.
        let n = grid.face_centroid(seed);
        let d = v_world.x * n.x + v_world.y * n.y + v_world.z * n.z;
        let t = DVec3::new(v_world.x - d * n.x, v_world.y - d * n.y, v_world.z - d * n.z);
        let tlen = (t.x * t.x + t.y * t.y + t.z * t.z).sqrt().max(1e-12);
        velocity.push(DVec3::new(t.x / tlen, t.y / tlen, t.z / tlen));

        let oc = splitmix64(world_seed.wrapping_add(0xCAFE_BABE_DEAD_F00D).wrapping_add(pid as u64));
        plate_is_ocean[pid] = ((oc & 0xFFFF) as u16) < cfg.ocean_prob_q16;
    }

    // Build elevation. Base by plate type, plus uplift along convergent
    // boundaries proportional to plate-velocity convergence.
    let mut elev_m = vec![0.0_f32; n_faces];
    for f in 0..n_faces {
        let pid = plate_id[f] as usize;
        if pid >= n_plates {
            // Should not happen (every face fills); fall back to oceanic.
            elev_m[f] = cfg.ocean_base_m;
            continue;
        }
        elev_m[f] = if plate_is_ocean[pid] { cfg.ocean_base_m } else { cfg.continent_base_m };

        // Boundary check: if any neighbour belongs to a different plate,
        // accumulate uplift based on the relative velocity dotted with
        // the face-to-neighbour direction.
        let centre = grid.face_centroid(f as FaceId);
        let mut uplift = 0.0f32;
        for n in grid.neighbours_of(f as FaceId) {
            if n == FaceId::MAX {
                continue;
            }
            let npid = plate_id[n as usize] as usize;
            if npid == pid {
                continue;
            }
            let dir_to_n = grid.face_centroid(n);
            let dir = DVec3::new(dir_to_n.x - centre.x, dir_to_n.y - centre.y, dir_to_n.z - centre.z);
            let dlen = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt().max(1e-12);
            let dir = DVec3::new(dir.x / dlen, dir.y / dlen, dir.z / dlen);
            // Convergence = (v_self - v_other) · dir_to_n. Positive means
            // the plates are crashing into each other along this edge.
            let v_self = velocity[pid];
            let v_other = velocity[npid];
            let rel = DVec3::new(v_self.x - v_other.x, v_self.y - v_other.y, v_self.z - v_other.z);
            let conv = (rel.x * dir.x + rel.y * dir.y + rel.z * dir.z) as f32;
            if conv > 0.0 {
                uplift += conv * cfg.uplift_per_convergence_m * 0.5;
            }
        }
        elev_m[f] += uplift;
    }

    (
        PlateMap { plate_id, seeds, velocity },
        ElevationField { elev_m },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_face_gets_a_plate() {
        let g = SurfaceGrid::new(3);
        let cfg = PlateConfig { plate_count: 24, ..PlateConfig::default() };
        let (plates, _elev) = generate_plates(&g, 0xCAFE_F00D, cfg);
        assert_eq!(plates.plate_id.len(), g.face_count());
        for (i, &p) in plates.plate_id.iter().enumerate() {
            assert!(p < cfg.plate_count, "face {i} got plate_id {p}");
        }
    }

    #[test]
    fn plate_count_below_face_count_is_respected() {
        let g = SurfaceGrid::new(2);
        let cfg = PlateConfig { plate_count: 8, ..PlateConfig::default() };
        let (plates, _) = generate_plates(&g, 0xDEAD_BEEF, cfg);
        assert_eq!(plates.seeds.len(), 8);
    }

    #[test]
    fn deterministic_from_seed() {
        let g = SurfaceGrid::new(2);
        let cfg = PlateConfig::default();
        let (a, ae) = generate_plates(&g, 0xABCD_1234, cfg);
        let (b, be) = generate_plates(&g, 0xABCD_1234, cfg);
        assert_eq!(a.plate_id, b.plate_id);
        assert_eq!(a.seeds, b.seeds);
        for i in 0..a.velocity.len() {
            assert_eq!(a.velocity[i].x.to_bits(), b.velocity[i].x.to_bits());
            assert_eq!(a.velocity[i].y.to_bits(), b.velocity[i].y.to_bits());
            assert_eq!(a.velocity[i].z.to_bits(), b.velocity[i].z.to_bits());
        }
        for i in 0..ae.elev_m.len() {
            assert_eq!(ae.elev_m[i].to_bits(), be.elev_m[i].to_bits());
        }
    }

    #[test]
    fn different_seed_yields_different_layout() {
        let g = SurfaceGrid::new(2);
        let cfg = PlateConfig::default();
        let (a, _) = generate_plates(&g, 0x1111, cfg);
        let (b, _) = generate_plates(&g, 0x2222, cfg);
        // At least one face's plate id should differ.
        assert!(a.plate_id != b.plate_id);
    }
}
