//! Flora strategies: L-system trees and blue-noise grass tufts.
//!
//! Shared turtle-graphics primitives live here so both submodules can stamp
//! voxels through one interpreter. Concrete `FloraStrategy` impls are in the
//! `lsystem` and `grass` submodules.

use std::collections::HashMap;

use atomr_worlds_core::seed::splitmix64;
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use super::workspace::BrickWorkspace;

pub mod grass;
pub mod lsystem;

pub use grass::BlueNoiseGrass;
pub use lsystem::LSystemTrees;

/// L-system grammar: axiom + symbol -> replacement rules, expanded
/// `iterations` times to produce the turtle command string.
#[derive(Debug, Clone)]
pub struct LSystemGrammar {
    pub axiom: String,
    pub rules: HashMap<char, String>,
    pub iterations: u32,
    pub params: TurtleParams,
}

/// Turtle-graphics parameters consumed by `TurtleInterp`.
#[derive(Debug, Clone, Copy)]
pub struct TurtleParams {
    pub branch_length_m: f32,
    pub pitch_deg: f32,
    pub yaw_deg: f32,
    pub branch_radius_voxels: u32,
}

impl Default for TurtleParams {
    fn default() -> Self {
        Self {
            branch_length_m: 1.5,
            pitch_deg: 22.5,
            yaw_deg: 22.5,
            branch_radius_voxels: 1,
        }
    }
}

impl LSystemGrammar {
    /// Simple bracketed tree grammar: F -> FF[+F][-F][&F][^F]. Three
    /// iterations from a single-F axiom is enough for a small canopy that
    /// fits inside one brick. Higher iteration counts blow up exponentially
    /// and would clip outside the AABB.
    pub fn default_tree() -> Self {
        let mut rules = HashMap::new();
        rules.insert('F', "FF[+F][-F][&F][^F]".to_string());
        Self {
            axiom: "F".to_string(),
            rules,
            iterations: 3,
            params: TurtleParams::default(),
        }
    }

    /// Expand the axiom by `iterations` rule applications. Symbols absent
    /// from `rules` are preserved verbatim.
    pub fn derive(&self) -> String {
        let mut cur = self.axiom.clone();
        for _ in 0..self.iterations {
            let mut next = String::with_capacity(cur.len() * 2);
            for ch in cur.chars() {
                if let Some(repl) = self.rules.get(&ch) {
                    next.push_str(repl);
                } else {
                    next.push(ch);
                }
            }
            cur = next;
        }
        cur
    }
}

/// 3x3 rotation matrix (row-major) representing the turtle's local frame.
#[derive(Debug, Clone, Copy)]
struct Rot([[f32; 3]; 3]);

impl Rot {
    fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    #[inline]
    fn mul_vec(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.0[0][0] * v[0] + self.0[0][1] * v[1] + self.0[0][2] * v[2],
            self.0[1][0] * v[0] + self.0[1][1] * v[1] + self.0[1][2] * v[2],
            self.0[2][0] * v[0] + self.0[2][1] * v[1] + self.0[2][2] * v[2],
        ]
    }

    #[inline]
    fn mul(&self, other: &Rot) -> Rot {
        let mut o = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] = self.0[i][0] * other.0[0][j]
                    + self.0[i][1] * other.0[1][j]
                    + self.0[i][2] * other.0[2][j];
            }
        }
        Rot(o)
    }

    fn rot_x(rad: f32) -> Self {
        let (s, c) = rad.sin_cos();
        Self([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    }

    fn rot_z(rad: f32) -> Self {
        let (s, c) = rad.sin_cos();
        Self([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    }
}

#[derive(Debug, Clone, Copy)]
struct TurtleState {
    /// World-meter position (brick-local).
    pos: [f32; 3],
    /// Orientation: heading = local +Y by default (trees grow up).
    rot: Rot,
}

/// Stateful turtle-graphics interpreter. Translates an expanded L-system
/// string into voxel stamps inside `ws`. The trunk material is stamped
/// along every `F`; the canopy material is stamped at branch tips (any
/// `F` followed by `]` or end-of-string).
#[derive(Debug)]
pub struct TurtleInterp<'a> {
    ws: &'a mut BrickWorkspace,
    params: TurtleParams,
    trunk: Voxel,
    canopy: Voxel,
    seed: u64,
    rng: u64,
    /// Per-step max angle jitter in radians (small noise for organic look).
    jitter_rad: f32,
}

impl<'a> TurtleInterp<'a> {
    pub fn new(
        ws: &'a mut BrickWorkspace,
        params: TurtleParams,
        trunk: Voxel,
        canopy: Voxel,
        seed: u64,
    ) -> Self {
        Self {
            ws,
            params,
            trunk,
            canopy,
            seed,
            rng: seed,
            jitter_rad: 0.08,
        }
    }

    fn jitter(&mut self) -> f32 {
        self.rng = splitmix64(self.rng);
        // Map u64 -> [-1, 1].
        let u = (self.rng >> 11) as f64 / (1u64 << 53) as f64;
        (u * 2.0 - 1.0) as f32 * self.jitter_rad
    }

    pub fn run_at(&mut self, origin: [f32; 3], program: &str) {
        let mut stack: Vec<TurtleState> = Vec::new();
        let mut state = TurtleState { pos: origin, rot: Rot::identity() };
        // Step through the program; when we see `F`, walk a branch segment;
        // when we see `]` or end-of-string after an F, also stamp canopy at
        // the tip.
        let bytes = program.as_bytes();
        let pitch = self.params.pitch_deg.to_radians();
        let yaw = self.params.yaw_deg.to_radians();
        let mut last_was_f = false;
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'F' => {
                    let from = state.pos;
                    let dir = state.rot.mul_vec([0.0, 1.0, 0.0]);
                    let to = [
                        from[0] + dir[0] * self.params.branch_length_m,
                        from[1] + dir[1] * self.params.branch_length_m,
                        from[2] + dir[2] * self.params.branch_length_m,
                    ];
                    self.stamp_branch(from, to, self.trunk);
                    state.pos = to;
                    last_was_f = true;
                }
                b'+' => {
                    let j = self.jitter();
                    state.rot = state.rot.mul(&Rot::rot_x(pitch + j));
                }
                b'-' => {
                    let j = self.jitter();
                    state.rot = state.rot.mul(&Rot::rot_x(-pitch + j));
                }
                b'&' => {
                    let j = self.jitter();
                    state.rot = state.rot.mul(&Rot::rot_z(yaw + j));
                }
                b'^' => {
                    let j = self.jitter();
                    state.rot = state.rot.mul(&Rot::rot_z(-yaw + j));
                }
                b'[' => {
                    stack.push(state);
                }
                b']' => {
                    if last_was_f {
                        self.stamp_sphere(state.pos, self.canopy);
                    }
                    if let Some(s) = stack.pop() {
                        state = s;
                    }
                    last_was_f = false;
                }
                _ => {}
            }
            // Canopy at the very tip if program ends on F.
            if i + 1 == bytes.len() && last_was_f {
                self.stamp_sphere(state.pos, self.canopy);
            }
        }
        // Silence unused warning when seed isn't otherwise referenced.
        let _ = self.seed;
    }

    fn stamp_branch(&mut self, from: [f32; 3], to: [f32; 3], v: Voxel) {
        let steps = ((to[0] - from[0]).powi(2)
            + (to[1] - from[1]).powi(2)
            + (to[2] - from[2]).powi(2))
        .sqrt()
        .ceil() as i32;
        let steps = steps.max(1);
        let r = self.params.branch_radius_voxels as i32;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let p = [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
                from[2] + (to[2] - from[2]) * t,
            ];
            self.stamp_disc(p, r, v);
        }
    }

    fn stamp_disc(&mut self, center: [f32; 3], r: i32, v: Voxel) {
        let cx = center[0].round() as i32;
        let cy = center[1].round() as i32;
        let cz = center[2].round() as i32;
        let r2 = r * r;
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy + dz * dz > r2 {
                        continue;
                    }
                    let x = cx + dx;
                    let y = cy + dy;
                    let z = cz + dz;
                    if in_brick(x, y, z) {
                        self.ws.set_material(x, y, z, v);
                    }
                }
            }
        }
    }

    fn stamp_sphere(&mut self, center: [f32; 3], v: Voxel) {
        let r = (self.params.branch_radius_voxels as i32 + 1).max(2);
        self.stamp_disc(center, r, v);
    }
}

#[inline]
fn in_brick(x: i32, y: i32, z: i32) -> bool {
    (0..BRICK_EDGE as i32).contains(&x)
        && (0..BRICK_EDGE as i32).contains(&y)
        && (0..BRICK_EDGE as i32).contains(&z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tree_derive_terminates() {
        let g = LSystemGrammar::default_tree();
        let s = g.derive();
        assert!(!s.is_empty());
        // Three iterations of FF[+F][-F][&F][^F] from F: bounded.
        assert!(s.len() < 1_000_000, "derive runaway: {}", s.len());
    }

    #[test]
    fn derive_preserves_unknown_symbols() {
        let mut rules = HashMap::new();
        rules.insert('A', "AB".to_string());
        let g = LSystemGrammar {
            axiom: "A+".to_string(),
            rules,
            iterations: 2,
            params: TurtleParams::default(),
        };
        // A -> AB -> ABB; '+' is preserved.
        assert_eq!(g.derive(), "ABB+");
    }
}
