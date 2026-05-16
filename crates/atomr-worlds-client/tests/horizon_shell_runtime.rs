//! Phase 19.2 — horizon-imposter shell integration test.
//!
//! The client crate is binary-only (no `lib.rs`), so this test exercises
//! the same `bake_polar_annulus` contract the `HorizonShellPlugin` in
//! `src/render/horizon_shell.rs` consumes, without spinning up Bevy. The
//! unit tests inside `atomr-worlds-view/src/derived/horizon_shell.rs`
//! already cover the baker's invariants (topology, determinism, vertex
//! cap, sphere curvature drop); this file is the end-to-end
//! "drift-driven rebuild → identical-pose deduplication" walk-through.
//!
//! Assertions:
//! 1. Identical `(macro_state, shape, observer, radii)` → bit-identical
//!    mesh. The plugin uses this property to dedupe re-bakes via
//!    `HorizonImposterMesh::source_digest`.
//! 2. Drifting the observer past the strategy's `rebuild_drift_m()`
//!    threshold (64 m default) changes the baked mesh — the
//!    elevation + biome sample lookups land on different macro faces.
//! 3. The shell mesh stays well below the vertex cap at the default
//!    `n_rings=32, n_sectors=128` configuration so a real-world bake
//!    never hits the cap fallback.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::{
    DefaultMacroGenerator, MacroConfig, MacroGenerator,
};
use atomr_worlds_view::{bake_polar_annulus, MAX_SHELL_VERTS};

fn cube_shape() -> WorldShape {
    WorldShape::Cube { edge_m: 1.0e7 }
}

fn sphere_shape() -> WorldShape {
    WorldShape::Sphere { radius_m: 6_371_000.0 }
}

fn build_macro_state(shape: WorldShape) -> std::sync::Arc<atomr_worlds_generate::WorldMacroState> {
    let gen = DefaultMacroGenerator::new(MacroConfig {
        grid_level: 2,
        ..MacroConfig::default()
    });
    gen.generate(0xDEAD_BEEF, shape)
}

#[test]
fn identical_inputs_produce_identical_mesh() {
    let shape = cube_shape();
    let macro_state = build_macro_state(shape);
    let observer = DVec3::new(100.0, 32.0, 200.0);
    let inner = 1024.0;
    let outer = 16_000.0;
    let a = bake_polar_annulus(&macro_state, shape, observer, inner, outer, 32, 128);
    let b = bake_polar_annulus(&macro_state, shape, observer, inner, outer, 32, 128);
    assert_eq!(a.vertices, b.vertices, "vertex positions must match");
    assert_eq!(a.colors, b.colors, "vertex colors must match");
    assert_eq!(a.indices, b.indices, "indices must match");
}

#[test]
fn drift_past_threshold_changes_mesh() {
    // Use a sphere world so the elevation field varies enough across a
    // 64 m drift that we see a visible color/elevation change. Cube
    // worlds have no macro state baked in practice so they'd hit the
    // default elevation field.
    let shape = sphere_shape();
    let macro_state = build_macro_state(shape);
    let inner = 1024.0;
    let outer = 16_000.0;
    let a = bake_polar_annulus(
        &macro_state,
        shape,
        DVec3::new(0.0, 0.0, 0.0),
        inner,
        outer,
        32,
        128,
    );
    // 65 m drift in +X — just past the strategy's 64 m rebuild_drift_m
    // default.
    let b = bake_polar_annulus(
        &macro_state,
        shape,
        DVec3::new(65.0, 0.0, 0.0),
        inner,
        outer,
        32,
        128,
    );
    // XZ components of each vertex are observer-relative offsets
    // (r·cos θ, r·sin θ) — pose-invariant across bakes. The Y
    // component, however, IS the macro-sampled elevation in world Y,
    // so it moves when the observer drifts to a face with a different
    // elevation. Plus the vertex color shifts when the sampled biome /
    // water-kind changes. Asserting on both keeps the test robust to
    // either signal.
    let xz_invariant = a
        .vertices
        .iter()
        .zip(b.vertices.iter())
        .all(|(va, vb)| va[0] == vb[0] && va[2] == vb[2]);
    assert!(xz_invariant, "vertex XZ should be pose-invariant (observer-relative)");
    let any_y_or_color_changed = a
        .vertices
        .iter()
        .zip(b.vertices.iter())
        .any(|(va, vb)| va[1] != vb[1])
        || a.colors
            .iter()
            .zip(b.colors.iter())
            .any(|(ca, cb)| ca != cb);
    assert!(
        any_y_or_color_changed,
        "drifting 65 m should land on a different macro face for at least one vertex",
    );
}

#[test]
fn default_config_stays_under_vertex_cap() {
    let shape = cube_shape();
    let macro_state = build_macro_state(shape);
    let observer = DVec3::new(0.0, 32.0, 0.0);
    // The HorizonShellPlugin's default config is n_rings=32, n_sectors=128.
    let baked = bake_polar_annulus(&macro_state, shape, observer, 1024.0, 16_000.0, 32, 128);
    assert!(
        baked.vertices.len() <= MAX_SHELL_VERTS,
        "default n_rings=32 n_sectors=128 → {} verts, must fit under MAX_SHELL_VERTS={}",
        baked.vertices.len(),
        MAX_SHELL_VERTS,
    );
    // Topology: (rings-1) * sectors * 6 indices.
    assert_eq!(baked.indices.len(), 31 * 128 * 6);
    assert_eq!(baked.r_inner_m, 1024.0);
    assert_eq!(baked.r_outer_m, 16_000.0);
}

#[test]
fn empty_when_outer_le_inner() {
    let shape = cube_shape();
    let macro_state = build_macro_state(shape);
    let observer = DVec3::new(0.0, 32.0, 0.0);
    // outer < inner triggers the early-return guard in
    // `HorizonShellPlugin::sync_horizon_shell` (which short-circuits
    // before spawning a bake). The baker handles the same case for
    // defense in depth.
    let baked = bake_polar_annulus(&macro_state, shape, observer, 2000.0, 1000.0, 32, 128);
    assert!(baked.vertices.is_empty());
    assert!(baked.indices.is_empty());
}
