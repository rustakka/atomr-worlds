//! Cross-brick GPU buffer + material cache for the DAG raymarcher.
//!
//! Procedural terrain produces vast numbers of *structurally identical* bricks
//! — solid stone, flat strata, open air rims. Without dedup, every such brick
//! uploads its own `nodes`/`colors` storage buffers and its own
//! [`RaymarchMaterial`], even though the bytes are identical. This cache keys
//! buffers by the DAG's stable content [`digest`](atomr_worlds_voxel::DagBrick::digest)
//! so all bricks with the same shape share one buffer set, and keys materials by
//! `(digest, tier)` so a tier flip reuses buffers and rebuilds only the tiny
//! material. With raymarch as the default render path this is what keeps VRAM
//! bounded as the world streams.
//!
//! ## Refcounting
//!
//! Two layers, both refcounted, kept in lockstep with [`crate::world_stream::LoadedChunks`]
//! eviction:
//! - a **material** entry per `(digest, tier)`, increfed once per spawned brick
//!   entity using it;
//! - a **buffer** entry per `digest`, increfed once per *distinct live material
//!   entry* referencing it (typically 1–3, one per in-use tier).
//!
//! When a material entry's count hits 0 the cache drops its strong
//! `Handle<RaymarchMaterial>` (freeing the asset) and decrefs the buffer entry;
//! when the buffer entry hits 0 the buffers are freed. Entities hold handle
//! clones, so GPU memory frees exactly when no entity references it.
//!
//! ## Future: edit-time rebuild
//!
//! No client voxel-edit path exists yet. When one lands, editing brick `B`
//! becomes: rebuild only `B`'s [`DagGpuWithDigest`] on the blocking pool,
//! [`release`](DagBufferCache::release) the old `(digest, tier)`,
//! [`acquire`](DagBufferCache::acquire) the new one, and swap the entity's
//! `MeshMaterial3d` handle. An edit producing an already-resident shape costs
//! zero new buffers — the dedup makes per-brick rebuild cheap.

use std::collections::HashMap;

use atomr_worlds_voxel::{DagGpuWithDigest, DAG_GPU_EMPTY_ROOT};
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;

use super::raymarch::{
    raymarch_material_from_parts, raymarch_meta, upload_dag_buffers, RaymarchMaterial,
    RaymarchShadingTier,
};

/// Shared `nodes`/`colors` buffers for one DAG shape, refcounted by the number
/// of live material entries (across tiers) that reference them.
struct BufferEntry {
    nodes: Handle<ShaderStorageBuffer>,
    colors: Handle<ShaderStorageBuffer>,
    refcount: u32,
    /// Resident GPU bytes (nodes + widened colors) — for VRAM accounting.
    bytes: usize,
}

/// A per-`(digest, tier)` material, refcounted by the number of live brick
/// entities using it.
struct MaterialEntry {
    material: Handle<RaymarchMaterial>,
    refcount: u32,
}

/// Outcome of an [`acquire`](DagBufferCache::acquire): the material handle to put
/// on the brick entity, plus whether it was a fresh build (for the hit/miss
/// counters). Resident VRAM / dedup ratio are read separately via
/// [`resident_bytes`](DagBufferCache::resident_bytes) /
/// [`buffer_count`](DagBufferCache::buffer_count).
pub struct Acquired {
    /// Material handle for the brick's proxy entity.
    pub material: Handle<RaymarchMaterial>,
    /// `true` when a new material asset was built (a `(digest, tier)` miss).
    pub material_miss: bool,
}

/// Refcounted dedup cache mapping DAG content → shared GPU buffers + materials.
#[derive(Resource, Default)]
pub struct DagBufferCache {
    buffers: HashMap<u64, BufferEntry>,
    materials: HashMap<(u64, RaymarchShadingTier), MaterialEntry>,
}

/// Lightweight per-session counters for the mesh-vs-raymarch perf comparison
/// (dumped by the harness `dump_brick_mem` event). Cumulative over the run;
/// combined with [`DagBufferCache::resident_bytes`]/[`buffer_count`](DagBufferCache::buffer_count)
/// at dump time. Bumped in `spawn_brick_entity` — counters only, no hot-path
/// systems.
#[derive(Resource, Default)]
pub struct BrickGpuStats {
    /// Raymarch proxies spawned (one per non-empty brick rendered via DAG).
    pub raymarch_spawns: u64,
    /// Cumulative main-thread time in [`DagBufferCache::acquire`] (ns). The heavy
    /// DAG build is off-thread, so this is just hashmap lookups + buffer uploads
    /// on a miss — perf-bar criterion 4 checks it's a tiny fraction of frame time.
    pub acquire_ns_total: u128,
    /// `(digest, tier)` material cache hits at spawn (dedup denominator).
    pub cache_hits: u64,
    /// `(digest, tier)` material cache misses at spawn (uploads/builds).
    pub cache_misses: u64,
    /// Bricks rendered via a mesh path (split or palette).
    pub mesh_spawns: u64,
    /// Cumulative mesh vertices uploaded (for a mesh-vs-DAG VRAM estimate).
    pub mesh_vertices: u64,
    /// Cumulative mesh indices uploaded.
    pub mesh_indices: u64,
}

impl DagBufferCache {
    /// Get-or-create the material for `bundle` at `tier`, increffing the relevant
    /// refcounts. Returns `None` for an empty DAG (no proxy is spawned). The
    /// caller must record `bundle.digest` + `tier` on the brick's
    /// [`LoadedChunk`](crate::world_stream::LoadedChunk) and call
    /// [`release`](Self::release) when it is evicted.
    pub fn acquire(
        &mut self,
        bundle: &DagGpuWithDigest,
        tier: RaymarchShadingTier,
        palette: Handle<ShaderStorageBuffer>,
        storage_buffers: &mut Assets<ShaderStorageBuffer>,
        materials: &mut Assets<RaymarchMaterial>,
    ) -> Option<Acquired> {
        let gpu = &bundle.gpu;
        if gpu.root == DAG_GPU_EMPTY_ROOT || gpu.nodes.is_empty() {
            return None;
        }
        let digest = bundle.digest;
        let mat_key = (digest, tier);

        // Material hit: same shape + same tier already built — just incref.
        if let Some(me) = self.materials.get_mut(&mat_key) {
            me.refcount += 1;
            return Some(Acquired { material: me.material.clone(), material_miss: false });
        }

        // Material miss. Ensure the shared buffers exist (buffer hit across a
        // different tier of the same shape, or a fresh upload), increffing the
        // buffer entry for this new material entry.
        let (nodes, colors) = if let Some(be) = self.buffers.get_mut(&digest) {
            be.refcount += 1;
            (be.nodes.clone(), be.colors.clone())
        } else {
            let (nodes, colors) = upload_dag_buffers(gpu, storage_buffers);
            let bytes = gpu.nodes.len() * 4 + gpu.colors.len() * 4; // widened u16→u32
            self.buffers.insert(
                digest,
                BufferEntry { nodes: nodes.clone(), colors: colors.clone(), refcount: 1, bytes },
            );
            (nodes, colors)
        };

        let meta = raymarch_meta(gpu, tier, bundle.aabb_min, bundle.aabb_max);
        let material = materials.add(raymarch_material_from_parts(nodes, colors, palette, meta));
        self.materials.insert(mat_key, MaterialEntry { material: material.clone(), refcount: 1 });
        Some(Acquired { material, material_miss: true })
    }

    /// Decref the `(digest, tier)` material; when it reaches 0 free the material
    /// and decref its shared buffers (freed when their last tier-variant dies).
    /// No-op if the key is unknown (mesh/empty bricks never acquired).
    pub fn release(&mut self, digest: u64, tier: RaymarchShadingTier) {
        let key = (digest, tier);
        let Some(me) = self.materials.get_mut(&key) else { return };
        me.refcount -= 1;
        if me.refcount == 0 {
            self.materials.remove(&key);
            if let Some(be) = self.buffers.get_mut(&digest) {
                be.refcount -= 1;
                if be.refcount == 0 {
                    self.buffers.remove(&digest);
                }
            }
        }
    }

    /// Number of distinct DAG shapes with resident buffers (dedup denominator).
    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Total resident GPU buffer bytes across all cached shapes.
    #[inline]
    pub fn resident_bytes(&self) -> usize {
        self.buffers.values().map(|b| b.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_voxel::{Brick, DagBrick, Voxel};

    fn uniform_bundle(material: u16) -> DagGpuWithDigest {
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    b.set(IVec3::new(x, y, z), Voxel::new(material));
                }
            }
        }
        DagBrick::from_brick(&b).to_gpu_with_digest(&b).unwrap()
    }

    fn sparse_bundle() -> DagGpuWithDigest {
        let mut b = Brick::new();
        b.set(IVec3::new(1, 2, 3), Voxel::new(7));
        DagBrick::from_brick(&b).to_gpu_with_digest(&b).unwrap()
    }

    fn stores() -> (Assets<ShaderStorageBuffer>, Assets<RaymarchMaterial>) {
        (Assets::default(), Assets::default())
    }

    #[test]
    fn identical_bricks_share_one_buffer_set() {
        let mut cache = DagBufferCache::default();
        let (mut sb, mut mats) = stores();
        let palette = Handle::default();
        let bundle = uniform_bundle(1);

        let a = cache.acquire(&bundle, RaymarchShadingTier::Lambert, palette.clone(), &mut sb, &mut mats).unwrap();
        let b = cache.acquire(&bundle, RaymarchShadingTier::Lambert, palette.clone(), &mut sb, &mut mats).unwrap();

        assert!(a.material_miss, "first acquire builds the material");
        assert!(!b.material_miss, "second acquire is a cache hit");
        assert_eq!(a.material.id(), b.material.id(), "identical bricks share the material");
        assert_eq!(cache.buffer_count(), 1, "one shape ⇒ one buffer set");
    }

    #[test]
    fn release_lifecycle_frees_buffers() {
        let mut cache = DagBufferCache::default();
        let (mut sb, mut mats) = stores();
        let palette = Handle::default();
        let bundle = uniform_bundle(2);
        let tier = RaymarchShadingTier::Lambert;

        cache.acquire(&bundle, tier, palette.clone(), &mut sb, &mut mats).unwrap();
        cache.acquire(&bundle, tier, palette.clone(), &mut sb, &mut mats).unwrap();
        assert_eq!(cache.buffer_count(), 1);

        cache.release(bundle.digest, tier); // one of two entities gone
        assert_eq!(cache.buffer_count(), 1, "still one live reference");
        cache.release(bundle.digest, tier); // last entity gone
        assert_eq!(cache.buffer_count(), 0, "buffers freed when last reference drops");
        assert_eq!(cache.resident_bytes(), 0);
    }

    #[test]
    fn different_tiers_share_buffers_distinct_materials() {
        let mut cache = DagBufferCache::default();
        let (mut sb, mut mats) = stores();
        let palette = Handle::default();
        let bundle = uniform_bundle(3);

        let lam = cache.acquire(&bundle, RaymarchShadingTier::Lambert, palette.clone(), &mut sb, &mut mats).unwrap();
        assert_eq!(cache.buffer_count(), 1, "first tier uploads buffers");
        let unlit = cache.acquire(&bundle, RaymarchShadingTier::Unlit, palette.clone(), &mut sb, &mut mats).unwrap();

        assert!(lam.material_miss && unlit.material_miss, "each tier builds its own material");
        assert_ne!(lam.material.id(), unlit.material.id(), "tiers get distinct materials");
        assert_eq!(cache.buffer_count(), 1, "both tiers share one buffer set");

        // Releasing one tier keeps the shared buffers alive for the other.
        cache.release(bundle.digest, RaymarchShadingTier::Lambert);
        assert_eq!(cache.buffer_count(), 1);
        cache.release(bundle.digest, RaymarchShadingTier::Unlit);
        assert_eq!(cache.buffer_count(), 0);
    }

    #[test]
    fn empty_dag_acquires_nothing() {
        use atomr_worlds_voxel::{DagGpu, DAG_GPU_EMPTY_ROOT};
        let mut cache = DagBufferCache::default();
        let (mut sb, mut mats) = stores();
        let empty = DagGpuWithDigest {
            gpu: DagGpu { nodes: vec![], colors: vec![], root: DAG_GPU_EMPTY_ROOT },
            digest: 0,
            aabb_min: [0, 0, 0],
            aabb_max: [0, 0, 0],
        };
        assert!(cache
            .acquire(&empty, RaymarchShadingTier::Lambert, Handle::default(), &mut sb, &mut mats)
            .is_none());
        assert_eq!(cache.buffer_count(), 0);
    }

    #[test]
    fn distinct_shapes_get_distinct_buffers() {
        let mut cache = DagBufferCache::default();
        let (mut sb, mut mats) = stores();
        let palette = Handle::default();
        cache.acquire(&uniform_bundle(1), RaymarchShadingTier::Lambert, palette.clone(), &mut sb, &mut mats).unwrap();
        cache.acquire(&sparse_bundle(), RaymarchShadingTier::Lambert, palette.clone(), &mut sb, &mut mats).unwrap();
        assert_eq!(cache.buffer_count(), 2, "two shapes ⇒ two buffer sets");
    }
}
