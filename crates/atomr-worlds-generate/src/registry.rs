//! Hierarchical generation-strategy registry.
//!
//! [`GeneratorRegistry`] holds a map of [`StrategyId`] → [`BrickGenerator`]
//! impls. A [`StrategySelector`] picks one of the registered ids
//! deterministically from a world seed; the [`GenerationPolicy`] (in
//! [`atomr-worlds-host::policy`]) decides whether to use that pick, force a
//! specific id, or short-circuit to empty.
//!
//! The selection rule is the *same source of truth* the host actor consults
//! once on spawn — see the actor's `resolve_on_spawn` flow in
//! `atomr_worlds_host::local`.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use atomr_worlds_core::addr::Address;

use crate::brick::BrickGenerator;

/// Stable strategy identifier (FNV-1a-64 of the strategy name).
pub type StrategyId = u64;

/// A choice of how generation should proceed for a given address. The host
/// crate's `policy.rs` produces values of this type from its `PolicyResolver`
/// and passes them to [`GeneratorRegistry::resolve`].
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum GenerationPolicy {
    /// Use the strategy chosen by the selector from the world seed.
    #[default]
    Seeded,
    /// Skip generation entirely; reads return `Voxel::EMPTY`. User writes
    /// still apply through the overlay.
    Empty,
    /// Force a specific registered strategy.
    Custom(StrategyId),
}

/// Const-time FNV-1a 64-bit hash of `name`. Used to build [`StrategyId`]
/// constants at compile time so the ids are stable across compilations.
pub const fn strategy_id(name: &str) -> StrategyId {
    let bytes = name.as_bytes();
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325; // FNV-1a 64 offset basis
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3); // FNV-1a 64 prime
        i += 1;
    }
    hash
}

pub const TERRAIN: StrategyId = strategy_id("terrain");
pub const GAS_GIANT: StrategyId = strategy_id("gas_giant");
pub const ASTEROID_BELT: StrategyId = strategy_id("asteroid_belt");
pub const EMPTY_PLANETOID: StrategyId = strategy_id("empty_planetoid");

/// Picks a strategy for an address deterministically from the seed.
/// Implementors must be pure.
pub trait StrategySelector: Send + Sync + Debug {
    fn pick(&self, addr: &Address, world_seed: u64) -> StrategyId;
}

/// Weighted-pick selector: chooses among registered ids using
/// `splitmix64(world_seed)` modulo the total weight. Iterates ids in sorted
/// order so registration order doesn't affect the pick — only the (id,
/// weight) pairs do.
#[derive(Debug, Clone)]
pub struct BuiltinSelector {
    pub weights: Vec<(StrategyId, u32)>,
}

impl BuiltinSelector {
    pub fn new(mut weights: Vec<(StrategyId, u32)>) -> Self {
        weights.sort_by_key(|(id, _)| *id);
        Self { weights }
    }

    /// Default mix used by [`default_registry`]: `terrain` only.
    pub fn terrain_only() -> Self {
        Self::new(vec![(TERRAIN, 1)])
    }
}

impl StrategySelector for BuiltinSelector {
    fn pick(&self, _addr: &Address, world_seed: u64) -> StrategyId {
        if self.weights.is_empty() {
            return TERRAIN;
        }
        let total: u64 = self.weights.iter().map(|(_, w)| *w as u64).sum();
        if total == 0 {
            return self.weights[0].0;
        }
        let mix = atomr_worlds_core::splitmix64(world_seed) % total;
        let mut acc: u64 = 0;
        for (id, w) in &self.weights {
            acc += *w as u64;
            if mix < acc {
                return *id;
            }
        }
        // Unreachable given `mix < total`, but be defensive.
        self.weights.last().unwrap().0
    }
}

/// Outcome of [`GeneratorRegistry::resolve`].
#[derive(Clone)]
pub enum Resolved {
    Generate { gen: Arc<dyn BrickGenerator>, strategy: StrategyId },
    Empty,
}

impl Debug for Resolved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Resolved::Generate { strategy, .. } => {
                f.debug_struct("Generate").field("strategy", strategy).finish_non_exhaustive()
            }
            Resolved::Empty => f.write_str("Empty"),
        }
    }
}

/// Registry of [`BrickGenerator`]s keyed by [`StrategyId`] plus a selector.
#[derive(Clone)]
pub struct GeneratorRegistry {
    by_id: HashMap<StrategyId, Arc<dyn BrickGenerator>>,
    selector: Arc<dyn StrategySelector>,
}

impl Debug for GeneratorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeneratorRegistry")
            .field("strategies", &self.by_id.keys().collect::<Vec<_>>())
            .field("selector", &self.selector)
            .finish()
    }
}

impl GeneratorRegistry {
    pub fn builder() -> GeneratorRegistryBuilder {
        GeneratorRegistryBuilder::default()
    }

    pub fn known_strategies(&self) -> Vec<StrategyId> {
        let mut v: Vec<_> = self.by_id.keys().copied().collect();
        v.sort();
        v
    }

    /// The one-shot resolution rule: given a policy and a world seed, return
    /// the generator (and the picked strategy id) or `Empty`.
    pub fn resolve(
        &self,
        addr: &Address,
        world_seed: u64,
        policy: GenerationPolicy,
    ) -> Result<Resolved, ResolveError> {
        match policy {
            GenerationPolicy::Empty => Ok(Resolved::Empty),
            GenerationPolicy::Custom(id) => match self.by_id.get(&id) {
                Some(gen) => Ok(Resolved::Generate { gen: gen.clone(), strategy: id }),
                None => Err(ResolveError::UnknownStrategy(id)),
            },
            GenerationPolicy::Seeded => {
                let id = self.selector.pick(addr, world_seed);
                match self.by_id.get(&id) {
                    Some(gen) => Ok(Resolved::Generate { gen: gen.clone(), strategy: id }),
                    None => Err(ResolveError::UnknownStrategy(id)),
                }
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct GeneratorRegistryBuilder {
    by_id: HashMap<StrategyId, Arc<dyn BrickGenerator>>,
    selector: Option<Arc<dyn StrategySelector>>,
}

impl Debug for GeneratorRegistryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeneratorRegistryBuilder")
            .field("strategies", &self.by_id.keys().collect::<Vec<_>>())
            .field("has_selector", &self.selector.is_some())
            .finish()
    }
}

impl GeneratorRegistryBuilder {
    pub fn register(mut self, id: StrategyId, g: Arc<dyn BrickGenerator>) -> Self {
        self.by_id.insert(id, g);
        self
    }

    pub fn selector(mut self, s: Arc<dyn StrategySelector>) -> Self {
        self.selector = Some(s);
        self
    }

    pub fn build(self) -> GeneratorRegistry {
        GeneratorRegistry {
            by_id: self.by_id,
            selector: self.selector.unwrap_or_else(|| Arc::new(BuiltinSelector::terrain_only())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("unknown strategy id: {0:#x}")]
    UnknownStrategy(StrategyId),
}

/// Default registry: terrain strategy registered with the [`TerrainGenerator`]
/// body. Existing callers using [`crate::WorldGen`] migrate via `.into()` or
/// by constructing this directly.
pub fn default_registry() -> GeneratorRegistry {
    use crate::strategies;
    GeneratorRegistry::builder()
        .register(TERRAIN, Arc::new(strategies::terrain::default_terrain()))
        .register(GAS_GIANT, Arc::new(strategies::gas_giant::GasGiantStub))
        .register(ASTEROID_BELT, Arc::new(strategies::asteroid_belt::AsteroidBeltStub))
        .register(EMPTY_PLANETOID, Arc::new(strategies::empty_planetoid::EmptyPlanetoidStrategy))
        .selector(Arc::new(BuiltinSelector::terrain_only()))
        .build()
}

impl From<crate::tiers::WorldGen> for GeneratorRegistry {
    fn from(wg: crate::tiers::WorldGen) -> Self {
        use crate::strategies;
        let terrain_arc: Arc<dyn BrickGenerator> = Arc::new(wg.brick_gen());
        GeneratorRegistry::builder()
            .register(TERRAIN, terrain_arc)
            .register(GAS_GIANT, Arc::new(strategies::gas_giant::GasGiantStub))
            .register(ASTEROID_BELT, Arc::new(strategies::asteroid_belt::AsteroidBeltStub))
            .register(EMPTY_PLANETOID, Arc::new(strategies::empty_planetoid::EmptyPlanetoidStrategy))
            .selector(Arc::new(BuiltinSelector::terrain_only()))
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::addr::{Address, WorldAddr};

    #[test]
    fn strategy_id_is_const_and_stable() {
        // Ensure const ids are non-zero and distinct for distinct names.
        assert_ne!(TERRAIN, 0);
        assert_ne!(TERRAIN, GAS_GIANT);
        assert_ne!(TERRAIN, ASTEROID_BELT);
        assert_ne!(TERRAIN, EMPTY_PLANETOID);
    }

    #[test]
    fn builtin_selector_ignores_registration_order() {
        let a = BuiltinSelector::new(vec![(TERRAIN, 1), (GAS_GIANT, 1)]);
        let b = BuiltinSelector::new(vec![(GAS_GIANT, 1), (TERRAIN, 1)]);
        let addr = Address::World(WorldAddr::ROOT);
        for seed in [0, 1, 2, 3, 4, 0xCAFE_BABE, 0xDEAD_BEEF] {
            assert_eq!(a.pick(&addr, seed), b.pick(&addr, seed), "seed={seed}");
        }
    }

    #[test]
    fn default_registry_picks_terrain() {
        let r = default_registry();
        let addr = Address::World(WorldAddr::ROOT);
        match r.resolve(&addr, 0xCAFE, GenerationPolicy::Seeded).unwrap() {
            Resolved::Generate { strategy, .. } => assert_eq!(strategy, TERRAIN),
            Resolved::Empty => panic!("expected Generate"),
        }
    }

    #[test]
    fn custom_override_beats_selector() {
        let r = default_registry();
        let addr = Address::World(WorldAddr::ROOT);
        match r.resolve(&addr, 0xCAFE, GenerationPolicy::Custom(GAS_GIANT)).unwrap() {
            Resolved::Generate { strategy, .. } => assert_eq!(strategy, GAS_GIANT),
            Resolved::Empty => panic!("expected Generate"),
        }
    }

    #[test]
    fn empty_policy_short_circuits() {
        let r = default_registry();
        let addr = Address::World(WorldAddr::ROOT);
        assert!(matches!(
            r.resolve(&addr, 0xCAFE, GenerationPolicy::Empty).unwrap(),
            Resolved::Empty
        ));
    }
}
