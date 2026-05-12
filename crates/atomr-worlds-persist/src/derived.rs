//! Persistence-backed store for derived byte blobs (feature-gated).
//!
//! The [`DerivedStore`] trait is a deliberately small key→bytes interface
//! used by Phase 14c/d/e view-mode caches that want to survive process
//! restarts (slice columns, surface rasters, overview pyramid tiles).
//! Phase 14 ships only the in-memory implementation here ([`InMemoryDerivedStore`]);
//! a SQL-backed implementation will live alongside `atomr-persistence-sql`
//! when the SQL feature graduates to needing it.
//!
//! Keys are arbitrary UTF-8 strings; callers are expected to pick a stable
//! prefix scheme (e.g., `"slice/<world>/<region>"`) so that
//! [`DerivedStore::delete_prefix`] can mass-evict on world unloads.

use std::collections::HashMap;
use std::sync::RwLock;

/// Errors returned by [`DerivedStore`] backends.
#[derive(Debug, thiserror::Error)]
pub enum DerivedStoreError {
    #[error("io error: {0}")]
    Io(String),
}

/// Storage backend for derived byte blobs keyed by UTF-8 strings.
pub trait DerivedStore: Send + Sync + std::fmt::Debug {
    /// Insert or overwrite the value at `key`.
    fn put(&self, key: &str, bytes: &[u8]) -> Result<(), DerivedStoreError>;
    /// Fetch the value at `key`, returning `None` if absent.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, DerivedStoreError>;
    /// Remove the value at `key`. Missing keys are not an error.
    fn delete(&self, key: &str) -> Result<(), DerivedStoreError>;
    /// Remove every entry whose key starts with `prefix`. Returns the count.
    fn delete_prefix(&self, prefix: &str) -> Result<usize, DerivedStoreError>;
}

/// In-memory [`DerivedStore`]; useful for tests and as the default when a
/// process has no persistent backend wired up.
#[derive(Debug, Default)]
pub struct InMemoryDerivedStore {
    entries: RwLock<HashMap<String, Vec<u8>>>,
}

impl InMemoryDerivedStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.read().expect("InMemoryDerivedStore lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().expect("InMemoryDerivedStore lock poisoned").is_empty()
    }
}

impl DerivedStore for InMemoryDerivedStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<(), DerivedStoreError> {
        let mut guard = self.entries.write().expect("InMemoryDerivedStore lock poisoned");
        guard.insert(key.to_owned(), bytes.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, DerivedStoreError> {
        let guard = self.entries.read().expect("InMemoryDerivedStore lock poisoned");
        Ok(guard.get(key).cloned())
    }

    fn delete(&self, key: &str) -> Result<(), DerivedStoreError> {
        let mut guard = self.entries.write().expect("InMemoryDerivedStore lock poisoned");
        guard.remove(key);
        Ok(())
    }

    fn delete_prefix(&self, prefix: &str) -> Result<usize, DerivedStoreError> {
        let mut guard = self.entries.write().expect("InMemoryDerivedStore lock poisoned");
        let before = guard.len();
        guard.retain(|k, _| !k.starts_with(prefix));
        Ok(before - guard.len())
    }
}

#[cfg(all(test, feature = "derived"))]
mod tests {
    use super::*;

    #[test]
    fn inmem_put_get_roundtrip() {
        let store = InMemoryDerivedStore::new();
        assert!(store.is_empty());
        store.put("k1", b"hello").unwrap();
        store.put("k2", b"world").unwrap();
        assert_eq!(store.len(), 2);

        assert_eq!(store.get("k1").unwrap().as_deref(), Some(&b"hello"[..]));
        assert_eq!(store.get("k2").unwrap().as_deref(), Some(&b"world"[..]));
        assert!(store.get("missing").unwrap().is_none());

        // Overwrite.
        store.put("k1", b"updated").unwrap();
        assert_eq!(store.get("k1").unwrap().as_deref(), Some(&b"updated"[..]));

        // Single delete.
        store.delete("k1").unwrap();
        assert!(store.get("k1").unwrap().is_none());
        // Missing delete is fine.
        store.delete("nonexistent").unwrap();
    }

    #[test]
    fn inmem_delete_prefix_counts() {
        let store = InMemoryDerivedStore::new();
        store.put("slice/world-a/0", b"a0").unwrap();
        store.put("slice/world-a/1", b"a1").unwrap();
        store.put("slice/world-b/0", b"b0").unwrap();
        store.put("overview/world-a/0", b"ov").unwrap();
        assert_eq!(store.len(), 4);

        let evicted = store.delete_prefix("slice/world-a/").unwrap();
        assert_eq!(evicted, 2);
        assert_eq!(store.len(), 2);
        assert!(store.get("slice/world-a/0").unwrap().is_none());
        assert!(store.get("slice/world-a/1").unwrap().is_none());
        assert!(store.get("slice/world-b/0").unwrap().is_some());
        assert!(store.get("overview/world-a/0").unwrap().is_some());

        // Prefix that matches nothing returns 0.
        let evicted = store.delete_prefix("does-not-exist/").unwrap();
        assert_eq!(evicted, 0);
        assert_eq!(store.len(), 2);
    }
}
