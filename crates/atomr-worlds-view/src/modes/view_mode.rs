//! `ViewMode` dispatch — convenience wrapper, not a forced abstraction.
//! Each mode's `render_*` function is independently callable.

/// Selects which Phase 14 display mode a caller wants.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ViewMode {
    FirstPerson,
    ThirdPerson,
    Slice,
    Rts,
    Overview,
}
