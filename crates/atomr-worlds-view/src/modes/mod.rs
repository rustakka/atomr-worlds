//! Phase 14 world display modes.
//!
//! Each submodule is a tailored rendering pipeline for one viewing
//! context — 1st-person walk, 3rd-person chase, Dwarf-Fortress-style
//! slice, RTS oblique strategy, or large-scale regional overview.
//! All are headless: `(camera, world_query, config) → Framebuffer`.

pub mod fp;
pub mod overview;
pub mod rts;
pub mod slice;
pub mod tp;
pub mod view_mode;
