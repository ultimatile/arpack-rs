//! Safe Rust wrapper around ARPACK-NG.
//!
//! Re-exports the raw FFI surface from `arpack-sys` for callers that
//! need to drive ARPACK manually, and provides safe drivers for the
//! supported eigenproblem variants.

pub use arpack_sys as sys;

pub mod arnoldi;
pub mod error;
mod lock;
pub mod symmetric;

pub use error::Error;
// Crate-root re-exports for the symmetric driver were the public API
// before the `arnoldi` module landed; preserve them so existing
// callers do not need to update their imports. The Arnoldi module's
// own `Options` lives at `arpack::arnoldi::Options` to avoid
// colliding with the symmetric one re-exported here.
pub use symmetric::{Options, smallest_eigenpair_f32, smallest_eigenpair_f64};
