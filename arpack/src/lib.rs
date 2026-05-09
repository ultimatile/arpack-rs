//! Safe Rust wrapper around ARPACK-NG.
//!
//! Provides safe drivers for the supported eigenproblem variants
//! (real-symmetric Lanczos and complex Arnoldi). Callers that need
//! to drive ARPACK manually should depend on `arpack-sys` directly.

pub mod arnoldi;
pub mod error;
mod lock;
mod solution;
pub mod symmetric;

pub use error::Error;
pub use solution::EigSolution;
// Crate-root re-exports for the symmetric driver were the public API
// before the `arnoldi` module landed; preserve them so existing
// callers do not need to update their imports. The Arnoldi module's
// own `Options` lives at `arpack::arnoldi::Options` to avoid
// colliding with the symmetric one re-exported here.
pub use symmetric::{Options, smallest_eigenpair_f32, smallest_eigenpair_f64};
