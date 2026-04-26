//! Safe Rust wrapper around ARPACK-NG.
//!
//! Re-exports the raw FFI surface from `arpack-sys` for callers that
//! need to drive ARPACK manually, and provides safe drivers for the
//! supported eigenproblem variants.

pub use arpack_sys as sys;

pub mod error;
pub mod symmetric;

pub use error::Error;
pub use symmetric::{Options, smallest_eigenpair_f64};
