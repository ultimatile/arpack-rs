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
