//! Safe Rust wrapper around ARPACK-NG.
//!
//! Currently a thin re-export of `arpack-sys`; the safe API will land in
//! follow-up commits.

pub use arpack_sys as sys;
