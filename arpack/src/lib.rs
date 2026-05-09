//! Safe Rust wrapper around ARPACK-NG.
//!
//! Provides safe drivers for the supported eigenproblem variants
//! (real-symmetric Lanczos and complex Arnoldi). Callers that need
//! to drive ARPACK manually should depend on `arpack-sys` directly.
//!
//! # Example
//!
//! Smallest eigenvalue of `diag(1, 2, 3)`:
//!
//! ```
//! use arpack::{Options, smallest_eigenpair_f64};
//!
//! let diag = [1.0_f64, 2.0, 3.0];
//! let n = diag.len();
//!
//! let solution = smallest_eigenpair_f64(
//!     n,
//!     |x, y| {
//!         for i in 0..n {
//!             y[i] = diag[i] * x[i];
//!         }
//!     },
//!     &Options::default(),
//! )
//! .expect("ARPACK converged");
//!
//! assert!((solution.eigenvalue - 1.0).abs() < 1e-9);
//! ```

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
