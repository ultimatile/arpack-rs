//! Return type shared across all driver entry points.

use std::os::raw::c_int;

/// The eigenpair returned by a driver, plus the diagnostic counters
/// ARPACK writes back into `iparam` during the iteration.
///
/// Even when ARPACK reports the iteration as successful (`info == 0`
/// or `info == 1` from `*aupd_c`), the caller may want to:
///
/// - tell fast convergence apart from a `max_iter` cap (`iters`);
/// - detect partial convergence (`nconv < nev` — for the
///   single-eigenpair drivers exposed today this means
///   `nconv == 0`, i.e. the returned pair is the best Ritz
///   approximation seen but did not satisfy `tol`);
/// - account the cost of operator applications (`n_matvec`),
///   which is the dominant cost in DMRG-style workloads.
///
/// Exposing these as fields means callers do not have to thread
/// custom counters through their matvec closures or guess at
/// "did it really converge or just hit max_iter."
#[derive(Debug, Clone)]
pub struct EigSolution<T> {
    /// Smallest (algebraic / real-part) eigenvalue.
    pub eigenvalue: T,
    /// Corresponding eigenvector of length `n`, unit-normalized.
    pub eigenvector: Vec<T>,
    /// Number of restart iterations actually performed
    /// (ARPACK's `iparam[2]` writeback). Strictly less than
    /// `Options::max_iter` for a normally-converging problem;
    /// equal to it when the iteration was capped.
    pub iters: usize,
    /// Number of converged Ritz values (`iparam[4]`). For the
    /// single-eigenpair drivers exposed today this is `0` or `1`:
    /// `0` means partial convergence (the eigenpair is the best
    /// Ritz approximation in the final Krylov subspace but did
    /// not meet `tol`), `1` means full convergence.
    pub nconv: usize,
    /// Total number of operator applications performed by ARPACK
    /// during the iteration (`iparam[8]`). This is the only cost
    /// term that scales with the actual matrix; everything else
    /// in the workspace is O(n * ncv + ncv^2).
    pub n_matvec: usize,
}

/// Convert a non-negative `iparam` writeback (the only kind ARPACK
/// produces for these slots) into `usize`. Values are inherently
/// non-negative — `iters`, `nconv`, and matvec counts cannot be
/// negative — so a negative reading would be a wrapper bug, not user
/// input; clamp at 0 and let downstream invariants catch the mismatch.
pub(crate) fn usize_from_iparam(value: c_int) -> usize {
    if value < 0 { 0 } else { value as usize }
}
