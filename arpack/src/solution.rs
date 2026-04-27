//! Return type shared across all driver entry points.

use std::os::raw::c_int;

/// The eigenpair returned by a driver, plus the diagnostic counters
/// ARPACK writes back into `iparam` during the iteration.
///
/// Returned only when ARPACK reaches full convergence (`info == 0`
/// from `*aupd_c`); the `max_iter`-reached case is reported through
/// [`crate::Error::MaxIterReached`] instead, which preserves the
/// same iparam counters but signals that no usable Ritz pair was
/// extracted. Callers thus see this struct only when the eigenpair
/// is meaningful.
///
/// The fields beyond `eigenvalue` / `eigenvector` let callers:
///
/// - tell fast convergence apart from a near-cap run (`iters`);
/// - confirm full convergence at a glance (`nconv >= nev`);
/// - account the cost of operator applications (`n_matvec`),
///   which is the dominant cost in DMRG-style workloads.
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
    /// single-eigenpair drivers exposed today this is always `1`
    /// when this struct is returned (`info == 0` from `*aupd_c`);
    /// the `max_iter`-reached case where `nconv` would be `0` is
    /// reported via [`crate::Error::MaxIterReached`] instead.
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
