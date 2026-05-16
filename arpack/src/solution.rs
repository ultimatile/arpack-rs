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
/// - account the cost of operator applications (`n_matvec`).
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

/// Multi-eigenpair result returned by the `eigenpairs_*` drivers.
///
/// Carries the converged eigenpairs plus the same diagnostic
/// counters as [`EigSolution`]. The crate distinguishes
/// `EigSolution` (singular, returned by `smallest_eigenpair_*`)
/// from `MultiEigSolution` (plural, returned by `eigenpairs_*`)
/// so the single-eigenpair API can stay ergonomic while the
/// multi-eigenpair API exposes per-vector storage.
///
/// `eigenvalues` and `eigenvectors` both have length
/// `min(nconv, nev_requested)`. ARPACK only guarantees the
/// first `nconv` slots of its output are converged; slots
/// beyond that count are undefined and never propagated. The
/// extra `min` against `nev_requested` accommodates the rare
/// case where ARPACK reports `nconv > nev_requested` (bonus
/// Ritz values that satisfied the convergence bound) — the
/// extraction buffer is only `nev_requested` long per the
/// documented `*eupd` interface, so any bonus values are
/// recorded in `nconv` as a diagnostic but not in the
/// eigenpair arrays. Callers verify `nconv >= nev_requested`
/// themselves if they need full convergence.
///
/// # Ordering
///
/// The real-symmetric Lanczos drivers (`{s,d}{sa,se}upd_c`)
/// return eigenvalues in ascending algebraic order regardless of
/// the `Which` selector — `Which::LargestAlgebraic` with `nev = 3`
/// on a Laplacian-style matrix yields the three largest
/// eigenvalues sorted ascending (smallest of the three first).
///
/// The complex Arnoldi drivers (`{c,z}{na,ne}upd_c`) do **not**
/// apply a final sort; the order depends on ARPACK's internal
/// selection state. Callers that need a stable order must sort
/// the returned vectors themselves.
#[derive(Debug, Clone)]
pub struct MultiEigSolution<T> {
    /// Converged Ritz values, length `min(nconv, nev_requested)`.
    /// See the type-level docstring for the per-family ordering
    /// convention and for why the count may be `< nconv` when ARPACK
    /// reports extra converged values.
    pub eigenvalues: Vec<T>,
    /// Converged eigenvectors, same length as `eigenvalues` (i.e.
    /// `min(nconv, nev_requested)`). Each inner `Vec` has length `n`
    /// and is unit-normalized per ARPACK's convention.
    /// `eigenvectors[k]` corresponds to `eigenvalues[k]`.
    pub eigenvectors: Vec<Vec<T>>,
    /// Number of eigenpairs the caller asked for. Carried for
    /// diagnostics — compare against `nconv` to detect partial
    /// convergence.
    pub nev_requested: usize,
    /// Number of converged Ritz values (`iparam[4]` writeback).
    /// Always at least `1` when this struct is returned (the
    /// `nconv == 0` case is reported as
    /// [`crate::Error::MaxIterReached`] instead). Usually
    /// `nconv <= nev_requested`, but ARPACK occasionally reports
    /// a slightly larger count when extra Ritz values converged
    /// to tolerance — only `min(nconv, nev_requested)` of those
    /// are surfaced in `eigenvalues` / `eigenvectors`.
    pub nconv: usize,
    /// Number of restart iterations actually performed
    /// (ARPACK's `iparam[2]` writeback).
    pub iters: usize,
    /// Total number of operator applications (`iparam[8]`).
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
