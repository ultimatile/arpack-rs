use std::fmt;

/// Errors raised by the safe ARPACK wrappers.
///
/// Documented ARPACK `info` codes are interpreted into named variants
/// so a non-expert consumer sees the meaning rather than a bare number,
/// while the raw `info` is retained as a field on the catch-all
/// variants so an ARPACK expert can still match it against the Users'
/// Guide. Each variant also carries the `iparam` writeback counters
/// (`iters`, `nconv`, `n_matvec`) describing the run that produced it.
///
/// Most documented codes are never observed here: the wrapper validates
/// `n` / `nev` / `ncv` / `max_iter` / `which` and hardcodes `bmat`,
/// `mode`, and `ishift` before the FFI call, so the parameter-misuse
/// band (`info = -1..-7`, `-10..-13`, the `howmny` codes) cannot be
/// returned by ARPACK. The reachable non-success `*aupd_c` codes are
/// `info = 1` ([`Error::MaxIterReached`] / partial `Ok`), `info = 3`
/// ([`Error::NoShiftsApplied`]), `info = -9999`
/// ([`Error::ArnoldiFactorizationFailed`]), and the rare internal
/// failures `-8` / `-9` (catch-all [`Error::AupdFailed`]).
///
/// `info = 1` from `*aupd_c` (max_iter reached before all `nev` Ritz
/// pairs converged) splits two ways depending on the converged count
/// reported in `iparam[4]`:
///
/// - `nconv == 0`: no usable Ritz pair. Mapped to
///   [`Error::MaxIterReached`] with the diagnostic counters
///   preserved so the caller can retry with a larger budget. The
///   wrapper does **not** call `*eupd_c` in this case (symmetric
///   `*seupd` would quick-return with d/z untouched; complex
///   `*neupd` would return `info = -14`).
/// - `0 < nconv < nev`: partial extraction. Mapped to
///   `Ok(MultiEigSolution { nconv, .. })` carrying the `nconv`
///   converged pairs ARPACK was able to produce. This branch is
///   only reachable from the multi-eigenpair drivers
///   (`eigenpairs_*`); the `nev = 1` wrappers cannot observe it
///   because there is no integer strictly between 0 and 1.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A wrapper-side parameter check failed before any FFI call.
    InvalidParam(&'static str),
    /// `*aupd_c` returned a non-zero `info` code that is not separately
    /// modelled — a rare internal failure such as `info = -8` (error
    /// from the LAPACK eigenvalue calculation) or `info = -9` (a
    /// degenerate starting vector). The raw `info` is retained for
    /// expert diagnosis; the iparam writeback counters describe the run.
    AupdFailed {
        /// The raw `*aupd_c` `info` code (always non-zero, and not one
        /// of the separately modelled values `1` / `3` / `-9999`).
        info: i32,
        /// `iparam[2]` writeback — restart iterations performed.
        iters: usize,
        /// `iparam[4]` writeback — converged Ritz value count.
        nconv: usize,
        /// `iparam[8]` writeback — operator applications performed.
        n_matvec: usize,
    },
    /// `*eupd_c` (eigenvector extraction) returned a non-zero `info`
    /// code. Extraction failures are rare internal LAPACK and
    /// converged-count-consistency errors whose numeric codes differ
    /// between the real-symmetric and complex families, so they are
    /// surfaced generically with the raw `info` retained rather than
    /// modelled per code.
    EupdFailed {
        /// The raw `*eupd_c` `info` code (always non-zero).
        info: i32,
        /// `iparam[2]` writeback — restart iterations performed.
        iters: usize,
        /// `iparam[4]` writeback — converged Ritz value count.
        nconv: usize,
        /// `iparam[8]` writeback — operator applications performed.
        n_matvec: usize,
    },
    /// `*aupd_c` requested an `ido` value the wrapper does not support
    /// (currently `ido = 2`, which only occurs for generalized
    /// eigenproblems with `bmat = 'G'`).
    UnexpectedIdo(i32),
    /// `*aupd_c` returned `info = 1` AND `iparam[4] == 0`: the
    /// iteration hit `max_iter` with zero converged Ritz pairs, so
    /// no eigenpair could be extracted. The iparam writeback
    /// counters are preserved so the caller can retry with a larger
    /// `max_iter`. The partial-extraction case (`0 < nconv < nev`)
    /// is reported through `Ok(MultiEigSolution { nconv, .. })`
    /// instead and never via this variant.
    MaxIterReached {
        /// `iparam[2]` writeback — restart iterations performed
        /// (equals `Options::max_iter` when the cap was hit).
        iters: usize,
        /// `iparam[4]` writeback — converged Ritz value count.
        /// Always `0` when this variant is returned (positive
        /// nconv routes through `Ok` instead).
        nconv: usize,
        /// `iparam[8]` writeback — operator applications performed.
        n_matvec: usize,
    },
    /// `*aupd_c` returned `info = 3`: no shifts could be applied during
    /// a cycle of the Implicitly Restarted Arnoldi/Lanczos iteration,
    /// so the restart stalled before convergence. The usual remedy is
    /// to increase `ncv` relative to `nev`. The iparam writeback
    /// counters are preserved.
    NoShiftsApplied {
        /// `iparam[2]` writeback — restart iterations performed.
        iters: usize,
        /// `iparam[4]` writeback — converged Ritz value count.
        nconv: usize,
        /// `iparam[8]` writeback — operator applications performed.
        n_matvec: usize,
    },
    /// `*aupd_c` returned `info = -9999`: ARPACK could not build an
    /// Arnoldi factorization of the requested size within `max_iter`.
    /// Increasing `max_iter` or `ncv` may help. The iparam writeback
    /// counters are preserved.
    ArnoldiFactorizationFailed {
        /// `iparam[2]` writeback — restart iterations performed.
        iters: usize,
        /// `iparam[4]` writeback — for this code, IPARAM(5) returns the
        /// size of the Arnoldi factorization ARPACK did manage to build.
        /// Named distinctly from the `nconv` (converged Ritz count)
        /// field on the other variants because the same slot means a
        /// different thing here.
        factorization_size: usize,
        /// `iparam[8]` writeback — operator applications performed.
        n_matvec: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            Error::AupdFailed {
                info,
                iters,
                nconv,
                n_matvec,
            } => write!(
                f,
                "ARPACK *aupd failed: info = {info} \
                 (iters = {iters}, nconv = {nconv}, n_matvec = {n_matvec})"
            ),
            Error::EupdFailed {
                info,
                iters,
                nconv,
                n_matvec,
            } => write!(
                f,
                "ARPACK *eupd failed: info = {info} \
                 (iters = {iters}, nconv = {nconv}, n_matvec = {n_matvec})"
            ),
            Error::UnexpectedIdo(ido) => write!(f, "ARPACK requested unsupported ido = {ido}"),
            Error::MaxIterReached {
                iters,
                nconv,
                n_matvec,
            } => write!(
                f,
                "ARPACK hit max_iter without convergence: iters = {iters}, \
                 nconv = {nconv}, n_matvec = {n_matvec}"
            ),
            Error::NoShiftsApplied {
                iters,
                nconv,
                n_matvec,
            } => write!(
                f,
                "ARPACK could not apply any shifts during a restart cycle \
                 (info = 3); increase ncv relative to nev. iters = {iters}, \
                 nconv = {nconv}, n_matvec = {n_matvec}"
            ),
            Error::ArnoldiFactorizationFailed {
                iters,
                factorization_size,
                n_matvec,
            } => write!(
                f,
                "ARPACK could not build an Arnoldi factorization (info = -9999); \
                 built a factorization of size {factorization_size}. Try increasing \
                 max_iter or ncv. iters = {iters}, n_matvec = {n_matvec}"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Map a `*aupd_c` exit `info` (with the `iparam` diagnostic counters
/// already read back) into the typed error it represents, or `None`
/// when the caller should proceed to the `*eupd_c` extraction step.
///
/// `None` covers the two "keep going" outcomes: `info == 0` (full
/// convergence) and `info == 1 && nconv >= 1` (max_iter hit but a
/// usable partial set converged — `*eupd_c` extracts the `nconv`
/// pairs). Every other `info` is terminal.
///
/// The match order is load-bearing: the named convergence codes (`3`,
/// `-9999`) must be caught before the generic catch-all, otherwise they
/// would collapse into [`Error::AupdFailed`]. Both driver families
/// share this mapping — the `*aupd` codes it interprets carry the same
/// meaning across the real-symmetric and complex routines.
pub(crate) fn aupd_error(info: i32, iters: usize, nconv: usize, n_matvec: usize) -> Option<Error> {
    match info {
        0 => None,
        1 if nconv == 0 => Some(Error::MaxIterReached {
            iters,
            nconv,
            n_matvec,
        }),
        1 => None,
        3 => Some(Error::NoShiftsApplied {
            iters,
            nconv,
            n_matvec,
        }),
        -9999 => Some(Error::ArnoldiFactorizationFailed {
            iters,
            factorization_size: nconv,
            n_matvec,
        }),
        info => Some(Error::AupdFailed {
            info,
            iters,
            nconv,
            n_matvec,
        }),
    }
}

/// Map a `*eupd_c` exit `info` into its typed error, or `None` on
/// success (`info == 0`). The extraction step has no separately
/// modelled codes — its reachable failures are rare internal LAPACK and
/// consistency errors whose numeric codes differ across the real and
/// complex families — so every non-zero value is surfaced through the
/// raw-code-carrying [`Error::EupdFailed`].
pub(crate) fn eupd_error(info: i32, iters: usize, nconv: usize, n_matvec: usize) -> Option<Error> {
    match info {
        0 => None,
        info => Some(Error::EupdFailed {
            info,
            iters,
            nconv,
            n_matvec,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The mapping is pure and family-invariant, so it is verified here
    // directly rather than by trying to coax ARPACK into emitting each
    // code (info = 3 / -9999 have no deterministic small-matrix trigger).

    #[test]
    fn aupd_success_and_partial_proceed_to_eupd() {
        assert!(aupd_error(0, 5, 2, 7).is_none());
        // info = 1 with at least one converged pair is the partial-Ok
        // path: not an error, extraction continues.
        assert!(aupd_error(1, 5, 1, 7).is_none());
    }

    #[test]
    fn aupd_maxiter_with_zero_nconv() {
        assert!(matches!(
            aupd_error(1, 5, 0, 7),
            Some(Error::MaxIterReached {
                iters: 5,
                nconv: 0,
                n_matvec: 7
            })
        ));
    }

    #[test]
    fn aupd_named_convergence_codes() {
        assert!(matches!(
            aupd_error(3, 5, 2, 7),
            Some(Error::NoShiftsApplied {
                iters: 5,
                nconv: 2,
                n_matvec: 7
            })
        ));
        assert!(matches!(
            aupd_error(-9999, 5, 2, 7),
            Some(Error::ArnoldiFactorizationFailed {
                iters: 5,
                factorization_size: 2,
                n_matvec: 7
            })
        ));
    }

    #[test]
    fn aupd_catch_all_retains_raw_code() {
        // -8 (LAPACK failure) and -9 (degenerate start) are not
        // separately modelled; they keep the raw info.
        assert!(matches!(
            aupd_error(-8, 5, 0, 7),
            Some(Error::AupdFailed { info: -8, .. })
        ));
        assert!(matches!(
            aupd_error(-9, 5, 0, 7),
            Some(Error::AupdFailed { info: -9, .. })
        ));
    }

    #[test]
    fn eupd_success_and_failure() {
        assert!(eupd_error(0, 5, 2, 7).is_none());
        // The count-mismatch code differs across families (-17 symmetric,
        // -15 complex); both surface generically with the raw code.
        assert!(matches!(
            eupd_error(-17, 5, 2, 7),
            Some(Error::EupdFailed { info: -17, .. })
        ));
        assert!(matches!(
            eupd_error(-15, 5, 2, 7),
            Some(Error::EupdFailed { info: -15, .. })
        ));
    }

    #[test]
    fn display_mentions_raw_code_and_remedy() {
        let no_shifts = Error::NoShiftsApplied {
            iters: 3,
            nconv: 1,
            n_matvec: 4,
        };
        let msg = no_shifts.to_string();
        assert!(msg.contains("info = 3"));
        assert!(msg.contains("ncv"));

        let build = Error::ArnoldiFactorizationFailed {
            iters: 3,
            factorization_size: 1,
            n_matvec: 4,
        };
        assert!(build.to_string().contains("-9999"));

        let aupd = Error::AupdFailed {
            info: -8,
            iters: 3,
            nconv: 0,
            n_matvec: 4,
        };
        assert!(aupd.to_string().contains("info = -8"));
    }
}
