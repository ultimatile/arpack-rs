use std::fmt;

/// Errors raised by the safe ARPACK wrappers.
///
/// `info` codes are passed through verbatim from the underlying `*aupd_c`
/// and `*eupd_c` routines so the caller can interpret them against the
/// ARPACK Users' Guide. Negative values indicate misuse or numerical
/// failure; positive values other than `1` indicate non-recoverable
/// convergence conditions (e.g. `info = 3` from `aupd` means no shifts
/// could be applied — try increasing `ncv`).
///
/// `info = 1` from `*aupd_c` (max_iter reached before all `nev` Ritz
/// pairs converged) is mapped to [`Error::MaxIterReached`], which
/// preserves the iparam diagnostic counters so the caller can decide
/// whether to retry with a larger budget. For the single-eigenpair
/// drivers (`nev = 1`) currently exposed, `info = 1` always means
/// `nconv = 0` and no usable Ritz pair was extracted.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A wrapper-side parameter check failed before any FFI call.
    InvalidParam(&'static str),
    /// The reverse-communication driver (`*aupd_c`) returned a non-zero
    /// `info` code that is not separately modelled.
    AupdFailed(i32),
    /// The eigenvector-extraction routine (`*eupd_c`) returned a
    /// non-zero `info` code.
    EupdFailed(i32),
    /// `*aupd_c` requested an `ido` value the wrapper does not support
    /// (currently `ido = 2`, which only occurs for generalized
    /// eigenproblems with `bmat = 'G'`).
    UnexpectedIdo(i32),
    /// `*aupd_c` returned `info = 1`: the iteration hit `max_iter`
    /// before the requested `nev` Ritz pairs converged. The iparam
    /// writeback counters are preserved so the caller can retry with
    /// a larger `max_iter` or report partial progress.
    MaxIterReached {
        /// `iparam[2]` writeback — restart iterations performed
        /// (equals `Options::max_iter` when the cap was hit).
        iters: usize,
        /// `iparam[4]` writeback — converged Ritz value count.
        /// For the single-eigenpair drivers this is always `0`
        /// when this variant is returned.
        nconv: usize,
        /// `iparam[8]` writeback — operator applications performed.
        n_matvec: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            Error::AupdFailed(info) => write!(f, "ARPACK *aupd returned info = {info}"),
            Error::EupdFailed(info) => write!(f, "ARPACK *eupd returned info = {info}"),
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
        }
    }
}

impl std::error::Error for Error {}
