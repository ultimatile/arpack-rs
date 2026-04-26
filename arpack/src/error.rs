use std::fmt;

/// Errors raised by the safe ARPACK wrappers.
///
/// `info` codes are passed through verbatim from the underlying `*aupd_c`
/// and `*eupd_c` routines so the caller can interpret them against the
/// ARPACK Users' Guide. Negative values indicate misuse or numerical
/// failure; positive values indicate convergence-related conditions
/// (e.g. `info = 1` from `aupd` means the maximum iteration count was
/// reached without convergence).
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A wrapper-side parameter check failed before any FFI call.
    InvalidParam(&'static str),
    /// The reverse-communication driver (`*aupd_c`) returned a non-zero
    /// `info` code.
    AupdFailed(i32),
    /// The eigenvector-extraction routine (`*eupd_c`) returned a
    /// non-zero `info` code.
    EupdFailed(i32),
    /// `*aupd_c` requested an `ido` value the wrapper does not support
    /// (currently `ido = 2`, which only occurs for generalized
    /// eigenproblems with `bmat = 'G'`).
    UnexpectedIdo(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            Error::AupdFailed(info) => write!(f, "ARPACK *aupd returned info = {info}"),
            Error::EupdFailed(info) => write!(f, "ARPACK *eupd returned info = {info}"),
            Error::UnexpectedIdo(ido) => write!(f, "ARPACK requested unsupported ido = {ido}"),
        }
    }
}

impl std::error::Error for Error {}
