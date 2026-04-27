//! Process-wide lock guarding every ARPACK call.
//!
//! ARPACK is implemented in Fortran and keeps state in `SAVE`
//! variables (the random-number generator seed, restart bookkeeping
//! inside the Lanczos / Arnoldi drivers, etc.). Concurrent calls
//! corrupt that state, so every `*aupd_c` + `*eupd_c` sequence
//! across the entire crate must serialize through one mutex. Each
//! driver module acquires this same lock at the top of its public
//! entry point.

use std::sync::{Mutex, MutexGuard};

static ARPACK_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the process-wide ARPACK lock.
///
/// Recovers from poisoning silently: ARPACK has no caller-observable
/// state we can inspect after a panic mid-iteration, and the Fortran
/// drivers reset themselves whenever the next `*aupd_c` is called
/// with `info = 0`. So an earlier panic does not contaminate later
/// callers.
pub(crate) fn lock() -> MutexGuard<'static, ()> {
    ARPACK_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
