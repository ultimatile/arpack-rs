//! Real-symmetric eigenvalue drivers backed by the ARPACK-NG
//! `{s,d}{sa,se}upd_c` family (Implicitly Restarted Lanczos).
//!
//! The crate exposes two layers:
//!
//! - [`eigenpairs_f64`] / [`eigenpairs_f32`] — general entry point
//!   accepting `nev >= 1` and a [`Which`] selector. Returns a
//!   [`MultiEigSolution<T>`] carrying the converged eigenpairs.
//! - [`smallest_eigenpair_f64`] / [`smallest_eigenpair_f32`] —
//!   convenience wrappers fixed to `nev = 1` and
//!   [`Which::SmallestAlgebraic`]. Returns the original
//!   [`EigSolution<T>`].
//!
//! Thread-safety: every entry point acquires a process-wide mutex
//! so the entire `*aupd_c` + `*eupd_c` sequence runs atomically
//! against ARPACK's Fortran SAVE state.

use std::os::raw::c_int;

use arpack_sys::{dsaupd_c, dseupd_c, ssaupd_c, sseupd_c};

use crate::error::Error;
use crate::lock::lock;
use crate::solution::{EigSolution, MultiEigSolution, usize_from_iparam};
use crate::which::Which;

/// Tunable parameters for the Lanczos driver.
#[derive(Clone, Debug)]
pub struct Options {
    /// Convergence tolerance. Pass `0.0` to accept ARPACK's default
    /// (machine epsilon for the working precision).
    pub tol: f64,
    /// Maximum number of restart iterations.
    pub max_iter: usize,
    /// Krylov-subspace dimension `ncv`. Must satisfy
    /// `nev < ncv < n` — the strict upper bound (rather than the
    /// `<= n` permitted by the ARPACK manual) ensures IRLM has at
    /// least one free Krylov dimension to restart against; the
    /// upstream code returns `info = -9999` when `ncv == n`.
    /// `None` selects `min(2*nev + 4, n - 1)` floored at `nev + 1`.
    pub ncv: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tol: 0.0,
            max_iter: 300,
            ncv: None,
        }
    }
}

/// Compute up to `nev` eigenpairs of a real symmetric operator.
///
/// The operator is provided as a matrix-vector closure: `matvec(x, y)`
/// must compute `y <- A x` where both slices have length `n`. The
/// [`Which`] selector controls which Ritz values to retain — must be
/// one of [`Which::SmallestAlgebraic`], [`Which::LargestAlgebraic`],
/// [`Which::SmallestMagnitude`], [`Which::LargestMagnitude`]
/// (per-family restriction enforced by the wrapper).
///
/// Returns a [`MultiEigSolution<f64>`] holding `nconv` converged
/// eigenpairs (with `nconv <= nev`) plus iparam diagnostics. The
/// real-symmetric drivers always return eigenvalues in **ascending
/// algebraic order**, regardless of `Which` — `LargestAlgebraic`
/// with `nev = 3` returns the three largest eigenvalues sorted
/// smallest-of-the-three first.
///
/// On `Options::max_iter` exhaustion with `nconv == 0`, returns
/// [`Error::MaxIterReached`]; on exhaustion with `0 < nconv < nev`,
/// returns `Ok(MultiEigSolution { nconv, .. })` carrying the
/// partial set (ARPACK's `*seupd` extracts the converged pairs
/// cleanly when `nconv >= 1`).
///
/// # Allocation
///
/// Workspace sizes scale as `O(n * ncv)`. Inputs whose byte size
/// exceeds `isize::MAX` cause the underlying `Vec` allocations to
/// panic rather than return [`Error::InvalidParam`] — same
/// convention as the standard library.
pub fn eigenpairs_f64<F>(
    n: usize,
    nev: usize,
    which: Which,
    matvec: F,
    options: &Options,
) -> Result<MultiEigSolution<f64>, Error>
where
    F: FnMut(&[f64], &mut [f64]),
{
    eigenpairs_f64_impl(n, nev, which, matvec, options)
}

/// Compute up to `nev` eigenpairs of a real symmetric operator,
/// f32 precision.
///
/// See [`eigenpairs_f64`] for the long-form contract. f32 is
/// rarely useful for production eigenvalue work — the achievable
/// convergence is bounded by the scalar's relative epsilon
/// (~`1.2e-7`). The tolerance is accepted as `f64` to keep
/// [`Options`] uniform and is cast to `f32` at the FFI boundary.
pub fn eigenpairs_f32<F>(
    n: usize,
    nev: usize,
    which: Which,
    matvec: F,
    options: &Options,
) -> Result<MultiEigSolution<f32>, Error>
where
    F: FnMut(&[f32], &mut [f32]),
{
    eigenpairs_f32_impl(n, nev, which, matvec, options)
}

/// Smallest algebraic eigenpair of a real symmetric operator.
///
/// Thin wrapper around [`eigenpairs_f64`] with `nev = 1` and
/// [`Which::SmallestAlgebraic`]. Returns a singular
/// [`EigSolution<f64>`].
///
/// On `Options::max_iter` exhaustion, returns
/// [`Error::MaxIterReached`] — at `nev = 1` the only legal
/// post-`info=1` state is `nconv = 0`, so the partial-Ok path
/// of [`eigenpairs_f64`] cannot fire here. See [`eigenpairs_f64`]
/// for the other failure modes.
pub fn smallest_eigenpair_f64<F>(
    n: usize,
    matvec: F,
    options: &Options,
) -> Result<EigSolution<f64>, Error>
where
    F: FnMut(&[f64], &mut [f64]),
{
    let multi = eigenpairs_f64(n, 1, Which::SmallestAlgebraic, matvec, options)?;
    Ok(singular_from_multi(multi))
}

/// Smallest algebraic eigenpair of a real symmetric operator,
/// f32 precision. See [`smallest_eigenpair_f64`] for the
/// long-form contract.
pub fn smallest_eigenpair_f32<F>(
    n: usize,
    matvec: F,
    options: &Options,
) -> Result<EigSolution<f32>, Error>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let multi = eigenpairs_f32(n, 1, Which::SmallestAlgebraic, matvec, options)?;
    Ok(singular_from_multi(multi))
}

fn singular_from_multi<T>(multi: MultiEigSolution<T>) -> EigSolution<T> {
    // info = 0 from *aupd guarantees nconv >= nev = 1, so the
    // vectors are non-empty. The partial-Ok branch (0 < nconv < nev)
    // is unreachable at nev = 1 — there is no integer strictly
    // between 0 and 1.
    let mut eigenvalues = multi.eigenvalues;
    let mut eigenvectors = multi.eigenvectors;
    let eigenvalue = eigenvalues
        .pop()
        .expect("eigenpairs_* with nev=1 returns at least one Ritz value on Ok");
    let eigenvector = eigenvectors
        .pop()
        .expect("eigenpairs_* with nev=1 returns at least one eigenvector on Ok");
    EigSolution {
        eigenvalue,
        eigenvector,
        iters: multi.iters,
        nconv: multi.nconv,
        n_matvec: multi.n_matvec,
    }
}

fn eigenpairs_f64_impl<F>(
    n: usize,
    nev: usize,
    which: Which,
    mut matvec: F,
    options: &Options,
) -> Result<MultiEigSolution<f64>, Error>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if nev == 0 {
        return Err(Error::InvalidParam("nev must be positive"));
    }
    if !which.accepted_by_symmetric() {
        return Err(Error::InvalidParam(
            "Which selector not accepted by the real-symmetric driver",
        ));
    }
    // Bound `nev` (caller-controlled) to the `c_int` range before
    // using it in `usize` arithmetic (`nev + 2`, `2 * nev + 4`,
    // `nev + 1`). On 64-bit targets — the only ones supported here
    // per the workspace's `compile_error!` — the bounded value
    // cannot overflow `usize` in those expressions; without this
    // upfront check, `nev = usize::MAX` panics in debug builds at
    // `nev + 2` before the existing `c_int_from_usize` calls fire.
    let nev_i32 = c_int_from_usize(nev)?;
    // IRLM enforces a strict `ncv < n` ceiling so it always has at
    // least one free Krylov dimension to restart against; the
    // smallest legal `ncv` is `nev + 1`, hence the precondition
    // `n >= nev + 2`.
    if n < nev + 2 {
        return Err(Error::InvalidParam(
            "n too small for ARPACK (require n >= nev + 2)",
        ));
    }
    let ncv = options
        .ncv
        .unwrap_or_else(|| (2 * nev + 4).min(n - 1).max(nev + 1));

    let n_i32 = c_int_from_usize(n)?;
    let ncv_i32 = c_int_from_usize(ncv)?;
    let max_iter_i32 = c_int_from_usize(options.max_iter)?;

    if !(nev_i32 < ncv_i32 && ncv_i32 < n_i32) {
        return Err(Error::InvalidParam("require nev < ncv < n"));
    }
    if max_iter_i32 <= 0 {
        return Err(Error::InvalidParam("max_iter must be positive"));
    }

    let v_len = n
        .checked_mul(ncv)
        .ok_or(Error::InvalidParam("n * ncv overflows usize"))?;
    let workd_len = n
        .checked_mul(3)
        .ok_or(Error::InvalidParam("3 * n overflows usize"))?;
    let lworkl = ncv
        .checked_add(8)
        .and_then(|s| ncv.checked_mul(s))
        .ok_or(Error::InvalidParam("ncv * (ncv + 8) overflows usize"))?;

    let lworkl_i32 = c_int_from_usize(lworkl)?;

    let mut resid = vec![0.0f64; n];
    let mut v = vec![0.0f64; v_len];
    let ldv = n_i32;
    let mut iparam = [0i32; 11];
    iparam[0] = 1; // exact shifts via ARPACK
    iparam[2] = max_iter_i32;
    iparam[3] = 1; // NB block size; ARPACK only supports NB = 1
    iparam[6] = 1; // mode 1: standard problem A x = lambda x
    let mut ipntr = [0i32; 11];
    let mut workd = vec![0.0f64; workd_len];
    let mut workl = vec![0.0f64; lworkl];

    let bmat = c"I".as_ptr();
    let which_ptr = which.as_c_str().as_ptr();

    let _guard = lock();

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut x_buf = vec![0.0f64; n];

    loop {
        // SAFETY: All pointer arguments alias `Vec`-owned buffers
        // that outlive this call; bound checks above guarantee the
        // lengths match what ARPACK reads/writes. ARPACK is
        // single-threaded here via the process-wide lock.
        unsafe {
            dsaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which_ptr,
                nev_i32,
                options.tol,
                resid.as_mut_ptr(),
                ncv_i32,
                v.as_mut_ptr(),
                ldv,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl_i32,
                &mut info,
            );
        }

        match ido {
            -1 | 1 => {
                let x_off = (ipntr[0] - 1) as usize;
                let y_off = (ipntr[1] - 1) as usize;
                debug_assert!(x_off + n <= workd.len() && y_off + n <= workd.len());
                x_buf.copy_from_slice(&workd[x_off..x_off + n]);
                matvec(&x_buf, &mut workd[y_off..y_off + n]);
            }
            99 => break,
            other => return Err(Error::UnexpectedIdo(other)),
        }
    }

    // info handling per the unified two-stage protocol:
    // - info = 0  : full convergence, extract nev Ritz pairs.
    // - info = 1  : max_iter hit. Read iparam[4] = nconv:
    //     * nconv == 0  → MaxIterReached (skip *eupd; dseupd would
    //                     quick-return via the `nconv .eq. 0`
    //                     guard in SRC/dseupd.f, leaving d/z
    //                     zeroed).
    //     * nconv >= 1  → call *eupd, extract `nconv` valid pairs
    //                     (dseupd accepts partial extraction).
    // - info < 0 / other info > 1 → AupdFailed.
    let nconv = usize_from_iparam(iparam[4]);
    let iters = usize_from_iparam(iparam[2]);
    let n_matvec = usize_from_iparam(iparam[8]);

    if info == 1 && nconv == 0 {
        return Err(Error::MaxIterReached {
            iters,
            nconv,
            n_matvec,
        });
    }
    if info != 0 && info != 1 {
        return Err(Error::AupdFailed(info));
    }
    // At this point: info == 0 (nconv typically == nev) or
    // info == 1 && nconv >= 1 (partial-Ok path). Both call *eupd.

    let rvec: c_int = 1;
    let howmny = c"A".as_ptr();
    let mut select = vec![0i32; ncv];
    // d is sized to nev (ARPACK's documented buffer size) but only
    // d[..nconv] is meaningful on return.
    let mut d = vec![0.0f64; nev];
    let sigma = 0.0f64;
    let mut info_eup: c_int = 0;

    // SAFETY: as above; v doubles as z (output eigenvector storage).
    unsafe {
        dseupd_c(
            rvec,
            howmny,
            select.as_mut_ptr(),
            d.as_mut_ptr(),
            v.as_mut_ptr(),
            ldv,
            sigma,
            bmat,
            n_i32,
            which_ptr,
            nev_i32,
            options.tol,
            resid.as_mut_ptr(),
            ncv_i32,
            v.as_mut_ptr(),
            ldv,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr(),
            workl.as_mut_ptr(),
            lworkl_i32,
            &mut info_eup,
        );
    }

    if info_eup != 0 {
        return Err(Error::EupdFailed(info_eup));
    }

    // ARPACK wrote d[..nconv] and v[.., 0..nconv] (column-major).
    // Slots beyond nconv are undefined and must not be exposed.
    let eigenvalues = d[..nconv].to_vec();
    let mut eigenvectors = Vec::with_capacity(nconv);
    for k in 0..nconv {
        eigenvectors.push(v[k * n..(k + 1) * n].to_vec());
    }

    Ok(MultiEigSolution {
        eigenvalues,
        eigenvectors,
        nev_requested: nev,
        nconv,
        iters,
        n_matvec,
    })
}

fn eigenpairs_f32_impl<F>(
    n: usize,
    nev: usize,
    which: Which,
    mut matvec: F,
    options: &Options,
) -> Result<MultiEigSolution<f32>, Error>
where
    F: FnMut(&[f32], &mut [f32]),
{
    if nev == 0 {
        return Err(Error::InvalidParam("nev must be positive"));
    }
    if !which.accepted_by_symmetric() {
        return Err(Error::InvalidParam(
            "Which selector not accepted by the real-symmetric driver",
        ));
    }
    // See `eigenpairs_f64_impl` for why `nev` is bounded here.
    let nev_i32 = c_int_from_usize(nev)?;
    if n < nev + 2 {
        return Err(Error::InvalidParam(
            "n too small for ARPACK (require n >= nev + 2)",
        ));
    }
    let ncv = options
        .ncv
        .unwrap_or_else(|| (2 * nev + 4).min(n - 1).max(nev + 1));

    let n_i32 = c_int_from_usize(n)?;
    let ncv_i32 = c_int_from_usize(ncv)?;
    let max_iter_i32 = c_int_from_usize(options.max_iter)?;

    if !(nev_i32 < ncv_i32 && ncv_i32 < n_i32) {
        return Err(Error::InvalidParam("require nev < ncv < n"));
    }
    if max_iter_i32 <= 0 {
        return Err(Error::InvalidParam("max_iter must be positive"));
    }

    let v_len = n
        .checked_mul(ncv)
        .ok_or(Error::InvalidParam("n * ncv overflows usize"))?;
    let workd_len = n
        .checked_mul(3)
        .ok_or(Error::InvalidParam("3 * n overflows usize"))?;
    let lworkl = ncv
        .checked_add(8)
        .and_then(|s| ncv.checked_mul(s))
        .ok_or(Error::InvalidParam("ncv * (ncv + 8) overflows usize"))?;

    let lworkl_i32 = c_int_from_usize(lworkl)?;

    let tol = options.tol as f32;
    let mut resid = vec![0.0f32; n];
    let mut v = vec![0.0f32; v_len];
    let ldv = n_i32;
    let mut iparam = [0i32; 11];
    iparam[0] = 1;
    iparam[2] = max_iter_i32;
    iparam[3] = 1;
    iparam[6] = 1;
    let mut ipntr = [0i32; 11];
    let mut workd = vec![0.0f32; workd_len];
    let mut workl = vec![0.0f32; lworkl];

    let bmat = c"I".as_ptr();
    let which_ptr = which.as_c_str().as_ptr();

    let _guard = lock();

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut x_buf = vec![0.0f32; n];

    loop {
        // SAFETY: identical reasoning to `eigenpairs_f64_impl`,
        // with `f32` storage instead of `f64`.
        unsafe {
            ssaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which_ptr,
                nev_i32,
                tol,
                resid.as_mut_ptr(),
                ncv_i32,
                v.as_mut_ptr(),
                ldv,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl_i32,
                &mut info,
            );
        }

        match ido {
            -1 | 1 => {
                let x_off = (ipntr[0] - 1) as usize;
                let y_off = (ipntr[1] - 1) as usize;
                debug_assert!(x_off + n <= workd.len() && y_off + n <= workd.len());
                x_buf.copy_from_slice(&workd[x_off..x_off + n]);
                matvec(&x_buf, &mut workd[y_off..y_off + n]);
            }
            99 => break,
            other => return Err(Error::UnexpectedIdo(other)),
        }
    }

    let nconv = usize_from_iparam(iparam[4]);
    let iters = usize_from_iparam(iparam[2]);
    let n_matvec = usize_from_iparam(iparam[8]);

    if info == 1 && nconv == 0 {
        return Err(Error::MaxIterReached {
            iters,
            nconv,
            n_matvec,
        });
    }
    if info != 0 && info != 1 {
        return Err(Error::AupdFailed(info));
    }

    let rvec: c_int = 1;
    let howmny = c"A".as_ptr();
    let mut select = vec![0i32; ncv];
    let mut d = vec![0.0f32; nev];
    let sigma = 0.0f32;
    let mut info_eup: c_int = 0;

    // SAFETY: as above; v doubles as z (output eigenvector storage).
    unsafe {
        sseupd_c(
            rvec,
            howmny,
            select.as_mut_ptr(),
            d.as_mut_ptr(),
            v.as_mut_ptr(),
            ldv,
            sigma,
            bmat,
            n_i32,
            which_ptr,
            nev_i32,
            tol,
            resid.as_mut_ptr(),
            ncv_i32,
            v.as_mut_ptr(),
            ldv,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr(),
            workl.as_mut_ptr(),
            lworkl_i32,
            &mut info_eup,
        );
    }

    if info_eup != 0 {
        return Err(Error::EupdFailed(info_eup));
    }

    let eigenvalues = d[..nconv].to_vec();
    let mut eigenvectors = Vec::with_capacity(nconv);
    for k in 0..nconv {
        eigenvectors.push(v[k * n..(k + 1) * n].to_vec());
    }

    Ok(MultiEigSolution {
        eigenvalues,
        eigenvectors,
        nev_requested: nev,
        nconv,
        iters,
        n_matvec,
    })
}

fn c_int_from_usize(value: usize) -> Result<c_int, Error> {
    c_int::try_from(value).map_err(|_| Error::InvalidParam("value does not fit in c_int"))
}
