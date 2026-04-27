//! Real-symmetric eigenvalue drivers backed by ARPACK-NG `dsaupd_c` /
//! `dseupd_c` (Implicitly Restarted Lanczos).
//!
//! The current surface is intentionally narrow: a single function that
//! returns the smallest algebraic eigenpair of a symmetric operator
//! supplied through a matrix-vector closure. Additional `Which` modes
//! and multi-eigenvalue extraction can be layered on later.
//!
//! Thread-safety: every entry point acquires the crate-wide
//! [`crate::lock`] guard so the entire `*aupd_c` + `*eupd_c`
//! sequence runs atomically against ARPACK's Fortran SAVE state.

use std::os::raw::c_int;

use arpack_sys::{dsaupd_c, dseupd_c, ssaupd_c, sseupd_c};

use crate::error::Error;
use crate::lock::lock;
use crate::solution::{usize_from_iparam, EigSolution};

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

/// Smallest algebraic eigenpair of a real symmetric operator.
///
/// The operator is provided as a matrix-vector closure: `matvec(x, y)`
/// must compute `y <- A x` where both slices have length `n`.
///
/// Returns an [`EigSolution`] with the eigenpair and the diagnostic
/// counters ARPACK wrote back into `iparam` (actual iterations, number
/// converged, matvec applications). The eigenvector is normalized per
/// ARPACK's convention (unit 2-norm). Partial convergence (max_iter
/// reached) is returned as `Ok` with `nconv < nev`; see [`Error`] for
/// the cases that surface as `Err`.
///
/// # Allocation
///
/// Workspace sizes scale as `O(n * ncv)`. Inputs whose byte size exceeds
/// `isize::MAX` (relevant in practice only on 32-bit targets) cause the
/// underlying `Vec` allocations to panic rather than return
/// [`Error::InvalidParam`] — this matches the standard library's own
/// allocation-failure convention.
pub fn smallest_eigenpair_f64<F>(
    n: usize,
    matvec: F,
    options: &Options,
) -> Result<EigSolution<f64>, Error>
where
    F: FnMut(&[f64], &mut [f64]),
{
    let nev: c_int = 1;
    let nev_usize = nev as usize;
    // The wrapper enforces a strict `ncv < n` ceiling so IRLM always
    // has at least one free Krylov dimension to restart against; this
    // requires `n >= nev + 2` so the smallest legal `ncv` (`nev + 1`)
    // still fits below `n`.
    if n < nev_usize + 2 {
        return Err(Error::InvalidParam(
            "n too small for ARPACK (require n >= nev + 2)",
        ));
    }
    // Default: target `2*nev + 4` Krylov vectors, capped strictly
    // below `n` and floored at `nev + 1`. The previous floor of
    // `nev + 2` was too conservative — for `n = nev + 2` it lifted
    // `ncv` back up to `n`, defeating the ceiling.
    let ncv = options.ncv.unwrap_or_else(|| {
        (2 * nev_usize + 4).min(n - 1).max(nev_usize + 1)
    });

    let n_i32 = c_int_from_usize(n, "n")?;
    let ncv_i32 = c_int_from_usize(ncv, "ncv")?;
    let max_iter_i32 = c_int_from_usize(options.max_iter, "max_iter")?;

    if !(nev > 0 && nev < ncv_i32 && ncv_i32 < n_i32) {
        return Err(Error::InvalidParam("require 0 < nev < ncv < n"));
    }
    if max_iter_i32 <= 0 {
        return Err(Error::InvalidParam("max_iter must be positive"));
    }

    // All buffer allocations multiply user-controlled `n` and `ncv`.
    // On 32-bit targets these products can overflow `usize` even
    // though the individual values pass the `c_int` range check, so
    // verify each one explicitly before requesting allocations that
    // ARPACK will then index using the un-overflowed `n` / `ncv`.
    let v_len = n.checked_mul(ncv).ok_or(Error::InvalidParam(
        "n * ncv overflows usize",
    ))?;
    let workd_len = n.checked_mul(3).ok_or(Error::InvalidParam(
        "3 * n overflows usize",
    ))?;
    let lworkl = ncv
        .checked_add(8)
        .and_then(|s| ncv.checked_mul(s))
        .ok_or(Error::InvalidParam(
            "ncv * (ncv + 8) overflows usize",
        ))?;

    // Convert every length we need to pass into ARPACK to `c_int`
    // *before* requesting allocations. This keeps the failure mode
    // for absurdly-large inputs as a fast `InvalidParam` rather than
    // an OOM after committing to multi-gigabyte vectors.
    let lworkl_i32 = c_int_from_usize(lworkl, "lworkl")?;

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
    let which = c"SA".as_ptr();

    let _guard = lock();

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut matvec = matvec;
    // Reusable input buffer so the matvec closure always sees a stable
    // read-only view, regardless of whether ARPACK's `ipntr` happens
    // to point the X and Y windows to overlapping (or identical)
    // sub-ranges of `workd`. This costs one copy per ido callback but
    // avoids a tricky borrowing case for in-place operator modes.
    let mut x_buf = vec![0.0f64; n];

    loop {
        // SAFETY: All pointer arguments alias `Vec`-owned buffers that
        // outlive this call; bound checks above guarantee the lengths
        // match what ARPACK reads/writes. ARPACK is single-threaded
        // here (no concurrent calls).
        unsafe {
            dsaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which,
                nev,
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

    // ARPACK convention: info < 0 is misuse / numerical failure;
    // info = 1 means the iteration hit `max_iter` before nev Ritz
    // pairs converged. Because `nev = 1` is hardcoded here, info = 1
    // always implies `nconv = 0`, which means `*seupd` quick-returns
    // without writing `d` / `z` — calling it would silently produce
    // a zeroed eigenvalue and the first Lanczos basis vector. Surface
    // this as `Error::MaxIterReached` (with the iparam diagnostics
    // preserved) instead of a bogus `Ok`. Other non-zero codes are
    // surfaced as `AupdFailed` (e.g. info = 3 means try a larger ncv).
    if info == 1 {
        return Err(Error::MaxIterReached {
            iters: usize_from_iparam(iparam[2]),
            nconv: usize_from_iparam(iparam[4]),
            n_matvec: usize_from_iparam(iparam[8]),
        });
    }
    if info != 0 {
        return Err(Error::AupdFailed(info));
    }

    // Extract eigenvalue and eigenvector. `z` aliases `v` in-place,
    // which is the standard ARPACK pattern and avoids an extra n*nev
    // allocation.
    let rvec: c_int = 1;
    let howmny = c"A".as_ptr();
    let mut select = vec![0i32; ncv];
    let mut d = vec![0.0f64; nev as usize];
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
            which,
            nev,
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

    let value = d[0];
    let mut vector = vec![0.0f64; n];
    vector.copy_from_slice(&v[..n]);
    Ok(EigSolution {
        eigenvalue: value,
        eigenvector: vector,
        iters: usize_from_iparam(iparam[2]),
        nconv: usize_from_iparam(iparam[4]),
        n_matvec: usize_from_iparam(iparam[8]),
    })
}

/// Smallest algebraic eigenpair of a real symmetric operator, f32
/// precision. See [`smallest_eigenpair_f64`] for the long-form
/// contract; this entry point is identical except for the working
/// precision, and accepts the tolerance as `f64` to keep
/// [`Options`] uniform across precisions (the value is cast to
/// `f32` at the FFI boundary).
///
/// f32 precision in eigenvalue computation is rarely useful for
/// production work — the achievable convergence is bounded by the
/// scalar's relative epsilon (~`1.2e-7`). This entry point exists
/// for completeness and for callers (e.g. mixed-precision
/// pipelines, GPU staging) that produce f32 operators.
pub fn smallest_eigenpair_f32<F>(
    n: usize,
    matvec: F,
    options: &Options,
) -> Result<EigSolution<f32>, Error>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let nev: c_int = 1;
    let nev_usize = nev as usize;
    if n < nev_usize + 2 {
        return Err(Error::InvalidParam(
            "n too small for ARPACK (require n >= nev + 2)",
        ));
    }
    let ncv = options
        .ncv
        .unwrap_or_else(|| (2 * nev_usize + 4).min(n - 1).max(nev_usize + 1));

    let n_i32 = c_int_from_usize(n, "n")?;
    let ncv_i32 = c_int_from_usize(ncv, "ncv")?;
    let max_iter_i32 = c_int_from_usize(options.max_iter, "max_iter")?;

    if !(nev > 0 && nev < ncv_i32 && ncv_i32 < n_i32) {
        return Err(Error::InvalidParam("require 0 < nev < ncv < n"));
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

    let lworkl_i32 = c_int_from_usize(lworkl, "lworkl")?;

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
    let which = c"SA".as_ptr();

    let _guard = lock();

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut matvec = matvec;
    let mut x_buf = vec![0.0f32; n];

    loop {
        // SAFETY: identical reasoning to `smallest_eigenpair_f64`,
        // with `f32` storage instead of `f64`.
        unsafe {
            ssaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which,
                nev,
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

    // See `smallest_eigenpair_f64` for the rationale on splitting
    // `info == 1` (max_iter, no usable pair) from generic `AupdFailed`.
    if info == 1 {
        return Err(Error::MaxIterReached {
            iters: usize_from_iparam(iparam[2]),
            nconv: usize_from_iparam(iparam[4]),
            n_matvec: usize_from_iparam(iparam[8]),
        });
    }
    if info != 0 {
        return Err(Error::AupdFailed(info));
    }

    let rvec: c_int = 1;
    let howmny = c"A".as_ptr();
    let mut select = vec![0i32; ncv];
    let mut d = vec![0.0f32; nev as usize];
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
            which,
            nev,
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

    let value = d[0];
    let mut vector = vec![0.0f32; n];
    vector.copy_from_slice(&v[..n]);
    Ok(EigSolution {
        eigenvalue: value,
        eigenvector: vector,
        iters: usize_from_iparam(iparam[2]),
        nconv: usize_from_iparam(iparam[4]),
        n_matvec: usize_from_iparam(iparam[8]),
    })
}

fn c_int_from_usize(value: usize, name: &'static str) -> Result<c_int, Error> {
    c_int::try_from(value).map_err(|_| {
        let _ = name; // kept for future error-context expansion
        Error::InvalidParam("value does not fit in c_int")
    })
}

