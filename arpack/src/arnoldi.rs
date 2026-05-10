//! Arnoldi-iteration eigenvalue drivers (`*naupd_c` / `*neupd_c`).
//!
//! ARPACK-NG ships two Arnoldi families: real-non-symmetric
//! (`{s,d}{na,ne}upd_c`) and complex (`{c,z}{na,ne}upd_c`). This
//! module wraps the complex family for both `Complex<f64>` and
//! `Complex<f32>` scalars; the real-non-symmetric family is not
//! wrapped yet.
//!
//! Hermitian operators have real eigenvalues but are still driven
//! through the complex Arnoldi routine; the returned eigenvalue
//! comes back complex and callers verify / discard the imaginary
//! part themselves.
//!
//! Thread-safety: every entry point acquires a process-wide mutex
//! so the entire `*aupd_c` + `*eupd_c` sequence runs atomically
//! against ARPACK's Fortran SAVE state.

use std::os::raw::c_int;

use arpack_sys::{__BindgenComplex, cnaupd_c, cneupd_c, znaupd_c, zneupd_c};
use num_complex::{Complex32, Complex64};

use crate::error::Error;
use crate::lock::lock;
use crate::solution::{EigSolution, usize_from_iparam};

/// Tunable parameters for the complex Arnoldi driver. The fields
/// have the same meaning as in [`crate::symmetric::Options`]; the
/// type is duplicated rather than shared so the two driver families
/// can evolve independently without breaking each other's API.
#[derive(Clone, Debug)]
pub struct Options {
    /// Convergence tolerance. `0.0` accepts ARPACK's default
    /// (machine epsilon for the working precision).
    pub tol: f64,
    /// Maximum number of restart iterations.
    pub max_iter: usize,
    /// Krylov-subspace dimension `ncv`. Must satisfy
    /// `nev + 2 <= ncv < n` — `zneupd` requires at least two extra
    /// Krylov vectors for restart-deflation (`ncv - nev >= 2`,
    /// stricter than the symmetric driver's `ncv > nev`), and
    /// `ncv == n` would leave IRLM no room to restart. `None`
    /// selects `min(2*nev + 4, n - 1)` floored at `nev + 2`.
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

/// Smallest-real-part eigenpair of a complex linear operator. For
/// Hermitian operators the eigenvalue's imaginary part is
/// numerically zero and the real part is the smallest eigenvalue.
///
/// `matvec(x, y)` must compute `y <- A x` where both slices have
/// length `n`.
///
/// # Allocation
///
/// Workspace size scales as `O(n * ncv + ncv^2)`. Inputs whose byte
/// size exceeds `isize::MAX` cause `Vec` allocations to panic
/// rather than return [`Error::InvalidParam`] — same convention as
/// the standard library and the `symmetric` module.
pub fn smallest_eigenpair_c64<F>(
    n: usize,
    matvec: F,
    options: &Options,
) -> Result<EigSolution<Complex64>, Error>
where
    F: FnMut(&[Complex64], &mut [Complex64]),
{
    let nev: c_int = 1;
    let nev_usize = nev as usize;
    // Complex Arnoldi has a tighter constraint than real symmetric:
    // `zneupd` rejects `ncv - nev < 2` with `info = -3`, so the
    // smallest legal `ncv` is `nev + 2` and the precondition on
    // `n` is `n >= nev + 3` (so `ncv = nev + 2 < n` still holds).
    if n < nev_usize + 3 {
        return Err(Error::InvalidParam(
            "n too small for complex Arnoldi (require n >= nev + 3)",
        ));
    }
    let ncv = options
        .ncv
        .unwrap_or_else(|| (2 * nev_usize + 4).min(n - 1).max(nev_usize + 2));

    let n_i32 = c_int_from_usize(n)?;
    let ncv_i32 = c_int_from_usize(ncv)?;
    let max_iter_i32 = c_int_from_usize(options.max_iter)?;

    // Strict `ncv >= nev + 2` and `ncv < n`.
    if !(nev > 0 && ncv_i32 >= nev + 2 && ncv_i32 < n_i32) {
        return Err(Error::InvalidParam(
            "require 0 < nev, nev + 2 <= ncv, and ncv < n",
        ));
    }
    if max_iter_i32 <= 0 {
        return Err(Error::InvalidParam("max_iter must be positive"));
    }

    // Workspace sizes. znaupd's `lworkl` requirement is
    // `3*ncv^2 + 5*ncv` (different from dsaupd's `ncv*(ncv+8)`),
    // and the eupd routine needs an extra `workev[2*ncv]`.
    let v_len = n
        .checked_mul(ncv)
        .ok_or(Error::InvalidParam("n * ncv overflows usize"))?;
    let workd_len = n
        .checked_mul(3)
        .ok_or(Error::InvalidParam("3 * n overflows usize"))?;
    let ncv_sq = ncv
        .checked_mul(ncv)
        .ok_or(Error::InvalidParam("ncv * ncv overflows usize"))?;
    let three_ncv_sq = ncv_sq
        .checked_mul(3)
        .ok_or(Error::InvalidParam("3 * ncv^2 overflows usize"))?;
    let five_ncv = ncv
        .checked_mul(5)
        .ok_or(Error::InvalidParam("5 * ncv overflows usize"))?;
    let lworkl = three_ncv_sq
        .checked_add(five_ncv)
        .ok_or(Error::InvalidParam("3*ncv^2 + 5*ncv overflows usize"))?;
    let workev_len = ncv
        .checked_mul(2)
        .ok_or(Error::InvalidParam("2 * ncv overflows usize"))?;

    let lworkl_i32 = c_int_from_usize(lworkl)?;

    // All buffers are owned in `Complex64` form for ergonomic
    // access; the FFI calls cast pointers to `__BindgenComplex<f64>`
    // since the two types are layout-compatible (`#[repr(C)]` with
    // identical fields).
    let zero = Complex64::new(0.0, 0.0);
    let mut resid = vec![zero; n];
    let mut v = vec![zero; v_len];
    let ldv = n_i32;
    let mut iparam = [0i32; 11];
    iparam[0] = 1; // exact shifts via ARPACK
    iparam[2] = max_iter_i32;
    iparam[3] = 1; // NB block size; ARPACK only supports NB = 1
    iparam[6] = 1; // mode 1: standard problem A x = lambda x
    // znaupd's ICB Fortran wrapper declares `ipntr(14)`, unlike
    // dsaupd's `ipntr(11)`. Allocating shorter is an OOB write.
    let mut ipntr = [0i32; 14];
    let mut workd = vec![zero; workd_len];
    let mut workl = vec![zero; lworkl];
    let mut rwork = vec![0.0f64; ncv];

    let bmat = c"I".as_ptr();
    let which = c"SR".as_ptr();

    let _guard = lock();

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut matvec = matvec;
    // Reusable input buffer; ARPACK can hand us X and Y windows
    // pointing into the same sub-range of `workd`.
    let mut x_buf = vec![zero; n];

    loop {
        // SAFETY: Every pointer aliases a Vec whose length matches
        // (or exceeds) what ARPACK reads/writes; the `Complex64` ↔
        // `__BindgenComplex<f64>` cast is sound because both are
        // `#[repr(C)] { re: f64, im: f64 }`. The process-wide lock
        // serializes ARPACK's Fortran SAVE state.
        unsafe {
            znaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which,
                nev,
                options.tol,
                resid.as_mut_ptr() as *mut __BindgenComplex<f64>,
                ncv_i32,
                v.as_mut_ptr() as *mut __BindgenComplex<f64>,
                ldv,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr() as *mut __BindgenComplex<f64>,
                workl.as_mut_ptr() as *mut __BindgenComplex<f64>,
                lworkl_i32,
                rwork.as_mut_ptr(),
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

    // info = 1: max_iter reached. With nev = 1 hardcoded, this always
    // means nconv = 0, and calling `*neupd` on that state returns
    // info = -14 ("did not find any eigenvalues to sufficient
    // accuracy") rather than a usable Ritz pair. Surface as
    // MaxIterReached with the iparam diagnostics intact.
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
    let mut d = vec![zero; nev_usize];
    let sigma = __BindgenComplex { re: 0.0, im: 0.0 };
    let mut workev = vec![zero; workev_len];
    let mut info_eup: c_int = 0;

    // SAFETY: as for znaupd_c above; v doubles as z (output
    // eigenvector storage), which is the standard ARPACK pattern.
    unsafe {
        zneupd_c(
            rvec,
            howmny,
            select.as_mut_ptr(),
            d.as_mut_ptr() as *mut __BindgenComplex<f64>,
            v.as_mut_ptr() as *mut __BindgenComplex<f64>,
            ldv,
            sigma,
            workev.as_mut_ptr() as *mut __BindgenComplex<f64>,
            bmat,
            n_i32,
            which,
            nev,
            options.tol,
            resid.as_mut_ptr() as *mut __BindgenComplex<f64>,
            ncv_i32,
            v.as_mut_ptr() as *mut __BindgenComplex<f64>,
            ldv,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr() as *mut __BindgenComplex<f64>,
            workl.as_mut_ptr() as *mut __BindgenComplex<f64>,
            lworkl_i32,
            rwork.as_mut_ptr(),
            &mut info_eup,
        );
    }

    if info_eup != 0 {
        return Err(Error::EupdFailed(info_eup));
    }

    let value = d[0];
    let mut vector = vec![zero; n];
    vector.copy_from_slice(&v[..n]);
    Ok(EigSolution {
        eigenvalue: value,
        eigenvector: vector,
        iters: usize_from_iparam(iparam[2]),
        nconv: usize_from_iparam(iparam[4]),
        n_matvec: usize_from_iparam(iparam[8]),
    })
}

/// Smallest-real-part eigenpair of a complex linear operator, f32
/// precision. See [`smallest_eigenpair_c64`] for the long-form
/// contract; this entry point is identical except for the working
/// precision, and accepts the tolerance as `f64` to keep
/// [`Options`] uniform across precisions (the value is cast to
/// `f32` at the FFI boundary).
pub fn smallest_eigenpair_c32<F>(
    n: usize,
    matvec: F,
    options: &Options,
) -> Result<EigSolution<Complex32>, Error>
where
    F: FnMut(&[Complex32], &mut [Complex32]),
{
    let nev: c_int = 1;
    let nev_usize = nev as usize;
    if n < nev_usize + 3 {
        return Err(Error::InvalidParam(
            "n too small for complex Arnoldi (require n >= nev + 3)",
        ));
    }
    let ncv = options
        .ncv
        .unwrap_or_else(|| (2 * nev_usize + 4).min(n - 1).max(nev_usize + 2));

    let n_i32 = c_int_from_usize(n)?;
    let ncv_i32 = c_int_from_usize(ncv)?;
    let max_iter_i32 = c_int_from_usize(options.max_iter)?;

    if !(nev > 0 && ncv_i32 >= nev + 2 && ncv_i32 < n_i32) {
        return Err(Error::InvalidParam(
            "require 0 < nev, nev + 2 <= ncv, and ncv < n",
        ));
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
    let ncv_sq = ncv
        .checked_mul(ncv)
        .ok_or(Error::InvalidParam("ncv * ncv overflows usize"))?;
    let three_ncv_sq = ncv_sq
        .checked_mul(3)
        .ok_or(Error::InvalidParam("3 * ncv^2 overflows usize"))?;
    let five_ncv = ncv
        .checked_mul(5)
        .ok_or(Error::InvalidParam("5 * ncv overflows usize"))?;
    let lworkl = three_ncv_sq
        .checked_add(five_ncv)
        .ok_or(Error::InvalidParam("3*ncv^2 + 5*ncv overflows usize"))?;
    let workev_len = ncv
        .checked_mul(2)
        .ok_or(Error::InvalidParam("2 * ncv overflows usize"))?;

    let lworkl_i32 = c_int_from_usize(lworkl)?;

    let tol = options.tol as f32;
    let zero = Complex32::new(0.0, 0.0);
    let mut resid = vec![zero; n];
    let mut v = vec![zero; v_len];
    let ldv = n_i32;
    let mut iparam = [0i32; 11];
    iparam[0] = 1;
    iparam[2] = max_iter_i32;
    iparam[3] = 1;
    iparam[6] = 1;
    let mut ipntr = [0i32; 14];
    let mut workd = vec![zero; workd_len];
    let mut workl = vec![zero; lworkl];
    let mut rwork = vec![0.0f32; ncv];

    let bmat = c"I".as_ptr();
    let which = c"SR".as_ptr();

    let _guard = lock();

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut matvec = matvec;
    let mut x_buf = vec![zero; n];

    loop {
        // SAFETY: identical reasoning to `smallest_eigenpair_c64`,
        // with `Complex32` storage instead of `Complex64`.
        unsafe {
            cnaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which,
                nev,
                tol,
                resid.as_mut_ptr() as *mut __BindgenComplex<f32>,
                ncv_i32,
                v.as_mut_ptr() as *mut __BindgenComplex<f32>,
                ldv,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr() as *mut __BindgenComplex<f32>,
                workl.as_mut_ptr() as *mut __BindgenComplex<f32>,
                lworkl_i32,
                rwork.as_mut_ptr(),
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

    // See `smallest_eigenpair_c64` for the rationale on splitting
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
    let mut d = vec![zero; nev_usize];
    let sigma = __BindgenComplex {
        re: 0.0_f32,
        im: 0.0_f32,
    };
    let mut workev = vec![zero; workev_len];
    let mut info_eup: c_int = 0;

    // SAFETY: as for cnaupd_c above; v doubles as z (output
    // eigenvector storage).
    unsafe {
        cneupd_c(
            rvec,
            howmny,
            select.as_mut_ptr(),
            d.as_mut_ptr() as *mut __BindgenComplex<f32>,
            v.as_mut_ptr() as *mut __BindgenComplex<f32>,
            ldv,
            sigma,
            workev.as_mut_ptr() as *mut __BindgenComplex<f32>,
            bmat,
            n_i32,
            which,
            nev,
            tol,
            resid.as_mut_ptr() as *mut __BindgenComplex<f32>,
            ncv_i32,
            v.as_mut_ptr() as *mut __BindgenComplex<f32>,
            ldv,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr() as *mut __BindgenComplex<f32>,
            workl.as_mut_ptr() as *mut __BindgenComplex<f32>,
            lworkl_i32,
            rwork.as_mut_ptr(),
            &mut info_eup,
        );
    }

    if info_eup != 0 {
        return Err(Error::EupdFailed(info_eup));
    }

    let value = d[0];
    let mut vector = vec![zero; n];
    vector.copy_from_slice(&v[..n]);
    Ok(EigSolution {
        eigenvalue: value,
        eigenvector: vector,
        iters: usize_from_iparam(iparam[2]),
        nconv: usize_from_iparam(iparam[4]),
        n_matvec: usize_from_iparam(iparam[8]),
    })
}

fn c_int_from_usize(value: usize) -> Result<c_int, Error> {
    c_int::try_from(value).map_err(|_| Error::InvalidParam("value does not fit in c_int"))
}
