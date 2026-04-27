//! Arnoldi-iteration eigenvalue drivers (`*naupd_c` / `*neupd_c`).
//!
//! ARPACK-NG ships two Arnoldi families: real-non-symmetric
//! (`{s,d}{na,ne}upd_c`) and complex (`{c,z}{na,ne}upd_c`). For now
//! we expose only the `Complex<f64>` driver because that covers
//! arnet's primary oracle use case (Hermitian operators in DMRG /
//! quantum-system simulations). `Complex<f32>` will mirror this
//! module with the smaller scalar.
//!
//! Hermitian operators have real eigenvalues but are still driven
//! through the complex Arnoldi routine; the returned eigenvalue
//! comes back as `Complex<f64>` and callers verify / discard the
//! imaginary part themselves.
//!
//! Thread-safety: every entry point acquires the crate-wide
//! [`crate::lock`] guard so the entire `*aupd_c` + `*eupd_c`
//! sequence runs atomically against ARPACK's Fortran SAVE state.

use std::os::raw::c_int;

use arpack_sys::{__BindgenComplex, znaupd_c, zneupd_c};
use num_complex::Complex64;

use crate::error::Error;
use crate::lock::lock;

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
    /// Krylov-subspace dimension `ncv`. See
    /// [`crate::symmetric::Options::ncv`] for why `ncv < n` is
    /// enforced strictly.
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
) -> Result<(Complex64, Vec<Complex64>), Error>
where
    F: FnMut(&[Complex64], &mut [Complex64]),
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

    let n_i32 = c_int_from_usize(n)?;
    let ncv_i32 = c_int_from_usize(ncv)?;
    let max_iter_i32 = c_int_from_usize(options.max_iter)?;

    if !(nev > 0 && nev < ncv_i32 && ncv_i32 < n_i32) {
        return Err(Error::InvalidParam("require 0 < nev < ncv < n"));
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
        // `#[repr(C)] { re: f64, im: f64 }`. The crate-wide lock
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
    Ok((value, vector))
}

fn c_int_from_usize(value: usize) -> Result<c_int, Error> {
    c_int::try_from(value).map_err(|_| Error::InvalidParam("value does not fit in c_int"))
}
