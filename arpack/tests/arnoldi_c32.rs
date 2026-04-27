//! Smoke tests for the f32 complex Arnoldi driver. Tolerances are
//! looser than the c64 suite — single precision can converge no
//! tighter than its relative epsilon (~1.2e-7).

use arpack::Error;
use arpack::arnoldi::{Options, smallest_eigenpair_c32};
use num_complex::Complex32;

fn c(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

#[test]
fn diagonal_real_spectrum_returns_smallest() {
    let diag = [c(-2.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(4.0, 0.0), c(7.0, 0.0)];
    let n = diag.len();

    let (lambda, vector) = smallest_eigenpair_c32(
        n,
        |x, y| {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
        },
        &Options {
            tol: 1e-6,
            max_iter: 200,
            ncv: None,
        },
    )
    .expect("driver should converge");

    assert!(
        (lambda.re + 2.0).abs() < 1e-5 && lambda.im.abs() < 1e-5,
        "lambda = {lambda} (expected -2 + 0i)"
    );
    assert!(
        vector[0].norm() > 0.99,
        "|vector[0]| = {} (expected near 1)",
        vector[0].norm()
    );
    let norm_sq: f32 = vector.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-5,
        "vector should be unit-normalized; norm^2 = {norm_sq}"
    );
}

#[test]
fn hermitian_tridiagonal_matches_analytical_smallest() {
    // 1D Laplacian-style Hermitian tridiagonal: diagonal 2,
    // off-diagonal -1. Eigenvalues 2 - 2 cos(k pi / (n+1)).
    let n = 32usize;
    let lambda_min_expected = 2.0_f32 - 2.0 * (std::f32::consts::PI / (n as f32 + 1.0)).cos();

    let (lambda, _vector) = smallest_eigenpair_c32(
        n,
        |x, y| {
            for i in 0..n {
                let center = c(2.0, 0.0) * x[i];
                let left = if i > 0 { c(-1.0, 0.0) * x[i - 1] } else { c(0.0, 0.0) };
                let right = if i + 1 < n { c(-1.0, 0.0) * x[i + 1] } else { c(0.0, 0.0) };
                y[i] = center + left + right;
            }
        },
        &Options {
            tol: 1e-6,
            max_iter: 500,
            ncv: None,
        },
    )
    .expect("driver should converge");

    assert!(
        lambda.im.abs() < 1e-5,
        "expected real eigenvalue, got imag = {}",
        lambda.im
    );
    let rel_err = (lambda.re - lambda_min_expected).abs() / lambda_min_expected.abs();
    assert!(
        rel_err < 1e-4,
        "lambda.re = {}, expected {lambda_min_expected}, rel_err = {rel_err}",
        lambda.re
    );
}

#[test]
fn hermitian_with_imaginary_off_diagonals() {
    // H_C = D H_R D^* where D = diag(i^k); off-diagonals become +/- i.
    let n = 8usize;
    let lambda_min_expected = 2.0_f32 - 2.0 * (std::f32::consts::PI / (n as f32 + 1.0)).cos();
    let im = c(0.0, 1.0);

    let (lambda, vector) = smallest_eigenpair_c32(
        n,
        |x, y| {
            for i in 0..n {
                let center = c(2.0, 0.0) * x[i];
                let upper = if i + 1 < n { im * x[i + 1] } else { c(0.0, 0.0) };
                let lower = if i > 0 { -im * x[i - 1] } else { c(0.0, 0.0) };
                y[i] = center + upper + lower;
            }
        },
        &Options {
            tol: 1e-6,
            max_iter: 500,
            ncv: None,
        },
    )
    .expect("driver should converge");

    assert!(
        lambda.im.abs() < 1e-5,
        "Hermitian eigenvalue should be real; got imag = {}",
        lambda.im
    );
    let rel_err = (lambda.re - lambda_min_expected).abs() / lambda_min_expected.abs();
    assert!(
        rel_err < 1e-4,
        "lambda.re = {}, expected {lambda_min_expected}, rel_err = {rel_err}",
        lambda.re
    );
    let total_imag: f32 = vector.iter().map(|c| c.im.abs()).sum();
    assert!(
        total_imag > 1e-2,
        "expected non-trivial imaginary content; sum |im| = {total_imag}"
    );
}

#[test]
fn explicit_ncv_equal_to_n_is_rejected() {
    let n = 8;
    let result = smallest_eigenpair_c32(
        n,
        |_x, _y| unreachable!("matvec should not run when params are rejected"),
        &Options {
            tol: 0.0,
            max_iter: 100,
            ncv: Some(n),
        },
    );
    assert!(matches!(result, Err(Error::InvalidParam(_))));
}

#[test]
fn explicit_ncv_equals_nev_plus_one_is_rejected() {
    let n = 8;
    let result = smallest_eigenpair_c32(
        n,
        |_x, _y| unreachable!("matvec should not run when params are rejected"),
        &Options {
            tol: 0.0,
            max_iter: 100,
            ncv: Some(2), // = nev + 1 with nev hardcoded to 1
        },
    );
    assert!(matches!(result, Err(Error::InvalidParam(_))));
}
