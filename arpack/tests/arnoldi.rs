//! Smoke tests for the complex Arnoldi smallest-eigenpair driver.
//! Matrices are analytical so the smallest eigenvalue is known in
//! closed form; the imaginary part of the returned eigenvalue is
//! verified against zero for Hermitian inputs.

use arpack::Error;
use arpack::arnoldi::{Options, smallest_eigenpair_c64};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn diagonal_real_spectrum_returns_smallest() {
    // Hermitian (in fact real-diagonal) matrix with spectrum
    // {-2, 0, 1, 4, 7}; expected smallest = -2 with eigenvector
    // aligned with e_0.
    let diag = [c(-2.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(4.0, 0.0), c(7.0, 0.0)];
    let n = diag.len();

    let (lambda, vector) = smallest_eigenpair_c64(
        n,
        |x, y| {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
        },
        &Options {
            tol: 1e-12,
            max_iter: 200,
            ncv: None,
        },
    )
    .expect("driver should converge");

    assert!(
        (lambda.re + 2.0).abs() < 1e-10 && lambda.im.abs() < 1e-10,
        "lambda = {lambda} (expected -2 + 0i)"
    );
    assert!(
        vector[0].norm() > 0.99,
        "|vector[0]| = {} (expected near 1)",
        vector[0].norm()
    );
    let norm_sq: f64 = vector.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-10,
        "vector should be unit-normalized; norm^2 = {norm_sq}"
    );
}

#[test]
fn complex_hermitian_diagonal_returns_smallest_real() {
    // Diagonal with phase: H[k,k] = real eigenvalue, but build via a
    // complex similarity that doesn't change spectrum.
    // Use H = diag(1, 2, 3, 4) (still real but stored as complex);
    // smallest = 1.
    let n = 4;
    let diag = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];

    let (lambda, _vector) = smallest_eigenpair_c64(
        n,
        |x, y| {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
        },
        &Options::default(),
    )
    .expect("driver should converge");

    assert!(
        (lambda.re - 1.0).abs() < 1e-10 && lambda.im.abs() < 1e-10,
        "lambda = {lambda} (expected 1 + 0i)"
    );
}

#[test]
fn hermitian_tridiagonal_matches_analytical_smallest() {
    // 1D Laplacian-style Hermitian tridiagonal: diagonal 2, off-
    // diagonal -1. Eigenvalues are
    //   lambda_k = 2 - 2 cos(k pi / (n + 1)),  k = 1..=n.
    // Smallest is at k = 1.
    let n = 32usize;
    let lambda_min_expected = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();

    let (lambda, _vector) = smallest_eigenpair_c64(
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
            tol: 1e-12,
            max_iter: 500,
            ncv: None,
        },
    )
    .expect("driver should converge");

    assert!(
        lambda.im.abs() < 1e-9,
        "expected real eigenvalue, got imag = {}",
        lambda.im
    );
    let rel_err = (lambda.re - lambda_min_expected).abs() / lambda_min_expected.abs();
    assert!(
        rel_err < 1e-9,
        "lambda.re = {}, expected {lambda_min_expected}, rel_err = {rel_err}",
        lambda.re
    );
}

#[test]
fn explicit_ncv_equal_to_n_is_rejected() {
    let n = 8;
    let result = smallest_eigenpair_c64(
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
