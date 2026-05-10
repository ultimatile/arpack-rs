//! Smoke tests for the f32 real-symmetric driver. Tolerances are
//! looser than the f64 suite — single precision can converge no
//! tighter than its relative epsilon (~1.2e-7).

use arpack::Error;
use arpack::symmetric::{Options, smallest_eigenpair_f32};

#[test]
fn diagonal_matrix_returns_smallest() {
    let diag = [-3.0_f32, -1.0, 0.0, 2.0, 5.0];
    let n = diag.len();

    let solution = smallest_eigenpair_f32(
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

    let lambda = solution.eigenvalue;
    let vector = &solution.eigenvector;
    assert!(
        (lambda + 3.0).abs() < 1e-5,
        "lambda = {lambda} (expected -3)"
    );
    assert!(
        vector[0].abs() > 0.99,
        "vector[0] = {} (expected near +/-1)",
        vector[0]
    );
    let norm_sq: f32 = vector.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-5,
        "vector should be unit-normalized; norm^2 = {norm_sq}"
    );
}

#[test]
fn tridiagonal_matrix_matches_analytical_smallest() {
    // 1D Laplacian-style tridiagonal: diagonal 2, off-diagonal -1.
    // Eigenvalues lambda_k = 2 - 2 cos(k pi / (n + 1)).
    let n = 32usize;
    let lambda_min_expected = 2.0_f32 - 2.0 * (std::f32::consts::PI / (n as f32 + 1.0)).cos();

    let solution = smallest_eigenpair_f32(
        n,
        |x, y| {
            for i in 0..n {
                let center = 2.0 * x[i];
                let left = if i > 0 { -x[i - 1] } else { 0.0 };
                let right = if i + 1 < n { -x[i + 1] } else { 0.0 };
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

    let lambda = solution.eigenvalue;
    let rel_err = (lambda - lambda_min_expected).abs() / lambda_min_expected.abs();
    assert!(
        rel_err < 1e-4,
        "lambda = {lambda}, expected {lambda_min_expected}, rel_err = {rel_err}"
    );
}

#[test]
fn explicit_ncv_equal_to_n_is_rejected() {
    let n = 8;
    let result = smallest_eigenpair_f32(
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
fn boundary_n_equals_nev_plus_two_uses_default_ncv() {
    // n = nev + 2 = 3; default heuristic must produce ncv = 2 = nev + 1.
    let n = 3;
    let diag = [1.0_f32, 4.0, 9.0]; // smallest = 1.0
    let solution = smallest_eigenpair_f32(
        n,
        |x, y| {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
        },
        &Options {
            tol: 1e-6,
            max_iter: 100,
            ncv: None,
        },
    )
    .expect("driver should converge at the smallest legal n");
    let lambda = solution.eigenvalue;
    assert!(
        (lambda - 1.0).abs() < 1e-5,
        "lambda = {lambda} (expected 1.0)"
    );
}
