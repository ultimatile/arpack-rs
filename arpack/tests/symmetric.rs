//! Smoke tests for the real-symmetric Lanczos driver. The matrices are
//! analytical (diagonal / tridiagonal) so the smallest eigenvalue is
//! known in closed form.

use arpack::{Options, smallest_eigenpair_f64};

#[test]
fn diagonal_matrix_returns_smallest() {
    // Spectrum {-3, -1, 0, 2, 5}; expected smallest = -3 with
    // eigenvector aligned with e_0 (up to a sign).
    let diag = [-3.0_f64, -1.0, 0.0, 2.0, 5.0];
    let n = diag.len();

    let (lambda, vector) = smallest_eigenpair_f64(
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
    .expect("ARPACK driver should converge");

    assert!(
        (lambda + 3.0).abs() < 1e-10,
        "lambda = {lambda} (expected -3)"
    );
    assert!(
        vector[0].abs() > 0.99,
        "vector[0] = {} (expected near +/-1)",
        vector[0]
    );

    let norm_sq: f64 = vector.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-10,
        "vector should be unit-normalized; norm^2 = {norm_sq}"
    );
}

#[test]
fn tridiagonal_matrix_matches_analytical_smallest() {
    // For the 1D Laplacian-style tridiagonal matrix with diagonal 2
    // and off-diagonal -1 of size n, eigenvalues are
    //   lambda_k = 2 - 2 cos(k pi / (n + 1)),  k = 1..=n.
    // Smallest eigenvalue is at k=1.
    let n = 32usize;
    let lambda_min_expected = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();

    let (lambda, _vector) = smallest_eigenpair_f64(
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
            tol: 1e-12,
            max_iter: 500,
            ncv: None,
        },
    )
    .expect("ARPACK driver should converge");

    let rel_err = (lambda - lambda_min_expected).abs() / lambda_min_expected.abs();
    assert!(
        rel_err < 1e-9,
        "lambda = {lambda}, expected {lambda_min_expected}, rel_err = {rel_err}"
    );
}
