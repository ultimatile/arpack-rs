//! Smoke tests for the real-symmetric Lanczos driver. The matrices are
//! analytical (diagonal / tridiagonal) so the smallest eigenvalue is
//! known in closed form.

use arpack::{Error, Options, smallest_eigenpair_f64};

#[test]
fn diagonal_matrix_returns_smallest() {
    // Spectrum {-3, -1, 0, 2, 5}; expected smallest = -3 with
    // eigenvector aligned with e_0 (up to a sign).
    let diag = [-3.0_f64, -1.0, 0.0, 2.0, 5.0];
    let n = diag.len();

    let solution = smallest_eigenpair_f64(
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

    let lambda = solution.eigenvalue;
    let vector = &solution.eigenvector;
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

    // The driver should fully converge for this trivial spectrum:
    // exactly one Ritz pair settles, and ARPACK reports the matvec count.
    assert_eq!(solution.nconv, 1, "expected full convergence (nconv = 1)");
    assert!(
        solution.iters >= 1 && solution.iters <= 200,
        "iters out of range: {}",
        solution.iters
    );
    assert!(
        solution.n_matvec >= 1,
        "matvec count should be positive: {}",
        solution.n_matvec
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

    let solution = smallest_eigenpair_f64(
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

    let lambda = solution.eigenvalue;
    let rel_err = (lambda - lambda_min_expected).abs() / lambda_min_expected.abs();
    assert!(
        rel_err < 1e-9,
        "lambda = {lambda}, expected {lambda_min_expected}, rel_err = {rel_err}"
    );
}

#[test]
fn explicit_ncv_equal_to_n_is_rejected() {
    // ARPACK fails with info = -9999 when ncv == n because IRLM has no
    // restart room. The wrapper must reject that configuration up front
    // rather than letting the upstream error surface as a runtime
    // failure deep in the reverse-communication loop.
    let n = 8;
    let result = smallest_eigenpair_f64(
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
    // With n = nev + 2 = 3 the only legal default ncv is nev + 1 = 2.
    // The earlier heuristic (floor `nev + 2`) would have produced
    // ncv = n = 3 here and triggered the upstream -9999 failure.
    let n = 3;
    let diag = [1.0_f64, 4.0, 9.0]; // smallest eigenvalue = 1.0
    let solution = smallest_eigenpair_f64(
        n,
        |x, y| {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
        },
        &Options {
            tol: 1e-12,
            max_iter: 100,
            ncv: None,
        },
    )
    .expect("driver should converge at the smallest legal n");
    let lambda = solution.eigenvalue;
    assert!(
        (lambda - 1.0).abs() < 1e-10,
        "lambda = {lambda} (expected 1.0)"
    );
}

#[test]
fn max_iter_too_small_returns_max_iter_reached() {
    // Force ARPACK to bail out via the `info = 1` branch. With nev = 1
    // hardcoded, that always implies `nconv = 0` — the wrapper must
    // surface `Error::MaxIterReached` rather than calling `*seupd`
    // (which would quick-return without writing the eigenpair) and
    // returning a bogus `Ok`.
    //
    // The 1D Laplacian on n = 64 has dense low-end spectrum, so a
    // single restart cycle is nowhere near convergence.
    let n = 64usize;
    let result = smallest_eigenpair_f64(
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
            tol: 1e-15,
            max_iter: 1,
            ncv: None,
        },
    );

    match result {
        Err(Error::MaxIterReached {
            iters,
            nconv,
            n_matvec,
        }) => {
            // For nev = 1 hardcoded, nconv must be 0 in this branch
            // (ARPACK only sets info = 1 when fewer than nev pairs
            // converged).
            assert_eq!(nconv, 0, "nev = 1 forces nconv = 0 on max_iter exit");
            assert!(iters >= 1, "iters should reflect the cap: {iters}");
            assert!(
                n_matvec >= 1,
                "matvec count should be positive on a non-trivial run: {n_matvec}"
            );
        }
        other => panic!("expected MaxIterReached, got {other:?}"),
    }
}
