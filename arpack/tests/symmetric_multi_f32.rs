//! f32 multi-eigenvalue tests for the real-symmetric driver.
//! Tolerances are looser than the f64 suite — single precision
//! can converge no tighter than its relative epsilon (~`1.2e-7`).

use arpack::Error;
use arpack::Which;
use arpack::symmetric::{Options, eigenpairs_f32};

#[test]
fn eigenpairs_smallest_three_diag_returns_sorted() {
    // Same fixture shape as the f64 mirror, with the zero
    // eigenvalue replaced (ARPACK loses it from a random start
    // on diagonal mode-1 operators). Spectrum = diagonal entries;
    // smallest three algebraic = {-3, -1, 1}.
    let diag = [-3.0_f32, -1.0, 1.0, 2.0, 5.0, 8.0, 11.0, 14.0];
    let n = diag.len();
    let nev = 3;

    let solution = eigenpairs_f32(
        n,
        nev,
        Which::SmallestAlgebraic,
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
        solution.nconv >= nev,
        "expected full convergence (nconv >= nev); nconv = {}, nev = {}",
        solution.nconv,
        nev
    );
    assert_eq!(solution.eigenvalues.len(), nev);
    assert_eq!(solution.eigenvectors.len(), nev);
    let expected = [-3.0_f32, -1.0, 1.0];
    for (k, (&got, &exp)) in solution.eigenvalues.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "eigenvalues[{k}] = {got} (expected {exp})"
        );
    }
}

#[test]
fn eigenpairs_largest_three_laplacian_returns_ascending() {
    let n = 32usize;
    let nev = 3;
    let pi = std::f32::consts::PI;
    let lambda = |k: usize| 2.0 - 2.0 * (k as f32 * pi / (n as f32 + 1.0)).cos();
    let expected = [lambda(n - 2), lambda(n - 1), lambda(n)];

    let solution = eigenpairs_f32(
        n,
        nev,
        Which::LargestAlgebraic,
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

    assert!(
        solution.nconv >= nev,
        "expected full convergence (nconv >= nev); nconv = {}, nev = {}",
        solution.nconv,
        nev
    );
    assert_eq!(solution.eigenvalues.len(), nev);
    assert_eq!(solution.eigenvectors.len(), nev);
    for (k, (&got, &exp)) in solution.eigenvalues.iter().zip(expected.iter()).enumerate() {
        let rel_err = (got - exp).abs() / exp.abs();
        assert!(
            rel_err < 1e-4,
            "eigenvalues[{k}] = {got}, expected {exp}, rel_err = {rel_err}"
        );
    }
}

#[test]
fn eigenpairs_rejects_complex_only_which() {
    let n = 8;
    let result = eigenpairs_f32(
        n,
        2,
        Which::SmallestRealPart,
        |_x, _y| unreachable!(),
        &Options::default(),
    );
    assert!(matches!(result, Err(Error::InvalidParam(_))));
}
