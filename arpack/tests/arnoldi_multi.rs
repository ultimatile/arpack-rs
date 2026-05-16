//! Multi-eigenvalue tests for the complex Arnoldi driver
//! (`eigenpairs_c64`). Unlike the symmetric extraction routine,
//! `zneupd` does not sort its output by `which` — tests sort the
//! returned vectors themselves before comparing to expected.

use arpack::Error;
use arpack::Which;
use arpack::arnoldi::{Options, eigenpairs_c64};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn sorted_by_re(eigenvalues: &[Complex64]) -> Vec<Complex64> {
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).expect("eigenvalues are finite"));
    sorted
}

#[test]
fn eigenpairs_smallest_real_part_three_laplacian() {
    // Hermitian 1D Laplacian on n=32 driven through the complex
    // Arnoldi routine. Plan D1 derivation: λ_k = 2 - 2 cos(k π /
    // (n+1)). With Which::SmallestRealPart the wrapper returns
    // the three smallest real-part eigenvalues, but `zneupd` does
    // not sort — we sort by re here. Expected smallest three are
    // λ_1, λ_2, λ_3.
    let n = 32usize;
    let nev = 3;
    let pi = std::f64::consts::PI;
    let lambda = |k: usize| 2.0 - 2.0 * (k as f64 * pi / (n as f64 + 1.0)).cos();
    let expected = [lambda(1), lambda(2), lambda(3)];

    let solution = eigenpairs_c64(
        n,
        nev,
        Which::SmallestRealPart,
        |x, y| {
            for i in 0..n {
                let center = c(2.0, 0.0) * x[i];
                let left = if i > 0 {
                    c(-1.0, 0.0) * x[i - 1]
                } else {
                    c(0.0, 0.0)
                };
                let right = if i + 1 < n {
                    c(-1.0, 0.0) * x[i + 1]
                } else {
                    c(0.0, 0.0)
                };
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

    assert_eq!(solution.nconv, nev);
    let sorted = sorted_by_re(&solution.eigenvalues);
    for (k, (&got, &exp)) in sorted.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.im.abs() < 1e-9,
            "Hermitian eigenvalue should be real; got im = {}",
            got.im
        );
        let rel_err = (got.re - exp).abs() / exp.abs();
        assert!(
            rel_err < 1e-9,
            "eigenvalue[{k}].re = {}, expected {exp}, rel_err = {rel_err}",
            got.re
        );
    }
}

#[test]
fn eigenpairs_largest_real_part_three_laplacian() {
    let n = 32usize;
    let nev = 3;
    let pi = std::f64::consts::PI;
    let lambda = |k: usize| 2.0 - 2.0 * (k as f64 * pi / (n as f64 + 1.0)).cos();
    let expected = [lambda(n - 2), lambda(n - 1), lambda(n)];

    let solution = eigenpairs_c64(
        n,
        nev,
        Which::LargestRealPart,
        |x, y| {
            for i in 0..n {
                let center = c(2.0, 0.0) * x[i];
                let left = if i > 0 {
                    c(-1.0, 0.0) * x[i - 1]
                } else {
                    c(0.0, 0.0)
                };
                let right = if i + 1 < n {
                    c(-1.0, 0.0) * x[i + 1]
                } else {
                    c(0.0, 0.0)
                };
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

    assert_eq!(solution.nconv, nev);
    let sorted = sorted_by_re(&solution.eigenvalues);
    for (k, (&got, &exp)) in sorted.iter().zip(expected.iter()).enumerate() {
        let rel_err = (got.re - exp).abs() / exp.abs();
        assert!(
            rel_err < 1e-9,
            "eigenvalue[{k}].re = {}, expected {exp}, rel_err = {rel_err}",
            got.re
        );
    }
}

#[test]
fn eigenpairs_smallest_magnitude_complex_diag() {
    // Hermitian diagonal (entries are complex but with zero imag
    // part) on n=8: spectrum {-5, -3, 1, 2, 4, 7, 10, 15}. By |λ|:
    // 1 < 2 < 3 < 4 < 5 < ..., so smallest three by magnitude
    // = {1, 2, -3}. Sort post-hoc by re for comparison.
    let diag = [
        c(-5.0, 0.0),
        c(-3.0, 0.0),
        c(1.0, 0.0),
        c(2.0, 0.0),
        c(4.0, 0.0),
        c(7.0, 0.0),
        c(10.0, 0.0),
        c(15.0, 0.0),
    ];
    let n = diag.len();
    let nev = 3;

    let solution = eigenpairs_c64(
        n,
        nev,
        Which::SmallestMagnitude,
        |x, y| {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
        },
        &Options {
            tol: 1e-12,
            max_iter: 300,
            ncv: None,
        },
    )
    .expect("driver should converge");

    assert_eq!(solution.nconv, nev);
    let sorted = sorted_by_re(&solution.eigenvalues);
    let expected_re = [-3.0_f64, 1.0, 2.0];
    for (k, (&got, &exp)) in sorted.iter().zip(expected_re.iter()).enumerate() {
        assert!(
            (got.re - exp).abs() < 1e-9 && got.im.abs() < 1e-9,
            "eigenvalue[{k}] = {got} (expected {exp} + 0i)"
        );
    }
}

#[test]
fn eigenpairs_hermitian_imaginary_off_diagonals_nev3() {
    // Plan Fixture D variant for nev=3: H_C = D H_R D^* with
    // D = diag(i^k) on the real 1D-Laplacian-style tridiagonal.
    // Spectrum is the same as the real Laplacian (similarity
    // preserves it). Off-diagonals become +/- i so the complex
    // arithmetic path actually runs.
    let n = 8usize;
    let nev = 3;
    let pi = std::f64::consts::PI;
    let lambda = |k: usize| 2.0 - 2.0 * (k as f64 * pi / (n as f64 + 1.0)).cos();
    let expected = [lambda(1), lambda(2), lambda(3)];
    let im = c(0.0, 1.0);

    let solution = eigenpairs_c64(
        n,
        nev,
        Which::SmallestRealPart,
        |x, y| {
            for i in 0..n {
                let center = c(2.0, 0.0) * x[i];
                let upper = if i + 1 < n {
                    im * x[i + 1]
                } else {
                    c(0.0, 0.0)
                };
                let lower = if i > 0 { -im * x[i - 1] } else { c(0.0, 0.0) };
                y[i] = center + upper + lower;
            }
        },
        &Options {
            tol: 1e-12,
            max_iter: 500,
            ncv: None,
        },
    )
    .expect("driver should converge");

    assert_eq!(solution.nconv, nev);
    let sorted = sorted_by_re(&solution.eigenvalues);
    for (k, (&got, &exp)) in sorted.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.im.abs() < 1e-9,
            "eigenvalue[{k}].im = {} (expected ~0 for Hermitian)",
            got.im
        );
        let rel_err = (got.re - exp).abs() / exp.abs();
        assert!(
            rel_err < 1e-9,
            "eigenvalue[{k}].re = {}, expected {exp}, rel_err = {rel_err}",
            got.re
        );
    }

    // Sanity: at least some eigenvector entry has non-trivial
    // imaginary content (D = diag(i^k) rotates each component).
    let total_imag: f64 = solution
        .eigenvectors
        .iter()
        .flat_map(|v| v.iter())
        .map(|c| c.im.abs())
        .sum();
    assert!(
        total_imag > 1e-2,
        "expected non-trivial imaginary content in eigenvectors; sum |im| = {total_imag}"
    );
}

#[test]
fn eigenpairs_rejects_symmetric_only_which_for_complex() {
    let n = 8;
    let result = eigenpairs_c64(
        n,
        2,
        Which::SmallestAlgebraic,
        |_x, _y| unreachable!(),
        &Options::default(),
    );
    assert!(matches!(result, Err(Error::InvalidParam(_))));
}

#[test]
fn eigenpairs_nev_zero_is_rejected() {
    let n = 8;
    let result = eigenpairs_c64(
        n,
        0,
        Which::SmallestRealPart,
        |_x, _y| unreachable!(),
        &Options::default(),
    );
    assert!(matches!(result, Err(Error::InvalidParam(_))));
}

#[test]
fn eigenpairs_max_iter_one_yields_max_iter_reached_or_partial() {
    // Tolerant partial-convergence test mirroring the symmetric
    // version. Accepts either Err(MaxIterReached { nconv: 0 }) or
    // Ok with 0 < nconv <= nev.
    let n = 64usize;
    let nev = 3;
    let matvec_fn = |x: &[Complex64], y: &mut [Complex64]| {
        for i in 0..n {
            let center = c(2.0, 0.0) * x[i];
            let left = if i > 0 {
                c(-1.0, 0.0) * x[i - 1]
            } else {
                c(0.0, 0.0)
            };
            let right = if i + 1 < n {
                c(-1.0, 0.0) * x[i + 1]
            } else {
                c(0.0, 0.0)
            };
            y[i] = center + left + right;
        }
    };
    let result = eigenpairs_c64(
        n,
        nev,
        Which::SmallestRealPart,
        matvec_fn,
        &Options {
            tol: 1e-15,
            max_iter: 1,
            ncv: None,
        },
    );

    match result {
        Err(Error::MaxIterReached { nconv, iters, .. }) => {
            assert_eq!(nconv, 0, "MaxIterReached must carry nconv = 0");
            assert!(iters >= 1);
        }
        Ok(solution) => {
            assert!(
                solution.nconv >= 1 && solution.nconv <= nev,
                "partial-Ok branch requires 0 < nconv <= nev, got nconv = {}",
                solution.nconv
            );
            assert_eq!(solution.eigenvalues.len(), solution.nconv);
            assert_eq!(solution.eigenvectors.len(), solution.nconv);
        }
        other => panic!("expected MaxIterReached or Ok partial, got {other:?}"),
    }
}
