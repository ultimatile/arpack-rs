//! f32 multi-eigenvalue tests for the complex Arnoldi driver.
//! Tolerances are looser than the c64 suite — single precision
//! can converge no tighter than its relative epsilon (~`1.2e-7`).

use arpack::Error;
use arpack::Which;
use arpack::arnoldi::{Options, eigenpairs_c32};
use num_complex::Complex32;

fn c(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

fn sorted_by_re(eigenvalues: &[Complex32]) -> Vec<Complex32> {
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).expect("eigenvalues are finite"));
    sorted
}

#[test]
fn eigenpairs_smallest_real_part_three_laplacian() {
    let n = 32usize;
    let nev = 3;
    let pi = std::f32::consts::PI;
    let lambda = |k: usize| 2.0 - 2.0 * (k as f32 * pi / (n as f32 + 1.0)).cos();
    let expected = [lambda(1), lambda(2), lambda(3)];

    let solution = eigenpairs_c32(
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
    let sorted = sorted_by_re(&solution.eigenvalues);
    for (k, (&got, &exp)) in sorted.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.im.abs() < 1e-5,
            "Hermitian eigenvalue should be real; got im = {}",
            got.im
        );
        let rel_err = (got.re - exp).abs() / exp.abs();
        assert!(
            rel_err < 1e-4,
            "eigenvalue[{k}].re = {}, expected {exp}, rel_err = {rel_err}",
            got.re
        );
    }
}

#[test]
fn eigenpairs_rejects_symmetric_only_which() {
    let n = 8;
    let result = eigenpairs_c32(
        n,
        2,
        Which::LargestAlgebraic,
        |_x, _y| unreachable!(),
        &Options::default(),
    );
    assert!(matches!(result, Err(Error::InvalidParam(_))));
}
