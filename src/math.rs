// src/math.rs

//! Mathematical utilities for the PICARD algorithm.

use crate::error::{PicardError, Result};
use ndarray::Array2;
use ndarray_linalg::Eigh;
use ndarray_linalg::UPLO;

/// Symmetric decorrelation: W <- (W · W^T)^{-1/2} · W
///
/// This ensures the rows of W are orthonormal.
pub fn sym_decorrelation(w: &Array2<f64>) -> Result<Array2<f64>> {
    let ww_t = w.dot(&w.t());
    let (eigenvalues, eigenvectors) =
        ww_t.eigh(UPLO::Lower)
            .map_err(|_| PicardError::ComputationError {
                message: "Eigendecomposition failed in symmetric decorrelation".into(),
            })?;

    // Check for near-zero eigenvalues
    let min_eigenvalue = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_eigenvalue < 1e-10 {
        return Err(PicardError::SingularMatrix);
    }

    let s_inv_sqrt = eigenvalues.mapv(|v| 1.0 / v.sqrt());

    // (U · diag(1/sqrt(s)) · U^T) · W
    let scaled = &eigenvectors * &s_inv_sqrt;
    let result = scaled.dot(&eigenvectors.t()).dot(w);

    Ok(result)
}

/// Compute matrix exponential using Taylor series.
///
/// This is used for the orthogonal updates in Picard-O.
pub fn matrix_exp(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();

    // Check if matrix is essentially zero
    let norm: f64 = a.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    if norm < 1e-15 {
        return Array2::eye(n);
    }

    // Scale the matrix to improve convergence
    let s = (norm.log2().ceil().max(0.0)) as i32;
    let scale = 2.0_f64.powi(s);
    let a_scaled = a / scale;

    // Taylor series expansion
    let mut result = Array2::eye(n);
    let mut term = Array2::eye(n);
    let max_terms = 30;
    let tolerance = 1e-16;

    for k in 1..=max_terms {
        term = term.dot(&a_scaled) / (k as f64);
        result = &result + &term;

        let term_norm: f64 = term.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        if term_norm < tolerance {
            break;
        }
    }

    // Square the result s times to undo scaling
    for _ in 0..s {
        result = result.dot(&result);
    }

    result
}

/// Compute determinant of a square matrix using LU decomposition.
pub fn determinant(m: &Array2<f64>) -> f64 {
    let n = m.nrows();

    if n == 1 {
        return m[[0, 0]];
    }
    if n == 2 {
        return m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    }

    // LU decomposition with partial pivoting
    let mut lu = m.clone();
    let mut det = 1.0;

    for i in 0..n {
        // Find pivot
        let mut max_val = lu[[i, i]].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if lu[[k, i]].abs() > max_val {
                max_val = lu[[k, i]].abs();
                max_row = k;
            }
        }

        if max_val < 1e-15 {
            return 0.0;
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..n {
                let tmp = lu[[i, j]];
                lu[[i, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
            det = -det;
        }

        det *= lu[[i, i]];

        // Eliminate below
        for k in (i + 1)..n {
            let factor = lu[[k, i]] / lu[[i, i]];
            for j in i..n {
                lu[[k, j]] -= factor * lu[[i, j]];
            }
        }
    }

    det
}

/// Make a matrix skew-symmetric: A <- (A - A^T) / 2
pub fn skew_symmetric(a: &Array2<f64>) -> Array2<f64> {
    (a - &a.t()) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sym_decorrelation() {
        let w = array![[1.0, 0.5], [0.5, 1.0]];
        let w_dec = sym_decorrelation(&w).unwrap();
        let ww_t = w_dec.dot(&w_dec.t());

        // Should be close to identity
        assert!((ww_t[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((ww_t[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(ww_t[[0, 1]].abs() < 1e-10);
        assert!(ww_t[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_matrix_exp_identity() {
        let zero = Array2::<f64>::zeros((3, 3));
        let exp_zero = matrix_exp(&zero);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((exp_zero[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_determinant() {
        let m = array![[1.0, 2.0], [3.0, 4.0]];
        let det = determinant(&m);
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_skew_symmetric() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let skew = skew_symmetric(&a);

        // Check A = -A^T
        assert!((skew[[0, 0]]).abs() < 1e-10);
        assert!((skew[[1, 1]]).abs() < 1e-10);
        assert!((skew[[0, 1]] + skew[[1, 0]]).abs() < 1e-10);
    }
}
