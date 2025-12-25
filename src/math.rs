// src/math.rs

//! Mathematical utilities for the PICARD algorithm.

use crate::error::{PicardError, Result};
use ndarray::Array2;
use ndarray_linalg::{Determinant, Eigh, UPLO};

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

/// Compute the signed log-determinant of a square matrix using LAPACK.
///
/// Returns (sign, log_abs_det) where:
/// - sign is 1.0, -1.0, or 0.0
/// - log_abs_det is ln(|det(m)|)
///
/// This is more numerically stable than computing det directly,
/// especially for the log|det| terms used in ICA objectives.
pub fn sln_det(m: &Array2<f64>) -> Result<(f64, f64)> {
    m.sln_det().map_err(|e| PicardError::ComputationError {
        message: format!("LU decomposition failed in determinant computation: {}", e),
    })
}

/// Compute determinant of a square matrix using LAPACK.
///
/// For ICA objectives that need log|det|, prefer using `sln_det` directly
/// to avoid numerical issues with very large or small determinants.
pub fn determinant(m: &Array2<f64>) -> Result<f64> {
    m.det().map_err(|e| PicardError::ComputationError {
        message: format!("LU decomposition failed in determinant computation: {}", e),
    })
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
        let det = determinant(&m).unwrap();
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sln_det() {
        let m = array![[1.0, 2.0], [3.0, 4.0]];
        let (sign, log_abs) = sln_det(&m).unwrap();
        assert!((sign - (-1.0)).abs() < 1e-10); // det is negative
        assert!((log_abs - 2.0_f64.ln()).abs() < 1e-10); // |det| = 2
    }

    #[test]
    fn test_sln_det_large_values() {
        // Test with a matrix that would overflow if computing det directly
        let m = array![[1e150, 0.0], [0.0, 1e150]];
        let (sign, log_abs) = sln_det(&m).unwrap();
        assert!((sign - 1.0).abs() < 1e-10);
        assert!((log_abs - 2.0 * 150.0 * 10.0_f64.ln()).abs() < 1e-6);
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
