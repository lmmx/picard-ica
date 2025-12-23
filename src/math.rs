// src/math.rs

//! Mathematical utilities for the PICARD algorithm.

use crate::error::{PicardError, Result};
use faer::{Mat, MatRef};
use faer::matrix_free::LinOp;

/// Symmetric decorrelation: W <- (W · W^T)^{-1/2} · W
///
/// This ensures the rows of W are orthonormal.
pub fn sym_decorrelation(w: MatRef<'_, f64>) -> Result<Mat<f64>> {
    let ww_t = w * w.transpose();

    // Self-adjoint eigendecomposition
    let eigen = ww_t.self_adjoint_eigen(faer::Side::Lower)?;
    let eigenvalues = eigen.S();
    let eigenvectors = eigen.U();

    // Check for near-zero eigenvalues
    let n = eigenvalues.ncols();
    let mut min_eigenvalue = f64::INFINITY;
    for i in 0..n {
        let ev = eigenvalues[i];
        if ev < min_eigenvalue {
            min_eigenvalue = ev;
        }
    }

    if min_eigenvalue < 1e-10 {
        return Err(PicardError::SingularMatrix);
    }

    // Compute S^{-1/2}
    let mut s_inv_sqrt_diag = Mat::zeros(n, n);
    for i in 0..n {
        s_inv_sqrt_diag[(i,i)] = 1.0 / eigenvalues[i].sqrt();
    }

    // (U · diag(1/sqrt(s)) · U^T) · W
    let result = &eigenvectors * &s_inv_sqrt_diag * eigenvectors.transpose() * w;

    Ok(result)
}

/// Compute matrix exponential using Taylor series.
///
/// This is used for the orthogonal updates in Picard-O.
pub fn matrix_exp(a: MatRef<'_, f64>) -> Mat<f64> {
    let n = a.nrows();

    // Check if matrix is essentially zero
    let mut norm: f64 = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            norm = norm.max(a[(i, j)].abs());
        }
    }

    if norm < 1e-15 {
        return Mat::identity(n, n);
    }

    // Scale the matrix to improve convergence
    let s = (norm.log2().ceil().max(0.0)) as i32;
    let scale = 2.0_f64.powi(s);
    let a_scaled = &a * faer::Scale(1.0 / scale);

    // Taylor series expansion
    let mut result = Mat::<f64>::identity(n, n);
    let mut term = Mat::<f64>::identity(n, n);
    let max_terms = 30;
    let tolerance = 1e-16;

    for k in 1..=max_terms {
        term = &term * &a_scaled * faer::Scale(1.0 / k as f64);
        result = &result + &term;

        let mut term_norm: f64 = 0.0;
        for j in 0..term.ncols() {
            for i in 0..term.nrows() {
                term_norm = term_norm.max(term[(i, j)].abs());
            }
        }
        if term_norm < tolerance {
            break;
        }
    }

    // Square the result s times to undo scaling
    for _ in 0..s {
        result = &result * &result;
    }

    result
}

/// Compute determinant of a square matrix using LU decomposition.
pub fn determinant(m: MatRef<'_, f64>) -> f64 {
    let n = m.nrows();

    if n == 1 {
        return m[(0, 0)];
    }
    if n == 2 {
        return m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
    }

    // LU decomposition with partial pivoting
    let mut lu = m.to_owned();
    let mut det = 1.0;

    for i in 0..n {
        // Find pivot
        let mut max_val = lu[(i, i)].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if lu[(k, i)].abs() > max_val {
                max_val = lu[(k, i)].abs();
                max_row = k;
            }
        }

        if max_val < 1e-15 {
            return 0.0;
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..n {
                let tmp = lu[(i, j)];
                lu[(i, j)] = lu[(max_row, j)];
                lu[(max_row, j)] = tmp;
            }
            det = -det;
        }

        det *= lu[(i, i)];

        // Eliminate below
        for k in (i + 1)..n {
            let factor = lu[(k, i)] / lu[(i, i)];
            for j in i..n {
                lu[(k, j)] -= factor * lu[(i, j)];
            }
        }
    }

    det
}

/// Make a matrix skew-symmetric: A <- (A - A^T) / 2
pub fn skew_symmetric(a: MatRef<'_, f64>) -> Mat<f64> {
    (&a - a.transpose()) * faer::Scale(0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_sym_decorrelation() {
        let w = mat![[1.0, 0.5], [0.5, 1.0]];
        let w_dec = sym_decorrelation(w.as_ref()).unwrap();
        let ww_t = &w_dec * w_dec.transpose();

        // Should be close to identity
        assert!((ww_t[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((ww_t[(1, 1)] - 1.0).abs() < 1e-10);
        assert!(ww_t[(0, 1)].abs() < 1e-10);
        assert!(ww_t[(1, 0)].abs() < 1e-10);
    }

    #[test]
    fn test_matrix_exp_identity() {
        let zero = Mat::<f64>::zeros(3, 3);
        let exp_zero = matrix_exp(zero.as_ref());

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((exp_zero[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_determinant() {
        let m = mat![[1.0, 2.0], [3.0, 4.0]];
        let det = determinant(m.as_ref());
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_skew_symmetric() {
        let a = mat![[1.0, 2.0], [3.0, 4.0]];
        let skew = skew_symmetric(a.as_ref());

        // Check A = -A^T
        assert!((skew[(0, 0)]).abs() < 1e-10);
        assert!((skew[(1, 1)]).abs() < 1e-10);
        assert!((skew[(0, 1)] + skew[(1, 0)]).abs() < 1e-10);
    }
}
