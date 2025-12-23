// src/whitening.rs

//! Data preprocessing: centering and whitening.

use crate::error::{PicardError, Result};
use faer::{Col, Mat, MatRef};
use faer::matrix_free::LinOp;

/// Result of whitening transformation.
pub struct WhiteningResult {
    /// Whitened data matrix.
    pub data: Mat<f64>,
    /// Whitening matrix K (n_components × n_features).
    pub whitening_matrix: Mat<f64>,
}

/// Center the data by subtracting the mean of each row.
///
/// # Arguments
/// * `x` - Data matrix of shape (n_features, n_samples)
///
/// # Returns
/// * Tuple of (centered_data, mean_vector)
pub fn center(x: MatRef<'_, f64>) -> (Mat<f64>, Col<f64>) {
    let (nrows, ncols) = (x.nrows(), x.ncols());

    // Compute row means
    let mut mean = Col::zeros(nrows);
    for i in 0..nrows {
        let mut sum = 0.0;
        for j in 0..ncols {
            sum += x[(i, j)];
        }
        mean[i] = sum / ncols as f64;
    }

    // Center the data
    let mut centered = Mat::zeros(nrows, ncols);
    for j in 0..ncols {
        for i in 0..nrows {
            centered[(i, j)] = x[(i, j)] - mean[i];
        }
    }

    (centered, mean)
}

/// Whiten the data using PCA.
///
/// Whitening transforms the data so that it has unit variance and
/// the components are uncorrelated.
///
/// # Arguments
/// * `x` - Centered data matrix of shape (n_features, n_samples)
/// * `n_components` - Number of components to keep
///
/// # Returns
/// * `WhiteningResult` containing whitened data and whitening matrix
pub fn whiten(x: MatRef<'_, f64>, n_components: usize) -> Result<WhiteningResult> {
    let (n_features, n_samples) = (x.nrows(), x.ncols());

    if n_components > n_features {
        return Err(PicardError::InvalidDimensions {
            message: format!(
                "n_components ({}) cannot exceed n_features ({})",
                n_components, n_features
            ),
        });
    }

    // Thin SVD decomposition
    let svd = x.thin_svd()?;
    let u = svd.U();
    let s = svd.S();

    // s is a diagonal matrix, get the number of singular values
    let num_singular_values = s.ncols().min(s.nrows());

    // Check for near-zero singular values
    let mut min_sv = f64::INFINITY;
    for i in 0..n_components.min(num_singular_values) {
        let sv = s[i];
        if sv < min_sv {
            min_sv = sv;
        }
    }
    if min_sv < 1e-10 {
        return Err(PicardError::SingularMatrix);
    }

    // Construct whitening matrix K = (U / s)^T[:n_components]
    // Scaled by sqrt(n_samples) for unit variance
    let scale = (n_samples as f64).sqrt();
    let mut k = Mat::zeros(n_components, n_features);

    for i in 0..n_components {
        for j in 0..n_features {
            k[(i, j)] = u[(j, i)] / s[i] * scale;
        }
    }

    // Enforce fixed sign for reproducibility (match MATLAB convention)
    for i in 0..k.nrows() {
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for j in 0..k.ncols() {
            if k[(i, j)].abs() > max_val {
                max_val = k[(i, j)].abs();
                max_idx = j;
            }
        }

        if k[(i, max_idx)] < 0.0 {
            for j in 0..k.ncols() {
                k[(i, j)] = -k[(i, j)];
            }
        }
    }

    // Apply whitening
    let whitened = &k * x;

    Ok(WhiteningResult {
        data: whitened,
        whitening_matrix: k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_center() {
        let x = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let (centered, mean) = center(x.as_ref());

        assert!((mean[0] - 2.0).abs() < 1e-10);
        assert!((mean[1] - 5.0).abs() < 1e-10);

        // Centered data should have zero mean
        let mut new_mean_0 = 0.0;
        let mut new_mean_1 = 0.0;
        for j in 0..centered.ncols() {
            new_mean_0 += centered[(0, j)];
            new_mean_1 += centered[(1, j)];
        }
        new_mean_0 /= centered.ncols() as f64;
        new_mean_1 /= centered.ncols() as f64;

        assert!(new_mean_0.abs() < 1e-10);
        assert!(new_mean_1.abs() < 1e-10);
    }

    #[test]
    fn test_whiten() {
        let x = mat![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 3.0, 2.0, 4.0]
        ];
        let (centered, _) = center(x.as_ref());
        let result = whiten(centered.as_ref(), 2).unwrap();

        assert_eq!(result.data.nrows(), 2);
        assert_eq!(result.data.ncols(), 4);
        assert_eq!(result.whitening_matrix.nrows(), 2);
        assert_eq!(result.whitening_matrix.ncols(), 3);
    }
}
