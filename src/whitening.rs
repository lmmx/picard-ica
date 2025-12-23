// src/whitening.rs

//! Data preprocessing: centering and whitening.

use crate::error::{PicardError, Result};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SVD;

/// Result of whitening transformation.
pub struct WhiteningResult {
    /// Whitened data matrix.
    pub data: Array2<f64>,
    /// Whitening matrix K (n_components × n_features).
    pub whitening_matrix: Array2<f64>,
}

/// Center the data by subtracting the mean of each row.
///
/// # Arguments
/// * `x` - Data matrix of shape (n_features, n_samples)
///
/// # Returns
/// * Tuple of (centered_data, mean_vector)
pub fn center(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mean = x.mean_axis(Axis(1)).unwrap();
    let mut centered = x.clone();

    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            centered[[i, j]] -= mean[i];
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
pub fn whiten(x: &Array2<f64>, n_components: usize) -> Result<WhiteningResult> {
    let (n_features, n_samples) = (x.nrows(), x.ncols());

    if n_components > n_features {
        return Err(PicardError::InvalidDimensions {
            message: format!(
                "n_components ({}) cannot exceed n_features ({})",
                n_components, n_features
            ),
        });
    }

    // SVD decomposition
    let (u, s, _) = x.svd(true, false).map_err(|e| PicardError::ComputationError {
        message: format!("SVD failed: {}", e),
    })?;

    let u = u.ok_or_else(|| PicardError::ComputationError {
        message: "SVD did not return U matrix".into(),
    })?;

    // Check for near-zero singular values
    let min_sv = s.iter().take(n_components).cloned().fold(f64::INFINITY, f64::min);
    if min_sv < 1e-10 {
        return Err(PicardError::SingularMatrix);
    }

    // Construct whitening matrix K = (U / s)^T[:n_components]
    // Scaled by sqrt(n_samples) for unit variance
    let scale = (n_samples as f64).sqrt();
    let mut k = Array2::zeros((n_components, n_features));

    for i in 0..n_components {
        for j in 0..n_features {
            k[[i, j]] = u[[j, i]] / s[i] * scale;
        }
    }

    // Enforce fixed sign for reproducibility (match MATLAB convention)
    for i in 0..k.nrows() {
        let max_idx = k
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if k[[i, max_idx]] < 0.0 {
            for j in 0..k.ncols() {
                k[[i, j]] = -k[[i, j]];
            }
        }
    }

    // Apply whitening
    let whitened = k.dot(x);

    Ok(WhiteningResult {
        data: whitened,
        whitening_matrix: k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_center() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let (centered, mean) = center(&x);

        assert!((mean[0] - 2.0).abs() < 1e-10);
        assert!((mean[1] - 5.0).abs() < 1e-10);

        // Centered data should have zero mean
        let new_mean = centered.mean_axis(Axis(1)).unwrap();
        assert!(new_mean[0].abs() < 1e-10);
        assert!(new_mean[1].abs() < 1e-10);
    }

    #[test]
    fn test_whiten() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 3.0, 2.0, 4.0]
        ];
        let (centered, _) = center(&x);
        let result = whiten(&centered, 2).unwrap();

        assert_eq!(result.data.nrows(), 2);
        assert_eq!(result.data.ncols(), 4);
        assert_eq!(result.whitening_matrix.shape(), &[2, 3]);
    }
}
