//! Data whitening (sphering) via PCA.

use ndarray::{Array1, Array2, Axis};

use crate::error::PicardError;
use crate::math::symmetric_eigen_sorted;

/// Result of whitening transformation.
#[derive(Debug, Clone)]
pub struct WhiteningResult {
    /// Whitened data (n_components x n_samples).
    pub data: Array2<f64>,
    /// Whitening matrix K such that X_white = K @ (X - mean).
    pub whitening_matrix: Array2<f64>,
    /// Mean of original data (n_features).
    pub mean: Array1<f64>,
    /// Explained variance ratio for each component.
    pub explained_variance_ratio: Array1<f64>,
}

/// Whiten data using PCA.
///
/// # Arguments
/// * `x` - Data matrix of shape (n_features, n_samples)
/// * `n_components` - Number of components to keep
///
/// # Returns
/// Whitened data and transformation parameters.
pub fn whiten(x: &Array2<f64>, n_components: usize) -> Result<WhiteningResult, PicardError> {
    let (n_features, n_samples) = (x.nrows(), x.ncols());

    if n_samples == 0 {
        return Err(PicardError::invalid_dimensions("Input has no samples"));
    }

    if n_features == 0 {
        return Err(PicardError::invalid_dimensions("Input has no features"));
    }

    let n_components = n_components.min(n_features).min(n_samples);

    // Center the data
    let mean = x.mean_axis(Axis(1)).unwrap();
    let mut x_centered = x.clone();
    for i in 0..n_features {
        for j in 0..n_samples {
            x_centered[[i, j]] -= mean[i];
        }
    }

    // Compute covariance matrix
    let cov = x_centered.dot(&x_centered.t()) / (n_samples as f64);

    // Eigendecomposition of covariance (sorted by decreasing eigenvalue)
    let (eigenvalues, eigenvectors) = symmetric_eigen_sorted(&cov);

    // Check for numerical issues
    let total_var: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();
    if total_var < 1e-10 {
        return Err(PicardError::whitening(
            "Data has zero or near-zero variance",
        ));
    }

    // Check if we have enough positive eigenvalues
    let n_positive = eigenvalues.iter().filter(|&&v| v > 1e-10).count();
    if n_positive < n_components {
        return Err(PicardError::whitening(format!(
            "Only {} positive eigenvalues, but {} components requested",
            n_positive, n_components
        )));
    }

    // Build whitening matrix: K = D^(-1/2) @ U^T for top n_components
    let mut whitening_matrix = Array2::zeros((n_components, n_features));
    for i in 0..n_components {
        let scale = 1.0 / eigenvalues[i].sqrt();
        for j in 0..n_features {
            whitening_matrix[[i, j]] = eigenvectors[[j, i]] * scale;
        }
    }

    // Compute explained variance ratio
    let mut explained_variance_ratio = Array1::zeros(n_components);
    for i in 0..n_components {
        explained_variance_ratio[i] = eigenvalues[i] / total_var;
    }

    // Apply whitening
    let data = whitening_matrix.dot(&x_centered);

    Ok(WhiteningResult {
        data,
        whitening_matrix,
        mean,
        explained_variance_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn generate_random_data(n_features: usize, n_samples: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::zeros((n_features, n_samples));
        let mut state = seed;

        for i in 0..n_features {
            for j in 0..n_samples {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                data[[i, j]] = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            }
        }

        data
    }

    #[test]
    fn test_whiten_identity_covariance() {
        let x = generate_random_data(5, 1000, 42);
        let result = whiten(&x, 5).unwrap();

        // Whitened data should have approximately identity covariance
        let cov = result.data.dot(&result.data.t()) / 1000.0;

        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(cov[[i, j]], expected, epsilon = 0.15);
            }
        }
    }

    #[test]
    fn test_whiten_explained_variance() {
        let x = generate_random_data(5, 1000, 42);
        let result = whiten(&x, 5).unwrap();

        // Explained variance ratios should sum to approximately 1
        let total: f64 = result.explained_variance_ratio.sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 0.01);

        // Should be in descending order
        for i in 1..5 {
            assert!(result.explained_variance_ratio[i] <= result.explained_variance_ratio[i - 1]);
        }
    }

    #[test]
    fn test_whiten_fewer_components() {
        let x = generate_random_data(10, 1000, 42);
        let result = whiten(&x, 3).unwrap();

        assert_eq!(result.data.nrows(), 3);
        assert_eq!(result.data.ncols(), 1000);
        assert_eq!(result.whitening_matrix.nrows(), 3);
        assert_eq!(result.whitening_matrix.ncols(), 10);
    }
}
