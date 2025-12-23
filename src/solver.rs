//! Main Picard ICA solver.

use ndarray::{Array1, Array2};

use crate::config::{PicardBuilder, PicardConfig};
use crate::error::PicardError;
use crate::lbfgs::LBFGSMemory;
use crate::math::{
    apply_hessian_inverse, compute_kurtosis_signs, frobenius_dot, hessian_approx, inf_norm,
    log_det, neg_log_likelihood, regularize_hessian, relative_gradient, score_derivative_extended,
    score_extended, score_tanh, score_tanh_derivative, symmetric_orthogonalize,
};
use crate::whitening::{whiten, WhiteningResult};

/// Result of Picard ICA decomposition.
#[derive(Debug, Clone)]
pub struct PicardResult {
    /// Unmixing matrix W (operates on whitened data).
    unmixing: Array2<f64>,
    /// Whitening matrix K.
    whitening: Array2<f64>,
    /// Mean of original data.
    mean: Array1<f64>,
    /// Estimated independent sources (n_components x n_samples).
    sources: Array2<f64>,
    /// Number of iterations until convergence.
    n_iter: usize,
    /// Final gradient infinity norm.
    final_gradient_norm: f64,
    /// Whether the algorithm converged.
    converged: bool,
    /// Explained variance ratio from whitening.
    explained_variance_ratio: Array1<f64>,
}

impl PicardResult {
    /// Get the unmixing matrix W (operates on whitened data).
    ///
    /// For whitened data: S = W @ X_white
    pub fn unmixing_matrix(&self) -> &Array2<f64> {
        &self.unmixing
    }

    /// Get the whitening matrix K.
    pub fn whitening_matrix(&self) -> &Array2<f64> {
        &self.whitening
    }

    /// Get the mean of the original data.
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    /// Get the estimated independent sources.
    pub fn sources(&self) -> &Array2<f64> {
        &self.sources
    }

    /// Get the number of iterations.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Get the final gradient norm.
    pub fn final_gradient_norm(&self) -> f64 {
        self.final_gradient_norm
    }

    /// Check if the algorithm converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Get explained variance ratio from whitening.
    pub fn explained_variance_ratio(&self) -> &Array1<f64> {
        &self.explained_variance_ratio
    }

    /// Get the full unmixing matrix that applies to centered data.
    ///
    /// Returns W_full such that S = W_full @ (X - mean)
    pub fn full_unmixing_matrix(&self) -> Array2<f64> {
        self.unmixing.dot(&self.whitening)
    }

    /// Transform new data using the fitted model.
    ///
    /// # Arguments
    /// * `x` - New data matrix (n_features x n_samples)
    ///
    /// # Returns
    /// Transformed data (n_components x n_samples)
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let (n_features, n_samples) = (x.nrows(), x.ncols());

        // Center using stored mean
        let mut x_centered = x.clone();
        for i in 0..n_features {
            for j in 0..n_samples {
                x_centered[[i, j]] -= self.mean[i];
            }
        }

        // Apply full unmixing: W @ K @ (X - mean)
        self.unmixing.dot(&self.whitening.dot(&x_centered))
    }
}

/// Picard ICA algorithm.
pub struct Picard;

impl Picard {
    /// Fit ICA model with default configuration.
    ///
    /// # Arguments
    /// * `x` - Data matrix of shape (n_features, n_samples)
    /// * `n_components` - Number of independent components to extract
    ///
    /// # Returns
    /// Fitted model result.
    pub fn fit(x: &Array2<f64>, n_components: usize) -> Result<PicardResult, PicardError> {
        Self::fit_with_config(x, PicardConfig::new(n_components))
    }

    /// Create a builder for configuring the algorithm.
    pub fn builder(n_components: usize) -> PicardBuilder {
        PicardBuilder::new(n_components)
    }

    /// Fit ICA model with custom configuration.
    ///
    /// # Arguments
    /// * `x` - Data matrix of shape (n_features, n_samples)
    /// * `config` - Algorithm configuration
    ///
    /// # Returns
    /// Fitted model result.
    pub fn fit_with_config(
        x: &Array2<f64>,
        config: PicardConfig,
    ) -> Result<PicardResult, PicardError> {
        config.validate()?;

        let (n_features, n_samples) = (x.nrows(), x.ncols());
        let n_components = config.n_components.min(n_features);

        // Validate dimensions
        if n_samples <= n_components {
            return Err(PicardError::insufficient_samples(n_samples, n_components));
        }
        if n_features < n_components {
            return Err(PicardError::insufficient_features(n_features, n_components));
        }

        // Whiten the data
        let WhiteningResult {
            data: x_white,
            whitening_matrix: k,
            mean,
            explained_variance_ratio,
        } = whiten(x, n_components)?;

        // Initialize unmixing matrix W
        let mut w = initialize_unmixing(n_components, config.random_seed);
        if config.ortho {
            w = symmetric_orthogonalize(&w);
        }

        // Current sources
        let mut y = w.dot(&x_white);

        // L-BFGS memory
        let mut memory = LBFGSMemory::new(config.memory_size);

        // Kurtosis signs for extended mode
        let mut signs: Option<Array1<f64>> = None;

        // Previous gradient for L-BFGS updates
        let mut prev_gradient: Option<Array2<f64>> = None;

        let mut n_iter = 0;
        let mut final_g_norm = f64::INFINITY;
        let mut converged = false;

        for iter in 0..config.max_iter {
            n_iter = iter + 1;

            // Update kurtosis signs periodically for extended mode
            if config.extended && (iter % 10 == 0 || signs.is_none()) {
                signs = Some(compute_kurtosis_signs(&y));
            }

            // Compute score function
            let psi = if config.extended {
                score_extended(&y, signs.as_ref().unwrap())
            } else {
                score_tanh(&y)
            };

            // Compute gradient
            let g = relative_gradient(&y, &psi);
            let g_norm = inf_norm(&g);
            final_g_norm = g_norm;

            if config.verbose && (iter % 10 == 0 || iter == 0) {
                eprintln!(
                    "[Picard] iter {:4}: gradient norm = {:.6e}",
                    iter, g_norm
                );
            }

            // Check convergence
            if g_norm < config.tol {
                converged = true;
                if config.verbose {
                    eprintln!("[Picard] Converged at iteration {}", iter);
                }
                break;
            }

            // Compute score derivative
            let psi_prime = if config.extended {
                score_derivative_extended(&y, signs.as_ref().unwrap())
            } else {
                score_tanh_derivative(&y)
            };

            // Compute and regularize Hessian approximation
            let mut h = hessian_approx(&y, &psi_prime);
            regularize_hessian(&mut h, config.lambda_min);

            // Apply Hessian inverse to gradient
            let h_inv_g = apply_hessian_inverse(&g, &h);

            // Compute search direction via L-BFGS
            let p = memory.compute_direction(&g, &h_inv_g);

            // Line search
            let (w_new, y_new, step_taken, ls_success) = line_search(
                &w,
                &x_white,
                &p,
                &g,
                config.extended,
                signs.as_ref(),
                config.max_line_search,
            );

            if ls_success {
                // Update L-BFGS memory
                if let Some(ref prev_g) = prev_gradient {
                    let s = &p * step_taken;
                    let y_diff = &g - prev_g;
                    memory.push(s, y_diff);
                }

                w = w_new;
                y = y_new;
            } else {
                // Line search failed - reset and take small gradient step
                if config.verbose {
                    eprintln!(
                        "[Picard] Line search failed at iter {}, resetting L-BFGS",
                        iter
                    );
                }
                memory.clear();

                let step = 0.1;
                let mut w_new = Array2::eye(n_components);
                for i in 0..n_components {
                    for j in 0..n_components {
                        w_new[[i, j]] -= step * h_inv_g[[i, j]];
                    }
                }
                w = w_new.dot(&w);
                y = w.dot(&x_white);
            }

            // Apply orthogonalization if requested
            if config.ortho {
                w = symmetric_orthogonalize(&w);
                y = w.dot(&x_white);
            }

            prev_gradient = Some(g);
        }

        if config.verbose && !converged {
            eprintln!(
                "[Picard] Did not converge after {} iterations (gradient norm: {:.2e})",
                n_iter, final_g_norm
            );
        }

        // Compute final sources using the same transform logic
        let sources = compute_sources(x, &w, &k, &mean);

        Ok(PicardResult {
            unmixing: w,
            whitening: k,
            mean,
            sources,
            n_iter,
            final_gradient_norm: final_g_norm,
            converged,
            explained_variance_ratio,
        })
    }
}

/// Compute sources from original data using unmixing and whitening matrices.
fn compute_sources(
    x: &Array2<f64>,
    w: &Array2<f64>,
    k: &Array2<f64>,
    mean: &Array1<f64>,
) -> Array2<f64> {
    let (n_features, n_samples) = (x.nrows(), x.ncols());

    // Center the data
    let mut x_centered = x.clone();
    for i in 0..n_features {
        for j in 0..n_samples {
            x_centered[[i, j]] -= mean[i];
        }
    }

    // Apply full unmixing: W @ K @ (X - mean)
    w.dot(&k.dot(&x_centered))
}

/// Initialize unmixing matrix.
fn initialize_unmixing(n: usize, seed: Option<u64>) -> Array2<f64> {
    match seed {
        Some(s) => {
            // Deterministic initialization with small random perturbation
            let mut w = Array2::eye(n);
            let mut state = s;

            for i in 0..n {
                for j in 0..n {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.01;
                    w[[i, j]] += noise;
                }
            }
            w
        }
        None => Array2::eye(n),
    }
}

/// Perform backtracking line search with Armijo condition.
fn line_search(
    w: &Array2<f64>,
    x_white: &Array2<f64>,
    direction: &Array2<f64>,
    gradient: &Array2<f64>,
    extended: bool,
    signs: Option<&Array1<f64>>,
    max_iter: usize,
) -> (Array2<f64>, Array2<f64>, f64, bool) {
    let n = w.nrows();
    let y = w.dot(x_white);
    let log_det_w = log_det(w);
    let current_loss = neg_log_likelihood(&y, log_det_w, extended, signs);
    let descent = frobenius_dot(gradient, direction);

    let c1 = 1e-4; // Armijo constant
    let mut alpha = 1.0;

    for _ in 0..max_iter {
        // W_new = (I + α*p) @ W
        let mut update = Array2::eye(n);
        for i in 0..n {
            for j in 0..n {
                update[[i, j]] += alpha * direction[[i, j]];
            }
        }
        let w_new = update.dot(w);
        let y_new = w_new.dot(x_white);

        let log_det_new = log_det(&w_new);
        let new_loss = neg_log_likelihood(&y_new, log_det_new, extended, signs);

        // Armijo condition
        if new_loss < current_loss + c1 * alpha * descent {
            return (w_new, y_new, alpha, true);
        }

        alpha *= 0.5;
    }

    // Return current state if line search failed
    (w.clone(), y, 0.0, false)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic ICA test data.
    fn generate_ica_data(
        n_sources: usize,
        n_samples: usize,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut sources = Array2::zeros((n_sources, n_samples));
        let mut state = seed;

        // Generate independent Laplace sources (super-Gaussian)
        for i in 0..n_sources {
            for j in 0..n_samples {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (state >> 33) as f64 / (1u64 << 31) as f64;
                sources[[i, j]] = if u < 0.5 {
                    (2.0 * u).ln()
                } else {
                    -(2.0 * (1.0 - u)).ln()
                };
            }
        }

        // Random mixing matrix
        let mut mixing = Array2::zeros((n_sources, n_sources));
        for i in 0..n_sources {
            for j in 0..n_sources {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                mixing[[i, j]] = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            }
        }

        let mixed = mixing.dot(&sources);

        (sources, mixing, mixed)
    }

    #[test]
    fn test_picard_basic() {
        let (_sources, _mixing, mixed) = generate_ica_data(3, 1000, 42);

        let result = Picard::fit(&mixed, 3).unwrap();

        assert!(result.converged(), "Algorithm should converge");
        assert!(result.n_iter() < 100, "Should converge in reasonable iterations");
        assert_eq!(result.sources().nrows(), 3);
        assert_eq!(result.sources().ncols(), 1000);
    }

    #[test]
    fn test_picard_builder() {
        let (_sources, _mixing, mixed) = generate_ica_data(3, 1000, 42);

        let result = Picard::builder(3)
            .max_iter(100)
            .tol(1e-6)
            .random_seed(42)
            .extended(true)
            .fit(&mixed)
            .unwrap();

        assert!(result.converged());
    }

    #[test]
    fn test_picard_fewer_components() {
        let (_, _, mixed) = generate_ica_data(5, 1000, 42);

        let result = Picard::fit(&mixed, 3).unwrap();

        assert_eq!(result.sources().nrows(), 3);
        assert_eq!(result.sources().ncols(), 1000);
    }

    #[test]
    fn test_picard_transform() {
        let (_, _, mixed) = generate_ica_data(3, 1000, 42);

        let result = Picard::fit(&mixed, 3).unwrap();

        // Transform should give same result as sources for training data
        let transformed = result.transform(&mixed);

        assert_eq!(transformed.nrows(), 3);
        assert_eq!(transformed.ncols(), 1000);

        // Values should be very close (may have small numerical differences)
        let max_diff: f64 = transformed
            .iter()
            .zip(result.sources().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < 1e-10,
            "Transform should match sources, max diff: {}",
            max_diff
        );
    }

    #[test]
    fn test_picard_insufficient_samples() {
        let (_, _, mixed) = generate_ica_data(3, 2, 42);

        let result = Picard::fit(&mixed, 3);
        assert!(matches!(result, Err(PicardError::InsufficientSamples { .. })));
    }

    #[test]
    fn test_picard_deterministic_without_seed() {
        // Without a seed but with ortho=true and no random init noise,
        // results should be deterministic
        let (_, _, mixed) = generate_ica_data(3, 1000, 42);

        let result1 = Picard::builder(3)
            .ortho(true)
            .extended(false) // Disable extended to reduce variability
            .fit(&mixed)
            .unwrap();

        let result2 = Picard::builder(3)
            .ortho(true)
            .extended(false)
            .fit(&mixed)
            .unwrap();

        // Should get identical iteration counts at minimum
        assert_eq!(result1.n_iter(), result2.n_iter());
    }
}
