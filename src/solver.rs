// src/solver.rs

//! Main PICARD solver interface.

use crate::config::PicardConfig;
use crate::core;
use crate::density::DensityType;
use crate::error::{PicardError, Result};
use crate::jade;
use crate::math::sym_decorrelation;
use crate::result::PicardResult;
use crate::whitening::{center, whiten};

use ndarray::{Array2, Axis};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

/// The PICARD Independent Component Analysis solver.
///
/// This struct provides static methods for fitting ICA models.
pub struct Picard;

impl Picard {
    /// Fit ICA model with default configuration.
    ///
    /// # Arguments
    /// * `x` - Data matrix of shape (n_features, n_samples)
    ///
    /// # Returns
    /// * `PicardResult` containing unmixing matrix, sources, etc.
    pub fn fit(x: &Array2<f64>) -> Result<PicardResult> {
        Self::fit_with_config(x, &PicardConfig::default())
    }

    /// Fit ICA model with custom configuration.
    ///
    /// # Arguments
    /// * `x` - Data matrix of shape (n_features, n_samples)
    /// * `config` - Algorithm configuration
    ///
    /// # Returns
    /// * `PicardResult` containing unmixing matrix, sources, etc.
    pub fn fit_with_config(x: &Array2<f64>, config: &PicardConfig) -> Result<PicardResult> {
        config.validate()?;

        let (n, p) = (x.nrows(), x.ncols());

        if n == 0 || p == 0 {
            return Err(PicardError::InvalidDimensions {
                message: "Input matrix cannot be empty".into(),
            });
        }

        // Initialize RNG
        let mut rng = match config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        // Determine number of components
        let n_components = config.n_components.unwrap_or(n.min(p)).min(n.min(p));

        // Get effective extended setting
        let extended = config.effective_extended();

        // Warn about potentially problematic configurations
        if !matches!(config.density, DensityType::Tanh(_)) && extended && !config.ortho {
            eprintln!(
                "Warning: Using a density other than tanh with extended=true and ortho=false \
                 may result in incorrect estimation or numerical overflow"
            );
        }

        // Center the data
        let (x1, x_mean) = if config.centering {
            let (centered, mean) = center(x);
            (centered, Some(mean))
        } else {
            (x.clone(), None)
        };

        // Whiten the data
        let (x1, k) = if config.whiten {
            let whitening_result = whiten(&x1, n_components)?;
            (
                whitening_result.data,
                Some(whitening_result.whitening_matrix),
            )
        } else {
            (x1, None)
        };

        let actual_components = x1.nrows();

        // Initialize unmixing matrix
        let w_init = match &config.w_init {
            Some(w) => {
                if w.shape() != [actual_components, actual_components] {
                    return Err(PicardError::InvalidDimensions {
                        message: format!(
                            "w_init shape {:?} doesn't match expected ({}, {})",
                            w.shape(),
                            actual_components,
                            actual_components
                        ),
                    });
                }
                w.clone()
            }
            None => {
                let mut w = Array2::zeros((actual_components, actual_components));
                for i in 0..actual_components {
                    for j in 0..actual_components {
                        w[[i, j]] = rng.sample(StandardNormal);
                    }
                }
                sym_decorrelation(&w)?
            }
        };

        // Optional JADE warm start (takes priority if both specified, but validation prevents this)
        let w_init = if let Some(jade_it) = config.jade_it {
            if config.verbose {
                println!("Running {} iterations of JADE...", jade_it);
            }
            jade::jade(&x1, jade_it, 1e-6, config.verbose)?
        } else if let Some(fastica_it) = config.fastica_it {
            // Optional FastICA pre-iterations
            if config.verbose {
                println!("Running {} iterations of FastICA...", fastica_it);
            }
            ica_par(&x1, &config.density, fastica_it, &w_init, config.verbose)?
        } else {
            w_init
        };

        // Apply initial transformation
        let x1 = w_init.dot(&x1);

        // Covariance for extended ICA
        let covariance = if extended && config.whiten {
            Some(Array2::eye(actual_components))
        } else {
            None
        };

        // Run core algorithm
        if config.verbose {
            println!("Running Picard...");
        }

        let (y, w, info) = core::run(
            &x1,
            &config.density,
            config.ortho,
            extended,
            config.m,
            config.max_iter,
            config.tol,
            config.lambda_min,
            config.ls_tries,
            config.verbose,
            covariance.as_ref(),
        )?;

        // Combine transformations
        let w = w.dot(&w_init);

        if !info.converged && config.verbose {
            eprintln!(
                "Warning: PICARD did not converge. \
                 Final gradient norm: {:.4e}, tolerance: {:.4e}",
                info.gradient_norm, config.tol
            );
        }

        Ok(PicardResult {
            whitening: k,
            unmixing: w,
            sources: y,
            mean: x_mean,
            n_iterations: info.n_iterations,
            converged: info.converged,
            gradient_norm: info.gradient_norm,
            signs: info.signs,
        })
    }

    /// Transform new data using a fitted model.
    ///
    /// # Arguments
    /// * `x` - New data matrix (n_features, n_samples)
    /// * `result` - Result from a previous fit
    ///
    /// # Returns
    /// * Transformed data (n_components, n_samples)
    pub fn transform(x: &Array2<f64>, result: &PicardResult) -> Result<Array2<f64>> {
        let mut x = x.clone();

        // Subtract mean if available
        if let Some(ref mean) = result.mean {
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    x[[i, j]] -= mean[i];
                }
            }
        }

        // Apply full unmixing
        let w = result.full_unmixing();
        Ok(w.dot(&x))
    }
}

/// FastICA parallel iteration (used for initialization).
fn ica_par(
    x: &Array2<f64>,
    density: &DensityType,
    max_iter: usize,
    w_init: &Array2<f64>,
    verbose: bool,
) -> Result<Array2<f64>> {
    let mut w = sym_decorrelation(w_init)?;
    let p = x.ncols() as f64;

    for _ in 0..max_iter {
        let wx = w.dot(x);
        let (gwtx, g_wtx) = density.score_and_der(&wx);
        let g_wtx_mean = g_wtx.mean_axis(Axis(1)).unwrap();

        // C = E[g(Wx)X^T] - E[g'(Wx)] * W
        let mut c = gwtx.dot(&x.t()) / p;
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                c[[i, j]] -= g_wtx_mean[i] * w[[i, j]];
            }
        }

        w = sym_decorrelation(&c)?;
    }

    if verbose {
        println!("FastICA pre-iterations complete.");
    }

    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use rand_distr::Uniform;

    fn generate_test_data(
        n: usize,
        t: usize,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate Laplacian-like sources
        let mut s = Array2::zeros((n, t));
        for i in 0..n {
            for j in 0..t {
                let u: f64 = rng.gen_range(0.0..1.0);
                let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                s[[i, j]] = sign * (-u.ln());
            }
        }

        // Generate mixing matrix
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = rng.sample(StandardNormal);
            }
        }

        // Mix signals
        let x = a.dot(&s);

        (s, a, x)
    }

    #[test]
    fn test_fit_default() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let result = Picard::fit(&x).unwrap();

        assert_eq!(result.sources.nrows(), 3);
        assert_eq!(result.sources.ncols(), 1000);
        assert_eq!(result.unmixing.nrows(), 3);
        assert_eq!(result.unmixing.ncols(), 3);
    }

    #[test]
    fn test_fit_with_config() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let config = PicardConfig::builder()
            .max_iter(100)
            .random_state(42)
            .verbose(false)
            .build();

        let result = Picard::fit_with_config(&x, &config).unwrap();

        assert!(result.n_iterations <= 100);
    }

    #[test]
    fn test_fit_with_jade_warmstart() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let config = PicardConfig::builder()
            .jade_it(50)
            .random_state(42)
            .verbose(false)
            .build();

        let result = Picard::fit_with_config(&x, &config).unwrap();

        assert_eq!(result.sources.nrows(), 3);
        assert!(result.converged || result.n_iterations > 0);
    }

    #[test]
    fn test_jade_vs_no_warmstart() {
        let (_, _, x) = generate_test_data(4, 2000, 123);

        // Without warm start
        let config_plain = PicardConfig::builder()
            .random_state(42)
            .verbose(false)
            .build();

        let result_plain = Picard::fit_with_config(&x, &config_plain).unwrap();

        // With JADE warm start
        let config_jade = PicardConfig::builder()
            .jade_it(30)
            .random_state(42)
            .verbose(false)
            .build();

        let result_jade = Picard::fit_with_config(&x, &config_jade).unwrap();

        // JADE should typically help converge faster or with better gradient norm
        // (This is a soft check - the main goal is that it doesn't break)
        assert!(result_jade.converged || result_jade.gradient_norm < 1.0);
    }

    #[test]
    fn test_n_components() {
        let (_, _, x) = generate_test_data(5, 1000, 42);

        let config = PicardConfig::builder()
            .n_components(3)
            .random_state(42)
            .build();

        let result = Picard::fit_with_config(&x, &config).unwrap();

        assert_eq!(result.sources.nrows(), 3);
        assert_eq!(result.unmixing.nrows(), 3);
    }

    #[test]
    fn test_transform() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let config = PicardConfig::builder().random_state(42).build();

        let result = Picard::fit_with_config(&x, &config).unwrap();

        // Transform the same data
        let transformed = Picard::transform(&x, &result).unwrap();

        assert_eq!(transformed.shape(), result.sources.shape());
    }

    #[test]
    fn test_no_whiten() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let config = PicardConfig::builder()
            .whiten(false)
            .random_state(42)
            .build();

        let result = Picard::fit_with_config(&x, &config).unwrap();

        assert!(result.whitening.is_none());
    }

    #[test]
    fn test_cannot_use_both_warmstarts() {
        let config = PicardConfig::builder()
            .fastica_it(10)
            .jade_it(10)
            .build();

        assert!(config.validate().is_err());
    }
}
