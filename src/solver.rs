//! Main PICARD solver interface.

use crate::config::PicardConfig;
use crate::core;
use crate::density::DensityType;
use crate::error::{PicardError, Result};
use crate::math::sym_decorrelation;
use crate::result::PicardResult;
use crate::whitening::{center, whiten};

use faer::{Mat, MatRef};
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
    pub fn fit(x: MatRef<'_, f64>) -> Result<PicardResult> {
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
    pub fn fit_with_config(x: MatRef<'_, f64>, config: &PicardConfig) -> Result<PicardResult> {
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
            (x.to_owned(), None)
        };

        // Whiten the data
        let (x1, k) = if config.whiten {
            let whitening_result = whiten(x1.as_ref(), n_components)?;
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
                if w.nrows() != actual_components || w.ncols() != actual_components {
                    return Err(PicardError::InvalidDimensions {
                        message: format!(
                            "w_init shape ({}, {}) doesn't match expected ({}, {})",
                            w.nrows(),
                            w.ncols(),
                            actual_components,
                            actual_components
                        ),
                    });
                }
                w.clone()
            }
            None => {
                let mut w = Mat::zeros(actual_components, actual_components);
                for j in 0..actual_components {
                    for i in 0..actual_components {
                        w[(i, j)] = rng.sample(StandardNormal);
                    }
                }
                sym_decorrelation(w.as_ref())?
            }
        };

        // Optional FastICA pre-iterations
        let w_init = if let Some(fastica_it) = config.fastica_it {
            if config.verbose {
                println!("Running {} iterations of FastICA...", fastica_it);
            }
            ica_par(x1.as_ref(), &config.density, fastica_it, w_init.as_ref(), config.verbose)?
        } else {
            w_init
        };

        // Apply initial transformation
        let x1 = &w_init * &x1;

        // Covariance for extended ICA
        let covariance = if extended && config.whiten {
            Some(Mat::<f64>::identity(actual_components, actual_components))
        } else {
            None
        };

        // Run core algorithm
        if config.verbose {
            println!("Running Picard...");
        }

        let (y, w, info) = core::run(
            x1.as_ref(),
            &config.density,
            config.ortho,
            extended,
            config.m,
            config.max_iter,
            config.tol,
            config.lambda_min,
            config.ls_tries,
            config.verbose,
            covariance.as_ref().map(|c| c.as_ref()),
        );

        // Combine transformations
        let w = &w * &w_init;

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
    pub fn transform(x: MatRef<'_, f64>, result: &PicardResult) -> Result<Mat<f64>> {
        let mut x = x.to_owned();

        // Subtract mean if available
        if let Some(ref mean) = result.mean {
            for j in 0..x.ncols() {
                for i in 0..x.nrows() {
                    x[(i, j)] -= mean[i];
                }
            }
        }

        // Apply full unmixing
        let w = result.full_unmixing();
        Ok(&w * &x)
    }
}

/// FastICA parallel iteration (used for initialization).
fn ica_par(
    x: MatRef<'_, f64>,
    density: &DensityType,
    max_iter: usize,
    w_init: MatRef<'_, f64>,
    verbose: bool,
) -> Result<Mat<f64>> {
    let mut w = sym_decorrelation(w_init)?;
    let p = x.ncols() as f64;

    for _ in 0..max_iter {
        let wx = &w * x;
        let (gwtx, g_wtx) = density.score_and_der(wx.as_ref());

        // Compute mean along axis 1
        let n = w.nrows();
        let t = g_wtx.ncols();
        let mut g_wtx_mean = faer::Col::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..t {
                sum += g_wtx[(i, j)];
            }
            g_wtx_mean[i] = sum / t as f64;
        }

        // C = E[g(Wx)X^T] - E[g'(Wx)] * W
        let mut c = &gwtx * x.transpose() * faer::Scale(1.0 / p);
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                c[(i, j)] -= g_wtx_mean[i] * w[(i, j)];
            }
        }

        w = sym_decorrelation(c.as_ref())?;
    }

    if verbose {
        println!("FastICA pre-iterations complete.");
    }

    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(
        n: usize,
        t: usize,
        seed: u64,
    ) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate Laplacian-like sources
        let mut s = Mat::zeros(n, t);
        for i in 0..n {
            for j in 0..t {
                let u: f64 = rng.gen_range(0.0..1.0);
                let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                s[(i, j)] = sign * (-u.ln());
            }
        }

        // Generate mixing matrix
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = rng.sample(StandardNormal);
            }
        }

        // Mix signals
        let x = &a * &s;

        (s, a, x)
    }

    #[test]
    fn test_fit_default() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let result = Picard::fit(x.as_ref()).unwrap();

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

        let result = Picard::fit_with_config(x.as_ref(), &config).unwrap();

        assert!(result.n_iterations <= 100);
    }

    #[test]
    fn test_n_components() {
        let (_, _, x) = generate_test_data(5, 1000, 42);

        let config = PicardConfig::builder()
            .n_components(3)
            .random_state(42)
            .build();

        let result = Picard::fit_with_config(x.as_ref(), &config).unwrap();

        assert_eq!(result.sources.nrows(), 3);
        assert_eq!(result.unmixing.nrows(), 3);
    }

    #[test]
    fn test_transform() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let config = PicardConfig::builder().random_state(42).build();

        let result = Picard::fit_with_config(x.as_ref(), &config).unwrap();

        // Transform the same data
        let transformed = Picard::transform(x.as_ref(), &result).unwrap();

        assert_eq!(transformed.nrows(), result.sources.nrows());
        assert_eq!(transformed.ncols(), result.sources.ncols());
    }

    #[test]
    fn test_no_whiten() {
        let (_, _, x) = generate_test_data(3, 1000, 42);

        let config = PicardConfig::builder()
            .whiten(false)
            .random_state(42)
            .build();

        let result = Picard::fit_with_config(x.as_ref(), &config).unwrap();

        assert!(result.whitening.is_none());
    }
}
