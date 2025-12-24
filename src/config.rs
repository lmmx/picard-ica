// src/config.rs

//! Configuration for the PICARD algorithm.

use crate::density::DensityType;
use crate::error::{PicardError, Result};
use ndarray::Array2;

/// Configuration parameters for the PICARD algorithm.
#[derive(Clone)]
pub struct PicardConfig {
    /// Density function to use for ICA.
    pub density: DensityType,

    /// Number of components to extract. If None, uses min(n_features, n_samples).
    pub n_components: Option<usize>,

    /// If true, uses Picard-O with orthogonal constraint.
    pub ortho: bool,

    /// If true, uses extended algorithm for sub/super-Gaussian sources.
    /// Defaults to same value as `ortho` if not specified.
    pub extended: Option<bool>,

    /// If true, perform whitening on the data.
    pub whiten: bool,

    /// If true, center the data before processing.
    pub centering: bool,

    /// Maximum number of iterations.
    pub max_iter: usize,

    /// Convergence tolerance for gradient norm.
    pub tol: f64,

    /// Size of L-BFGS memory.
    pub m: usize,

    /// Maximum line search attempts.
    pub ls_tries: usize,

    /// Minimum eigenvalue for Hessian regularization.
    pub lambda_min: f64,

    /// Initial unmixing matrix. If None, uses random initialization.
    pub w_init: Option<Array2<f64>>,

    /// Number of FastICA iterations before PICARD. If None, skip FastICA.
    pub fastica_it: Option<usize>,

    /// Number of JADE iterations before PICARD. If None, skip JADE.
    /// JADE (Joint Approximate Diagonalization of Eigenmatrices) can provide
    /// a better warm start than FastICA for some data distributions.
    pub jade_it: Option<usize>,

    /// Random seed for reproducibility.
    pub random_state: Option<u64>,

    /// If true, print progress information.
    pub verbose: bool,
}

impl Default for PicardConfig {
    fn default() -> Self {
        Self {
            density: DensityType::default(),
            n_components: None,
            ortho: true,
            extended: None,
            whiten: true,
            centering: true,
            max_iter: 500,
            tol: 1e-7,
            m: 7,
            ls_tries: 10,
            lambda_min: 0.01,
            w_init: None,
            fastica_it: None,
            jade_it: None,
            random_state: None,
            verbose: false,
        }
    }
}

impl PicardConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for constructing a configuration.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::new()
    }

    /// Get the effective value of `extended` (defaults to `ortho` if not set).
    pub fn effective_extended(&self) -> bool {
        self.extended.unwrap_or(self.ortho)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_iter == 0 {
            return Err(PicardError::InvalidConfig {
                parameter: "max_iter".into(),
                message: "must be greater than 0".into(),
            });
        }

        if self.tol <= 0.0 {
            return Err(PicardError::InvalidConfig {
                parameter: "tol".into(),
                message: "must be positive".into(),
            });
        }

        if self.lambda_min <= 0.0 {
            return Err(PicardError::InvalidConfig {
                parameter: "lambda_min".into(),
                message: "must be positive".into(),
            });
        }

        if self.m == 0 {
            return Err(PicardError::InvalidConfig {
                parameter: "m".into(),
                message: "L-BFGS memory size must be at least 1".into(),
            });
        }

        if self.fastica_it.is_some() && self.jade_it.is_some() {
            return Err(PicardError::InvalidConfig {
                parameter: "jade_it".into(),
                message: "cannot use both fastica_it and jade_it; choose one warm start method"
                    .into(),
            });
        }

        Ok(())
    }
}

/// Builder for constructing `PicardConfig` with a fluent API.
#[derive(Default)]
pub struct ConfigBuilder {
    config: PicardConfig,
}

impl ConfigBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            config: PicardConfig::default(),
        }
    }

    /// Set the density function.
    pub fn density(mut self, density: DensityType) -> Self {
        self.config.density = density;
        self
    }

    /// Set the number of components to extract.
    pub fn n_components(mut self, n: usize) -> Self {
        self.config.n_components = Some(n);
        self
    }

    /// Enable or disable orthogonal constraint (Picard-O).
    pub fn ortho(mut self, ortho: bool) -> Self {
        self.config.ortho = ortho;
        self
    }

    /// Enable or disable extended algorithm for mixed sub/super-Gaussian sources.
    pub fn extended(mut self, extended: bool) -> Self {
        self.config.extended = Some(extended);
        self
    }

    /// Enable or disable whitening.
    pub fn whiten(mut self, whiten: bool) -> Self {
        self.config.whiten = whiten;
        self
    }

    /// Enable or disable centering.
    pub fn centering(mut self, centering: bool) -> Self {
        self.config.centering = centering;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the L-BFGS memory size.
    pub fn m(mut self, m: usize) -> Self {
        self.config.m = m;
        self
    }

    /// Set the maximum line search attempts.
    pub fn ls_tries(mut self, ls_tries: usize) -> Self {
        self.config.ls_tries = ls_tries;
        self
    }

    /// Set the minimum eigenvalue for Hessian regularization.
    pub fn lambda_min(mut self, lambda_min: f64) -> Self {
        self.config.lambda_min = lambda_min;
        self
    }

    /// Set the initial unmixing matrix.
    pub fn w_init(mut self, w_init: Array2<f64>) -> Self {
        self.config.w_init = Some(w_init);
        self
    }

    /// Set the number of FastICA pre-iterations.
    ///
    /// Note: Cannot be used together with `jade_it`.
    pub fn fastica_it(mut self, iterations: usize) -> Self {
        self.config.fastica_it = Some(iterations);
        self
    }

    /// Set the number of JADE pre-iterations.
    ///
    /// JADE (Joint Approximate Diagonalization of Eigenmatrices) uses
    /// fourth-order cumulants and Jacobi rotations for joint diagonalization.
    /// It can provide a better warm start than FastICA for some distributions.
    ///
    /// Note: Cannot be used together with `fastica_it`.
    pub fn jade_it(mut self, iterations: usize) -> Self {
        self.config.jade_it = Some(iterations);
        self
    }

    /// Set the random seed.
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Enable or disable verbose output.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> PicardConfig {
        self.config
    }

    /// Build and validate the configuration.
    pub fn build_validated(self) -> Result<PicardConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}