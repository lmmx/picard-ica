// src/config.rs

//! Configuration for Picard ICA algorithm.

/// Configuration parameters for Picard ICA.
#[derive(Debug, Clone)]
pub struct PicardConfig {
    /// Number of independent components to extract.
    pub n_components: usize,

    /// Maximum number of iterations.
    pub max_iter: usize,

    /// Convergence tolerance (infinity norm of gradient).
    pub tol: f64,

    /// L-BFGS memory size (number of past iterations to store).
    pub memory_size: usize,

    /// Minimum eigenvalue for Hessian regularization.
    pub lambda_min: f64,

    /// Maximum line search iterations per step.
    pub max_line_search: usize,

    /// Use extended mode for sub-Gaussian sources.
    ///
    /// When true, the algorithm adapts the score function based on
    /// the estimated kurtosis of each component.
    pub extended: bool,

    /// Random seed for reproducible initialization.
    pub random_seed: Option<u64>,

    /// Print progress information to stderr.
    pub verbose: bool,

    /// Whether to whiten the data before ICA.
    ///
    /// If false, assumes input is already whitened.
    pub whiten: bool,

    /// Orthogonal constraint (Picard-O variant).
    ///
    /// When true, maintains orthogonality of the unmixing matrix.
    pub ortho: bool,
}

impl Default for PicardConfig {
    fn default() -> Self {
        Self {
            n_components: 10,
            max_iter: 200,
            tol: 1e-7,
            memory_size: 7,
            lambda_min: 1e-2,
            max_line_search: 10,
            extended: true,
            random_seed: None,
            verbose: false,
            whiten: true,
            ortho: true,
        }
    }
}

impl PicardConfig {
    /// Create a new configuration with the specified number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Default::default()
        }
    }

    /// Validate configuration parameters.
    pub(crate) fn validate(&self) -> Result<(), crate::PicardError> {
        if self.n_components == 0 {
            return Err(crate::PicardError::invalid_dimensions(
                "n_components must be at least 1",
            ));
        }
        if self.max_iter == 0 {
            return Err(crate::PicardError::invalid_dimensions(
                "max_iter must be at least 1",
            ));
        }
        if self.tol <= 0.0 {
            return Err(crate::PicardError::invalid_dimensions(
                "tol must be positive",
            ));
        }
        if self.memory_size == 0 {
            return Err(crate::PicardError::invalid_dimensions(
                "memory_size must be at least 1",
            ));
        }
        Ok(())
    }
}

/// Builder for [`PicardConfig`].
///
/// # Example
///
/// ```rust
/// use picard::PicardBuilder;
///
/// let config = PicardBuilder::new(10)
///     .max_iter(100)
///     .tol(1e-6)
///     .extended(true)
///     .random_seed(42)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct PicardBuilder {
    config: PicardConfig,
}

impl PicardBuilder {
    /// Create a new builder with the specified number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            config: PicardConfig::new(n_components),
        }
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
    pub fn memory_size(mut self, memory_size: usize) -> Self {
        self.config.memory_size = memory_size;
        self
    }

    /// Set the minimum eigenvalue for Hessian regularization.
    pub fn lambda_min(mut self, lambda_min: f64) -> Self {
        self.config.lambda_min = lambda_min;
        self
    }

    /// Set the maximum line search iterations.
    pub fn max_line_search(mut self, max_line_search: usize) -> Self {
        self.config.max_line_search = max_line_search;
        self
    }

    /// Enable or disable extended mode for sub-Gaussian sources.
    pub fn extended(mut self, extended: bool) -> Self {
        self.config.extended = extended;
        self
    }

    /// Set a random seed for reproducible results.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Enable or disable verbose output.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Enable or disable whitening.
    pub fn whiten(mut self, whiten: bool) -> Self {
        self.config.whiten = whiten;
        self
    }

    /// Enable or disable orthogonal constraint.
    pub fn ortho(mut self, ortho: bool) -> Self {
        self.config.ortho = ortho;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> PicardConfig {
        self.config
    }

    /// Build the configuration and fit to data.
    ///
    /// This is a convenience method equivalent to calling `build()` followed by
    /// `Picard::fit_with_config()`.
    pub fn fit(self, x: &ndarray::Array2<f64>) -> Result<crate::PicardResult, crate::PicardError> {
        crate::Picard::fit_with_config(x, self.config)
    }
}