// src/density.rs

//! Density functions for ICA.
//!
//! The density function determines the non-linearity used in the ICA algorithm.
//! Different densities are suited for different source distributions.

use ndarray::{Array1, Array2};

/// Trait for density functions used in ICA.
///
/// A density must provide methods for computing the log-likelihood,
/// score function (derivative of log-likelihood), and score derivative.
pub trait Density: Clone + Send + Sync {
    /// Compute the log-likelihood for a 1D signal.
    fn log_lik(&self, y: &Array1<f64>) -> Array1<f64>;

    /// Compute the score function and its derivative for a 2D signal matrix.
    ///
    /// Returns `(score, score_derivative)` where both have the same shape as input.
    fn score_and_der(&self, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>);
}

/// Hyperbolic tangent density.
///
/// This is the default and most commonly used density. It works well for
/// super-Gaussian sources (e.g., speech, sparse signals).
///
/// The log-likelihood is: `|y| + log(1 + exp(-2α|y|)) / α`
#[derive(Clone, Debug)]
pub struct Tanh {
    /// Scaling parameter (default: 1.0).
    pub alpha: f64,
}

impl Default for Tanh {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Tanh {
    /// Create a new Tanh density with the given alpha parameter.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Density for Tanh {
    fn log_lik(&self, y: &Array1<f64>) -> Array1<f64> {
        let alpha = self.alpha;
        y.mapv(|v| {
            let abs_y = v.abs();
            abs_y + (1.0 + (-2.0 * alpha * abs_y).exp()).ln() / alpha
        })
    }

    fn score_and_der(&self, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let alpha = self.alpha;
        let score = y.mapv(|v| (alpha * v).tanh());
        let score_der = score.mapv(|s| alpha * (1.0 - s * s));
        (score, score_der)
    }
}

/// Exponential density.
///
/// Suited for super-Gaussian sources with heavy tails.
///
/// The log-likelihood is: `-exp(-αy²/2) / α`
#[derive(Clone, Debug)]
pub struct Exp {
    /// Scaling parameter (default: 1.0).
    pub alpha: f64,
}

impl Default for Exp {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Exp {
    /// Create a new Exp density with the given alpha parameter.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Density for Exp {
    fn log_lik(&self, y: &Array1<f64>) -> Array1<f64> {
        let a = self.alpha;
        y.mapv(|v| -(-a * v * v / 2.0).exp() / a)
    }

    fn score_and_der(&self, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let a = self.alpha;
        let y_sq = y.mapv(|v| v * v);
        let k = y_sq.mapv(|v| (-a / 2.0 * v).exp());
        let score = y * &k;
        let score_der = (1.0 - a * &y_sq) * k;
        (score, score_der)
    }
}

/// Cubic density.
///
/// Suited for sub-Gaussian sources (e.g., uniform distributions).
///
/// The log-likelihood is: `y⁴/4`
#[derive(Clone, Debug, Default)]
pub struct Cube;

impl Cube {
    /// Create a new Cube density.
    pub fn new() -> Self {
        Self
    }
}

impl Density for Cube {
    fn log_lik(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|v| v.powi(4) / 4.0)
    }

    fn score_and_der(&self, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let score = y.mapv(|v| v.powi(3));
        let score_der = y.mapv(|v| 3.0 * v * v);
        (score, score_der)
    }
}

/// Enumeration of built-in density types.
///
/// This allows specifying a density without type parameters.
#[derive(Clone, Debug)]
pub enum DensityType {
    /// Hyperbolic tangent density.
    Tanh(Tanh),
    /// Exponential density.
    Exp(Exp),
    /// Cubic density.
    Cube(Cube),
}

impl Default for DensityType {
    fn default() -> Self {
        DensityType::Tanh(Tanh::default())
    }
}

impl DensityType {
    /// Create a Tanh density with default parameters.
    pub fn tanh() -> Self {
        DensityType::Tanh(Tanh::default())
    }

    /// Create a Tanh density with custom alpha.
    pub fn tanh_with_alpha(alpha: f64) -> Self {
        DensityType::Tanh(Tanh::new(alpha))
    }

    /// Create an Exp density with default parameters.
    pub fn exp() -> Self {
        DensityType::Exp(Exp::default())
    }

    /// Create an Exp density with custom alpha.
    pub fn exp_with_alpha(alpha: f64) -> Self {
        DensityType::Exp(Exp::new(alpha))
    }

    /// Create a Cube density.
    pub fn cube() -> Self {
        DensityType::Cube(Cube::new())
    }

    /// Compute the log-likelihood.
    pub fn log_lik(&self, y: &Array1<f64>) -> Array1<f64> {
        match self {
            DensityType::Tanh(d) => d.log_lik(y),
            DensityType::Exp(d) => d.log_lik(y),
            DensityType::Cube(d) => d.log_lik(y),
        }
    }

    /// Compute the score function and its derivative.
    pub fn score_and_der(&self, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        match self {
            DensityType::Tanh(d) => d.score_and_der(y),
            DensityType::Exp(d) => d.score_and_der(y),
            DensityType::Cube(d) => d.score_and_der(y),
        }
    }
}