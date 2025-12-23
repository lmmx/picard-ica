//! Error types for Picard ICA.

use core::fmt;

/// Errors that can occur during Picard ICA fitting.
#[derive(Debug, Clone)]
pub enum PicardError {
    /// Not enough samples for the requested number of components.
    InsufficientSamples {
        n_samples: usize,
        n_components: usize,
    },

    /// Not enough features for the requested number of components.
    InsufficientFeatures {
        n_features: usize,
        n_components: usize,
    },

    /// Input matrix has invalid dimensions.
    InvalidDimensions { message: String },

    /// Numerical error during computation.
    NumericalError { message: String },

    /// Algorithm did not converge within the maximum number of iterations.
    ConvergenceError { n_iter: usize, gradient_norm: f64 },

    /// Error during whitening/PCA step.
    WhiteningError { message: String },
}

impl fmt::Display for PicardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSamples { n_samples, n_components } => {
                write!(
                    f,
                    "insufficient samples: need more than {} samples, got {}",
                    n_components, n_samples
                )
            }
            Self::InsufficientFeatures { n_features, n_components } => {
                write!(
                    f,
                    "insufficient features: need at least {} features, got {}",
                    n_components, n_features
                )
            }
            Self::InvalidDimensions { message } => {
                write!(f, "invalid input dimensions: {}", message)
            }
            Self::NumericalError { message } => {
                write!(f, "numerical error: {}", message)
            }
            Self::ConvergenceError { n_iter, gradient_norm } => {
                write!(
                    f,
                    "failed to converge after {} iterations (gradient norm: {:.2e})",
                    n_iter, gradient_norm
                )
            }
            Self::WhiteningError { message } => {
                write!(f, "whitening failed: {}", message)
            }
        }
    }
}

impl std::error::Error for PicardError {}

impl PicardError {
    pub(crate) fn insufficient_samples(n_samples: usize, n_components: usize) -> Self {
        Self::InsufficientSamples {
            n_samples,
            n_components,
        }
    }

    pub(crate) fn insufficient_features(n_features: usize, n_components: usize) -> Self {
        Self::InsufficientFeatures {
            n_features,
            n_components,
        }
    }

    pub(crate) fn invalid_dimensions(message: impl Into<String>) -> Self {
        Self::InvalidDimensions {
            message: message.into(),
        }
    }

    pub(crate) fn whitening(message: impl Into<String>) -> Self {
        Self::WhiteningError {
            message: message.into(),
        }
    }
}
