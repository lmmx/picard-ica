// src/error.rs

//! Error types for the Picard crate.

use std::fmt;

/// Errors that can occur during PICARD computation.
#[derive(Debug, Clone)]
pub enum PicardError {
    /// Algorithm did not converge within the maximum number of iterations.
    NotConverged {
        /// Final gradient norm achieved.
        gradient_norm: f64,
        /// Requested tolerance.
        tolerance: f64,
        /// Number of iterations performed.
        iterations: usize,
    },

    /// Input dimensions are invalid.
    InvalidDimensions {
        /// Description of the dimension error.
        message: String,
    },

    /// A singular matrix was encountered during computation.
    SingularMatrix,

    /// General computation error.
    ComputationError {
        /// Description of what went wrong.
        message: String,
    },

    /// Invalid configuration parameter.
    InvalidConfig {
        /// Name of the invalid parameter.
        parameter: String,
        /// Description of why it's invalid.
        message: String,
    },
}

impl fmt::Display for PicardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PicardError::NotConverged {
                gradient_norm,
                tolerance,
                iterations,
            } => {
                write!(
                    f,
                    "PICARD did not converge after {} iterations. \
                     Final gradient norm: {:.4e}, requested tolerance: {:.4e}. \
                     Consider increasing max_iter or tolerance.",
                    iterations, gradient_norm, tolerance
                )
            }
            PicardError::InvalidDimensions { message } => {
                write!(f, "Invalid dimensions: {}", message)
            }
            PicardError::SingularMatrix => {
                write!(f, "Singular matrix encountered during computation")
            }
            PicardError::ComputationError { message } => {
                write!(f, "Computation error: {}", message)
            }
            PicardError::InvalidConfig { parameter, message } => {
                write!(f, "Invalid configuration for '{}': {}", parameter, message)
            }
        }
    }
}

impl std::error::Error for PicardError {}

// Implement From for faer's error types
impl From<faer::linalg::svd::SvdError> for PicardError {
    fn from(e: faer::linalg::svd::SvdError) -> Self {
        PicardError::ComputationError {
            message: format!("SVD failed: {:?}", e),
        }
    }
}

impl From<faer::linalg::evd::EvdError> for PicardError {
    fn from(e: faer::linalg::evd::EvdError) -> Self {
        PicardError::ComputationError {
            message: format!("Eigendecomposition failed: {:?}", e),
        }
    }
}

/// Convenience type alias for Results with PicardError.
pub type Result<T> = std::result::Result<T, PicardError>;