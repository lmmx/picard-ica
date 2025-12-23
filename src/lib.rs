// src/lib.rs

//! # Picard
//!
//! Fast Independent Component Analysis using preconditioned L-BFGS optimization.
//!
//! This crate implements the Picard algorithm from:
//!
//! > Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort.
//! > "Faster independent component analysis by preconditioning with Hessian approximations"
//! > IEEE Transactions on Signal Processing, 2018
//!
//! ## Example
//!
//! ```rust
//! use picard::Picard;
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), picard::PicardError> {
//! // Generate some test data (n_features x n_samples)
//! let x = Array2::<f64>::zeros((10, 1000));
//!
//! // Fit ICA
//! let result = Picard::fit(&x, 5)?;
//!
//! // Get the independent sources
//! let sources = result.sources();
//! # Ok(())
//! # }
//! ```

mod config;
mod error;
mod lbfgs;
mod math;
mod solver;
mod whitening;

pub use config::{PicardBuilder, PicardConfig};
pub use error::PicardError;
pub use solver::{Picard, PicardResult};

// Re-export ndarray for convenience
pub use ndarray;