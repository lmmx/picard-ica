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
//! ```rust,no_run
//! use picard::{Picard, PicardConfig};
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), picard::PicardError> {
//! // Generate some test data (n_features x n_samples)
//! let x = Array2::<f64>::zeros((10, 1000));
//!
//! // Fit ICA with default settings
//! let result = Picard::fit(&x)?;
//!
//! // Or with custom configuration
//! let config = PicardConfig::builder()
//!     .n_components(5)
//!     .max_iter(200)
//!     .ortho(true)
//!     .build();
//! let result = Picard::fit_with_config(&x, &config)?;
//!
//! // Access results
//! let sources = &result.sources;
//! let unmixing = &result.unmixing;
//! # Ok(())
//! # }
//! ```

mod config;
mod core;
mod density;
mod error;
mod lbfgs;
mod math;
mod result;
mod solver;
mod whitening;

pub use config::{ConfigBuilder, PicardConfig};
pub use density::{Cube, Density, DensityType, Exp, Tanh};
pub use error::PicardError;
pub use result::PicardResult;
pub use solver::Picard;

// Utility functions
pub mod utils;

// Re-export ndarray for convenience
pub use ndarray;