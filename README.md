# Picard

**P**reconditioned **I**CA for **R**eal **D**ata — A fast and robust Independent Component Analysis implementation in Rust.

Based on the paper:
> Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort.
> "Faster independent component analysis by preconditioning with Hessian approximations"
> IEEE Transactions on Signal Processing, 2018
> https://arxiv.org/abs/1706.08171

## Features

- **Fast convergence**: Uses L-BFGS with Hessian preconditioning for faster convergence than FastICA
- **Robust**: Works well on real data where ICA assumptions don't perfectly hold
- **Orthogonal mode (Picard-O)**: Enforces orthogonal unmixing matrix for whitened data
- **Extended mode**: Handles both sub-Gaussian and super-Gaussian sources automatically
- **Multiple density functions**: Tanh (default), Exp, and Cube for different source distributions
- **Comprehensive output**: Access to unmixing matrices, sources, convergence info, and more

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
picard = "0.1"
ndarray = "0.15"
```

You'll also need a BLAS backend for `ndarray-linalg`. For example, with OpenBLAS:
```toml
[dependencies]
ndarray-linalg = { version = "0.16", features = ["openblas-system"] }
```

## Quick Start
```rust
use picard::{Picard, PicardConfig, DensityType};
use ndarray::Array2;

// Your data matrix: (n_features × n_samples)
let x: Array2<f64> = /* ... */;

// Fit ICA with default settings
let result = Picard::fit(&x)?;

// Access results
let sources = &result.sources;        // Estimated sources (n_components × n_samples)
let unmixing = &result.unmixing;      // Unmixing matrix W
let converged = result.converged;     // Whether algorithm converged
let n_iter = result.n_iterations;     // Number of iterations

// Get the full unmixing matrix (W @ K if whitening was used)
let full_w = result.full_unmixing();

// Get the mixing matrix (pseudo-inverse of full unmixing)
let mixing = result.mixing();
```

## Configuration

Use the builder pattern for fine-grained control:
```rust
use picard::{Picard, PicardConfig, DensityType};

let config = PicardConfig::builder()
    .n_components(10)           // Number of components to extract
    .ortho(true)                // Use orthogonal constraint (Picard-O)
    .extended(true)             // Handle sub and super-Gaussian sources
    .whiten(true)               // Perform PCA whitening
    .centering(true)            // Center the data
    .density(DensityType::tanh()) // Density function
    .max_iter(500)              // Maximum iterations
    .tol(1e-7)                  // Convergence tolerance
    .m(7)                       // L-BFGS memory size
    .random_state(42)           // For reproducibility
    .verbose(true)              // Print progress
    .build();

let result = Picard::fit_with_config(&x, &config)?;
```

## Density Functions

Choose the density function based on your source distributions:
```rust
use picard::DensityType;

// Tanh (default) - good for super-Gaussian sources (e.g., speech, sparse signals)
let density = DensityType::tanh();
let density = DensityType::tanh_with_alpha(1.0);  // Custom alpha

// Exp - for heavy-tailed super-Gaussian sources
let density = DensityType::exp();
let density = DensityType::exp_with_alpha(0.1);

// Cube - for sub-Gaussian sources (e.g., uniform distributions)
let density = DensityType::cube();
```

## Transforming New Data
```rust
// Fit on training data
let result = Picard::fit(&x_train)?;

// Transform new data using the fitted model
let sources_new = Picard::transform(&x_new, &result)?;
```

## Evaluating Separation Quality

When you know the true mixing matrix (e.g., in simulations):
```rust
use picard::utils::{amari_distance, permute};

// Amari distance: 0 = perfect separation
let distance = amari_distance(&result.full_unmixing(), &true_mixing);

// Permute W @ A to visualize how close it is to identity
let wa = result.full_unmixing().dot(&true_mixing);
let permuted = permute(&wa, true);  // Should be close to identity
```

## Algorithm

Picard combines two key ideas:

1. **Sparse Hessian approximations** for ICA that are cheap to compute and invert
2. **L-BFGS optimization** that refines these approximations using gradient history

This yields an algorithm that:
- Has the same cost per iteration as simple quasi-Newton methods
- Achieves much better convergence on real data
- Is typically 2-5× faster than FastICA

### Picard vs Picard-O

- **Picard** (`ortho: false`): Standard ICA, minimizes mutual information
- **Picard-O** (`ortho: true`): Adds orthogonality constraint, faster for whitened data

### Extended Mode

When `extended: true`, the algorithm automatically detects and handles both:
- **Super-Gaussian** sources (positive kurtosis): speech, sparse signals
- **Sub-Gaussian** sources (negative kurtosis): uniform distributions

## Output Structure
```rust
pub struct PicardResult {
    pub whitening: Option<Array2<f64>>,   // Whitening matrix K (if whiten=true)
    pub unmixing: Array2<f64>,            // Unmixing matrix W
    pub sources: Array2<f64>,             // Estimated sources S
    pub mean: Option<Array1<f64>>,        // Data mean (if centering=true)
    pub n_iterations: usize,              // Iterations performed
    pub converged: bool,                  // Convergence status
    pub gradient_norm: f64,               // Final gradient norm
    pub signs: Option<Array1<f64>>,       // Component signs (if extended=true)
}
```

## Error Handling
```rust
use picard::{Picard, PicardError};

match Picard::fit(&x) {
    Ok(result) => {
        if !result.converged {
            eprintln!("Warning: did not converge, gradient norm: {:.2e}", 
                      result.gradient_norm);
        }
        // Use result...
    }
    Err(PicardError::NotConverged { gradient_norm, tolerance, iterations }) => {
        eprintln!("Failed after {} iterations", iterations);
    }
    Err(PicardError::InvalidDimensions { message }) => {
        eprintln!("Bad input: {}", message);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## License

BSD-3-Clause, matching the original Python implementation.