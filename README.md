# README.md

# Picard

**P**reconditioned **I**CA for **R**eal **D**ata - A fast and robust Independent Component Analysis implementation in Rust.

Based on the paper:
> Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort.
> "Faster independent component analysis by preconditioning with Hessian approximations"
> IEEE Transactions on Signal Processing, 2018
> https://arxiv.org/abs/1706.08171

## Features

- **Fast convergence**: Uses L-BFGS with Hessian preconditioning for faster convergence than FastICA
- **Robust**: Works well on real data where ICA assumptions don't perfectly hold
- **Extended mode**: Handles both sub-Gaussian and super-Gaussian sources
- **No external dependencies**: Pure Rust implementation (no LAPACK required)
- **Well-tested**: Comprehensive test suite with synthetic and edge cases

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
picard = "0.1"
```

## Quick Start
```rust
use picard::{Picard, PicardConfig};
use ndarray::Array2;

// Your data matrix: (n_features x n_samples)
let x: Array2<f64> = /* ... */;

// Fit ICA with default settings
let result = Picard::fit(&x, 10)?; // Extract 10 components

// Access results
let sources = result.sources();           // (n_components x n_samples)
let unmixing = result.unmixing_matrix();  // W such that S = W @ X

// Or use the builder for more control
let result = Picard::builder()
    .n_components(10)
    .max_iter(200)
    .tol(1e-7)
    .extended(true)
    .random_seed(42)
    .fit(&x)?;
```

## Algorithm

Picard combines two key ideas:

1. **Sparse Hessian approximations** for ICA that are cheap to compute and invert
2. **L-BFGS optimization** that refines these approximations using gradient history

This yields an algorithm that:
- Has the same cost per iteration as simple quasi-Newton methods
- Achieves much better convergence on real data
- Is typically 2-5x faster than FastICA

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.