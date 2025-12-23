//! Result types for the PICARD algorithm.

use faer::{Col, Mat};

/// Result of running the PICARD algorithm.
#[derive(Debug, Clone)]
pub struct PicardResult {
    /// Whitening matrix K (n_components × n_features).
    /// None if whitening was disabled.
    pub whitening: Option<Mat<f64>>,

    /// Unmixing matrix W (n_components × n_components).
    pub unmixing: Mat<f64>,

    /// Estimated independent sources (n_components × n_samples).
    pub sources: Mat<f64>,

    /// Mean of input features (n_features,).
    /// None if centering was disabled.
    pub mean: Option<Col<f64>>,

    /// Number of iterations performed.
    pub n_iterations: usize,

    /// Whether the algorithm converged.
    pub converged: bool,

    /// Final gradient norm.
    pub gradient_norm: f64,

    /// Signs for extended ICA (indicates sub/super-Gaussian for each component).
    pub signs: Option<Col<f64>>,
}

impl PicardResult {
    /// Get the full unmixing matrix that transforms original data to sources.
    ///
    /// This is `W @ K` if whitening was used, otherwise just `W`.
    pub fn full_unmixing(&self) -> Mat<f64> {
        match &self.whitening {
            Some(k) => &self.unmixing * k,
            None => self.unmixing.clone(),
        }
    }

    /// Get the mixing matrix (pseudo-inverse of full unmixing).
    ///
    /// This transforms sources back to the original feature space.
    pub fn mixing(&self) -> Mat<f64> {
        let full_w = self.full_unmixing();
        // Compute pseudo-inverse using (W^T W)^{-1} W^T
        let wt = full_w.transpose();
        let wtw = &wt * &full_w;

        // Simple matrix inverse for square case
        if wtw.nrows() == wtw.ncols() {
            if let Ok(inv) = invert_matrix(&wtw) {
                return &inv * wt;
            }
        }

        // Fallback: return transpose (valid for orthogonal W)
        wt.to_owned()
    }
}

/// Simple matrix inversion for small matrices.
fn invert_matrix(m: &Mat<f64>) -> Result<Mat<f64>, ()> {
    let n = m.nrows();
    if n != m.ncols() {
        return Err(());
    }

    // Gauss-Jordan elimination
    let mut aug = Mat::zeros(n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = m[(i, j)];
        }
        aug[(i, n + i)] = 1.0;
    }

    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[(k, i)].abs() > aug[(max_row, i)].abs() {
                max_row = k;
            }
        }

        // Swap rows
        for j in 0..(2 * n) {
            let tmp = aug[(i, j)];
            aug[(i, j)] = aug[(max_row, j)];
            aug[(max_row, j)] = tmp;
        }

        if aug[(i, i)].abs() < 1e-15 {
            return Err(());
        }

        // Scale pivot row
        let pivot = aug[(i, i)];
        for j in 0..(2 * n) {
            aug[(i, j)] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[(k, i)];
                for j in 0..(2 * n) {
                    aug[(k, j)] -= factor * aug[(i, j)];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inv[(i, j)] = aug[(i, n + j)];
        }
    }

    Ok(inv)
}
