// src/math.rs

//! Mathematical utilities for Picard ICA.

use ndarray::{Array1, Array2, Zip};

// ============================================================================
// Score Functions
// ============================================================================

/// Score function ψ(y) = tanh(y/2) for logcosh density (super-Gaussian).
#[inline]
pub fn score_tanh(y: &Array2<f64>) -> Array2<f64> {
    y.mapv(|v| (v * 0.5).tanh())
}

/// Derivative of tanh score: ψ'(y) = 0.5 * sech²(y/2).
#[inline]
pub fn score_tanh_derivative(y: &Array2<f64>) -> Array2<f64> {
    y.mapv(|v| {
        let t = (v * 0.5).tanh();
        0.5 * (1.0 - t * t)
    })
}

// ============================================================================
// Extended Mode (Adaptive Score Functions)
// ============================================================================

/// Compute kurtosis sign for each row.
///
/// Returns +1.0 for super-Gaussian (positive excess kurtosis),
/// -1.0 for sub-Gaussian (negative excess kurtosis).
pub fn compute_kurtosis_signs(y: &Array2<f64>) -> Array1<f64> {
    let n = y.nrows();
    let t = y.ncols() as f64;
    let mut signs = Array1::zeros(n);

    for i in 0..n {
        let row = y.row(i);

        // Compute standardized fourth moment (kurtosis)
        let mean: f64 = row.sum() / t;
        let var: f64 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / t;
        let std = var.sqrt().max(1e-10);

        let kurt: f64 = row.iter().map(|&x| ((x - mean) / std).powi(4)).sum::<f64>() / t;

        // Excess kurtosis = kurt - 3
        // Positive => super-Gaussian, negative => sub-Gaussian
        signs[i] = if kurt > 3.0 { 1.0 } else { -1.0 };
    }

    signs
}

/// Compute score function with adaptive selection based on kurtosis signs.
pub fn score_extended(y: &Array2<f64>, signs: &Array1<f64>) -> Array2<f64> {
    let (n, t) = (y.nrows(), y.ncols());
    let mut result = Array2::zeros((n, t));

    Zip::from(result.rows_mut())
        .and(y.rows())
        .and(signs)
        .for_each(|mut out, row, &sign| {
            if sign > 0.0 {
                // Super-Gaussian: tanh(y/2)
                Zip::from(&mut out)
                    .and(&row)
                    .for_each(|o, &v| *o = (v * 0.5).tanh());
            } else {
                // Sub-Gaussian: y - tanh(y)
                Zip::from(&mut out)
                    .and(&row)
                    .for_each(|o, &v| *o = v - v.tanh());
            }
        });

    result
}

/// Compute score derivative with adaptive selection based on kurtosis signs.
pub fn score_derivative_extended(y: &Array2<f64>, signs: &Array1<f64>) -> Array2<f64> {
    let (n, t) = (y.nrows(), y.ncols());
    let mut result = Array2::zeros((n, t));

    Zip::from(result.rows_mut())
        .and(y.rows())
        .and(signs)
        .for_each(|mut out, row, &sign| {
            if sign > 0.0 {
                // d/dy tanh(y/2) = 0.5 * sech²(y/2)
                Zip::from(&mut out).and(&row).for_each(|o, &v| {
                    let th = (v * 0.5).tanh();
                    *o = 0.5 * (1.0 - th * th);
                });
            } else {
                // d/dy (y - tanh(y)) = 1 - sech²(y) = tanh²(y)
                Zip::from(&mut out).and(&row).for_each(|o, &v| {
                    let th = v.tanh();
                    *o = th * th;
                });
            }
        });

    result
}

// ============================================================================
// Gradient and Hessian
// ============================================================================

/// Compute relative gradient: G = (1/T) * ψ(Y) * Y^T - I.
pub fn relative_gradient(y: &Array2<f64>, psi: &Array2<f64>) -> Array2<f64> {
    let t = y.ncols() as f64;
    let n = y.nrows();

    let mut g = psi.dot(&y.t());
    g /= t;

    // Subtract identity
    for i in 0..n {
        g[[i, i]] -= 1.0;
    }

    g
}

/// Compute Hessian approximation H̃₂.
///
/// h_ij = E[ψ'_i(y_i) * y_j²]
pub fn hessian_approx(y: &Array2<f64>, psi_prime: &Array2<f64>) -> Array2<f64> {
    let (n, t) = (y.nrows(), y.ncols());
    let t_f64 = t as f64;

    // Precompute y² for efficiency
    let y_sq = y.mapv(|v| v * v);

    let mut h = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..t {
                sum += psi_prime[[i, k]] * y_sq[[j, k]];
            }
            h[[i, j]] = sum / t_f64;
        }
    }

    h
}

/// Regularize Hessian to ensure positive definiteness.
///
/// For each 2x2 off-diagonal block, shifts eigenvalues if below `lambda_min`.
pub fn regularize_hessian(h: &mut Array2<f64>, lambda_min: f64) {
    let n = h.nrows();

    // Regularize diagonal elements (1 + h_ii should be >= lambda_min)
    for i in 0..n {
        let diag_val = 1.0 + h[[i, i]];
        if diag_val < lambda_min {
            h[[i, i]] = lambda_min - 1.0;
        }
    }

    // Regularize off-diagonal 2x2 blocks
    for i in 0..n {
        for j in (i + 1)..n {
            let a_ij = h[[i, j]];
            let a_ji = h[[j, i]];

            // Smallest eigenvalue of 2x2 block:
            // λ_min = 0.5 * (a_ij + a_ji - sqrt((a_ij - a_ji)² + 4))
            let diff = a_ij - a_ji;
            let discriminant = diff * diff + 4.0;
            let lambda = 0.5 * (a_ij + a_ji - discriminant.sqrt());

            if lambda < lambda_min {
                let shift = lambda_min - lambda;
                h[[i, j]] += shift;
                h[[j, i]] += shift;
            }
        }
    }
}

/// Apply inverse of block-diagonal Hessian approximation to matrix.
///
/// For off-diagonal: [H̃⁻¹G]_ij = (a_ji * G_ij - G_ji) / (a_ij * a_ji - 1)
/// For diagonal: [H̃⁻¹G]_ii = G_ii / (1 + h_ii)
pub fn apply_hessian_inverse(g: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
    let n = g.nrows();
    let mut result = Array2::zeros((n, n));

    for i in 0..n {
        // Diagonal
        result[[i, i]] = g[[i, i]] / (1.0 + h[[i, i]]);

        // Off-diagonal blocks
        for j in (i + 1)..n {
            let a_ij = h[[i, j]];
            let a_ji = h[[j, i]];
            let denom = a_ij * a_ji - 1.0;

            if denom.abs() < 1e-10 {
                // Fallback if near-singular
                result[[i, j]] = g[[i, j]];
                result[[j, i]] = g[[j, i]];
            } else {
                result[[i, j]] = (a_ji * g[[i, j]] - g[[j, i]]) / denom;
                result[[j, i]] = (a_ij * g[[j, i]] - g[[i, j]]) / denom;
            }
        }
    }

    result
}

// ============================================================================
// Likelihood
// ============================================================================

/// Compute negative log-likelihood for line search.
pub fn neg_log_likelihood(
    y: &Array2<f64>,
    log_det_w: f64,
    extended: bool,
    signs: Option<&Array1<f64>>,
) -> f64 {
    let (n, t) = (y.nrows(), y.ncols());
    let t_f64 = t as f64;

    let mut log_p_sum = 0.0;

    match (extended, signs) {
        (true, Some(s)) => {
            for i in 0..n {
                let sign = s[i];
                for k in 0..t {
                    let yi = y[[i, k]];
                    if sign > 0.0 {
                        // Super-Gaussian: log p(y) ∝ -log(cosh(y))
                        log_p_sum -= yi.cosh().ln();
                    } else {
                        // Sub-Gaussian: log p(y) ∝ -y²/2 + log(cosh(y))
                        log_p_sum += -yi * yi / 2.0 + yi.cosh().ln();
                    }
                }
            }
        }
        _ => {
            // Standard logcosh (super-Gaussian only)
            for i in 0..n {
                for k in 0..t {
                    log_p_sum -= y[[i, k]].cosh().ln();
                }
            }
        }
    }

    -log_det_w - log_p_sum / t_f64
}

// ============================================================================
// Linear Algebra Utilities
// ============================================================================

/// Frobenius inner product: <A, B> = Tr(A^T B) = Σ_ij A_ij * B_ij.
#[inline]
pub fn frobenius_dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    Zip::from(a).and(b).fold(0.0, |acc, &x, &y| acc + x * y)
}

/// Infinity norm: max_ij |A_ij|.
#[inline]
pub fn inf_norm(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

/// Compute log determinant via LU decomposition.
pub fn log_det(w: &Array2<f64>) -> f64 {
    let n = w.nrows();
    let mut a = w.clone();
    let mut log_det = 0.0;
    let mut sign = 1.0_f64;

    for i in 0..n {
        // Partial pivoting
        let mut max_idx = i;
        let mut max_val = a[[i, i]].abs();
        for k in (i + 1)..n {
            if a[[k, i]].abs() > max_val {
                max_val = a[[k, i]].abs();
                max_idx = k;
            }
        }

        if max_idx != i {
            for j in 0..n {
                a.swap([i, j], [max_idx, j]);
            }
            sign = -sign;
        }

        let pivot = a[[i, i]];
        if pivot.abs() < 1e-15 {
            return f64::NEG_INFINITY;
        }

        log_det += pivot.abs().ln();
        if pivot < 0.0 {
            sign = -sign;
        }

        for k in (i + 1)..n {
            let factor = a[[k, i]] / pivot;
            for j in i..n {
                a[[k, j]] -= factor * a[[i, j]];
            }
        }
    }

    log_det
}

/// Symmetric orthogonalization: W = W @ (W @ W^T)^(-1/2).
pub fn symmetric_orthogonalize(w: &Array2<f64>) -> Array2<f64> {
    let wwt = w.dot(&w.t());
    let (eigenvalues, eigenvectors) = symmetric_eigen(&wwt);

    let n = w.nrows();
    let mut d_inv_sqrt = Array2::zeros((n, n));
    for i in 0..n {
        d_inv_sqrt[[i, i]] = 1.0 / eigenvalues[i].max(1e-10).sqrt();
    }

    let inv_sqrt = eigenvectors.dot(&d_inv_sqrt).dot(&eigenvectors.t());
    inv_sqrt.dot(w)
}

/// Symmetric eigendecomposition using Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column vectors.
pub fn symmetric_eigen(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut v = Array2::eye(n);
    let mut d = a.clone();

    let max_sweeps = 50;
    for _ in 0..max_sweeps {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                if d[[i, j]].abs() > max_val {
                    max_val = d[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-14 {
            break;
        }

        // Compute Jacobi rotation angle
        let theta = if (d[[q, q]] - d[[p, p]]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * d[[p, q]] / (d[[p, p]] - d[[q, q]])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to d
        let mut new_d = d.clone();
        for i in 0..n {
            if i != p && i != q {
                new_d[[i, p]] = c * d[[i, p]] + s * d[[i, q]];
                new_d[[p, i]] = new_d[[i, p]];
                new_d[[i, q]] = -s * d[[i, p]] + c * d[[i, q]];
                new_d[[q, i]] = new_d[[i, q]];
            }
        }
        new_d[[p, p]] = c * c * d[[p, p]] + 2.0 * s * c * d[[p, q]] + s * s * d[[q, q]];
        new_d[[q, q]] = s * s * d[[p, p]] - 2.0 * s * c * d[[p, q]] + c * c * d[[q, q]];
        new_d[[p, q]] = 0.0;
        new_d[[q, p]] = 0.0;
        d = new_d;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip + s * viq;
            v[[i, q]] = -s * vip + c * viq;
        }
    }

    let eigenvalues = Array1::from_iter((0..n).map(|i| d[[i, i]]));
    (eigenvalues, v)
}

/// Symmetric eigendecomposition with eigenvalues sorted in descending order.
pub fn symmetric_eigen_sorted(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let (eigenvalues, eigenvectors) = symmetric_eigen(a);
    let n = a.nrows();

    // Sort by decreasing eigenvalue
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

    let sorted_eigenvalues = Array1::from_iter(indices.iter().map(|&i| eigenvalues[i]));

    let mut sorted_eigenvectors = Array2::zeros((n, n));
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for i in 0..n {
            sorted_eigenvectors[[i, new_idx]] = eigenvectors[[i, old_idx]];
        }
    }

    (sorted_eigenvalues, sorted_eigenvectors)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_score_tanh() {
        let y = array![[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]];
        let s = score_tanh(&y);

        assert_abs_diff_eq!(s[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s[[0, 1]], 0.5_f64.tanh(), epsilon = 1e-10);
        assert_abs_diff_eq!(s[[0, 2]], (-0.5_f64).tanh(), epsilon = 1e-10);
    }

    #[test]
    fn test_frobenius_dot() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 0.0], [0.0, 1.0]];
        assert_abs_diff_eq!(frobenius_dot(&a, &b), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inf_norm() {
        let a = array![[1.0, -3.0], [2.0, -1.5]];
        assert_abs_diff_eq!(inf_norm(&a), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_eigen() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let (vals, vecs) = symmetric_eigen(&a);

        // Eigenvalues should sum to trace
        let trace = a[[0, 0]] + a[[1, 1]];
        assert_abs_diff_eq!(vals.sum(), trace, epsilon = 1e-6);

        // Eigenvectors should be orthogonal
        let vtv = vecs.t().dot(&vecs);
        assert_abs_diff_eq!(vtv[[0, 1]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(vtv[[1, 0]], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_log_det() {
        let a = array![[2.0, 0.0], [0.0, 3.0]];
        assert_abs_diff_eq!(log_det(&a), (6.0_f64).ln(), epsilon = 1e-10);

        let b = array![[1.0, 2.0], [3.0, 4.0]];
        // det = 1*4 - 2*3 = -2
        assert_abs_diff_eq!(log_det(&b), (2.0_f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_kurtosis_signs() {
        // Laplace distribution has positive excess kurtosis (super-Gaussian)
        // Uniform distribution has negative excess kurtosis (sub-Gaussian)
        let n = 10000;
        let mut super_gauss = Array2::zeros((1, n));
        let mut sub_gauss = Array2::zeros((1, n));

        let mut state = 42u64;
        for i in 0..n {
            // Laplace via inverse CDF
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            super_gauss[[0, i]] = if u < 0.5 {
                (2.0 * u).ln()
            } else {
                -(2.0 * (1.0 - u)).ln()
            };

            // Uniform
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            sub_gauss[[0, i]] = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        }

        let super_signs = compute_kurtosis_signs(&super_gauss);
        let sub_signs = compute_kurtosis_signs(&sub_gauss);

        assert!(super_signs[0] > 0.0, "Laplace should be super-Gaussian");
        assert!(sub_signs[0] < 0.0, "Uniform should be sub-Gaussian");
    }
}
