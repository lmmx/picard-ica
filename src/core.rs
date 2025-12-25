//! Core PICARD algorithm implementation.

use crate::density::DensityType;
use crate::error::{PicardError, Result};
use crate::lbfgs::{compute_direction, regularize_hessian, LbfgsMemory};
use crate::math::{matrix_exp, skew_symmetric, sln_det};
use ndarray::{Array1, Array2, Axis};

/// Information returned from core PICARD iteration.
pub struct CoreInfo {
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Final gradient norm.
    pub gradient_norm: f64,
    /// Number of iterations performed.
    pub n_iterations: usize,
    /// Signs for extended ICA (sub/super-Gaussian).
    pub signs: Option<Array1<f64>>,
}

/// Result of computing the loss function.
pub enum LossResult {
    /// Successfully computed loss value.
    Value(f64),
    /// Matrix is singular - optimization should handle this gracefully.
    Singular,
    /// Computation failed with an error.
    Error(PicardError),
}

/// Compute the loss function.
///
/// Uses LAPACK-backed log-determinant computation for numerical stability.
/// This avoids the numerical issues with manual LU decomposition and
/// provides smoother gradients for L-BFGS optimization.
///
/// Returns a `LossResult` to distinguish between normal values, singularity,
/// and actual errors.
pub fn compute_loss(
    y: &Array2<f64>,
    w: &Array2<f64>,
    density: &DensityType,
    signs: &Array1<f64>,
    ortho: bool,
    extended: bool,
) -> LossResult {
    let n = y.nrows();
    let t = y.ncols() as f64;

    // Log-determinant term (only for non-orthogonal)
    let mut loss = if !ortho {
        // Use sln_det for numerical stability - it returns (sign, log|det|)
        // directly, avoiding overflow/underflow issues with large/small determinants
        match sln_det(w) {
            Ok((sign, log_abs_det)) => {
                if sign == 0.0 {
                    // Matrix is singular
                    return LossResult::Singular;
                }
                // Normal case: -log|det(W)|
                -log_abs_det
            }
            Err(e) => {
                // LU decomposition failed entirely
                return LossResult::Error(e);
            }
        }
    } else {
        0.0
    };

    // Density terms
    for i in 0..n {
        let row = y.row(i).to_owned();
        let log_lik = density.log_lik(&row);
        loss += signs[i] * log_lik.sum() / t;

        if extended && !ortho {
            let sq_sum: f64 = row.iter().map(|&x| x * x).sum();
            loss += 0.5 * sq_sum / t;
        }
    }

    LossResult::Value(loss)
}

/// Convert LossResult to f64 for use in line search comparisons.
/// Singular matrices get a large penalty value to push optimization away.
/// Errors also get a large penalty to avoid that direction.
fn loss_to_f64(result: LossResult) -> f64 {
    match result {
        LossResult::Value(v) => v,
        LossResult::Singular => 1e15,
        LossResult::Error(_) => 1e15,
    }
}

/// Perform backtracking line search.
pub fn line_search(
    y: &Array2<f64>,
    w: &Array2<f64>,
    density: &DensityType,
    direction: &Array2<f64>,
    signs: &Array1<f64>,
    current_loss: f64,
    ls_tries: usize,
    ortho: bool,
    extended: bool,
) -> LineSearchResult {
    let n = w.nrows();
    let mut alpha = 1.0;

    let mut y_new = y.clone();
    let mut w_new = w.clone();
    let mut new_loss = current_loss;

    for _ in 0..ls_tries {
        let transform = if ortho {
            matrix_exp(&(direction * alpha))
        } else {
            Array2::eye(n) + alpha * direction
        };

        y_new = transform.dot(y);
        w_new = transform.dot(w);

        let loss_result = compute_loss(&y_new, &w_new, density, signs, ortho, extended);
        new_loss = loss_to_f64(loss_result);

        if new_loss < current_loss {
            return LineSearchResult {
                success: true,
                y: y_new,
                w: w_new,
                loss: new_loss,
                step: direction * alpha,
            };
        }

        alpha /= 2.0;
    }

    LineSearchResult {
        success: false,
        y: y_new,
        w: w_new,
        loss: new_loss,
        step: direction * alpha,
    }
}

/// Result of line search.
pub struct LineSearchResult {
    pub success: bool,
    pub y: Array2<f64>,
    pub w: Array2<f64>,
    pub loss: f64,
    pub step: Array2<f64>,
}

/// Minimum curvature threshold for L-BFGS updates.
/// Updates with s·y below this are rejected to maintain positive definiteness.
const LBFGS_CURVATURE_MIN: f64 = 1e-10;

/// Run the core PICARD algorithm.
pub fn run(
    x: &Array2<f64>,
    density: &DensityType,
    ortho: bool,
    extended: bool,
    m: usize,
    max_iter: usize,
    tol: f64,
    lambda_min: f64,
    ls_tries: usize,
    verbose: bool,
    covariance: Option<&Array2<f64>>,
) -> Result<(Array2<f64>, Array2<f64>, CoreInfo)> {
    let (n, t) = (x.nrows(), x.ncols());
    let t_f = t as f64;

    let mut w = Array2::eye(n);
    let mut y = x.clone();

    let mut memory = LbfgsMemory::new(m);
    let mut signs = Array1::ones(n);
    let mut old_signs = signs.clone();

    let initial_loss = compute_loss(&y, &w, density, &signs, ortho, extended);
    let mut current_loss = match initial_loss {
        LossResult::Value(v) => v,
        LossResult::Singular => {
            return Err(PicardError::SingularMatrix);
        }
        LossResult::Error(e) => {
            return Err(e);
        }
    };

    let mut gradient_norm = 1.0;
    let mut converged = false;

    let mut c = if extended {
        covariance
            .map(|cov| cov.clone())
            .unwrap_or_else(|| y.dot(&y.t()) / t_f)
    } else {
        Array2::eye(n)
    };

    let mut g_old: Option<Array2<f64>> = None;
    let mut prev_step: Option<Array2<f64>> = None;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter;

        // Compute score function and derivative
        let (psi_y, mut psi_dy) = density.score_and_der(&y);

        // Compute relative gradient: G = E[ψ(Y)Y^T]
        let mut g = psi_y.dot(&y.t()) / t_f;

        // Squared signals for Hessian
        let y_square = y.mapv(|v| v * v);

        // Handle extended ICA
        let mut sign_change = false;
        if extended {
            let psi_dy_mean = psi_dy.mean_axis(Axis(1)).unwrap();
            let c_diag = Array1::from_iter((0..n).map(|i| c[[i, i]]));
            let g_diag = Array1::from_iter((0..n).map(|i| g[[i, i]]));

            // Kurtosis-based sign detection
            let k = &psi_dy_mean * &c_diag - &g_diag;
            signs = k.mapv(|v| v.signum());

            if iter > 0 {
                sign_change = signs.iter().zip(old_signs.iter()).any(|(&a, &b)| a != b);
            }
            old_signs.assign(&signs);

            // Apply signs to gradient and score derivative
            for i in 0..n {
                for j in 0..g.ncols() {
                    g[[i, j]] *= signs[i];
                }
                for j in 0..psi_dy.ncols() {
                    psi_dy[[i, j]] *= signs[i];
                }
            }

            if !ortho {
                g = &g + &c;
                psi_dy = &psi_dy + 1.0;
            }
        }

        // Compute Hessian components
        let h_off = if ortho {
            Array1::from_iter((0..n).map(|i| g[[i, i]]))
        } else {
            Array1::ones(n)
        };

        // Compute and regularize Hessian approximation
        let h = if ortho {
            let psi_dy_mean = psi_dy.mean_axis(Axis(1)).unwrap();
            let mut h = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    h[[i, j]] = 0.5 * (psi_dy_mean[i] + psi_dy_mean[j] - h_off[i] - h_off[j]);
                    h[[i, j]] = h[[i, j]].max(lambda_min);
                }
            }
            h
        } else {
            let mut h = psi_dy.dot(&y_square.t()) / t_f;
            regularize_hessian(&mut h, &h_off, lambda_min);
            h
        };

        // Project gradient
        if ortho {
            g = skew_symmetric(&g);
        } else {
            for i in 0..n {
                g[[i, i]] -= 1.0;
            }
        }

        // Check convergence
        gradient_norm = g.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        if gradient_norm < tol {
            converged = true;
            break;
        }

        // L-BFGS memory update
        if iter > 0 {
            if let (Some(step), Some(ref g_prev)) = (prev_step.take(), &g_old) {
                let y_diff = &g - g_prev;
                let sy = (&step * &y_diff).sum();

                // Only accept update if curvature is sufficiently positive
                // This maintains positive definiteness of the inverse Hessian approximation
                if sy > LBFGS_CURVATURE_MIN {
                    let r = 1.0 / sy;
                    memory.s_list.push(step);
                    memory.y_list.push(y_diff);
                    memory.r_list.push(r);
                    // Trim to size m
                    if memory.s_list.len() > m {
                        memory.s_list.remove(0);
                        memory.y_list.remove(0);
                        memory.r_list.remove(0);
                    }
                }
                // If curvature condition fails, skip this update
                // The L-BFGS will use existing memory or fall back to steepest descent
            }
        }
        g_old = Some(g.clone());

        // Flush memory on sign change
        if extended && sign_change {
            let loss_result = compute_loss(&y, &w, density, &signs, ortho, extended);
            current_loss = match loss_result {
                LossResult::Value(v) => v,
                LossResult::Singular => {
                    // During iteration, try to continue rather than fail immediately
                    // The line search should push us away from singularity
                    1e15
                }
                LossResult::Error(e) => {
                    return Err(e);
                }
            };
            memory.clear();
        }

        // Compute search direction
        let direction = compute_direction(&g, &h, &h_off, &memory, ortho);

        // Line search
        let result = line_search(
            &y,
            &w,
            density,
            &direction,
            &signs,
            current_loss,
            ls_tries,
            ortho,
            extended,
        );

        let (new_y, new_w, new_loss, step) = if !result.success {
            // Fall back to gradient descent
            memory.clear();
            let neg_g = -&g;
            let fallback = line_search(
                &y,
                &w,
                density,
                &neg_g,
                &signs,
                current_loss,
                10,
                ortho,
                extended,
            );
            (fallback.y, fallback.w, fallback.loss, fallback.step)
        } else {
            (result.y, result.w, result.loss, result.step)
        };

        // Store step for next iteration's memory update
        prev_step = Some(step);

        y = new_y;
        w = new_w;

        if extended {
            if let Some(cov) = covariance {
                c = w.dot(cov).dot(&w.t());
            }
        }

        current_loss = new_loss;

        if verbose {
            println!(
                "iteration {}, gradient norm = {:.4e}, loss = {:.4e}",
                iter + 1,
                gradient_norm,
                current_loss
            );
        }
    }

    let info = CoreInfo {
        converged,
        gradient_norm,
        n_iterations: n_iter + 1,
        signs: if extended { Some(signs) } else { None },
    };

    Ok((y, w, info))
}
