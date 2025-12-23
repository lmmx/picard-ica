//! Core PICARD algorithm implementation.

use crate::density::DensityType;
use crate::lbfgs::{compute_direction, regularize_hessian, LbfgsMemory};
use crate::math::{determinant, matrix_exp, skew_symmetric};
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

/// Compute the loss function.
pub fn compute_loss(
    y: &Array2<f64>,
    w: &Array2<f64>,
    density: &DensityType,
    signs: &Array1<f64>,
    ortho: bool,
    extended: bool,
) -> f64 {
    let n = y.nrows();
    let t = y.ncols() as f64;

    // Log-determinant term (only for non-orthogonal)
    let mut loss = if !ortho {
        let det = determinant(w);
        if det.abs() < 1e-15 {
            return f64::MAX;
        }
        -det.abs().ln()
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

    loss
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
        new_loss = compute_loss(&y_new, &w_new, density, signs, ortho, extended);

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
) -> (Array2<f64>, Array2<f64>, CoreInfo) {
    let (n, t) = (x.nrows(), x.ncols());
    let t_f = t as f64;

    let mut w = Array2::eye(n);
    let mut y = x.clone();

    let mut memory = LbfgsMemory::new(m);
    let mut signs = Array1::ones(n);
    let mut old_signs = signs.clone();

    let mut current_loss = compute_loss(&y, &w, density, &signs, ortho, extended);
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

        // Update L-BFGS memory
        if let Some(ref g_prev) = g_old {
            let y_diff = &g - g_prev;
            if let Some(last_step) = memory.s_list.last() {
                // Already have the step stored, just add gradient diff
                memory.y_list.push(y_diff.clone());
                let r = 1.0 / (last_step * &y_diff).sum();
                if r.is_finite() {
                    memory.r_list.push(r);
                } else {
                    // Invalid curvature, pop the step
                    memory.s_list.pop();
                }
            }
        }
        g_old = Some(g.clone());

        // Flush memory on sign change
        if extended && sign_change {
            current_loss = compute_loss(&y, &w, density, &signs, ortho, extended);
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

        // Store step for next L-BFGS update
        memory.s_list.push(step);

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

    (y, w, info)
}
