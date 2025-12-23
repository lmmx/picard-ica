// src/core.rs

//! Core PICARD algorithm implementation.

use crate::density::DensityType;
use crate::lbfgs::{compute_direction, regularize_hessian, LbfgsMemory};
use crate::math::{determinant, matrix_exp, skew_symmetric};
use faer::{Col, Mat, MatRef};

/// Information returned from core PICARD iteration.
pub struct CoreInfo {
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Final gradient norm.
    pub gradient_norm: f64,
    /// Number of iterations performed.
    pub n_iterations: usize,
    /// Signs for extended ICA (sub/super-Gaussian).
    pub signs: Option<Col<f64>>,
}

/// Compute the loss function.
pub fn compute_loss(
    y: MatRef<'_, f64>,
    w: MatRef<'_, f64>,
    density: &DensityType,
    signs: &Col<f64>,
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
        let row = Col::from_fn(y.ncols(), |j| y[(i, j)]);
        let log_lik = density.log_lik(&row);
        let log_lik_sum: f64 = (0..log_lik.nrows()).map(|j| log_lik[j]).sum();
        loss += signs[i] * log_lik_sum / t;

        if extended && !ortho {
            let sq_sum: f64 = (0..row.nrows()).map(|j| row[j] * row[j]).sum();
            loss += 0.5 * sq_sum / t;
        }
    }

    loss
}

/// Perform backtracking line search.
pub fn line_search(
    y: MatRef<'_, f64>,
    w: MatRef<'_, f64>,
    density: &DensityType,
    direction: MatRef<'_, f64>,
    signs: &Col<f64>,
    current_loss: f64,
    ls_tries: usize,
    ortho: bool,
    extended: bool,
) -> LineSearchResult {
    let n = w.nrows();
    let mut alpha = 1.0;

    let mut y_new = y.to_owned();
    let mut w_new = w.to_owned();
    let mut new_loss = current_loss;

    for _ in 0..ls_tries {
        let transform = if ortho {
            matrix_exp((&direction * faer::Scale(alpha)).as_ref())
        } else {
            &Mat::<f64>::identity(n, n) + &direction * faer::Scale(alpha)
        };

        y_new = &transform * y;
        w_new = &transform * w;
        new_loss = compute_loss(y_new.as_ref(), w_new.as_ref(), density, signs, ortho, extended);

        if new_loss < current_loss {
            return LineSearchResult {
                success: true,
                y: y_new,
                w: w_new,
                loss: new_loss,
                step: &direction * faer::Scale(alpha),
            };
        }

        alpha /= 2.0;
    }

    LineSearchResult {
        success: false,
        y: y_new,
        w: w_new,
        loss: new_loss,
        step: &direction * faer::Scale(alpha),
    }
}

/// Result of line search.
pub struct LineSearchResult {
    pub success: bool,
    pub y: Mat<f64>,
    pub w: Mat<f64>,
    pub loss: f64,
    pub step: Mat<f64>,
}

/// Run the core PICARD algorithm.
pub fn run(
    x: MatRef<'_, f64>,
    density: &DensityType,
    ortho: bool,
    extended: bool,
    m: usize,
    max_iter: usize,
    tol: f64,
    lambda_min: f64,
    ls_tries: usize,
    verbose: bool,
    covariance: Option<MatRef<'_, f64>>,
) -> (Mat<f64>, Mat<f64>, CoreInfo) {
    let (n, t) = (x.nrows(), x.ncols());
    let t_f = t as f64;

    let mut w = Mat::<f64>::identity(n, n);
    let mut y = x.to_owned();

    let mut memory = LbfgsMemory::new(m);
    let mut signs = Col::from_fn(n, |_| 1.0);
    let mut old_signs = signs.clone();

    let mut current_loss = compute_loss(y.as_ref(), w.as_ref(), density, &signs, ortho, extended);
    let mut gradient_norm: f64 = 1.0;
    let mut converged = false;

    let mut c = if extended {
        covariance
            .map(|cov| cov.to_owned())
            .unwrap_or_else(|| &y * y.transpose() * faer::Scale(1.0 / t_f))
    } else {
        Mat::<f64>::identity(n, n)
    };

    let mut g_old: Option<Mat<f64>> = None;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter;

        // Compute score function and derivative
        let (psi_y, mut psi_dy) = density.score_and_der(y.as_ref());

        // Compute relative gradient: G = E[ψ(Y)Y^T]
        let mut g = &psi_y * y.transpose() * faer::Scale(1.0 / t_f);

        // Squared signals for Hessian
        let y_square = Mat::from_fn(y.nrows(), y.ncols(), |i, j| {
            let v = y[(i, j)];
            v * v
        });

        // Handle extended ICA
        let mut sign_change = false;
        if extended {
            // Compute mean along axis 1 (columns)
            let mut psi_dy_mean = Col::zeros(n);
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..t {
                    sum += psi_dy[(i, j)];
                }
                psi_dy_mean[i] = sum / t_f;
            }

            let c_diag = Col::from_fn(n, |i| c[(i, i)]);
            let g_diag = Col::from_fn(n, |i| g[(i, i)]);

            // Kurtosis-based sign detection
            let k = Col::from_fn(n, |i| psi_dy_mean[i] * c_diag[i] - g_diag[i]);
            signs = Col::from_fn(n, |i| k[i].signum());

            if iter > 0 {
                sign_change = (0..n).any(|i| signs[i] != old_signs[i]);
            }
            old_signs = signs.clone();

            // Apply signs to gradient and score derivative
            for i in 0..n {
                for j in 0..g.ncols() {
                    g[(i, j)] *= signs[i];
                }
                for j in 0..psi_dy.ncols() {
                    psi_dy[(i, j)] *= signs[i];
                }
            }

            if !ortho {
                g = &g + &c;
                for j in 0..psi_dy.ncols() {
                    for i in 0..psi_dy.nrows() {
                        psi_dy[(i, j)] += 1.0;
                    }
                }
            }
        }

        // Compute Hessian components
        let h_off = if ortho {
            Col::from_fn(n, |i| g[(i, i)])
        } else {
            Col::from_fn(n, |_| 1.0)
        };

        // Compute and regularize Hessian approximation
        let h = if ortho {
            // Compute mean along axis 1
            let mut psi_dy_mean = Col::zeros(n);
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..t {
                    sum += psi_dy[(i, j)];
                }
                psi_dy_mean[i] = sum / t_f;
            }

            let mut h = Mat::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    h[(i, j)] = 0.5 * (psi_dy_mean[i] + psi_dy_mean[j] - h_off[i] - h_off[j]);
                    h[(i, j)] = h[(i, j)].max(lambda_min);
                }
            }
            h
        } else {
            let mut h = &psi_dy * y_square.transpose() * faer::Scale(1.0 / t_f);
            regularize_hessian(&mut h, &h_off, lambda_min);
            h
        };

        // Project gradient
        if ortho {
            g = skew_symmetric(g.as_ref());
        } else {
            for i in 0..n {
                g[(i, i)] -= 1.0;
            }
        }

        // Check convergence
        gradient_norm = 0.0;
        for j in 0..g.ncols() {
            for i in 0..g.nrows() {
                gradient_norm = gradient_norm.max(g[(i, j)].abs());
            }
        }
        if gradient_norm < tol {
            converged = true;
            break;
        }

        // Update L-BFGS memory
        if let Some(ref g_prev) = g_old {
            let y_diff = &g - g_prev;
            if let Some(last_step) = memory.s_list.last() {
                // Already have the step stored, just add gradient diff
                let mut r_sum = 0.0;
                for j in 0..last_step.ncols() {
                    for i in 0..last_step.nrows() {
                        r_sum += last_step[(i, j)] * y_diff[(i, j)];
                    }
                }
                let r = 1.0 / r_sum;
                if r.is_finite() {
                    memory.y_list.push(y_diff.clone());
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
            current_loss = compute_loss(y.as_ref(), w.as_ref(), density, &signs, ortho, extended);
            memory.clear();
        }

        // Compute search direction
        let direction = compute_direction(g.as_ref(), h.as_ref(), &h_off, &memory, ortho);

        // Line search
        let result = line_search(
            y.as_ref(),
            w.as_ref(),
            density,
            direction.as_ref(),
            &signs,
            current_loss,
            ls_tries,
            ortho,
            extended,
        );

        let (new_y, new_w, new_loss, step) = if !result.success {
            // Fall back to gradient descent
            memory.clear();
            let neg_g = &g * faer::Scale(-1.0);
            let fallback = line_search(
                y.as_ref(),
                w.as_ref(),
                density,
                neg_g.as_ref(),
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
                c = &w * cov * w.transpose();
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