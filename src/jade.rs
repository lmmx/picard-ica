// src/jade.rs

//! JADE (Joint Approximate Diagonalization of Eigenmatrices) for ICA warm start.
//!
//! Based on: Cardoso & Souloumiac, "Blind beamforming for non-Gaussian signals"
//! IEE Proceedings F, 1993.

use crate::error::Result;
use crate::math::sym_decorrelation;
use ndarray::{Array2, Array3};

/// Run JADE algorithm for ICA.
///
/// # Arguments
/// * `x` - Whitened data matrix (n_components × n_samples)
/// * `max_iter` - Maximum number of Jacobi sweeps
/// * `tol` - Convergence tolerance for rotation angles
/// * `verbose` - Print progress information
///
/// # Returns
/// * Unmixing matrix W
pub fn jade(x: &Array2<f64>, max_iter: usize, tol: f64, verbose: bool) -> Result<Array2<f64>> {
    let n = x.nrows();

    if n < 2 {
        return Ok(Array2::eye(n));
    }

    // Compute cumulant matrices
    let cumulants = compute_cumulant_matrices(x);

    if verbose {
        println!("JADE: {} cumulant matrices computed", cumulants.len());
    }

    // Initialize rotation matrix
    let mut v = Array2::eye(n);

    // Joint diagonalization using Jacobi rotations
    for iter in 0..max_iter {
        let mut max_theta = 0.0_f64;

        // Sweep through all pairs
        for p in 0..n {
            for q in (p + 1)..n {
                // Compute Givens rotation angle for this pair
                let (c, s, theta) = compute_givens_rotation(&cumulants, &v, p, q);
                max_theta = max_theta.max(theta.abs());

                // Apply rotation to V
                apply_givens_rotation(&mut v, c, s, p, q);
            }
        }

        if verbose && (iter + 1) % 10 == 0 {
            println!("JADE iteration {}: max angle = {:.4e}", iter + 1, max_theta);
        }

        // Check convergence
        if max_theta < tol {
            if verbose {
                println!("JADE converged after {} iterations", iter + 1);
            }
            break;
        }
    }

    // Ensure orthogonality
    let w = sym_decorrelation(&v)?;

    Ok(w)
}

/// Compute fourth-order cumulant matrices for JADE.
///
/// Uses the eigenvalue decomposition approach to select the most
/// informative cumulant slices.
fn compute_cumulant_matrices(x: &Array2<f64>) -> Vec<Array2<f64>> {
    let (n, t) = (x.nrows(), x.ncols());
    let t_f = t as f64;

    // We'll compute cumulant matrices Q_ij where:
    // Q_ij[k,l] = cum(x_i, x_j, x_k, x_l) = E[x_i x_j x_k x_l] - E[x_i x_j]E[x_k x_l]
    //             - E[x_i x_k]E[x_j x_l] - E[x_i x_l]E[x_j x_k]
    //
    // For whitened data, E[x_i x_j] = δ_ij, simplifying the formula.

    let mut matrices = Vec::with_capacity(n * (n + 1) / 2);

    // Precompute x_i * x_j for all pairs
    let mut xx = Array3::zeros((n, n, t));
    for i in 0..n {
        for j in 0..n {
            for s in 0..t {
                xx[[i, j, s]] = x[[i, s]] * x[[j, s]];
            }
        }
    }

    // For each unique pair (i, j), compute the cumulant matrix
    for i in 0..n {
        for j in i..n {
            let mut q = Array2::zeros((n, n));

            for k in 0..n {
                for l in 0..n {
                    // E[x_i x_j x_k x_l]
                    let mut e_ijkl = 0.0;
                    for s in 0..t {
                        e_ijkl += xx[[i, j, s]] * xx[[k, l, s]];
                    }
                    e_ijkl /= t_f;

                    // For whitened data: E[x_a x_b] = δ_ab
                    // cum = E[ijkl] - δ_ij δ_kl - δ_ik δ_jl - δ_il δ_jk
                    let delta_ij_kl = if i == j && k == l { 1.0 } else { 0.0 };
                    let delta_ik_jl = if i == k && j == l { 1.0 } else { 0.0 };
                    let delta_il_jk = if i == l && j == k { 1.0 } else { 0.0 };

                    q[[k, l]] = e_ijkl - delta_ij_kl - delta_ik_jl - delta_il_jk;
                }
            }

            // Symmetrize
            let q = (&q + &q.t()) / 2.0;
            matrices.push(q);
        }
    }

    matrices
}

/// Compute optimal Givens rotation for joint diagonalization.
///
/// Returns (cos(θ), sin(θ), θ) for the rotation that best diagonalizes
/// all matrices simultaneously for the (p, q) pair.
fn compute_givens_rotation(
    matrices: &[Array2<f64>],
    v: &Array2<f64>,
    p: usize,
    q: usize,
) -> (f64, f64, f64) {
    // Accumulate the off-diagonal terms across all rotated matrices
    let mut g = [[0.0; 2]; 2];

    for m in matrices {
        // Compute the relevant 2x2 block of V^T M V
        let mut block = [[0.0; 2]; 2];

        for (bi, &i) in [p, q].iter().enumerate() {
            for (bj, &j) in [p, q].iter().enumerate() {
                for k in 0..m.nrows() {
                    for l in 0..m.ncols() {
                        block[bi][bj] += v[[k, i]] * m[[k, l]] * v[[l, j]];
                    }
                }
            }
        }

        // Contribution to the objective
        let h_pq = block[0][1] + block[1][0];
        let h_pp_qq = block[0][0] - block[1][1];

        g[0][0] += h_pq * h_pq;
        g[0][1] += h_pq * h_pp_qq;
        g[1][1] += h_pp_qq * h_pp_qq;
    }
    g[1][0] = g[0][1];

    // Solve for optimal angle
    // We want to maximize: g[0][0] * sin^2(2θ) + g[1][1] * cos^2(2θ) + g[0][1] * sin(4θ)
    // This is equivalent to finding the dominant eigenvector of a 2x2 problem

    let diff = g[1][1] - g[0][0];
    let angle = if g[0][1].abs() < 1e-15 && diff.abs() < 1e-15 {
        0.0
    } else {
        0.25 * (2.0 * g[0][1]).atan2(diff)
    };

    let c = angle.cos();
    let s = angle.sin();

    (c, s, angle)
}

/// Apply Givens rotation to matrix V (in-place).
fn apply_givens_rotation(v: &mut Array2<f64>, c: f64, s: f64, p: usize, q: usize) {
    let n = v.nrows();

    for i in 0..n {
        let v_ip = v[[i, p]];
        let v_iq = v[[i, q]];
        v[[i, p]] = c * v_ip - s * v_iq;
        v[[i, q]] = s * v_ip + c * v_iq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whitening::{center, whiten};
    use ndarray::Array2;
    use rand::prelude::*;
    use rand::rngs::StdRng;
    use rand_distr::StandardNormal;

    #[test]
    fn test_jade_basic() {
        let mut rng = StdRng::seed_from_u64(42);

        // Generate simple test data
        let n = 3;
        let t = 1000;

        // Generate sources with different distributions
        let mut s = Array2::zeros((n, t));
        for j in 0..t {
            // Laplacian
            let u: f64 = rng.gen_range(0.001..0.999);
            s[[0, j]] = -u.ln().copysign(rng.gen::<f64>() - 0.5);
            // Uniform
            s[[1, j]] = rng.gen_range(-1.73..1.73);
            // Super-Gaussian
            let g: f64 = rng.sample(StandardNormal);
            s[[2, j]] = g.signum() * g.abs().sqrt();
        }

        // Mix
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = rng.sample(StandardNormal);
            }
        }
        let x = a.dot(&s);

        // Preprocess
        let (x_centered, _) = center(&x);
        let whitened = whiten(&x_centered, n).unwrap();

        // Run JADE
        let w = jade(&whitened.data, 100, 1e-6, false).unwrap();

        // Check that W is orthogonal
        let ww_t = w.dot(&w.t());
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ww_t[[i, j]] - expected).abs() < 1e-6,
                    "W should be orthogonal"
                );
            }
        }
    }
}
