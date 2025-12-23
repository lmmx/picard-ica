//! Utility functions for ICA analysis.

use faer::{Mat, MatRef};

/// Permute and scale a matrix to be close to identity.
///
/// This is useful for evaluating separation quality when the
/// true mixing matrix is known.
///
/// # Arguments
/// * `a` - Matrix to permute (typically W @ A where W is unmixing and A is mixing)
/// * `scale` - If true, scale rows to have unit diagonal
///
/// # Returns
/// * Permuted (and optionally scaled) matrix
pub fn permute(a: MatRef<'_, f64>, scale: bool) -> Mat<f64> {
    let n = a.nrows();
    let mut a = a.to_owned();

    // Iteratively swap rows to maximize diagonal elements
    let mut done = false;
    while !done {
        done = true;
        for i in 0..n {
            for j in 0..i {
                let diag_sq = a[(i, i)].powi(2) + a[(j, j)].powi(2);
                let off_sq = a[(i, j)].powi(2) + a[(j, i)].powi(2);

                if diag_sq < off_sq {
                    // Swap rows i and j
                    for col in 0..a.ncols() {
                        let tmp = a[(i, col)];
                        a[(i, col)] = a[(j, col)];
                        a[(j, col)] = tmp;
                    }
                    done = false;
                }
            }
        }
    }

    // Scale by diagonal
    if scale {
        for i in 0..n {
            let diag = a[(i, i)];
            if diag.abs() > 1e-10 {
                for j in 0..a.ncols() {
                    a[(i, j)] /= diag;
                }
            }
        }
    }

    // Sort by column sum for consistent ordering
    let col_sums: Vec<f64> = (0..n)
        .map(|j| (0..n).map(|i| a[(i, j)].abs()).sum())
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| col_sums[i].partial_cmp(&col_sums[j]).unwrap());

    let mut result = Mat::zeros(n, n);
    for (new_i, &old_i) in order.iter().enumerate() {
        for (new_j, &old_j) in order.iter().enumerate() {
            result[(new_i, new_j)] = a[(old_i, old_j)];
        }
    }

    result
}

/// Compute the Amari distance between two matrices.
///
/// The Amari distance measures how close `W @ A` is to a permutation
/// and scaling matrix. It equals 0 when W perfectly unmixes A.
///
/// # Arguments
/// * `w` - Unmixing matrix
/// * `a` - Mixing matrix
///
/// # Returns
/// * Amari distance (0 = perfect separation)
pub fn amari_distance(w: MatRef<'_, f64>, a: MatRef<'_, f64>) -> f64 {
    let p = w * a;
    let n = p.nrows() as f64;

    let s = |r: &Mat<f64>| -> f64 {
        let mut sum = 0.0;
        for i in 0..r.nrows() {
            let row_sq: Vec<f64> = (0..r.ncols()).map(|j| r[(i, j)] * r[(i, j)]).collect();
            let row_sum: f64 = row_sq.iter().sum();
            let row_max: f64 = row_sq.iter().cloned().fold(0.0, f64::max);
            if row_max > 1e-15 {
                sum += row_sum / row_max - 1.0;
            }
        }
        sum
    };

    let p_abs = Mat::from_fn(p.nrows(), p.ncols(), |i, j| p[(i, j)].abs());
    let p_abs_t = p_abs.transpose().to_owned();

    (s(&p_abs) + s(&p_abs_t)) / (2.0 * n)
}

/// Check if a density function is valid by verifying gradient consistency.
///
/// This numerically checks that the score function is the derivative
/// of the log-likelihood.
#[cfg(test)]
pub fn check_density<D: crate::density::Density>(density: &D, tol: f64) -> bool {
    use faer::{col, mat, Col};

    let test_points: [f64; 7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let eps = 1e-7;

    for &y in test_points.iter() {
        let y_arr = Col::from_fn(1, |_| y);
        let y_plus = Col::from_fn(1, |_| y + eps);
        let y_minus = Col::from_fn(1, |_| y - eps);

        let log_lik = density.log_lik(&y_arr)[0];
        let log_lik_plus = density.log_lik(&y_plus)[0];
        let log_lik_minus = density.log_lik(&y_minus)[0];

        let numerical_score = (log_lik_plus - log_lik_minus) / (2.0 * eps);

        let y_2d = Mat::from_fn(1, 1, |_, _| y);
        let (score, _) = density.score_and_der(y_2d.as_ref());
        let analytical_score = score[(0, 0)];

        if (numerical_score - analytical_score).abs() > tol {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::{Cube, Exp, Tanh};
    use crate::math::determinant;
    use faer::mat;

    // Simple matrix inverse for testing
    fn simple_inverse(m: &Mat<f64>) -> Mat<f64> {
        let n = m.nrows();
        let mut aug = Mat::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug[(i, j)] = m[(i, j)];
            }
            aug[(i, n + i)] = 1.0;
        }

        for i in 0..n {
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[(k, i)].abs() > aug[(max_row, i)].abs() {
                    max_row = k;
                }
            }

            for j in 0..(2 * n) {
                let tmp = aug[(i, j)];
                aug[(i, j)] = aug[(max_row, j)];
                aug[(max_row, j)] = tmp;
            }

            let pivot = aug[(i, i)];
            for j in 0..(2 * n) {
                aug[(i, j)] /= pivot;
            }

            for k in 0..n {
                if k != i {
                    let factor = aug[(k, i)];
                    for j in 0..(2 * n) {
                        aug[(k, j)] -= factor * aug[(i, j)];
                    }
                }
            }
        }

        let mut inv = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv[(i, j)] = aug[(i, n + j)];
            }
        }
        inv
    }

    #[test]
    fn test_amari_distance_perfect() {
        // W = A^{-1} should give distance ~0
        let a = mat![[1.0, 0.5, 0.2], [0.3, 1.0, 0.4], [0.1, 0.2, 1.0]];

        let w = simple_inverse(&a);
        let dist = amari_distance(w.as_ref(), a.as_ref());

        assert!(dist < 1e-10, "Amari distance should be ~0, got {}", dist);
    }

    #[test]
    fn test_amari_distance_permutation() {
        // Permuted inverse should also give distance ~0
        let a = mat![[1.0, 0.5], [0.3, 1.0]];
        let w_inv = simple_inverse(&a);

        // Swap rows (permutation)
        let mut w = Mat::zeros(2, 2);
        w[(0, 0)] = w_inv[(1, 0)];
        w[(0, 1)] = w_inv[(1, 1)];
        w[(1, 0)] = w_inv[(0, 0)];
        w[(1, 1)] = w_inv[(0, 1)];

        let dist = amari_distance(w.as_ref(), a.as_ref());
        assert!(dist < 1e-10, "Amari distance should be ~0, got {}", dist);
    }

    #[test]
    fn test_permute() {
        let a = mat![[0.1, 0.9], [0.95, 0.05]];

        let p = permute(a.as_ref(), true);

        // After permutation and scaling, diagonal should be 1
        assert!(
            (p[(0, 0)] - 1.0).abs() < 1e-6,
            "Diagonal should be 1, got {}",
            p[(0, 0)]
        );
        assert!(
            (p[(1, 1)] - 1.0).abs() < 1e-6,
            "Diagonal should be 1, got {}",
            p[(1, 1)]
        );
    }

    #[test]
    fn test_density_tanh() {
        let density = Tanh::default();
        assert!(check_density(&density, 1e-5));
    }

    #[test]
    fn test_density_exp() {
        let density = Exp::new(0.1);
        assert!(check_density(&density, 1e-5));
    }

    #[test]
    fn test_density_cube() {
        let density = Cube::new();
        assert!(check_density(&density, 1e-5));
    }
}
