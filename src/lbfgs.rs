//! L-BFGS optimization for the PICARD algorithm.

use ndarray::{Array1, Array2};

/// L-BFGS memory storage.
pub struct LbfgsMemory {
    /// Step differences (s_k = x_{k+1} - x_k).
    pub s_list: Vec<Array2<f64>>,
    /// Gradient differences (y_k = g_{k+1} - g_k).
    pub y_list: Vec<Array2<f64>>,
    /// Curvature estimates (r_k = 1 / (s_k · y_k)).
    pub r_list: Vec<f64>,
    #[allow(unused)]
    /// Maximum memory size.
    max_size: usize,
}

impl LbfgsMemory {
    /// Create a new L-BFGS memory with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            s_list: Vec::with_capacity(max_size),
            y_list: Vec::with_capacity(max_size),
            r_list: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Clear all stored memory.
    pub fn clear(&mut self) {
        self.s_list.clear();
        self.y_list.clear();
        self.r_list.clear();
    }

    #[allow(unused)]
    /// Update the memory with a new step.
    ///
    /// # Arguments
    /// * `s` - Step difference (direction * alpha)
    /// * `y` - Gradient difference
    pub fn update(&mut self, s: Array2<f64>, y: Array2<f64>) {
        let sy: f64 = (&s * &y).sum();

        // Only update if curvature condition is satisfied (s·y > 0)
        if sy > 1e-10 {
            let r = 1.0 / sy;

            if self.s_list.len() >= self.max_size {
                self.s_list.remove(0);
                self.y_list.remove(0);
                self.r_list.remove(0);
            }

            self.s_list.push(s);
            self.y_list.push(y);
            self.r_list.push(r);
        }
    }

    #[allow(unused)]
    /// Check if memory is empty.
    pub fn is_empty(&self) -> bool {
        self.s_list.is_empty()
    }

    /// Get the number of stored pairs.
    pub fn len(&self) -> usize {
        self.s_list.len()
    }
}

/// Compute the L-BFGS search direction.
///
/// # Arguments
/// * `g` - Current gradient
/// * `h` - Hessian approximation (diagonal or full)
/// * `h_off` - Hessian off-diagonal elements
/// * `memory` - L-BFGS memory
/// * `ortho` - Whether using orthogonal constraint
///
/// # Returns
/// * Search direction (negative of preconditioned gradient)
pub fn compute_direction(
    g: &Array2<f64>,
    h: &Array2<f64>,
    h_off: &Array1<f64>,
    memory: &LbfgsMemory,
    ortho: bool,
) -> Array2<f64> {
    let mut q = g.clone();
    let mut alpha_list = Vec::with_capacity(memory.len());

    // First loop: backward through memory
    for ((s, y), &r) in memory
        .s_list
        .iter()
        .zip(memory.y_list.iter())
        .zip(memory.r_list.iter())
        .rev()
    {
        let alpha = r * (s * &q).sum();
        alpha_list.push(alpha);
        q = &q - alpha * y;
    }
    alpha_list.reverse();

    // Apply preconditioner (inverse Hessian approximation)
    let mut z = if ortho {
        // For orthogonal case, use diagonal scaling
        let mut z = &q / h;
        // Make skew-symmetric
        z = (&z - &z.t()) / 2.0;
        z
    } else {
        // Solve the 2x2 systems for each pair of components
        solve_hessian_system(h, h_off, &q)
    };

    // Second loop: forward through memory
    for (((s, y), &r), &alpha) in memory
        .s_list
        .iter()
        .zip(memory.y_list.iter())
        .zip(memory.r_list.iter())
        .zip(alpha_list.iter())
    {
        let beta = r * (y * &z).sum();
        z = &z + (alpha - beta) * s;
    }

    -z
}

/// Solve the Hessian system for the non-orthogonal case.
///
/// For each (i,j) pair, solves the 2x2 system:
/// [[h[i,j], h_off[i]], [h_off[j], h[j,i]]] * [x, y]^T = [g[i,j], g[j,i]]^T
///
/// Uses a numerically stable approach that avoids hard thresholds.
fn solve_hessian_system(h: &Array2<f64>, h_off: &Array1<f64>, g: &Array2<f64>) -> Array2<f64> {
    let n = h.nrows();
    let mut result = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            // 2x2 system: [[a, b], [c, d]] * [x, y]^T = [e, f]^T
            // where a = h[i,j], b = h_off[i], c = h_off[j], d = h[j,i]
            //       e = g[i,j], f = g[j,i]
            let a = h[[i, j]];
            let b = h_off[i];
            let c = h_off[j];
            let d = h[[j, i]];
            let e = g[[i, j]];
            let f = g[[j, i]];

            let det = a * d - b * c;

            // Use relative tolerance based on matrix scale
            // This avoids the sharp threshold that creates discontinuities
            let scale = (a.abs() + d.abs() + b.abs() + c.abs()) * 0.25;
            let tol = scale * 1e-12 + 1e-15; // Relative + absolute tolerance

            if det.abs() > tol {
                // Standard Cramer's rule solution
                result[[i, j]] = (d * e - b * f) / det;
            } else {
                // Near-singular case: use regularized pseudoinverse approach
                // Add small regularization to make the system well-conditioned
                let reg = scale * 1e-6 + 1e-12;
                let det_reg = det + reg.copysign(det + 1e-30);
                result[[i, j]] = (d * e - b * f) / det_reg;
            }
        }
    }

    result
}

/// Regularize the Hessian approximation.
///
/// Ensures eigenvalues are at least `lambda_min` for numerical stability.
/// For non-symmetric 2x2 blocks, regularizes both elements symmetrically.
pub fn regularize_hessian(h: &mut Array2<f64>, h_off: &Array1<f64>, lambda_min: f64) {
    let n = h.nrows();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                // For the 2x2 block [[h[i,j], h_off[i]], [h_off[j], h[j,i]]]
                // compute the smaller eigenvalue
                let a = h[[i, j]];
                let d = h[[j, i]];
                let bc = h_off[i] * h_off[j];

                let trace = a + d;
                let det = a * d - bc;

                // Eigenvalues: (trace ± sqrt(trace² - 4*det)) / 2
                let discriminant = trace * trace - 4.0 * det;

                if discriminant >= 0.0 {
                    let sqrt_disc = discriminant.sqrt();
                    let lambda_min_eigen = 0.5 * (trace - sqrt_disc);

                    if lambda_min_eigen < lambda_min {
                        // Shift both diagonal elements to ensure minimum eigenvalue
                        let shift = lambda_min - lambda_min_eigen;
                        h[[i, j]] += shift;
                        // Note: we only modify h[i,j] here because h[j,i] will be
                        // modified when we process the (j,i) pair
                    }
                } else {
                    // Complex eigenvalues (shouldn't happen for well-formed Hessians)
                    // Regularize conservatively
                    if h[[i, j]] < lambda_min {
                        h[[i, j]] = lambda_min;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lbfgs_memory() {
        let mut memory = LbfgsMemory::new(3);
        assert!(memory.is_empty());

        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![[0.5, 0.0], [0.0, 0.5]];
        memory.update(s, y);

        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_memory_overflow() {
        let mut memory = LbfgsMemory::new(2);

        for i in 0..5 {
            let s = Array2::from_elem((2, 2), i as f64 + 1.0);
            let y = Array2::from_elem((2, 2), 1.0);
            memory.update(s, y);
        }

        // Should only keep last 2
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn test_curvature_condition() {
        let mut memory = LbfgsMemory::new(3);

        // Positive curvature - should be accepted
        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        memory.update(s.clone(), y);
        assert_eq!(memory.len(), 1);

        // Negative curvature - should be rejected
        let y_neg = array![[-1.0, 0.0], [0.0, -1.0]];
        memory.update(s.clone(), y_neg);
        assert_eq!(memory.len(), 1); // Still 1, not 2

        // Zero curvature - should be rejected
        let y_zero = array![[0.0, 1.0], [-1.0, 0.0]]; // Orthogonal to s
        memory.update(s, y_zero);
        assert_eq!(memory.len(), 1); // Still 1
    }

    #[test]
    fn test_solve_hessian_near_singular() {
        // Test that near-singular systems don't produce NaN or huge values
        let h = array![[1.0, 1.0], [1.0, 1.0]]; // Nearly singular when h_off = [1, 1]
        let h_off = array![0.99, 0.99]; // det ≈ 0.02
        let g = array![[1.0, 0.0], [0.0, 1.0]];

        let result = solve_hessian_system(&h, &h_off, &g);

        // Should produce finite results
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
