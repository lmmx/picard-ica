// src/lbfgs.rs

//! L-BFGS optimization for the PICARD algorithm.

use faer::{Col, Mat, MatRef};

/// L-BFGS memory storage.
pub struct LbfgsMemory {
    /// Step differences (s_k = x_{k+1} - x_k).
    pub s_list: Vec<Mat<f64>>,
    /// Gradient differences (y_k = g_{k+1} - g_k).
    pub y_list: Vec<Mat<f64>>,
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
    pub fn update(&mut self, s: Mat<f64>, y: Mat<f64>) {
        let sy = mat_dot(&s, &y);

        // Only update if curvature condition is satisfied
        if sy.abs() > 1e-15 {
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

/// Element-wise dot product of two matrices (sum of element-wise products).
fn mat_dot(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut sum = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            sum += a[(i, j)] * b[(i, j)];
        }
    }
    sum
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
    g: MatRef<'_, f64>,
    h: MatRef<'_, f64>,
    h_off: &Col<f64>,
    memory: &LbfgsMemory,
    ortho: bool,
) -> Mat<f64> {
    let mut q = g.to_owned();
    let mut alpha_list = Vec::with_capacity(memory.len());

    // First loop: backward through memory
    for ((s, y), &r) in memory
        .s_list
        .iter()
        .zip(memory.y_list.iter())
        .zip(memory.r_list.iter())
        .rev()
    {
        let alpha = r * mat_dot(s, &q);
        alpha_list.push(alpha);
        q = &q - y * faer::Scale(alpha);
    }
    alpha_list.reverse();

    // Apply preconditioner (inverse Hessian approximation)
    let mut z = if ortho {
        // For orthogonal case, use diagonal scaling
        let mut z = Mat::zeros(q.nrows(), q.ncols());
        for j in 0..q.ncols() {
            for i in 0..q.nrows() {
                z[(i, j)] = q[(i, j)] / h[(i, j)];
            }
        }
        // Make skew-symmetric
        z = (&z - z.transpose()) * faer::Scale(0.5);
        z
    } else {
        // Solve the 2x2 systems for each pair of components
        solve_hessian_system(h, h_off, g)
    };

    // Second loop: forward through memory
    for (((s, y), &r), &alpha) in memory
        .s_list
        .iter()
        .zip(memory.y_list.iter())
        .zip(memory.r_list.iter())
        .zip(alpha_list.iter())
    {
        let beta = r * mat_dot(y, &z);
        z = &z + s * faer::Scale(alpha - beta);
    }

    z * faer::Scale(-1.0)
}

/// Solve the Hessian system for the non-orthogonal case.
fn solve_hessian_system(h: MatRef<'_, f64>, h_off: &Col<f64>, g: MatRef<'_, f64>) -> Mat<f64> {
    let n = h.nrows();
    let mut result = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let det = h[(i, j)] * h[(j, i)] - h_off[i] * h_off[j];
            if det.abs() > 1e-15 {
                result[(i, j)] = (h[(j, i)] * g[(i, j)] - h_off[i] * g[(j, i)]) / det;
            }
        }
    }

    result
}

/// Regularize the Hessian approximation.
///
/// Ensures eigenvalues are at least `lambda_min` for numerical stability.
pub fn regularize_hessian(h: &mut Mat<f64>, h_off: &Col<f64>, lambda_min: f64) {
    let n = h.nrows();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = h[(i, j)] - h[(j, i)];
                let discr = (diff * diff + 4.0 * h_off[i] * h_off[j]).sqrt();
                let eigenvalue = 0.5 * (h[(i, j)] + h[(j, i)] - discr);

                if eigenvalue < lambda_min {
                    h[(i, j)] += lambda_min - eigenvalue;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_lbfgs_memory() {
        let mut memory = LbfgsMemory::new(3);
        assert!(memory.is_empty());

        let s = mat![[1.0, 0.0], [0.0, 1.0]];
        let y = mat![[0.5, 0.0], [0.0, 0.5]];
        memory.update(s, y);

        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_memory_overflow() {
        let mut memory = LbfgsMemory::new(2);

        for i in 0..5 {
            let s = Mat::from_fn(2, 2, |_, _| i as f64 + 1.0);
            let y = Mat::from_fn(2, 2, |_, _| 1.0);
            memory.update(s, y);
        }

        // Should only keep last 2
        assert_eq!(memory.len(), 2);
    }
}
