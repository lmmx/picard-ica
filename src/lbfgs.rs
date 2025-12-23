//! L-BFGS optimization with preconditioning.

use ndarray::Array2;

use crate::math::frobenius_dot;

/// L-BFGS memory for storing past iterations.
#[derive(Debug)]
pub struct LBFGSMemory {
    /// s_k = α_k * p_k (step taken)
    s_history: Vec<Array2<f64>>,
    /// y_k = G_{k+1} - G_k (gradient difference)
    y_history: Vec<Array2<f64>>,
    /// ρ_k = 1 / <s_k, y_k>
    rho_history: Vec<f64>,
    /// Maximum memory size
    capacity: usize,
}

impl LBFGSMemory {
    /// Create a new L-BFGS memory with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            s_history: Vec::with_capacity(capacity),
            y_history: Vec::with_capacity(capacity),
            rho_history: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a new (s, y) pair to memory.
    ///
    /// Skips if curvature condition s^T y <= 0 is not satisfied.
    pub fn push(&mut self, s: Array2<f64>, y: Array2<f64>) {
        let sy = frobenius_dot(&s, &y);

        // Skip if curvature condition not satisfied
        if sy <= 1e-10 {
            return;
        }

        let rho = 1.0 / sy;

        // Remove oldest if at capacity
        if self.s_history.len() >= self.capacity {
            self.s_history.remove(0);
            self.y_history.remove(0);
            self.rho_history.remove(0);
        }

        self.s_history.push(s);
        self.y_history.push(y);
        self.rho_history.push(rho);
    }

    /// Clear all memory.
    pub fn clear(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
    }

    /// Check if memory is empty.
    pub fn is_empty(&self) -> bool {
        self.s_history.is_empty()
    }

    /// Compute search direction using two-loop recursion with preconditioner.
    ///
    /// Implements Algorithm 3 from the Picard paper:
    /// The preconditioner H̃⁻¹ is applied between the two loops.
    pub fn compute_direction(
        &self,
        gradient: &Array2<f64>,
        preconditioned_gradient: &Array2<f64>,
    ) -> Array2<f64> {
        // If no history, return preconditioned steepest descent
        if self.is_empty() {
            return -preconditioned_gradient.clone();
        }

        let k = self.s_history.len();
        let mut q = -gradient.clone();
        let mut alphas = vec![0.0; k];

        // First loop (backward through history)
        for i in (0..k).rev() {
            alphas[i] = self.rho_history[i] * frobenius_dot(&self.s_history[i], &q);
            q = &q - &(&self.y_history[i] * alphas[i]);
        }

        // Apply preconditioner H̃⁻¹
        // Scale q based on the ratio of preconditioned to original gradient
        let g_norm_sq = frobenius_dot(gradient, gradient);
        let scale = if g_norm_sq > 1e-15 {
            frobenius_dot(preconditioned_gradient, gradient) / g_norm_sq
        } else {
            1.0
        };
        let mut r = &q * scale;

        // Second loop (forward through history)
        for i in 0..k {
            let beta = self.rho_history[i] * frobenius_dot(&self.y_history[i], &r);
            r = &r + &(&self.s_history[i] * (alphas[i] - beta));
        }

        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lbfgs_memory_push() {
        let mut mem = LBFGSMemory::new(3);

        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![[0.5, 0.0], [0.0, 0.5]];

        mem.push(s.clone(), y.clone());
        assert_eq!(mem.s_history.len(), 1);

        mem.push(s.clone(), y.clone());
        mem.push(s.clone(), y.clone());
        mem.push(s.clone(), y.clone());

        // Should cap at 3
        assert_eq!(mem.s_history.len(), 3);
    }

    #[test]
    fn test_lbfgs_skip_bad_curvature() {
        let mut mem = LBFGSMemory::new(3);

        // s^T y = 1 - 1 = 0 (bad curvature)
        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![[1.0, 0.0], [0.0, -1.0]];

        mem.push(s, y);
        assert!(mem.is_empty(), "Should skip pair with non-positive curvature");
    }

    #[test]
    fn test_lbfgs_direction_no_history() {
        let mem = LBFGSMemory::new(3);
        let g = array![[1.0, 2.0], [3.0, 4.0]];
        let h_inv_g = array![[0.5, 1.0], [1.5, 2.0]];

        let dir = mem.compute_direction(&g, &h_inv_g);

        // Should return -h_inv_g
        assert!((dir[[0, 0]] + 0.5).abs() < 1e-10);
        assert!((dir[[0, 1]] + 1.0).abs() < 1e-10);
    }
}
