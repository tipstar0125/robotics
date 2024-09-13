use nalgebra::{DMatrix, DVector, Matrix};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;

#[derive(Debug)]
pub struct MultivariateNormal {
    pub n: usize,
    pub mu: DMatrix<f64>,
    pub l: DMatrix<f64>,
    pub rng: Pcg64Mcg,
}

impl MultivariateNormal {
    pub fn new(std: &Vec<f64>) -> Self {
        let mu = DMatrix::from_vec(std.len(), 1, (0..std.len()).map(|_| 0.0_f64).collect());
        let std2: Vec<_> = std.clone().iter().cloned().map(|x| x.powf(2.0)).collect();
        let sig = DMatrix::from_diagonal(&DVector::from_row_slice(&std2));

        Self {
            n: std.len(),
            mu,
            l: sig.cholesky().unwrap().l(),
            rng: Pcg64Mcg::seed_from_u64(0),
        }
    }
    pub fn sample(&mut self) -> Vec<f64> {
        let r = DMatrix::from_vec(
            self.n,
            1,
            (0..self.n)
                .map(|_| self.rng.sample(StandardNormal))
                .collect::<Vec<f64>>(),
        );
        let sample = self.l.clone() * r + self.mu.clone();
        let ret: Vec<f64> = sample.iter().cloned().collect();
        ret
    }
}
