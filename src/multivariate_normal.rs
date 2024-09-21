use nalgebra::{DMatrix, DVector, Dyn};
use nalgebra_mvn::MultivariateNormal;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;

#[derive(Debug)]
pub struct MyMultivariateNormal {
    pub n: usize,
    pub mu: DMatrix<f64>,
    pub l: DMatrix<f64>,
    pub rng: Pcg64Mcg,
    pub mvn: MultivariateNormal<f64, Dyn>,
}

impl MyMultivariateNormal {
    pub fn new(mu_vec: Vec<f64>, sig_vec: Vec<f64>) -> Self {
        assert!(mu_vec.len() * mu_vec.len() == sig_vec.len());
        let n = mu_vec.len();
        let sig = DMatrix::from_row_slice(mu_vec.len(), 2, &sig_vec);
        let mvn =
            MultivariateNormal::from_mean_and_covariance(&DVector::from_row_slice(&mu_vec), &sig)
                .unwrap();
        let mu = DMatrix::from_vec(mu_vec.len(), 1, mu_vec);

        Self {
            n,
            mu,
            l: sig.cholesky().unwrap().l(),
            rng: Pcg64Mcg::seed_from_u64(0),
            mvn,
        }
    }
    pub fn new_without_correlation(mu_vec: Vec<f64>, std_vec: Vec<f64>) -> Self {
        assert!(mu_vec.len() == std_vec.len());
        let n = mu_vec.len();
        let std2: Vec<_> = std_vec.iter().map(|x| x.powf(2.0)).collect();
        let sig = DMatrix::from_diagonal(&DVector::from_row_slice(&std2));
        let mvn =
            MultivariateNormal::from_mean_and_covariance(&DVector::from_row_slice(&mu_vec), &sig)
                .unwrap();
        let mu = DMatrix::from_vec(mu_vec.len(), 1, mu_vec);

        Self {
            n,
            mu,
            l: sig.cholesky().unwrap().l(),
            rng: Pcg64Mcg::seed_from_u64(0),
            mvn,
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
    pub fn possibility(&self, xs_vec: Vec<f64>) -> f64 {
        assert!(self.n == xs_vec.len());
        let xs = DMatrix::from_row_slice(1, self.n, &xs_vec);
        let result = self.mvn.pdf(&xs);
        result[0]
    }
}
