use std::{collections::VecDeque, f64::consts::PI};

use rand::Rng;
use rand_pcg::Pcg64Mcg;

#[derive(Debug)]
pub struct Normal {
    mu: f64,
    std: f64,
    queue: VecDeque<f64>,
}

impl Normal {
    pub fn new(mu: f64, std: f64) -> Self {
        Self {
            mu,
            std,
            queue: VecDeque::new(),
        }
    }
    pub fn sample(&mut self, rng: &mut Pcg64Mcg) -> f64 {
        if self.queue.is_empty() {
            let (x, y) = box_muller(rng, self.mu, self.std);
            self.queue.push_back(x);
            self.queue.push_back(y);
        }
        self.queue.pop_front().unwrap()
    }
    pub fn pdf(&self, x: f64) -> f64 {
        normal_pdf(x, self.mu, self.std)
    }
}

fn box_muller(rng: &mut Pcg64Mcg, mu: f64, std: f64) -> (f64, f64) {
    let u1 = rng.gen::<f64>();
    let u2 = rng.gen::<f64>();

    (
        mu + (-2.0 * u1.ln() * std.powf(2.0)).sqrt() * (2.0 * PI * u2).cos(),
        mu + (-2.0 * u1.ln() * std.powf(2.0)).sqrt() * (2.0 * PI * u2).sin(),
    )
}

fn normal_pdf(x: f64, mu: f64, std: f64) -> f64 {
    let v = (x - mu) / std;
    // (-0.5 * v * v).exp()
    (-0.5 * v * v).exp() / ((2.0 * PI).sqrt() * std)
}
