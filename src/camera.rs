use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::{
    agent::Pose,
    common::{convert_radian_in_range, Coord},
};

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub dist: f64,
    pub angle: f64,
}

impl std::fmt::Display for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dist: {}, angle: {}", self.dist, self.angle)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct Camera {
    pub noise: ObservationNoise,
    pub bias: ObservationBias,
    pub phantom: Phantom,
    pub oversight: Oversight,
    pub occlusion: Occlusion,
}

impl Camera {
    const VIS_DISTANCE_RANGE: std::ops::Range<f64> = 0.5..6.0;
    const VIS_DIRECTION_RANGE: std::ops::Range<f64> = -PI / 3.0..PI / 3.0;
    pub fn new() -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        Self {
            noise: ObservationNoise::new(0.0, 0.0),
            bias: ObservationBias::new(&mut rng, 0.0, 0.0),
            phantom: Phantom::new(0.0, 0.0, 0.0),
            oversight: Oversight::new(0.0),
            occlusion: Occlusion::new(0.0),
        }
    }
    pub fn is_visible(&self, dist: &f64, angle: &f64) -> bool {
        Self::VIS_DISTANCE_RANGE.contains(dist) && Self::VIS_DIRECTION_RANGE.contains(angle)
    }
    pub fn set_noise(&mut self, distance_noise_rate: f64, direction_noise: f64) {
        self.noise = ObservationNoise::new(distance_noise_rate, direction_noise)
    }
    pub fn set_bias(
        &mut self,
        rng: &mut ChaCha20Rng,
        distance_bias_rate_std: f64,
        direction_bias_std: f64,
    ) {
        self.bias = ObservationBias::new(rng, distance_bias_rate_std, direction_bias_std);
    }
    pub fn set_phantom(&mut self, prob: f64, width: f64, height: f64) {
        self.phantom = Phantom::new(prob, width, height);
    }
    pub fn set_oversight(&mut self, prob: f64) {
        self.oversight = Oversight::new(prob);
    }
    pub fn set_occlusion(&mut self, prob: f64) {
        self.occlusion = Occlusion::new(prob);
    }
    pub fn observe(
        &mut self,
        rng: &mut ChaCha20Rng,
        pose: Pose,
        landmarks: &[Coord],
    ) -> Vec<Observation> {
        let mut obs = vec![];
        for mark in landmarks.iter() {
            let mut mark_x = mark.x;
            let mut mark_y = mark.y;
            self.phantom.occur(rng, &mut mark_x, &mut mark_y);
            let dx = mark_x - pose.coord.x;
            let dy = mark_y - pose.coord.y;
            let mut dist = (dx.powf(2.0) + dy.powf(2.0)).sqrt();
            let mut angle = dy.atan2(dx) - pose.theta;
            angle = convert_radian_in_range(angle);
            self.occlusion
                .occur(rng, &mut dist, Self::VIS_DISTANCE_RANGE);
            if !self.oversight.occur(rng) && self.is_visible(&dist, &angle) {
                self.bias.on(&mut dist, &mut angle);
                self.noise.occur(rng, &mut dist, &mut angle);
                angle = convert_radian_in_range(angle);
                obs.push(Observation { dist, angle })
            }
        }
        obs
    }
}

#[derive(Debug)]
pub struct ObservationNoise {
    pub distance_noise_rate: f64,
    pub direction_noise: f64,
}

impl ObservationNoise {
    pub fn new(distance_noise_rate: f64, direction_noise: f64) -> Self {
        Self {
            distance_noise_rate,
            direction_noise,
        }
    }
    pub fn occur(&self, rng: &mut ChaCha20Rng, dist: &mut f64, angle: &mut f64) {
        *dist = Normal::new(*dist, *dist * self.distance_noise_rate)
            .unwrap()
            .sample(rng);
        *angle = Normal::new(*angle, self.direction_noise)
            .unwrap()
            .sample(rng);
    }
}

#[derive(Debug)]
pub struct ObservationBias {
    pub distance_bias_rate_std: f64,
    pub direction_bias_std: f64,
    pub distance_bias_rate: f64,
    pub direction_bias: f64,
}

impl ObservationBias {
    pub fn new(
        rng: &mut ChaCha20Rng,
        distance_bias_rate_std: f64,
        direction_bias_std: f64,
    ) -> Self {
        Self {
            distance_bias_rate_std,
            direction_bias_std,
            distance_bias_rate: Normal::new(0.0, distance_bias_rate_std)
                .unwrap()
                .sample(rng),
            direction_bias: Normal::new(0.0, direction_bias_std).unwrap().sample(rng),
        }
    }
    pub fn on(&self, dist: &mut f64, angle: &mut f64) {
        *dist += *dist * self.distance_bias_rate;
        *angle += self.direction_bias;
    }
}

#[derive(Debug)]

pub struct Phantom {
    pub prob: f64,
    pub x_dist: Uniform<f64>,
    pub y_dist: Uniform<f64>,
}

impl Phantom {
    pub fn new(prob: f64, width: f64, height: f64) -> Self {
        let x_range = -width / 2.0..=width / 2.0;
        let y_range = -height / 2.0..=height / 2.0;
        Self {
            prob,
            x_dist: Uniform::from(x_range),
            y_dist: Uniform::from(y_range),
        }
    }
    pub fn occur(&mut self, rng: &mut ChaCha20Rng, x: &mut f64, y: &mut f64) {
        if rng.gen_range(0.0..=1.0) < self.prob {
            *x = self.x_dist.sample(rng);
            *y = self.y_dist.sample(rng);
        }
    }
}

#[derive(Debug)]
pub struct Oversight {
    pub prob: f64,
}

impl Oversight {
    fn new(prob: f64) -> Self {
        Self { prob }
    }
    fn occur(&self, rng: &mut ChaCha20Rng) -> bool {
        rng.gen_range(0.0..=1.0) < self.prob
    }
}

#[derive(Debug)]
pub struct Occlusion {
    pub prob: f64,
}

impl Occlusion {
    fn new(prob: f64) -> Self {
        Self { prob }
    }
    fn occur(&self, rng: &mut ChaCha20Rng, dist: &mut f64, dist_range: std::ops::Range<f64>) {
        if rng.gen_range(0.0..=1.0) < self.prob {
            *dist += rng.gen_range(0.0..=1.0) * (dist_range.end - dist_range.start);
        }
    }
}
