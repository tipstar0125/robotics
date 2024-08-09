use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};

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
}

impl Camera {
    const VIS_DISTANCE_RANGE: std::ops::Range<f64> = 0.5..6.0;
    const VIS_DIRECTION_RANGE: std::ops::Range<f64> = -PI / 3.0..PI / 3.0;
    pub fn new() -> Self {
        Self {
            noise: ObservationNoise::new(0.0, 0.0),
        }
    }
    pub fn is_visible(&self, dist: &f64, angle: &f64) -> bool {
        Self::VIS_DISTANCE_RANGE.contains(dist) && Self::VIS_DIRECTION_RANGE.contains(angle)
    }
    pub fn set_noise(&mut self, distance_noise_rate: f64, direction_noise: f64) {
        self.noise = ObservationNoise::new(distance_noise_rate, direction_noise)
    }
    pub fn observe(
        &self,
        rng: &mut ChaCha20Rng,
        pose: Pose,
        landmarks: &[Coord],
    ) -> Vec<Observation> {
        let mut obs = vec![];
        for mark in landmarks.iter() {
            let dx = mark.x - pose.coord.x;
            let dy = mark.y - pose.coord.y;
            let mut dist = (dx.powf(2.0) + dy.powf(2.0)).sqrt();
            let mut angle = dy.atan2(dx) - pose.theta;
            angle = convert_radian_in_range(angle);
            if self.is_visible(&dist, &angle) {
                self.noise.occur(rng, &mut dist, &mut angle);
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
