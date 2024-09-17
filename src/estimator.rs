use rand::distributions::WeightedIndex;
use rand::SeedableRng;
use rand_distr::Distribution;
use rand_pcg::Pcg64Mcg;

use crate::agent::Pose;
use crate::camera::{observe_landmark, Observation};
use crate::common::Coord;
use crate::motion::state_transition;
use crate::multivariate_normal::MyMultivariateNormal;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub pose: Pose,
    pub weight: f64,
}

impl Particle {
    fn new(init_pose: Pose, init_weight: f64) -> Self {
        Self {
            pose: init_pose,
            weight: init_weight,
        }
    }
}

#[derive(Debug)]
pub struct MotionNoisePdf {
    pub nn: f64,
    pub no: f64,
    pub on: f64,
    pub oo: f64,
    pub pdf: MyMultivariateNormal,
}

impl MotionNoisePdf {
    pub fn new(nn: f64, no: f64, on: f64, oo: f64) -> Self {
        Self {
            nn,
            no,
            on,
            oo,
            pdf: MyMultivariateNormal::new_without_correlation(vec![0.0; 4], vec![nn, no, on, oo]),
        }
    }
    pub fn sample(&mut self) -> Vec<f64> {
        self.pdf.sample()
    }
    pub fn possibility(&self, xs_vec: Vec<f64>) -> f64 {
        self.pdf.possibility(xs_vec)
    }
}

#[derive(Debug)]
pub struct ObservationNoiseStd {
    pub distance_rate_std: f64,
    pub direction_std: f64,
}

#[derive(Debug)]
pub struct ObservationNoisePdf {
    pub distance_mu: f64,
    pub direction_mu: f64,
    pub distance_std: f64,
    pub direction_std: f64,
    pub pdf: MyMultivariateNormal,
}

impl ObservationNoisePdf {
    pub fn new(distance_mu: f64, direction_mu: f64, distance_std: f64, direction_std: f64) -> Self {
        Self {
            distance_mu,
            direction_mu,
            distance_std,
            direction_std,
            pdf: MyMultivariateNormal::new_without_correlation(
                vec![distance_mu, direction_mu],
                vec![distance_std, direction_std],
            ),
        }
    }
    pub fn sample(&mut self) -> Vec<f64> {
        self.pdf.sample()
    }
    pub fn possibility(&self, xs_vec: Vec<f64>) -> f64 {
        self.pdf.possibility(xs_vec)
    }
}

#[derive(Debug)]
pub struct Estimator {
    pub nu: f64,
    pub omega: f64,
    pub prev_nu: f64,
    pub prev_omega: f64,
    pub time_interval: f64,
    pub radius: f64,
    pub particles: Vec<Particle>,
    pub motion_noise_pdf: MotionNoisePdf,
    pub observation_noise_std: ObservationNoiseStd,
    pub pose_records: Vec<Vec<Pose>>,
    pub weight_records: Vec<Vec<f64>>,
    pub rng: Pcg64Mcg,
}

impl Estimator {
    pub fn new(
        nu: f64,
        omega: f64,
        time_interval: f64,
        radius: f64,
        init_pose: Pose,
        particle_num: usize,
        motion_noise_pdf: MotionNoisePdf,
        observation_noise_std: ObservationNoiseStd,
    ) -> Self {
        Self {
            nu,
            omega,
            prev_nu: 0.0,
            prev_omega: 0.0,
            time_interval,
            radius,
            particles: vec![Particle::new(init_pose, 1.0_f64 / particle_num as f64); particle_num],
            motion_noise_pdf,
            observation_noise_std,
            pose_records: vec![vec![init_pose; particle_num]],
            weight_records: vec![vec![1.0_f64 / particle_num as f64; particle_num]],
            rng: Pcg64Mcg::seed_from_u64(0),
        }
    }
    pub fn update_motion(&mut self, prev_nu: f64, prev_omega: f64) {
        let mut poses = vec![];
        for particle in self.particles.iter_mut() {
            let ns = self.motion_noise_pdf.sample();
            let nn_noise = ns[0];
            let no_noise = ns[1];
            let on_noise = ns[2];
            let oo_noise = ns[3];
            let noised_nu = prev_nu
                + nn_noise * (prev_nu.abs() / self.time_interval).sqrt()
                + no_noise * (prev_omega.abs() / self.time_interval);
            let noised_omega = prev_omega
                + on_noise * (prev_nu.abs() / self.time_interval).sqrt()
                + oo_noise * (prev_omega.abs() / self.time_interval);
            particle.pose =
                state_transition(particle.pose, noised_nu, noised_omega, self.time_interval);
            poses.push(particle.pose.clone());
        }
        self.pose_records.push(poses);
    }
    pub fn updater_observation(&mut self, observation: &Vec<Observation>, landmarks: &[Coord]) {
        let mut weights = vec![];
        for particle in self.particles.iter_mut() {
            for obs in observation.iter() {
                let mark = landmarks[obs.id];
                let obs_particle = observe_landmark(&particle.pose, &mark, obs.id);
                let distance_std = self.observation_noise_std.distance_rate_std * obs_particle.dist;
                let pdf = MyMultivariateNormal::new_without_correlation(
                    vec![obs_particle.dist, obs_particle.angle],
                    vec![distance_std, self.observation_noise_std.direction_std],
                );
                particle.weight *= pdf.possibility(vec![obs.dist, obs.angle]);
            }
            weights.push(particle.weight);
        }
        self.weight_records.push(weights);
    }
    pub fn resampling(&mut self) {
        let sum_weight = self.particles.iter().map(|x| x.weight).sum::<f64>();
        if sum_weight < 1e-100 {
            self.particles.iter_mut().for_each(|x| x.weight += 1e-100);
        }
        let weights: Vec<_> = self.particles.iter().map(|x| x.weight).collect();
        let weight_choice = WeightedIndex::new(weights).unwrap();
        let particles: Vec<_> = (0..self.particles.len())
            .map(|_| self.particles[weight_choice.sample(&mut self.rng)])
            .collect();
        self.particles = particles;
    }
    pub fn decision(&mut self, observation: &Vec<Observation>, landmarks: &[Coord]) {
        self.update_motion(self.prev_nu, self.prev_omega);
        self.prev_nu = self.nu;
        self.prev_omega = self.omega;
        self.updater_observation(observation, landmarks);
        self.resampling();
    }
}
