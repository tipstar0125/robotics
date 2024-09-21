use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

use crate::agent::Pose;
use crate::camera::{observe_landmark, Observation};
use crate::common::Coord;
use crate::motion::state_transition;
use crate::normal::Normal;

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
    pub nn_pdf: Normal,
    pub no_pdf: Normal,
    pub on_pdf: Normal,
    pub oo_pdf: Normal,
}

impl MotionNoisePdf {
    pub fn new(nn: f64, no: f64, on: f64, oo: f64) -> Self {
        Self {
            nn_pdf: Normal::new(0.0, nn),
            no_pdf: Normal::new(0.0, no),
            on_pdf: Normal::new(0.0, on),
            oo_pdf: Normal::new(0.0, oo),
        }
    }
    pub fn sample(&mut self, rng: &mut Pcg64Mcg) -> (f64, f64, f64, f64) {
        let nn_noise = self.nn_pdf.sample(rng);
        let no_noise = self.no_pdf.sample(rng);
        let on_noise = self.on_pdf.sample(rng);
        let oo_noise = self.oo_pdf.sample(rng);
        (nn_noise, no_noise, on_noise, oo_noise)
    }
}

#[derive(Debug)]
pub struct Estimator {
    pub rng: Pcg64Mcg,
    pub time_interval: f64,
    pub radius: f64,
    pub nu: f64,
    pub omega: f64,
    pub prev_nu: f64,
    pub prev_omega: f64,
    pub particles: Vec<Particle>,
    pub motion_noise_pdf: MotionNoisePdf,
    pub distance_rate_std: f64,
    pub direction_std: f64,
    pub pose_records: Vec<Vec<Pose>>,
    pub best_weight_records: Vec<usize>,
}

impl Estimator {
    pub fn new(
        time_interval: f64,
        init_pose: Pose,
        radius: f64,
        nu: f64,
        omega: f64,
        particle_num: usize,
        motion_noise_pdf: MotionNoisePdf,
        distance_rate_std: f64,
        direction_std: f64,
    ) -> Self {
        Self {
            rng: Pcg64Mcg::seed_from_u64(0),
            time_interval,
            radius,
            nu,
            omega,
            prev_nu: 0.0,
            prev_omega: 0.0,
            particles: vec![Particle::new(init_pose, 1.0); particle_num],
            motion_noise_pdf,
            distance_rate_std,
            direction_std,
            pose_records: vec![vec![init_pose; particle_num]],
            best_weight_records: vec![0],
        }
    }
    pub fn update_motion(&mut self, prev_nu: f64, prev_omega: f64) {
        let mut poses = vec![];
        for particle in self.particles.iter_mut() {
            let (nn_noise, no_noise, on_noise, oo_noise) =
                self.motion_noise_pdf.sample(&mut self.rng);
            let noised_nu = prev_nu
                + nn_noise * (prev_nu.abs() / self.time_interval).sqrt()
                + no_noise * (prev_omega.abs() / self.time_interval);
            let noised_omega = prev_omega
                + on_noise * (prev_nu.abs() / self.time_interval).sqrt()
                + oo_noise * (prev_omega.abs() / self.time_interval);
            particle.pose =
                state_transition(self.time_interval, particle.pose, noised_nu, noised_omega);
            poses.push(particle.pose.clone());
        }
        self.pose_records.push(poses);
    }
    pub fn updater_observation(&mut self, observation: &Vec<Observation>, landmarks: &[Coord]) {
        for particle in self.particles.iter_mut() {
            for obs in observation.iter() {
                let mark = landmarks[obs.id];
                let obs_particle = observe_landmark(&particle.pose, &mark, obs.id);
                let distance_std = self.distance_rate_std * obs_particle.dist;
                let distance_normal = Normal::new(obs_particle.dist, distance_std);
                let direction_normal = Normal::new(obs_particle.angle, self.direction_std);
                particle.weight *= distance_normal.pdf(obs.dist);
                particle.weight *= direction_normal.pdf(obs.angle);
            }
        }
    }
    // 系統サンプリング
    pub fn resampling(&mut self) {
        let mut ws = vec![];
        let mut s = 0.0;
        self.particles.iter().for_each(|particle| {
            s += particle.weight;
            ws.push(s);
        });
        if s < 1e-100 {
            ws = ws.iter().map(|x| x + 1e-100).collect();
            s += 1e-100;
        }
        let step = s / self.particles.len() as f64;
        let mut r = self.rng.gen_range(0.0..step);
        let mut pos = 0;
        let mut particle = vec![];
        let mut best_particle_idx = 0;
        let mut best_weight = std::f64::MIN;
        while particle.len() < self.particles.len() {
            if r < ws[pos] {
                if best_weight < self.particles[pos].weight {
                    best_weight = self.particles[pos].weight;
                    best_particle_idx = particle.len();
                }
                self.particles[pos].weight = 1.0;
                particle.push(self.particles[pos]);
                r += step;
            } else {
                pos += 1;
            }
        }
        self.particles = particle;
        self.best_weight_records.push(best_particle_idx);
    }
    pub fn decision(&mut self, observation: &Vec<Observation>, landmarks: &[Coord]) {
        self.update_motion(self.prev_nu, self.prev_omega);
        self.prev_nu = self.nu;
        self.prev_omega = self.omega;
        self.updater_observation(observation, landmarks);
        self.resampling();
    }
}
