use crate::agent::Pose;
use crate::common::Coord;
use crate::move_::state_transition;
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
pub struct MotionNoise {
    pub nn: f64,
    pub no: f64,
    pub on: f64,
    pub oo: f64,
    pub mvn: MyMultivariateNormal,
}

impl MotionNoise {
    pub fn new(nn: f64, no: f64, on: f64, oo: f64) -> Self {
        Self {
            nn,
            no,
            on,
            oo,
            mvn: MyMultivariateNormal::new_without_correlation(vec![0.0; 4], vec![nn, no, on, oo]),
        }
    }
}

#[derive(Debug)]
pub struct CameraNoise {
    pub distance_rate: f64,
    pub direction: f64,
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
    pub motion_noise: MotionNoise,
    pub pose_records: Vec<Vec<Pose>>,
}

impl Estimator {
    pub fn new(
        nu: f64,
        omega: f64,
        time_interval: f64,
        radius: f64,
        init_pose: Pose,
        particle_num: usize,
        motion_noise: MotionNoise,
    ) -> Self {
        Self {
            nu,
            omega,
            prev_nu: 0.0,
            prev_omega: 0.0,
            time_interval,
            radius,
            particles: vec![Particle::new(init_pose, 1.0_f64 / particle_num as f64); particle_num],
            motion_noise,
            pose_records: vec![vec![init_pose; particle_num]],
        }
    }
    pub fn update_motion(&mut self, prev_nu: f64, prev_omega: f64) {
        let mut poses = vec![];
        for particle in self.particles.iter_mut() {
            let ns = self.motion_noise.mvn.sample();
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
    pub fn updater_observation(&mut self, landmarks: &[Coord]) {
        for particle in self.particles.iter_mut() {
            for mark in landmarks.iter() {}
        }
    }
    pub fn decision(&mut self, landmarks: &[Coord]) {
        self.update_motion(self.prev_nu, self.prev_omega);
        self.prev_nu = self.nu;
        self.prev_omega = self.omega;
        self.updater_observation(landmarks);
    }
}
