use crate::agent::Pose;
use crate::move_::state_transition;
use crate::multivariate_normal::MultivariateNormal;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub pose: Pose,
}

impl Particle {
    fn new(init_pose: Pose) -> Self {
        Self { pose: init_pose }
    }
}

#[derive(Debug)]
pub struct MotionNoiseStd {
    pub nn: f64,
    pub no: f64,
    pub on: f64,
    pub oo: f64,
}

impl MotionNoiseStd {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.nn, self.no, self.on, self.oo]
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
    pub motion_noise_rate_pdf: MultivariateNormal,
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
        motion_noise: MotionNoiseStd,
    ) -> Self {
        Self {
            nu,
            omega,
            prev_nu: 0.0,
            prev_omega: 0.0,
            time_interval,
            radius,
            particles: vec![Particle::new(init_pose); particle_num],
            motion_noise_rate_pdf: MultivariateNormal::new(&motion_noise.to_vec()),
            pose_records: vec![vec![init_pose; particle_num]],
        }
    }
    pub fn update_motion(&mut self, prev_nu: f64, prev_omega: f64) {
        let mut poses = vec![];
        for p in self.particles.iter_mut() {
            let ns = self.motion_noise_rate_pdf.sample();
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
            p.pose = state_transition(p.pose, noised_nu, noised_omega, self.time_interval);
            poses.push(p.pose.clone());
        }
        self.pose_records.push(poses);
    }
    pub fn decision(&mut self) {
        self.update_motion(self.prev_nu, self.prev_omega);
        self.prev_nu = self.nu;
        self.prev_omega = self.omega;
    }
}
