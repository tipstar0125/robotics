use crate::agent::Pose;

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
pub struct Estimator {
    pub nu: f64,
    pub omega: f64,
    pub radius: f64,
    pub particles: Vec<Particle>,
}

impl Estimator {
    pub fn new(nu: f64, omega: f64, radius: f64, init_pose: Pose, num: usize) -> Self {
        Self {
            nu,
            omega,
            radius,
            particles: vec![Particle::new(init_pose); num],
        }
    }
}
