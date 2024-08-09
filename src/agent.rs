use crate::{
    camera::{Camera, Observation},
    common::{convert_radian_in_range, Coord},
    move_::Move,
};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Exp, Normal, Uniform};

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub coord: Coord,
    pub theta: f64,
}

impl std::fmt::Display for Pose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "x: {}, y: {}, theta: {}",
            self.coord.x, self.coord.y, self.theta
        )?;
        Ok(())
    }
}

impl std::ops::Add<Pose> for Pose {
    type Output = Pose;
    fn add(self, rhs: Pose) -> Self::Output {
        Pose {
            coord: Coord {
                x: self.coord.x + rhs.coord.x,
                y: self.coord.y + rhs.coord.y,
            },
            theta: self.theta + rhs.theta,
        }
    }
}

#[derive(Debug)]
pub struct Agent {
    pub id: u64,
    pub radius: f64, // ロボット半径
    pub pose: Pose,
    pub nu: f64,    // ロボットの前方方向の速度
    pub omega: f64, // ロボットの中心の角速度
    pub move_: Move,
    pub camera: Camera,
    pub rng: ChaCha20Rng,
    pub obs_records: Vec<Vec<Observation>>,
    pub pose_records: Vec<Pose>,
}

impl Agent {
    pub fn new(id: u64, radius: f64, pose: Pose, nu: f64, omega: f64) -> Self {
        Agent {
            id,
            radius,
            pose,
            nu,
            omega,
            move_: Move::new(),
            camera: Camera::new(),
            rng: ChaCha20Rng::seed_from_u64(id),
            obs_records: vec![vec![]],
            pose_records: vec![pose],
        }
    }
    pub fn set_move_noise(&mut self, noise_per_meter: f64, noise_std: f64) {
        self.move_
            .set_noise(&mut self.rng, noise_per_meter, noise_std);
    }
    pub fn set_move_bias(&mut self, nu_bias_rate_std: f64, omega_bias_rate_std: f64) {
        self.move_
            .set_bias(&mut self.rng, nu_bias_rate_std, omega_bias_rate_std);
    }
    pub fn set_stuck(&mut self, expected_stuck_time: f64, expected_escape_time: f64) {
        self.move_
            .set_stuck(&mut self.rng, expected_stuck_time, expected_escape_time);
    }
    pub fn set_kidnap(&mut self, expected_kidnap_time: f64, width: f64, height: f64) {
        self.move_
            .set_kidnap(&mut self.rng, expected_kidnap_time, width, height);
    }
    pub fn action(&mut self, dt: f64, landmarks: &[Coord]) {
        self.move_.state_transition_with_noise(
            &mut self.rng,
            &mut self.pose,
            self.nu,
            self.omega,
            self.radius,
            dt,
        );
        self.pose_records.push(self.pose);
        self.obs_records
            .push(self.camera.observe(self.pose, landmarks));
    }
}
