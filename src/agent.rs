use crate::{
    camera::{Camera, Observation},
    common::{convert_radian_in_range, Coord},
    motion::Motion,
};
use rand::prelude::*;
use rand_distr::{Distribution, Exp, Normal, Uniform};
use rand_pcg::Pcg64Mcg;
use std::f64::consts::PI;

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
    pub rng: Pcg64Mcg,
    pub time_interval: f64,
    pub pose: Pose,
    pub radius: f64, // ロボット半径
    pub nu: f64,     // ロボットの前方方向の速度
    pub omega: f64,  // ロボットの中心の角速度
    pub motion: Motion,
    pub camera: Camera,
    pub obs_records: Vec<Vec<Observation>>, //  ビジュアライザ用観測記録
    pub pose_records: Vec<Pose>,            //  ビジュアライザ用姿勢記録
}

impl Agent {
    pub fn new(
        id: u64,
        time_interval: f64,
        init_pose: Pose,
        radius: f64,
        nu: f64,
        omega: f64,
    ) -> Self {
        Agent {
            id,
            rng: Pcg64Mcg::seed_from_u64(id),
            time_interval,
            pose: init_pose,
            radius,
            nu,
            omega,
            motion: Motion::new(),         // 理想の動き
            camera: Camera::new(),         // 理想観測
            obs_records: vec![vec![]],     // t=0では観測はしない
            pose_records: vec![init_pose], // t=0は初期姿勢
        }
    }
    pub fn set_motion_noise(&mut self, noise_per_meter: f64, noise_std: f64) {
        self.motion
            .set_noise(&mut self.rng, noise_per_meter, noise_std);
    }
    pub fn set_motion_bias(&mut self, nu_bias_rate_std: f64, omega_bias_rate_std: f64) {
        self.motion
            .set_bias(&mut self.rng, nu_bias_rate_std, omega_bias_rate_std);
    }
    pub fn set_stuck(&mut self, expected_stuck_time: f64, expected_escape_time: f64) {
        self.motion
            .set_stuck(&mut self.rng, expected_stuck_time, expected_escape_time);
    }
    pub fn set_kidnap(&mut self, expected_kidnap_time: f64, width: f64, height: f64) {
        self.motion
            .set_kidnap(&mut self.rng, expected_kidnap_time, width, height);
    }
    pub fn set_camera_noise(&mut self, distance_noise_rate: f64, direction_noise: f64) {
        self.camera.set_noise(distance_noise_rate, direction_noise);
    }
    pub fn set_camera_bias(&mut self, distance_bias_rate_std: f64, direction_bias_std: f64) {
        self.camera
            .set_bias(&mut self.rng, distance_bias_rate_std, direction_bias_std);
    }
    pub fn set_camera_phantom(&mut self, prob: f64, width: f64, height: f64) {
        self.camera.set_phantom(prob, width, height);
    }
    pub fn set_camera_oversight(&mut self, prob: f64) {
        self.camera.set_oversight(prob);
    }
    pub fn set_camera_occlusion(&mut self, prob: f64) {
        self.camera.set_occlusion(prob);
    }
    pub fn action(&mut self, landmarks: &[Coord]) -> Vec<Observation> {
        self.motion.state_transition_with_noise(
            &mut self.rng,
            self.time_interval,
            &mut self.pose,
            self.radius,
            self.nu,
            self.omega,
        );
        self.pose_records.push(self.pose);
        let obs = self.camera.observe(&mut self.rng, self.pose, landmarks);
        self.obs_records.push(obs.clone());
        obs
    }
}
