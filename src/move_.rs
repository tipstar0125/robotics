use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Exp, Normal, Uniform};

use crate::{agent::Pose, common::Coord};

const PI: f64 = std::f64::consts::PI;

#[derive(Debug)]
pub struct Move {
    pub noise: MoveNoise,
    pub bias: MoveBias,
    pub stuck: Stuck,
    pub kidnap: Kidnap,
}

impl Move {
    pub fn new() -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        Self {
            noise: MoveNoise::new(&mut rng, 0.0, 0.0),
            bias: MoveBias::new(&mut rng, 0.0, 0.0),
            stuck: Stuck::new(&mut rng, f64::INFINITY, 0.0),
            kidnap: Kidnap::new(&mut rng, 1e100, 0.0, 0.0),
        }
    }
    pub fn set_noise(&mut self, rng: &mut ChaCha20Rng, noise_per_meter: f64, noise_std: f64) {
        self.noise = MoveNoise::new(rng, noise_per_meter, noise_std);
    }
    pub fn set_bias(
        &mut self,
        rng: &mut ChaCha20Rng,
        nu_bias_rate_std: f64,
        omega_bias_rate_std: f64,
    ) {
        self.bias = MoveBias::new(rng, nu_bias_rate_std, omega_bias_rate_std);
    }
    pub fn set_stuck(
        &mut self,
        rng: &mut ChaCha20Rng,
        expected_stuck_time: f64,
        expected_escape_time: f64,
    ) {
        self.stuck = Stuck::new(rng, expected_stuck_time, expected_escape_time);
    }
    pub fn set_kidnap(
        &mut self,
        rng: &mut ChaCha20Rng,
        expected_kidnap_time: f64,
        width: f64,
        height: f64,
    ) {
        self.kidnap = Kidnap::new(rng, expected_kidnap_time, width, height);
    }
    pub fn state_transition_with_noise(
        &mut self,
        rng: &mut ChaCha20Rng,
        pose: &mut Pose,
        mut nu: f64,
        mut omega: f64,
        radius: f64,
        dt: f64,
    ) {
        self.bias.on(&mut nu, &mut omega);
        if self.stuck.occur(rng, dt) {
            nu = 0.0;
            omega = 0.0;
        }
        *pose = state_transition(*pose, nu, omega, dt);
        pose.theta += self.noise.occur(rng, nu * dt + radius * omega.abs() * dt);
        self.kidnap.occur(rng, dt, pose);
    }
}

pub fn state_transition(pose: Pose, nu: f64, omega: f64, dt: f64) -> Pose {
    let delta = if omega.abs() < 1e-10 {
        Pose {
            coord: Coord {
                x: nu * pose.theta.cos() * dt,
                y: nu * pose.theta.sin() * dt,
            },
            theta: omega * dt,
        }
    } else {
        Pose {
            coord: Coord {
                x: nu / omega * ((pose.theta + omega * dt).sin() - pose.theta.sin()),
                y: nu / omega * (-(pose.theta + omega * dt).cos() + pose.theta.cos()),
            },
            theta: omega * dt,
        }
    };
    pose + delta
}

#[derive(Debug)]
pub struct MoveNoise {
    noise_per_meter: f64,     // 道のりあたりに踏みつける小石の期待値
    noise_std: f64,           // ロボットが小石を踏んだときにずれる向きの標準偏差
    noise_pdf: Exp<f64>,      // 小石を踏む確率密度関数(指数分布)
    theta_noise: Normal<f64>, // 小石を踏んだ時にずれる角度の確率密度関数(正規分布)
    dist_until_noise: f64,    // 小石を踏むまでの距離
}

impl MoveNoise {
    pub fn new(rng: &mut ChaCha20Rng, noise_per_meter: f64, noise_std: f64) -> Self {
        let noise_pdf = Exp::new(noise_per_meter).unwrap();
        let dist_until_noise = noise_pdf.sample(rng);
        Self {
            noise_per_meter,
            noise_std,
            noise_pdf: Exp::new(noise_per_meter).unwrap(),
            theta_noise: Normal::new(0.0, noise_std).unwrap(),
            dist_until_noise,
        }
    }
    pub fn occur(&mut self, rng: &mut ChaCha20Rng, dist: f64) -> f64 {
        self.dist_until_noise -= dist;
        if self.dist_until_noise <= 0.0 {
            self.dist_until_noise += self.noise_pdf.sample(rng);
            self.theta_noise.sample(rng)
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct MoveBias {
    pub nu_bias_rate_std: f64,
    pub omega_bias_rate_std: f64,
    pub nu_bias: f64,
    pub omega_bias: f64,
}

impl MoveBias {
    pub fn new(rng: &mut ChaCha20Rng, nu_bias_rate_std: f64, omega_bias_rate_std: f64) -> Self {
        Self {
            nu_bias_rate_std,
            omega_bias_rate_std,
            nu_bias: Normal::new(1.0, nu_bias_rate_std).unwrap().sample(rng),
            omega_bias: Normal::new(1.0, omega_bias_rate_std).unwrap().sample(rng),
        }
    }
    pub fn on(&self, nu: &mut f64, omega: &mut f64) {
        *nu *= self.nu_bias;
        *omega *= self.omega_bias;
    }
}

#[derive(Debug)]
pub struct Stuck {
    pub expected_stuck_time: f64,
    pub expected_escape_time: f64,
    pub stuck_pdf: Exp<f64>,  //  スタックするまで時間の確率密度関数(指数分布)
    pub escape_pdf: Exp<f64>, //  スタックから逃れるまでの時間の確率密度関数(指数分布)
    pub time_until_stuck: f64, // スタックするまでの時間
    pub time_until_escape: f64, // スタックから逃れるまでの時間
    pub is_stuck: bool,
}

impl Stuck {
    pub fn new(rng: &mut ChaCha20Rng, expected_stuck_time: f64, expected_escape_time: f64) -> Self {
        let stuck_pdf = Exp::new(1.0 / expected_stuck_time).unwrap();
        let escape_pdf = Exp::new(1.0 / expected_escape_time).unwrap();
        Self {
            expected_stuck_time,
            expected_escape_time,
            stuck_pdf,
            escape_pdf,
            time_until_stuck: stuck_pdf.sample(rng),
            time_until_escape: escape_pdf.sample(rng),
            is_stuck: false,
        }
    }
    pub fn occur(&mut self, rng: &mut ChaCha20Rng, dt: f64) -> bool {
        if self.is_stuck {
            self.time_until_escape -= dt;
            if self.time_until_escape <= 0.0 {
                self.time_until_escape += self.escape_pdf.sample(rng);
                self.is_stuck = false;
            }
        } else {
            self.time_until_stuck -= dt;
            if self.time_until_stuck <= 0.0 {
                self.time_until_stuck += self.stuck_pdf.sample(rng);
                self.is_stuck = true;
            }
        }
        self.is_stuck
    }
}

#[derive(Debug)]
pub struct Kidnap {
    pub expected_kidnap_time: f64,
    pub kidnap_pdf: Exp<f64>,   // 誘拐されるまでの確率密度関数(指数分布)
    pub time_until_kidnap: f64, // 誘拐されるまでの時間
    pub kidnap_x_dist: Uniform<f64>, // 誘拐された時のx座標の確率密度関数(一様分布)
    pub kidnap_y_dist: Uniform<f64>, // 誘拐された時のy座標の確率密度関数(一様分布)
    pub kidnap_theta_dist: Uniform<f64>, // 誘拐された時の向きの確率密度関数(一様分布)
}

impl Kidnap {
    pub fn new(rng: &mut ChaCha20Rng, expected_kidnap_time: f64, width: f64, height: f64) -> Self {
        let kidnap_range_x = -width / 2.0..=width / 2.0;
        let kidnap_range_y = -height / 2.0..=height / 2.0;
        let kidnap_pdf = Exp::new(1.0 / expected_kidnap_time).unwrap();
        let kidnap_x_dist = Uniform::from(kidnap_range_x);
        let kidnap_y_dist = Uniform::from(kidnap_range_y);
        let kidnap_theta_dist = Uniform::from(0.0..2.0 * PI);
        Self {
            expected_kidnap_time,
            kidnap_pdf,
            time_until_kidnap: kidnap_pdf.sample(rng),
            kidnap_x_dist,
            kidnap_y_dist,
            kidnap_theta_dist,
        }
    }
    pub fn occur(&mut self, rng: &mut ChaCha20Rng, dt: f64, pose: &mut Pose) {
        self.time_until_kidnap -= dt;
        if self.time_until_kidnap <= 0.0 {
            self.time_until_kidnap += self.kidnap_pdf.sample(rng);
            pose.coord.x = self.kidnap_x_dist.sample(rng);
            pose.coord.y = self.kidnap_y_dist.sample(rng);
            pose.theta = self.kidnap_theta_dist.sample(rng);
        }
    }
}
