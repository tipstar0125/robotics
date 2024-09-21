#![allow(dead_code)]

mod agent;
mod camera;
mod common;
mod estimator;
mod motion;
mod normal;
mod vis;

use agent::{Agent, Pose};
use common::{convert_radian_in_range, Coord};
use estimator::{Estimator, MotionNoisePdf, ObservationNoiseStd};
use std::f64::consts::PI;

fn main() {
    let input = Input {
        time_span: 100.0,    // sec
        time_interval: 0.1, // sec
        height: 10,
        width: 10,
        landmarks: vec![
            Coord { x: -4.0, y: 2.0 },
            Coord { x: 2.0, y: -3.0 },
            Coord { x: 3.0, y: 3.0 },
        ],
        // ロボット初期姿勢
        init_pose: Pose {
            coord: Coord { x: 0.0, y: 0.0 },
            theta: 0.0,
        },
        radius: 0.2,                  // ロボット半径, m
        nu: 0.2,                      // ロボットの前方方向の速度, m/s
        omega: 10.0_f64.to_radians(), // ロボットの中心の角速度, rad/s
    };

    let mut agent = Agent::new(
        0, // 乱数シード
        input.time_interval,
        input.init_pose,
        input.radius,
        input.nu,
        input.omega,
    );

    // 動作ノイズ(実際は未知のパラメータ)
    let noise_per_meter = 5.0; // 道のりあたりに踏みつける小石の期待値
    let noise_std = PI / 60.0; // 小石を踏んだ時にずれる角度の確率密度関数(正規分布)
    agent.set_motion_noise(noise_per_meter, noise_std);

    // 観測ノイズ(実際は未知のパラメータ)
    let distance_noise_rate = 0.1; // 単位観測長当たりの観測距離ノイズの標準偏差
    let direction_noise = PI / 90.0; // 観測角度ノイズの標準偏差
    agent.set_camera_noise(distance_noise_rate, direction_noise);

    let particle_num = 100;
    let mut estimator = Estimator::new(
        input.time_interval,
        input.init_pose,
        input.radius,
        input.nu,
        input.omega,
        particle_num,
        MotionNoisePdf::new(0.19, 0.001, 0.13, 0.2),
        ObservationNoiseStd {
            distance_rate_std: 0.14,
            direction_std: 0.05,
        },
    );

    let max_turn = (input.time_span / input.time_interval) as usize;
    for _ in 0..max_turn {
        let obs = agent.action(&input.landmarks);
        estimator.decision(&obs, &input.landmarks);
    }

    let output = Output {
        agents: vec![agent],
        estimator,
    };
    vis::visualizer(input, output, max_turn);
}

pub struct Input {
    time_span: f64,
    time_interval: f64,
    height: usize,
    width: usize,
    landmarks: Vec<Coord>,
    init_pose: Pose,
    radius: f64,
    nu: f64,
    omega: f64,
}

pub struct Output {
    agents: Vec<Agent>,
    estimator: Estimator,
}
