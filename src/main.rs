#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

mod agent;
mod camera;
mod common;
mod estimator;
mod motion;
mod multivariate_normal;
mod plot;
mod vis;

use agent::{Agent, Pose};
use common::{convert_radian_in_range, Coord};
use eframe::egui::Color32;
use estimator::{Estimator, MotionNoisePdf, ObservationNoiseStd};
use multivariate_normal::MyMultivariateNormal;
use plot::scatter;
use proconio::input;
use rand::{distributions::WeightedIndex, SeedableRng};
use rand_distr::Distribution;
use rand_pcg::Pcg64Mcg;
use std::f64::consts::PI;

fn main() {
    let input = Input {
        time_span: 30.0,
        time_interval: 0.1,
        height: 10,
        width: 10,
        landmarks: vec![
            Coord { x: -4.0, y: 2.0 },
            Coord { x: 2.0, y: -3.0 },
            Coord { x: 3.0, y: 3.0 },
        ],
    };

    let max_turn = (input.time_span / input.time_interval) as usize;

    let init_pose = Pose {
        coord: Coord { x: 0.0, y: 0.0 },
        theta: 0.0,
    };
    let nu = 0.2;
    let omega = 10.0_f64.to_radians();
    let radius = 0.2;

    let mut agent = Agent::new(
        0, // id
        radius, init_pose, nu, omega,
    );
    agent.set_motion_noise(5.0, PI / 60.0);
    agent.set_motion_bias(0.1, 0.1);
    agent.set_camera_noise(0.1, PI / 90.0);
    agent.set_camera_bias(0.1, PI / 90.0);

    let particle_num = 100;
    let mut estimator = Estimator::new(
        nu,
        omega,
        input.time_interval,
        radius,
        init_pose,
        particle_num,
        MotionNoisePdf::new(0.19, 0.001, 0.13, 0.2),
        ObservationNoiseStd {
            distance_rate_std: 0.14,
            direction_std: 0.05,
        },
    );

    for _ in 0..max_turn {
        let obs = agent.action(input.time_interval, &input.landmarks);
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
}

pub struct Output {
    agents: Vec<Agent>,
    estimator: Estimator,
}
