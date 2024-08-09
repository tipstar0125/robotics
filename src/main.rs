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
mod move_;
mod vis;
use agent::{Agent, Pose};
use common::{convert_radian_in_range, Coord};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use eframe::egui::Color32;
use proconio::input;

const PI: f64 = std::f64::consts::PI;

fn main() {
    let input = Input {
        time_span: 30.0,
        time_interval: 0.1,
        height: 10,
        width: 10,
        landmarks: vec![
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 3.0, y: 2.0 },
            Coord { x: -3.0, y: -2.0 },
        ],
    };

    let max_turn = (input.time_span / input.time_interval) as usize;
    let mut agents = vec![];

    for i in 0..4 {
        let mut agent = Agent::new(
            i,   // id
            0.2, // radius
            Pose {
                coord: Coord { x: 0.0, y: 0.0 },
                theta: 0.0,
            },
            0.2,                   // nu
            10.0_f64.to_radians(), // omega
        );
        agent.set_move_noise(5.0, 3.0_f64.to_radians());
        // agent.set_move_bias(0.1, 0.1);
        // agent.set_stuck(60.0, 60.0);
        // agent.set_kidnap(5.0, input.width as f64, input.height as f64);

        for _ in 0..=max_turn {
            agent.action(input.time_interval, &input.landmarks);
        }
        agents.push(agent);
    }
    let output = Output { agents };
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
}
