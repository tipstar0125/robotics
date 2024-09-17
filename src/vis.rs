use crate::{convert_radian_in_range, Agent, Coord, Estimator, Input, Output, Pose};

use eframe::egui::{
    show_tooltip_at_pointer, widgets, Align2, CentralPanel, Color32, Context, FontFamily, FontId,
    Id, Key, Pos2, Rect, RichText, Slider, Stroke, TextEdit, Ui, Vec2,
};
use eframe::{run_native, App, CreationContext, Frame, NativeOptions, Storage, Theme};
use std::time::{Duration, Instant};

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 800.0;
const VIS_WIDTH: f32 = 600.0;
const VIS_HEIGHT: f32 = 600.0;
const OFFSET_WIDTH: f32 = (WIDTH - VIS_WIDTH) / 2.0;
const OFFSET_HEIGHT: f32 = (HEIGHT - VIS_HEIGHT) / 2.0;
const SPEED_MIN: usize = 1;
const SPEED_MAX: usize = 10;
const AGENT_COLORS: [Color32; 4] = [Color32::RED, Color32::BLUE, Color32::GREEN, Color32::BROWN];

pub struct Egui {
    input: Input,
    output: Output,
    turn: usize,
    max_turn: usize,
    checked: bool,
    play: bool,
    speed: usize,
    instant: Instant,
    cnt: usize,
}

impl Egui {
    fn new(input: Input, output: Output, max_turn: usize) -> Self {
        Egui {
            input,
            output,
            turn: 0,
            max_turn,
            checked: true,
            play: false,
            speed: 5,
            instant: Instant::now(),
            cnt: 0,
        }
    }
}

impl App for Egui {
    fn save(&mut self, _storage: &mut dyn Storage) {}
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        ctx.request_repaint_after(Duration::from_millis(5));
        if self.instant.elapsed() >= Duration::from_millis(10) {
            self.cnt += 1;
            if self.cnt % (SPEED_MIN + SPEED_MAX - self.speed) == 0
                && self.play
                && self.turn < self.max_turn
            {
                self.turn += 1;
            }
            self.instant = Instant::now();
        }

        CentralPanel::default().show(ctx, |ui| {
            let width = self.input.width;
            let height = self.input.height;
            let d = VIS_WIDTH / (height.max(width)) as f32;

            view_world(ui, &self.input, d);
            for (id, &coord) in self.input.landmarks.iter().enumerate() {
                view_landmark(ui, &self.input, d, id, coord);
            }

            view_estimator(ui, &self.input, d, &self.output.estimator, self.turn);

            for (idx, agent) in self.output.agents.iter().enumerate() {
                view_agent(ui, &self.input, d, agent, self.turn, AGENT_COLORS[idx]);
            }

            ui.horizontal(|ui| {
                ui.label(RichText::new("Turn: ").size(20.0));
                ui.add(Slider::new(&mut self.turn, 0..=self.max_turn));
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("Speed: ").size(20.0));
                ui.add(Slider::new(&mut self.speed, SPEED_MIN..=SPEED_MAX));
            });

            if ctx.input().key_released(Key::Space) {
                self.play = !self.play;
            };
            if self.turn == self.max_turn {
                self.play = false;
            }
            if ctx.input().key_pressed(Key::ArrowRight) && self.turn < self.max_turn {
                self.turn += 1;
            };
            if ctx.input().key_pressed(Key::ArrowLeft) && self.turn > 0 {
                self.turn -= 1;
            };
        });
    }
}

pub fn visualizer(input: Input, output: Output, max_turn: usize) {
    let options = NativeOptions {
        initial_window_size: Some((WIDTH, HEIGHT).into()),
        initial_window_pos: Some(Pos2 { x: 100.0, y: 100.0 }),
        resizable: false,
        default_theme: Theme::Light,
        ..NativeOptions::default()
    };
    let gui = Egui::new(input, output, max_turn);
    run_native("visualizer", options, Box::new(|_cc| Box::new(gui)));
}

// 0 <= val <= 1
pub fn color32(mut val: f32) -> Color32 {
    val = val.min(1.0);
    val = val.max(0.0);
    let (r, g, b) = if val < 0.5 {
        let x = val * 2.0;
        (
            30. * (1.0 - x) + 144. * x,
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
        )
    } else {
        let x = val * 2.0 - 1.0;
        (
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
            30. * (1.0 - x) + 70. * x,
        )
    };
    Color32::from_rgb(r.round() as u8, g.round() as u8, b.round() as u8)
}
pub fn txt(ui: &mut Ui, txt: &str, mut pos: Pos2, size: f32, color: Color32) {
    pos.x += OFFSET_WIDTH;
    pos.y += OFFSET_HEIGHT;
    let anchor = Align2::CENTER_CENTER;
    let font_id = FontId::new(size, FontFamily::Monospace);
    ui.painter().text(pos, anchor, txt, font_id, color);
}
pub fn line(ui: &mut Ui, mut pos1: Pos2, mut pos2: Pos2, color: Color32, stroke_width: f32) {
    pos1.x += OFFSET_WIDTH;
    pos2.x += OFFSET_WIDTH;
    pos1.y += OFFSET_HEIGHT;
    pos2.y += OFFSET_HEIGHT;
    let points = [pos1, pos2];
    let stroke = Stroke {
        width: stroke_width,
        color,
    };
    ui.painter().line_segment(points, stroke);
}
pub fn rect(
    ui: &mut Ui,
    mut pos1: Pos2,
    mut pos2: Pos2,
    fill_color: Color32,
    stroke_color: Color32,
) -> Rect {
    pos1.x += OFFSET_WIDTH;
    pos2.x += OFFSET_WIDTH;
    pos1.y += OFFSET_HEIGHT;
    pos2.y += OFFSET_HEIGHT;

    let rect = Rect {
        min: pos1,
        max: pos2,
    };
    let rounding = 0.0;
    let stroke = Stroke {
        width: 3.0,
        color: stroke_color,
    };
    ui.painter().rect(rect, rounding, fill_color, stroke);
    rect
}
pub fn circle(
    ui: &mut Ui,
    mut center: Pos2,
    radius: f32,
    fill_color: Color32,
    stroke_color: Color32,
) -> Rect {
    center.x += OFFSET_WIDTH;
    center.y += OFFSET_HEIGHT;
    let stroke = Stroke {
        width: 3.0,
        color: stroke_color,
    };
    ui.painter().circle(center, radius, fill_color, stroke);

    Rect {
        min: Pos2 {
            x: center.x - radius,
            y: center.y - radius,
        },
        max: Pos2 {
            x: center.x + radius,
            y: center.y + radius,
        },
    }
}
pub fn arrow(ui: &mut Ui, mut origin: Pos2, vec: Vec2, stroke_color: Color32, stroke_width: f32) {
    origin.x += OFFSET_WIDTH;
    origin.y += OFFSET_HEIGHT;
    let stroke = Stroke {
        width: stroke_width,
        color: stroke_color,
    };
    ui.painter().arrow(origin, vec, stroke);
}
pub fn view_world(ui: &mut Ui, input: &Input, d: f32) {
    let view_top_left_pos = Pos2 { x: 0.0, y: 0.0 };
    let view_bottom_right_pos = Pos2 {
        x: d * input.width as f32,
        y: d * input.height as f32,
    };

    let view_center_left_pos = Pos2 {
        x: 0.0,
        y: d * input.height as f32 / 2.0,
    };
    let view_center_right_pos = Pos2 {
        x: d * input.width as f32,
        y: d * input.height as f32 / 2.0,
    };

    let view_top_center_pos = Pos2 {
        x: d * input.width as f32 / 2.0,
        y: 0.0,
    };
    let view_bottom_center_pos = Pos2 {
        x: d * input.width as f32 / 2.0,
        y: d * input.height as f32,
    };

    line(
        ui,
        view_center_left_pos,
        view_center_right_pos,
        Color32::LIGHT_GRAY,
        3.0,
    );

    line(
        ui,
        view_top_center_pos,
        view_bottom_center_pos,
        Color32::LIGHT_GRAY,
        3.0,
    );

    rect(
        ui,
        view_top_left_pos,
        view_bottom_right_pos,
        Color32::TRANSPARENT,
        Color32::BLACK,
    );

    let offset = 50.0;
    let offset_scale = 15.0;
    let ticks = 1;

    txt(
        ui,
        "X",
        Pos2 {
            x: view_bottom_center_pos.x,
            y: view_bottom_center_pos.y + offset,
        },
        30.0,
        Color32::BLACK,
    );

    let mut x = 0;
    while x <= input.width as isize / 2 {
        txt(
            ui,
            &x.to_string(),
            Pos2 {
                x: view_bottom_center_pos.x + d * x as f32,
                y: view_bottom_center_pos.y + offset_scale,
            },
            15.0,
            Color32::BLACK,
        );
        txt(
            ui,
            &(-x).to_string(),
            Pos2 {
                x: view_bottom_center_pos.x + d * (-x) as f32,
                y: view_bottom_center_pos.y + offset_scale,
            },
            15.0,
            Color32::BLACK,
        );
        x += ticks;
    }

    txt(
        ui,
        "Y",
        Pos2 {
            x: -offset,
            y: view_center_left_pos.y,
        },
        30.0,
        Color32::BLACK,
    );

    let mut y = 0;
    while y <= input.height as isize / 2 {
        txt(
            ui,
            &y.to_string(),
            Pos2 {
                x: -offset_scale,
                y: view_center_left_pos.y + d * (-y) as f32,
            },
            15.0,
            Color32::BLACK,
        );
        txt(
            ui,
            &(-y).to_string(),
            Pos2 {
                x: -offset_scale,
                y: view_center_left_pos.y + d * y as f32,
            },
            15.0,
            Color32::BLACK,
        );
        y += ticks;
    }
}
pub fn view_agent(ui: &mut Ui, input: &Input, d: f32, agent: &Agent, turn: usize, color: Color32) {
    let pose = agent.pose_records[turn];
    let x_center = d * input.width as f32 / 2.0;
    let y_center = d * input.height as f32 / 2.0;
    let center_pos = Pos2 {
        x: x_center + d * pose.coord.x as f32,
        y: y_center + d * (-pose.coord.y) as f32,
    };

    // エージェント描画
    let radius = agent.radius as f32 * d;
    let rect = circle(ui, center_pos, radius, Color32::TRANSPARENT, color);
    let dir_pos = Pos2 {
        x: center_pos.x + radius * pose.theta.cos() as f32,
        y: center_pos.y - radius * pose.theta.sin() as f32,
    };
    line(ui, center_pos, dir_pos, color, 3.0);

    // 観測結果描画
    for obs in agent.obs_records[turn].iter() {
        let pos = Pos2 {
            x: center_pos.x + d * (obs.dist * (obs.angle + pose.theta).cos()) as f32,
            y: center_pos.y - d * (obs.dist * (obs.angle + pose.theta).sin()) as f32,
        };
        line(ui, center_pos, pos, Color32::LIGHT_RED, 2.0);
    }

    // エージェント軌跡描画
    let trajectory_num = 200;
    let trajectory_st = if turn <= trajectory_num {
        0
    } else {
        turn - trajectory_num
    };
    for i in trajectory_st..turn {
        let pos = Pos2 {
            x: x_center + d * agent.pose_records[i].coord.x as f32,
            y: y_center + d * (-agent.pose_records[i].coord.y) as f32,
        };
        circle(ui, pos, 1.0, color, Color32::TRANSPARENT);
    }

    let hover_pos = ui.input().pointer.hover_pos();
    let theta = convert_radian_in_range(pose.theta);
    if let Some(hover_pos) = hover_pos {
        if rect.contains(hover_pos) {
            show_tooltip_at_pointer(ui.ctx(), Id::new("hover tooltip"), |ui| {
                ui.label(format!(
                    "id{}: (x, y, theta) = ({:.2}, {:.2}, {:.1})",
                    agent.id,
                    pose.coord.x,
                    pose.coord.y,
                    theta.to_degrees(),
                ));
            });
        }
    }
}
pub fn view_landmark(ui: &mut Ui, input: &Input, d: f32, id: usize, coord: Coord) {
    let x_center = d * input.width as f32 / 2.0;
    let y_center = d * input.height as f32 / 2.0;
    let center_pos = Pos2 {
        x: x_center + d * coord.x as f32,
        y: y_center + d * (-coord.y) as f32,
    };
    let radius = 5.0;
    let rect = circle(ui, center_pos, radius, Color32::YELLOW, Color32::BLACK);
    let hover_pos = ui.input().pointer.hover_pos();
    if let Some(hover_pos) = hover_pos {
        if rect.contains(hover_pos) {
            show_tooltip_at_pointer(ui.ctx(), Id::new("hover tooltip"), |ui| {
                ui.label(format!(
                    "id{}: (x, y) = ({:.2}, {:.2})",
                    id, coord.x, coord.y
                ));
            });
        }
    }
}
pub fn view_estimator(ui: &mut Ui, input: &Input, d: f32, estimator: &Estimator, turn: usize) {
    let x_center = d * input.width as f32 / 2.0;
    let y_center = d * input.height as f32 / 2.0;
    let size = estimator.radius as f32 * d;

    for (pose, &weight) in estimator.pose_records[turn]
        .iter()
        .zip(estimator.weight_records[turn].iter())
    {
        let origin = Pos2 {
            x: x_center + d * pose.coord.x as f32,
            y: y_center + d * (-pose.coord.y) as f32,
        };
        let vec = Vec2 {
            x: size * pose.theta.cos() as f32,
            y: -size * pose.theta.sin() as f32,
        };
        let opacity = ((weight * 255.0) as u8).min(255);
        let color = Color32::from_rgba_unmultiplied(0, 0, 255, opacity);
        arrow(ui, origin, vec, color, 2.0);
    }
}
