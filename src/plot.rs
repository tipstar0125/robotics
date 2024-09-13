use plotly::color::{NamedColor, Rgb};
use plotly::common::{Marker, Mode};
use plotly::layout::{Axis, ColorAxis, TicksDirection};
use plotly::{Layout, Plot, Scatter};

pub fn scatter(x: Vec<f64>, y: Vec<f64>, x_range: Vec<i32>, y_range: Vec<i32>, marker_size: usize) {
    let trace = Scatter::new(x, y)
        .mode(Mode::Markers)
        .marker(Marker::new().size(marker_size));
    let mut plot = Plot::new();
    let layout = Layout::new()
        .width(800)
        .height(800)
        .x_axis(
            Axis::new()
                .range(x_range)
                .zero_line(false)
                .grid_color(NamedColor::LightGray)
                .line_color(NamedColor::Black)
                .mirror(true),
        )
        .y_axis(
            Axis::new()
                .range(y_range)
                .zero_line(false)
                .grid_color(NamedColor::LightGray)
                .line_color(NamedColor::Black)
                .mirror(true),
        );
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.show();
}
