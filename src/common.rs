pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 1.0
        }
        #[cfg(not(feature = "local"))]
        {
            ms - STIME
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

impl std::fmt::Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x: {}, y: {}", self.x, self.y)?;
        Ok(())
    }
}

impl Coord {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

const PI: f64 = std::f64::consts::PI;

pub fn convert_radian_in_range(mut radian: f64) -> f64 {
    while radian > PI {
        radian -= 2.0 * PI;
    }
    while radian < -PI {
        radian += 2.0 * PI;
    }
    radian
}
