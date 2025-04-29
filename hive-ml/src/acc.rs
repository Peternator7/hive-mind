use tch::Tensor;

#[derive(Default)]
pub struct Accumulator {
    count: usize,
    value: f64,
}

impl Accumulator {
    pub fn accumulate(&mut self, xs: &Tensor) {
        self.count += xs.numel();
        self.value += f64::try_from(xs.sum(None)).unwrap();
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        self.value / self.count as f64
    }
}
