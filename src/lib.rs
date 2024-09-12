use std::{ops::RangeInclusive, str::SplitInclusive};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type MyRng = ChaCha8Rng;

/// Samplers generate samples and allow for retrieving their probability.
/// In case of BRDF sampling, this will be a 2D continuous uniform distribution over the hemisphere.
/// In case of NEE, this will be discrete uniform (or weighted by emission) distribution over lights followed by continuous distribution over their specific points.
trait Sampler {
    /// Returns a random sample.
    fn sample(&mut self) -> f32;
    /// Returns the probability of picking this sample.
    fn probability(&self, sample: f32) -> f32;

    fn sample_and_probability(&mut self) -> (f32, f32) {
        let s = self.sample();
        let p = self.probability(s);
        (s, p)
    }
}

/// Simple 1D continuous sampler.
struct UniformSampler {
    /// Source RNG.
    rng: MyRng,
    /// Domain of the samples.
    range: RangeInclusive<f32>,
}

impl UniformSampler {
    /// Creates a new uniform sampler from the provided seed over provided range.
    pub fn new(seed: u128, range: RangeInclusive<f32>) -> Self {
        Self {
            rng: ChaCha8Rng::from_seed(seed.to_le_bytes().repeat(2).try_into().unwrap()),
            range,
        }
    }
}

impl Sampler for UniformSampler {
    fn sample(&mut self) -> f32 {
        self.rng.gen_range(self.range.clone())
    }

    fn probability(&self, _sample: f32) -> f32 {
        0.1
    }
}

/// Weighted reservoir sampling allows for selecting one sample out of many, based on their weights.
struct WeightedReservoirSampler {
    rng: MyRng,
    sample: Option<f32>,
    weights: f32,
}

impl WeightedReservoirSampler {
    /// Creates a new instance from the provided seed.
    pub fn new(seed: u128) -> Self {
        Self {
            rng: ChaCha8Rng::from_seed(seed.to_le_bytes().repeat(2).try_into().unwrap()),
            sample: None,
            weights: 0.0,
        }
    }

    /// Adds a new sample to the reservoir.
    pub fn add(&mut self, sample: f32, weight: f32) {
        self.weights += weight;
        if self.rng.gen_bool((weight / self.weights) as f64) {
            self.sample = Some(sample)
        }
    }

    /// Finalize the result.
    ///
    /// PANIC: This will panic if no samples were added.
    pub fn take(self) -> (f32, f32) {
        assert!(
            self.sample.is_some(),
            "Reservoir requires at least one sample."
        );
        (self.sample.unwrap(), self.weights)
    }
}

/// Helper for calculating mean value.
struct Mean {
    /// Sum of all samples.
    sum: f32,
    /// Count of all samples.
    count: usize,
}

impl Mean {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }

    /// Adds a new value for averaging.
    pub fn add(&mut self, value: f32) {
        self.sum += value;
        self.count += 1;
    }

    /// Returns the mean value of all provided values.
    ///
    /// PANIC: This will panic if no values were added.
    pub fn finalize(self) -> f32 {
        assert!(self.count > 0, "Mean requires at least one value.");
        self.sum / self.count as f32
    }
}

/// Importance Sampling integrator.
/// In case of uniform sampler, this is a Monte Carlo integrator.
struct IsIntegrator<'a> {
    sampler: &'a mut dyn Sampler,
    target: fn(f32) -> f32,
    sample_count: usize,
}

impl<'a> IsIntegrator<'a> {
    /// Creates a new integrator.
    pub fn new(sampler: &'a mut dyn Sampler, target: fn(f32) -> f32, sample_count: usize) -> Self {
        Self {
            sampler,
            target,
            sample_count,
        }
    }

    /// Calculates the result.
    pub fn integrate(self) -> f32 {
        let mut mean = Mean::new();
        for _ in 0..self.sample_count {
            let (s, p) = self.sampler.sample_and_probability();
            let w = 1.0 / p;
            mean.add((self.target)(s) * w);
        }
        mean.finalize()
    }
}

/// Resampled Importance Sampling integrator.
/// Each true sample for computatin is based on [`Self::approximate_sample_count`] approximations for better result.
struct RisIntegrator<'a> {
    sampler: &'a mut dyn Sampler,
    true_target: fn(f32) -> f32,
    true_sample_count: usize,
    approximate_target: fn(f32) -> f32,
    approximate_sample_count: usize,
}

impl<'a> RisIntegrator<'a> {
    /// Creates a new integrator.
    pub fn new(
        sampler: &'a mut dyn Sampler,
        true_target: fn(f32) -> f32,
        true_sample_count: usize,
        approximate_target: fn(f32) -> f32,
        approximate_sample_count: usize,
    ) -> Self {
        Self {
            sampler,
            true_target,
            true_sample_count,
            approximate_target,
            approximate_sample_count,
        }
    }

    /// Calculates the result.
    pub fn integrate(self) -> f32 {
        let mut mean = Mean::new();
        for _ in 0..self.true_sample_count {
            let mut reservoir = WeightedReservoirSampler::new(0);
            for _ in 0..self.approximate_sample_count {
                let (s, p) = self.sampler.sample_and_probability();
                let y = (self.approximate_target)(s);
                let w = 1.0 / p;
                let m = 1.0 / self.approximate_sample_count as f32;
                reservoir.add(s, m * y * w);
            }
            let (s, weights) = reservoir.take();
            let w = weights / (self.approximate_target)(s);
            mean.add((self.true_target)(s) * w);
        }
        mean.finalize()
    }
}

/// Numerical integrator.
/// Values of target function are calculated in the provided [`Self::range`] for evenly distributed [`Self::samples`].
struct NumericalIntegrator {
    range: RangeInclusive<f32>,
    samples: usize,
    target: fn(f32) -> f32,
}

impl NumericalIntegrator {
    /// Creates a new integrator.
    pub fn new(range: RangeInclusive<f32>, samples: usize, target: fn(f32) -> f32) -> Self {
        Self {
            range,
            samples,
            target,
        }
    }

    /// Calculates the result.
    pub fn integrate(self) -> f32 {
        let mut acc = 0.0;
        let start = *self.range.start();
        let end = *self.range.end();
        let length = end - start;
        let sample_to_x = (self.samples - 1) as f32;
        for sample in 0..self.samples {
            let t = sample as f32 / sample_to_x;
            let x = start + t * length;
            acc += (self.target)(x);
        }
        let width = length / self.samples as f32;
        acc * width
    }
}

#[cfg(test)]
mod tests {
    use plotters::{
        chart::ChartBuilder,
        prelude::{BitMapBackend, DiscreteRanged, IntoDrawingArea, IntoLinspace},
        series::LineSeries,
        style::{BLUE, WHITE},
    };

    use crate::{IsIntegrator, NumericalIntegrator, RisIntegrator, UniformSampler};

    /// The actual function we're trying to integrate.
    /// This function is usually more expensive to compute than [`approximate_target`],
    /// which is why we try to compute it less often.
    fn true_target(s: f32) -> f32 {
        let visible = (((3.5643 * s).sin() + (1.4352 * s).sin() + 0.5) >= 0.0) as u32 as f32;
        approximate_target(s) * visible
    }

    /// Simplified version of [`true_target`].
    fn approximate_target(s: f32) -> f32 {
        let inbound = (-5.0 < s && s < 5.0) as u32 as f32;
        let value = 1.0 + s.cos() + 3.0 * std::f32::consts::E.powf(-((s - 3.0) * (s - 3.0)));
        value * inbound
    }

    #[test]
    fn numerical_integral() {
        let true_sample_count = 1_000_000;
        let integrator = NumericalIntegrator::new(-5.0..=5.0, true_sample_count, true_target);
        println!("{:?}", integrator.integrate())
    }

    #[test]
    fn monte_carlo_integral() {
        let mut sampler = UniformSampler::new(0, -5.0..=5.0);
        let true_sample_count = 1_000_000;
        let integrator = IsIntegrator::new(&mut sampler, true_target, true_sample_count);
        println!("{:?}", integrator.integrate());
    }

    #[test]
    fn resampled_importance_sampling_integral() {
        let mut sampler = UniformSampler::new(0, -5.0..=5.0);
        let true_sample_count = 100;
        let approximate_sample_count = 10_000;
        let integrator = RisIntegrator::new(
            &mut sampler,
            true_target,
            true_sample_count,
            approximate_target,
            approximate_sample_count,
        );
        println!("{:?}", integrator.integrate());
    }

    /// Helper for plotting target functions.
    fn plot_fn(dst: &str, func: fn(f32) -> f32) -> Result<(), Box<dyn std::error::Error>> {
        let x_range: (f32, f32) = (-6.0, 6.0);
        let y_range: (f32, f32) = (-1.0, 4.0);
        let pixels_per_unit: u32 = 100;

        let x_size = x_range.1 - x_range.0;
        let y_size = y_range.1 - y_range.0;
        let root_area = BitMapBackend::new(
            dst,
            (
                x_size.ceil() as u32 * pixels_per_unit,
                y_size.ceil() as u32 * pixels_per_unit,
            ),
        )
        .into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let step = 1.0 / pixels_per_unit as f32;
        let x_axis = (x_range.0..x_range.1).step(step);

        let mut cc = ChartBuilder::on(&root_area)
            .margin(5.0)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        cc.draw_series(LineSeries::new(
            x_axis.values().map(|v| (v, func(v))),
            &BLUE,
        ))?;

        Ok(())
    }

    #[test]
    fn plot() {
        plot_fn("imgs/true_target.png", true_target).unwrap();
        plot_fn("imgs/approximate_target.png", approximate_target).unwrap();
    }
}
