//! Soft limiter, ported from the one OmniConverter already ships. \[1\]

#[derive(Debug, Clone)]
pub struct Limiter {
    loudness_l: f64,
    loudness_r: f64,
    velocity_l: f64,
    velocity_r: f64,
    attack: f64,
    falloff: f64,
    min_thresh: f64,
    /// 0 disables the effect, 1 is full strength.
    pub strength: f64,
    /// Optional high-frequency energy limiter, off by default.
    pub reduce_high_pitch: bool,
    velocity_thresh: f64,
    first_sample: bool,
}

impl Limiter {
    pub fn new(sample_rate: u32) -> Self {
        Limiter {
            loudness_l: 1.0,
            loudness_r: 1.0,
            velocity_l: 0.0,
            velocity_r: 0.0,
            attack: 100.0,
            falloff: sample_rate as f64 / 3.0,
            min_thresh: 0.4,
            strength: 1.0,
            reduce_high_pitch: false,
            velocity_thresh: 1.0,
            first_sample: true,
        }
    }

    pub fn with_frequency_reduce(mut self, frequency_reduce: f64) -> Self {
        self.reduce_high_pitch = true;
        self.velocity_thresh = 1.0 / frequency_reduce;
        self
    }

    /// Process one interleaved stereo block in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len() % 2, 0);
        let attack = self.attack;
        let falloff = self.falloff;

        for i in (0..buf.len()).step_by(2) {
            let in_l = buf[i] as f64;
            let in_r = buf[i + 1] as f64;
            let l_abs = in_l.abs();
            let r_abs = in_r.abs();

            self.loudness_l = if self.loudness_l > l_abs {
                (self.loudness_l * falloff + l_abs) / (falloff + 1.0)
            } else {
                (self.loudness_l * attack + l_abs) / (attack + 1.0)
            };
            self.loudness_r = if self.loudness_r > r_abs {
                (self.loudness_r * falloff + r_abs) / (falloff + 1.0)
            } else {
                (self.loudness_r * attack + r_abs) / (attack + 1.0)
            };

            if self.loudness_l < self.min_thresh {
                self.loudness_l = self.min_thresh;
            }
            if self.loudness_r < self.min_thresh {
                self.loudness_r = self.min_thresh;
            }

            let mut l = in_l / (self.loudness_l * self.strength + 2.0 * (1.0 - self.strength)) / 2.0;
            let mut r = in_r / (self.loudness_r * self.strength + 2.0 * (1.0 - self.strength)) / 2.0;

            if !self.first_sample {
                let dl = (in_l - l).abs();
                let dr = (in_r - r).abs();
                self.velocity_l = if self.velocity_l > dl {
                    (self.velocity_l * falloff + dl) / (falloff + 1.0)
                } else {
                    (self.velocity_l * attack + dl) / (attack + 1.0)
                };
                self.velocity_r = if self.velocity_r > dr {
                    (self.velocity_r * falloff + dr) / (falloff + 1.0)
                } else {
                    (self.velocity_r * attack + dr) / (attack + 1.0)
                };
            }
            self.first_sample = false;

            if self.reduce_high_pitch {
                if self.velocity_l > self.velocity_thresh {
                    l = l / self.velocity_l * self.velocity_thresh;
                }
                if self.velocity_r > self.velocity_thresh {
                    r = r / self.velocity_r * self.velocity_thresh;
                }
            }

            buf[i] = l as f32;
            buf[i + 1] = r as f32;
        }
    }
}

/// Hard clamp, always applied last so nothing leaves the renderer out of range. \[2\]
pub fn clamp_block(buf: &mut [f32]) -> u64 {
    let mut n = 0u64;
    for v in buf.iter_mut() {
        if *v > 1.0 || *v < -1.0 {
            n += 1;
            *v = v.clamp(-1.0, 1.0);
        }
    }
    n
}

// [3]

/// Which limiter runs on the mixed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterMode {
    /// No limiting. `clamp_block` still runs, so loud material hard-clips.
    Off,
    /// The port of the realtime limiter OmniConverter ships, above. \[4\]
    Omni,
    /// Lookahead true-peak brickwall. Guarantees the output never exceeds the \[5\]
    Brickwall,
}

impl LimiterMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" => Some(LimiterMode::Off),
            "omni" | "kiva" | "legacy" => Some(LimiterMode::Omni),
            "brickwall" | "brick" | "peak" => Some(LimiterMode::Brickwall),
            _ => None,
        }
    }
}

/// Phases of the true-peak oversampler, and taps per phase.
const TP_PHASES: usize = 4;
const TP_TAPS: usize = 8;

/// Lookahead true-peak brickwall limiter. \[6\]
pub struct Brickwall {
    ceiling: f64,
    look: usize,
    release_coef: f64,
    /// Attack and release coefficients of the sustained stage. Both zero when \[7\]
    sustain_atk: f64,
    sustain_rel: f64,
    /// Gain the sustained stage is holding: a slow envelope of the \[8\]
    g_slow: f64,
    /// Gain the fast stage is holding, on top of the sustained one.
    g_fast: f64,
    /// Interleaved stereo delay line, `look + group` frames.
    delay: Vec<f32>,
    dpos: usize,
    /// Monotonic deque for the sliding maximum, decreasing, of (value, index).
    dq: std::collections::VecDeque<(f64, u64)>,
    /// Ring of `gr` values with a running sum, for the moving average.
    gr_ring: Vec<f64>,
    gr_pos: usize,
    gr_sum: f64,
    /// Release-only smoother state.
    gain: f64,
    /// Oversampler history, one per channel.
    hist: [[f64; TP_TAPS]; 2],
    /// Polyphase coefficients, indexed by phase then tap.
    poly: [[f64; TP_TAPS]; TP_PHASES],
    /// Frames the audio is delayed by: the lookahead plus the detector's own \[9\]
    delay_frames: usize,
    true_peak: bool,
    idx: u64,
    /// Largest true peak seen at the input, for reporting.
    pub peak_in: f64,
    /// Smallest gain the limiter had to apply, for reporting. **Not wired up** \[10\]
    pub min_gain: f64,
}

impl Brickwall {
    pub fn new(
        sample_rate: u32,
        ceiling: f64,
        lookahead_ms: f64,
        release_ms: f64,
        sustain_ms: f64,
        true_peak: bool,
    ) -> Self {
        let sr = sample_rate as f64;
        let look = ((lookahead_ms * 1e-3 * sr).round() as usize).max(1);
        let rel = (release_ms * 1e-3 * sr).max(1.0);
        // [11]
        let (sustain_atk, sustain_rel) = if sustain_ms > 0.0 {
            let a = (sustain_ms * 1e-3 * sr).max(1.0);
            (
                1.0 - (-1.0f64 / a).exp(),
                1.0 - (-1.0f64 / (a * 4.0)).exp(),
            )
        } else {
            (0.0, 0.0)
        };
        // [12]
        let release_coef = 1.0 - (-1.0 / rel).exp();
        // [13]
        let group = if true_peak { TP_TAPS / 2 - 1 } else { 0 };
        let delay_frames = look + group;
        Brickwall {
            ceiling,
            look,
            release_coef,
            sustain_atk,
            sustain_rel,
            g_slow: 1.0,
            g_fast: 1.0,
            delay: vec![0.0; delay_frames * 2],
            dpos: 0,
            dq: std::collections::VecDeque::with_capacity(look + 2),
            gr_ring: vec![1.0; look + 1],
            gr_pos: 0,
            gr_sum: (look + 1) as f64,
            gain: 1.0,
            hist: [[0.0; TP_TAPS]; 2],
            poly: design_polyphase(),
            delay_frames,
            true_peak,
            idx: 0,
            peak_in: 0.0,
            min_gain: 1.0,
        }
    }

    /// Frames of latency this adds to the render.
    pub fn latency(&self) -> usize {
        self.delay_frames
    }

    /// True peak of one frame: the largest magnitude of the 4x oversampled \[14\]
    fn detect(&mut self, l: f64, r: f64) -> f64 {
        if !self.true_peak {
            return l.abs().max(r.abs());
        }
        let mut peak = 0.0f64;
        for (ch, v) in [l, r].into_iter().enumerate() {
            let h = &mut self.hist[ch];
            h.copy_within(0..TP_TAPS - 1, 1);
            h[0] = v;
            for ph in &self.poly {
                let mut acc = 0.0;
                for (c, x) in ph.iter().zip(h.iter()) {
                    acc += c * x;
                }
                let a = acc.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
        peak
    }

    /// Process one interleaved stereo block in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len() % 2, 0);
        for i in (0..buf.len()).step_by(2) {
            let in_l = buf[i] as f64;
            let in_r = buf[i + 1] as f64;

            // Pull the delayed frame out, push the new one in.
            let d = self.dpos * 2;
            let out_l = self.delay[d] as f64;
            let out_r = self.delay[d + 1] as f64;
            self.delay[d] = buf[i];
            self.delay[d + 1] = buf[i + 1];
            self.dpos = (self.dpos + 1) % self.delay_frames;

            // Sliding maximum of the detected true peak.
            let det = self.detect(in_l, in_r);
            if det > self.peak_in {
                self.peak_in = det;
            }
            while let Some(&(v, _)) = self.dq.back() {
                if v <= det {
                    self.dq.pop_back();
                } else {
                    break;
                }
            }
            self.dq.push_back((det, self.idx));
            let oldest = self.idx.saturating_sub(self.look as u64);
            while let Some(&(_, j)) = self.dq.front() {
                if j < oldest {
                    self.dq.pop_front();
                } else {
                    break;
                }
            }
            let env = self.dq.front().map(|&(v, _)| v).unwrap_or(0.0);

            // The gain that envelope needs, then the moving average of it.
            let gr = if env > self.ceiling {
                self.ceiling / env
            } else {
                1.0
            };
            self.gr_sum += gr - self.gr_ring[self.gr_pos];
            self.gr_ring[self.gr_pos] = gr;
            self.gr_pos = (self.gr_pos + 1) % self.gr_ring.len();
            let ma = self.gr_sum / self.gr_ring.len() as f64;

            // [15]
            if self.sustain_atk > 0.0 {
                let c = if ma < self.g_slow {
                    self.sustain_atk
                } else {
                    self.sustain_rel
                };
                self.g_slow += (ma - self.g_slow) * c;
                self.g_slow = self.g_slow.clamp(1e-9, 1.0);
            }
            let req = (ma / self.g_slow).min(1.0);
            if req < self.g_fast {
                self.g_fast = req;
            } else {
                self.g_fast += (req - self.g_fast) * self.release_coef;
                if self.g_fast > req {
                    self.g_fast = req;
                }
            }
            self.gain = self.g_slow * self.g_fast;
            // The product can only drift above `ma` through rounding; pin it.
            if self.gain > ma {
                self.gain = ma;
            }

            buf[i] = (out_l * self.gain) as f32;
            buf[i + 1] = (out_r * self.gain) as f32;
            self.idx += 1;
        }
    }
}

/// A 4x polyphase interpolator for true-peak detection: a Blackman-windowed \[16\]
fn design_polyphase() -> [[f64; TP_TAPS]; TP_PHASES] {
    let mut out = [[0.0f64; TP_TAPS]; TP_PHASES];
    let n = (TP_TAPS * TP_PHASES) as f64;
    for (p, phase) in out.iter_mut().enumerate() {
        for (t, c) in phase.iter_mut().enumerate() {
            // Where this tap sits in the prototype, in input samples.
            let x = t as f64 - (TP_TAPS as f64 / 2.0 - 1.0) - p as f64 / TP_PHASES as f64;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let k = (t * TP_PHASES + p) as f64;
            let w = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * k / (n - 1.0)).cos()
                + 0.08 * (4.0 * std::f64::consts::PI * k / (n - 1.0)).cos();
            *c = sinc * w;
        }
        // [17]
        let sum: f64 = phase.iter().sum();
        if sum.abs() > 1e-12 {
            for c in phase.iter_mut() {
                *c /= sum;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_signal_passes_through_scaled() {
        let mut lim = Limiter::new(48000);
        // [18]
        let mut buf = vec![0.1f32; 48000 * 6];
        lim.process(&mut buf);
        for &v in &buf[buf.len() - 1000..] {
            assert!((v - 0.125).abs() < 0.005, "got {v}");
        }
    }

    #[test]
    fn loud_signal_is_pulled_down() {
        let mut lim = Limiter::new(48000);
        let mut buf = vec![8.0f32; 48000 * 2];
        lim.process(&mut buf);
        let tail = &buf[buf.len() - 100..];
        for &v in tail {
            assert!(v.abs() <= 1.0, "limiter left {v} above full scale");
        }
    }

    #[test]
    fn is_deterministic_across_identical_input() {
        let make = || {
            let mut lim = Limiter::new(48000);
            let mut buf: Vec<f32> = (0..8192)
                .map(|i| ((i as f32) * 0.01).sin() * 3.0)
                .collect();
            lim.process(&mut buf);
            buf
        };
        assert_eq!(make(), make());
    }

    /// The whole point of a brickwall: whatever goes in, nothing comes out \[19\]
    #[test]
    fn brickwall_never_exceeds_the_ceiling() {
        for ceiling in [1.0f64, 0.5] {
            let mut bw = Brickwall::new(48000, ceiling, 2.0, 60.0, 400.0, true);
            // [20]
            let mut buf: Vec<f32> = (0..48000 * 2)
                .map(|i| {
                    let t = i / 2;
                    let base = ((t as f64) * 0.05).sin() * 40.0;
                    let spike = if t % 977 == 0 { 300.0 } else { 0.0 };
                    (base + spike) as f32
                })
                .collect();
            bw.process(&mut buf);
            let peak = buf.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            assert!(
                peak as f64 <= ceiling + 1e-4,
                "brickwall at ceiling {ceiling} let {peak} through"
            );
            assert_eq!(
                clamp_block(&mut buf),
                0,
                "brickwall output still needed hard clipping at ceiling {ceiling}"
            );
        }
    }

    /// The reason it exists. A brief transient must not pull down the material \[21\]
    #[test]
    fn brickwall_recovers_quickly_after_a_transient() {
        let quiet = 0.5f32;
        let make = |limit: bool| {
            let mut buf: Vec<f32> = vec![0.0; 48000 * 2];
            for (i, v) in buf.iter_mut().enumerate() {
                let t = i / 2;
                // One 1 ms burst at 0.1 s, quiet steady tone either side.
                *v = if (4800..4848).contains(&t) { 60.0 } else { quiet };
            }
            if limit {
                Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut buf);
            }
            buf
        };
        let out = make(true);
        let level = |sec: f64| {
            let a = (sec * 48000.0) as usize * 2;
            let b = a + 4800;
            out[a..b].iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        };
        // Well before the burst the signal is under the ceiling and untouched.
        assert!(
            (level(0.02) - quiet).abs() < 0.02,
            "quiet material before the transient was attenuated: {}",
            level(0.02)
        );
        // [22]
        assert!(
            (level(0.35) - quiet).abs() < 0.02,
            "still ducking 250 ms after a 1 ms transient: {}",
            level(0.35)
        );
    }

    /// True-peak detection has to catch overshoot that sample-peak detection \[23\]
    #[test]
    fn true_peak_detection_catches_intersample_overshoot() {
        // A half-Nyquist tone whose samples sit exactly at full scale.
        let make = || -> Vec<f32> {
            (0..48000 * 2)
                .map(|i| {
                    let t = (i / 2) as f64;
                    // [24]
                    ((t * std::f64::consts::PI / 2.0 + std::f64::consts::PI / 4.0).sin()
                        * std::f64::consts::SQRT_2) as f32
                })
                .collect()
        };
        let mut tp = make();
        let mut sp = make();
        Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut tp);
        Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, false).process(&mut sp);
        let peak = |b: &[f32]| b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(
            peak(&tp) < peak(&sp) - 0.05,
            "true-peak mode did not pull further than sample-peak mode: {} against {}",
            peak(&tp),
            peak(&sp)
        );
    }

    #[test]
    fn brickwall_is_deterministic() {
        let run = || {
            let mut buf: Vec<f32> = (0..8192)
                .map(|i| ((i as f32) * 0.017).sin() * 9.0)
                .collect();
            Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut buf);
            buf
        };
        assert_eq!(run(), run());
    }

    /// The point of the sustained stage. \[25\]
    #[test]
    fn the_sustained_stage_holds_the_gain_steady_on_a_loud_passage() {
        let wobble = |sustain_ms: f64| -> f64 {
            // [26]
            let src: Vec<f32> = (0..48000 * 2 * 2)
                .map(|i| {
                    let t = (i / 2) as f64 / 48000.0;
                    let a = (t * 1000.0 * std::f64::consts::TAU).sin();
                    let b = (t * 1050.0 * std::f64::consts::TAU).sin();
                    ((a + b) * 20.0) as f32
                })
                .collect();
            let mut buf = src.clone();
            // Deliberately short fast release: the setting that misbehaves.
            let mut bw = Brickwall::new(48000, 1.0, 2.0, 5.0, sustain_ms, true);
            let d = bw.latency() * 2;
            bw.process(&mut buf);
            // [27]
            let start = src.len() / 2;
            let g: Vec<f64> = (start..src.len() - d)
                .filter(|&k| src[k].abs() > 4.0)
                .map(|k| (buf[k + d] as f64) / (src[k] as f64))
                .collect();
            assert!(g.len() > 10000, "not enough usable samples at {sustain_ms}");
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            let var = g.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / g.len() as f64;
            var.sqrt() / mean
        };

        let single = wobble(0.0);
        let staged = wobble(400.0);
        // [28]
        assert!(
            staged < single * 0.85,
            "the sustained stage did not steady the gain: it wobbles by              {staged:.4} with the stage and {single:.4} without"
        );
    }

    /// And it must not have cost the thing the fast release was for: a brief \[29\]
    #[test]
    fn the_sustained_stage_does_not_reintroduce_pumping() {
        let quiet = 0.5f32;
        let mut buf: Vec<f32> = vec![0.0; 48000 * 2];
        for (i, v) in buf.iter_mut().enumerate() {
            let t = i / 2;
            *v = if (4800..4848).contains(&t) { 60.0 } else { quiet };
        }
        Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut buf);
        let level = |sec: f64| {
            let a = (sec * 48000.0) as usize * 2;
            buf[a..a + 4800].iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        };
        assert!(
            (level(0.35) - quiet).abs() < 0.02,
            "still ducking 250 ms after a 1 ms transient with the sustained \
             stage on: {}",
            level(0.35)
        );
    }
}
