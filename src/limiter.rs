// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// NOT ENTIRELY KESTREL'S TO LICENSE. The `Limiter` type below -- the
// `--limiter omni` path -- is a port of
// `OmniConverter/Extensions/Audio/Limiter.cs`, itself from Kiva by Arduano:
//
//     Copyright (C) 2020 Arduano
//     DON'T BE A DICK PUBLIC LICENSE, Version 1.1
//     https://github.com/arduano/Kiva
//
// DBAD is permissive and not copyleft, so it neither extends to the rest of
// this file nor conflicts with the MPL; what it asks for is credit, which this
// notice and `THIRD-PARTY.md` are. `Brickwall`, the default and the only
// limiter recommended for rendering, is original work and shares no code with
// it. If `omni` is ever removed, this notice goes with it.

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

/// What every finished block goes through on its way out, in this order: \[2\]
pub struct OutputStage {
    limiter: Limiter,
    brickwall: Brickwall,
    enabled: bool,
    mode: LimiterMode,
    clamp: bool,
    nan_guard: bool,
    /// `master_volume` where it is over 1, else 1.
    pre_gain: f32,
    /// `master_volume` where it is at most 1, else 1.
    post_gain: f32,
    dc: Option<DcBlocker>,
    /// The largest magnitude the stage has written, before the clamp.
    peak: f32,
}

impl OutputStage {
    pub fn new(cfg: &crate::config::Config) -> Self {
        OutputStage {
            limiter: Limiter::new(cfg.sample_rate),
            brickwall: Brickwall::new(
                cfg.sample_rate,
                cfg.limiter_ceiling(),
                cfg.limiter_lookahead_ms,
                cfg.limiter_release_ms,
                cfg.limiter_sustain_ms,
                cfg.limiter_true_peak,
            ),
            enabled: cfg.limiter,
            mode: cfg.limiter_mode,
            clamp: cfg.clamp_output,
            nan_guard: cfg.nan_guard,
            pre_gain: cfg.master_volume.max(1.0),
            post_gain: cfg.master_volume.min(1.0),
            dc: cfg.dc_blocker.then(|| DcBlocker::new(cfg.sample_rate, cfg.dc_blocker_hz)),
            peak: 0.0,
        }
    }

    /// The largest magnitude this stage has written so far, before the clamp: \[3\]
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// Scale, filter, limit, clamp and check block number `block` in place. \[4\]
    pub fn process(&mut self, out: &mut [f32], block: u64) -> anyhow::Result<u64> {
        // [5]
        if self.pre_gain != 1.0 {
            out.iter_mut().for_each(|v| *v *= self.pre_gain);
        }
        if let Some(dc) = &mut self.dc {
            dc.process(out);
        }
        if self.enabled {
            match self.mode {
                LimiterMode::Off => {}
                // [6]
                LimiterMode::Omni => {
                    self.limiter.process(out);
                    self.brickwall.process(out);
                }
                LimiterMode::Brickwall => self.brickwall.process(out),
            }
        }
        // [7]
        if self.post_gain != 1.0 {
            out.iter_mut().for_each(|v| *v *= self.post_gain);
        }
        self.peak = out.iter().fold(self.peak, |m, v| m.max(v.abs()));
        // [8]
        let clipped = if self.clamp { clamp_block(out) } else { 0 };

        if self.nan_guard {
            if let Some(i) = out.iter().position(|v| !v.is_finite()) {
                anyhow::bail!(
                    "block {} sample {} is {}; the synth produced a non-finite value",
                    block,
                    i,
                    out[i]
                );
            }
        }
        Ok(clipped)
    }
}

/// Second-order Butterworth high-pass on an interleaved stereo stream: the DC \[9\]
#[derive(Debug, Clone)]
pub struct DcBlocker {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: [f64; 2],
    z2: [f64; 2],
}

impl DcBlocker {
    pub fn new(sample_rate: u32, hz: f64) -> Self {
        let w0 = std::f64::consts::TAU * hz / sample_rate as f64;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / std::f64::consts::SQRT_2;
        let a0 = 1.0 + alpha;
        DcBlocker {
            b0: (1.0 + cos) / 2.0 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: [0.0; 2],
            z2: [0.0; 2],
        }
    }

    /// Filter one interleaved stereo block in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len() % 2, 0);
        for frame in buf.chunks_exact_mut(2) {
            for (c, s) in frame.iter_mut().enumerate() {
                let x = *s as f64;
                let y = self.b0 * x + self.z1[c];
                self.z1[c] = self.b1 * x - self.a1 * y + self.z2[c];
                self.z2[c] = self.b2 * x - self.a2 * y;
                *s = y as f32;
            }
        }
    }
}

/// Hard clamp, always applied last so nothing leaves the renderer out of range. \[10\]
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

// [11]

/// Which limiter runs on the mixed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterMode {
    /// No limiting. `clamp_block` still runs, so loud material hard-clips.
    Off,
    /// The port of the realtime limiter OmniConverter ships, above, with \[12\]
    Omni,
    /// Lookahead true-peak brickwall. Guarantees the output never exceeds the \[13\]
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

/// Lookahead true-peak brickwall limiter. \[14\]
pub struct Brickwall {
    ceiling: f64,
    look: usize,
    release_coef: f64,
    /// Attack and release coefficients of the sustained stage. Both zero when \[15\]
    sustain_atk: f64,
    sustain_rel: f64,
    /// Gain the sustained stage is holding: a slow envelope of the \[16\]
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
    /// Frames the audio is delayed by: the lookahead plus the detector's own \[17\]
    delay_frames: usize,
    true_peak: bool,
    idx: u64,
    /// Largest true peak seen at the input, for reporting.
    pub peak_in: f64,
    /// Smallest gain the limiter had to apply, for reporting. **Not wired up** \[18\]
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
        // [19]
        let (sustain_atk, sustain_rel) = if sustain_ms > 0.0 {
            let a = (sustain_ms * 1e-3 * sr).max(1.0);
            (
                1.0 - (-1.0f64 / a).exp(),
                1.0 - (-1.0f64 / (a * 4.0)).exp(),
            )
        } else {
            (0.0, 0.0)
        };
        // [20]
        let release_coef = 1.0 - (-1.0 / rel).exp();
        // [21]
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

    /// True peak of one frame: the largest magnitude of the 4x oversampled \[22\]
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
        if !self.skip_if_settled(buf) {
            self.process_frames(buf);
        }
    }

    /// Pass a block of exact zeros through a limiter that has settled, without \[23\]
    fn skip_if_settled(&mut self, buf: &[f32]) -> bool {
        let n = buf.len() / 2;
        if n == 0
            || !buf.iter().all(|v| v.to_bits() == 0)
            || !self.delay.iter().all(|v| v.to_bits() == 0)
            || !self.hist.iter().flatten().all(|&v| v == 0.0)
            || self.dq.front().is_some_and(|&(v, _)| v != 0.0)
            || !self.gr_ring.iter().all(|&g| g == 1.0)
        {
            return false;
        }
        let before = (self.g_slow, self.g_fast, self.gain);
        self.step_gain(self.gr_sum / self.gr_ring.len() as f64);
        let bits = |(a, b, c): (f64, f64, f64)| (a.to_bits(), b.to_bits(), c.to_bits());
        if bits((self.g_slow, self.g_fast, self.gain)) != bits(before) {
            (self.g_slow, self.g_fast, self.gain) = before;
            return false;
        }
        // [24]
        if self.true_peak {
            let k = n.min(TP_TAPS);
            for h in &mut self.hist {
                h.copy_within(0..TP_TAPS - k, k);
                h[..k].fill(0.0);
            }
        }
        self.dq.clear();
        self.dq.push_back((0.0, self.idx + n as u64 - 1));
        self.dpos = (self.dpos + n) % self.delay_frames;
        self.gr_pos = (self.gr_pos + n) % self.gr_ring.len();
        self.idx += n as u64;
        true
    }

    /// The per-sample limiter, which `process` runs on anything that is not \[25\]
    fn process_frames(&mut self, buf: &mut [f32]) {
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
            self.step_gain(ma);

            buf[i] = (out_l * self.gain) as f32;
            buf[i + 1] = (out_r * self.gain) as f32;
            self.idx += 1;
        }
    }

    /// One frame's gain, from the moving average of the requirement.
    #[inline]
    fn step_gain(&mut self, ma: f64) {
        // [26]
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
    }
}

/// A 4x polyphase interpolator for true-peak detection: a Blackman-windowed \[27\]
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
        // [28]
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
        // [29]
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

    /// The whole point of a brickwall: whatever goes in, nothing comes out \[30\]
    #[test]
    fn brickwall_never_exceeds_the_ceiling() {
        for ceiling in [1.0f64, 0.5] {
            let mut bw = Brickwall::new(48000, ceiling, 2.0, 60.0, 400.0, true);
            // [31]
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

    /// The reason it exists. A brief transient must not pull down the material \[32\]
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
        // [33]
        assert!(
            (level(0.35) - quiet).abs() < 0.02,
            "still ducking 250 ms after a 1 ms transient: {}",
            level(0.35)
        );
    }

    /// True-peak detection has to catch overshoot that sample-peak detection \[34\]
    #[test]
    fn true_peak_detection_catches_intersample_overshoot() {
        // A half-Nyquist tone whose samples sit exactly at full scale.
        let make = || -> Vec<f32> {
            (0..48000 * 2)
                .map(|i| {
                    let t = (i / 2) as f64;
                    // [35]
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

    /// Skipping a settled limiter's silence has to be invisible: the same \[36\]
    #[test]
    fn skipping_settled_silence_changes_nothing() {
        fn state(b: &Brickwall) -> Vec<u64> {
            let mut v = vec![
                b.g_slow.to_bits(),
                b.g_fast.to_bits(),
                b.gain.to_bits(),
                b.gr_sum.to_bits(),
                b.dpos as u64,
                b.gr_pos as u64,
                b.idx,
                b.peak_in.to_bits(),
            ];
            v.extend(b.delay.iter().map(|x| x.to_bits() as u64));
            v.extend(b.hist.iter().flatten().map(|x| x.to_bits()));
            v.extend(b.gr_ring.iter().map(|x| x.to_bits()));
            v.extend(b.dq.iter().flat_map(|&(x, j)| [x.to_bits(), j]));
            v
        }
        // [37]
        let tone = |len: usize, amp: f64| -> Vec<f32> {
            (0..len * 2).map(|i| (((i / 2) as f64 * 0.07).sin() * amp) as f32).collect()
        };
        let blocks = |settle: usize| {
            let mut blocks: Vec<Vec<f32>> = Vec::new();
            blocks.push(tone(4096, 30.0));
            for _ in 0..settle {
                blocks.push(vec![0.0; 4096 * 2]);
            }
            let mut neg = vec![0.0f32; 4096 * 2];
            neg[100] = -0.0;
            blocks.push(neg);
            for len in [3, 5, 4096, 4096, 7, 4096] {
                blocks.push(vec![0.0; len * 2]);
            }
            blocks.push(tone(1000, 0.5));
            blocks.push(tone(3000, 12.0));
            for _ in 0..settle {
                blocks.push(vec![0.0; 4096 * 2]);
            }
            blocks
        };

        for (sustain, true_peak) in [(0.0, true), (400.0, true), (0.0, false), (400.0, false)] {
            let blocks = blocks(if sustain > 0.0 { 700 } else { 40 });
            let mut fast = Brickwall::new(48000, 1.0, 2.0, 60.0, sustain, true_peak);
            let mut slow = Brickwall::new(48000, 1.0, 2.0, 60.0, sustain, true_peak);
            let mut skipped = 0;
            for (k, b) in blocks.iter().enumerate() {
                let mut a = b.clone();
                let mut c = b.clone();
                if fast.skip_if_settled(&a) {
                    skipped += 1;
                } else {
                    fast.process_frames(&mut a);
                }
                slow.process_frames(&mut c);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a), bits(&c), "block {k} output, sustain {sustain} tp {true_peak}");
                assert_eq!(state(&fast), state(&slow), "block {k} state, sustain {sustain} tp {true_peak}");
            }
            assert!(skipped > 40, "only {skipped} blocks skipped at sustain {sustain} tp {true_peak}");
            // And `process` is the two put together.
            let mut via = Brickwall::new(48000, 1.0, 2.0, 60.0, sustain, true_peak);
            for b in &blocks {
                via.process(&mut b.clone());
            }
            assert_eq!(state(&via), state(&slow));
        }
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

    /// The point of the sustained stage. \[38\]
    #[test]
    fn the_sustained_stage_holds_the_gain_steady_on_a_loud_passage() {
        let wobble = |sustain_ms: f64| -> f64 {
            // [39]
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
            // [40]
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
        // [41]
        assert!(
            staged < single * 0.85,
            "the sustained stage did not steady the gain: it wobbles by \
             {staged:.4} with the stage and {single:.4} without"
        );
    }

    /// And it must not have cost the thing the fast release was for: a brief \[42\]
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

    /// `--limiter omni` runs the brickwall behind the follower as a safety \[43\]
    #[test]
    fn omni_with_the_safety_stage_never_exceeds_the_ceiling() {
        // [44]
        let src: Vec<f32> = (0..48000 * 2 * 2)
            .map(|i| {
                let t = i / 2;
                let x = ((t as f64) * 0.03).sin();
                (if t < 48000 { x * 0.3 } else { x * 12.0 }) as f32
            })
            .collect();
        let peak = |b: &[f32]| b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let mut alone = src.clone();
        Limiter::new(48000).process(&mut alone);
        assert!(
            peak(&alone) > 1.5,
            "the follower caught the burst by itself ({}), so this proves nothing",
            peak(&alone)
        );

        for ceiling in [1.0f64, 10f64.powf(-1.0 / 20.0)] {
            let mut both = src.clone();
            let mut bw = Brickwall::new(48000, ceiling, 2.0, 60.0, 0.0, true);
            let d = bw.latency() * 2;
            Limiter::new(48000).process(&mut both);
            bw.process(&mut both);
            assert!(
                peak(&both) as f64 <= ceiling + 1e-4,
                "omni with the safety stage let {} through at ceiling {ceiling}",
                peak(&both)
            );
            assert!(
                both[d..96000] == alone[..96000 - d],
                "the safety stage changed material under the ceiling at {ceiling}"
            );
        }
    }

    /// Interleaved stereo, the same signal in both channels.
    fn stereo(frames: usize, f: impl Fn(f64) -> f64) -> Vec<f32> {
        (0..frames).flat_map(|i| { let s = f(i as f64 / 48000.0) as f32; [s, s] }).collect()
    }

    /// Gain of the DC blocker at `hz`, in dB, from the RMS of a settled tone.
    fn dc_gain_db(hz: f64) -> f64 {
        let secs = (20.0 / hz).max(2.0);
        let n = (48000.0 * secs) as usize;
        let mut buf = stereo(n, |t| (std::f64::consts::TAU * hz * t).sin() * 0.5);
        DcBlocker::new(48000, 15.0).process(&mut buf);
        let rms = |b: &[f32]| (b.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
        let tail = &buf[buf.len() / 2..];
        20.0 * (rms(tail) / (0.5 / std::f64::consts::SQRT_2)).log10()
    }

    /// The pedestal goes and the music stays: DC to nothing, the 2.6 Hz swell \[45\]
    #[test]
    fn dc_blocker_removes_the_pedestal_and_keeps_the_music() {
        let mut dc = stereo(96000, |_| 0.5);
        DcBlocker::new(48000, 15.0).process(&mut dc);
        let settled = dc[48000..].iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(settled < 1e-4, "DC left after a second: {settled}");

        assert!(dc_gain_db(2.63) < -25.0, "2.63 Hz is only {:.1} dB down", dc_gain_db(2.63));
        for hz in [27.5, 55.0, 440.0, 4186.0] {
            let g = dc_gain_db(hz);
            assert!(g.abs() < 0.5, "{hz} Hz moved {g:.2} dB");
        }
    }

    /// State carries across blocks, so cutting the stream differently \[46\]
    #[test]
    fn dc_blocker_does_not_depend_on_the_block_size() {
        let src = stereo(48000, |t| 0.3 + (std::f64::consts::TAU * 3.0 * t).sin() * 0.4);
        let run = |block: usize| {
            let mut out = src.clone();
            let mut f = DcBlocker::new(48000, 15.0);
            out.chunks_mut(block * 2).for_each(|b| f.process(b));
            out
        };
        assert!(run(4096) == run(512), "block size changed the output");
    }

    /// `--volume` at or below 100% scales the limited output, so a dense \[47\]
    #[test]
    fn volume_sets_the_level_of_the_limited_output() {
        // Far over full scale, as a dense mix is.
        let src = stereo(48000, |t| (std::f64::consts::TAU * 220.0 * t).sin() * 8.0);
        let run = |volume: f32, limiter: bool| {
            let cfg = crate::config::Config { master_volume: volume, limiter, ..Default::default() };
            let mut out = src.clone();
            let mut stage = OutputStage::new(&cfg);
            out.chunks_mut(8192).enumerate().for_each(|(i, b)| {
                stage.process(b, i as u64).unwrap();
            });
            (out, stage.peak())
        };
        let (full, full_peak) = run(1.0, true);
        let (half, half_peak) = run(0.5, true);
        assert!(full_peak > 0.9, "the limiter was not engaged ({full_peak})");
        assert!(
            half.iter().zip(&full).all(|(h, f)| *h == f * 0.5),
            "50% is not exactly half of the 100% render"
        );
        assert!((half_peak as f64 - full_peak as f64 * 0.5).abs() < 1e-6);

        let (_, loud_peak) = run(2.0, true);
        assert!(loud_peak as f64 <= 1.0 + 1e-4, "200% went over the ceiling: {loud_peak}");

        let cfg = crate::config::Config { master_volume: 0.5, limiter: false, clamp_output: false, ..Default::default() };
        let mut raw = src.clone();
        OutputStage::new(&cfg).process(&mut raw, 0).unwrap();
        assert!(raw.iter().zip(&src).all(|(r, s)| *r == s * 0.5), "unlimited, 50% is not a plain gain");

        let untouched = crate::config::Config { limiter: false, clamp_output: false, ..Default::default() };
        let mut same = src.clone();
        OutputStage::new(&untouched).process(&mut same, 0).unwrap();
        assert!(same == src, "100% with nothing else on changed the samples");
    }
}
