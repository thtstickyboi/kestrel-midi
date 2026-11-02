// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Building the cached waveforms the FDL convolves against. \[1\]

use crate::backend::Backend;
use crate::bank::Bank;
use crate::config::Config;
use crate::cpu::CpuSynth;
use crate::merlin::fft::{r2c_once, C32, Fft};
use crate::voice::SpawnCmd;
use anyhow::{bail, Context, Result};
use std::sync::Arc;

/// One entry's waveform, before a release is chosen.
#[derive(Debug)]
pub struct Kernel {
    /// Mono, unit gain, unpanned, and **still sounding**: no note-off was \[2\]
    pub sustain: Vec<f32>,
    /// The command this was rendered from, kept so [`Kernel::grown`] can \[3\]
    pub cmd: SpawnCmd,
    /// Per-frame release ratio, read from the region's own params so the \[4\]
    pub release_coef: f32,
    /// Frames the release takes to reach the envelope floor from here. This is \[5\]
    pub tail: usize,
    /// First frame at which the amplitude envelope has stopped moving. \[6\]
    pub flat_from: usize,
}

impl Kernel {
    /// Render one voice for `frames` with no note-off.
    pub fn build(
        cfg: &Config,
        bank: &Arc<Bank>,
        region: u32,
        key: u8,
        vel: u8,
        frames: usize,
    ) -> Result<Self> {
        let v = bank
            .build_layer(region, key, vel, cfg, 0.0)
            .with_context(|| format!("region {region} produced no voice at key {key} vel {vel}"))?;
        Self::from_cmd(cfg, bank, &cmd_for(&v, region), frames, None)
    }

    /// The same, from an already-resolved spawn command. \[7\]
    pub fn from_cmd(
        cfg: &Config,
        bank: &Arc<Bank>,
        src: &SpawnCmd,
        frames: usize,
        nofilter: Option<&std::sync::Arc<Vec<crate::bank::RegionParams>>>,
    ) -> Result<Self> {
        let region = src.region;
        let table: &[crate::bank::RegionParams] =
            nofilter.map_or(&bank.params[..], |a| &a[..]);
        let p = table
            .get(src.params as usize)
            .with_context(|| format!("region {region} has no params at index {}", src.params))?;

        let flat_from = flat_from(p, cfg);
        let release_coef = p.release_coef;
        let tail = release_frames(p, cfg);

        // [8]
        let cmd = SpawnCmd {
            gate_slot: 0,
            ordinal: 1,
            start_rel: 0,
            note_id_lo: 0,
            note_id_hi: 0,
            gain_l: 1.0,
            gain_r: 0.0,
            row_bias: 0,
            // [9]
            variant: 0,
            ..*src
        };

        let mut synth = CpuSynth::new(cfg, bank.clone());
        // [10]
        if let Some(nf) = nofilter {
            synth.set_base_params(nf.clone());
        }
        synth.spawn(std::slice::from_ref(&cmd))?;

        let block = cfg.block_frames as usize;
        let mut buf = vec![0.0f32; cfg.block_samples()];
        let mut sustain = Vec::with_capacity(frames);
        while sustain.len() < frames {
            synth.render(&mut buf)?;
            for f in 0..block {
                if sustain.len() == frames {
                    break;
                }
                sustain.push(buf[f * 2]);
            }
        }
        Ok(Self { sustain, cmd, release_coef, tail, flat_from })
    }

    /// A kernel whose sustain was rendered somewhere else -- the device's \[11\]
    pub fn from_rendered(
        cfg: &Config,
        src: &SpawnCmd,
        sustain: Vec<f32>,
        table: &[crate::bank::RegionParams],
    ) -> Result<Self> {
        let p = table
            .get(src.params as usize)
            .with_context(|| format!("region {} has no params at index {}", src.region, src.params))?;
        let cmd = SpawnCmd {
            gate_slot: 0,
            ordinal: 1,
            start_rel: 0,
            note_id_lo: 0,
            note_id_hi: 0,
            gain_l: 1.0,
            gain_r: 0.0,
            row_bias: 0,
            variant: 0,
            ..*src
        };
        Ok(Self {
            sustain,
            cmd,
            release_coef: p.release_coef,
            tail: release_frames(p, cfg),
            flat_from: flat_from(p, cfg),
        })
    }

    /// The same waveform rendered to at least `frames`. \[12\]
    pub fn grown(
        &self,
        cfg: &Config,
        bank: &Arc<Bank>,
        nofilter: Option<&std::sync::Arc<Vec<crate::bank::RegionParams>>>,
        frames: usize,
    ) -> Result<Self> {
        Self::from_cmd(cfg, bank, &self.cmd, frames.max(self.sustain.len()), nofilter)
    }

    pub fn frames(&self) -> usize {
        self.sustain.len()
    }

    /// The kernel for release reference point `dj`, per section 3b. \[13\]
    pub fn at_release(&self, dj: usize) -> Result<Vec<f32>> {
        self.segment(dj, 0, self.sustain.len())
    }

    /// The same kernel, but only `[from, from + len)` of it. \[14\]
    pub fn segment(&self, dj: usize, from: usize, len: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; len];
        self.segment_rows(dj, from, len, len, 1, &mut out)?;
        Ok(out)
    }

    /// [`Kernel::segment`] and [`partition`] in one pass: the segment \[15\]
    pub fn segment_partitioned(
        &self,
        dj: usize,
        from: usize,
        p: usize,
        lags: usize,
        out: &mut [f32],
    ) -> Result<()> {
        self.segment_rows(dj, from, p, 2 * p, lags, out)
    }

    /// The shared core: `rows` rows of `row` frames of the kernel starting at \[16\]
    fn segment_rows(
        &self,
        dj: usize,
        from: usize,
        row: usize,
        stride: usize,
        rows: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if dj < self.flat_from {
            bail!(
                "release reference {dj} is inside this region's moving envelope, which is flat \
                 only from frame {}. Section 3b's collapse does not hold there, so this bucket \
                 needs a render with a real note-off rather than an analytic release.",
                self.flat_from
            );
        }
        if row > stride || out.len() != rows * stride {
            bail!(
                "a segment of {rows} rows of {row} frames at stride {stride} is {} samples, \
                 not the {} it was given room for",
                rows * stride,
                out.len()
            );
        }
        if out.is_empty() {
            return Ok(());
        }
        // Frames of the kernel that exist at all, counted from `from`.
        let valid = self.sustain.len().saturating_sub(from).min(row * rows);

        // [17]
        let r = self.release_coef as f64;
        let mut g = if from > dj { r.powi((from - dj) as i32) } else { 1.0 };
        for (k, dst) in out.chunks_exact_mut(stride).enumerate() {
            let a = k * row;
            let m = valid.saturating_sub(a).min(row);
            if m == 0 {
                dst.fill(0.0);
                continue;
            }
            let (body, pad) = dst.split_at_mut(m);
            let src = &self.sustain[from + a..from + a + m];
            // [18]
            let plain = dj.saturating_sub(from + a).min(m);
            body[..plain].copy_from_slice(&src[..plain]);
            for (d, &s) in body[plain..].iter_mut().zip(&src[plain..]) {
                g *= r;
                *d = s * g as f32;
            }
            pad.fill(0.0);
        }
        Ok(())
    }
}

/// Split a kernel into `lags` partitions of `p` frames, each zero-padded to \[19\]
pub fn partition(kernel: &[f32], p: usize, lags: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; lags * 2 * p];
    for k in 0..lags {
        let src = k * p;
        if src >= kernel.len() {
            break;
        }
        let n = p.min(kernel.len() - src);
        out[k * 2 * p..k * 2 * p + n].copy_from_slice(&kernel[src..src + n]);
    }
    out
}

/// Partition a kernel and transform it into the `lags * bins` complex the FDL \[20\]
pub fn transform(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fft: &Fft,
    kernel: &[f32],
    lags: usize,
) -> Result<Vec<C32>> {
    r2c_once(device, queue, fft, &partition(kernel, fft.n(), lags))
}

/// A spawn command for one freshly-built voice, for the `(region, key, vel)` \[21\]
fn cmd_for(v: &crate::bank::VoiceSpawn, region: u32) -> SpawnCmd {
    SpawnCmd {
        phase_lo: v.phase.lo(),
        phase_hi: v.phase.hi(),
        step_lo: v.step.lo(),
        step_hi: v.step.hi(),
        smp_base: v.smp_base,
        smp_len: v.smp_len,
        loop_start: v.loop_start,
        loop_end: v.loop_end,
        flags: v.flags,
        params: v.params,
        variant: 0,
        region,
        gate_slot: 0,
        ordinal: 1,
        start_rel: 0,
        note_id_lo: 0,
        note_id_hi: 0,
        gain_l: 1.0,
        gain_r: 0.0,
        row_bias: 0,
        // Analytic phase is not a merlin feature; the shader never reads it.
        rotation: crate::phase::Coefficients::default(),
    }
}

/// The first frame at which a region's amplitude envelope has stopped moving. \[22\]
pub fn flat_from(p: &crate::bank::RegionParams, cfg: &Config) -> usize {
    let n_attack = if p.attack_rate > 0.0 {
        (p.attack_end / p.attack_rate).ceil().max(0.0) as usize
    } else {
        0
    };
    // The decay starts at 1.0, whatever the attack's target was.
    let n_decay = if p.decay_target >= 1.0 {
        0
    } else if cfg.decay_curve == crate::config::EnvelopeCurve::Exponential {
        if p.decay_coef >= 1.0 || p.decay_coef <= 0.0 {
            0
        } else {
            (p.decay_target.ln() / p.decay_coef.ln()).ceil().max(0.0) as usize
        }
    } else if p.decay_coef > 0.0 {
        ((1.0 - p.decay_target) / p.decay_coef).ceil().max(0.0) as usize
    } else {
        0
    };
    n_attack.saturating_add(n_decay)
}

/// The amplitude envelope of a **sustaining** note at frame `n`. \[23\]
pub fn env_at(p: &crate::bank::RegionParams, cfg: &Config, n: u32) -> f32 {
    let n = n as f64;
    // [24]
    let n_ah = if p.attack_rate > 0.0 {
        (p.attack_end as f64 / p.attack_rate as f64).ceil().max(0.0)
    } else {
        // A rate of zero never reaches the target, so the note never sounds.
        return 0.0;
    };
    if n + 1.0 < n_ah {
        return (((n + 1.0) * p.attack_rate as f64) as f32).min(1.0);
    }
    if n + 1.0 == n_ah {
        return 1.0;
    }
    let k = n - n_ah + 1.0;
    let lvl = match cfg.decay_curve {
        crate::config::EnvelopeCurve::Exponential => {
            if p.decay_coef <= 0.0 {
                0.0
            } else {
                (p.decay_coef as f64).powf(k)
            }
        }
        crate::config::EnvelopeCurve::Linear => 1.0 - p.decay_coef as f64 * k,
    } as f32;
    if lvl <= p.decay_target {
        // [25]
        if p.sustain <= cfg.env_floor {
            0.0
        } else {
            p.sustain
        }
    } else {
        lvl
    }
}

/// How many frames a region's release takes to reach the envelope floor. \[26\]
pub fn release_frames(p: &crate::bank::RegionParams, cfg: &Config) -> usize {
    let floor = cfg.env_floor as f64;
    let level = p.sustain as f64;
    if level <= floor {
        // [27]
        return 0;
    }
    let r = p.release_coef as f64;
    let n = match cfg.release_curve {
        crate::config::EnvelopeCurve::Exponential => {
            if r <= 0.0 {
                0.0
            } else if r >= 1.0 {
                return usize::MAX;
            } else {
                (floor / level).ln() / r.ln()
            }
        }
        crate::config::EnvelopeCurve::Linear => {
            if r <= 0.0 {
                return usize::MAX;
            }
            (level - floor) / r
        }
    };
    n.ceil().max(0.0) as usize
}

/// A copy of `params` with every filter disabled. \[28\]
pub fn nofilter_params(params: &[crate::bank::RegionParams]) -> Vec<crate::bank::RegionParams> {
    params
        .iter()
        .map(|p| {
            let mut q = *p;
            q.flags &= !crate::bank::RP_FILTER;
            q
        })
        .collect()
}
