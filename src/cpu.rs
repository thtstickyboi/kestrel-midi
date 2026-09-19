// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! CPU reference synthesizer. \[1\]

use crate::backend::{Backend, BlockStats};
use crate::bank::{
    biquad_lowpass_pre, cents_to_hz, lfo_tri, mod_env_level, mod_env_pitch_index, Bank,
    ModEnvParams, RegionParams, NO_RELEASE_AGE, RP_FILTER, RP_MOD_ENV, RP_SHORT_RELEASE, VF_LOOP,
    VF_LOOP_UNTIL_RELEASE,
};
use crate::config::{AdmitRule, Config, EnvelopeCurve, Interpolation, StealRule};
use crate::fixed::{Fixed, FRAC_SCALE_F32};
use crate::voice::*;
use anyhow::Result;
use std::sync::Arc;

/// Structure of arrays, one Vec per field. `phase` and `step` are kept as u64 \[2\]
#[derive(Default)]
struct Pool {
    phase: Vec<Fixed>,
    step: Vec<Fixed>,
    smp_base: Vec<u32>,
    smp_len: Vec<u32>,
    loop_start: Vec<u32>,
    loop_end: Vec<u32>,
    flags: Vec<u32>,
    env_stage: Vec<u32>,
    env_level: Vec<f32>,
    gain_l: Vec<f32>,
    gain_r: Vec<f32>,
    filt_z1: Vec<f32>,
    filt_z2: Vec<f32>,
    params: Vec<u32>,
    region: Vec<u32>,
    gate_slot: Vec<u32>,
    ordinal: Vec<u32>,
    start_rel: Vec<u32>,
    /// Variant the voice was born under, plus one, or zero once it has \[3\]
    born_variant: Vec<u32>,
    /// Frame in this block at which a stolen voice starts fading, plus one. \[4\]
    stop_rel: Vec<u32>,
    /// Frames alive. Read by the LFOs and by the modulation envelope. Mirrors \[5\]
    age: Vec<u32>,
    /// Age at which this voice entered release, or `NO_RELEASE_AGE` while it \[6\]
    rel_age: Vec<u32>,
    note_id: Vec<u64>,
}

impl Pool {
    fn len(&self) -> usize {
        self.phase.len()
    }

    /// `env_phase` is the block's position on the envelope grid, from which \[7\]
    fn push(&mut self, c: &SpawnCmd, env_phase: u32, step: u32) {
        self.phase.push(Fixed::from_parts(c.phase_hi, c.phase_lo));
        self.step.push(Fixed::from_parts(c.step_hi, c.step_lo));
        self.smp_base.push(c.smp_base);
        self.smp_len.push(c.smp_len);
        self.loop_start.push(c.loop_start);
        self.loop_end.push(c.loop_end);
        self.flags.push(c.flags);
        self.env_stage.push(ENV_ATTACK);
        self.env_level.push(0.0);
        self.gain_l.push(c.gain_l);
        self.gain_r.push(c.gain_r);
        self.filt_z1.push(0.0);
        self.filt_z2.push(0.0);
        self.params.push(c.params);
        self.region.push(c.region);
        let grid = (env_phase + c.start_rel) % step;
        self.gate_slot.push(c.gate_slot | (grid << GRID_SHIFT));
        self.ordinal.push(c.ordinal);
        self.start_rel.push(c.start_rel);
        // [8]
        self.born_variant.push((c.variant + 1) | (c.row_bias << 16));
        self.stop_rel.push(0);
        // [9]
        self.age.push(0u32.wrapping_sub(c.start_rel));
        self.rel_age.push(NO_RELEASE_AGE);
        self.note_id
            .push(((c.note_id_hi as u64) << 32) | c.note_id_lo as u64);
    }

    fn swap_remove_compact(&mut self, keep: &[bool]) {
        // [10]
        let mut w = 0usize;
        for (r, &alive) in keep.iter().enumerate().take(self.len()) {
            if alive {
                if w != r {
                    self.phase[w] = self.phase[r];
                    self.step[w] = self.step[r];
                    self.smp_base[w] = self.smp_base[r];
                    self.smp_len[w] = self.smp_len[r];
                    self.loop_start[w] = self.loop_start[r];
                    self.loop_end[w] = self.loop_end[r];
                    self.flags[w] = self.flags[r];
                    self.env_stage[w] = self.env_stage[r];
                    self.env_level[w] = self.env_level[r];
                    self.gain_l[w] = self.gain_l[r];
                    self.gain_r[w] = self.gain_r[r];
                    self.filt_z1[w] = self.filt_z1[r];
                    self.filt_z2[w] = self.filt_z2[r];
                    self.params[w] = self.params[r];
                    self.region[w] = self.region[r];
                    self.gate_slot[w] = self.gate_slot[r];
                    self.ordinal[w] = self.ordinal[r];
                    self.start_rel[w] = self.start_rel[r];
                    self.born_variant[w] = self.born_variant[r];
                    self.stop_rel[w] = self.stop_rel[r];
                    self.age[w] = self.age[r];
                    self.rel_age[w] = self.rel_age[r];
                    self.note_id[w] = self.note_id[r];
                }
                w += 1;
            }
        }
        self.truncate(w);
    }

    fn truncate(&mut self, n: usize) {
        self.phase.truncate(n);
        self.step.truncate(n);
        self.smp_base.truncate(n);
        self.smp_len.truncate(n);
        self.loop_start.truncate(n);
        self.loop_end.truncate(n);
        self.flags.truncate(n);
        self.env_stage.truncate(n);
        self.env_level.truncate(n);
        self.gain_l.truncate(n);
        self.gain_r.truncate(n);
        self.filt_z1.truncate(n);
        self.filt_z2.truncate(n);
        self.params.truncate(n);
        self.region.truncate(n);
        self.gate_slot.truncate(n);
        self.ordinal.truncate(n);
        self.start_rel.truncate(n);
        self.born_variant.truncate(n);
        self.stop_rel.truncate(n);
        self.age.truncate(n);
        self.rel_age.truncate(n);
        self.note_id.truncate(n);
    }
}

/// The tremolo's linear gain at `age`: exactly one while the modulation LFO is \[11\]
#[inline]
fn lfo_tremolo(p: &RegionParams, age: u32) -> f32 {
    let d = p.mod_lfo_delay();
    let mv = p.mod_lfo_to_volume();
    let on = if age >= d && mv != 0.0 { 1.0f32 } else { 0.0 };
    let m = lfo_tri(age.wrapping_sub(d).wrapping_mul(p.mod_lfo_inc)) * on;
    let x = -(m * mv) * (1.0 / 60.205_999);
    1.0 + (x.exp2() - 1.0) * on
}

/// A voice's age at a block frame, from the age stored for the top of the \[12\]
#[inline]
fn note_age(raw: u32) -> u32 {
    if raw >= 0x8000_0000 {
        0
    } else {
        raw
    }
}

/// What a release subtracts every frame, or zero to multiply by the region's \[13\]
#[inline]
fn release_fall(p: &RegionParams, level: f32, left: u32, exp_release: bool) -> f32 {
    if p.flags & RP_SHORT_RELEASE != 0 {
        level / left as f32
    } else if exp_release {
        0.0
    } else {
        p.release_coef
    }
}

pub struct CpuSynth {
    cfg: Config,
    bank: Arc<Bank>,
    pool: Pool,
    /// Interleaved f64 accumulator for one block.
    mix: Vec<f64>,
    off_meta: Vec<u32>,
    off_runs: Vec<u32>,
    chan_rows: Vec<u32>,
    /// Copies of the params table beyond the bank's own. Index 0 is the \[14\]
    variants: Vec<Vec<RegionParams>>,
    /// Modulation-envelope tables, one per params variant beyond zero.
    menv_variants: Vec<Vec<ModEnvParams>>,
    bend_active: bool,
    gain_active: bool,
    variant_active: bool,
    cut_active: bool,
    tiles: usize,
    tile_frames: usize,
    /// Portamento's speeds and exponent table, the same numbers the device \[15\]
    glide_tab: Vec<u32>,
    /// This block's position on the envelope grid. Mirrors `GpuSynth`'s.
    env_phase: u32,
    stolen: u64,
    dropped: u64,
    peak: f32,
}

impl CpuSynth {
    pub fn new(cfg: &Config, bank: Arc<Bank>) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        CpuSynth {
            cfg: cfg.clone(),
            glide_tab: crate::porta::tables(cfg.sample_rate),
            bank,
            pool: Pool::default(),
            mix: vec![0.0; cfg.block_samples()],
            off_meta: vec![0; (GATE_SLOTS + 1) * 2],
            off_runs: Vec::new(),
            // One row past the last tile; see `voice::ChannelTable::row_bias`.
            chan_rows: vec![0; (tiles + 1) * BEND_CHANNELS * CHAN_FIELDS],
            variants: Vec::new(),
            menv_variants: Vec::new(),
            bend_active: false,
            gain_active: false,
            variant_active: false,
            cut_active: false,
            tiles,
            tile_frames: cfg.gate_frames as usize,
            env_phase: 0,
            stolen: 0,
            dropped: 0,
            peak: 0.0,
        }
    }

    /// Read one pool sample, normalised the same way `unpack2x16snorm` does on \[16\]
    #[inline(always)]
    fn fetch(&self, base: u32, idx: u32) -> f32 {
        let i = (base + idx) as usize;
        let v = *self.bank.pool.get(i).unwrap_or(&0);
        (v as f32 * (1.0 / 32767.0)).max(-1.0)
    }

    /// Which queued spawn the `i`-th accepted one is. The driver has already \[17\]
    #[inline(always)]
    fn pick(&self, i: usize, want: usize, take: usize) -> usize {
        match self.cfg.admit_rule {
            AdmitRule::Loudest => i,
            AdmitRule::Even => spawn_pick(i, want, take),
        }
    }

    /// The 64-bit key voice stealing selects the k smallest of. Mirrors \[18\]
    #[inline(always)]
    fn steal_key(&self, i: usize) -> u64 {
        let id = self.pool.note_id[i];
        if self.cfg.steal_rule != StealRule::Quietest {
            return id;
        }
        let level = self.pool.env_level[i].clamp(0.0, 1.0);
        let q = (level * 65535.0) as u32 as u64;
        (q << 48) | (id & 0x0000_FFFF_FFFF_FFFF)
    }

    /// One region's DSP constants out of one copy of the params table. \[19\]
    #[inline(always)]
    fn params_of(&self, variant: u32, params_base: usize) -> RegionParams {
        match variant.checked_sub(1) {
            None => self.bank.params[params_base],
            Some(i) => self
                .variants
                .get(i as usize)
                .and_then(|t| t.get(params_base))
                .copied()
                .unwrap_or(self.bank.params[params_base]),
        }
    }

    /// The modulation-envelope entry beside `params_of`'s, from the same \[20\]
    #[inline(always)]
    fn menv_of(&self, variant: u32, params_base: usize) -> ModEnvParams {
        let fallback = || {
            self.bank
                .menv
                .get(params_base)
                .copied()
                .unwrap_or_default()
        };
        match variant.checked_sub(1) {
            None => fallback(),
            Some(i) => self
                .menv_variants
                .get(i as usize)
                .and_then(|t| t.get(params_base))
                .copied()
                .unwrap_or_else(fallback),
        }
    }

    /// Index of the sample `off` frames after `idx`, honouring the loop. \[21\]
    #[inline(always)]
    fn advance_index(idx: u32, off: i32, looping: bool, loop_start: u32, loop_end: u32, len: u32) -> u32 {
        let raw = idx as i64 + off as i64;
        if looping {
            let ls = loop_start as i64;
            let le = loop_end as i64;
            if raw >= le {
                let span = (le - ls).max(1);
                (ls + (raw - ls) % span) as u32
            } else if raw < 0 {
                0
            } else {
                raw as u32
            }
        } else {
            raw.clamp(0, len as i64 - 1) as u32
        }
    }

    #[allow(clippy::too_many_arguments)] // mirrors the shader's signature
    #[inline(always)]
    fn interpolate(
        &self,
        base: u32,
        idx: u32,
        frac: f32,
        looping: bool,
        loop_start: u32,
        loop_end: u32,
        len: u32,
    ) -> f32 {
        match self.cfg.interpolation {
            Interpolation::Nearest => self.fetch(base, idx),
            Interpolation::Linear => {
                let i1 = Self::advance_index(idx, 1, looping, loop_start, loop_end, len);
                let s0 = self.fetch(base, idx);
                let s1 = self.fetch(base, i1);
                s0 + (s1 - s0) * frac
            }
            Interpolation::Cubic => {
                let im1 = Self::advance_index(idx, -1, looping, loop_start, loop_end, len);
                let i1 = Self::advance_index(idx, 1, looping, loop_start, loop_end, len);
                let i2 = Self::advance_index(idx, 2, looping, loop_start, loop_end, len);
                let sm1 = self.fetch(base, im1);
                let s0 = self.fetch(base, idx);
                let s1 = self.fetch(base, i1);
                let s2 = self.fetch(base, i2);
                catmull_rom(sm1, s0, s1, s2, frac)
            }
        }
    }
}

/// Catmull-Rom, written in the same Horner form as the shader.
#[inline(always)]
pub fn catmull_rom(sm1: f32, s0: f32, s1: f32, s2: f32, t: f32) -> f32 {
    let a = -0.5 * sm1 + 1.5 * s0 - 1.5 * s1 + 0.5 * s2;
    let b = sm1 - 2.5 * s0 + 2.0 * s1 - 0.5 * s2;
    let c = -0.5 * sm1 + 0.5 * s1;
    ((a * t + b) * t + c) * t + s0
}

impl Backend for CpuSynth {
    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, variant: bool, cut: bool) -> Result<()> {
        self.chan_rows.copy_from_slice(rows);
        self.bend_active = bend;
        self.gain_active = gain;
        self.variant_active = variant;
        self.cut_active = cut;
        Ok(())
    }

    fn set_params_variant(
        &mut self,
        index: u32,
        data: &[RegionParams],
        menv: &[ModEnvParams],
    ) -> Result<()> {
        if index == 0 {
            return Ok(());
        }
        let i = index as usize - 1;
        if self.variants.len() <= i {
            self.variants.resize(i + 1, Vec::new());
        }
        self.variants[i] = data.to_vec();
        if self.menv_variants.len() <= i {
            self.menv_variants.resize(i + 1, Vec::new());
        }
        self.menv_variants[i] = menv.to_vec();
        Ok(())
    }

    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()> {
        self.off_meta.copy_from_slice(meta);
        self.off_runs.clear();
        self.off_runs.extend_from_slice(runs);
        Ok(())
    }

    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        let cap = self.cfg.max_voices as usize;
        let live = self.pool.len();
        let want = cmds.len();
        let (phase, step) = (self.env_phase, self.cfg.env_step_frames());

        if live + want > cap {
            match self.cfg.steal_rule {
                StealRule::DropNew => {
                    let take = want.min(cap.saturating_sub(live));
                    self.dropped += (want - take) as u64;
                    for i in 0..take {
                        self.pool.push(&cmds[self.pick(i, want, take)], phase, step);
                    }
                    return Ok(());
                }
                StealRule::Oldest | StealRule::Quietest => {
                    // [22]
                    let need = (live + want).saturating_sub(cap);
                    // [23]
                    let need = need.min(live).min(self.cfg.max_steal() as usize);
                    if need > 0 {
                        let mut order: Vec<u32> = (0..live as u32).collect();
                        order.select_nth_unstable_by_key(need - 1, |&i| {
                            self.steal_key(i as usize)
                        });
                        // [24]
                        let span = self.cfg.steal_span();
                        for &i in &order[..need] {
                            let id = self.pool.note_id[i as usize] as u32;
                            self.pool.stop_rel[i as usize] = id % span + 1;
                        }
                        self.stolen += need as u64;
                    }
                    // [25]
                    let take = want.min(cap - (live - need));
                    self.dropped += (want - take) as u64;
                    for i in 0..take {
                        self.pool.push(&cmds[self.pick(i, want, take)], phase, step);
                    }
                    return Ok(());
                }
            }
        }

        for c in cmds {
            self.pool.push(c, phase, step);
        }
        Ok(())
    }

    /// The reference backend has no device, so there is nothing to overlap: \[26\]
    fn submit(&mut self) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        let block = self.cfg.block_frames as usize;
        debug_assert_eq!(out.len(), block * 2);
        self.mix.iter_mut().for_each(|v| *v = 0.0);

        let exp_decay = self.cfg.decay_curve == EnvelopeCurve::Exponential;
        let exp_release = self.cfg.release_curve == EnvelopeCurve::Exponential;
        let floor = self.cfg.env_floor;
        let fade = self.cfg.steal_fade_frames as usize;
        let step_frames = self.cfg.env_step_frames();
        let n = self.pool.len();

        for v in 0..n {
            let stage0 = self.pool.env_stage[v];
            if stage0 == ENV_DEAD {
                continue;
            }
            let params_base = self.pool.params[v] as usize;
            // [27]
            let packed = self.pool.born_variant[v];
            let born_variant = packed & 0xFFFF;
            // [28]
            let born_bias = (packed >> 16) as usize;
            let mut variant = born_variant.saturating_sub(1);
            let mut p: RegionParams = self.params_of(variant, params_base);
            let mut use_filter = p.flags & RP_FILTER != 0;
            // [29]
            let (mut cb0, mut cb1, mut ca1, mut ca2) = (p.b0, p.b1, p.a1, p.a2);
            let (mut db0, mut db1, mut da1, mut da2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            // [30]
            let mut filter_mix = if use_filter { 1.0f32 } else { 0.0f32 };
            let mut d_filter_mix = 0.0f32;

            let base = self.pool.smp_base[v];
            let len = self.pool.smp_len[v];
            let loop_start = self.pool.loop_start[v];
            let loop_end = self.pool.loop_end[v];
            let vflags = self.pool.flags[v];
            let loop_enabled = vflags & VF_LOOP != 0;
            let loop_until_release = vflags & VF_LOOP_UNTIL_RELEASE != 0;
            // The top byte carries a glide's speed and direction; see `porta`.
            let slot_word = self.pool.gate_slot[v];
            let gate_slot = (slot_word & crate::porta::SLOT_MASK) as usize;
            let ordinal = self.pool.ordinal[v];
            // [31]
            let base_gain_l = self.pool.gain_l[v];
            let base_gain_r = self.pool.gain_r[v];
            let mut gain_l = base_gain_l;
            let mut gain_r = base_gain_r;
            // [32]
            let mut d_gain_l = 0.0f32;
            let mut d_gain_r = 0.0f32;
            let inv_gate_tile = 1.0f32 / self.tile_frames as f32;
            let base_step = self.pool.step[v];
            // Step after bend but before the LFOs; see `render.wgsl`.
            let mut bent = base_step;
            let age0 = self.pool.age[v];
            // [33]
            let mut lfo_factor = 1u32 << 24;
            let rtile = self.cfg.reduce_tile as usize;
            let channel = ChannelTable::channel_of(gate_slot as u32);
            let mut step = base_step;

            let mut phase = self.pool.phase[v];
            let mut stage = stage0;
            let mut level = self.pool.env_level[v];
            let mut z1 = self.pool.filt_z1[v];
            let mut z2 = self.pool.filt_z2[v];
            let start_rel = self.pool.start_rel[v] as usize;
            let mut stop_rel = self.pool.stop_rel[v] as usize;
            // [34]
            let born_tile = start_rel / self.tile_frames;
            // [35]
            let grid = self.cfg.note_grid;
            let phase_now = self.env_phase;
            let born_here = born_variant != 0;
            let mut release_frame = usize::MAX;
            if stage0 < ENV_RELEASE {
                if ordinal <= self.off_meta[gate_slot * 2] {
                    // [36]
                    release_frame = if grid && !born_here {
                        (fall_left(phase_now, slot_word, 0, step_frames) % step_frames) as usize
                    } else {
                        0
                    };
                } else if let Some(off) =
                    off_frame(&self.off_meta, &self.off_runs, gate_slot, ordinal)
                {
                    release_frame = if grid && !(born_here && off as usize <= start_rel) {
                        release_start(phase_now, slot_word, off, step_frames) as usize
                    } else {
                        off as usize
                    };
                }
            }
            // What the release subtracts every frame. See `release_fall`.
            let mut rel_d = if grid && stage0 == ENV_RELEASE {
                let left = fall_left(phase_now, slot_word, 0, step_frames);
                release_fall(&p, self.pool.env_level[v], left, exp_release)
            } else {
                0.0f32
            };

            // [37]
            let sr = self.cfg.sample_rate as f32;
            let mut use_menv = self.cfg.mod_env_enabled && p.flags & RP_MOD_ENV != 0;
            let mut mp = if use_menv {
                self.menv_of(variant, params_base)
            } else {
                ModEnvParams::default()
            };
            let mut rel_age = self.pool.rel_age[v];
            if use_menv && release_frame != usize::MAX {
                rel_age = note_age(age0.wrapping_add(release_frame as u32));
            }
            let mut menv_filter = use_menv && mp.to_filter != 0.0;

            'voice: for tile in 0..self.tiles {
                // [38]
                let mut tgt_l = base_gain_l;
                let mut tgt_r = base_gain_r;
                let mut snap = tile == 0;

                if self.bend_active || self.gain_active || self.variant_active || self.cut_active
                {
                    let ci = (tile * BEND_CHANNELS + channel) * CHAN_FIELDS;
                    // [39]
                    let bias = if born_variant != 0 && tile <= born_tile { born_bias } else { 0 };
                    let gi = ((tile + bias) * BEND_CHANNELS + channel) * CHAN_FIELDS;
                    // [40]
                    if self.cut_active {
                        let cut = self.chan_rows[ci + CHAN_CUT] as usize;
                        if cut != 0 && stop_rel == 0 {
                            let cut_id = (self.chan_rows[ci + CHAN_CUT_ID_HI] as u64) << 32
                                | self.chan_rows[ci + CHAN_CUT_ID_LO] as u64;
                            if self.pool.note_id[v] < cut_id {
                                stop_rel = cut;
                            }
                        }
                    }
                    if self.bend_active {
                        bent = base_step.scale(self.chan_rows[gi + CHAN_BEND]);
                        step = bent;
                    }
                    if self.gain_active {
                        tgt_l = base_gain_l * f32::from_bits(self.chan_rows[gi + CHAN_GAIN_L]);
                        tgt_r = base_gain_r * f32::from_bits(self.chan_rows[gi + CHAN_GAIN_R]);
                        // [41]
                        if self.cfg.gain_ramp && tile > born_tile {
                            d_gain_l = (tgt_l - gain_l) * inv_gate_tile;
                            d_gain_r = (tgt_r - gain_r) * inv_gate_tile;
                        } else {
                            gain_l = tgt_l;
                            gain_r = tgt_r;
                            d_gain_l = 0.0;
                            d_gain_r = 0.0;
                            snap = true;
                        }
                    }
                    // [42]
                    db0 = 0.0;
                    db1 = 0.0;
                    da1 = 0.0;
                    da2 = 0.0;
                    d_filter_mix = 0.0;
                    if self.variant_active {
                        let want = self.chan_rows[ci + CHAN_VARIANT];
                        // [43]
                        let born_here = born_variant != 0 && tile <= born_tile;
                        if want != variant && !born_here {
                            variant = want;
                            let was_filtering = use_filter;
                            p = self.params_of(variant, params_base);
                            // [44]
                            use_filter = p.flags & RP_FILTER != 0;
                            // [45]
                            if grid && stage == ENV_RELEASE {
                                let f = (tile * self.tile_frames) as u32;
                                let left = fall_left(phase_now, slot_word, f, step_frames);
                                rel_d = release_fall(&p, level, left, exp_release);
                            }
                            // [46]
                            use_menv = self.cfg.mod_env_enabled && p.flags & RP_MOD_ENV != 0;
                            if use_menv {
                                mp = self.menv_of(variant, params_base);
                            }
                            menv_filter = use_menv && mp.to_filter != 0.0;
                            // [47]
                            let ramp_ok = self.cfg.filter_ramp && tile > born_tile;
                            if ramp_ok && use_filter && was_filtering {
                                db0 = (p.b0 - cb0) * inv_gate_tile;
                                db1 = (p.b1 - cb1) * inv_gate_tile;
                                da1 = (p.a1 - ca1) * inv_gate_tile;
                                da2 = (p.a2 - ca2) * inv_gate_tile;
                            } else if ramp_ok && use_filter != was_filtering {
                                // [48]
                                if use_filter {
                                    z1 = 0.0;
                                    z2 = 0.0;
                                    cb0 = p.b0;
                                    cb1 = p.b1;
                                    ca1 = p.a1;
                                    ca2 = p.a2;
                                    filter_mix = 0.0;
                                    d_filter_mix = inv_gate_tile;
                                } else {
                                    filter_mix = 1.0;
                                    d_filter_mix = -inv_gate_tile;
                                }
                                db0 = 0.0;
                                db1 = 0.0;
                                da1 = 0.0;
                                da2 = 0.0;
                            } else {
                                cb0 = p.b0;
                                cb1 = p.b1;
                                ca1 = p.a1;
                                ca2 = p.a2;
                                db0 = 0.0;
                                db1 = 0.0;
                                da1 = 0.0;
                                da2 = 0.0;
                                filter_mix = if use_filter { 1.0 } else { 0.0 };
                            }
                        }
                    }
                }

                // [49]
                if vflags > crate::porta::FLAG_BITS {
                    let from = if self.bend_active { bent } else { base_step };
                    let f = (tile * self.tile_frames).max(start_rel) as u32;
                    bent = from.scale(crate::porta::factor(vflags, slot_word, f, &self.glide_tab));
                    step = bent;
                }

                // [50]
                if self.cfg.lfo_enabled {
                    let f = tile * self.tile_frames;
                    let age = note_age(age0.wrapping_add(f as u32));
                    let next = note_age(age0.wrapping_add((f + self.tile_frames) as u32));
                    let now = lfo_tremolo(&p, age);
                    let trem = lfo_tremolo(&p, next);
                    if snap {
                        gain_l = tgt_l * now;
                        gain_r = tgt_r * now;
                    }
                    d_gain_l = (tgt_l * trem - gain_l) * inv_gate_tile;
                    d_gain_r = (tgt_r * trem - gain_r) * inv_gate_tile;
                    let (vd, md) = (p.vib_lfo_delay(), p.mod_lfo_delay());
                    let (vp, mlp) = (p.vib_lfo_to_pitch(), p.mod_lfo_to_pitch());
                    let on_vib = (age >= vd && vp != 0.0) as u32;
                    let on_mod = (age >= md && mlp != 0.0) as u32;
                    let vib = lfo_tri(age.wrapping_sub(vd).wrapping_mul(p.vib_lfo_inc)) * on_vib as f32;
                    let modl =
                        lfo_tri(age.wrapping_sub(md).wrapping_mul(p.mod_lfo_inc)) * on_mod as f32;
                    let cents = vp * vib + mlp * modl;
                    // [51]
                    let one = 1u32 << 24;
                    let fx = ((cents * (1.0 / 1200.0)).exp2() * 16_777_216.0) as u32;
                    lfo_factor = one.wrapping_add(fx.wrapping_sub(one).wrapping_mul(on_vib | on_mod));
                    step = bent.scale(lfo_factor);
                }

                let f0 = tile * self.tile_frames;
                for i in 0..self.tile_frames {
                    let f = f0 + i;

                    // [52]
                    if f.is_multiple_of(rtile) && use_menv {
                        let age = note_age(age0.wrapping_add(f as u32));
                        let mut menv_factor = 0u32;
                        if use_menv {
                            let l = mod_env_level(&mp, age, rel_age, &self.bank.menv_log2);
                            // [53]
                            if mp.to_pitch != 0.0 {
                                let i = mod_env_pitch_index(
                                    mp.to_pitch * l,
                                    self.bank.menv_factor_half,
                                );
                                menv_factor =
                                    self.bank.menv_factors.get(i as usize).copied().unwrap_or(0);
                            }
                            if menv_filter {
                                // [54]
                                let fc = cents_to_hz(mp.fc_cents + mp.to_filter * l);
                                let c = biquad_lowpass_pre(fc, mp.q_gain, mp.q_inv_2q, sr);
                                cb0 = c.0;
                                cb1 = c.1;
                                ca1 = c.2;
                                ca2 = c.3;
                                db0 = 0.0;
                                db1 = 0.0;
                                da1 = 0.0;
                                da2 = 0.0;
                            }
                        }
                        // [55]
                        step = bent.scale(lfo_factor);
                        if menv_factor != 0 {
                            step = step.scale(menv_factor);
                        }
                    }

                    if f < start_rel {
                        continue;
                    }
                    if stage == ENV_DEAD {
                        break 'voice;
                    }

                    // [56]
                    if stage < ENV_RELEASE && f >= release_frame {
                        stage = ENV_RELEASE;
                        level = level.min(1.0);
                        release_frame = usize::MAX;
                        if grid {
                            rel_d = release_fall(&p, level, step_frames, exp_release);
                        }
                    }

                    let looping =
                        loop_enabled && !(loop_until_release && stage >= ENV_RELEASE);

                    let idx = phase.hi();
                    if !looping && idx >= len {
                        stage = ENV_DEAD;
                        break 'voice;
                    }

                    // [57]
                    match stage {
                        ENV_ATTACK => {
                            level += p.attack_rate;
                            if level >= p.attack_end {
                                stage = ENV_DECAY;
                                level = 1.0;
                            }
                        }
                        ENV_DECAY => {
                            level = if exp_decay {
                                level * p.decay_coef
                            } else {
                                level - p.decay_coef
                            };
                            if level <= p.decay_target {
                                if p.sustain <= floor {
                                    stage = ENV_DEAD;
                                    level = 0.0;
                                } else {
                                    stage = ENV_SUSTAIN;
                                    level = p.sustain;
                                }
                            }
                        }
                        ENV_SUSTAIN => {}
                        ENV_RELEASE => {
                            level = if grid {
                                if rel_d != 0.0 {
                                    level - rel_d
                                } else {
                                    level * p.release_coef
                                }
                            } else if exp_release {
                                level * p.release_coef
                            } else {
                                level - p.release_coef
                            };
                            if level <= floor {
                                stage = ENV_DEAD;
                                level = 0.0;
                            }
                        }
                        _ => {}
                    }

                    let s = self.interpolate(
                        base,
                        idx,
                        phase.lo() as f32 * FRAC_SCALE_F32,
                        looping,
                        loop_start,
                        loop_end,
                        len,
                    );

                    let g = level.min(1.0);
                    let x = s * g;

                    // Transposed direct form II. b2 == b0.
                    let y = if d_filter_mix != 0.0 {
                        // [58]
                        let fy = cb0 * x + z1;
                        z1 = cb1 * x - ca1 * fy + z2;
                        z2 = cb0 * x - ca2 * fy;
                        let out = x + (fy - x) * filter_mix;
                        // Advanced here, as in `render.wgsl`.
                        filter_mix += d_filter_mix;
                        out
                    } else if use_filter {
                        let y = cb0 * x + z1;
                        z1 = cb1 * x - ca1 * y + z2;
                        z2 = cb0 * x - ca2 * y;
                        y
                    } else {
                        x
                    };

                    // [59]
                    let y = if stop_rel != 0 && f + 1 >= stop_rel {
                        let d = f + 1 - stop_rel;
                        if d >= fade {
                            stage = ENV_DEAD;
                            0.0
                        } else {
                            y * (1.0 - d as f32 / fade as f32)
                        }
                    } else {
                        y
                    };
                    self.mix[f * 2] += (y * gain_l) as f64;
                    self.mix[f * 2 + 1] += (y * gain_r) as f64;
                    // [60]
                    gain_l += d_gain_l;
                    gain_r += d_gain_r;
                    cb0 += db0;
                    cb1 += db1;
                    ca1 += da1;
                    ca2 += da2;

                    // ---- advance the phase ----
                    phase = phase.wrapping_add(step);
                    if looping {
                        let hi = phase.hi();
                        if hi >= loop_end {
                            let span = (loop_end - loop_start).max(1);
                            let wrapped = loop_start + (hi - loop_start) % span;
                            phase = Fixed::from_parts(wrapped, phase.lo());
                        }
                    }
                }
            }

            self.pool.phase[v] = phase;
            self.pool.env_stage[v] = stage;
            self.pool.env_level[v] = level;
            self.pool.filt_z1[v] = z1;
            self.pool.filt_z2[v] = z2;
            self.pool.start_rel[v] = 0;
            self.pool.born_variant[v] = 0;
            self.pool.stop_rel[v] = 0;
            if vflags > crate::porta::FLAG_BITS {
                self.pool.flags[v] = crate::porta::advance(vflags, self.cfg.block_frames);
            }
            // [61]
            self.pool.age[v] = age0.wrapping_add(self.cfg.block_frames);
            self.pool.rel_age[v] = rel_age;
        }

        // Reduce to the output block.
        let mut peak = 0.0f32;
        for (o, m) in out.iter_mut().zip(&self.mix).take(block * 2) {
            let v = *m as f32;
            *o = v;
            let a = v.abs();
            if a > peak {
                peak = a;
            }
        }
        self.peak = peak;

        // Compaction. Dead voices leave, order is preserved.
        let keep: Vec<bool> = self
            .pool
            .env_stage
            .iter()
            .map(|&s| s != ENV_DEAD)
            .collect();
        if keep.iter().any(|k| !k) {
            self.pool.swap_remove_compact(&keep);
        }

        self.env_phase = (self.env_phase + self.cfg.block_frames) % step_frames;
        Ok(())
    }

    fn stats(&self) -> BlockStats {
        BlockStats {
            active_voices: self.pool.len() as u64,
            stolen: self.stolen,
            dropped: self.dropped,
            peak: self.peak,
        }
    }

    fn name(&self) -> &'static str {
        "cpu-reference"
    }
}
