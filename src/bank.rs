// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Format-independent instrument bank. \[1\]

use anyhow::{bail, Result};
use crate::config::{Config, EnvelopeCurve};
use crate::fixed::Fixed;
use std::f64::consts::PI;

/// Voice flag bits. Mirrored by `shaders/common.wgsl`.
pub const VF_LOOP: u32 = 1 << 0;
/// Loop only until the note is released, then run out the tail (SF2 mode 3).
pub const VF_LOOP_UNTIL_RELEASE: u32 = 1 << 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    NoLoop,
    Continuous,
    UntilRelease,
}

impl LoopMode {
    pub fn flags(self) -> u32 {
        match self {
            LoopMode::NoLoop => 0,
            LoopMode::Continuous => VF_LOOP,
            LoopMode::UntilRelease => VF_LOOP | VF_LOOP_UNTIL_RELEASE,
        }
    }
}

/// One sample in the flat pool.
#[derive(Debug, Clone)]
pub struct SampleInfo {
    /// Index of the first frame in `Bank::pool`.
    pub start: u32,
    pub len: u32,
    /// Loop points relative to `start`. Always valid to read: when the source \[2\]
    pub loop_start: u32,
    pub loop_end: u32,
    /// Whether the *source* actually declared those points -- a `smpl` chunk \[3\]
    pub declared_loop: bool,
    /// Rate the data is stored at. After a pool rebuild this equals \[4\]
    pub rate: u32,
    pub root_key: u8,
    /// Sample-level tuning, in cents.
    pub correction_cents: f32,
    /// Pool frames per source frame. Region address offsets are quoted in \[5\]
    pub resample_ratio: f32,
    pub name: String,
}

/// Effective sample coordinates shared by voice spawning and phase preparation.
pub(crate) fn sample_geometry(s: &SampleInfo, r: &Region) -> (u32, u32, u32, u32, u32) {
    let scale = |v: i32| (v as f32 * s.resample_ratio).round() as i64;
    let start = scale(r.addr_start).clamp(0, s.len as i64 - 1) as u32;
    let len = (s.len as i64 + scale(r.addr_end)).clamp(1, s.len as i64) as u32;
    let ls = (s.loop_start as i64 + scale(r.addr_loop_start)).clamp(0, len as i64 - 1) as u32;
    let le = (s.loop_end as i64 + scale(r.addr_loop_end)).clamp(0, len as i64) as u32;
    let flags = if le <= ls + 1 { 0 } else { r.loop_mode.flags() };
    (start, len, ls, le, flags)
}

/// A key/velocity zone with every SF2 generator already applied.
#[derive(Debug, Clone)]
pub struct Region {
    pub sample: u32,
    pub key_lo: u8,
    pub key_hi: u8,
    pub vel_lo: u8,
    pub vel_hi: u8,
    /// Overrides the sample's root key when >= 0.
    pub root_key_override: i16,
    /// Fixed key/velocity for drum-style regions; -1 when unset.
    pub fixed_key: i16,
    pub fixed_vel: i16,
    pub coarse_tune: i16,
    pub fine_tune: i16,
    /// Cents of pitch change per key. 100 is normal, 0 pins the pitch.
    pub scale_tuning: i16,
    pub attenuation_cb: f32,
    /// SFZ `amp_veltrack`, in percent. 100 is full velocity tracking, 0 pins \[6\]
    pub amp_veltrack: f32,
    /// -1.0 hard left to 1.0 hard right.
    pub pan: f32,
    pub loop_mode: LoopMode,
    /// SF2 address offset generators, in source frames, applied on top of the \[7\]
    pub addr_start: i32,
    pub addr_end: i32,
    pub addr_loop_start: i32,
    pub addr_loop_end: i32,
    pub exclusive_class: u8,
    /// SFZ `lorand`/`hirand`: this region matches only when the note-on's \[8\]
    pub rand_lo: f32,
    pub rand_hi: f32,
    /// SFZ velocity crossfade, `xfin_lovel`/`xfin_hivel` (fade in) and \[9\]
    pub xfin_lo: u8,
    pub xfin_hi: u8,
    pub xfout_lo: u8,
    pub xfout_hi: u8,

    // Volume envelope, in seconds, before key scaling.
    pub delay: f32,
    pub attack: f32,
    pub hold: f32,
    pub decay: f32,
    /// Sustain as a linear level in [0, 1].
    pub sustain: f32,
    pub release: f32,
    /// SF2 keynumToVolEnvHold / Decay, in timecents per key relative to key 60.
    pub keynum_to_hold: i16,
    pub keynum_to_decay: i16,

    pub filter_fc_cents: f32,
    pub filter_q_cb: f32,
    /// SFZ `fil_veltrack`: how far the cutoff opens at full velocity, in \[10\]
    pub filter_veltrack_cents: f32,

    // [11]
    pub mod_lfo_delay: f32,
    pub vib_lfo_delay: f32,
    /// Rates in Hz, from `freqModLFO` / `freqVibLFO`.
    pub mod_lfo_hz: f32,
    pub vib_lfo_hz: f32,
    /// Peak deviation the LFOs apply, in the SF2 units of each destination: \[12\]
    pub mod_lfo_to_pitch: f32,
    pub vib_lfo_to_pitch: f32,
    pub mod_lfo_to_volume: f32,

    // [13]
    pub mod_env_delay: f32,
    pub mod_env_attack: f32,
    pub mod_env_hold: f32,
    /// Seconds for a *full-scale* fall. Reaching a sustain of 0.5 takes half \[14\]
    pub mod_env_decay: f32,
    /// Level the decay settles at, in [0, 1], already converted from the \[15\]
    pub mod_env_sustain: f32,
    /// Seconds for a full-scale fall, from wherever the level stands.
    pub mod_env_release: f32,
    /// Cents of pitch at full modulation.
    pub mod_env_to_pitch: f32,
    /// Cents of filter cutoff at full modulation.
    pub mod_env_to_filter: f32,

    /// First entry of this region's block in `Bank::params`.
    pub params_base: u32,
    /// 0 when one entry covers every key, 1 when there is an entry per key.
    pub params_stride: u32,
    /// Entries per velocity step in this region's block, 1 when the cutoff \[16\]
    pub params_vel_span: u32,
}

impl Default for Region {
    fn default() -> Self {
        Region {
            sample: 0,
            key_lo: 0,
            key_hi: 127,
            vel_lo: 0,
            vel_hi: 127,
            root_key_override: -1,
            fixed_key: -1,
            fixed_vel: -1,
            coarse_tune: 0,
            fine_tune: 0,
            scale_tuning: 100,
            attenuation_cb: 0.0,
            amp_veltrack: 100.0,
            pan: 0.0,
            loop_mode: LoopMode::NoLoop,
            addr_start: 0,
            addr_end: 0,
            addr_loop_start: 0,
            addr_loop_end: 0,
            exclusive_class: 0,
            rand_lo: 0.0,
            rand_hi: 1.0,
            xfin_lo: 0,
            xfin_hi: 0,
            xfout_lo: 127,
            xfout_hi: 127,
            delay: 0.0,
            attack: 0.001,
            hold: 0.0,
            decay: 0.001,
            sustain: 1.0,
            release: 0.001,
            keynum_to_hold: 0,
            keynum_to_decay: 0,
            filter_fc_cents: 13500.0,
            filter_veltrack_cents: 0.0,
            mod_lfo_delay: 0.0,
            vib_lfo_delay: 0.0,
            // 8.176 Hz is absolute cents zero, the SF2 default for both.
            mod_lfo_hz: 8.176,
            vib_lfo_hz: 8.176,
            mod_lfo_to_pitch: 0.0,
            vib_lfo_to_pitch: 0.0,
            mod_lfo_to_volume: 0.0,
            mod_env_delay: 0.0,
            mod_env_attack: 0.0,
            mod_env_hold: 0.0,
            mod_env_decay: 0.0,
            // No drop below the peak, which is `sustainModEnv = 0`.
            mod_env_sustain: 1.0,
            mod_env_release: 0.0,
            mod_env_to_pitch: 0.0,
            mod_env_to_filter: 0.0,
            filter_q_cb: 0.0,
            params_base: 0,
            params_stride: 0,
            params_vel_span: 1,
        }
    }
}

/// Per-(region, key) DSP constants, uploaded once and read by the render pass \[17\]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegionParams {
    /// Per-frame increment during attack.
    pub attack_rate: f32,
    /// Attack ends when the raw level reaches this. Hold is folded in here: \[18\]
    pub attack_end: f32,
    /// Multiplier (exponential curve) or decrement (linear curve) per frame.
    pub decay_coef: f32,
    /// Decay stops here: max(sustain, env_floor).
    pub decay_target: f32,
    pub sustain: f32,
    pub release_coef: f32,
    pub b0: f32,
    pub b1: f32,
    pub a1: f32,
    pub a2: f32,
    /// bit 0: run the filter (`RP_FILTER`); bit 1, `RP_MOD_ENV`; bit 2, \[19\]
    pub flags: u32,

    // [20]
    pub mod_lfo_inc: u32,
    pub vib_lfo_inc: u32,
    /// Start delays in frames, `mod` in the low half and `vib` in the high.
    pub lfo_delays: u32,
    /// Peak pitch deviation in cents as two `i16`, `mod` low and `vib` high.
    pub lfo_pitch: u32,
}

/// SF2's LFO waveform: a triangle starting at zero, in `[-1, 1]`. \[21\]
#[inline]
pub fn lfo_tri(phase: u32) -> f32 {
    // [22]
    let t = phase.wrapping_add(0x4000_0000) as f32 * (1.0 / 4_294_967_296.0);
    1.0 - (4.0 * t - 2.0).abs()
}

/// Per-(region, key, velocity, variant) constants for the modulation envelope, \[23\]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModEnvParams {
    pub delay_frames: u32,
    /// Zero means instant, which is the default and the common case.
    pub attack_frames: u32,
    pub hold_frames: u32,
    /// Per-frame fall during decay, as a fraction of full scale.
    pub decay_rate: f32,
    pub sustain: f32,
    /// Per-frame fall during release, as a fraction of full scale.
    pub release_rate: f32,
    /// Cents of pitch at full modulation.
    pub to_pitch: f32,
    /// Cents of cutoff at full modulation.
    pub to_filter: f32,
    /// The unmodulated cutoff this entry was built for, in absolute cents. \[24\]
    pub fc_cents: f32,
    /// The two resonance-dependent terms of `biquad_lowpass`, precomputed so \[25\]
    pub q_gain: f32,
    pub q_inv_2q: f32,
    pub _pad: [f32; 5],
}

impl Default for ModEnvParams {
    fn default() -> Self {
        ModEnvParams {
            delay_frames: 0,
            attack_frames: 0,
            hold_frames: 0,
            decay_rate: 0.0,
            sustain: 1.0,
            release_rate: 0.0,
            to_pitch: 0.0,
            to_filter: 0.0,
            fc_cents: 13500.0,
            q_gain: 1.0,
            q_inv_2q: 1.0,
            _pad: [0.0; 5],
        }
    }
}

/// How far into its attack the modulation envelope is, given a linear ramp. \[26\]
#[inline]
pub fn mod_env_attack_level(u: f32, log2_tab: &[u32]) -> f32 {
    if u <= 0.0 {
        return 0.0;
    }
    (1.0 + quantise_level(log2_exact(u, log2_tab) * MOD_ENV_ATTACK_SCALE)).clamp(0.0, 1.0)
}

/// Entries in the shared `log2` mantissa table, as a power of two.
pub const MOD_ENV_LOG2_BITS: u32 = 10;
/// Bits of the mantissa used to interpolate between two entries.
pub const MOD_ENV_LOG2_FRAC_BITS: u32 = 8;

/// `log2(1 + i / 1024)` for `i` in `0..=1024`, in 0.30 fixed point. \[27\]
pub fn build_log2_tab() -> Vec<u32> {
    let n = 1usize << MOD_ENV_LOG2_BITS;
    (0..=n)
        .map(|i| {
            let v = (1.0 + i as f64 / n as f64).log2() * (1u64 << 30) as f64;
            v.round().clamp(0.0, (1u64 << 30) as f64) as u32
        })
        .collect()
}

/// `log2` computed identically on the host and on the device. \[28\]
#[inline]
pub fn log2_exact(x: f32, tab: &[u32]) -> f32 {
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let man = bits & 0x007F_FFFF;
    let i = (man >> (23 - MOD_ENV_LOG2_BITS)) as usize;
    let frac = (man >> (23 - MOD_ENV_LOG2_BITS - MOD_ENV_LOG2_FRAC_BITS))
        & ((1 << MOD_ENV_LOG2_FRAC_BITS) - 1);
    let (a, b) = match (tab.get(i), tab.get(i + 1)) {
        (Some(a), Some(b)) => (*a, *b),
        _ => return 0.0,
    };
    let v = a + (((b - a) * frac) >> MOD_ENV_LOG2_FRAC_BITS);
    e as f32 + v as f32 * (1.0 / 1_073_741_824.0)
}

/// `(20 / 48) * log10(2)`, the slope of the attack curve against `log2(u)`. \[29\]
pub const MOD_ENV_ATTACK_SCALE: f32 = 0.125_429_17;

/// Steps per cent in the modulation envelope's pitch-factor table. \[30\]
pub const MOD_ENV_CENTS_STEPS: f32 = 64.0;

/// Index into `Bank::menv_factors` for a pitch offset in cents. \[31\]
#[inline]
pub fn mod_env_pitch_index(cents: f32, half: u32) -> u32 {
    // [32]
    let q = (cents * MOD_ENV_CENTS_STEPS).round_ties_even();
    let i = q + half as f32;
    i.clamp(0.0, (half * 2) as f32) as u32
}

/// Round to a multiple of 2^-22, through an integer. \[33\]
#[inline]
pub fn quantise_level(x: f32) -> f32 {
    const S: f32 = 4_194_304.0; // 2^22
    let t = (x.clamp(-2.0, 2.0) * S) as i32;
    t as f32 * (1.0 / S)
}

/// A release age no voice can reach, meaning "this voice has not released". \[34\]
pub const NO_RELEASE_AGE: u32 = u32::MAX;

/// The modulation envelope's level at a given age, in [0, 1]. \[35\]
#[inline]
pub fn mod_env_level(p: &ModEnvParams, age: u32, release_age: u32, log2_tab: &[u32]) -> f32 {
    // [36]
    let pre = mod_env_pre_release(p, age.min(release_age), log2_tab);
    if age <= release_age {
        return pre;
    }
    // [37]
    (pre - quantise_level((age - release_age) as f32 * p.release_rate)).max(0.0)
}

#[inline]
fn mod_env_pre_release(p: &ModEnvParams, age: u32, log2_tab: &[u32]) -> f32 {
    if age < p.delay_frames {
        return 0.0;
    }
    let a = age - p.delay_frames;
    if a < p.attack_frames {
        return mod_env_attack_level(a as f32 / p.attack_frames as f32, log2_tab);
    }
    let a = a - p.attack_frames;
    if a < p.hold_frames {
        return 1.0;
    }
    let a = a - p.hold_frames;
    (1.0 - quantise_level(a as f32 * p.decay_rate)).max(p.sustain)
}

/// `biquad_lowpass` with the resonance-dependent terms already computed. \[38\]
#[inline]
pub fn biquad_lowpass_pre(fc: f32, q_gain: f32, inv_2q: f32, sr: f32) -> (f32, f32, f32, f32) {
    let w0 = 2.0 * std::f32::consts::PI * (fc / sr).clamp(1.0e-5, 0.49);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 * inv_2q;

    let a0 = 1.0 + alpha;
    let b0 = q_gain * (1.0 - cos_w0) * 0.5 / a0;
    let b1 = q_gain * (1.0 - cos_w0) / a0;
    let a1 = -2.0 * cos_w0 / a0;
    let a2 = (1.0 - alpha) / a0;
    (b0, b1, a1, a2)
}

pub const RP_FILTER: u32 = 1 << 0;
/// This region drives a modulation envelope with at least one live \[39\]
pub const RP_MOD_ENV: u32 = 1 << 1;
/// This region's release would finish inside one step of the envelope grid, so \[40\]
pub const RP_SHORT_RELEASE: u32 = 1 << 2;

impl RegionParams {
    /// The three packed LFO amounts, unpacked. Mirrored in `render.wgsl`.
    #[inline]
    pub fn mod_lfo_delay(&self) -> u32 {
        self.lfo_delays & 0xFFFF
    }
    #[inline]
    pub fn vib_lfo_delay(&self) -> u32 {
        self.lfo_delays >> 16
    }
    #[inline]
    pub fn mod_lfo_to_pitch(&self) -> f32 {
        ((self.lfo_pitch << 16) as i32 >> 16) as f32
    }
    #[inline]
    pub fn vib_lfo_to_pitch(&self) -> f32 {
        (self.lfo_pitch as i32 >> 16) as f32
    }
    #[inline]
    pub fn mod_lfo_to_volume(&self) -> f32 {
        (self.flags as i32 >> 16) as f32
    }
}

/// What a channel's sound controllers do to a region's DSP constants. \[41\]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamMod {
    /// CC74 brightness, as an offset to the region's cutoff in cents.
    pub cutoff_cents: f32,
    /// CC71 resonance, as an offset in centibels of Q. Never negative: see \[42\]
    pub q_cb: f32,
    /// CC73, CC75, CC72: what happens to each envelope time.
    pub attack: TimeMod,
    pub decay: TimeMod,
    pub release: TimeMod,
}

/// What a sound controller does to one envelope time. \[43\]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeMod {
    pub scale: f32,
    /// Seconds added, on top of the scaled time.
    pub add: f32,
}

impl TimeMod {
    pub const NONE: TimeMod = TimeMod { scale: 1.0, add: 0.0 };

    #[inline]
    pub fn apply(self, secs: f32) -> f32 {
        secs * self.scale + self.add
    }

    /// One controller's contribution, 64 being neutral. \[44\]
    pub fn from_controller(v: u8) -> TimeMod {
        let d = v as f32 - 64.0;
        if d <= 0.0 {
            TimeMod { scale: (d / CC_ENV_STEPS_PER_OCTAVE).exp2(), add: 0.0 }
        } else {
            TimeMod { scale: 1.0, add: CC_ENV_ADD_PER_CUBE * d * d * d }
        }
    }
}

impl Default for TimeMod {
    fn default() -> Self {
        TimeMod::NONE
    }
}

impl Default for ParamMod {
    fn default() -> Self {
        ParamMod {
            cutoff_cents: 0.0,
            q_cb: 0.0,
            attack: TimeMod::NONE,
            decay: TimeMod::NONE,
            release: TimeMod::NONE,
        }
    }
}

// [45]

/// Cents of cutoff offset per CC74 step. \[46\]
pub const CC_CUTOFF_CENTS_PER_STEP: f32 = 75.0;

/// Centibels of resonance per CC71 step **above 64**. Below 64 BASSMIDI does \[47\]
pub const CC_Q_CB_PER_STEP: f32 = 240.0 / 63.0;

/// Controller steps per octave of envelope time below 64. The time halves every \[48\]
pub const CC_ENV_STEPS_PER_OCTAVE: f32 = 8.0;

/// Seconds of envelope time *added* per cubed step above 64. \[49\]
pub const CC_ENV_ADD_PER_CUBE: f32 = 6.0e-5;

/// The shortest fade BASSMIDI performs, in seconds. \[50\]
pub const CC_MIN_FADE_SECS: f32 = 0.004;

impl ParamMod {
    pub fn is_neutral(&self) -> bool {
        *self == ParamMod::default()
    }

    /// Build from the raw controller values, 64 being the neutral position for \[51\]
    pub fn from_controllers(cc71: u8, cc72: u8, cc73: u8, cc74: u8, cc75: u8) -> Self {
        ParamMod {
            cutoff_cents: (cc74 as f32 - 64.0) * CC_CUTOFF_CENTS_PER_STEP,
            // [52]
            q_cb: (cc71 as f32 - 64.0).max(0.0) * CC_Q_CB_PER_STEP,
            attack: TimeMod::from_controller(cc73),
            decay: TimeMod::from_controller(cc75),
            release: TimeMod::from_controller(cc72),
        }
    }
}

/// A bank/program pair with its zones flattened.
#[derive(Debug, Clone)]
pub struct Preset {
    pub bank: u16,
    pub program: u16,
    pub name: String,
    /// Indices into `Bank::regions`.
    pub regions: Vec<u32>,
    /// `key_index[k] .. key_index[k+1]` slices `key_regions` for key `k`.
    pub key_index: Vec<u32>,
    pub key_regions: Vec<u32>,
}

impl Preset {
    fn build_key_index(&mut self, regions: &[Region]) {
        let mut per_key: Vec<Vec<u32>> = vec![Vec::new(); 128];
        for &r in &self.regions {
            let reg = &regions[r as usize];
            for k in reg.key_lo..=reg.key_hi.min(127) {
                per_key[k as usize].push(r);
            }
        }
        self.key_index = Vec::with_capacity(129);
        self.key_regions = Vec::new();
        for list in per_key {
            self.key_index.push(self.key_regions.len() as u32);
            self.key_regions.extend_from_slice(&list);
        }
        self.key_index.push(self.key_regions.len() as u32);
    }
}

/// Everything the renderer needs from a soundfont.
pub struct Bank {
    pub pool: Vec<i16>,
    /// Rate every pool sample is stored at, or 0 when the pool is mixed-rate \[53\]
    pub pool_rate: u32,
    pub samples: Vec<SampleInfo>,
    pub regions: Vec<Region>,
    pub params: Vec<RegionParams>,
    /// Modulation-envelope constants, parallel to `params` and read with the \[54\]
    pub menv: Vec<ModEnvParams>,
    /// Phase-step factors in 8.24 fixed point for the modulation envelope's \[55\]
    pub menv_factors: Vec<u32>,
    pub menv_factor_half: u32,
    /// Shared `log2` mantissa table; see `log2_exact`. Empty when the bank \[56\]
    pub menv_log2: Vec<u32>,
    pub presets: Vec<Preset>,
    /// Sorted (bank, program) -> preset index.
    pub(crate) index: Vec<((u16, u16), u32)>,
    pub name: String,
    /// True when any region drives an LFO anywhere. \[57\]
    pub uses_lfo: bool,
    /// Whether any region drives the tremolo, and whether any drives pitch \[58\]
    pub uses_lfo_volume: bool,
    pub uses_lfo_pitch: bool,
    /// True when any region narrows `lorand`/`hirand`. \[59\]
    pub uses_rand: bool,
    /// True when any region drives a modulation envelope destination. \[60\]
    pub uses_mod_env: bool,

    // [61]
    pub(crate) gain_table: Vec<[f32; 2]>,
    /// `build_voice`'s `delay_frames` per region. A region constant, and the \[62\]
    pub(crate) delay_frames: Vec<u32>,
    /// Bitset over `region * 128 + key`, set when `build_voice` would return \[63\]
    pub(crate) key_ok: Vec<u64>,
}

/// One layer a note-on would produce, as far as admission needs to know it. \[64\]
#[derive(Debug, Clone, Copy, Default)]
pub struct PreviewLayer {
    pub region: u32,
    pub gain_l: f32,
    pub gain_r: f32,
    /// `VoiceSpawn::delay_frames`. Admission never sees a delayed voice -- it \[65\]
    pub delay_frames: u32,
}

/// Everything the device needs to start one voice. Produced by `note_on`.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoiceSpawn {
    pub phase: Fixed,
    pub step: Fixed,
    pub smp_base: u32,
    pub smp_len: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    pub flags: u32,
    pub params: u32,
    pub region: u32,
    pub gain_l: f32,
    pub gain_r: f32,
    /// Frames to wait before the voice starts, from SF2 delayVolEnv.
    pub delay_frames: u32,
    pub exclusive_class: u8,
}

#[inline]
pub fn timecents_to_secs(tc: f32) -> f32 {
    // -32768 is the SF2 idiom for "instant".
    if tc <= -12000.0 {
        0.0
    } else {
        (2.0f32).powf(tc / 1200.0)
    }
}

#[inline]
pub fn cents_to_hz(cents: f32) -> f32 {
    8.176 * (2.0f32).powf(cents / 1200.0)
}

/// Where a channel's volume (CC7) sits before a file sends one: 100, General \[66\]
pub const POWER_ON_VOLUME: u8 = 100;

/// The amplitude power-on channel volume stands for, `(100/127)^2`. It is part \[67\]
pub const POWER_ON_GAIN: f32 =
    (POWER_ON_VOLUME as f32 / 127.0) * (POWER_ON_VOLUME as f32 / 127.0);

#[inline]
pub fn cb_to_gain(cb: f32) -> f32 {
    (10.0f32).powf(-cb / 200.0)
}

/// The random draw for one note-on, in `[0, 1)`. \[68\]
#[inline]
pub fn rand_draw(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 24 bits into an f32 mantissa, so the result is exact and always < 1.0.
    ((z >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Linear gain from SFZ velocity crossfade, in `[0, 1]`. \[69\]
#[inline]
pub fn xfade_gain(r: &Region, vel: u8) -> f32 {
    // [70]
    let fade_in = if vel >= r.xfin_hi {
        1.0
    } else if vel <= r.xfin_lo {
        0.0
    } else {
        (vel - r.xfin_lo) as f32 / (r.xfin_hi - r.xfin_lo) as f32
    };
    // Fade out: open at and below `lo`, silent above `hi`. Default 127..127.
    let fade_out = if vel <= r.xfout_lo {
        1.0
    } else if vel >= r.xfout_hi {
        0.0
    } else {
        1.0 - (vel - r.xfout_lo) as f32 / (r.xfout_hi - r.xfout_lo) as f32
    };
    fade_in * fade_out
}

/// Velocity to attenuation, in centibels, scaled by SFZ `amp_veltrack`. \[71\]
#[inline]
pub fn velocity_atten_cb(vel: u8, veltrack: f32) -> f32 {
    let v = vel.max(1) as f32 / 127.0;
    let t = veltrack.clamp(-100.0, 100.0) / 100.0;
    if t == 1.0 {
        // [72]
        return (-400.0 * v.log10()).clamp(0.0, 960.0);
    }
    let gain = if t >= 0.0 {
        1.0 - t * (1.0 - v * v)
    } else {
        1.0 + t * v * v
    };
    if gain <= 0.0 {
        return 960.0;
    }
    (-200.0 * gain.log10()).clamp(0.0, 960.0)
}

impl Bank {
    /// Layer `other` on top of this bank. \[73\]
    pub fn merge(&mut self, mut other: Bank) {
        let pool_off = self.pool.len() as u32;
        let sample_off = self.samples.len() as u32;
        let region_off = self.regions.len() as u32;

        // [74]
        if self.pool_rate != other.pool_rate {
            self.pool_rate = 0;
        }

        self.pool.extend_from_slice(&other.pool);
        for s in &mut other.samples {
            s.start += pool_off;
        }
        self.samples.append(&mut other.samples);

        for r in &mut other.regions {
            // [75]
            if r.sample != u32::MAX {
                r.sample += sample_off;
            }
        }
        self.regions.append(&mut other.regions);

        for p in &mut other.presets {
            for r in &mut p.regions {
                *r += region_off;
            }
        }
        for p in other.presets {
            self.presets.retain(|q| (q.bank, q.program) != (p.bank, p.program));
            self.presets.push(p);
        }

        self.name = format!("{} + {}", self.name, other.name);
    }

    /// Move this bank's single preset onto the given programs of bank 0. \[76\]
    pub fn remap_to_programs(&mut self, programs: &[u16]) -> Result<()> {
        if self.presets.len() != 1 {
            bail!(
                "--sf-programs needs a soundfont with exactly one preset, but {} has {}",
                self.name,
                self.presets.len()
            );
        }
        let base = self.presets.remove(0);
        for &prog in programs {
            let mut p = base.clone();
            p.bank = 0;
            p.program = prog;
            self.presets.push(p);
        }
        Ok(())
    }

    pub fn finish(&mut self) {
        for p in &mut self.presets {
            p.build_key_index(&self.regions);
        }
        self.uses_rand = self
            .regions
            .iter()
            .any(|r| r.rand_lo > 0.0 || r.rand_hi < 1.0);
        self.uses_lfo_volume = self.regions.iter().any(|r| r.mod_lfo_to_volume != 0.0);
        self.uses_lfo_pitch = self
            .regions
            .iter()
            .any(|r| r.mod_lfo_to_pitch != 0.0 || r.vib_lfo_to_pitch != 0.0);
        self.uses_lfo = self.uses_lfo_volume || self.uses_lfo_pitch;
        // [77]
        self.uses_mod_env = self
            .regions
            .iter()
            .any(|r| r.mod_env_to_pitch != 0.0 || r.mod_env_to_filter != 0.0);
        self.index = self
            .presets
            .iter()
            .enumerate()
            .map(|(i, p)| ((p.bank, p.program), i as u32))
            .collect();
        self.index.sort_by_key(|e| e.0);
    }

    /// Resolve a (bank, program) pair, falling back the way hardware does: \[78\]
    pub fn find_preset(&self, bank: u16, program: u16) -> Option<u32> {
        if let Ok(i) = self.index.binary_search_by_key(&(bank, program), |e| e.0) {
            return Some(self.index[i].1);
        }
        if bank == 128 {
            // Kit 0 is Standard, and is the kit every GM device falls back to.
            if let Ok(i) = self.index.binary_search_by_key(&(128, 0), |e| e.0) {
                return Some(self.index[i].1);
            }
            if let Some(e) = self.index.iter().find(|e| e.0 .0 == 128) {
                return Some(e.1);
            }
            // [79]
        }
        if let Ok(i) = self.index.binary_search_by_key(&(0, program), |e| e.0) {
            return Some(self.index[i].1);
        }
        self.index.first().map(|e| e.1)
    }

    /// Build the voices for one note-on. Pushes into `out` rather than \[80\]
    #[allow(clippy::too_many_arguments)]
    pub fn note_on(
        &self,
        preset: u32,
        key: u8,
        vel: u8,
        seed: u64,
        cfg: &Config,
        max_layers: usize,
        out: &mut Vec<VoiceSpawn>,
    ) {
        let Some(p) = self.presets.get(preset as usize) else {
            return;
        };
        let key = key.min(127);
        let lo = p.key_index[key as usize] as usize;
        let hi = p.key_index[key as usize + 1] as usize;
        let mut layers = 0usize;
        // [81]
        let draw = if self.uses_rand { rand_draw(seed) } else { 0.0 };

        for &ri in &p.key_regions[lo..hi] {
            if layers >= max_layers {
                break;
            }
            let r = &self.regions[ri as usize];
            if vel < r.vel_lo || vel > r.vel_hi {
                continue;
            }
            if self.uses_rand && (draw < r.rand_lo || draw >= r.rand_hi) {
                continue;
            }
            if let Some(v) = self.build_voice(r, ri, key, vel, cfg) {
                out.push(v);
                layers += 1;
            }
        }
    }

    /// Build the one layer `preview_note_on` named, by its region index. \[82\]
    pub fn build_layer(
        &self,
        region: u32,
        key: u8,
        vel: u8,
        cfg: &Config,
    ) -> Option<VoiceSpawn> {
        let r = self.regions.get(region as usize)?;
        self.build_voice(r, region, key.min(127), vel, cfg)
    }

    fn build_voice(
        &self,
        r: &Region,
        region_idx: u32,
        key: u8,
        vel: u8,
        cfg: &Config,
    ) -> Option<VoiceSpawn> {
        let s = self.samples.get(r.sample as usize)?;
        if s.len == 0 {
            return None;
        }

        let eff_key = if r.fixed_key >= 0 { r.fixed_key as u8 } else { key };
        let eff_vel = if r.fixed_vel >= 0 { r.fixed_vel as u8 } else { vel };

        let root = if r.root_key_override >= 0 {
            r.root_key_override as f64
        } else {
            s.root_key as f64
        };

        // Pitch, computed in f64 and only then frozen to fixed point.
        let cents = (eff_key as f64 - root) * r.scale_tuning as f64
            + r.coarse_tune as f64 * 100.0
            + r.fine_tune as f64
            + s.correction_cents as f64;
        let mut ratio = (cents / 1200.0).exp2();

        // [83]
        if self.pool_rate == 0 {
            ratio *= s.rate as f64 / cfg.sample_rate as f64;
        } else {
            ratio *= self.pool_rate as f64 / cfg.sample_rate as f64;
        }
        if !(ratio.is_finite() && ratio > 0.0) {
            return None;
        }

        let atten = r.attenuation_cb + velocity_atten_cb(eff_vel, r.amp_veltrack);
        // [84]
        let gain = cb_to_gain(atten) * cfg.master_volume * POWER_ON_GAIN * xfade_gain(r, eff_vel);

        // Constant-power pan.
        let theta = (r.pan.clamp(-1.0, 1.0) + 1.0) * 0.5 * (PI as f32 * 0.5);
        let gain_l = gain * theta.cos();
        let gain_r = gain * theta.sin();

        // [85]
        let (start_offset, smp_len, loop_start, loop_end, flags) = sample_geometry(s, r);

        // [86]
        let keys = if r.params_stride != 0 { 128u32 } else { 1 };
        let ki = if r.params_stride != 0 { eff_key as u32 } else { 0 };
        let vi = if r.params_vel_span > 1 {
            (vel.saturating_sub(r.vel_lo) as u32).min(r.params_vel_span - 1)
        } else {
            0
        };
        let params = r.params_base + vi * keys + ki;

        Some(VoiceSpawn {
            phase: Fixed::from_int(start_offset),
            step: Fixed::from_f64(ratio),
            smp_base: s.start,
            smp_len,
            loop_start,
            loop_end,
            flags,
            params,
            region: region_idx,
            gain_l,
            gain_r,
            delay_frames: (r.delay * cfg.sample_rate as f32) as u32,
            exclusive_class: r.exclusive_class,
        })
    }

    /// Fill `params` from `regions`. Call once after all regions exist.
    pub fn build_params(&mut self, cfg: &Config) {
        // [87]
        let mut n = 0u32;
        for r in &mut self.regions {
            let per_key = r.keynum_to_decay != 0 || r.keynum_to_hold != 0;
            // [88]
            let vel_span = if r.filter_veltrack_cents != 0.0 {
                (r.vel_hi.saturating_sub(r.vel_lo) as u32 + 1).min(128)
            } else {
                1
            };
            r.params_base = n;
            r.params_stride = if per_key { 1 } else { 0 };
            r.params_vel_span = vel_span;
            n += vel_span * if per_key { 128 } else { 1 };
        }
        self.params = self.build_variant(cfg, &ParamMod::default());
        self.menv = self.build_menv_variant(cfg, &ParamMod::default());
        self.build_menv_factors();
        self.build_admission_tables(cfg);
    }

    /// Precompute what admission needs to rank a note-on without building it. \[89\]
    fn build_admission_tables(&mut self, cfg: &Config) {
        let n = self.regions.len();
        self.gain_table = vec![[0.0, 0.0]; n * 128];
        self.key_ok = vec![0u64; (n * 128).div_ceil(64)];
        self.delay_frames = self
            .regions
            .iter()
            .map(|r| (r.delay * cfg.sample_rate as f32) as u32)
            .collect();

        for (ri, r) in self.regions.iter().enumerate() {
            // [90]
            let theta = (r.pan.clamp(-1.0, 1.0) + 1.0) * 0.5 * (PI as f32 * 0.5);
            let (pan_l, pan_r) = (theta.cos(), theta.sin());
            for v in 0..128u8 {
                let eff_vel = if r.fixed_vel >= 0 { r.fixed_vel as u8 } else { v };
                let atten = r.attenuation_cb + velocity_atten_cb(eff_vel, r.amp_veltrack);
                let gain = cb_to_gain(atten) * cfg.master_volume * POWER_ON_GAIN * xfade_gain(r, eff_vel);
                self.gain_table[ri * 128 + v as usize] = [gain * pan_l, gain * pan_r];
            }

            // [91]
            let Some(s) = self.samples.get(r.sample as usize) else {
                continue;
            };
            if s.len == 0 {
                continue;
            }
            for k in 0..128u8 {
                let eff_key = if r.fixed_key >= 0 { r.fixed_key as u8 } else { k };
                let root = if r.root_key_override >= 0 {
                    r.root_key_override as f64
                } else {
                    s.root_key as f64
                };
                let cents = (eff_key as f64 - root) * r.scale_tuning as f64
                    + r.coarse_tune as f64 * 100.0
                    + r.fine_tune as f64
                    + s.correction_cents as f64;
                let mut ratio = (cents / 1200.0).exp2();
                if self.pool_rate == 0 {
                    ratio *= s.rate as f64 / cfg.sample_rate as f64;
                } else {
                    ratio *= self.pool_rate as f64 / cfg.sample_rate as f64;
                }
                if ratio.is_finite() && ratio > 0.0 {
                    let bit = ri * 128 + k as usize;
                    self.key_ok[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
    }

    #[inline]
    fn region_key_ok(&self, region: u32, key: u8) -> bool {
        let bit = region as usize * 128 + key as usize;
        self.key_ok[bit / 64] >> (bit % 64) & 1 != 0
    }

    /// The layers `note_on` would produce, and each one's opening gain, without \[92\]
    pub fn preview_note_on(
        &self,
        preset: u32,
        key: u8,
        vel: u8,
        seed: u64,
        max_layers: usize,
        out: &mut Vec<PreviewLayer>,
    ) {
        let Some(p) = self.presets.get(preset as usize) else {
            return;
        };
        let key = key.min(127);
        let lo = p.key_index[key as usize] as usize;
        let hi = p.key_index[key as usize + 1] as usize;
        let mut layers = 0usize;
        // [93]
        let draw = if self.uses_rand { rand_draw(seed) } else { 0.0 };

        for &ri in &p.key_regions[lo..hi] {
            if layers >= max_layers {
                break;
            }
            let r = &self.regions[ri as usize];
            if vel < r.vel_lo || vel > r.vel_hi {
                continue;
            }
            if self.uses_rand && (draw < r.rand_lo || draw >= r.rand_hi) {
                continue;
            }
            if !self.region_key_ok(ri, key) {
                continue;
            }
            let g = self.gain_table[ri as usize * 128 + vel as usize];
            out.push(PreviewLayer {
                region: ri,
                gain_l: g[0],
                gain_r: g[1],
                delay_frames: self.delay_frames[ri as usize],
            });
            layers += 1;
        }
    }

    /// Build one copy of the params table with a channel's sound controllers \[94\]
    pub fn build_variant(&self, cfg: &Config, m: &ParamMod) -> Vec<RegionParams> {
        let sr = cfg.sample_rate as f32;
        let mut params = Vec::with_capacity(self.params.len());
        for r in &self.regions {
            let per_key = r.params_stride != 0;
            for v in 0..r.params_vel_span {
                let vel = (r.vel_lo as u32 + v).min(127) as u8;
                if per_key {
                    for k in 0..128u32 {
                        params.push(make_params(r, k as u8, vel, sr, cfg, m));
                    }
                } else {
                    params.push(make_params(r, 60, vel, sr, cfg, m));
                }
            }
        }
        params
    }

    /// The modulation-envelope table for one controller state, laid out to the \[95\]
    pub fn build_menv_variant(&self, cfg: &Config, m: &ParamMod) -> Vec<ModEnvParams> {
        // [96]
        if self.mod_env_regions() == 0 {
            return vec![ModEnvParams::default()];
        }
        let sr = cfg.sample_rate as f32;
        let mut out = Vec::with_capacity(self.params.len());
        for r in &self.regions {
            let per_key = r.params_stride != 0;
            for v in 0..r.params_vel_span {
                let vel = (r.vel_lo as u32 + v).min(127) as u8;
                if per_key {
                    for k in 0..128u32 {
                        out.push(make_menv(r, k as u8, vel, sr, m));
                    }
                } else {
                    out.push(make_menv(r, 60, vel, sr, m));
                }
            }
        }
        out
    }

    /// Build the pitch-factor table, sized to this bank's own largest \[97\]
    fn build_menv_factors(&mut self) {
        let max_cents = self
            .regions
            .iter()
            .map(|r| r.mod_env_to_pitch.abs())
            .fold(0.0f32, f32::max);
        self.menv_log2 = if self.mod_env_regions() == 0 {
            Vec::new()
        } else {
            build_log2_tab()
        };
        if max_cents <= 0.0 {
            self.menv_factors = vec![1 << 24];
            self.menv_factor_half = 0;
            return;
        }
        let half = (max_cents.ceil() * MOD_ENV_CENTS_STEPS) as u32;
        let n = half as usize * 2 + 1;
        self.menv_factors = (0..n)
            .map(|i| {
                let cents = (i as f64 - half as f64) / MOD_ENV_CENTS_STEPS as f64;
                let f = (cents / 1200.0).exp2() * 16_777_216.0;
                f.round().clamp(1.0, u32::MAX as f64) as u32
            })
            .collect();
        self.menv_factor_half = half;
    }

    /// Regions driving at least one modulation-envelope destination.
    pub fn mod_env_regions(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| r.mod_env_to_pitch != 0.0 || r.mod_env_to_filter != 0.0)
            .count()
    }

    pub fn pool_bytes(&self) -> u64 {
        self.pool.len() as u64 * 2
    }

    /// Regions driving at least one LFO destination.
    pub fn lfo_regions(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| {
                r.mod_lfo_to_pitch != 0.0
                    || r.vib_lfo_to_pitch != 0.0
                    || r.mod_lfo_to_volume != 0.0
            })
            .count()
    }

    pub fn describe(&self) -> String {
        format!(
            "{}: {} presets, {} regions, {} samples, {:.1} MiB pool @ {} Hz",
            self.name,
            self.presets.len(),
            self.regions.len(),
            self.samples.len(),
            self.pool_bytes() as f64 / (1024.0 * 1024.0),
            if self.pool_rate == 0 {
                "mixed".to_string()
            } else {
                self.pool_rate.to_string()
            }
        )
    }
}

fn make_params(r: &Region, key: u8, vel: u8, sr: f32, cfg: &Config, m: &ParamMod) -> RegionParams {
    // SF2 key scaling is expressed in timecents per key, relative to key 60.
    let key_delta = 60.0 - key as f32;
    let hold = r.hold * (2.0f32).powf(r.keynum_to_hold as f32 * key_delta / 1200.0);
    let decay = r.decay * (2.0f32).powf(r.keynum_to_decay as f32 * key_delta / 1200.0);

    // [98]
    let floor = |t: f32, base: f32| t.max(CC_MIN_FADE_SECS.min(base));

    let attack_frames = (m.attack.apply(r.attack) * sr).max(0.0);
    let hold_frames = (hold * sr).max(0.0);
    let decay_frames = (floor(m.decay.apply(decay), decay) * sr).max(1.0);
    let release_frames = (floor(m.release.apply(r.release), r.release) * sr).max(1.0);

    let attack_rate = if attack_frames < 1.0 { 1.0 } else { 1.0 / attack_frames };
    let attack_end = 1.0 + hold_frames * attack_rate;

    let sustain = r.sustain.clamp(0.0, 1.0);
    let floor = cfg.env_floor;

    // [99]
    let (decay_coef, release_coef) = match cfg.decay_curve {
        EnvelopeCurve::Exponential => (
            (10.0f32).powf(-100.0 / (20.0 * decay_frames)),
            match cfg.release_curve {
                EnvelopeCurve::Exponential => (10.0f32).powf(-100.0 / (20.0 * release_frames)),
                EnvelopeCurve::Linear => 1.0 / release_frames,
            },
        ),
        EnvelopeCurve::Linear => (
            1.0 / decay_frames,
            match cfg.release_curve {
                EnvelopeCurve::Exponential => (10.0f32).powf(-100.0 / (20.0 * release_frames)),
                EnvelopeCurve::Linear => 1.0 / release_frames,
            },
        ),
    };

    // [100]
    let fc_cents = r.filter_fc_cents + r.filter_veltrack_cents * (vel as f32 / 127.0)
        + m.cutoff_cents;
    let fc = cents_to_hz(fc_cents);
    let nyq_guard = sr * 0.49;
    // [101]
    let fc_lo_cents = fc_cents.min(fc_cents + r.mod_env_to_filter);
    let fc_lo = cents_to_hz(fc_lo_cents);
    let use_filter =
        cfg.filter_enabled && fc_lo_cents < 13500.0 && fc_lo < nyq_guard && fc_lo > 20.0;

    let (b0, b1, a1, a2) = if use_filter {
        biquad_lowpass(fc, r.filter_q_cb + m.q_cb, sr)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };

    // [102]
    let inc = |hz: f32| -> u32 {
        let turns = (hz.max(0.0) / sr) as f64;
        (turns * 4_294_967_296.0) as u32
    };
    let delay = |secs: f32| -> u32 { (secs.max(0.0) * sr) as u32 };

    let pack16 = |lo: i32, hi: i32| -> u32 {
        ((lo as u32) & 0xFFFF) | ((hi as u32) << 16)
    };
    RegionParams {
        mod_lfo_inc: inc(r.mod_lfo_hz),
        vib_lfo_inc: inc(r.vib_lfo_hz),
        lfo_delays: pack16(
            delay(r.mod_lfo_delay).min(0xFFFF) as i32,
            delay(r.vib_lfo_delay).min(0xFFFF) as i32,
        ),
        lfo_pitch: pack16(
            r.mod_lfo_to_pitch.clamp(-32768.0, 32767.0) as i32,
            r.vib_lfo_to_pitch.clamp(-32768.0, 32767.0) as i32,
        ),
        attack_rate,
        attack_end,
        decay_coef,
        decay_target: sustain.max(floor),
        sustain,
        release_coef,
        b0,
        b1,
        a1,
        a2,
        flags: if use_filter { RP_FILTER } else { 0 }
            | if r.mod_env_to_pitch != 0.0 || r.mod_env_to_filter != 0.0 {
                RP_MOD_ENV
            } else {
                0
            }
            | if release_frames <= cfg.env_step_frames() as f32 {
                RP_SHORT_RELEASE
            } else {
                0
            }
            | (((r.mod_lfo_to_volume.clamp(-32768.0, 32767.0) as i32) as u32) << 16),
    }
}

/// The modulation-envelope entry matching one `make_params` entry. \[103\]
fn make_menv(r: &Region, _key: u8, vel: u8, sr: f32, m: &ParamMod) -> ModEnvParams {
    // [104]
    let rate = |secs: f32| -> f32 {
        let frames = secs.max(0.0) * sr;
        if frames < 1.0 {
            1.0
        } else {
            1.0 / frames
        }
    };

    // [105]
    let fc_cents = r.filter_fc_cents + r.filter_veltrack_cents * (vel as f32 / 127.0)
        + m.cutoff_cents;

    // [106]
    const Q_DEFAULT: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let q_db = (r.filter_q_cb + m.q_cb) / 10.0 - 3.01;
    let q = (10.0f32).powf(q_db / 20.0).max(0.001);

    ModEnvParams {
        delay_frames: (r.mod_env_delay.max(0.0) * sr) as u32,
        // [107]
        attack_frames: {
            let f = r.mod_env_attack.max(0.0) * sr;
            if f < 1.0 {
                0
            } else {
                f as u32
            }
        },
        hold_frames: (r.mod_env_hold.max(0.0) * sr) as u32,
        decay_rate: rate(r.mod_env_decay),
        sustain: r.mod_env_sustain.clamp(0.0, 1.0),
        release_rate: rate(r.mod_env_release),
        to_pitch: r.mod_env_to_pitch,
        to_filter: r.mod_env_to_filter,
        fc_cents,
        q_gain: (Q_DEFAULT / q).sqrt(),
        q_inv_2q: 1.0 / (2.0 * q),
        _pad: [0.0; 5],
    }
}

/// RBJ low-pass, normalised so resonance does not raise the passband level. \[108\]
pub fn biquad_lowpass(fc: f32, q_cb: f32, sr: f32) -> (f32, f32, f32, f32) {
    const Q_DEFAULT: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let q_db = q_cb / 10.0 - 3.01;
    let q = (10.0f32).powf(q_db / 20.0).max(0.001);
    let gain = (Q_DEFAULT / q).sqrt();

    let w0 = 2.0 * std::f32::consts::PI * (fc / sr).clamp(1.0e-5, 0.49);
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = gain * (1.0 - cos_w0) * 0.5 / a0;
    let b1 = gain * (1.0 - cos_w0) / a0;
    let a1 = -2.0 * cos_w0 / a0;
    let a2 = (1.0 - alpha) / a0;
    (b0, b1, a1, a2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_curve_is_sane() {
        assert!(velocity_atten_cb(127, 100.0) < 0.01);
        // Half velocity should be around -12 dB, not -48 and not -3.
        let cb = velocity_atten_cb(64, 100.0);
        assert!((100.0..140.0).contains(&cb), "vel 64 gave {cb} cB");
        assert!(velocity_atten_cb(1, 100.0) >= 800.0);
    }

    #[test]
    fn amp_veltrack_scales_the_velocity_curve() {
        // 0 means velocity does not touch the amplitude at all.
        for vel in [1u8, 40, 64, 100, 127] {
            assert_eq!(velocity_atten_cb(vel, 0.0), 0.0, "vel {vel} at veltrack 0");
        }
        // Half tracking sits between no tracking and full, at every velocity.
        for vel in [1u8, 40, 64, 100] {
            let half = velocity_atten_cb(vel, 50.0);
            assert!(
                half > 0.0 && half < velocity_atten_cb(vel, 100.0),
                "vel {vel} gave {half} cB at veltrack 50"
            );
        }
        // Full velocity is unattenuated whatever the tracking.
        assert!(velocity_atten_cb(127, 50.0) < 0.01);
        // [109]
        assert!(velocity_atten_cb(127, -100.0) >= 960.0);
        assert!(velocity_atten_cb(1, -100.0) < 0.01);
    }

    #[test]
    fn lowpass_is_unity_at_dc() {
        let (b0, b1, a1, a2) = biquad_lowpass(1000.0, 0.0, 48000.0);
        // H(1) = (b0 + b1 + b2) / (1 + a1 + a2), with b2 == b0.
        let h = (2.0 * b0 + b1) / (1.0 + a1 + a2);
        assert!((h - 1.0).abs() < 0.02, "dc gain was {h}");
        // [110]
        let (b0, b1, a1, a2) = biquad_lowpass(1000.0, 120.0, 48000.0);
        let h_res = (2.0 * b0 + b1) / (1.0 + a1 + a2);
        assert!(h_res < h, "resonant filter raised dc gain to {h_res}");
    }

    #[test]
    fn timecents_round_trip() {
        assert_eq!(timecents_to_secs(-12000.0), 0.0);
        assert!((timecents_to_secs(0.0) - 1.0).abs() < 1e-6);
        assert!((timecents_to_secs(1200.0) - 2.0).abs() < 1e-5);
    }
}
