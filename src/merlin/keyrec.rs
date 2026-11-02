// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What merlin's key parser (`keyparse`) needs to turn a paired note into the \[1\]

use crate::bank::{Bank, ModEnvParams, PreviewLayer, PreviewMemo, RegionParams};
use crate::config::Config;
use crate::midi::{Event, CHANNELS};
use crate::voice::SpawnCmd;

/// What a file asked of the channel state that is not built yet, a bit each.
pub mod unsupported {
    /// Portamento on, or a CC84 source key.
    pub const PORTA: u32 = 4;
    /// Modulation, vibrato or tremolo away from rest.
    pub const MOD: u32 = 8;
    /// MIDI Tuning Standard messages.
    pub const TUNE: u32 = 16;
    /// `Config::phase` analytic, or a bank with random regions.
    pub const CONFIG: u32 = 32;

    /// The names of the bits set in `flags`.
    pub fn names(flags: u32) -> Vec<&'static str> {
        [(PORTA, "portamento"), (MOD, "modulation"), (TUNE, "tuning"), (CONFIG, "phase or random regions")]
            .iter()
            .filter(|(b, _)| flags & b != 0)
            .map(|&(_, n)| n)
            .collect()
    }
}

/// `Driver`'s RPN selection when an NRPN is selected or the selection is \[2\]
const RPN_NONE: u16 = u16::MAX;

/// Every channel's state, as the driver keeps it, for what a record reads.
#[derive(Clone)]
pub struct Channels {
    program: [u8; CHANNELS],
    bank_msb: [u8; CHANNELS],
    drum_map: [u8; CHANNELS],
    preset: [u32; CHANNELS],
    cc_volume: [u8; CHANNELS],
    lsb_volume: [u8; CHANNELS],
    cc_expression: [u8; CHANNELS],
    lsb_expression: [u8; CHANNELS],
    cc_pan: [u8; CHANNELS],
    lsb_pan: [u8; CHANNELS],
    cc_soft: [u8; CHANNELS],
    gain: [[f32; 2]; CHANNELS],
    bend_val: [i16; CHANNELS],
    bend_range: [f64; CHANNELS],
    coarse_tune: [f64; CHANNELS],
    fine_tune: [f64; CHANNELS],
    fine_tune_raw: [u16; CHANNELS],
    rpn_sel: [u16; CHANNELS],
    /// The pitch as `Driver::refresh_bend` publishes it: `fixed::bend_factor` \[3\]
    bend_raw: [u32; CHANNELS],
    cc_sound: [[u8; 5]; CHANNELS],
    porta_on: [bool; CHANNELS],
    porta_note: [u8; CHANNELS],
    cc_mod: [u8; CHANNELS],
    cc_vib_depth: [u8; CHANNELS],
    cc_tremolo: [u8; CHANNELS],
    /// What a note asked for that is not built yet, [`unsupported`] bits.
    pub flags: u32,
}

/// `Driver`'s default drum parts: channel 10 of every port.
fn default_drum_map() -> [u8; CHANNELS] {
    let mut m = [0u8; CHANNELS];
    let mut c = 9;
    while c < CHANNELS {
        m[c] = 1;
        c += 16;
    }
    m
}

impl Channels {
    pub fn new(bank: &Bank, cfg: &Config) -> Self {
        let mut c = Self {
            program: [0; CHANNELS],
            bank_msb: [0; CHANNELS],
            drum_map: default_drum_map(),
            preset: [0; CHANNELS],
            cc_volume: [crate::bank::POWER_ON_VOLUME; CHANNELS],
            lsb_volume: [0; CHANNELS],
            cc_expression: [127; CHANNELS],
            lsb_expression: [0; CHANNELS],
            cc_pan: [64; CHANNELS],
            lsb_pan: [0; CHANNELS],
            cc_soft: [0; CHANNELS],
            gain: [[1.0; 2]; CHANNELS],
            bend_val: [0; CHANNELS],
            bend_range: [2.0; CHANNELS],
            coarse_tune: [0.0; CHANNELS],
            fine_tune: [0.0; CHANNELS],
            fine_tune_raw: [8192; CHANNELS],
            rpn_sel: [0; CHANNELS],
            bend_raw: [crate::fixed::BEND_ONE; CHANNELS],
            cc_sound: [[64; 5]; CHANNELS],
            porta_on: [false; CHANNELS],
            porta_note: [0; CHANNELS],
            cc_mod: [0; CHANNELS],
            cc_vib_depth: [64; CHANNELS],
            cc_tremolo: [0; CHANNELS],
            flags: 0,
        };
        if cfg.phase.active() || bank.uses_rand {
            c.flags |= unsupported::CONFIG;
        }
        for ch in 0..CHANNELS {
            c.refresh_preset(bank, ch, 0);
            c.refresh_gain(ch);
        }
        c
    }

    fn refresh_preset(&mut self, bank: &Bank, ch: usize, program: u8) {
        self.program[ch] = program;
        let bank_num = if self.drum_map[ch] != 0 { 128u16 } else { self.bank_msb[ch] as u16 };
        self.preset[ch] = bank.find_preset(bank_num, program as u16).unwrap_or(0);
    }

    /// `Driver::refresh_gain`, to the bit.
    fn refresh_gain(&mut self, c: usize) {
        let fine = |msb: u8, lsb: u8| {
            if lsb == 0 {
                msb as f32 / 127.0
            } else {
                (msb as f32 * 128.0 + lsb as f32) / 16383.0
            }
        };
        let v = fine(self.cc_volume[c], self.lsb_volume[c]) / (crate::bank::POWER_ON_VOLUME as f32 / 127.0);
        let e = fine(self.cc_expression[c], self.lsb_expression[c]);
        let soft = 1.0 - 0.5 * (self.cc_soft[c] as f32 / 127.0);
        let amp = (v * v) * (e * e) * soft;
        let theta = fine(self.cc_pan[c], self.lsb_pan[c]) * std::f32::consts::FRAC_PI_2;
        let (l, r) = if self.cc_pan[c] == 64 && self.lsb_pan[c] == 0 {
            (1.0, 1.0)
        } else {
            (theta.cos() * std::f32::consts::SQRT_2, theta.sin() * std::f32::consts::SQRT_2)
        };
        self.gain[c] = [amp * l, amp * r];
    }

    /// `Driver::refresh_bend`, to the bit.
    fn refresh_bend(&mut self, c: usize) {
        let semitones = (self.bend_val[c] as f64 / 8192.0) * self.bend_range[c] + self.coarse_tune[c] + self.fine_tune[c] / 100.0;
        self.bend_raw[c] = crate::fixed::bend_factor(semitones);
    }

    /// CC121's share, as `Driver::reset_controllers` has it.
    pub fn reset(&mut self, ch: u8) {
        let c = ch as usize;
        self.porta_on[c] = false;
        self.bend_val[c] = 0;
        self.rpn_sel[c] = RPN_NONE;
        self.cc_expression[c] = 127;
        self.lsb_expression[c] = 0;
        self.cc_mod[c] = 0;
        self.cc_soft[c] = 0;
        self.refresh_bend(c);
        self.refresh_gain(c);
    }

    /// A channel event the pairing does not take, as the driver handles it.
    pub fn apply(&mut self, bank: &Bank, ev: &Event) {
        match *ev {
            Event::Cc { ch, num, val } => {
                let c = ch as usize;
                match num {
                    0 => self.bank_msb[c] = val,
                    1 => self.cc_mod[c] = val,
                    5 => {}
                    6 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = val as f64;
                        self.refresh_bend(c);
                    }
                    38 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = self.bend_range[c].trunc() + val as f64 / 100.0;
                        self.refresh_bend(c);
                    }
                    6 if self.rpn_sel[c] == 1 => {
                        self.fine_tune_raw[c] = (self.fine_tune_raw[c] & 0x7F) | ((val as u16) << 7);
                        self.fine_tune[c] = (self.fine_tune_raw[c] as f64 - 8192.0) / 8192.0 * 100.0;
                        self.refresh_bend(c);
                    }
                    38 if self.rpn_sel[c] == 1 => {
                        self.fine_tune_raw[c] = (self.fine_tune_raw[c] & 0x3F80) | val as u16;
                        self.fine_tune[c] = (self.fine_tune_raw[c] as f64 - 8192.0) / 8192.0 * 100.0;
                        self.refresh_bend(c);
                    }
                    6 if self.rpn_sel[c] == 2 => {
                        self.coarse_tune[c] = val as f64 - 64.0;
                        self.refresh_bend(c);
                    }
                    96 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = (self.bend_range[c] + 1.0).min(127.0);
                        self.refresh_bend(c);
                    }
                    97 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = (self.bend_range[c] - 1.0).max(0.0);
                        self.refresh_bend(c);
                    }
                    98 | 99 => self.rpn_sel[c] = RPN_NONE,
                    100 => {
                        let keep = if self.rpn_sel[c] == RPN_NONE { 0 } else { self.rpn_sel[c] };
                        self.rpn_sel[c] = (keep & 0x3F80) | val as u16;
                    }
                    101 => {
                        let keep = if self.rpn_sel[c] == RPN_NONE { 0 } else { self.rpn_sel[c] };
                        self.rpn_sel[c] = (keep & 0x7F) | ((val as u16) << 7);
                    }
                    7 => {
                        self.cc_volume[c] = val;
                        self.lsb_volume[c] = 0;
                        self.refresh_gain(c);
                    }
                    10 => {
                        self.cc_pan[c] = val;
                        self.lsb_pan[c] = 0;
                        self.refresh_gain(c);
                    }
                    11 => {
                        self.cc_expression[c] = val;
                        self.lsb_expression[c] = 0;
                        self.refresh_gain(c);
                    }
                    39 => {
                        self.lsb_volume[c] = val;
                        self.refresh_gain(c);
                    }
                    42 => {
                        self.lsb_pan[c] = val;
                        self.refresh_gain(c);
                    }
                    43 => {
                        self.lsb_expression[c] = val;
                        self.refresh_gain(c);
                    }
                    65 => self.porta_on[c] = val >= 64,
                    84 => self.porta_note[c] = val,
                    67 => {
                        self.cc_soft[c] = val;
                        self.refresh_gain(c);
                    }
                    71..=75 => self.cc_sound[c][(num - 71) as usize] = val,
                    77 => self.cc_vib_depth[c] = val,
                    92 => self.cc_tremolo[c] = val,
                    _ => {}
                }
            }
            Event::Program { ch, val } => self.refresh_preset(bank, ch as usize, val),
            Event::DrumPart { ch, map } => {
                let c = ch as usize;
                if self.drum_map[c] != map {
                    self.drum_map[c] = map;
                    self.refresh_preset(bank, c, self.program[c]);
                }
            }
            Event::ResetParts => {
                for (c, want) in default_drum_map().into_iter().enumerate() {
                    if self.drum_map[c] != want {
                        self.drum_map[c] = want;
                        self.refresh_preset(bank, c, self.program[c]);
                    }
                }
            }
            Event::PitchBend { ch, val } => {
                self.bend_val[ch as usize] = val;
                self.refresh_bend(ch as usize);
            }
            Event::Tune { .. } | Event::KeyTune { .. } => self.flags |= unsupported::TUNE,
            _ => {}
        }
    }

    /// The channel as a note struck now takes it, and whatever it asks for \[4\]
    pub fn strike(&mut self, ch: u8) -> Strike {
        let c = ch as usize;
        if self.porta_on[c] || self.porta_note[c] != 0 {
            self.flags |= unsupported::PORTA;
        }
        if self.cc_mod[c] != 0 || self.cc_tremolo[c] != 0 {
            self.flags |= unsupported::MOD;
        }
        // A CC84 is spent by the note it names a source for.
        self.porta_note[c] = 0;
        Strike { preset: self.preset[c], gain: self.gain[c], bend: self.bend_raw[c], sound: self.cc_sound[c] }
    }

    /// `fixed::bend_factor` of the channel's pitch now, `BEND_ONE` at rest.
    pub fn bend(&self, ch: u8) -> u32 {
        self.bend_raw[ch as usize]
    }

    /// The channel's gain now, which a release tail is fired at.
    pub fn gain(&self, ch: u8) -> [f32; 2] {
        self.gain[ch as usize]
    }
}

/// What a note takes from its channel at the note-on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strike {
    pub preset: u32,
    pub gain: [f32; 2],
    /// `fixed::bend_factor` of the channel's pitch, `BEND_ONE` at rest.
    pub bend: u32,
    /// CC71-75 as they stand, raw.
    pub sound: [u8; 5],
}

/// CC71-75 at rest: the bank's own params.
const SOUND_REST: [u8; 5] = [64; 5];

/// One layer of a note, as key-major files it: the command with the \[5\]
#[derive(Clone, Copy)]
pub struct Layer {
    pub cmd: SpawnCmd,
    pub params: RegionParams,
    pub menv: Option<ModEnvParams>,
    pub start: u64,
    pub duration: u32,
    pub gain_end: [f32; 2],
    /// The layer's own gains, before the channel's at either end.
    pub base: [f32; 2],
    /// The right half of a stereo sample, `cmd.smp_base` for a mono layer.
    pub smp_base_r: u32,
}

/// A gain this far under the other side is the rounding of a hard pan: a \[6\]
const HARD_PAN: f32 = 1e-6;

/// The two halves of a stereo sample as one layer, wherever a note has them. \[7\]
pub fn merge_stereo(out: &mut Vec<Layer>) {
    let hard = |g: [f32; 2]| g[1].abs() <= g[0].abs() * HARD_PAN && g[0] != 0.0;
    let same = |a: &Layer, b: &Layer| {
        (a.cmd.phase_lo, a.cmd.phase_hi, a.cmd.step_lo, a.cmd.step_hi, a.cmd.smp_len, a.cmd.loop_start, a.cmd.loop_end, a.cmd.flags, a.cmd.variant)
            == (b.cmd.phase_lo, b.cmd.phase_hi, b.cmd.step_lo, b.cmd.step_hi, b.cmd.smp_len, b.cmd.loop_start, b.cmd.loop_end, b.cmd.flags, b.cmd.variant)
            && (a.start, a.duration) == (b.start, b.duration)
            && a.cmd.smp_base != b.cmd.smp_base
            && bytemuck::bytes_of(&a.params) == bytemuck::bytes_of(&b.params)
            && a.menv.as_ref().map(bytemuck::bytes_of) == b.menv.as_ref().map(bytemuck::bytes_of)
    };
    let mut i = 0;
    while i < out.len() {
        let found = (i + 1..out.len()).find(|&j| {
            let (a, b) = (&out[i], &out[j]);
            let sides = (hard(a.base) && hard([b.base[1], b.base[0]])) || (hard(b.base) && hard([a.base[1], a.base[0]]));
            sides && a.smp_base_r == a.cmd.smp_base && b.smp_base_r == b.cmd.smp_base && same(a, b)
        });
        if let Some(j) = found {
            let other = out.remove(j);
            let this = out[i];
            let (l, r) = if hard(this.base) { (this, other) } else { (other, this) };
            let mut cmd = l.cmd;
            cmd.gain_r = r.cmd.gain_r;
            out[i] = Layer { cmd, gain_end: [l.gain_end[0], r.gain_end[1]], base: [l.base[0], r.base[1]], smp_base_r: r.cmd.smp_base, ..l };
        }
        i += 1;
    }
}

/// The params tables for one set of raw CC71-75 values.
type Variant = (Vec<RegionParams>, Vec<ModEnvParams>);

/// A note's layers, built by the driver's own code.
pub struct Layers<'a> {
    bank: &'a Bank,
    cfg: &'a Config,
    memo: PreviewMemo,
    prev: Vec<PreviewLayer>,
    /// Params tables by the raw CC71-75 values they were built from.
    variants: rustc_hash::FxHashMap<[u8; 5], Variant>,
}

/// Merlin's bend grid, `BucketPass::quantise_bend` without its memo.
fn quantise_bend(cfg: &Config, raw: u32) -> u32 {
    let w = cfg.merlin_bend_cents;
    if w <= 0.0 || raw == crate::fixed::BEND_ONE {
        return raw;
    }
    let cents = 1200.0 * (raw as f64 / crate::fixed::BEND_ONE as f64).log2();
    crate::fixed::bend_factor((cents / w).round() * w / 100.0)
}

impl<'a> Layers<'a> {
    pub fn new(bank: &'a Bank, cfg: &'a Config) -> Self {
        Self { bank, cfg, memo: PreviewMemo::new(bank, cfg.max_layers as usize), prev: Vec::new(), variants: Default::default() }
    }

    /// The params row `base` reads under `sound`, and its modulation envelope \[8\]
    fn params_of(&mut self, sound: [u8; 5], base: usize) -> Option<(RegionParams, Option<ModEnvParams>)> {
        let (bank, cfg) = (self.bank, self.cfg);
        let (p, m) = if sound == SOUND_REST {
            (*bank.params.get(base)?, bank.menv.get(base).copied())
        } else {
            let v = self.variants.entry(sound).or_insert_with(|| {
                let m = crate::bank::ParamMod::from_controllers(sound[0], sound[1], sound[2], sound[3], sound[4]);
                (bank.build_variant(cfg, &m), bank.build_menv_variant(cfg, &m))
            });
            (*v.0.get(base)?, v.1.get(base).copied().or_else(|| bank.menv.get(base).copied()))
        };
        let menv = if cfg.mod_env_enabled && p.flags & crate::bank::RP_MOD_ENV != 0 { m } else { None };
        Some((p, menv))
    }

    /// The layers of `n`, into `out`.
    pub fn build(&mut self, n: &crate::merlin::keyparse::Note, out: &mut Vec<Layer>) {
        out.clear();
        let (ch, key) = ((n.slot / 128) as u8, (n.slot % 128) as u8);
        self.prev.clear();
        self.memo.preview(self.bank, n.preset, key, n.vel, 0, &mut self.prev);
        let prev = std::mem::take(&mut self.prev);
        for p in &prev {
            let Some(v) = self.bank.build_layer(p.region, key, n.vel, self.cfg, 0.0) else { continue };
            let start = n.start + v.delay_frames as u64;
            let end = n.end.max(start);
            let mut cmd = crate::driver::Driver::make_cmd(&v, 0, crate::voice::GateTable::slot(ch, key) as u32, 0, 0, 0, 0);
            if n.bend != crate::fixed::BEND_ONE {
                let step = crate::fixed::Fixed::from_parts(cmd.step_hi, cmd.step_lo).scale(quantise_bend(self.cfg, n.bend));
                cmd.step_lo = step.lo();
                cmd.step_hi = step.hi();
            }
            let base = [cmd.gain_l, cmd.gain_r];
            cmd.gain_l *= n.gain_on[0];
            cmd.gain_r *= n.gain_on[1];
            let Some((params, menv)) = self.params_of(n.sound, cmd.params as usize) else { continue };
            out.push(Layer {
                cmd,
                params,
                menv,
                start,
                duration: (end - start) as u32,
                gain_end: [base[0] * n.gain_off[0], base[1] * n.gain_off[1]],
                base,
                smp_base_r: cmd.smp_base,
            });
        }
        self.prev = prev;
    }
}
