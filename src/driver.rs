// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Block-at-a-time render driver. \[1\]

use crate::backend::Backend;
use crate::bank::{Bank, ParamMod, PreviewLayer, VoiceSpawn};
use crate::config::{AdmitRule, Config};
use crate::limiter::{clamp_block, Brickwall, Limiter, LimiterMode};
use crate::midi::{Event, MidiStream, TempoClock};
use crate::fixed::bend_factor;
use crate::voice::{ChannelTable, GateTable, SpawnCmd};
use anyhow::{bail, Result};

/// Time strata a saturated block is cut into before ranking admission. \[2\]
const ADMIT_STRATA: usize = 64;
use std::path::Path;
use std::sync::Arc;

/// A selected RPN that is not one this synth acts on, including the null RPN \[3\]
pub const RPN_NONE: u16 = 0x4000;

/// How far CC71-CC75 are quantised before they become a params variant. \[4\]
pub const SOUND_CC_SHIFT: u32 = 1;

/// What this synth does with a controller number. \[5\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcRole {
    /// Acted on.
    Applied,
    /// Recognised, and correctly does nothing to an offline render.
    Inert(&'static str),
    /// Would change what is heard, and is not implemented. The string says \[6\]
    Missing(&'static str),
}

/// Name and role for one controller number.
pub fn cc_role(num: u8) -> (&'static str, CcRole) {
    use CcRole::*;
    const NO_DEFAULT: CcRole =
        Inert("no default mapping in GM; a soundfont would have to bind it");
    const UNDEFINED: CcRole = Inert("undefined by the MIDI spec");
    const GENERAL: CcRole = Inert("general purpose, no defined destination");
    const LSB_UNUSED: CcRole = Inert("fine resolution for a controller that is not implemented");
    match num {
        0 => ("bank select msb", Applied),
        1 => ("modulation", Applied),
        2 => ("breath", NO_DEFAULT),
        3 => ("", UNDEFINED),
        4 => ("foot controller", NO_DEFAULT),
        5 => ("portamento time", Missing("portamento needs per-voice glide state")),
        6 => ("data entry msb", Applied),
        7 => ("channel volume", Applied),
        8 => ("balance", Inert("for two-source channels; a sampler has one")),
        9 => ("", UNDEFINED),
        10 => ("pan", Applied),
        11 => ("expression", Applied),
        12 | 13 => ("effect control", Missing("effects are not implemented")),
        14 | 15 => ("", UNDEFINED),
        16..=19 => ("general purpose", GENERAL),
        20..=31 => ("", UNDEFINED),
        32 => ("bank select lsb", Applied),
        33 => ("modulation lsb", Applied),
        38 => ("data entry lsb", Applied),
        39 => ("channel volume lsb", Applied),
        42 => ("pan lsb", Applied),
        43 => ("expression lsb", Applied),
        34..=63 => ("", LSB_UNUSED),
        64 => ("sustain pedal", Applied),
        65 => ("portamento", Missing("portamento needs per-voice glide state")),
        66 => ("sostenuto", Applied),
        67 => ("soft pedal", Applied),
        68 => ("legato footswitch", Missing("legato needs mono voice allocation")),
        69 => ("hold 2", Missing("a second hold that lengthens release rather than damping")),
        70 => ("sound variation", NO_DEFAULT),
        71 => ("resonance", Applied),
        72 => ("release time", Applied),
        73 => ("attack time", Applied),
        74 => ("brightness", Applied),
        75 => ("decay time", Applied),
        76 => ("vibrato rate", Applied),
        77 => ("vibrato depth", Applied),
        78 => ("vibrato delay", Missing("the LFO starts immediately")),
        79 => ("sound controller 10", NO_DEFAULT),
        80..=83 => ("general purpose", GENERAL),
        84 => ("portamento control", Missing("portamento needs per-voice glide state")),
        85..=87 => ("", UNDEFINED),
        88 => ("high resolution velocity", Missing("velocity is read as 7 bits")),
        89 | 90 => ("", UNDEFINED),
        91 => ("reverb send", Missing("effects are not implemented")),
        92 => ("tremolo depth", Applied),
        93 => ("chorus send", Missing("effects are not implemented")),
        94 => ("celeste depth", Missing("effects are not implemented")),
        95 => ("phaser depth", Missing("effects are not implemented")),
        96 => ("data increment", Applied),
        97 => ("data decrement", Applied),
        98 | 99 => ("nrpn select", Applied),
        100 | 101 => ("rpn select", Applied),
        102..=119 => ("", UNDEFINED),
        120 => ("all sound off", Applied),
        121 => ("reset controllers", Applied),
        122 => ("local control", Inert("there is no keyboard attached to an offline render")),
        123 => ("all notes off", Applied),
        124 => ("omni off", Applied),
        125 => ("omni on", Applied),
        126 => ("mono mode on", Applied),
        127 => ("poly mode on", Applied),
        _ => ("", UNDEFINED),
    }
}

/// Whether a controller changes anything about the render.
pub fn handles_cc(num: u8) -> bool {
    cc_role(num).1 == CcRole::Applied
}

/// How long to keep rendering after the last MIDI event before giving up on \[7\]
const MAX_TAIL_SECONDS: f64 = 60.0;

#[derive(Debug, Clone, Copy, Default)]
pub struct DriverStats {
    /// Cumulative microseconds inside `next_block`, split so the host/device \[8\]
    pub us_total: u64,
    /// The event drain loop: pulling events and running `handle_event`.
    pub us_drain: u64,
    /// Admission, ranking and `materialise`.
    pub us_admit: u64,
    /// `backend.spawn`, which uploads the block's new voices.
    pub us_spawn: u64,
    /// `backend.render`, which is the dispatch plus the blocking readback.
    pub us_render: u64,
    /// Copies of the params table CC71-CC75 asked for and got.
    pub param_variants: u32,
    /// Times a sound-controller change had to reuse an approximate copy \[9\]
    pub variant_fallbacks: u64,
    /// Distinct quantised states the file asked for, whether or not they \[10\]
    pub variant_states: u32,
    /// Times a slot was reused for a different state. Correct, just work.
    pub variant_rebuilds: u64,
    pub frames: u64,
    pub blocks: u64,
    pub events: u64,
    /// Note-ons the block could not admit. Counted here rather than in the \[11\]
    pub dropped: u64,
    pub notes: u64,
    pub voices_spawned: u64,
    pub peak: f32,
    /// Samples the final clamp had to pull back to full scale. Every run of \[12\]
    pub clipped: u64,

    // [13]
    pub last_live: u32,
    /// Note-on layers this block queued.
    pub last_want: u64,
    /// Layers admission let through.
    pub last_take: u32,
    /// Voices the backend killed to make room for them.
    pub last_stolen: u64,
    /// Summed opening amplitude of every layer the block queued, and of the \[14\]
    pub last_want_energy: u64,
    pub last_take_energy: u64,
}

/// Pack a note-on into one word: everything `build_layer` and `SpawnCmd` will \[15\]
#[inline]
fn note_pack(ch: u8, key: u8, vel: u8, variant: u32, rel: u32, row_bias: u32) -> u64 {
    debug_assert!(variant < 64 && rel < 65536 && row_bias < 2);
    (ch as u64 & 0xF)
        | ((key as u64 & 0x7F) << 4)
        | ((vel as u64 & 0x7F) << 11)
        | ((variant as u64 & 0x3F) << 18)
        | ((rel as u64 & 0xFFFF) << 24)
        | ((row_bias as u64 & 1) << 40)
}

#[inline]
fn note_unpack(w: u64) -> (u8, u8, u8, u32, u32, u32) {
    (
        (w & 0xF) as u8,
        ((w >> 4) & 0x7F) as u8,
        ((w >> 11) & 0x7F) as u8,
        ((w >> 18) & 0x3F) as u32,
        ((w >> 24) & 0xFFFF) as u32,
        ((w >> 40) & 1) as u32,
    )
}

/// One layer a block might admit, before anything has been built for it. \[16\]
#[derive(Clone, Copy)]
struct Cand {
    key: u64,
    /// Index into `notes`, or `DEFERRED` for a voice built in an earlier block \[17\]
    note: u32,
    /// The region to build, so materialising never has to preview again. For a \[18\]
    region: u32,
    /// Note id, as an offset from `block_first_id`. Unused when `DEFERRED`, \[19\]
    id_off: u64,
}

/// `Cand::note` for a candidate that came off the deferred queue.
const DEFERRED: u32 = u32::MAX;

/// Which channels are drum parts before a file says anything. Channel 10 -- \[20\]
const DEFAULT_DRUM_MAP: [u8; 16] = {
    let mut m = [0u8; 16];
    m[9] = 1;
    m
};

/// `admit_key`, reached without building a `SpawnCmd` first.
#[inline]
fn rank_key(gain_l: f32, gain_r: f32, index: u64) -> u64 {
    (crate::voice::rank_gain_q(gain_l, gain_r) << 48) | crate::voice::mix48(index)
}

pub struct Driver {
    cfg: Config,
    bank: Arc<Bank>,
    stream: MidiStream,
    clock: TempoClock,
    gates: GateTable,
    chan: ChannelTable,
    limiter: Limiter,
    brickwall: Brickwall,

    bank_msb: [u8; 16],
    bank_lsb: [u8; 16],
    /// Last program change per channel, kept because the drum-part SysEx can \[21\]
    program: [u8; 16],
    /// GS part type: 0 melodic, 1 or 2 for the two drum maps. Channel 10 is \[22\]
    drum_map: [u8; 16],
    preset: [u32; 16],
    /// Raw pitch bend, -8192..8191 relative to centre.
    bend_val: [i16; 16],
    /// Bend range in semitones, RPN 0. Two is the GM default.
    bend_range: [f64; 16],
    /// Channel coarse tuning, RPN 2, in semitones. The wire value is an MSB \[23\]
    coarse_tune: [f64; 16],
    /// Channel fine tuning, RPN 1, in cents. A 14-bit value centred on 8192 \[24\]
    fine_tune: [f64; 16],
    /// The raw 14-bit value behind it, kept so an MSB and an LSB written at \[25\]
    fine_tune_raw: [u16; 16],
    /// Currently selected RPN, from CC101 (MSB) and CC100 (LSB).
    rpn_sel: [u16; 16],
    /// CC7 channel volume, CC11 expression, CC10 pan.
    cc_volume: [u8; 16],
    cc_expression: [u8; 16],
    cc_pan: [u8; 16],
    /// CC71 resonance, CC72 release, CC73 attack, CC74 brightness, CC75 decay. \[26\]
    cc_sound: [[u8; 5]; 16],
    /// CC1 modulation depth, CC76 vibrato rate, CC77 vibrato depth, CC92 \[27\]
    cc_mod: [u8; 16],
    cc_vib_rate: [u8; 16],
    cc_vib_depth: [u8; 16],
    cc_tremolo: [u8; 16],
    cc_soft: [u8; 16],
    /// Fine halves for the four controllers that have one worth reading. A \[28\]
    lsb_mod: [u8; 16],
    lsb_volume: [u8; 16],
    lsb_pan: [u8; 16],
    lsb_expression: [u8; 16],
    /// LFO phase per channel, in cycles, carried across blocks so vibrato does \[29\]
    lfo_phase: [f64; 16],
    /// Quantised sound-controller states, indexed by variant number.
    variants: Vec<[u8; 5]>,
    /// Every quantised state the file has asked for, for diagnostics.
    seen_states: Vec<[u8; 5]>,
    /// Which variant each channel is currently on.
    cur_variant: [u32; 16],
    /// Bitmask of variants any channel has pointed at during this block. A \[30\]
    variant_used: u64,
    /// Monotonic tick per variant, for choosing the stalest one to reuse.
    variant_seen: Vec<u64>,
    variant_clock: u64,
    /// Variants asked for this block but not yet built and uploaded.
    pending_variants: Vec<(u32, ParamMod)>,
    /// Set when a voice queued this block was born under a non-zero variant. \[31\]
    spawn_variants: bool,

    next_note_id: u64,
    block_index: u64,
    /// Event pulled from the stream that belongs to a later block.
    pending: Option<(u64, Event)>,
    stream_done: bool,
    end_sent: bool,
    /// Whether block 0's tables have been built. Every later block is prepared \[32\]
    primed: bool,
    tail_blocks_left: u64,

    spawn_buf: Vec<SpawnCmd>,
    /// This block's note-ons, one packed entry each, plus their ordinals. \[33\]
    notes: Vec<u64>,
    note_ordinal: Vec<u32>,
    /// One entry per admissible layer this block, up to `cand_cap`. See `Cand` \[34\]
    cands: Vec<Cand>,
    /// Candidates offered this block, counting the ones `cand_stride` skipped: \[35\]
    cand_seen: u64,
    /// Offer only candidates whose index is a multiple of this. One until the \[36\]
    cand_stride: u64,
    /// `Config::max_block_candidates`, never below twice the pool, so thinning \[37\]
    cand_cap: usize,
    /// This block's share of the deferred queue, already built. Kept apart from \[38\]
    deferred_now: Vec<SpawnCmd>,
    /// Scratch for `Bank::preview_note_on`.
    preview_buf: Vec<PreviewLayer>,
    /// Note id of this block's first candidate. Ids are handed out in \[39\]
    block_first_id: u64,
    /// Voices whose SF2 delay pushes their start past this block.
    deferred: Vec<(u64, SpawnCmd)>,

    pub stats: DriverStats,
}

impl Driver {
    pub fn open(cfg: &Config, bank: Arc<Bank>, midi: impl AsRef<Path>) -> Result<Self> {
        cfg.validate()?;
        let stream = MidiStream::open(midi)?;
        let clock = TempoClock::new(stream.division, cfg.sample_rate);

        let mut d = Driver {
            cfg: cfg.clone(),
            bank,
            stream,
            clock,
            gates: GateTable::new(cfg),
            chan: ChannelTable::new(cfg),
            limiter: Limiter::new(cfg.sample_rate),
            brickwall: Brickwall::new(
                cfg.sample_rate,
                cfg.limiter_ceiling(),
                cfg.limiter_lookahead_ms,
                cfg.limiter_release_ms,
                cfg.limiter_sustain_ms,
                cfg.limiter_true_peak,
            ),
            bank_msb: [0; 16],
            bank_lsb: [0; 16],
            program: [0; 16],
            drum_map: DEFAULT_DRUM_MAP,
            preset: [0; 16],
            bend_val: [0; 16],
            bend_range: [2.0; 16],
            coarse_tune: [0.0; 16],
            fine_tune: [0.0; 16],
            fine_tune_raw: [8192; 16],
            rpn_sel: [0; 16],
            cc_volume: [crate::bank::POWER_ON_VOLUME; 16],
            cc_expression: [127; 16],
            cc_pan: [64; 16],
            cc_sound: [[64; 5]; 16],
            cc_mod: [0; 16],
            cc_vib_rate: [64; 16],
            cc_vib_depth: [64; 16],
            cc_tremolo: [0; 16],
            cc_soft: [0; 16],
            lsb_mod: [0; 16],
            lsb_volume: [0; 16],
            lsb_pan: [0; 16],
            lsb_expression: [0; 16],
            lfo_phase: [0.0; 16],
            variants: vec![[64 >> SOUND_CC_SHIFT; 5]],
            seen_states: vec![[64 >> SOUND_CC_SHIFT; 5]],
            cur_variant: [0; 16],
            variant_used: 1,
            variant_seen: vec![0],
            variant_clock: 0,
            pending_variants: Vec::new(),
            spawn_variants: false,
            next_note_id: 1,
            block_index: 0,
            pending: None,
            stream_done: false,
            end_sent: false,
            primed: false,
            tail_blocks_left: (MAX_TAIL_SECONDS * cfg.sample_rate as f64
                / cfg.block_frames as f64) as u64,
            spawn_buf: Vec::new(),
            notes: Vec::new(),
            note_ordinal: Vec::new(),
            cands: Vec::new(),
            cand_seen: 0,
            cand_stride: 1,
            cand_cap: (cfg.max_block_candidates as usize).max(cfg.pool_slots() as usize * 2),
            deferred_now: Vec::new(),
            preview_buf: Vec::new(),
            block_first_id: 0,
            deferred: Vec::new(),
            stats: DriverStats::default(),
        };
        for ch in 0..16 {
            d.refresh_preset(ch, 0);
        }
        Ok(d)
    }

    pub fn track_count(&self) -> u16 {
        self.stream.track_count
    }

    /// Track data decoded so far and the file's total, in bytes. See \[40\]
    pub fn input_bytes(&self) -> (u64, u64) {
        (self.stream.bytes_read(), self.stream.bytes_total())
    }

    fn refresh_preset(&mut self, ch: usize, program: u8) {
        self.program[ch] = program;
        // [41]
        let bank_num = if self.drum_map[ch] != 0 {
            128u16
        } else {
            self.bank_msb[ch] as u16
        };
        self.preset[ch] = self
            .bank
            .find_preset(bank_num, program as u16)
            .unwrap_or(0);
    }

    /// Render one block. Returns false once the file and its tail are done. \[42\]
    fn prepare_block(&mut self) -> Result<()> {
        let prof = self.cfg.profile;
        let t_block = std::time::Instant::now();
        let block_frames = self.cfg.block_frames as u64;
        let block_start = self.block_index * block_frames;
        let block_end = block_start + block_frames;

        self.gates.begin_block();
        self.chan.begin_block();
        // [43]
        self.variant_used = 1;
        for v in self.cur_variant {
            self.variant_used |= 1u64 << v;
        }
        self.spawn_buf.clear();
        self.notes.clear();
        self.note_ordinal.clear();
        self.cands.clear();
        self.cand_seen = 0;
        self.cand_stride = 1;
        self.deferred_now.clear();
        self.block_first_id = self.next_note_id;
        self.spawn_variants = false;

        // [44]
        let mut i = 0;
        while i < self.deferred.len() {
            if self.deferred[i].0 < block_end {
                let (frame, mut cmd) = self.deferred.swap_remove(i);
                cmd.start_rel = frame.saturating_sub(block_start) as u32;
                if cmd.variant != 0 {
                    self.variant_used |= 1u64 << cmd.variant;
                    self.spawn_variants = true;
                }
                if self.cands.len() >= self.cand_cap {
                    self.thin_cands();
                }
                if let Some(index) = self.offer_cand() {
                    self.cands.push(Cand {
                        key: rank_key(cmd.gain_l, cmd.gain_r, index),
                        note: DEFERRED,
                        region: self.deferred_now.len() as u32,
                        id_off: 0,
                    });
                }
                self.deferred_now.push(cmd);
            } else {
                i += 1;
            }
        }

        // Drain events belonging to this block.
        loop {
            let (tick, ev) = match self.pending.take() {
                Some(p) => p,
                None => match self.stream.next() {
                    Some(p) => p,
                    None => {
                        self.stream_done = true;
                        break;
                    }
                },
            };

            if let Event::Tempo(us) = ev {
                // [45]
                self.clock.set_tempo(tick, us);
                self.stats.events += 1;
                continue;
            }

            let frame = self.clock.frame_at(tick);
            let frame = if frame < 0.0 { 0 } else { frame as u64 };
            if frame >= block_end {
                self.pending = Some((tick, ev));
                break;
            }
            let rel = frame.saturating_sub(block_start) as u32;
            let rel = rel.min(self.cfg.block_frames - 1);
            self.stats.events += 1;
            self.handle_event(ev, rel, block_start);
        }

        if prof {
            self.stats.us_drain += t_block.elapsed().as_micros() as u64;
        }

        // Once the file runs out, release everything so looping voices stop.
        if self.stream_done && !self.end_sent {
            for ch in 0..16u8 {
                // [46]
                self.gates.all_sound_off(ch, 0);
            }
            self.end_sent = true;
        }

        self.block_index += 1;
        Ok(())
    }

    pub fn next_block(&mut self, backend: &mut dyn Backend, out: &mut [f32]) -> Result<bool> {
        if out.len() != self.cfg.block_samples() {
            bail!(
                "output block is {} samples, expected {}",
                out.len(),
                self.cfg.block_samples()
            );
        }

        let prof = self.cfg.profile;
        let t_block = std::time::Instant::now();
        let block_frames = self.cfg.block_frames as u64;

        // [47]
        if !self.primed {
            self.prepare_block()?;
            self.primed = true;
        }

        for (i, m) in std::mem::take(&mut self.pending_variants) {
            let data = self.bank.build_variant(&self.cfg, &m);
            let menv = self.bank.build_menv_variant(&self.cfg, &m);
            backend.set_params_variant(i, &data, &menv)?;
        }

        self.gates.end_block();
        self.chan.end_block();
        self.apply_modulation();
        self.chan.refresh_active();
        backend.set_gates(&self.gates.off_meta, &self.gates.off_runs)?;
        backend.set_channels(
            &self.chan.rows,
            self.chan.bend_active(),
            self.chan.gain_active(),
            // [48]
            self.chan.variant_active() || self.spawn_variants,
            self.chan.cut_active(),
        )?;
        // [49]
        let t_admit = std::time::Instant::now();
        let live = backend.stats().active_voices as u32;
        // [50]
        let want = self.cands.len();
        let take = self.cfg.admit_take(live, want as u32) as usize;
        self.stats.dropped += self.cand_seen - take as u64;
        let stolen_before = backend.stats().stolen;
        self.stats.last_live = live;
        self.stats.last_want = self.cand_seen;
        self.stats.last_take = take as u32;

        if take < want && take > 0 {
            match self.cfg.admit_rule {
                // [51]
                AdmitRule::Even => {
                    for i in 0..take {
                        self.cands[i] = self.cands[crate::voice::spawn_pick(i, want, take)];
                    }
                }
                AdmitRule::Loudest => self.rank_candidates(want, take),
            }
        }
        // [52]
        self.stats.last_want_energy = self.cands.iter().map(|c| (c.key >> 48) & 0x7FFF).sum();
        self.stats.last_take_energy =
            self.cands[..take.min(self.cands.len())].iter().map(|c| (c.key >> 48) & 0x7FFF).sum();
        self.materialise(take);
        if prof {
            self.stats.us_admit += t_admit.elapsed().as_micros() as u64;
        }

        let t_spawn = std::time::Instant::now();
        backend.spawn(&self.spawn_buf)?;
        if prof {
            self.stats.us_spawn += t_spawn.elapsed().as_micros() as u64;
        }
        self.stats.last_stolen = backend.stats().stolen - stolen_before;
        // [53]
        self.stats.voices_spawned += self.cand_seen;
        backend.submit()?;

        // [54]
        let was_done = self.stream_done;
        let had_deferred = !self.deferred.is_empty();

        // [55]
        self.prepare_block()?;

        let t_render = std::time::Instant::now();
        backend.finish(out)?;
        if prof {
            self.stats.us_render += t_render.elapsed().as_micros() as u64;
        }

        if self.cfg.limiter {
            match self.cfg.limiter_mode {
                LimiterMode::Off => {}
                // [56]
                LimiterMode::Omni => {
                    self.limiter.process(out);
                    self.brickwall.process(out);
                }
                LimiterMode::Brickwall => self.brickwall.process(out),
            }
        }
        // [57]
        if self.cfg.clamp_output {
            self.stats.clipped += clamp_block(out);
        }

        if self.cfg.nan_guard {
            if let Some(i) = out.iter().position(|v| !v.is_finite()) {
                bail!(
                    "block {} sample {} is {}; the synth produced a non-finite value",
                    self.stats.blocks,
                    i,
                    out[i]
                );
            }
        }

        let st = backend.stats();
        self.stats.peak = self.stats.peak.max(st.peak);
        self.stats.frames += block_frames;
        self.stats.blocks += 1;
        if prof {
            self.stats.us_total += t_block.elapsed().as_micros() as u64;
        }

        let more = if !was_done || had_deferred {
            true
        } else if st.active_voices > 0 {
            self.tail_blocks_left = self.tail_blocks_left.saturating_sub(1);
            if self.tail_blocks_left == 0 {
                log::warn!(
                    "tail exceeded {MAX_TAIL_SECONDS} s with {} voices still alive; stopping",
                    st.active_voices
                );
                false
            } else {
                true
            }
        } else {
            false
        };
        Ok(more)
    }

    /// Assemble the device-side command for one built layer.
    #[allow(clippy::too_many_arguments)]
    fn make_cmd(
        v: &VoiceSpawn,
        variant: u32,
        gate_slot: u32,
        ordinal: u32,
        start_rel: u32,
        id: u64,
        row_bias: u32,
    ) -> SpawnCmd {
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
            // [58]
            variant,
            region: v.region,
            gate_slot,
            ordinal,
            start_rel,
            note_id_lo: id as u32,
            note_id_hi: (id >> 32) as u32,
            gain_l: v.gain_l,
            gain_r: v.gain_r,
            // [59]
            row_bias,
        }
    }

    /// Move the `take` best candidates to the front, ranked *within* time \[60\]
    fn rank_candidates(&mut self, want: usize, take: usize) {
        let cands = &mut self.cands;
        let strata = take.min(ADMIT_STRATA);
        for s in 0..strata {
            let lo = (s as u64 * want as u64 / strata as u64) as usize;
            let hi = ((s + 1) as u64 * want as u64 / strata as u64) as usize;
            let olo = (s as u64 * take as u64 / strata as u64) as usize;
            let ohi = ((s + 1) as u64 * take as u64 / strata as u64) as usize;
            let q = ohi - olo;
            if q == 0 {
                continue;
            }
            // Ranking is a plain sort over one word.
            let key = |c: &Cand| std::cmp::Reverse(c.key);
            let seg = &mut cands[lo..hi];
            let n = seg.len();
            if q < n {
                seg.select_nth_unstable_by_key(q, key);
            }
            // [61]

            // [62]
            for j in 0..q {
                cands.swap(olo + j, lo + j);
            }
        }
    }


    /// Number the next admission candidate, and say whether the stride keeps \[63\]
    #[inline]
    fn offer_cand(&mut self) -> Option<u64> {
        let index = self.cand_seen;
        self.cand_seen += 1;
        (index & (self.cand_stride - 1) == 0).then_some(index)
    }

    /// Halve this block's candidates once the list reaches `cand_cap`: keep \[64\]
    fn thin_cands(&mut self) {
        let mut kept = 0;
        let mut notes = 0;
        let mut last = DEFERRED;
        for r in (0..self.cands.len()).step_by(2) {
            let mut c = self.cands[r];
            if c.note != DEFERRED {
                // [65]
                if c.note != last {
                    last = c.note;
                    self.notes[notes] = self.notes[c.note as usize];
                    self.note_ordinal[notes] = self.note_ordinal[c.note as usize];
                    notes += 1;
                }
                c.note = notes as u32 - 1;
            }
            self.cands[kept] = c;
            kept += 1;
        }
        self.cands.truncate(kept);
        self.notes.truncate(notes);
        self.note_ordinal.truncate(notes);
        self.cand_stride *= 2;
    }

    /// Build the voices for the first `take` candidates, appending them to \[66\]
    fn materialise(&mut self, take: usize) {
        let cands = std::mem::take(&mut self.cands);
        for c in &cands[..take.min(cands.len())] {
            if c.note == DEFERRED {
                self.spawn_buf.push(self.deferred_now[c.region as usize]);
                continue;
            }
            let (ch, key, vel, variant, rel, row_bias) =
                note_unpack(self.notes[c.note as usize]);
            let ordinal = self.note_ordinal[c.note as usize];
            let Some(v) = self.bank.build_layer(c.region, key, vel, &self.cfg) else {
                // [67]
                debug_assert!(false, "preview named a layer build_layer will not build");
                continue;
            };
            // [68]
            let start_rel = if v.delay_frames == 0 {
                rel
            } else {
                rel + v.delay_frames
            };
            self.spawn_buf.push(Self::make_cmd(
                &v,
                variant,
                GateTable::slot(ch, key) as u32,
                ordinal,
                start_rel,
                self.block_first_id + c.id_off,
                row_bias,
            ));
        }
        self.cands = cands;
    }

    fn handle_event(&mut self, ev: Event, rel: u32, block_start: u64) {
        match ev {
            Event::NoteOn { ch, key, vel } => {
                let ordinal = self.gates.note_on(ch, key, rel);
                self.stats.notes += 1;
                let variant = self.cur_variant[ch as usize & 15];

                // [69]
                let mut prev = std::mem::take(&mut self.preview_buf);
                prev.clear();
                // [70]
                self.bank.preview_note_on(
                    self.preset[ch as usize & 15],
                    key,
                    vel,
                    self.next_note_id,
                    self.cfg.max_layers as usize,
                    &mut prev,
                );

                // [71]
                if self.cands.len() + prev.len() > self.cand_cap {
                    self.thin_cands();
                }
                let note = self.notes.len() as u32;
                let mut recorded = false;
                for p in prev.iter() {
                    // [72]
                    let id = self.next_note_id;
                    self.next_note_id += 1;

                    if p.delay_frames != 0 {
                        let start = block_start + rel as u64 + p.delay_frames as u64;
                        if start >= block_start + self.cfg.block_frames as u64 {
                            // [73]
                            if let Some(v) =
                                self.bank.build_layer(p.region, key, vel, &self.cfg)
                            {
                                // [74]
                                let cmd = Self::make_cmd(
                                    &v,
                                    variant,
                                    GateTable::slot(ch, key) as u32,
                                    ordinal,
                                    rel,
                                    id,
                                    0,
                                );
                                self.deferred.push((start, cmd));
                            }
                            continue;
                        }
                    }

                    // [75]
                    if variant != 0 {
                        self.variant_used |= 1u64 << variant;
                        self.spawn_variants = true;
                    }
                    let Some(index) = self.offer_cand() else {
                        continue;
                    };

                    if !recorded {
                        self.notes.push(note_pack(
                            ch,
                            key,
                            vel,
                            variant,
                            rel,
                            self.chan.row_bias(ch, rel),
                        ));
                        self.note_ordinal.push(ordinal);
                        recorded = true;
                    }
                    self.cands.push(Cand {
                        key: rank_key(p.gain_l, p.gain_r, index),
                        note,
                        region: p.region,
                        id_off: id - self.block_first_id,
                    });
                }
                self.preview_buf = prev;
            }
            Event::NoteOff { ch, key } => {
                self.gates.note_off(ch, key, rel);
            }
            Event::Cc { ch, num, val } => {
                let c = ch as usize & 15;
                match num {
                    0 => self.bank_msb[c] = val,
                    1 => {
                        self.cc_mod[c] = val;
                        self.lsb_mod[c] = 0;
                    }
                    33 => self.lsb_mod[c] = val,
                    39 => {
                        self.lsb_volume[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    42 => {
                        self.lsb_pan[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    43 => {
                        self.lsb_expression[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    32 => self.bank_lsb[c] = val,
                    // [76]
                    6 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = val as f64;
                        self.refresh_bend(c, rel);
                    }
                    38 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = self.bend_range[c].trunc() + val as f64 / 100.0;
                        self.refresh_bend(c, rel);
                    }
                    // [77]
                    6 if self.rpn_sel[c] == 1 => {
                        self.fine_tune_raw[c] = (self.fine_tune_raw[c] & 0x7F) | ((val as u16) << 7);
                        self.fine_tune[c] = (self.fine_tune_raw[c] as f64 - 8192.0) / 8192.0 * 100.0;
                        self.refresh_bend(c, rel);
                    }
                    38 if self.rpn_sel[c] == 1 => {
                        self.fine_tune_raw[c] = (self.fine_tune_raw[c] & 0x3F80) | val as u16;
                        self.fine_tune[c] = (self.fine_tune_raw[c] as f64 - 8192.0) / 8192.0 * 100.0;
                        self.refresh_bend(c, rel);
                    }
                    // [78]
                    6 if self.rpn_sel[c] == 2 => {
                        self.coarse_tune[c] = val as f64 - 64.0;
                        self.refresh_bend(c, rel);
                    }
                    96 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = (self.bend_range[c] + 1.0).min(127.0);
                        self.refresh_bend(c, rel);
                    }
                    97 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = (self.bend_range[c] - 1.0).max(0.0);
                        self.refresh_bend(c, rel);
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
                        self.refresh_gain(c, rel);
                    }
                    10 => {
                        self.cc_pan[c] = val;
                        self.lsb_pan[c] = 0;
                        self.refresh_gain(c, rel);
                    }
                    11 => {
                        self.cc_expression[c] = val;
                        self.lsb_expression[c] = 0;
                        self.refresh_gain(c, rel);
                    }
                    64 => self.gates.set_sustain(ch, val >= 64, rel),
                    66 => self.gates.set_sostenuto(ch, val >= 64, rel),
                    67 => {
                        self.cc_soft[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    71 => self.set_sound_cc(ch, 0, val, rel),
                    72 => self.set_sound_cc(ch, 1, val, rel),
                    73 => self.set_sound_cc(ch, 2, val, rel),
                    74 => self.set_sound_cc(ch, 3, val, rel),
                    75 => self.set_sound_cc(ch, 4, val, rel),
                    76 => self.cc_vib_rate[c] = val,
                    77 => self.cc_vib_depth[c] = val,
                    92 => self.cc_tremolo[c] = val,
                    120 => {
                        // [79]
                        self.gates.all_sound_off(ch, rel);
                        self.chan.set_sound_off(ch, rel, self.next_note_id);
                    }
                    121 => self.reset_controllers(ch, rel),
                    // [80]
                    123..=127 => self.gates.all_notes_off(ch, rel),
                    _ => {}
                }
            }
            Event::Program { ch, val } => {
                self.refresh_preset(ch as usize & 15, val);
            }
            Event::DrumPart { ch, map } => {
                let c = ch as usize & 15;
                if self.drum_map[c] != map {
                    self.drum_map[c] = map;
                    self.refresh_preset(c, self.program[c]);
                }
            }
            Event::ResetParts => {
                for (c, want) in DEFAULT_DRUM_MAP.iter().enumerate() {
                    if self.drum_map[c] != *want {
                        self.drum_map[c] = *want;
                        self.refresh_preset(c, self.program[c]);
                    }
                }
            }
            Event::PitchBend { ch, val } => {
                self.bend_val[ch as usize & 15] = val;
                self.refresh_bend(ch as usize & 15, rel);
            }
            Event::Tempo(_) | Event::Other => {}
        }
    }

    /// Take one of CC71-CC75 and move the channel onto whichever copy of the \[81\]
    fn set_sound_cc(&mut self, ch: u8, which: usize, val: u8, rel: u32) {
        let c = ch as usize & 15;
        if self.cc_sound[c][which] == val {
            return;
        }
        self.cc_sound[c][which] = val;

        let mut want = [0u8; 5];
        for (i, w) in want.iter_mut().enumerate() {
            *w = self.cc_sound[c][i] >> SOUND_CC_SHIFT;
        }
        if !self.seen_states.contains(&want) {
            self.seen_states.push(want);
            self.stats.variant_states = self.seen_states.len() as u32;
        }
        let cap = self.cfg.max_param_variants.clamp(1, 63);
        let idx = match self.variants.iter().position(|v| *v == want) {
            Some(i) => i as u32,
            None if (self.variants.len() as u32) < cap => {
                let i = self.variants.len() as u32;
                self.variants.push(want);
                self.variant_seen.push(0);
                self.stats.param_variants = i + 1;
                self.build_variant_at(i, c);
                i
            }
            // [82]
            None => {
                let victim = (1..self.variants.len())
                    .filter(|i| self.variant_used & (1u64 << i) == 0)
                    .min_by_key(|i| self.variant_seen[*i]);
                match victim {
                    Some(i) => {
                        self.variants[i] = want;
                        self.build_variant_at(i as u32, c);
                        self.stats.variant_rebuilds += 1;
                        i as u32
                    }
                    // Every slot is spoken for within this block already.
                    None => {
                        self.stats.variant_fallbacks += 1;
                        let mut best = 0u32;
                        let mut best_d = u32::MAX;
                        for (i, v) in self.variants.iter().enumerate() {
                            let d: u32 = v
                                .iter()
                                .zip(&want)
                                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                                .sum();
                            if d < best_d {
                                best_d = d;
                                best = i as u32;
                            }
                        }
                        best
                    }
                }
            }
        };
        self.variant_clock += 1;
        self.variant_seen[idx as usize] = self.variant_clock;
        self.variant_used |= 1u64 << idx;
        self.cur_variant[c] = idx;
        self.chan.set_variant(ch, idx, rel);
    }

    /// Queue the build of variant `i` from channel `c`'s current controllers. \[83\]
    fn build_variant_at(&mut self, i: u32, c: usize) {
        let m = ParamMod::from_controllers(
            self.cc_sound[c][0],
            self.cc_sound[c][1],
            self.cc_sound[c][2],
            self.cc_sound[c][3],
            self.cc_sound[c][4],
        );
        // A slot rebuilt twice in one block only needs its last state.
        self.pending_variants.retain(|(j, _)| *j != i);
        self.pending_variants.push((i, m));
    }

    /// CC121, Reset All Controllers. \[84\]
    fn reset_controllers(&mut self, ch: u8, rel: u32) {
        let c = ch as usize & 15;
        // Pitch wheel to centre -- but not its range.
        self.bend_val[c] = 0;
        self.rpn_sel[c] = RPN_NONE;
        self.cc_expression[c] = 127;
        self.lsb_expression[c] = 0;
        self.cc_mod[c] = 0;
        self.lsb_mod[c] = 0;
        self.cc_soft[c] = 0;
        self.refresh_bend(c, rel);
        self.refresh_gain(c, rel);
        self.gates.reset_controllers(ch, rel);
    }

    /// Lay the vibrato and tremolo LFOs over the controller rows. \[85\]
    fn apply_modulation(&mut self) {
        let tiles = self.chan.tiles;
        let tile_seconds = self.cfg.gate_frames as f64 / self.cfg.sample_rate as f64;
        for c in 0..16usize {
            let vib = (self.cc_mod[c] as f64 * 128.0 + self.lsb_mod[c] as f64) / 16383.0
                * (self.cc_vib_depth[c] as f64 / 64.0)
                * 50.0;
            let trem = self.cc_tremolo[c] as f64 / 127.0 * 0.25;
            let rate = 5.0 * (2.0f64).powf((self.cc_vib_rate[c] as f64 - 64.0) / 32.0);
            if vib <= 0.0 && trem <= 0.0 {
                // [86]
                self.lfo_phase[c] =
                    (self.lfo_phase[c] + rate * tile_seconds * tiles as f64).fract();
                continue;
            }
            let mut phase = self.lfo_phase[c];
            for t in 0..tiles {
                let s = (phase * std::f64::consts::TAU).sin();
                if vib > 0.0 {
                    self.chan
                        .modulate_bend(c as u8, t, bend_factor(vib * s / 100.0));
                }
                if trem > 0.0 {
                    self.chan
                        .modulate_gain(c as u8, t, (1.0 - trem + trem * s) as f32);
                }
                phase = (phase + rate * tile_seconds).fract();
            }
            self.lfo_phase[c] = phase;
        }
    }

    /// Freeze this channel's bend into the factor the backends multiply by. \[87\]
    fn refresh_bend(&mut self, c: usize, rel: u32) {
        // [88]
        let semitones = (self.bend_val[c] as f64 / 8192.0) * self.bend_range[c]
            + self.coarse_tune[c]
            + self.fine_tune[c] / 100.0;
        self.chan.set_bend(c as u8, bend_factor(semitones), rel);
    }

    /// Fold CC7, CC11 and CC10 into the pair of gains a voice multiplies by. \[89\]
    fn refresh_gain(&mut self, c: usize, rel: u32) {
        // [90]
        let fine = |msb: u8, lsb: u8| {
            if lsb == 0 {
                msb as f32 / 127.0
            } else {
                (msb as f32 * 128.0 + lsb as f32) / 16383.0
            }
        };
        // [91]
        let v = fine(self.cc_volume[c], self.lsb_volume[c])
            / (crate::bank::POWER_ON_VOLUME as f32 / 127.0);
        let e = fine(self.cc_expression[c], self.lsb_expression[c]);
        // [92]
        let soft = 1.0 - 0.5 * (self.cc_soft[c] as f32 / 127.0);
        let amp = (v * v) * (e * e) * soft;
        let theta = fine(self.cc_pan[c], self.lsb_pan[c]) * std::f32::consts::FRAC_PI_2;
        // [93]
        let (l, r) = if self.cc_pan[c] == 64 && self.lsb_pan[c] == 0 {
            (1.0, 1.0)
        } else {
            (theta.cos() * std::f32::consts::SQRT_2, theta.sin() * std::f32::consts::SQRT_2)
        };
        self.chan.set_gain(c as u8, amp * l, amp * r, rel);
    }

    pub fn seconds_rendered(&self) -> f64 {
        self.stats.frames as f64 / self.cfg.sample_rate as f64
    }

}
