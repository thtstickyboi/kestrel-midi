// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Block-at-a-time render driver. \[1\]

use crate::backend::Backend;
use crate::bank::{Bank, ParamMod, PreviewLayer, VoiceSpawn};
use crate::config::{AdmitRule, Config};
use crate::limiter::OutputStage;
use crate::midi::{Event, MidiStream, TempoClock, TrackSelection, CHANNELS};
use crate::fixed::bend_factor;
use crate::porta::{self, Glide};
use crate::snap::{Dec, Enc};
use crate::{edo31, mts};
use crate::voice::{ChannelTable, GateTable, SpawnCmd};
use anyhow::{bail, Context, Result};

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
        5 => ("portamento time", Applied),
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
        65 => ("portamento", Applied),
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
        84 => ("portamento control", Applied),
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
    /// Blocks written as zeros on the host without a dispatch, because no \[11\]
    pub silent_blocks: u64,
    pub events: u64,
    /// Note-ons the block could not admit. Counted here rather than in the \[12\]
    pub dropped: u64,
    pub notes: u64,
    /// Note-ons `Config::min_velocity` skipped. Not in `notes`, which counts \[13\]
    pub notes_skipped: u64,
    pub voices_spawned: u64,
    pub peak: f32,
    /// Samples the final clamp had to pull back to full scale. Every run of \[14\]
    pub clipped: u64,

    // [15]
    pub last_live: u32,
    /// Note-on layers this block queued.
    pub last_want: u64,
    /// Layers admission let through.
    pub last_take: u32,
    /// Voices the backend killed to make room for them.
    pub last_stolen: u64,
    /// Microseconds the host waited on the device for this block: the render \[16\]
    pub last_wait_us: u64,
    /// Summed opening amplitude of every layer the block queued, and of the \[17\]
    pub last_want_energy: u64,
    pub last_take_energy: u64,
}

impl DriverStats {
    /// The counters a render adds up as it goes. The microsecond timings, which \[18\]
    fn save(&self, e: &mut Enc) {
        let DriverStats {
            us_total: _,
            us_drain: _,
            us_admit: _,
            us_spawn: _,
            us_render: _,
            param_variants,
            variant_fallbacks,
            variant_states,
            variant_rebuilds,
            frames,
            blocks,
            silent_blocks,
            events,
            dropped,
            notes,
            notes_skipped,
            voices_spawned,
            peak,
            clipped,
            last_live: _,
            last_want: _,
            last_take: _,
            last_stolen: _,
            last_wait_us: _,
            last_want_energy: _,
            last_take_energy: _,
        } = self;
        e.u32(*param_variants);
        e.u64(*variant_fallbacks);
        e.u32(*variant_states);
        e.u64(*variant_rebuilds);
        e.u64(*frames);
        e.u64(*blocks);
        e.u64(*silent_blocks);
        e.u64(*events);
        e.u64(*dropped);
        e.u64(*notes);
        e.u64(*notes_skipped);
        e.u64(*voices_spawned);
        e.f32(*peak);
        e.u64(*clipped);
    }

    fn load(&mut self, d: &mut Dec) -> Result<()> {
        self.param_variants = d.u32()?;
        self.variant_fallbacks = d.u64()?;
        self.variant_states = d.u32()?;
        self.variant_rebuilds = d.u64()?;
        self.frames = d.u64()?;
        self.blocks = d.u64()?;
        self.silent_blocks = d.u64()?;
        self.events = d.u64()?;
        self.dropped = d.u64()?;
        self.notes = d.u64()?;
        self.notes_skipped = d.u64()?;
        self.voices_spawned = d.u64()?;
        self.peak = d.f32()?;
        self.clipped = d.u64()?;
        Ok(())
    }
}

/// Pack a note-on into one word: everything `build_layer` and `SpawnCmd` will \[19\]
#[inline]
fn note_pack(ch: u8, key: u8, vel: u8, variant: u32, rel: u32, row_bias: u32, glide: u16) -> u64 {
    debug_assert!(variant < 64 && rel < 65536 && row_bias < 2);
    (ch as u64)
        | ((key as u64 & 0x7F) << 8)
        | ((vel as u64 & 0x7F) << 15)
        | ((variant as u64 & 0x3F) << 22)
        | ((rel as u64 & 0xFFFF) << 28)
        | ((row_bias as u64 & 1) << 44)
        | ((glide as u64) << 45)
}

#[inline]
fn note_unpack(w: u64) -> (u8, u8, u8, u32, u32, u32, u16) {
    (
        (w & 0xFF) as u8,
        ((w >> 8) & 0x7F) as u8,
        ((w >> 15) & 0x7F) as u8,
        ((w >> 22) & 0x3F) as u32,
        ((w >> 28) & 0xFFFF) as u32,
        ((w >> 44) & 1) as u32,
        (w >> 45) as u16,
    )
}

/// How a note-on is tuned: what `Driver::tune_note` made of its key.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NoteTune {
    /// The key whose sample and parameters the voice is built from.
    key: u8,
    /// Cents from that key's pitch, from the key tuning and the channel's \[20\]
    cents: f32,
}

/// One MIDI Tuning Standard tuning program: every key's offset from 12-tone, in \[21\]
#[derive(Clone, Debug, PartialEq, Eq)]
struct MtsTable {
    bank: u8,
    program: u8,
    offsets: Box<[i32; 128]>,
}

/// A note-on's portamento in 16 bits: the interval it glides across, as the \[22\]
#[inline]
fn glide_pack(semitones: i32, cc5: u8) -> u16 {
    (semitones as i8 as u8 as u16) | ((cc5 as u16 & 0x7F) << 8)
}

#[inline]
fn glide_unpack(g: u16) -> (i32, u8) {
    (g as u8 as i8 as i32, (g >> 8) as u8)
}

/// `Driver::porta_last` for a channel that has not played a note.
const NO_KEY: u8 = 0xFF;

/// One layer a block might admit, before anything has been built for it. \[23\]
#[derive(Clone, Copy)]
struct Cand {
    key: u64,
    /// Index into `notes`, or `DEFERRED` for a voice built in an earlier block \[24\]
    note: u32,
    /// The region to build, so materialising never has to preview again. For a \[25\]
    region: u32,
    /// Note id, as an offset from `block_first_id`. Unused when `DEFERRED`, \[26\]
    id_off: u64,
}

/// `Cand::note` for a candidate that came off the deferred queue.
const DEFERRED: u32 = u32::MAX;

/// Admitted candidates per thread below which `materialise` stays on one.
pub(crate) const MATERIALISE_MIN: usize = 8192;
/// Candidates a `materialise` thread takes at a time.
const MATERIALISE_CHUNK: usize = 2048;

/// Everything building an admitted candidate reads, and nothing it writes, so \[27\]
struct Builder<'a> {
    bank: &'a Bank,
    cfg: &'a Config,
    /// Only when analytic phase is on.
    phase: Option<&'a crate::phase::PhaseBank>,
    glide: &'a Glide,
    notes: &'a [u64],
    note_ordinal: &'a [u32],
    note_ticks: &'a [u64],
    /// Each note's key to build on and its tuning in cents, once anything has been \[28\]
    note_tune: &'a [NoteTune],
    deferred_now: &'a [SpawnCmd],
    first_id: u64,
    block_start: u64,
}

impl Builder<'_> {
    /// The command for one candidate. `memo` holds the last note's analytic \[29\]
    #[inline]
    fn build(&self, c: &Cand, memo: &mut Option<(u32, crate::phase::Angle)>) -> Option<SpawnCmd> {
        if c.note == DEFERRED {
            return Some(self.deferred_now[c.region as usize]);
        }
        let (ch, key, vel, variant, rel, row_bias, glide) = note_unpack(self.notes[c.note as usize]);
        let ordinal = self.note_ordinal[c.note as usize];
        // [30]
        let NoteTune { key: built_on, cents: tune } =
            self.note_tune.get(c.note as usize).copied().unwrap_or(NoteTune { key, cents: 0.0 });
        let Some(v) = self.bank.build_layer(c.region, built_on, vel, self.cfg, tune) else {
            // [31]
            debug_assert!(false, "preview named a layer build_layer will not build");
            return None;
        };
        // [32]
        let start_rel = if v.delay_frames == 0 {
            rel
        } else {
            rel + v.delay_frames
        };
        let mut cmd = Driver::make_cmd(
            &v,
            variant,
            GateTable::slot(ch, key) as u32,
            ordinal,
            start_rel,
            self.first_id + c.id_off,
            row_bias,
        );
        if let Some(pb) = self.phase {
            let angle = match *memo {
                Some((note, a)) if note == c.note => a,
                _ => {
                    let a = pb.angle(self.note_ticks[c.note as usize], ch, key);
                    *memo = Some((c.note, a));
                    a
                }
            };
            cmd.rotation = pb.coefficients(v.region, angle);
        }
        let (semitones, cc5) = glide_unpack(glide);
        let (fl, sl) = self.glide.spawn(semitones, cc5, start_rel);
        cmd.flags |= fl;
        cmd.note_id_hi |= sl;
        Some(cmd)
    }

    /// Where this voice's glide ends, or 0 if it has none.
    #[inline]
    fn glide_end(&self, cmd: &SpawnCmd) -> u64 {
        if cmd.flags > porta::FLAG_BITS {
            self.block_start + (cmd.flags >> porta::REM_SHIFT) as u64
        } else {
            0
        }
    }
}

/// For `Config::min_velocity`: which of each (channel, key)'s unanswered \[33\]
struct QuietPairs {
    runs: Vec<std::collections::VecDeque<(bool, u64)>>,
}

impl QuietPairs {
    fn new() -> Self {
        Self { runs: (0..CHANNELS * 128).map(|_| Default::default()).collect() }
    }

    /// For a snapshot: the keys that have something unanswered, and their runs. \[34\]
    fn save(&self, e: &mut Enc) {
        let busy: Vec<usize> = (0..self.runs.len()).filter(|&i| !self.runs[i].is_empty()).collect();
        e.len_of(self.runs.len());
        e.len_of(busy.len());
        for i in busy {
            e.u32(i as u32);
            e.len_of(self.runs[i].len());
            for &(skipped, n) in &self.runs[i] {
                e.bool(skipped);
                e.u64(n);
            }
        }
    }

    fn load(&mut self, d: &mut Dec) -> Result<()> {
        let keys = d.u64()? as usize;
        if keys != self.runs.len() {
            bail!("the saved note pairing has {keys} keys and this render has {}", self.runs.len());
        }
        for q in &mut self.runs {
            q.clear();
        }
        let busy = d.len_of(12)?;
        for _ in 0..busy {
            let i = d.u32()? as usize;
            let n = d.len_of(9)?;
            let Some(q) = self.runs.get_mut(i) else {
                bail!("the saved note pairing names key {i}, which there is not");
            };
            for _ in 0..n {
                let (skipped, count) = (d.bool()?, d.u64()?);
                q.push_back((skipped, count));
            }
        }
        Ok(())
    }

    fn push(&mut self, slot: usize, skipped: bool) {
        let q = &mut self.runs[slot];
        match q.back_mut() {
            Some((s, n)) if *s == skipped => *n += 1,
            _ => q.push_back((skipped, 1)),
        }
    }

    /// Whether the note-on a note-off answers was skipped. A note-off with \[35\]
    fn pop(&mut self, slot: usize) -> bool {
        let q = &mut self.runs[slot];
        let Some((skipped, n)) = q.front_mut() else {
            return false;
        };
        let skipped = *skipped;
        *n -= 1;
        if *n == 0 {
            q.pop_front();
        }
        skipped
    }
}

/// Which channels are drum parts before a file says anything. Channel 10 -- \[36\]
const DEFAULT_DRUM_MAP: [u8; CHANNELS] = {
    let mut m = [0u8; CHANNELS];
    let mut c = 9;
    while c < CHANNELS {
        m[c] = 1;
        c += 16;
    }
    m
};

/// `admit_key`, reached without building a `SpawnCmd` first.
#[inline]
fn rank_key(gain_l: f32, gain_r: f32, index: u64) -> u64 {
    (crate::voice::rank_gain_q(gain_l, gain_r) << 48) | crate::voice::mix48(index)
}

/// A block `submit_block` has handed to the backend and `finish_block` has \[37\]
#[derive(Debug, Clone, Copy)]
struct InFlight {
    t_block: std::time::Instant,
    silent: bool,
    was_done: bool,
    had_deferred: bool,
}

pub struct Driver {
    phase_bank: Arc<crate::phase::PhaseBank>,
    note_ticks: Vec<u64>,
    /// Scale/octave tuning, `mts`: per channel, how far each pitch class is \[38\]
    tune: [i32; CHANNELS * 12],
    /// Whether anything has been tuned: a channel's scale/octave tuning, a key \[39\]
    tuned: bool,
    /// This block's tuned note-ons, one per entry of `notes` once `tuned`, and empty \[40\]
    note_tune: Vec<NoteTune>,
    /// The tuning of every key that the configuration gives (`Config::tuning`, which \[41\]
    key_static: Option<Arc<crate::tuning::Tuning>>,
    /// MIDI Tuning Standard tuning programs a file has set (`Config::mts_notes`): each a \[42\]
    mts_tables: Vec<MtsTable>,
    /// Each channel's selected tuning program, `bank << 7 | program`, RPN 3 and 4. \[43\]
    tuning_sel: [u16; CHANNELS],
    cfg: Config,
    bank: Arc<Bank>,
    stream: MidiStream,
    clock: TempoClock,
    gates: GateTable,
    chan: ChannelTable,
    output: OutputStage,

    bank_msb: [u8; CHANNELS],
    bank_lsb: [u8; CHANNELS],
    /// Last program change per channel, kept because the drum-part SysEx can \[44\]
    program: [u8; CHANNELS],
    /// GS part type: 0 melodic, 1 or 2 for the two drum maps. Channel 10 is \[45\]
    drum_map: [u8; CHANNELS],
    preset: [u32; CHANNELS],
    /// Raw pitch bend, -8192..8191 relative to centre.
    bend_val: [i16; CHANNELS],
    /// Bend range in semitones, RPN 0. Two is the GM default.
    bend_range: [f64; CHANNELS],
    /// Channel coarse tuning, RPN 2, in semitones. The wire value is an MSB \[46\]
    coarse_tune: [f64; CHANNELS],
    /// Channel fine tuning, RPN 1, in cents. A 14-bit value centred on 8192 \[47\]
    fine_tune: [f64; CHANNELS],
    /// The raw 14-bit value behind it, kept so an MSB and an LSB written at \[48\]
    fine_tune_raw: [u16; CHANNELS],
    /// Currently selected RPN, from CC101 (MSB) and CC100 (LSB).
    rpn_sel: [u16; CHANNELS],
    /// CC7 channel volume, CC11 expression, CC10 pan.
    cc_volume: [u8; CHANNELS],
    cc_expression: [u8; CHANNELS],
    cc_pan: [u8; CHANNELS],
    /// CC71 resonance, CC72 release, CC73 attack, CC74 brightness, CC75 decay. \[49\]
    cc_sound: [[u8; 5]; CHANNELS],
    /// CC1 modulation depth, CC76 vibrato rate, CC77 vibrato depth, CC92 \[50\]
    cc_mod: [u8; CHANNELS],
    cc_vib_rate: [u8; CHANNELS],
    cc_vib_depth: [u8; CHANNELS],
    cc_tremolo: [u8; CHANNELS],
    cc_soft: [u8; CHANNELS],
    /// Fine halves for the four controllers that have one worth reading. A \[51\]
    lsb_mod: [u8; CHANNELS],
    lsb_volume: [u8; CHANNELS],
    lsb_pan: [u8; CHANNELS],
    lsb_expression: [u8; CHANNELS],
    /// LFO phase per channel, in cycles, carried across blocks so vibrato does \[52\]
    lfo_phase: [f64; CHANNELS],
    /// Portamento: CC65, CC5, a pending CC84 (zero for none), and the key of \[53\]
    porta_on: [bool; CHANNELS],
    porta_time: [u8; CHANNELS],
    porta_note: [u8; CHANNELS],
    porta_last: [u8; CHANNELS],
    glide: Glide,
    /// Frame by which every glide handed to the backend has landed. A backend \[54\]
    glide_until: u64,
    /// Quantised sound-controller states, indexed by variant number.
    variants: Vec<[u8; 5]>,
    /// The controller values each variant's table was built from, which are not \[55\]
    variant_ccs: Vec<[u8; 5]>,
    /// Every quantised state the file has asked for, for diagnostics.
    seen_states: Vec<[u8; 5]>,
    /// Which variant each channel is currently on.
    cur_variant: [u32; CHANNELS],
    /// Bitmask of variants any channel has pointed at during this block. A \[56\]
    variant_used: u64,
    /// Monotonic tick per variant, for choosing the stalest one to reuse.
    variant_seen: Vec<u64>,
    variant_clock: u64,
    /// Variants asked for this block but not yet built and uploaded.
    pending_variants: Vec<(u32, ParamMod)>,
    /// Set when a voice queued this block was born under a non-zero variant. \[57\]
    spawn_variants: bool,

    next_note_id: u64,
    block_index: u64,
    /// Event pulled from the stream that belongs to a later block.
    pending: Option<(u64, Event)>,
    /// For a per-track stream, the frame the whole file's last event falls \[58\]
    hold_frame: Option<u64>,
    /// Between `submit_block` and `finish_block`: what the second needs of \[59\]
    in_flight: Option<InFlight>,
    stream_done: bool,
    end_sent: bool,
    /// Whether block 0's tables have been built. Every later block is prepared \[60\]
    primed: bool,
    tail_blocks_left: u64,

    /// This block's admitted voices are the first `spawn_len`; see \[61\]
    spawn_buf: Vec<SpawnCmd>,
    spawn_len: usize,
    /// This block's note-ons, one packed entry each, plus their ordinals. \[62\]
    notes: Vec<u64>,
    note_ordinal: Vec<u32>,
    /// `Some` when `Config::min_velocity` skips anything: which note-offs \[63\]
    quiet: Option<QuietPairs>,
    /// One entry per admissible layer this block, up to `cand_cap`. See `Cand` \[64\]
    cands: Vec<Cand>,
    /// Candidates offered this block, counting the ones `cand_stride` skipped: \[65\]
    cand_seen: u64,
    /// Note-ons offered this block, counting the ones `cand_stride` skipped, \[66\]
    unit_seen: u64,
    /// Offer only units whose number is a multiple of this. One until the \[67\]
    cand_stride: u64,
    /// `Config::max_block_candidates`, never below twice the pool, so thinning \[68\]
    cand_cap: usize,
    /// This block's share of the deferred queue, already built. Kept apart from \[69\]
    deferred_now: Vec<SpawnCmd>,
    /// Scratch for `Bank::preview_note_on`.
    preview_buf: Vec<PreviewLayer>,
    /// Note id of this block's first candidate. Ids are handed out in \[70\]
    block_first_id: u64,
    /// Voices whose SF2 delay pushes their start past this block, with the \[71\]
    deferred: Vec<(u64, SpawnCmd, u16)>,
    /// Fill `DriverStats::last_want_energy` and `last_take_energy`; see \[72\]
    block_energy: bool,

    pub stats: DriverStats,
}

impl Driver {
    pub fn open(cfg: &Config, bank: Arc<Bank>, midi: impl AsRef<Path>) -> Result<Self> {
        let phase = crate::phase::PhaseBank::prepare(&bank, &cfg.phase)?;
        Self::open_prepared(cfg, bank, midi, phase)
    }

    pub(crate) fn open_prepared(cfg: &Config, bank: Arc<Bank>, midi: impl AsRef<Path>,
        phase_bank: Arc<crate::phase::PhaseBank>) -> Result<Self> {
        cfg.validate()?;
        Self::from_stream(cfg, bank, MidiStream::open_keys(midi, cfg.edo31)?, phase_bank)
    }

    /// Render the tracks `sel` names, alone: the per-track path. See \[73\]
    pub fn open_tracks(cfg: &Config, bank: Arc<Bank>, midi: impl AsRef<Path>, sel: &TrackSelection) -> Result<Self> {
        let phase = crate::phase::PhaseBank::prepare(&bank, &cfg.phase)?;
        Self::open_tracks_prepared(cfg, bank, midi, sel, phase)
    }

    pub(crate) fn open_tracks_prepared(cfg: &Config, bank: Arc<Bank>, midi: impl AsRef<Path>,
        sel: &TrackSelection, phase_bank: Arc<crate::phase::PhaseBank>) -> Result<Self> {
        cfg.validate()?;
        Self::from_stream(cfg, bank, MidiStream::open_tracks_keys(midi, sel, cfg.edo31)?, phase_bank)
    }

    fn from_stream(cfg: &Config, bank: Arc<Bank>, stream: MidiStream,
        phase_bank: Arc<crate::phase::PhaseBank>) -> Result<Self> {
        let clock = TempoClock::new(stream.division, cfg.sample_rate);
        let hold_frame = stream.end_frame(cfg.sample_rate);

        let mut d = Driver {
            hold_frame,
            in_flight: None,
            phase_bank,
            note_ticks: Vec::new(),
            tune: [0; CHANNELS * 12],
            // A tuning from the configuration is on from the first note.
            tuned: cfg.tuning.is_some(),
            note_tune: Vec::new(),
            key_static: cfg.tuning.clone(),
            mts_tables: Vec::new(),
            tuning_sel: [0; CHANNELS],
            cfg: cfg.clone(),
            bank,
            stream,
            clock,
            gates: GateTable::new(cfg),
            chan: ChannelTable::new(cfg),
            output: OutputStage::new(cfg),
            bank_msb: [0; CHANNELS],
            bank_lsb: [0; CHANNELS],
            program: [0; CHANNELS],
            drum_map: DEFAULT_DRUM_MAP,
            preset: [0; CHANNELS],
            bend_val: [0; CHANNELS],
            bend_range: [2.0; CHANNELS],
            coarse_tune: [0.0; CHANNELS],
            fine_tune: [0.0; CHANNELS],
            fine_tune_raw: [8192; CHANNELS],
            rpn_sel: [0; CHANNELS],
            cc_volume: [crate::bank::POWER_ON_VOLUME; CHANNELS],
            cc_expression: [127; CHANNELS],
            cc_pan: [64; CHANNELS],
            cc_sound: [[64; 5]; CHANNELS],
            cc_mod: [0; CHANNELS],
            cc_vib_rate: [64; CHANNELS],
            cc_vib_depth: [64; CHANNELS],
            cc_tremolo: [0; CHANNELS],
            cc_soft: [0; CHANNELS],
            lsb_mod: [0; CHANNELS],
            lsb_volume: [0; CHANNELS],
            lsb_pan: [0; CHANNELS],
            lsb_expression: [0; CHANNELS],
            lfo_phase: [0.0; CHANNELS],
            porta_on: [false; CHANNELS],
            porta_time: [0; CHANNELS],
            porta_note: [0; CHANNELS],
            porta_last: [NO_KEY; CHANNELS],
            glide: Glide::new(cfg),
            glide_until: 0,
            variants: vec![[64 >> SOUND_CC_SHIFT; 5]],
            variant_ccs: vec![[64; 5]],
            seen_states: vec![[64 >> SOUND_CC_SHIFT; 5]],
            cur_variant: [0; CHANNELS],
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
            spawn_len: 0,
            notes: Vec::new(),
            note_ordinal: Vec::new(),
            quiet: (cfg.min_velocity > 1).then(QuietPairs::new),
            cands: Vec::new(),
            cand_seen: 0,
            unit_seen: 0,
            cand_stride: 1,
            cand_cap: (cfg.max_block_candidates as usize).max(cfg.pool_slots() as usize * 2),
            deferred_now: Vec::new(),
            preview_buf: Vec::new(),
            block_first_id: 0,
            deferred: Vec::new(),
            block_energy: false,
            stats: DriverStats::default(),
        };
        for ch in 0..CHANNELS {
            d.refresh_preset(ch, 0);
        }
        if cfg.edo31 {
            d.tune_lanes();
        }
        Ok(d)
    }

    /// Set the scale/octave tuning of a channel's pitch class, in \[74\]
    fn set_tune(&mut self, ch: u8, pc: u8, units: i32) {
        self.tune[ch as usize * 12 + pc as usize % 12] = units;
        if units != 0 {
            self.start_tuning();
        }
    }

    /// Something is tuned from here on. Every note this block has recorded so far \[75\]
    #[cold]
    #[inline(never)]
    fn start_tuning(&mut self) {
        if !self.tuned {
            self.tuned = true;
            let notes = &self.notes;
            self.note_tune.clear();
            self.note_tune.extend(notes.iter().map(|&w| NoteTune { key: note_unpack(w).1, cents: 0.0 }));
        }
    }

    /// How a note-on is tuned: the key to build it on and the cents from there, and \[76\]
    #[inline(never)]
    fn tune_note(&self, ch: u8, key: u8) -> (NoteTune, bool) {
        let scale = mts::cents(self.tune[ch as usize * 12 + key as usize % 12]);
        if self.drum_map[ch as usize] != 0 {
            return (NoteTune { key, cents: scale }, false);
        }
        let silent = self.key_static.as_ref().is_some_and(|t| t.is_silent(key));
        let sel = self.tuning_sel[ch as usize];
        let offset = match self.mts_tables.iter().find(|t| (t.bank, t.program) == ((sel >> 7) as u8, (sel & 0x7F) as u8)) {
            Some(t) => t.offsets[key as usize],
            None => self.key_static.as_ref().map_or(0, |t| t.offsets[key as usize]),
        };
        if offset == 0 {
            return (NoteTune { key, cents: scale }, silent);
        }
        let (nearest, cents) = crate::tuning::lookup(key, offset);
        (NoteTune { key: nearest, cents: cents + scale }, silent)
    }

    /// One key of a tuning program set by a message (`Event::KeyTune`): the table \[77\]
    #[cold]
    #[inline(never)]
    fn set_key_tune(&mut self, bank: u8, program: u8, key: u8, units: i32) {
        if !self.cfg.mts_notes {
            // [78]
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!(
                    "this file retunes keys with MIDI Tuning Standard single-note or bulk messages, \
                     which BASSMIDI ignores and this render does too"
                );
            }
            return;
        }
        let at = match self.mts_tables.iter().position(|t| (t.bank, t.program) == (bank, program)) {
            Some(i) => i,
            None => {
                let base = self.key_static.as_ref().map_or([0; 128], |t| t.offsets);
                self.mts_tables.push(MtsTable { bank, program, offsets: Box::new(base) });
                self.mts_tables.len() - 1
            }
        };
        self.mts_tables[at].offsets[key as usize & 0x7F] = units;
        self.start_tuning();
    }

    /// Put the three lane tunings of the 31-EDO template on the lane \[79\]
    fn tune_lanes(&mut self) {
        for msg in crate::edo31::TUNING {
            let m = mts::parse(&msg).expect("the built-in tuning is a scale/octave message");
            for port in 0..crate::midi::PORTS {
                for c in (0..16u8).filter(|&c| m.channels >> c & 1 != 0) {
                    for (pc, &u) in m.units.iter().enumerate() {
                        self.set_tune(port * 16 + c, pc as u8, u);
                    }
                }
            }
        }
    }

    pub fn track_count(&self) -> u16 {
        self.stream.track_count
    }

    /// Track data decoded so far and the file's total, in bytes. See \[80\]
    pub fn input_bytes(&self) -> (u64, u64) {
        (self.stream.bytes_read(), self.stream.bytes_total())
    }

    /// Measure each block's queued and admitted energy into \[81\]
    pub fn measure_block_energy(&mut self, on: bool) {
        self.block_energy = on;
    }

    fn refresh_preset(&mut self, ch: usize, program: u8) {
        self.program[ch] = program;
        // [82]
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

    /// Render one block. Returns false once the file and its tail are done. \[83\]
    fn prepare_block(&mut self) -> Result<()> {
        let prof = self.cfg.profile;
        let t_block = std::time::Instant::now();
        let block_frames = self.cfg.block_frames as u64;
        let block_start = self.block_index * block_frames;
        let block_end = block_start + block_frames;

        self.gates.begin_block();
        self.chan.begin_block();
        // [84]
        self.variant_used = 1;
        for v in self.cur_variant {
            self.variant_used |= 1u64 << v;
        }
        self.spawn_len = 0;
        self.notes.clear();
        self.note_ticks.clear();
        self.note_tune.clear();
        self.note_ordinal.clear();
        self.cands.clear();
        self.cand_seen = 0;
        self.unit_seen = 0;
        self.cand_stride = 1;
        self.deferred_now.clear();
        self.block_first_id = self.next_note_id;
        self.spawn_variants = false;

        // [85]
        let mut i = 0;
        while i < self.deferred.len() {
            if self.deferred[i].0 < block_end {
                let (frame, mut cmd, glide) = self.deferred.swap_remove(i);
                cmd.start_rel = frame.saturating_sub(block_start) as u32;
                let (semitones, cc5) = glide_unpack(glide);
                let (fl, sl) = self.glide.spawn(semitones, cc5, cmd.start_rel);
                cmd.flags |= fl;
                cmd.note_id_hi |= sl;
                if cmd.variant != 0 {
                    self.variant_used |= 1u64 << cmd.variant;
                    self.spawn_variants = true;
                }
                if self.cands.len() >= self.cand_cap {
                    self.thin_cands();
                }
                // [86]
                let index = self.next_cand_index();
                if self.offer_unit() {
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
                        if self.hold_frame.is_some_and(|f| f >= block_end) {
                            break;
                        }
                        self.stream_done = true;
                        break;
                    }
                },
            };

            if let Event::Tempo(us) = ev {
                // [87]
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
            self.handle_event(ev, tick, rel, block_start);
        }

        if prof {
            self.stats.us_drain += t_block.elapsed().as_micros() as u64;
        }

        // Once the file runs out, release everything so looping voices stop.
        if self.stream_done && !self.end_sent {
            // [88]
            for ch in 0..self.gates.channels() {
                // [89]
                self.gates.all_sound_off(ch as u8, 0);
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
        self.submit_block(backend)?;
        self.prepare_ahead()?;
        self.finish_block(backend, out)
    }

    /// The first of `next_block`'s three steps, split out so that a batch can \[90\]
    pub fn submit_block(&mut self, backend: &mut dyn Backend) -> Result<()> {
        if self.in_flight.is_some() {
            bail!("submit_block while a block is still in flight");
        }
        let prof = self.cfg.profile;
        let t_block = std::time::Instant::now();
        let block_frames = self.cfg.block_frames as u64;

        // [91]
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
        let live = backend.stats().active_voices as u32;
        // [92]
        let silent = self.cfg.skip_silence && live == 0 && self.cand_seen == 0;
        if !silent {
            backend.set_gates(&self.gates.off_meta, &self.gates.off_runs)?;
            backend.set_channels(
                &self.chan.rows,
                self.chan.bend_active(),
                self.chan.gain_active(),
                // [93]
                self.chan.variant_active() || self.spawn_variants,
                self.chan.cut_active(),
            )?;
        }
        // [94]
        let t_admit = std::time::Instant::now();
        // [95]
        let want = self.cands.len();
        let take = self.cfg.admit_take(live, want as u32) as usize;
        self.stats.dropped += self.cand_seen - take as u64;
        let stolen_before = backend.stats().stolen;
        self.stats.last_live = live;
        self.stats.last_want = self.cand_seen;
        self.stats.last_take = take as u32;

        if take < want && take > 0 {
            match self.cfg.admit_rule {
                // [96]
                AdmitRule::Even => {
                    for i in 0..take {
                        self.cands[i] = self.cands[crate::voice::spawn_pick(i, want, take)];
                    }
                }
                AdmitRule::Loudest => self.rank_candidates(want, take),
            }
        }
        // [97]
        if self.block_energy {
            self.stats.last_want_energy = self.cands.iter().map(|c| (c.key >> 48) & 0x7FFF).sum();
            self.stats.last_take_energy =
                self.cands[..take.min(self.cands.len())].iter().map(|c| (c.key >> 48) & 0x7FFF).sum();
        }
        // [98]
        let block_start = (self.block_index - 1) * block_frames;
        self.materialise(take, block_start);
        if prof {
            self.stats.us_admit += t_admit.elapsed().as_micros() as u64;
        }
        debug_assert!(!silent || self.spawn_len == 0, "a silent block admitted a voice");
        if !silent {
            backend.set_glide(self.glide_until > block_start)?;
        }

        let t_spawn = std::time::Instant::now();
        if !silent {
            backend.spawn(&self.spawn_buf[..self.spawn_len])?;
        }
        if prof {
            self.stats.us_spawn += t_spawn.elapsed().as_micros() as u64;
        }
        self.stats.last_stolen = backend.stats().stolen - stolen_before;
        // [99]
        self.stats.voices_spawned += self.cand_seen;
        if !silent {
            backend.submit()?;
        }

        // [100]
        self.in_flight = Some(InFlight {
            t_block,
            silent,
            was_done: self.stream_done,
            had_deferred: !self.deferred.is_empty(),
        });
        Ok(())
    }

    /// The second of `next_block`'s three steps: build the next block's \[101\]
    pub fn prepare_ahead(&mut self) -> Result<()> {
        self.prepare_block()
    }

    /// The last of `next_block`'s three steps: take the block back from the \[102\]
    pub fn finish_block(&mut self, backend: &mut dyn Backend, out: &mut [f32]) -> Result<bool> {
        if out.len() != self.cfg.block_samples() {
            bail!(
                "output block is {} samples, expected {}",
                out.len(),
                self.cfg.block_samples()
            );
        }
        let InFlight { t_block, silent, was_done, had_deferred } =
            self.in_flight.take().context("finish_block with no block submitted")?;
        let prof = self.cfg.profile;
        let block_frames = self.cfg.block_frames as u64;

        let t_render = std::time::Instant::now();
        if silent {
            out.fill(0.0);
            backend.skip_block()?;
            self.stats.silent_blocks += 1;
        } else {
            backend.finish(out)?;
        }
        let waited = t_render.elapsed().as_micros() as u64;
        self.stats.last_wait_us = waited;
        if prof {
            self.stats.us_render += waited;
        }

        self.stats.clipped += self.output.process(out, self.stats.blocks)?;

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
        // The top of the high word is the glide's; see `voice::NOTE_HI_MASK`.
        debug_assert!(id >> 48 == 0, "note id {id} is past 48 bits");
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
            // [103]
            variant,
            region: v.region,
            gate_slot,
            ordinal,
            start_rel,
            note_id_lo: id as u32,
            note_id_hi: (id >> 32) as u32,
            gain_l: v.gain_l,
            gain_r: v.gain_r,
            // [104]
            row_bias,
            // Filled in by the caller when analytic phase is on.
            rotation: crate::phase::Coefficients::default(),
        }
    }

    /// Move the `take` best candidates to the front, ranked *within* time \[105\]
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
            // [106]

            // [107]
            for j in 0..q {
                cands.swap(olo + j, lo + j);
            }
        }
    }


    /// Number the next admission candidate. Every layer that reaches \[108\]
    #[inline]
    fn next_cand_index(&mut self) -> u64 {
        let index = self.cand_seen;
        self.cand_seen += 1;
        index
    }

    /// Number the next unit -- a note-on, or one deferred voice -- and say \[109\]
    #[inline]
    fn offer_unit(&mut self) -> bool {
        let unit = self.unit_seen;
        self.unit_seen += 1;
        unit & (self.cand_stride - 1) == 0
    }

    /// Halve this block's candidates once the list reaches `cand_cap`: keep \[110\]
    fn thin_cands(&mut self) {
        let mut kept = 0;
        let mut notes = 0;
        let mut unit = 0usize;
        let len = self.cands.len();
        let mut r = 0;
        while r < len {
            let first = self.cands[r];
            // [111]
            let mut end = r + 1;
            if first.note != DEFERRED {
                while end < len && self.cands[end].note == first.note {
                    end += 1;
                }
            }
            if unit.is_multiple_of(2) {
                // [112]
                if first.note != DEFERRED {
                    let n = first.note as usize;
                    self.notes[notes] = self.notes[n];
                    self.note_ordinal[notes] = self.note_ordinal[n];
                    if self.cfg.phase.active() {
                        self.note_ticks[notes] = self.note_ticks[n];
                    }
                    if self.tuned {
                        self.note_tune[notes] = self.note_tune[n];
                    }
                    notes += 1;
                }
                for j in r..end {
                    let mut c = self.cands[j];
                    if c.note != DEFERRED {
                        c.note = notes as u32 - 1;
                    }
                    self.cands[kept] = c;
                    kept += 1;
                }
            }
            unit += 1;
            r = end;
        }
        self.cands.truncate(kept);
        self.notes.truncate(notes);
        self.note_ordinal.truncate(notes);
        self.note_ticks.truncate(notes);
        self.note_tune.truncate(notes);
        self.cand_stride *= 2;
    }

    /// Build the voices for the first `take` candidates, appending them to \[113\]
    fn materialise(&mut self, take: usize, block_start: u64) {
        let cands = std::mem::take(&mut self.cands);
        let admitted = &cands[..take.min(cands.len())];
        let b = Builder {
            bank: &self.bank,
            cfg: &self.cfg,
            phase: self.cfg.phase.active().then_some(&*self.phase_bank),
            glide: &self.glide,
            notes: &self.notes,
            note_ordinal: &self.note_ordinal,
            note_ticks: &self.note_ticks,
            note_tune: &self.note_tune,
            deferred_now: &self.deferred_now,
            first_id: self.block_first_id,
            block_start,
        };
        let n = admitted.len();
        let base = self.spawn_len;
        // [114]
        if self.spawn_buf.len() < base + n {
            self.spawn_buf.resize(base + n, SpawnCmd::default());
        }
        let out = &mut self.spawn_buf[base..base + n];
        // [115]
        let fill = |out: &mut [SpawnCmd], cs: &[Cand]| -> (usize, u64) {
            let mut memo = None;
            let (mut k, mut until) = (0, 0);
            for c in cs {
                if let Some(cmd) = b.build(c, &mut memo) {
                    until = until.max(b.glide_end(&cmd));
                    out[k] = cmd;
                    k += 1;
                }
            }
            (k, until)
        };
        let threads = (n / MATERIALISE_MIN).clamp(1, self.cfg.materialise_threads.max(1));
        let (built, glide_until) = if threads == 1 {
            fill(out, admitted)
        } else {
            // [116]
            let mut done = {
                let pieces = std::sync::Mutex::new(
                    out.chunks_mut(MATERIALISE_CHUNK).zip(admitted.chunks(MATERIALISE_CHUNK)).enumerate(),
                );
                let run = || {
                    let mut done = Vec::new();
                    loop {
                        let next = pieces.lock().unwrap().next();
                        let Some((i, (o, cs))) = next else { break };
                        let (k, until) = fill(o, cs);
                        done.push((i, k, until));
                    }
                    done
                };
                std::thread::scope(|s| {
                    let handles: Vec<_> = (1..threads).map(|_| s.spawn(run)).collect();
                    let mut done = run();
                    for h in handles {
                        done.extend(h.join().expect("materialise thread"));
                    }
                    done
                })
            };
            done.sort_unstable_by_key(|d| d.0);
            // [117]
            let (mut end, mut until) = (0, 0);
            for (i, k, u) in done {
                let from = i * MATERIALISE_CHUNK;
                if from != end {
                    out.copy_within(from..from + k, end);
                }
                end += k;
                until = until.max(u);
            }
            (end, until)
        };
        self.spawn_len = base + built;
        self.glide_until = self.glide_until.max(glide_until);
        self.cands = cands;
    }

    /// Grow the published tables to cover channel `ch`, a port at a time. The \[118\]
    #[inline]
    fn cover(&mut self, ch: u8) {
        let ch = ch as usize;
        if ch >= self.gates.channels() {
            let channels = (ch / 16 + 1) * 16;
            self.gates.grow(channels);
            self.chan.grow(channels);
        }
    }

    /// `tick` is the event's own MIDI tick, before it rounds to a frame: the \[119\]
    fn handle_event(&mut self, ev: Event, tick: u64, rel: u32, block_start: u64) {
        if !self.cfg.edo31 {
            return self.handle_one(ev, tick, rel, block_start);
        }
        let on = |ch| match edo31::triplet(ch) {
            Some(t) => t.map(Some),
            None => [Some(ch), None, None],
        };
        match ev {
            Event::NoteOn { ch, key, vel } => {
                let (ch, key) = edo31::map_note(ch, key);
                self.handle_one(Event::NoteOn { ch, key, vel }, tick, rel, block_start);
            }
            Event::NoteOff { ch, key } => {
                let (ch, key) = edo31::map_note(ch, key);
                self.handle_one(Event::NoteOff { ch, key }, tick, rel, block_start);
            }
            Event::Cc { ch, num, val } => {
                for ch in on(ch).into_iter().flatten() {
                    self.handle_one(Event::Cc { ch, num, val }, tick, rel, block_start);
                }
            }
            Event::Program { ch, val } => {
                for ch in on(ch).into_iter().flatten() {
                    self.handle_one(Event::Program { ch, val }, tick, rel, block_start);
                }
            }
            Event::PitchBend { ch, val } => {
                for ch in on(ch).into_iter().flatten() {
                    self.handle_one(Event::PitchBend { ch, val }, tick, rel, block_start);
                }
            }
            // Not channel messages, or not the template's to move.
            _ => self.handle_one(ev, tick, rel, block_start),
        }
    }

    fn handle_one(&mut self, ev: Event, tick: u64, rel: u32, block_start: u64) {
        match ev {
            Event::NoteOn { ch, .. }
            | Event::NoteOff { ch, .. }
            | Event::Cc { ch, .. }
            | Event::PitchBend { ch, .. } => self.cover(ch),
            _ => {}
        }
        match ev {
            Event::NoteOn { ch, key, vel } => {
                // [120]
                if let Some(q) = &mut self.quiet {
                    let skip = vel < self.cfg.min_velocity;
                    q.push(GateTable::slot(ch, key), skip);
                    if skip {
                        self.stats.notes_skipped += 1;
                        return;
                    }
                }
                let ordinal = self.gates.note_on(ch, key, rel);
                self.stats.notes += 1;
                let variant = self.cur_variant[ch as usize];
                let (nt, silent) = if self.tuned {
                    self.tune_note(ch, key)
                } else {
                    (NoteTune { key, cents: 0.0 }, false)
                };

                // [121]
                let c = ch as usize;
                let from = if self.porta_note[c] != 0 {
                    self.porta_note[c]
                } else if self.porta_on[c] {
                    self.porta_last[c]
                } else {
                    NO_KEY
                };
                self.porta_note[c] = 0;
                self.porta_last[c] = key;
                let glide = if from == NO_KEY {
                    0
                } else {
                    glide_pack(from as i32 - key as i32, self.porta_time[c])
                };

                // [122]
                let mut prev = std::mem::take(&mut self.preview_buf);
                prev.clear();
                // [123]
                if !silent {
                    self.bank.preview_note_on(
                        self.preset[ch as usize],
                        nt.key,
                        vel,
                        self.next_note_id,
                        self.cfg.max_layers as usize,
                        &mut prev,
                    );
                }

                // [124]
                if self.cands.len() + prev.len() > self.cand_cap {
                    self.thin_cands();
                }
                let note = self.notes.len() as u32;
                let mut recorded = false;
                // [125]
                let mut kept = None;
                // Every layer of one note-on shares one angle, parked or not.
                let mut deferred_angle = None;
                for p in prev.iter() {
                    // [126]
                    let id = self.next_note_id;
                    self.next_note_id += 1;

                    if p.delay_frames != 0 {
                        let start = block_start + rel as u64 + p.delay_frames as u64;
                        if start >= block_start + self.cfg.block_frames as u64 {
                            // [127]
                            if let Some(v) =
                                self.bank.build_layer(p.region, nt.key, vel, &self.cfg, nt.cents)
                            {
                                // [128]
                                let mut cmd = Self::make_cmd(
                                    &v,
                                    variant,
                                    GateTable::slot(ch, key) as u32,
                                    ordinal,
                                    rel,
                                    id,
                                    0,
                                );
                                if self.cfg.phase.active() {
                                    let angle = *deferred_angle.get_or_insert_with(||
                                        self.phase_bank.angle(tick, ch, key));
                                    cmd.rotation = self.phase_bank.coefficients(v.region, angle);
                                }
                                self.deferred.push((start, cmd, glide));
                            }
                            continue;
                        }
                    }

                    // [129]
                    if variant != 0 {
                        self.variant_used |= 1u64 << variant;
                        self.spawn_variants = true;
                    }
                    let index = self.next_cand_index();
                    if !*kept.get_or_insert_with(|| self.offer_unit()) {
                        continue;
                    }

                    if !recorded {
                        self.notes.push(note_pack(
                            ch,
                            key,
                            vel,
                            variant,
                            rel,
                            self.chan.row_bias(ch, rel),
                            glide,
                        ));
                        self.note_ordinal.push(ordinal);
                        if self.cfg.phase.active() {
                            self.note_ticks.push(tick);
                        }
                        if self.tuned {
                            self.note_tune.push(nt);
                        }
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
                if let Some(q) = &mut self.quiet {
                    if q.pop(GateTable::slot(ch, key)) {
                        return;
                    }
                }
                self.gates.note_off(ch, key, rel);
            }
            Event::Cc { ch, num, val } => {
                let c = ch as usize;
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
                    // [130]
                    6 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = val as f64;
                        self.refresh_bend(c, rel);
                    }
                    38 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = self.bend_range[c].trunc() + val as f64 / 100.0;
                        self.refresh_bend(c, rel);
                    }
                    // [131]
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
                    // [132]
                    6 if self.rpn_sel[c] == 2 => {
                        self.coarse_tune[c] = val as f64 - 64.0;
                        self.refresh_bend(c, rel);
                    }
                    // [133]
                    6 if self.rpn_sel[c] == 3 && self.cfg.mts_notes => {
                        self.tuning_sel[c] = (self.tuning_sel[c] & 0x3F80) | val as u16;
                    }
                    6 if self.rpn_sel[c] == 4 && self.cfg.mts_notes => {
                        self.tuning_sel[c] = (self.tuning_sel[c] & 0x7F) | ((val as u16) << 7);
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
                    // [134]
                    5 => self.porta_time[c] = val,
                    65 => self.porta_on[c] = val >= 64,
                    84 => self.porta_note[c] = val,
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
                        // [135]
                        self.gates.all_sound_off(ch, rel);
                        self.chan.set_sound_off(ch, rel, self.next_note_id);
                    }
                    121 => self.reset_controllers(ch, rel),
                    // [136]
                    123..=127 => self.gates.all_notes_off(ch, rel),
                    _ => {}
                }
            }
            Event::Program { ch, val } => {
                self.refresh_preset(ch as usize, val);
            }
            Event::DrumPart { ch, map } => {
                let c = ch as usize;
                if self.drum_map[c] != map {
                    self.drum_map[c] = map;
                    self.refresh_preset(c, self.program[c]);
                }
            }
            // Every port, whichever track sent it, as BASSMIDI resets.
            Event::ResetParts => {
                for (c, want) in DEFAULT_DRUM_MAP.iter().enumerate() {
                    if self.drum_map[c] != *want {
                        self.drum_map[c] = *want;
                        self.refresh_preset(c, self.program[c]);
                    }
                }
                // [137]
                self.tune.fill(0);
                if self.cfg.edo31 {
                    self.tune_lanes();
                }
                // [138]
                self.mts_tables.clear();
                self.tuning_sel.fill(0);
            }
            Event::Tune { ch, pc, units } => self.set_tune(ch, pc, units),
            Event::KeyTune { bank, prog, key, units } => self.set_key_tune(bank, prog, key, units),
            Event::PitchBend { ch, val } => {
                self.bend_val[ch as usize] = val;
                self.refresh_bend(ch as usize, rel);
            }
            Event::Tempo(_) | Event::Other => {}
        }
    }

    /// Take one of CC71-CC75 and move the channel onto whichever copy of the \[139\]
    fn set_sound_cc(&mut self, ch: u8, which: usize, val: u8, rel: u32) {
        let c = ch as usize;
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
            // [140]
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

    /// Queue the build of variant `i` from channel `c`'s current controllers. \[141\]
    fn build_variant_at(&mut self, i: u32, c: usize) {
        if self.variant_ccs.len() <= i as usize {
            self.variant_ccs.resize(i as usize + 1, [64; 5]);
        }
        self.variant_ccs[i as usize] = self.cc_sound[c];
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

    /// CC121, Reset All Controllers. \[142\]
    fn reset_controllers(&mut self, ch: u8, rel: u32) {
        let c = ch as usize;
        self.porta_on[c] = false;
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

    /// Lay the vibrato and tremolo LFOs over the controller rows. \[143\]
    fn apply_modulation(&mut self) {
        let tiles = self.chan.tiles;
        let tile_seconds = self.cfg.gate_frames as f64 / self.cfg.sample_rate as f64;
        // [144]
        for c in 0..CHANNELS {
            let vib = (self.cc_mod[c] as f64 * 128.0 + self.lsb_mod[c] as f64) / 16383.0
                * (self.cc_vib_depth[c] as f64 / 64.0)
                * 50.0;
            let trem = self.cc_tremolo[c] as f64 / 127.0 * 0.25;
            let rate = 5.0 * (2.0f64).powf((self.cc_vib_rate[c] as f64 - 64.0) / 32.0);
            if vib <= 0.0 && trem <= 0.0 {
                // [145]
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

    /// Freeze this channel's bend into the factor the backends multiply by. \[146\]
    fn refresh_bend(&mut self, c: usize, rel: u32) {
        // [147]
        let semitones = (self.bend_val[c] as f64 / 8192.0) * self.bend_range[c]
            + self.coarse_tune[c]
            + self.fine_tune[c] / 100.0;
        self.chan.set_bend(c as u8, bend_factor(semitones), rel);
    }

    /// Fold CC7, CC11 and CC10 into the pair of gains a voice multiplies by. \[148\]
    fn refresh_gain(&mut self, c: usize, rel: u32) {
        // [149]
        let fine = |msb: u8, lsb: u8| {
            if lsb == 0 {
                msb as f32 / 127.0
            } else {
                (msb as f32 * 128.0 + lsb as f32) / 16383.0
            }
        };
        // [150]
        let v = fine(self.cc_volume[c], self.lsb_volume[c])
            / (crate::bank::POWER_ON_VOLUME as f32 / 127.0);
        let e = fine(self.cc_expression[c], self.lsb_expression[c]);
        // [151]
        let soft = 1.0 - 0.5 * (self.cc_soft[c] as f32 / 127.0);
        let amp = (v * v) * (e * e) * soft;
        let theta = fine(self.cc_pan[c], self.lsb_pan[c]) * std::f32::consts::FRAC_PI_2;
        // [152]
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

    // ---- stopping and going on ----------------------------------------------

    /// Like `next_block`, with one difference: after the block is submitted, \[153\]
    pub fn next_block_or_hold(
        &mut self,
        backend: &mut dyn Backend,
        out: &mut [f32],
        hold: impl FnOnce() -> bool,
    ) -> Result<(bool, bool)> {
        if out.len() != self.cfg.block_samples() {
            bail!("output block is {} samples, expected {}", out.len(), self.cfg.block_samples());
        }
        self.submit_block(backend)?;
        let held = hold();
        if held {
            // The next call to `submit_block` prepares it, as a fresh driver's does.
            self.primed = false;
        } else {
            self.prepare_ahead()?;
        }
        let more = self.finish_block(backend, out)?;
        Ok((more, held))
    }

    /// Whether the driver is at a block boundary: no block on the device and \[154\]
    pub fn at_boundary(&self) -> bool {
        self.in_flight.is_none() && !self.primed
    }

    /// Everything this render has come to that is not rebuilt from the \[155\]
    pub fn save_state(&self, e: &mut Enc) -> Result<()> {
        if !self.at_boundary() {
            bail!("a render can be saved only between blocks, with none prepared");
        }
        let Driver {
            // Rebuilt from the configuration and the soundfont.
            phase_bank: _,
            cfg: _,
            bank: _,
            key_static: _,
            glide: _,
            cand_cap: _,
            hold_frame: _,
            block_energy: _,
            // [156]
            note_ticks: _,
            note_tune: _,
            variant_used: _,
            pending_variants: _,
            spawn_variants: _,
            in_flight: _,
            primed: _,
            spawn_buf: _,
            spawn_len: _,
            notes: _,
            note_ordinal: _,
            cands: _,
            cand_seen: _,
            unit_seen: _,
            cand_stride: _,
            deferred_now: _,
            preview_buf: _,
            block_first_id: _,
            // Saved.
            stream,
            clock,
            gates,
            chan,
            output,
            bank_msb,
            bank_lsb,
            program,
            drum_map,
            preset,
            bend_val,
            bend_range,
            coarse_tune,
            fine_tune,
            fine_tune_raw,
            tune,
            tuned,
            mts_tables,
            tuning_sel,
            rpn_sel,
            cc_volume,
            cc_expression,
            cc_pan,
            cc_sound,
            cc_mod,
            cc_vib_rate,
            cc_vib_depth,
            cc_tremolo,
            cc_soft,
            lsb_mod,
            lsb_volume,
            lsb_pan,
            lsb_expression,
            lfo_phase,
            porta_on,
            porta_time,
            porta_note,
            porta_last,
            glide_until,
            variants,
            variant_ccs,
            seen_states,
            cur_variant,
            variant_seen,
            variant_clock,
            next_note_id,
            block_index,
            pending,
            stream_done,
            end_sent,
            tail_blocks_left,
            quiet,
            deferred,
            stats,
        } = self;
        stream.save_state(e);
        clock.save_state(e);
        gates.save_state(e);
        chan.save_state(e);
        output.save_state(e);

        e.u8s(bank_msb);
        e.u8s(bank_lsb);
        e.u8s(program);
        e.u8s(drum_map);
        e.u32s(preset);
        e.i16s(bend_val);
        e.f64s(bend_range);
        e.f64s(coarse_tune);
        e.f64s(fine_tune);
        e.u16s(fine_tune_raw);
        e.i32s(tune);
        e.bool(*tuned);
        // The tuning programs a file has set, and the one each channel plays on.
        e.len_of(mts_tables.len());
        for t in mts_tables {
            e.u8(t.bank);
            e.u8(t.program);
            e.i32s(&t.offsets[..]);
        }
        e.u16s(tuning_sel);
        e.u16s(rpn_sel);
        e.u8s(cc_volume);
        e.u8s(cc_expression);
        e.u8s(cc_pan);
        for per in cc_sound {
            e.raw(per);
        }
        e.u8s(cc_mod);
        e.u8s(cc_vib_rate);
        e.u8s(cc_vib_depth);
        e.u8s(cc_tremolo);
        e.u8s(cc_soft);
        e.u8s(lsb_mod);
        e.u8s(lsb_volume);
        e.u8s(lsb_pan);
        e.u8s(lsb_expression);
        e.f64s(lfo_phase);
        e.bools(porta_on);
        e.u8s(porta_time);
        e.u8s(porta_note);
        e.u8s(porta_last);
        e.u64(*glide_until);

        e.len_of(variants.len());
        for v in variants {
            e.raw(v);
        }
        e.len_of(variant_ccs.len());
        for v in variant_ccs {
            e.raw(v);
        }
        e.len_of(seen_states.len());
        for v in seen_states {
            e.raw(v);
        }
        e.u32s(cur_variant);
        e.u64s(variant_seen);
        e.u64(*variant_clock);

        e.u64(*next_note_id);
        e.u64(*block_index);
        match pending {
            Some((tick, ev)) => {
                e.bool(true);
                e.u64(*tick);
                ev.save(e);
            }
            None => e.bool(false),
        }
        e.bool(*stream_done);
        e.bool(*end_sent);
        e.u64(*tail_blocks_left);

        match quiet {
            Some(q) => {
                e.bool(true);
                q.save(e);
            }
            None => e.bool(false),
        }
        e.len_of(deferred.len());
        for (frame, cmd, glide) in deferred {
            e.u64(*frame);
            e.pod(std::slice::from_ref(cmd));
            e.u16(*glide);
        }
        stats.save(e);
        Ok(())
    }

    /// Go on from a saved state: the other half of `save_state`, into a driver \[157\]
    pub fn load_state(&mut self, d: &mut Dec) -> Result<()> {
        if !self.at_boundary() || self.block_index != 0 {
            bail!("a saved state can be loaded only into a render that has not begun");
        }
        self.stream.load_state(d).context("the MIDI stream")?;
        self.clock.load_state(d).context("the tempo clock")?;
        self.gates.load_state(d).context("the gate table")?;
        self.chan.load_state(d).context("the channel table")?;
        self.output.load_state(d).context("the output stage")?;

        d.fill_u8s(&mut self.bank_msb, "channels")?;
        d.fill_u8s(&mut self.bank_lsb, "channels")?;
        d.fill_u8s(&mut self.program, "channels")?;
        d.fill_u8s(&mut self.drum_map, "channels")?;
        d.fill_u32s(&mut self.preset, "channels")?;
        d.fill_i16s(&mut self.bend_val, "channels")?;
        d.fill_f64s(&mut self.bend_range, "channels")?;
        d.fill_f64s(&mut self.coarse_tune, "channels")?;
        d.fill_f64s(&mut self.fine_tune, "channels")?;
        d.fill_u16s(&mut self.fine_tune_raw, "channels")?;
        d.fill_i32s(&mut self.tune, "scale/octave tunings")?;
        self.tuned = d.bool()?;
        let tables = d.len_of(2 + 4 * 128)?;
        self.mts_tables.clear();
        for _ in 0..tables {
            let (bank, program) = (d.u8()?, d.u8()?);
            let mut offsets = Box::new([0i32; 128]);
            d.fill_i32s(&mut offsets[..], "tuning program offsets")?;
            self.mts_tables.push(MtsTable { bank, program, offsets });
        }
        d.fill_u16s(&mut self.tuning_sel, "channels")?;
        d.fill_u16s(&mut self.rpn_sel, "channels")?;
        d.fill_u8s(&mut self.cc_volume, "channels")?;
        d.fill_u8s(&mut self.cc_expression, "channels")?;
        d.fill_u8s(&mut self.cc_pan, "channels")?;
        for per in &mut self.cc_sound {
            per.copy_from_slice(d.raw(5)?);
        }
        d.fill_u8s(&mut self.cc_mod, "channels")?;
        d.fill_u8s(&mut self.cc_vib_rate, "channels")?;
        d.fill_u8s(&mut self.cc_vib_depth, "channels")?;
        d.fill_u8s(&mut self.cc_tremolo, "channels")?;
        d.fill_u8s(&mut self.cc_soft, "channels")?;
        d.fill_u8s(&mut self.lsb_mod, "channels")?;
        d.fill_u8s(&mut self.lsb_volume, "channels")?;
        d.fill_u8s(&mut self.lsb_pan, "channels")?;
        d.fill_u8s(&mut self.lsb_expression, "channels")?;
        d.fill_f64s(&mut self.lfo_phase, "channels")?;
        d.fill_bools(&mut self.porta_on, "channels")?;
        d.fill_u8s(&mut self.porta_time, "channels")?;
        d.fill_u8s(&mut self.porta_note, "channels")?;
        d.fill_u8s(&mut self.porta_last, "channels")?;
        self.glide_until = d.u64()?;

        let five = |d: &mut Dec| -> Result<Vec<[u8; 5]>> {
            let n = d.len_of(5)?;
            (0..n).map(|_| Ok(d.raw(5)?.try_into().expect("five bytes"))).collect()
        };
        self.variants = five(d)?;
        self.variant_ccs = five(d)?;
        self.seen_states = five(d)?;
        if self.variants.is_empty()
            || self.variant_ccs.len() != self.variants.len()
            || self.variants.len() > 64
            || self.seen_states.is_empty()
        {
            bail!("the saved sound-controller variants are not consistent");
        }
        d.fill_u32s(&mut self.cur_variant, "channels")?;
        if self.cur_variant.iter().any(|&v| v as usize >= self.variants.len()) {
            bail!("a saved channel names a sound-controller variant there is not");
        }
        self.variant_seen = d.u64s()?;
        if self.variant_seen.len() != self.variants.len() {
            bail!("the saved sound-controller variants are not consistent");
        }
        self.variant_clock = d.u64()?;

        self.next_note_id = d.u64()?;
        self.block_index = d.u64()?;
        self.pending = if d.bool()? {
            let tick = d.u64()?;
            Some((tick, Event::load(d)?))
        } else {
            None
        };
        self.stream_done = d.bool()?;
        self.end_sent = d.bool()?;
        self.tail_blocks_left = d.u64()?;

        match (&mut self.quiet, d.bool()?) {
            (Some(q), true) => q.load(d)?,
            (None, false) => {}
            _ => bail!("the saved state and this render disagree about --min-velocity"),
        }
        let n = d.len_of(8 + std::mem::size_of::<SpawnCmd>() + 2)?;
        self.deferred = Vec::with_capacity(n);
        for _ in 0..n {
            let frame = d.u64()?;
            let cmd = d.pod::<SpawnCmd>()?;
            let [cmd] = <[SpawnCmd; 1]>::try_from(cmd).map_err(|_| anyhow::anyhow!("a saved voice is the wrong size"))?;
            self.deferred.push((frame, cmd, d.u16()?));
        }
        self.stats.load(d)?;
        // [158]
        self.primed = false;
        Ok(())
    }

    /// Hand the backend the params tables the sound controllers built before \[159\]
    pub fn reinstall_variants(&self, backend: &mut dyn Backend) -> Result<()> {
        for (i, c) in self.variant_ccs.iter().enumerate().skip(1) {
            let m = ParamMod::from_controllers(c[0], c[1], c[2], c[3], c[4]);
            let data = self.bank.build_variant(&self.cfg, &m);
            let menv = self.bank.build_menv_variant(&self.cfg, &m);
            backend.set_params_variant(i as u32, &data, &menv)?;
        }
        Ok(())
    }

    /// The largest magnitude written to the output so far, after `--volume` \[160\]
    pub fn output_peak(&self) -> f32 {
        self.output.peak()
    }

}
