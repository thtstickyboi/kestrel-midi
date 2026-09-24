// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Voice pool layout, spawn commands, and the note-off gate table. \[1\]

use crate::config::Config;
use crate::fixed::BEND_ONE;

// Envelope stages. Part of the host/device contract, so do not renumber them.
pub const ENV_ATTACK: u32 = 0;
pub const ENV_DECAY: u32 = 1;
pub const ENV_SUSTAIN: u32 = 2;
pub const ENV_RELEASE: u32 = 3;
pub const ENV_DEAD: u32 = 4;

/// Channels the gate table and the controller rows start with: one port. \[2\]
pub const BASE_CHANNELS: usize = 16;

/// Bits of a gate slot, `channel * 128 + key`, at the most channels a file can \[3\]
pub const SLOT_BITS: u32 = 15;
pub const SLOT_MASK: u32 = (1 << SLOT_BITS) - 1;
const _: () = assert!(crate::midi::CHANNELS * 128 == 1 << SLOT_BITS);

/// The part of a voice's note-id high word that is the id. \[4\]
pub const NOTE_HI_MASK: u32 = 0xFFFF;

/// Everything needed to start one voice, in the exact order the spawn shader \[5\]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpawnCmd {
    pub phase_lo: u32,
    pub phase_hi: u32,
    pub step_lo: u32,
    pub step_hi: u32,
    pub smp_base: u32,
    pub smp_len: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    pub flags: u32,
    pub params: u32,
    /// The params variant the channel was on at this voice's own note-on. \[6\]
    pub variant: u32,
    pub region: u32,
    /// `channel * 128 + key`, the note-off gate this voice listens to, with \[7\]
    pub gate_slot: u32,
    /// Which note-on of that slot this voice belongs to, counting from 1. \[8\]
    pub ordinal: u32,
    /// Frames into the block before the voice starts. Gives sample-accurate \[9\]
    pub start_rel: u32,
    pub note_id_lo: u32,
    /// The note id's high 16 bits -- ids are 48 bits, see `NOTE_HI_MASK` -- \[10\]
    pub note_id_hi: u32,
    pub gain_l: f32,
    pub gain_r: f32,
    /// Which channel row this voice takes its **opening** bend and gain from, \[11\]
    pub row_bias: u32,
    /// Fixed analytic coefficients, computed before GPU dispatch. The identity \[12\]
    pub rotation: crate::phase::Coefficients,
}

/// Which of `want` queued spawns the `i`-th accepted one should be, when only \[13\]
#[inline]
pub fn spawn_pick(i: usize, want: usize, take: usize) -> usize {
    debug_assert!(take > 0 && take <= want && i < take);
    (i as u64 * want as u64 / take as u64) as usize
}

/// The 64-bit key admission ranks queued spawns by, highest kept first. \[14\]
#[inline]
pub fn admit_key(cmd: &SpawnCmd, index: u64) -> u64 {
    (rank_gain_q(cmd.gain_l, cmd.gain_r) << 48) | mix48(index)
}

/// The tiebreak field of the ranking keys: a scrambled function of a \[15\]
#[inline]
pub(crate) fn mix48(index: u64) -> u64 {
    const M: u64 = 0x0000_FFFF_FFFF_FFFF;
    let mut z = index & M;
    z = (z ^ (z >> 24)) & M;
    z = z.wrapping_mul(0x9E37_79B9_7F4B) & M;
    z = (z ^ (z >> 23)) & M;
    z = z.wrapping_mul(0xBF58_476D_1CE5) & M;
    z = (z ^ (z >> 25)) & M;
    z
}

/// Quantise a voice's opening gain to the 15 bits `admit_key` has for it. \[16\]
#[inline]
pub fn rank_gain_q(gain_l: f32, gain_r: f32) -> u64 {
    let g = gain_l.abs() + gain_r.abs();
    // NaN as well as zero and negatives: a NaN gain must not become a rank.
    if g.is_nan() || g <= 0.0 {
        return 0;
    }
    ((g.to_bits() >> 16) & 0x7FFF) as u64
}

/// Per-block note-off state, sampled once per reduce tile. \[17\]
pub struct GateTable {
    /// Note-off count per slot at the *start* of this block, and the offset of \[18\]
    pub off_meta: Vec<u32>,
    /// The exact frame of every note-off published in this block, grouped by \[19\]
    pub off_runs: Vec<u32>,
    pub tiles: usize,
    /// Channels the table currently covers. See `BASE_CHANNELS` and `grow`.
    channels: usize,
    /// Live counters, carried across blocks.
    on_count: Vec<u32>,
    off_count: Vec<u32>,
    /// Note-offs that arrived while the channel's sustain pedal was down and \[20\]
    pending_off: Vec<u32>,
    /// Per channel, on every port: set while CC64 is down.
    sustain: [bool; crate::midi::CHANNELS],
    /// Per channel: set while CC66 is down.
    sostenuto: [bool; crate::midi::CHANNELS],
    /// How many notes at each slot the sostenuto pedal caught. Sostenuto only \[21\]
    sost_held: Vec<u32>,
    /// Runs published so far this block, as parallel (slot, frame, count) \[22\]
    ev_slot: Vec<u32>,
    ev_frame: Vec<u32>,
    ev_count: Vec<u32>,
    /// Each slot's latest run in those lists this block, or `NO_RUN`. A \[23\]
    last_run: Vec<u32>,
    /// Scatter cursors for that sort, kept to avoid a per-block allocation.
    scatter: Vec<u32>,
}

impl GateTable {
    pub fn new(cfg: &Config) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        let slots = BASE_CHANNELS * 128;
        GateTable {
            off_meta: vec![0; (slots + 1) * 2],
            off_runs: Vec::new(),
            tiles,
            channels: BASE_CHANNELS,
            on_count: vec![0; slots],
            off_count: vec![0; slots],
            pending_off: vec![0; slots],
            sustain: [false; crate::midi::CHANNELS],
            sostenuto: [false; crate::midi::CHANNELS],
            sost_held: vec![0; slots],
            ev_slot: Vec::new(),
            ev_frame: Vec::new(),
            ev_count: Vec::new(),
            last_run: vec![NO_RUN; slots],
            scatter: vec![0; slots],
        }
    }

    #[inline]
    pub fn slot(ch: u8, key: u8) -> usize {
        ch as usize * 128 + (key as usize & 127)
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    pub fn slots(&self) -> usize {
        self.channels * 128
    }

    /// Cover `channels` channels, keeping everything already counted. Safe at \[24\]
    pub fn grow(&mut self, channels: usize) {
        if channels <= self.channels {
            return;
        }
        self.channels = channels;
        let slots = self.slots();
        self.off_meta.resize((slots + 1) * 2, 0);
        self.on_count.resize(slots, 0);
        self.off_count.resize(slots, 0);
        self.pending_off.resize(slots, 0);
        self.sost_held.resize(slots, 0);
        self.last_run.resize(slots, NO_RUN);
        self.scatter.resize(slots, 0);
    }

    /// Start a new block. The per-slot base is the count as it stands now, \[25\]
    pub fn begin_block(&mut self) {
        for s in 0..self.slots() {
            self.off_meta[s * 2] = self.off_count[s];
        }
        self.last_run.fill(NO_RUN);
        self.ev_slot.clear();
        self.ev_frame.clear();
        self.ev_count.clear();
    }

    /// Record `n` published note-offs at one exact frame. \[26\]
    #[inline]
    fn publish_off(&mut self, s: usize, frame: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.off_count[s] = self.off_count[s].wrapping_add(n);
        let last = self.last_run[s] as usize;
        if last < self.ev_frame.len() && self.ev_frame[last] == frame {
            self.ev_count[last] = self.ev_count[last].wrapping_add(n);
            return;
        }
        self.last_run[s] = self.ev_slot.len() as u32;
        self.ev_slot.push(s as u32);
        self.ev_frame.push(frame);
        self.ev_count.push(n);
    }

    /// Register a note-on. Returns the ordinal the voice should carry.
    pub fn note_on(&mut self, ch: u8, key: u8, frame: u32) -> u32 {
        let _ = frame;
        let s = Self::slot(ch, key);
        self.on_count[s] = self.on_count[s].wrapping_add(1);
        self.on_count[s]
    }

    /// Register a note-off. A note-off with nothing sounding is ignored, which \[27\]
    pub fn note_off(&mut self, ch: u8, key: u8, frame: u32) {
        let s = Self::slot(ch, key);
        if self.sostenuto[ch as usize] && self.sost_held[s] > 0 {
            // [28]
            if self.off_count[s].wrapping_add(self.pending_off[s]) != self.on_count[s] {
                self.pending_off[s] += 1;
                self.sost_held[s] -= 1;
            }
            return;
        }
        if self.sustained(ch) {
            // [29]
            if self.off_count[s].wrapping_add(self.pending_off[s]) != self.on_count[s] {
                self.pending_off[s] += 1;
            }
        } else if self.off_count[s] != self.on_count[s] {
            self.publish_off(s, frame, 1);
        }
    }

    #[inline]
    pub fn sustained(&self, ch: u8) -> bool {
        self.sustain[ch as usize]
    }

    /// CC64. Pressing holds every later note-off on the channel; releasing \[30\]
    pub fn set_sustain(&mut self, ch: u8, down: bool, frame: u32) {
        if down == self.sustained(ch) {
            return;
        }
        self.sustain[ch as usize] = down;
        if down {
            return;
        }
        if self.sostenuto[ch as usize] {
            // Sostenuto is still down and still holding what it caught.
            return;
        }
        self.flush_pending(ch, frame);
    }

    /// CC66. Holds only the notes already sounding when it goes down; notes \[31\]
    pub fn set_sostenuto(&mut self, ch: u8, down: bool, frame: u32) {
        if down == self.sostenuto[ch as usize] {
            return;
        }
        self.sostenuto[ch as usize] = down;
        let base = ch as usize * 128;
        if down {
            for k in 0..128 {
                let s = base + k;
                self.sost_held[s] = self.on_count[s]
                    .wrapping_sub(self.off_count[s])
                    .wrapping_sub(self.pending_off[s]);
            }
            return;
        }
        for k in 0..128 {
            self.sost_held[base + k] = 0;
        }
        if self.sustained(ch) {
            // The other pedal is still down, so nothing damps yet.
            return;
        }
        self.flush_pending(ch, frame);
    }

    /// Publish everything the pedal was holding, all at `frame`. That frame is \[32\]
    fn flush_pending(&mut self, ch: u8, frame: u32) {
        let base = ch as usize * 128;
        for k in 0..128 {
            let s = base + k;
            let n = self.pending_off[s];
            if n != 0 {
                self.pending_off[s] = 0;
                self.publish_off(s, frame, n);
            }
        }
    }

    /// CC123. Releases what is sounding, but a held pedal still holds: the \[33\]
    pub fn all_notes_off(&mut self, ch: u8, frame: u32) {
        let base = ch as usize * 128;
        if self.sustained(ch) {
            for k in 0..128 {
                let s = base + k;
                self.pending_off[s] = self.on_count[s].wrapping_sub(self.off_count[s]);
            }
            return;
        }
        for k in 0..128 {
            let s = base + k;
            let n = self.on_count[s].wrapping_sub(self.off_count[s]);
            self.publish_off(s, frame, n);
        }
    }

    /// CC120. Stops everything on the channel now, pedal or not, and drops \[34\]
    pub fn all_sound_off(&mut self, ch: u8, frame: u32) {
        let base = ch as usize * 128;
        for k in 0..128 {
            let s = base + k;
            let n = self.on_count[s].wrapping_sub(self.off_count[s]);
            self.publish_off(s, frame, n);
            self.pending_off[s] = 0;
            self.sost_held[s] = 0;
        }
    }

    /// CC121. Lifting the pedal is part of resetting a channel's controllers, \[35\]
    pub fn reset_controllers(&mut self, ch: u8, frame: u32) {
        self.set_sostenuto(ch, false, frame);
        self.set_sustain(ch, false, frame);
    }

    /// Group this block's runs by slot, in ordinal order, and make each run's \[36\]
    pub fn end_block(&mut self) {
        let n = self.ev_slot.len();
        for s in 0..=self.slots() {
            self.off_meta[s * 2 + 1] = 0;
        }
        for &s in &self.ev_slot {
            self.off_meta[(s as usize + 1) * 2 + 1] += 1;
        }
        for s in 0..self.slots() {
            self.off_meta[(s + 1) * 2 + 1] += self.off_meta[s * 2 + 1];
        }
        self.off_runs.clear();
        self.off_runs.resize(n * 2, 0);
        for s in 0..self.slots() {
            self.scatter[s] = self.off_meta[s * 2 + 1];
        }
        for i in 0..n {
            let s = self.ev_slot[i] as usize;
            let r = self.scatter[s] as usize;
            let before = if r > self.off_meta[s * 2 + 1] as usize {
                self.off_runs[(r - 1) * 2]
            } else {
                0
            };
            self.off_runs[r * 2] = before.wrapping_add(self.ev_count[i]);
            self.off_runs[r * 2 + 1] = self.ev_frame[i];
            self.scatter[s] += 1;
        }
    }

    /// The note-off count for `slot` as it stood at the start of this block.
    #[inline]
    pub fn off_base(&self, slot: usize) -> u32 {
        self.off_meta[slot * 2]
    }

    /// The runs published for `slot` in this block, as interleaved \[37\]
    #[inline]
    pub fn off_runs_for(&self, slot: usize) -> &[u32] {
        let lo = self.off_meta[slot * 2 + 1] as usize;
        let hi = self.off_meta[(slot + 1) * 2 + 1] as usize;
        &self.off_runs[lo * 2..hi * 2]
    }

    /// Number of notes started but not yet released, across all slots. Notes \[38\]
    pub fn sounding(&self) -> u64 {
        self.on_count
            .iter()
            .zip(&self.off_count)
            .map(|(a, b)| a.wrapping_sub(*b) as u64)
            .sum()
    }
}

/// `GateTable::last_run` for a slot with no run yet this block.
const NO_RUN: u32 = u32::MAX;

/// The frame on which the voice holding `ordinal` at `slot` is released, read \[39\]
pub fn off_frame(meta: &[u32], runs: &[u32], slot: usize, ordinal: u32) -> Option<u32> {
    let base = meta[slot * 2];
    if ordinal <= base {
        return Some(0);
    }
    let first = meta[slot * 2 + 1];
    let end = meta[(slot + 1) * 2 + 1];
    let j = ordinal - base;
    if end <= first || j > runs[(end as usize - 1) * 2] {
        return None;
    }
    let (mut lo, mut hi) = (first, end - 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if runs[mid as usize * 2] >= j {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Some(runs[lo as usize * 2 + 1])
}

/// Where a voice's position on the envelope grid sits in its gate slot word: \[40\]
pub const GRID_SHIFT: u32 = SLOT_BITS;
pub const GRID_MASK: u32 = 0x1FFF;
const _: () = assert!(GRID_SHIFT + 13 <= 32);

/// A voice's position on its envelope grid at block frame `f`: frames since its \[41\]
#[inline]
pub fn grid_pos(env_phase: u32, slot_word: u32, f: u32, step: u32) -> u32 {
    (env_phase + f + step - ((slot_word >> GRID_SHIFT) & GRID_MASK)) % step
}

/// The frame a release starts falling on, for a note-off at block frame `off`: \[42\]
#[inline]
pub fn release_start(env_phase: u32, slot_word: u32, off: u32, step: u32) -> u32 {
    off + (step - grid_pos(env_phase, slot_word, off, step)) % step + 1
}

/// Frames of fall left from block frame `f` for a voice already releasing, \[43\]
#[inline]
pub fn fall_left(env_phase: u32, slot_word: u32, f: u32, step: u32) -> u32 {
    step - (grid_pos(env_phase, slot_word, f, step) + step - 1) % step
}

/// Words per channel in a `ChannelTable` row: bend factor, left gain, right \[44\]
pub const CHAN_FIELDS: usize = 8;
pub const CHAN_BEND: usize = 0;
pub const CHAN_GAIN_L: usize = 1;
pub const CHAN_GAIN_R: usize = 2;
/// Which copy of the params table this channel's voices read, see `ParamMod`.
pub const CHAN_VARIANT: usize = 3;
/// Frame within the block at which CC120 silenced this channel, plus one. \[45\]
pub const CHAN_CUT: usize = 4;
/// The note id that cut applies *below*, low and high words. \[46\]
pub const CHAN_CUT_ID_LO: usize = 5;
pub const CHAN_CUT_ID_HI: usize = 6;

/// Per-channel controller state, published the same way the note-off gate is: \[47\]
pub struct ChannelTable {
    /// `(tiles + 1) * channels * CHAN_FIELDS` entries, tile-major. Gains are \[48\]
    pub rows: Vec<u32>,
    pub tiles: usize,
    /// Channels a row covers. See `BASE_CHANNELS` and `grow`.
    channels: usize,
    /// Current state per channel, carried across blocks.
    now: Vec<u32>,
    cursor: usize,
    tile_frames: u32,
    bend_active: bool,
    gain_active: bool,
    variant_active: bool,
    cut_active: bool,
}

/// A channel's entry before anything has touched it: no bend, unity gains, \[49\]
const CHAN_UNTOUCHED: [u32; CHAN_FIELDS] = {
    let mut e = [0u32; CHAN_FIELDS];
    e[CHAN_BEND] = BEND_ONE;
    e[CHAN_GAIN_L] = 1.0f32.to_bits();
    e[CHAN_GAIN_R] = 1.0f32.to_bits();
    e
};

impl ChannelTable {
    pub fn new(cfg: &Config) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        let now = CHAN_UNTOUCHED.repeat(BASE_CHANNELS);
        let w = now.len();
        // [50]
        let mut rows = vec![0u32; (tiles + 1) * w];
        for t in 0..tiles {
            rows[t * w..t * w + w].copy_from_slice(&now);
        }
        ChannelTable {
            rows,
            tiles,
            channels: BASE_CHANNELS,
            now,
            cursor: 0,
            tile_frames: cfg.gate_frames,
            bend_active: false,
            gain_active: false,
            variant_active: false,
            cut_active: false,
        }
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Cover `channels` channels. Every row already written is re-laid at the \[51\]
    pub fn grow(&mut self, channels: usize) {
        if channels <= self.channels {
            return;
        }
        let (old, new) = (self.channels * CHAN_FIELDS, channels * CHAN_FIELDS);
        let fresh = CHAN_UNTOUCHED.repeat(channels - self.channels);
        let mut rows = Vec::with_capacity((self.tiles + 1) * new);
        for row in self.rows.chunks_exact(old) {
            rows.extend_from_slice(row);
            rows.extend_from_slice(&fresh);
        }
        self.rows = rows;
        self.now.extend_from_slice(&fresh);
        self.channels = channels;
    }

    pub fn begin_block(&mut self) {
        self.cursor = 0;
    }

    #[inline]
    fn advance_to(&mut self, frame: u32) {
        let tile = (frame / self.tile_frames) as usize;
        let w = self.channels * CHAN_FIELDS;
        while self.cursor <= tile && self.cursor < self.tiles {
            let base = self.cursor * w;
            self.rows[base..base + w].copy_from_slice(&self.now);
            self.cursor += 1;
        }
    }

    /// Set a channel's bend factor from this frame on.
    pub fn set_bend(&mut self, ch: u8, factor: u32, frame: u32) {
        let i = ch as usize * CHAN_FIELDS + CHAN_BEND;
        if self.now[i] == factor {
            return;
        }
        // [52]
        self.advance_to(frame);
        self.now[i] = factor;
    }

    /// CC120, All Sound Off: silence this channel *now*, ignoring release. \[53\]
    pub fn set_sound_off(&mut self, ch: u8, frame: u32, note_id: u64) {
        self.advance_to(frame);
        let tile = (frame / self.tile_frames) as usize;
        if tile < self.tiles {
            let i = (tile * self.channels + ch as usize) * CHAN_FIELDS;
            self.rows[i + CHAN_CUT] = frame + 1;
            self.rows[i + CHAN_CUT_ID_LO] = note_id as u32;
            self.rows[i + CHAN_CUT_ID_HI] = (note_id >> 32) as u32;
        }
        self.cut_active = true;
    }

    /// Set a channel's output gains from this frame on. These multiply the \[54\]
    pub fn set_gain(&mut self, ch: u8, l: f32, r: f32, frame: u32) {
        let base = ch as usize * CHAN_FIELDS;
        let (lb, rb) = (l.to_bits(), r.to_bits());
        if self.now[base + CHAN_GAIN_L] == lb && self.now[base + CHAN_GAIN_R] == rb {
            return;
        }
        // [55]
        self.advance_to(frame);
        self.now[base + CHAN_GAIN_L] = lb;
        self.now[base + CHAN_GAIN_R] = rb;
    }

    /// Select which copy of the params table this channel reads from.
    pub fn set_variant(&mut self, ch: u8, variant: u32, frame: u32) {
        let i = ch as usize * CHAN_FIELDS + CHAN_VARIANT;
        if self.now[i] == variant {
            return;
        }
        // [56]
        self.advance_to(frame);
        self.now[i] = variant;
    }

    /// Fill any tiles no event reached. Call before `modulate` and \[57\]
    pub fn end_block(&mut self) {
        let w = self.channels * CHAN_FIELDS;
        // [58]
        while self.cursor <= self.tiles {
            let base = self.cursor * w;
            self.rows[base..base + w].copy_from_slice(&self.now);
            self.cursor += 1;
        }
    }

    /// Which row a note struck at `frame` should take its opening bend and gain \[59\]
    pub fn row_bias(&self, ch: u8, frame: u32) -> u32 {
        let tile = (frame / self.tile_frames) as usize;
        if tile >= self.tiles || self.cursor <= tile {
            return 0;
        }
        let row = (tile * self.channels + ch as usize) * CHAN_FIELDS;
        let now = ch as usize * CHAN_FIELDS;
        for f in [CHAN_BEND, CHAN_GAIN_L, CHAN_GAIN_R] {
            if self.rows[row + f] != self.now[now + f] {
                return 1;
            }
        }
        0
    }

    /// Multiply a tile's already-published bend factor, for an LFO the host \[60\]
    pub fn modulate_bend(&mut self, ch: u8, tile: usize, factor: u32) {
        let i = (tile * self.channels + ch as usize) * CHAN_FIELDS + CHAN_BEND;
        let scaled = ((self.rows[i] as u64 * factor as u64) >> 24) as u32;
        self.rows[i] = scaled.max(1);
    }

    /// Scale a tile's already-published gains, for a tremolo LFO.
    pub fn modulate_gain(&mut self, ch: u8, tile: usize, factor: f32) {
        let base = (tile * self.channels + ch as usize) * CHAN_FIELDS;
        for f in [CHAN_GAIN_L, CHAN_GAIN_R] {
            let v = f32::from_bits(self.rows[base + f]) * factor;
            self.rows[base + f] = v.to_bits();
        }
    }

    pub fn refresh_active(&mut self) {
        let one = 1.0f32.to_bits();
        self.bend_active = false;
        self.gain_active = false;
        self.variant_active = false;
        self.cut_active = false;
        for e in self.rows.chunks_exact(CHAN_FIELDS) {
            if e[CHAN_BEND] != BEND_ONE {
                self.bend_active = true;
            }
            if e[CHAN_GAIN_L] != one || e[CHAN_GAIN_R] != one {
                self.gain_active = true;
            }
            if e[CHAN_VARIANT] != 0 {
                self.variant_active = true;
            }
            if e[CHAN_CUT] != 0 {
                self.cut_active = true;
            }
        }
    }

    /// False when every channel read the untouched params table, which lets \[61\]
    #[inline]
    pub fn variant_active(&self) -> bool {
        self.variant_active
    }

    /// True when any of the three needs the controller path at all.
    #[inline]
    pub fn any_active(&self) -> bool {
        self.bend_active || self.gain_active || self.variant_active || self.cut_active
    }

    /// True when some channel was silenced by CC120 in this block.
    #[inline]
    pub fn cut_active(&self) -> bool {
        self.cut_active
    }

    /// False when nothing in this block is bent, which lets both backends skip \[62\]
    #[inline]
    pub fn bend_active(&self) -> bool {
        self.bend_active
    }

    /// False when every channel sat at unity gain, which lets both backends \[63\]
    #[inline]
    pub fn gain_active(&self) -> bool {
        self.gain_active
    }

    #[inline]
    pub fn row(&self, tile: usize) -> &[u32] {
        let w = self.channels * CHAN_FIELDS;
        &self.rows[tile * w..tile * w + w]
    }

    /// The channel a voice belongs to, recovered from its gate slot. Voices \[64\]
    #[inline]
    pub fn channel_of(gate_slot: u32) -> usize {
        (gate_slot >> 7) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            block_frames: 64,
            reduce_tile: 8,
            gate_frames: 16,
            ..Default::default()
        }
    }

    impl GateTable {
        /// One frame per note-off, the layout the runs replaced, so these \[65\]
        fn off_frames_for(&self, slot: usize) -> Vec<u32> {
            let mut out = Vec::new();
            let mut done = 0u32;
            for run in self.off_runs_for(slot).chunks_exact(2) {
                out.resize(out.len() + (run[0] - done) as usize, run[1]);
                done = run[0];
            }
            out
        }
    }

    /// A file reaching a new port mid-block grows the controller rows, and \[66\]
    #[test]
    fn channel_rows_grow_mid_block_without_moving_what_was_published() {
        let mut t = ChannelTable::new(&cfg());
        let bent = BEND_ONE * 2;
        t.begin_block();
        t.set_bend(3, bent, 40); // tile 2, so tiles 0..=2 freeze unbent
        t.grow(32);
        assert_eq!(t.channels(), 32);
        t.set_gain(20, 0.5, 0.25, 50); // tile 3
        t.end_block();

        let at = |tile: usize, ch: usize, f: usize| t.row(tile)[ch * CHAN_FIELDS + f];
        for tile in 0..=t.tiles {
            let want = if tile <= 2 { BEND_ONE } else { bent };
            assert_eq!(at(tile, 3, CHAN_BEND), want, "channel 3's bend in tile {tile}");
            let (l, r) = if tile <= 3 { (1.0f32, 1.0f32) } else { (0.5, 0.25) };
            assert_eq!(at(tile, 20, CHAN_GAIN_L), l.to_bits(), "channel 20's gain in tile {tile}");
            assert_eq!(at(tile, 20, CHAN_GAIN_R), r.to_bits());
            assert_eq!(at(tile, 31, CHAN_BEND), BEND_ONE, "channel 31 was never touched");
        }
        // The same channel number on the first port saw none of it.
        assert_eq!(at(t.tiles, 4, CHAN_GAIN_L), 1.0f32.to_bits());
    }

    /// The gate table grows the same way: what was counted before stays \[67\]
    #[test]
    fn the_gate_table_grows_mid_block_and_keeps_its_counts() {
        let mut g = GateTable::new(&cfg());
        g.begin_block();
        g.note_on(5, 60, 0);
        g.note_on(5, 60, 0);
        g.set_sustain(5, true, 0);
        g.grow(48);
        g.note_on(16 + 5, 60, 8);
        g.note_off(16 + 5, 60, 9);
        g.note_off(5, 60, 10); // held by channel 5's pedal, not channel 21's
        g.end_block();

        assert_eq!(g.slots(), 48 * 128);
        assert_eq!(g.off_frames_for(GateTable::slot(16 + 5, 60)), &[9]);
        assert!(g.off_frames_for(GateTable::slot(5, 60)).is_empty());
        assert_eq!(g.sounding(), 2, "channel 5's two notes, one of them pedalled");
    }

    /// A pedal lift or an all-notes-off publishes any number of note-offs on \[68\]
    #[test]
    fn a_flood_of_note_offs_on_one_frame_is_one_run() {
        let mut g = GateTable::new(&cfg());
        let s = GateTable::slot(3, 40);
        g.begin_block();
        for _ in 0..100_000 {
            g.note_on(3, 40, 0);
        }
        for _ in 0..1000 {
            g.note_off(3, 40, 9);
        }
        g.all_notes_off(3, 9);
        g.end_block();

        assert_eq!(g.off_runs_for(s), &[100_000, 9]);
        assert_eq!(off_frame(&g.off_meta, &g.off_runs, s, 1), Some(9));
        assert_eq!(off_frame(&g.off_meta, &g.off_runs, s, 100_000), Some(9));
        assert_eq!(off_frame(&g.off_meta, &g.off_runs, s, 100_001), None);
    }

    /// The binary search reads exactly the frame an index into one entry per \[69\]
    #[test]
    fn the_run_search_reads_what_one_entry_per_note_off_would() {
        let mut g = GateTable::new(&Config {
            block_frames: 4096,
            reduce_tile: 16,
            gate_frames: 32,
            ..Default::default()
        });
        let mut seed = 0x2545_F491u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for block in 0..4 {
            g.begin_block();
            let mut frame = 0u32;
            for _ in 0..20_000 {
                let key = (next() % 3) as u8;
                match next() % 6 {
                    0 | 1 => {
                        g.note_on(0, key, frame);
                    }
                    2 | 3 => g.note_off(0, key, frame),
                    4 => g.set_sustain(0, next() % 3 == 0, frame),
                    _ => frame = (frame + next() % 2).min(4095),
                }
            }
            if block == 3 {
                g.all_sound_off(0, 0);
            }
            g.end_block();

            for key in 0..3u8 {
                let s = GateTable::slot(0, key);
                let base = g.off_base(s);
                let frames = g.off_frames_for(s);
                assert!(g.off_runs_for(s).len() / 2 <= 4096 + 1);
                for ordinal in 1..=base + frames.len() as u32 + 2 {
                    let want = if ordinal <= base {
                        Some(0)
                    } else {
                        frames.get((ordinal - base - 1) as usize).copied()
                    };
                    assert_eq!(
                        off_frame(&g.off_meta, &g.off_runs, s, ordinal),
                        want,
                        "block {block} key {key} ordinal {ordinal}"
                    );
                }
            }
        }
    }

    /// A note-off is published at the frame it happened on, not at the start \[70\]
    #[test]
    fn a_note_off_carries_its_exact_frame() {
        let mut g = GateTable::new(&cfg());
        g.begin_block();
        let ord = g.note_on(0, 60, 0);
        assert_eq!(ord, 1);
        g.note_off(0, 60, 40);
        g.end_block();

        let s = GateTable::slot(0, 60);
        assert_eq!(g.off_base(s), 0, "nothing was released before this block");
        assert_eq!(g.off_frames_for(s), &[40], "frame 40, not tile 2's start");
    }

    /// Several note-offs in one gate tile stay distinct, which is the case the \[71\]
    #[test]
    fn note_offs_inside_one_tile_keep_their_own_frames() {
        let mut g = GateTable::new(&cfg());
        g.begin_block();
        g.note_on(0, 60, 0);
        g.note_on(0, 60, 0);
        g.note_on(0, 60, 0);
        // All three land in tile 2 (frames 32..47).
        g.note_off(0, 60, 33);
        g.note_off(0, 60, 39);
        g.note_off(0, 60, 45);
        g.end_block();

        let s = GateTable::slot(0, 60);
        assert_eq!(g.off_frames_for(s), &[33, 39, 45]);
    }

    #[test]
    fn the_pedal_defers_note_offs_and_releases_them_together() {
        let mut g = GateTable::new(&cfg());
        let s = GateTable::slot(0, 60);
        g.begin_block();
        g.note_on(0, 60, 0);
        g.set_sustain(0, true, 0);
        g.note_off(0, 60, 16); // held
        g.end_block();
        assert!(
            g.off_frames_for(s).is_empty(),
            "a pedalled note-off must not be published"
        );
        assert_eq!(g.sounding(), 1, "the held note is still sounding");

        g.begin_block();
        g.set_sustain(0, false, 16);
        g.end_block();
        // [72]
        assert_eq!(g.off_frames_for(s), &[16]);
        assert_eq!(g.sounding(), 0);
    }

    /// Restriking a key while the pedal is down leaves two notes sounding and \[73\]
    #[test]
    fn a_restrike_under_the_pedal_releases_only_what_was_lifted() {
        let mut g = GateTable::new(&cfg());
        let s = GateTable::slot(0, 60);
        g.begin_block();
        g.set_sustain(0, true, 0);
        g.note_on(0, 60, 0);
        g.note_off(0, 60, 0);
        g.note_on(0, 60, 16);
        // A second off with only one note left un-lifted is still legal.
        g.note_off(0, 60, 16);
        // [74]
        g.note_off(0, 60, 16);
        g.end_block();
        assert_eq!(g.sounding(), 2, "both strikes are held by the pedal");

        g.begin_block();
        g.set_sustain(0, false, 0);
        g.end_block();
        assert_eq!(
            g.off_frames_for(s),
            &[0, 0],
            "both held offs publish at the lift, the third does not"
        );
        assert_eq!(g.sounding(), 0);
    }

    #[test]
    fn unmatched_note_off_does_not_pre_release() {
        let mut g = GateTable::new(&cfg());
        g.begin_block();
        g.note_off(0, 60, 0); // nothing sounding
        let ord = g.note_on(0, 60, 16);
        g.end_block();
        let s = GateTable::slot(0, 60);
        assert_eq!(ord, 1);
        assert!(
            g.off_frames_for(s).is_empty(),
            "stray note-off must not release the next note"
        );
    }

    #[test]
    fn retrigger_releases_in_order() {
        let mut g = GateTable::new(&cfg());
        g.begin_block();
        let a = g.note_on(0, 60, 0);
        let b = g.note_on(0, 60, 0);
        g.note_off(0, 60, 16);
        g.end_block();
        assert_eq!((a, b), (1, 2));
        let s = GateTable::slot(0, 60);
        // [75]
        assert_eq!(g.off_base(s), 0);
        assert_eq!(g.off_frames_for(s), &[16]);
    }
}
