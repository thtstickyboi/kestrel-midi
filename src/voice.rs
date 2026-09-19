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

/// Slots in the gate table: 16 MIDI channels by 128 keys.
pub const GATE_SLOTS: usize = 16 * 128;

/// Everything needed to start one voice, in the exact order the spawn shader \[2\]
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
    /// The params variant the channel was on at this voice's own note-on. \[3\]
    pub variant: u32,
    pub region: u32,
    /// `channel * 128 + key`, the note-off gate this voice listens to.
    pub gate_slot: u32,
    /// Which note-on of that slot this voice belongs to, counting from 1. \[4\]
    pub ordinal: u32,
    /// Frames into the block before the voice starts. Gives sample-accurate \[5\]
    pub start_rel: u32,
    pub note_id_lo: u32,
    pub note_id_hi: u32,
    pub gain_l: f32,
    pub gain_r: f32,
    /// Which channel row this voice takes its **opening** bend and gain from, \[6\]
    pub row_bias: u32,
}

/// Which of `want` queued spawns the `i`-th accepted one should be, when only \[7\]
#[inline]
pub fn spawn_pick(i: usize, want: usize, take: usize) -> usize {
    debug_assert!(take > 0 && take <= want && i < take);
    (i as u64 * want as u64 / take as u64) as usize
}

/// The 64-bit key admission ranks queued spawns by, highest kept first. \[8\]
#[inline]
pub fn admit_key(cmd: &SpawnCmd, index: u64) -> u64 {
    (rank_gain_q(cmd.gain_l, cmd.gain_r) << 48) | mix48(index)
}

/// The tiebreak field of the ranking keys: a scrambled function of a \[9\]
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

/// Quantise a voice's opening gain to the 15 bits `admit_key` has for it. \[10\]
#[inline]
pub fn rank_gain_q(gain_l: f32, gain_r: f32) -> u64 {
    let g = gain_l.abs() + gain_r.abs();
    // NaN as well as zero and negatives: a NaN gain must not become a rank.
    if g.is_nan() || g <= 0.0 {
        return 0;
    }
    ((g.to_bits() >> 16) & 0x7FFF) as u64
}

/// Per-block note-off state, sampled once per reduce tile. \[11\]
pub struct GateTable {
    /// Note-off count per slot at the *start* of this block, and the offset of \[12\]
    pub off_meta: Vec<u32>,
    /// The exact frame of every note-off published in this block, grouped by \[13\]
    pub off_runs: Vec<u32>,
    pub tiles: usize,
    /// Live counters, carried across blocks.
    on_count: Vec<u32>,
    off_count: Vec<u32>,
    /// Note-offs that arrived while the channel's sustain pedal was down and \[14\]
    pending_off: Vec<u32>,
    /// One bit per channel, set while CC64 is down.
    sustain: u16,
    /// One bit per channel, set while CC66 is down.
    sostenuto: u16,
    /// How many notes at each slot the sostenuto pedal caught. Sostenuto only \[15\]
    sost_held: Vec<u32>,
    /// Runs published so far this block, as parallel (slot, frame, count) \[16\]
    ev_slot: Vec<u32>,
    ev_frame: Vec<u32>,
    ev_count: Vec<u32>,
    /// Each slot's latest run in those lists this block, or `NO_RUN`. A \[17\]
    last_run: Vec<u32>,
    /// Scatter cursors for that sort, kept to avoid a per-block allocation.
    scatter: Vec<u32>,
}

impl GateTable {
    pub fn new(cfg: &Config) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        GateTable {
            off_meta: vec![0; (GATE_SLOTS + 1) * 2],
            off_runs: Vec::new(),
            tiles,
            on_count: vec![0; GATE_SLOTS],
            off_count: vec![0; GATE_SLOTS],
            pending_off: vec![0; GATE_SLOTS],
            sustain: 0,
            sostenuto: 0,
            sost_held: vec![0; GATE_SLOTS],
            ev_slot: Vec::new(),
            ev_frame: Vec::new(),
            ev_count: Vec::new(),
            last_run: vec![NO_RUN; GATE_SLOTS],
            scatter: vec![0; GATE_SLOTS],
        }
    }

    #[inline]
    pub fn slot(ch: u8, key: u8) -> usize {
        (ch as usize & 15) * 128 + (key as usize & 127)
    }

    /// Start a new block. The per-slot base is the count as it stands now, \[18\]
    pub fn begin_block(&mut self) {
        for s in 0..GATE_SLOTS {
            self.off_meta[s * 2] = self.off_count[s];
        }
        self.last_run.fill(NO_RUN);
        self.ev_slot.clear();
        self.ev_frame.clear();
        self.ev_count.clear();
    }

    /// Record `n` published note-offs at one exact frame. \[19\]
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

    /// Register a note-off. A note-off with nothing sounding is ignored, which \[20\]
    pub fn note_off(&mut self, ch: u8, key: u8, frame: u32) {
        let s = Self::slot(ch, key);
        if self.sostenuto & (1u16 << (ch & 15)) != 0 && self.sost_held[s] > 0 {
            // [21]
            if self.off_count[s].wrapping_add(self.pending_off[s]) != self.on_count[s] {
                self.pending_off[s] += 1;
                self.sost_held[s] -= 1;
            }
            return;
        }
        if self.sustained(ch) {
            // [22]
            if self.off_count[s].wrapping_add(self.pending_off[s]) != self.on_count[s] {
                self.pending_off[s] += 1;
            }
        } else if self.off_count[s] != self.on_count[s] {
            self.publish_off(s, frame, 1);
        }
    }

    #[inline]
    pub fn sustained(&self, ch: u8) -> bool {
        self.sustain & (1u16 << (ch & 15)) != 0
    }

    /// CC64. Pressing holds every later note-off on the channel; releasing \[23\]
    pub fn set_sustain(&mut self, ch: u8, down: bool, frame: u32) {
        if down == self.sustained(ch) {
            return;
        }
        let bit = 1u16 << (ch & 15);
        if down {
            self.sustain |= bit;
            return;
        }
        self.sustain &= !bit;
        if self.sostenuto & bit != 0 {
            // Sostenuto is still down and still holding what it caught.
            return;
        }
        self.flush_pending(ch, frame);
    }

    /// CC66. Holds only the notes already sounding when it goes down; notes \[24\]
    pub fn set_sostenuto(&mut self, ch: u8, down: bool, frame: u32) {
        let bit = 1u16 << (ch & 15);
        if down == (self.sostenuto & bit != 0) {
            return;
        }
        let base = (ch as usize & 15) * 128;
        if down {
            self.sostenuto |= bit;
            for k in 0..128 {
                let s = base + k;
                self.sost_held[s] = self.on_count[s]
                    .wrapping_sub(self.off_count[s])
                    .wrapping_sub(self.pending_off[s]);
            }
            return;
        }
        self.sostenuto &= !bit;
        for k in 0..128 {
            self.sost_held[base + k] = 0;
        }
        if self.sustained(ch) {
            // The other pedal is still down, so nothing damps yet.
            return;
        }
        self.flush_pending(ch, frame);
    }

    /// Publish everything the pedal was holding, all at `frame`. That frame is \[25\]
    fn flush_pending(&mut self, ch: u8, frame: u32) {
        let base = (ch as usize & 15) * 128;
        for k in 0..128 {
            let s = base + k;
            let n = self.pending_off[s];
            if n != 0 {
                self.pending_off[s] = 0;
                self.publish_off(s, frame, n);
            }
        }
    }

    /// CC123. Releases what is sounding, but a held pedal still holds: the \[26\]
    pub fn all_notes_off(&mut self, ch: u8, frame: u32) {
        let base = (ch as usize & 15) * 128;
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

    /// CC120. Stops everything on the channel now, pedal or not, and drops \[27\]
    pub fn all_sound_off(&mut self, ch: u8, frame: u32) {
        let base = (ch as usize & 15) * 128;
        for k in 0..128 {
            let s = base + k;
            let n = self.on_count[s].wrapping_sub(self.off_count[s]);
            self.publish_off(s, frame, n);
            self.pending_off[s] = 0;
            self.sost_held[s] = 0;
        }
    }

    /// CC121. Lifting the pedal is part of resetting a channel's controllers, \[28\]
    pub fn reset_controllers(&mut self, ch: u8, frame: u32) {
        self.set_sostenuto(ch, false, frame);
        self.set_sustain(ch, false, frame);
    }

    /// Group this block's runs by slot, in ordinal order, and make each run's \[29\]
    pub fn end_block(&mut self) {
        let n = self.ev_slot.len();
        for s in 0..=GATE_SLOTS {
            self.off_meta[s * 2 + 1] = 0;
        }
        for &s in &self.ev_slot {
            self.off_meta[(s as usize + 1) * 2 + 1] += 1;
        }
        for s in 0..GATE_SLOTS {
            self.off_meta[(s + 1) * 2 + 1] += self.off_meta[s * 2 + 1];
        }
        self.off_runs.clear();
        self.off_runs.resize(n * 2, 0);
        for s in 0..GATE_SLOTS {
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

    /// The runs published for `slot` in this block, as interleaved \[30\]
    #[inline]
    pub fn off_runs_for(&self, slot: usize) -> &[u32] {
        let lo = self.off_meta[slot * 2 + 1] as usize;
        let hi = self.off_meta[(slot + 1) * 2 + 1] as usize;
        &self.off_runs[lo * 2..hi * 2]
    }

    /// Number of notes started but not yet released, across all slots. Notes \[31\]
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

/// The frame on which the voice holding `ordinal` at `slot` is released, read \[32\]
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

/// Where a voice's position on the envelope grid sits in its gate slot word: \[33\]
pub const GRID_SHIFT: u32 = 11;
pub const GRID_MASK: u32 = 0x1FFF;

/// A voice's position on its envelope grid at block frame `f`: frames since its \[34\]
#[inline]
pub fn grid_pos(env_phase: u32, slot_word: u32, f: u32, step: u32) -> u32 {
    (env_phase + f + step - ((slot_word >> GRID_SHIFT) & GRID_MASK)) % step
}

/// The frame a release starts falling on, for a note-off at block frame `off`: \[35\]
#[inline]
pub fn release_start(env_phase: u32, slot_word: u32, off: u32, step: u32) -> u32 {
    off + (step - grid_pos(env_phase, slot_word, off, step)) % step + 1
}

/// Frames of fall left from block frame `f` for a voice already releasing, \[36\]
#[inline]
pub fn fall_left(env_phase: u32, slot_word: u32, f: u32, step: u32) -> u32 {
    step - (grid_pos(env_phase, slot_word, f, step) + step - 1) % step
}

pub const BEND_CHANNELS: usize = 16;
/// Words per channel in a `ChannelTable` row: bend factor, left gain, right \[37\]
pub const CHAN_FIELDS: usize = 8;
pub const CHAN_BEND: usize = 0;
pub const CHAN_GAIN_L: usize = 1;
pub const CHAN_GAIN_R: usize = 2;
/// Which copy of the params table this channel's voices read, see `ParamMod`.
pub const CHAN_VARIANT: usize = 3;
/// Frame within the block at which CC120 silenced this channel, plus one. \[38\]
pub const CHAN_CUT: usize = 4;
/// The note id that cut applies *below*, low and high words. \[39\]
pub const CHAN_CUT_ID_LO: usize = 5;
pub const CHAN_CUT_ID_HI: usize = 6;

/// Per-channel controller state, published the same way the note-off gate is: \[40\]
pub struct ChannelTable {
    /// `tiles * BEND_CHANNELS * CHAN_FIELDS` entries, tile-major. Gains are \[41\]
    pub rows: Vec<u32>,
    pub tiles: usize,
    /// Current state per channel, carried across blocks.
    now: [u32; BEND_CHANNELS * CHAN_FIELDS],
    cursor: usize,
    tile_frames: u32,
    bend_active: bool,
    gain_active: bool,
    variant_active: bool,
    cut_active: bool,
}

impl ChannelTable {
    pub fn new(cfg: &Config) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        let mut now = [0u32; BEND_CHANNELS * CHAN_FIELDS];
        for c in 0..BEND_CHANNELS {
            now[c * CHAN_FIELDS + CHAN_BEND] = BEND_ONE;
            now[c * CHAN_FIELDS + CHAN_GAIN_L] = 1.0f32.to_bits();
            now[c * CHAN_FIELDS + CHAN_GAIN_R] = 1.0f32.to_bits();
            now[c * CHAN_FIELDS + CHAN_VARIANT] = 0;
        }
        // [42]
        let mut rows = vec![0u32; (tiles + 1) * BEND_CHANNELS * CHAN_FIELDS];
        for t in 0..tiles {
            let base = t * BEND_CHANNELS * CHAN_FIELDS;
            rows[base..base + BEND_CHANNELS * CHAN_FIELDS].copy_from_slice(&now);
        }
        ChannelTable {
            rows,
            tiles,
            now,
            cursor: 0,
            tile_frames: cfg.gate_frames,
            bend_active: false,
            gain_active: false,
            variant_active: false,
            cut_active: false,
        }
    }

    pub fn begin_block(&mut self) {
        self.cursor = 0;
    }

    #[inline]
    fn advance_to(&mut self, frame: u32) {
        let tile = (frame / self.tile_frames) as usize;
        let w = BEND_CHANNELS * CHAN_FIELDS;
        while self.cursor <= tile && self.cursor < self.tiles {
            let base = self.cursor * w;
            self.rows[base..base + w].copy_from_slice(&self.now);
            self.cursor += 1;
        }
    }

    /// Set a channel's bend factor from this frame on.
    pub fn set_bend(&mut self, ch: u8, factor: u32, frame: u32) {
        let i = (ch as usize & 15) * CHAN_FIELDS + CHAN_BEND;
        if self.now[i] == factor {
            return;
        }
        // [43]
        self.advance_to(frame);
        self.now[i] = factor;
    }

    /// CC120, All Sound Off: silence this channel *now*, ignoring release. \[44\]
    pub fn set_sound_off(&mut self, ch: u8, frame: u32, note_id: u64) {
        self.advance_to(frame);
        let tile = (frame / self.tile_frames) as usize;
        if tile < self.tiles {
            let i = (tile * BEND_CHANNELS + (ch as usize & 15)) * CHAN_FIELDS;
            self.rows[i + CHAN_CUT] = frame + 1;
            self.rows[i + CHAN_CUT_ID_LO] = note_id as u32;
            self.rows[i + CHAN_CUT_ID_HI] = (note_id >> 32) as u32;
        }
        self.cut_active = true;
    }

    /// Set a channel's output gains from this frame on. These multiply the \[45\]
    pub fn set_gain(&mut self, ch: u8, l: f32, r: f32, frame: u32) {
        let base = (ch as usize & 15) * CHAN_FIELDS;
        let (lb, rb) = (l.to_bits(), r.to_bits());
        if self.now[base + CHAN_GAIN_L] == lb && self.now[base + CHAN_GAIN_R] == rb {
            return;
        }
        // [46]
        self.advance_to(frame);
        self.now[base + CHAN_GAIN_L] = lb;
        self.now[base + CHAN_GAIN_R] = rb;
    }

    /// Select which copy of the params table this channel reads from.
    pub fn set_variant(&mut self, ch: u8, variant: u32, frame: u32) {
        let i = (ch as usize & 15) * CHAN_FIELDS + CHAN_VARIANT;
        if self.now[i] == variant {
            return;
        }
        // [47]
        self.advance_to(frame);
        self.now[i] = variant;
    }

    /// Fill any tiles no event reached. Call before `modulate` and \[48\]
    pub fn end_block(&mut self) {
        let w = BEND_CHANNELS * CHAN_FIELDS;
        // [49]
        while self.cursor <= self.tiles {
            let base = self.cursor * w;
            self.rows[base..base + w].copy_from_slice(&self.now);
            self.cursor += 1;
        }
    }

    /// Which row a note struck at `frame` should take its opening bend and gain \[50\]
    pub fn row_bias(&self, ch: u8, frame: u32) -> u32 {
        let tile = (frame / self.tile_frames) as usize;
        if tile >= self.tiles || self.cursor <= tile {
            return 0;
        }
        let row = (tile * BEND_CHANNELS + (ch as usize & 15)) * CHAN_FIELDS;
        let now = (ch as usize & 15) * CHAN_FIELDS;
        for f in [CHAN_BEND, CHAN_GAIN_L, CHAN_GAIN_R] {
            if self.rows[row + f] != self.now[now + f] {
                return 1;
            }
        }
        0
    }

    /// Multiply a tile's already-published bend factor, for an LFO the host \[51\]
    pub fn modulate_bend(&mut self, ch: u8, tile: usize, factor: u32) {
        let i = (tile * BEND_CHANNELS + (ch as usize & 15)) * CHAN_FIELDS + CHAN_BEND;
        let scaled = ((self.rows[i] as u64 * factor as u64) >> 24) as u32;
        self.rows[i] = scaled.max(1);
    }

    /// Scale a tile's already-published gains, for a tremolo LFO.
    pub fn modulate_gain(&mut self, ch: u8, tile: usize, factor: f32) {
        let base = (tile * BEND_CHANNELS + (ch as usize & 15)) * CHAN_FIELDS;
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

    /// False when every channel read the untouched params table, which lets \[52\]
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

    /// False when nothing in this block is bent, which lets both backends skip \[53\]
    #[inline]
    pub fn bend_active(&self) -> bool {
        self.bend_active
    }

    /// False when every channel sat at unity gain, which lets both backends \[54\]
    #[inline]
    pub fn gain_active(&self) -> bool {
        self.gain_active
    }

    #[inline]
    pub fn row(&self, tile: usize) -> &[u32] {
        let w = BEND_CHANNELS * CHAN_FIELDS;
        &self.rows[tile * w..tile * w + w]
    }

    /// The channel a voice belongs to, recovered from its gate slot. Voices \[55\]
    #[inline]
    pub fn channel_of(gate_slot: u32) -> usize {
        (gate_slot >> 7) as usize & 15
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
        /// One frame per note-off, the layout the runs replaced, so these \[56\]
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

    /// A pedal lift or an all-notes-off publishes any number of note-offs on \[57\]
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

    /// The binary search reads exactly the frame an index into one entry per \[58\]
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

    /// A note-off is published at the frame it happened on, not at the start \[59\]
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

    /// Several note-offs in one gate tile stay distinct, which is the case the \[60\]
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
        // [61]
        assert_eq!(g.off_frames_for(s), &[16]);
        assert_eq!(g.sounding(), 0);
    }

    /// Restriking a key while the pedal is down leaves two notes sounding and \[62\]
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
        // [63]
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
        // [64]
        assert_eq!(g.off_base(s), 0);
        assert_eq!(g.off_frames_for(s), &[16]);
    }
}
