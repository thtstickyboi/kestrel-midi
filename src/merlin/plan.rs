// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Turning note-ons into impulses. \[1\]

use crate::merlin::fdl::Impulse;
use rayon::prelude::*;
use std::collections::HashMap;

/// What makes two notes share a cached **waveform**. \[2\]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct WaveKey {
    pub smp_base: u32,
    /// The right half of a stereo sample, `smp_base` for a mono waveform. See \[3\]
    pub smp_base_r: u32,
    pub smp_len: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    /// 32.32 pitch step. Bend lives here, along with the key's transposition.
    pub step: u64,
    /// `SpawnCmd` flags, plus the params flags with `RP_FILTER` cleared.
    pub flags: u32,
    /// Envelope and LFO, **by value**: attack, decay, sustain, release and the \[4\]
    pub env: [u32; 10],
    /// The modulation envelope, also by value, and zero when the region drives \[5\]
    pub menv: [u32; 7],
}

impl WaveKey {
    /// `menv` is the region's modulation-envelope row, or `None` when the bank \[6\]
    pub fn new(
        cmd: &crate::voice::SpawnCmd,
        p: &crate::bank::RegionParams,
        menv: Option<&crate::bank::ModEnvParams>,
    ) -> Self {
        Self {
            smp_base: cmd.smp_base,
            smp_base_r: cmd.smp_base,
            smp_len: cmd.smp_len,
            loop_start: cmd.loop_start,
            loop_end: cmd.loop_end,
            step: ((cmd.step_hi as u64) << 32) | cmd.step_lo as u64,
            flags: cmd.flags ^ ((p.flags & crate::bank::RP_FILTER) << 16),
            env: [
                p.attack_rate.to_bits(),
                p.attack_end.to_bits(),
                p.decay_coef.to_bits(),
                p.decay_target.to_bits(),
                p.sustain.to_bits(),
                p.release_coef.to_bits(),
                p.mod_lfo_inc,
                p.vib_lfo_inc,
                p.lfo_delays,
                p.lfo_pitch,
            ],
            menv: match menv {
                None => [0; 7],
                Some(m) => [
                    m.delay_frames,
                    m.attack_frames,
                    m.hold_frames,
                    m.decay_rate.to_bits(),
                    m.sustain.to_bits(),
                    m.release_rate.to_bits(),
                    m.to_pitch.to_bits(),
                ],
            },
        }
    }
}

/// The biquad a note's velocity selects. Notes sharing one of these share a bus.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FilterKey {
    pub on: bool,
    pub coef: [u32; 4],
}

impl FilterKey {
    pub fn new(p: &crate::bank::RegionParams) -> Self {
        let on = p.flags & crate::bank::RP_FILTER != 0;
        Self {
            on,
            // [7]
            coef: if on {
                [p.b0.to_bits(), p.b1.to_bits(), p.a1.to_bits(), p.a2.to_bits()]
            } else {
                [0; 4]
            },
        }
    }

    /// The key of what `bus_filter` returned.
    pub fn of(coef: Option<[f32; 4]>) -> Self {
        Self { on: coef.is_some(), coef: coef.map_or([0; 4], |c| c.map(f32::to_bits)) }
    }
}

/// How finely a snapped filter keeps its resonance, in centibels.
const Q_GRID_CB: f64 = 10.0;

/// The biquad the per-key render runs a region's bus through, `None` with its \[8\]
pub fn bus_filter(p: &crate::bank::RegionParams, cents: f64) -> Option<[f32; 4]> {
    (p.flags & crate::bank::RP_FILTER != 0).then(|| {
        let c = [p.b0, p.b1, p.a1, p.a2];
        if cents > 0.0 {
            snap_lowpass(c, cents)
        } else {
            c
        }
    })
}

/// `c` with its cutoff on a `cents` grid of `log2(w)`, its resonance on \[9\]
fn snap_lowpass(c: [f32; 4], cents: f64) -> [f32; 4] {
    let [b0, b1, a1, a2] = c.map(f64::from);
    // `1 + a2 = 2 / a0`, and `1 - cos w` is what stays accurate at low cutoffs.
    let two_over_a0 = 1.0 + a2;
    let one_minus_cos = (two_over_a0 + a1) / two_over_a0;
    if !(two_over_a0 > 0.0 && one_minus_cos > 0.0 && one_minus_cos < 2.0 && b0 > 0.0 && (b1 - 2.0 * b0).abs() <= 1e-5 * b1) {
        return c;
    }
    let w = 2.0 * (one_minus_cos / 2.0).sqrt().asin();
    let alpha = 2.0 / two_over_a0 - 1.0;
    let q = w.sin() / (2.0 * alpha);
    let q_cb = (20.0 * q.log10() + 3.01) * 10.0;
    let step = cents / 1200.0;
    let w = 2f64.powf((w.log2() / step).round() * step);
    let q_cb = (q_cb / Q_GRID_CB).round() * Q_GRID_CB;
    let (b0, b1, a1, a2) = crate::bank::biquad_lowpass((w / std::f64::consts::TAU) as f32, q_cb as f32, 1.0);
    [b0, b1, a1, a2]
}

/// A waveform plus the release reference it was built at: one cache entry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct EntryKey {
    pub wave: WaveKey,
    /// Which release reference point, counting from the note's start.
    pub release: u16,
}

/// Interns keys into dense ids, in first-seen order. \[10\]
#[derive(Debug)]
pub struct Table<K> {
    map: HashMap<K, u32>,
    keys: Vec<K>,
}

impl<K> Default for Table<K> {
    fn default() -> Self {
        Self { map: HashMap::new(), keys: Vec::new() }
    }
}

impl<K: Copy + Eq + std::hash::Hash> Table<K> {
    pub fn intern(&mut self, key: K) -> u32 {
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        let id = self.keys.len() as u32;
        self.keys.push(key);
        self.map.insert(key, id);
        id
    }

    pub fn get(&self, key: &K) -> Option<u32> {
        self.map.get(key).copied()
    }

    pub fn key(&self, id: u32) -> Option<&K> {
        self.keys.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Interns entry keys into the dense ids the kernel cache is indexed by.
#[derive(Default, Debug)]
pub struct EntryTable {
    map: HashMap<EntryKey, u32>,
    keys: Vec<EntryKey>,
}

impl EntryTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// The id for `key`, assigning a new one if this is the first sighting. \[11\]
    pub fn intern(&mut self, key: EntryKey) -> u32 {
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        let id = self.keys.len() as u32;
        self.keys.push(key);
        self.map.insert(key, id);
        id
    }

    pub fn get(&self, key: &EntryKey) -> Option<u32> {
        self.map.get(key).copied()
    }

    pub fn key(&self, id: u32) -> Option<&EntryKey> {
        self.keys.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// How a note's duration becomes a release bucket and a gain. \[12\]
#[derive(Clone, Copy, Debug)]
pub struct ReleaseBuckets {
    pub width: u32,
}

impl ReleaseBuckets {
    /// From a release tail in seconds and the count section 3b's dynamic-range \[13\]
    pub fn from_tail(tail_secs: f64, rate: u32, buckets: u32) -> Self {
        let width = (tail_secs * rate as f64 / buckets.max(1) as f64).round() as u32;
        Self { width: width.max(1) }
    }

    /// The bucket a note of duration `d` frames belongs to, and that bucket's \[14\]
    pub fn bucket(&self, d: u32) -> (u16, u32) {
        let j = d / self.width;
        (j.min(u16::MAX as u32) as u16, j * self.width)
    }

    /// The same, with the reference point held back to where the region's \[15\]
    pub fn reference(&self, d: u32, flat_from: u32) -> (u16, u32) {
        let (j, dj) = self.bucket(d);
        (j, dj.max(flat_from))
    }

    /// Section 3b's collapse: `env(D) / env(Dj) * r^(Dj - D)`. \[16\]
    pub fn gain(&self, release_coef: f64, dj: u32, d: u32, env_ratio: f32) -> f32 {
        if release_coef <= 0.0 {
            // [17]
            return env_ratio;
        }
        let e = dj as i64 - d as i64;
        (release_coef.powi(e as i32) * env_ratio as f64) as f32
    }
}

/// One note, already resolved to an entry and a filter bus. \[18\]
#[derive(Clone, Copy, Debug)]
pub struct PlannedNote {
    pub entry: u32,
    /// Frame of the note-on **within its step**, not an absolute frame. \[19\]
    pub start_rel: u16,
    /// Which filter this note's velocity selected. Its two output buses are \[20\]
    pub filter: u16,
    /// Left and right gain, including the pan and the section 3b scalar.
    pub gain: [f32; 2],
    /// Identical notes this impulse stands for (THE PLAN, Phase 1: per run, not \[21\]
    pub weight: u16,
}

/// One block's worth of work, in the three arrays `Fdl::encode_block` takes.
#[derive(Default, Debug)]
pub struct BlockPlan {
    /// Entry ids, grouped by bus and ascending within each group.
    pub fired: Vec<u32>,
    /// `[lo, hi)` per bus into `fired`.
    pub bus_range: Vec<u32>,
    /// Impulses, whose `slot` indexes `fired` rather than naming an entry.
    pub impulses: Vec<Impulse>,
}

/// A planned note with its sort key built, which is what `rebuild` sorts. \[22\]
#[derive(Clone, Copy, Debug)]
struct Keyed {
    key: u64,
    gain: [f32; 2],
    /// The planned note's weight, carried to its impulses.
    weight: u32,
}

/// The notes of one filter, as a range of the sorted `Keyed` list.
#[derive(Clone, Copy, Debug)]
struct FilterRun {
    filter: u32,
    lo: usize,
    hi: usize,
}

/// One bus that has something on it: which run and channel, and how many rows \[23\]
#[derive(Clone, Copy, Debug)]
struct Stream {
    run: usize,
    ch: usize,
    rows: u32,
    imps: u32,
}

/// What one bus of one filter run holds: rows (distinct entries) and impulses.
#[derive(Clone, Copy, Debug, Default)]
struct Tally {
    rows: u32,
    imps: u32,
}

/// Buffers `BlockPlan::rebuild` reuses from one step to the next. \[24\]
#[derive(Default)]
pub struct PlanScratch {
    keyed: Vec<Keyed>,
    runs: Vec<FilterRun>,
    tally: Vec<[Tally; 2]>,
    streams: Vec<Stream>,
}

/// What the caller knows about the largest values in a step without looking: \[25\]
#[derive(Clone, Copy, Debug)]
pub struct PlanBounds {
    pub max_entry: u32,
    pub max_depth: u32,
}

impl PlanBounds {
    /// The tightest bounds for these notes, by looking. For callers that are not \[26\]
    pub fn of(notes: &[PlannedNote], depth: impl Fn(u32) -> u32) -> Self {
        let mut b = Self { max_entry: 0, max_depth: 0 };
        for n in notes {
            b.max_entry = b.max_entry.max(n.entry);
            b.max_depth = b.max_depth.max(depth(n.entry));
        }
        b
    }
}

/// Key for a note that has nothing to play, and for a note that is out of \[27\]
const DROPPED: u64 = u64::MAX;
const INVALID: u64 = u64::MAX - 1;

/// Bit widths for the packed sort key. `None` when they do not fit in 62 bits, \[28\]
#[derive(Clone, Copy, Debug)]
struct KeyBits {
    /// Left-shift of each field. Frame is the lowest, then the entry, then the \[29\]
    entry: u32,
    depth: u32,
    filter: u32,
    entry_bits: u32,
    rel_mask: u64,
    max_depth: u32,
}

impl KeyBits {
    fn new(max_rel: u32, max_entry: u32, max_depth: u32, max_filter: u32) -> Option<Self> {
        let bits = |v: u32| u32::BITS - v.leading_zeros();
        let (rel_bits, entry_bits, depth_bits, filter_bits) =
            (bits(max_rel), bits(max_entry), bits(max_depth), bits(max_filter));
        // [30]
        if rel_bits + entry_bits + depth_bits + filter_bits > 62 {
            return None;
        }
        Some(Self {
            entry: rel_bits,
            depth: rel_bits + entry_bits,
            filter: rel_bits + entry_bits + depth_bits,
            entry_bits,
            rel_mask: (1u64 << rel_bits) - 1,
            max_depth,
        })
    }
}

impl BlockPlan {
    /// Assemble the notes starting in `[block_start, block_start + frames)`. \[31\]
    pub fn build(
        notes: &[PlannedNote],
        block_start: u64,
        frames: u32,
        buses: usize,
        depth: impl Fn(u32) -> u32 + Sync,
    ) -> Self {
        let mut plan = BlockPlan::default();
        let bounds = PlanBounds::of(notes, &depth);
        plan.rebuild(&mut PlanScratch::default(), notes, block_start, frames, buses, bounds, depth);
        plan
    }

    /// [`Self::build`] into `self`, reusing its buffers and `scratch`'s. \[32\]
    #[allow(clippy::too_many_arguments)]
    pub fn rebuild(
        &mut self,
        scratch: &mut PlanScratch,
        notes: &[PlannedNote],
        block_start: u64,
        frames: u32,
        buses: usize,
        bounds: PlanBounds,
        depth: impl Fn(u32) -> u32 + Sync,
    ) {
        // [33]
        debug_assert!(
            block_start.is_multiple_of(frames as u64),
            "a step starts at a multiple of the partition"
        );
        self.fired.clear();
        self.impulses.clear();
        self.bus_range.clear();
        self.bus_range.resize(buses * 2, 0);
        if notes.is_empty() {
            return;
        }

        // [34]
        let max_filter = (buses.saturating_sub(1) / 2) as u32;
        let Some(kb) =
            KeyBits::new(frames.saturating_sub(1), bounds.max_entry, bounds.max_depth, max_filter)
        else {
            self.rebuild_wide(notes, buses, depth);
            return;
        };

        // [35]
        let keyed = &mut scratch.keyed;
        keyed.clear();
        keyed.par_extend(notes.par_iter().map(|n| {
            let live = |ch: usize| {
                let bus = n.filter as usize * 2 + ch;
                bus < buses && n.gain[ch] != 0.0
            };
            let gain =
                [if live(0) { n.gain[0] } else { 0.0 }, if live(1) { n.gain[1] } else { 0.0 }];
            let weight = n.weight as u32;
            if gain[0] == 0.0 && gain[1] == 0.0 {
                return Keyed { key: DROPPED, gain, weight };
            }
            let d = depth(n.entry);
            if n.entry > bounds.max_entry || d > bounds.max_depth || n.start_rel as u32 >= frames {
                return Keyed { key: INVALID, gain, weight };
            }
            let key = ((n.filter as u64) << kb.filter)
                | (((kb.max_depth - d) as u64) << kb.depth)
                | ((n.entry as u64) << kb.entry)
                | n.start_rel as u64;
            Keyed { key, gain, weight }
        }));
        // [36]
        keyed.par_sort_unstable_by_key(|k| k.key);
        let kept = keyed.partition_point(|k| k.key < INVALID);
        if keyed.get(kept).is_some_and(|k| k.key == INVALID) {
            self.rebuild_wide(notes, buses, depth);
            return;
        }
        keyed.truncate(kept);
        let keyed = &scratch.keyed;

        // Where each filter's notes start and stop.
        let runs = &mut scratch.runs;
        runs.clear();
        let mut lo = 0;
        while lo < keyed.len() {
            let filter = (keyed[lo].key >> kb.filter) as u32;
            let mut hi = lo + 1;
            while hi < keyed.len() && (keyed[hi].key >> kb.filter) as u32 == filter {
                hi += 1;
            }
            runs.push(FilterRun { filter, lo, hi });
            lo = hi;
        }

        // [37]
        let stream = |run: &FilterRun, ch: usize| {
            keyed[run.lo..run.hi]
                .iter()
                .filter(move |k| k.gain[ch] != 0.0)
                .map(move |k| (k.key >> kb.entry, (k.key & kb.rel_mask) as u32, k.gain[ch], k.weight))
        };
        let tally = &mut scratch.tally;
        tally.clear();
        tally.par_extend(runs.par_iter().map(|run| {
            let mut t = [Tally::default(); 2];
            for (ch, t) in t.iter_mut().enumerate() {
                let mut prev = u64::MAX;
                for (row, _, _, _) in stream(run, ch) {
                    t.rows += (row != prev) as u32;
                    prev = row;
                    t.imps += 1;
                }
            }
            t
        }));

        // [38]
        let streams = &mut scratch.streams;
        streams.clear();
        let (mut rows, mut imps) = (0usize, 0usize);
        let mut at = 0;
        for bus in 0..buses {
            self.bus_range[bus * 2] = rows as u32;
            let filter = (bus / 2) as u32;
            while at < runs.len() && runs[at].filter < filter {
                at += 1;
            }
            if at < runs.len() && runs[at].filter == filter {
                let t = tally[at][bus % 2];
                if t.imps > 0 {
                    streams.push(Stream { run: at, ch: bus % 2, rows: t.rows, imps: t.imps });
                }
                rows += t.rows as usize;
                imps += t.imps as usize;
            }
            self.bus_range[bus * 2 + 1] = rows as u32;
        }

        // [39]
        let zero = Impulse::new(0, 0, 0.0);
        self.fired.resize(rows, 0);
        self.impulses.par_extend((0..imps).into_par_iter().map(|_| zero));

        // Each bus then fills its own disjoint stretch of both, in parallel.
        let mut jobs: Vec<(&FilterRun, usize, &mut [u32], &mut [Impulse])> = Vec::new();
        let (mut fired, mut impulses) = (&mut self.fired[..], &mut self.impulses[..]);
        for s in streams.iter() {
            let (f, rest) = std::mem::take(&mut fired).split_at_mut(s.rows as usize);
            fired = rest;
            let (i, rest) = std::mem::take(&mut impulses).split_at_mut(s.imps as usize);
            impulses = rest;
            jobs.push((&runs[s.run], s.ch, f, i));
        }
        let entry_mask = (1u64 << kb.entry_bits) - 1;
        // The row each bus's first entry takes: how many rows come before it.
        let mut first_row = 0u32;
        let firsts: Vec<u32> = streams
            .iter()
            .map(|s| {
                let at = first_row;
                first_row += s.rows;
                at
            })
            .collect();
        jobs.par_iter_mut().zip(firsts.par_iter()).for_each(
            |((run, ch, fired, impulses), &first)| {
                let mut rows = 0usize;
                let mut prev = u64::MAX;
                for (n, (row, rel, gain, weight)) in stream(run, *ch).enumerate() {
                    if row != prev {
                        prev = row;
                        fired[rows] = (row & entry_mask) as u32;
                        rows += 1;
                    }
                    impulses[n] = Impulse::weighted(first + rows as u32 - 1, rel, gain, weight);
                }
            },
        );
    }

    /// `rebuild` for a step whose values do not pack into a `u64`, or go past \[40\]
    fn rebuild_wide(
        &mut self,
        notes: &[PlannedNote],
        buses: usize,
        depth: impl Fn(u32) -> u32 + Sync,
    ) {
        // [41]
        let mut flat: Vec<(u32, u32, u32, u32, f32, u32)> = notes
            .par_iter()
            .flat_map_iter(|n| {
                let rel = n.start_rel as u32;
                let d = depth(n.entry);
                (0..2usize).filter_map(move |ch| {
                    let bus = n.filter as u32 * 2 + ch as u32;
                    if n.gain[ch] == 0.0 || bus as usize >= buses {
                        return None;
                    }
                    Some((bus, d, n.entry, rel, n.gain[ch], n.weight as u32))
                })
            })
            .collect();
        flat.par_sort_unstable_by_key(|&(bus, d, e, f, _, _)| (bus, std::cmp::Reverse(d), e, f));

        let mut i = 0usize;
        for bus in 0..buses as u32 {
            let lo = self.fired.len() as u32;
            while i < flat.len() && flat[i].0 == bus {
                let entry = flat[i].2;
                let row = self.fired.len() as u32;
                self.fired.push(entry);
                // Every impulse of this entry on this bus shares the row.
                while i < flat.len() && flat[i].0 == bus && flat[i].2 == entry {
                    self.impulses.push(Impulse::weighted(row, flat[i].3, flat[i].4, flat[i].5));
                    i += 1;
                }
            }
            self.bus_range[bus as usize * 2] = lo;
            self.bus_range[bus as usize * 2 + 1] = self.fired.len() as u32;
        }
    }

    pub fn is_silent(&self) -> bool {
        self.impulses.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;

    /// A lowpass as the bank builds it, at `fc` as a share of the sample rate.
    fn lowpass(fc: f32, q_cb: f32) -> [f32; 4] {
        let (b0, b1, a1, a2) = crate::bank::biquad_lowpass(fc, q_cb, 1.0);
        [b0, b1, a1, a2]
    }

    /// The bank's velocity sweep: `cutoff=89`, `fil_veltrack=9600`.
    fn sweep() -> impl Iterator<Item = f32> {
        (1..128).map(|v| 89.0 * 2f32.powf(v as f32 / 127.0 * 8.0) / 48000.0)
    }

    /// `|H(e^jw)|` in dB.
    fn gain_db(c: [f32; 4], w: f64) -> f64 {
        let [b0, b1, a1, a2] = c.map(f64::from);
        let (c1, s1, c2, s2) = (w.cos(), w.sin(), (2.0 * w).cos(), (2.0 * w).sin());
        let num = (b0 + b1 * c1 + b0 * c2).hypot(b1 * s1 + b0 * s2);
        let den = (1.0 + a1 * c1 + a2 * c2).hypot(a1 * s1 + a2 * s2);
        20.0 * (num / den).log10()
    }

    #[test]
    fn the_filter_grid_shares_buses_across_a_velocity_sweep() {
        let mut exact = std::collections::HashSet::new();
        let mut grid = std::collections::HashSet::new();
        for fc in sweep() {
            let c = lowpass(fc, 0.0);
            exact.insert(c.map(f32::to_bits));
            grid.insert(snap_lowpass(c, 400.0).map(f32::to_bits));
        }
        assert_eq!(exact.len(), 127);
        assert!(grid.len() <= 25, "{} filters on a 400-cent grid", grid.len());
    }

    #[test]
    fn a_snapped_cutoff_stays_within_half_a_step() {
        for cents in [200.0f64, 400.0, 800.0] {
            // [42]
            for fc in sweep().filter(|&fc| fc < 0.1) {
                let c = lowpass(fc, 0.0);
                let s = snap_lowpass(c, cents);
                // [43]
                let w = std::f64::consts::TAU * fc as f64;
                assert!((gain_db(s, w / 10.0) - gain_db(c, w / 10.0)).abs() < 0.2, "{cents} cents at {fc}: passband");
                let past = w * 3.0;
                let bound = cents / 200.0 + 0.5;
                let err = (gain_db(s, past) - gain_db(c, past)).abs();
                assert!(err < bound, "{cents} cents at {fc}: {err:.2} dB past the cutoff");
            }
        }
    }

    #[test]
    fn a_snapped_filter_keeps_its_resonance() {
        let c = lowpass(0.05, 120.0);
        let s = snap_lowpass(c, 1e-3);
        let w = std::f64::consts::TAU * 0.05;
        assert!((gain_db(s, w) - gain_db(c, w)).abs() < 0.6, "{} against {}", gain_db(s, w), gain_db(c, w));
    }

    #[test]
    fn a_filter_that_does_not_read_back_is_left_alone() {
        // Not `b1 = 2 b0`: no lowpass of ours.
        let c = [0.1, 0.5, -1.2, 0.5];
        assert_eq!(snap_lowpass(c, 400.0), c);
    }

    /// `fired`, `bus_range` and `(row, frame, gain bits)` per impulse.
    type Plain = (Vec<u32>, Vec<u32>, Vec<(u32, u32, u32)>);

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound.max(1)
        }
    }

    fn notes(rng: &mut Lcg, n: usize, filters: u64, entries: u64, frames: u64) -> Vec<PlannedNote> {
        (0..n)
            .map(|_| {
                let gain = |rng: &mut Lcg| match rng.next(5) {
                    0 => 0.0,
                    1 => -0.0,
                    _ => 0.01 + rng.next(100) as f32 / 64.0,
                };
                PlannedNote {
                    entry: rng.next(entries) as u32,
                    start_rel: rng.next(frames) as u16,
                    filter: rng.next(filters) as u16,
                    gain: [gain(rng), gain(rng)],
                    weight: 1,
                }
            })
            .collect()
    }

    /// The plan written the plain way, one bus at a time: no packing, no \[44\]
    fn oracle(notes: &[PlannedNote], buses: usize, depth: &dyn Fn(u32) -> u32) -> Plain {
        let (mut fired, mut range, mut imps) = (Vec::new(), vec![0u32; buses * 2], Vec::new());
        for bus in 0..buses {
            let mut items = Vec::new();
            for n in notes {
                for ch in 0..2 {
                    if n.filter as usize * 2 + ch == bus && n.gain[ch] != 0.0 {
                        items.push((
                            Reverse(depth(n.entry)),
                            n.entry,
                            n.start_rel as u32,
                            n.gain[ch].to_bits(),
                        ));
                    }
                }
            }
            items.sort();
            range[bus * 2] = fired.len() as u32;
            let mut prev = None;
            for (_, e, frame, g) in items {
                if prev != Some(e) {
                    fired.push(e);
                    prev = Some(e);
                }
                imps.push((fired.len() as u32 - 1, frame, g));
            }
            range[bus * 2 + 1] = fired.len() as u32;
        }
        (fired, range, imps)
    }

    fn check(plan: &BlockPlan, want: &Plain, what: &str) {
        assert_eq!(plan.fired, want.0, "{what}: fired");
        assert_eq!(plan.bus_range, want.1, "{what}: bus_range");
        let order: Vec<(u32, u32)> = plan.impulses.iter().map(|i| (i.slot, i.frame)).collect();
        assert!(order.windows(2).all(|w| w[0] <= w[1]), "{what}: not in (row, frame) order");
        let mut got: Vec<(u32, u32, u32)> =
            plan.impulses.iter().map(|i| (i.slot, i.frame, i.gain.to_bits())).collect();
        got.sort();
        assert_eq!(got, want.2, "{what}: impulses");
    }

    #[test]
    fn the_packed_sort_matches_the_plain_construction() {
        let mut rng = Lcg(3);
        // [45]
        let mut scratch = PlanScratch::default();
        let mut plan = BlockPlan::default();
        for (n, filters, entries, buses) in [
            (3000, 4, 40, 8),
            (10, 2, 3, 4),
            (4000, 5, 900, 9),
            (0, 1, 1, 2),
            (700, 4, 30, 6),
            (1, 1, 1, 1),
        ] {
            let list = notes(&mut rng, n, filters, entries, 512);
            let depth = |e: u32| 1 + e % 7;
            let want = oracle(&list, buses, &depth);
            plan.rebuild(&mut scratch, &list, 0, 512, buses, PlanBounds::of(&list, depth), depth);
            check(&plan, &want, &format!("{n} notes, {filters} filters, {buses} buses"));
            // And the fresh-allocating entry point agrees with the reusing one.
            check(&BlockPlan::build(&list, 0, 512, buses, depth), &want, "build");
        }
    }

    /// Fields too wide to pack into 63 bits go to the unpacked sort, which has \[46\]
    #[test]
    fn a_step_too_wide_to_pack_takes_the_unpacked_sort_and_agrees() {
        let mut rng = Lcg(5);
        let mut list = notes(&mut rng, 600, 4, 50, 512);
        list[0].entry = u32::MAX;
        list[1].entry = u32::MAX - 1;
        let depth = |e: u32| if e >= u32::MAX - 1 { u32::MAX } else { e % 5 };
        let (rel, entry, depth_max, filter) = (511u32, u32::MAX, u32::MAX, 3u32);
        assert!(
            KeyBits::new(rel, entry, depth_max, filter).is_none(),
            "the fixture is meant to overflow the key"
        );
        let want = oracle(&list, 8, &depth);
        check(&BlockPlan::build(&list, 0, 512, 8, depth), &want, "unpacked");
    }

    /// The bounds are only a speed: a note past them sends the step to the \[47\]
    #[test]
    fn a_note_past_the_bounds_gives_the_same_plan() {
        let mut rng = Lcg(13);
        let list = notes(&mut rng, 800, 4, 60, 512);
        let depth = |e: u32| 1 + e % 7;
        let want = oracle(&list, 8, &depth);
        let mut scratch = PlanScratch::default();
        let mut plan = BlockPlan::default();
        for bounds in [
            PlanBounds { max_entry: 59, max_depth: 7 },
            PlanBounds { max_entry: 30, max_depth: 7 },
            PlanBounds { max_entry: 59, max_depth: 3 },
        ] {
            plan.rebuild(&mut scratch, &list, 0, 512, 8, bounds, depth);
            check(&plan, &want, &format!("{bounds:?}"));
        }
        // A frame that is not inside the partition is the same kind of surprise.
        plan.rebuild(&mut scratch, &list, 0, 100, 8, PlanBounds::of(&list, depth), depth);
        let narrow: Vec<PlannedNote> = list.clone();
        check(&plan, &oracle(&narrow, 8, &depth), "frames narrower than the notes");
    }

    #[test]
    fn rebuilding_into_a_used_plan_leaves_nothing_of_the_last_step() {
        let mut rng = Lcg(9);
        let mut scratch = PlanScratch::default();
        let mut plan = BlockPlan::default();
        let big = notes(&mut rng, 5000, 6, 200, 512);
        let bounds = PlanBounds { max_entry: 200, max_depth: 4 };
        plan.rebuild(&mut scratch, &big, 0, 512, 12, bounds, |e| e % 4);
        let small = notes(&mut rng, 20, 6, 200, 512);
        plan.rebuild(&mut scratch, &small, 0, 512, 12, bounds, |e| e % 4);
        check(&plan, &oracle(&small, 12, &|e| e % 4), "small after big");
        plan.rebuild(&mut scratch, &[], 0, 512, 12, bounds, |e| e % 4);
        assert!(plan.is_silent() && plan.fired.is_empty());
        assert_eq!(plan.bus_range, vec![0; 24]);
    }
}
