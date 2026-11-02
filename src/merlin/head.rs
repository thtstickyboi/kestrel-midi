// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The head correction: what the frequency-domain delay line cannot say. \[1\]

use crate::merlin::plan::EntryKey;
use std::collections::HashMap;

/// One note's correction, clipped to one partition step. Mirrors the WGSL \[2\]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Head {
    /// Where this note's waveform starts in the head cache. \[3\]
    pub base: u32,
    pub out0: u32,
    pub src0: u32,
    pub len: u32,
    pub gain_l: f32,
    pub gain_r: f32,
    pub f0: f32,
    pub r: f32,
    pub hold: u32,
    pub bus: u32,
    /// Frames from `src0` over which the delay line contributes nothing, so the \[4\]
    pub dead: u32,
    /// Frames this waveform's pages hold, which is what `src` is bounded by. \[5\]
    pub cap: u32,
    /// Added to `f` after each multiply by `r`: the envelope recurrence is \[6\]
    pub fade: f32,
    /// Per-frame gain increments, for a channel whose volume moves under the \[7\]
    pub dg_l: f32,
    pub dg_r: f32,
    /// The gain the delay line's impulse carries, which the correction has to \[8\]
    pub gi_l: f32,
    pub gi_r: f32,
    /// Host-side only, in what was padding: non-zero on the second piece of a \[9\]
    pub split: u32,
    /// Identical notes this correction stands for (THE PLAN, Phase 1: per run, \[10\]
    pub weight: u32,
}

/// A note needing correction, in absolute frames.
#[derive(Clone, Copy, Debug)]
pub struct HeadNote {
    pub wave: u32,
    pub start: u64,
    /// First frame of the note this record covers. \[11\]
    pub from: u32,
    /// `D`, the note's length. The correction runs over `[from, D)`.
    pub d: u32,
    /// `Dj`, the release reference the kernel was built at.
    pub dj: u32,
    pub r: f32,
    pub gain: [f32; 2],
    /// Where the gain gets to by frame `d`, because a channel's volume moves \[12\]
    pub gain_end: [f32; 2],
    /// Frames of linear damp after `d`, or zero for an ordinary note; after \[13\]
    pub fade: u16,
    /// Frames of the region's own exponential release after `d`, played out by \[14\]
    pub release: u16,
    /// Left output bus; right is `bus + 1`. The head has to go through the same \[15\]
    pub bus: u16,
    /// Identical notes this record stands for (THE PLAN, Phase 1: per run, not \[16\]
    pub weight: u16,
    /// `env(D) / env(Dj)`, section 3b's envelope ratio. \[17\]
    pub env_ratio: f32,
}

impl HeadNote {
    /// Frames from the note-on to the end of everything this record plays: \[18\]
    pub fn span(&self) -> u32 {
        self.d
            .saturating_add(self.fade as u32)
            .saturating_add(self.release as u32)
    }

    /// Frames of the kernel the delay line does not hold, because its impulse \[19\]
    pub fn skip(&self) -> u32 {
        if self.dj <= self.d {
            self.dj
        } else {
            0
        }
    }

    /// `f` at head frame `n`: the kernel's envelope, which the correction \[20\]
    fn f_at(&self, n: u32) -> f32 {
        // [21]
        let e = if n < self.dj {
            self.dj as i64 - self.d as i64
        } else {
            n as i64 - self.d as i64 + 1
        };
        if self.r <= 0.0 {
            // [22]
            return if n < self.dj { self.env_ratio } else { 0.0 };
        }
        ((self.r as f64).powi(e as i32) * self.env_ratio as f64) as f32
    }

    /// The frames `[lo, hi)` this record plays, in absolute frames, or `None` \[23\]
    pub fn reach(&self, cached: u32) -> Option<(u64, u64)> {
        let lo = self.start + self.from as u64;
        let hi = self.start + self.span().min(cached) as u64;
        (lo < hi).then_some((lo, hi))
    }

    /// The first and last partition step, as `frame / step_len`, that \[24\]
    pub fn steps(&self, step_len: u32, cached: u32) -> Option<(u64, u64)> {
        let (lo, hi) = self.reach(cached)?;
        Some((lo / step_len as u64, (hi - 1) / step_len as u64))
    }

    /// Where in this record and in the step the frames `clip` plays are: \[25\]
    fn window(&self, step_start: u64, step_len: u32, cached: u32) -> Option<(u32, u32, u32)> {
        let (reach_lo, reach_hi) = self.reach(cached)?;
        let lo = reach_lo.max(step_start);
        let hi = reach_hi.min(step_start + step_len as u64);
        if hi <= lo {
            return None;
        }
        Some(((lo - self.start) as u32, (lo - step_start) as u32, (hi - lo) as u32))
    }

    /// Which of the two pieces [`Self::clip`] returns exist, as a bitmask: \[26\]
    pub fn pieces(&self, step_start: u64, step_len: u32, cached: u32) -> u8 {
        let Some((src0, _, len)) = self.window(step_start, step_len, cached) else {
            return 0;
        };
        if self.fade == 0 && self.release != 0 {
            // [27]
            u8::from(src0 < self.d) | (u8::from(src0 + len > self.d) << 1)
        } else {
            1
        }
    }

    /// The part of this correction landing in `[step_start, step_start + len)`. \[28\]
    pub fn clip(&self, step_start: u64, step_len: u32, cached: u32) -> [Option<Head>; 2] {
        // [29]
        let Some((src0, out0, len)) = self.window(step_start, step_len, cached) else {
            return [None, None];
        };
        // [30]
        let dg = |i: usize| {
            if self.d == 0 || self.gain_end[i] == self.gain[i] {
                0.0
            } else {
                (self.gain_end[i] - self.gain[i]) / self.d as f32
            }
        };
        let (dg_l, dg_r) = (dg(0), dg(1));
        if self.fade != 0 && self.release == 0 {
            // [31]
            let cut = self.d.saturating_sub(src0);
            return [Some(Head {
                base: self.wave,
                out0,
                src0,
                len,
                gain_l: self.gain[0] + dg_l * src0 as f32,
                gain_r: self.gain[1] + dg_r * src0 as f32,
                f0: 0.0,
                r: 1.0,
                hold: cut + 1,
                bus: self.bus as u32,
                dead: cut,
                cap: 0,
                fade: 1.0 / self.fade as f32,
                dg_l,
                dg_r,
                // [32]
                gi_l: self.gain_end[0],
                gi_r: self.gain_end[1],
                split: 0,
                weight: self.weight as u32,
            }), None];
        }
        if self.release != 0 {
            // [33]
            let r = self.r as f64;
            // A damp after the release is `damp`'s.
            let end = (src0 + len).min(self.d + self.release as u32);
            let body = (src0 < self.d).then(|| {
                let blen = end.min(self.d) - src0;
                Head {
                    base: self.wave,
                    out0,
                    src0,
                    len: blen,
                    gain_l: self.gain[0] + dg_l * src0 as f32,
                    gain_r: self.gain[1] + dg_r * src0 as f32,
                    f0: 0.0,
                    r: 1.0,
                    // Never reached: `f` stays at zero for the whole piece.
                    hold: u32::MAX,
                    bus: self.bus as u32,
                    dead: 0,
                    cap: 0,
                    fade: 0.0,
                    dg_l,
                    dg_r,
                    gi_l: self.gain_end[0],
                    gi_r: self.gain_end[1],
                    split: 0,
                weight: self.weight as u32,
                }
            });
            let release = (end > self.d && end > src0).then(|| {
                let rsrc = src0.max(self.d);
                Head {
                    base: self.wave,
                    out0: out0 + (rsrc - src0),
                    src0: rsrc,
                    len: end - rsrc,
                    gain_l: self.gain_end[0],
                    gain_r: self.gain_end[1],
                    // `f` at `rsrc`, and moving from the first frame.
                    f0: (1.0 - r.powi((rsrc - self.d + 1) as i32)) as f32,
                    r: self.r,
                    hold: 0,
                    bus: self.bus as u32,
                    dead: 0,
                    cap: 0,
                    fade: (1.0 - r) as f32,
                    dg_l: 0.0,
                    dg_r: 0.0,
                    gi_l: self.gain_end[0],
                    gi_r: self.gain_end[1],
                    // The second piece of a note already counted this step.
                    split: u32::from(body.is_some()),
                    weight: self.weight as u32,
                }
            });
            return [body, release];
        }
        [Some(Head {
            base: self.wave,
            out0,
            src0,
            len,
            gain_l: self.gain[0] + dg_l * src0 as f32,
            gain_r: self.gain[1] + dg_r * src0 as f32,
            f0: self.f_at(src0),
            r: self.r,
            // Frames from `src0` until `f` starts moving.
            hold: self.dj.saturating_sub(src0),
            bus: self.bus as u32,
            // [34]
            dead: self.skip().saturating_sub(src0),
            cap: 0,
            fade: 0.0,
            dg_l,
            dg_r,
            gi_l: self.gain_end[0],
            gi_r: self.gain_end[1],
            split: 0,
                weight: self.weight as u32,
        }), None]
    }
}

impl HeadNote {
    /// The third piece of a note released on the head and cut in its tail, a \[35\]
    pub fn damp(&self, step_start: u64, step_len: u32, cached: u32) -> Option<Head> {
        if self.fade == 0 || self.release == 0 {
            return None;
        }
        let (src0, out0, len) = self.window(step_start, step_len, cached)?;
        let (from, end) = (self.d + self.release as u32, src0 + len);
        if end <= from {
            return None;
        }
        let rsrc = src0.max(from);
        let level = (self.r as f64).powi(self.release as i32 + 1);
        let step = level / self.fade as f64;
        Some(Head {
            base: self.wave,
            out0: out0 + (rsrc - src0),
            src0: rsrc,
            len: end - rsrc,
            gain_l: self.gain_end[0],
            gain_r: self.gain_end[1],
            f0: (1.0 - level + step * (rsrc - from) as f64) as f32,
            r: 1.0,
            hold: 0,
            bus: self.bus as u32,
            dead: 0,
            cap: 0,
            fade: step as f32,
            dg_l: 0.0,
            dg_r: 0.0,
            gi_l: self.gain_end[0],
            gi_r: self.gain_end[1],
            split: 0,
            weight: self.weight as u32,
        })
    }

    /// Every piece this record plays in the step: [`Self::clip`]'s and \[36\]
    pub fn played(&self, step_start: u64, step_len: u32, cached: u32) -> impl Iterator<Item = Head> {
        self.clip(step_start, step_len, cached).into_iter().flatten().chain(self.damp(step_start, step_len, cached))
    }
}

/// One change to what a run of notes plays, `n` frames from their note-on: to \[37\]
#[derive(Clone, Copy, Default)]
struct Turn {
    n: u32,
    a: [f64; 2],
    c: [f64; 2],
    b: [f64; 2],
    notes: i32,
    tails: i32,
}

impl HeadNote {
    /// Where this note changes what the run plays, by the same cases as \[38\]
    fn turns(&self, out: &mut Vec<Turn>) {
        let (r, w, ratio) = (self.r as f64, self.weight as f64, self.env_ratio as f64);
        let (d, dj) = (self.d, self.dj);
        let g = self.gain.map(|x| x as f64 * w);
        let ge = self.gain_end.map(|x| x as f64 * w);
        let dg: [f64; 2] = std::array::from_fn(|i| if d == 0 { 0.0 } else { (ge[i] - g[i]) / d as f64 });
        let at = |n: u32, i: usize| g[i] + dg[i] * n as f64;
        let neg = |x: [f64; 2]| x.map(|v| -v);
        let on = |a: [f64; 2]| Turn { n: self.from, a, c: dg, notes: 1, ..Turn::default() };
        let off = |a: [f64; 2]| Turn { n: d, a: neg(a), c: neg(dg), notes: -1, ..Turn::default() };
        let start = std::array::from_fn(|i| at(self.from, i));
        let end = std::array::from_fn(|i| at(d, i));
        if self.release != 0 {
            let tail_end = r.powi(self.release as i32 + 1);
            out.push(on(start));
            out.push(Turn { b: ge.map(|v| v * r), tails: 1, ..off(end) });
            out.push(Turn { n: d + self.release as u32, b: ge.map(|v| -v * tail_end), tails: -1, ..Turn::default() });
        } else if dj > d {
            // [39]
            let f = if r <= 0.0 { ratio } else { r.powi(dj as i32 - d as i32) * ratio };
            let level: [f64; 2] = std::array::from_fn(|i| g[i] - ge[i] * f);
            out.push(on(level));
            out.push(off(std::array::from_fn(|i| level[i] + dg[i] * d as f64)));
        } else if r > 0.0 && dj < d {
            let first = r.powi(dj as i32 - d as i32 + 1) * ratio;
            out.push(on(start));
            out.push(Turn { n: dj, b: ge.map(|v| -v * first), tails: 1, ..Turn::default() });
            out.push(Turn { b: ge.map(|v| v * r * ratio), tails: -1, ..off(end) });
        } else {
            out.push(on(start));
            out.push(off(end));
        }
    }
}

/// Notes of one waveform struck on one frame, played as one. \[40\]
#[derive(Default)]
pub struct Fold {
    turns: Vec<Turn>,
}

impl Fold {
    /// `notes`, all of one waveform and one start, as the pieces playing their \[41\]
    pub fn run<'a>(&mut self, notes: impl Iterator<Item = &'a HeadNote>, step_start: u64, step_len: u32, out: &mut Vec<Head>) {
        self.turns.clear();
        let mut shared = None;
        for h in notes {
            if h.fade != 0 {
                out.extend(h.played(step_start, step_len, u32::MAX));
                continue;
            }
            debug_assert!(shared.is_none_or(|(w, s, r, _)| (w, s, r) == (h.wave, h.start, h.r.to_bits())));
            shared = Some((h.wave, h.start, h.r.to_bits(), h.bus));
            h.turns(&mut self.turns);
        }
        let Some((wave, start, r, bus)) = shared else { return };
        let r = f32::from_bits(r);
        // The step, in frames from the note-on.
        let lo = step_start.saturating_sub(start);
        let hi = (step_start + step_len as u64).saturating_sub(start);
        self.turns.sort_unstable_by_key(|t| t.n);
        let turns = &self.turns;
        let rr = r as f64;
        let (mut a, mut c, mut b) = ([0f64; 2], [0f64; 2], [0f64; 2]);
        let (mut notes, mut tails) = (0i32, 0i32);
        // [42]
        let mut open: Option<u32> = None;
        let mut i = 0;
        while i < turns.len() {
            let n = turns[i].n;
            while i < turns.len() && turns[i].n == n {
                let t = &turns[i];
                for ch in 0..2 {
                    a[ch] += t.a[ch];
                    c[ch] += t.c[ch];
                    b[ch] += t.b[ch];
                }
                notes += t.notes;
                tails += t.tails;
                i += 1;
            }
            if notes == 0 {
                a = [0.0; 2];
                c = [0.0; 2];
            }
            if tails == 0 {
                b = [0.0; 2];
            }
            let Some(next) = turns.get(i).map(|t| t.n) else { break };
            if n as u64 >= hi {
                break;
            }
            // [43]
            let joins = (next as u64) < hi
                && turns[i..].iter().take_while(|t| t.n == next).all(|t| t.a == [0.0; 2] && t.c == [0.0; 2] && t.notes == 0 && t.tails > 0);
            if tails != 0 || !joins {
                let from = open.take().unwrap_or(n);
                if notes != 0 || tails != 0 {
                    if let Some(p) = piece(from, n, next, lo, hi, [a, c, b], (tails != 0, r)) {
                        out.push(Head { base: wave, out0: (start + p.src0 as u64 - step_start) as u32, bus: bus as u32, ..p });
                    }
                }
            } else {
                open.get_or_insert(n);
            }
            let k = next - n;
            for ch in 0..2 {
                a[ch] += c[ch] * k as f64;
                b[ch] *= if tails != 0 { rr.powi(k as i32) } else { 0.0 };
            }
        }
    }
}

/// The piece for the stretch `[from, end)` of a run, cut to `[lo, hi)`: `live` \[44\]
fn piece(from: u32, live: u32, end: u32, lo: u64, hi: u64, [a, c, b]: [[f64; 2]; 3], (tail, r): (bool, f32)) -> Option<Head> {
    let src = (from as u64).max(lo);
    let stop = (end as u64).min(hi);
    if stop <= src {
        return None;
    }
    let (src, len) = (src as u32, (stop - src) as u32);
    // Frames to the tail, or the whole piece when it has none here.
    let dead = if tail { live.saturating_sub(src).min(len) } else { len };
    // `b` where the tail is first heard, which is `f = 1` at `dead`.
    let late = src.saturating_sub(live);
    let scale = if late == 0 { 1.0 } else { (r as f64).powi(late as i32) };
    let level = |ch: usize| (a[ch] + c[ch] * (src as f64 - live as f64)) as f32;
    let live_here = dead < len;
    Some(Head {
        base: 0,
        out0: 0,
        src0: src,
        len,
        gain_l: level(0),
        gain_r: level(1),
        f0: if live_here { 1.0 } else { 0.0 },
        r,
        hold: dead.saturating_add(1),
        bus: 0,
        dead,
        cap: 0,
        fade: 0.0,
        dg_l: c[0] as f32,
        dg_r: c[1] as f32,
        gi_l: if live_here { -(b[0] * scale) as f32 } else { 0.0 },
        gi_r: if live_here { -(b[1] * scale) as f32 } else { 0.0 },
        split: 0,
        weight: 1,
    })
}

/// The live [`HeadNote`]s, filed under every partition step they reach. \[45\]
pub struct HeadSteps {
    /// The step, as `frame / partition`, that `lists[0]` is. Everything before \[46\]
    base: u64,
    lists: std::collections::VecDeque<Vec<HeadNote>>,
    /// Emptied lists, kept for their capacity: a list is filled over the ~10 \[47\]
    spare: Vec<Vec<HeadNote>>,
    /// Records held across every list, counting a record once per step.
    held: usize,
    /// `log2` of the partition when it is a power of two, which it is wherever \[48\]
    shift: Option<u32>,
    step_len: u32,
}

impl HeadSteps {
    pub fn new(step_len: u32) -> Self {
        Self {
            base: 0,
            lists: Default::default(),
            spare: Vec::new(),
            held: 0,
            shift: step_len.is_power_of_two().then(|| step_len.trailing_zeros()),
            step_len,
        }
    }

    /// Records held, counted once per step they are filed under. This is the \[49\]
    pub fn len(&self) -> usize {
        self.held
    }

    pub fn is_empty(&self) -> bool {
        self.held == 0
    }

    /// File `h` under every step it reaches that has not been taken yet. \[50\]
    pub fn push(&mut self, h: HeadNote, cached: u32) {
        let (first, last) = match (h.reach(cached), self.shift) {
            (None, _) => return,
            (Some((lo, hi)), Some(s)) => (lo >> s, (hi - 1) >> s),
            (Some((lo, hi)), None) => {
                let l = self.step_len as u64;
                (lo / l, (hi - 1) / l)
            }
        };
        let first = first.max(self.base);
        if last < first {
            return;
        }
        let end = (last - self.base) as usize + 1;
        while self.lists.len() < end {
            self.lists.push_back(self.spare.pop().unwrap_or_default());
        }
        let at = (first - self.base) as usize;
        for list in self.lists.range_mut(at..end) {
            list.push(h);
        }
        self.held += end - at;
    }

    /// The records reaching `step`, in the order they were filed, and every \[51\]
    pub fn take(&mut self, step: u64) -> Vec<HeadNote> {
        while self.base < step {
            if let Some(mut l) = self.lists.pop_front() {
                self.held -= l.len();
                l.clear();
                self.spare.push(l);
            }
            self.base += 1;
        }
        if self.base > step {
            return Vec::new();
        }
        self.base += 1;
        match self.lists.pop_front() {
            Some(l) => {
                self.held -= l.len();
                l
            }
            None => self.spare.pop().unwrap_or_default(),
        }
    }

    pub fn recycle(&mut self, mut list: Vec<HeadNote>) {
        // [52]
        let keep = (list.len() * 2).max(4096);
        list.clear();
        list.shrink_to(keep);
        if self.spare.len() < 16 {
            self.spare.push(list);
        }
    }
}

/// Sustaining waveforms, packed into a pool of pages that are **reused**. \[53\]
pub struct HeadCache {
    frames: usize,
    /// Frames per page. Small, because the point is that the median waveform \[54\]
    page: usize,
    ids: HashMap<EntryKey, u32>,
    /// Frames of each *waveform* any note has asked for, so a row rebound later \[55\]
    want: Vec<u32>,
    /// Where each waveform's frames are, when it has any: `(first page, pages)`.
    at: Vec<Option<(u32, u32)>>,
    /// Frames actually written, per waveform, into the pages it currently \[56\]
    filled: Vec<u32>,
    free: Vec<Vec<(u32, u32)>>,
    pages: usize,
    pub res: crate::merlin::residency::Residency,
    /// Bytes uploaded refilling waveforms whose pages had been taken. The cost \[57\]
    pub refilled_bytes: u64,
    pub buffer: wgpu::Buffer,
}

impl HeadCache {
    /// `rows` is a budget in *full-length rows*, kept as the unit the flag and \[58\]
    pub fn new(device: &wgpu::Device, rows: usize, frames: usize) -> Self {
        // [59]
        let page = 1024.min(frames.max(1));
        let pages = rows * frames.div_ceil(page);
        Self {
            frames,
            page,
            ids: HashMap::new(),
            want: Vec::new(),
            at: Vec::new(),
            filled: Vec::new(),
            free: vec![vec![(0u32, pages as u32)]],
            pages,
            res: Default::default(),
            refilled_bytes: 0,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("merlin::head::cache"),
                size: (pages * page * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Frames of `A[n]` a single waveform may hold. \[60\]
    pub fn frames(&self) -> u32 {
        self.frames as u32
    }

    /// Pages held and the pool's capacity, for the caller's report.
    pub fn pages_used(&self) -> (usize, usize) {
        let free: usize = self.free[0].iter().map(|&(_, len)| len as usize).sum();
        (self.pages - free, self.pages)
    }

    pub fn page_frames(&self) -> usize {
        self.page
    }

    /// The logical id for `wave`, assigning one on first sight. \[61\]
    pub fn intern(&mut self, wave: EntryKey, need: u32) -> u32 {
        let id = match self.ids.get(&wave) {
            Some(&id) => id,
            None => {
                let id = self.ids.len() as u32;
                self.ids.insert(wave, id);
                self.want.push(0);
                self.at.push(None);
                self.filled.push(0);
                id
            }
        };
        self.want_frames(id, need);
        id
    }

    /// Note that some note wants `need` frames of waveform `id`.
    #[inline]
    pub fn want_frames(&mut self, id: u32, need: u32) {
        let w = &mut self.want[id as usize];
        *w = (*w).max(need.min(self.frames as u32));
    }

    /// Give waveform `id` pages and fill them to what has been asked for, \[62\]
    pub fn bind(
        &mut self,
        queue: &wgpu::Queue,
        id: u32,
        sustain: &[f32],
    ) -> Option<(u32, u32, bool)> {
        let want = (self.want[id as usize] as usize).min(self.frames).min(sustain.len()).max(1);
        // [63]
        let need = want.div_ceil(self.page).next_power_of_two() as u32;
        let mut hit = true;
        // [64]
        if let Some((_, have)) = self.at[id as usize] {
            if have < need {
                self.drop_pages(id);
                hit = false;
            }
        }
        if self.at[id as usize].is_none() {
            hit = false;
            loop {
                if let Some((_, start)) =
                    crate::merlin::residency::take_run(&mut self.free, need)
                {
                    self.at[id as usize] = Some((start, need));
                    self.filled[id as usize] = 0;
                    break;
                }
                let victim = self.res.victim()?;
                self.drop_pages(victim);
            }
        }
        self.res.admit(id);
        self.res.pin(id);
        let (start, _) = self.at[id as usize]?;
        let base = start as usize * self.page;
        let have = self.filled[id as usize] as usize;
        if want > have {
            queue.write_buffer(
                &self.buffer,
                ((base + have) * 4) as u64,
                bytemuck::cast_slice(&sustain[have..want]),
            );
            self.filled[id as usize] = want as u32;
            if !hit {
                self.refilled_bytes += ((want - have) * 4) as u64;
            }
        }
        Some((base as u32, (need as usize * self.page) as u32, hit))
    }

    fn drop_pages(&mut self, id: u32) {
        if let Some((start, len)) = self.at[id as usize].take() {
            crate::merlin::residency::put_run(&mut self.free[0], start, len);
            self.filled[id as usize] = 0;
        }
    }

    /// Distinct waveforms seen, which is not the same as waveforms held.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// One still-open note's contribution to one partition step: the waveform, with \[65\]
#[allow(clippy::too_many_arguments)]
pub fn sustaining(
    wave: u32,
    bus: u32,
    gain: [f32; 2],
    d_gain: [f32; 2],
    gain_from: u64,
    note_start: u64,
    note_end: u64,
    step_start: u64,
    step_len: u32,
    cached: u32,
    weight: u32,
) -> Option<Head> {
    let lo = note_start.max(step_start);
    let hi = note_end.min(step_start + step_len as u64).min(note_start + cached as u64);
    if hi <= lo {
        return None;
    }
    // [66]
    let off = lo.saturating_sub(gain_from) as f32;
    let (gl, gr) = (gain[0] + d_gain[0] * off, gain[1] + d_gain[1] * off);
    Some(Head {
        base: wave,
        out0: (lo - step_start) as u32,
        src0: (lo - note_start) as u32,
        len: (hi - lo) as u32,
        gain_l: gl,
        gain_r: gr,
        f0: 0.0,
        r: 0.0,
        hold: 0,
        bus,
        // [67]
        dead: (hi - lo) as u32,
        cap: 0,
        fade: 0.0,
        dg_l: d_gain[0],
        dg_r: d_gain[1],
        // [68]
        gi_l: gl,
        gi_r: gr,
        split: 0,
                weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic generator, so these tests need no dependency and \[69\]
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound.max(1)
        }
    }

    fn note(rng: &mut Lcg, wave: u32) -> HeadNote {
        let d = rng.next(9000) as u32;
        let kind = rng.next(4);
        HeadNote {
            wave,
            start: rng.next(60_000),
            // [70]
            from: if rng.next(3) == 0 { rng.next(d as u64 + 200) as u32 } else { 0 },
            d,
            dj: rng.next(9000) as u32,
            r: 0.9,
            gain: [1.0, 0.5],
            gain_end: [1.0, 0.5],
            fade: if kind == 1 { 1 + rng.next(300) as u16 } else { 0 },
            release: if kind == 2 { 1 + rng.next(3000) as u16 } else { 0 },
            bus: 2,
            weight: 1,
            env_ratio: 1.0,
        }
    }

    /// `steps` is what `HeadSteps` files by, and `clip` is what plays: they \[71\]
    #[test]
    fn steps_agree_with_clip() {
        let mut rng = Lcg(7);
        for p in [2048u32, 1024, 96] {
            for cached in [480_000u32, 5_000, 1] {
                for i in 0..400 {
                    let h = note(&mut rng, i);
                    let steps = h.steps(p, cached);
                    for s in 0..(70_000 / p as u64 + 4) {
                        let pieces = h.clip(s * p as u64, p, cached);
                        let mask = h.pieces(s * p as u64, p, cached);
                        assert_eq!(
                            mask,
                            u8::from(pieces[0].is_some()) | (u8::from(pieces[1].is_some()) << 1),
                            "pieces: p {p} cached {cached} step {s} for {h:?}"
                        );
                        let hit = pieces.iter().any(|x| x.is_some());
                        let named = steps.is_some_and(|(a, b)| a <= s && s <= b);
                        assert_eq!(
                            hit, named,
                            "p {p} cached {cached} step {s}: clip says {hit}, steps {steps:?} for {h:?}"
                        );
                    }
                }
            }
        }
    }

    /// What `pieces` add to each frame of a step, read through a waveform of \[72\]
    fn play(pieces: &[Head], out: &mut [[f64; 2]]) {
        for h in pieces {
            let r = h.r as f64;
            let h1 = h.hold.max(1);
            for j in 0..h.len {
                let m = if j >= h1 { j - h1 + 1 } else { 0 };
                let f = if m == 0 {
                    h.f0 as f64
                } else if h.r == 1.0 {
                    h.f0 as f64 + h.fade as f64 * m as f64
                } else {
                    let rm = if h.r > 0.0 { r.powi(m as i32) } else { 0.0 };
                    h.f0 as f64 * rm + h.fade as f64 * (1.0 - rm) / (1.0 - r)
                };
                let fe = if j >= h.dead { f } else { 0.0 };
                let o = &mut out[(h.out0 + j) as usize];
                let w = h.weight as f64;
                o[0] += (h.gain_l as f64 + h.dg_l as f64 * j as f64 - fe * h.gi_l as f64) * w;
                o[1] += (h.gain_r as f64 + h.dg_r as f64 * j as f64 - fe * h.gi_r as f64) * w;
            }
        }
    }

    /// A note released on the head and cut in its tail plays its body, its \[73\]
    #[test]
    fn a_release_cut_on_the_head_plays_three_pieces() {
        let h = HeadNote { wave: 0, start: 40, from: 0, d: 1000, dj: 900, r: 0.999, gain: [0.5, 0.25], gain_end: [0.4, 0.2], fade: 64, release: 300, bus: 0, weight: 3, env_ratio: 1.0 };
        let span = h.span() as usize;
        assert_eq!(span, 1364);
        let (r, level) = (h.r as f64, (h.r as f64).powi(301));
        for step in [100_000u32, 977, 97, 1] {
            let mut out = vec![[0f64; 2]; 40 + span + 8];
            for s in 0..(40 + span + 8) as u32 / step + 1 {
                let at = (s * step) as usize;
                if at < out.len() {
                    play(&h.played(at as u64, step, u32::MAX).collect::<Vec<_>>(), &mut out[at..]);
                }
            }
            for (i, o) in out.iter().enumerate() {
                let n = i as i64 - 40;
                for (ch, &got) in o.iter().enumerate() {
                    let (g, e) = (h.gain[ch] as f64, h.gain_end[ch] as f64);
                    let want = if n < 0 {
                        0.0
                    } else if n < 1000 {
                        g + (e - g) * n as f64 / 1000.0
                    } else if n < 1300 {
                        e * r.powi((n - 1000 + 1) as i32)
                    } else if n < 1364 {
                        e * level * (1.0 - (n - 1300) as f64 / 64.0)
                    } else {
                        0.0
                    };
                    assert!((got - want * 3.0).abs() < 2e-5, "steps of {step}, frame {n}: {got} against {}", want * 3.0);
                }
            }
        }
    }

    /// A chord of `n` notes struck together, of every kind `plan_key` makes.
    fn chord(rng: &mut Lcg, n: usize, start: u64, r: f32, width: u32) -> Vec<HeadNote> {
        (0..n)
            .map(|_| {
                let d = 1 + rng.next(20_000) as u32;
                let (j, flat) = (d / width, 200);
                let dj = (j * width).max(flat);
                let gain = [0.05 + rng.next(100) as f32 / 400.0, 0.05 + rng.next(100) as f32 / 400.0];
                let gain_end = if rng.next(3) == 0 { [gain[0] * 0.5, gain[1] * 1.25] } else { gain };
                let cut = rng.next(5) == 0;
                // `plan_key`'s rule: past this gain the note has no kernel.
                let on_head = !cut && r > 0.0 && r < 1.0 && dj < d && (r as f64).powi(dj as i32 - d as i32) > 16.0;
                HeadNote {
                    wave: 3,
                    start,
                    from: 0,
                    d,
                    dj,
                    r,
                    gain,
                    gain_end,
                    fade: if cut { 96 } else { 0 },
                    release: if on_head { 1 + rng.next(4000) as u16 } else { 0 },
                    bus: 2,
                    weight: 1 + rng.next(4) as u16,
                    env_ratio: if dj > d { 0.25 + rng.next(3) as f32 / 4.0 } else { 1.0 },
                }
            })
            .collect()
    }

    /// `Fold`'s pieces play what each note's own pieces play, for every kind of \[74\]
    #[test]
    fn a_folded_run_plays_what_its_notes_play() {
        let mut rng = Lcg(23);
        let (step_len, start) = (4096u32, 30_000u64);
        let mut fold = Fold::default();
        let mut worst = 0f64;
        for round in 0..600 {
            let r = [0.9995, 0.998, 0.9, 1.0, 0.0][round % 5];
            let notes = chord(&mut rng, 2 + round % 40, start, r, 300 + 700 * (round as u32 % 3));
            for step in [28_672u64, 30_000, 32_768, 36_864, 45_056] {
                let mut each = Vec::new();
                for h in &notes {
                    each.extend(h.clip(step, step_len, u32::MAX).into_iter().flatten());
                }
                let mut folded = Vec::new();
                fold.run(notes.iter(), step, step_len, &mut folded);
                let (mut a, mut b) = (vec![[0f64; 2]; step_len as usize], vec![[0f64; 2]; step_len as usize]);
                play(&each, &mut a);
                play(&folded, &mut b);
                for (x, y) in a.iter().zip(&b) {
                    for ch in 0..2 {
                        worst = worst.max((x[ch] - y[ch]).abs());
                        assert!((x[ch] - y[ch]).abs() < 1e-4, "r {r}, step {step}, round {round}: {} against {}", x[ch], y[ch]);
                    }
                }
                assert!(folded.iter().all(|h| h.base == 3 && h.out0 + h.len <= step_len), "a piece outside the step");
            }
        }
        assert!(worst > 0.0, "nothing was compared");
    }

    /// Notes with one length are one piece in the step, however many there are \[75\]
    #[test]
    fn a_stack_of_one_length_is_one_piece() {
        let h = HeadNote {
            wave: 1,
            start: 100,
            from: 0,
            d: 3000,
            dj: 2000,
            r: 0.999,
            gain: [0.5, 0.25],
            gain_end: [0.5, 0.25],
            fade: 0,
            release: 0,
            bus: 0,
            weight: 2,
            env_ratio: 1.0,
        };
        let notes = vec![h; 50];
        let mut out = Vec::new();
        Fold::default().run(notes.iter(), 0, 8192, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!((out[0].dead, out[0].len, out[0].out0), (2000, 3000, 100));
        assert!((out[0].gain_l - 50.0).abs() < 1e-3, "{}", out[0].gain_l);
    }

    /// The contract of `HeadSteps`: step by step, the list it hands over is the \[76\]
    #[test]
    fn each_step_gets_what_a_scan_would_find_in_the_same_order() {
        let (p, cached) = (2048u32, 20_000u32);
        let mut rng = Lcg(11);
        let mut all: Vec<HeadNote> = Vec::new();
        let mut steps = HeadSteps::new(p);
        // [77]
        let mut next_step = 0u64;
        for round in 0..60u64 {
            for _ in 0..rng.next(40) {
                let wave = rng.next(50) as u32;
                let mut h = note(&mut rng, wave);
                h.start += round * p as u64 * 2;
                // [78]
                let floor = next_step * p as u64;
                if h.start + h.from as u64 + 1 < floor {
                    h.from = (floor - h.start) as u32;
                }
                all.push(h);
                steps.push(h, cached);
            }
            for _ in 0..2 {
                let s = next_step;
                next_step += 1;
                let want: Vec<(u32, u64)> = all
                    .iter()
                    .filter(|h| h.clip(s * p as u64, p, cached).iter().any(|x| x.is_some()))
                    .map(|h| (h.wave, h.start))
                    .collect();
                let list = steps.take(s);
                let got: Vec<(u32, u64)> = list.iter().map(|h| (h.wave, h.start)).collect();
                assert_eq!(got, want, "step {s}");
                steps.recycle(list);
            }
        }
        assert!(!steps.is_empty(), "the test never left anything filed for a later step");
    }
}
