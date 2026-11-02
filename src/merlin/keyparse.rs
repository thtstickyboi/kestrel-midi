// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Merlin's own MIDI front end for the per-key render: the file read straight \[1\]

use crate::bank::Bank;
use crate::config::Config;
use crate::merlin::keystore::Ev;
use crate::midi::{Division, Event, SmfHeader, TempoClock, TrackReader};
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Count buckets to a quarter note: a sixty-fourth note, the finest a window \[2\]
pub const BUCKETS_PER_QN: u64 = 16;

/// MIDI keys.
const KEYS: usize = 128;

/// No pairing: a key outside the group.
const NONE: u8 = u8::MAX;
/// The read windows a group pass holds for all its tracks together. It runs \[3\]
const EXTRACT_READ_BUDGET: u64 = crate::midi::READ_BUDGET / 4;

/// Channels once ports are folded in: 16 ports of 16.
const CHANNELS: usize = 256;

/// What a controller does to note ends, in stream order with the tempo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ctl {
    Tempo(u32),
    Sustain(bool),
    Sostenuto(bool),
    /// CC120.
    SoundOff,
    /// CC121.
    Reset,
    /// CC123 to 127.
    NotesOff,
    /// Any other channel event, for the channel state records read.
    Chan(Event),
}

/// An event's place in the stream: the merge takes ticks in order, ties by \[4\]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Place {
    tick: u64,
    track: u32,
    seq: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CtlEv {
    at: Place,
    ch: u8,
    ctl: Ctl,
}

/// A controller on disk.
const CTL_BYTES: usize = 32;

impl CtlEv {
    /// The bytes of a controller in a run on disk.
    fn pack(&self) -> [u8; CTL_BYTES] {
        let (tag, a, b, d, c): (u8, u8, u8, u8, i32) = match self.ctl {
            Ctl::Tempo(us) => (0, 0, 0, 0, us as i32),
            Ctl::Sustain(down) => (1, down as u8, 0, 0, 0),
            Ctl::Sostenuto(down) => (2, down as u8, 0, 0, 0),
            Ctl::SoundOff => (3, 0, 0, 0, 0),
            Ctl::Reset => (4, 0, 0, 0, 0),
            Ctl::NotesOff => (5, 0, 0, 0, 0),
            Ctl::Chan(Event::Cc { num, val, .. }) => (6, num, val, 0, 0),
            Ctl::Chan(Event::Program { val, .. }) => (7, val, 0, 0, 0),
            Ctl::Chan(Event::PitchBend { val, .. }) => (8, 0, 0, 0, val as i32),
            Ctl::Chan(Event::DrumPart { map, .. }) => (9, map, 0, 0, 0),
            Ctl::Chan(Event::Tune { pc, units, .. }) => (10, pc, 0, 0, units),
            Ctl::Chan(Event::KeyTune { bank, prog, key, units }) => (11, bank, prog, key, units),
            Ctl::Chan(Event::ResetParts) => (12, 0, 0, 0, 0),
            Ctl::Chan(_) => unreachable!("`ctl_of` keeps no other event"),
        };
        let mut o = [0u8; CTL_BYTES];
        o[..8].copy_from_slice(&self.at.tick.to_le_bytes());
        o[8..16].copy_from_slice(&self.at.seq.to_le_bytes());
        o[16..20].copy_from_slice(&self.at.track.to_le_bytes());
        o[20..24].copy_from_slice(&c.to_le_bytes());
        o[24] = self.ch;
        o[25] = tag;
        o[26] = a;
        o[27] = b;
        o[28] = d;
        o
    }

    fn unpack(b: &[u8]) -> Self {
        let word = |r: std::ops::Range<usize>| -> [u8; 8] {
            let mut w = [0u8; 8];
            w[..r.len()].copy_from_slice(&b[r]);
            w
        };
        let at = Place { tick: u64::from_le_bytes(word(0..8)), seq: u64::from_le_bytes(word(8..16)), track: u64::from_le_bytes(word(16..20)) as u32 };
        let c = u64::from_le_bytes(word(20..24)) as u32 as i32;
        let (ch, x, y, z) = (b[24], b[26], b[27], b[28]);
        let ctl = match b[25] {
            0 => Ctl::Tempo(c as u32),
            1 => Ctl::Sustain(x != 0),
            2 => Ctl::Sostenuto(x != 0),
            3 => Ctl::SoundOff,
            4 => Ctl::Reset,
            5 => Ctl::NotesOff,
            6 => Ctl::Chan(Event::Cc { ch, num: x, val: y }),
            7 => Ctl::Chan(Event::Program { ch, val: x }),
            8 => Ctl::Chan(Event::PitchBend { ch, val: c as i16 }),
            9 => Ctl::Chan(Event::DrumPart { ch, map: x }),
            10 => Ctl::Chan(Event::Tune { ch, pc: x, units: c }),
            11 => Ctl::Chan(Event::KeyTune { bank: x, prog: y, key: z, units: c }),
            _ => Ctl::Chan(Event::ResetParts),
        };
        Self { at, ch, ctl }
    }
}

/// Controllers a track holds before it hands them to the scan's `CtlSink`.
const CTL_TRACK_BATCH: usize = 1 << 16;

/// Controllers of one run read at a time.
const CTL_READ: usize = 1 << 16;

/// A channel's pitch along the song as a clock: `tau` is how far a note on it \[5\]
#[derive(Clone, Debug)]
pub struct Warp {
    /// Each stretch of one pitch: its first frame, `tau` there, `tau` a frame.
    at: Vec<(u64, f64, f64)>,
}

impl Default for Warp {
    fn default() -> Self {
        Self { at: vec![(0, 0.0, 1.0)] }
    }
}

impl Warp {
    /// From `frame` on the pitch is `ratio`. Frames arrive in order.
    fn change(&mut self, frame: u64, ratio: f64) {
        let &(f0, t0, r0) = self.at.last().expect("a first stretch");
        if frame == f0 {
            self.at.last_mut().expect("a first stretch").2 = ratio;
        } else {
            self.at.push((frame, t0 + (frame - f0) as f64 * r0, ratio));
        }
    }

    /// Whether it ever moves.
    pub fn moves_at_all(&self) -> bool {
        self.at.len() > 1
    }

    fn stretch(&self, f: u64) -> usize {
        self.at.partition_point(|s| s.0 <= f) - 1
    }

    /// `tau` at frame `f`.
    pub fn tau(&self, f: u64) -> f64 {
        let (f0, t0, r) = self.at[self.stretch(f)];
        t0 + (f - f0) as f64 * r
    }

    /// The frame at `tau`, not rounded.
    pub fn frame(&self, tau: f64) -> f64 {
        let (f0, t0, r) = self.at[self.at.partition_point(|s| s.1 <= tau).max(1) - 1];
        f0 as f64 + (tau - t0) / r
    }

    /// Whether the pitch changes after frame `a`, up to frame `b`.
    pub fn moves(&self, a: u64, b: u64) -> bool {
        self.stretch(a) != self.stretch(b)
    }

    /// The song's frames `[lo, hi)` as runs of at most `most`, none across a \[6\]
    pub fn runs(&self, lo: u64, hi: u64, most: u64) -> Vec<[u32; 5]> {
        let mut out = Vec::new();
        let mut f = lo;
        while f < hi {
            let i = self.stretch(f);
            let (f0, t0, r) = self.at[i];
            let stop = self.at.get(i + 1).map_or(u64::MAX, |s| s.0);
            let n = most.min(stop - f).min(hi - f);
            let t = t0 + (f - f0) as f64 * r;
            let whole = t.floor();
            out.push([f as u32, whole as u32, ((t - whole) as f32).to_bits(), (r as f32).to_bits(), n as u32]);
            f += n;
        }
        out
    }
}

/// `Config::merlin_km_ctl_held`'s default: about 450 MiB.
pub const CTL_HELD: u32 = 8 << 20;

/// Controllers worth a note-on, for what a pass window holds.
const CTL_WEIGHT: u32 = 4;

/// The controllers on disk: runs, each sorted, each `(offset, bytes)`.
struct CtlDisk {
    file: std::fs::File,
    next: u64,
    runs: Vec<(u64, u64)>,
}

/// The scan's controllers: those in memory, and, past the budget and where \[7\]
#[derive(Default)]
struct CtlSink {
    held: Vec<CtlEv>,
    disk: Option<CtlDisk>,
    count: u64,
    /// The tempo changes and CC120s, which always stay here: what placing a \[8\]
    marks: Vec<CtlEv>,
    pitch: bool,
}

impl CtlSink {
    /// `evs` taken on. What is held past `budget` goes to a run; if it cannot, \[9\]
    fn add(&mut self, evs: &mut Vec<CtlEv>, budget: usize, dir: Option<&Path>) {
        self.count += evs.len() as u64;
        let pitch = self.pitch;
        self.marks.extend(evs.iter().filter(|c| match c.ctl {
            Ctl::Tempo(_) | Ctl::SoundOff => true,
            Ctl::Reset | Ctl::Chan(Event::PitchBend { .. }) => pitch,
            Ctl::Chan(Event::Cc { num, .. }) => pitch && matches!(num, 6 | 38 | 96..=101),
            _ => false,
        }));
        self.held.append(evs);
        let Some(dir) = dir.filter(|_| self.held.len() >= budget) else { return };
        if self.disk.is_none() {
            match crate::merlin::keystore::temp_file(dir) {
                Ok(file) => self.disk = Some(CtlDisk { file, next: 0, runs: Vec::new() }),
                Err(e) => {
                    log::warn!("merlin's controllers stay in memory, as no file could be made for them: {e}");
                    return;
                }
            }
        }
        // By hand, not with rayon: this runs under the sink's lock, on a worker.
        self.held.sort_unstable_by_key(|c| c.at);
        let d = self.disk.as_mut().expect("made above");
        let mut bytes = Vec::with_capacity(self.held.len() * CTL_BYTES);
        for c in &self.held {
            bytes.extend_from_slice(&c.pack());
        }
        match crate::merlin::keystore::write_all_at(&d.file, &bytes, d.next) {
            Ok(()) => {
                d.runs.push((d.next, bytes.len() as u64));
                d.next += bytes.len() as u64;
                self.held.clear();
            }
            Err(e) => log::warn!("merlin's controllers stay in memory, as they could not be written: {e}"),
        }
    }
}

/// One run on disk as a pass walks it.
struct CtlRun {
    off: u64,
    left: u64,
    buf: Vec<CtlEv>,
    i: usize,
}

impl CtlRun {
    fn head(&mut self, file: &std::fs::File, bytes: &mut Vec<u8>) -> Result<Option<CtlEv>> {
        if self.i == self.buf.len() {
            if self.left == 0 {
                return Ok(None);
            }
            let n = self.left.min((CTL_READ * CTL_BYTES) as u64) as usize;
            bytes.resize(n, 0);
            crate::merlin::bucket::read_exact_at(file, bytes, self.off)?;
            self.off += n as u64;
            self.left -= n as u64;
            self.buf.clear();
            self.buf.extend(bytes.chunks_exact(CTL_BYTES).map(CtlEv::unpack));
            self.i = 0;
        }
        Ok(Some(self.buf[self.i]))
    }
}

/// A pass's walk along the file's controllers in stream order, a window at a \[10\]
struct CtlWalk<'a> {
    mem: &'a [CtlEv],
    file: Option<&'a std::fs::File>,
    runs: Vec<CtlRun>,
    bytes: Vec<u8>,
    out: Vec<CtlEv>,
}

impl<'a> CtlWalk<'a> {
    /// The controllers before tick `end`.
    fn window(&mut self, end: u64) -> Result<&[CtlEv]> {
        let Some(file) = self.file else {
            let n = self.mem.partition_point(|c| c.at.tick < end);
            let (now, rest) = self.mem.split_at(n);
            self.mem = rest;
            return Ok(now);
        };
        self.out.clear();
        loop {
            let mut best: Option<(usize, CtlEv)> = None;
            for i in 0..self.runs.len() {
                if let Some(c) = self.runs[i].head(file, &mut self.bytes)? {
                    if c.at.tick < end && best.is_none_or(|(_, b)| c.at < b.at) {
                        best = Some((i, c));
                    }
                }
            }
            let m = self.mem.first().copied().filter(|c| c.at.tick < end);
            match (m, best) {
                (Some(c), Some((_, b))) if c.at < b.at => {
                    self.out.push(c);
                    self.mem = &self.mem[1..];
                }
                (Some(c), None) => {
                    self.out.push(c);
                    self.mem = &self.mem[1..];
                }
                (_, Some((i, b))) => {
                    self.out.push(b);
                    self.runs[i].i += 1;
                }
                (None, None) => return Ok(&self.out),
            }
        }
    }
}

/// Windows that wait between one stage of a pass and the next.
const STAGE_AHEAD: usize = 2;

/// A window as each worker read it, by pairing.
type ReadBufs = Vec<Vec<Vec<NoteEv>>>;

/// A window merged, per pairing, and the counting sort's bins.
type MergeBufs = Vec<(Vec<NoteEv>, Vec<u32>)>;

/// `Config::merlin_told_events`' default: events a track's worker reads before \[11\]
pub const TOLD_EVERY: u32 = 1 << 20;

/// Who the pre-pass tells how it is getting on, and who can stop it. The \[12\]
pub trait Watch: Send + Sync {
    /// The pre-pass begins over `bytes` of tracks.
    fn begin(&self, bytes: u64);
    /// `bytes` more read and `notes` more note-ons counted, from any worker, at \[13\]
    fn tell(&self, bytes: u64, notes: u64) -> Result<()>;
    /// Whether the render has been stopped, which the passes after this one \[14\]
    fn stopped(&self) -> bool;
}

impl Watch for crate::merlin::progress::MerlinProgress {
    fn begin(&self, bytes: u64) {
        self.begin_arming(bytes);
    }

    fn tell(&self, bytes: u64, notes: u64) -> Result<()> {
        self.check()?;
        self.add_pre(bytes, notes);
        Ok(())
    }

    fn stopped(&self) -> bool {
        self.is_cancelled()
    }
}

/// What the pre-pass read.
pub struct Prepass {
    path: PathBuf,
    header: SmfHeader,
    /// The controllers in memory, sorted; with `ctl_disk`, the runs past them.
    ctls: Vec<CtlEv>,
    ctl_disk: Option<CtlDisk>,
    ctl_count: u64,
    /// Controllers per bucket, which a pass's windows are cut by too.
    ctl_counts: Vec<u32>,
    /// Each channel's CC120s, as frames in stream order. A note leaves pairing \[15\]
    cuts: std::sync::Arc<Vec<Vec<u64>>>,
    /// What placing each channel's pitch along the song needs, with \[16\]
    marks: Vec<CtlEv>,
    clock: TempoClock,
    /// Every key's note events, when they could be filed.
    store: Option<crate::merlin::keystore::KeyStore>,
    /// Note-ons per (bucket, key).
    counts: Vec<[u32; KEYS]>,
    bucket_ticks: u64,
    /// The tick of the stream's last event that is not a tempo change, the \[17\]
    last_tick: Option<u64>,
    /// A per-key render's progress and stop, which the pre-pass and the passes \[18\]
    watch: Option<Arc<dyn Watch>>,
    /// Frames the song runs, by its last event and its tempo changes.
    song_frames: u64,
    pub notes: u64,
    pub events: u64,
    pub secs: f64,
}

/// One worker's share of the pre-pass.
#[derive(Default)]
struct TrackScan {
    counts: Vec<[u32; KEYS]>,
    ctls: Vec<u32>,
    last_tick: Option<u64>,
    notes: u64,
    events: u64,
}

impl TrackScan {
    /// Two shares as one, into the larger table.
    fn merge(self, o: Self) -> Self {
        let (mut big, small) = if self.counts.len() >= o.counts.len() { (self, o) } else { (o, self) };
        for (to, from) in big.counts.iter_mut().zip(&small.counts) {
            for (t, &f) in to.iter_mut().zip(from) {
                *t = t.saturating_add(f);
            }
        }
        if big.ctls.len() < small.ctls.len() {
            big.ctls.resize(small.ctls.len(), 0);
        }
        for (t, &f) in big.ctls.iter_mut().zip(&small.ctls) {
            *t = t.saturating_add(f);
        }
        big.notes += small.notes;
        big.events += small.events;
        big.last_tick = match (big.last_tick, small.last_tick) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        big
    }
}

fn ctl_of(ev: &Event) -> Option<(u8, Ctl)> {
    match *ev {
        Event::Tempo(us) => Some((0, Ctl::Tempo(us))),
        Event::Cc { ch, num, val } => match num {
            64 => Some((ch, Ctl::Sustain(val >= 64))),
            66 => Some((ch, Ctl::Sostenuto(val >= 64))),
            120 => Some((ch, Ctl::SoundOff)),
            121 => Some((ch, Ctl::Reset)),
            123..=127 => Some((ch, Ctl::NotesOff)),
            _ => Some((ch, Ctl::Chan(*ev))),
        },
        Event::Program { ch, .. } | Event::PitchBend { ch, .. } | Event::DrumPart { ch, .. } | Event::Tune { ch, .. } => Some((ch, Ctl::Chan(*ev))),
        Event::ResetParts | Event::KeyTune { .. } => Some((0, Ctl::Chan(*ev))),
        _ => None,
    }
}

impl Prepass {
    pub fn scan(path: &Path, cfg: &Config) -> Result<Self> {
        Self::scan_with(path, cfg, None, None, None)
    }

    /// `scan`, and every note event filed by key in a temporary file in `dir`. \[19\]
    pub fn scan_filed(path: &Path, cfg: &Config, dir: &Path) -> Result<Self> {
        Self::scan_watched(path, cfg, dir, None)
    }

    /// `scan_filed`, reporting to `watch` as it reads and stopping, with \[20\]
    pub fn scan_watched(path: &Path, cfg: &Config, dir: &Path, watch: Option<Arc<dyn Watch>>) -> Result<Self> {
        use crate::merlin::keystore::{store_bytes, Filing};
        let need = std::fs::metadata(path).map_or(0, |m| store_bytes(m.len()));
        let free = fs4::available_space(dir).unwrap_or(u64::MAX);
        if free < need.saturating_add(cfg.merlin_disk_floor) {
            log::warn!(
                "merlin's key store needs about {:.1} GB and {} has {:.1} GB free, so the file is read a group of keys at a time, which is much slower on a big file; \
                 point TMP at a drive with room to avoid it",
                need as f64 / 1e9,
                dir.display(),
                free as f64 / 1e9
            );
            return Self::scan_with(path, cfg, None, Some(dir), watch.as_ref());
        }
        let filing = Filing::new(dir, cfg.merlin_disk_floor);
        if let Err(e) = &filing {
            log::warn!("merlin's key store could not be made, falling back to reading the file a group at a time: {e}");
        }
        Self::scan_with(path, cfg, filing.ok(), Some(dir), watch.as_ref())
    }

    /// Frames the song runs, by its last event and its tempo changes.
    pub fn song_frames(&self) -> u64 {
        self.song_frames
    }

    /// The render this feeds was stopped.
    fn stopped(&self) -> bool {
        self.watch.as_ref().is_some_and(|w| w.stopped())
    }

    /// The store's size and the events in it, when there is one.
    pub fn store_size(&self) -> Option<(u64, u64)> {
        self.store.as_ref().map(|s| (s.bytes, s.events))
    }

    fn scan_with(path: &Path, cfg: &Config, filing: Option<crate::merlin::keystore::Filing>, dir: Option<&Path>, watch: Option<&Arc<dyn Watch>>) -> Result<Self> {
        if cfg.edo31 {
            bail!("merlin's key parser does not read 31-EDO files yet");
        }
        if cfg.min_velocity > 1 {
            bail!("merlin's key parser does not skip quiet notes yet (--min-velocity)");
        }
        let t = std::time::Instant::now();
        let header = SmfHeader::read(path)?;
        if header.tracks.is_empty() {
            bail!("{}: no MTrk chunks", path.display());
        }
        let bucket_ticks = match header.division {
            Division::Ppq(ppq) => (ppq as u64 / BUCKETS_PER_QN).max(1),
            Division::Smpte { fps, ticks_per_frame } => (fps as u64 * ticks_per_frame as u64 / BUCKETS_PER_QN).max(1),
        };
        let lens: Vec<u64> = header.tracks.iter().map(|&(_, l)| l).collect();
        if let Some(w) = watch {
            w.begin(lens.iter().sum());
            w.tell(0, 0)?;
        }
        let window = crate::midi::window_share(&lens, crate::midi::READ_BUDGET);
        // [21]
        let held = cfg.merlin_km_ctl_held.max(1) as usize;
        let sink = std::sync::Mutex::new(CtlSink { pitch: cfg.merlin_km_warp, ..CtlSink::default() });
        // [22]
        let counted = std::sync::atomic::AtomicU64::new(0);
        let tally = |n: u64, bytes: u64| -> Result<()> {
            let seen = counted.fetch_add(n, std::sync::atomic::Ordering::Relaxed) + n;
            if seen > cfg.merlin_max_notes {
                bail!(
                    "{} has more than {} note-ons, merlin's limit: at least {seen} before it stopped reading, so at least {} over",
                    path.display(),
                    cfg.merlin_max_notes,
                    seen - cfg.merlin_max_notes
                );
            }
            if let Some(w) = watch {
                w.tell(bytes, n)?;
            }
            Ok(())
        };
        let scan: TrackScan = header
            .tracks
            .par_iter()
            .enumerate()
            .try_fold(
                TrackScan::default,
                |mut s, (track, &(off, len))| -> Result<_> {
                    let mut r = TrackReader::open(path, off, len, window)?;
                    let mut filer = filing.as_ref().map(|f| f.track(track as u32));
                    let mut mine = Vec::new();
                    let (mut seq, mut untold) = (0u64, 0u64);
                    let (mut quiet, mut told_bytes) = (0u32, 0u64);
                    let told_every = cfg.merlin_told_events.max(1);
                    while let Some(ev) = r.next_event() {
                        let tick = r.tick;
                        let at = Place { tick, track: track as u32, seq };
                        s.events += 1;
                        quiet += 1;
                        if quiet == told_every {
                            quiet = 0;
                            let at = len.saturating_sub(r.unread());
                            tally(std::mem::take(&mut untold), at.saturating_sub(told_bytes))?;
                            told_bytes = at;
                        }
                        if !matches!(ev, Event::Tempo(_)) {
                            s.last_tick = Some(s.last_tick.map_or(tick, |l| l.max(tick)));
                        }
                        if let Event::NoteOn { key, .. } = ev {
                            let b = (tick / bucket_ticks) as usize;
                            if s.counts.len() <= b {
                                s.counts.resize(b + 1, [0; KEYS]);
                            }
                            s.counts[b][key as usize & 127] += 1;
                            s.notes += 1;
                            untold += 1;
                        } else if let Some((ch, ctl)) = ctl_of(&ev) {
                            let b = (tick / bucket_ticks) as usize;
                            if s.ctls.len() <= b {
                                s.ctls.resize(b + 1, 0);
                            }
                            s.ctls[b] = s.ctls[b].saturating_add(1);
                            mine.push(CtlEv { at, ch, ctl });
                            seq += 1;
                            if mine.len() >= CTL_TRACK_BATCH {
                                sink.lock().expect("no scan panics").add(&mut mine, held, dir);
                            }
                        }
                        if let Some(f) = &mut filer {
                            match ev {
                                Event::NoteOn { ch, key, vel } => f.note(key, Ev { tick, seq, ch, vel, count: 1 }),
                                Event::NoteOff { ch, key } => f.note(key, Ev { tick, seq, ch, vel: 0, count: 1 }),
                                _ => {}
                            }
                        }
                    }
                    if let Some(f) = filer {
                        f.end();
                    }
                    if !mine.is_empty() {
                        sink.lock().expect("no scan panics").add(&mut mine, held, dir);
                    }
                    tally(untold, len.saturating_sub(told_bytes))?;
                    Ok(s)
                },
            )
            .try_reduce(TrackScan::default, |a, b| Ok(a.merge(b)))?;
        let TrackScan { counts, ctls: ctl_counts, last_tick, notes, events } = scan;
        let CtlSink { held: mut ctls, disk: ctl_disk, count: ctl_count, mut marks, .. } = sink.into_inner().expect("no scan panics");
        // [23]
        ctls.par_sort_unstable_by_key(|c| c.at);
        marks.sort_unstable_by_key(|c| c.at);
        let clock0 = TempoClock::with_timing(header.division, cfg.sample_rate, cfg.tempo_timing);
        let song_frames = last_tick.map_or(0, |t| {
            let mut c = clock0.clone();
            for m in &marks {
                if let Ctl::Tempo(us) = m.ctl {
                    c.set_tempo(m.at.tick, us);
                }
            }
            frame_of(&c, t)
        });
        let mut clock = clock0.clone();
        let mut cuts: Vec<Vec<u64>> = vec![Vec::new(); CHANNELS];
        if marks.iter().any(|c| c.ctl == Ctl::SoundOff) {
            for c in &marks {
                match c.ctl {
                    Ctl::Tempo(us) => clock.set_tempo(c.at.tick, us),
                    Ctl::SoundOff => cuts[c.ch as usize].push(frame_of(&clock, c.at.tick)),
                    _ => {}
                }
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            header,
            ctls,
            ctl_disk,
            ctl_count,
            ctl_counts,
            cuts: std::sync::Arc::new(cuts),
            marks: if cfg.merlin_km_warp { marks } else { Vec::new() },
            clock: clock0,
            store: filing.and_then(|f| f.finish()),
            counts,
            bucket_ticks,
            last_tick,
            watch: watch.cloned(),
            song_frames,
            notes,
            events,
            secs: t.elapsed().as_secs_f64(),
        })
    }

    /// Count buckets, and controllers kept.
    pub fn sizes(&self) -> (usize, usize) {
        (self.counts.len(), self.ctl_count as usize)
    }

    /// Note-ons on each key.
    pub fn key_notes(&self) -> [u64; KEYS] {
        let mut n = [0u64; KEYS];
        for c in &self.counts {
            for k in 0..KEYS {
                n[k] += c[k] as u64;
            }
        }
        n
    }

    /// Keys in groups of consecutive keys, each holding at most `budget` \[24\]
    pub fn groups(&self, budget: u64) -> Vec<Vec<u8>> {
        let n = self.key_notes();
        let mut out: Vec<Vec<u8>> = Vec::new();
        let mut held = 0u64;
        for (k, &nk) in n.iter().enumerate() {
            if nk == 0 {
                continue;
            }
            match out.last_mut() {
                Some(g) if held + nk <= budget => {
                    g.push(k as u8);
                    held += nk;
                }
                _ => {
                    out.push(vec![k as u8]);
                    held = nk;
                }
            }
        }
        out
    }

    /// Each channel's clock (`Warp`), by the channel state's own reading of \[25\]
    pub fn warps(&self, bank: &Bank, cfg: &Config) -> Vec<std::sync::Arc<Warp>> {
        let mut chans = crate::merlin::keyrec::Channels::new(bank, cfg);
        let mut clock = self.clock.clone();
        let mut out: Vec<Warp> = vec![Warp::default(); CHANNELS];
        for c in &self.marks {
            let ch = c.ch;
            let was = chans.bend(ch);
            match c.ctl {
                Ctl::Tempo(us) => {
                    clock.set_tempo(c.at.tick, us);
                    continue;
                }
                Ctl::Reset => chans.reset(ch),
                Ctl::Chan(ev) => chans.apply(bank, &ev),
                _ => continue,
            }
            if chans.bend(ch) != was {
                out[ch as usize].change(frame_of(&clock, c.at.tick), chans.bend(ch) as f64 / crate::fixed::BEND_ONE as f64);
            }
        }
        out.into_iter().map(std::sync::Arc::new).collect()
    }

    /// A pass's walk along the controllers.
    fn ctl_walk(&self) -> CtlWalk<'_> {
        let runs = self.ctl_disk.iter().flat_map(|d| &d.runs).map(|&(off, left)| CtlRun { off, left, buf: Vec::new(), i: 0 }).collect();
        CtlWalk { mem: &self.ctls, file: self.ctl_disk.as_ref().map(|d| &d.file), runs, bytes: Vec::new(), out: Vec::new() }
    }

    /// Window ends, in ticks, for `keys`: each window as many whole buckets as \[26\]
    fn windows(&self, keys: &[u8], notes: u64) -> Vec<u64> {
        let notes = notes.max(1);
        let mut ends = Vec::new();
        let mut held = 0u64;
        for (b, c) in self.counts.iter().enumerate() {
            let ctls = self.ctl_counts.get(b).map_or(0, |&n| n / CTL_WEIGHT);
            let n: u64 = keys.iter().map(|&k| c[k as usize] as u64).sum::<u64>() + ctls as u64;
            let b0 = b as u64 * self.bucket_ticks;
            if held > 0 && held + n > notes {
                ends.push(b0);
                held = 0;
            }
            if n > notes {
                let parts = n.div_ceil(notes).min(self.bucket_ticks);
                for i in 1..parts {
                    ends.push(b0 + self.bucket_ticks * i / parts);
                }
                ends.push(b0 + self.bucket_ticks);
                continue;
            }
            held += n;
        }
        ends.push(u64::MAX);
        ends
    }
}

/// Paired notes, as the driver pairs them: `count` identical copies on `slot` \[27\]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    pub start: u64,
    pub end: u64,
    pub slot: u32,
    pub vel: u8,
    pub count: u32,
    /// The channel's preset, gain, bend and CC71-75 at the note-on.
    pub preset: u32,
    pub gain_on: [f32; 2],
    pub bend: u32,
    pub sound: [u8; 5],
    /// The channel's gain at the release, which the release tail is fired at.
    pub gain_off: [f32; 2],
    /// Released by a CC120, which stops it rather than releasing it.
    pub cut: bool,
    /// The frame of the first CC120 on its channel after its release, which \[28\]
    pub cut_at: u64,
}

/// Copies of one note-on still sounding, oldest first in their slot's queue.
#[derive(Clone, Copy)]
struct Run {
    start: u64,
    vel: u8,
    count: u32,
    at: crate::merlin::keyrec::Strike,
}

/// The driver's `GateTable`, for one group's keys, with each slot's sounding \[29\]
struct Pairing {
    keys: Vec<u8>,
    /// Every channel's state, as records read it.
    chans: crate::merlin::keyrec::Channels,
    on: Vec<u32>,
    off: Vec<u32>,
    pending: Vec<u32>,
    sost_held: Vec<u32>,
    sustain: [bool; CHANNELS],
    sostenuto: [bool; CHANNELS],
    open: Vec<VecDeque<Run>>,
    /// Runs held across all slots, and the most at once.
    live: usize,
    peak_live: usize,
    /// Each slot's last release, held until the next one differs, so copies \[30\]
    out: Vec<Option<Note>>,
    /// Notes handed to the sink, and the copies they stand for.
    runs: u64,
    copies: u64,
    note_ons: u64,
    /// Notes released since the caller last took them.
    emitted: Vec<Note>,
    /// Each channel's CC120 frames, and how many of them the stream has passed.
    cuts: std::sync::Arc<Vec<Vec<u64>>>,
    cuts_seen: [u32; CHANNELS],
}

impl Pairing {
    fn new(keys: Vec<u8>, chans: crate::merlin::keyrec::Channels, cuts: std::sync::Arc<Vec<Vec<u64>>>) -> Self {
        let slots = CHANNELS * KEYS;
        Self {
            keys,
            chans,
            on: vec![0; slots],
            off: vec![0; slots],
            pending: vec![0; slots],
            sost_held: vec![0; slots],
            sustain: [false; CHANNELS],
            sostenuto: [false; CHANNELS],
            open: (0..slots).map(|_| VecDeque::new()).collect(),
            live: 0,
            peak_live: 0,
            out: vec![None; slots],
            runs: 0,
            copies: 0,
            note_ons: 0,
            emitted: Vec::new(),
            cuts,
            cuts_seen: [0; CHANNELS],
        }
    }

    fn slot(ch: u8, key: u8) -> usize {
        ch as usize * KEYS + (key as usize & 127)
    }

    fn note_on(&mut self, ch: u8, key: u8, vel: u8, frame: u64, n: u32) {
        let s = Self::slot(ch, key);
        self.on[s] = self.on[s].wrapping_add(n);
        self.note_ons += n as u64;
        let at = self.chans.strike(ch);
        let same = |r: &Run| r.start == frame && r.vel == vel && r.at == at;
        let q = &mut self.open[s];
        match q.back_mut() {
            Some(r) if same(r) && r.count.checked_add(n).is_some() => r.count += n,
            _ => {
                q.push_back(Run { start: frame, vel, count: n, at });
                self.live += 1;
                self.peak_live = self.peak_live.max(self.live);
            }
        }
    }

    /// `GateTable::publish_off`: the next `n` sounding copies of `s` end at \[31\]
    fn publish(&mut self, s: usize, frame: u64, n: u32) {
        self.publish_as(s, frame, n, false)
    }

    /// `publish`, saying whether a CC120 is what ends them.
    fn publish_as(&mut self, s: usize, frame: u64, mut n: u32, cut: bool) {
        self.off[s] = self.off[s].wrapping_add(n);
        let c = s / KEYS;
        let gain_off = self.chans.gain(c as u8);
        let cut_at = if cut { u64::MAX } else { self.cuts[c].get(self.cuts_seen[c] as usize).copied().unwrap_or(u64::MAX) };
        let q = &mut self.open[s];
        while n > 0 {
            let Some(r) = q.front_mut() else { break };
            let k = n.min(r.count);
            let note = Note { start: r.start, end: frame.max(r.start), slot: s as u32, vel: r.vel, count: k, preset: r.at.preset, gain_on: r.at.gain, bend: r.at.bend, sound: r.at.sound, gain_off, cut, cut_at };
            // [32]
            match &mut self.out[s] {
                Some(o) if Note { count: o.count, ..note } == *o && o.count.checked_add(k).is_some() => o.count += k,
                o => {
                    if let Some(prev) = o.replace(note) {
                        self.emitted.push(prev);
                        self.runs += 1;
                    }
                }
            }
            self.copies += k as u64;
            r.count -= k;
            n -= k;
            if r.count == 0 {
                q.pop_front();
                self.live -= 1;
            }
        }
    }

    fn note_off(&mut self, ch: u8, key: u8, frame: u64) {
        let s = Self::slot(ch, key);
        let c = ch as usize;
        if self.sostenuto[c] && self.sost_held[s] > 0 {
            if self.off[s].wrapping_add(self.pending[s]) != self.on[s] {
                self.pending[s] += 1;
                self.sost_held[s] -= 1;
            }
            return;
        }
        if self.sustain[c] {
            if self.off[s].wrapping_add(self.pending[s]) != self.on[s] {
                self.pending[s] += 1;
            }
        } else if self.off[s] != self.on[s] {
            self.publish(s, frame, 1);
        }
    }

    /// `n` of `note_off` at once. Identical note-offs reach here folded, and \[33\]
    fn note_offs(&mut self, ch: u8, key: u8, frame: u64, n: u32) {
        let c = ch as usize;
        if self.sustain[c] || self.sostenuto[c] {
            for _ in 0..n {
                self.note_off(ch, key, frame);
            }
            return;
        }
        let s = Self::slot(ch, key);
        let k = n.min(self.on[s].wrapping_sub(self.off[s]));
        if k != 0 {
            self.publish(s, frame, k);
        }
    }

    fn set_sustain(&mut self, ch: u8, down: bool, frame: u64) {
        let c = ch as usize;
        if down == self.sustain[c] {
            return;
        }
        self.sustain[c] = down;
        if down || self.sostenuto[c] {
            return;
        }
        self.flush_pending(ch, frame);
    }

    fn set_sostenuto(&mut self, ch: u8, down: bool, frame: u64) {
        let c = ch as usize;
        if down == self.sostenuto[c] {
            return;
        }
        self.sostenuto[c] = down;
        if down {
            for &k in &self.keys {
                let s = Self::slot(ch, k);
                self.sost_held[s] = self.on[s].wrapping_sub(self.off[s]).wrapping_sub(self.pending[s]);
            }
            return;
        }
        for &k in &self.keys {
            self.sost_held[Self::slot(ch, k)] = 0;
        }
        if self.sustain[c] {
            return;
        }
        self.flush_pending(ch, frame);
    }

    fn flush_pending(&mut self, ch: u8, frame: u64) {
        for i in 0..self.keys.len() {
            let s = Self::slot(ch, self.keys[i]);
            let n = self.pending[s];
            if n != 0 {
                self.pending[s] = 0;
                self.publish(s, frame, n);
            }
        }
    }

    fn all_notes_off(&mut self, ch: u8, frame: u64) {
        if self.sustain[ch as usize] {
            for &k in &self.keys {
                let s = Self::slot(ch, k);
                self.pending[s] = self.on[s].wrapping_sub(self.off[s]);
            }
            return;
        }
        for i in 0..self.keys.len() {
            let s = Self::slot(ch, self.keys[i]);
            let n = self.on[s].wrapping_sub(self.off[s]);
            if n != 0 {
                self.publish(s, frame, n);
            }
        }
    }

    /// CC120 when `cut`; the end of the file, which releases, when not.
    fn all_sound_off(&mut self, ch: u8, frame: u64, cut: bool) {
        for i in 0..self.keys.len() {
            let s = Self::slot(ch, self.keys[i]);
            let n = self.on[s].wrapping_sub(self.off[s]);
            if n != 0 {
                self.publish_as(s, frame, n, cut);
            }
            self.pending[s] = 0;
            self.sost_held[s] = 0;
        }
    }

    fn reset_controllers(&mut self, ch: u8, frame: u64) {
        self.set_sostenuto(ch, false, frame);
        self.set_sustain(ch, false, frame);
    }

    /// The earliest start of a note still sounding.
    fn earliest_held(&self) -> u64 {
        let mut f = u64::MAX;
        for &k in &self.keys {
            for ch in 0..CHANNELS {
                if let Some(r) = self.open[Self::slot(ch as u8, k)].front() {
                    f = f.min(r.start);
                }
            }
        }
        f
    }

    /// Hand the sink every release still held, in slot order.
    fn flush_out(&mut self) {
        let mut keys = self.keys.clone();
        keys.sort_unstable();
        for ch in 0..CHANNELS {
            for &k in &keys {
                if let Some(n) = self.out[Self::slot(ch as u8, k)].take() {
                    self.emitted.push(n);
                    self.runs += 1;
                }
            }
        }
    }
}

/// How a group pass is cut.
#[derive(Clone, Copy, Debug)]
pub struct ExtractOpts {
    /// About how many of the group's note-ons one window of ticks holds.
    pub window_notes: u64,
}

impl Default for ExtractOpts {
    fn default() -> Self {
        // [34]
        Self { window_notes: 1 << 18 }
    }
}

/// What a group pass did.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExtractStats {
    pub windows: usize,
    pub note_ons: u64,
    /// Notes handed to the sink, and the copies they stand for.
    pub runs: u64,
    pub copies: u64,
    /// Note-ons in the largest window, and runs sounding at once.
    pub peak_window: usize,
    pub peak_live: usize,
    /// The tracks read into their buffers, then each pairing's window merged \[35\]
    pub read_secs: f64,
    pub pair_secs: f64,
    /// Within those, summed over the pairings: keys read from the store, the \[36\]
    pub load_secs: f64,
    pub merge_secs: f64,
    pub match_secs: f64,
    pub sink_secs: f64,
    pub after_secs: f64,
    /// What the group's notes asked of the channel state that records do \[37\]
    pub unsupported: u32,
    /// The frame everything still held is released at, the start of the \[38\]
    pub eof: Option<u64>,
}

/// A note event of the group, in its place.
#[derive(Clone, Copy)]
struct NoteEv {
    at: Place,
    ch: u8,
    key: u8,
    /// Zero for a note-off.
    vel: u8,
    /// Identical note events read back to back in one track on one tick, \[39\]
    count: u32,
}

/// One track's stream of one key in a group pass, holding the event it read \[40\]
struct FiledKey {
    cursor: crate::merlin::keystore::Cursor,
    /// The pairing it feeds.
    sub: u8,
    held: Option<NoteEv>,
}

impl FiledKey {
    /// Its events of the window ending at tick `end`, into `buf`.
    fn take(&mut self, end: u64, track: u32, bytes: &[u8], buf: &mut Vec<NoteEv>) {
        loop {
            let e = match self.held.take() {
                Some(e) => e,
                None => match self.cursor.next_event(bytes) {
                    Some(e) => NoteEv { at: Place { tick: e.tick, track, seq: e.seq }, ch: e.ch, key: self.cursor.key, vel: e.vel, count: e.count },
                    None => return,
                },
            };
            if e.at.tick >= end {
                self.held = Some(e);
                return;
            }
            buf.push(e);
        }
    }
}

/// One track's reader in a group pass, holding the event it read past the \[41\]
struct TrackPass {
    reader: TrackReader,
    held: Option<(u64, Event)>,
    seq: u64,
    track: u32,
}

/// A run of neighbouring tracks, read by one worker into one buffer per \[42\]
struct Worker {
    tracks: Vec<TrackPass>,
    /// With a store, what stands for `tracks`: each track's keys.
    filed: Vec<(u32, Vec<FiledKey>)>,
    /// Whether the last event in each buffer may take more copies.
    fold_ok: Vec<bool>,
}

impl Worker {
    /// Its tracks' events of the window ending at tick `end`, the group's \[43\]
    fn fill(&mut self, end: u64, sub_of: &[u8; KEYS], buf: &mut [Vec<NoteEv>], bytes: &[u8], shared: &[bool]) {
        for b in buf.iter_mut() {
            b.clear();
        }
        let mut mark = vec![0usize; buf.len()];
        for (track, keys) in &mut self.filed {
            for (m, b) in mark.iter_mut().zip(buf.iter()) {
                *m = b.len();
            }
            for k in keys.iter_mut() {
                k.take(end, *track, bytes, &mut buf[k.sub as usize]);
            }
            for ((b, &m), _) in buf.iter_mut().zip(&mark).zip(shared).filter(|(_, &s)| s) {
                b[m..].sort_by_key(|e| e.at);
            }
        }
        let keep: [bool; 128] = std::array::from_fn(|k| sub_of[k] != NONE);
        for tp in &mut self.tracks {
            self.fold_ok.fill(false);
            loop {
                // [44]
                if tp.held.is_none() {
                    tp.reader.skip_unkept(end, &keep);
                }
                let (tick, ev) = match tp.held.take() {
                    Some(h) => h,
                    None => match tp.reader.next_event() {
                        Some(ev) => (tp.reader.tick, ev),
                        None => break,
                    },
                };
                if tick >= end {
                    tp.held = Some((tick, ev));
                    break;
                }
                let at = Place { tick, track: tp.track, seq: tp.seq };
                let (ch, key, vel) = match ev {
                    Event::NoteOn { ch, key, vel } => (ch, key, vel),
                    Event::NoteOff { ch, key } => (ch, key, 0),
                    _ => {
                        if ctl_of(&ev).is_some() {
                            tp.seq += 1;
                            self.fold_ok.fill(false);
                        }
                        continue;
                    }
                };
                let sub = sub_of[key as usize & 127];
                if sub == NONE {
                    continue;
                }
                // [45]
                let (buf, ok) = (&mut buf[sub as usize], &mut self.fold_ok[sub as usize]);
                if let Some(last) = buf.last_mut().filter(|l| *ok && l.at.tick == tick && l.at.track == tp.track && l.ch == ch && l.key == key && l.vel == vel && l.count < u32::MAX) {
                    last.count += 1;
                    continue;
                }
                buf.push(NoteEv { at, ch, key, vel, count: 1 });
                *ok = true;
            }
        }
    }
}

/// The workers' buffers for pairing `sub` merged into `out` in stream order: \[46\]
fn merge_by_tick(bufs: &[Vec<Vec<NoteEv>>], sub: usize, window: (u64, u64), out: &mut Vec<NoteEv>, bins: &mut Vec<u32>) {
    let n: usize = bufs.iter().map(|w| w[sub].len()).sum();
    let Some(&first) = bufs.iter().flat_map(|w| w[sub].first()).next() else {
        out.clear();
        return;
    };
    let all = || bufs.iter().flat_map(|w| w[sub].iter());
    let limit = (n as u64 * 4).max(1 << 16);
    // [47]
    let (lo, range) = if window.1 != u64::MAX && window.1 - window.0 <= limit {
        (window.0, window.1 - window.0)
    } else {
        let (lo, hi) = all().fold((u64::MAX, 0), |(l, h), e| (l.min(e.at.tick), h.max(e.at.tick)));
        (lo, hi - lo + 1)
    };
    if range > limit || n >= u32::MAX as usize {
        out.clear();
        out.extend(all().copied());
        out.sort_by_key(|e| e.at.tick);
        return;
    }
    bins.clear();
    bins.resize(range as usize + 1, 0);
    for w in bufs {
        for e in &w[sub] {
            bins[(e.at.tick - lo) as usize + 1] += 1;
        }
    }
    for i in 1..bins.len() {
        bins[i] += bins[i - 1];
    }
    out.resize(n, first);
    for w in bufs {
        for e in &w[sub] {
            let b = &mut bins[(e.at.tick - lo) as usize];
            out[*b as usize] = *e;
            *b += 1;
        }
    }
}

/// One pass over the file for `keys`: their notes paired and handed to `sink` \[48\]
pub fn extract(pre: &Prepass, cfg: &Config, bank: &Bank, keys: &[u8], opts: ExtractOpts, mut sink: impl FnMut(Note)) -> Result<ExtractStats> {
    let (st, _) = extract_core::<(), fn(&mut (), Note)>(pre, cfg, bank, keys, opts, None, None, &mut sink)?;
    Ok(st)
}

/// Each pairing's keys and its sink.
pub type Pairings<S> = Vec<(Vec<u8>, S)>;

/// What makes a pairing's sink, and what hands it a note.
type PerPairing<'a, S, F> = (&'a dyn Fn(&[u8]) -> S, &'a F);

/// What [`extract_par`] calls after each window but the last, with the sinks \[49\]
pub type AfterWindow<'a, S> = &'a mut dyn FnMut(&mut [Option<S>], u64) -> bool;

/// [`extract`] with a sink per pairing, each fed on a worker as its pairing \[50\]
#[allow(clippy::too_many_arguments)]
pub fn extract_par<S: Send, F: Fn(&mut S, Note) + Sync>(
    pre: &Prepass,
    cfg: &Config,
    bank: &Bank,
    keys: &[u8],
    opts: ExtractOpts,
    make: impl Fn(&[u8]) -> S,
    note: F,
    after: Option<AfterWindow<'_, S>>,
) -> Result<(ExtractStats, Pairings<S>)> {
    extract_core(pre, cfg, bank, keys, opts, Some((&make as &dyn Fn(&[u8]) -> S, &note)), after, &mut |_| {})
}

#[allow(clippy::too_many_arguments)]
fn extract_core<S: Send, F: Fn(&mut S, Note) + Sync>(
    pre: &Prepass,
    cfg: &Config,
    bank: &Bank,
    keys: &[u8],
    opts: ExtractOpts,
    par: Option<PerPairing<'_, S, F>>,
    mut after: Option<AfterWindow<'_, S>>,
    sink: &mut dyn FnMut(Note),
) -> Result<(ExtractStats, Pairings<S>)> {
    let mut st = ExtractStats::default();
    // [51]
    let counts = pre.key_notes();
    let mut order: Vec<u8> = keys.to_vec();
    order.sort_by_key(|&k| (std::cmp::Reverse(counts[k as usize & 127]), k));
    order.dedup();
    let subs_n = rayon::current_num_threads().clamp(1, order.len().max(1));
    let mut sub_keys: Vec<Vec<u8>> = vec![Vec::new(); subs_n];
    let mut load = vec![0u64; subs_n];
    let mut sub_of = [NONE; KEYS];
    for &k in &order {
        let i = (0..subs_n).min_by_key(|&i| (load[i], i)).expect("one at least");
        load[i] += counts[k as usize & 127];
        sub_keys[i].push(k & 127);
        sub_of[k as usize & 127] = i as u8;
    }
    // [52]
    let runs = (rayon::current_num_threads() * 4).max(1) as u64;
    let mut workers: Vec<Worker> = Vec::new();
    let mut acc = 0u64;
    let mut bytes: Vec<u8> = Vec::new();
    let shared: Vec<bool> = sub_keys.iter().map(|k| k.len() > 1).collect();
    if let Some(store) = &pre.store {
        let t = std::time::Instant::now();
        let loaded = store.load(&order)?;
        st.load_secs += t.elapsed().as_secs_f64();
        st.read_secs += t.elapsed().as_secs_f64();
        bytes = loaded.bytes;
        // Runs of neighbouring tracks again, by what each has of these keys.
        let total = loaded.tracks.iter().map(|t| t.1.len() as u64).sum::<u64>();
        for (track, keys) in loaded.tracks {
            if workers.is_empty() || acc >= total.div_ceil(runs) {
                workers.push(Worker { tracks: Vec::new(), filed: Vec::new(), fold_ok: Vec::new() });
                acc = 0;
            }
            acc += keys.len() as u64;
            let keys = keys.into_iter().map(|cursor| FiledKey { sub: sub_of[cursor.key as usize & 127], cursor, held: None }).collect();
            workers.last_mut().expect("pushed").filed.push((track, keys));
        }
    } else {
        let total: u64 = pre.header.tracks.iter().map(|&(_, l)| l).sum();
        let lens: Vec<u64> = pre.header.tracks.iter().map(|&(_, l)| l).collect();
        let window = crate::midi::window_share(&lens, EXTRACT_READ_BUDGET);
        let opened: Vec<TrackPass> = pre
            .header
            .tracks
            .par_iter()
            .enumerate()
            .map(|(t, &(off, len))| Ok(TrackPass { reader: TrackReader::open(&pre.path, off, len, window)?, held: None, seq: 0, track: t as u32 }))
            .collect::<Result<_>>()?;
        for (tp, &(_, len)) in opened.into_iter().zip(&pre.header.tracks) {
            if workers.is_empty() || acc >= total.div_ceil(runs) {
                workers.push(Worker { tracks: Vec::new(), filed: Vec::new(), fold_ok: vec![false; subs_n] });
                acc = 0;
            }
            acc += len;
            workers.last_mut().expect("pushed").tracks.push(tp);
        }
    }
    let clock0 = TempoClock::with_timing(pre.header.division, cfg.sample_rate, cfg.tempo_timing);
    let chans = crate::merlin::keyrec::Channels::new(bank, cfg);
    let mut states: Vec<Option<S>> = sub_keys.iter().map(|k| par.map(|(make, _)| make(k))).collect();
    let mut subs: Vec<(Pairing, TempoClock)> = sub_keys.into_iter().map(|k| (Pairing::new(k, chans.clone(), pre.cuts.clone()), clock0.clone())).collect();
    let ends = pre.windows(keys, opts.window_notes);
    let note = par.map(|(_, f)| f);
    let spent = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
    let clocked = |i: usize, t: std::time::Instant| {
        spent[i].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    };
    let mut walk = pre.ctl_walk();
    // [53]
    let (filled_tx, filled_rx) = std::sync::mpsc::sync_channel::<ReadBufs>(STAGE_AHEAD);
    let (merged_tx, merged_rx) = std::sync::mpsc::sync_channel::<MergeBufs>(STAGE_AHEAD);
    let (paired_tx, paired_rx) = std::sync::mpsc::sync_channel::<(Vec<Vec<Note>>, Option<u64>)>(STAGE_AHEAD);
    let (read_back, read_free) = std::sync::mpsc::channel::<ReadBufs>();
    let (merge_back, merge_free) = std::sync::mpsc::channel::<MergeBufs>();
    let (note_back, note_free) = std::sync::mpsc::channel::<Vec<Vec<Note>>>();
    for _ in 0..STAGE_AHEAD + 2 {
        let _ = read_back.send(vec![vec![Vec::new(); subs_n]; workers.len()]);
        let _ = merge_back.send((0..subs_n).map(|_| (Vec::new(), Vec::new())).collect());
        let _ = note_back.send(vec![Vec::new(); subs_n]);
    }
    let flush = after.is_some();
    let (ends, sub_of, bytes, shared, clocked) = (&ends, &sub_of, &bytes, &shared, &clocked);
    let (running, run_workers) = (&mut subs, &mut workers);
    let (read_secs, peak_window, pair_secs, after_secs) = std::thread::scope(|scope| -> Result<(f64, usize, f64, f64)> {
        let reading = scope.spawn(move || {
            let mut secs = 0.0;
            for &end in ends {
                let Ok(mut b) = read_free.recv() else { break };
                let t = std::time::Instant::now();
                run_workers.par_iter_mut().zip(b.par_iter_mut()).for_each(|(w, b)| w.fill(end, sub_of, b, bytes, shared));
                secs += t.elapsed().as_secs_f64();
                if filled_tx.send(b).is_err() {
                    break;
                }
            }
            secs
        });
        let merging = scope.spawn(move || {
            let mut peak = 0;
            for (w, &end) in ends.iter().enumerate() {
                let (Ok(b), Ok(mut m)) = (filled_rx.recv(), merge_free.recv()) else { break };
                let start = if w == 0 { 0 } else { ends[w - 1] };
                let t = std::time::Instant::now();
                m.par_iter_mut().enumerate().for_each(|(i, (notes, bins))| merge_by_tick(&b, i, (start, end), notes, bins));
                clocked(0, t);
                peak = peak.max(m.iter().map(|n| n.0.len()).sum());
                let _ = read_back.send(b);
                if merged_tx.send(m).is_err() {
                    break;
                }
            }
            peak
        });
        let pairing = scope.spawn(move || -> Result<f64> {
            let mut secs = 0.0;
            for (w, &end) in ends.iter().enumerate() {
                let (Ok(m), Ok(mut notes)) = (merged_rx.recv(), note_free.recv()) else { break };
                let ctls = walk.window(end)?;
                let t = std::time::Instant::now();
                running.par_iter_mut().zip(m.par_iter()).for_each(|((pair, clock), (notes, _))| pair_window(bank, pair, clock, ctls, notes));
                // [54]
                let mut frontier = None;
                if flush && w + 1 < ends.len() {
                    running.iter_mut().for_each(|(pair, _)| pair.flush_out());
                    frontier = Some(running.iter().map(|(pair, _)| pair.earliest_held()).min().unwrap_or(u64::MAX).min(frame_of(&running[0].1, end)));
                }
                for (to, (pair, _)) in notes.iter_mut().zip(running.iter_mut()) {
                    std::mem::swap(to, &mut pair.emitted);
                }
                clocked(1, t);
                secs += t.elapsed().as_secs_f64();
                let _ = merge_back.send(m);
                if paired_tx.send((notes, frontier)).is_err() {
                    break;
                }
            }
            Ok(secs)
        });
        // Here, on the caller's thread: `sink` and `after` need not be `Send`.
        let mut after_secs = 0.0;
        let mut stopped = false;
        for (mut notes, frontier) in paired_rx.iter() {
            if pre.stopped() {
                stopped = true;
                break;
            }
            let t = std::time::Instant::now();
            match note {
                Some(f) => states.par_iter_mut().zip(notes.par_iter_mut()).for_each(|(state, notes)| {
                    if let Some(s) = state.as_mut() {
                        notes.drain(..).for_each(|n| f(s, n));
                    }
                }),
                None => notes.iter_mut().flat_map(|notes| notes.drain(..)).for_each(&mut *sink),
            }
            clocked(2, t);
            let _ = note_back.send(notes);
            if let (Some(f), Some(at)) = (after.as_mut(), frontier) {
                let t = std::time::Instant::now();
                if !f(&mut states, at) {
                    stopped = true;
                    break;
                }
                after_secs += t.elapsed().as_secs_f64();
            }
        }
        // The queues closing is what tells the stages to stop.
        drop(paired_rx);
        drop(note_back);
        if stopped {
            if pre.stopped() {
                return Err(crate::merlin::progress::Cancelled.into());
            }
            bail!("stopped");
        }
        let pair_secs = pairing.join().expect("no stage panics")?;
        Ok((reading.join().expect("no stage panics"), merging.join().expect("no stage panics"), pair_secs, after_secs))
    })?;
    st.windows = ends.len();
    st.read_secs += read_secs;
    st.peak_window = peak_window;
    st.pair_secs = pair_secs;
    st.after_secs = after_secs;
    [st.merge_secs, st.match_secs, st.sink_secs] = spent.map(|n| n.into_inner() as f64 / 1e9);
    // [55]
    let eof = match pre.last_tick {
        Some(t) => frame_of(&subs[0].1, t) / cfg.block_frames as u64 * cfg.block_frames as u64,
        None => 0,
    };
    st.eof = pre.last_tick.map(|_| eof);
    for ((pair, _), state) in subs.iter_mut().zip(states.iter_mut()) {
        for ch in 0..CHANNELS {
            pair.all_sound_off(ch as u8, eof, false);
        }
        pair.flush_out();
        for n in pair.emitted.drain(..) {
            match (note, state.as_mut()) {
                (Some(f), Some(s)) => f(s, n),
                _ => sink(n),
            }
        }
        st.peak_live += pair.peak_live;
        st.runs += pair.runs;
        st.copies += pair.copies;
        st.note_ons += pair.note_ons;
        st.unsupported |= pair.chans.flags;
    }
    let out = subs.into_iter().zip(states).filter_map(|((pair, _), s)| s.map(|s| (pair.keys, s))).collect();
    Ok((st, out))
}

/// A frame from the clock, as the driver takes it.
fn frame_of(clock: &TempoClock, tick: u64) -> u64 {
    let f = clock.frame_at(tick);
    if f < 0.0 {
        0
    } else {
        f as u64
    }
}

/// `frame_of`, remembering the last tick: nearly every event of black MIDI is \[56\]
fn frame_now(clock: &mut TempoClock, tick: u64) -> u64 {
    let f = clock.frame_at_cached(tick);
    if f < 0.0 {
        0
    } else {
        f as u64
    }
}

/// One window through one pairing: its notes and the window's controllers, \[57\]
fn pair_window(bank: &Bank, pair: &mut Pairing, clock: &mut TempoClock, ctls: &[CtlEv], notes: &[NoteEv]) {
    let (mut ci, mut ni) = (0usize, 0usize);
    loop {
        let take_ctl = match (ctls.get(ci), notes.get(ni)) {
            (Some(c), Some(n)) => c.at < n.at,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => return,
        };
        if take_ctl {
            let c = ctls[ci];
            ci += 1;
            if let Ctl::Tempo(us) = c.ctl {
                clock.set_tempo(c.at.tick, us);
                continue;
            }
            let f = frame_now(clock, c.at.tick);
            match c.ctl {
                Ctl::Sustain(down) => pair.set_sustain(c.ch, down, f),
                Ctl::Sostenuto(down) => pair.set_sostenuto(c.ch, down, f),
                Ctl::SoundOff => {
                    pair.all_sound_off(c.ch, f, true);
                    pair.cuts_seen[c.ch as usize] += 1;
                }
                Ctl::Reset => {
                    pair.chans.reset(c.ch);
                    pair.reset_controllers(c.ch, f);
                }
                Ctl::Chan(ev) => pair.chans.apply(bank, &ev),
                Ctl::NotesOff => pair.all_notes_off(c.ch, f),
                Ctl::Tempo(_) => unreachable!(),
            }
        } else {
            let n = notes[ni];
            ni += 1;
            let f = frame_now(clock, n.at.tick);
            if n.vel == 0 {
                pair.note_offs(n.ch, n.key, f, n.count);
            } else {
                pair.note_on(n.ch, n.key, n.vel, f, n.count);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound.max(1)
        }
    }

    /// The merge is a stable sort by tick over the buffers laid end to end, \[58\]
    #[test]
    fn a_warp_is_the_integral_of_its_pitch() {
        let mut w = Warp::default();
        assert!(!w.moves_at_all());
        assert_eq!(w.tau(1234), 1234.0);
        // A pitch set at the start and left there moves nothing.
        w.change(0, 2.0);
        assert!(!w.moves_at_all());
        assert_eq!(w.tau(10), 20.0);
        w.change(0, 1.0);
        w.change(100, 2.0);
        w.change(300, 0.5);
        w.change(300, 0.25);
        w.change(1000, 1.0);
        assert!(w.moves_at_all());
        for (f, t) in [(0u64, 0.0), (100, 100.0), (150, 200.0), (300, 500.0), (500, 550.0), (1000, 675.0), (1010, 685.0)] {
            assert_eq!(w.tau(f), t, "frame {f}");
            assert_eq!(w.frame(t), f as f64, "tau {t}");
        }
        assert!(!w.moves(0, 99) && w.moves(0, 100) && w.moves(99, 100) && !w.moves(100, 299) && !w.moves(1000, 5000));
        let runs = w.runs(50, 1100, 64);
        let mut f = 50u64;
        for r in &runs {
            assert_eq!(r[0] as u64, f, "a gap in the runs");
            let t = r[1] as f64 + f32::from_bits(r[2]) as f64;
            assert_eq!(t, w.tau(f));
            assert_eq!(f32::from_bits(r[3]) as f64, w.tau(f + 1) - w.tau(f));
            assert!(r[4] <= 64 && !w.moves(f, f + r[4] as u64 - 1), "a run across a change");
            f += r[4] as u64;
        }
        assert_eq!(f, 1100);
    }

    #[test]
    fn a_controller_reads_back_as_it_was_written() {
        let at = Place { tick: (1 << 40) + 5, track: 70_000, seq: (1 << 33) + 9 };
        let ctls = [
            Ctl::Tempo(u32::MAX - 3),
            Ctl::Sustain(true),
            Ctl::Sustain(false),
            Ctl::Sostenuto(true),
            Ctl::SoundOff,
            Ctl::Reset,
            Ctl::NotesOff,
            Ctl::Chan(Event::Cc { ch: 200, num: 74, val: 127 }),
            Ctl::Chan(Event::Program { ch: 200, val: 9 }),
            Ctl::Chan(Event::PitchBend { ch: 200, val: -8192 }),
            Ctl::Chan(Event::DrumPart { ch: 200, map: 2 }),
            Ctl::Chan(Event::Tune { ch: 200, pc: 11, units: -123_456 }),
            Ctl::Chan(Event::KeyTune { bank: 3, prog: 4, key: 5, units: 987_654 }),
            Ctl::Chan(Event::ResetParts),
        ];
        for ctl in ctls {
            let ch = if matches!(ctl, Ctl::Chan(Event::KeyTune { .. } | Event::ResetParts) | Ctl::Tempo(_)) { 0 } else { 200 };
            let c = CtlEv { at, ch, ctl };
            assert_eq!(CtlEv::unpack(&c.pack()), c);
        }
    }

    /// Runs on disk and what stayed in memory merge into the stream's order, \[59\]
    #[test]
    fn controllers_on_disk_walk_in_stream_order() {
        let mut rng = Lcg(5);
        let mut want: Vec<CtlEv> = (0..5000u64)
            .map(|seq| CtlEv { at: Place { tick: rng.next(900), track: rng.next(7) as u32, seq }, ch: rng.next(256) as u8, ctl: Ctl::Chan(Event::Cc { ch: 0, num: 7, val: 1 }) })
            .map(|c| CtlEv { ctl: Ctl::Chan(Event::Cc { ch: c.ch, num: 7, val: (c.at.seq % 128) as u8 }), ..c })
            .collect();
        let dir = std::env::temp_dir();
        for budget in [1usize, 7, 1000, 1 << 20] {
            let mut sink = CtlSink::default();
            for part in want.chunks(13) {
                sink.add(&mut part.to_vec(), budget, Some(&dir));
            }
            assert_eq!(sink.count, want.len() as u64);
            assert_eq!(sink.disk.is_some(), budget <= 1000, "budget {budget}");
            let mut mem = std::mem::take(&mut sink.held);
            mem.sort_unstable_by_key(|c| c.at);
            let runs = sink.disk.iter().flat_map(|d| &d.runs).map(|&(off, left)| CtlRun { off, left, buf: Vec::new(), i: 0 }).collect();
            let mut walk = CtlWalk { mem: &mem, file: sink.disk.as_ref().map(|d| &d.file), runs, bytes: Vec::new(), out: Vec::new() };
            let mut got: Vec<CtlEv> = Vec::new();
            for end in (100..=1000).step_by(100) {
                let w = walk.window(end).unwrap();
                assert!(w.iter().all(|c| c.at.tick < end && c.at.tick + 100 >= end), "budget {budget}: a window holds a stranger");
                got.extend_from_slice(w);
            }
            let mut sorted = want.clone();
            sorted.sort_unstable_by_key(|c| c.at);
            assert_eq!(got, sorted, "budget {budget}");
        }
        want.clear();
    }

    #[test]
    fn the_merge_is_a_stable_sort_by_tick() {
        let mut rng = Lcg(29);
        let (mut out, mut bins) = (Vec::new(), Vec::new());
        // Events, the ticks they fall on from the window's start, its length.
        let cases: [(usize, u64, u64); 7] =
            [(3000, 500, 500), (40, 500, 500), (3000, 500, u64::MAX / 2), (3000, 90_000, 90_000), (500, 1 << 40, 1 << 40), (1, 1, 1), (0, 1, 1)];
        for (n, span, len) in cases {
            for start in [0u64, 7_000_000] {
                let mut bufs: Vec<Vec<Vec<NoteEv>>> = vec![vec![Vec::new(), Vec::new()]; 3];
                for i in 0..n {
                    let w = rng.next(3) as usize;
                    let e = NoteEv { at: Place { tick: start + rng.next(span), track: w as u32, seq: i as u64 }, ch: 0, key: 60, vel: 1, count: 1 };
                    bufs[w][rng.next(2) as usize].push(e);
                }
                for sub in 0..2 {
                    let mut want: Vec<NoteEv> = bufs.iter().flat_map(|w| w[sub].iter().copied()).collect();
                    want.sort_by_key(|e| e.at.tick);
                    let end = if len == u64::MAX / 2 { u64::MAX } else { start + len };
                    merge_by_tick(&bufs, sub, (start, end), &mut out, &mut bins);
                    let places = |v: &[NoteEv]| v.iter().map(|e| e.at).collect::<Vec<_>>();
                    assert_eq!(places(&out), places(&want), "{n} events over {span} ticks of a window of {len}");
                }
            }
        }
    }
}
