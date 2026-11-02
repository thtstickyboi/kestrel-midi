// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Duplicate-note coalescing, `Config::coalesce`: N note-ons that would build N \[1\]

use crate::bank::PreviewLayer;
use crate::config::Config;
use crate::driver::QuietPairs;
use crate::edo31;
use crate::midi::{Event, MidiStream, CHANNELS};
use crate::snap::Dec;
use crate::voice::GateTable;
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// The release token of a note-on that is still sounding when the file ends.
pub const EOF: u64 = u64::MAX - 2;
/// Not released yet.
const OPEN: u64 = u64::MAX;
/// Note-ons read between sweeps of every slot's open queue for spent entries. \[2\]
const SWEEP_EVERY: u64 = 1 << 22;
/// Decided unmerged: nothing else was struck on its slot on its tick, it was \[3\]
const SOLO: u64 = u64::MAX - 1;

/// One note-on's decision, from the driver's next one on.
#[derive(Clone, Copy)]
struct Entry {
    /// `OPEN`, `SOLO`, `EOF`, or the tick that released it.
    token: u64,
    /// Its slot and velocity, `slot << 8 | vel`, which the driver's note-on \[4\]
    check: u32,
}

/// In `Entry::check`: decided unmerged because it outlived the window. Above \[5\]
const FORCED: u32 = 1 << 31;

/// The second reader: the stream, the pairing, and the note-ons read and not \[6\]
struct Reader {
    stream: MidiStream,
    /// For a fresh gate table on `resync`, and the routing and pairing options.
    cfg: Config,
    gates: GateTable,
    quiet: Option<QuietPairs>,
    /// The next event to read, ahead of the stream: the one the driver had \[7\]
    injected: Option<(u64, Event)>,
    /// One entry per note-on read and not yet decided and handed on. The first \[8\]
    fifo: VecDeque<Entry>,
    head: u64,
    /// The number the next note-on read takes.
    read: u64,
    /// The tick being read, and the number of its first note-on.
    tick: u64,
    tick_first: u64,
    /// Per slot, its note-ons not yet released or decided, as (ordinal, \[9\]
    open: Vec<VecDeque<(u32, u64)>>,
    /// Note-ons read since the last sweep of every slot for spent entries.
    since_sweep: u64,
    /// Per slot, note-ons struck on it on the tick being read, and the first \[10\]
    slot_gen: Vec<u32>,
    slot_count: Vec<u32>,
    slot_first: Vec<u64>,
    tick_gen: u32,
    touched: Vec<u32>,
    window: u64,
    /// The stream is over, and every entry is decided.
    done: bool,
}

impl Reader {
    fn new(cfg: &Config, stream: MidiStream) -> Self {
        let slots = CHANNELS * 128;
        let mut gates = GateTable::new(cfg);
        gates.grow(CHANNELS);
        Reader {
            stream,
            cfg: cfg.clone(),
            gates,
            quiet: (cfg.min_velocity > 1).then(QuietPairs::new),
            injected: None,
            fifo: VecDeque::new(),
            head: 0,
            read: 0,
            tick: u64::MAX,
            tick_first: 0,
            open: (0..slots).map(|_| VecDeque::new()).collect(),
            slot_gen: vec![u32::MAX; slots],
            slot_count: vec![0; slots],
            slot_first: vec![0; slots],
            tick_gen: 0,
            touched: Vec::new(),
            since_sweep: 0,
            window: cfg.coalesce_window.max(1) as u64,
            done: false,
        }
    }

    /// See `Lookahead::resync`.
    fn resync(
        &mut self,
        stream_state: &[u8],
        pending: Option<(u64, Event)>,
        gates_state: &[u8],
        quiet: Option<QuietPairs>,
    ) -> Result<()> {
        let mut d = Dec::new(stream_state);
        self.stream.load_state(&mut d).context("the look-ahead's MIDI stream")?;
        d.finish()?;
        let mut gates = GateTable::new(&self.cfg);
        let mut d = Dec::new(gates_state);
        gates.load_state(&mut d).context("the look-ahead's gate table")?;
        d.finish()?;
        gates.grow(CHANNELS);
        self.gates = gates;
        self.quiet = quiet;
        self.injected = pending;
        self.fifo.clear();
        self.head = 0;
        self.read = 0;
        self.tick = u64::MAX;
        self.tick_first = 0;
        for q in &mut self.open {
            q.clear();
        }
        self.slot_gen.fill(u32::MAX);
        self.tick_gen = 0;
        self.touched.clear();
        self.since_sweep = 0;
        self.done = false;
        Ok(())
    }

    /// The next note-on's decision, in stream order; `None` once the stream \[11\]
    fn next_decision(&mut self) -> Option<Entry> {
        loop {
            if let Some(&e) = self.fifo.front() {
                let decided = e.token == SOLO || self.done || (e.token != OPEN && self.head < self.tick_first);
                if decided {
                    self.fifo.pop_front();
                    self.head += 1;
                    debug_assert_ne!(e.token, OPEN);
                    return Some(e);
                }
                // [12]
                if self.read - self.head >= self.window {
                    self.fifo[0].token = SOLO;
                    self.fifo[0].check |= FORCED;
                    continue;
                }
            }
            if !self.step() && self.fifo.is_empty() {
                return None;
            }
        }
    }

    /// Read one event. False at the end of the stream.
    fn step(&mut self) -> bool {
        let Some((tick, ev)) = self.injected.take().or_else(|| self.stream.next()) else {
            if !self.done {
                self.end_tick();
                for e in &mut self.fifo {
                    if e.token == OPEN {
                        e.token = EOF;
                    }
                }
                for q in &mut self.open {
                    q.clear();
                }
                self.done = true;
            }
            return false;
        };
        if tick != self.tick {
            self.end_tick();
            self.tick = tick;
            self.tick_first = self.read;
        }
        if !self.cfg.edo31 {
            self.event(ev, tick);
            return true;
        }
        // [13]
        match ev {
            Event::NoteOn { ch, key, vel } => {
                let (ch, key) = edo31::map_note(ch, key);
                self.event(Event::NoteOn { ch, key, vel }, tick);
            }
            Event::NoteOff { ch, key } => {
                let (ch, key) = edo31::map_note(ch, key);
                self.event(Event::NoteOff { ch, key }, tick);
            }
            Event::Cc { ch, num, val } => match edo31::triplet(ch) {
                Some(t) => {
                    for ch in t {
                        self.event(Event::Cc { ch, num, val }, tick);
                    }
                }
                None => self.event(ev, tick),
            },
            _ => {}
        }
        true
    }

    /// One event on one channel, as the driver's `handle_one` moves the gates.
    fn event(&mut self, ev: Event, tick: u64) {
        match ev {
            Event::NoteOn { ch, key, vel } => {
                let s = GateTable::slot(ch, key);
                let n = self.read;
                self.read += 1;
                let check = ((s as u32) << 8) | vel as u32;
                if let Some(q) = &mut self.quiet {
                    let skip = vel < self.cfg.min_velocity;
                    q.push(s, skip);
                    if skip {
                        self.fifo.push_back(Entry { token: SOLO, check });
                        return;
                    }
                }
                let ordinal = self.gates.note_on(ch, key, 0);
                self.fifo.push_back(Entry { token: OPEN, check });
                let head = self.head;
                let q = &mut self.open[s];
                // [14]
                while q.front().is_some_and(|&(_, m)| m < head) {
                    q.pop_front();
                }
                q.push_back((ordinal, n));
                // And every slot now and then, for the ones no note touches again.
                self.since_sweep += 1;
                if self.since_sweep >= SWEEP_EVERY {
                    self.since_sweep = 0;
                    for q in &mut self.open {
                        while q.front().is_some_and(|&(_, m)| m < head) {
                            q.pop_front();
                        }
                        // A queue that grew large in a burst gives its memory back.
                        if q.is_empty() && q.capacity() > 1024 {
                            q.shrink_to_fit();
                        }
                    }
                }
                if self.slot_gen[s] != self.tick_gen {
                    self.slot_gen[s] = self.tick_gen;
                    self.slot_count[s] = 1;
                    self.slot_first[s] = n;
                    self.touched.push(s as u32);
                } else {
                    self.slot_count[s] += 1;
                }
                return;
            }
            Event::NoteOff { ch, key } => {
                if let Some(q) = &mut self.quiet {
                    if q.pop(GateTable::slot(ch, key)) {
                        return;
                    }
                }
                self.gates.note_off(ch, key, 0);
            }
            Event::Cc { ch, num, val } => match num {
                64 => self.gates.set_sustain(ch, val >= 64, 0),
                66 => self.gates.set_sostenuto(ch, val >= 64, 0),
                120 => self.gates.all_sound_off(ch, 0),
                121 => self.gates.reset_controllers(ch, 0),
                123..=127 => self.gates.all_notes_off(ch, 0),
                _ => return,
            },
            _ => return,
        }
        let (open, fifo, head) = (&mut self.open, &mut self.fifo, self.head);
        self.gates.drain_published(|s, before, n| {
            let q = &mut open[s];
            while let Some(&(ordinal, number)) = q.front() {
                // [15]
                if ordinal.wrapping_sub(before).wrapping_sub(1) >= n {
                    break;
                }
                q.pop_front();
                if number >= head {
                    let e = &mut fifo[(number - head) as usize];
                    if e.token == OPEN {
                        e.token = tick;
                    }
                }
            }
        });
    }

    /// The tick being read is over: a note-on that was alone on its slot on it \[16\]
    fn end_tick(&mut self) {
        for &s in &self.touched {
            let s = s as usize;
            if self.slot_count[s] != 1 {
                continue;
            }
            let n = self.slot_first[s];
            if n >= self.head {
                self.fifo[(n - self.head) as usize].token = SOLO;
            }
            // Struck last on its slot, so at the back if it is still there.
            if self.open[s].back().is_some_and(|&(_, m)| m == n) {
                self.open[s].pop_back();
            }
        }
        self.touched.clear();
        self.tick_gen = self.tick_gen.wrapping_add(1);
    }
}

// ---- the reader on a thread of its own ---------------------------------------

/// Decisions handed from the reader's thread to the drain's at once. A channel \[17\]
const BATCH: usize = 4096;
/// Batches the reader may have ready before it waits for the drain. With the \[18\]
const BATCHES: usize = 4;

struct Batch {
    entries: Vec<Entry>,
    /// The reader has no more note-ons after these.
    last: bool,
}

/// The reader running on its own thread, as the drain's end sees it.
struct Running {
    rx: Receiver<Batch>,
    /// Emptied batches, back to the reader to fill again.
    back: Sender<Vec<Entry>>,
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Box<Reader>>,
}

/// The look-ahead, as the driver asks it: one decision per note-on, in order. \[19\]
pub struct Lookahead {
    /// The reader while it is not running on its thread: before the first \[20\]
    reader: Option<Box<Reader>>,
    threaded: bool,
    run: Option<Running>,
    /// Decisions handed over and not yet asked for, from `at` on.
    batch: Vec<Entry>,
    at: usize,
    /// The reader has handed over its last decision.
    over: bool,
    /// The driver asked for a note-on that is not the one read here.
    broken: bool,
    /// Note-ons decided unmerged because they outlived the window, counted as \[21\]
    pub forced: u64,
    /// Time the driver has spent waiting for the reader's next batch.
    pub waited: std::time::Duration,
}

impl Lookahead {
    /// A reader at the start of `stream`, which must be a second opening of the \[22\]
    pub fn new(cfg: &Config, stream: MidiStream) -> Self {
        Lookahead {
            reader: Some(Box::new(Reader::new(cfg, stream))),
            threaded: cfg.drain_threads > 0,
            run: None,
            batch: Vec::new(),
            at: 0,
            over: false,
            broken: false,
            forced: 0,
            waited: std::time::Duration::ZERO,
        }
    }

    /// Put the reader where a driver that has just loaded a saved state is: \[23\]
    pub fn resync(
        &mut self,
        stream_state: &[u8],
        pending: Option<(u64, Event)>,
        gates_state: &[u8],
        quiet: Option<QuietPairs>,
    ) -> Result<()> {
        self.stop(true);
        self.batch.clear();
        self.at = 0;
        self.over = false;
        self.broken = false;
        let reader = self.reader.as_mut().expect("a stopped reader is back");
        reader.resync(stream_state, pending, gates_state, quiet)
    }

    /// Whether the driver has asked for a note-on this reader did not read.
    pub fn broken(&self) -> bool {
        self.broken
    }

    /// The release token of the driver's next note-on, which is on `ch` and \[24\]
    #[inline]
    pub fn next_release(&mut self, ch: u8, key: u8, vel: u8) -> Option<u64> {
        let check = ((GateTable::slot(ch, key) as u32) << 8) | vel as u32;
        let next = if !self.threaded {
            self.reader.as_mut().expect("an inline reader").next_decision()
        } else if self.at < self.batch.len() {
            self.at += 1;
            Some(self.batch[self.at - 1])
        } else {
            self.refill()
        };
        let Some(e) = next else {
            // A note-on past the last one the reader read.
            self.broken = true;
            return None;
        };
        if e.check & !FORCED != check {
            self.broken = true;
            return None;
        }
        self.forced += (e.check & FORCED != 0) as u64;
        (e.token != SOLO).then_some(e.token)
    }

    /// The next batch from the thread, which is started if it is not running, \[25\]
    #[cold]
    fn refill(&mut self) -> Option<Entry> {
        while !self.over {
            if self.run.is_none() {
                self.start();
            }
            let run = self.run.as_ref().expect("started");
            // A clock read a batch only, and only when the reader is behind.
            let ready = run.rx.try_recv().ok();
            let got = ready.map(Ok).unwrap_or_else(|| {
                let t = std::time::Instant::now();
                let b = run.rx.recv();
                self.waited += t.elapsed();
                b
            });
            match got {
                Ok(b) => {
                    let used = std::mem::replace(&mut self.batch, b.entries);
                    // The reader may have stopped; then it needs it no more.
                    let _ = run.back.send(used);
                    self.over = b.last;
                    if !self.batch.is_empty() {
                        self.at = 1;
                        return Some(self.batch[0]);
                    }
                    self.at = 0;
                }
                // [26]
                Err(_) => {
                    self.stop(true);
                    self.over = true;
                }
            }
        }
        None
    }

    fn start(&mut self) {
        let mut reader = self.reader.take().expect("a reader not yet running");
        let (tx, rx) = sync_channel::<Batch>(BATCHES);
        let (back, spent) = channel::<Vec<Entry>>();
        let stop = Arc::new(AtomicBool::new(false));
        let halt = stop.clone();
        let thread = std::thread::Builder::new()
            .name("kestrel-coalesce".into())
            .spawn(move || {
                loop {
                    let mut entries = spent.try_recv().unwrap_or_else(|_| Vec::with_capacity(BATCH));
                    entries.clear();
                    let mut last = false;
                    while entries.len() < BATCH {
                        match reader.next_decision() {
                            Some(e) => entries.push(e),
                            None => {
                                last = true;
                                break;
                            }
                        }
                    }
                    if halt.load(Ordering::Relaxed) || tx.send(Batch { entries, last }).is_err() || last {
                        return reader;
                    }
                }
            })
            .expect("a thread for the coalescing look-ahead");
        self.run = Some(Running { rx, back, stop, thread });
    }

    /// Stop the thread, if it runs, and take the reader back. A panic on it is \[27\]
    fn stop(&mut self, raise: bool) {
        let Some(run) = self.run.take() else { return };
        run.stop.store(true, Ordering::Relaxed);
        // A reader waiting to hand a batch over is woken by its queue closing.
        drop(run.rx);
        drop(run.back);
        match run.thread.join() {
            Ok(reader) => self.reader = Some(reader),
            Err(p) if raise && !std::thread::panicking() => std::panic::resume_unwind(p),
            Err(_) => {}
        }
    }
}

impl Drop for Lookahead {
    fn drop(&mut self) {
        self.stop(false);
    }
}

// ---- the driver's side: groups on one tick ------------------------------------

/// No leader, no note.
const NONE: u32 = u32::MAX;

/// What two note-ons must share, beside their previewed layers, to be one \[28\]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupKey {
    /// `driver::note_pack`'s word.
    pub word: u64,
    pub release: u64,
    pub preset: u32,
    /// `NoteTune`: the key built on, and the cents' bits.
    pub tune_key: u8,
    pub tune_cents: u32,
    pub epoch: u32,
}

/// The first note-on of a group: the one that is built.
struct Leader {
    key: GroupKey,
    /// The slot's next older leader, or `NONE`.
    next: u32,
    slot: u32,
    /// Its previewed layers, in `Coalescer::layers`.
    layer0: u32,
    layers: u32,
    /// Its note in the driver's `notes`, or `NONE` if none was recorded.
    note: u32,
    /// Its candidates in the driver's `cands`, and their gains in \[29\]
    cand0: u32,
    cands: u32,
    gain0: u32,
    /// Its layers parked on the driver's deferred queue.
    deferred0: u32,
    deferred: u32,
    weight: u32,
}

/// A group as it closes, for the driver to weight what its leader built.
pub(crate) struct Closed<'a> {
    pub weight: u32,
    pub note: Option<usize>,
    pub cand0: usize,
    /// The opening gains of its candidates, in order from `cand0`.
    pub gains: &'a [(f32, f32)],
    pub deferred: std::ops::Range<usize>,
}

/// The driver's coalescing state: the reader, the channels' epochs, and the \[30\]
pub struct Coalescer {
    pub look: Lookahead,
    tick: u64,
    epoch: Vec<u32>,
    leaders: Vec<Leader>,
    /// Per slot, its newest open leader, chained through `Leader::next`.
    slot_head: Vec<u32>,
    layers: Vec<PreviewLayer>,
    gains: Vec<(f32, f32)>,
}

impl Coalescer {
    pub fn new(cfg: &Config, stream: MidiStream) -> Self {
        Coalescer {
            look: Lookahead::new(cfg, stream),
            tick: u64::MAX,
            epoch: vec![0; CHANNELS],
            leaders: Vec::new(),
            slot_head: vec![NONE; CHANNELS * 128],
            layers: Vec::new(),
            gains: Vec::new(),
        }
    }

    /// An event, on one channel after the 31-EDO mapping, as the driver \[31\]
    #[inline]
    pub fn observe(&mut self, ev: &Event) {
        match *ev {
            Event::NoteOn { .. } | Event::NoteOff { .. } | Event::Tempo(_) | Event::Other => {}
            Event::Cc { ch, .. }
            | Event::Program { ch, .. }
            | Event::PitchBend { ch, .. }
            | Event::DrumPart { ch, .. }
            | Event::Tune { ch, .. } => {
                let e = &mut self.epoch[ch as usize];
                *e = e.wrapping_add(1);
            }
            Event::ResetParts | Event::KeyTune { .. } => {
                for e in &mut self.epoch {
                    *e = e.wrapping_add(1);
                }
            }
        }
    }

    #[inline]
    pub fn epoch(&self, ch: u8) -> u32 {
        self.epoch[ch as usize]
    }

    /// Whether the groups open are on `tick`. When they are not, the driver \[32\]
    #[inline]
    pub fn on_tick(&self, tick: u64) -> bool {
        self.tick == tick
    }

    #[inline]
    pub fn set_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Whether any group is open.
    #[inline]
    pub fn any_open(&self) -> bool {
        !self.leaders.is_empty()
    }

    /// Take a note-on into the open group it is the same voice as, if there is \[33\]
    pub fn absorb(&mut self, slot: usize, key: &GroupKey, prev: &[PreviewLayer]) -> bool {
        let mut l = self.slot_head[slot];
        while l != NONE {
            let g = &mut self.leaders[l as usize];
            if g.key == *key {
                let theirs = &self.layers[g.layer0 as usize..(g.layer0 + g.layers) as usize];
                let same = theirs.len() == prev.len()
                    && theirs.iter().zip(prev).all(|(a, b)| {
                        a.region == b.region
                            && a.gain_l.to_bits() == b.gain_l.to_bits()
                            && a.gain_r.to_bits() == b.gain_r.to_bits()
                            && a.delay_frames == b.delay_frames
                    });
                if same {
                    g.weight += 1;
                    return true;
                }
            }
            l = g.next;
        }
        false
    }

    /// The gains of a candidate a note-on that is about to lead has pushed.
    #[inline]
    pub fn push_gain(&mut self, gain_l: f32, gain_r: f32) {
        self.gains.push((gain_l, gain_r));
    }

    /// Where a leader's gains start, before its candidates are pushed.
    #[inline]
    pub fn gain_mark(&self) -> u32 {
        self.gains.len() as u32
    }

    /// Open a group led by a note-on the driver has just recorded.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        &mut self,
        slot: usize,
        key: GroupKey,
        prev: &[PreviewLayer],
        note: Option<usize>,
        cands: std::ops::Range<usize>,
        gain0: u32,
        deferred: std::ops::Range<usize>,
    ) {
        let i = self.leaders.len() as u32;
        self.leaders.push(Leader {
            key,
            next: self.slot_head[slot],
            slot: slot as u32,
            layer0: self.layers.len() as u32,
            layers: prev.len() as u32,
            note: note.map_or(NONE, |n| n as u32),
            cand0: cands.start as u32,
            cands: cands.len() as u32,
            gain0,
            deferred0: deferred.start as u32,
            deferred: deferred.len() as u32,
            weight: 1,
        });
        self.slot_head[slot] = i;
        self.layers.extend_from_slice(prev);
    }

    /// Close every open group, handing each one that took a duplicate to `f`.
    pub(crate) fn close(&mut self, mut f: impl FnMut(&Closed)) {
        for g in &self.leaders {
            self.slot_head[g.slot as usize] = NONE;
            if g.weight > 1 {
                f(&Closed {
                    weight: g.weight,
                    note: (g.note != NONE).then_some(g.note as usize),
                    cand0: g.cand0 as usize,
                    gains: &self.gains[g.gain0 as usize..(g.gain0 + g.cands) as usize],
                    deferred: g.deferred0 as usize..(g.deferred0 + g.deferred) as usize,
                });
            }
        }
        self.leaders.clear();
        self.layers.clear();
        self.gains.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::MidiWriter;

    type Track = Vec<(u64, Vec<u8>)>;

    fn file(name: &str, tracks: Vec<Track>) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("kestrel_coalesce_unit");
        std::fs::create_dir_all(&dir).unwrap();
        let mut w = MidiWriter::new(480);
        for t in tracks {
            w.raw_track(t);
        }
        let p = dir.join(name);
        w.save(&p).unwrap();
        p
    }

    /// Every note-on's token, in order, as a driver would ask for them, and how \[34\]
    fn tokens_forced(cfg: &Config, path: &std::path::Path) -> (Vec<Option<u64>>, u64) {
        let run = |threads: usize| {
            let cfg = Config { drain_threads: threads, ..cfg.clone() };
            let mut look = Lookahead::new(&cfg, MidiStream::open(path).unwrap());
            assert_eq!(look.threaded, threads > 0);
            let mut s = MidiStream::open(path).unwrap();
            let mut out = Vec::new();
            while let Some((_, ev)) = s.next() {
                if let Event::NoteOn { ch, key, vel } = ev {
                    out.push(look.next_release(ch, key, vel));
                }
            }
            assert!(!look.broken());
            (out, look.forced)
        };
        let inline = run(0);
        assert!(inline == run(1), "the reader on its thread decided differently");
        inline
    }

    fn tokens(cfg: &Config, path: &std::path::Path) -> Vec<Option<u64>> {
        tokens_forced(cfg, path).0
    }

    fn on(t: u64, key: u8, vel: u8) -> (u64, Vec<u8>) {
        (t, vec![0x90, key, vel])
    }

    fn off(t: u64, key: u8) -> (u64, Vec<u8>) {
        (t, vec![0x80, key, 0])
    }

    /// A note alone on its slot on its tick is decided at the end of the tick, \[35\]
    #[test]
    fn tokens_pair_as_the_gate_table_does() {
        let p = file(
            "pair.mid",
            vec![
                vec![on(0, 60, 100), off(100, 60), on(0, 62, 100), off(50, 62)],
                vec![on(0, 60, 100), off(120, 60)],
            ],
        );
        let t = tokens(&Config::default(), &p);
        // Track 0's events on tick 0 come first: 60, 62, then track 1's 60.
        assert_eq!(t, vec![Some(100), None, Some(120)]);
    }

    /// The pedal holds the note-offs of both, and they are published at the \[36\]
    #[test]
    fn the_pedal_publishes_at_its_lift() {
        let p = file(
            "pedal.mid",
            vec![vec![
                on(0, 60, 100),
                on(0, 60, 100),
                (10, vec![0xB0, 64, 127]),
                off(20, 60),
                off(30, 60),
                (40, vec![0xB0, 64, 0]),
                on(50, 61, 90),
                on(50, 61, 90),
                off(60, 61),
                (60, vec![0xB0, 64, 127]),
                off(60, 61),
                (90, vec![0xB0, 64, 0]),
            ]],
        );
        assert_eq!(tokens(&Config::default(), &p), vec![Some(40), Some(40), Some(60), Some(90)]);
    }

    /// Held to the end: the end-of-file release. Past the window: forced.
    #[test]
    fn the_end_of_the_file_and_the_window() {
        let mut ev = vec![on(0, 60, 100), on(0, 60, 100)];
        for i in 0..20u64 {
            ev.push(on(10 + i, 70, 80));
            ev.push(off(11 + i, 70));
        }
        let p = file("eof.mid", vec![ev]);
        let t = tokens(&Config::default(), &p);
        assert_eq!(&t[..2], &[Some(EOF), Some(EOF)]);
        assert!(t[2..].iter().all(|t| t.is_none()));
        let cfg = Config { coalesce_window: 5, ..Config::default() };
        let mut look = Lookahead::new(&cfg, MidiStream::open(&p).unwrap());
        assert_eq!(look.next_release(0, 60, 100), None, "held past the window");
        assert_eq!(look.next_release(0, 60, 100), None);
        assert_eq!(look.forced, 2);
    }

    /// A note-on the reader did not read is an error, not a guess, wherever the \[37\]
    #[test]
    fn a_driver_out_of_step_is_caught() {
        let p = file("step.mid", vec![vec![on(0, 60, 100), off(10, 60)]]);
        for threads in [0, 1] {
            let cfg = Config { drain_threads: threads, ..Config::default() };
            let mut look = Lookahead::new(&cfg, MidiStream::open(&p).unwrap());
            assert_eq!(look.next_release(0, 61, 100), None);
            assert!(look.broken());
            let mut look = Lookahead::new(&cfg, MidiStream::open(&p).unwrap());
            look.next_release(0, 60, 100);
            assert!(!look.broken());
            look.next_release(0, 60, 100);
            assert!(look.broken(), "a note-on past the end of the file");
        }
    }

    /// Black MIDI's layering, many batches long, with the window short enough \[38\]
    #[test]
    fn the_reader_on_its_thread_decides_what_it_does_inline() {
        let dir = std::env::temp_dir().join("kestrel_coalesce_unit");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("threaded.mid");
        crate::testkit::layered_midi(&p, 6, 3_000, 77).unwrap();
        let mut merged = 0;
        for window in [1, 7, 300, 1 << 22] {
            let cfg = Config { coalesce_window: window, ..Config::default() };
            let (t, forced) = tokens_forced(&cfg, &p);
            assert!(t.len() > 3 * BATCH, "{} note-ons is not several batches", t.len());
            assert_eq!(forced > 0, window < 1 << 22, "window {window}: {forced} forced");
            merged = merged.max(t.iter().filter(|t| t.is_some()).count());
        }
        assert!(merged > 10_000, "only {merged} note-ons could merge");
        let _ = std::fs::remove_file(&p);
    }

    /// A driver that goes, or resumes somewhere else, while its reader is \[39\]
    #[test]
    fn a_reader_running_ahead_is_stopped_and_resynced() {
        let dir = std::env::temp_dir().join("kestrel_coalesce_unit");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("stopped.mid");
        crate::testkit::layered_midi(&p, 6, 3_000, 78).unwrap();
        let cfg = Config::default();
        let first = |look: &mut Lookahead, s: &mut MidiStream, n: usize| {
            let mut out = Vec::new();
            while out.len() < n {
                let Some((_, ev)) = s.next() else { break };
                if let Event::NoteOn { ch, key, vel } = ev {
                    out.push(look.next_release(ch, key, vel));
                }
            }
            out
        };
        // Dropped with its queue full.
        let mut look = Lookahead::new(&cfg, MidiStream::open(&p).unwrap());
        let mut s = MidiStream::open(&p).unwrap();
        let a = first(&mut look, &mut s, 10);
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(look);
        // Resynced to the start while running: the same tokens again.
        let mut look = Lookahead::new(&cfg, MidiStream::open(&p).unwrap());
        let mut s = MidiStream::open(&p).unwrap();
        let mut e = crate::snap::Enc::new();
        s.save_state(&mut e).unwrap();
        let fresh = GateTable::new(&cfg);
        let mut g = crate::snap::Enc::new();
        fresh.save_state(&mut g);
        assert_eq!(first(&mut look, &mut s, 10), a);
        look.resync(e.as_bytes(), None, g.as_bytes(), None).unwrap();
        let mut s = MidiStream::open(&p).unwrap();
        assert_eq!(first(&mut look, &mut s, 10), a);
        assert!(!look.broken());
        let _ = std::fs::remove_file(&p);
    }
}
