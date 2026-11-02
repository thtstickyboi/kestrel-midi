// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The MIDI merge on worker threads, `MidiStream::set_threads` \[1\]

use super::{Event, Held, Layout, Merge, TrackReader};
use anyhow::{bail, Result};
use std::cmp::Reverse;
use std::collections::binary_heap::PeekMut;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Chunks a worker may have ready before it waits for the caller.
const DEPTH: usize = 2;
/// Events a chunk holds: at least this many, and at least `PER_TRACK` a track \[2\]
const CHUNK_MIN: usize = 16 * 1024;
const PER_TRACK: usize = 32;
const CHUNK_MAX: usize = 1 << 20;
/// The window of a reader opened for a replay, which reads one chunk's events \[3\]
const REPLAY_WINDOW: usize = 4096;

/// A run of events from one worker, and where its tracks stood before them.
struct Chunk {
    events: Vec<(u64, Event)>,
    /// The worker's tracks, in order, before the first of `events`.
    snap: Vec<Held>,
    /// Bytes of the run not yet decoded at `snap`, for `MidiStream::bytes_read`.
    unread: u64,
    /// The run is over after these.
    last: bool,
}

/// One worker, as the caller sees it.
struct Part {
    /// Its first track, in the stream's numbering.
    first: usize,
    rx: Receiver<Chunk>,
    /// Chunks taken, back to the worker to fill again.
    back: Sender<Chunk>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Merge>>,
    /// The chunk being taken, and how far. Never used up while `!done`: the \[4\]
    cur: Chunk,
    at: usize,
    done: bool,
    /// Time the caller has spent waiting for a chunk from this worker.
    waited: std::time::Duration,
}

/// The merge on worker threads.
pub(crate) struct Par {
    parts: Vec<Part>,
    /// (tick of a part's next event, part) for each part not done. Empty, and \[5\]
    heap: BinaryHeap<Reverse<(u64, u32)>>,
}

/// How many tracks go to each of `k` runs of neighbours, balanced by `weights`: \[6\]
fn split(weights: &[u64], k: usize) -> Vec<usize> {
    let n = weights.len();
    let k = k.clamp(1, n.max(1));
    let total: u128 = weights.iter().map(|&w| w as u128 + 1).sum();
    let (mut sizes, mut acc, mut start) = (Vec::with_capacity(k), 0u128, 0usize);
    for (i, &w) in weights.iter().enumerate() {
        acc += w as u128 + 1;
        let closed = sizes.len();
        if closed + 1 < k
            && (acc * k as u128 >= total * (closed as u128 + 1) || n - (i + 1) == k - closed - 1)
        {
            sizes.push(i + 1 - start);
            start = i + 1;
        }
    }
    sizes.push(n - start);
    sizes
}

/// A worker: merge the run into chunks until it is over or the caller goes.
fn work(mut m: Merge, cap: usize, tx: SyncSender<Chunk>, back: Receiver<Chunk>, stop: Arc<AtomicBool>) -> Merge {
    loop {
        let mut c = back.try_recv().unwrap_or_else(|_| Chunk {
            events: Vec::with_capacity(cap),
            snap: Vec::with_capacity(m.readers.len()),
            unread: 0,
            last: false,
        });
        m.snapshot(&mut c.snap);
        c.unread = m.unread();
        c.events.clear();
        c.last = false;
        while c.events.len() < cap {
            match m.next() {
                Some(e) => c.events.push(e),
                None => {
                    c.last = true;
                    break;
                }
            }
        }
        let last = c.last;
        if stop.load(Ordering::Relaxed) || tx.send(c).is_err() || last {
            return m;
        }
    }
}

impl Par {
    /// Share `merge`'s tracks among `threads` workers, and start them, with \[7\]
    pub(crate) fn start(merge: Merge, threads: usize, chunk: usize) -> Par {
        let Merge { readers, pending, .. } = merge;
        let weights: Vec<u64> = readers.iter().map(|r| if r.ended { 0 } else { r.unread() }).collect();
        let sizes = split(&weights, threads);
        let (mut readers, mut pending) = (readers.into_iter(), pending.into_iter());
        let mut parts = Vec::with_capacity(sizes.len());
        let mut first = 0;
        for size in sizes {
            let m = Merge::new(readers.by_ref().take(size).collect(), pending.by_ref().take(size).collect());
            let cap = match chunk {
                0 => (size * PER_TRACK).clamp(CHUNK_MIN, CHUNK_MAX),
                n => n,
            };
            let (tx, rx) = sync_channel(DEPTH);
            let (back, spent) = channel();
            let stop = Arc::new(AtomicBool::new(false));
            let halt = stop.clone();
            let thread = std::thread::Builder::new()
                .name("kestrel-midi".into())
                .spawn(move || work(m, cap, tx, spent, halt))
                .expect("a thread to read the MIDI file on");
            let empty = Chunk { events: Vec::new(), snap: Vec::new(), unread: 0, last: false };
            parts.push(Part {
                first,
                rx,
                back,
                stop,
                thread: Some(thread),
                cur: empty,
                at: 0,
                done: false,
                waited: std::time::Duration::ZERO,
            });
            first += size;
        }
        let mut par = Par { parts, heap: BinaryHeap::new() };
        for p in &mut par.parts {
            p.fetch();
        }
        if par.parts.len() > 1 {
            par.heap = (par.parts.iter().enumerate())
                .filter(|(_, p)| !p.done)
                .map(|(i, p)| Reverse((p.cur.events[p.at].0, i as u32)))
                .collect();
        }
        par
    }

    #[inline]
    pub(crate) fn next(&mut self) -> Option<(u64, Event)> {
        if let [p] = &mut self.parts[..] {
            if p.done {
                return None;
            }
            return Some(p.take());
        }
        let mut top = self.heap.peek_mut()?;
        let Reverse((_, i)) = *top;
        let p = &mut self.parts[i as usize];
        let e = p.take();
        if p.done {
            PeekMut::pop(top);
        } else {
            *top = Reverse((p.cur.events[p.at].0, i));
        }
        Some(e)
    }

    /// The tick of the event `next` would return.
    pub(crate) fn peek_tick(&self) -> Option<u64> {
        if let [p] = &self.parts[..] {
            return (!p.done).then(|| p.cur.events[p.at].0);
        }
        self.heap.peek().map(|&Reverse((t, _))| t)
    }

    /// Time the caller has spent waiting for the workers, in all.
    pub(crate) fn waited(&self) -> std::time::Duration {
        self.parts.iter().map(|p| p.waited).sum()
    }

    /// Bytes not yet decoded as of the chunks being taken.
    pub(crate) fn unread(&self) -> u64 {
        self.parts.iter().filter(|p| !p.done).map(|p| p.cur.unread).sum()
    }

    /// Every track as a merge on the caller's thread would hold it after the \[8\]
    pub(crate) fn replay(&self, layout: &Layout) -> Result<Vec<Held>> {
        let mut out = Vec::with_capacity(layout.chunks.len());
        for p in &self.parts {
            out.extend(p.replay(layout)?);
        }
        Ok(out)
    }

    /// Stop the workers and take their tracks back, as one merge in the \[9\]
    pub(crate) fn stop(mut self) -> std::thread::Result<Merge> {
        let mut readers = Vec::new();
        let mut pending = Vec::new();
        let mut lost = None;
        for p in std::mem::take(&mut self.parts) {
            match p.stop() {
                Some(Ok(m)) => {
                    readers.extend(m.readers);
                    pending.extend(m.pending);
                }
                Some(Err(e)) => lost = Some(e),
                None => {}
            }
        }
        match lost {
            Some(e) => Err(e),
            None => Ok(Merge::new(readers, pending)),
        }
    }
}

impl Drop for Par {
    fn drop(&mut self) {
        // [10]
        for p in std::mem::take(&mut self.parts) {
            let _ = p.stop();
        }
    }
}

impl Part {
    /// The next event of the chunk being taken, which there must be, and the \[11\]
    #[inline]
    fn take(&mut self) -> (u64, Event) {
        let e = self.cur.events[self.at];
        self.at += 1;
        if self.at == self.cur.events.len() {
            self.fetch();
        }
        e
    }

    /// Move to the worker's next chunk with an event in it, or to `done`.
    #[cold]
    fn fetch(&mut self) {
        while !self.cur.last {
            // A clock read a chunk only, and only when the worker is behind.
            let ready = self.rx.try_recv().ok();
            let next = match ready.map(Ok).unwrap_or_else(|| {
                let t = std::time::Instant::now();
                let c = self.rx.recv();
                self.waited += t.elapsed();
                c
            }) {
                Ok(c) => c,
                // [12]
                Err(_) => match self.thread.take().map(JoinHandle::join) {
                    Some(Err(e)) => std::panic::resume_unwind(e),
                    _ => panic!("a thread reading the MIDI file stopped before the end of it"),
                },
            };
            let used = std::mem::replace(&mut self.cur, next);
            // [13]
            if used.events.capacity() > 0 {
                let _ = self.back.send(used);
            }
            self.at = 0;
            if !self.cur.events.is_empty() {
                return;
            }
        }
        // The last chunk stays, used up: it is what `replay` reads from.
        self.at = self.cur.events.len();
        self.done = true;
    }

    /// Stop the worker and take its merge back; `None` if it had been taken.
    fn stop(mut self) -> Option<std::thread::Result<Merge>> {
        self.stop.store(true, Ordering::Relaxed);
        let thread = self.thread.take()?;
        // A worker waiting to hand a chunk over is woken by its queue closing.
        drop(self.rx);
        drop(self.back);
        Some(thread.join())
    }

    /// This run's tracks as a merge on the caller's thread would hold them \[14\]
    fn replay(&self, layout: &Layout) -> Result<Vec<Held>> {
        let mut held = self.cur.snap.clone();
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = (held.iter().enumerate())
            .filter(|(_, h)| h.pending.is_some())
            .map(|(i, h)| Reverse((h.pos.tick, i as u32)))
            .collect();
        // [15]
        let mut readers: Vec<Option<TrackReader>> = Vec::new();
        readers.resize_with(held.len(), || None);
        for &(tick, ev) in &self.cur.events[..self.at] {
            let Some(Reverse((t, i))) = heap.pop() else {
                bail!("reading the MIDI file again from a snapshot ran out of events early; this is a defect");
            };
            let i = i as usize;
            if (t, held[i].pending.take()) != (tick, Some(ev)) {
                bail!("reading the MIDI file again from a snapshot gave a different event; this is a defect");
            }
            let r = match &mut readers[i] {
                Some(r) => r,
                slot => {
                    let mut r = layout.reader(self.first + i, REPLAY_WINDOW)?;
                    r.set_pos(held[i].pos.clone())?;
                    slot.insert(r)
                }
            };
            if let Some(next) = r.next_event() {
                held[i].pending = Some(next);
                heap.push(Reverse((r.tick, i as u32)));
            }
        }
        for (h, r) in held.iter_mut().zip(&readers) {
            if let Some(r) = r {
                h.pos = r.pos();
            }
        }
        Ok(held)
    }
}

#[cfg(test)]
mod tests {
    use super::split;

    #[test]
    fn runs_are_neighbours_balanced_and_never_empty() {
        assert_eq!(split(&[], 4), vec![0]);
        assert_eq!(split(&[5], 4), vec![1]);
        assert_eq!(split(&[100, 1, 1], 3), vec![1, 1, 1]);
        assert_eq!(split(&[1, 1, 1, 1], 2), vec![2, 2]);
        assert_eq!(split(&[1, 1, 1, 100], 2), vec![3, 1]);
        assert_eq!(split(&[100, 1, 1, 1], 2), vec![1, 3]);
        for n in 1..40usize {
            for k in 1..8usize {
                let w: Vec<u64> = (0..n).map(|i| (i as u64 * 7919) % 13).collect();
                let s = split(&w, k);
                assert_eq!(s.len(), k.min(n));
                assert_eq!(s.iter().sum::<usize>(), n);
                assert!(s.iter().all(|&x| x > 0));
            }
        }
    }
}
