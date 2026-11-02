// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Knowing a note's length before rendering the block it starts in. \[1\]

use std::collections::VecDeque;

/// A note-on that has been matched with its note-off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PairedNote {
    pub channel: u8,
    pub key: u8,
    pub vel: u8,
    /// Absolute frame of the note-on. This is where the impulse goes.
    pub start: u64,
    pub duration: u32,
    /// True when `max_wait` ran out before the note-off arrived, so `duration` \[2\]
    pub clamped: bool,
}

#[derive(Clone, Copy)]
struct Open {
    channel: u8,
    key: u8,
    vel: u8,
    start: u64,
    duration: Option<u32>,
    emitted: bool,
}

/// Pairs note-ons with note-offs from a time-ordered stream.
pub struct Horizon {
    max_wait: u64,
    /// Open notes in note-on order. Retired from the front once emitted *and* \[3\]
    open: VecDeque<Open>,
    /// Absolute index of `open.front()`, so slot entries stay valid as the \[4\]
    base: u64,
    /// `channel * 128 + key` -> absolute indices, oldest first, with `channel` \[5\]
    slots: Vec<VecDeque<u64>>,
    unpaired_offs: u64,
}

impl Horizon {
    pub fn new(max_wait: u64) -> Self {
        Self {
            max_wait: max_wait.max(1),
            open: VecDeque::new(),
            base: 0,
            slots: vec![VecDeque::new(); crate::midi::CHANNELS * 128],
            unpaired_offs: 0,
        }
    }

    fn slot(channel: u8, key: u8) -> usize {
        channel as usize * 128 + (key as usize & 127)
    }

    pub fn note_on(&mut self, channel: u8, key: u8, vel: u8, frame: u64) {
        let idx = self.base + self.open.len() as u64;
        self.open.push_back(Open {
            channel,
            key,
            vel,
            start: frame,
            duration: None,
            emitted: false,
        });
        self.slots[Self::slot(channel, key)].push_back(idx);
    }

    /// Pair a note-off. Silently counted if it matches nothing, which real \[6\]
    pub fn note_off(&mut self, channel: u8, key: u8, frame: u64) {
        let Some(idx) = self.slots[Self::slot(channel, key)].pop_front() else {
            self.unpaired_offs += 1;
            return;
        };
        let Some(o) = self.at_mut(idx) else {
            // [7]
            return;
        };
        if o.duration.is_none() {
            o.duration = Some((frame.saturating_sub(o.start)) as u32);
        }
    }

    fn at_mut(&mut self, idx: u64) -> Option<&mut Open> {
        let off = idx.checked_sub(self.base)? as usize;
        self.open.get_mut(off)
    }

    /// Emit every note that is resolved as of stream position `now`. \[8\]
    pub fn advance(&mut self, now: u64, out: &mut Vec<PairedNote>) {
        let mut i = 0usize;
        while i < self.open.len() {
            let o = self.open[i];
            if o.emitted {
                i += 1;
                continue;
            }
            let (duration, clamped) = match o.duration {
                Some(d) => (d, false),
                None if o.start + self.max_wait <= now => (self.max_wait as u32, true),
                // [9]
                None => break,
            };
            out.push(PairedNote {
                channel: o.channel,
                key: o.key,
                vel: o.vel,
                start: o.start,
                duration,
                clamped,
            });
            self.open[i].emitted = true;
            if clamped {
                // [10]
                self.open[i].duration = Some(duration);
            }
            i += 1;
        }
        self.retire();
    }

    /// Drop entries the front no longer needs. An entry is only safe to retire \[11\]
    fn retire(&mut self) {
        while let Some(o) = self.open.front() {
            if !o.emitted {
                break;
            }
            let idx = self.base;
            let still_queued = self.slots[Self::slot(o.channel, o.key)]
                .front()
                .is_some_and(|&q| q == idx);
            if still_queued {
                break;
            }
            self.open.pop_front();
            self.base += 1;
        }
    }

    /// The frame through which every note-on has been resolved. \[12\]
    pub fn ready_through(&self, now: u64) -> u64 {
        self.open
            .iter()
            .find(|o| !o.emitted)
            .map(|o| o.start)
            .unwrap_or(now)
    }

    /// Notes currently held open. The lookahead's real memory cost, and the \[13\]
    pub fn open_notes(&self) -> usize {
        self.open.iter().filter(|o| !o.emitted).count()
    }

    pub fn unpaired_offs(&self) -> u64 {
        self.unpaired_offs
    }

    /// End of stream: everything still open is clamped and emitted.
    pub fn finish(&mut self, out: &mut Vec<PairedNote>) {
        let end = self
            .open
            .iter()
            .filter(|o| !o.emitted)
            .map(|o| o.start)
            .max()
            .map(|s| s + self.max_wait)
            .unwrap_or(0);
        self.advance(end, out);
    }
}
