// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Streaming SMF parser and track merger. \[1\]

use crate::mts;
use crate::snap::{Dec, Enc};
use anyhow::{bail, Context, Result};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod par;

/// The largest window a track reads through. A reader that is alone, like the \[2\]
pub(crate) const TRACK_BUF: usize = 256 * 1024;

/// The smallest window a stream gives a track, however many tracks share the \[3\]
const TRACK_BUF_MIN: usize = 8 * 1024;

/// Bytes of read window one render's MIDI readers share: every track of every \[4\]
pub const READ_BUDGET: u64 = 1 << 30;

/// Bytes of an FF 03 track name a reader keeps; the rest is skipped. Names in \[5\]
pub(crate) const NAME_MAX: u64 = 255;

/// Bytes of a SysEx a reader looks at: a scale/octave tuning message is 31 \[6\]
const SYSEX_HEAD: usize = 32;

/// MIDI ports a file can address. A port is one more set of sixteen channels, \[7\]
pub const PORTS: u8 = 16;
/// Channels across every port, and the range of an event's `ch`: all 256 \[8\]
pub const CHANNELS: usize = 16 * PORTS as usize;

/// One decoded event. \[9\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    NoteOn { ch: u8, key: u8, vel: u8 },
    NoteOff { ch: u8, key: u8 },
    Cc { ch: u8, num: u8, val: u8 },
    Program { ch: u8, val: u8 },
    PitchBend { ch: u8, val: i16 },
    /// Roland GS "USE FOR RHYTHM PART". `map` is 0 for a melodic part and 1 or \[10\]
    DrumPart { ch: u8, map: u8 },
    /// GM System On, GM System Off or GS Reset. Puts every channel back to the \[11\]
    ResetParts,
    /// Microseconds per quarter note.
    Tempo(u32),
    /// MIDI Tuning Standard scale/octave tuning, one pitch class of one \[12\]
    Tune { ch: u8, pc: u8, units: i32 },
    /// MIDI Tuning Standard single-note change or bulk dump, one key of one tuning \[13\]
    KeyTune { bank: u8, prog: u8, key: u8, units: i32 },
    /// Anything the synth does not act on. Kept in the stream so callers can \[14\]
    Other,
}

impl Event {
    /// For a snapshot: a tag, then the fields.
    pub fn save(&self, e: &mut Enc) {
        match *self {
            Event::NoteOn { ch, key, vel } => {
                e.u8(0);
                e.u8(ch);
                e.u8(key);
                e.u8(vel);
            }
            Event::NoteOff { ch, key } => {
                e.u8(1);
                e.u8(ch);
                e.u8(key);
            }
            Event::Cc { ch, num, val } => {
                e.u8(2);
                e.u8(ch);
                e.u8(num);
                e.u8(val);
            }
            Event::Program { ch, val } => {
                e.u8(3);
                e.u8(ch);
                e.u8(val);
            }
            Event::PitchBend { ch, val } => {
                e.u8(4);
                e.u8(ch);
                e.i16(val);
            }
            Event::DrumPart { ch, map } => {
                e.u8(5);
                e.u8(ch);
                e.u8(map);
            }
            Event::ResetParts => e.u8(6),
            Event::Tempo(us) => {
                e.u8(7);
                e.u32(us);
            }
            Event::Other => e.u8(8),
            Event::Tune { ch, pc, units } => {
                e.u8(9);
                e.u8(ch);
                e.u8(pc);
                e.i32(units);
            }
            Event::KeyTune { bank, prog, key, units } => {
                e.u8(10);
                e.u8(bank);
                e.u8(prog);
                e.u8(key);
                e.i32(units);
            }
        }
    }

    pub fn load(d: &mut Dec) -> Result<Event> {
        Ok(match d.u8()? {
            0 => Event::NoteOn { ch: d.u8()?, key: d.u8()?, vel: d.u8()? },
            1 => Event::NoteOff { ch: d.u8()?, key: d.u8()? },
            2 => Event::Cc { ch: d.u8()?, num: d.u8()?, val: d.u8()? },
            3 => Event::Program { ch: d.u8()?, val: d.u8()? },
            4 => Event::PitchBend { ch: d.u8()?, val: d.i16()? },
            5 => Event::DrumPart { ch: d.u8()?, map: d.u8()? },
            6 => Event::ResetParts,
            7 => Event::Tempo(d.u32()?),
            8 => Event::Other,
            9 => Event::Tune { ch: d.u8()?, pc: d.u8()?, units: d.i32()? },
            10 => Event::KeyTune { bank: d.u8()?, prog: d.u8()?, key: d.u8()?, units: d.i32()? },
            t => bail!("the saved state holds a MIDI event of kind {t}, which this build does not know"),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Division {
    /// Ticks per quarter note.
    Ppq(u16),
    /// SMPTE: frames per second and ticks per frame.
    Smpte { fps: u8, ticks_per_frame: u8 },
}

/// What is left of a tuning message, which is read as a run of events, one for \[15\]
#[derive(Debug, Clone, PartialEq)]
enum Burst {
    Scale {
        msg: mts::ScaleOctave,
        /// The next of the 16 channels x 12 pitch classes to look at.
        at: u16,
    },
    Keys {
        msg: mts::KeyChanges,
        /// The next key of `msg` to hand out.
        at: u16,
    },
}

impl Burst {
    fn next(&mut self, port_base: u8) -> Option<Event> {
        match self {
            Burst::Scale { msg, at } => {
                while *at < 16 * 12 {
                    let (c, pc) = ((*at / 12) as u8, (*at % 12) as u8);
                    *at += 1;
                    if msg.channels >> c & 1 != 0 {
                        return Some(Event::Tune { ch: port_base | c, pc, units: msg.units[pc as usize] });
                    }
                }
                None
            }
            Burst::Keys { msg, at } => {
                let &(key, units) = msg.keys.get(*at as usize)?;
                *at += 1;
                Some(Event::KeyTune { bank: msg.bank, prog: msg.program, key, units })
            }
        }
    }
}

/// Where a track's reader is, apart from its window: how many bytes of its \[16\]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReaderPos {
    unread: u64,
    tick: u64,
    running: u8,
    ended: bool,
    port_base: u8,
    /// A tuning message part way through its run.
    burst: Option<Box<Burst>>,
}

impl ReaderPos {
    /// For a snapshot. The window is not kept; it is read again from the byte \[17\]
    fn save(&self, e: &mut Enc) {
        e.u64(self.unread);
        e.u64(self.tick);
        e.u8(self.running);
        e.bool(self.ended);
        e.u8(self.port_base);
        // [18]
        match self.burst.as_deref() {
            None => e.u8(0),
            Some(Burst::Scale { msg, at }) => {
                e.u8(1);
                e.u16(msg.channels);
                for u in msg.units {
                    e.i32(u);
                }
                e.u16(*at);
            }
            Some(Burst::Keys { msg, at }) => {
                e.u8(2);
                e.u8(msg.bank);
                e.u8(msg.program);
                e.len_of(msg.keys.len());
                for &(key, units) in &msg.keys {
                    e.u8(key);
                    e.i32(units);
                }
                e.u16(*at);
            }
        }
    }

    fn load(d: &mut Dec) -> Result<Self> {
        let unread = d.u64()?;
        let (tick, running, ended, port_base) = (d.u64()?, d.u8()?, d.bool()?, d.u8()?);
        let burst = match d.u8()? {
            0 => None,
            1 => {
                let channels = d.u16()?;
                let mut units = [0i32; 12];
                for u in &mut units {
                    *u = d.i32()?;
                }
                let at = d.u16()?;
                if at > 16 * 12 {
                    bail!("the saved tuning message is {at} events in and has {}", 16 * 12);
                }
                Some(Box::new(Burst::Scale { msg: mts::ScaleOctave { channels, units }, at }))
            }
            2 => {
                let (bank, program) = (d.u8()?, d.u8()?);
                let n = d.len_of(5)?;
                let mut keys = Vec::with_capacity(n);
                for _ in 0..n {
                    keys.push((d.u8()?, d.i32()?));
                }
                let at = d.u16()?;
                if at as usize > keys.len() {
                    bail!("the saved tuning message is {at} keys in and has {}", keys.len());
                }
                Some(Box::new(Burst::Keys { msg: mts::KeyChanges { bank, program, keys }, at }))
            }
            t => bail!("the saved state holds a tuning message of kind {t}, which this build does not know"),
        };
        Ok(ReaderPos { unread, tick, running, ended, port_base, burst })
    }
}

/// The window each of a stream's tracks reads through, from their chunk \[19\]
pub(crate) fn window_share(lens: &[u64], budget: u64) -> usize {
    let mut lens = lens.to_vec();
    lens.sort_unstable();
    let mut left = budget;
    for (i, &len) in lens.iter().enumerate() {
        let share = left / (lens.len() - i) as u64;
        if len > share {
            // [20]
            return share.clamp(TRACK_BUF_MIN as u64, TRACK_BUF as u64) as usize;
        }
        left -= len;
    }
    TRACK_BUF
}

/// A single track chunk with its own file cursor and read window.
pub(crate) struct TrackReader {
    file: File,
    buf: Box<[u8]>,
    pos: usize,
    filled: usize,
    /// Bytes of the track chunk not yet pulled into `buf`.
    remaining: u64,
    /// Where the chunk ends in the file, which is how a saved position is \[21\]
    end: u64,
    /// The tick of the event `next_event` last returned.
    pub(crate) tick: u64,
    running: u8,
    ended: bool,
    /// This track's port, already folded to `PORTS` and shifted into place: \[22\]
    port_base: u8,
    /// Keep the track's first FF 03 name in `name`. Only the track scan asks; \[23\]
    pub(crate) keep_name: bool,
    pub(crate) name: Option<Vec<u8>>,
    /// Ignore FF 21, so every event stays on port A. A per-track render sets \[24\]
    fold_ports: bool,
    /// What a note key is ANDed with: `0x7F`, which makes it a key, or `0xFF` \[25\]
    key_mask: u8,
    /// A tuning message being handed out. Checked once an event, which costs \[26\]
    burst: Option<Box<Burst>>,
}

impl TrackReader {
    /// A reader over the `len` bytes of chunk at `offset`, pulling `window` \[27\]
    pub(crate) fn open(path: &Path, offset: u64, len: u64, window: usize) -> Result<Self> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        Ok(TrackReader {
            file,
            buf: vec![0u8; len.min(window.max(1) as u64) as usize].into_boxed_slice(),
            pos: 0,
            filled: 0,
            remaining: len,
            end: offset + len,
            tick: 0,
            running: 0,
            ended: false,
            port_base: 0,
            keep_name: false,
            name: None,
            fold_ports: false,
            key_mask: 0x7F,
            burst: None,
        })
    }

    /// Bytes of the chunk not yet decoded: still on disk, or read into the \[28\]
    #[inline]
    pub(crate) fn unread(&self) -> u64 {
        self.remaining + (self.filled - self.pos) as u64
    }

    /// Where the reader is, apart from its window. See `ReaderPos`.
    pub(crate) fn pos(&self) -> ReaderPos {
        ReaderPos {
            unread: self.unread(),
            tick: self.tick,
            running: self.running,
            ended: self.ended,
            port_base: self.port_base,
            burst: self.burst.clone(),
        }
    }

    /// Go on from `pos`, which a reader of this same chunk was at. The window \[29\]
    pub(crate) fn set_pos(&mut self, pos: ReaderPos) -> Result<()> {
        let Some(start) = self.end.checked_sub(pos.unread) else {
            bail!("the saved position of a MIDI track is before the start of the file");
        };
        self.file.seek(SeekFrom::Start(start))?;
        self.remaining = pos.unread;
        self.pos = 0;
        self.filled = 0;
        self.tick = pos.tick;
        self.running = pos.running;
        self.ended = pos.ended;
        self.port_base = pos.port_base;
        self.burst = pos.burst;
        Ok(())
    }

    fn load(&mut self, d: &mut Dec) -> Result<()> {
        self.set_pos(ReaderPos::load(d)?)
    }

    #[inline]
    fn fill(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        let want = self.remaining.min(self.buf.len() as u64) as usize;
        let mut got = 0usize;
        while got < want {
            match self.file.read(&mut self.buf[got..want]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        self.remaining -= got as u64;
        self.pos = 0;
        self.filled = got;
        got > 0
    }

    #[inline(always)]
    fn byte(&mut self) -> Option<u8> {
        if self.pos == self.filled && !self.fill() {
            return None;
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Some(b)
    }

    #[inline]
    fn varlen(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for _ in 0..8 {
            let b = self.byte()?;
            v = (v << 7) | (b & 0x7F) as u64;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    fn skip(&mut self, mut n: u64) -> Option<()> {
        while n > 0 {
            if self.pos == self.filled && !self.fill() {
                return None;
            }
            let avail = (self.filled - self.pos) as u64;
            let take = avail.min(n);
            self.pos += take as usize;
            n -= take;
        }
        Some(())
    }

    /// A channel message and its delta, decoded straight out of the window when \[30\]
    #[inline(always)]
    fn channel_event_in_window(&mut self) -> Option<Event> {
        if self.filled - self.pos < 16 {
            return None;
        }
        let b: &[u8; 16] = self.buf[self.pos..self.pos + 16].try_into().ok()?;
        let (mut delta, mut n) = (0u64, 0usize);
        loop {
            let x = b[n & 15];
            delta = (delta << 7) | (x & 0x7F) as u64;
            n += 1;
            if x & 0x80 == 0 {
                break;
            }
            if n == 8 {
                return None;
            }
        }
        let explicit = b[n & 15] >= 0x80;
        let (status, first, mut used) = if explicit {
            (b[n & 15], b[(n + 1) & 15], n + 2)
        } else {
            (self.running, b[n & 15], n + 1)
        };
        // Running status with nothing running, a meta event, SysEx: not decided here.
        if !(0x80..0xF0).contains(&status) {
            return None;
        }
        let ch = self.port_base | (status & 0x0F);
        let ev = match status & 0xF0 {
            0x80 => {
                used += 1;
                Event::NoteOff { ch, key: first & self.key_mask }
            }
            0x90 => {
                let vel = b[used & 15];
                used += 1;
                if vel == 0 {
                    Event::NoteOff { ch, key: first & self.key_mask }
                } else {
                    Event::NoteOn { ch, key: first & self.key_mask, vel: vel & 0x7F }
                }
            }
            0xA0 => {
                used += 1;
                Event::Other
            }
            0xB0 => {
                let val = b[used & 15];
                used += 1;
                Event::Cc { ch, num: first & 0x7F, val: val & 0x7F }
            }
            0xC0 => Event::Program { ch, val: first & 0x7F },
            0xD0 => Event::Other,
            _ => {
                let msb = b[used & 15];
                used += 1;
                let raw = ((msb as i16 & 0x7F) << 7) | (first as i16 & 0x7F);
                Event::PitchBend { ch, val: raw - 8192 }
            }
        };
        self.tick += delta;
        self.pos += used;
        if explicit {
            self.running = status;
        }
        Some(ev)
    }

    /// Step over the channel events a pass has no use for -- notes on keys \[31\]
    #[inline]
    pub(crate) fn skip_unkept(&mut self, end: u64, keep: &[bool; 128]) -> u64 {
        if self.burst.is_some() || self.ended {
            return 0;
        }
        let (mut pos, mut tick, mut running) = (self.pos, self.tick, self.running);
        let mut n = 0u64;
        while self.filled - pos >= 16 {
            let b: &[u8; 16] = self.buf[pos..pos + 16].try_into().expect("16 bytes");
            let (mut delta, mut i) = (0u64, 0usize);
            let mut long = false;
            loop {
                let x = b[i & 15];
                delta = (delta << 7) | (x & 0x7F) as u64;
                i += 1;
                if x & 0x80 == 0 {
                    break;
                }
                if i == 8 {
                    long = true;
                    break;
                }
            }
            if long || tick + delta >= end {
                break;
            }
            let explicit = b[i & 15] >= 0x80;
            let (status, first, mut used) = if explicit { (b[i & 15], b[(i + 1) & 15], i + 2) } else { (running, b[i & 15], i + 1) };
            match status & 0xF0 {
                0x80 | 0x90 if status < 0xF0 && !keep[(first & 0x7F) as usize] => used += 1,
                0xA0 if status < 0xF0 => used += 1,
                0xD0 if status < 0xF0 => {}
                _ => break,
            }
            tick += delta;
            pos += used;
            if explicit {
                running = status;
            }
            n += 1;
        }
        self.pos = pos;
        self.tick = tick;
        self.running = running;
        n
    }

    /// Decode one event, advancing `tick`. Returns None at end of track.
    #[inline]
    pub(crate) fn next_event(&mut self) -> Option<Event> {
        if self.burst.is_none() && !self.ended {
            if let Some(ev) = self.channel_event_in_window() {
                return Some(ev);
            }
        }
        self.next_event_general()
    }

    /// `next_event` a byte at a time: every kind of event, and the window's end.
    fn next_event_general(&mut self) -> Option<Event> {
        if let Some(b) = self.burst.as_mut() {
            if let Some(ev) = b.next(self.port_base) {
                return Some(ev);
            }
            self.burst = None;
        }
        if self.ended {
            return None;
        }
        let delta = match self.varlen() {
            Some(d) => d,
            None => {
                self.ended = true;
                return None;
            }
        };
        self.tick += delta;

        let mut status = match self.byte() {
            Some(b) => b,
            None => {
                self.ended = true;
                return None;
            }
        };

        // [32]
        let first_data;
        if status < 0x80 {
            first_data = status;
            status = self.running;
            if status < 0x80 {
                // Garbage before any status byte was ever seen.
                self.ended = true;
                return None;
            }
        } else {
            if status < 0xF0 {
                self.running = status;
            } else if status != 0xF7 && status != 0xF0 {
                // [33]
                if status == 0xFF {
                    self.running = 0;
                }
            }
            first_data = match status {
                0xFF | 0xF0 | 0xF7 => 0,
                _ => match self.byte() {
                    Some(b) => b,
                    None => {
                        self.ended = true;
                        return None;
                    }
                },
            };
        }

        let ch = self.port_base | (status & 0x0F);
        match status & 0xF0 {
            0x80 => {
                let _vel = self.byte()?;
                Some(Event::NoteOff { ch, key: first_data & self.key_mask })
            }
            0x90 => {
                let vel = self.byte()?;
                if vel == 0 {
                    Some(Event::NoteOff { ch, key: first_data & self.key_mask })
                } else {
                    Some(Event::NoteOn {
                        ch,
                        key: first_data & self.key_mask,
                        vel: vel & 0x7F,
                    })
                }
            }
            0xA0 => {
                let _ = self.byte()?;
                Some(Event::Other)
            }
            0xB0 => {
                let val = self.byte()?;
                Some(Event::Cc {
                    ch,
                    num: first_data & 0x7F,
                    val: val & 0x7F,
                })
            }
            0xC0 => Some(Event::Program { ch, val: first_data & 0x7F }),
            0xD0 => Some(Event::Other),
            0xE0 => {
                let msb = self.byte()?;
                let raw = ((msb as i16 & 0x7F) << 7) | (first_data as i16 & 0x7F);
                Some(Event::PitchBend { ch, val: raw - 8192 })
            }
            _ => match status {
                0xFF => {
                    let meta = self.byte()?;
                    let len = self.varlen()?;
                    if meta == 0x2F {
                        self.skip(len)?;
                        self.ended = true;
                        return Some(Event::Other);
                    }
                    if meta == 0x51 && len == 3 {
                        let a = self.byte()? as u32;
                        let b = self.byte()? as u32;
                        let c = self.byte()? as u32;
                        return Some(Event::Tempo((a << 16) | (b << 8) | c));
                    }
                    // [34]
                    if meta == 0x21 && len == 1 {
                        let port = self.byte()?;
                        if !self.fold_ports {
                            self.port_base = (port % PORTS) << 4;
                        }
                        return Some(Event::Other);
                    }
                    if meta == 0x03 && self.keep_name && self.name.is_none() {
                        let n = len.min(NAME_MAX);
                        let mut name = Vec::with_capacity(n as usize);
                        for _ in 0..n {
                            name.push(self.byte()?);
                        }
                        self.skip(len - n)?;
                        self.name = Some(name);
                        return Some(Event::Other);
                    }
                    self.skip(len)?;
                    Some(Event::Other)
                }
                0xF0 | 0xF7 => {
                    let len = self.varlen()?;
                    // [35]
                    let mut head = [0u8; SYSEX_HEAD];
                    let n = len.min(head.len() as u64) as usize;
                    for h in head.iter_mut().take(n) {
                        *h = self.byte()?;
                    }
                    // [36]
                    let mut whole: Option<Vec<u8>> = None;
                    let mut left = len - n as u64;
                    if left > 0 && mts::is_key_tuning(&head[..n]) {
                        let mut v = head[..n].to_vec();
                        let take = left.min((mts::KEYS_MESSAGE_MAX - n) as u64);
                        for _ in 0..take {
                            v.push(self.byte()?);
                        }
                        left -= take;
                        whole = Some(v);
                    }
                    self.skip(left)?;
                    let message: &[u8] = whole.as_deref().unwrap_or(&head[..n]);
                    // [37]
                    if let Some(msg) = mts::parse(message) {
                        let mut b = Burst::Scale { msg, at: 0 };
                        return Some(match b.next(self.port_base) {
                            Some(ev) => {
                                self.burst = Some(Box::new(b));
                                ev
                            }
                            None => Event::Other,
                        });
                    }
                    if let Some(msg) = mts::parse_keys(message) {
                        let mut b = Burst::Keys { msg, at: 0 };
                        return Some(match b.next(self.port_base) {
                            Some(ev) => {
                                self.burst = Some(Box::new(b));
                                ev
                            }
                            None => Event::Other,
                        });
                    }
                    // A part message addresses a channel of this track's port.
                    Some(match sysex_event(&head[..n]) {
                        Event::DrumPart { ch, map } => Event::DrumPart { ch: self.port_base | ch, map },
                        e => e,
                    })
                }
                _ => Some(Event::Other),
            },
        }
    }
}

/// Recognise the SysEx messages that change how a channel resolves. \[38\]
fn sysex_event(p: &[u8]) -> Event {
    // Roland GS DT1: 41 <dev> 42 12 <addr hi mid lo> <data..> <sum> F7.
    if p.len() >= 8 && p[0] == 0x41 && p[2] == 0x42 && p[3] == 0x12 {
        // [39]
        if p[4] == 0x40 && p[5] & 0xF0 == 0x10 && p[6] == 0x15 {
            let block = p[5] & 0x0F;
            let ch = match block {
                0 => 9,
                1..=9 => block - 1,
                _ => block,
            };
            return Event::DrumPart { ch, map: p[7] & 0x7F };
        }
        // GS Reset, address 40 00 7F, data 00.
        if p[4] == 0x40 && p[5] == 0x00 && p[6] == 0x7F {
            return Event::ResetParts;
        }
    }
    // GM System On / Off: 7E <dev> 09 <01|02|03>.
    if p.len() >= 4 && p[0] == 0x7E && p[2] == 0x09 && matches!(p[3], 0x01..=0x03) {
        return Event::ResetParts;
    }
    Event::Other
}

/// How much of a file `uses_extended_keys` reads, so that asking costs a \[40\]
const EXTENDED_PER_TRACK: u64 = 128 * 1024;
const EXTENDED_TRACKS: usize = 1024;
const EXTENDED_TOTAL: u64 = 16 << 20;

/// Whether a file has a note-on whose key is over 127, which is how a file \[41\]
pub fn uses_extended_keys(path: impl AsRef<Path>) -> Result<bool> {
    let path = path.as_ref();
    let h = SmfHeader::read(path)?;
    let mut read = 0u64;
    for &(off, len) in h.tracks.iter().take(EXTENDED_TRACKS) {
        if read >= EXTENDED_TOTAL {
            break;
        }
        let want = len.min(EXTENDED_PER_TRACK);
        let mut r = TrackReader::open(path, off, len, TRACK_BUF)?;
        r.key_mask = 0xFF;
        while let Some(ev) = r.next_event() {
            if matches!(ev, Event::NoteOn { key, .. } if key > 127) {
                return Ok(true);
            }
            if len - r.unread() >= want {
                break;
            }
        }
        read += want;
    }
    Ok(false)
}

/// What a standard MIDI file says about itself before any event is decoded: \[42\]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmfHeader {
    pub format: u16,
    pub division: Division,
    /// Tracks the MThd chunk claims. A truncated or hand-edited file can \[43\]
    pub declared_tracks: u16,
    /// `(data offset, length)` of every MTrk chunk, each length clamped to \[44\]
    pub tracks: Vec<(u64, u64)>,
    /// MTrk chunks whose declared length ran past the end of the file, which \[45\]
    pub truncated_tracks: u32,
    /// Bytes after the last chunk that are not chunks: exporter padding.
    pub trailing_bytes: u64,
    pub file_len: u64,
}

impl SmfHeader {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let file_len = file.metadata()?.len();

        let mut hdr = [0u8; 14];
        file.read_exact(&mut hdr)
            .with_context(|| format!("{}: too short for an SMF header", path.display()))?;
        if &hdr[0..4] != b"MThd" {
            bail!("{}: not a standard MIDI file", path.display());
        }
        let hdr_len = u32::from_be_bytes(hdr[4..8].try_into().unwrap()) as u64;
        let format = u16::from_be_bytes(hdr[8..10].try_into().unwrap());
        let declared_tracks = u16::from_be_bytes(hdr[10..12].try_into().unwrap());
        let div_raw = i16::from_be_bytes(hdr[12..14].try_into().unwrap());
        let division = if div_raw > 0 {
            Division::Ppq(div_raw as u16)
        } else {
            Division::Smpte {
                fps: (-(div_raw >> 8)) as u8,
                ticks_per_frame: (div_raw & 0xFF) as u8,
            }
        };

        // [46]
        let mut offset = 8 + hdr_len;
        let mut tracks: Vec<(u64, u64)> = Vec::new();
        let mut truncated_tracks = 0u32;
        let mut trailing_bytes = 0u64;
        while offset + 8 <= file_len {
            file.seek(SeekFrom::Start(offset))?;
            let mut ch = [0u8; 8];
            if file.read_exact(&mut ch).is_err() {
                break;
            }
            let declared = u32::from_be_bytes(ch[4..8].try_into().unwrap()) as u64;
            let data_start = offset + 8;
            let avail = file_len.saturating_sub(data_start);
            let len = declared.min(avail);
            if &ch[0..4] == b"MTrk" {
                if declared > avail {
                    truncated_tracks += 1;
                }
                tracks.push((data_start, len));
            } else if len == 0 {
                // [47]
                trailing_bytes = file_len - offset;
                break;
            }
            offset = data_start + len;
        }

        Ok(SmfHeader {
            format,
            division,
            declared_tracks,
            tracks,
            truncated_tracks,
            trailing_bytes,
            file_len,
        })
    }
}

/// What a per-track render reads: some of a file's tracks, timed by the whole \[48\]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackSelection {
    /// Tracks to read, 0-based: the one being rendered, and the setup tracks \[49\]
    pub tracks: Vec<usize>,
    /// Every FF 51 in the file, in the merged stream's order. Played in place \[50\]
    pub tempo: Arc<[(u64, u32)]>,
    /// The file's chunk table, as the scan read it. Shared for the same \[51\]
    pub header: Arc<SmfHeader>,
    /// The tick of the whole file's last event other than a tempo change. The \[52\]
    pub end_tick: u64,
}

/// A k-way merge of some tracks on one thread: each track's reader, the event \[53\]
pub(crate) struct Merge {
    readers: Vec<TrackReader>,
    /// (tick, track index) so ties break on track order, deterministically.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
    pending: Vec<Option<Event>>,
}

impl Merge {
    /// The readers as they stand, each holding the event it decoded last, if \[54\]
    fn new(readers: Vec<TrackReader>, pending: Vec<Option<Event>>) -> Self {
        let heap = readers
            .iter()
            .zip(&pending)
            .enumerate()
            .filter(|(_, (_, held))| held.is_some())
            .map(|(i, (r, _))| Reverse((r.tick, i as u32)))
            .collect();
        Merge { readers, heap, pending }
    }

    #[inline]
    fn next(&mut self) -> Option<(u64, Event)> {
        // [55]
        let mut top = self.heap.peek_mut()?;
        let Reverse((tick, idx)) = *top;
        let i = idx as usize;
        let ev = self.pending[i].take().expect("heap entry without pending event");

        match self.readers[i].next_event() {
            Some(next) => {
                self.pending[i] = Some(next);
                // One sift-down. The track keeps its slot in the heap.
                *top = Reverse((self.readers[i].tick, idx));
            }
            // [56]
            None => {
                std::collections::binary_heap::PeekMut::pop(top);
            }
        }
        Some((tick, ev))
    }

    /// The tick of the event `next` would return.
    #[inline]
    fn peek_tick(&self) -> Option<u64> {
        self.heap.peek().map(|&Reverse((t, _))| t)
    }

    /// Bytes not yet decoded, a track that has ended counting none.
    fn unread(&self) -> u64 {
        self.readers.iter().filter(|r| !r.ended).map(TrackReader::unread).sum()
    }

    /// Every track's position and held event, in track order, into `out`.
    fn snapshot(&self, out: &mut Vec<Held>) {
        out.clear();
        out.extend(self.readers.iter().zip(&self.pending).map(|(r, &pending)| Held { pos: r.pos(), pending }));
    }
}

/// One track of a snapshot: where its reader is, and the event it holds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Held {
    pos: ReaderPos,
    pending: Option<Event>,
}

impl Held {
    fn save(&self, e: &mut Enc) {
        self.pos.save(e);
        match &self.pending {
            Some(ev) => {
                e.bool(true);
                ev.save(e);
            }
            None => e.bool(false),
        }
    }
}

/// What it takes to open a stream's readers again, from anywhere: the file, \[57\]
pub(crate) struct Layout {
    path: PathBuf,
    /// `(data offset, length)` of each track the stream reads, in its order.
    chunks: Vec<(u64, u64)>,
    fold_ports: bool,
    key_mask: u8,
}

impl Layout {
    /// A reader of the stream's track `t` at its start, with a window of \[58\]
    fn reader(&self, t: usize, window: usize) -> Result<TrackReader> {
        let (off, len) = self.chunks[t];
        let mut r = TrackReader::open(&self.path, off, len, window)?;
        r.fold_ports = self.fold_ports;
        r.key_mask = self.key_mask;
        Ok(r)
    }
}

/// Tick-ordered merge of every track in a standard MIDI file, or of the tracks \[59\]
pub struct MidiStream {
    /// The merge, while it runs on the caller's thread. Its readers move to \[60\]
    merge: Merge,
    /// The merge on worker threads, from the first event on when `threads` is \[61\]
    par: Option<par::Par>,
    threads: usize,
    /// Events a worker's chunk holds; 0 for `par`'s own choice. The tests set \[62\]
    chunk: usize,
    layout: Arc<Layout>,
    /// Sum of every track chunk's length, for `bytes_read`.
    bytes_total: u64,
    /// A selection's tempo map, played beside the tracks, and how far it has \[63\]
    tempo: Arc<[(u64, u32)]>,
    tempo_pos: usize,
    /// `TrackSelection::end_tick`, for a selection.
    end_tick: Option<u64>,
    pub division: Division,
    pub format: u16,
    /// Tracks in the file, whether or not a selection reads them all.
    pub track_count: u16,
    pub path: PathBuf,
}

impl MidiStream {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_keys(path, false, READ_BUDGET)
    }

    /// `open`, reading note keys as the 31-EDO template writes them -- \[64\]
    pub fn open_keys(path: impl AsRef<Path>, wide: bool, read_budget: u64) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let h = SmfHeader::read(&path)?;
        let all: Vec<usize> = (0..h.tracks.len()).collect();
        Self::from_header(path, &h, &all, None, wide, read_budget)
    }

    /// Read only the tracks `sel` names, with every port folded onto port A \[65\]
    pub fn open_tracks(path: impl AsRef<Path>, sel: &TrackSelection) -> Result<Self> {
        Self::open_tracks_keys(path, sel, false, READ_BUDGET)
    }

    /// `open_tracks`, with `open_keys`' `wide` and `read_budget`. The budget \[66\]
    pub fn open_tracks_keys(path: impl AsRef<Path>, sel: &TrackSelection, wide: bool,
        read_budget: u64) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let h = &*sel.header;
        let mut tracks = sel.tracks.clone();
        tracks.sort_unstable();
        tracks.dedup();
        if let Some(&bad) = tracks.iter().find(|&&t| t >= h.tracks.len()) {
            bail!(
                "{}: there is no track {}; the file has {}",
                path.display(),
                bad + 1,
                h.tracks.len()
            );
        }
        Self::from_header(path, h, &tracks, Some(sel), wide, read_budget)
    }

    fn from_header(path: PathBuf, h: &SmfHeader, tracks: &[usize], sel: Option<&TrackSelection>,
        wide: bool, read_budget: u64) -> Result<Self> {
        // [67]
        let whole = sel.is_none();
        if whole && h.trailing_bytes > 0 {
            log::warn!(
                "{}: {} bytes after the last chunk are not chunks; ignored",
                path.display(),
                h.trailing_bytes
            );
        }
        if h.tracks.is_empty() {
            bail!("{}: no MTrk chunks", path.display());
        }
        if whole && h.tracks.len() != h.declared_tracks as usize {
            log::warn!(
                "{}: header claims {} tracks, found {}",
                path.display(),
                h.declared_tracks,
                h.tracks.len()
            );
        }

        let layout = Arc::new(Layout {
            path: path.clone(),
            chunks: tracks.iter().map(|&t| h.tracks[t]).collect(),
            fold_ports: sel.is_some(),
            key_mask: if wide { 0xFF } else { 0x7F },
        });
        // [68]
        let lens: Vec<u64> = layout.chunks.iter().map(|&(_, len)| len).collect();
        let window = window_share(&lens, read_budget);
        let mut readers = Vec::with_capacity(tracks.len());
        let mut pending = Vec::with_capacity(tracks.len());
        for t in 0..tracks.len() {
            let mut r = layout.reader(t, window)?;
            pending.push(r.next_event());
            readers.push(r);
        }

        Ok(MidiStream {
            merge: Merge::new(readers, pending),
            par: None,
            threads: 0,
            chunk: 0,
            layout,
            bytes_total: tracks.iter().map(|&t| h.tracks[t].1).sum(),
            tempo: sel.map_or_else(|| Arc::from(Vec::new()), |s| s.tempo.clone()),
            tempo_pos: 0,
            end_tick: sel.map(|s| s.end_tick),
            division: h.division,
            format: h.format,
            track_count: h.tracks.len() as u16,
            path,
        })
    }

    /// For a selection, the frame at `rate` its `end_tick` falls on, which is \[69\]
    pub fn end_frame(&self, rate: u32, timing: TempoTiming) -> Option<u64> {
        let end = self.end_tick?;
        let mut clock = TempoClock::with_timing(self.division, rate, timing);
        for &(tick, us) in self.tempo.iter().take_while(|&&(tick, _)| tick <= end) {
            clock.set_tempo(tick, us);
        }
        let f = clock.frame_at(end);
        Some(if f < 0.0 { 0 } else { f as u64 })
    }

    /// Tracks this stream reads: the file's, or a selection's.
    pub fn tracks_read(&self) -> usize {
        self.layout.chunks.len()
    }

    /// Bytes of track data in the file, summed over every MTrk chunk.
    pub fn bytes_total(&self) -> u64 {
        self.bytes_total
    }

    /// Bytes of track data decoded so far. \[70\]
    pub fn bytes_read(&self) -> u64 {
        let unread = match &self.par {
            Some(p) => p.unread(),
            None => self.merge.unread(),
        };
        self.bytes_total.saturating_sub(unread)
    }

    /// Time spent waiting for the worker threads to hand over events: what the \[71\]
    pub fn waited(&self) -> std::time::Duration {
        self.par.as_ref().map_or(std::time::Duration::ZERO, par::Par::waited)
    }

    /// Merge on `threads` worker threads from the first event on, or on the \[72\]
    pub fn set_threads(&mut self, threads: usize) {
        debug_assert!(self.par.is_none(), "the merge has already started");
        self.threads = threads;
    }

    /// `set_threads`, with chunks of `chunk` events rather than `par`'s choice.
    #[cfg(test)]
    pub(crate) fn set_threads_chunk(&mut self, threads: usize, chunk: usize) {
        self.set_threads(threads);
        self.chunk = chunk;
    }

    /// Where every track is, for a snapshot: how far into its chunk, and the \[73\]
    pub fn save_state(&self, e: &mut Enc) -> Result<()> {
        e.len_of(self.layout.chunks.len());
        match &self.par {
            None => {
                for (r, &pending) in self.merge.readers.iter().zip(&self.merge.pending) {
                    Held { pos: r.pos(), pending }.save(e);
                }
            }
            Some(p) => {
                for held in p.replay(&self.layout)? {
                    held.save(e);
                }
            }
        }
        e.u64(self.tempo_pos as u64);
        Ok(())
    }

    /// Go on from a saved position. The stream must have been opened on the \[74\]
    pub fn load_state(&mut self, d: &mut Dec) -> Result<()> {
        if let Some(p) = self.par.take() {
            self.merge = p.stop().unwrap_or_else(|e| std::panic::resume_unwind(e));
        }
        let m = &mut self.merge;
        d.len_is(m.readers.len(), "MIDI tracks", 20)?;
        m.heap.clear();
        for i in 0..m.readers.len() {
            m.readers[i].load(d)?;
            m.pending[i] = if d.bool()? { Some(Event::load(d)?) } else { None };
            if m.pending[i].is_some() {
                m.heap.push(Reverse((m.readers[i].tick, i as u32)));
            }
        }
        let pos = d.u64()? as usize;
        if pos > self.tempo.len() {
            bail!("the saved tempo position is past the end of the tempo map");
        }
        self.tempo_pos = pos;
        Ok(())
    }

    /// Next event in tick order, or None at the end of the file. \[75\]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(u64, Event)> {
        if self.end_tick.is_some() {
            return self.next_selected();
        }
        self.next_merged()
    }

    /// A selection's next event: its tempo map merged with its tracks, whose \[76\]
    fn next_selected(&mut self) -> Option<(u64, Event)> {
        loop {
            if let Some(&(tick, us)) = self.tempo.get(self.tempo_pos) {
                // First on a tie; see `open_tracks` for why that moves nothing.
                if self.peek_tick().is_none_or(|t| tick <= t) {
                    self.tempo_pos += 1;
                    return Some((tick, Event::Tempo(us)));
                }
            }
            match self.next_merged()? {
                (_, Event::Tempo(_)) => continue,
                e => return Some(e),
            }
        }
    }

    #[inline]
    fn next_merged(&mut self) -> Option<(u64, Event)> {
        if self.par.is_none() {
            if self.threads == 0 {
                return self.merge.next();
            }
            self.start();
        }
        self.par.as_mut().expect("started").next()
    }

    /// The tick of the event `next_merged` would return.
    fn peek_tick(&mut self) -> Option<u64> {
        if self.par.is_none() && self.threads > 0 {
            self.start();
        }
        match &self.par {
            Some(p) => p.peek_tick(),
            None => self.merge.peek_tick(),
        }
    }

    /// Hand the merge to `threads` workers.
    #[cold]
    fn start(&mut self) {
        let merge = std::mem::replace(&mut self.merge, Merge::new(Vec::new(), Vec::new()));
        self.par = Some(par::Par::start(merge, self.threads, self.chunk));
    }
}

/// How a tempo change moves [`TempoClock`]: `--tempo-timing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TempoTiming {
    /// BASSMIDI's own MIDI file player, and so Keppy's MIDI Converter: a change \[77\]
    #[default]
    Bassmidi,
    /// The tempo map exactly: every event on the first whole sample at or after \[78\]
    Exact,
}

impl TempoTiming {
    pub fn parse(s: &str) -> Option<TempoTiming> {
        match s {
            "bassmidi" => Some(TempoTiming::Bassmidi),
            "exact" => Some(TempoTiming::Exact),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TempoTiming::Bassmidi => "bassmidi",
            TempoTiming::Exact => "exact",
        }
    }
}

/// Converts ticks to absolute output frames the way BASSMIDI does, tracking \[79\]
#[derive(Debug, Clone)]
pub struct TempoClock {
    division: Division,
    sample_rate: u64,
    /// Which way a tempo change moves the clock. From the configuration.
    timing: TempoTiming,
    /// Under the exact timing, how far past `base_frame` the tick `base_tick` \[80\]
    carry: u64,
    /// The whole frame the current tempo took effect on.
    base_frame: u64,
    /// The tick of the tempo change that set it.
    base_tick: u64,
    /// How far past `base_tick` the tick position had already run by \[81\]
    overshoot: f64,
    /// Ticks to frames is `ticks * num / den` under the current tempo.
    num: u64,
    den: u64,
    /// `overshoot` in frames of the current tempo, `overshoot * num / den`.
    overshoot_frames: f64,
    /// The last tick `frame_at_cached` was asked about and the answer, under \[82\]
    memo: Option<(u64, f64)>,
}

impl TempoClock {
    /// A clock with BASSMIDI's timing, the default.
    pub fn new(division: Division, sample_rate: u32) -> Self {
        Self::with_timing(division, sample_rate, TempoTiming::Bassmidi)
    }

    pub fn with_timing(division: Division, sample_rate: u32, timing: TempoTiming) -> Self {
        let mut c = TempoClock {
            division,
            sample_rate: sample_rate as u64,
            timing,
            carry: 0,
            base_frame: 0,
            base_tick: 0,
            overshoot: 0.0,
            num: 1,
            den: 1,
            overshoot_frames: 0.0,
            memo: None,
        };
        c.set_ratio(500_000); // 120 bpm until told otherwise
        c
    }

    /// Take a new tempo's ratio, and the carried overshoot in its frames. The \[83\]
    fn set_ratio(&mut self, us_per_qn: u64) {
        (self.num, self.den) = match self.division {
            Division::Ppq(ppq) => (us_per_qn * self.sample_rate, 1_000_000 * ppq.max(1) as u64),
            Division::Smpte { fps, ticks_per_frame } => {
                (self.sample_rate, fps.max(1) as u64 * ticks_per_frame.max(1) as u64)
            }
        };
        self.overshoot_frames = self.overshoot * self.num as f64 / self.den as f64;
    }

    /// Whole frames from `base_frame` to the first one whose tick position has \[84\]
    #[inline]
    fn frames_to(&self, tick: u64) -> u64 {
        let d = tick.saturating_sub(self.base_tick);
        let (whole, rem) = match d.checked_mul(self.num).and_then(|x| x.checked_add(self.carry)) {
            Some(exact) => (exact / self.den, exact % self.den),
            // Only a span of billions of ticks at a slow tempo gets here.
            None => {
                let exact = d as u128 * self.num as u128 + self.carry as u128;
                let den = self.den as u128;
                ((exact / den) as u64, (exact % den) as u64)
            }
        };
        // [85]
        let frac = rem as f64 / self.den as f64 - self.overshoot_frames;
        (whole as f64 + frac.ceil()).max(0.0) as u64
    }

    /// For a snapshot. Everything but the division and the rate, which the \[86\]
    pub fn save_state(&self, e: &mut Enc) {
        match self.division {
            Division::Ppq(p) => {
                e.u8(0);
                e.u32(p as u32);
            }
            Division::Smpte { fps, ticks_per_frame } => {
                e.u8(1);
                e.u32(((fps as u32) << 8) | ticks_per_frame as u32);
            }
        }
        e.u64(self.sample_rate);
        e.u64(self.base_frame);
        e.u64(self.base_tick);
        e.f64(self.overshoot);
        e.u64(self.num);
        e.u64(self.den);
        e.f64(self.overshoot_frames);
        if self.timing == TempoTiming::Exact {
            e.u64(self.carry);
        }
    }

    pub fn load_state(&mut self, d: &mut Dec) -> Result<()> {
        let division = match d.u8()? {
            0 => Division::Ppq(d.u32()? as u16),
            1 => {
                let v = d.u32()?;
                Division::Smpte { fps: (v >> 8) as u8, ticks_per_frame: (v & 0xFF) as u8 }
            }
            t => bail!("the saved tempo clock has a division of kind {t}"),
        };
        if division != self.division || d.u64()? != self.sample_rate {
            bail!("the saved tempo clock was made for another time division or sample rate");
        }
        self.base_frame = d.u64()?;
        self.base_tick = d.u64()?;
        self.overshoot = d.f64()?;
        self.num = d.u64()?;
        self.den = d.u64()?;
        self.overshoot_frames = d.f64()?;
        if self.timing == TempoTiming::Exact {
            self.carry = d.u64()?;
            if self.carry >= self.den {
                bail!("the saved tempo clock's part frame is not under a frame");
            }
        }
        self.memo = None;
        Ok(())
    }

    /// Apply a tempo change at `tick`, which must not be earlier than the last.
    pub fn set_tempo(&mut self, tick: u64, us_per_qn: u32) {
        if matches!(self.division, Division::Smpte { .. }) {
            return; // SMPTE division ignores tempo meta events
        }
        // [87]
        self.memo = None;
        match self.timing {
            TempoTiming::Bassmidi => {
                let n = self.frames_to(tick);
                // [88]
                let past = n as i128 * self.den as i128
                    - tick.saturating_sub(self.base_tick) as i128 * self.num as i128;
                self.overshoot = (self.overshoot + past as f64 / self.num as f64).max(0.0);
                self.base_frame += n;
            }
            // [89]
            TempoTiming::Exact => {
                let exact = tick.saturating_sub(self.base_tick) as u128 * self.num as u128 + self.carry as u128;
                let den = self.den as u128;
                self.base_frame += (exact / den) as u64;
                self.carry = (exact % den) as u64;
            }
        }
        self.base_tick = tick;
        self.set_ratio(us_per_qn.max(1) as u64);
    }

    /// The frame an event at `tick` fires on. Always a whole frame; `f64` so \[90\]
    #[inline]
    pub fn frame_at(&self, tick: u64) -> f64 {
        (self.base_frame + self.frames_to(tick)) as f64
    }

    /// `frame_at`, which remembers the tick it last answered for. \[91\]
    #[inline]
    pub fn frame_at_cached(&mut self, tick: u64) -> f64 {
        if let Some((t, f)) = self.memo {
            if t == tick {
                return f;
            }
        }
        let f = self.frame_at(tick);
        self.memo = Some((tick, f));
        f
    }
}

// [92]

/// Minimal SMF writer. Only used by tests and the `gen-test-midi` CLI command, \[93\]
pub struct MidiWriter {
    tracks: Vec<Vec<u8>>,
    ppq: u16,
}

impl MidiWriter {
    pub fn new(ppq: u16) -> Self {
        MidiWriter { tracks: Vec::new(), ppq }
    }

    pub fn track(&mut self, events: Vec<(u64, [u8; 3], usize)>) {
        // events: (absolute tick, message bytes, message length)
        self.raw_track(events.into_iter().map(|(t, m, n)| (t, m[..n].to_vec())).collect());
    }

    /// A track of arbitrary events, each written verbatim after its delta: \[94\]
    pub fn raw_track(&mut self, events: Vec<(u64, Vec<u8>)>) {
        let mut sorted = events;
        sorted.sort_by_key(|e| e.0);
        let mut buf = Vec::new();
        let mut last = 0u64;
        for (tick, msg) in sorted {
            write_varlen(&mut buf, tick - last);
            last = tick;
            buf.extend_from_slice(&msg);
        }
        write_varlen(&mut buf, 0);
        buf.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        self.tracks.push(buf);
    }

    /// The FF 21 meta event that moves a track onto `port`.
    pub fn port_event(port: u8) -> Vec<u8> {
        vec![0xFF, 0x21, 0x01, port]
    }

    pub fn tempo_track(&mut self, us_per_qn: u32) {
        let mut buf = Vec::new();
        write_varlen(&mut buf, 0);
        buf.extend_from_slice(&[0xFF, 0x51, 0x03]);
        buf.extend_from_slice(&us_per_qn.to_be_bytes()[1..4]);
        write_varlen(&mut buf, 0);
        buf.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        self.tracks.push(buf);
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(File::create(path)?);
        f.write_all(b"MThd")?;
        f.write_all(&6u32.to_be_bytes())?;
        f.write_all(&1u16.to_be_bytes())?;
        f.write_all(&(self.tracks.len() as u16).to_be_bytes())?;
        f.write_all(&self.ppq.to_be_bytes())?;
        for t in &self.tracks {
            f.write_all(b"MTrk")?;
            f.write_all(&(t.len() as u32).to_be_bytes())?;
            f.write_all(t)?;
        }
        f.flush()?;
        Ok(())
    }
}

fn write_varlen(buf: &mut Vec<u8>, mut v: u64) {
    let mut stack = [0u8; 10];
    let mut n = 0;
    stack[n] = (v & 0x7F) as u8;
    n += 1;
    v >>= 7;
    while v > 0 {
        stack[n] = ((v & 0x7F) as u8) | 0x80;
        n += 1;
        v >>= 7;
    }
    for i in (0..n).rev() {
        buf.push(stack[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tempo change between two samples fires on the later one and carries \[95\]
    #[test]
    fn a_tempo_change_between_samples_carries_what_it_overshot() {
        let mut c = TempoClock::new(Division::Ppq(1920), 48_000);
        c.set_tempo(1920, 2250);
        c.set_tempo(3600, 2_727_273);
        assert_eq!(c.frame_at(3600), 24_095.0);
        assert_eq!(c.frame_at(3618), 24_717.0);
    }

    /// On a sample boundary nothing carries: the same pattern with an 800-tick \[96\]
    #[test]
    fn a_tempo_change_on_a_sample_boundary_carries_nothing() {
        let mut c = TempoClock::new(Division::Ppq(1920), 48_000);
        c.set_tempo(1920, 2250);
        c.set_tempo(2720, 2_727_273);
        assert_eq!(c.frame_at(2720), 24_045.0);
        // 18 ticks at 22 bpm is 1227.27 frames, so the next whole one.
        assert_eq!(c.frame_at(2738), 25_273.0);
    }

    /// The same burst under `--tempo-timing exact`: nothing carries, so tick \[97\]
    #[test]
    fn the_exact_timing_plays_the_tempo_map_as_written() {
        let mut c = TempoClock::with_timing(Division::Ppq(1920), 48_000, TempoTiming::Exact);
        c.set_tempo(1920, 2250);
        c.set_tempo(3600, 2_727_273);
        // 24,000 plus the fast segment's 94.5 frames.
        assert_eq!(c.frame_at(3600), 24_095.0);
        assert_eq!(c.frame_at(3618), 25_322.0);
        // [98]
        let mut c = TempoClock::with_timing(Division::Ppq(1920), 48_000, TempoTiming::Exact);
        c.set_tempo(1920, 2250);
        c.set_tempo(2720, 2_727_273);
        assert_eq!((c.frame_at(2720), c.frame_at(2738)), (24_045.0, 25_273.0));
    }

    /// A user's file, in small (2026-10-23): a tempo that wobbles every 10 \[99\]
    #[test]
    fn a_wobbling_tempo_that_adds_up_lands_where_a_steady_one_does() {
        // [100]
        let mut steady = TempoClock::with_timing(Division::Ppq(960), 48_000, TempoTiming::Exact);
        steady.set_tempo(0, 3000);
        for timing in [TempoTiming::Exact, TempoTiming::Bassmidi] {
            let mut c = TempoClock::with_timing(Division::Ppq(960), 48_000, timing);
            let mut drift = 0.0f64;
            for pair in 0..100u64 {
                c.set_tempo(pair * 20, 3001);
                c.set_tempo(pair * 20 + 10, 2999);
                let bar = (pair + 1) * 20;
                assert_eq!(steady.frame_at(bar), 3.0 * (pair + 1) as f64);
                drift = drift.max((c.frame_at(bar) - steady.frame_at(bar)).abs());
            }
            match timing {
                TempoTiming::Exact => assert_eq!(drift, 0.0),
                TempoTiming::Bassmidi => assert!(drift >= 1.0, "BASSMIDI's player drifted {drift} frames"),
            }
        }
    }

    /// The exact timing is the exact rational time of each tick, rounded up to a \[101\]
    #[test]
    fn the_exact_timing_is_exact_arithmetic_rounded_up() {
        let mut z = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            z
        };
        for (ppq, rate) in [(96u16, 44_100u32), (480, 48_000), (960, 48_000), (1920, 96_000), (3840, 22_050)] {
            let den = 1_000_000u128 * ppq as u128;
            let mut c = TempoClock::with_timing(Division::Ppq(ppq), rate, TempoTiming::Exact);
            let (mut sum, mut last, mut us) = (0u128, 0u64, 500_000u128);
            let mut tick = 0u64;
            for _ in 0..20_000 {
                tick += next() % 50;
                if next() % 4 == 0 {
                    // [102]
                    let new = 2 + (next() % 3_000_000) as u32;
                    sum += (tick - last) as u128 * us * rate as u128;
                    (last, us) = (tick, new as u128);
                    c.set_tempo(tick, new);
                }
                let exact = sum + (tick - last) as u128 * us * rate as u128;
                let want = exact.div_ceil(den) as f64;
                assert_eq!(c.frame_at(tick), want, "tick {tick} at {ppq} PPQ, {rate} Hz");
            }
        }
    }

    /// A checkpoint of an exact clock carries its part frame: one loaded from it \[103\]
    #[test]
    fn a_checkpoint_carries_the_exact_timings_part_frame() {
        for timing in [TempoTiming::Exact, TempoTiming::Bassmidi] {
            let mut c = TempoClock::with_timing(Division::Ppq(960), 48_000, timing);
            for pair in 0..37u64 {
                c.set_tempo(pair * 20, 3001);
                c.set_tempo(pair * 20 + 10, 2999);
            }
            let mut e = Enc::new();
            c.save_state(&mut e);
            let bytes = e.into_bytes();
            // [104]
            let base = 5 + 7 * 8;
            assert_eq!(bytes.len(), if timing == TempoTiming::Exact { base + 8 } else { base });
            let mut back = TempoClock::with_timing(Division::Ppq(960), 48_000, timing);
            let mut d = Dec::new(&bytes);
            back.load_state(&mut d).unwrap();
            for tick in [740, 741, 745, 750, 999, 5_000] {
                assert_eq!(back.frame_at(tick), c.frame_at(tick), "tick {tick} under {timing:?}");
            }
        }
    }

    /// An event fires on the first whole frame its tick has reached.
    #[test]
    fn an_event_fires_on_the_first_whole_frame() {
        let c = TempoClock::new(Division::Ppq(480), 44_100);
        // One tick at 120 bpm is 45.9375 frames.
        assert_eq!(c.frame_at(0), 0.0);
        assert_eq!(c.frame_at(1), 46.0);
    }

    /// The clock that remembers its last answer gives the bits the one that does \[105\]
    #[test]
    fn the_remembered_frame_is_the_frame_the_clock_gives() {
        let divisions = [Division::Ppq(1920), Division::Ppq(96), Division::Smpte { fps: 25, ticks_per_frame: 40 }];
        for (division, timing) in divisions.into_iter().flat_map(|d| [(d, TempoTiming::Bassmidi), (d, TempoTiming::Exact)]) {
            let mut cached = TempoClock::with_timing(division, 48_000, timing);
            let mut plain = cached.clone();
            let mut z = 0x2545_F491_4F6C_DD1Du64;
            let mut next = move || {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                z
            };
            let mut tick = 0u64;
            for _ in 0..200_000 {
                match next() % 8 {
                    // The same tick again, as nearly every event of a black MIDI is.
                    0..=3 => {}
                    4 | 5 => tick += next() % 3,
                    6 => tick += next() % 5000,
                    _ => {
                        // [106]
                        let held = tick;
                        tick += 1 + next() % 3;
                        let us = 1 + (next() % 3_000_000) as u32;
                        cached.set_tempo(tick, us);
                        plain.set_tempo(tick, us);
                        let (a, b) = (cached.frame_at_cached(held), plain.frame_at(held));
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "tick {held} after a change at {tick} under {division:?}"
                        );
                    }
                }
                let (a, b) = (cached.frame_at_cached(tick), plain.frame_at(tick));
                assert_eq!(a.to_bits(), b.to_bits(), "tick {tick} under {division:?}");
            }
        }
    }

    /// A track chunk of bytes that mostly make events, and a few that do not: \[107\]
    fn random_track(seed: u64, events: usize, tail: &[u8]) -> Vec<u8> {
        let mut z = seed | 1;
        let mut next = move || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            z
        };
        let data = |x: u64| (x % 128) as u8;
        // [108]
        let mut t = vec![0x00, 0x90, 60, 100];
        let mut running = 0x90u8; // zero when a meta event has cleared it
        for _ in 0..events {
            match next() % 16 {
                0..=9 => t.push(0),
                10 | 11 => t.push(data(next())),
                12 => t.extend([0x80 | data(next()), data(next())]),
                13 => t.extend([0x81, 0x80, 0x80, data(next())]),
                // [109]
                _ => t.extend([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, data(next())]),
            }
            let data_bytes = |status: u8| if matches!(status & 0xF0, 0xC0 | 0xD0) { 1 } else { 2 };
            // Every delta is followed by an event, so the stream stays in step.
            let mut kind = next() % 36;
            if running == 0 && matches!(kind, 15..=29 | 34) {
                kind = 0;
            }
            match kind {
                // A channel message with its status byte, and without one.
                15..=29 => {
                    for _ in 0..data_bytes(running) {
                        t.push(data(next()));
                    }
                }
                // Tempo, port, an ignored meta event: each clears running status.
                30 => {
                    t.extend([0xFF, 0x51, 0x03, data(next()) % 8, data(next()), data(next())]);
                    running = 0;
                }
                31 => {
                    t.extend([0xFF, 0x21, 0x01, (next() % 20) as u8]);
                    running = 0;
                }
                32 => {
                    t.extend([0xFF, 0x01, 0x02, b'a', b'b']);
                    running = 0;
                }
                // A SysEx that is nothing, which leaves running status alone.
                33 => t.extend([0xF0, 0x03, 0x7E, 0x7F, 0xF7]),
                // [110]
                34 => {
                    t.push(running);
                    for _ in 0..data_bytes(running) {
                        t.push(data(next()));
                    }
                }
                _ => {
                    running = 0x80 | (next() % 0x70) as u8;
                    t.push(running);
                    for _ in 0..data_bytes(running) {
                        t.push(data(next()));
                    }
                }
            }
            // [111]
            if next() % 30_000 == 0 {
                t.push(next() as u8);
            }
        }
        t.extend_from_slice(tail);
        t.extend([0x00, 0xFF, 0x2F, 0x00]);
        t
    }

    /// The decoder that takes a channel message straight out of the window and \[112\]
    #[test]
    fn the_window_fast_path_decodes_what_the_general_path_does() {
        let dir = std::env::temp_dir().join("kestrel_midi_fast_path");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("random.mid");
        let mut total = 0u64;
        for (seed, key_mask) in [(1u64, 0x7Fu8), (2, 0xFF), (3, 0x7F), (4, 0xFF), (5, 0x7F), (6, 0x7F)] {
            // Seed 2 ends in a delta no reader accepts; the others end as a track does.
            let tail: &[u8] = if seed == 2 { &[0xFF; 8] } else { &[] };
            let body = random_track(seed * 0x9E37_79B9_7F4A_7C15, 250_000, tail);
            let mut file = b"MThd\0\0\0\x06\0\0\0\x01\x01\xE0".to_vec();
            file.extend_from_slice(b"MTrk");
            file.extend_from_slice(&(body.len() as u32).to_be_bytes());
            file.extend_from_slice(&body);
            std::fs::write(&path, &file).unwrap();
            assert!(body.len() > 3 * TRACK_BUF, "the track must cross the window");

            let (off, len) = SmfHeader::read(&path).unwrap().tracks[0];
            let (mut fast, mut slow) = (
                TrackReader::open(&path, off, len, TRACK_BUF).unwrap(),
                TrackReader::open(&path, off, len, TRACK_BUF).unwrap(),
            );
            fast.key_mask = key_mask;
            slow.key_mask = key_mask;
            let mut n = 0u64;
            loop {
                let (a, b) = (fast.next_event(), slow.next_event_general());
                assert_eq!(a, b, "seed {seed}: event {n} (tick {})", slow.tick);
                assert_eq!(fast.tick, slow.tick, "seed {seed}: tick after event {n}");
                assert_eq!(fast.unread(), slow.unread(), "seed {seed}: position after event {n}");
                assert_eq!(fast.port_base, slow.port_base, "seed {seed}: port after event {n}");
                assert_eq!(fast.running, slow.running, "seed {seed}: running status after event {n}");
                assert_eq!(fast.ended, slow.ended, "seed {seed}: end of track after event {n}");
                n += 1;
                if a.is_none() {
                    break;
                }
            }
            total += n;
        }
        assert!(total > 600_000, "only {total} events came out of all of them");
        let _ = std::fs::remove_file(&path);
    }

    /// Stepping over the notes of other keys a run at a time lands where \[113\]
    #[test]
    fn skipping_other_keys_lands_where_decoding_them_does() {
        fn step(r: &mut TrackReader, fast: bool, end: u64, keep: &[bool; 128], seq: &mut u64) -> Option<(u64, Event)> {
            loop {
                if fast {
                    *seq += r.skip_unkept(end, keep);
                }
                let ev = r.next_event()?;
                *seq += 1;
                let unkept = match ev {
                    Event::NoteOn { key, .. } | Event::NoteOff { key, .. } => !keep[key as usize & 127],
                    Event::Other => true,
                    _ => false,
                };
                if !(unkept && r.tick < end) {
                    return Some((r.tick, ev));
                }
            }
        }
        let dir = std::env::temp_dir().join("kestrel_midi_skip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("random.mid");
        let mut kept = 0u64;
        for seed in 1..=4u64 {
            let body = random_track(seed.wrapping_mul(0xC2B2_AE3D_27D4_EB4F), 250_000, &[]);
            let mut file = b"MThd\0\0\0\x06\0\0\0\x01\x01\xE0".to_vec();
            file.extend_from_slice(b"MTrk");
            file.extend_from_slice(&(body.len() as u32).to_be_bytes());
            file.extend_from_slice(&body);
            std::fs::write(&path, &file).unwrap();
            let (off, len) = SmfHeader::read(&path).unwrap().tracks[0];
            let (mut fast, mut slow) = (TrackReader::open(&path, off, len, TRACK_BUF).unwrap(), TrackReader::open(&path, off, len, TRACK_BUF).unwrap());
            let keep: [bool; 128] = std::array::from_fn(|k| (k as u64 + seed).is_multiple_of(5));
            let (mut end, mut sf, mut ss) = (1u64, 0u64, 0u64);
            loop {
                let (a, b) = (step(&mut fast, true, end, &keep, &mut sf), step(&mut slow, false, end, &keep, &mut ss));
                assert_eq!(a, b, "seed {seed}: after {ss} events");
                assert_eq!(sf, ss, "seed {seed}: events stepped over");
                assert_eq!(fast.unread(), slow.unread(), "seed {seed}: position after {ss} events");
                assert_eq!(fast.running, slow.running, "seed {seed}: running status after {ss} events");
                let Some((t, _)) = a else { break };
                kept += 1;
                if t >= end {
                    end = t + 1 + (t ^ seed) % 700;
                }
            }
        }
        assert!(kept > 1000, "only {kept} events stopped the skip");
        let _ = std::fs::remove_file(&path);
    }

    /// A file of `bodies` as format-1 tracks, 480 ticks a quarter.
    fn raw_tracks_file(path: &Path, bodies: &[Vec<u8>]) {
        let mut file = b"MThd\0\0\0\x06\0\x01".to_vec();
        file.extend_from_slice(&(bodies.len() as u16).to_be_bytes());
        file.extend_from_slice(b"\x01\xE0");
        for body in bodies {
            file.extend_from_slice(b"MTrk");
            file.extend_from_slice(&(body.len() as u32).to_be_bytes());
            file.extend_from_slice(body);
        }
        std::fs::write(path, &file).unwrap();
    }

    /// The window is how often the file is read and nothing else: a track read \[114\]
    #[test]
    fn a_window_of_any_size_decodes_the_same_track() {
        let dir = std::env::temp_dir().join("kestrel_midi_window_sizes");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("random.mid");
        let tail: &[u8] = &[];
        raw_tracks_file(&path, &[random_track(0xD1B5_4A32_D192_ED03, 30_000, tail)]);
        let (off, len) = SmfHeader::read(&path).unwrap().tracks[0];
        assert!(len > 4 * TRACK_BUF_MIN as u64, "the track must cross the smaller windows");
        for window in [1, 2, 15, 16, 17, 100, TRACK_BUF_MIN, TRACK_BUF] {
            let mut want = TrackReader::open(&path, off, len, TRACK_BUF).unwrap();
            let mut got = TrackReader::open(&path, off, len, window).unwrap();
            assert_eq!(got.buf.len(), window.min(len as usize));
            let mut n = 0u64;
            loop {
                let (a, b) = (got.next_event(), want.next_event());
                assert_eq!(a, b, "window {window}: event {n}");
                assert_eq!(got.tick, want.tick, "window {window}: tick after event {n}");
                assert_eq!(got.unread(), want.unread(), "window {window}: position after event {n}");
                n += 1;
                if a.is_none() {
                    break;
                }
            }
            assert!(n > 25_000, "window {window}: only {n} events");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A stream's windows add up to its budget at most, a track never gets more \[115\]
    #[test]
    fn track_windows_share_the_budget() {
        let total = |lens: &[u64], w: usize| lens.iter().map(|&l| l.min(w as u64)).sum::<u64>();
        const K: u64 = 1024;
        // Few tracks: the largest window, whatever the budget left over.
        assert_eq!(window_share(&[12, 300 * K, 50 << 20], READ_BUDGET), TRACK_BUF);
        assert_eq!(window_share(&[], READ_BUDGET), TRACK_BUF);
        // Four long tracks, half a MiB between them.
        let long = [K << 10; 4];
        assert_eq!(window_share(&long, 512 * K), 128 * K as usize);
        // Two short tracks take their length, and the long ones share the rest.
        let mixed = [K << 10, 10, K << 10, 30];
        let w = window_share(&mixed, 200 * K);
        assert_eq!(w as u64, (200 * K - 40) / 2);
        assert_eq!(total(&mixed, w), 200 * K);
        // The floor wins: a thousand long tracks in a mebibyte.
        assert_eq!(window_share(&[K << 10; 1000], K << 10), TRACK_BUF_MIN);
        // Never over the budget above the floor, on lengths of every size.
        let mut z = 0x2545_F491_4F6C_DD1Du64;
        let lens: Vec<u64> = (0..5000)
            .map(|_| {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                (z % (1 << 20)) >> (z % 20)
            })
            .collect();
        for budget in [0, 1, 64 * K, 20 << 20, 300 << 20, READ_BUDGET / 2, READ_BUDGET] {
            let w = window_share(&lens, budget);
            assert!((TRACK_BUF_MIN..=TRACK_BUF).contains(&w));
            assert!(w == TRACK_BUF_MIN || total(&lens, w) <= budget, "budget {budget}: window {w}");
            // And it is the largest that fits: a byte more would not.
            assert!(w == TRACK_BUF || total(&lens, w + 1) > budget || (w == TRACK_BUF_MIN && total(&lens, w) > budget));
        }
    }

    /// A saved position is in bytes of the chunk, not in windows, so a stream \[116\]
    #[test]
    fn a_stream_goes_on_from_a_state_saved_with_other_windows() {
        let dir = std::env::temp_dir().join("kestrel_midi_window_state");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("three.mid");
        let tail: &[u8] = &[];
        let bodies: Vec<Vec<u8>> =
            (1..=3u64).map(|s| random_track(s * 0xA076_1D64_78BD_642F, 12_000, tail)).collect();
        assert!(bodies.iter().all(|b| b.len() > 4 * TRACK_BUF_MIN));
        raw_tracks_file(&path, &bodies);
        let read_all = |s: &mut MidiStream| {
            let mut v = Vec::new();
            while let Some(e) = s.next() {
                v.push(e);
            }
            v
        };
        // A budget of nothing puts every track at the floor.
        let events = read_all(&mut MidiStream::open(&path).unwrap());
        assert_eq!(read_all(&mut MidiStream::open_keys(&path, false, 0).unwrap()), events);
        assert!(events.len() > 30_000);
        for (from, to) in [(0, READ_BUDGET), (READ_BUDGET, 0)] {
            for taken in [1, 777, events.len() / 2, events.len() - 1] {
                let mut s = MidiStream::open_keys(&path, false, from).unwrap();
                for _ in 0..taken {
                    s.next().unwrap();
                }
                let mut e = Enc::new();
                s.save_state(&mut e).unwrap();
                let bytes = e.into_bytes();
                let mut r = MidiStream::open_keys(&path, false, to).unwrap();
                let mut d = Dec::new(&bytes);
                r.load_state(&mut d).unwrap();
                d.finish().unwrap();
                assert_eq!(read_all(&mut r), events[taken..], "budget {from} to {to}, after {taken} events");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Trailing padding must not be walked eight bytes at a time. \[117\]
    #[test]
    fn trailing_padding_does_not_stall_the_chunk_walk() {
        let dir = std::env::temp_dir().join("kestrel_midi_padding");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("padded.mid");

        let mut w = MidiWriter::new(960);
        w.tempo_track(500_000);
        for _ in 0..4 {
            w.track(vec![(0, [0x90, 60, 90], 3), (480, [0x80, 60, 0], 3)]);
        }
        w.save(&path).unwrap();

        let clean = std::fs::metadata(&path).unwrap().len();
        // 16 MiB of zeros: 2M chunk headers if the walk steps over them.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&vec![0u8; 16 << 20]).unwrap();
        }

        let t0 = std::time::Instant::now();
        let s = MidiStream::open(&path).unwrap();
        let took = t0.elapsed();

        assert_eq!(s.track_count, 5, "tempo track plus four");
        assert!(
            took < std::time::Duration::from_secs(2),
            "opening a padded file took {took:?}; the walk is stepping through \
             the padding rather than stopping at it"
        );
        assert!(std::fs::metadata(&path).unwrap().len() > clean);
        let _ = std::fs::remove_file(&path);
    }

    /// A file that moves its drums with SysEx must not be read as melodic. \[118\]
    #[test]
    fn gs_rhythm_part_sysex_maps_blocks_to_the_right_channels() {
        // (block, data) -> (channel index, map)
        for (block, data, ch, map) in [
            (0x00u8, 0x01u8, 9u8, 1u8),   // block 0 is channel 10
            (0x01, 0x00, 0, 0),           // blocks 1-9 are channels 1-9
            (0x09, 0x01, 8, 1),
            (0x0A, 0x02, 10, 2),          // blocks A-F are channels 11-16
            (0x0F, 0x01, 15, 1),
        ] {
            let msg = [0x41, 0x10, 0x42, 0x12, 0x40, 0x10 | block, 0x15, data, 0x00, 0xF7];
            assert_eq!(
                sysex_event(&msg),
                Event::DrumPart { ch, map },
                "block {block:#04x}"
            );
        }
    }

    #[test]
    fn the_two_resets_are_recognised_and_nothing_else_is() {
        // GS Reset and GM System On.
        assert_eq!(
            sysex_event(&[0x41, 0x10, 0x42, 0x12, 0x40, 0x00, 0x7F, 0x00, 0x41, 0xF7]),
            Event::ResetParts
        );
        assert_eq!(sysex_event(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]), Event::ResetParts);
        // [119]
        assert_eq!(
            sysex_event(&[0x41, 0x10, 0x42, 0x12, 0x40, 0x11, 0x02, 0x40, 0x00, 0xF7]),
            Event::Other
        );
        assert_eq!(sysex_event(&[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x7F, 0xF7]), Event::Other);
        assert_eq!(sysex_event(&[0x41, 0x10]), Event::Other);
        assert_eq!(sysex_event(&[]), Event::Other);
    }

    /// The head-and-skip read has to leave the cursor exactly at the end of the \[120\]
    #[test]
    fn a_sysex_longer_than_the_peek_buffer_does_not_desync_the_track() {
        let dir = std::env::temp_dir().join("kestrel_midi_sysex");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sysex.mid");

        // [121]
        let mut track: Vec<u8> = Vec::new();
        let mut ev = |delta: u8, bytes: &[u8]| {
            track.push(delta);
            track.extend_from_slice(bytes);
        };
        ev(0, &[0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        let mut bulk = vec![0xF0, 40u8, 0x41, 0x10, 0x42, 0x12, 0x48, 0x00, 0x00];
        bulk.extend(std::iter::repeat_n(0x00u8, 40 - 8));
        bulk.push(0xF7);
        ev(0, &bulk);
        ev(0, &[0x90, 60, 90]);
        ev(96, &[0x80, 60, 0]);
        ev(0, &[0xFF, 0x2F, 0x00]);

        let mut file: Vec<u8> = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6u32.to_be_bytes());
        file.extend_from_slice(&[0, 0, 0, 1, 0x03, 0xC0]);
        file.extend_from_slice(b"MTrk");
        file.extend_from_slice(&(track.len() as u32).to_be_bytes());
        file.extend_from_slice(&track);
        std::fs::write(&path, &file).unwrap();

        let mut s = MidiStream::open(&path).unwrap();
        let mut got = Vec::new();
        while let Some((tick, e)) = s.next() {
            got.push((tick, e));
        }
        assert!(
            got.contains(&(0, Event::ResetParts)),
            "the 5-byte reset was not seen: {got:?}"
        );
        assert!(
            got.contains(&(0, Event::NoteOn { ch: 0, key: 60, vel: 90 })),
            "the note after the oversized SysEx was lost, so the skip left the \
             cursor in the wrong place: {got:?}"
        );
        assert!(got.contains(&(96, Event::NoteOff { ch: 0, key: 60 })));
        let _ = std::fs::remove_file(&path);
    }

    /// The FF 21 port meta event, read the way BASSMIDI was measured \[122\]
    #[test]
    fn a_port_event_moves_its_own_track_from_that_point_on() {
        let dir = std::env::temp_dir().join("kestrel_midi_ports");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ports.mid");
        let cc = |ch: u8, v: u8| vec![0xB0 | ch, 7, v];
        // USE FOR RHYTHM PART, block 1 -- channel 1 of whichever port.
        let drums = vec![0xF0, 0x0A, 0x41, 0x10, 0x42, 0x12, 0x40, 0x11, 0x15, 0x01, 0x19, 0xF7];

        let mut w = MidiWriter::new(480);
        // [123]
        w.raw_track(vec![
            (0, MidiWriter::port_event(1)),
            (0, cc(2, 1)),
            (10, MidiWriter::port_event(0)),
            (10, cc(2, 2)),
            (20, MidiWriter::port_event(17)),
            (20, cc(2, 3)),
            (21, vec![7, 4]),
        ]);
        // No FF 21 at all: port 0, whatever the track before it said.
        w.raw_track(vec![(30, cc(3, 5))]);
        // Malformed, so ignored, then a part message on port 15, the last.
        w.raw_track(vec![
            (0, vec![0xFF, 0x21, 0x02, 0x03, 0x00]),
            (40, cc(4, 6)),
            (50, MidiWriter::port_event(15)),
            (50, drums),
        ]);
        // Port 9 is a port of its own -- BASSMIDI would fold it onto port 1.
        w.raw_track(vec![(0, MidiWriter::port_event(9)), (60, cc(0, 7))]);
        w.save(&path).unwrap();

        let mut s = MidiStream::open(&path).unwrap();
        let mut got = Vec::new();
        while let Some((tick, e)) = s.next() {
            if !matches!(e, Event::Other) {
                got.push((tick, e));
            }
        }
        let want = [
            (0, Event::Cc { ch: 16 + 2, num: 7, val: 1 }),
            (10, Event::Cc { ch: 2, num: 7, val: 2 }),
            (20, Event::Cc { ch: 16 + 2, num: 7, val: 3 }),
            (21, Event::Cc { ch: 16 + 2, num: 7, val: 4 }),
            (30, Event::Cc { ch: 3, num: 7, val: 5 }),
            (40, Event::Cc { ch: 4, num: 7, val: 6 }),
            (50, Event::DrumPart { ch: 15 * 16, map: 1 }),
            (60, Event::Cc { ch: 9 * 16, num: 7, val: 7 }),
        ];
        assert_eq!(got, want);
        let _ = std::fs::remove_file(&path);
    }

    /// A scale/octave tuning message is read as one event for each pitch class \[124\]
    fn tuning_file(dir: &Path) -> PathBuf {
        let path = dir.join("tuning.mid");
        // [125]
        let mut msg = vec![0xF0, 31 + 1, 0x7F, 0x7F, 0x08, 0x09, 0x02, 0x00, 0x01];
        for pc in 0..12i32 {
            let cents = if pc == 0 { 20.0 } else { -(pc as f64) };
            let v = (8192.0 + cents * 8192.0 / 100.0).round() as i32;
            msg.extend([(v >> 7) as u8, (v & 0x7F) as u8]);
        }
        msg.push(0xF7);
        let mut w = MidiWriter::new(480);
        w.raw_track(vec![
            (0, MidiWriter::port_event(2)),
            (5, msg),
            (10, vec![0x90, 60, 100]),
        ]);
        w.raw_track(vec![(5, vec![0xB0, 7, 99])]);
        w.save(&path).unwrap();
        path
    }

    #[test]
    fn a_scale_octave_message_is_read_as_a_run_of_tuning_events() {
        let dir = std::env::temp_dir().join("kestrel_midi_tuning");
        std::fs::create_dir_all(&dir).unwrap();
        let path = tuning_file(&dir);
        let mut s = MidiStream::open(&path).unwrap();
        let mut got = Vec::new();
        while let Some((tick, e)) = s.next() {
            if !matches!(e, Event::Other) {
                got.push((tick, e));
            }
        }
        let units = |cents: f64| ((8192.0 + cents * 8192.0 / 100.0).round() as i32 - 8192) * 25;
        let mut want = Vec::new();
        for c in [32u8, 32 + 15] {
            for pc in 0..12u8 {
                let cents = if pc == 0 { 20.0 } else { -(pc as f64) };
                want.push((5, Event::Tune { ch: c, pc, units: units(cents) }));
            }
        }
        // [126]
        want.push((5, Event::Cc { ch: 0, num: 7, val: 99 }));
        want.push((10, Event::NoteOn { ch: 32, key: 60, vel: 100 }));
        assert_eq!(got, want);
        let _ = std::fs::remove_file(&path);
    }

    /// A stop can fall anywhere in that run -- the next block's first event is \[127\]
    #[test]
    fn a_run_of_tuning_events_goes_on_from_any_place_in_it() {
        let dir = std::env::temp_dir().join("kestrel_midi_tuning_save");
        std::fs::create_dir_all(&dir).unwrap();
        let path = tuning_file(&dir);
        let mut all = MidiStream::open(&path).unwrap();
        let mut events = Vec::new();
        while let Some(e) = all.next() {
            events.push(e);
        }
        assert!(events.len() > 26);
        for taken in 0..=events.len() {
            let mut s = MidiStream::open(&path).unwrap();
            for _ in 0..taken {
                s.next().unwrap();
            }
            let mut e = Enc::new();
            s.save_state(&mut e).unwrap();
            let bytes = e.into_bytes();

            let mut r = MidiStream::open(&path).unwrap();
            let mut d = Dec::new(&bytes);
            r.load_state(&mut d).unwrap();
            d.finish().unwrap();
            let mut rest = Vec::new();
            while let Some(e) = r.next() {
                rest.push(e);
            }
            assert_eq!(rest, events[taken..], "after {taken} events");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A single-note change with more keys than the 32-byte head holds, a bulk dump of \[128\]
    fn key_tuning_file(dir: &Path) -> PathBuf {
        let path = dir.join("key_tuning.mid");
        let triple = |semis: f64| {
            let v = (semis * 16384.0).round() as u32;
            [(v >> 14) as u8, ((v >> 7) & 0x7F) as u8, (v & 0x7F) as u8]
        };
        // `F0`, the length, the body and `F7`.
        let sysex = |body: Vec<u8>| {
            let n = body.len() + 1;
            let mut v = vec![0xF0];
            if n >= 128 {
                v.push(0x80 | (n >> 7) as u8);
            }
            v.push((n & 0x7F) as u8);
            v.extend(body);
            v.push(0xF7);
            v
        };
        // Program 3, thirty keys: 60 + i to 60.5 + i, which is past the head.
        let mut single = vec![0x7F, 0x7F, 0x08, 0x02, 3, 30];
        for i in 0..30u8 {
            single.push(60 + i);
            single.extend(triple(60.5 + i as f64));
        }
        // Bank 1, program 4, one key, the form with a bank.
        let mut banked = vec![0x7F, 0x7F, 0x08, 0x07, 1, 4, 1, 61];
        banked.extend(triple(61.25));
        // A bulk dump of program 9: every key on its own pitch but key 60, and a checksum.
        let mut dump = vec![0x7E, 0x7F, 0x08, 0x01, 9];
        dump.extend(std::iter::repeat_n(b'n', 16));
        for k in 0..128u8 {
            dump.extend(triple(if k == 60 { 59.5 } else { k as f64 }));
        }
        dump.push(0x12);
        let (single, banked, dump) = (sysex(single), sysex(banked), sysex(dump));
        let mut w = MidiWriter::new(480);
        w.raw_track(vec![
            (5, single),
            (6, vec![0x90, 60, 100]),
            (7, banked),
            (8, dump),
            (9, vec![0x80, 60, 0]),
        ]);
        w.save(&path).unwrap();
        path
    }

    #[test]
    fn a_single_note_change_and_a_bulk_dump_are_read_whole_as_runs_of_key_tuning_events() {
        let dir = std::env::temp_dir().join("kestrel_midi_key_tuning");
        std::fs::create_dir_all(&dir).unwrap();
        let path = key_tuning_file(&dir);
        let mut s = MidiStream::open(&path).unwrap();
        let mut got = Vec::new();
        while let Some((tick, e)) = s.next() {
            if !matches!(e, Event::Other) {
                got.push((tick, e));
            }
        }
        // 30 keys of a single-note change on tick 5, each half a semitone up (50 cents).
        let up = |cents: i32| cents * mts::UNITS_PER_CENT;
        for i in 0..30u8 {
            assert_eq!(got[i as usize], (5, Event::KeyTune { bank: 0, prog: 3, key: 60 + i, units: up(50) }), "key {i}");
        }
        assert_eq!(got[30], (6, Event::NoteOn { ch: 0, key: 60, vel: 100 }));
        assert_eq!(got[31], (7, Event::KeyTune { bank: 1, prog: 4, key: 61, units: up(25) }));
        // The dump: all 128 keys of program 9 on tick 8, and key 60 is down half a semitone.
        let dumped: Vec<_> = got[32..160].iter().map(|&(t, e)| (t, e)).collect();
        assert!(dumped.iter().all(|&(t, _)| t == 8));
        assert_eq!(dumped[60].1, Event::KeyTune { bank: 0, prog: 9, key: 60, units: up(-50) });
        assert_eq!(dumped[61].1, Event::KeyTune { bank: 0, prog: 9, key: 61, units: 0 });
        assert_eq!(got[160], (9, Event::NoteOff { ch: 0, key: 60 }));
        assert_eq!(got.len(), 161);
        let _ = std::fs::remove_file(&path);
    }

    /// And a stop can fall anywhere in those runs, as it can in a scale/octave message's.
    #[test]
    fn a_run_of_key_tuning_events_goes_on_from_any_place_in_it() {
        let dir = std::env::temp_dir().join("kestrel_midi_key_tuning_save");
        std::fs::create_dir_all(&dir).unwrap();
        let path = key_tuning_file(&dir);
        let mut all = MidiStream::open(&path).unwrap();
        let mut events = Vec::new();
        while let Some(e) = all.next() {
            events.push(e);
        }
        assert!(events.len() > 160);
        for taken in 0..=events.len() {
            let mut s = MidiStream::open(&path).unwrap();
            for _ in 0..taken {
                s.next().unwrap();
            }
            let mut e = Enc::new();
            s.save_state(&mut e).unwrap();
            let bytes = e.into_bytes();
            let mut r = MidiStream::open(&path).unwrap();
            let mut d = Dec::new(&bytes);
            r.load_state(&mut d).unwrap();
            d.finish().unwrap();
            let mut rest = Vec::new();
            while let Some(e) = r.next() {
                rest.push(e);
            }
            assert_eq!(rest, events[taken..], "after {taken} events");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Without being asked a note key is its low seven bits, as BASSMIDI reads \[129\]
    #[test]
    fn a_key_over_127_is_masked_unless_the_stream_is_asked_for_it_whole() {
        let dir = std::env::temp_dir().join("kestrel_midi_wide");
        std::fs::create_dir_all(&dir).unwrap();
        let wide = dir.join("wide.mid");
        let plain = dir.join("plain.mid");
        let mut w = MidiWriter::new(480);
        w.raw_track(vec![
            (0, vec![0x90, 200, 100]),
            (0, vec![0x93, 255, 100]),
            (10, vec![0x80, 200, 0]),
            (10, vec![0x93, 255, 0]),
        ]);
        w.save(&wide).unwrap();
        let mut w = MidiWriter::new(480);
        w.raw_track(vec![(0, vec![0x90, 100, 100]), (10, vec![0x80, 100, 0])]);
        w.save(&plain).unwrap();

        let events = |wide_keys: bool| {
            let mut s = MidiStream::open_keys(&wide, wide_keys, READ_BUDGET).unwrap();
            let mut v = Vec::new();
            while let Some((_, e)) = s.next() {
                if !matches!(e, Event::Other) {
                    v.push(e);
                }
            }
            v
        };
        assert_eq!(
            events(false),
            [
                Event::NoteOn { ch: 0, key: 200 & 0x7F, vel: 100 },
                Event::NoteOn { ch: 3, key: 0x7F, vel: 100 },
                Event::NoteOff { ch: 0, key: 200 & 0x7F },
                Event::NoteOff { ch: 3, key: 0x7F },
            ]
        );
        assert_eq!(
            events(true),
            [
                Event::NoteOn { ch: 0, key: 200, vel: 100 },
                Event::NoteOn { ch: 3, key: 255, vel: 100 },
                Event::NoteOff { ch: 0, key: 200 },
                Event::NoteOff { ch: 3, key: 255 },
            ]
        );
        assert!(uses_extended_keys(&wide).unwrap());
        assert!(!uses_extended_keys(&plain).unwrap());
        let _ = std::fs::remove_file(&wide);
        let _ = std::fs::remove_file(&plain);
    }

    /// A file cut short mid-track still opens, and the header reader says so. \[130\]
    #[test]
    fn a_truncated_track_is_counted_and_the_stream_still_opens() {
        let dir = std::env::temp_dir().join("kestrel_midi_truncated");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("truncated.mid");

        let mut w = MidiWriter::new(960);
        w.tempo_track(500_000);
        let notes: Vec<(u64, [u8; 3], usize)> = (0..64u64)
            .flat_map(|i| [(i * 120, [0x90, 60, 90], 3), (i * 120 + 60, [0x80, 60, 0], 3)])
            .collect();
        w.track(notes);
        w.save(&path).unwrap();

        let whole = SmfHeader::read(&path).unwrap();
        assert_eq!((whole.tracks.len(), whole.truncated_tracks), (2, 0));

        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 40]).unwrap();
        let cut = SmfHeader::read(&path).unwrap();
        assert_eq!(cut.declared_tracks, 2);
        assert_eq!(cut.tracks.len(), 2);
        assert_eq!(cut.truncated_tracks, 1, "the last track runs past the end");
        assert!(MidiStream::open(&path).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    /// Progress climbs monotonically and lands on exactly the total when the \[131\]
    #[test]
    fn bytes_read_reaches_the_total_exactly_when_the_stream_ends() {
        let dir = std::env::temp_dir().join("kestrel_midi_progress");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("progress.mid");

        let mut w = MidiWriter::new(960);
        w.tempo_track(500_000);
        for t in 0..3u64 {
            let key = 40 + t as u8;
            let notes: Vec<(u64, [u8; 3], usize)> = (0..2000u64)
                .flat_map(|i| {
                    let at = i * 48 + t;
                    [(at, [0x90, key, 90], 3), (at + 24, [0x80, key, 0], 3)]
                })
                .collect();
            w.track(notes);
        }
        w.save(&path).unwrap();

        let mut s = MidiStream::open(&path).unwrap();
        let total = s.bytes_total();
        let file_len = std::fs::metadata(&path).unwrap().len();
        assert!(total > 0 && total < file_len, "track data is the file less its chunk headers");

        let mut last = s.bytes_read();
        let mut events = 0u64;
        while s.next().is_some() {
            events += 1;
            if events.is_multiple_of(500) {
                let now = s.bytes_read();
                assert!(now >= last && now <= total, "{last} -> {now} of {total}");
                last = now;
            }
        }
        assert_eq!(s.bytes_read(), total);
        let _ = std::fs::remove_file(&path);
    }

    /// Tracks enough to share several ways and of every size, all starting at \[132\]
    fn many_tracks_file(dir: &Path) -> PathBuf {
        let path = dir.join("many.mid");
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        for i in 0..23u64 {
            let events = [5usize, 400, 3000, 20_000][i as usize % 4];
            chunks.push(random_track(i * 0x9E37_79B9_7F4A_7C15 + 11, events, &[]));
            if i == 7 {
                let mut t = Vec::new();
                for k in 0..300usize {
                    t.push((k % 3 == 0) as u8);
                    t.extend([0xF0, 20, 0x7F, 0x7F, 0x08, 0x08, 0x03, 0x7F, 0x7F]);
                    t.extend((0..12).map(|pc| 54 + ((k + pc) % 20) as u8));
                    t.push(0xF7);
                    t.extend([0x00, 0x90, 60, 100]);
                }
                t.extend([0x00, 0xFF, 0x2F, 0x00]);
                chunks.push(t);
                chunks.push(random_track(99, 3000, &[0xFF; 8]));
                chunks.push(vec![0x00, 0xFF, 0x2F, 0x00]);
            }
        }
        let mut file = b"MThd\0\0\0\x06\0\x01".to_vec();
        file.extend((chunks.len() as u16).to_be_bytes());
        file.extend([0x01, 0xE0]);
        for c in chunks {
            file.extend_from_slice(b"MTrk");
            file.extend_from_slice(&(c.len() as u32).to_be_bytes());
            file.extend_from_slice(&c);
        }
        std::fs::write(&path, &file).unwrap();
        path
    }

    fn all(s: &mut MidiStream) -> Vec<(u64, Event)> {
        let mut v = Vec::new();
        while let Some(e) = s.next() {
            v.push(e);
        }
        v
    }

    fn first_difference(a: &[(u64, Event)], b: &[(u64, Event)]) -> String {
        match a.iter().zip(b).position(|(x, y)| x != y) {
            Some(i) => format!("event {i}: {:?} against {:?}", a[i], b[i]),
            None => format!("{} events against {}", a.len(), b.len()),
        }
    }

    /// The merge on worker threads hands out every event the one-thread merge \[133\]
    #[test]
    fn the_merge_on_worker_threads_hands_out_the_one_thread_order() {
        let dir = std::env::temp_dir().join("kestrel_midi_par_order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = many_tracks_file(&dir);
        const B: u64 = READ_BUDGET;
        for wide in [false, true] {
            let want = all(&mut MidiStream::open_keys(&path, wide, B).unwrap());
            assert!(want.len() > 100_000, "only {} events", want.len());
            assert!(want.iter().any(|e| matches!(e.1, Event::Tune { .. })));
            // A budget of 0 puts every window at the floor.
            for (threads, chunk, budget) in
                [(1, 0, B), (2, 0, 0), (3, 0, B), (5, 0, 0), (64, 0, B), (1, 1, 0), (3, 7, B), (4, 1000, 0)]
            {
                let mut s = MidiStream::open_keys(&path, wide, budget).unwrap();
                s.set_threads_chunk(threads, chunk);
                let mut got = Vec::new();
                let mut last = s.bytes_read();
                while let Some(e) = s.next() {
                    got.push(e);
                    if got.len() % 997 == 0 {
                        let now = s.bytes_read();
                        assert!(now >= last && now <= s.bytes_total(), "{last} -> {now}");
                        last = now;
                    }
                }
                assert!(
                    got == want,
                    "{threads} threads, chunks of {chunk}, budget {budget}: {}",
                    first_difference(&got, &want)
                );
                assert_eq!(s.bytes_read(), s.bytes_total());
                assert_eq!(s.next(), None);
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A stream on worker threads saves, at any event, the bytes the stream on \[134\]
    #[test]
    fn a_stream_on_worker_threads_saves_and_loads_what_one_thread_does() {
        let dir = std::env::temp_dir().join("kestrel_midi_par_state");
        std::fs::create_dir_all(&dir).unwrap();
        let path = many_tracks_file(&dir);
        let total = all(&mut MidiStream::open(&path).unwrap()).len();
        let mut stops: Vec<usize> = (0..total).step_by(9_973).collect();
        for k in 1..(total / 16_384) {
            stops.extend([k * 16_384 - 1, k * 16_384, k * 16_384 + 1]);
        }
        stops.extend([total - 1, total]);
        stops.sort_unstable();
        stops.dedup();
        // A budget of 0, windows at the floor, saves the same bytes as well.
        for (threads, chunk, budget) in [(1, 0, READ_BUDGET), (3, 0, 0), (2, 101, READ_BUDGET), (5, 1, 0)] {
            let mut one = MidiStream::open(&path).unwrap();
            let mut many = MidiStream::open_keys(&path, false, budget).unwrap();
            many.set_threads_chunk(threads, chunk);
            let mut at = 0;
            for &stop in &stops {
                while at < stop {
                    assert_eq!(many.next(), one.next(), "event {at}");
                    at += 1;
                }
                let (mut a, mut b) = (Enc::new(), Enc::new());
                one.save_state(&mut a).unwrap();
                many.save_state(&mut b).unwrap();
                assert!(a.as_bytes() == b.as_bytes(), "{threads} threads, chunks of {chunk}: the state at event {at}");
            }
        }
        for stop in [0, 1, 5_000, 16_384, 70_001, total] {
            let mut one = MidiStream::open(&path).unwrap();
            for _ in 0..stop {
                one.next();
            }
            let mut e = Enc::new();
            one.save_state(&mut e).unwrap();
            let rest = all(&mut one);
            for ahead in [0, 3, 40_000] {
                let mut s = MidiStream::open(&path).unwrap();
                s.set_threads(3);
                for _ in 0..ahead {
                    s.next();
                }
                let mut d = Dec::new(e.as_bytes());
                s.load_state(&mut d).unwrap();
                d.finish().unwrap();
                let got = all(&mut s);
                assert!(got == rest, "loaded at {stop} after {ahead}: {}", first_difference(&got, &rest));
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A selection -- some tracks, with the whole file's tempo map played \[135\]
    #[test]
    fn a_selection_on_worker_threads_is_the_selection_on_one() {
        let dir = std::env::temp_dir().join("kestrel_midi_par_selection");
        std::fs::create_dir_all(&dir).unwrap();
        let path = many_tracks_file(&dir);
        let header = Arc::new(SmfHeader::read(&path).unwrap());
        let sel = TrackSelection {
            tracks: vec![0, 3, 8, 9, 11, 12, 19],
            tempo: (0..400u64).map(|i| (i * 37, 400_000 + i as u32)).collect::<Vec<_>>().into(),
            header,
            end_tick: 1 << 40,
        };
        let want = all(&mut MidiStream::open_tracks(&path, &sel).unwrap());
        assert!(want.iter().filter(|e| matches!(e.1, Event::Tempo(_))).count() == 400);
        for threads in [1, 2, 4] {
            let mut s = MidiStream::open_tracks(&path, &sel).unwrap();
            s.set_threads_chunk(threads, 333);
            let mut one = MidiStream::open_tracks(&path, &sel).unwrap();
            let mut got = Vec::new();
            while let Some(e) = s.next() {
                assert_eq!(Some(e), one.next());
                got.push(e);
                if got.len() % 4_999 == 0 {
                    let (mut a, mut b) = (Enc::new(), Enc::new());
                    one.save_state(&mut a).unwrap();
                    s.save_state(&mut b).unwrap();
                    assert!(a.as_bytes() == b.as_bytes(), "{threads} threads: the state at event {}", got.len());
                }
            }
            assert!(got == want, "{threads} threads: {}", first_difference(&got, &want));
        }
        let _ = std::fs::remove_file(&path);
    }
}
