// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Streaming SMF parser and track merger. \[1\]

use anyhow::{bail, Context, Result};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const TRACK_BUF: usize = 256 * 1024;

/// Bytes of an FF 03 track name a reader keeps; the rest is skipped. Names in \[2\]
pub(crate) const NAME_MAX: u64 = 255;

/// MIDI ports a file can address. A port is one more set of sixteen channels, \[3\]
pub const PORTS: u8 = 16;
/// Channels across every port, and the range of an event's `ch`: all 256 \[4\]
pub const CHANNELS: usize = 16 * PORTS as usize;

/// One decoded event. \[5\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    NoteOn { ch: u8, key: u8, vel: u8 },
    NoteOff { ch: u8, key: u8 },
    Cc { ch: u8, num: u8, val: u8 },
    Program { ch: u8, val: u8 },
    PitchBend { ch: u8, val: i16 },
    /// Roland GS "USE FOR RHYTHM PART". `map` is 0 for a melodic part and 1 or \[6\]
    DrumPart { ch: u8, map: u8 },
    /// GM System On, GM System Off or GS Reset. Puts every channel back to the \[7\]
    ResetParts,
    /// Microseconds per quarter note.
    Tempo(u32),
    /// Anything the synth does not act on. Kept in the stream so callers can \[8\]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Division {
    /// Ticks per quarter note.
    Ppq(u16),
    /// SMPTE: frames per second and ticks per frame.
    Smpte { fps: u8, ticks_per_frame: u8 },
}

/// A single track chunk with its own file cursor and read window.
pub(crate) struct TrackReader {
    file: File,
    buf: Box<[u8]>,
    pos: usize,
    filled: usize,
    /// Bytes of the track chunk not yet pulled into `buf`.
    remaining: u64,
    /// The tick of the event `next_event` last returned.
    pub(crate) tick: u64,
    running: u8,
    ended: bool,
    /// This track's port, already folded to `PORTS` and shifted into place: \[9\]
    port_base: u8,
    /// Keep the track's first FF 03 name in `name`. Only the track scan asks; \[10\]
    pub(crate) keep_name: bool,
    pub(crate) name: Option<Vec<u8>>,
    /// Ignore FF 21, so every event stays on port A. A per-track render sets \[11\]
    fold_ports: bool,
}

impl TrackReader {
    pub(crate) fn open(path: &Path, offset: u64, len: u64) -> Result<Self> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        Ok(TrackReader {
            file,
            buf: vec![0u8; TRACK_BUF].into_boxed_slice(),
            pos: 0,
            filled: 0,
            remaining: len,
            tick: 0,
            running: 0,
            ended: false,
            port_base: 0,
            keep_name: false,
            name: None,
            fold_ports: false,
        })
    }

    /// Bytes of the chunk not yet decoded: still on disk, or read into the \[12\]
    #[inline]
    pub(crate) fn unread(&self) -> u64 {
        self.remaining + (self.filled - self.pos) as u64
    }

    #[inline]
    fn fill(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        let want = self.remaining.min(TRACK_BUF as u64) as usize;
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

    /// Decode one event, advancing `tick`. Returns None at end of track.
    pub(crate) fn next_event(&mut self) -> Option<Event> {
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

        // [13]
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
                // [14]
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
                Some(Event::NoteOff { ch, key: first_data & 0x7F })
            }
            0x90 => {
                let vel = self.byte()?;
                if vel == 0 {
                    Some(Event::NoteOff { ch, key: first_data & 0x7F })
                } else {
                    Some(Event::NoteOn {
                        ch,
                        key: first_data & 0x7F,
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
                    // [15]
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
                    // [16]
                    let mut head = [0u8; 10];
                    let n = len.min(head.len() as u64) as usize;
                    for h in head.iter_mut().take(n) {
                        *h = self.byte()?;
                    }
                    self.skip(len - n as u64)?;
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

/// Recognise the SysEx messages that change how a channel resolves. \[17\]
fn sysex_event(p: &[u8]) -> Event {
    // Roland GS DT1: 41 <dev> 42 12 <addr hi mid lo> <data..> <sum> F7.
    if p.len() >= 8 && p[0] == 0x41 && p[2] == 0x42 && p[3] == 0x12 {
        // [18]
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

/// What a standard MIDI file says about itself before any event is decoded: \[19\]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmfHeader {
    pub format: u16,
    pub division: Division,
    /// Tracks the MThd chunk claims. A truncated or hand-edited file can \[20\]
    pub declared_tracks: u16,
    /// `(data offset, length)` of every MTrk chunk, each length clamped to \[21\]
    pub tracks: Vec<(u64, u64)>,
    /// MTrk chunks whose declared length ran past the end of the file, which \[22\]
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

        // [23]
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
                // [24]
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

/// What a per-track render reads: some of a file's tracks, timed by the whole \[25\]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackSelection {
    /// Tracks to read, 0-based: the one being rendered, and the setup tracks \[26\]
    pub tracks: Vec<usize>,
    /// Every FF 51 in the file, in the merged stream's order. Played in place \[27\]
    pub tempo: Arc<[(u64, u32)]>,
    /// The file's chunk table, as the scan read it. Shared for the same \[28\]
    pub header: Arc<SmfHeader>,
    /// The tick of the whole file's last event other than a tempo change. The \[29\]
    pub end_tick: u64,
}

/// Tick-ordered merge of every track in a standard MIDI file, or of the tracks \[30\]
pub struct MidiStream {
    readers: Vec<TrackReader>,
    /// (tick, track index) so ties break on track order, deterministically.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
    pending: Vec<Option<Event>>,
    /// Sum of every track chunk's length, for `bytes_read`.
    bytes_total: u64,
    /// A selection's tempo map, played beside the tracks, and how far it has \[31\]
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
        let path = path.as_ref().to_path_buf();
        let h = SmfHeader::read(&path)?;
        let all: Vec<usize> = (0..h.tracks.len()).collect();
        Self::from_header(path, &h, &all, None)
    }

    /// Read only the tracks `sel` names, with every port folded onto port A \[32\]
    pub fn open_tracks(path: impl AsRef<Path>, sel: &TrackSelection) -> Result<Self> {
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
        Self::from_header(path, h, &tracks, Some(sel))
    }

    fn from_header(path: PathBuf, h: &SmfHeader, tracks: &[usize], sel: Option<&TrackSelection>) -> Result<Self> {
        // [33]
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

        let mut readers = Vec::with_capacity(tracks.len());
        for &t in tracks {
            let (off, len) = h.tracks[t];
            let mut r = TrackReader::open(&path, off, len)?;
            r.fold_ports = sel.is_some();
            readers.push(r);
        }

        let mut pending = vec![None; readers.len()];
        let mut heap = BinaryHeap::with_capacity(readers.len());
        for (i, r) in readers.iter_mut().enumerate() {
            if let Some(ev) = r.next_event() {
                pending[i] = Some(ev);
                heap.push(Reverse((r.tick, i as u32)));
            }
        }

        Ok(MidiStream {
            readers,
            heap,
            pending,
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

    /// For a selection, the frame at `rate` its `end_tick` falls on, which is \[34\]
    pub fn end_frame(&self, rate: u32) -> Option<u64> {
        let end = self.end_tick?;
        let mut clock = TempoClock::new(self.division, rate);
        for &(tick, us) in self.tempo.iter().take_while(|&&(tick, _)| tick <= end) {
            clock.set_tempo(tick, us);
        }
        let f = clock.frame_at(end);
        Some(if f < 0.0 { 0 } else { f as u64 })
    }

    /// Bytes of track data in the file, summed over every MTrk chunk.
    pub fn bytes_total(&self) -> u64 {
        self.bytes_total
    }

    /// Bytes of track data decoded so far. \[35\]
    pub fn bytes_read(&self) -> u64 {
        let unread: u64 = self
            .readers
            .iter()
            .filter(|r| !r.ended)
            .map(TrackReader::unread)
            .sum();
        self.bytes_total.saturating_sub(unread)
    }

    /// Next event in tick order, or None at the end of the file. \[36\]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(u64, Event)> {
        if self.end_tick.is_some() {
            return self.next_selected();
        }
        self.next_merged()
    }

    /// A selection's next event: its tempo map merged with its tracks, whose \[37\]
    fn next_selected(&mut self) -> Option<(u64, Event)> {
        loop {
            if let Some(&(tick, us)) = self.tempo.get(self.tempo_pos) {
                // First on a tie; see `open_tracks` for why that moves nothing.
                if self.heap.peek().is_none_or(|&Reverse((t, _))| tick <= t) {
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
        // [38]
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
            // [39]
            None => {
                std::collections::binary_heap::PeekMut::pop(top);
            }
        }
        Some((tick, ev))
    }
}

/// Converts ticks to absolute output frames the way BASSMIDI does, tracking \[40\]
#[derive(Debug, Clone)]
pub struct TempoClock {
    division: Division,
    sample_rate: u64,
    /// The whole frame the current tempo took effect on.
    base_frame: u64,
    /// The tick of the tempo change that set it.
    base_tick: u64,
    /// How far past `base_tick` the tick position had already run by \[41\]
    overshoot: f64,
    /// Ticks to frames is `ticks * num / den` under the current tempo.
    num: u64,
    den: u64,
    /// `overshoot` in frames of the current tempo, `overshoot * num / den`.
    overshoot_frames: f64,
}

impl TempoClock {
    pub fn new(division: Division, sample_rate: u32) -> Self {
        let mut c = TempoClock {
            division,
            sample_rate: sample_rate as u64,
            base_frame: 0,
            base_tick: 0,
            overshoot: 0.0,
            num: 1,
            den: 1,
            overshoot_frames: 0.0,
        };
        c.set_ratio(500_000); // 120 bpm until told otherwise
        c
    }

    /// Take a new tempo's ratio, and the carried overshoot in its frames. The \[42\]
    fn set_ratio(&mut self, us_per_qn: u64) {
        (self.num, self.den) = match self.division {
            Division::Ppq(ppq) => (us_per_qn * self.sample_rate, 1_000_000 * ppq.max(1) as u64),
            Division::Smpte { fps, ticks_per_frame } => {
                (self.sample_rate, fps.max(1) as u64 * ticks_per_frame.max(1) as u64)
            }
        };
        self.overshoot_frames = self.overshoot * self.num as f64 / self.den as f64;
    }

    /// Whole frames from `base_frame` to the first one whose tick position has \[43\]
    #[inline]
    fn frames_to(&self, tick: u64) -> u64 {
        let d = tick.saturating_sub(self.base_tick);
        let (whole, rem) = match d.checked_mul(self.num) {
            Some(exact) => (exact / self.den, exact % self.den),
            // Only a span of billions of ticks at a slow tempo gets here.
            None => {
                let exact = d as u128 * self.num as u128;
                let den = self.den as u128;
                ((exact / den) as u64, (exact % den) as u64)
            }
        };
        // [44]
        let frac = rem as f64 / self.den as f64 - self.overshoot_frames;
        (whole as f64 + frac.ceil()).max(0.0) as u64
    }

    /// Apply a tempo change at `tick`, which must not be earlier than the last.
    pub fn set_tempo(&mut self, tick: u64, us_per_qn: u32) {
        if matches!(self.division, Division::Smpte { .. }) {
            return; // SMPTE division ignores tempo meta events
        }
        let n = self.frames_to(tick);
        // [45]
        let past = n as i128 * self.den as i128
            - tick.saturating_sub(self.base_tick) as i128 * self.num as i128;
        self.overshoot = (self.overshoot + past as f64 / self.num as f64).max(0.0);
        self.base_frame += n;
        self.base_tick = tick;
        self.set_ratio(us_per_qn.max(1) as u64);
    }

    /// The frame an event at `tick` fires on. Always a whole frame; `f64` so \[46\]
    #[inline]
    pub fn frame_at(&self, tick: u64) -> f64 {
        (self.base_frame + self.frames_to(tick)) as f64
    }
}

// [47]

/// Minimal SMF writer. Only used by tests and the `gen-test-midi` CLI command, \[48\]
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

    /// A track of arbitrary events, each written verbatim after its delta: \[49\]
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

    /// A tempo change between two samples fires on the later one and carries \[50\]
    #[test]
    fn a_tempo_change_between_samples_carries_what_it_overshot() {
        let mut c = TempoClock::new(Division::Ppq(1920), 48_000);
        c.set_tempo(1920, 2250);
        c.set_tempo(3600, 2_727_273);
        assert_eq!(c.frame_at(3600), 24_095.0);
        assert_eq!(c.frame_at(3618), 24_717.0);
    }

    /// On a sample boundary nothing carries: the same pattern with an 800-tick \[51\]
    #[test]
    fn a_tempo_change_on_a_sample_boundary_carries_nothing() {
        let mut c = TempoClock::new(Division::Ppq(1920), 48_000);
        c.set_tempo(1920, 2250);
        c.set_tempo(2720, 2_727_273);
        assert_eq!(c.frame_at(2720), 24_045.0);
        // 18 ticks at 22 bpm is 1227.27 frames, so the next whole one.
        assert_eq!(c.frame_at(2738), 25_273.0);
    }

    /// An event fires on the first whole frame its tick has reached.
    #[test]
    fn an_event_fires_on_the_first_whole_frame() {
        let c = TempoClock::new(Division::Ppq(480), 44_100);
        // One tick at 120 bpm is 45.9375 frames.
        assert_eq!(c.frame_at(0), 0.0);
        assert_eq!(c.frame_at(1), 46.0);
    }

    /// Trailing padding must not be walked eight bytes at a time. \[52\]
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

    /// A file that moves its drums with SysEx must not be read as melodic. \[53\]
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
        // [54]
        assert_eq!(
            sysex_event(&[0x41, 0x10, 0x42, 0x12, 0x40, 0x11, 0x02, 0x40, 0x00, 0xF7]),
            Event::Other
        );
        assert_eq!(sysex_event(&[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x7F, 0xF7]), Event::Other);
        assert_eq!(sysex_event(&[0x41, 0x10]), Event::Other);
        assert_eq!(sysex_event(&[]), Event::Other);
    }

    /// The head-and-skip read has to leave the cursor exactly at the end of the \[55\]
    #[test]
    fn a_sysex_longer_than_the_peek_buffer_does_not_desync_the_track() {
        let dir = std::env::temp_dir().join("kestrel_midi_sysex");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sysex.mid");

        // [56]
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

    /// The FF 21 port meta event, read the way BASSMIDI was measured \[57\]
    #[test]
    fn a_port_event_moves_its_own_track_from_that_point_on() {
        let dir = std::env::temp_dir().join("kestrel_midi_ports");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ports.mid");
        let cc = |ch: u8, v: u8| vec![0xB0 | ch, 7, v];
        // USE FOR RHYTHM PART, block 1 -- channel 1 of whichever port.
        let drums = vec![0xF0, 0x0A, 0x41, 0x10, 0x42, 0x12, 0x40, 0x11, 0x15, 0x01, 0x19, 0xF7];

        let mut w = MidiWriter::new(480);
        // [58]
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

    /// A file cut short mid-track still opens, and the header reader says so. \[59\]
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

    /// Progress climbs monotonically and lands on exactly the total when the \[60\]
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
}
