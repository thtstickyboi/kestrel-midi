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

const TRACK_BUF: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    NoteOn { ch: u8, key: u8, vel: u8 },
    NoteOff { ch: u8, key: u8 },
    Cc { ch: u8, num: u8, val: u8 },
    Program { ch: u8, val: u8 },
    PitchBend { ch: u8, val: i16 },
    /// Roland GS "USE FOR RHYTHM PART". `map` is 0 for a melodic part and 1 or \[2\]
    DrumPart { ch: u8, map: u8 },
    /// GM System On, GM System Off or GS Reset. Puts every channel back to the \[3\]
    ResetParts,
    /// Microseconds per quarter note.
    Tempo(u32),
    /// Anything the synth does not act on. Kept in the stream so callers can \[4\]
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
struct TrackReader {
    file: File,
    buf: Box<[u8]>,
    pos: usize,
    filled: usize,
    /// Bytes of the track chunk not yet pulled into `buf`.
    remaining: u64,
    tick: u64,
    running: u8,
    ended: bool,
}

impl TrackReader {
    fn open(path: &Path, offset: u64, len: u64) -> Result<Self> {
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
        })
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
    fn next_event(&mut self) -> Option<Event> {
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

        // [5]
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
                // [6]
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

        let ch = status & 0x0F;
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
                    self.skip(len)?;
                    Some(Event::Other)
                }
                0xF0 | 0xF7 => {
                    let len = self.varlen()?;
                    // [7]
                    let mut head = [0u8; 10];
                    let n = len.min(head.len() as u64) as usize;
                    for h in head.iter_mut().take(n) {
                        *h = self.byte()?;
                    }
                    self.skip(len - n as u64)?;
                    Some(sysex_event(&head[..n]))
                }
                _ => Some(Event::Other),
            },
        }
    }
}

/// Recognise the SysEx messages that change how a channel resolves. \[8\]
fn sysex_event(p: &[u8]) -> Event {
    // Roland GS DT1: 41 <dev> 42 12 <addr hi mid lo> <data..> <sum> F7.
    if p.len() >= 8 && p[0] == 0x41 && p[2] == 0x42 && p[3] == 0x12 {
        // [9]
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

/// What a standard MIDI file says about itself before any event is decoded: \[10\]
#[derive(Debug, Clone)]
pub struct SmfHeader {
    pub format: u16,
    pub division: Division,
    /// Tracks the MThd chunk claims. A truncated or hand-edited file can \[11\]
    pub declared_tracks: u16,
    /// `(data offset, length)` of every MTrk chunk, each length clamped to \[12\]
    pub tracks: Vec<(u64, u64)>,
    /// MTrk chunks whose declared length ran past the end of the file, which \[13\]
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

        // [14]
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
                // [15]
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

/// Tick-ordered merge of every track in a standard MIDI file.
pub struct MidiStream {
    readers: Vec<TrackReader>,
    /// (tick, track index) so ties break on track order, deterministically.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
    pending: Vec<Option<Event>>,
    /// Sum of every track chunk's length, for `bytes_read`.
    bytes_total: u64,
    pub division: Division,
    pub format: u16,
    pub track_count: u16,
    pub path: PathBuf,
}

impl MidiStream {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let h = SmfHeader::read(&path)?;

        if h.trailing_bytes > 0 {
            log::warn!(
                "{}: {} bytes after the last chunk are not chunks; ignored",
                path.display(),
                h.trailing_bytes
            );
        }
        if h.tracks.is_empty() {
            bail!("{}: no MTrk chunks", path.display());
        }
        if h.tracks.len() != h.declared_tracks as usize {
            log::warn!(
                "{}: header claims {} tracks, found {}",
                path.display(),
                h.declared_tracks,
                h.tracks.len()
            );
        }

        let mut readers = Vec::with_capacity(h.tracks.len());
        for (off, len) in &h.tracks {
            readers.push(TrackReader::open(&path, *off, *len)?);
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
            bytes_total: h.tracks.iter().map(|&(_, len)| len).sum(),
            division: h.division,
            format: h.format,
            track_count: h.tracks.len() as u16,
            path,
        })
    }

    /// Bytes of track data in the file, summed over every MTrk chunk.
    pub fn bytes_total(&self) -> u64 {
        self.bytes_total
    }

    /// Bytes of track data decoded so far. \[16\]
    pub fn bytes_read(&self) -> u64 {
        let unread: u64 = self
            .readers
            .iter()
            .filter(|r| !r.ended)
            .map(|r| r.remaining + (r.filled - r.pos) as u64)
            .sum();
        self.bytes_total.saturating_sub(unread)
    }

    /// Next event in tick order, or None at the end of the file. \[17\]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(u64, Event)> {
        // [18]
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
            // [19]
            None => {
                std::collections::binary_heap::PeekMut::pop(top);
            }
        }
        Some((tick, ev))
    }
}

/// Converts ticks to absolute output frames the way BASSMIDI does, tracking \[20\]
#[derive(Debug, Clone)]
pub struct TempoClock {
    division: Division,
    sample_rate: u64,
    /// The whole frame the current tempo took effect on.
    base_frame: u64,
    /// The tick of the tempo change that set it.
    base_tick: u64,
    /// How far past `base_tick` the tick position had already run by \[21\]
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

    /// Take a new tempo's ratio, and the carried overshoot in its frames. The \[22\]
    fn set_ratio(&mut self, us_per_qn: u64) {
        (self.num, self.den) = match self.division {
            Division::Ppq(ppq) => (us_per_qn * self.sample_rate, 1_000_000 * ppq.max(1) as u64),
            Division::Smpte { fps, ticks_per_frame } => {
                (self.sample_rate, fps.max(1) as u64 * ticks_per_frame.max(1) as u64)
            }
        };
        self.overshoot_frames = self.overshoot * self.num as f64 / self.den as f64;
    }

    /// Whole frames from `base_frame` to the first one whose tick position has \[23\]
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
        // [24]
        let frac = rem as f64 / self.den as f64 - self.overshoot_frames;
        (whole as f64 + frac.ceil()).max(0.0) as u64
    }

    /// Apply a tempo change at `tick`, which must not be earlier than the last.
    pub fn set_tempo(&mut self, tick: u64, us_per_qn: u32) {
        if matches!(self.division, Division::Smpte { .. }) {
            return; // SMPTE division ignores tempo meta events
        }
        let n = self.frames_to(tick);
        // [25]
        let past = n as i128 * self.den as i128
            - tick.saturating_sub(self.base_tick) as i128 * self.num as i128;
        self.overshoot = (self.overshoot + past as f64 / self.num as f64).max(0.0);
        self.base_frame += n;
        self.base_tick = tick;
        self.set_ratio(us_per_qn.max(1) as u64);
    }

    /// The frame an event at `tick` fires on. Always a whole frame; `f64` so \[26\]
    #[inline]
    pub fn frame_at(&self, tick: u64) -> f64 {
        (self.base_frame + self.frames_to(tick)) as f64
    }
}

// [27]

/// Minimal SMF writer. Only used by tests and the `gen-test-midi` CLI command, \[28\]
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
        let mut sorted = events;
        sorted.sort_by_key(|e| e.0);
        let mut buf = Vec::new();
        let mut last = 0u64;
        for (tick, msg, len) in sorted {
            write_varlen(&mut buf, tick - last);
            last = tick;
            buf.extend_from_slice(&msg[..len]);
        }
        write_varlen(&mut buf, 0);
        buf.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        self.tracks.push(buf);
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

    /// A tempo change between two samples fires on the later one and carries \[29\]
    #[test]
    fn a_tempo_change_between_samples_carries_what_it_overshot() {
        let mut c = TempoClock::new(Division::Ppq(1920), 48_000);
        c.set_tempo(1920, 2250);
        c.set_tempo(3600, 2_727_273);
        assert_eq!(c.frame_at(3600), 24_095.0);
        assert_eq!(c.frame_at(3618), 24_717.0);
    }

    /// On a sample boundary nothing carries: the same pattern with an 800-tick \[30\]
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

    /// Trailing padding must not be walked eight bytes at a time. \[31\]
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

    /// A file that moves its drums with SysEx must not be read as melodic. \[32\]
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
        // [33]
        assert_eq!(
            sysex_event(&[0x41, 0x10, 0x42, 0x12, 0x40, 0x11, 0x02, 0x40, 0x00, 0xF7]),
            Event::Other
        );
        assert_eq!(sysex_event(&[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x7F, 0xF7]), Event::Other);
        assert_eq!(sysex_event(&[0x41, 0x10]), Event::Other);
        assert_eq!(sysex_event(&[]), Event::Other);
    }

    /// The head-and-skip read has to leave the cursor exactly at the end of the \[34\]
    #[test]
    fn a_sysex_longer_than_the_peek_buffer_does_not_desync_the_track() {
        let dir = std::env::temp_dir().join("kestrel_midi_sysex");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sysex.mid");

        // [35]
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

    /// A file cut short mid-track still opens, and the header reader says so. \[36\]
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

    /// Progress climbs monotonically and lands on exactly the total when the \[37\]
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
