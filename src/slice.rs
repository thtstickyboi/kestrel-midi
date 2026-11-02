// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Note-slicing: one MIDI read once into `N` compact per-slice event streams. \[1\]

use crate::midi::{Division, Event, MidiStream};
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Stream magic. The trailing digit is the format version; a reader that does \[2\]
pub const MAGIC: [u8; 8] = *b"KSTRLSL3";
const MAGIC_V2: [u8; 8] = *b"KSTRLSL2";
const MAGIC_V1: [u8; 8] = *b"KSTRLSL1";

/// Bytes of fixed header before the first record.
pub const HEADER_BYTES: usize = 32;

/// File extension a slice stream is written with. \[3\]
pub const EXT: &str = "ksl";

/// Slot count, `channels * 128 keys` with the channels of every MIDI port. \[4\]
const SLOTS: usize = crate::midi::CHANNELS * 128;

/// Staging before a write, and the read window. 256 KiB matches \[5\]
const BUF: usize = 256 * 1024;

// [6]
const T_NOTE_ON: u8 = 0x00;
const T_NOTE_OFF: u8 = 0x10;
const T_CC: u8 = 0x20;
const T_PROGRAM: u8 = 0x30;
const T_BEND: u8 = 0x40;
const T_DRUM_PART: u8 = 0x50;
const T_RESET_PARTS: u8 = 0x60;
const T_TEMPO: u8 = 0x70;
/// Tempo, tuning and key tuning share a kind because none of them has a \[7\]
const T_TUNE: u8 = 0x71;
const T_KEY_TUNE: u8 = 0x72;
/// Set on a channel record's tag when a port byte follows it: the record's \[8\]
const T_PORT: u8 = 0x80;

/// What one stream says about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceHeader {
    pub division: Division,
    /// This slice's index, `0..slices`.
    pub slice: u16,
    /// How many slices the source was cut into.
    pub slices: u16,
    /// Track count of the source MIDI, carried only so a render can report the \[9\]
    pub tracks: u16,
    /// Records in this stream, `Event::Other` already excluded.
    pub events: u64,
    /// Note-ons in this stream.
    pub notes: u64,
}

/// Where slice `k` of `n` lands under `dir`, given the source's file stem. \[10\]
pub fn slice_path(dir: &Path, stem: &str, k: u16, n: u16) -> PathBuf {
    let width = (n.saturating_sub(1)).to_string().len().max(2);
    dir.join(format!("{stem}.s{k:0width$}.{EXT}", width = width))
}

/// The raw SMF division word, so a slice round-trips a source's timebase \[11\]
fn division_word(d: Division) -> i16 {
    match d {
        Division::Ppq(ppq) => ppq as i16,
        Division::Smpte { fps, ticks_per_frame } => {
            ((-(fps as i16)) << 8) | (ticks_per_frame as i16 & 0xFF)
        }
    }
}

fn division_from_word(w: i16) -> Division {
    if w > 0 {
        Division::Ppq(w as u16)
    } else {
        Division::Smpte {
            fps: (-(w >> 8)) as u8,
            ticks_per_frame: (w & 0xFF) as u8,
        }
    }
}

fn write_varlen(buf: &mut Vec<u8>, mut v: u64) {
    // [12]
    let mut tmp = [0u8; 10];
    let mut n = 0;
    tmp[n] = (v & 0x7F) as u8;
    n += 1;
    v >>= 7;
    while v != 0 {
        tmp[n] = ((v & 0x7F) as u8) | 0x80;
        n += 1;
        v >>= 7;
    }
    for i in (0..n).rev() {
        buf.push(tmp[i]);
    }
}

/// One slice's output stream. \[13\]
pub struct SliceWriter {
    file: File,
    buf: Vec<u8>,
    path: PathBuf,
    last_tick: u64,
    slice: u16,
    slices: u16,
    tracks: u16,
    division: Division,
    events: u64,
    notes: u64,
}

impl SliceWriter {
    pub fn create(
        path: impl AsRef<Path>,
        division: Division,
        slice: u16,
        slices: u16,
        tracks: u16,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::create(&path)
            .with_context(|| format!("creating {}", path.display()))?;
        // [14]
        file.write_all(&[0u8; HEADER_BYTES])?;
        Ok(SliceWriter {
            file,
            buf: Vec::with_capacity(BUF + 64),
            path,
            last_tick: 0,
            slice,
            slices,
            tracks,
            division,
            events: 0,
            notes: 0,
        })
    }

    #[inline]
    fn flush_if_full(&mut self) -> Result<()> {
        if self.buf.len() >= BUF {
            self.file.write_all(&self.buf)?;
            self.buf.clear();
        }
        Ok(())
    }

    /// A channel record's tag, and the port byte after it when the channel \[15\]
    #[inline]
    fn tag(&mut self, kind: u8, ch: u8) {
        if ch < 16 {
            self.buf.push(kind | ch);
        } else {
            self.buf.push(kind | T_PORT | (ch & 15));
            self.buf.push(ch >> 4);
        }
    }

    /// Append one event. `tick` is absolute and must not go backwards; the \[16\]
    pub fn push(&mut self, tick: u64, ev: Event) -> Result<()> {
        let delta = tick.saturating_sub(self.last_tick);
        let before = self.buf.len();
        write_varlen(&mut self.buf, delta);
        match ev {
            Event::NoteOn { ch, key, vel } => {
                self.tag(T_NOTE_ON, ch);
                self.buf.push(key & 127);
                self.buf.push(vel);
                self.notes += 1;
            }
            Event::NoteOff { ch, key } => {
                self.tag(T_NOTE_OFF, ch);
                self.buf.push(key & 127);
            }
            Event::Cc { ch, num, val } => {
                self.tag(T_CC, ch);
                self.buf.push(num);
                self.buf.push(val);
            }
            Event::Program { ch, val } => {
                self.tag(T_PROGRAM, ch);
                self.buf.push(val);
            }
            Event::PitchBend { ch, val } => {
                self.tag(T_BEND, ch);
                self.buf.extend_from_slice(&val.to_le_bytes());
            }
            Event::DrumPart { ch, map } => {
                self.tag(T_DRUM_PART, ch);
                self.buf.push(map);
            }
            Event::ResetParts => self.buf.push(T_RESET_PARTS),
            Event::Tempo(us) => {
                self.buf.push(T_TEMPO);
                self.buf.extend_from_slice(&us.to_be_bytes()[1..4]);
            }
            // [17]
            Event::Tune { ch, pc, units } => {
                self.buf.push(T_TUNE);
                self.buf.push(ch);
                self.buf.push(pc);
                self.buf.extend_from_slice(&units.to_le_bytes());
            }
            Event::KeyTune { bank, prog, key, units } => {
                self.buf.push(T_KEY_TUNE);
                self.buf.push(bank);
                self.buf.push(prog);
                self.buf.push(key);
                self.buf.extend_from_slice(&units.to_le_bytes());
            }
            // [18]
            Event::Other => {
                self.buf.truncate(before);
                return Ok(());
            }
        }
        self.last_tick = tick;
        self.events += 1;
        self.flush_if_full()
    }

    /// Flush, stamp the real header, and report what was written.
    pub fn finish(mut self) -> Result<SliceHeader> {
        if !self.buf.is_empty() {
            self.file.write_all(&self.buf)?;
            self.buf.clear();
        }
        let hdr = SliceHeader {
            division: self.division,
            slice: self.slice,
            slices: self.slices,
            tracks: self.tracks,
            events: self.events,
            notes: self.notes,
        };
        let mut h = [0u8; HEADER_BYTES];
        h[0..8].copy_from_slice(&MAGIC);
        h[8..10].copy_from_slice(&division_word(hdr.division).to_le_bytes());
        h[10..12].copy_from_slice(&hdr.slice.to_le_bytes());
        h[12..14].copy_from_slice(&hdr.slices.to_le_bytes());
        h[14..16].copy_from_slice(&hdr.tracks.to_le_bytes());
        h[16..24].copy_from_slice(&hdr.events.to_le_bytes());
        h[24..32].copy_from_slice(&hdr.notes.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&h)?;
        self.file.flush()?;
        Ok(hdr)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Reads one slice stream back as `(tick, Event)`, the same shape \[19\]
pub struct SliceReader {
    file: File,
    buf: Box<[u8]>,
    pos: usize,
    filled: usize,
    tick: u64,
    /// Bytes of the file read into `buf` so far, the header included, and the \[20\]
    fetched: u64,
    total: u64,
    pub header: SliceHeader,
    pub path: PathBuf,
}

impl SliceReader {
    /// The stream's length in bytes. The counterpart of \[21\]
    pub fn bytes_total(&self) -> u64 {
        self.total
    }

    /// Bytes of the stream decoded so far.
    pub fn bytes_read(&self) -> u64 {
        self.fetched - (self.filled - self.pos) as u64
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mut h = [0u8; HEADER_BYTES];
        file.read_exact(&mut h)
            .with_context(|| format!("{}: too short for a slice header", path.display()))?;
        if h[0..8] != MAGIC && h[0..8] != MAGIC_V2 && h[0..8] != MAGIC_V1 {
            bail!(
                "{}: not a Kestrel slice stream (expected {:?})",
                path.display(),
                std::str::from_utf8(&MAGIC).unwrap()
            );
        }
        let header = SliceHeader {
            division: division_from_word(i16::from_le_bytes(h[8..10].try_into().unwrap())),
            slice: u16::from_le_bytes(h[10..12].try_into().unwrap()),
            slices: u16::from_le_bytes(h[12..14].try_into().unwrap()),
            tracks: u16::from_le_bytes(h[14..16].try_into().unwrap()),
            events: u64::from_le_bytes(h[16..24].try_into().unwrap()),
            notes: u64::from_le_bytes(h[24..32].try_into().unwrap()),
        };
        let total = file.metadata().map(|m| m.len()).unwrap_or(HEADER_BYTES as u64);
        Ok(SliceReader {
            file,
            buf: vec![0u8; BUF].into_boxed_slice(),
            pos: 0,
            filled: 0,
            tick: 0,
            fetched: HEADER_BYTES as u64,
            total,
            header,
            path,
        })
    }

    #[inline]
    fn fill(&mut self) -> bool {
        let mut got = 0usize;
        while got == 0 {
            match self.file.read(&mut self.buf) {
                Ok(0) => break,
                Ok(n) => got = n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        self.pos = 0;
        self.filled = got;
        self.fetched += got as u64;
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

    /// A little-endian `i32`, as the tuning records carry their offsets.
    #[inline]
    fn i32(&mut self) -> Option<i32> {
        let mut b = [0u8; 4];
        for x in &mut b {
            *x = self.byte()?;
        }
        Some(i32::from_le_bytes(b))
    }

    #[inline]
    fn varlen(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for _ in 0..10 {
            let b = self.byte()?;
            v = (v << 7) | (b & 0x7F) as u64;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    /// Next event, or `None` at the end of the stream. \[22\]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(u64, Event)> {
        let delta = self.varlen()?;
        let tag = self.byte()?;
        let ch = if tag & T_PORT != 0 {
            (self.byte()? << 4) | (tag & 15)
        } else {
            tag & 15
        };
        let ev = match tag & 0x70 {
            T_NOTE_ON => {
                let key = self.byte()?;
                let vel = self.byte()?;
                Event::NoteOn { ch, key, vel }
            }
            T_NOTE_OFF => Event::NoteOff { ch, key: self.byte()? },
            T_CC => {
                let num = self.byte()?;
                let val = self.byte()?;
                Event::Cc { ch, num, val }
            }
            T_PROGRAM => Event::Program { ch, val: self.byte()? },
            T_BEND => {
                let lo = self.byte()? as u16;
                let hi = self.byte()? as u16;
                Event::PitchBend { ch, val: ((hi << 8) | lo) as i16 }
            }
            T_DRUM_PART => Event::DrumPart { ch, map: self.byte()? },
            T_RESET_PARTS => Event::ResetParts,
            T_TEMPO => match tag & 15 {
                0 => {
                    let a = self.byte()? as u32;
                    let b = self.byte()? as u32;
                    let c = self.byte()? as u32;
                    Event::Tempo((a << 16) | (b << 8) | c)
                }
                n if n == T_TUNE & 15 => {
                    let ch = self.byte()?;
                    let pc = self.byte()?;
                    Event::Tune { ch, pc, units: self.i32()? }
                }
                n if n == T_KEY_TUNE & 15 => {
                    let bank = self.byte()?;
                    let prog = self.byte()?;
                    let key = self.byte()?;
                    Event::KeyTune { bank, prog, key, units: self.i32()? }
                }
                _ => return None,
            },
            // [23]
            _ => return None,
        };
        self.tick += delta;
        Some((self.tick, ev))
    }
}

/// What one slice ended up holding.
#[derive(Debug, Clone)]
pub struct SliceStats {
    pub path: PathBuf,
    pub header: SliceHeader,
    pub bytes: u64,
}

/// Fans a tick-ordered event stream out to `N` slice writers.
pub struct Slicer {
    writers: Vec<SliceWriter>,
    n: u32,
    /// Note-ons seen per `(channel, key)` slot, and note-offs paired against \[24\]
    on_seen: Vec<u32>,
    off_seen: Vec<u32>,
    /// Note-offs with nothing outstanding, dropped exactly as `GateTable` \[25\]
    pub strays: u64,
    pub notes: u64,
    pub events: u64,
}

impl Slicer {
    pub fn create(dir: &Path, stem: &str, n: u16, division: Division, tracks: u16) -> Result<Self> {
        if n < 1 {
            bail!("a slice count of {n} makes no sense; use 1 or more");
        }
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        let mut writers = Vec::with_capacity(n as usize);
        for k in 0..n {
            writers.push(SliceWriter::create(
                slice_path(dir, stem, k, n),
                division,
                k,
                n,
                tracks,
            )?);
        }
        // [26]
        let phase: Vec<u32> = (0..SLOTS).map(|s| (s as u32) % (n as u32)).collect();
        Ok(Slicer {
            writers,
            n: n as u32,
            on_seen: phase.clone(),
            off_seen: phase,
            strays: 0,
            notes: 0,
            events: 0,
        })
    }

    #[inline]
    fn slot(ch: u8, key: u8) -> usize {
        ch as usize * 128 + (key as usize & 127)
    }

    /// Route one event. Notes go to one slice, everything else to all of them.
    pub fn push(&mut self, tick: u64, ev: Event) -> Result<()> {
        match ev {
            Event::NoteOn { ch, key, .. } => {
                let s = Self::slot(ch, key);
                let k = (self.on_seen[s] % self.n) as usize;
                self.on_seen[s] = self.on_seen[s].wrapping_add(1);
                self.notes += 1;
                self.events += 1;
                self.writers[k].push(tick, ev)?;
            }
            Event::NoteOff { ch, key } => {
                let s = Self::slot(ch, key);
                if self.off_seen[s] == self.on_seen[s] {
                    // [27]
                    self.strays += 1;
                    return Ok(());
                }
                let k = (self.off_seen[s] % self.n) as usize;
                self.off_seen[s] = self.off_seen[s].wrapping_add(1);
                self.events += 1;
                self.writers[k].push(tick, ev)?;
            }
            Event::Other => {}
            ev => {
                // [28]
                if let Event::Cc { ch, num, .. } = ev {
                    if num == 120 || (123..=127).contains(&num) {
                        let base = ch as usize * 128;
                        self.off_seen[base..base + 128]
                            .copy_from_slice(&self.on_seen[base..base + 128]);
                    }
                }
                self.events += 1;
                for w in &mut self.writers {
                    w.push(tick, ev)?;
                }
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<SliceStats>> {
        let mut out = Vec::with_capacity(self.writers.len());
        for w in self.writers {
            let path = w.path().to_path_buf();
            let header = w.finish()?;
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            out.push(SliceStats { path, header, bytes });
        }
        Ok(out)
    }
}

/// Everything one `slice` run produced.
#[derive(Debug, Clone)]
pub struct SliceSummary {
    pub slices: Vec<SliceStats>,
    pub source_events: u64,
    pub source_notes: u64,
    pub strays: u64,
    pub manifest: PathBuf,
}

/// Read `midi` once and write `n` slice streams under `dir`. \[29\]
pub fn slice_file(
    midi: &Path,
    dir: &Path,
    n: u16,
    mut progress: impl FnMut(u64),
) -> Result<SliceSummary> {
    let mut stream = MidiStream::open(midi)?;
    let stem = midi
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "slice".to_string());
    let mut slicer = Slicer::create(dir, &stem, n, stream.division, stream.track_count)?;

    let mut source_events = 0u64;
    let mut next_report = 1u64 << 22;
    while let Some((tick, ev)) = stream.next() {
        source_events += 1;
        slicer.push(tick, ev)?;
        if slicer.notes >= next_report {
            progress(slicer.notes);
            next_report = slicer.notes + (1 << 22);
        }
    }

    let source_notes = slicer.notes;
    let strays = slicer.strays;
    let slices = slicer.finish()?;
    let manifest = write_manifest(dir, &stem, midi, &slices, source_events, source_notes, strays)?;
    Ok(SliceSummary { slices, source_events, source_notes, strays, manifest })
}

/// A plain-text index beside the streams. \[30\]
#[allow(clippy::too_many_arguments)]
fn write_manifest(
    dir: &Path,
    stem: &str,
    midi: &Path,
    slices: &[SliceStats],
    source_events: u64,
    source_notes: u64,
    strays: u64,
) -> Result<PathBuf> {
    let path = dir.join(format!("{stem}.slices.txt"));
    let mut f = std::io::BufWriter::new(File::create(&path)?);
    writeln!(f, "source        {}", midi.display())?;
    writeln!(f, "slices        {}", slices.len())?;
    writeln!(f, "source notes  {source_notes}")?;
    writeln!(f, "source events {source_events} (Event::Other not carried)")?;
    if strays > 0 {
        writeln!(f, "stray offs    {strays} (note-offs with nothing sounding, dropped)")?;
    }
    writeln!(f)?;
    // [31]
    let w = slices
        .iter()
        .filter_map(|s| s.path.file_name().map(|n| n.to_string_lossy().chars().count()))
        .max()
        .unwrap_or(32)
        .max(4);
    writeln!(f, "{:<w$} {:>16} {:>16} {:>14}", "file", "notes", "events", "bytes")?;
    for s in slices {
        let name = s.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        writeln!(
            f,
            "{:<w$} {:>16} {:>16} {:>14}",
            name, s.header.notes, s.header.events, s.bytes
        )?;
    }
    writeln!(f)?;
    writeln!(
        f,
        "Render each with:\n\
         \x20   --note-slice <file> --backend merlin --merlin-promote-after 0 \\\n\
         \x20   --limiter off --format float32\n\
         then sum the stems with `cargo run --release --example merge_stems`.\n\
         \n\
         `--merlin-promote-after 0` is required and the renderer refuses without it.\n\
         The fallback router promotes a waveform to the convolution path on its Nth\n\
         note, and a slice holds a fraction of each waveform's notes -- so it would\n\
         promote different waveforms at different points than the whole file, and a\n\
         note on the per-voice fallback is not the same samples as one convolved.\n\
         \n\
         The limiter must run once on the sum: it is not a linear operator, so\n\
         limiting each slice and adding them neither bounds the total nor sounds\n\
         like the mix."
    )?;
    f.flush()?;
    Ok(path)
}

/// Whether this path names a slice stream rather than a MIDI file.
pub fn is_slice_path(p: &Path) -> bool {
    p.extension()
        .map(|e| e.eq_ignore_ascii_case(EXT))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip through a real file, under a name of this call's own. \[32\]
    fn roundtrip(name: &str, evs: &[(u64, Event)], division: Division) -> Vec<(u64, Event)> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch")
            .join("slice-unit");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.{EXT}"));
        let mut w = SliceWriter::create(&path, division, 0, 1, 3).unwrap();
        for (t, e) in evs {
            w.push(*t, *e).unwrap();
        }
        let hdr = w.finish().unwrap();
        assert_eq!(hdr.division, division);
        let mut r = SliceReader::open(&path).unwrap();
        assert_eq!(r.header, hdr);
        let mut out = Vec::new();
        while let Some(p) = r.next() {
            out.push(p);
        }
        std::fs::remove_file(&path).ok();
        out
    }

    #[test]
    fn every_event_kind_survives_a_round_trip() {
        let evs = vec![
            (0, Event::Tempo(500_000)),
            (0, Event::Program { ch: 3, val: 40 }),
            (0, Event::ResetParts),
            (5, Event::NoteOn { ch: 0, key: 60, vel: 100 }),
            (5, Event::Cc { ch: 15, num: 7, val: 90 }),
            (7, Event::PitchBend { ch: 9, val: -8192 }),
            (7, Event::PitchBend { ch: 9, val: 8191 }),
            (9, Event::DrumPart { ch: 10, map: 2 }),
            (1_000_000_000, Event::NoteOff { ch: 0, key: 60 }),
        ];
        assert_eq!(roundtrip("kinds", &evs, Division::Ppq(960)), evs);
    }

    /// Channels past the first port -- up to the last channel of port 16 -- \[33\]
    #[test]
    fn every_channel_of_every_port_survives_a_round_trip() {
        let mut evs = Vec::new();
        for (i, ch) in [0u8, 15, 16, 31, 146, 255].into_iter().enumerate() {
            let t = i as u64 * 10;
            evs.push((t, Event::NoteOn { ch, key: 60, vel: 100 }));
            evs.push((t, Event::Cc { ch, num: 7, val: 90 }));
            evs.push((t, Event::Program { ch, val: 40 }));
            evs.push((t, Event::PitchBend { ch, val: -8192 }));
            evs.push((t, Event::DrumPart { ch, map: 1 }));
            evs.push((t + 5, Event::NoteOff { ch, key: 60 }));
        }
        assert_eq!(roundtrip("ports", &evs, Division::Ppq(960)), evs);
    }

    /// A record on port 0 is exactly what version 1 wrote, and one on another \[34\]
    #[test]
    fn a_port_costs_one_byte_and_port_zero_costs_none() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch")
            .join("slice-unit");
        std::fs::create_dir_all(&dir).unwrap();
        let len = |name: &str, ch: u8| {
            let path = dir.join(format!("{name}.{EXT}"));
            let mut w = SliceWriter::create(&path, Division::Ppq(960), 0, 1, 1).unwrap();
            w.push(0, Event::NoteOn { ch, key: 60, vel: 100 }).unwrap();
            w.finish().unwrap();
            let n = std::fs::metadata(&path).unwrap().len();
            std::fs::remove_file(&path).ok();
            n
        };
        // Delta, tag, key and velocity.
        assert_eq!(len("port0", 5), HEADER_BYTES as u64 + 4);
        assert_eq!(len("port1", 16 + 5), HEADER_BYTES as u64 + 5);
    }

    /// Streams written before ports carry the version 1 magic, and read.
    #[test]
    fn a_version_1_stream_still_reads() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch")
            .join("slice-unit");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("v1.{EXT}"));
        let evs = vec![(3, Event::NoteOn { ch: 2, key: 64, vel: 90 }), (9, Event::NoteOff { ch: 2, key: 64 })];
        let mut w = SliceWriter::create(&path, Division::Ppq(960), 0, 1, 1).unwrap();
        for (t, e) in &evs {
            w.push(*t, *e).unwrap();
        }
        w.finish().unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0..8].copy_from_slice(&MAGIC_V1);
        std::fs::write(&path, &bytes).unwrap();
        let mut r = SliceReader::open(&path).unwrap();
        let mut out = Vec::new();
        while let Some(p) = r.next() {
            out.push(p);
        }
        std::fs::remove_file(&path).ok();
        assert_eq!(out, evs);
    }

    /// Tuning records share the tempo tag's kind and say which they are in its \[35\]
    #[test]
    fn tuning_records_survive_a_round_trip_beside_tempo() {
        let evs = vec![
            (0, Event::Tempo(500_000)),
            (0, Event::Tune { ch: 0, pc: 0, units: 0 }),
            (0, Event::Tune { ch: 146, pc: 11, units: -123_456 }),
            (1, Event::Tempo(0xFF_FFFF)),
            (1, Event::KeyTune { bank: 0, prog: 0, key: 60, units: 77_000 }),
            (1, Event::KeyTune { bank: 127, prog: 127, key: 127, units: i32::MIN }),
            (2, Event::Tune { ch: 255, pc: 5, units: i32::MAX }),
            (3, Event::Tempo(250_000)),
            (4, Event::NoteOn { ch: 146, key: 60, vel: 100 }),
        ];
        assert_eq!(roundtrip("tuning", &evs, Division::Ppq(960)), evs);
    }

    /// Version 2 streams, written before the tuning records, carry none of them \[36\]
    #[test]
    fn a_version_2_stream_still_reads() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch")
            .join("slice-unit");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("v2.{EXT}"));
        let evs = vec![
            (0, Event::Tempo(600_000)),
            (3, Event::NoteOn { ch: 40, key: 64, vel: 90 }),
            (9, Event::NoteOff { ch: 40, key: 64 }),
        ];
        let mut w = SliceWriter::create(&path, Division::Ppq(960), 0, 1, 1).unwrap();
        for (t, e) in &evs {
            w.push(*t, *e).unwrap();
        }
        w.finish().unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0..8].copy_from_slice(&MAGIC_V2);
        std::fs::write(&path, &bytes).unwrap();
        let mut r = SliceReader::open(&path).unwrap();
        let mut out = Vec::new();
        while let Some(p) = r.next() {
            out.push(p);
        }
        std::fs::remove_file(&path).ok();
        assert_eq!(out, evs);
    }

    #[test]
    fn smpte_division_round_trips() {
        let evs = vec![(1, Event::NoteOn { ch: 1, key: 64, vel: 1 })];
        let d = Division::Smpte { fps: 25, ticks_per_frame: 40 };
        assert_eq!(roundtrip("smpte", &evs, d), evs);
        // And the word itself, since this is the half a MIDI header carries.
        assert_eq!(division_from_word(division_word(d)), d);
    }

    #[test]
    fn varlen_covers_a_u64_gap() {
        let mut b = Vec::new();
        write_varlen(&mut b, u64::MAX >> 1);
        assert!(b.len() <= 10);
    }
}
