// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What each track of a MIDI holds, read before a per-track render is set up. \[1\]

use crate::midi::{Division, Event, SmfHeader, TempoClock, TrackReader};
use anyhow::{bail, Context, Result};
use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// What one track chunk holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackInfo {
    /// The track's first FF 03 name, at most 255 bytes of it. Bytes, not a \[2\]
    pub name: Option<Vec<u8>>,
    /// Note-ons at each velocity. `[0]` is always zero, a velocity-0 note-on \[3\]
    pub by_velocity: [u64; 128],
    /// Tick of the first note-on.
    pub first_note: Option<u64>,
    /// Tick of the last note event after the first note-on, note-on or \[4\]
    pub last_note: Option<u64>,
    /// MIDI channels a note-on used, one bit per channel of its port. The port \[5\]
    pub channels: u16,
    /// Controllers, program changes, pitch bends and the SysEx messages that \[6\]
    pub controls: u64,
    /// FF 51 events on this track.
    pub tempos: u64,
    /// Tick of the track's last event other than a tempo change, its \[7\]
    pub end_tick: u64,
    /// The chunk's length in bytes.
    pub bytes: u64,
}

/// What a track is to a per-track render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    /// Note-ons: a stem of its own.
    Notes,
    /// Controllers, programs, bends or part SysEx and no notes: applied to \[8\]
    Setup,
    /// Neither. Names, text, tempo -- and tempo reaches every track through \[9\]
    Empty,
}

impl TrackInfo {
    pub fn notes(&self) -> u64 {
        self.by_velocity.iter().sum()
    }

    /// Note-ons `--min-velocity min` keeps: velocity `min` and up, as the \[10\]
    pub fn notes_from(&self, min: u8) -> u64 {
        self.by_velocity[(min as usize).min(127)..].iter().sum()
    }

    pub fn kind(&self) -> TrackKind {
        if self.notes() > 0 {
            TrackKind::Notes
        } else if self.controls > 0 {
            TrackKind::Setup
        } else {
            TrackKind::Empty
        }
    }

    /// Lowest and highest velocity any note-on used.
    pub fn velocity_range(&self) -> Option<(u8, u8)> {
        let lo = self.by_velocity.iter().position(|&n| n > 0)?;
        let hi = self.by_velocity.iter().rposition(|&n| n > 0)?;
        Some((lo as u8, hi as u8))
    }

    /// The name for display, trimmed of the padding and control bytes \[11\]
    pub fn display_name(&self) -> Option<String> {
        let name = decode_name(self.name.as_deref()?);
        let name = name.trim_matches(|c: char| c.is_whitespace() || c.is_control());
        (!name.is_empty()).then(|| name.to_string())
    }
}

/// A track name's bytes as text: UTF-8 when they are valid UTF-8, which \[12\]
pub fn decode_name(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return text.into();
    }
    encoding_rs::SHIFT_JIS
        .decode_without_bom_handling_and_without_replacement(bytes)
        .unwrap_or_else(|| String::from_utf8_lossy(bytes))
}

/// Every track of a MIDI, and the tempo map that times them all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackScan {
    pub path: PathBuf,
    pub format: u16,
    pub division: Division,
    /// One per MTrk chunk, in file order.
    pub tracks: Vec<TrackInfo>,
    /// Every FF 51 in the file as (tick, microseconds per quarter note), in \[13\]
    pub tempo: Arc<[(u64, u32)]>,
    /// The chunk table, which every track's render opens the file through.
    pub header: Arc<SmfHeader>,
}

/// Whether a per-track render hears the file's setup tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SetupTracks {
    /// Their controllers, programs, bends and part SysEx reach the track as \[14\]
    #[default]
    Apply,
    /// Left out, so the track hears only its own.
    Ignore,
}

/// One track of a file rendered on its own, and whether the setup tracks \[15\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackPick {
    /// 0-based, where `kestrel tracks` and `--track` count from 1.
    pub index: usize,
    pub setup: SetupTracks,
}

/// Which tracks a stems render takes, as `--tracks` spells them: `all`, or \[16\]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackList {
    /// Every track with notes.
    All,
    /// Inclusive 0-based ranges, `None` for an open end, as given.
    Pick(Vec<(usize, Option<usize>)>),
}

impl TrackList {
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.eq_ignore_ascii_case("all") {
            return Ok(TrackList::All);
        }
        let number = |s: &str| -> Result<usize> {
            match s.trim().parse::<usize>() {
                Ok(0) => bail!("--tracks counts from 1, as `kestrel tracks` does; 0 is not a track"),
                Ok(n) => Ok(n - 1),
                Err(_) => bail!("--tracks {spec:?}: {:?} is not a track number", s.trim()),
            }
        };
        let mut out = Vec::new();
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part.split_once('-') {
                Some((a, b)) => {
                    let lo = number(a)?;
                    let hi = if b.trim().is_empty() { None } else { Some(number(b)?) };
                    if hi.is_some_and(|hi| hi < lo) {
                        bail!("--tracks range {part:?} runs backwards");
                    }
                    out.push((lo, hi));
                }
                None => {
                    let n = number(part)?;
                    out.push((n, Some(n)));
                }
            }
        }
        if out.is_empty() {
            bail!("--tracks is empty; give `all`, or numbers and ranges such as 1-40,57,90-");
        }
        Ok(TrackList::Pick(out))
    }

    /// The tracks this names in `scan`, in order and once each, split into \[17\]
    pub fn resolve(&self, scan: &TrackScan) -> Result<(Vec<usize>, Vec<usize>)> {
        let n = scan.tracks.len();
        let named: Vec<usize> = match self {
            TrackList::All => return Ok((scan.of_kind(TrackKind::Notes).collect(), Vec::new())),
            TrackList::Pick(ranges) => {
                let mut v = Vec::new();
                for &(lo, hi) in ranges {
                    let last = hi.unwrap_or(n.saturating_sub(1));
                    if lo >= n || last >= n {
                        bail!(
                            "{}: --tracks names track {}, and the file has {}",
                            scan.path.display(),
                            lo.max(last) + 1,
                            n
                        );
                    }
                    v.extend(lo..=last);
                }
                v.sort_unstable();
                v.dedup();
                v
            }
        };
        Ok(named.into_iter().partition(|&t| scan.tracks[t].kind() == TrackKind::Notes))
    }
}

/// A per-track render: one file per track, all in a folder named after the \[18\]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stems {
    pub tracks: TrackList,
    pub setup: SetupTracks,
    /// Threads doing the host's share of the tracks. Held to `max_jobs`.
    pub jobs: usize,
    /// The container every stem is written in: `wav`, `flac`, `opus`, ... \[19\]
    pub ext: String,
    /// One file, every track rendered alone and then summed, limited and \[20\]
    pub merge: bool,
    /// The file's scan, when the caller has made one already -- the guided \[21\]
    pub scanned: Option<Arc<TrackScan>>,
}

/// Default and ceiling for `Stems::jobs`, decided with the user 2026-09-23: \[22\]
pub fn default_jobs() -> usize {
    8.min(max_jobs())
}

pub fn max_jobs() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .saturating_sub(2)
        .max(1)
}

/// Seconds each track of a per-track render runs: to where the file ends and \[23\]
pub fn render_secs(scan: &TrackScan, rate: u32, seconds: Option<f64>) -> f64 {
    let secs = scan.duration(rate) + 1.0;
    seconds.map_or(secs, |s| secs.min(s))
}

/// Bytes a per-track render writes as `files` files of `secs` each in the \[24\]
pub fn output_bytes(files: usize, secs: f64, rate: u32, ext: &str, float32: bool) -> (u64, bool) {
    let rate = rate as f64;
    let (per_sec, exact) = match ext {
        "wav" => (rate * 2.0 * if float32 { 4.0 } else { 2.0 }, true),
        "flac" => (rate * 2.0 * 3.0, false),
        "opus" => (160_000.0 / 8.0, false),
        "mp3" => (320_000.0 / 8.0, false),
        "ogg" => (256_000.0 / 8.0, false),
        _ => (256_000.0 / 8.0, false),
    };
    ((files as f64 * (per_sec * secs + 4096.0)) as u64, exact)
}

/// The per-block candidate cap each track of a per-track render gets, from \[25\]
pub fn candidates_each(cfg: &crate::config::Config) -> u32 {
    let floor = 1u64 << 16;
    let ceiling = (cfg.max_block_candidates as u64 / 256).max(floor);
    (cfg.pool_slots() as u64 * 64).clamp(floor, ceiling) as u32
}

/// Each stem's share of a voice total, split evenly, as decided with the user \[26\]
pub fn voices_each(total: u32, stems: usize) -> u32 {
    (total / stems.max(1) as u32).max(1)
}

/// The guided renderer's default voice total for a per-track render, set by \[27\]
pub const GUIDED_VOICES: u32 = 60_000_000;

/// The most voices a per-track render may be given in all, set by the user \[28\]
pub const MAX_TOTAL_VOICES: u32 = 2_000_000_000;

/// `cfg` as a per-track render lays out its voices for `bank`.
fn laid_out(cfg: &crate::config::Config, bank: &crate::bank::Bank, voices: u32) -> crate::config::Config {
    let mut cfg = crate::config::Config { max_voices: voices, ..cfg.clone() };
    crate::session::fit_to_bank(&mut cfg, bank);
    cfg
}

/// The most voices one track holds on an adapter that binds `binding_bytes` \[29\]
pub fn max_voices_alone(cfg: &crate::config::Config, bank: &crate::bank::Bank, binding_bytes: u64) -> u32 {
    crate::gpu::GpuBatch::max_voices_each(&laid_out(cfg, bank, 1), bank, binding_bytes, 1)
}

/// The most voices a per-track render of `tracks` tracks can be given with as \[30\]
pub fn max_voices_at_once(cfg: &crate::config::Config, bank: &crate::bank::Bank, binding_bytes: u64, tracks: usize) -> u32 {
    use crate::gpu::GpuBatch;
    let tracks = tracks.max(1);
    // [31]
    let lanes = GpuBatch::lanes_that_bind(&laid_out(cfg, bank, 1), bank, binding_bytes, tracks.min(crate::gpu::LANES_MAX)).max(1);
    let each = GpuBatch::max_voices_each(&laid_out(cfg, bank, 1), bank, binding_bytes, lanes);
    (each as u64 * tracks as u64).min(MAX_TOTAL_VOICES as u64) as u32
}

/// How many of `tracks` tracks share the device at a time when they are given \[32\]
pub fn tracks_at_once(cfg: &crate::config::Config, bank: &crate::bank::Bank, binding_bytes: u64, tracks: usize, total: u32) -> usize {
    let lanes = tracks.clamp(1, crate::gpu::LANES_MAX);
    crate::gpu::GpuBatch::lanes_that_bind(&laid_out(cfg, bank, voices_each(total, tracks)), bank, binding_bytes, lanes)
}

/// The file a stem is written to, inside the folder: the track number padded \[33\]
pub fn stem_file_name(index: usize, track_count: usize, name: Option<&str>, ext: &str) -> String {
    let width = track_count.max(1).to_string().len();
    let mut clean: String = name
        .unwrap_or("")
        .chars()
        .map(|c| if c.is_control() || r#"<>:"/\|?*"#.contains(c) { '_' } else { c })
        .take(80)
        .collect();
    while clean.ends_with(['.', ' ']) {
        clean.pop();
    }
    let clean = clean.trim_start();
    if clean.is_empty() {
        format!("{:0width$}.{ext}", index + 1)
    } else {
        format!("{:0width$} {clean}.{ext}", index + 1)
    }
}

impl SetupTracks {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "apply" => Some(SetupTracks::Apply),
            "ignore" => Some(SetupTracks::Ignore),
            _ => None,
        }
    }
}

impl TrackScan {
    /// Tick of the file's last event on any track, tempo changes aside; see \[34\]
    pub fn end_tick(&self) -> u64 {
        self.tracks.iter().map(|t| t.end_tick).max().unwrap_or(0)
    }

    /// What a render of `track` alone reads: that track, the setup tracks \[35\]
    pub fn selection(&self, track: usize, setup: SetupTracks) -> Result<crate::midi::TrackSelection> {
        Ok(self.selections(&[track], setup)?.remove(0))
    }

    /// `selection` for each of `tracks`. The setup tracks and the end are the \[36\]
    pub fn selections(&self, tracks: &[usize], setup: SetupTracks) -> Result<Vec<crate::midi::TrackSelection>> {
        let setups: Vec<usize> = match setup {
            SetupTracks::Apply => self.of_kind(TrackKind::Setup).collect(),
            SetupTracks::Ignore => Vec::new(),
        };
        let end_tick = self.end_tick();
        tracks.iter().map(|&t| self.select(t, &setups, end_tick)).collect()
    }

    fn select(&self, track: usize, setups: &[usize], end_tick: u64) -> Result<crate::midi::TrackSelection> {
        let Some(t) = self.tracks.get(track) else {
            bail!(
                "{}: there is no track {}; the file has {}",
                self.path.display(),
                track + 1,
                self.tracks.len()
            );
        };
        match t.kind() {
            TrackKind::Notes => {}
            kind => bail!(
                "{}: track {} has no notes to render ({}); `kestrel tracks` lists the ones that do",
                self.path.display(),
                track + 1,
                if kind == TrackKind::Setup { "it is a setup track" } else { "it is empty" }
            ),
        }
        let mut tracks = vec![track];
        tracks.extend_from_slice(setups);
        tracks.sort_unstable();
        Ok(crate::midi::TrackSelection {
            tracks,
            tempo: self.tempo.clone(),
            header: self.header.clone(),
            end_tick,
        })
    }

    pub fn notes(&self) -> u64 {
        self.tracks.iter().map(TrackInfo::notes).sum()
    }

    /// The track with the most note-ons, the first of them on a tie. `None` \[37\]
    pub fn busiest(&self) -> Option<usize> {
        let (i, t) = self
            .tracks
            .iter()
            .enumerate()
            .max_by_key(|&(i, t)| (t.notes(), Reverse(i)))?;
        (t.notes() > 0).then_some(i)
    }

    /// Tracks of one kind, by index.
    pub fn of_kind(&self, kind: TrackKind) -> impl Iterator<Item = usize> + '_ {
        self.tracks
            .iter()
            .enumerate()
            .filter(move |(_, t)| t.kind() == kind)
            .map(|(i, _)| i)
    }

    /// The output frame each of `ticks` falls on at `rate`, by the same clock \[38\]
    pub fn frames_at(&self, ticks: &[u64], rate: u32) -> Vec<f64> {
        let mut order: Vec<usize> = (0..ticks.len()).collect();
        order.sort_by_key(|&i| ticks[i]);
        let mut clock = TempoClock::new(self.division, rate);
        let mut tempo = self.tempo.iter().peekable();
        let mut out = vec![0.0; ticks.len()];
        for i in order {
            // [39]
            while let Some(&&(t, us)) = tempo.peek() {
                if t > ticks[i] {
                    break;
                }
                clock.set_tempo(t, us);
                tempo.next();
            }
            out[i] = clock.frame_at(ticks[i]);
        }
        out
    }

    /// Seconds from the start to the file's last event at `rate`. The render \[40\]
    pub fn duration(&self, rate: u32) -> f64 {
        self.frames_at(&[self.end_tick()], rate)[0] / rate as f64
    }
}

/// A scan in flight, for a front end to show and to stop.
#[derive(Debug, Default)]
pub struct ScanProgress {
    /// Track bytes in the file, set as the scan starts.
    pub bytes_total: AtomicU64,
    /// Track bytes decoded so far. Reaches `bytes_total` exactly when the scan \[41\]
    pub bytes_read: AtomicU64,
    /// Set to stop the scan, which then returns an error saying so.
    pub cancel: AtomicBool,
}

/// Threads a scan of `tracks` tracks runs on when asked for `jobs`: one per \[42\]
pub fn jobs_for(jobs: usize, tracks: usize) -> usize {
    match jobs {
        0 => std::thread::available_parallelism().map_or(4, |n| n.get()),
        n => n,
    }
    .clamp(1, tracks.max(1))
}

/// Scan every track of `path` on up to `jobs` threads, 0 for one per core.
pub fn scan(path: impl AsRef<Path>, jobs: usize, progress: Option<&ScanProgress>) -> Result<TrackScan> {
    let path = path.as_ref();
    let h = SmfHeader::read(path)?;
    if h.tracks.is_empty() {
        bail!("{}: no MTrk chunks", path.display());
    }
    // [43]
    if h.trailing_bytes > 0 {
        log::warn!("{}: {} bytes after the last chunk are not chunks; ignored", path.display(), h.trailing_bytes);
    }
    if h.tracks.len() != h.declared_tracks as usize {
        log::warn!("{}: header claims {} tracks, found {}", path.display(), h.declared_tracks, h.tracks.len());
    }
    let own = ScanProgress::default();
    let progress = progress.unwrap_or(&own);
    progress
        .bytes_total
        .store(h.tracks.iter().map(|&(_, len)| len).sum(), Ordering::Relaxed);

    // [44]
    let mut order: Vec<usize> = (0..h.tracks.len()).collect();
    order.sort_by_key(|&i| Reverse(h.tracks[i].1));
    let jobs = jobs_for(jobs, order.len());

    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let mut done: Vec<(usize, Result<Scanned>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    while !failed.load(Ordering::Relaxed) && !progress.cancel.load(Ordering::Relaxed) {
                        let Some(&i) = order.get(next.fetch_add(1, Ordering::Relaxed)) else {
                            break;
                        };
                        let (offset, len) = h.tracks[i];
                        let r = scan_track(path, offset, len, progress);
                        if r.is_err() {
                            failed.store(true, Ordering::Relaxed);
                        }
                        mine.push((i, r));
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("a track scan panicked"))
            .collect()
    });
    if progress.cancel.load(Ordering::Relaxed) {
        bail!("{}: track scan cancelled", path.display());
    }
    done.sort_by_key(|&(i, _)| i);

    let mut tracks = Vec::with_capacity(done.len());
    let mut tempo = Vec::new();
    for (i, r) in done {
        let (info, changes) = r.with_context(|| format!("{}: reading track {}", path.display(), i + 1))?;
        tempo.extend(changes);
        tracks.push(info);
    }
    // [45]
    tempo.sort_by_key(|&(tick, _)| tick);

    Ok(TrackScan {
        path: path.to_path_buf(),
        format: h.format,
        division: h.division,
        tracks,
        tempo: tempo.into(),
        header: Arc::new(h),
    })
}

/// One track read: what it holds, and its tempo changes in its own order.
type Scanned = (TrackInfo, Vec<(u64, u32)>);

/// Read one track chunk start to finish.
fn scan_track(path: &Path, offset: u64, len: u64, progress: &ScanProgress) -> Result<Scanned> {
    let mut r = TrackReader::open(path, offset, len)?;
    r.keep_name = true;
    let mut t = TrackInfo {
        name: None,
        by_velocity: [0; 128],
        first_note: None,
        last_note: None,
        channels: 0,
        controls: 0,
        tempos: 0,
        end_tick: 0,
        bytes: len,
    };
    let mut tempo = Vec::new();
    let mut reported = 0u64;
    let mut events = 0u32;
    while let Some(ev) = r.next_event() {
        let tick = r.tick;
        if !matches!(ev, Event::Tempo(_)) {
            t.end_tick = tick;
        }
        match ev {
            Event::NoteOn { ch, vel, .. } => {
                t.by_velocity[vel as usize] += 1;
                t.first_note.get_or_insert(tick);
                t.last_note = Some(tick);
                t.channels |= 1 << (ch & 15);
            }
            Event::NoteOff { .. } => {
                if t.first_note.is_some() {
                    t.last_note = Some(tick);
                }
            }
            Event::Cc { .. }
            | Event::Program { .. }
            | Event::PitchBend { .. }
            | Event::DrumPart { .. }
            | Event::ResetParts => t.controls += 1,
            Event::Tempo(us) => tempo.push((tick, us)),
            Event::Other => {}
        }
        events = events.wrapping_add(1);
        if events & 0xFFFF == 0 {
            let read = len - r.unread();
            progress.bytes_read.fetch_add(read - reported, Ordering::Relaxed);
            reported = read;
            if progress.cancel.load(Ordering::Relaxed) {
                break;
            }
        }
    }
    // [46]
    progress.bytes_read.fetch_add(len - reported, Ordering::Relaxed);
    t.name = r.name.take();
    t.tempos = tempo.len() as u64;
    Ok((t, tempo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::{MidiStream, MidiWriter};

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("kestrel_tracks");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn tempo(tick: u64, us: u32) -> (u64, Vec<u8>) {
        let b = us.to_be_bytes();
        (tick, vec![0xFF, 0x51, 0x03, b[1], b[2], b[3]])
    }

    fn name(tick: u64, bytes: &[u8]) -> (u64, Vec<u8>) {
        let mut m = vec![0xFF, 0x03, bytes.len() as u8];
        m.extend_from_slice(bytes);
        (tick, m)
    }

    /// Five tracks of every kind, with tempo changes on two of them -- one on \[47\]
    fn five_tracks(path: &Path) {
        let mut w = MidiWriter::new(480);
        // 1: a conductor track. Name, tempo, nothing a channel hears.
        w.raw_track(vec![
            name(0, b"Conductor"),
            tempo(0, 400_000),
            tempo(960, 250_000),
            tempo(5000, 2_000_000),
        ]);
        // [48]
        w.raw_track(vec![
            name(0, b"Piano"),
            (0, vec![0xB0, 7, 100]),
            (10, vec![0x90, 60, 1]),
            (20, vec![0x80, 60, 0]),
            (30, vec![0x91, 64, 127]),
            // Running status: a second note-on, then a velocity-0 note-off.
            (30, vec![67, 5]),
            (40, vec![67, 0]),
            (50, vec![0x81, 64, 0]),
            tempo(960, 600_000),
            tempo(3000, 300_000),
            (4000, vec![0x90, 72, 5]),
            (4100, vec![0x80, 72, 0]),
        ]);
        // 3: a setup track, controllers and a program and nothing else.
        w.raw_track(vec![
            (0, vec![0xC0, 5]),
            (0, vec![0xB0, 10, 20]),
            (200, vec![0xE0, 0, 0x50]),
        ]);
        // 4: an empty track with a Shift-JIS name, "piano" in katakana.
        w.raw_track(vec![name(0, &[0x83, 0x73, 0x83, 0x41, 0x83, 0x6D]), (9000, vec![0xFF, 0x01, 0x01, b'x'])]);
        // 5: port B, channel 10. The port is dropped, the channel kept.
        w.raw_track(vec![
            (0, MidiWriter::port_event(1)),
            (100, vec![0x99, 36, 90]),
            (7000, vec![0x89, 36, 0]),
        ]);
        w.save(path).unwrap();
    }

    /// 64 candidates a pool slot, between 2^16 and a 256th of a render's cap.
    #[test]
    fn a_track_gets_candidates_by_its_pool() {
        let at = |v: u32| candidates_each(&crate::config::Config { max_voices: v, ..Default::default() });
        // [49]
        assert_eq!(at(1082), 86_528);
        assert_eq!(at(18), 1 << 16);
        assert_eq!(at(1 << 20), (1 << 27) / 256);
        let small = crate::config::Config { max_block_candidates: 1000, ..Default::default() };
        assert_eq!(candidates_each(&small), 1 << 16);
    }

    /// UTF-8 when it is UTF-8, Shift-JIS when it is not, and the old lossy \[50\]
    #[test]
    fn a_name_is_utf8_or_shift_jis() {
        assert_eq!(decode_name(b"Piano"), "Piano");
        assert_eq!(decode_name("東方ピアノ".as_bytes()), "東方ピアノ");
        // "ピアノ" and "東方" as a Japanese sequencer writes them.
        assert_eq!(decode_name(&[0x83, 0x73, 0x83, 0x41, 0x83, 0x6D]), "ピアノ");
        assert_eq!(decode_name(&[0x93, 0x8C, 0x95, 0xFB, b' ', b'1']), "東方 1");
        // Half-width katakana, one byte each in Shift-JIS.
        assert_eq!(decode_name(&[0xB1, 0xB2, 0xB3]), "ｱｲｳ");
        // A lead byte with nothing after it is neither: replaced, as before.
        assert_eq!(decode_name(&[b'A', 0x83]), "A\u{FFFD}");
        let info = TrackInfo {
            name: Some(vec![b' ', 0x83, 0x73, 0x83, 0x41, 0x83, 0x6D, 0x00]),
            by_velocity: [0; 128],
            first_note: None,
            last_note: None,
            channels: 0,
            controls: 0,
            tempos: 0,
            end_tick: 0,
            bytes: 0,
        };
        assert_eq!(info.display_name().as_deref(), Some("ピアノ"));
    }

    #[test]
    fn a_scan_reads_each_track_for_what_it_holds() {
        let path = temp("five.mid");
        five_tracks(&path);
        let s = scan(&path, 0, None).unwrap();

        assert_eq!(s.tracks.len(), 5);
        let kinds: Vec<_> = s.tracks.iter().map(TrackInfo::kind).collect();
        use TrackKind::*;
        assert_eq!(kinds, [Empty, Notes, Setup, Empty, Notes]);
        assert_eq!(s.of_kind(Setup).collect::<Vec<_>>(), [2]);

        let piano = &s.tracks[1];
        assert_eq!(piano.display_name().as_deref(), Some("Piano"));
        assert_eq!(piano.notes(), 4);
        assert_eq!((piano.by_velocity[1], piano.by_velocity[5], piano.by_velocity[127]), (1, 2, 1));
        assert_eq!(piano.velocity_range(), Some((1, 127)));
        assert_eq!(piano.notes_from(5), 3);
        assert_eq!(piano.notes_from(6), 1);
        assert_eq!((piano.first_note, piano.last_note), (Some(10), Some(4100)));
        assert_eq!(piano.channels, 0b11);
        assert_eq!((piano.controls, piano.tempos), (1, 2));

        assert_eq!(s.tracks[2].controls, 3);
        assert_eq!(s.tracks[2].first_note, None);
        // Kept as bytes, not decoded into something else.
        assert_eq!(s.tracks[3].name.as_deref(), Some(&[0x83, 0x73, 0x83, 0x41, 0x83, 0x6D][..]));
        assert_eq!(s.tracks[3].end_tick, 9000);
        assert_eq!(s.tracks[4].channels, 1 << 9);
        assert_eq!(s.end_tick(), 9000);
        assert_eq!(s.busiest(), Some(1));
        let _ = std::fs::remove_file(&path);
    }

    /// The scan's tempo map and note count are what the merged stream plays, \[51\]
    #[test]
    fn a_scan_agrees_with_the_merged_stream() {
        let path = temp("agree.mid");
        five_tracks(&path);
        let s = scan(&path, 0, None).unwrap();

        let mut m = MidiStream::open(&path).unwrap();
        let mut clock = TempoClock::new(m.division, 48_000);
        let (mut tempo, mut notes, mut ticks, mut frames) = (Vec::new(), 0u64, Vec::new(), Vec::new());
        while let Some((tick, ev)) = m.next() {
            match ev {
                Event::Tempo(us) => {
                    clock.set_tempo(tick, us);
                    tempo.push((tick, us));
                }
                Event::NoteOn { .. } => notes += 1,
                _ => {}
            }
            ticks.push(tick);
            frames.push(clock.frame_at(tick));
        }
        assert_eq!(*s.tempo, tempo[..]);
        assert_eq!(s.tempo[1..3], [(960, 250_000), (960, 600_000)], "same tick: by track");
        assert_eq!(s.notes(), notes);
        assert_eq!(s.frames_at(&ticks, 48_000), frames);
        assert_eq!(s.duration(48_000), frames.last().unwrap() / 48_000.0);
        let _ = std::fs::remove_file(&path);
    }

    /// Forty tracks of assorted lengths, scanned on one thread and on eight, \[52\]
    #[test]
    fn the_scan_is_the_same_on_any_number_of_threads() {
        let path = temp("forty.mid");
        let mut w = MidiWriter::new(960);
        w.tempo_track(500_000);
        for t in 0..40u64 {
            let key = 30 + t as u8;
            let count = 1 + (t * 7919) % 3000;
            let mut ev: Vec<(u64, Vec<u8>)> = (0..count)
                .flat_map(|i| {
                    let at = i * 10 + t;
                    let vel = 1 + ((i + t) % 127) as u8;
                    [(at, vec![0x90 | (t % 16) as u8, key, vel]), (at + 5, vec![0x80 | (t % 16) as u8, key, 0])]
                })
                .collect();
            if t % 5 == 0 {
                ev.push(tempo(t * 100, 300_000 + t as u32 * 1000));
            }
            w.raw_track(ev);
        }
        w.save(&path).unwrap();

        let progress = ScanProgress::default();
        let one = scan(&path, 1, Some(&progress)).unwrap();
        assert_eq!(
            progress.bytes_read.load(Ordering::Relaxed),
            progress.bytes_total.load(Ordering::Relaxed)
        );
        let eight = scan(&path, 8, None).unwrap();
        assert_eq!(one, eight);
        assert_eq!(one.tracks.len(), 41);
        assert_eq!(one.tempo.len(), 9);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_cancelled_scan_says_so() {
        let path = temp("cancel.mid");
        five_tracks(&path);
        let progress = ScanProgress::default();
        progress.cancel.store(true, Ordering::Relaxed);
        let e = scan(&path, 2, Some(&progress)).unwrap_err();
        assert!(format!("{e:#}").contains("cancelled"), "{e:#}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_track_list_reads_as_kestrel_tracks_numbers_them() {
        assert_eq!(TrackList::parse("all").unwrap(), TrackList::All);
        assert_eq!(TrackList::parse(" ALL ").unwrap(), TrackList::All);
        assert_eq!(
            TrackList::parse("1-3, 5,7-").unwrap(),
            TrackList::Pick(vec![(0, Some(2)), (4, Some(4)), (6, None)])
        );
        for (bad, says) in [("0", "counts from 1"), ("3-1", "backwards"), ("x", "not a track"), (" , ", "empty")] {
            let e = format!("{:#}", TrackList::parse(bad).unwrap_err());
            assert!(e.contains(says), "{bad:?}: {e}");
        }

        let path = temp("list.mid");
        five_tracks(&path);
        let s = scan(&path, 0, None).unwrap();
        // Tracks 2 and 5 have notes; 1 and 4 are empty, 3 is a setup track.
        assert_eq!(TrackList::All.resolve(&s).unwrap(), (vec![1, 4], vec![]));
        assert_eq!(TrackList::parse("2-").unwrap().resolve(&s).unwrap(), (vec![1, 4], vec![2, 3]));
        assert_eq!(TrackList::parse("5,2,2").unwrap().resolve(&s).unwrap(), (vec![1, 4], vec![]));
        let e = format!("{:#}", TrackList::parse("4-6").unwrap().resolve(&s).unwrap_err());
        assert!(e.contains("track 6") && e.contains("has 5"), "{e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_stem_is_named_by_number_then_name_and_sorts_in_track_order() {
        assert_eq!(stem_file_name(2, 30, Some("Piano"), "flac"), "03 Piano.flac");
        assert_eq!(stem_file_name(2, 4393, Some("Piano"), "wav"), "0003 Piano.wav");
        assert_eq!(stem_file_name(8, 9, None, "wav"), "9.wav");
        assert_eq!(stem_file_name(0, 100, Some("  "), "wav"), "001.wav");
        // Windows refuses these, and drops a trailing dot or space.
        assert_eq!(stem_file_name(0, 10, Some("a/b:c*?\"<>|d. "), "wav"), "01 a_b_c______d.wav");
        let long = "x".repeat(200);
        assert_eq!(stem_file_name(0, 10, Some(&long), "wav").len(), "01 ".len() + 80 + ".wav".len());
        assert_eq!(voices_each(1_048_576, 34), 30_840);
        assert_eq!(voices_each(60_000_000, 243), 246_913);
        assert_eq!(voices_each(3, 10), 1);
    }

    /// A name longer than the reader keeps is cut, and the events after it \[53\]
    #[test]
    fn a_long_name_is_cut_and_the_track_still_reads() {
        let path = temp("longname.mid");
        let long = vec![b'n'; 300];
        let mut m = vec![0xFF, 0x03, 0x82, 0x2C]; // varlen 300
        m.extend_from_slice(&long);
        let mut w = MidiWriter::new(480);
        w.raw_track(vec![(0, m), (10, vec![0x90, 60, 64]), (20, vec![0x80, 60, 0])]);
        w.save(&path).unwrap();
        let s = scan(&path, 0, None).unwrap();
        assert_eq!(s.tracks[0].name.as_ref().map(Vec::len), Some(255));
        assert_eq!(s.tracks[0].notes(), 1);
        let _ = std::fs::remove_file(&path);
    }
}
