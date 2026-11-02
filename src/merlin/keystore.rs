// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The file split once by key: what lets merlin's per-key render stop reading \[1\]

use anyhow::Result;
use rayon::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

/// MIDI keys.
const KEYS: usize = 128;
/// What a worker holds for a key before it writes a chunk.
const CHUNK_BYTES: usize = 64 << 10;
/// `Config::merlin_disk_floor`'s default: what the rest of the machine and the \[2\]
pub const DISK_FLOOR: u64 = 8 << 30;
/// Bytes of store to allow for a byte of MIDI file: 0.23 to 0.57 measured on \[3\]
const STORE_SHARE: f64 = 0.65;
/// How often the store looks at what is left of its disk.
const WATCH_BYTES: u64 = 256 << 20;

/// One note event as a stream holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ev {
    pub tick: u64,
    /// Controllers before it in its track: all its order against them needs.
    pub seq: u64,
    pub ch: u8,
    /// Zero for a note-off.
    pub vel: u8,
    /// Identical events back to back in the key's stream.
    pub count: u32,
}

/// What an event is written relative to.
#[derive(Clone, Copy, Default)]
struct State {
    tick: u64,
    seq: u64,
    ch: u8,
}

const HAS_COUNT: u8 = 1;
const HAS_CH: u8 = 2;
const HAS_SEQ: u8 = 4;

fn put_var(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_var(b: &[u8], p: &mut usize) -> u64 {
    let (mut v, mut shift) = (0u64, 0);
    loop {
        let x = b[*p];
        *p += 1;
        v |= ((x & 0x7F) as u64) << shift;
        if x < 0x80 {
            return v;
        }
        shift += 7;
    }
}

/// The velocity, its top bit saying a byte of flags follows for what is rare; \[4\]
fn encode(out: &mut Vec<u8>, st: &mut State, e: &Ev) {
    let ds = e.seq - st.seq;
    let rare = if e.count != 1 { HAS_COUNT } else { 0 } | if e.ch != st.ch { HAS_CH } else { 0 } | if ds != 0 { HAS_SEQ } else { 0 };
    out.push(e.vel | if rare != 0 { 0x80 } else { 0 });
    if rare != 0 {
        out.push(rare);
        if rare & HAS_COUNT != 0 {
            put_var(out, e.count as u64);
        }
        if rare & HAS_CH != 0 {
            out.push(e.ch);
        }
        if rare & HAS_SEQ != 0 {
            put_var(out, ds);
        }
    }
    put_var(out, e.tick - st.tick);
    *st = State { tick: e.tick, seq: e.seq, ch: e.ch };
}

fn decode(b: &[u8], p: &mut usize, st: &mut State) -> Ev {
    let v = b[*p];
    *p += 1;
    let rare = if v & 0x80 != 0 {
        *p += 1;
        b[*p - 1]
    } else {
        0
    };
    let count = if rare & HAS_COUNT != 0 { get_var(b, p) as u32 } else { 1 };
    if rare & HAS_CH != 0 {
        st.ch = b[*p];
        *p += 1;
    }
    if rare & HAS_SEQ != 0 {
        st.seq += get_var(b, p);
    }
    st.tick += get_var(b, p);
    Ev { tick: st.tick, seq: st.seq, ch: st.ch, vel: v & 0x7F, count }
}

pub(crate) fn temp_file(dir: &Path) -> std::io::Result<File> {
    open_temp(&temp_path(dir))
}

fn temp_path(dir: &Path) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    dir.join(format!("kestrel-keys-{}-{nanos}.bin", std::process::id()))
}

fn open_temp(path: &Path) -> std::io::Result<File> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true).write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_ATTRIBUTE_TEMPORARY and FILE_FLAG_DELETE_ON_CLOSE.
        o.attributes(0x100).custom_flags(0x0400_0000);
    }
    let f = o.open(path)?;
    #[cfg(not(windows))]
    std::fs::remove_file(path)?;
    Ok(f)
}

/// A handle on a store for each of rayon's workers. Windows runs the reads on \[5\]
fn reader_handles(path: &Path) -> Vec<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_RANDOM_ACCESS: a key's chunks lie all over the file.
        let open = |_| std::fs::OpenOptions::new().read(true).custom_flags(0x1000_0000).open(path);
        (0..rayon::current_num_threads()).map_while(|i| open(i).ok()).collect()
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Vec::new()
    }
}

pub(crate) fn write_all_at(file: &File, mut buf: &[u8], mut at: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_write(file, buf, at)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::write_at(file, buf, at)?;
        if n == 0 {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        buf = &buf[n..];
        at += n as u64;
    }
    Ok(())
}

/// A chunk of one key's stream: where it is, and its segments' `(track, first \[6\]
struct Chunk {
    off: u64,
    len: u32,
    segs: Box<[(u32, u32)]>,
}

struct Shared {
    file: File,
    next: AtomicU64,
    chunks: Vec<Mutex<Vec<Chunk>>>,
    failed: AtomicBool,
    /// Its disk, and what to leave free on it.
    dir: PathBuf,
    floor: u64,
    /// Where it is, to read it by.
    path: Option<PathBuf>,
}

impl Shared {
    /// True, once, as the store has to stop: the disk is down to the floor.
    fn out_of_room(&self) -> bool {
        match fs4::available_space(&self.dir) {
            Ok(free) if free < self.floor => {
                log::warn!(
                    "merlin's key store stopped at {:.1} GB with {:.1} GB left on its disk ({}), falling back to reading the file a group at a time, which is much slower on a big file",
                    self.next.load(Ordering::Relaxed) as f64 / 1e9,
                    free as f64 / 1e9,
                    self.dir.display()
                );
                true
            }
            _ => false,
        }
    }
}

/// About what a MIDI file of `midi_bytes` takes as a store, on the high side.
pub fn store_bytes(midi_bytes: u64) -> u64 {
    (midi_bytes as f64 * STORE_SHARE) as u64
}

/// Where the key store can go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    /// The temporary folder has the room, or cannot say.
    Temp,
    /// Not the temporary folder, but the output's does.
    Output,
    /// Neither: the file is read a group of keys at a time instead, which is \[7\]
    Neither,
}

/// Where a store needing `need` bytes, with `floor` left free, goes, given the \[8\]
pub fn site(need: u64, floor: u64, temp_free: Option<u64>, out_free: Option<u64>) -> Site {
    let need = need.saturating_add(floor);
    if temp_free.is_none_or(|free| free >= need) {
        Site::Temp
    } else if out_free.is_some_and(|free| free >= need) {
        Site::Output
    } else {
        Site::Neither
    }
}

/// The folder the output goes in, which has to have room for a render anyway.
pub fn output_folder(out: &Path) -> &Path {
    out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
}

/// Where a render's temporary files go: the temporary folder, or, without \[9\]
pub fn filing_dir(midi: &Path, out: &Path, floor: u64) -> PathBuf {
    let temp = std::env::temp_dir();
    let need = std::fs::metadata(midi).map_or(0, |m| store_bytes(m.len()));
    let folder = output_folder(out);
    match site(need, floor, fs4::available_space(&temp).ok(), fs4::available_space(folder).ok()) {
        Site::Output => {
            log::info!("merlin's key store needs about {:.1} GB, more than {} has free, so it goes in {}", need.saturating_add(floor) as f64 / 1e9, temp.display(), folder.display());
            folder.to_path_buf()
        }
        Site::Temp | Site::Neither => temp,
    }
}

#[derive(Default)]
struct KeyBuf {
    bytes: Vec<u8>,
    segs: Vec<(u32, u32)>,
    /// A segment of the current track is open in `bytes`.
    open: bool,
    st: State,
    /// The last event, which the next may fold into.
    pend: Option<Ev>,
    /// Events written. Counted here because a counter every worker shares, \[10\]
    events: u64,
}

impl KeyBuf {
    fn flush(&mut self, sh: &Shared, key: usize) {
        self.open = false;
        if self.bytes.is_empty() {
            return;
        }
        if !sh.failed.load(Ordering::Relaxed) {
            let len = self.bytes.len() as u64;
            let off = sh.next.fetch_add(len, Ordering::Relaxed);
            if (off == 0 || (off + len) / WATCH_BYTES != off / WATCH_BYTES) && sh.out_of_room() {
                sh.failed.store(true, Ordering::Relaxed);
                self.bytes.clear();
                self.segs.clear();
                return;
            }
            match write_all_at(&sh.file, &self.bytes, off) {
                Ok(()) => {
                    let segs = std::mem::take(&mut self.segs).into_boxed_slice();
                    sh.chunks[key].lock().expect("no writer panics").push(Chunk { off, len: self.bytes.len() as u32, segs });
                }
                Err(e) => {
                    log::warn!("merlin's key store could not be written, falling back to reading the file a group at a time: {e}");
                    sh.failed.store(true, Ordering::Relaxed);
                }
            }
        }
        self.bytes.clear();
        self.segs.clear();
    }

    fn commit(&mut self, sh: &Shared, key: usize, track: u32, e: &Ev) {
        if !self.open {
            self.segs.push((track, self.bytes.len() as u32));
            self.st = State::default();
            self.open = true;
        }
        encode(&mut self.bytes, &mut self.st, e);
        self.events += 1;
        if self.bytes.len() >= CHUNK_BYTES {
            self.flush(sh, key);
        }
    }
}

/// A worker's buffers, one per key.
struct Filer {
    keys: Vec<KeyBuf>,
    /// Keys with an event of the current track.
    touched: Vec<u8>,
}

/// The file being written, and a set of buffers for each worker. A set is \[11\]
pub struct Filing {
    sh: Shared,
    filers: Vec<Mutex<Filer>>,
}

impl Filing {
    /// A store in `dir`, which leaves `floor` bytes free on its disk.
    pub fn new(dir: &Path, floor: u64) -> std::io::Result<Self> {
        let path = temp_path(dir);
        Self::on(open_temp(&path)?, dir, floor, Some(path))
    }

    fn on(file: File, dir: &Path, floor: u64, path: Option<PathBuf>) -> std::io::Result<Self> {
        let sh = Shared {
            file,
            next: AtomicU64::new(0),
            chunks: (0..KEYS).map(|_| Mutex::new(Vec::new())).collect(),
            failed: AtomicBool::new(false),
            dir: dir.to_path_buf(),
            floor,
            path,
        };
        let filers = (0..rayon::current_num_threads() + 1)
            .map(|_| Mutex::new(Filer { keys: (0..KEYS).map(|_| KeyBuf::default()).collect(), touched: Vec::new() }))
            .collect();
        Ok(Self { sh, filers })
    }

    /// A worker's buffers for `track`.
    pub fn track(&self, track: u32) -> TrackFiler<'_> {
        let i = rayon::current_thread_index().unwrap_or(self.filers.len() - 1) % self.filers.len();
        let f = (0..self.filers.len())
            .find_map(|d| self.filers[(i + d) % self.filers.len()].try_lock().ok())
            .unwrap_or_else(|| self.filers[i].lock().expect("no writer panics"));
        TrackFiler { sh: &self.sh, f, track }
    }

    /// Everything written, or `None` if any of it could not be.
    pub fn finish(self) -> Option<KeyStore> {
        let Self { sh, filers } = self;
        let mut events = 0;
        for f in filers {
            let f = f.into_inner().expect("no writer panics");
            for (k, mut b) in f.keys.into_iter().enumerate() {
                b.flush(&sh, k);
                events += b.events;
            }
        }
        if sh.failed.load(Ordering::Relaxed) {
            return None;
        }
        let Shared { file, next, chunks, path, .. } = sh;
        Some(KeyStore {
            file,
            readers: path.as_deref().map_or_else(Vec::new, reader_handles),
            bytes: next.into_inner(),
            events,
            chunks: chunks.into_iter().map(|c| c.into_inner().expect("no writer panics")).collect(),
        })
    }
}

/// One track's note events on their way into their keys' streams.
pub struct TrackFiler<'a> {
    sh: &'a Shared,
    f: MutexGuard<'a, Filer>,
    track: u32,
}

impl TrackFiler<'_> {
    pub fn note(&mut self, key: u8, e: Ev) {
        let k = key as usize & 127;
        let Filer { keys, touched } = &mut *self.f;
        let b = &mut keys[k];
        match b.pend {
            Some(p) if p.tick == e.tick && p.seq == e.seq && p.ch == e.ch && p.vel == e.vel && p.count < u32::MAX => {
                b.pend = Some(Ev { count: p.count + 1, ..p });
                return;
            }
            Some(p) => b.commit(self.sh, k, self.track, &p),
            None => touched.push(k as u8),
        }
        b.pend = Some(e);
    }

    /// The track's last events into their streams.
    pub fn end(mut self) {
        let Filer { keys, touched } = &mut *self.f;
        for k in touched.drain(..) {
            let k = k as usize;
            let b = &mut keys[k];
            if let Some(p) = b.pend.take() {
                b.commit(self.sh, k, self.track, &p);
            }
            b.open = false;
        }
    }
}

/// Every key's stream, in the temporary file.
pub struct KeyStore {
    file: File,
    readers: Vec<File>,
    pub bytes: u64,
    /// Events written, identical repeats counted once.
    pub events: u64,
    chunks: Vec<Vec<Chunk>>,
}

/// A walk along one key's stream in one track.
pub struct Cursor {
    pub key: u8,
    /// `[lo, hi)` in `Loaded::bytes`, in stream order.
    segs: Vec<(usize, usize)>,
    i: usize,
    pos: usize,
    st: State,
}

impl Cursor {
    pub fn next_event(&mut self, bytes: &[u8]) -> Option<Ev> {
        loop {
            let &(lo, hi) = self.segs.get(self.i)?;
            let p = self.pos.max(lo);
            if p < hi {
                let mut p = p;
                let e = decode(bytes, &mut p, &mut self.st);
                self.pos = p;
                return Some(e);
            }
            self.i += 1;
            self.st = State::default();
        }
    }
}

/// Some keys' streams in memory: their tracks in order, each with a cursor per \[12\]
pub struct Loaded {
    pub bytes: Vec<u8>,
    pub tracks: Vec<(u32, Vec<Cursor>)>,
}

impl KeyStore {
    /// `keys`' streams read back.
    pub fn load(&self, keys: &[u8]) -> Result<Loaded> {
        let total: usize = keys.iter().flat_map(|&k| &self.chunks[k as usize & 127]).map(|c| c.len as usize).sum();
        let mut bytes = vec![0u8; total];
        let mut rest = &mut bytes[..];
        let mut at = 0usize;
        let mut jobs: Vec<(&mut [u8], u64)> = Vec::new();
        let mut segs: Vec<(u32, u8, usize, usize)> = Vec::new();
        for &key in keys {
            for c in &self.chunks[key as usize & 127] {
                let (mine, tail) = rest.split_at_mut(c.len as usize);
                rest = tail;
                jobs.push((mine, c.off));
                for (i, &(track, lo)) in c.segs.iter().enumerate() {
                    let hi = c.segs.get(i + 1).map_or(c.len, |s| s.1);
                    segs.push((track, key, at + lo as usize, at + hi as usize));
                }
                at += c.len as usize;
            }
        }
        jobs.into_par_iter().try_for_each(|(buf, off)| {
            let mine = rayon::current_thread_index().and_then(|i| self.readers.get(i));
            crate::merlin::bucket::read_exact_at(mine.unwrap_or(&self.file), buf, off)
        })?;
        // Stable, so a track's segments of a key stay in stream order.
        segs.sort_by_key(|s| (s.0, s.1));
        let mut tracks: Vec<(u32, Vec<Cursor>)> = Vec::new();
        for (track, key, lo, hi) in segs {
            if tracks.last().is_none_or(|t| t.0 != track) {
                tracks.push((track, Vec::new()));
            }
            let t = &mut tracks.last_mut().expect("pushed").1;
            match t.last_mut() {
                Some(c) if c.key == key => c.segs.push((lo, hi)),
                _ => t.push(Cursor { key, segs: vec![(lo, hi)], i: 0, pos: 0, st: State::default() }),
            }
        }
        Ok(Loaded { bytes, tracks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_read_back_as_they_were_written() {
        let evs = [
            Ev { tick: 0, seq: 0, ch: 0, vel: 100, count: 1 },
            Ev { tick: 1, seq: 0, ch: 0, vel: 0, count: 1 },
            Ev { tick: 1, seq: 3, ch: 200, vel: 1, count: 70_000 },
            Ev { tick: 1 << 40, seq: 1 << 33, ch: 200, vel: 127, count: u32::MAX },
        ];
        let mut out = Vec::new();
        let mut st = State::default();
        for e in &evs {
            encode(&mut out, &mut st, e);
        }
        let (mut p, mut st) = (0, State::default());
        for e in &evs {
            assert_eq!(decode(&out, &mut p, &mut st), *e);
        }
        assert_eq!(p, out.len());
    }

    /// Past several chunks on every key, tracks written side by side, and each \[13\]
    fn file_a_chunk(filing: &Filing) {
        let mut t = filing.track(0);
        for i in 0..CHUNK_BYTES as u64 {
            t.note(60, Ev { tick: i * 1000, seq: 0, ch: 0, vel: 64, count: 1 });
        }
        t.end();
    }

    #[test]
    fn the_store_goes_where_there_is_room_for_it() {
        const GB: u64 = 1_000_000_000;
        // 100 GB and an 8 GB floor: 108 GB wanted.
        assert_eq!(site(100 * GB, 8 * GB, Some(108 * GB), Some(0)), Site::Temp, "exactly enough");
        assert_eq!(site(100 * GB, 8 * GB, Some(108 * GB - 1), Some(500 * GB)), Site::Output);
        assert_eq!(site(100 * GB, 8 * GB, Some(10 * GB), Some(107 * GB)), Site::Neither);
        assert_eq!(site(100 * GB, 8 * GB, Some(10 * GB), None), Site::Neither, "an output folder that cannot say has no room to promise");
        assert_eq!(site(100 * GB, 8 * GB, None, Some(0)), Site::Temp, "a temporary folder that cannot say is taken at its word");
        assert_eq!(site(0, 0, Some(0), Some(0)), Site::Temp);
        assert_eq!(output_folder(Path::new("song.wav")), Path::new("."));
        assert_eq!(output_folder(Path::new("out/song.wav")), Path::new("out"));
    }

    #[test]
    fn a_store_stops_at_its_floor() {
        let filing = Filing::new(&std::env::temp_dir(), u64::MAX).unwrap();
        file_a_chunk(&filing);
        assert!(filing.finish().is_none(), "a store below its floor went on");
    }

    /// A real refusal from the operating system, which is how a full disk \[14\]
    #[test]
    fn a_store_that_cannot_be_written_is_dropped() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("kestrel-readonly-{}.bin", std::process::id()));
        std::fs::write(&path, b"").unwrap();
        let file = File::open(&path).unwrap();
        let filing = Filing::on(file, &dir, 0, None).unwrap();
        file_a_chunk(&filing);
        assert!(filing.finish().is_none(), "a failed write went unnoticed");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn streams_come_back_by_track_and_key_through_chunks() {
        let dir = std::env::temp_dir();
        let filing = Filing::new(&dir, 0).unwrap();
        (0..8u32).into_par_iter().for_each(|track| {
            let mut f = filing.track(track);
            for i in 0..100_000u64 {
                let (key, tick) = ((i % 3) as u8 * 40, i / 2 * 1000 + track as u64);
                f.note(key, Ev { tick, seq: i / 7, ch: track as u8, vel: (i % 100) as u8 + 1, count: 1 });
                if i % 10 == 0 {
                    f.note(key, Ev { tick, seq: i / 7, ch: track as u8, vel: (i % 100) as u8 + 1, count: 1 });
                }
            }
            f.end();
        });
        let store = filing.finish().unwrap();
        assert_eq!(store.events, 800_000);
        let mut l = store.load(&[0, 40, 80]).unwrap();
        assert_eq!(l.tracks.iter().map(|t| t.0).collect::<Vec<_>>(), (0..8).collect::<Vec<_>>());
        for (track, cursors) in &mut l.tracks {
            assert_eq!(cursors.len(), 3);
            let mut got = Vec::new();
            for c in cursors.iter_mut() {
                let k = c.key as u64 / 40;
                let mut want = (0..100_000u64).filter(|i| i % 3 == k);
                while let Some(e) = c.next_event(&l.bytes) {
                    let i = want.next().expect("as many as went in");
                    assert_eq!(e.tick, i / 2 * 1000 + *track as u64);
                    assert_eq!(e.count, if i % 10 == 0 { 2 } else { 1 });
                    got.push(e);
                }
                assert!(want.next().is_none(), "track {track} key {} came back short", c.key);
            }
            assert_eq!(got.len(), 100_000);
        }
    }

    #[test]
    fn a_file_that_cannot_be_written_gives_no_store() {
        let filing = Filing::new(&std::env::temp_dir(), 0).unwrap();
        filing.sh.failed.store(true, Ordering::Relaxed);
        let mut f = filing.track(0);
        f.note(60, Ev { tick: 0, seq: 0, ch: 0, vel: 1, count: 1 });
        f.end();
        assert!(filing.finish().is_none());
    }
}
