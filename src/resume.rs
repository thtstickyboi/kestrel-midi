// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Resuming a per-track render: the checkpoint file, `.krsm`, and what makes \[1\]

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Version of the header's layout and the payload's. A file with another is \[2\]
pub const FORMAT: u32 = 1;

const MAGIC: &[u8; 8] = b"KRSMCHK\x01";

/// The default insurance interval for a per-track render, in minutes: how \[3\]
pub const DEFAULT_EVERY_MINUTES: u64 = 10;

// ---- the hash --------------------------------------------------------------

const K1: u64 = 0x9E37_79B9_7F4A_7C15;
const K2: u64 = 0xBF58_476D_1CE4_E5B9;

/// A 64-bit hash that takes eight bytes at a time, for telling a file that was \[4\]
#[derive(Clone)]
pub struct Hash64 {
    h: u64,
    tail: [u8; 8],
    n: usize,
    len: u64,
}

impl Default for Hash64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Hash64 {
    pub fn new() -> Self {
        Hash64 { h: 0x243F_6A88_85A3_08D3, tail: [0; 8], n: 0, len: 0 }
    }

    fn mix(&mut self, w: u64) {
        self.h = (self.h ^ w).wrapping_mul(K1);
        self.h ^= self.h >> 29;
        self.h = self.h.wrapping_mul(K2);
        self.h ^= self.h >> 32;
    }

    pub fn update(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len() as u64;
        while self.n > 0 && !bytes.is_empty() {
            self.tail[self.n] = bytes[0];
            self.n += 1;
            bytes = &bytes[1..];
            if self.n == 8 {
                let w = u64::from_le_bytes(self.tail);
                self.mix(w);
                self.n = 0;
            }
        }
        if self.n > 0 {
            // The bytes ran out inside a word still being filled.
            return;
        }
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.mix(u64::from_le_bytes(c.try_into().expect("chunks of eight")));
        }
        let rest = chunks.remainder();
        self.tail[..rest.len()].copy_from_slice(rest);
        self.n = rest.len();
    }

    pub fn finish(mut self) -> u64 {
        let mut last = [0u8; 8];
        last[..self.n].copy_from_slice(&self.tail[..self.n]);
        let n = self.n as u64;
        self.mix(u64::from_le_bytes(last) ^ (n << 56));
        let len = self.len;
        self.mix(len);
        self.h
    }
}

pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = Hash64::new();
    h.update(bytes);
    h.finish()
}

// ---- who made it -----------------------------------------------------------

/// The build that is running: the version, and a hash of the executable. \[5\]
pub fn build_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let hash = std::env::current_exe().ok().and_then(|p| std::fs::read(p).ok()).map(|b| hash_bytes(&b));
        match hash {
            Some(h) => format!("{}-{h:016x}", env!("CARGO_PKG_VERSION")),
            None => format!("{}-unknown", env!("CARGO_PKG_VERSION")),
        }
    })
}

/// How much of each end of a file is hashed: enough to see a re-export, and \[6\]
const ENDS: u64 = 1 << 20;

/// A file as it was, close enough to know whether it is the same file. \[7\]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub path: String,
    pub size: u64,
    pub modified: u64,
    pub head: u64,
    pub tail: u64,
}

impl Fingerprint {
    pub fn of(path: &Path) -> Result<Self> {
        let mut f = File::open(path).with_context(|| format!("reading {}", path.display()))?;
        let meta = f.metadata()?;
        let size = meta.len();
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let mut buf = vec![0u8; ENDS.min(size) as usize];
        f.read_exact(&mut buf)?;
        let head = hash_bytes(&buf);
        let tail = if size > ENDS {
            f.seek(SeekFrom::Start(size - ENDS))?;
            f.read_exact(&mut buf)?;
            hash_bytes(&buf)
        } else {
            head
        };
        Ok(Fingerprint { path: path.display().to_string(), size, modified, head, tail })
    }

    /// What is different about `now`, or `None` if it is the same file.
    fn difference(&self, now: &Fingerprint, what: &str) -> Option<String> {
        if self.size != now.size {
            Some(format!("{what} {} was {} bytes and is now {}", now.path, self.size, now.size))
        } else if self.head != now.head || self.tail != now.tail {
            Some(format!("{what} {} has different contents", now.path))
        } else {
            None
        }
    }
}

/// A soundfont: its own file, and for an SFZ the sample files it names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoundfontPrint {
    pub file: Fingerprint,
    /// Hash of the name and size of each sample file, sorted.
    pub samples: u64,
    pub sample_files: usize,
}

impl SoundfontPrint {
    pub fn of(path: &Path) -> Result<Self> {
        let file = Fingerprint::of(path)?;
        let is_sfz = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("sfz"));
        let (samples, sample_files) = if is_sfz {
            let mut named: Vec<(String, u64)> = crate::sfz::sample_files(path)?
                .into_iter()
                .map(|p| {
                    let size = std::fs::metadata(&p).map_or(0, |m| m.len());
                    (p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), size)
                })
                .collect();
            named.sort();
            let mut h = Hash64::new();
            for (name, size) in &named {
                h.update(name.as_bytes());
                h.update(&size.to_le_bytes());
            }
            (h.finish(), named.len())
        } else {
            (0, 0)
        };
        Ok(SoundfontPrint { file, samples, sample_files })
    }

    fn difference(&self, now: &SoundfontPrint) -> Option<String> {
        self.file.difference(&now.file, "the soundfont").or_else(|| {
            (self.samples != now.samples || self.sample_files != now.sample_files)
                .then(|| format!("the sample files of {} have changed", now.file.path))
        })
    }
}

// ---- the header ------------------------------------------------------------

/// What a track had come to when its file was finished or its blocks were all \[8\]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub bytes: u64,
    pub audio_secs: f64,
    pub notes: u64,
    pub notes_skipped: u64,
    pub voices_spawned: u64,
    pub peak_voices: u64,
    pub stolen: u64,
    pub dropped: u64,
    pub peak_level: f32,
    pub clipped: u64,
    pub silent_blocks: u64,
    pub blocks: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finished {
    pub track: usize,
    pub stats: Stats,
}

/// A track that had begun: `blocks` of it were in the mix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InFlight {
    pub track: usize,
    pub blocks: u64,
}

/// What kind of audio file a single render's checkpoint stands behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioKind {
    /// The WAV being written, itself: its first `bytes` of audio data are the \[9\]
    Wav,
    /// The float samples an encoder is being fed, kept in a file of their own, \[10\]
    Pcm,
}

/// The audio a single render had written when its checkpoint was made.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Audio {
    pub kind: AudioKind,
    pub path: String,
    /// Bytes of audio, counted from the start of the data: after the 44-byte \[11\]
    pub bytes: u64,
    /// Hash of the last mebibyte of those bytes, to tell that the file is the \[12\]
    pub tail: u64,
}

/// The size of a canonical WAV header, which `wav::WavWriter` writes.
pub const WAV_HEADER: u64 = 44;

impl Audio {
    fn start(&self) -> u64 {
        match self.kind {
            AudioKind::Wav => WAV_HEADER,
            AudioKind::Pcm => 0,
        }
    }

    /// The hash of the last `ENDS` bytes of the `bytes` that begin at `start`.
    pub fn tail_of(path: &Path, start: u64, bytes: u64) -> Result<u64> {
        let mut f = File::open(path).with_context(|| format!("reading {}", path.display()))?;
        let n = bytes.min(ENDS);
        let mut buf = vec![0u8; n as usize];
        f.seek(SeekFrom::Start(start + bytes - n))?;
        f.read_exact(&mut buf)
            .with_context(|| format!("{} is shorter than the checkpoint says it was", path.display()))?;
        Ok(hash_bytes(&buf))
    }

    /// The file is there, is as long as the checkpoint says at least, and ends \[13\]
    pub fn check(&self) -> Result<()> {
        let path = Path::new(&self.path);
        let len = std::fs::metadata(path)
            .with_context(|| format!("the audio rendered so far, {}, is not there", path.display()))?
            .len();
        if len < self.start() + self.bytes {
            bail!(
                "the audio rendered so far, {}, is {len} bytes, and the checkpoint was made when it held {}",
                path.display(),
                self.start() + self.bytes
            );
        }
        if Audio::tail_of(path, self.start(), self.bytes)? != self.tail {
            bail!("the audio rendered so far, {}, is not the file this checkpoint was made beside", path.display());
        }
        Ok(())
    }
}

/// What a checkpoint of a single render holds beyond what every one does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Single {
    /// Blocks rendered and written.
    pub blocks: u64,
    /// The most voices that were alive after any block, which only the render \[14\]
    pub peak_voices: u64,
    pub audio: Audio,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub format: u32,
    pub kestrel: String,
    pub build: String,
    /// The command as it was typed, which a resume parses again through the same \[15\]
    pub argv: Vec<String>,
    /// `cpu`, or `gpu: <adapter>`: DX12 and Vulkan write different bytes for the \[16\]
    pub backend: String,
    pub midi: Fingerprint,
    pub soundfonts: Vec<SoundfontPrint>,
    pub merge: bool,
    /// The tracks being rendered, in track order.
    pub tracks: Vec<usize>,
    /// The voice limit each track ran under, after it was held to the card.
    pub voices_each: u32,
    pub block_samples: usize,
    pub seconds: Option<f64>,
    pub finished: Vec<Finished>,
    pub in_flight: Vec<InFlight>,
    /// Blocks the mix runs for, silent ones included. Merge only.
    pub mix_len: u64,
    /// Set for a render of one file or one track, which saves the render itself \[17\]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub single: Option<Single>,
}

/// What a resume is checked against, worked out from the render that is about to \[18\]
pub struct Identity {
    pub midi: Fingerprint,
    pub soundfonts: Vec<SoundfontPrint>,
    pub merge: bool,
    pub tracks: Vec<usize>,
    pub block_samples: usize,
    pub seconds: Option<f64>,
    /// A render of one file, not one track at a time.
    pub single: bool,
}

impl Header {
    /// Every way this checkpoint does not fit the render about to run, said in \[19\]
    pub fn check(&self, now: &Identity) -> Result<()> {
        let mut why: Vec<String> = Vec::new();
        if self.format != FORMAT {
            why.push(format!("it is checkpoint format {}, and this build reads {FORMAT}", self.format));
        }
        if self.build != build_id() {
            why.push(format!(
                "it was made by build {}, and this is build {}: a track rendered by another build is not \
                 guaranteed to be the same bytes, so the render has to start over",
                self.build,
                build_id()
            ));
        }
        why.extend(self.midi.difference(&now.midi, "the MIDI"));
        if self.soundfonts.len() != now.soundfonts.len() {
            why.push(format!(
                "it used {} soundfont(s) and this render names {}",
                self.soundfonts.len(),
                now.soundfonts.len()
            ));
        } else {
            why.extend(self.soundfonts.iter().zip(&now.soundfonts).filter_map(|(a, b)| a.difference(b)));
        }
        if self.single.is_some() != now.single {
            why.push(if self.single.is_some() {
                "it was a render of the whole file, not of its tracks one by one".into()
            } else {
                "it was a render of the tracks one by one, not of the whole file".to_string()
            });
        } else if self.merge != now.merge {
            why.push(if self.merge { "it was a merged render".into() } else { "it was a render to stems".to_string() });
        }
        if self.tracks != now.tracks {
            why.push(format!(
                "it rendered {} track(s) and this render has {}, or different ones",
                self.tracks.len(),
                now.tracks.len()
            ));
        }
        if self.block_samples != now.block_samples {
            why.push("the block size is different".into());
        }
        if self.seconds != now.seconds {
            why.push("the length (--seconds) is different".into());
        }
        if why.is_empty() {
            Ok(())
        } else {
            bail!("this checkpoint cannot be resumed: {}", why.join("; "))
        }
    }
}

// ---- what a render is told ---------------------------------------------------

/// What a per-track render is told about checkpoints: where, how often, the \[20\]
#[derive(Debug, Clone)]
pub struct Spec {
    /// The `.krsm` to write, and to delete when the render is whole. `None` \[21\]
    pub path: Option<PathBuf>,
    /// How often a running merge saves its progress; `None` only on a stop \[22\]
    pub every: Option<Duration>,
    /// The command, as `Header::argv` keeps it.
    pub argv: Vec<String>,
    /// A checkpoint to continue from.
    pub restore: Option<Arc<Checkpoint>>,
    /// For tests and for trying a resume by hand: stop as if cancelled once this \[23\]
    pub stop_after_blocks: Option<u64>,
}

impl Spec {
    /// Checkpoints on, beside the output, at the default interval.
    pub fn new(argv: Vec<String>) -> Spec {
        Spec {
            path: None,
            every: Some(Duration::from_secs(DEFAULT_EVERY_MINUTES * 60)),
            argv,
            restore: None,
            stop_after_blocks: None,
        }
    }
}

impl PartialEq for Spec {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.every == other.every
            && self.argv == other.argv
            && self.stop_after_blocks == other.stop_after_blocks
            && match (&self.restore, &other.restore) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

impl Eq for Spec {}

/// Where a per-track render's checkpoint is by default: `<file>.krsm` beside a \[24\]
pub fn default_path(out: &Path, midi: &Path, merged: bool) -> PathBuf {
    if merged {
        let mut p = out.as_os_str().to_owned();
        p.push(".krsm");
        PathBuf::from(p)
    } else {
        let folder = midi.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "stems".into());
        out.join(folder).join("stems.krsm")
    }
}

/// `song.wav` -> `song.partial.wav`: the name a file has while it is being \[25\]
pub fn partial_of(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let named = match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => format!("{stem}.partial.{ext}"),
        _ => format!("{name}.partial"),
    };
    path.with_file_name(named)
}

// ---- the container ---------------------------------------------------------

struct HashingWriter<W: Write> {
    inner: W,
    hash: Hash64,
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hash.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// A checkpoint being written. Nothing is at its real name until `commit`.
pub struct Writer {
    out: HashingWriter<BufWriter<File>>,
    tmp: PathBuf,
    dest: PathBuf,
}

impl Writer {
    pub fn create(dest: &Path, header: &Header) -> Result<Writer> {
        let tmp = {
            let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
            name.push(".tmp");
            dest.with_file_name(name)
        };
        let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let mut out = HashingWriter { inner: BufWriter::with_capacity(1 << 20, file), hash: Hash64::new() };
        let json = serde_json::to_vec(header)?;
        out.write_all(MAGIC)?;
        out.write_all(&(json.len() as u64).to_le_bytes())?;
        out.write_all(&json)?;
        Ok(Writer { out, tmp, dest: dest.to_path_buf() })
    }

    /// The hash, the sync and the rename. Returns the size of the file.
    pub fn commit(self) -> Result<u64> {
        let Writer { mut out, tmp, dest } = self;
        let hash = out.hash.clone().finish();
        out.inner.write_all(&hash.to_le_bytes())?;
        out.inner.flush()?;
        let file = out.inner.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        let size = file.metadata()?.len();
        drop(file);
        std::fs::rename(&tmp, &dest).with_context(|| format!("renaming {} to {}", tmp.display(), dest.display()))?;
        Ok(size)
    }

    /// Give up: the temporary file goes and nothing is left.
    pub fn abandon(self) {
        let Writer { out, tmp, .. } = self;
        drop(out);
        let _ = std::fs::remove_file(tmp);
    }
}

impl Write for Writer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.out.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

/// A checkpoint that has been read and checked: whole, and a format this build \[26\]
#[derive(Debug)]
pub struct Checkpoint {
    pub header: Header,
    pub path: PathBuf,
    payload_at: u64,
    payload_len: u64,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Checkpoint> {
        let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let len = f.metadata()?.len();
        let mut magic = [0u8; 8];
        if len < 8 + 8 + 8 || f.read_exact(&mut magic).is_err() || &magic != MAGIC {
            bail!("{} is not a Kestrel checkpoint", path.display());
        }
        // Whole, before anything in it is believed.
        let mut r = BufReader::with_capacity(1 << 20, f);
        r.seek(SeekFrom::Start(0))?;
        let mut hash = Hash64::new();
        let mut left = len - 8;
        let mut buf = vec![0u8; 1 << 20];
        while left > 0 {
            let n = left.min(buf.len() as u64) as usize;
            r.read_exact(&mut buf[..n])?;
            hash.update(&buf[..n]);
            left -= n as u64;
        }
        let mut stored = [0u8; 8];
        r.read_exact(&mut stored)?;
        if hash.finish() != u64::from_le_bytes(stored) {
            bail!(
                "{} is damaged, or was cut off while it was being written, and cannot be resumed",
                path.display()
            );
        }
        r.seek(SeekFrom::Start(8))?;
        let mut n = [0u8; 8];
        r.read_exact(&mut n)?;
        let json_len = u64::from_le_bytes(n);
        if json_len > len {
            bail!("{} is damaged: its header is longer than the file", path.display());
        }
        let mut json = vec![0u8; json_len as usize];
        r.read_exact(&mut json)?;
        let header: Header = serde_json::from_slice(&json)
            .with_context(|| format!("{}: the header is not one this build understands", path.display()))?;
        let payload_at = 8 + 8 + json_len;
        Ok(Checkpoint { header, path: path.to_path_buf(), payload_at, payload_len: len - 8 - payload_at })
    }

    /// The payload, from its start.
    pub fn payload(&self) -> Result<impl Read> {
        let mut f = File::open(&self.path).with_context(|| format!("opening {}", self.path.display()))?;
        f.seek(SeekFrom::Start(self.payload_at))?;
        Ok(BufReader::with_capacity(1 << 20, f).take(self.payload_len))
    }

    pub fn payload_len(&self) -> u64 {
        self.payload_len
    }

    /// Where the payload starts in the file, for the front end that tells how \[27\]
    pub fn size(&self) -> u64 {
        self.payload_at + self.payload_len + 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join("kestrel_resume").join(format!("{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn header() -> Header {
        Header {
            format: FORMAT,
            kestrel: "1.2.3".into(),
            build: build_id().into(),
            argv: vec!["kestrel".into(), "render".into()],
            backend: "cpu".into(),
            midi: Fingerprint { path: "a.mid".into(), size: 10, modified: 1, head: 2, tail: 3 },
            soundfonts: vec![],
            merge: true,
            tracks: vec![0, 2, 5],
            voices_each: 1000,
            block_samples: 8192,
            seconds: None,
            finished: vec![Finished { track: 2, stats: Stats { notes: 7, ..Default::default() } }],
            in_flight: vec![InFlight { track: 5, blocks: 123 }],
            mix_len: 456,
            single: None,
        }
    }

    #[test]
    fn the_hash_does_not_depend_on_how_the_bytes_arrive() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        let whole = hash_bytes(&data);
        for split in [1usize, 3, 7, 8, 9, 64, 999] {
            let mut h = Hash64::new();
            for c in data.chunks(split) {
                h.update(c);
            }
            assert_eq!(h.finish(), whole, "split {split}");
        }
        // And it sees a changed byte, a missing one, and an added zero.
        let mut changed = data.clone();
        changed[500] ^= 1;
        assert_ne!(hash_bytes(&changed), whole);
        assert_ne!(hash_bytes(&data[..999]), whole);
        let mut longer = data.clone();
        longer.push(0);
        assert_ne!(hash_bytes(&longer), whole);
        assert_ne!(hash_bytes(&[]), hash_bytes(&[0]));
    }

    #[test]
    fn a_checkpoint_comes_back_whole() {
        let d = scratch("whole");
        let path = d.join("a.krsm");
        let mut w = Writer::create(&path, &header()).unwrap();
        let payload: Vec<u8> = (0..100_000u32).map(|i| (i % 253) as u8).collect();
        w.write_all(&payload).unwrap();
        // Nothing is at the real name until it is committed.
        assert!(!path.exists());
        w.commit().unwrap();
        let c = Checkpoint::open(&path).unwrap();
        assert_eq!(c.header, header());
        assert_eq!(c.payload_len(), payload.len() as u64);
        let mut back = Vec::new();
        c.payload().unwrap().read_to_end(&mut back).unwrap();
        assert_eq!(back, payload);
        // The temporary name is gone.
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1);
    }

    #[test]
    fn a_checkpoint_cut_off_or_changed_is_never_taken_for_whole() {
        let d = scratch("damaged");
        let path = d.join("a.krsm");
        let mut w = Writer::create(&path, &header()).unwrap();
        w.write_all(&vec![7u8; 5000]).unwrap();
        w.commit().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        for cut in [bytes.len() - 1, bytes.len() - 9, bytes.len() / 2, 40, 12] {
            std::fs::write(&path, &bytes[..cut]).unwrap();
            assert!(Checkpoint::open(&path).is_err(), "cut to {cut}");
        }
        for at in [10usize, 100, 3000, bytes.len() - 3] {
            let mut flipped = bytes.clone();
            flipped[at] ^= 0x40;
            std::fs::write(&path, &flipped).unwrap();
            assert!(Checkpoint::open(&path).is_err(), "byte {at} changed");
        }
        std::fs::write(&path, b"not a checkpoint at all, just some text of enough length").unwrap();
        let e = Checkpoint::open(&path).unwrap_err().to_string();
        assert!(e.contains("not a Kestrel checkpoint"), "{e}");
        std::fs::write(&path, &bytes).unwrap();
        assert!(Checkpoint::open(&path).is_ok());
    }

    fn identity() -> Identity {
        let h = header();
        Identity {
            midi: h.midi,
            soundfonts: vec![],
            merge: true,
            tracks: vec![0, 2, 5],
            block_samples: 8192,
            seconds: None,
            single: false,
        }
    }

    #[test]
    fn every_refusal_fires_and_they_are_all_named_at_once() {
        assert!(header().check(&identity()).is_ok());

        let mut other_build = header();
        other_build.build = "1.2.3-0000000000000000".into();
        let e = other_build.check(&identity()).unwrap_err().to_string();
        assert!(e.contains("another build") || e.contains("made by build"), "{e}");

        let mut moved = identity();
        moved.midi.size = 11;
        assert!(header().check(&moved).unwrap_err().to_string().contains("the MIDI a.mid was 10 bytes and is now 11"));
        let mut edited = identity();
        edited.midi.tail = 9;
        assert!(header().check(&edited).unwrap_err().to_string().contains("different contents"));
        // The modified time alone is not a change: a copy has another.
        let mut copied = identity();
        copied.midi.modified = 99;
        copied.midi.path = "elsewhere/a.mid".into();
        assert!(header().check(&copied).is_ok());

        let mut tracks = identity();
        tracks.tracks = vec![0, 2];
        assert!(header().check(&tracks).unwrap_err().to_string().contains("track"));
        let mut mode = identity();
        mode.merge = false;
        assert!(header().check(&mode).is_err());
        let mut secs = identity();
        secs.seconds = Some(30.0);
        assert!(header().check(&secs).unwrap_err().to_string().contains("--seconds"));

        // Three things changed: all three are said.
        let mut many = identity();
        many.midi.size = 11;
        many.seconds = Some(30.0);
        many.block_samples = 4096;
        let e = header().check(&many).unwrap_err().to_string();
        assert!(e.contains("the MIDI") && e.contains("--seconds") && e.contains("block size"), "{e}");
    }

    #[test]
    fn a_soundfont_is_known_by_its_file_and_its_samples() {
        let d = scratch("sfz");
        let sfz = d.join("a.sfz");
        std::fs::write(d.join("one.wav"), vec![1u8; 100]).unwrap();
        std::fs::write(&sfz, "<region> sample=one.wav key=60\n").unwrap();
        let a = SoundfontPrint::of(&sfz).unwrap();
        assert_eq!(a.sample_files, 1);
        assert!(a.difference(&SoundfontPrint::of(&sfz).unwrap()).is_none());
        // [28]
        std::fs::write(d.join("one.wav"), vec![1u8; 101]).unwrap();
        let b = SoundfontPrint::of(&sfz).unwrap();
        assert!(a.difference(&b).unwrap().contains("sample files"));
        std::fs::write(&sfz, "<region> sample=one.wav key=61\n").unwrap();
        assert!(a.difference(&SoundfontPrint::of(&sfz).unwrap()).is_some());
    }

    #[test]
    fn a_file_in_progress_has_a_name_no_finished_one_has() {
        assert_eq!(partial_of(Path::new("o/song.wav")), Path::new("o/song.partial.wav"));
        assert_eq!(partial_of(Path::new("o/01 Piano (a.b).flac")), Path::new("o/01 Piano (a.b).partial.flac"));
        assert_eq!(partial_of(Path::new("o/song")), Path::new("o/song.partial"));
    }

    #[test]
    fn a_big_file_is_fingerprinted_by_its_ends() {
        let d = scratch("ends");
        let p = d.join("big.bin");
        let mut data = vec![0u8; (ENDS * 3) as usize];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        std::fs::write(&p, &data).unwrap();
        let a = Fingerprint::of(&p).unwrap();
        // [29]
        data[(ENDS + 5) as usize] ^= 1;
        std::fs::write(&p, &data).unwrap();
        assert_eq!(Fingerprint::of(&p).unwrap().head, a.head);
        data[5] ^= 1;
        std::fs::write(&p, &data).unwrap();
        assert_ne!(Fingerprint::of(&p).unwrap().head, a.head);
        data[5] ^= 1;
        let n = data.len();
        data[n - 5] ^= 1;
        std::fs::write(&p, &data).unwrap();
        assert_ne!(Fingerprint::of(&p).unwrap().tail, a.tail);
    }
}
