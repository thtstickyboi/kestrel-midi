// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Key-major merlin, step 1: the bucket pass. \[1\]

use crate::backend::{Backend, BlockStats};
use crate::bank::{Bank, ModEnvParams, RegionParams};
use crate::config::Config;
use crate::midi::CHANNELS;
use crate::voice::{
    ChannelTable, SpawnCmd, BASE_CHANNELS, CHAN_BEND, CHAN_CUT, CHAN_CUT_ID_HI, CHAN_CUT_ID_LO,
    CHAN_FIELDS, CHAN_GAIN_L, CHAN_GAIN_R, NOTE_HI_MASK, SLOT_BITS, SLOT_MASK,
};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Groups the store is filed by: one per MIDI key, every channel's notes on \[2\]
pub const KEYS: usize = 128;

/// A CC120 damped this run rather than releasing it. Merlin fires no impulse \[3\]
pub const REC_CUT: u16 = 1;
/// The note glided (portamento) and the glide was stripped, which is what \[4\]
pub const REC_GLIDE: u16 = 2;
/// The flags a record carries; the bits above are the open run's own.
const REC_MASK: u16 = 0xFF;
/// Merlin would render this run on its per-voice fallback, so it is not in \[5\]
const OPEN_FALLBACK: u16 = 0x100;
/// Merlin would already have planned what is left of this run, clamped to its \[6\]
const OPEN_CLAMPED: u16 = 0x200;

/// How far back along a slot's runs a copy looks for the run it continues: a \[7\]
const FOLD_LOOKBACK: usize = 8;

/// One run of identical notes, as the bucket store holds it. \[8\]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RunRec {
    /// The note-on's frame, from the start of the render.
    pub start: u64,
    /// Frames from the note-on to its note-off, with no ceiling.
    pub duration: u32,
    /// Row of [`BucketPass::voices`].
    pub voice: u32,
    /// The command's whole `gate_slot` word: `channel * 128 + key`.
    pub slot: u32,
    pub count: u16,
    /// [`REC_CUT`], [`REC_GLIDE`].
    pub flags: u16,
    /// The note's gains at its note-on with its channel's folded in, which is \[9\]
    pub gain: [f32; 2],
    /// The note's own gains times the channel's at the note-off: what the \[10\]
    pub gain_end: [f32; 2],
}

const REC: usize = std::mem::size_of::<RunRec>();
const _: () = assert!(REC == 40);

/// What one row of the voice table stands for.
#[derive(Clone, Copy)]
pub struct VoiceDesc {
    /// The command with every word that differs per note cleared: the ids, the \[11\]
    pub cmd: SpawnCmd,
    /// The params row it reads, from its own variant, as it stood at the \[12\]
    pub params: RegionParams,
    /// The modulation-envelope row beside it, `None` when nothing reads one.
    pub menv: Option<ModEnvParams>,
    /// The right half of a stereo sample, `cmd.smp_base` for a mono voice.
    pub smp_base_r: u32,
}

impl VoiceDesc {
    /// The command a record of this voice stands for, as merlin holds it once \[13\]
    pub fn cmd_for(&self, r: &RunRec) -> SpawnCmd {
        let mut c = self.cmd;
        c.gate_slot = r.slot;
        c.gain_l = r.gain[0];
        c.gain_r = r.gain[1];
        c
    }
}

/// Clear what differs between copies of one note -- its id, its ordinal on the \[14\]
fn strip_note(c: &SpawnCmd) -> SpawnCmd {
    let mut c = *c;
    c.note_id_lo = 0;
    c.note_id_hi &= !NOTE_HI_MASK;
    c.ordinal = 0;
    c.start_rel = 0;
    c.row_bias = 0;
    c
}

/// And the per-note words a record carries itself.
pub(crate) fn strip_voice(c: &SpawnCmd) -> SpawnCmd {
    let mut c = strip_note(c);
    c.gate_slot = 0;
    c.gain_l = 0.0;
    c.gain_r = 0.0;
    c
}

/// A voice table key: the stripped command, compared and hashed by its bytes, \[15\]
#[derive(Clone, Copy)]
struct CmdKey {
    cmd: SpawnCmd,
    generation: u32,
}

impl PartialEq for CmdKey {
    fn eq(&self, o: &Self) -> bool {
        self.generation == o.generation && bytemuck::bytes_of(&self.cmd) == bytemuck::bytes_of(&o.cmd)
    }
}

impl Eq for CmdKey {}

impl Hash for CmdKey {
    fn hash<H: Hasher>(&self, h: &mut H) {
        h.write(bytemuck::bytes_of(&self.cmd));
        h.write_u32(self.generation);
    }
}

/// `MerlinSynth`'s `VoiceKey`, which its promotion count is keyed on: the \[16\]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PromoteKey {
    params: u32,
    variant: u32,
    step_lo: u32,
    step_hi: u32,
    smp_base: u32,
    smp_len: u32,
    loop_start: u32,
    loop_end: u32,
    flags: u32,
}

impl PromoteKey {
    fn new(c: &SpawnCmd) -> Self {
        Self {
            params: c.params,
            variant: c.variant,
            step_lo: c.step_lo,
            step_hi: c.step_hi,
            smp_base: c.smp_base,
            smp_len: c.smp_len,
            loop_start: c.loop_start,
            loop_end: c.loop_end,
            flags: c.flags,
        }
    }
}

/// An order-free digest of a list of notes, copy by copy. \[17\]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoteDigest {
    pub copies: u64,
    /// Held frames, summed: a figure a person can read.
    pub frames: u64,
    pub a: u64,
    pub b: u64,
}

impl NoteDigest {
    pub fn add(&mut self, cmd: &SpawnCmd, start: u64, duration: u32, gain_end: [f32; 2], cut: bool, count: u64) {
        let c = strip_note(cmd);
        let (mut a, mut b) = (0x243F_6A88_85A3_08D3u64, 0x1319_8A2E_0370_7344u64);
        let mut eat = |w: u32| {
            a = (a ^ w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            a ^= a >> 32;
            b = (b ^ (w as u64).rotate_left(17)).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            b ^= b >> 29;
        };
        for w in bytemuck::bytes_of(&c).chunks_exact(4) {
            eat(u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
        }
        eat(start as u32);
        eat((start >> 32) as u32);
        eat(duration);
        eat(gain_end[0].to_bits());
        eat(gain_end[1].to_bits());
        eat(cut as u32);
        self.copies += count;
        self.frames = self.frames.wrapping_add(duration as u64 * count);
        self.a = self.a.wrapping_add(finish(a).wrapping_mul(count));
        self.b = self.b.wrapping_add(finish(b ^ 0x5851_F42D_4C95_7F2D).wrapping_mul(count));
    }
}

impl std::fmt::Display for NoteDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} copies, {} frames, {:016x}{:016x}", self.copies, self.frames, self.a, self.b)
    }
}

/// splitmix64's finaliser.
fn finish(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The store's memory for `cfg`, in bytes. See `Config::merlin_bucket_mb`.
pub fn store_budget(cfg: &Config) -> usize {
    if cfg.merlin_bucket_mb > 0 {
        return (cfg.merlin_bucket_mb as usize) << 20;
    }
    match crate::falconeye::winsys::memory() {
        Some(m) => ((m.total / 16) as usize).clamp(256 << 20, 2 << 30),
        None => 1 << 30,
    }
}

/// Records a key's list holds in one sealed chunk: 160 KiB. A list grows by \[18\]
pub const CHUNK: usize = 4 << 10;

/// One key's records: on disk, then sealed in memory, then the open tail, which \[19\]
#[derive(Default)]
struct KeyList {
    /// (first record in the spill file, records), oldest first.
    on_disk: Vec<(u64, u32)>,
    sealed: Vec<Vec<RunRec>>,
    tail: Vec<RunRec>,
}

impl KeyList {
    fn held(&self) -> usize {
        (self.sealed.len() * CHUNK + self.tail.capacity()) * REC
    }
}

/// The spill file, shared by the store that writes it and the sealed windows \[20\]
struct Spill {
    file: Option<std::fs::File>,
    path: PathBuf,
}

impl Drop for Spill {
    fn drop(&mut self) {
        // Closed first: Windows will not remove a file that is still open.
        drop(self.file.take());
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Spill {
    fn file(&self) -> &std::fs::File {
        self.file.as_ref().expect("open until dropped")
    }
}

/// One window's lists, one per key.
type WindowLists = Vec<KeyList>;

fn window_lists() -> WindowLists {
    (0..KEYS).map(|_| KeyList::default()).collect()
}

/// The records, one list per (window, key), inside one memory budget that \[21\]
pub struct BucketStore {
    /// The windows not yet sealed, oldest first: `windows[i]` is window \[22\]
    windows: VecDeque<WindowLists>,
    base: u64,
    /// Record storage held in memory by the windows not yet sealed, as \[23\]
    held: usize,
    /// The same for sealed windows still held by their reader, which gives it \[24\]
    sealed: Arc<AtomicUsize>,
    budget: usize,
    dir: PathBuf,
    spill: Option<Arc<Spill>>,
    /// Records written to the spill file.
    spill_len: u64,
    pub peak_held: usize,
    /// Records filed, each once however many windows it went into.
    pub records: u64,
    /// Records filed counting every window: what the windows hold.
    pub copies: u64,
    pub spilled: u64,
    pub spill_writes: u64,
    pub spill_secs: f64,
}

impl BucketStore {
    pub fn new(budget: usize, dir: &Path) -> Self {
        Self {
            windows: VecDeque::new(),
            base: 0,
            held: 0,
            sealed: Arc::new(AtomicUsize::new(0)),
            // [25]
            budget: budget.max(KEYS * CHUNK * REC),
            dir: dir.to_path_buf(),
            spill: None,
            spill_len: 0,
            peak_held: 0,
            records: 0,
            copies: 0,
            spilled: 0,
            spill_writes: 0,
            spill_secs: 0.0,
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// What the budget covers now: the unsealed windows and the sealed ones \[26\]
    pub fn held(&self) -> usize {
        self.held + self.sealed.load(Ordering::Relaxed)
    }

    /// The oldest window not yet sealed.
    pub fn base(&self) -> u64 {
        self.base
    }


    fn list_mut(&mut self, w: u64, key: usize) -> Result<&mut KeyList> {
        if w < self.base {
            anyhow::bail!("a record for window {w}, which was sealed already (the oldest open is {})", self.base);
        }
        let i = (w - self.base) as usize;
        while self.windows.len() <= i {
            self.windows.push_back(window_lists());
        }
        Ok(&mut self.windows[i][key])
    }

    /// File `rec` under `key` in every window of `windows`.
    pub fn push(&mut self, windows: std::ops::RangeInclusive<u64>, key: usize, rec: RunRec) -> Result<()> {
        for w in windows {
            let l = self.list_mut(w, key)?;
            let before = l.held();
            if l.tail.len() == CHUNK {
                let full = std::mem::take(&mut l.tail);
                l.sealed.push(full);
            }
            if l.tail.len() == l.tail.capacity() {
                let want = (l.tail.capacity() * 2).clamp(16, CHUNK);
                l.tail.reserve_exact(want - l.tail.len());
            }
            l.tail.push(rec);
            let after = l.held();
            self.held = self.held + after - before;
            self.copies += 1;
        }
        self.records += 1;
        self.peak_held = self.peak_held.max(self.held());
        if self.held() > self.budget {
            self.spill_down()?;
        }
        Ok(())
    }

    /// Write whole lists out, largest first, until a quarter of the budget is \[27\]
    fn spill_down(&mut self) -> Result<()> {
        let t = std::time::Instant::now();
        let target = self.budget / 4 * 3;
        while self.held() > target {
            // [28]
            let mut best = (0usize, 0usize, 0usize);
            for (i, lists) in self.windows.iter().enumerate() {
                for (k, l) in lists.iter().enumerate() {
                    if l.held() > best.2 {
                        best = (i, k, l.held());
                    }
                }
            }
            if best.2 == 0 {
                break;
            }
            let (i, k) = (best.0, best.1);
            let mut l = std::mem::take(&mut self.windows[i][k]);
            self.held -= l.held();
            self.spill_list(&mut l)?;
            self.windows[i][k] = l;
        }
        self.spill_secs += t.elapsed().as_secs_f64();
        Ok(())
    }

    fn spill_list(&mut self, l: &mut KeyList) -> Result<()> {
        if self.spill.is_none() {
            std::fs::create_dir_all(&self.dir)
                .with_context(|| format!("creating the bucket spill folder {}", self.dir.display()))?;
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let path = self.dir.join(format!("kestrel-buckets-{}-{nanos}.bin", std::process::id()));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .with_context(|| format!("creating the bucket spill file {}", path.display()))?;
            self.spill = Some(Arc::new(Spill { file: Some(file), path }));
        }
        let sp = self.spill.clone().expect("opened above");
        let sealed = std::mem::take(&mut l.sealed);
        let tail = std::mem::take(&mut l.tail);
        for chunk in sealed.iter().chain(std::iter::once(&tail)) {
            if chunk.is_empty() {
                continue;
            }
            write_all_at(sp.file(), bytemuck::cast_slice(chunk), self.spill_len * REC as u64)
                .with_context(|| format!("writing the bucket spill file {}", sp.path.display()))?;
            l.on_disk.push((self.spill_len, chunk.len() as u32));
            self.spill_len += chunk.len() as u64;
            self.spilled += chunk.len() as u64;
            self.spill_writes += 1;
        }
        Ok(())
    }

    /// Hand over the oldest open window, whole, to be read: everything filed \[29\]
    pub fn seal_front(&mut self) -> Result<SealedWindow> {
        let mut lists = self.windows.pop_front().unwrap_or_else(window_lists);
        let index = self.base;
        self.base += 1;
        let mut bytes: usize = lists.iter().map(KeyList::held).sum();
        self.held -= bytes;
        let target = self.budget / 4 * 3;
        while self.held() + bytes > self.budget {
            // Its largest list first, ties to the lowest key.
            let (k, h) = lists.iter().enumerate().fold((0, 0), |b, (k, l)| if l.held() > b.1 { (k, l.held()) } else { b });
            if h == 0 {
                break;
            }
            let t = std::time::Instant::now();
            self.spill_list(&mut lists[k])?;
            self.spill_secs += t.elapsed().as_secs_f64();
            bytes -= h;
            if self.held() + bytes <= target {
                break;
            }
        }
        self.sealed.fetch_add(bytes, Ordering::Relaxed);
        Ok(SealedWindow { index, lists, bytes, sealed: self.sealed.clone(), spill: self.spill.clone() })
    }

    /// Bytes in the spill file.
    pub fn spill_bytes(&self) -> u64 {
        self.spill_len * REC as u64
    }

    fn front(&self) -> Option<&WindowLists> {
        self.windows.front()
    }

    /// Records filed under `key` in the oldest open window: all of them, \[30\]
    pub fn key_len(&self, key: usize) -> u64 {
        self.front().map_or(0, |w| w[key].len())
    }

    /// `key`'s records in the oldest open window -- all of them, without \[31\]
    pub fn key_records(&self, key: usize) -> Result<KeyRecords<'_>> {
        match self.front() {
            Some(w) => w[key].records(self.spill.as_deref()),
            None => Ok(KeyRecords { disk: Vec::new(), mem: Vec::new() }),
        }
    }

    /// Hand `f` every record filed under `key` in the oldest open window, in \[32\]
    pub fn read_key(&mut self, key: usize, mut f: impl FnMut(&[RunRec]) -> Result<()>) -> Result<()> {
        let recs = self.key_records(key)?;
        for c in recs.chunks() {
            f(c)?;
        }
        Ok(())
    }
}

impl KeyList {
    /// Records in the list.
    fn len(&self) -> u64 {
        self.on_disk.iter().map(|&(_, n)| n as u64).sum::<u64>() + (self.sealed.len() * CHUNK + self.tail.len()) as u64
    }

    /// The records in the order they were filed, without copying what is in \[33\]
    fn records<'a>(&'a self, spill: Option<&Spill>) -> Result<KeyRecords<'a>> {
        self.records_between(spill, 0, usize::MAX)
    }

    /// Records `[lo, hi)` of the list, the same way: of what is on disk, only \[34\]
    fn records_between<'a>(&'a self, spill: Option<&Spill>, lo: usize, hi: usize) -> Result<KeyRecords<'a>> {
        let mut disk: Vec<RunRec> = Vec::new();
        let mut at_rec = 0usize;
        for &(at, n) in &self.on_disk {
            let (s, e) = (at_rec.max(lo), (at_rec + n as usize).min(hi));
            if s < e {
                let sp = spill.context("a list on disk with no spill file")?;
                let from = disk.len();
                disk.resize(from + (e - s), RunRec::zeroed());
                let pos = (at + (s - at_rec) as u64) * REC as u64;
                read_exact_at(sp.file(), bytemuck::cast_slice_mut(&mut disk[from..]), pos)
                    .with_context(|| format!("reading the bucket spill file {}", sp.path.display()))?;
            }
            at_rec += n as usize;
        }
        let mut mem: Vec<&[RunRec]> = Vec::new();
        for c in self.sealed.iter().map(|c| &c[..]).chain(std::iter::once(&self.tail[..])) {
            let (s, e) = (at_rec.max(lo), (at_rec + c.len()).min(hi));
            if s < e {
                mem.push(&c[s - at_rec..e - at_rec]);
            }
            at_rec += c.len();
        }
        Ok(KeyRecords { disk, mem })
    }
}

/// A window the pass has finished filing, handed whole to its reader: every \[35\]
pub struct SealedWindow {
    pub index: u64,
    lists: WindowLists,
    bytes: usize,
    sealed: Arc<AtomicUsize>,
    spill: Option<Arc<Spill>>,
}

impl SealedWindow {
    /// Records under `key`.
    pub fn key_len(&self, key: usize) -> u64 {
        self.lists[key].len()
    }

    /// `key`'s records, in the order they were filed.
    pub fn key_records(&self, key: usize) -> Result<KeyRecords<'_>> {
        self.lists[key].records(self.spill.as_deref())
    }

    /// `key`'s records `[lo, hi)`, reading from disk only what is in range.
    pub fn key_records_between(&self, key: usize, lo: usize, hi: usize) -> Result<KeyRecords<'_>> {
        self.lists[key].records_between(self.spill.as_deref(), lo, hi)
    }
}

impl Drop for SealedWindow {
    fn drop(&mut self) {
        self.sealed.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// One key's records, from `key_records`: what was on disk, then what is in \[36\]
pub struct KeyRecords<'a> {
    disk: Vec<RunRec>,
    mem: Vec<&'a [RunRec]>,
}

impl KeyRecords<'_> {
    /// The records a chunk at a time, in filing order.
    pub fn chunks(&self) -> Vec<&[RunRec]> {
        std::iter::once(&self.disk[..]).filter(|c| !c.is_empty()).chain(self.mem.iter().copied()).collect()
    }
}

/// `buf.len()` bytes of `file` from `at`, without its cursor mattering, so \[37\]
pub(crate) fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut at: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(file, buf, at)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(file, buf, at)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        at += n as u64;
    }
    Ok(())
}

/// `buf` into `file` at `at`, the same way.
fn write_all_at(file: &std::fs::File, mut buf: &[u8], mut at: u64) -> std::io::Result<()> {
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

/// One installed variant: its params table and modulation-envelope table.
type VariantTables = (Arc<Vec<RegionParams>>, Arc<Vec<ModEnvParams>>);

/// One channel's bend and gains from `frame` until its next point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtlPoint {
    pub frame: u64,
    /// 8.24, as the channel row carries it, before `--merlin-bend-cents`.
    pub bend: u32,
    pub gain: [f32; 2],
}

const UNITY: (u32, [f32; 2]) = (crate::fixed::BEND_ONE, [1.0, 1.0]);

/// A run of identical note-ons waiting for their note-offs.
#[derive(Clone, Copy)]
struct OpenRun {
    start: u64,
    /// The first copy's, which is what merlin compares a CC120's against.
    note_id: u64,
    voice: u32,
    slot: u32,
    /// The first copy's ordinal; the others follow it by one each.
    ordinal: u32,
    count: u16,
    flags: u16,
    gain: [f32; 2],
    /// The note's own gains, before the channel's were folded in.
    base: [f32; 2],
}

impl OpenRun {
    /// `count` of this run's copies from the `from`-th on.
    fn part(&self, from: u16, count: u16) -> OpenRun {
        let mut o = *self;
        o.ordinal = o.ordinal.wrapping_add(from as u32);
        o.count = count;
        o
    }
}

/// Where the pass's wall clock goes, cumulative seconds. \[38\]
#[derive(Default, Clone, Copy)]
struct Profile {
    /// `route`: the glide rule, the channel, the voice table and folding.
    route: f64,
    /// `resolve`: pairing note-offs and filing the records, spill writes \[39\]
    resolve: f64,
    /// `set_channels`' curve scan.
    curves: f64,
}

/// The bucket pass as a [`Backend`]. See the module documentation.
pub struct BucketPass {
    cfg: Config,
    bank: Arc<Bank>,
    block: u64,
    block_frames: u64,

    tile_frames: u32,
    tiles: u32,
    chan_rows: Vec<u32>,
    chan_count: usize,
    bend_active: bool,
    gain_active: bool,
    cut_active: bool,
    bend_memo: [(u32, u32); CHANNELS],
    cut_in_block: Vec<bool>,
    off_meta: Vec<u32>,
    off_runs: Vec<u32>,

    /// The variant tables as installed, index minus one, and how many times \[40\]
    variants: Vec<Option<VariantTables>>,
    generation: Vec<u32>,
    voice_of: FxHashMap<CmdKey, u32>,
    voices: Vec<VoiceDesc>,
    /// The last two voice rows each slot found, newest first (`u32::MAX` \[41\]
    voice_memo: Vec<[(CmdKey, u32); 2]>,

    /// Open runs per slot, oldest first: ordinal order, which is the order the \[42\]
    open: Vec<VecDeque<OpenRun>>,
    live: usize,
    parts_open: bool,
    /// Whether this block's mirrored head-cache clamp has run.
    clamp_scanned: bool,

    store: BucketStore,
    curves: Vec<Vec<CtlPoint>>,

    /// Mirroring merlin for the gate: its promotion threshold, its head-cache \[43\]
    mirror: bool,
    promote: u32,
    head_frames: u64,
    fb_uses: FxHashMap<PromoteKey, u32>,
    digest_merlin: NoteDigest,
    digest_all: NoteDigest,

    pub notes: u64,
    pub runs: u64,
    pub folded: u64,
    pub glides: u64,
    pub cut_notes: u64,
    pub bent_notes: u64,
    pub fallback_notes: u64,
    pub clamped_notes: u64,
    pub peak_open: usize,
    prof: Profile,

    /// Filing by window, for the per-key render: the window's length in \[44\]
    window: Option<u64>,
    /// Each voice row's release tail as key-major plans it, its raw release \[45\]
    voice_tail: Vec<u32>,
    tail_cap: u32,
    /// Two partitions, which key-major adds to where a note stops reaching.
    two_p: u64,
    ready: VecDeque<SealedWindow>,
    /// Where the last record stops reaching the output.
    reach_end: u64,
}

/// Frames past the last note's reach that key-major renders, for the bus \[46\]
pub const RING_OUT: u64 = 96_000;

impl BucketPass {
    /// Spills, past the budget, into `dir`.
    pub fn new(cfg: &Config, bank: Arc<Bank>, dir: &Path) -> Result<Self> {
        let tiles = cfg.block_frames / cfg.gate_frames;
        let w = BASE_CHANNELS * CHAN_FIELDS;
        let mut chan_rows = vec![0u32; (tiles as usize + 1) * w];
        for r in chan_rows.chunks_exact_mut(CHAN_FIELDS) {
            r[CHAN_BEND] = crate::fixed::BEND_ONE;
            r[CHAN_GAIN_L] = 1.0f32.to_bits();
            r[CHAN_GAIN_R] = 1.0f32.to_bits();
        }
        let store = BucketStore::new(store_budget(cfg), dir);
        let shape = crate::merlin::keymajor::Shape::new(cfg, &bank);
        log::info!(
            "merlin bucket pass: store budget {} MiB, spilling to {}",
            store.budget() >> 20,
            dir.display()
        );
        Ok(Self {
            cfg: cfg.clone(),
            bank,
            block: 0,
            block_frames: cfg.block_frames as u64,
            tile_frames: cfg.gate_frames,
            tiles,
            chan_rows,
            chan_count: BASE_CHANNELS,
            bend_active: false,
            gain_active: false,
            cut_active: false,
            bend_memo: [(crate::fixed::BEND_ONE, crate::fixed::BEND_ONE); CHANNELS],
            cut_in_block: Vec::new(),
            off_meta: Vec::new(),
            off_runs: Vec::new(),
            variants: Vec::new(),
            generation: Vec::new(),
            voice_of: FxHashMap::default(),
            voices: Vec::new(),
            voice_memo: vec![[(CmdKey { cmd: SpawnCmd::zeroed(), generation: 0 }, u32::MAX); 2]; 1usize << SLOT_BITS],
            open: (0..1usize << SLOT_BITS).map(|_| VecDeque::new()).collect(),
            live: 0,
            parts_open: false,
            clamp_scanned: false,
            store,
            curves: Vec::new(),
            mirror: cfg.merlin_digest,
            promote: cfg.merlin_promote_after,
            head_frames: cfg.merlin_head() as u64,
            fb_uses: FxHashMap::default(),
            digest_merlin: NoteDigest::default(),
            digest_all: NoteDigest::default(),
            notes: 0,
            runs: 0,
            folded: 0,
            glides: 0,
            cut_notes: 0,
            bent_notes: 0,
            fallback_notes: 0,
            clamped_notes: 0,
            peak_open: 0,
            prof: Profile::default(),
            window: None,
            voice_tail: Vec::new(),
            tail_cap: shape.tail_cap as u32,
            two_p: 2 * shape.p as u64,
            ready: VecDeque::new(),
            reach_end: 0,
        })
    }

    /// File by windows of `frames`, every window sealed once the file is read \[47\]
    pub fn windowed(mut self, frames: u64) -> Self {
        self.window = Some(frames.max(1));
        self
    }

    /// The next window sealed, in order.
    pub fn take_ready(&mut self) -> Option<SealedWindow> {
        self.ready.pop_front()
    }

    /// Where key-major's render ends: the last record's reach and the bus \[48\]
    pub fn end(&self) -> u64 {
        self.reach_end + RING_OUT
    }

    /// After the file is read: seal every window up to the render's end.
    pub fn seal_all(&mut self) -> Result<()> {
        let Some(w) = self.window else { return Ok(()) };
        let windows = self.end().div_ceil(w);
        if self.live > 0 {
            anyhow::bail!("{} runs still open at the end of the file", self.live);
        }
        while self.store.base() < windows {
            let sw = self.store.seal_front()?;
            self.ready.push_back(sw);
        }
        Ok(())
    }

    pub fn voices(&self) -> &[VoiceDesc] {
        &self.voices
    }

    pub fn store(&mut self) -> &mut BucketStore {
        &mut self.store
    }

    /// The store, shared, for `BucketStore::key_records`.
    pub fn store_ref(&self) -> &BucketStore {
        &self.store
    }

    pub fn curves(&self) -> &[Vec<CtlPoint>] {
        &self.curves
    }

    /// What merlin would plan, by [`NoteDigest`]. Only kept with \[49\]
    pub fn merlin_digest(&self) -> Option<NoteDigest> {
        self.mirror.then_some(self.digest_merlin)
    }

    /// Every note filed, as it was filed. Only kept with \[50\]
    pub fn filed_digest(&self) -> NoteDigest {
        self.digest_all
    }

    /// Runs still waiting for a note-off.
    pub fn open_runs(&self) -> usize {
        self.live
    }

    /// Every note in the store, read back.
    pub fn stored_digest(&mut self) -> Result<NoteDigest> {
        let mut d = NoteDigest::default();
        let voices = &self.voices;
        for k in 0..KEYS {
            self.store.read_key(k, |recs| {
                for r in recs {
                    let v = voices.get(r.voice as usize).context("a record names no voice")?;
                    if (r.slot & SLOT_MASK) as usize % KEYS != k {
                        anyhow::bail!("a record for key {} is filed under key {k}", r.slot % KEYS as u32);
                    }
                    d.add(&v.cmd_for(r), r.start, r.duration, r.gain_end, r.flags & REC_CUT != 0, r.count as u64);
                }
                Ok(())
            })?;
        }
        Ok(d)
    }

    /// `MerlinSynth::quantise_bend`.
    fn quantise_bend(&mut self, ch: usize, raw: u32) -> u32 {
        let w = self.cfg.merlin_bend_cents;
        if w <= 0.0 || raw == crate::fixed::BEND_ONE {
            return raw;
        }
        let memo = &mut self.bend_memo[ch & (CHANNELS - 1)];
        if memo.0 == raw {
            return memo.1;
        }
        let cents = 1200.0 * (raw as f64 / crate::fixed::BEND_ONE as f64).log2();
        let q = crate::fixed::bend_factor((cents / w).round() * w / 100.0);
        *memo = (raw, q);
        q
    }

    /// `MerlinSynth::apply_channel`: the channel's bend into the step and its \[51\]
    fn apply_channel(&mut self, c: &mut SpawnCmd) -> [f32; 2] {
        let base = [c.gain_l, c.gain_r];
        if !self.bend_active && !self.gain_active {
            return base;
        }
        let ch = ChannelTable::channel_of(c.gate_slot);
        let tile = (c.start_rel / self.tile_frames).min(self.tiles.saturating_sub(1));
        let row = ((tile + c.row_bias.min(1)) as usize * self.chan_count + ch) * CHAN_FIELDS;
        if self.bend_active {
            let raw = self.chan_rows[row + CHAN_BEND];
            if raw != crate::fixed::BEND_ONE {
                self.bent_notes += 1;
                let f = self.quantise_bend(ch, raw);
                let step = crate::fixed::Fixed::from_parts(c.step_hi, c.step_lo).scale(f);
                c.step_lo = step.lo();
                c.step_hi = step.hi();
            }
        }
        if !self.gain_active {
            return base;
        }
        c.gain_l *= f32::from_bits(self.chan_rows[row + CHAN_GAIN_L]);
        c.gain_r *= f32::from_bits(self.chan_rows[row + CHAN_GAIN_R]);
        base
    }

    /// `MerlinSynth::chan_gain_at`.
    fn chan_gain_at(&self, gate_slot: u32, rel: u32) -> [f32; 2] {
        if !self.gain_active {
            return [1.0, 1.0];
        }
        let ch = ChannelTable::channel_of(gate_slot);
        let tile = (rel / self.tile_frames).min(self.tiles.saturating_sub(1));
        let row = (tile as usize * self.chan_count + ch) * CHAN_FIELDS;
        [f32::from_bits(self.chan_rows[row + CHAN_GAIN_L]), f32::from_bits(self.chan_rows[row + CHAN_GAIN_R])]
    }

    /// `MerlinSynth::cut_at`: by note id, not by frame.
    fn cut_at(&self, gate_slot: u32, note_id: u64, rel: u32) -> bool {
        if !self.cut_active {
            return false;
        }
        let ch = ChannelTable::channel_of(gate_slot);
        let tile = (rel / self.tile_frames).min(self.tiles.saturating_sub(1));
        let row = (tile as usize * self.chan_count + ch) * CHAN_FIELDS;
        if self.chan_rows[row + CHAN_CUT] == 0 {
            return false;
        }
        let cut_id = (self.chan_rows[row + CHAN_CUT_ID_HI] as u64) << 32 | self.chan_rows[row + CHAN_CUT_ID_LO] as u64;
        note_id < cut_id
    }

    /// `MerlinSynth::release_frame`, for the `k`-th copy of a run.
    fn release_frame(&self, o: &OpenRun, k: u16, block_start: u64) -> Option<u64> {
        let s = (o.slot & SLOT_MASK) as usize;
        if s + 1 >= self.off_meta.len() / 2 {
            return None;
        }
        let ordinal = o.ordinal.wrapping_add(k as u32);
        if ordinal <= self.off_meta[s * 2] {
            return None;
        }
        crate::voice::off_frame(&self.off_meta, &self.off_runs, s, ordinal).map(|f| block_start + f as u64)
    }

    /// `MerlinSynth::params_of`, and its fallback to the bank's row for a \[52\]
    fn params_of(&self, variant: u32, base: usize) -> Option<RegionParams> {
        match variant.checked_sub(1) {
            None => self.bank.params.get(base).copied(),
            Some(i) => self
                .variants
                .get(i as usize)
                .and_then(|v| v.as_ref())
                .and_then(|v| v.0.get(base))
                .copied()
                .or_else(|| self.bank.params.get(base).copied()),
        }
    }

    /// `MerlinSynth::menv_of`.
    fn menv_of(&self, variant: u32, base: usize, p: &RegionParams) -> Option<ModEnvParams> {
        if !self.cfg.mod_env_enabled || p.flags & crate::bank::RP_MOD_ENV == 0 {
            return None;
        }
        let fallback = || self.bank.menv.get(base).copied();
        match variant.checked_sub(1) {
            None => fallback(),
            Some(i) => self
                .variants
                .get(i as usize)
                .and_then(|v| v.as_ref())
                .and_then(|v| v.1.get(base))
                .copied()
                .or_else(fallback),
        }
    }

    /// The voice table row for a resolved command, made on first sight with \[53\]
    fn voice_id(&mut self, m: &SpawnCmd) -> Result<u32> {
        let generation = match m.variant.checked_sub(1) {
            None => 0,
            Some(i) => self.generation.get(i as usize).copied().unwrap_or(0),
        };
        let key = CmdKey { cmd: strip_voice(m), generation };
        let memo = &mut self.voice_memo[(m.gate_slot & SLOT_MASK) as usize];
        if memo[0].1 != u32::MAX && memo[0].0 == key {
            return Ok(memo[0].1);
        }
        if memo[1].1 != u32::MAX && memo[1].0 == key {
            memo.swap(0, 1);
            return Ok(memo[0].1);
        }
        if let Some(&v) = self.voice_of.get(&key) {
            memo[1] = memo[0];
            memo[0] = (key, v);
            return Ok(v);
        }
        let params = self
            .params_of(m.variant, m.params as usize)
            .with_context(|| format!("region {} has no params", m.region))?;
        let menv = self.menv_of(m.variant, m.params as usize, &params);
        let v = self.voices.len() as u32;
        let tail = crate::merlin::kernel::release_frames(&params, &self.cfg).min(self.tail_cap as usize) as u32;
        self.voice_tail.push(tail);
        self.voices.push(VoiceDesc { cmd: key.cmd, params, menv, smp_base_r: key.cmd.smp_base });
        self.voice_of.insert(key, v);
        let memo = &mut self.voice_memo[(m.gate_slot & SLOT_MASK) as usize];
        memo[1] = memo[0];
        memo[0] = (key, v);
        Ok(v)
    }

    /// `MerlinSynth::open_block`'s part: which channels a CC120 cuts this \[54\]
    fn open_block(&mut self) {
        self.clamp_scanned = false;
        self.cut_in_block.clear();
        self.cut_in_block.resize(self.chan_count, false);
        if self.cut_active {
            for tile in 0..self.tiles as usize {
                for ch in 0..self.chan_count {
                    let row = (tile * self.chan_count + ch) * CHAN_FIELDS;
                    if self.chan_rows[row + CHAN_CUT] != 0 {
                        self.cut_in_block[ch] = true;
                    }
                }
            }
        }
    }

    /// `MerlinSynth::fold_into_run`'s rule, on the voice row rather than the \[55\]
    fn fold(&mut self, m: &SpawnCmd, voice: u32, start: u64, base: [f32; 2], flags: u16) -> bool {
        let ch = ChannelTable::channel_of(m.gate_slot);
        if self.cut_in_block.get(ch).copied().unwrap_or(true) {
            return false;
        }
        let same_gain = |a: [f32; 2], b: [f32; 2]| a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits();
        let q = &mut self.open[(m.gate_slot & SLOT_MASK) as usize];
        for o in q.iter_mut().rev().take(FOLD_LOOKBACK) {
            if o.count < u16::MAX
                && o.start == start
                && m.ordinal == o.ordinal.wrapping_add(o.count as u32)
                && o.voice == voice
                && o.slot == m.gate_slot
                && o.flags == flags
                && same_gain(o.gain, [m.gain_l, m.gain_r])
                && same_gain(o.base, base)
            {
                o.count += 1;
                self.folded += 1;
                return true;
            }
        }
        false
    }

    /// `MerlinSynth::route`, for every note: none goes anywhere else, so the \[56\]
    fn route(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        let t = std::time::Instant::now();
        let block_start = self.block * self.block_frames;
        for c in cmds {
            self.notes += 1;
            let mut m = *c;
            let mut flags = 0u16;
            if m.flags > crate::porta::FLAG_BITS {
                // [57]
                if self.mirror && self.promote > 0 {
                    flags |= OPEN_FALLBACK;
                }
                m.flags &= crate::porta::FLAG_BITS;
                m.note_id_hi &= NOTE_HI_MASK;
                flags |= REC_GLIDE;
                self.glides += 1;
            }
            let base = self.apply_channel(&mut m);
            if self.mirror && self.promote > 0 && flags & OPEN_FALLBACK == 0 {
                // Counted on the resolved command, as merlin counts it.
                let n = self.fb_uses.entry(PromoteKey::new(&m)).or_insert(0);
                *n += 1;
                if *n < self.promote {
                    flags |= OPEN_FALLBACK;
                }
            }
            if flags & OPEN_FALLBACK != 0 {
                self.fallback_notes += 1;
            }
            let start = block_start + m.start_rel as u64;
            let voice = self.voice_id(&m)?;
            if self.fold(&m, voice, start, base, flags) {
                continue;
            }
            self.open[(m.gate_slot & SLOT_MASK) as usize].push_back(OpenRun {
                start,
                note_id: (((m.note_id_hi & NOTE_HI_MASK) as u64) << 32) | m.note_id_lo as u64,
                voice,
                slot: m.gate_slot,
                ordinal: m.ordinal,
                count: 1,
                flags,
                gain: [m.gain_l, m.gain_r],
                base,
            });
            self.live += 1;
        }
        self.peak_open = self.peak_open.max(self.live);
        self.prof.route += t.elapsed().as_secs_f64();
        Ok(())
    }

    /// File `count` copies of `o`, all released at `end`; `rel` is that frame \[58\]
    fn file(&mut self, o: &OpenRun, count: u16, end: u64, rel: u32) -> Result<()> {
        let c = self.chan_gain_at(o.slot, rel);
        let gain_end = [o.base[0] * c[0], o.base[1] * c[1]];
        let cut = self.cut_at(o.slot, o.note_id, rel);
        let rec = RunRec {
            start: o.start,
            duration: (end - o.start) as u32,
            voice: o.voice,
            slot: o.slot,
            count,
            flags: (o.flags & REC_MASK) | if cut { REC_CUT } else { 0 },
            gain: o.gain,
            gain_end,
        };
        if self.mirror {
            let cmd = self.voices[o.voice as usize].cmd_for(&rec);
            self.digest_all.add(&cmd, rec.start, rec.duration, gain_end, cut, count as u64);
            if o.flags & (OPEN_FALLBACK | OPEN_CLAMPED) == 0 {
                self.digest_merlin.add(&cmd, rec.start, rec.duration, gain_end, cut, count as u64);
            }
        }
        if cut {
            self.cut_notes += count as u64;
        }
        self.runs += 1;
        // [59]
        let tail = self.voice_tail[o.voice as usize] as u64;
        let heard = rec.duration as u64 + self.cfg.steal_fade_frames.max(1) as u64 + tail + 1;
        self.reach_end = self.reach_end.max(rec.start + rec.duration as u64 + self.cfg.steal_fade_frames as u64 + tail + self.two_p);
        let windows = match self.window {
            Some(w) => rec.start / w..=(rec.start + heard - 1) / w,
            None => 0..=0,
        };
        self.store.push(windows, (o.slot & SLOT_MASK) as usize % KEYS, rec)
    }

    /// `MerlinSynth::resolve`, visiting only the slots this block's gates \[60\]
    fn resolve(&mut self) -> Result<()> {
        let t = std::time::Instant::now();
        let block_start = self.block * self.block_frames;
        // The table's last pair closes the last slot, as `release_frame` reads it.
        let slots = (self.off_meta.len() / 2).saturating_sub(1).min(self.open.len());
        for s in 0..slots {
            let (first, end) = (self.off_meta[s * 2 + 1], self.off_meta[(s + 1) * 2 + 1]);
            if end <= first || self.open[s].is_empty() {
                continue;
            }
            let mut q = std::mem::take(&mut self.open[s]);
            let base = self.off_meta[s * 2];
            let offs = self.off_runs[(end as usize - 1) * 2];
            // [61]
            let (mut kept, mut holes) = (false, false);
            for o in q.iter_mut() {
                if o.ordinal > base && o.ordinal - base > offs {
                    break;
                }
                let done = self.release_run(o, block_start)?;
                if done == o.count {
                    o.count = 0;
                    self.live -= 1;
                    holes |= kept;
                } else {
                    if done > 0 {
                        *o = o.part(done, o.count - done);
                    }
                    kept = true;
                }
            }
            while q.front().is_some_and(|o| o.count == 0) {
                q.pop_front();
            }
            if holes {
                q.retain(|o| o.count > 0);
            }
            self.open[s] = q;
        }
        // [62]
        if self.mirror && (!self.clamp_scanned || self.head_frames < self.block_frames) {
            self.mirror_clamp(block_start);
            self.clamp_scanned = true;
        }
        self.prof.resolve += t.elapsed().as_secs_f64();
        Ok(())
    }

    /// File what this block releases of one run, from its front, and say how \[63\]
    fn release_run(&mut self, o: &OpenRun, block_start: u64) -> Result<u16> {
        let mut done: u16 = 0;
        while done < o.count {
            let at = |k: u16| self.release_frame(o, k, block_start).map(|f| f.max(o.start));
            let Some(end) = at(done) else { break };
            let mut k: u16 = 1;
            while done + k < o.count && at(done + k) == Some(end) {
                k += 1;
            }
            let rel = end.saturating_sub(block_start).min(self.block_frames - 1) as u32;
            self.file(&o.part(done, k), k, end, rel)?;
            done += k;
        }
        Ok(done)
    }

    /// Merlin's head-cache ceiling, for its digest only. A note merlin still \[64\]
    fn mirror_clamp(&mut self, block_start: u64) {
        let (bf, hf) = (self.block_frames, self.head_frames);
        for q in self.open.iter_mut() {
            for o in q.iter_mut() {
                if o.flags & (OPEN_FALLBACK | OPEN_CLAMPED) != 0 {
                    continue;
                }
                let held = (block_start + bf).wrapping_sub(o.start);
                if held <= hf {
                    continue;
                }
                let mut cmd = self.voices[o.voice as usize].cmd;
                cmd.gate_slot = o.slot;
                cmd.gain_l = o.gain[0];
                cmd.gain_r = o.gain[1];
                self.digest_merlin.add(&cmd, o.start, hf as u32, o.gain, false, o.count as u64);
                self.clamped_notes += o.count as u64;
                o.flags |= OPEN_CLAMPED;
            }
        }
    }

    /// Record this block's rows where they move. See [`CtlPoint`].
    fn record_curves(&mut self) {
        let t = std::time::Instant::now();
        let block_start = self.block * self.block_frames;
        if self.curves.len() < self.chan_count {
            self.curves.resize_with(self.chan_count, Vec::new);
        }
        for ch in 0..self.chan_count {
            let tiles = if self.bend_active || self.gain_active { self.tiles as usize } else { 1 };
            for tile in 0..tiles {
                let row = (tile * self.chan_count + ch) * CHAN_FIELDS;
                let bend = if self.bend_active { self.chan_rows[row + CHAN_BEND] } else { UNITY.0 };
                let gain = if self.gain_active {
                    [f32::from_bits(self.chan_rows[row + CHAN_GAIN_L]), f32::from_bits(self.chan_rows[row + CHAN_GAIN_R])]
                } else {
                    UNITY.1
                };
                let last = self.curves[ch].last().map_or(UNITY, |p| (p.bend, p.gain));
                if last.0 != bend || last.1[0].to_bits() != gain[0].to_bits() || last.1[1].to_bits() != gain[1].to_bits() {
                    self.curves[ch].push(CtlPoint { frame: block_start + (tile as u64 * self.tile_frames as u64), bend, gain });
                }
            }
        }
        self.prof.curves += t.elapsed().as_secs_f64();
    }

    /// Everything the pass counted, as one line.
    pub fn summary(&self) -> String {
        let points: usize = self.curves.iter().map(Vec::len).sum();
        format!(
            "{} notes in {} runs ({} folded), {} voices; store {:.1} MiB in {} records, peak {:.1} MiB \
             held of {} MiB, {:.1} MiB spilled in {} writes ({:.2}s); peak {} runs open, {} still open; \
             {} curve points on {} channels; {} glides stripped, {} cut by CC120, {} on bent channels; \
             route {:.2}s, resolve and file {:.2}s, curves {:.2}s",
            self.notes,
            self.runs,
            self.folded,
            self.voices.len(),
            (self.store.records as usize * REC) as f64 / 1048576.0,
            self.store.records,
            self.store.peak_held as f64 / 1048576.0,
            self.store.budget() >> 20,
            self.store.spill_bytes() as f64 / 1048576.0,
            self.store.spill_writes,
            self.store.spill_secs,
            self.peak_open,
            self.live,
            points,
            self.curves.iter().filter(|c| !c.is_empty()).count(),
            self.glides,
            self.cut_notes,
            self.bent_notes,
            self.prof.route,
            self.prof.resolve,
            self.prof.curves,
        )
    }
}

impl Backend for BucketPass {
    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()> {
        self.off_meta.clear();
        self.off_meta.extend_from_slice(meta);
        self.off_runs.clear();
        self.off_runs.extend_from_slice(runs);
        Ok(())
    }

    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, _variant: bool, cut: bool) -> Result<()> {
        self.chan_rows.clear();
        self.chan_rows.extend_from_slice(rows);
        self.chan_count = rows.len() / ((self.tiles as usize + 1) * CHAN_FIELDS);
        self.bend_active = bend;
        self.gain_active = gain;
        self.cut_active = cut;
        self.record_curves();
        Ok(())
    }

    fn set_params_variant(&mut self, index: u32, data: Arc<Vec<RegionParams>>, menv: Arc<Vec<ModEnvParams>>) -> Result<()> {
        let Some(i) = index.checked_sub(1).map(|i| i as usize) else {
            return Ok(());
        };
        if self.variants.len() <= i {
            self.variants.resize(i + 1, None);
            self.generation.resize(i + 1, 0);
        }
        self.variants[i] = Some((data, menv));
        self.generation[i] += 1;
        Ok(())
    }

    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        self.open_block();
        self.route(cmds)?;
        self.resolve()?;
        self.block += 1;
        Ok(())
    }

    fn spawn_parts(&self) -> bool {
        true
    }

    fn spawn_part(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        if !self.parts_open {
            self.open_block();
            self.parts_open = true;
        }
        self.route(cmds)?;
        self.resolve()
    }

    fn spawn_end(&mut self) -> Result<()> {
        if !self.parts_open {
            self.open_block();
        }
        self.parts_open = false;
        self.block += 1;
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        out.fill(0.0);
        Ok(())
    }

    fn skip_block(&mut self) -> Result<()> {
        anyhow::bail!("the bucket pass has no pool, so the driver never steps over its blocks")
    }

    /// Outstanding while any run waits for its note-off, which keeps the \[65\]
    fn stats(&self) -> BlockStats {
        BlockStats { active_voices: self.live as u64, ..BlockStats::default() }
    }

    fn name(&self) -> &'static str {
        "merlin-buckets"
    }

    fn has_pool(&self) -> bool {
        false
    }
}

impl Drop for BucketPass {
    fn drop(&mut self) {
        log::info!("merlin bucket pass: {}", self.summary());
        if self.mirror {
            log::info!("merlin bucket pass: filed {}", self.digest_all);
            log::info!(
                "merlin bucket pass: merlin would plan {} ({} notes to its fallback, {} clamped to {:.1}s)",
                self.digest_merlin,
                self.fallback_notes,
                self.clamped_notes,
                self.head_frames as f64 / self.cfg.sample_rate as f64
            );
            let t = std::time::Instant::now();
            match self.stored_digest() {
                Ok(d) => log::info!(
                    "merlin bucket pass: read back {d} in {:.2}s -- {}",
                    t.elapsed().as_secs_f64(),
                    if d == self.digest_all { "the same as filed" } else { "NOT what was filed" }
                ),
                Err(e) => log::error!("merlin bucket pass: reading the store back failed: {e:#}"),
            }
        }
    }
}
