// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Merlin as a [`Backend`], so the existing driver can render through it. \[1\]

use crate::backend::{Backend, BlockStats};
use crate::bank::{Bank, ModEnvParams, RegionParams};
use crate::config::Config;
use crate::merlin::fdl::Fdl;
use crate::merlin::head::{Head, HeadCache, HeadNote, HeadSteps};
use crate::merlin::fft::{Fft, R2cBatch};
use crate::merlin::gpubuild::GpuBuilder;
use crate::merlin::kernel::Kernel;
use crate::merlin::plan::{
    BlockPlan, EntryKey, EntryTable, FilterKey, PlanBounds, PlanScratch, PlannedNote,
    ReleaseBuckets, WaveKey,
};
use crate::midi::CHANNELS;
use crate::voice::{
    SpawnCmd, BASE_CHANNELS, CHAN_BEND, CHAN_FIELDS, CHAN_GAIN_L, CHAN_GAIN_R, NOTE_HI_MASK,
    SLOT_MASK,
};
use anyhow::{bail, Context, Result};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::Arc;

/// Where merlin's wall clock goes, cumulative seconds. \[2\]
#[derive(Default, Clone, Copy)]
struct Profile {
    /// `resolve`, i.e. pairing note-offs and planning every note. Includes the \[3\]
    plan: f64,
    /// `CpuSynth` renders for new or grown waveforms. Also `build_secs`.
    build: f64,
    /// `ensure_kernels`: allocating a missing entry's pages, assembling every \[4\]
    transform: f64,
    /// The still-open-note scan at the top of `render`.
    open: f64,
    /// `BlockPlan::rebuild`: the per-step impulse sort.
    block: f64,
    /// `Fdl::encode_block`, host side -- the ordering check, the uniform and \[5\]
    encode: f64,
    /// `head_pass`, host side: clipping every live head to the step, sorting by \[6\]
    head: f64,
    /// `bus_pass` and `queue.submit`. Submission is asynchronous, so this is \[7\]
    submit: f64,
    /// `read_mixed`: the copy, the `poll(wait)`, the map and the copy out. \[8\]
    read: f64,
    /// The per-voice fallback's whole `render`, which is a second backend's \[9\]
    fallback: f64,
    /// `set_params_variant`, merlin's side: taking a CC71-CC75 params table \[10\]
    variants: f64,
    /// `spawn`'s routing loop: resolving each command's channel, counting its \[11\]
    route: f64,
}

impl Profile {
    fn rows(&self) -> [(&'static str, f64); 11] {
        [
            ("route", self.route),
            ("variants", self.variants),
            ("plan", self.plan),
            ("build", self.build),
            ("transform", self.transform),
            ("open", self.open),
            ("block", self.block),
            ("encode", self.encode),
            ("head", self.head),
            ("read+gpu", self.read + self.submit),
            ("fallback", self.fallback),
        ]
    }

    fn total(&self) -> f64 {
        self.rows().iter().map(|r| r.1).sum()
    }
}

/// Time `f`, adding the elapsed seconds to `slot`.
fn timed<T>(slot: &mut f64, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let out = f();
    *slot += t.elapsed().as_secs_f64();
    out
}

/// One block's mixed output on its way back from the device. \[12\]
struct Readback {
    buf: wgpu::Buffer,
    rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    idx: Option<wgpu::SubmissionIndex>,
}

/// A note-on waiting for its length.
#[derive(Clone, Copy)]
struct Open {
    cmd: SpawnCmd,
    start: u64,
    /// The note's own gains -- velocity, region pan, layer -- *before* the \[13\]
    base: [f32; 2],
    /// Filled once the note is old enough that the emit block has reached it. \[14\]
    row: Option<OpenRow>,
    /// THE PLAN, Phase 1 -- per run, not per note. How many identical copies \[15\]
    count: u16,
}

impl Open {
    /// `count` of this run's copies from the `from`-th on.
    fn part(&self, from: u16, count: u16) -> Open {
        let mut o = *self;
        o.cmd.ordinal = o.cmd.ordinal.wrapping_add(from as u32);
        o.count = count;
        o
    }
}

/// Whether two commands build the same note but for which note it is: equal in \[16\]
fn same_but_note(a: &SpawnCmd, b: &SpawnCmd) -> bool {
    let strip = |c: &SpawnCmd| {
        let mut c = *c;
        c.note_id_lo = 0;
        c.note_id_hi &= !NOTE_HI_MASK;
        c.ordinal = 0;
        c
    };
    bytemuck::bytes_of(&strip(a)) == bytemuck::bytes_of(&strip(b))
}

/// What a still-open note needs in order to sound: no part of it depends on the \[17\]
#[derive(Clone, Copy)]
struct OpenRow {
    wave: u32,
    bus: u32,
    /// The note's own gain, without the channel's, so the block being emitted \[18\]
    base: [f32; 2],
}

/// The waveform-determining words of a spawn command, and nothing else. \[19\]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct VoiceKey {
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

impl VoiceKey {
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

/// A channel table with nothing happening in it: unity gains, no bend, variant \[20\]
fn neutral_chan_rows(tiles: usize) -> Vec<u32> {
    let w = BASE_CHANNELS * CHAN_FIELDS;
    let mut rows = vec![0u32; (tiles + 1) * w];
    for r in rows.chunks_exact_mut(CHAN_FIELDS) {
        r[CHAN_BEND] = crate::fixed::BEND_ONE;
        r[CHAN_GAIN_L] = 1.0f32.to_bits();
        r[CHAN_GAIN_R] = 1.0f32.to_bits();
    }
    rows
}

/// What it takes to rebuild one entry's kernel from nothing. \[21\]
#[derive(Clone, Copy, Default)]
struct EntryInfo {
    /// The dense waveform id the sustain render is held under, which is the \[22\]
    wave_id: u32,
    /// The release reference, and the frames of it the delay line does not hold \[23\]
    dj: u32,
    skip: u32,
    /// Partitions. Zero means this id has never been planned against.
    depth: u32,
}

/// One installed copy of the params table, as `set_params_variant` left it. \[24\]
struct Variant {
    /// What the driver installed, filters and all. This is what `FilterKey` and \[25\]
    params: Arc<Vec<RegionParams>>,
    /// The same with `RP_FILTER` cleared, which is what a kernel is rendered \[26\]
    nofilter: Option<Arc<Vec<RegionParams>>>,
    menv: Arc<Vec<ModEnvParams>>,
}

/// Everything about a voice that does not depend on how long its note is held, \[27\]
struct Voice {
    /// The biquad bus, the head-cache row, and the key `built` holds the \[28\]
    filter: u32,
    wave_id: u32,
    wave: EntryKey,
    /// The region's release tail and the frame its envelope settles, both of \[29\]
    tail: usize,
    flat: u32,
    /// The region's release, as a per-frame factor: what section 3b's scalar \[30\]
    r: f32,
    /// Release bucket to entry id, in first-seen order. Linear, because a \[31\]
    entries: Vec<(u16, u32)>,
}

/// What rebuilding one host sustain render needs, kept for the whole run. \[32\]
struct HostDesc {
    cmd: SpawnCmd,
    /// The filterless params table this waveform was built under, **captured \[33\]
    nf: Arc<Vec<RegionParams>>,
    /// The exact length last held, and what a rebuild reproduces. \[34\]
    frames: u32,
}

/// Host-side sustain renders under a byte budget. \[35\]
#[derive(Default)]
struct HostKernels {
    desc: Vec<Option<HostDesc>>,
    held: Vec<Option<Arc<Kernel>>>,
    res: crate::merlin::residency::Residency,
    bytes: u64,
    peak_bytes: u64,
    budget: u64,
    rebuilds: u64,
    rebuild_secs: f64,
}

impl HostKernels {
    fn new(budget: u64) -> Self {
        Self { budget, ..Default::default() }
    }

    fn reserve(&mut self, id: u32) {
        let n = id as usize + 1;
        if self.desc.len() < n {
            self.desc.resize_with(n, || None);
            self.held.resize(n, None);
        }
        self.res.reserve(id);
    }

    /// The render for `id`, or `None` if it was evicted or never built.
    fn get(&self, id: u32) -> Option<&Arc<Kernel>> {
        self.held.get(id as usize).and_then(|k| k.as_ref())
    }

    /// The length `id` was last held at, which a rebuild has to reach.
    fn frames(&self, id: u32) -> usize {
        self.desc.get(id as usize).and_then(|d| d.as_ref()).map_or(0, |d| d.frames as usize)
    }

    /// Hold `k` for `id`, remembering what it would take to make it again.
    fn store(&mut self, id: u32, k: Arc<Kernel>, nf: Arc<Vec<RegionParams>>) {
        self.reserve(id);
        let i = id as usize;
        if let Some(old) = self.held[i].take() {
            self.bytes -= (old.sustain.len() * 4) as u64;
        }
        self.bytes += (k.sustain.len() * 4) as u64;
        self.peak_bytes = self.peak_bytes.max(self.bytes);
        self.desc[i] = Some(HostDesc {
            cmd: k.cmd,
            nf,
            frames: k.sustain.len() as u32,
        });
        self.held[i] = Some(k);
        self.res.admit(id);
    }

    /// Drop least-recently-used renders until the budget is met. \[36\]
    fn trim(&mut self) -> Option<u32> {
        if self.budget == 0 {
            return Some(0);
        }
        let mut dropped = 0;
        while self.bytes > self.budget {
            let victim = self.res.victim()?;
            if let Some(k) = self.held[victim as usize].take() {
                self.bytes -= (k.sustain.len() * 4) as u64;
            }
            dropped += 1;
        }
        Some(dropped)
    }
}


/// Prices the design's option 2 for CC7 without changing a sample. \[37\]
#[derive(Default)]
struct TailProbe {
    /// `(entry, channel)` packed, to the step its tail dies. \[38\]
    live: HashMap<u64, u64>,
    /// `(waveform, channel, fire epoch)` packed, to the step its tail dies. \[39\]
    cells: HashMap<u64, u64>,
    /// Keys expiring at each step, indexed by `expiry % len`.
    expire: Vec<Vec<u64>>,
    cell_expire: Vec<Vec<u64>>,
    per_chan: Vec<u32>,
    cell_per_chan: Vec<u32>,
    /// Impulses option 2 would have fired, and the worst single block.
    total: u64,
    worst_block: u64,
    cell_total: u64,
    cell_worst_block: u64,
    cell_peak: usize,
    /// Live pairs, summed per block and at peak, for the state cost.
    live_sum: u64,
    live_peak: usize,
    blocks: u64,
}

impl TailProbe {
    fn new(lags: usize) -> Self {
        Self {
            expire: vec![Vec::new(); lags + 2],
            cell_expire: vec![Vec::new(); lags + 2],
            per_chan: vec![0; CHANNELS],
            cell_per_chan: vec![0; CHANNELS],
            ..Default::default()
        }
    }

    #[inline]
    fn cell_key(wave: u32, chan: usize, epoch: u64) -> u64 {
        ((wave as u64 & 0xFF_FFFF) << 40) | ((chan as u64 & 0xFF) << 32) | (epoch & 0xFFFF_FFFF)
    }

    #[inline]
    fn key(entry: u32, chan: usize) -> u64 {
        ((entry as u64) << 8) | chan as u64
    }

    /// An entry fired on a channel at `step`, with a kernel `depth` steps long. \[40\]
    fn fire(&mut self, entry: u32, wave: u32, chan: usize, step: u64, depth: u32, epoch: u64) {
        let ch = chan.min(CHANNELS - 1);
        let expiry = step + depth.max(1) as u64;
        let n = self.expire.len();

        let k = Self::key(entry, chan);
        if self.live.insert(k, expiry).is_none() {
            self.per_chan[ch] += 1;
        }
        self.expire[(expiry % n as u64) as usize].push(k);

        let c = Self::cell_key(wave, chan, epoch);
        if self.cells.insert(c, expiry).is_none() {
            self.cell_per_chan[ch] += 1;
        }
        self.cell_expire[(expiry % n as u64) as usize].push(c);
    }

    /// Retire whatever died at `step`.
    fn advance(&mut self, step: u64) {
        let n = self.expire.len();
        let slot = (step % n as u64) as usize;

        let mut bucket = std::mem::take(&mut self.expire[slot]);
        for k in bucket.drain(..) {
            // Refired since, so this entry in the bucket is stale.
            if self.live.get(&k) == Some(&step) {
                self.live.remove(&k);
                let ch = (k & 0xFF) as usize;
                self.per_chan[ch.min(CHANNELS - 1)] -= 1;
            }
        }
        self.expire[slot] = bucket;

        let mut cb = std::mem::take(&mut self.cell_expire[slot]);
        for k in cb.drain(..) {
            if self.cells.get(&k) == Some(&step) {
                self.cells.remove(&k);
                let ch = ((k >> 32) & 0xFF) as usize;
                self.cell_per_chan[ch.min(CHANNELS - 1)] -= 1;
            }
        }
        self.cell_expire[slot] = cb;
    }

    /// Charge one block: `changes[ch]` gain moves, each correcting every pair \[41\]
    fn charge(&mut self, changes: &[u32; CHANNELS]) {
        let mut block = 0u64;
        for (n, live) in changes.iter().zip(self.per_chan.iter()) {
            block += *n as u64 * *live as u64;
        }
        self.total += block;
        self.worst_block = self.worst_block.max(block);

        let mut cblock = 0u64;
        for (n, live) in changes.iter().zip(self.cell_per_chan.iter()) {
            cblock += *n as u64 * *live as u64;
        }
        self.cell_total += cblock;
        self.cell_worst_block = self.cell_worst_block.max(cblock);
        self.cell_peak = self.cell_peak.max(self.cells.len());

        self.live_sum += self.live.len() as u64;
        self.live_peak = self.live_peak.max(self.live.len());
        self.blocks += 1;
    }
}

/// The most section 3b's scalar may amplify a kernel before the note is \[42\]
const MAX_KERNEL_GAIN: f64 = 16.0;

/// Sustains the device's kernel builder renders a batch: rows of one block \[43\]
const GPU_BUILD_ROWS: u32 = 512;

/// Host memory the kernel builder's batches in flight may hold in their \[44\]
const INFLIGHT_BYTES: u64 = 512 << 20;

/// A batch of sustain renders on the device, not yet read: its jobs in \[45\]
struct InFlightBuild {
    seq: u64,
    jobs: Vec<(u32, SpawnCmd, Arc<Vec<RegionParams>>)>,
    groups: Vec<(Vec<usize>, crate::merlin::gpubuild::Batch)>,
    bytes: u64,
}

/// The smallest kernel cache [`fit_video_memory`] will shrink to before it \[46\]
const MIN_KERNEL_CACHE: u64 = 256 << 20;

/// Room left for what the estimate in [`fit_video_memory`] does not count: the \[47\]
const VRAM_SLACK: u64 = 512 << 20;

/// The kernel cache, shrunk if merlin would otherwise ask for more video memory \[48\]
fn fit_video_memory(
    cfg: &Config,
    bank: &Bank,
    adapter: &wgpu::AdapterInfo,
    pool_bytes: u64,
    ring_bytes: u64,
) -> Result<u64> {
    let Some(m) = crate::gpu::vram::sample_quick(adapter.vendor, adapter.device) else {
        return Ok(pool_bytes);
    };
    let (Some(budget), Some(used)) = (m.process_budget, m.process_used) else {
        return Ok(pool_bytes);
    };
    let free = budget.saturating_sub(used);
    let head = cfg.merlin_max_waves as u64 * cfg.merlin_head() as u64 * 4;
    // [49]
    let pool = if cfg.merlin_promote_after > 0 || cfg.merlin_gpu_build { bank.pool_bytes() } else { 0 };
    let fallback = pool
        + if cfg.merlin_promote_after > 0 { crate::gpu::pool_device_bytes(cfg) } else { 0 };
    let rest = ring_bytes + head + fallback + VRAM_SLACK;
    let mib = |b: u64| b >> 20;
    if pool_bytes + rest <= free {
        return Ok(pool_bytes);
    }
    let fits = free.saturating_sub(rest);
    if fits < MIN_KERNEL_CACHE {
        bail!(
            "merlin needs about {} MiB of video memory (kernel cache {}, ring {}, head cache {}, \
             fallback {}, slack {}) and the OS lets this process use {} MiB of it right now; other \
             programs hold the rest -- video players, browsers and games all use it. Close some of \
             them, lower --merlin-head-ms, or render with --backend gpu, which needs less.",
            mib(MIN_KERNEL_CACHE + rest),
            mib(MIN_KERNEL_CACHE),
            mib(ring_bytes),
            mib(head),
            mib(fallback),
            mib(VRAM_SLACK),
            mib(free),
        );
    }
    log::warn!(
        "merlin: kernel cache {} -> {} MiB to fit the {} MiB of video memory this process may \
         use now (other programs hold the rest). The output is the same; a render that evicts \
         more is slower.",
        mib(pool_bytes),
        mib(fits),
        mib(free),
    );
    Ok(fits)
}

/// The host sustain store's budget, capped at a quarter of the RAM that is free \[50\]
fn fit_host_memory(cfg: &Config) -> u64 {
    let want = cfg.merlin_host_mb as u64 * (1 << 20);
    if want == 0 {
        return 0;
    }
    let Some(m) = crate::falconeye::winsys::memory() else {
        return want;
    };
    let cap = (m.available / 4).max(256 << 20);
    if want <= cap {
        return want;
    }
    log::warn!(
        "merlin: host sustain store {} -> {} MiB, a quarter of the {} MiB of RAM free now. The \
         output is the same; a render that evicts more is slower.",
        want >> 20,
        cap >> 20,
        m.available >> 20,
    );
    cap
}

/// Partitions one batched kernel transform takes at most, and so the size of \[51\]
const KERNEL_BATCH_ROWS: usize = 2048;

/// A kernel `ensure_kernels` has allocated pages for and not yet filled.
struct KernelJob {
    kern: Arc<Kernel>,
    slot: crate::merlin::fdl::KernelSlot,
    dj: usize,
    skip: usize,
    depth: usize,
}

pub struct MerlinSynth {
    cfg: Config,
    bank: Arc<Bank>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The adapter's name and PCI vendor and device id, for the front ends' \[52\]
    adapter_name: String,
    adapter_ids: (u32, u32),
    /// The real transform the kernel cache is built with, and the buffers it \[53\]
    fft: Fft,
    kbatch: R2cBatch,
    fdl: Fdl,

    /// Partition size, and how many FDL steps make one driver block.
    p: usize,
    steps: usize,

    entries: EntryTable,
    /// Distinct biquads, each owning two output buses. Section 3(c)'s \[54\]
    filters: crate::merlin::plan::Table<FilterKey>,
    buses: usize,
    /// Whether `buses` grows. See `Config::merlin_max_filters`.
    auto_buses: bool,
    /// The most `buses` may reach: what one storage binding holds at `lags`. \[55\]
    max_buses: usize,
    /// The bank's own distinct filters, which automatic sizing starts at.
    static_filters: usize,
    /// How many times `grow_buses` has widened the bus dimension.
    bus_growths: u32,
    /// Set for the lifetime of a step's command encoder. `grow_buses` swaps \[56\]
    encoding: bool,
    /// The bank's params with every filter cleared, for building kernels.
    nofilter: Arc<Vec<crate::bank::RegionParams>>,
    /// Copies of the params table beyond the bank's own, as `set_params_variant` \[57\]
    variants: Vec<Variant>,

    /// This block's channel rows, as `set_channels` published them, and whether \[58\]
    chan_rows: Vec<u32>,
    /// Channels a row of `chan_rows` covers: sixteen per MIDI port the file \[59\]
    chan_count: usize,
    bend_active: bool,
    gain_active: bool,
    /// Whether any channel was silenced by CC120 in this block. See `cut_at`.
    cut_active: bool,
    /// Notes damped by All Sound Off rather than released, cumulative.
    cut_notes: u64,
    tile_frames: u32,
    tiles: u32,
    /// Raw 8.24 bend factor to the quantised one, per channel. \[60\]
    bend_memo: [(u32, u32); CHANNELS],
    /// Distinct voices whose release ran past `tail_cap` and was cut there, and \[61\]
    release_truncated: u64,
    /// Prices the head-correction split. See the report at the end of `render`. \[62\]
    head_total: u64,
    head_still: u64,
    head_span_now: u64,
    head_span_split: u64,
    /// Kernels the ring could not hold in full, and the deepest one asked for. \[63\]
    depth_truncated: u64,
    depth_longest: usize,
    release_longest: usize,
    /// Note-ons whose channel was bent, and how many of those became a \[64\]
    bent_notes: u64,
    buckets: ReleaseBuckets,
    /// Sustain renders, under a byte budget, indexed by the same dense \[65\]
    host: HostKernels,
    /// Everything a kernel needs to be **re**built, indexed by entry id. \[66\]
    entry_info: Vec<EntryInfo>,
    /// Which kernels are on the card, and which one goes when the room runs \[67\]
    res: crate::merlin::residency::Residency,
    /// The per-voice backend that renders waveforms merlin has not promoted. \[68\]
    fallback: Option<Box<crate::gpu::GpuSynth>>,
    /// Notes seen per voice key, which is what promotion counts.
    fb_uses: FxHashMap<VoiceKey, u32>,
    /// The device's kernel builder, or `None` when sustains are rendered on \[69\]
    gpu_build: Option<GpuBuilder>,
    /// Sustain renders asked for since the last `start_builds`, by waveform: \[70\]
    pending_builds: FxHashMap<u32, (SpawnCmd, Arc<Vec<RegionParams>>, usize)>,
    /// A block's commands are arriving in parts (`Backend::spawn_part`) and \[71\]
    parts_open: bool,
    /// Batches queued on the device and not yet read, oldest first, and for \[72\]
    inflight: std::collections::VecDeque<InFlightBuild>,
    inflight_of: FxHashMap<u32, (u64, usize)>,
    inflight_seq: u64,
    inflight_bytes: u64,
    /// The bank's modulation envelopes, shared, for the kernel builder: what \[73\]
    base_menv: Arc<Vec<ModEnvParams>>,
    /// The fallback's output, delayed into merlin's frame. \[74\]
    fb_ring: Vec<Vec<f32>>,
    fb_scratch: Vec<f32>,
    /// Cmds routed this block, reused so a dense block does not allocate.
    fb_batch: Vec<SpawnCmd>,
    fb_notes: u64,
    fb_promoted: u64,
    /// Gliding notes (CC5/CC65/CC84) sent to the fallback because a kernel \[75\]
    glides_fallback: u64,
    glides_flattened: u64,
    /// Notes whose kernel would have needed more than `MAX_KERNEL_GAIN`, and \[76\]
    released_on_head: u64,
    /// High-water marks for the two lists that scale with how many notes are \[77\]
    peak_open: usize,
    /// Head records held at the peak, counted once per partition step they are \[78\]
    peak_head_notes: usize,
    /// Prices CC7's per-entry tail correction. See `TailProbe`.
    probe: Option<TailProbe>,
    /// What `plan_note` was handed, copy by copy, with `--merlin-digest`: the \[79\]
    digest: Option<crate::merlin::bucket::NoteDigest>,
    /// Partition steps encoded, which is the clock every residency table ages \[80\]
    step_clock: u64,
    /// Kernels rebuilt after eviction, and the seconds spent doing it.
    kernel_rebuilds: u64,
    rebuild_secs: f64,
    /// The per-note memo. See [`VoiceKey`].
    voices: Vec<Voice>,
    voice_of: FxHashMap<VoiceKey, u32>,

    /// The head correction: its cache, its pipelines, and the notes awaiting it.
    heads: HeadCache,
    head_scatter: wgpu::ComputePipeline,
    head_merge: wgpu::ComputePipeline,
    head_layout: wgpu::BindGroupLayout,
    head_u: wgpu::Buffer,
    head_acc: wgpu::Buffer,
    head_buf: wgpu::Buffer,
    head_cap: usize,
    /// Records this device can bind at once, which is the hard ceiling on one \[81\]
    head_chunk: usize,
    head_chunks: u64,
    /// The workgroup size both head pipelines were compiled at. Was assumed to \[82\]
    head_wg: u32,
    /// The notes awaiting their correction, filed under each partition step \[83\]
    head_steps: HeadSteps,
    head_truncated: u64,

    /// The per-velocity filter buses and the mix to stereo.
    bus_biquad: wgpu::ComputePipeline,
    bus_mix: wgpu::ComputePipeline,
    bus_layout: wgpu::BindGroupLayout,
    bus_u: wgpu::Buffer,
    bus_coef: wgpu::Buffer,
    bus_enabled: wgpu::Buffer,
    bus_state: wgpu::Buffer,
    bus_mixed: wgpu::Buffer,
    /// The two in-flight readbacks, and which one this block writes.
    rb: [Readback; 2],
    rb_turn: usize,
    /// Scratch for `head_pass`, reused every step. See the counting sort there.
    head_clip: Vec<Head>,
    /// Each open note's correction in this step, if it has one, before they \[84\]
    head_open: Vec<Option<Head>>,
    /// Where each correction of the step comes from -- `2 * record + piece` in \[85\]
    head_srcs: Vec<u32>,
    head_counts: Vec<u32>,
    /// The sort key per clipped record -- its waveform's number among this \[86\]
    head_keys: Vec<u32>,
    head_local: Vec<u32>,
    head_local_stamp: Vec<u64>,
    /// This step's distinct waveforms, in first-sighting order, and the \[87\]
    head_waves: Vec<u32>,
    head_at: Vec<(u32, u32)>,
    /// Index into `head_keys` and `head_srcs` of each correction, in the order \[88\]
    head_order: Vec<u32>,
    /// The block plan and the buffers that build it, reused every step.
    plan_buf: BlockPlan,
    plan_scratch: PlanScratch,
    /// Filters already uploaded, so coefficients are written once.
    bus_uploaded: Vec<bool>,

    open: Vec<Open>,
    /// Impulses waiting to fire, bucketed by the partition step they land in. \[89\]
    planned: Vec<Vec<PlannedNote>>,
    /// Impulses in the ring, so the driver's drain check does not walk it.
    planned_live: usize,
    /// Driver blocks seen by `spawn`.
    block: u64,
    latency_blocks: u64,
    /// Per-channel gains at the start and end of each recent block, ringed. \[90\]
    chan_hist: Vec<[[f32; 4]; CHANNELS]>,
    /// Most recent gate table, as published for the block being spawned: the \[91\]
    off_meta: Vec<u32>,
    off_runs: Vec<u32>,

    stats: BlockStats,
    /// Notes held longer than the head cache and clamped to it. They sound and \[92\]
    clamped_long: u64,
    /// THE PLAN, Phase 1. The runs this block has opened, by the one note-on \[93\]
    run_next: FxHashMap<u64, u32>,
    /// Channels a CC120 cut lands on anywhere in this block. Their note-ons do \[94\]
    cut_in_block: Vec<bool>,
    /// Note-ons folded into an earlier copy's run rather than opened alone.
    run_folded: u64,
    /// Longest kernel any entry may hold, in frames: the bank's longest release \[95\]
    tail_cap: usize,
    /// Merlin's longest note, i.e. the head cache's row length.
    head_frames: usize,
    /// Impulses that resolved to a frame already emitted and were nudged \[96\]
    impulses_late: u64,
    /// Wall time spent rendering kernels on the CPU, and how many. \[97\]
    build_secs: f64,
    /// Where the wall clock goes. See [`Profile`].
    prof: Profile,
    /// Frame past which nothing can still be sounding. \[98\]
    sounding_until: u64,
}

impl MerlinSynth {
    pub fn new(cfg: &Config, bank: Arc<Bank>) -> Result<Self> {
        let (device, queue, adapter_info, limits, has_timestamps) =
            crate::gpu::open_device_info(cfg)?;
        let adapter_name = crate::gpu::adapter_label(&adapter_info);

        // [99]
        let block = cfg.block_frames as usize;
        let p = block.min(2048);
        if !block.is_multiple_of(p) {
            bail!("block_frames {block} must be a multiple of the {p}-frame partition");
        }
        let steps = block / p;

        // [100]
        let fft = Fft::new(&device, &limits, p, 256)?;
        // [101]
        let static_filters = bank
            .params
            .iter()
            .map(FilterKey::new)
            .collect::<std::collections::HashSet<_>>()
            .len()
            .max(1);
        let auto_buses = cfg.merlin_max_filters == 0;
        let buses =
            if auto_buses { static_filters } else { cfg.merlin_max_filters as usize } * 2;
        // [102]
        let tails: Vec<usize> = bank
            .params
            .iter()
            .map(|rp| crate::merlin::kernel::release_frames(rp, cfg))
            .collect();
        let tail_max = tails.iter().copied().max().unwrap_or(0);
        // [103]
        let derived = tail_max.min(30 * cfg.sample_rate as usize);
        // [104]
        let tail_cap = if cfg.merlin_kernel_frames == 0 {
            derived.max(p)
        } else {
            (cfg.merlin_kernel_frames as usize).max(p)
        };
        if tail_cap < tail_max {
            log::warn!(
                "merlin: the bank's longest release is {:.2} s but kernels are capped at \
                 {:.2} s, so those regions stop mid-decay. Raise --merlin-kernel-ms.",
                tail_max as f64 / cfg.sample_rate as f64,
                tail_cap as f64 / cfg.sample_rate as f64,
            );
        }
        // [105]
        let flats: Vec<usize> = bank
            .params
            .iter()
            .map(|rp| crate::merlin::kernel::flat_from(rp, cfg))
            .collect();
        let deepest = tails
            .iter()
            .zip(&flats)
            .map(|(&t, &f)| f + t.min(tail_cap))
            .max()
            .unwrap_or(tail_cap)
            .max(tail_cap);
        // [106]
        let ring_lag_bytes = buses as u64 * fft.spec_len() as u64 * 8;
        let ring_cap =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let fits = (ring_cap / ring_lag_bytes.max(1)) as usize;
        let lags = deepest.div_ceil(p).min(fits.max(1)).max(1);
        if deepest.div_ceil(p) > lags {
            log::warn!(
                "merlin: this bank's deepest kernel is {:.1} s ({} partitions) and the ring \
                 holds {:.1} s ({lags}) -- one storage binding is {:.2} GiB at {buses} buses{}. \
                 Those tails stop early and the count is reported at the end.",
                deepest as f64 / cfg.sample_rate as f64,
                deepest.div_ceil(p),
                (lags * p) as f64 / cfg.sample_rate as f64,
                (fits as u64 * ring_lag_bytes) as f64 / 1073741824.0,
                if auto_buses {
                    ", which is this bank's own filters with nothing to spare. The regions \
                     that lose are the ones whose envelope never settles"
                } else {
                    ". --merlin-filters halves the buses and doubles this"
                },
            );
        }
        // [107]
        let max_buses = (if auto_buses {
            ((ring_cap / (lags as u64 * fft.spec_len() as u64 * 8)) as usize)
                .min(u16::MAX as usize - 1)
                & !1
        } else {
            buses
        })
        .max(buses);

        // [108]
        if buses > u16::MAX as usize {
            bail!(
                "--merlin-filters {} means {buses} buses, and a head record holds the bus \
                 index in 16 bits. The ring would want {:.1} GiB at this count anyway.",
                cfg.merlin_max_filters,
                buses as f64 * 8.0 * 1024.0 / 1073741824.0,
            );
        }
        if cfg.steal_fade_frames > u16::MAX as u32 {
            bail!(
                "--steal-fade-frames {} does not fit the 16 bits a head record holds the \
                 CC120 damp in",
                cfg.steal_fade_frames,
            );
        }

        // [109]
        let pool_bytes = if cfg.merlin_cache_mb > 0 {
            cfg.merlin_cache_mb as u64 * (1 << 20)
        } else {
            let mean = if tails.is_empty() {
                tail_cap
            } else {
                tails.iter().map(|&t| t.min(tail_cap)).sum::<usize>() / tails.len()
            };
            let per = (mean.div_ceil(p) + 1) as u64
                * fft.spec_len() as u64
                * std::mem::size_of::<crate::merlin::fft::C32>() as u64;
            // [110]
            (cfg.merlin_max_entries as u64 * per).min(2 << 30)
        };
        let pool_bytes = fit_video_memory(cfg, &bank, &adapter_info, pool_bytes, lags as u64 * ring_lag_bytes)?;
        // [111]
        let bind_cap = if cfg.merlin_max_binding_kb > 0 {
            cfg.merlin_max_binding_kb as u64 * 1024
        } else {
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size)
        };
        let fdl = Fdl::new(
            &device,
            &limits,
            p,
            lags,
            cfg.merlin_max_entries as usize,
            buses,
            cfg.merlin_max_fired as usize,
            cfg.merlin_max_impulses as usize,
            pool_bytes,
            bind_cap,
        )?;
        // [112]
        let kbatch_rows = ((bind_cap / (fft.spec_len() as u64 * 8)) as usize)
            .min(KERNEL_BATCH_ROWS)
            .max(lags);
        let kbatch = R2cBatch::new(&device, &fft, kbatch_rows);

        // [113]
        let wg = 256u32.min(limits.max_compute_invocations_per_workgroup);
        let head_src =
            include_str!("../../shaders/head.wgsl").replace("{{HEAD_WG}}", &wg.to_string());
        let head_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("merlin::head"),
            source: wgpu::ShaderSource::Wgsl(head_src.into()),
        });
        let head_layout = crate::gpu::device::bind_layout(
            &device,
            "merlin::head",
            &[true, true, true, false, false],
        );
        let head_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("merlin::head"),
            bind_group_layouts: &[&head_layout],
            push_constant_ranges: &[],
        });
        let head_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("merlin::head::{entry}")),
                layout: Some(&head_pl),
                module: &head_mod,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (head_scatter, head_merge) = (head_pipe("scatter"), head_pipe("merge"));

        let bus_src =
            include_str!("../../shaders/bus.wgsl").replace("{{BUS_WG}}", &wg.to_string());
        let bus_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("merlin::bus"),
            source: wgpu::ShaderSource::Wgsl(bus_src.into()),
        });
        let bus_layout = crate::gpu::device::bind_layout(
            &device,
            "merlin::bus",
            &[true, true, true, false, false, false],
        );
        let bus_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("merlin::bus"),
            bind_group_layouts: &[&bus_layout],
            push_constant_ranges: &[],
        });
        let bus_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("merlin::bus::{entry}")),
                layout: Some(&bus_pl),
                module: &bus_mod,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (bus_biquad, bus_mix) = (bus_pipe("biquad"), bus_pipe("mix"));
        let mk_buf = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let filters = buses / 2;
        let bus_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merlin::bus::u"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // [114]
        let copy = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let bus_coef = mk_buf("merlin::bus::coef", (filters * 16) as u64, copy);
        let bus_enabled = mk_buf("merlin::bus::enabled", (filters * 4) as u64, copy);
        let bus_state = mk_buf("merlin::bus::state", (buses * 8) as u64, copy);
        let bus_mixed = mk_buf("merlin::bus::mixed", (2 * p * 4) as u64, wgpu::BufferUsages::COPY_SRC);
        let readback: [Readback; 2] = std::array::from_fn(|i| Readback {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("merlin::readback{i}")),
                size: (steps * 2 * p * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            rx: None,
            idx: None,
        });
        // [115]
        let head_frames = cfg.merlin_head() as usize;
        let head_bytes = cfg.merlin_max_waves as u64 * head_frames as u64 * 4;
        let head_cap =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if head_bytes > head_cap {
            bail!(
                "the head cache needs {:.2} GiB ({} waveforms x {head_frames} frames) but one \
                 storage binding holds {:.2} GiB. Lower --merlin-waves or --merlin-head-ms.",
                head_bytes as f64 / 1073741824.0,
                cfg.merlin_max_waves,
                head_cap as f64 / 1073741824.0,
            );
        }
        let heads = HeadCache::new(&device, cfg.merlin_max_waves as usize, head_frames);
        // [116]
        log::info!(
            "merlin: kernels up to {:.2}s ({lags} partitions, {:.0} MiB of ring), longest note \
             {:.1}s ({:.2} GiB of head cache), kernel cache {:.2} GiB",
            tail_cap as f64 / cfg.sample_rate as f64,
            (buses * lags * fft.spec_len() * 8) as f64 / 1048576.0,
            head_frames as f64 / cfg.sample_rate as f64,
            head_bytes as f64 / 1073741824.0,
            pool_bytes as f64 / 1073741824.0,
        );
        // [117]
        log::info!(
            "merlin: {buses} filter buses ({}) for a bank with {static_filters} distinct \
             filters, growing to at most {max_buses} ({:.0} MiB of ring at that width)",
            if auto_buses { "automatic" } else { "pinned by --merlin-filters" },
            (max_buses * lags * fft.spec_len() * 8) as f64 / 1048576.0,
        );
        let head_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merlin::head::u"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let head_acc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merlin::head::acc"),
            size: (buses * p * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // [118]
        let head_chunk = ((bind_cap / std::mem::size_of::<Head>() as u64) as usize / wg as usize
            * wg as usize)
            .max(wg as usize);
        // [119]
        let head_cap = (cfg.merlin_max_impulses as usize).min(head_chunk);
        let head_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merlin::head::notes"),
            size: (head_cap * std::mem::size_of::<Head>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let nofilter = Arc::new(crate::merlin::kernel::nofilter_params(&bank.params));
        let tiles = cfg.block_frames / cfg.gate_frames;

        let buckets = ReleaseBuckets::from_tail(
            cfg.merlin_release_tail,
            cfg.sample_rate,
            cfg.merlin_release_buckets,
        );
        // [120]
        let latency_blocks = 1 + (buckets.width as u64).div_ceil(cfg.block_frames as u64);
        // [121]
        log::info!(
            "merlin: {} release buckets over {:.2}s, {} frames each -- emit lag {} blocks              ({:.0} ms)",
            cfg.merlin_release_buckets,
            cfg.merlin_release_tail,
            buckets.width,
            latency_blocks,
            latency_blocks as f64 * cfg.block_frames as f64 * 1000.0 / cfg.sample_rate as f64,
        );

        // [122]
        let want_fallback = cfg.merlin_promote_after > 0;
        let shared = if want_fallback || cfg.merlin_gpu_build {
            let phase = crate::phase::PhaseBank::prepare(&bank, &cfg.phase)?;
            Some(crate::gpu::GpuShared::on_device(
                cfg,
                &bank,
                &phase,
                device.clone(),
                queue.clone(),
                adapter_info.clone(),
                limits.clone(),
                has_timestamps,
            )?)
        } else {
            None
        };
        let gpu_build = match (&shared, cfg.merlin_gpu_build) {
            (Some(sh), true) => Some(GpuBuilder::new(cfg, bank.clone(), sh, GPU_BUILD_ROWS)?),
            _ => None,
        };

        Ok(Self {
            cfg: cfg.clone(),
            bank: bank.clone(),
            device: device.clone(),
            queue: queue.clone(),
            adapter_name,
            adapter_ids: (adapter_info.vendor, adapter_info.device),
            fft,
            kbatch,
            fdl,
            p,
            steps,
            entries: EntryTable::new(),
            filters: Default::default(),
            buses,
            auto_buses,
            max_buses,
            static_filters,
            bus_growths: 0,
            encoding: false,
            nofilter,
            variants: Vec::new(),
            // [123]
            chan_rows: neutral_chan_rows(tiles as usize),
            chan_count: BASE_CHANNELS,
            bend_active: false,
            gain_active: false,
            cut_active: false,
            cut_notes: 0,
            tile_frames: cfg.gate_frames,
            tiles,
            bend_memo: [(crate::fixed::BEND_ONE, crate::fixed::BEND_ONE); CHANNELS],
            release_truncated: 0,
            head_total: 0,
            head_still: 0,
            head_span_now: 0,
            head_span_split: 0,
            depth_truncated: 0,
            depth_longest: 0,
            release_longest: 0,
            bent_notes: 0,
            buckets,
            fallback: if cfg.merlin_promote_after > 0 {
                // [124]
                let mut fc = cfg.clone();
                // [125]
                fc.max_voices = cfg.max_voices;
                let sh = shared.as_ref().context("the fallback's sample pool was not uploaded")?;
                let f = crate::gpu::GpuSynth::on_shared(&fc, bank.clone(), sh)?;
                f.log_buffers();
                Some(Box::new(f))
            } else {
                None
            },
            fb_uses: FxHashMap::default(),
            gpu_build,
            pending_builds: FxHashMap::default(),
            parts_open: false,
            inflight: std::collections::VecDeque::new(),
            inflight_of: FxHashMap::default(),
            inflight_seq: 0,
            inflight_bytes: 0,
            base_menv: Arc::new(bank.menv.clone()),
            fb_ring: vec![
                vec![0.0; cfg.block_samples()];
                latency_blocks as usize + 3
            ],
            fb_scratch: vec![0.0; cfg.block_samples()],
            fb_batch: Vec::new(),
            fb_notes: 0,
            fb_promoted: 0,
            glides_fallback: 0,
            glides_flattened: 0,
            released_on_head: 0,
            peak_open: 0,
            peak_head_notes: 0,
            probe: if cfg.merlin_probe_tails { Some(TailProbe::new(lags)) } else { None },
            digest: cfg.merlin_digest.then(Default::default),
            host: HostKernels::new(fit_host_memory(cfg)),
            entry_info: vec![EntryInfo::default(); cfg.merlin_max_entries as usize],
            res: Default::default(),
            step_clock: 0,
            kernel_rebuilds: 0,
            rebuild_secs: 0.0,
            voices: Vec::new(),
            voice_of: FxHashMap::default(),
            heads,
            head_scatter,
            head_merge,
            head_layout,
            head_u,
            head_acc,
            head_buf,
            head_cap,
            head_chunk,
            head_chunks: 0,
            head_wg: wg,
            head_steps: HeadSteps::new(p as u32),
            head_truncated: 0,
            bus_biquad,
            bus_mix,
            bus_layout,
            bus_u,
            bus_coef,
            bus_enabled,
            bus_state,
            bus_mixed,
            rb: readback,
            rb_turn: 0,
            head_clip: Vec::new(),
            head_open: Vec::new(),
            head_srcs: Vec::new(),
            head_counts: Vec::new(),
            head_keys: Vec::new(),
            head_local: Vec::new(),
            head_local_stamp: Vec::new(),
            head_waves: Vec::new(),
            head_at: Vec::new(),
            head_order: Vec::new(),
            plan_buf: BlockPlan::default(),
            plan_scratch: PlanScratch::default(),
            bus_uploaded: vec![false; buses / 2],
            open: Vec::new(),
            planned: vec![Vec::new(); (latency_blocks as usize + 3) * steps],
            planned_live: 0,
            block: 0,
            latency_blocks,
            chan_hist: vec![
                [[1.0; 4]; CHANNELS];
                latency_blocks as usize + 3
            ],
            off_meta: vec![0; (BASE_CHANNELS * 128 + 1) * 2],
            off_runs: Vec::new(),
            stats: BlockStats::default(),
            clamped_long: 0,
            run_next: FxHashMap::default(),
            cut_in_block: Vec::new(),
            run_folded: 0,
            tail_cap,
            head_frames,
            impulses_late: 0,
            build_secs: 0.0,
            prof: Profile::default(),
            sounding_until: 0,
        })
    }

    /// Frames of leading output the caller must discard. \[126\]
    pub fn latency_frames(&self) -> u64 {
        (self.latency_blocks + 1) * self.cfg.block_frames as u64
    }

    /// The adapter this renders on, named as `GpuSynth::adapter_name` names it.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// PCI vendor and device id of the adapter this renders on.
    pub fn adapter_ids(&self) -> (u32, u32) {
        self.adapter_ids
    }

    /// Notes clamped to the head cache because they were held longer than it. \[127\]
    pub fn clamped_long(&self) -> u64 {
        self.clamped_long
    }

    /// What `plan_note` was handed, with `Config::merlin_digest`.
    pub fn plan_digest(&self) -> Option<crate::merlin::bucket::NoteDigest> {
        self.digest
    }

    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    /// The frame this note's slot releases it, if this block's gate table says \[128\]
    fn release_frame(&self, o: &Open, k: u16, block_start: u64) -> Option<u64> {
        // The bits above the slot carry a glide; see `porta`.
        let s = (o.cmd.gate_slot & SLOT_MASK) as usize;
        if s + 1 >= self.off_meta.len() / 2 {
            return None;
        }
        let ordinal = o.cmd.ordinal.wrapping_add(k as u32);
        if ordinal <= self.off_meta[s * 2] {
            // [129]
            return None;
        }
        crate::voice::off_frame(&self.off_meta, &self.off_runs, s, ordinal)
            .map(|f| block_start + f as u64)
    }

    /// Resolve whatever this block's gates release, and plan it.
    fn resolve(&mut self, block_start: u64) -> Result<()> {
        let mut still_open = Vec::with_capacity(self.open.len());
        let taken = std::mem::take(&mut self.open);
        // [130]
        let mut moved: Vec<u32> = if self.run_next.is_empty() { Vec::new() } else { vec![u32::MAX; taken.len()] };
        for (index, mut o) in taken.into_iter().enumerate() {
            // [131]
            let mut done: u16 = 0;
            while done < o.count {
                let at = |k: u16| self.release_frame(&o, k, block_start).map(|f| f.max(o.start));
                let Some(end) = at(done) else { break };
                let mut k: u16 = 1;
                while done + k < o.count && at(done + k) == Some(end) {
                    k += 1;
                }
                // [132]
                let rel = end.saturating_sub(block_start).min(self.cfg.block_frames as u64 - 1) as u32;
                self.plan_note(&o.part(done, k), (end - o.start) as u32, Some(rel))?;
                done += k;
            }
            if done == o.count {
                continue;
            }
            if done > 0 {
                o = o.part(done, o.count - done);
            }
            // [133]
            let held = block_start + self.cfg.block_frames as u64 - o.start;
            if held > self.head_frames as u64 {
                self.clamped_long += o.count as u64;
                // [134]
                self.plan_note(&o, self.head_frames as u32, None)?;
                continue;
            }
            if let Some(m) = moved.get_mut(index) {
                *m = still_open.len() as u32;
            }
            still_open.push(o);
        }
        if !moved.is_empty() {
            self.run_next.retain(|_, i| {
                let to = moved[*i as usize];
                *i = to;
                to != u32::MAX
            });
        }
        self.open = still_open;
        Ok(())
    }

    /// The params row a command reads, from its own variant. \[135\]
    fn params_of(&self, variant: u32, base: usize) -> Option<RegionParams> {
        match variant.checked_sub(1) {
            None => self.bank.params.get(base).copied(),
            Some(i) => self
                .variants
                .get(i as usize)
                .and_then(|v| v.params.get(base))
                .copied()
                .or_else(|| self.bank.params.get(base).copied()),
        }
    }

    /// The modulation-envelope row beside it, or `None` when nothing can read \[136\]
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
                .and_then(|v| v.menv.get(base))
                .copied()
                .or_else(fallback),
        }
    }

    /// The filterless params table a kernel for `variant` is rendered through, \[137\]
    fn nofilter_of(&mut self, variant: u32) -> Arc<Vec<RegionParams>> {
        let Some(i) = variant.checked_sub(1).map(|i| i as usize) else {
            return self.nofilter.clone();
        };
        let Some(v) = self.variants.get_mut(i) else {
            return self.nofilter.clone();
        };
        if v.nofilter.is_none() {
            v.nofilter =
                Some(Arc::new(crate::merlin::kernel::nofilter_params(&v.params)));
        }
        v.nofilter.clone().unwrap_or_else(|| self.nofilter.clone())
    }

    /// The 8.24 bend factor rounded to the configured grid. \[138\]
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

    /// Fold this block's channel controllers into one note-on's command. \[139\]
    fn apply_channel(&mut self, c: &mut SpawnCmd) -> [f32; 2] {
        let base = [c.gain_l, c.gain_r];
        if !self.bend_active && !self.gain_active {
            return base;
        }
        let ch = crate::voice::ChannelTable::channel_of(c.gate_slot);
        // [140]
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

    /// The channel gains in force at `frame`, which must be inside the block \[141\]
    fn chan_gain_at(&self, gate_slot: u32, rel: u32) -> [f32; 2] {
        if !self.gain_active {
            return [1.0, 1.0];
        }
        let ch = crate::voice::ChannelTable::channel_of(gate_slot);
        let tile = (rel / self.tile_frames).min(self.tiles.saturating_sub(1));
        let row = (tile as usize * self.chan_count + ch) * CHAN_FIELDS;
        [
            f32::from_bits(self.chan_rows[row + CHAN_GAIN_L]),
            f32::from_bits(self.chan_rows[row + CHAN_GAIN_R]),
        ]
    }

    /// Whether a CC120 in this block silenced this note, and at which frame. \[142\]
    fn cut_at(&self, gate_slot: u32, note_id: u64, rel: u32) -> bool {
        if !self.cut_active {
            return false;
        }
        let ch = crate::voice::ChannelTable::channel_of(gate_slot);
        let tile = (rel / self.tile_frames).min(self.tiles.saturating_sub(1));
        let row = (tile as usize * self.chan_count + ch) * CHAN_FIELDS;
        if self.chan_rows[row + crate::voice::CHAN_CUT] == 0 {
            return false;
        }
        let cut_id = (self.chan_rows[row + crate::voice::CHAN_CUT_ID_HI] as u64) << 32
            | self.chan_rows[row + crate::voice::CHAN_CUT_ID_LO] as u64;
        note_id < cut_id
    }

    /// The filter bus a command's velocity selects, interning it on first sight \[143\]
    fn filter_of(&mut self, p: &RegionParams) -> Result<u32> {
        let filter = self.filters.intern(FilterKey::new(p));
        let need = filter as usize * 2 + 2;
        if need > self.buses {
            if !self.auto_buses {
                bail!(
                    "{} distinct filter settings need {need} buses, but --merlin-filters pins \
                     this instance at {}. Leave the flag out and merlin sizes it: the count \
                     grows as a file needs more, up to what the ring holds at this \
                     soundfont's deepest kernel.",
                    self.filters.len(),
                    self.buses
                );
            }
            // [144]
            if need > self.max_buses {
                bail!(
                    "{} distinct filter settings need {need} buses, and one storage binding \
                     holds {} at this soundfont's kernel depth of {:.1} s. Filters accumulate \
                     for the whole render -- every CC71-CC75 combination is a new biquad and \
                     none is released -- so a controller-heavy file can outrun any ceiling. \
                     Render it with the per-voice fallback on (the default \
                     --merlin-promote-after), which keeps most notes off these buses.",
                    self.filters.len(),
                    self.max_buses,
                    (self.fdl.lags * self.p) as f64 / self.cfg.sample_rate as f64,
                );
            }
            self.grow_buses((self.buses * 2).max(need).min(self.max_buses))?;
        }
        if !self.bus_uploaded[filter as usize] {
            let on = p.flags & crate::bank::RP_FILTER != 0;
            let c: [f32; 4] = if on { [p.b0, p.b1, p.a1, p.a2] } else { [0.0; 4] };
            self.queue
                .write_buffer(&self.bus_coef, filter as u64 * 16, bytemuck::cast_slice(&c));
            self.queue.write_buffer(
                &self.bus_enabled,
                filter as u64 * 4,
                bytemuck::cast_slice(&[on as u32]),
            );
            self.bus_uploaded[filter as usize] = true;
        }
        Ok(filter)
    }

    /// Widen every buffer with a bus dimension to `buses`, keeping its contents. \[145\]
    fn grow_buses(&mut self, buses: usize) -> Result<()> {
        if self.encoding {
            bail!(
                "merlin tried to widen its filter buses while a step's encoder was open; that \
                 encoder has the old buffers bound and what it wrote would be lost. This is a \
                 sequencing bug, not a limit."
            );
        }
        let (old, p) = (self.buses, self.p);
        if buses <= old {
            return Ok(());
        }
        log::info!(
            "merlin: filter buses {old} -> {buses} ({} distinct filters, at most {})",
            self.filters.len(),
            self.max_buses
        );
        self.fdl.grow_buses(&self.device, &self.queue, buses);
        let grown = |label: &str, from: &wgpu::Buffer, old_bytes: u64, bytes: u64| {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("merlin::grow_buses"),
            });
            enc.copy_buffer_to_buffer(from, 0, &b, 0, old_bytes);
            self.queue.submit([enc.finish()]);
            b
        };
        let (of, nf) = ((old / 2) as u64, (buses / 2) as u64);
        let coef = grown("merlin::bus::coef", &self.bus_coef, of * 16, nf * 16);
        let enabled = grown("merlin::bus::enabled", &self.bus_enabled, of * 4, nf * 4);
        let state = grown("merlin::bus::state", &self.bus_state, old as u64 * 8, buses as u64 * 8);
        self.bus_coef = coef;
        self.bus_enabled = enabled;
        self.bus_state = state;
        // [146]
        self.head_acc = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merlin::head::acc"),
            size: (buses * p * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.bus_uploaded.resize(buses / 2, false);
        self.buses = buses;
        self.bus_growths += 1;
        Ok(())
    }

    /// Distinct filters interned, buses allocated, and how many times the bus \[147\]
    pub fn filter_stats(&self) -> (usize, usize, u32) {
        (self.filters.len(), self.buses, self.bus_growths)
    }

    /// The sustain render for a command's waveform, at least `need` frames of it. \[148\]
    fn kernel_for(&mut self, cmd: &SpawnCmd, id: u32, need: usize) -> Result<()> {
        self.host.reserve(id);
        // [149]
        self.host.res.touch(id);
        let held = self.host.get(id).cloned();
        if let Some(k) = held.as_ref() {
            if k.frames() >= need {
                self.host.res.count_hit();
                return Ok(());
            }
        }
        let was = self.host.frames(id);
        let want = need.max(was * 2).max(was);
        // [150]
        let nf = match self.host.desc.get(id as usize).and_then(|d| d.as_ref()) {
            Some(d) => d.nf.clone(),
            None => self.nofilter_of(cmd.variant),
        };
        // [151]
        if self.gpu_build.is_some() {
            // Already on its way at a length that covers this: nothing to ask.
            let coming = self.inflight_of.get(&id).map_or(0, |&(_, f)| f);
            if coming >= need {
                return Ok(());
            }
            let was = was.max(coming);
            let want = need.max(was * 2).max(was);
            let src = held.as_ref().map_or(*cmd, |k| k.cmd);
            let e = self.pending_builds.entry(id).or_insert((src, nf, 0));
            e.2 = e.2.max(want);
            self.host.res.count_miss();
            return Ok(());
        }
        let t = std::time::Instant::now();
        let k = Arc::new(match held.as_ref() {
            Some(k) => k.grown(&self.cfg, &self.bank, Some(&nf), want)?,
            None => Kernel::from_cmd(&self.cfg, &self.bank, cmd, want, Some(&nf))?,
        });
        let secs = t.elapsed().as_secs_f64();
        self.build_secs += secs;
        self.prof.build += secs;
        // [152]
        if held.is_none() && was > 0 {
            self.host.res.count_miss();
            self.host.rebuilds += 1;
            self.host.rebuild_secs += secs;
        } else if held.is_none() {
            self.host.res.count_miss();
        }
        self.host.store(id, k, nf);
        if self.host.trim().is_none() {
            let (bytes, budget) = (self.host.bytes, self.host.budget);
            bail!(
                "this partition step needs more sustain renders held at once than \
                 --merlin-host-mb allows ({:.2} of {:.2} GiB, and everything still held is \
                 needed by this very step). Raise it, or lower the waveform count with \
                 --merlin-bend-cents. Unlike the device caches this one costs a CpuSynth \
                 render per miss, so it is sized generously on purpose.",
                bytes as f64 / 1073741824.0,
                budget as f64 / 1073741824.0,
            );
        }
        Ok(())
    }

    /// Queue every sustain asked for since the last call on the device, as one \[153\]
    fn start_builds(&mut self) -> Result<()> {
        if self.pending_builds.is_empty() || self.gpu_build.is_none() {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let mut jobs: Vec<(u32, SpawnCmd, Arc<Vec<RegionParams>>, usize)> =
            self.pending_builds.drain().map(|(id, (c, nf, f))| (id, c, nf, f)).collect();
        jobs.sort_unstable_by_key(|j| j.0);
        let mut tables: Vec<(Arc<Vec<RegionParams>>, Vec<usize>)> = Vec::new();
        for (i, j) in jobs.iter().enumerate() {
            match tables.iter_mut().find(|g| Arc::ptr_eq(&g.0, &j.2)) {
                Some(g) => g.1.push(i),
                None => tables.push((j.2.clone(), vec![i])),
            }
        }
        let mut groups = Vec::with_capacity(tables.len());
        if let Some(builder) = self.gpu_build.as_mut() {
            for (nf, idx) in tables {
                let list: Vec<(SpawnCmd, usize)> = idx.iter().map(|&i| (jobs[i].1, jobs[i].3)).collect();
                groups.push((idx, builder.begin(&nf, &self.base_menv, &list)?));
            }
        }
        let seq = self.inflight_seq;
        self.inflight_seq += 1;
        for j in &jobs {
            self.inflight_of.insert(j.0, (seq, j.3));
        }
        let bytes: u64 = groups.iter().map(|g| g.1.bytes()).sum();
        self.inflight_bytes += bytes;
        self.inflight.push_back(InFlightBuild {
            seq,
            jobs: jobs.into_iter().map(|(id, c, nf, _)| (id, c, nf)).collect(),
            groups,
            bytes,
        });
        let secs = t.elapsed().as_secs_f64();
        self.build_secs += secs;
        self.prof.build += secs;
        // [154]
        while self.inflight_bytes > INFLIGHT_BYTES && self.inflight.len() > 1 {
            self.collect_build()?;
        }
        Ok(())
    }

    /// Wait for the oldest batch in flight and store what it rendered.
    fn collect_build(&mut self) -> Result<()> {
        let Some(b) = self.inflight.pop_front() else {
            return Ok(());
        };
        let t = std::time::Instant::now();
        self.inflight_bytes -= b.bytes;
        let mut sustains: Vec<Vec<f32>> = vec![Vec::new(); b.jobs.len()];
        if let Some(builder) = self.gpu_build.as_mut() {
            for (idx, batch) in b.groups {
                for (row, &i) in builder.wait(batch)?.into_iter().zip(idx.iter()) {
                    sustains[i] = row;
                }
            }
        }
        for ((id, cmd, nf), sustain) in b.jobs.into_iter().zip(sustains) {
            if self.inflight_of.get(&id).is_some_and(|&(seq, _)| seq == b.seq) {
                self.inflight_of.remove(&id);
            }
            let k = Kernel::from_rendered(&self.cfg, &cmd, sustain, &nf)?;
            self.host.store(id, Arc::new(k), nf);
            if self.host.trim().is_none() {
                bail!(
                    "the renders asked for at once need more than --merlin-host-mb holds ({:.2} GiB). \
                     Raise it, or lower the waveform count with --merlin-bend-cents.",
                    self.host.budget as f64 / 1073741824.0,
                );
            }
        }
        let secs = t.elapsed().as_secs_f64();
        self.build_secs += secs;
        self.prof.build += secs;
        Ok(())
    }

    /// The sustain render for waveform `id`, made again if it was evicted. \[155\]
    fn host_kernel(&mut self, id: u32) -> Result<Arc<Kernel>> {
        // [156]
        if self.pending_builds.contains_key(&id) {
            self.start_builds()?;
        }
        if let Some(&(seq, _)) = self.inflight_of.get(&id) {
            while self.inflight.front().is_some_and(|b| b.seq <= seq) {
                self.collect_build()?;
            }
        }
        self.host.reserve(id);
        if let Some(k) = self.host.get(id).cloned() {
            self.host.res.count_hit();
            self.host.res.touch(id);
            return Ok(k);
        }
        let d = self
            .host
            .desc
            .get(id as usize)
            .and_then(|d| d.as_ref())
            .with_context(|| format!("waveform {id} was never built, so it cannot be remade"))?;
        let (cmd, nf, frames) = (d.cmd, d.nf.clone(), d.frames as usize);
        self.host.res.touch(id);
        self.host.res.count_miss();
        let t = std::time::Instant::now();
        let k = Arc::new(match self.gpu_build.as_mut() {
            Some(b) => {
                let mut rows = b.render(&nf, &self.base_menv, &[(cmd, frames)])?;
                Kernel::from_rendered(&self.cfg, &cmd, rows.pop().unwrap_or_default(), &nf)?
            }
            None => Kernel::from_cmd(&self.cfg, &self.bank, &cmd, frames, Some(&nf))?,
        });
        let secs = t.elapsed().as_secs_f64();
        self.build_secs += secs;
        self.prof.build += secs;
        self.host.rebuilds += 1;
        self.host.rebuild_secs += secs;
        self.host.store(id, k.clone(), nf);
        if self.host.trim().is_none() {
            bail!(
                "this partition step reads more distinct waveforms than --merlin-host-mb \
                 holds at once ({:.2} GiB). Raise it, or lower the waveform count with \
                 --merlin-bend-cents.",
                self.host.budget as f64 / 1073741824.0,
            );
        }
        Ok(k)
    }

    /// Everything a note that is still **open** needs in order to sound. \[157\]
    fn voice_index(&mut self, cmd: &SpawnCmd) -> Result<u32> {
        let vk = VoiceKey::new(cmd);
        if let Some(&i) = self.voice_of.get(&vk) {
            return Ok(i);
        }
        let p = self
            .params_of(cmd.variant, cmd.params as usize)
            .with_context(|| format!("region {} has no params", cmd.region))?;
        let menv = self.menv_of(cmd.variant, cmd.params as usize, &p);
        let filter = self.filter_of(&p)?;
        let wave = EntryKey { wave: WaveKey::new(cmd, &p, menv.as_ref()), release: 0 };
        // [158]
        let raw = crate::merlin::kernel::release_frames(&p, &self.cfg);
        let tail = raw.min(self.tail_cap);
        if tail < raw {
            self.release_truncated += 1;
            self.release_longest = self.release_longest.max(raw);
        }
        let flat = crate::merlin::kernel::flat_from(&p, &self.cfg) as u32;
        // [159]
        let wave_id = self.heads.intern(wave, 0);
        self.kernel_for(cmd, wave_id, tail.div_ceil(self.p) * self.p + self.p)?;
        let i = self.voices.len() as u32;
        self.voices.push(Voice {
            filter,
            wave_id,
            wave,
            tail,
            flat,
            r: p.release_coef,
            entries: Vec::new(),
        });
        self.voice_of.insert(vk, i);
        Ok(i)
    }

    /// The sustain render for memo row `vi`, grown to at least `need`.
    fn kern_at(&mut self, cmd: &SpawnCmd, vi: usize, need: usize) -> Result<()> {
        let id = self.voices[vi].wave_id;
        self.kernel_for(cmd, id, need)
    }

    /// Keep an already-interned open note's waveform ahead of where it is being \[160\]
    fn grow_open(&mut self, o: &Open, need: usize) -> Result<()> {
        let vi = self.voice_index(&o.cmd)? as usize;
        self.kern_at(&o.cmd, vi, need)?;
        let id = self.voices[vi].wave_id;
        self.heads.want_frames(id, need as u32);
        Ok(())
    }

    fn open_row(&mut self, o: &Open, need: usize) -> Result<OpenRow> {
        let vi = self.voice_index(&o.cmd)? as usize;
        self.kern_at(&o.cmd, vi, need)?;
        let v = &self.voices[vi];
        let (id, filter) = (v.wave_id, v.filter);
        self.heads.want_frames(id, need as u32);
        Ok(OpenRow { wave: id, bus: filter * 2, base: o.base })
    }

    /// The channel gains at the start and end of block `block`, from the ring.
    fn chan_span(&self, block: u64, ch: usize) -> [f32; 4] {
        if self.chan_hist.is_empty() {
            return [1.0; 4];
        }
        self.chan_hist[block as usize % self.chan_hist.len()][ch]
    }

    fn plan_note(&mut self, o: &Open, duration: u32, off_rel: Option<u32>) -> Result<()> {
        // [161]
        let vi = self.voice_index(&o.cmd)? as usize;
        let v = &self.voices[vi];
        let (filter, wave, tail, flat, release_r) = (v.filter, v.wave, v.tail, v.flat, v.r);
        let (j, dj) = self.buckets.reference(duration, flat);

        // [162]
        let gain_end = match off_rel {
            None => [o.cmd.gain_l, o.cmd.gain_r],
            Some(r) => {
                let c = self.chan_gain_at(o.cmd.gate_slot, r);
                [o.base[0] * c[0], o.base[1] * c[1]]
            }
        };

        // [163]
        let fade = match off_rel {
            Some(r) if self.cut_at(o.cmd.gate_slot, (((o.cmd.note_id_hi & NOTE_HI_MASK) as u64) << 32) | o.cmd.note_id_lo as u64, r) => {
                self.cut_notes += o.count as u64;
                self.cfg.steal_fade_frames.max(1)
            }
            _ => 0,
        };
        if let Some(d) = self.digest.as_mut() {
            d.add(&o.cmd, o.start, duration, gain_end, fade != 0, o.count as u64);
        }

        // [164]
        if fade == 0
            && self.cfg.release_curve == crate::config::EnvelopeCurve::Exponential
            && dj < duration
            && release_r > 0.0
            && release_r < 1.0
            && tail <= u16::MAX as usize
            && (release_r as f64).powi(dj as i32 - duration as i32) > MAX_KERNEL_GAIN
        {
            return self.release_on_head(o, vi, duration, dj, gain_end);
        }

        // [165]
        let skip = if dj <= duration { dj as usize } else { 0 };
        // [166]
        let want = (dj as usize + tail).saturating_sub(skip).div_ceil(self.p).max(1);
        let depth = want.min(self.fdl.lags);
        if want > depth {
            self.depth_truncated += o.count as u64;
            self.depth_longest = self.depth_longest.max(want);
        }

        // [167]
        let jkey = if dj <= duration { j } else { u16::MAX };

        // [168]
        let need = (duration as usize).max(dj as usize + depth * self.p);
        self.kern_at(&o.cmd, vi, need)?;
        let wave_id = self.voices[vi].wave_id;

        // [169]
        let id = match self.voices[vi].entries.iter().find(|&&(b, _)| b == jkey) {
            Some(&(_, id)) => id,
            None => {
                let id = self.entries.intern(EntryKey { wave: wave.wave, release: jkey });
                if id as usize >= self.entry_info.len() {
                    bail!(
                        "entry {id} exceeds --merlin-entries {}. Raise it -- an entry costs \
                         16 bytes of `kmeta` whether or not its kernel is resident, so this \
                         ceiling is cheap now that the cache evicts. Or quantise bend with \
                         --merlin-bend-cents: {} of this render's note-ons opened on a bent \
                         channel, and a standing bend is a resample rather than anything \
                         that collapses, so each distinct factor is its own entry.",
                        self.entry_info.len(),
                        self.bent_notes
                    );
                }
                // [170]
                self.entry_info[id as usize] =
                    EntryInfo { wave_id, dj, skip: skip as u32, depth: depth as u32 };
                self.voices[vi].entries.push((jkey, id));
                id
            }
        };
        // [171]
        self.res.plan(id);

        // [172]
        let env_ratio = if dj > duration {
            let p = self
                .params_of(o.cmd.variant, o.cmd.params as usize)
                .with_context(|| format!("region {} has no params", o.cmd.region))?;
            // [173]
            let at_dj = crate::merlin::kernel::env_at(&p, &self.cfg, dj);
            if at_dj > 0.0 {
                crate::merlin::kernel::env_at(&p, &self.cfg, duration.saturating_sub(1)) / at_dj
            } else {
                1.0
            }
        } else {
            // [174]
            1.0
        };
        // [175]
        let g = self.buckets.gain(release_r as f64, dj, duration, env_ratio);
        let gain = [gain_end[0] * g, gain_end[1] * g];

        // [176]
        let emit_start = self.emit_start();
        let mut start = o.start + skip as u64;
        if start < emit_start {
            self.impulses_late += o.count as u64;
            start = emit_start;
        }
        let step = (start / self.p as u64) as usize;
        let first = (emit_start / self.p as u64) as usize;
        if step >= first + self.planned.len() {
            bail!(
                "an impulse at frame {start} is {} steps past the block being encoded, which the \
                 {}-step ring cannot hold. `Dj` is never past a note's own note-off, so this \
                 means the emit lag and the ring have drifted apart.",
                step - first,
                self.planned.len()
            );
        }
        let slot = step % self.planned.len();
        if fade == 0 {
            // [177]
            self.planned[slot].push(PlannedNote {
                entry: id,
                start_rel: (start % self.p as u64) as u16,
                filter: filter as u16,
                gain,
                weight: o.count,
            });
            self.planned_live += 1;
            if self.probe.is_some() {
                let ch = crate::voice::ChannelTable::channel_of(o.cmd.gate_slot);
                let wave = self.voices[vi].wave_id;
                // [178]
                let epoch = start / self.buckets.width.max(1) as u64;
                if let Some(pr) = self.probe.as_mut() {
                    pr.fire(id, wave, ch, step as u64, depth as u32, epoch);
                }
            }
        }

        // [179]
        let wave_id = self.voices[vi].wave_id;
        let span = duration.saturating_add(fade);
        self.heads.want_frames(wave_id, span);
        if span > self.heads.frames() {
            self.head_truncated += o.count as u64;
        }
        let from = emit_start.saturating_sub(o.start).min(span as u64) as u32;
        let head = HeadNote {
            wave: wave_id,
            start: o.start,
            from,
            d: duration,
            dj,
            r: release_r,
            gain: [o.cmd.gain_l, o.cmd.gain_r],
            gain_end,
            // [180]
            fade: fade as u16,
            release: 0,
            bus: (filter * 2) as u16,
            weight: o.count,
            env_ratio,
        };
        debug_assert_eq!(
            head.skip(),
            skip as u32,
            "`HeadNote::skip` no longer derives what `plan_note` fired the impulse with"
        );
        self.head_steps.push(head, self.heads.frames());
        self.head_total += o.count as u64;
        if fade == 0 && gain_end == [o.cmd.gain_l, o.cmd.gain_r] {
            self.head_still += o.count as u64;
        }
        self.head_span_now += (span.saturating_sub(from)) as u64 * o.count as u64;
        self.head_span_split += dj.abs_diff(duration) as u64 * o.count as u64;
        // [181]
        let rings_until = if fade == 0 { start + (depth * self.p) as u64 } else { 0 };
        self.sounding_until =
            self.sounding_until.max(rings_until).max(o.start + span as u64);
        Ok(())
    }

    /// The rest of `plan_note` for a note released on the head: no impulse, no \[182\]
    fn release_on_head(
        &mut self,
        o: &Open,
        vi: usize,
        duration: u32,
        dj: u32,
        gain_end: [f32; 2],
    ) -> Result<()> {
        let (tail, filter, r) = {
            let v = &self.voices[vi];
            (v.tail as u32, v.filter, v.r)
        };
        let span = duration.saturating_add(tail);
        // [183]
        self.kern_at(&o.cmd, vi, span as usize)?;
        let wave_id = self.voices[vi].wave_id;
        self.heads.want_frames(wave_id, span);
        if span > self.heads.frames() {
            self.head_truncated += o.count as u64;
        }
        let emit_start = self.emit_start();
        let from = emit_start.saturating_sub(o.start).min(span as u64) as u32;
        self.head_steps.push(
            HeadNote {
                wave: wave_id,
                start: o.start,
                from,
                d: duration,
                dj,
                r,
                gain: [o.cmd.gain_l, o.cmd.gain_r],
                gain_end,
                fade: 0,
                release: tail as u16,
                bus: (filter * 2) as u16,
                weight: o.count,
                // Past `flat_from`, so the envelope ratio is one by construction.
                env_ratio: 1.0,
            },
            self.heads.frames(),
        );
        self.head_total += o.count as u64;
        if gain_end == [o.cmd.gain_l, o.cmd.gain_r] {
            self.head_still += o.count as u64;
        }
        self.head_span_now += (span.saturating_sub(from)) as u64 * o.count as u64;
        self.head_span_split += dj.abs_diff(duration) as u64 * o.count as u64;
        self.released_on_head += o.count as u64;
        self.sounding_until = self.sounding_until.max(o.start + span as u64);
        Ok(())
    }

    /// First frame of the block `render` will emit next. \[184\]
    fn emit_start(&self) -> u64 {
        self.block.saturating_sub(self.latency_blocks) * self.cfg.block_frames as u64
    }

    /// Put every kernel this step fires on the card, building and evicting as \[185\]
    fn ensure_kernels(&mut self, fired: &[u32]) -> Result<()> {
        self.res.begin_step(self.step_clock);
        self.host.res.begin_step(self.step_clock);
        if let Some(pr) = self.probe.as_mut() {
            pr.advance(self.step_clock);
        }
        self.step_clock += 1;
        for &e in fired {
            self.res.pin(e);
        }
        let mut jobs: Vec<KernelJob> = Vec::new();
        for &e in fired {
            if self.res.is_resident(e) {
                self.res.count_hit();
                continue;
            }
            self.res.count_miss();
            let info = self.entry_info[e as usize];
            if info.depth == 0 {
                bail!("entry {e} fired but was never planned, so there is nothing to build");
            }
            // [186]
            let kern = self.host_kernel(info.wave_id)?;
            let t = std::time::Instant::now();
            // [187]
            let slot = loop {
                if let Some(slot) =
                    self.fdl.reserve_kernel(&self.queue, e as usize, info.depth as usize)?
                {
                    break slot;
                }
                let Some(victim) = self.res.victim() else {
                    let (used, pool) = self.fdl.cache_used();
                    let c32 = std::mem::size_of::<crate::merlin::fft::C32>();
                    bail!(
                        "this partition step fires {} entries and they do not fit the kernel \
                         cache at once ({:.2} of {:.2} GiB used, nothing left that is not \
                         needed by this very step). Eviction cannot help a single step: \
                         raise --merlin-cache-mb, or lower the entry count with \
                         --merlin-bend-cents. The number to design against is the peak \
                         block, not the mean -- one test file's peak fires 3,428 distinct \
                         entries, which is 2.55 GiB resident simultaneously.",
                        fired.len(),
                        (used * c32) as f64 / 1073741824.0,
                        (pool * c32) as f64 / 1073741824.0,
                    );
                };
                self.fdl.evict_kernel(victim as usize);
            };
            self.res.admit(e);
            self.kernel_rebuilds += 1;
            jobs.push(KernelJob {
                kern,
                slot,
                dj: info.dj as usize,
                skip: info.skip as usize,
                depth: info.depth as usize,
            });
            let secs = t.elapsed().as_secs_f64();
            self.rebuild_secs += secs;
            self.prof.transform += secs;
        }
        if !jobs.is_empty() {
            let t = std::time::Instant::now();
            self.build_kernels(&jobs)?;
            let secs = t.elapsed().as_secs_f64();
            self.rebuild_secs += secs;
            self.prof.transform += secs;
        }
        // [188]
        Ok(())
    }

    /// Fill the pages `ensure_kernels` reserved: assemble every kernel in \[189\]
    fn build_kernels(&mut self, jobs: &[KernelJob]) -> Result<()> {
        let p = self.p;
        let bins = self.fft.spec_len();
        let cap = self.kbatch.max_rows();
        let mut from = 0;
        while from < jobs.len() {
            // Whole kernels only, and at least one however deep it is.
            let mut to = from;
            let mut rows = 0;
            while to < jobs.len() && (to == from || rows + jobs[to].depth <= cap) {
                rows += jobs[to].depth;
                to += 1;
            }
            let chunk = &jobs[from..to];
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("merlin::kernels"),
            });
            self.kbatch.transform(&self.device, &self.queue, &self.fft, &mut enc, rows, |out| {
                // [190]
                let mut rest = out;
                let mut pieces = Vec::with_capacity(chunk.len());
                for j in chunk {
                    let (piece, tail) = rest.split_at_mut(j.depth * 2 * p);
                    pieces.push(piece);
                    rest = tail;
                }
                chunk.par_iter().zip(pieces.into_par_iter()).try_for_each(|(j, piece)| {
                    j.kern.segment_partitioned(j.dj, j.skip, p, j.depth, piece)
                })
            })?;
            let mut row = 0;
            for j in chunk {
                debug_assert_eq!(
                    j.slot.bytes(),
                    (j.depth * bins * std::mem::size_of::<crate::merlin::fft::C32>()) as u64
                );
                self.fdl.copy_kernel(
                    &mut enc,
                    self.kbatch.spectra(),
                    (row * bins * std::mem::size_of::<crate::merlin::fft::C32>()) as u64,
                    &j.slot,
                );
                row += j.depth;
            }
            self.queue.submit([enc.finish()]);
            from = to;
        }
        Ok(())
    }
}

impl MerlinSynth {
    /// Record the head correction for one partition step. \[191\]
    pub fn chunking(&self) -> ((u64, u64), (usize, usize)) {
        let (imp_chunks, imp_cap) = self.fdl.impulse_chunking();
        ((self.head_chunks, imp_chunks), (self.head_chunk, imp_cap))
    }

    fn head_pass(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        start: u64,
        sustaining: &[(OpenRow, u64, u64, u32, u16)],
        emit_start: u64,
    ) -> Result<()> {
        let cached = self.heads.frames();
        // [192]
        let mut clipped = std::mem::take(&mut self.head_clip);
        clipped.clear();
        // [193]
        let p = self.p as u32;
        let due = self.head_steps.take(start / p as u64);
        // [194]
        let frames = self.cfg.block_frames as f32;
        // [195]
        let emit_block = emit_start / self.cfg.block_frames as u64;
        let mut spans = [[0.0f32; 4]; CHANNELS];
        for (ch, sp) in spans.iter_mut().enumerate() {
            *sp = self.chan_span(emit_block, ch);
        }
        // [196]
        let mut open_heads = std::mem::take(&mut self.head_open);
        open_heads.clear();
        open_heads.par_extend(sustaining.par_iter().map(|&(r, s, e, slot, weight)| {
            let ch = crate::voice::ChannelTable::channel_of(slot);
            let span = spans[ch.min(CHANNELS - 1)];
            let gain = [r.base[0] * span[0], r.base[1] * span[1]];
            let d_gain = [
                (r.base[0] * span[2] - gain[0]) / frames,
                (r.base[1] * span[3] - gain[1]) / frames,
            ];
            crate::merlin::head::sustaining(
                r.wave,
                r.bus,
                gain,
                d_gain,
                emit_start,
                s,
                e,
                start,
                p,
                cached,
                weight as u32,
            )
        }));

        // [197]
        self.heads.res.begin_step(self.step_clock);
        // [198]
        let stamp = self.step_clock + 1;
        let mut waves = std::mem::take(&mut self.head_waves);
        waves.clear();
        let mut keys = std::mem::take(&mut self.head_keys);
        keys.clear();
        let mut srcs = std::mem::take(&mut self.head_srcs);
        srcs.clear();
        let mut counts = std::mem::take(&mut self.head_counts);
        counts.clear();
        counts.push(0);
        let mut split = 0usize;
        let opens = due.len() * 2;
        {
            let (stamps, local) = (&mut self.head_local_stamp, &mut self.head_local);
            let mut number = |wave: u32, n: u32| -> u32 {
                let w = wave as usize;
                if stamps.len() <= w {
                    stamps.resize(w + 1, 0);
                    local.resize(w + 1, 0);
                }
                if stamps[w] != stamp {
                    stamps[w] = stamp;
                    local[w] = waves.len() as u32;
                    waves.push(wave);
                    counts.push(0);
                }
                let k = local[w];
                counts[k as usize + 1] += n;
                k
            };
            for (i, h) in due.iter().enumerate() {
                let pieces = h.pieces(start, p, cached);
                if pieces == 0 {
                    continue;
                }
                let k = number(h.wave, pieces.count_ones());
                for piece in 0..2 {
                    if pieces >> piece & 1 != 0 {
                        keys.push(k);
                        srcs.push((i * 2 + piece) as u32);
                    }
                }
                // [199]
                split += if pieces == 0b11 { h.weight as usize } else { 0 };
            }
            for (j, h) in open_heads.iter().enumerate() {
                if let Some(h) = h {
                    keys.push(number(h.base, 1));
                    srcs.push((opens + j) as u32);
                }
            }
        }
        let total = keys.len();
        for &w in &waves {
            self.heads.res.pin(w);
        }
        let mut at = std::mem::take(&mut self.head_at);
        at.clear();
        for &w in &waves {
            // [200]
            let kern = self.host_kernel(w)?;
            let Some((base, cap, hit)) = self.heads.bind(&self.queue, w, &kern.sustain) else {
                let (used, cap) = self.heads.pages_used();
                bail!(
                    "this partition step reads {} distinct waveforms and none of the head \
                     cache's {cap} pages can be freed -- every one of them is needed by this \
                     very step ({used} in use). Eviction cannot help a single step: raise \
                     --merlin-waves, which buys pages at `--merlin-head-ms * 48 * 4 B` a \
                     row, or lower the waveform count with --merlin-bend-cents.",
                    waves.len(),
                );
            };
            if hit {
                self.heads.res.count_hit();
            } else {
                self.heads.res.count_miss();
            }
            at.push((base, cap));
        }
        let repeats = (total - waves.len()) as u64;
        self.heads.res.count_hits(repeats);
        self.host.res.count_hits(repeats);

        // [201]
        if total > 0 {
            for i in 0..waves.len() {
                counts[i + 1] += counts[i];
            }
            let mut order = std::mem::take(&mut self.head_order);
            order.clear();
            order.resize(total, 0);
            for (i, &k) in keys.iter().enumerate() {
                let slot = &mut counts[k as usize];
                order[*slot as usize] = i as u32;
                *slot += 1;
            }
            clipped.par_extend(order.par_iter().map(|&i| {
                let (src, k) = (srcs[i as usize] as usize, keys[i as usize] as usize);
                let mut h = if src < opens {
                    due[src >> 1].clip(start, p, cached)[src & 1]
                } else {
                    open_heads[src - opens]
                }
                .expect("`srcs` names only corrections that exist");
                (h.base, h.cap) = at[k];
                h
            }));
            self.head_order = order;
        }
        self.head_steps.recycle(due);
        self.head_open = open_heads;
        self.head_keys = keys;
        self.head_srcs = srcs;
        self.head_counts = counts;
        self.head_waves = waves;
        self.head_at = at;

        if clipped.is_empty() {
            self.head_clip = clipped;
            return Ok(());
        }
        // [202]
        if clipped.len().min(self.head_chunk) > self.head_cap {
            let n = clipped
                .len()
                .next_power_of_two()
                .max(self.head_cap * 2)
                .min(self.head_chunk);
            log::info!("merlin: head buffer {} -> {n} corrections", self.head_cap);
            self.head_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("merlin::head::notes"),
                size: (n * std::mem::size_of::<Head>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.head_cap = n;
        }

        // [203]
        let notes: usize = clipped.iter().map(|h| h.weight as usize).sum();
        let scale = (i32::MAX as f32 / (notes - split) as f32).min(1.0e6);
        let bind = crate::gpu::device::bind(
            &self.device,
            &self.head_layout,
            &[
                &self.head_u,
                &self.heads.buffer,
                &self.head_buf,
                &self.head_acc,
                self.fdl.out_buffer(),
            ],
        );
        let wg = self.head_wg;
        let uniforms = |n: usize| {
            [
                n as u32,
                self.p as u32,
                (self.filters.len() * 2).min(self.buses) as u32,
                cached,
                scale.to_bits(),
                0,
                0,
                0,
            ]
        };
        if clipped.len() <= self.head_cap {
            // [204]
            self.queue
                .write_buffer(&self.head_buf, 0, bytemuck::cast_slice(&clipped));
            self.queue
                .write_buffer(&self.head_u, 0, bytemuck::cast_slice(&uniforms(clipped.len())));
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merlin::head::scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.head_scatter);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((clipped.len() as u32).div_ceil(wg).min(65535), 1, 1);
        } else {
            // [205]
            self.head_chunks += 1;
            log::info!(
                "merlin: {} corrections this step, past the {} one binding holds -- \
                 scattering in {} passes",
                clipped.len(),
                self.head_chunk,
                clipped.len().div_ceil(self.head_cap),
            );
            for part in clipped.chunks(self.head_cap) {
                self.queue
                    .write_buffer(&self.head_buf, 0, bytemuck::cast_slice(part));
                self.queue
                    .write_buffer(&self.head_u, 0, bytemuck::cast_slice(&uniforms(part.len())));
                let mut e =
                    self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("merlin::head::chunk"),
                    });
                {
                    let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("merlin::head::scatter"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.head_scatter);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups((part.len() as u32).div_ceil(wg).min(65535), 1, 1);
                }
                self.queue.submit([e.finish()]);
            }
            // [206]
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merlin::head::merge"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.head_merge);
            pass.set_bind_group(0, &bind, &[]);
            let live = (self.filters.len() * 2).min(self.buses).max(1) as u32;
            pass.dispatch_workgroups((self.p as u32).div_ceil(wg), live, 1);
        }
        self.head_clip = clipped;
        Ok(())
    }

    /// Filter each bus, then sum the buses to stereo. \[207\]
    fn bus_pass(&mut self, enc: &mut wgpu::CommandEncoder) {
        // [208]
        let live = (self.filters.len() * 2).min(self.buses) as u32;
        self.queue.write_buffer(
            &self.bus_u,
            0,
            bytemuck::cast_slice(&[self.p as u32, live, 0, 0]),
        );
        let bind = crate::gpu::device::bind(
            &self.device,
            &self.bus_layout,
            &[
                &self.bus_u,
                &self.bus_coef,
                &self.bus_enabled,
                &self.bus_state,
                self.fdl.out_buffer(),
                &self.bus_mixed,
            ],
        );
        let wg = 256u32;
        for (pipe, n, label) in [
            (&self.bus_biquad, live.max(1), "merlin::bus::biquad"),
            (&self.bus_mix, self.p as u32, "merlin::bus::mix"),
        ] {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n.div_ceil(wg), 1, 1);
        }
    }

    /// Collect the readback started a block ago, if there is one. \[209\]
    fn take_mixed(&mut self, slot: usize) -> Result<Option<Vec<f32>>> {
        let Some(rx) = self.rb[slot].rx.take() else {
            return Ok(None);
        };
        let submission_index = self.rb[slot].idx.take();
        self.device.poll(wgpu::PollType::Wait { submission_index, timeout: None })?;
        rx.recv()??;
        let data = {
            let view = self.rb[slot].buf.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        self.rb[slot].buf.unmap();
        Ok(Some(data))
    }

    /// Start the readback for the block just submitted, without waiting on it.
    fn start_readback(&mut self, slot: usize, idx: wgpu::SubmissionIndex) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.rb[slot].buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.rb[slot].rx = Some(rx);
        self.rb[slot].idx = Some(idx);
    }

    /// Notes whose head ran past what the cache holds.
    pub fn head_truncated(&self) -> u64 {
        self.head_truncated
    }

    /// What the two device caches had to throw away: kernel evictions, kernels \[210\]
    pub fn residency_stats(&self) -> (u64, u64, u64) {
        (self.res.evictions, self.kernel_rebuilds, self.heads.res.evictions)
    }
}

impl MerlinSynth {
    /// The top of a block's `spawn`, before any of its commands.
    fn open_block(&mut self) {
        // [211]
        if !self.chan_hist.is_empty() {
            let slot = self.block as usize % self.chan_hist.len();
            let last = self.tiles.saturating_sub(1);
            for ch in 0..self.chan_count {
                let at = |tile: u32| {
                    let row = (tile as usize * self.chan_count + ch) * CHAN_FIELDS;
                    [
                        f32::from_bits(self.chan_rows[row + CHAN_GAIN_L]),
                        f32::from_bits(self.chan_rows[row + CHAN_GAIN_R]),
                    ]
                };
                let (a, b) = (at(0), at(last));
                self.chan_hist[slot][ch] = [a[0], a[1], b[0], b[1]];
            }
        }
        self.fb_batch.clear();
        self.run_next.clear();
        self.cut_in_block.clear();
        self.cut_in_block.resize(self.chan_count, false);
        if self.cut_active {
            for tile in 0..self.tiles as usize {
                for ch in 0..self.chan_count {
                    let row = (tile * self.chan_count + ch) * CHAN_FIELDS;
                    if self.chan_rows[row + crate::voice::CHAN_CUT] != 0 {
                        self.cut_in_block[ch] = true;
                    }
                }
            }
        }
    }

    /// Route commands to the fallback or open them for convolving. A block's \[212\]
    fn fold_into_run(&mut self, m: &SpawnCmd, start: u64, base: [f32; 2]) -> bool {
        let ch = crate::voice::ChannelTable::channel_of(m.gate_slot);
        if self.cut_in_block.get(ch).copied().unwrap_or(true) {
            return false;
        }
        let s = (m.gate_slot & SLOT_MASK) as u64;
        let key = s << 32 | m.ordinal as u64;
        let Some(&i) = self.run_next.get(&key) else { return false };
        let o = &mut self.open[i as usize];
        if o.count < u16::MAX
            && o.start == start
            && m.ordinal == o.cmd.ordinal.wrapping_add(o.count as u32)
            && o.base[0].to_bits() == base[0].to_bits()
            && o.base[1].to_bits() == base[1].to_bits()
            && same_but_note(&o.cmd, m)
        {
            o.count += 1;
            self.run_folded += 1;
            self.run_next.remove(&key);
            self.run_next.insert(s << 32 | m.ordinal.wrapping_add(1) as u64, i);
            return true;
        }
        false
    }

    fn route(&mut self, cmds: &[SpawnCmd]) {
        let block_start = self.block * self.cfg.block_frames as u64;
        // [213]
        let t_route = std::time::Instant::now();
        let mut batch = std::mem::take(&mut self.fb_batch);
        let promote = self.cfg.merlin_promote_after;
        for c in cmds {
            // [214]
            let mut m = *c;
            // [215]
            if m.flags > crate::porta::FLAG_BITS {
                if self.fallback.is_some() {
                    batch.push(*c);
                    self.fb_notes += 1;
                    self.glides_fallback += 1;
                    continue;
                }
                // [216]
                m.flags &= crate::porta::FLAG_BITS;
                m.note_id_hi &= NOTE_HI_MASK;
                self.glides_flattened += 1;
            }
            let base = self.apply_channel(&mut m);
            if promote > 0 {
                let n = self.fb_uses.entry(VoiceKey::new(&m)).or_insert(0);
                *n += 1;
                if *n < promote {
                    // [217]
                    batch.push(*c);
                    self.fb_notes += 1;
                    continue;
                }
                if *n == promote {
                    self.fb_promoted += 1;
                }
            }
            let start = block_start + m.start_rel as u64;
            if self.fold_into_run(&m, start, base) {
                continue;
            }
            let s = (m.gate_slot & SLOT_MASK) as u64;
            self.run_next.insert(s << 32 | m.ordinal.wrapping_add(1) as u64, self.open.len() as u32);
            self.open.push(Open { cmd: m, start, base, row: None, count: 1 });
        }
        self.prof.route += t_route.elapsed().as_secs_f64();
        self.fb_batch = batch;
    }

    /// **And started at once.** The fallback has everything its block needs \[218\]
    fn start_fallback(&mut self) -> Result<()> {
        if let Some(f) = self.fallback.as_mut() {
            let t = std::time::Instant::now();
            f.spawn(&self.fb_batch)?;
            f.submit()?;
            self.prof.fallback += t.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// Plan whatever this block's gates release among the open notes.
    fn plan_block(&mut self) -> Result<()> {
        let block_start = self.block * self.cfg.block_frames as u64;
        // [219]
        let (b0, x0) = (self.prof.build, self.prof.transform);
        let t = std::time::Instant::now();
        let r = self.resolve(block_start);
        let inner = (self.prof.build - b0) + (self.prof.transform - x0);
        self.prof.plan += (t.elapsed().as_secs_f64() - inner).max(0.0);
        r?;
        Ok(())
    }

    /// The end of a block's `spawn`: its renders on their way and the block \[220\]
    fn close_block(&mut self) -> Result<()> {
        // [221]
        self.start_builds()?;
        self.block += 1;
        Ok(())
    }
}

impl Backend for MerlinSynth {
    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()> {
        self.off_meta.clear();
        self.off_meta.extend_from_slice(meta);
        self.off_runs.clear();
        self.off_runs.extend_from_slice(runs);
        // [222]
        if let Some(f) = self.fallback.as_mut() {
            f.set_gates(meta, runs)?;
        }
        Ok(())
    }

    /// Forwarded, and only that: a gliding note never reaches the convolution \[223\]
    fn set_glide(&mut self, active: bool) -> Result<()> {
        if let Some(f) = self.fallback.as_mut() {
            f.set_glide(active)?;
        }
        Ok(())
    }

    /// Take this block's channel rows. `spawn` reads them per note-on; see \[224\]
    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, _v: bool, cut: bool) -> Result<()> {
        self.chan_rows.clear();
        self.chan_rows.extend_from_slice(rows);
        self.chan_count = rows.len() / ((self.tiles as usize + 1) * CHAN_FIELDS);
        self.bend_active = bend;
        self.gain_active = gain;
        self.cut_active = cut;
        // [225]
        if let Some(f) = self.fallback.as_mut() {
            f.set_channels(rows, bend, gain, _v, cut)?;
        }
        Ok(())
    }

    /// Install one copy of the params table. \[226\]
    fn set_params_variant(
        &mut self,
        index: u32,
        data: Arc<Vec<RegionParams>>,
        menv: Arc<Vec<ModEnvParams>>,
    ) -> Result<()> {
        if let Some(f) = self.fallback.as_mut() {
            let t = std::time::Instant::now();
            f.set_params_variant(index, data.clone(), menv.clone())?;
            self.prof.fallback += t.elapsed().as_secs_f64();
        }
        let Some(i) = index.checked_sub(1).map(|i| i as usize) else {
            // Variant 0 is the bank's own and merlin reads it from there.
            return Ok(());
        };
        let t = std::time::Instant::now();
        while self.variants.len() <= i {
            self.variants.push(Variant {
                params: Arc::new(Vec::new()),
                nofilter: None,
                menv: Arc::new(Vec::new()),
            });
        }
        let rebuilt = !self.variants[i].params.is_empty();
        // [227]
        self.variants[i] = Variant {
            params: data,
            nofilter: None,
            menv,
        };
        if rebuilt {
            // [228]
            self.voice_of.retain(|k, _| k.variant != index);
        }
        self.prof.variants += t.elapsed().as_secs_f64();
        Ok(())
    }

    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        self.open_block();
        self.route(cmds);
        self.start_fallback()?;
        self.plan_block()?;
        self.close_block()
    }

    /// Yes: a part is routed and its released notes planned, and the order in \[229\]
    fn spawn_parts(&self) -> bool {
        true
    }

    fn spawn_part(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        if !self.parts_open {
            self.open_block();
            self.parts_open = true;
        }
        self.route(cmds);
        self.plan_block()
    }

    fn spawn_end(&mut self) -> Result<()> {
        if !self.parts_open {
            self.open_block();
        }
        self.parts_open = false;
        self.start_fallback()?;
        self.close_block()
    }

    /// Nothing left to start. The fallback's block went to the device in \[230\]
    fn submit(&mut self) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        out.fill(0.0);
        self.peak_open = self.peak_open.max(self.open.len());
        self.peak_head_notes = self.peak_head_notes.max(self.head_steps.len());

        // [231]
        if self.fallback.is_some() {
            let t = std::time::Instant::now();
            let scratch = &mut self.fb_scratch;
            if let Some(f) = self.fallback.as_mut() {
                f.finish(scratch)?;
            }
            if let Some(fb_block) = self.block.checked_sub(1) {
                let n = self.fb_ring.len() as u64;
                let slot = (fb_block % n) as usize;
                self.fb_ring[slot].copy_from_slice(&self.fb_scratch);
            }
            self.prof.fallback += t.elapsed().as_secs_f64();
        }

        // [232]
        let turn = self.rb_turn;
        let prev = turn ^ 1;
        let t = std::time::Instant::now();
        let ready = self.take_mixed(prev)?;
        self.prof.read += t.elapsed().as_secs_f64();
        if let Some(got) = ready {
            for step in 0..self.steps {
                let base = step * 2 * self.p;
                for f in 0..self.p {
                    out[(step * self.p + f) * 2] = got[base + f];
                    out[(step * self.p + f) * 2 + 1] = got[base + self.p + f];
                }
            }
        }
        self.rb_turn = prev;

        // [233]
        if self.probe.is_some() {
            let mut changes = [0u32; CHANNELS];
            let tiles = self.tiles.max(1) as usize;
            for (ch, moves) in changes.iter_mut().enumerate().take(self.chan_count) {
                let at = |tile: usize| {
                    let row = (tile * self.chan_count + ch) * CHAN_FIELDS;
                    (self.chan_rows[row + CHAN_GAIN_L], self.chan_rows[row + CHAN_GAIN_R])
                };
                let mut prev_row = at(0);
                for t in 1..tiles {
                    let now = at(t);
                    if now != prev_row {
                        *moves += 1;
                        prev_row = now;
                    }
                }
            }
            if let Some(pr) = self.probe.as_mut() {
                pr.charge(&changes);
            }
        }

        // [234]
        if self.fallback.is_some() {
            if let Some(w) = self.block.checked_sub(2 + self.latency_blocks) {
                let n = self.fb_ring.len() as u64;
                let src = &self.fb_ring[(w % n) as usize];
                for (o, s) in out.iter_mut().zip(src.iter()) {
                    *o += *s;
                }
            }
        }

        // [235]
        let Some(emit) = (self.block).checked_sub(1 + self.latency_blocks) else {
            return Ok(());
        };
        let mut last: Option<wgpu::SubmissionIndex> = None;

        let frames = self.cfg.block_frames as usize;
        let block_end = (emit + 1) * frames as u64;

        // [236]
        let t_open = std::time::Instant::now();
        let (b0, x0) = (self.prof.build, self.prof.transform);
        let mut open = std::mem::take(&mut self.open);
        let mut sustaining: Vec<(OpenRow, u64, u64, u32, u16)> = Vec::new();
        for o in open.iter_mut() {
            if o.start >= block_end {
                continue;
            }
            let need = ((block_end - o.start) as usize).min(self.head_frames);
            let row = match o.row {
                Some(r) => {
                    // [237]
                    self.grow_open(o, need)?;
                    r
                }
                None => {
                    let r = self.open_row(o, need)?;
                    o.row = Some(r);
                    r
                }
            };
            sustaining.push((row, o.start, o.start + need as u64, o.cmd.gate_slot, o.count));
        }
        self.open = open;
        let inner = (self.prof.build - b0) + (self.prof.transform - x0);
        self.prof.open += (t_open.elapsed().as_secs_f64() - inner).max(0.0);

        for step in 0..self.steps {
            let start = emit * frames as u64 + (step * self.p) as u64;
            let slot = (start / self.p as u64) as usize % self.planned.len();
            let planned = &self.planned[slot];
            let (p, buses) = (self.p as u32, self.buses);
            // [238]
            let info = &self.entry_info;
            // [239]
            let bounds = PlanBounds {
                max_entry: info.len().saturating_sub(1) as u32,
                max_depth: self.fdl.lags as u32,
            };
            // [240]
            let mut plan = std::mem::take(&mut self.plan_buf);
            let scratch = &mut self.plan_scratch;
            timed(&mut self.prof.block, || {
                plan.rebuild(scratch, planned, start, p, buses, bounds, |e| {
                    info.get(e as usize).map_or(0, |i| i.depth)
                })
            });
            // [241]
            self.ensure_kernels(&plan.fired)?;
            let mut enc = self.device.create_command_encoder(&Default::default());
            // [242]
            self.encoding = true;
            let t = std::time::Instant::now();
            self.fdl.encode_block(
                &self.device,
                &self.queue,
                &mut enc,
                &plan.fired,
                &plan.bus_range,
                &plan.impulses,
                (self.filters.len() * 2).min(self.buses),
            )?;
            self.prof.encode += t.elapsed().as_secs_f64();
            self.plan_buf = plan;
            let t = std::time::Instant::now();
            self.head_pass(&mut enc, start, &sustaining, emit * frames as u64)?;
            self.prof.head += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            self.bus_pass(&mut enc);
            // [243]
            enc.copy_buffer_to_buffer(
                &self.bus_mixed,
                0,
                &self.rb[turn].buf,
                (step * 2 * self.p * 4) as u64,
                (2 * self.p * 4) as u64,
            );
            last = Some(self.queue.submit([enc.finish()]));
            self.encoding = false;
            self.prof.submit += t.elapsed().as_secs_f64();
            // [244]
            self.planned_live -= self.planned[slot].len();
            // [245]
            let mut notes = std::mem::take(&mut self.planned[slot]);
            for n in &notes {
                self.res.unplan(n.entry);
            }
            notes.clear();
            self.planned[slot] = notes;
        }
        if let Some(idx) = last {
            self.start_readback(turn, idx);
        }

        // [246]

        // [247]
        let emitted_through = emit * frames as u64;
        let ringing = emitted_through < self.sounding_until;
        let outstanding = self.planned_live > 0 || !self.open.is_empty() || ringing;
        self.stats.active_voices = outstanding as u64;
        self.stats.peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        Ok(())
    }

    /// Merlin's own counts plus the fallback's stolen and dropped. Merlin never \[248\]
    fn stats(&self) -> BlockStats {
        let mut s = self.stats;
        if let Some(f) = self.fallback.as_ref() {
            let fs = f.stats();
            s.stolen += fs.stolen;
            s.dropped += fs.dropped;
        }
        s
    }

    fn name(&self) -> &'static str {
        "merlin"
    }

    /// No pool, so no cap. Merlin renders every note-on it is given, and \[249\]
    fn has_pool(&self) -> bool {
        false
    }

    fn output_latency_frames(&self) -> u64 {
        self.latency_frames()
    }

    /// Never asked for: the driver steps over a block only for a backend with a \[250\]
    fn skip_block(&mut self) -> Result<()> {
        anyhow::bail!("the merlin backend has no block it can step over without rendering")
    }

    fn timings(&self) -> Vec<(&'static str, f64)> {
        // [251]
        self.prof.rows().iter().map(|&(n, s)| (n, s * 1000.0)).collect()
    }
}

impl Drop for MerlinSynth {
    fn drop(&mut self) {
        let (used, pool) = self.fdl.cache_used();
        log::info!(
            "merlin: {} entries, {} waveforms, {:.2}s building kernels, kernel cache \
             {:.2}/{:.2} GiB, {} heads truncated, {} notes clamped to {:.1}s",
            self.entries.len(),
            self.host.desc.iter().filter(|d| d.is_some()).count(),
            self.build_secs,
            (used * std::mem::size_of::<crate::merlin::fft::C32>()) as f64 / 1073741824.0,
            (pool * std::mem::size_of::<crate::merlin::fft::C32>()) as f64 / 1073741824.0,
            self.head_truncated,
            self.clamped_long,
            self.head_frames as f64 / self.cfg.sample_rate as f64,
        );
        // [252]
        log::info!(
            "merlin: {} distinct filters on {} buses ({}; the bank alone has {}, {} growths, \
             room for {} buses)",
            self.filters.len(),
            self.buses,
            if self.auto_buses { "automatic" } else { "pinned" },
            self.static_filters,
            self.bus_growths,
            self.max_buses,
        );
        // [253]
        log::info!(
            "merlin: host sustain renders {:.2} GiB held at peak of {} budget, {} remade \
             after eviction ({:.2}s), {:.1}% hit",
            self.host.peak_bytes as f64 / 1073741824.0,
            if self.host.budget == 0 {
                "no".to_string()
            } else {
                format!("{:.2} GiB", self.host.budget as f64 / 1073741824.0)
            },
            self.host.rebuilds,
            self.host.rebuild_secs,
            100.0 * self.host.res.hit_rate(),
        );
        // [254]
        if self.cfg.merlin_promote_after > 0 {
            log::info!(
                "merlin: per-voice fallback took {} notes across {} waveforms; {} waveforms                  reached {} uses and were promoted to kernels, {:.2}s in the fallback",
                self.fb_notes,
                self.fb_uses.len() as u64 - self.fb_promoted,
                self.fb_promoted,
                self.cfg.merlin_promote_after,
                self.prof.fallback,
            );
        }
        if self.glides_fallback > 0 {
            log::info!(
                "merlin: {} gliding notes (portamento) rendered on the per-voice fallback, \
                 which a kernel cannot do; they are counted in the fallback's notes above",
                self.glides_fallback
            );
        }
        if self.released_on_head > 0 {
            log::info!(
                "merlin: {} notes released on the head rather than through a kernel: \
                 held so far past their release reference that the kernel would have \
                 needed amplifying more than {}x",
                self.released_on_head,
                MAX_KERNEL_GAIN
            );
        }
        if self.glides_flattened > 0 {
            log::warn!(
                "merlin: {} gliding notes (portamento) were convolved at their own pitch with \
                 the glide stripped, because --merlin-promote-after 0 leaves no per-voice \
                 fallback to glide them on",
                self.glides_flattened
            );
        }
        if let Some(d) = self.digest.as_ref() {
            log::info!("merlin: planned {d}");
        }
        if let Some(pr) = self.probe.as_ref() {
            log::info!(
                "merlin: CC7 tail correction would fire {} impulses, worst block {} --                  that is the real cost, one per (waveform, channel, age bucket). The                  (entry, channel) floor is {} and {}, which is what the design that does                  not work would have cost. Live: {} cells at peak ({:.1} MiB of state),                  {} pairs at peak, {:.0} pairs mean",
                pr.cell_total,
                pr.cell_worst_block,
                pr.total,
                pr.worst_block,
                pr.cell_peak,
                pr.cell_peak as f64 * 4.0 / 1048576.0,
                pr.live_peak,
                pr.live_sum as f64 / pr.blocks.max(1) as f64,
            );
        }
        // [255]
        let (imp_chunks, imp_cap) = self.fdl.impulse_chunking();
        if self.head_chunks > 0 || imp_chunks > 0 {
            log::info!(
                "merlin: {} steps needed more than one head scatter and {} more than one \
                 impulse scatter (this device binds {} corrections or {} impulses at once)",
                self.head_chunks,
                imp_chunks,
                self.head_chunk,
                imp_cap,
            );
        }
        // [256]
        let mb = |n: usize, sz: usize| (n * sz) as f64 / 1048576.0;
        log::info!(
            "merlin: host peak {:.0} MiB sustain renders, {:.0} MiB open notes ({} runs at \
             peak; {} note-ons folded into runs in all), {:.0} MiB head notes ({} at peak); \
             at exit {:.0} MiB planned ({}), {:.0} MiB voice memo ({}), {:.0} MiB entry \
             table ({})",
            self.host.peak_bytes as f64 / 1048576.0,
            mb(self.peak_open, std::mem::size_of::<Open>()),
            self.peak_open,
            self.run_folded,
            mb(self.peak_head_notes, std::mem::size_of::<HeadNote>()),
            self.peak_head_notes,
            mb(
                self.planned.iter().map(|p| p.len()).sum::<usize>(),
                std::mem::size_of::<PlannedNote>()
            ),
            self.planned.iter().map(|p| p.len()).sum::<usize>(),
            mb(self.voices.len(), std::mem::size_of::<Voice>()),
            self.voices.len(),
            mb(self.entry_info.len(), std::mem::size_of::<EntryInfo>()),
            self.entry_info.len(),
        );
        // [257]
        if self.res.evictions > 0 || self.heads.res.evictions > 0 {
            log::info!(
                "merlin: kernels {:.1}% resident ({} rebuilds of {} entries, {} evictions, \
                 {:.2}s), heads {:.1}% resident ({} evictions, {:.0} MiB refilled). Worst \
                 step evicted {} kernels and {} head rows.",
                100.0 * self.res.hit_rate(),
                self.kernel_rebuilds,
                self.entries.len(),
                self.res.evictions,
                self.rebuild_secs,
                100.0 * self.heads.res.hit_rate(),
                self.heads.res.evictions,
                self.heads.refilled_bytes as f64 / 1048576.0,
                self.res.worst_step,
                self.heads.res.worst_step,
            );
        }
        let total = self.prof.total();
        if total > 0.0 {
            let rows: Vec<String> = self
                .prof
                .rows()
                .iter()
                .filter(|&&(_, s)| s > 0.0005)
                .map(|&(n, s)| format!("{n} {s:.2}s ({:.0}%)", 100.0 * s / total))
                .collect();
            log::info!("merlin: {:.2}s accounted for -- {}", total, rows.join(", "));
        }
        if self.head_total > 0 {
            log::info!(
                "merlin: {} head notes, {:.1}% with nothing moving under them; span                  {} frames now against {} if the body came from a kernel ({:.1}x shorter,                  mean {:.0} -> {:.1} frames a note)",
                self.head_total,
                100.0 * self.head_still as f64 / self.head_total as f64,
                self.head_span_now,
                self.head_span_split,
                self.head_span_now as f64 / self.head_span_split.max(1) as f64,
                self.head_span_now as f64 / self.head_total as f64,
                self.head_span_split as f64 / self.head_total as f64,
            );
        }
        if self.depth_truncated > 0 {
            log::warn!(
                "merlin: {} notes wanted a kernel of up to {} partitions and the ring holds                  {} -- their tails stop early. The ring is sized from the bank's deepest                  `attack + release`, so this means the two arithmetics have drifted apart                  and is a defect rather than a setting.",
                self.depth_truncated,
                self.depth_longest,
                self.fdl.lags,
            );
        }
        if self.release_truncated > 0 {
            log::warn!(
                "merlin: {} distinct voices have a release longer than the {:.2} s the ring \
                 was allocated for -- the longest is {:.2} s -- so they stop mid-decay. The \
                 ring is sized once, before any controller has been seen, and CC72 adds up \
                 to 15.0 s to a release. Re-render with --merlin-kernel-ms {:.0} to cover it.",
                self.release_truncated,
                self.tail_cap as f64 / self.cfg.sample_rate as f64,
                self.release_longest as f64 / self.cfg.sample_rate as f64,
                (self.release_longest as f64 / self.cfg.sample_rate as f64) * 1000.0,
            );
        }
        if self.cut_notes > 0 {
            log::info!(
                "merlin: {} notes damped by CC120 rather than released -- no impulse, and a \
                 {}-frame linear ramp in the head. The gate table publishes a note-off at the \
                 cut frame like any other, so this count is the only way to see that All \
                 Sound Off reached merlin at all.",
                self.cut_notes,
                self.cfg.steal_fade_frames.max(1),
            );
        }
        if self.bent_notes > 0 {
            log::info!(
                "merlin: {} note-ons opened on a bent channel, quantised to {} cents. Bend \
                 does not collapse -- each distinct factor is its own waveform and its own \
                 entries -- so this is the number to read the entry count against.",
                self.bent_notes,
                if self.cfg.merlin_bend_cents > 0.0 {
                    format!("{:.3}", self.cfg.merlin_bend_cents)
                } else {
                    "no".into()
                },
            );
        }
        if self.impulses_late > 0 {
            log::warn!(
                "merlin: {} impulses resolved into a block already emitted and were nudged \
                 forward. They sound, a few frames late; see `plan_note`.",
                self.impulses_late
            );
        }
    }
}
