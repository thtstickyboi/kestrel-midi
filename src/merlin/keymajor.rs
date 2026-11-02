// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Key-major merlin, step 2: one window, the static case, on the CPU. \[1\]

use crate::backend::{Backend, BlockStats};
use crate::bank::{Bank, ModEnvParams, RegionParams};
use crate::config::Config;
use crate::merlin::bucket::{BucketPass, RunRec, SealedWindow, VoiceDesc, KEYS, REC_CUT, RING_OUT};
use crate::merlin::head::{Head, HeadNote};
use crate::merlin::kernel::Kernel;
use crate::merlin::plan::{FilterKey, ReleaseBuckets, Table, WaveKey};
use crate::voice::SpawnCmd;
use anyhow::{Context, Result};
use crate::merlin::progress::MerlinProgress;
use bytemuck::Zeroable;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::Arc;

/// `MerlinSynth::plan_note`'s ceiling on section 3b's scalar, past which a note \[2\]
const MAX_KERNEL_GAIN: f64 = 16.0;

/// The device's filter buses, which set the window: `filters * (window + \[3\]
const BUS_BYTES: u64 = 1536 << 20;

/// The longest window, in frames.
const WINDOW_MAX: usize = 1 << 20;

/// Sustains kept on the host from one window to the next, so a waveform heard \[4\]
const SUSTAIN_CACHE_BYTES: usize = 1 << 30;

/// Slices read and planned side by side, at most, before they are handed over \[5\]
const PLAN_BATCH: usize = 16;

/// The most records planned side by side: what a window's planning holds at \[6\]
const BATCH_RECORDS: usize = 4 << 20;

/// The most of one key's records in one plan, `Config::merlin_km_slice_records`'s \[7\]
pub const SLICE_RECORDS: usize = 1 << 20;

/// Windows of one key prepared side by side before they go to the device.
const WINDOWS_AHEAD: usize = 1024;

/// Bytes of prepared windows the parsed render hands the device's thread \[8\]
pub const AHEAD_BYTES: u64 = 512 << 20;

/// What the packing thread has handed the device's thread and the device has \[9\]
struct Ahead {
    /// Bytes held, and whether the device's thread is gone.
    state: std::sync::Mutex<(u64, bool)>,
    room: std::sync::Condvar,
    limit: u64,
}

impl Ahead {
    fn new(limit: u64) -> Self {
        Self { state: std::sync::Mutex::new((0, false)), room: std::sync::Condvar::new(), limit }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, (u64, bool)> {
        // Two counters that a panic cannot leave wrong.
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Wait for room for `bytes` and take it. Always room when nothing is held, \[10\]
    fn reserve(&self, bytes: u64) -> bool {
        let mut st = self.lock();
        loop {
            if st.1 {
                return false;
            }
            if st.0 == 0 || st.0 + bytes <= self.limit {
                st.0 += bytes;
                return true;
            }
            st = self.room.wait(st).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// The device has put `bytes`.
    fn release(&self, bytes: u64) {
        let mut st = self.lock();
        st.0 = st.0.saturating_sub(bytes);
        self.room.notify_all();
    }

    /// The device's thread is gone: whoever waits is let go, and no one waits again.
    fn close(&self) {
        self.lock().1 = true;
        self.room.notify_all();
    }
}

/// Closes an `Ahead` when dropped, so the device's thread closes it whether it \[11\]
struct CloseOnDrop(Arc<Ahead>);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Sustains longer than this are rendered on the host rather than by the \[12\]
const CPU_LONG_FRAMES: usize = 1 << 18;

/// `Config::merlin_km_pool_frames`'s default: 640 MiB.
pub const POOL_FRAMES: usize = 160 << 20;

/// `Config::merlin_km_plan_records`'s default.
pub const PLAN_RECORDS: usize = 8 << 20;

/// `Config::merlin_km_stream_notes`'s default.
pub const STREAM_NOTES: u32 = 32 << 20;

/// `Config::merlin_km_stream_part`'s default.
pub const STREAM_PART: u32 = 4 << 20;

/// `Config::merlin_km_stream_held`'s default: about 960 MiB.
pub const STREAM_HELD: u32 = 40 << 20;

/// Records a binning run takes, side by side with the others.
const BIN_RUN: usize = 1 << 18;

/// Marks a sustain segment's entry in `plan_key`, whose release buckets are \[13\]
const SEGMENT: u32 = 1 << 31;

/// Records of a key the parsed render plans and packs side by side at once, \[14\]
const KEY_BATCH_RECORDS: usize = 8 << 20;


/// The constants time-major merlin derives from the bank and the config, which \[15\]
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// The partition a kernel's length is rounded up to.
    pub p: usize,
    /// The deepest kernel, in partitions.
    pub lags: usize,
    /// The longest release a kernel holds.
    pub tail_cap: usize,
    pub buckets: ReleaseBuckets,
    /// Frames of a sustain segment, a whole number of partitions: zero on the \[16\]
    pub seg: u32,
}

impl Shape {
    /// As `MerlinSynth::new` sets them, less the cap one storage binding puts \[17\]
    pub fn new(cfg: &Config, bank: &Bank) -> Self {
        let p = (cfg.block_frames as usize).min(2048);
        let tails: Vec<usize> =
            bank.params.iter().map(|rp| crate::merlin::kernel::release_frames(rp, cfg)).collect();
        let tail_max = tails.iter().copied().max().unwrap_or(0);
        let derived = tail_max.min(30 * cfg.sample_rate as usize);
        let tail_cap = if cfg.merlin_kernel_frames == 0 {
            derived.max(p)
        } else {
            (cfg.merlin_kernel_frames as usize).max(p)
        };
        let deepest = tails
            .iter()
            .zip(bank.params.iter())
            .map(|(&t, rp)| crate::merlin::kernel::flat_from(rp, cfg) + t.min(tail_cap))
            .max()
            .unwrap_or(tail_cap)
            .max(tail_cap);
        let buckets =
            ReleaseBuckets::from_tail(cfg.merlin_release_tail, cfg.sample_rate, cfg.merlin_release_buckets);
        let seg = if cfg.merlin_key_major_gpu { (cfg.merlin_km_segment as usize).next_multiple_of(p) as u32 } else { 0 };
        Self { p, lags: deepest.div_ceil(p).max(1), tail_cap, buckets, seg }
    }

    /// Whole segments of a body `frames` long, for a note whose gain holds \[18\]
    fn segments(&self, frames: u32, r: &RunRec, cut: bool) -> u32 {
        if self.seg == 0 || cut || r.gain != r.gain_end {
            0
        } else {
            frames / self.seg
        }
    }

    /// Of a note's first `whole` segments, those that start in `[w0, w1)`.
    fn fired(&self, whole: u32, start: u64, w0: u64, w1: u64) -> std::ops::Range<u32> {
        if whole == 0 {
            return 0..0;
        }
        let seg = self.seg as u64;
        let hi = (w1 - start).div_ceil(seg).min(whole as u64) as u32;
        let lo = w0.saturating_sub(start).div_ceil(seg).min(hi as u64) as u32;
        lo..hi
    }
}

/// What a note of one voice needs that does not depend on how long it is held: \[19\]
#[derive(Clone, Copy)]
struct VoiceInfo {
    filter: u32,
    tail: usize,
    flat: u32,
    r: f32,
    wave: WaveKey,
}

/// One waveform of a key: what its sustain is rendered from, and how much of it \[20\]
struct Wave {
    cmd: SpawnCmd,
    params: RegionParams,
    need: usize,
    /// The waveform by value, which the sustain cache is keyed on.
    key: WaveKey,
}

impl Wave {
    /// Sustains it takes in the pool: its left and right, end to end, for one \[21\]
    fn channels(&self) -> usize {
        1 + usize::from(self.key.smp_base_r != self.key.smp_base)
    }

    /// Its channels' commands, each with the word in the pool it renders to.
    fn channel_cmds(&self, at: u32, len: usize) -> Vec<(crate::voice::SpawnCmd, u32)> {
        let right = crate::voice::SpawnCmd { smp_base: self.key.smp_base_r, ..self.cmd };
        std::iter::once((self.cmd, at)).chain((self.channels() == 2).then_some((right, at + len as u32))).collect()
    }
}

/// One cache entry: a waveform released at one reference point.
#[derive(Clone, Copy)]
struct Entry {
    wave: u32,
    dj: u32,
    skip: u32,
    depth: u32,
}

#[derive(Clone, Copy)]
struct Impulse {
    entry: u32,
    frame: u64,
    gain: [f32; 2],
    weight: u16,
}

/// One (key, filter) group's notes.
#[derive(Default)]
struct Group {
    impulses: Vec<Impulse>,
    /// Impulses a CC120 cuts in their tails, with the frames from each to it.
    stopped: Vec<(Impulse, u32)>,
    heads: Vec<HeadNote>,
    /// Frames `[lo, hi)` anything in it reaches.
    lo: u64,
    hi: u64,
}

/// Counts for the summary.
#[derive(Default, Clone, Copy, Debug)]
pub struct Counts {
    pub notes: u64,
    pub groups: u64,
    pub waves: u64,
    pub entries: u64,
    pub released_on_head: u64,
    pub cut: u64,
    /// Notes a CC120 cut in their tails.
    pub tails_cut: u64,
    pub kernel_frames: u64,
    pub head_frames: u64,
    pub render_secs: f64,
    /// Rendering the sustains and cutting the kernels from them.
    pub build_secs: f64,
    /// Playing the groups and adding them into the master.
    pub mix_secs: f64,
    /// The device's convolution, readback included, when it does it.
    pub conv_secs: f64,
    /// Reading each key's records and planning them.
    pub plan_secs: f64,
}

/// Sustains by waveform, and the bytes they hold.
type SustainCache = (FxHashMap<WaveKey, Arc<Vec<f32>>>, usize);

/// The renderer. Holds what is shared across keys: the filters in first-seen \[22\]
pub struct KeyMajorCpu {
    cfg: Config,
    bank: Arc<Bank>,
    shape: Shape,
    nofilter: Arc<Vec<RegionParams>>,
    filters: Table<FilterKey>,
    filter_coef: Vec<Option<[f32; 4]>>,
    /// The convolution on the device, in place of adding every kernel frame \[23\]
    gpu: Option<crate::merlin::kmgpu::KmGpu>,
    /// Sustain renders by waveform, inside `SUSTAIN_CACHE_BYTES`. \[24\]
    sustains: std::sync::Mutex<SustainCache>,
    pub counts: Counts,
    /// The voice rows taken on so far, and what each needs, for the streamed \[25\]
    voices: Vec<VoiceDesc>,
    infos: Vec<VoiceInfo>,
    begun: bool,
}

impl KeyMajorCpu {
    pub fn new(cfg: &Config, bank: Arc<Bank>) -> Self {
        let shape = Shape::new(cfg, &bank);
        let nofilter = Arc::new(crate::merlin::kernel::nofilter_params(&bank.params));
        Self {
            cfg: cfg.clone(),
            bank,
            shape,
            nofilter,
            filters: Table::default(),
            filter_coef: Vec::new(),
            gpu: None,
            sustains: std::sync::Mutex::new((FxHashMap::default(), 0)),
            counts: Counts::default(),
            voices: Vec::new(),
            infos: Vec::new(),
            begun: false,
        }
    }

    /// The same, convolving on the device.
    pub fn with_gpu(cfg: &Config, bank: Arc<Bank>) -> Result<Self> {
        let mut km = Self::new(cfg, bank);
        km.gpu = Some(crate::merlin::kmgpu::KmGpu::new(cfg)?);
        Ok(km)
    }

    pub fn gpu_profile(&self) -> Option<crate::merlin::kmgpu::GpuProfile> {
        self.gpu.as_ref().map(|g| g.prof)
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    fn voice_info(&mut self, v: &VoiceDesc) -> VoiceInfo {
        let p = &v.params;
        let coef = crate::merlin::plan::bus_filter(p, self.cfg.merlin_filter_cents);
        let filter = self.filters.intern(FilterKey::of(coef));
        if filter as usize == self.filter_coef.len() {
            self.filter_coef.push(coef);
        }
        let raw = crate::merlin::kernel::release_frames(p, &self.cfg);
        VoiceInfo {
            filter,
            tail: raw.min(self.shape.tail_cap),
            flat: crate::merlin::kernel::flat_from(p, &self.cfg) as u32,
            r: p.release_coef,
            // [26]
            wave: WaveKey { smp_base_r: v.smp_base_r, ..WaveKey::new(&v.cmd, p, v.menv.as_ref()) },
        }
    }

    /// Render every note the bucket pass filed, here, the whole song as one \[27\]
    pub fn render(&mut self, pass: &mut BucketPass) -> Result<Vec<f32>> {
        let t = std::time::Instant::now();
        let voices: Vec<VoiceDesc> = pass.voices().to_vec();
        let infos: Vec<VoiceInfo> = voices.iter().map(|v| self.voice_info(v)).collect();
        let mut master: Vec<f64> = Vec::new();
        for key in 0..KEYS {
            let recs = read_key(pass, key)?;
            if !recs.is_empty() {
                self.render_key_cpu(&recs, &voices, &infos, &mut master)?;
            }
        }
        self.counts.render_secs = t.elapsed().as_secs_f64();
        Ok(master.iter().map(|&x| x as f32).collect())
    }

    /// What the render did, for the log.
    fn log_summary(&self, cfg: &Config, windows: u64, frames: usize, carry: usize) {
        let c = &self.counts;
        if let Some(g) = self.gpu_profile() {
            log::info!(
                "merlin key-major: {windows} window(s) of {frames} frames, {carry}-frame carry, {} filter buses; \
                 on the device {:.2}s handing keys over and {:.2}s ending windows -- kernel spectra {:.2}s, \
                 convolution {:.2}s (of it direct {:.2}s), heads {:.2}s, waiting on uploads {:.2}s, biquads, \
                 mix and readback {:.2}s; {} transforms in {} chunks for {} (block, entry) pairs in {} blocks, \
                 {} impulses direct in {} (page, bus) items, {} head pieces in {} (page, bus) items",
                self.filter_coef.len(),
                c.conv_secs,
                c.mix_secs,
                g.kernels,
                g.conv,
                g.direct,
                g.heads,
                g.throttled,
                g.window,
                g.transforms,
                g.chunks,
                g.pairs,
                g.blocks,
                g.direct_imps,
                g.direct_items,
                g.head_pieces,
                g.head_items,
            );
        }
        log::info!(
            "merlin key-major ({}): render {:.2}s ({:.2}s reading and planning, {:.2}s building kernels, \
             {:.2}s handing keys to the device, {:.2}s playing groups or ending windows); {} notes in {} \
             (key, filter) groups, {} waveforms, {} entries, {} released on the head, {} cut, {} cut in their tails; {:.2} G kernel \
             frames and {:.2} G head frames added here",
            if cfg.merlin_key_major_gpu { "device convolution, streamed" } else { "cpu reference" },
            c.render_secs,
            c.plan_secs,
            c.build_secs,
            c.conv_secs,
            c.mix_secs,
            c.notes,
            c.groups,
            c.waves,
            c.entries,
            c.released_on_head,
            c.cut,
            c.tails_cut,
            c.kernel_frames as f64 / 1e9,
            c.head_frames as f64 / 1e9,
        );
    }

    /// Take on voice rows the pass has made, their filters interned in voice \[28\]
    fn add_voices(&mut self, new: Vec<VoiceDesc>) -> Result<()> {
        for v in new {
            let info = self.voice_info(&v);
            self.infos.push(info);
            self.voices.push(v);
        }
        let coefs = self.filter_coef.clone();
        if let (Some(gpu), true) = (self.gpu.as_mut(), self.begun) {
            gpu.grow(&coefs)?;
        }
        Ok(())
    }

    /// One sealed window of `key`, the current key, on the device, added into \[29\]
    fn render_window(&mut self, sw: &SealedWindow, key: usize, frames: usize, carry: usize) -> Result<()> {
        let w0 = sw.index * frames as u64;
        let w1 = w0 + frames as u64;
        self.render_slices(sw.key_len(key) as usize, w0, frames, carry, |sh, cfg, nf, voices, infos, lo, hi| {
            let recs = sw.key_records_between(key, lo, hi)?;
            Ok(Self::plan_key(sh, cfg, nf, recs.chunks().into_iter().flatten().map(|r| (*r, NO_CUT)), voices, infos, w0, w1))
        })
    }

    /// The device's buses made, at the window and carry the render uses.
    fn begin(&mut self, frames: usize, carry: usize) -> Result<()> {
        if !self.begun {
            let coefs = self.filter_coef.clone();
            self.gpu.as_mut().context("no device")?.begin(&coefs, frames, carry)?;
            self.begun = true;
        }
        Ok(())
    }

    /// A window's `n` records of the current key, planned by `plan` a slice \[30\]
    fn render_slices<F>(&mut self, n: usize, w0: u64, frames: usize, carry: usize, plan: F) -> Result<()>
    where
        F: Fn(Shape, &Config, usize, &[VoiceDesc], &[VoiceInfo], usize, usize) -> Result<KeyPlan> + Sync,
    {
        self.begin(frames, carry)?;
        let pc = self.gpu.as_ref().context("no device")?.pack_cfg()?;
        let p = self.prepare(&pc, n, w0, frames, plan)?;
        self.counts.absorb(&p.counts);
        let gpu = self.gpu.as_mut().context("no device")?;
        let (put, mix) = submit(gpu, p)?;
        self.counts.conv_secs += put;
        self.counts.mix_secs += mix;
        Ok(())
    }

    /// A window's `n` records of the current key, planned by `plan` a slice \[31\]
    fn prepare<F>(&self, pc: &crate::merlin::kmgpu::PackCfg, n: usize, w0: u64, frames: usize, plan: F) -> Result<Prepared>
    where
        F: Fn(Shape, &Config, usize, &[VoiceDesc], &[VoiceInfo], usize, usize) -> Result<KeyPlan> + Sync,
    {
        let mut counts = Counts::default();
        let mut packs = Vec::new();
        // Slices: (first record, end) in record order.
        let slices: Vec<(usize, usize)> = (0..n).step_by(SLICE_RECORDS).map(|lo| (lo, (lo + SLICE_RECORDS).min(n))).collect();
        let mut i = 0;
        while i < slices.len() {
            let mut j = i;
            let mut held = 0;
            while j < slices.len() && j - i < PLAN_BATCH && (j == i || held + slices[j].1 - slices[j].0 <= BATCH_RECORDS) {
                held += slices[j].1 - slices[j].0;
                j += 1;
            }
            let t = std::time::Instant::now();
            let (sh, cfg, nf) = (self.shape, &self.cfg, self.filter_coef.len());
            let (voices, infos) = (&self.voices, &self.infos);
            let plans: Vec<KeyPlan> = slices[i..j].par_iter().map(|&(lo, hi)| plan(sh, cfg, nf, voices, infos, lo, hi)).collect::<Result<_>>()?;
            counts.plan_secs += t.elapsed().as_secs_f64();
            for plan in plans {
                counts.add_plan(plan.counts);
                if !plan.groups.is_empty() {
                    packs.push(self.pack_plan(&plan, w0, frames, pc, &mut counts)?);
                }
            }
            i = j;
        }
        Ok(Prepared { w0, packs, counts, ends: true })
    }


    /// The next key: its buses and filter state clear. The voice rows and the \[32\]
    fn start_key(&mut self) -> Result<()> {
        if self.begun {
            self.gpu.as_mut().context("no device")?.start_key()?;
        }
        Ok(())
    }

    /// One key's plan for the window, packed for the device with pools of its \[33\]
    fn pack_plan(&self, plan: &KeyPlan, w0: u64, frames: usize, pc: &crate::merlin::kmgpu::PackCfg, counts: &mut Counts) -> Result<(Option<Vec<f32>>, crate::merlin::kmgpu::Packed)> {
        let t = std::time::Instant::now();
        let (sustains, kernels) = self.build(plan, counts)?;
        counts.build_secs += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let mut at_wave = Vec::with_capacity(sustains.len());
        let mut at = 0u32;
        for s in &sustains {
            at_wave.push((at, s.len() as u32));
            at += s.len() as u32;
        }
        let ident: Vec<u32> = (0..self.filter_coef.len() as u32).collect();
        let (ins, heads) = pieces(plan, w0, frames, &at_wave, &ident);
        let ks: Vec<&[f32]> = kernels.iter().map(|k| &k[..]).collect();
        let mut pool: Vec<f32> = sustains.iter().flat_map(|s| s.iter().copied()).collect();
        let table = crate::merlin::kmgpu::append_whole(&mut pool, &ks);
        let packed = crate::merlin::kmgpu::KmGpu::pack(pc, &table, &ins, heads)?;
        counts.conv_secs += t.elapsed().as_secs_f64();
        Ok((Some(pool), packed))
    }

    /// Pass one over a key: every window planned, only for the waveforms and \[34\]
    fn plan_windows(&self, kr: &KeyRecs, by_window: &[Vec<Vec<u32>>], first: u64, frames: usize) -> Vec<Vec<KeyPlan>> {
        let (sh, cfg, nf) = (self.shape, &self.cfg, self.filter_coef.len());
        let (voices, infos) = (&self.voices, &self.infos);
        by_window
            .par_iter()
            .enumerate()
            .map(|(i, wr)| {
                let w0 = (first + i as u64) * frames as u64;
                wr.par_iter()
                    .map(|c| {
                        let recs = c.iter().map(|&i| {
                            let r = &kr.recs[i as usize];
                            (kr.run(r), kr.cut_after(r))
                        });
                        Self::plan_key(sh, cfg, nf, recs, voices, infos, w0, w0 + frames as u64)
                    })
                    .collect()
            })
            .collect()
    }

    /// The pools for a run of a key's windows, from their plans.
    fn key_pools(&self, parts: &[WindowPart]) -> Result<KeyPools> {
        let t = std::time::Instant::now();
        let (sh, cfg) = (self.shape, &self.cfg);
        // [35]
        let mut waves: FxHashMap<WaveKey, usize> = FxHashMap::default();
        let mut wave_list: Vec<Wave> = Vec::new();
        let mut entries: FxHashMap<EntryKey, usize> = FxHashMap::default();
        let mut entry_list: Vec<(usize, Entry)> = Vec::new();
        for plan in parts.iter().flat_map(|p| &p.plans) {
            for wv in &plan.waves {
                let at = *waves.entry(wv.key).or_insert_with(|| {
                    wave_list.push(Wave { need: 0, ..*wv });
                    wave_list.len() - 1
                });
                wave_list[at].need = wave_list[at].need.max(wv.need);
            }
            for e in &plan.entries {
                let wk = plan.waves[e.wave as usize].key;
                entries.entry(EntryKey::new(wk, e)).or_insert_with(|| {
                    entry_list.push((waves[&wk], *e));
                    entry_list.len() - 1
                });
            }
        }
        let mut counts = Counts::default();
        let plan_secs = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        // Each sustain rendered into its own place in the pool, side by side.
        let lens: Vec<usize> = wave_list.iter().map(|wv| wv.need.max(1)).collect();
        let mut at = Vec::with_capacity(lens.len());
        let mut total = 0usize;
        for (&l, wv) in lens.iter().zip(&wave_list) {
            at.push(total as u32);
            total += l * wv.channels();
        }
        if (total as u64) * 4 > 2047 << 20 {
            anyhow::bail!("one waveform's sustain ({} MiB) passes one binding's 2047 MiB", total >> 18);
        }
        let build = {
            // [36]
            type RowCopy = (Vec<RegionParams>, FxHashMap<usize, RegionParams>);
            type Set = (Option<RowCopy>, Vec<(crate::voice::SpawnCmd, usize)>, Vec<u64>);
            let mut sets: Vec<Set> = vec![(None, Vec::new(), Vec::new())];
            // [37]
            use crate::merlin::gpubuild::{direct_ok, DirectJob};
            let direct: Vec<DirectJob> = wave_list
                .iter()
                .zip(&at)
                .filter(|(wv, _)| direct_ok(&wv.cmd, &wv.params))
                .flat_map(|(wv, &a)| wv.channel_cmds(a, wv.need.max(1)).into_iter().map(|(cmd, a)| DirectJob { cmd, params: wv.params, frames: wv.need.max(1), at: a as u64 }))
                .collect();
            // [38]
            let cpu: Vec<(u64, Vec<f32>)> = wave_list
                .par_iter()
                .zip(at.par_iter())
                .filter(|(wv, _)| wv.need.max(1) > CPU_LONG_FRAMES && !direct_ok(&wv.cmd, &wv.params))
                .flat_map(|(wv, &a)| wv.channel_cmds(a, wv.need.max(1)).into_par_iter().map(move |(cmd, a)| Ok((a as u64, self.render_sustain(&Wave { cmd, ..*wv }, wv.need.max(1))?))))
                .collect::<Result<_>>()?;
            for (wv, &a) in wave_list.iter().zip(&at) {
                if wv.need.max(1) > CPU_LONG_FRAMES || direct_ok(&wv.cmd, &wv.params) {
                    continue;
                }
                let cmd = wv.cmd;
                let mut own = wv.params;
                own.flags &= !crate::bank::RP_FILTER;
                let row = cmd.params as usize;
                let same = |b: &RegionParams| bytemuck::bytes_of(b) == bytemuck::bytes_of(&own);
                let i = if self.nofilter.get(row).is_some_and(same) {
                    0
                } else {
                    let found = sets.iter().skip(1).position(|(t, _, _)| t.as_ref().is_some_and(|(_, set)| set.get(&row).is_none_or(same)));
                    match found {
                        Some(j) => j + 1,
                        None => {
                            if row >= self.nofilter.len() {
                                anyhow::bail!("a voice names a params row the bank does not have");
                            }
                            sets.push((Some(((*self.nofilter).clone(), FxHashMap::default())), Vec::new(), Vec::new()));
                            sets.len() - 1
                        }
                    }
                };
                let (t, jobs, job_at) = &mut sets[i];
                if let Some((table, set)) = t.as_mut() {
                    table[row] = own;
                    set.insert(row, own);
                }
                for (cmd, a) in wv.channel_cmds(a, wv.need.max(1)) {
                    jobs.push((cmd, wv.need.max(1)));
                    job_at.push(a as u64);
                }
            }
            let tables = sets
                .into_iter()
                .filter(|s| !s.1.is_empty())
                .map(|(t, jobs, at)| (t.map_or_else(|| self.nofilter.clone(), |(table, _)| Arc::new(table)), jobs, at))
                .collect();
            PoolBuild { tables, total, cpu, direct }
        };
        counts.waves += wave_list.len() as u64;
        counts.entries += entry_list.len() as u64;
        let mut kp = KeyPools { build, wave_at: FxHashMap::default(), entry_at: FxHashMap::default(), counts };
        for (wk, &i) in &waves {
            // A head reads a stereo waveform by the top bit of its `cap`.
            let stereo = (wave_list[i].channels() as u32 - 1) << 31;
            kp.wave_at.insert(*wk, (at[i], lens[i] as u32 | stereo));
        }
        // [39]
        for (ek, &i) in &entries {
            let (w, e) = &entry_list[i];
            let wv = &wave_list[*w];
            let (dj, from, len) = (e.dj as usize, e.skip as usize, e.depth as usize * sh.p);
            let flat = crate::merlin::kernel::flat_from(&wv.params, cfg);
            if dj < flat {
                anyhow::bail!("release reference {dj} is inside this region's moving envelope, which is flat only from frame {flat}");
            }
            let r = wv.params.release_coef as f64;
            let desc = crate::merlin::kmgpu::KDesc {
                at: at[*w] + from as u32,
                len: len as u32,
                valid: lens[*w].saturating_sub(from).min(len) as u32,
                plain: dj.saturating_sub(from).min(len) as u32,
                rel0: from.saturating_sub(dj) as u32 + 1,
                log2r: if r > 0.0 { r.log2() as f32 } else { f32::NEG_INFINITY },
                right: if wv.channels() == 2 { lens[*w] as u32 } else { 0 },
            };
            kp.entry_at.insert(*ek, desc);
        }
        kp.counts.plan_secs += plan_secs;
        kp.counts.build_secs += t.elapsed().as_secs_f64();
        Ok(kp)
    }

    /// Pass two over a run of a key's windows: pass one's plans, each part's \[40\]
    fn prepare_key(&self, pc: &crate::merlin::kmgpu::PackCfg, kp: &KeyPools, parts: Vec<WindowPart>, frames: usize, slot: &[u32], mut send: impl FnMut(Prepared) -> bool) -> Result<bool> {
        let held_of = |w: &WindowPart| w.plans.iter().map(|p| p.groups.iter().map(|(_, g)| g.impulses.len() + g.stopped.len() + g.heads.len()).sum::<usize>()).sum::<usize>();
        let mut parts = parts.into_iter().peekable();
        while parts.peek().is_some() {
            let mut batch: Vec<WindowPart> = Vec::new();
            let mut held = 0;
            while let Some(w) = parts.next_if(|w| batch.is_empty() || held + held_of(w) <= KEY_BATCH_RECORDS) {
                held += held_of(&w);
                batch.push(w);
            }
            let heads: Vec<(u64, bool)> = batch.iter().map(|b| (b.w, b.ends)).collect();
            let items: Vec<(usize, u64, KeyPlan)> = batch.into_iter().enumerate().flat_map(|(i, w)| {
                let at = w.w;
                w.plans.into_iter().map(move |p| (i, at, p))
            }).collect();
            let done: Vec<(usize, Counts, Option<crate::merlin::kmgpu::Packed>)> = items
                .into_par_iter()
                .map(|(i, w, plan)| {
                    let (c, pk) = self.pack_slice(pc, kp, w * frames as u64, frames, plan, slot)?;
                    Ok((i, c, pk))
                })
                .collect::<Result<_>>()?;
            let mut done = done.into_iter().peekable();
            for (i, (w, ends)) in heads.into_iter().enumerate() {
                let mut p = Prepared { w0: w * frames as u64, packs: Vec::new(), counts: Counts::default(), ends };
                while let Some((_, c, pk)) = done.next_if(|d| d.0 == i) {
                    p.counts.absorb(&c);
                    p.packs.extend(pk.map(|pk| (None, pk)));
                }
                if !send(p) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// One slice's plan packed against the key's pools, and what it counted. \[41\]
    fn pack_slice(
        &self,
        pc: &crate::merlin::kmgpu::PackCfg,
        kp: &KeyPools,
        w0: u64,
        frames: usize,
        plan: KeyPlan,
        slot: &[u32],
    ) -> Result<(Counts, Option<crate::merlin::kmgpu::Packed>)> {
        let mut counts = Counts::default();
        counts.add_plan(plan.counts);
        if plan.groups.is_empty() {
            return Ok((counts, None));
        }
        let at_wave: Vec<(u32, u32)> = plan.waves.iter().map(|wv| kp.wave_at[&wv.key]).collect();
        let table: Vec<crate::merlin::kmgpu::KDesc> = plan.entries.iter().map(|e| kp.entry_at[&EntryKey::new(plan.waves[e.wave as usize].key, e)]).collect();
        let (ins, heads) = pieces(&plan, w0, frames, &at_wave, slot);
        drop(plan);
        let packed = crate::merlin::kmgpu::KmGpu::pack(pc, &table, &ins, heads)?;
        Ok((counts, Some(packed)))
    }

    /// One waveform rendered through the filterless table with the voice's \[42\]
    fn render_sustain(&self, wv: &Wave, frames: usize) -> Result<Vec<f32>> {
        let row = wv.cmd.params as usize;
        let mut own = wv.params;
        own.flags &= !crate::bank::RP_FILTER;
        let nofilter = &self.nofilter;
        let table = if nofilter.get(row).is_some_and(|b| bytemuck::bytes_of(b) == bytemuck::bytes_of(&own)) {
            nofilter.clone()
        } else {
            let mut t = (**nofilter).clone();
            *t.get_mut(row).context("a voice names a params row the bank does not have")? = own;
            Arc::new(t)
        };
        Ok(Kernel::from_cmd(&self.cfg, &self.bank, &wv.cmd, frames, Some(&table))?.sustain)
    }

    /// One key's notes that reach `[w0, w1)`: their waveforms, the entries \[43\]
    #[allow(clippy::too_many_arguments)]
    fn plan_key(
        sh: Shape,
        cfg: &Config,
        filters: usize,
        recs: impl Iterator<Item = (RunRec, u32)>,
        voices: &[VoiceDesc],
        infos: &[VoiceInfo],
        w0: u64,
        w1: u64,
    ) -> KeyPlan {
        let mut waves: Table<WaveKey> = Table::default();
        let mut wave_list: Vec<Wave> = Vec::new();
        let mut entries: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut entry_list: Vec<Entry> = Vec::new();
        let mut memo: Vec<u32> = vec![u32::MAX; voices.len()];
        let mut group_of: Vec<u32> = vec![u32::MAX; filters];
        let mut groups: Vec<(u32, Group)> = Vec::new();
        let mut counts = PlanCounts::default();

        for (r, cut_after) in recs {
            let r = &r;
            let vi = &infos[r.voice as usize];
            let reach = r.duration as u64 + cfg.steal_fade_frames as u64 + vi.tail as u64 + 1;
            if r.start >= w1 || r.start + reach <= w0 {
                continue;
            }
            let ours = r.start >= w0;
            if ours {
                counts.notes += r.count as u64;
            }
            let v = &voices[r.voice as usize];
            let p = &v.params;
            let d = r.duration;
            let (j, dj) = sh.buckets.reference(d, vi.flat);
            let fade = if r.flags & REC_CUT != 0 {
                if ours {
                    counts.cut += r.count as u64;
                }
                cfg.steal_fade_frames.max(1)
            } else {
                0
            };
            let bus = (vi.filter * 2) as u16;
            // Whether a CC120 comes inside its tail.
            let tail_cut = (cut_after as usize) < vi.tail;
            if tail_cut && ours {
                counts.tails_cut += r.count as u64;
            }
            let mut wave = |wave_list: &mut Vec<Wave>, need: usize| {
                let m = &mut memo[r.voice as usize];
                if *m == u32::MAX {
                    *m = waves.intern(vi.wave);
                    if *m as usize == wave_list.len() {
                        wave_list.push(Wave { cmd: v.cmd_for(r), params: *p, need: 0, key: vi.wave });
                    }
                }
                let wv = &mut wave_list[*m as usize];
                wv.need = wv.need.max(need);
                *m
            };
            let mut group = |groups: &mut Vec<(u32, Group)>| -> usize {
                let g = &mut group_of[vi.filter as usize];
                if *g == u32::MAX {
                    *g = groups.len() as u32;
                    groups.push((vi.filter, Group { lo: u64::MAX, ..Group::default() }));
                }
                *g as usize
            };
            // [44]
            let seg = sh.seg;
            let segments = |g: &mut Group, entries: &mut FxHashMap<(u32, u32), u32>, entry_list: &mut Vec<Entry>, w: u32, ks: std::ops::Range<u32>| {
                for k in ks {
                    let next = entry_list.len() as u32;
                    let e = *entries.entry((w, SEGMENT | k)).or_insert(next);
                    if e == next {
                        // [45]
                        entry_list.push(Entry { wave: w, dj: ((k + 1) * seg).max(vi.flat), skip: k * seg, depth: seg / sh.p as u32 });
                    }
                    g.impulses.push(Impulse { entry: e, frame: r.start + (k * seg) as u64, gain: r.gain, weight: r.count });
                }
            };

            // [46]
            let held = (dj < d && vi.r > 0.0).then(|| (vi.r as f64).powi(dj as i32 - d as i32));

            // [47]
            if fade == 0
                && cfg.release_curve == crate::config::EnvelopeCurve::Exponential
                && dj < d
                && vi.r > 0.0
                && vi.r < 1.0
                && vi.tail <= u16::MAX as usize
                && held.is_some_and(|p| p > MAX_KERNEL_GAIN)
            {
                // [48]
                let (release, damp) = if tail_cut { (cut_after, cfg.steal_fade_frames.max(1)) } else { (vi.tail as u32, 0) };
                let span = d.saturating_add(release).saturating_add(damp);
                if r.start + span as u64 <= w0 {
                    continue;
                }
                let whole = sh.segments(d, r, false);
                let ks = sh.fired(whole, r.start, w0, w1);
                let from = whole * seg;
                let w = wave(&mut wave_list, ((span as u64).min(w1 - r.start) as usize).max((ks.end * seg) as usize));
                let gi = group(&mut groups);
                let g = &mut groups[gi].1;
                segments(g, &mut entries, &mut entry_list, w, ks);
                g.heads.push(HeadNote {
                    wave: w,
                    start: r.start,
                    from,
                    d,
                    dj,
                    r: vi.r,
                    gain: r.gain,
                    gain_end: r.gain_end,
                    fade: damp as u16,
                    release: release as u16,
                    bus,
                    weight: r.count,
                    env_ratio: 1.0,
                });
                g.lo = g.lo.min(r.start);
                g.hi = g.hi.max(r.start + span as u64);
                if ours {
                    counts.released_on_head += r.count as u64;
                }
                continue;
            }

            let skip = if dj <= d { dj } else { 0 };
            let want = (dj as usize + vi.tail).saturating_sub(skip as usize).div_ceil(sh.p).max(1);
            let depth = want.min(sh.lags);
            let jkey = if dj <= d { j } else { u16::MAX };
            let env_ratio = if dj > d {
                let at_dj = crate::merlin::kernel::env_at(p, cfg, dj);
                if at_dj > 0.0 {
                    crate::merlin::kernel::env_at(p, cfg, d.saturating_sub(1)) / at_dj
                } else {
                    1.0
                }
            } else {
                1.0
            };
            let whole = sh.segments(skip, r, fade != 0);
            let ks = sh.fired(whole, r.start, w0, w1);
            let from = whole * seg;
            let frame = r.start + skip as u64;
            let span = d as u64 + fade as u64;
            let fires = fade == 0 && frame >= w0 && frame < w1;
            let heard = r.start + span > w0 && span > from as u64 && r.start + (from as u64) < w1;
            if !fires && !heard && ks.is_empty() {
                continue;
            }
            let need = if fires { dj as usize + depth * sh.p } else { 0 }
                .max(if heard { span.min(w1 - r.start) as usize } else { 0 })
                .max((ks.end * seg) as usize);
            let w = wave(&mut wave_list, need);
            let gi = group(&mut groups);
            let g = &mut groups[gi].1;
            segments(g, &mut entries, &mut entry_list, w, ks);
            if fires {
                let next = entry_list.len() as u32;
                let e = *entries.entry((w, jkey as u32)).or_insert(next);
                if e == next {
                    entry_list.push(Entry { wave: w, dj, skip, depth: depth as u32 });
                }
                let s = held.map_or_else(|| sh.buckets.gain(vi.r as f64, dj, d, env_ratio), |p| (p * env_ratio as f64) as f32);
                let imp = Impulse { entry: e, frame, gain: [r.gain_end[0] * s, r.gain_end[1] * s], weight: r.count };
                if tail_cut {
                    g.stopped.push((imp, d - skip + cut_after));
                } else {
                    g.impulses.push(imp);
                }
                g.hi = g.hi.max(frame + (depth * sh.p) as u64);
            }
            if heard {
                g.heads.push(HeadNote {
                    wave: w,
                    start: r.start,
                    from,
                    d,
                    dj,
                    r: vi.r,
                    gain: r.gain,
                    gain_end: r.gain_end,
                    fade: fade as u16,
                    release: 0,
                    bus,
                    weight: r.count,
                    env_ratio,
                });
                g.hi = g.hi.max(r.start + span);
            }
            g.lo = g.lo.min(r.start);
        }
        groups.sort_unstable_by_key(|(f, _)| *f);
        KeyPlan { waves: wave_list, entries: entry_list, groups, counts }
    }

    /// A plan's waveforms -- rendered through the filterless table with the \[49\]
    fn build(&self, plan: &KeyPlan, counts: &mut Counts) -> Result<Built> {
        let cfg = &self.cfg;
        let sh = self.shape;
        let (bank, nofilter) = (&self.bank, &self.nofilter);
        let keys: Vec<WaveKey> = plan.waves.iter().map(|wv| wv.key).collect();
        // What the cache holds for these waveforms, taken under the lock once.
        let held: Vec<Option<Arc<Vec<f32>>>> = {
            let c = self.sustains.lock().expect("the sustain cache's lock");
            keys.iter().map(|k| c.0.get(k).cloned()).collect()
        };
        let sustains: Vec<(Arc<Vec<f32>>, bool)> = plan
            .waves
            .par_iter()
            .zip(&held)
            .map(|(wv, had)| -> Result<(Arc<Vec<f32>>, bool)> {
                let need = wv.need.max(1);
                if let Some(s) = had.as_ref().filter(|s| s.len() >= need) {
                    return Ok((s.clone(), false));
                }
                // [50]
                let frames = need.max(had.as_ref().map_or(0, |s| s.len() * 2));
                let row = wv.cmd.params as usize;
                let mut own = wv.params;
                own.flags &= !crate::bank::RP_FILTER;
                let table = if nofilter.get(row).is_some_and(|b| bytemuck::bytes_of(b) == bytemuck::bytes_of(&own)) {
                    nofilter.clone()
                } else {
                    let mut t = (**nofilter).clone();
                    *t.get_mut(row).context("a voice names a params row the bank does not have")? = own;
                    Arc::new(t)
                };
                Ok((Arc::new(Kernel::from_cmd(cfg, bank, &wv.cmd, frames, Some(&table))?.sustain), true))
            })
            .collect::<Result<_>>()?;
        {
            let mut c = self.sustains.lock().expect("the sustain cache's lock");
            for ((s, fresh), key) in sustains.iter().zip(&keys) {
                if *fresh {
                    let old = c.0.get(key).map_or(0, |o| o.len() * 4);
                    if c.1 - old + s.len() * 4 <= SUSTAIN_CACHE_BYTES {
                        c.1 = c.1 - old + s.len() * 4;
                        c.0.insert(*key, s.clone());
                    }
                }
            }
        }
        let sustains: Vec<Arc<Vec<f32>>> = sustains.into_iter().map(|(s, _)| s).collect();
        // [51]
        let kernels: Vec<Vec<f32>> = plan
            .entries
            .par_iter()
            .map(|e| -> Result<Vec<f32>> {
                let wv = &plan.waves[e.wave as usize];
                let mut c = wv.cmd;
                c.params = 0;
                let k = Kernel::from_rendered(cfg, &c, sustains[e.wave as usize].to_vec(), std::slice::from_ref(&wv.params))?;
                k.segment(e.dj as usize, e.skip as usize, e.depth as usize * sh.p)
            })
            .collect::<Result<_>>()?;
        counts.waves += plan.waves.len() as u64;
        counts.entries += plan.entries.len() as u64;
        Ok((sustains, kernels))
    }

    /// One key, the whole song, here: plan it, render the waveforms it reads, \[52\]
    fn render_key_cpu(&mut self, recs: &[RunRec], voices: &[VoiceDesc], infos: &[VoiceInfo], master: &mut Vec<f64>) -> Result<()> {
        let t = std::time::Instant::now();
        let plan = Self::plan_key(self.shape, &self.cfg, self.filter_coef.len(), recs.iter().map(|r| (*r, NO_CUT)), voices, infos, 0, u64::MAX);
        self.counts.add_plan(plan.counts);
        self.counts.plan_secs += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let mut built = Counts::default();
        let (sustains, kernels) = self.build(&plan, &mut built)?;
        self.counts.absorb(&built);
        self.counts.build_secs += t.elapsed().as_secs_f64();
        let sustains: Vec<&[f32]> = sustains.iter().map(|s| &s[..]).collect();
        self.counts.groups += plan.groups.len() as u64;
        let coefs = &self.filter_coef;
        let chunk = rayon::current_num_threads().max(1);
        let t = std::time::Instant::now();
        for part in plan.groups.chunks(chunk) {
            let outs: Vec<Played> =
                part.par_iter().map(|(f, g)| play_group(g, &kernels, &sustains, coefs[*f as usize])).collect();
            for p in outs {
                self.counts.kernel_frames += p.kernel_frames;
                self.counts.head_frames += p.head_frames;
                let need = 2 * (p.lo as usize + p.l.len());
                if master.len() < need {
                    master.resize(need, 0.0);
                }
                let m = &mut master[2 * p.lo as usize..];
                for (i, (&a, &b)) in p.l.iter().zip(&p.r).enumerate() {
                    m[2 * i] += a as f64;
                    m[2 * i + 1] += b as f64;
                }
            }
        }
        self.counts.mix_secs += t.elapsed().as_secs_f64();
        Ok(())
    }
}

/// Every record filed under `key`.
fn read_key(pass: &mut BucketPass, key: usize) -> Result<Vec<RunRec>> {
    let mut recs: Vec<RunRec> = Vec::new();
    pass.store().read_key(key, |chunk| {
        recs.extend_from_slice(chunk);
        Ok(())
    })?;
    Ok(recs)
}

/// A plan's sustains, by wave, and its kernels, by entry.
type Built = (Vec<Arc<Vec<f32>>>, Vec<Vec<f32>>);

/// One key's notes over a window, planned.
struct KeyPlan {
    waves: Vec<Wave>,
    entries: Vec<Entry>,
    groups: Vec<(u32, Group)>,
    counts: PlanCounts,
}

/// What a plan counts, once per note, by the window it starts in.
#[derive(Default, Clone, Copy)]
struct PlanCounts {
    notes: u64,
    cut: u64,
    released_on_head: u64,
    tails_cut: u64,
}

impl KeyPlan {
    /// This plan cut by waveform, a plan for each of `n` classes of `class_of`: \[53\]
    fn split(self, class_of: &FxHashMap<WaveKey, u32>, n: usize) -> Vec<KeyPlan> {
        let cls: Vec<usize> = self.waves.iter().map(|w| class_of[&w.key] as usize).collect();
        let mut out: Vec<KeyPlan> = (0..n).map(|_| KeyPlan { waves: Vec::new(), entries: Vec::new(), groups: Vec::new(), counts: PlanCounts::default() }).collect();
        out[0].counts = self.counts;
        let mut wave_new = vec![0u32; cls.len()];
        for (i, wv) in self.waves.into_iter().enumerate() {
            wave_new[i] = out[cls[i]].waves.len() as u32;
            out[cls[i]].waves.push(wv);
        }
        let entry_old: Vec<u32> = self.entries.iter().map(|e| e.wave).collect();
        let mut entry_new = vec![0u32; entry_old.len()];
        for (i, e) in self.entries.into_iter().enumerate() {
            let c = cls[e.wave as usize];
            entry_new[i] = out[c].entries.len() as u32;
            out[c].entries.push(Entry { wave: wave_new[e.wave as usize], ..e });
        }
        for (f, g) in self.groups {
            let mut parts: Vec<Group> = (0..n).map(|_| Group { lo: g.lo, hi: g.hi, ..Group::default() }).collect();
            for i in g.impulses {
                parts[cls[entry_old[i.entry as usize] as usize]].impulses.push(Impulse { entry: entry_new[i.entry as usize], ..i });
            }
            for (i, stop) in g.stopped {
                parts[cls[entry_old[i.entry as usize] as usize]].stopped.push((Impulse { entry: entry_new[i.entry as usize], ..i }, stop));
            }
            for h in g.heads {
                parts[cls[h.wave as usize]].heads.push(HeadNote { wave: wave_new[h.wave as usize], ..h });
            }
            for (o, p) in out.iter_mut().zip(parts) {
                if !p.impulses.is_empty() || !p.stopped.is_empty() || !p.heads.is_empty() {
                    o.groups.push((f, p));
                }
            }
        }
        out
    }
}

/// Some of the slices of window `w`, planned; `ends` on the last of them, whose \[54\]
struct WindowPart {
    w: u64,
    plans: Vec<KeyPlan>,
    ends: bool,
}

/// A prepared window onto the device, in the order windows are submitted, \[55\]
fn submit(gpu: &mut crate::merlin::kmgpu::KmGpu, p: Prepared) -> Result<(f64, f64)> {
    let t = std::time::Instant::now();
    for (pools, pk) in p.packs {
        if let Some(pl) = pools {
            gpu.set_pool(&pl)?;
        }
        gpu.put(pk)?;
    }
    let put = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    if p.ends {
        gpu.end_window_into(p.w0 as usize)?;
    }
    Ok((put, t.elapsed().as_secs_f64()))
}

/// A key's sustains for the device's kernel builder: each job's command and \[56\]
struct PoolBuild {
    tables: Vec<crate::merlin::kmgpu::BuildTable>,
    total: usize,
    /// Sustains rendered here instead, at their words.
    cpu: Vec<(u64, Vec<f32>)>,
    /// Sustains for the device's frame-parallel pass.
    direct: Vec<crate::merlin::gpubuild::DirectJob>,
}

/// What the parsed render's thread hands the device's thread, in order.
enum Dev {
    /// The buses made, at these filters; the reply is what `pack` reads.
    Begin(Vec<Option<[f32; 4]>>, std::sync::mpsc::SyncSender<Result<crate::merlin::kmgpu::PackCfg>>),
    /// The next key, with the coefficients of the filters it uses, which \[57\]
    StartKey(Vec<Option<[f32; 4]>>, usize),
    /// The next key, streamed: its filters so far, its windows' frames, and \[58\]
    StartStream(Vec<Option<[f32; 4]>>, usize, usize),
    /// A streamed key's buses, as it has met more filters.
    Grow(Vec<Option<[f32; 4]>>),
    /// The windows that follow are a channel's bent notes.
    WarpBegin,
    /// Those windows read back through this clock, from one `tau` to the other.
    WarpEnd(Arc<crate::merlin::keyparse::Warp>, u64, u64),
    /// The pool for the windows that follow, rendered on the device.
    Build(PoolBuild),
    Window(Prepared),
    /// The window just before was `index` of `key`: what the progress is told \[59\]
    At(u8, u64),
    /// The key's last window was just before.
    KeyDone(u8),
}

impl Prepared {
    /// What it holds, for `AHEAD_BYTES`.
    fn bytes(&self) -> u64 {
        self.packs.iter().map(|(pl, pk)| pk.held() + pl.as_ref().map_or(0, |v| v.len() as u64 * 4)).sum::<u64>().max(1)
    }
}

/// A window ready for the device: its packs in order, and what preparing it \[60\]
struct Prepared {
    w0: u64,
    /// Each pack, after the pool it reads when the pool is its own.
    packs: Vec<(Option<Vec<f32>>, crate::merlin::kmgpu::Packed)>,
    counts: Counts,
    /// The last of its window: the window ends after it.
    ends: bool,
}

/// The pool for a run of a key's windows: its sustains as the device renders them end \[61\]
struct KeyPools {
    build: PoolBuild,
    wave_at: FxHashMap<WaveKey, (u32, u32)>,
    entry_at: FxHashMap<EntryKey, crate::merlin::kmgpu::KDesc>,
    counts: Counts,
}

/// What each waveform of `part` takes of a pool, in the order they appear.
fn part_needs(part: &WindowPart) -> (Vec<WaveKey>, FxHashMap<WaveKey, usize>) {
    let (mut order, mut need): (Vec<WaveKey>, FxHashMap<WaveKey, usize>) = (Vec::new(), FxHashMap::default());
    for wv in part.plans.iter().flat_map(|p| p.waves.iter()) {
        let m = need.entry(wv.key).or_insert_with(|| {
            order.push(wv.key);
            0
        });
        *m = (*m).max(wv.need.max(1) * wv.channels());
    }
    (order, need)
}

/// `part` cut by waveform, for a window that alone needs more than one pool: \[62\]
fn split_part(part: WindowPart, order: &[WaveKey], need: &FxHashMap<WaveKey, usize>, budget: usize) -> Vec<WindowPart> {
    let (mut class_of, mut class, mut total) = (FxHashMap::default(), 0u32, 0usize);
    for k in order {
        if total > 0 && total + need[k] > budget {
            class += 1;
            total = 0;
        }
        class_of.insert(*k, class);
        total += need[k];
    }
    let n = class as usize + 1;
    let WindowPart { w, plans, ends } = part;
    let mut out: Vec<WindowPart> = (0..n).map(|_| WindowPart { w, plans: Vec::new(), ends: false }).collect();
    for plan in plans {
        for (o, p) in out.iter_mut().zip(plan.split(&class_of, n)) {
            o.plans.push(p);
        }
    }
    out[n - 1].ends = ends;
    out
}

/// Runs of `parts`, in order, each run's waveforms at their longest need within \[63\]
fn pool_runs(parts: Vec<WindowPart>, budget: usize) -> Vec<Vec<WindowPart>> {
    let mut runs: Vec<Vec<WindowPart>> = Vec::new();
    let mut run: Vec<WindowPart> = Vec::new();
    let mut held: FxHashMap<WaveKey, usize> = FxHashMap::default();
    let mut total = 0usize;
    for part in parts {
        let (order, mine) = part_needs(&part);
        if mine.values().sum::<usize>() > budget {
            if !run.is_empty() {
                runs.push(std::mem::take(&mut run));
                held.clear();
                total = 0;
            }
            runs.extend(split_part(part, &order, &mine, budget).into_iter().map(|p| vec![p]));
            continue;
        }
        let grow: usize = mine.iter().map(|(k, &v)| v.saturating_sub(held.get(k).copied().unwrap_or(0))).sum();
        if !run.is_empty() && total + grow > budget {
            runs.push(std::mem::take(&mut run));
            held.clear();
            total = 0;
        }
        for (k, v) in mine {
            let e = held.entry(k).or_insert(0);
            if v > *e {
                total += v - *e;
                *e = v;
            }
        }
        run.push(part);
    }
    if !run.is_empty() {
        runs.push(run);
    }
    runs
}

/// A kernel by what it is cut from: its waveform, and `Entry`'s cut.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EntryKey {
    wave: WaveKey,
    dj: u32,
    skip: u32,
    depth: u32,
}

impl EntryKey {
    fn new(wave: WaveKey, e: &Entry) -> Self {
        Self { wave, dj: e.dj, skip: e.skip, depth: e.depth }
    }
}

/// A plan's impulses and head pieces for the window at `w0`, each head reading \[64\]
fn pieces(plan: &KeyPlan, w0: u64, frames: usize, at_wave: &[(u32, u32)], slot: &[u32]) -> (Vec<crate::merlin::kmgpu::GroupIn>, Vec<crate::merlin::head::Head>) {
    let mut ins = Vec::with_capacity(plan.groups.len());
    let mut heads: Vec<crate::merlin::head::Head> = Vec::new();
    let mut fold = crate::merlin::head::Fold::default();
    let mut order: Vec<(u64, u32, u32)> = Vec::new();
    for (f, g) in &plan.groups {
        // The key's bus for this filter.
        let bus = slot[*f as usize];
        let weighed = |i: &Impulse| {
            let w = i.weight as f32;
            (i.entry, i.frame - w0, [i.gain[0] * w, i.gain[1] * w])
        };
        ins.push(crate::merlin::kmgpu::GroupIn {
            bus,
            impulses: g.impulses.iter().map(weighed).collect(),
            stopped: g
                .stopped
                .iter()
                .map(|(i, stop)| {
                    let (e, f, gain) = weighed(i);
                    (e, f, gain, *stop)
                })
                .collect(),
        });
        // [65]
        order.clear();
        order.extend(g.heads.iter().enumerate().map(|(i, h)| (h.start, h.wave, i as u32)));
        order.sort_unstable();
        let first = heads.len();
        heads.reserve(order.len());
        for run in order.chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)) {
            match run {
                [(_, _, i)] => heads.extend(g.heads[*i as usize].played(w0, frames as u32, u32::MAX)),
                _ => fold.run(run.iter().map(|x| &g.heads[x.2 as usize]), w0, frames as u32, &mut heads),
            }
        }
        // [66]
        let (mut r, mut log2r) = (f32::NAN, 0u32);
        for piece in &mut heads[first..] {
            (piece.base, piece.cap) = at_wave[piece.base as usize];
            piece.bus = bus;
            if piece.r.to_bits() != r.to_bits() {
                r = piece.r;
                log2r = if r > 0.0 { ((r as f64).log2() as f32).to_bits() } else { 0 };
            }
            piece.split = log2r;
        }
    }
    (ins, heads)
}

impl Counts {
    fn absorb(&mut self, o: &Counts) {
        self.notes += o.notes;
        self.groups += o.groups;
        self.waves += o.waves;
        self.entries += o.entries;
        self.released_on_head += o.released_on_head;
        self.cut += o.cut;
        self.tails_cut += o.tails_cut;
        self.kernel_frames += o.kernel_frames;
        self.head_frames += o.head_frames;
        self.render_secs += o.render_secs;
        self.build_secs += o.build_secs;
        self.mix_secs += o.mix_secs;
        self.conv_secs += o.conv_secs;
        self.plan_secs += o.plan_secs;
    }

    fn add_plan(&mut self, c: PlanCounts) {
        self.notes += c.notes;
        self.cut += c.cut;
        self.released_on_head += c.released_on_head;
        self.tails_cut += c.tails_cut;
    }
}

/// One group's output: where it starts, both channels, and the kernel and head \[67\]
struct Played {
    lo: u64,
    l: Vec<f32>,
    r: Vec<f32>,
    kernel_frames: u64,
    head_frames: u64,
}

/// One group: every kernel at its impulses and every head correction, in f64 \[68\]
fn play_group(g: &Group, kernels: &[Vec<f32>], sustains: &[&[f32]], coef: Option<[f32; 4]>) -> Played {
    let lo = g.lo.min(g.impulses.iter().map(|i| i.frame).min().unwrap_or(u64::MAX));
    let len = (g.hi - lo) as usize;
    let mut l = vec![0f64; len];
    let mut r = vec![0f64; len];
    let mut kf = 0u64;
    for imp in &g.impulses {
        let k = &kernels[imp.entry as usize];
        let at = (imp.frame - lo) as usize;
        let (gl, gr, w) = (imp.gain[0] as f64, imp.gain[1] as f64, imp.weight as f64);
        for (i, &x) in k.iter().enumerate() {
            let x = x as f64 * w;
            l[at + i] += x * gl;
            r[at + i] += x * gr;
        }
        kf += k.len() as u64 * imp.weight as u64;
    }
    let mut hf = 0u64;
    for h in &g.heads {
        let a = &sustains[h.wave as usize];
        for piece in h.clip(h.start, h.span(), u32::MAX).into_iter().flatten() {
            hf += play_head(&piece, a, &mut l, &mut r, (h.start - lo) as usize) * h.weight as u64;
        }
    }
    let mut lf: Vec<f32> = l.iter().map(|&x| x as f32).collect();
    let mut rf: Vec<f32> = r.iter().map(|&x| x as f32).collect();
    if let Some(c) = coef {
        let zl = biquad(&mut lf, c);
        let zr = biquad(&mut rf, c);
        // [69]
        let (tl, tr) = (ring_out(zl, c), ring_out(zr, c));
        let n = tl.len().max(tr.len());
        lf.extend(tl.iter().copied().chain(std::iter::repeat(0.0)).take(n));
        rf.extend(tr.iter().copied().chain(std::iter::repeat(0.0)).take(n));
    }
    Played { lo, l: lf, r: rf, kernel_frames: kf, head_frames: hf }
}

/// What a biquad left in state `z` plays with nothing more going in, until the \[70\]
fn ring_out(z: (f32, f32), c: [f32; 4]) -> Vec<f32> {
    let (mut z1, mut z2) = z;
    let mut out = Vec::new();
    while (z1.abs() > 1e-12 || z2.abs() > 1e-12) && out.len() < 96_000 {
        let y = z1;
        z1 = -c[2] * y + z2;
        z2 = -c[3] * y;
        out.push(y);
    }
    out
}

/// `head.wgsl`'s `apply`, for a piece clipped from its note's own start: the \[71\]
fn play_head(h: &Head, a: &[f32], l: &mut [f64], r: &mut [f64], at: usize) -> u64 {
    let mut f = h.f0 as f64;
    let (rr, fade) = (h.r as f64, h.fade as f64);
    let w = h.weight as f64;
    let mut n = 0u32;
    while n < h.len {
        let src = (h.src0 + n) as usize;
        if src >= a.len() {
            break;
        }
        let out = at + (h.out0 + n) as usize;
        let fe = if n < h.dead { 0.0 } else { f };
        let x = a[src] as f64;
        let (gl, gr) = (h.gain_l as f64 + h.dg_l as f64 * n as f64, h.gain_r as f64 + h.dg_r as f64 * n as f64);
        l[out] += x * (gl - fe * h.gi_l as f64) * w;
        r[out] += x * (gr - fe * h.gi_r as f64) * w;
        if n + 1 >= h.hold {
            f = f * rr + fade;
        }
        n += 1;
    }
    n as u64
}

/// `bus.wgsl`'s biquad: transposed direct form II, `b2 == b0`, from rest. \[72\]
fn biquad(x: &mut [f32], c: [f32; 4]) -> (f32, f32) {
    let (mut z1, mut z2) = (0f32, 0f32);
    for s in x.iter_mut() {
        let v = *s;
        let y = c[0] * v + z1;
        z1 = c[1] * v - c[2] * y + z2;
        z2 = c[0] * v - c[3] * y;
        *s = y;
    }
    (z1, z2)
}

/// What the bucket pass knows once the file is read.
#[derive(Clone, Copy)]
struct PassEnd {
    /// Blocks its driver committed with events still to come, plus the first \[73\]
    stream_blocks: u64,
    driver_stats: crate::driver::DriverStats,
    /// Where the render ends, `BucketPass::end`.
    end: u64,
    /// The most of a group's records in memory at once: the parsed render's.
    peak_records: u64,
}

/// The bucket pass through `driver`, to the end of the file. Windowed, every \[74\]
fn run_pass_on(
    cfg: &Config,
    bank: Arc<Bank>,
    mut driver: crate::Driver,
    dir: &Path,
    window: Option<u64>,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<(BucketPass, PassEnd)> {
    let mut pass = BucketPass::new(cfg, bank, dir)?;
    if let Some(w) = window {
        pass = pass.windowed(w);
    }
    let mut out = vec![0f32; cfg.block_samples()];
    let mut stream_blocks = None;
    loop {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!("stopped");
        }
        let more = driver.next_block(&mut pass, &mut out)?;
        if stream_blocks.is_none() && !driver.stats.last_events_pending {
            stream_blocks = Some(driver.stats.blocks);
        }
        if !more {
            break;
        }
    }
    pass.seal_all()?;
    let end = PassEnd {
        stream_blocks: stream_blocks.unwrap_or(driver.stats.blocks),
        driver_stats: driver.stats,
        end: pass.end(),
        peak_records: 0,
    };
    Ok((pass, end))
}

/// The per-key render's window and the carry past it, fixed before the file \[75\]
pub fn window_for(cfg: &Config, bank: &Bank) -> (usize, usize) {
    let sh = Shape::new(cfg, bank);
    let carry = crate::merlin::kmgpu::KmGpu::size_for(sh.lags * sh.p);
    let mut filters: Table<FilterKey> = Table::default();
    for p in bank.params.iter() {
        filters.intern(FilterKey::of(crate::merlin::plan::bus_filter(p, cfg.merlin_filter_cents)));
    }
    let most = if cfg.merlin_km_window == 0 { WINDOW_MAX } else { cfg.merlin_km_window as usize };
    let frames = crate::merlin::kmgpu::KmGpu::window_frames(BUS_BYTES, filters.len(), carry).min(most).max(carry);
    (frames, carry)
}

/// The render thread, which runs the whole per-key render: \[76\]
fn run_keys(
    cfg: &Config,
    bank: Arc<Bank>,
    midi: &Path,
    dir: &Path,
    frames: usize,
    carry: usize,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<Rendered> {
    let t = std::time::Instant::now();
    let mut km = KeyMajorCpu::with_gpu(cfg, bank.clone())?;
    let mut pcfg = cfg.clone();
    pcfg.coalesce = false;
    let driver = crate::Driver::open(&pcfg, bank.clone(), midi)?;
    let (mut pass, end) = run_pass_on(&pcfg, bank, driver, dir, Some(frames as u64), stop)?;
    let pass_secs = t.elapsed().as_secs_f64();
    let mut windows: Vec<SealedWindow> = Vec::new();
    while let Some(sw) = pass.take_ready() {
        windows.push(sw);
    }
    let spilled = pass.store_ref().spill_bytes();
    km.add_voices(pass.voices().to_vec())?;
    // The windows hold the spill file; the rest of the pass is done with.
    drop(pass);
    if km.filter_coef.len() * (frames + carry) * 8 > BUS_BYTES as usize {
        log::warn!(
            "merlin key-major: {} filters, past what the window was sized for; the buses grow past their budget",
            km.filter_coef.len()
        );
    }
    let keys: Vec<usize> = (0..KEYS).filter(|&k| windows.iter().any(|w| w.key_len(k) > 0)).collect();
    log::info!(
        "merlin key-major: {} notes parsed once in {pass_secs:.2}s, {} windows, {} keys with notes, {:.1} MiB spilled",
        end.driver_stats.notes,
        windows.len(),
        keys.len(),
        spilled as f64 / 1048576.0
    );
    // A key's ring-out reaches this many windows past its last record.
    let ring = RING_OUT.div_ceil(frames as u64) as usize;
    let (mut rendered, mut song) = (0u64, 0usize);
    for (j, &key) in keys.iter().enumerate() {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!("stopped");
        }
        let t_key = std::time::Instant::now();
        let first = windows.iter().position(|w| w.key_len(key) > 0).expect("a key with records");
        let last = windows.iter().rposition(|w| w.key_len(key) > 0).expect("a key with records");
        km.start_key()?;
        for sw in &windows[first..=(last + ring).min(windows.len() - 1)] {
            km.render_window(sw, key, frames, carry)?;
            song = song.max((sw.index as usize + 1) * frames);
            rendered += 1;
        }
        log::info!("merlin key-major: key {key:03} ({}/{}) in {:.2}s", j + 1, keys.len(), t_key.elapsed().as_secs_f64());
    }
    drop(windows);
    let master = km.gpu.as_mut().context("no device")?.read_song(song)?;
    km.counts.render_secs = t.elapsed().as_secs_f64();
    km.log_summary(cfg, rendered, frames, carry);
    Ok((master, end))
}

/// Note-ons a key group of the parsed render holds at most, unless one key \[77\]
const GROUP_NOTE_ONS: u64 = 32 << 20;

/// Records a key group is sized to hold: about 960 MiB at 24 bytes each.
const GROUP_RECORDS: u64 = 40 << 20;

/// Note-ons a group takes at most, whatever the last one folded to, so a \[78\]
const GROUP_NOTE_ONS_MAX: u64 = 96 << 20;

/// Some of one key's windows planned, for the packing thread.
struct PlannedKey {
    key: u8,
    parts: Vec<WindowPart>,
    /// What the buses need first: the key's first hand-over, or more of them.
    start: Option<Buses>,
    /// The last of the key: what its log line waits for.
    last: bool,
    /// A channel's bent notes: on the first of them the windows start adding \[79\]
    warp_begin: bool,
    warp_end: Option<(Arc<crate::merlin::keyparse::Warp>, u64, u64)>,
    frames: usize,
    /// The key's bus for each filter of the file.
    slot: Arc<Vec<u32>>,
    pc: crate::merlin::kmgpu::PackCfg,
    began: std::time::Instant,
}

/// A key's buses as a hand-over changes them.
enum Buses {
    /// A key whose filters are known: their coefficients, its buses in order.
    Start(Vec<Option<[f32; 4]>>),
    /// A streamed key's first: the coefficients so far, and room for more.
    Stream(Vec<Option<[f32; 4]>>, usize),
    /// More filters on a streamed key: all its coefficients.
    Grow(Vec<Option<[f32; 4]>>),
}

/// What the packing thread did, for the render's log and its length.
#[derive(Default)]
struct PackTally {
    pool_secs: f64,
    pack_secs: f64,
    queue_secs: f64,
    direct_jobs: u64,
    voice_jobs: u64,
    song: usize,
    rendered: u64,
    counts: Counts,
}

/// The parsed render's second half, on a thread of its own: each planned key's \[80\]
fn pack_keys(km: &KeyMajorCpu, keys: std::sync::mpsc::Receiver<PlannedKey>, dtx: std::sync::mpsc::SyncSender<Dev>, ahead: &Ahead) -> Result<PackTally> {
    let mut t = PackTally::default();
    let budget = km.cfg.merlin_km_pool_frames as usize;
    'keys: for job in keys {
        let PlannedKey { key, parts, start, last, warp_begin, warp_end, frames, slot, pc, began } = job;
        let sent = match start {
            None => true,
            Some(Buses::Start(coefs)) => dtx.send(Dev::StartKey(coefs, frames)).is_ok(),
            Some(Buses::Stream(coefs, room)) => dtx.send(Dev::StartStream(coefs, frames, room)).is_ok(),
            Some(Buses::Grow(coefs)) => dtx.send(Dev::Grow(coefs)).is_ok(),
        };
        if !sent || (warp_begin && dtx.send(Dev::WarpBegin).is_err()) {
            break;
        }
        for run in pool_runs(parts, budget) {
            let tp = std::time::Instant::now();
            let mut kp = km.key_pools(&run)?;
            t.pool_secs += tp.elapsed().as_secs_f64();
            t.counts.absorb(&kp.counts);
            let build = std::mem::replace(&mut kp.build, PoolBuild { tables: Vec::new(), total: 0, cpu: Vec::new(), direct: Vec::new() });
            t.direct_jobs += build.direct.len() as u64;
            t.voice_jobs += build.cpu.len() as u64 + build.tables.iter().map(|b| b.1.len() as u64).sum::<u64>();
            if dtx.send(Dev::Build(build)).is_err() {
                break 'keys;
            }
            // [81]
            let tp = std::time::Instant::now();
            let queued = t.queue_secs;
            let sent = km.prepare_key(&pc, &kp, run, frames, &slot, |p| {
                let index = p.w0 / frames as u64;
                t.counts.absorb(&p.counts);
                if p.ends {
                    t.song = t.song.max((index as usize + 1) * frames);
                    t.rendered += 1;
                }
                let bytes = p.bytes();
                let tq = std::time::Instant::now();
                let ok = ahead.reserve(bytes) && dtx.send(Dev::Window(p)).is_ok() && dtx.send(Dev::At(key, index)).is_ok();
                t.queue_secs += tq.elapsed().as_secs_f64();
                ok
            })?;
            t.pack_secs += tp.elapsed().as_secs_f64() - (t.queue_secs - queued);
            if !sent {
                break 'keys;
            }
        }
        if let Some((w, lo, hi)) = warp_end {
            t.song = t.song.max(w.frame(hi as f64).ceil() as usize);
            if dtx.send(Dev::WarpEnd(w, lo, hi)).is_err() {
                break;
            }
        }
        if last {
            log::info!("merlin key-major: key {key:03} in {:.2}s", began.elapsed().as_secs_f64());
            if dtx.send(Dev::KeyDone(key)).is_err() {
                break;
            }
        }
    }
    Ok(t)
}

/// A voice row as the parsed render interns it: compared and hashed by its \[82\]
#[derive(Clone, Copy)]
struct VoiceKey(VoiceDesc);

impl VoiceKey {
    fn words(&self) -> (&[u8], &[u8], Option<&[u8]>, u32) {
        let v = &self.0;
        (bytemuck::bytes_of(&v.cmd), bytemuck::bytes_of(&v.params), v.menv.as_ref().map(bytemuck::bytes_of), v.smp_base_r)
    }
}

impl PartialEq for VoiceKey {
    fn eq(&self, o: &Self) -> bool {
        self.words() == o.words()
    }
}

impl Eq for VoiceKey {}

impl std::hash::Hash for VoiceKey {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.words().hash(h);
    }
}

/// The per-key render fed by merlin's own MIDI front end, with no driver, no \[83\]
fn run_keys_parsed(cfg: &Config, bank: Arc<Bank>, midi: &Path, dir: &Path, frames: usize, carry: usize, progress: &Arc<MerlinProgress>) -> Result<Rendered> {
    use crate::merlin::keyparse::Prepass;
    use crate::merlin::keyrec::unsupported;
    let t = std::time::Instant::now();
    let stop = &*progress.cancel_flag();
    let mut km = KeyMajorCpu::with_gpu(cfg, bank.clone())?;
    let pre = Prepass::scan_watched(midi, cfg, dir, Some(progress.clone() as Arc<dyn crate::merlin::keyparse::Watch>))?;
    progress.begin_rendering(&pre.key_notes(), pre.song_frames().div_ceil(frames as u64).max(1), pre.store_size());
    log::info!("merlin key-major: key parser's pre-pass {:.2}s, {} note-ons over {} events", pre.secs, pre.notes, pre.events);
    if let Some((bytes, events)) = pre.store_size() {
        log::info!("merlin key-major: key store {:.2} GB, {events} events, {:.2} bytes each", bytes as f64 / 1e9, bytes as f64 / events.max(1) as f64);
    }
    let mut voice_of: FxHashMap<VoiceKey, u32> = FxHashMap::default();
    let (fade, two_p) = (cfg.steal_fade_frames as u64, 2 * km.shape.p as u64);
    let (mut end, mut eof) = (0u64, None);
    let (mut flags, mut note_ons, mut parse_secs, mut peak_records) = (0u32, 0u64, 0f64, 0u64);
    let mut stats = crate::driver::DriverStats::default();
    // [84]
    let (pre, bank_ref) = (&pre, &bank);
    let warps = cfg.merlin_km_warp.then(|| pre.warps(&bank, cfg));
    let warps = warps.as_deref();
    let (mut warped_notes, mut warp_passes) = (0u64, 0u64);
    let mut wait_secs = 0f64;
    let mut inside = crate::merlin::keyparse::ExtractStats::default();
    km.gpu.as_mut().context("no device")?.attach_builder(cfg, &bank)?;
    km.gpu.as_mut().context("no device")?.time_device();
    let mut gpu = km.gpu.take().context("no device")?;
    let (frames_c, carry_c) = (frames, carry);
    // [85]
    let (mut bin_secs, mut plan_secs, mut hand_secs) = (0f64, 0f64, 0f64);
    let mut bin_stages = [0f64; 3];
    let packing = KeyMajorCpu::new(cfg, bank.clone());
    let scoped = std::thread::scope(|scope| -> Result<((f64, f64), PackTally)> {
        // [86]
        let (dtx, drx) = std::sync::mpsc::sync_channel::<Dev>(WINDOWS_AHEAD);
        // [87]
        let ahead = Arc::new(Ahead::new(cfg.merlin_km_ahead_bytes));
        let fail_at = cfg.merlin_km_fail_device_at;
        let ahead_dev = ahead.clone();
        let gpu_ref = &mut gpu;
        let said = progress.clone();
        let dev = scope.spawn(move || -> Result<(f64, f64)> {
            // Closed however this ends, or the packing thread waits for room for ever.
            let _closes = CloseOnDrop(ahead_dev.clone());
            let (mut put, mut mix) = (0f64, 0f64);
            let (mut idle, mut build, mut start) = (0f64, 0f64, 0f64);
            let (mut waited, mut windows) = (0u64, 0u32);
            loop {
                let t = std::time::Instant::now();
                let Ok(m) = drx.recv() else { break };
                idle += t.elapsed().as_secs_f64();
                waited += t.elapsed().as_nanos() as u64;
                let t = std::time::Instant::now();
                match m {
                    Dev::Begin(coefs, reply) => {
                        let r = gpu_ref.begin(&coefs, frames_c, carry_c).and_then(|_| gpu_ref.pack_cfg());
                        let failed = r.is_err();
                        let _ = reply.send(r);
                        if failed {
                            anyhow::bail!("the device did not begin");
                        }
                    }
                    Dev::StartKey(coefs, frames) => {
                        gpu_ref.start_key_with(&coefs, frames)?;
                        start += t.elapsed().as_secs_f64();
                    }
                    Dev::StartStream(coefs, frames, room) => {
                        gpu_ref.start_key_room(&coefs, frames, room)?;
                        start += t.elapsed().as_secs_f64();
                    }
                    Dev::Grow(coefs) => gpu_ref.grow(&coefs)?,
                    Dev::WarpBegin => gpu_ref.warp_begin(),
                    Dev::WarpEnd(w, lo, hi) => gpu_ref.warp_end(&w, lo, hi)?,
                    Dev::Build(b) => {
                        gpu_ref.build_pool(&b.tables, b.total, &b.cpu, &b.direct)?;
                        build += t.elapsed().as_secs_f64();
                    }
                    Dev::At(key, index) => said.at(key, index),
                    Dev::KeyDone(key) => said.key_done(key),
                    Dev::Window(p) => {
                        let bytes = p.bytes();
                        windows += 1;
                        if windows == fail_at {
                            anyhow::bail!("the device failed at window {windows} (merlin_km_fail_device_at)");
                        }
                        let (a, b) = submit(gpu_ref, p)?;
                        put += a;
                        mix += b;
                        ahead_dev.release(bytes);
                    }
                }
                said.add_device(t.elapsed().as_nanos() as u64, std::mem::take(&mut waited));
            }
            let t = std::time::Instant::now();
            gpu_ref.time_flush();
            log::info!(
                "merlin key-major: device's thread: waiting for work {idle:.2}s, encoding windows {put:.2}s, ending windows {mix:.2}s, \
                 sustain pools {build:.2}s, starting keys {start:.2}s, last wait {:.2}s",
                t.elapsed().as_secs_f64()
            );
            if let Some(g) = gpu_ref.times.as_ref() {
                let a = g.secs;
                log::info!(
                    "merlin key-major: device time: kernel spectra {:.2}s, direct {:.2}s, convolution {:.2}s, heads {:.2}s, \
                     master growth {:.2}s, key clears {:.2}s, sustains {:.2}s, filters, mix and carry {:.2}s",
                    a[0],
                    a[1],
                    a[2],
                    a[3],
                    a[4],
                    a[5],
                    a[6],
                    a[7]
                );
            }
            Ok((put, mix))
        });
        // [88]
        let (ktx, krx) = std::sync::mpsc::sync_channel::<PlannedKey>(0);
        let packer = {
            let (dtx, ahead, packing) = (dtx.clone(), ahead.clone(), &packing);
            scope.spawn(move || pack_keys(packing, krx, dtx, &ahead))
        };
        let mut hand = HandOver { ktx: &ktx, dtx: &dtx, pc: None, carry, bin_secs: 0.0, bin_stages: [0.0; 3], plan_secs: 0.0, hand_secs: 0.0 };
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Parsed>>(0);
        scope.spawn(move || {
            // [89]
            let notes = pre.key_notes();
            let keys: Vec<u8> = (0..KEYS as u8).filter(|&k| notes[k as usize] > 0).collect();
            let (mut at, mut budget) = (0usize, GROUP_NOTE_ONS);
            while at < keys.len() {
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let (from, mut held) = (at, 0u64);
                let big = |k: u8| notes[k as usize] > cfg.merlin_km_stream_notes as u64;
                while at < keys.len() && (at == from || (held + notes[keys[at] as usize] <= budget && !big(keys[at]) && !big(keys[from]))) {
                    held += notes[keys[at] as usize];
                    at += 1;
                }
                let streamed = big(keys[from]);
                let r = if streamed {
                    let send = |part: KeyPart| tx.send(Ok(Parsed::Part(keys[from], part))).is_ok();
                    parse_group(pre, cfg, bank_ref, keys[from..at].to_vec(), warps, Some(&send))
                } else {
                    parse_group(pre, cfg, bank_ref, keys[from..at].to_vec(), warps, None)
                };
                if let Ok(g) = &r {
                    budget = (GROUP_RECORDS.saturating_mul(held) / g.records.max(1)).clamp(GROUP_NOTE_ONS / 4, GROUP_NOTE_ONS_MAX);
                }
                let failed = r.is_err();
                if tx.send(r.map(Parsed::Group)).is_err() || failed {
                    break;
                }
            }
        });
        let mut stream: Option<KeyRun> = None;
        let stream_held = cfg.merlin_km_stream_held as usize;
        loop {
            let tw = std::time::Instant::now();
            let Ok(item) = rx.recv() else { break };
            wait_secs += tw.elapsed().as_secs_f64();
            let g = match item? {
                Parsed::Part(key, part) => {
                    if stream.is_none() {
                        km.voices.clear();
                        km.infos.clear();
                        stream = Some(KeyRun::streaming(key, frames));
                    }
                    let run = stream.as_mut().expect("made above");
                    let frontier = part.frontier;
                    run.feed(&mut km, part, fade, two_p)?;
                    peak_records = peak_records.max(run.recs.recs.len() as u64);
                    if frontier != u64::MAX {
                        // [90]
                        let sealed = (frontier / frames as u64).min(run.last.saturating_add(1));
                        if !run.hand(&km, &mut hand, sealed, false)? {
                            break;
                        }
                        if run.recs.recs.len() > stream_held {
                            run.spill_out(dir)?;
                        }
                    }
                    continue;
                }
                Parsed::Group(g) => g,
            };
            let ParsedGroup { keys: g, mut built, st, secs, records, warped } = g;
            warp_passes += warped.len() as u64;
            warped_notes += warped.iter().map(|w| w.2.iter().map(|r| r.count as u64).sum::<u64>()).sum::<u64>();
            // [91]
            let bent = |key: u8, layers: &[(u32, [f32; 2])], gains: &[ChannelEnds], km: &KeyMajorCpu, hand: &mut HandOver<'_>, end: &mut u64, warped: &mut Vec<(u8, u8, Vec<KeyRec>)>| -> Result<bool> {
                let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(warped).into_iter().partition(|w| w.0 == key);
                *warped = rest;
                for (_, ch, recs) in mine {
                    let kr = KeyRecs { recs, layers: layers.to_vec(), gains: gains.to_vec() };
                    let mut run = KeyRun::whole(key, kr, km, frames, carry, fade, two_p);
                    let w = warps.context("bent notes with no clocks")?[ch as usize].clone();
                    *end = (*end).max(w.frame(run.key_end as f64).ceil() as u64 + RING_OUT);
                    let stop_at = run.stop_at();
                    run.warp = Some(w);
                    if !run.hand(km, hand, stop_at + 1, true)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            };
            let mut warped = warped;
            if stream.is_none() {
                peak_records = peak_records.max(records);
            }
            parse_secs += secs;
            inside.load_secs += st.load_secs;
            inside.read_secs += st.read_secs - st.load_secs;
            inside.merge_secs += st.merge_secs;
            inside.match_secs += st.match_secs;
            inside.sink_secs += st.sink_secs;
            inside.after_secs += st.after_secs;
            flags |= st.unsupported;
            note_ons += st.note_ons;
            eof = eof.or(st.eof);
            if let Some(mut run) = stream.take() {
                // Its last part came ahead of this: everything is released.
                if run.first != u64::MAX {
                    log::info!("merlin key-major: key {:03} streamed, {records} records, {:.2}s parsing", run.key, secs);
                    end = end.max(run.key_end + RING_OUT);
                    let stop_at = run.stop_at();
                    if !run.hand(&km, &mut hand, stop_at + 1, true)? {
                        break;
                    }
                }
                let (key, layers, gains) = (run.key, std::mem::take(&mut run.recs.layers), std::mem::take(&mut run.recs.gains));
                drop(run);
                if !bent(key, &layers, &gains, &km, &mut hand, &mut end, &mut warped)? {
                    break;
                }
                continue;
            }
            let mut recs: Vec<KeyRecs> = (0..KEYS).map(|_| KeyRecs::default()).collect();
            // [92]
            voice_of.clear();
            km.voices.clear();
            km.infos.clear();
            let mut fresh: Vec<VoiceDesc> = Vec::new();
            for (&key, (descs, out)) in g.iter().zip(built.iter_mut()) {
                let ids: Vec<u32> = descs
                    .iter()
                    .map(|d| {
                        let next = (km.voices.len() + fresh.len()) as u32;
                        *voice_of.entry(VoiceKey(*d)).or_insert_with(|| {
                            fresh.push(*d);
                            next
                        })
                    })
                    .collect();
                out.layers.iter_mut().for_each(|l| l.0 = ids[l.0 as usize]);
                recs[key as usize] = std::mem::take(out);
            }
            drop(built);
            km.add_voices(fresh)?;
            for &key in &g {
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    anyhow::bail!("stopped");
                }
                let key_recs = std::mem::take(&mut recs[key as usize]);
                let tables = (key_recs.layers.clone(), key_recs.gains.clone());
                if !key_recs.recs.is_empty() {
                    let mut run = KeyRun::whole(key, key_recs, &km, frames, carry, fade, two_p);
                    end = end.max(run.key_end + RING_OUT);
                    let stop_at = run.stop_at();
                    if !run.hand(&km, &mut hand, stop_at + 1, true)? {
                        break;
                    }
                }
                if !bent(key, &tables.0, &tables.1, &km, &mut hand, &mut end, &mut warped)? {
                    break;
                }
            }
        }
        (bin_secs, plan_secs, hand_secs) = (hand.bin_secs, hand.plan_secs, hand.hand_secs);
        bin_stages = hand.bin_stages;
        km.counts.plan_secs += plan_secs;
        drop(ktx);
        let packed = packer.join().map_err(|_| anyhow::anyhow!("the packing thread panicked"));
        drop(dtx);
        let device = dev.join().map_err(|_| anyhow::anyhow!("the device's thread panicked"))?;
        Ok((device?, packed??))
    });
    // Stopped, whatever else the threads said on the way out.
    progress.check()?;
    let ((put_secs, mix_secs), tally) = scoped?;
    km.gpu = Some(gpu);
    km.counts.conv_secs += put_secs;
    km.counts.mix_secs += mix_secs;
    km.counts.absorb(&tally.counts);
    let PackTally { pool_secs, pack_secs, queue_secs, direct_jobs, voice_jobs, song, rendered, .. } = tally;
    if flags != 0 {
        log::warn!("merlin key-major: the key parser does not build these yet, and rendered without them: {}", unsupported::names(flags).join(", "));
    }
    let master = km.gpu.as_mut().context("no device")?.read_song(song)?;
    km.counts.render_secs = t.elapsed().as_secs_f64();
    log::info!(
        "merlin key-major: the render's thread: waiting for the parser {wait_secs:.2}s, binning {bin_secs:.2}s, planning {plan_secs:.2}s, \
         waiting for the packing thread {hand_secs:.2}s; the packing thread: pools {pool_secs:.2}s, packing {pack_secs:.2}s, \
         waiting for the device's queue {queue_secs:.2}s; {direct_jobs} sustains frame by frame, {voice_jobs} by voice"
    );
    log::info!(
        "merlin key-major: binning, of it: counting {:.2}s, slicing {:.2}s, dropping what is done {:.2}s; the parser, of it: keys from the store {:.2}s, decoding {:.2}s,          merging {:.2}s, pairing {:.2}s, building records {:.2}s, held by the render {:.2}s",
        bin_stages[0],
        bin_stages[1],
        bin_stages[2],
        inside.load_secs,
        inside.read_secs,
        inside.merge_secs,
        inside.match_secs,
        inside.sink_secs,
        inside.after_secs
    );
    log::info!(
        "merlin key-major: key parser and layers {parse_secs:.2}s alongside the render, {wait_secs:.2}s of it waited for, of {:.2}s; {} voice rows",
        km.counts.render_secs,
        km.voices.len()
    );
    km.log_summary(cfg, rendered, frames, carry);
    let bf = cfg.block_frames as u64;
    stats.notes = note_ons;
    stats.events = pre.events;
    let stream_blocks = eof.map_or(1, |f| f / bf + 1);
    log::info!("merlin key-major: at most {peak_records} records of a group in memory at once");
    if cfg.merlin_km_warp {
        log::info!("merlin key-major: {warped_notes} notes a bend moves under, in {warp_passes} (key, channel) passes");
    }
    Ok((master, PassEnd { stream_blocks, driver_stats: stats, end, peak_records }))
}

/// A key group as merlin's front end hands it to the render: the group, each \[93\]
struct ParsedGroup {
    keys: Vec<u8>,
    built: Vec<(Vec<VoiceDesc>, KeyRecs)>,
    st: crate::merlin::keyparse::ExtractStats,
    secs: f64,
    records: u64,
    /// The notes a bend moves under: `(key, channel, records)`, in the tables \[94\]
    warped: Vec<(u8, u8, Vec<KeyRec>)>,
}

/// What the parser's thread hands the render.
enum Parsed {
    /// A key alone over the budget, part of it.
    Part(u8, KeyPart),
    /// A group, or the last of a key that came in parts.
    Group(ParsedGroup),
}

/// A streamed key's next records: what its tables gained, and the records, \[95\]
struct KeyPart {
    descs: Vec<VoiceDesc>,
    layers: Vec<(u32, [f32; 2])>,
    gains: Vec<ChannelEnds>,
    recs: Vec<KeyRec>,
    /// A frame every note starting before which has been handed over.
    frontier: u64,
}

/// What a key's hand-over shares with every other.
struct HandOver<'a> {
    ktx: &'a std::sync::mpsc::SyncSender<PlannedKey>,
    dtx: &'a std::sync::mpsc::SyncSender<Dev>,
    /// What `pack` reads, from the device, once its buses are made.
    pc: Option<crate::merlin::kmgpu::PackCfg>,
    carry: usize,
    bin_secs: f64,
    /// Within `bin_secs`: counting, slicing, and dropping what is done.
    bin_stages: [f64; 3],
    plan_secs: f64,
    hand_secs: f64,
}

/// A key's records past what it may hold in memory, in a temporary file by the \[96\]
struct KeySpill {
    file: std::fs::File,
    next: u64,
    /// Each window's `(offset, records)`.
    bins: FxHashMap<u64, Vec<(u64, u32)>>,
}

impl KeySpill {
    fn add(&mut self, w: u64, recs: &[KeyRec]) -> Result<()> {
        let bytes: &[u8] = bytemuck::cast_slice(recs);
        crate::merlin::keystore::write_all_at(&self.file, bytes, self.next)?;
        self.bins.entry(w).or_default().push((self.next, recs.len() as u32));
        self.next += bytes.len() as u64;
        Ok(())
    }

    /// Records of window `w`.
    fn count(&self, w: u64) -> usize {
        self.bins.get(&w).map_or(0, |b| b.iter().map(|c| c.1 as usize).sum())
    }

    /// Window `w`'s records, onto `out`.
    fn read(&self, w: u64, out: &mut Vec<KeyRec>) -> Result<()> {
        for &(off, n) in self.bins.get(&w).map_or(&[][..], |b| &b[..]) {
            let at = out.len();
            out.resize(at + n as usize, KeyRec::zeroed());
            crate::merlin::bucket::read_exact_at(&self.file, bytemuck::cast_slice_mut(&mut out[at..]), off)?;
        }
        Ok(())
    }

    /// Windows before `w` are done with.
    fn done_below(&mut self, w: u64) {
        self.bins.retain(|&b, _| b >= w);
        if self.bins.is_empty() {
            self.next = 0;
        }
    }
}

/// One key on its way to the packing thread: its records, its buses, and the \[97\]
struct KeyRun {
    key: u8,
    began: std::time::Instant,
    frames: usize,
    recs: KeyRecs,
    /// Each of `recs`, the first and last window it reaches: worked out once, \[98\]
    spans: Vec<[u32; 2]>,
    /// The filter of each bus, and each filter's bus.
    used: Vec<u32>,
    slot: Vec<u32>,
    /// Buses the device knows of, so a hand-over can say there are more.
    told: usize,
    streamed: bool,
    /// The first window any record reaches, the last, and where the key ends.
    first: u64,
    last: u64,
    key_end: u64,
    /// The next window to hand over, once one has been.
    at: Option<u64>,
    /// Voice rows taken so far.
    voices: usize,
    spill: Option<KeySpill>,
    /// For one channel's bent notes, the clock they are read back through.
    warp: Option<Arc<crate::merlin::keyparse::Warp>>,
}

/// `plan_key`'s reach, as windows: the first and last a record reaches, its \[99\]
fn reach(r: &KeyRec, tail: u64, fr: u64, fade: u64, two_p: u64) -> (u64, u64, u64) {
    let heard = r.duration as u64 + fade.max(1) + tail + 1;
    (r.start / fr, (r.start + heard - 1) / fr, r.start + r.duration as u64 + fade + tail + two_p)
}

impl KeyRun {
    /// A key whose records are all here. Its buses are the filters they use, in \[100\]
    fn whole(key: u8, recs: KeyRecs, km: &KeyMajorCpu, frames: usize, carry: usize, fade: u64, two_p: u64) -> Self {
        let began = std::time::Instant::now();
        let mut used: Vec<u32> = recs.recs.par_iter().map(|r| km.infos[recs.voice(r) as usize].filter).collect();
        used.par_sort_unstable();
        used.dedup();
        let mut slot = vec![u32::MAX; km.filter_coef.len()];
        for (i, &f) in used.iter().enumerate() {
            slot[f as usize] = i as u32;
        }
        let frames = crate::merlin::kmgpu::KmGpu::window_frames(BUS_BYTES, used.len(), carry).min(frames).max(carry);
        let mut run = Self { key, began, frames, recs, spans: Vec::new(), used, slot, told: 0, streamed: false, first: u64::MAX, last: 0, key_end: 0, at: None, voices: 0, spill: None, warp: None };
        run.reckon(km, fade, two_p, 0);
        run
    }

    /// A key that will come in parts, at the render's own windows: its filters \[101\]
    fn streaming(key: u8, frames: usize) -> Self {
        Self {
            key,
            began: std::time::Instant::now(),
            frames,
            recs: KeyRecs::default(),
            spans: Vec::new(),
            used: Vec::new(),
            slot: Vec::new(),
            told: 0,
            streamed: true,
            first: u64::MAX,
            last: 0,
            key_end: 0,
            at: None,
            voices: 0,
            spill: None,
            warp: None,
        }
    }

    /// `first`, `last` and `key_end` over the records from `from` on.
    fn reckon(&mut self, km: &KeyMajorCpu, fade: u64, two_p: u64, from: usize) {
        let (kr, fr) = (&self.recs, self.frames as u64);
        self.spans.resize(kr.recs.len(), [0; 2]);
        let (a, b, e) = kr.recs[from..]
            .par_iter()
            .zip(self.spans[from..].par_iter_mut())
            .map(|(r, span)| {
                let (a, b, e) = reach(r, km.infos[kr.voice(r) as usize].tail as u64, fr, fade, two_p);
                *span = [a as u32, b as u32];
                (a, b, e)
            })
            .reduce(|| (u64::MAX, 0, 0), |a, b| (a.0.min(b.0), a.1.max(b.1), a.2.max(b.2)));
        self.first = self.first.min(a);
        self.last = self.last.max(b);
        self.key_end = self.key_end.max(e);
    }

    /// The last window the key is heard in.
    fn stop_at(&self) -> u64 {
        let fr = self.frames as u64;
        (self.last + RING_OUT.div_ceil(fr)).min((self.key_end + RING_OUT).div_ceil(fr).max(1) - 1).max(self.last)
    }

    /// A part's tables and voice rows taken on, new filters given buses in the \[102\]
    fn feed(&mut self, km: &mut KeyMajorCpu, part: KeyPart, fade: u64, two_p: u64) -> Result<()> {
        let from = self.recs.recs.len();
        let KeyPart { descs, layers, gains, recs, .. } = part;
        km.add_voices(descs)?;
        for i in self.voices..km.infos.len() {
            let f = km.infos[i].filter as usize;
            self.slot.resize(km.filter_coef.len(), u32::MAX);
            if self.slot[f] == u32::MAX {
                self.slot[f] = self.used.len() as u32;
                self.used.push(f as u32);
            }
        }
        self.voices = km.infos.len();
        self.recs.layers.extend(layers);
        self.recs.gains.extend(gains);
        self.recs.recs.extend(recs);
        self.reckon(km, fade, two_p, from);
        Ok(())
    }

    /// Everything in memory into its windows' bins on disk.
    fn spill_out(&mut self, dir: &Path) -> Result<()> {
        if self.spill.is_none() {
            self.spill = Some(KeySpill { file: crate::merlin::keystore::temp_file(dir)?, next: 0, bins: FxHashMap::default() });
        }
        let t = std::time::Instant::now();
        let at = self.at.unwrap_or(0);
        let mut by: std::collections::BTreeMap<u64, Vec<KeyRec>> = Default::default();
        for (r, &[a, b]) in self.recs.recs.iter().zip(&self.spans) {
            for w in (a as u64).max(at)..=b as u64 {
                by.entry(w).or_default().push(*r);
            }
        }
        let sp = self.spill.as_mut().expect("made above");
        for (w, v) in by {
            sp.add(w, &v)?;
        }
        log::info!("merlin key-major: key {:03}: {} records to disk in {:.2}s, {:.1} MiB so far", self.key, self.recs.recs.len(), t.elapsed().as_secs_f64(), sp.next as f64 / 1048576.0);
        self.recs.recs = Vec::new();
        self.spans = Vec::new();
        Ok(())
    }

    /// Windows up to `upto` to the packing thread, a batch of them at a time, \[103\]
    fn hand(&mut self, km: &KeyMajorCpu, h: &mut HandOver<'_>, upto: u64, last: bool) -> Result<bool> {
        let t_bin = std::time::Instant::now();
        let mut warp_begin = self.warp.is_some() && self.at.is_none();
        // [104]
        let at = self.at.unwrap_or(self.first);
        if at >= upto {
            if last {
                let job = PlannedKey { key: self.key, parts: Vec::new(), start: None, last, warp_begin: false, warp_end: None, frames: self.frames, slot: Arc::new(Vec::new()), pc: h.pc.as_ref().context("a key with nothing handed over")?.with_frames(self.frames), began: self.began };
                return Ok(h.ktx.send(job).is_ok());
            }
            return Ok(true);
        }
        let (frames, fr) = (self.frames, self.frames as u64);
        let nw = (upto - at) as usize;
        let cfg = &km.cfg;
        let base = self.recs.recs.len();
        // [105]
        let hits: Vec<Vec<u32>> = self
            .spans
            .chunks(BIN_RUN)
            .map(|c| {
                let mut n = vec![0u32; nw];
                for &[a, b] in c {
                    for w in (a as u64).max(at)..=(b as u64).min(upto - 1) {
                        n[(w - at) as usize] += 1;
                    }
                }
                n
            })
            .collect();
        h.bin_stages[0] += t_bin.elapsed().as_secs_f64();
        let spilled = |w: usize| self.spill.as_ref().map_or(0, |s| s.count(at + w as u64));
        let held: Vec<usize> = (0..nw).map(|w| hits.iter().map(|h| h[w] as usize).sum::<usize>() + spilled(w)).collect();
        let slice_records = cfg.merlin_km_slice_records.max(1) as usize;
        if h.pc.is_none() {
            let (rtx, rrx) = std::sync::mpsc::sync_channel(1);
            if h.dtx.send(Dev::Begin(vec![None], rtx)).is_err() {
                return Ok(false);
            }
            h.pc = Some(rrx.recv().context("the device's thread stopped")??);
        }
        let pcr = h.pc.as_ref().expect("begun above").with_frames(frames);
        h.bin_secs += t_bin.elapsed().as_secs_f64();
        let plan_records = cfg.merlin_km_plan_records.max(1) as usize;
        let slot = Arc::new(self.slot.clone());
        let coefs = |run: &Self| -> Vec<Option<[f32; 4]>> { run.used.iter().map(|&f| km.filter_coef[f as usize]).collect() };
        let mut start = if self.told == 0 {
            Some(if self.streamed {
                let room = (BUS_BYTES as usize / (8 * (frames + h.carry))).max(self.used.len());
                Buses::Stream(coefs(self), room)
            } else {
                Buses::Start(coefs(self))
            })
        } else if self.used.len() > self.told {
            Some(Buses::Grow(coefs(self)))
        } else {
            None
        };
        self.told = self.used.len();
        let mut w = 0usize;
        while w < nw {
            let (mut e, mut sum) = (w, 0usize);
            while e < nw && (e == w || sum + held[e] <= plan_records) {
                sum += held[e];
                e += 1;
            }
            let tb = std::time::Instant::now();
            // [106]
            let mut loaded: Vec<(usize, usize)> = Vec::with_capacity(e - w);
            if let Some(sp) = &self.spill {
                for win in w..e {
                    let lo = self.recs.recs.len();
                    sp.read(at + win as u64, &mut self.recs.recs)?;
                    loaded.push((lo, self.recs.recs.len()));
                }
            }
            let kr = &self.recs;
            // [107]
            let spans = &self.spans;
            let slices = |win: usize| -> Vec<Vec<u32>> {
                let parts = held[win].div_ceil(slice_records).max(1);
                let now = (at + win as u64) as u32;
                let mut v: Vec<Vec<u32>> = (0..parts).map(|_| Vec::with_capacity(held[win] / parts + held[win] / parts / 8)).collect();
                let file = |v: &mut Vec<Vec<u32>>, i: usize| {
                    let hash = kr.recs[i].start.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32;
                    v[((hash * parts as u64) >> 32) as usize].push(i as u32);
                };
                if let Some(&(lo, hi)) = loaded.get(win - w) {
                    (lo..hi).for_each(|i| file(&mut v, i));
                }
                // [108]
                let filed: Vec<Vec<Vec<u32>>> = spans
                    .par_chunks(BIN_RUN)
                    .zip(&hits)
                    .enumerate()
                    .filter(|(_, (_, h))| h[win] != 0)
                    .map(|(run, (c, _))| {
                        let mut mine: Vec<Vec<u32>> = vec![Vec::new(); parts];
                        for (i, &[a, b]) in c.iter().enumerate() {
                            if a <= now && now <= b {
                                file(&mut mine, run * BIN_RUN + i);
                            }
                        }
                        mine
                    })
                    .collect();
                for mine in &filed {
                    for (to, from) in v.iter_mut().zip(mine) {
                        to.extend_from_slice(from);
                    }
                }
                v
            };
            let by_window: Vec<Vec<Vec<u32>>> = (w..e).into_par_iter().map(&slices).collect();
            h.bin_secs += tb.elapsed().as_secs_f64();
            h.bin_stages[1] += tb.elapsed().as_secs_f64();
            // One window over the budget alone goes by slices.
            let batches: Vec<Vec<Vec<Vec<u32>>>> = if sum > plan_records {
                let n = plan_records.div_ceil(slice_records).max(1);
                by_window.into_iter().next().expect("one window").chunks(n).map(|c| vec![c.to_vec()]).collect()
            } else {
                vec![by_window]
            };
            let n_b = batches.len();
            for (bi, batch) in batches.into_iter().enumerate() {
                let first = at + w as u64;
                let tp = std::time::Instant::now();
                let plans = km.plan_windows(kr, &batch, first, frames);
                h.plan_secs += tp.elapsed().as_secs_f64();
                drop(batch);
                let parts: Vec<WindowPart> = plans.into_iter().enumerate().map(|(i, plans)| WindowPart { w: first + i as u64, plans, ends: bi + 1 == n_b }).collect();
                let ends_key = last && e == nw && bi + 1 == n_b;
                let warp_end = self.warp.clone().filter(|_| ends_key).map(|c| (c, self.first * fr, upto * fr));
                let job = PlannedKey {
                    key: self.key,
                    parts,
                    start: start.take(),
                    last: ends_key,
                    warp_begin: std::mem::take(&mut warp_begin),
                    warp_end,
                    frames,
                    slot: slot.clone(),
                    pc: pcr.with_frames(frames),
                    began: self.began,
                };
                let tq = std::time::Instant::now();
                let handed = h.ktx.send(job).is_ok();
                h.hand_secs += tq.elapsed().as_secs_f64();
                if !handed {
                    return Ok(false);
                }
            }
            self.recs.recs.truncate(base);
            w = e;
        }
        self.at = Some(upto);
        if !last {
            let t = std::time::Instant::now();
            // In place: a new vector each window was most of the binning.
            let (recs, spans) = (&mut self.recs.recs, &mut self.spans);
            let mut kept = 0;
            for i in 0..spans.len() {
                if spans[i][1] as u64 >= upto {
                    recs[kept] = recs[i];
                    spans[kept] = spans[i];
                    kept += 1;
                }
            }
            recs.truncate(kept);
            spans.truncate(kept);
            if let Some(sp) = self.spill.as_mut() {
                sp.done_below(upto);
            }
            h.bin_secs += t.elapsed().as_secs_f64();
            h.bin_stages[2] += t.elapsed().as_secs_f64();
        }
        Ok(true)
    }
}

/// A record as the parsed render holds a group, 24 bytes where a `RunRec` is \[109\]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KeyRec {
    start: u64,
    duration: u32,
    lid: u32,
    gid: u32,
    slot: u16,
    count: u16,
}

const KEY_REC_CUT: u32 = 1 << 31;

/// A tail no CC120 cuts.
const NO_CUT: u32 = u32::MAX;

/// A record's channel at each end: its gains at the note-on and the note-off, \[110\]
type ChannelEnds = ([f32; 2], [f32; 2], u64);

/// One key's records of a group, and the tables they index: each layer's \[111\]
#[derive(Default)]
struct KeyRecs {
    recs: Vec<KeyRec>,
    layers: Vec<(u32, [f32; 2])>,
    gains: Vec<ChannelEnds>,
}

impl KeyRecs {
    fn voice(&self, r: &KeyRec) -> u32 {
        self.layers[r.lid as usize].0
    }

    /// Frames from a record's note-off to the CC120 that cuts its tail, or \[112\]
    fn cut_after(&self, r: &KeyRec) -> u32 {
        match self.gains[(r.gid & !KEY_REC_CUT) as usize].2 {
            u64::MAX => NO_CUT,
            at => at.saturating_sub(r.start + r.duration as u64).min(NO_CUT as u64 - 1) as u32,
        }
    }

    fn run(&self, r: &KeyRec) -> RunRec {
        let (voice, base) = self.layers[r.lid as usize];
        let (on, off, _) = self.gains[(r.gid & !KEY_REC_CUT) as usize];
        RunRec {
            start: r.start,
            duration: r.duration,
            voice,
            slot: r.slot as u32,
            count: r.count,
            flags: if r.gid & KEY_REC_CUT != 0 { REC_CUT } else { 0 },
            gain: [base[0] * on[0], base[1] * on[1]],
            gain_end: [base[0] * off[0], base[1] * off[1]],
        }
    }
}

/// One key group through merlin's front end: its notes paired, then each \[113\]
fn parse_group(pre: &crate::merlin::keyparse::Prepass, cfg: &Config, bank: &Bank, g: Vec<u8>, warps: Option<&[Arc<crate::merlin::keyparse::Warp>]>, send: Option<&dyn Fn(KeyPart) -> bool>) -> Result<ParsedGroup> {
    use crate::merlin::keyparse::{extract_par, ExtractOpts, Note};
    use crate::merlin::keyrec::Layers;
    let tg = std::time::Instant::now();
    // [114]
    type Sound = (u32, u8, u32, [u8; 5]);
    struct Key {
        key: u8,
        local: FxHashMap<VoiceKey, u32>,
        descs: Vec<VoiceDesc>,
        out: KeyRecs,
        sounds: FxHashMap<Sound, Vec<(u32, u64, u64)>>,
        lid_of: FxHashMap<(u32, [u32; 2]), u32>,
        gid_of: FxHashMap<([u32; 4], u64), u32>,
        warped: std::collections::BTreeMap<u8, Vec<KeyRec>>,
    }
    struct Sub<'a> {
        layers: Layers<'a>,
        tmp: Vec<crate::merlin::keyrec::Layer>,
        keys: Vec<Key>,
    }
    let make = |ks: &[u8]| Sub {
        layers: Layers::new(bank, cfg),
        tmp: Vec::new(),
        keys: ks
            .iter()
            .map(|&key| Key {
                key,
                local: FxHashMap::default(),
                descs: Vec::new(),
                out: KeyRecs::default(),
                sounds: FxHashMap::default(),
                lid_of: FxHashMap::default(),
                gid_of: FxHashMap::default(),
                warped: Default::default(),
            })
            .collect(),
    };
    let note = |s: &mut Sub, n: Note| {
        let Sub { layers, tmp, keys } = s;
        let k = (n.slot & 127) as u8;
        let Key { local, descs, out, sounds, lid_of, gid_of, warped, .. } = keys.iter_mut().find(|e| e.key == k).expect("a key of this pairing");
        let mut build = |n: &Note| -> Vec<(u32, u64, u64)> {
            layers.build(n, tmp);
            crate::merlin::keyrec::merge_stereo(tmp);
            tmp.iter()
                .map(|l| {
                    let desc = VoiceDesc { cmd: crate::merlin::bucket::strip_voice(&l.cmd), params: l.params, menv: l.menv, smp_base_r: l.smp_base_r };
                    let next = descs.len() as u32;
                    let voice = *local.entry(VoiceKey(desc)).or_insert_with(|| {
                        descs.push(desc);
                        next
                    });
                    let lid = *lid_of.entry((voice, l.base.map(f32::to_bits))).or_insert_with(|| {
                        out.layers.push((voice, l.base));
                        out.layers.len() as u32 - 1
                    });
                    (lid, l.start - n.start, crate::merlin::kernel::release_frames(&l.params, cfg) as u64)
                })
                .collect()
        };
        let sound_of = |n: &Note| -> Sound { (n.preset, n.vel, n.bend, n.sound) };
        // [115]
        let ch = (n.slot / 128) as u8;
        let clock = warps.map(|w| &*w[ch as usize]).filter(|w| w.moves_at_all());
        let mut n = n;
        let moved = clock.filter(|w| {
            let plain = Note { bend: crate::fixed::BEND_ONE, ..n };
            let tail = sounds.entry(sound_of(&plain)).or_insert_with(|| build(&plain)).iter().map(|l| l.2).max().unwrap_or(0);
            let moved = w.moves(n.start, n.end + tail + cfg.steal_fade_frames as u64);
            if moved {
                n = plain;
            }
            moved
        });
        let sound = sounds.entry(sound_of(&n)).or_insert_with(|| build(&n));
        let tau = |f: u64| moved.map_or(f, |w| w.tau(f).round() as u64);
        let slot = crate::voice::GateTable::slot((n.slot / 128) as u8, k) as u16;
        let cut_at = if n.cut_at == u64::MAX { u64::MAX } else { tau(n.cut_at) };
        let gk = ([n.gain_on[0].to_bits(), n.gain_on[1].to_bits(), n.gain_off[0].to_bits(), n.gain_off[1].to_bits()], cut_at);
        let gid = *gid_of.entry(gk).or_insert_with(|| {
            out.gains.push((n.gain_on, n.gain_off, cut_at));
            out.gains.len() as u32 - 1
        });
        let recs = if moved.is_some() { warped.entry(ch).or_default() } else { &mut out.recs };
        for &(lid, delay, _) in sound.iter() {
            let at = n.start + delay;
            let (start, end) = (tau(at), tau(n.end.max(at)));
            let mut left = n.count;
            while left > 0 {
                let count = left.min(u16::MAX as u32);
                left -= count;
                recs.push(KeyRec { start, duration: (end - start) as u32, lid, gid: gid | if n.cut { KEY_REC_CUT } else { 0 }, slot, count: count as u16 });
            }
        }
    };
    // [116]
    let mut told = (0usize, 0usize, 0usize);
    let mut records = 0u64;
    let mut part = move |k: &mut Key, frontier: u64| {
        let p = KeyPart {
            descs: k.descs[told.0..].to_vec(),
            layers: k.out.layers[told.1..].to_vec(),
            gains: k.out.gains[told.2..].to_vec(),
            recs: std::mem::take(&mut k.out.recs),
            frontier,
        };
        told = (k.descs.len(), k.out.layers.len(), k.out.gains.len());
        p
    };
    let Some(send) = send else {
        let (st, subs) = extract_par(pre, cfg, bank, &g, ExtractOpts::default(), make, note, None)?;
        let mut by_key: FxHashMap<u8, (Vec<VoiceDesc>, KeyRecs)> = FxHashMap::default();
        let mut warped = Vec::new();
        for (_, sub) in subs {
            for mut k in sub.keys {
                k.out.recs.shrink_to_fit();
                records += k.out.recs.len() as u64;
                warped.extend(std::mem::take(&mut k.warped).into_iter().map(|(ch, r)| (k.key, ch, r)));
                by_key.insert(k.key, (k.descs, k.out));
            }
        }
        let built: Vec<(Vec<VoiceDesc>, KeyRecs)> = g.iter().map(|k| by_key.remove(k).unwrap_or_default()).collect();
        return Ok(ParsedGroup { keys: g, built, st, secs: tg.elapsed().as_secs_f64(), records, warped });
    };
    let part_records = cfg.merlin_km_stream_part.max(1) as usize;
    let opts = ExtractOpts { window_notes: ExtractOpts::default().window_notes.min(part_records as u64) };
    let mut after = |states: &mut [Option<Sub>], frontier: u64| -> bool {
        let k = &mut states[0].as_mut().expect("a sink for the one pairing").keys[0];
        if k.out.recs.len() < part_records {
            return true;
        }
        records += k.out.recs.len() as u64;
        send(part(k, frontier))
    };
    let (st, mut subs) = extract_par(pre, cfg, bank, &g, opts, make, note, Some(&mut after))?;
    let k = &mut subs[0].1.keys[0];
    records += k.out.recs.len() as u64;
    if !send(part(k, u64::MAX)) {
        anyhow::bail!("stopped");
    }
    let key = k.key;
    let warped = std::mem::take(&mut k.warped).into_iter().map(|(ch, r)| (key, ch, r)).collect();
    Ok(ParsedGroup { keys: g, built: Vec::new(), st, secs: tg.elapsed().as_secs_f64(), records, warped })
}

/// The song's master and what the passes counted.
type Rendered = (Vec<f32>, PassEnd);

/// The per-key render, as the backend holds it.
struct Stream {
    rx: Option<std::sync::mpsc::Receiver<Result<Rendered>>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    progress: Arc<MerlinProgress>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Stream {
    /// Stops the passes at their next block and waits for the render, so the \[117\]
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.rx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// One a backend, so the driver counters' few hundred bytes are not worth a box.
#[allow(clippy::large_enum_variant)]
enum Source {
    /// The whole song's master: the CPU reference's when the backend is made, \[118\]
    Whole { master: Vec<f32>, end: PassEnd },
    /// The device still rendering keys.
    Keys(Stream),
}

/// Key-major merlin as a backend, which the driver's output stage and the \[119\]
pub struct KeyMajor {
    src: Source,
    pos: usize,
    block_samples: usize,
}

impl KeyMajor {
    pub fn new(cfg: &Config, bank: Arc<Bank>, midi: &Path, spill_dir: &Path) -> Result<Self> {
        Self::with_progress(cfg, bank, midi, spill_dir, MerlinProgress::new())
    }

    /// `new`, its render writing `progress` as it goes and stopping, with \[120\]
    pub fn with_progress(cfg: &Config, bank: Arc<Bank>, midi: &Path, spill_dir: &Path, progress: Arc<MerlinProgress>) -> Result<Self> {
        let block_samples = cfg.block_samples();
        if !cfg.merlin_key_major_gpu {
            let t = std::time::Instant::now();
            let stop = std::sync::atomic::AtomicBool::new(false);
            let driver = crate::Driver::open(cfg, bank.clone(), midi)?;
            let (mut pass, end) = run_pass_on(cfg, bank.clone(), driver, spill_dir, None, &stop)?;
            let pass_secs = t.elapsed().as_secs_f64();
            let mut km = KeyMajorCpu::new(cfg, bank);
            let master = km.render(&mut pass)?;
            log::info!("merlin key-major (cpu reference): bucket pass {pass_secs:.2}s");
            km.log_summary(cfg, 1, 0, 0);
            return Ok(Self { src: Source::Whole { master, end }, pos: 0, block_samples });
        }
        let (frames, carry) = window_for(cfg, &bank);
        log::info!(
            "merlin key-major: true per-key, windows of {frames} frames ({:.1} s) inside a key, {carry}-frame carry",
            frames as f64 / cfg.sample_rate as f64
        );
        let stop = progress.cancel_flag();
        let (otx, orx) = std::sync::mpsc::sync_channel(1);
        let thread = {
            let (cfg, bank, midi, dir, stop, progress) = (cfg.clone(), bank.clone(), midi.to_path_buf(), spill_dir.to_path_buf(), stop.clone(), progress.clone());
            std::thread::Builder::new().name("km-render".into()).spawn(move || {
                let r = if cfg.merlin_key_parse {
                    run_keys_parsed(&cfg, bank, &midi, &dir, frames, carry, &progress)
                } else {
                    run_keys(&cfg, bank, &midi, &dir, frames, carry, &stop)
                };
                let _ = otx.send(r);
            })?
        };
        let stream = Stream { rx: Some(orx), stop, progress, thread: Some(thread) };
        Ok(Self { src: Source::Keys(stream), pos: 0, block_samples })
    }

    /// The whole render, interleaved stereo, once it is done; empty while the \[121\]
    pub fn master(&self) -> &[f32] {
        match &self.src {
            Source::Whole { master, .. } => master,
            Source::Keys(_) => &[],
        }
    }

    /// Wait for the last key, and take the master, cut to the render's end as \[122\]
    fn wait_master(&mut self) -> Result<()> {
        let Source::Keys(s) = &mut self.src else { return Ok(()) };
        let rx = s.rx.as_ref().context("the key-major render was stopped")?;
        let got = rx.recv();
        // Stopped is stopped, whatever the render said on its way out.
        s.progress.check()?;
        let (mut master, end) = match got {
            Ok(r) => r?,
            Err(_) => anyhow::bail!("the key-major render stopped without saying why"),
        };
        master.resize(2 * end.end as usize, 0.0);
        s.progress.begin_mastering((master.len() as u64).div_ceil(self.block_samples as u64));
        self.src = Source::Whole { master, end };
        Ok(())
    }

    /// The most of a group's records the render held in memory at once, for \[123\]
    pub fn peak_records(&self) -> u64 {
        self.pass_end().map_or(0, |e| e.peak_records)
    }

    fn pass_end(&self) -> Option<PassEnd> {
        match &self.src {
            Source::Whole { end, .. } => Some(*end),
            Source::Keys(_) => None,
        }
    }
}

impl Backend for KeyMajor {
    fn set_gates(&mut self, _meta: &[u32], _runs: &[u32]) -> Result<()> {
        Ok(())
    }

    fn set_channels(&mut self, _rows: &[u32], _b: bool, _g: bool, _v: bool, _c: bool) -> Result<()> {
        Ok(())
    }

    fn set_params_variant(&mut self, _i: u32, _d: Arc<Vec<RegionParams>>, _m: Arc<Vec<ModEnvParams>>) -> Result<()> {
        Ok(())
    }

    fn spawn(&mut self, _cmds: &[SpawnCmd]) -> Result<()> {
        Ok(())
    }

    fn spawn_parts(&self) -> bool {
        true
    }

    fn submit(&mut self) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        out.fill(0.0);
        self.wait_master()?;
        if let Source::Whole { master, .. } = &self.src {
            let from = self.pos.min(master.len());
            let to = (self.pos + out.len()).min(master.len());
            out[..to - from].copy_from_slice(&master[from..to]);
        }
        self.pos += self.block_samples;
        Ok(())
    }

    fn skip_block(&mut self) -> Result<()> {
        anyhow::bail!("key-major merlin has no pool, so the driver never steps over its blocks")
    }

    /// Outstanding until the render has been handed out.
    fn stats(&self) -> BlockStats {
        let left = match &self.src {
            Source::Whole { master, .. } => self.pos < master.len(),
            Source::Keys(_) => true,
        };
        BlockStats { active_voices: u64::from(left), ..BlockStats::default() }
    }

    fn name(&self) -> &'static str {
        "merlin-key-major"
    }

    fn has_pool(&self) -> bool {
        false
    }

    /// The driver's old end, without its reading the file again: on while \[124\]
    fn prerendered_blocks(&self) -> Option<(u64, u64)> {
        let bs = self.block_samples as u64;
        Some(match &self.src {
            Source::Whole { master, end } => (end.stream_blocks, (master.len() as u64).div_ceil(bs)),
            Source::Keys(_) => (u64::MAX, u64::MAX),
        })
    }

    fn event_counts(&self) -> Option<crate::driver::DriverStats> {
        self.pass_end().map(|e| e.driver_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A thread that asks for `bytes`, and a channel that says what it was told.
    fn asking(ahead: &Arc<Ahead>, bytes: u64) -> std::sync::mpsc::Receiver<bool> {
        let (tx, rx) = std::sync::mpsc::channel();
        let ahead = ahead.clone();
        std::thread::spawn(move || {
            let _ = tx.send(ahead.reserve(bytes));
        });
        rx
    }

    const SHORT: std::time::Duration = std::time::Duration::from_millis(150);
    const LONG: std::time::Duration = std::time::Duration::from_secs(5);

    /// Room is taken up to the limit, and one more waits until the device has \[125\]
    #[test]
    fn the_device_queue_holds_back_past_its_limit_and_lets_go_when_it_has_put() {
        let a = Arc::new(Ahead::new(100));
        // On a thread, so that a queue that turns it away fails this and does not hang it.
        assert_eq!(asking(&a, 1_000).recv_timeout(LONG), Ok(true), "a window larger than the limit goes when nothing is held");
        a.release(1_000);
        assert!(a.reserve(60));
        let waiting = asking(&a, 60);
        assert!(waiting.recv_timeout(SHORT).is_err(), "120 bytes went into a queue of 100");
        a.release(60);
        assert_eq!(waiting.recv_timeout(LONG), Ok(true), "the put did not make room");
        // More released than held is not a debt.
        a.release(10_000);
        assert!(a.reserve(100));
    }

    /// The device's thread going, whatever it was doing, lets the packing \[126\]
    #[test]
    fn a_device_thread_that_is_gone_lets_the_waiting_thread_go() {
        let a = Arc::new(Ahead::new(100));
        assert!(a.reserve(80));
        let waiting = asking(&a, 80);
        assert!(waiting.recv_timeout(SHORT).is_err(), "it did not wait");
        a.close();
        assert_eq!(waiting.recv_timeout(LONG), Ok(false), "a waiter was not told the device is gone");
        // And no one waits afterwards, whatever room there is.
        a.release(80);
        assert!(!a.reserve(1), "a closed queue took a window");
    }

    /// The device thread's guard closes the queue when the thread ends in any \[127\]
    #[test]
    fn a_device_thread_that_fails_closes_the_queue_as_it_goes() {
        let a = Arc::new(Ahead::new(100));
        assert!(a.reserve(100));
        let waiting = asking(&a, 100);
        assert!(waiting.recv_timeout(SHORT).is_err());
        let device = {
            let a = a.clone();
            std::thread::spawn(move || -> Result<()> {
                let _closes = CloseOnDrop(a);
                anyhow::bail!("the device did not begin")
            })
        };
        assert!(device.join().unwrap().is_err());
        assert_eq!(waiting.recv_timeout(LONG), Ok(false), "a failed device thread left the packing thread waiting");
    }

    fn wave(id: u32, need: usize) -> Wave {
        Wave { cmd: SpawnCmd::default(), params: bytemuck::Zeroable::zeroed(), need, key: WaveKey { smp_base: id, smp_base_r: id, ..WaveKey::default() } }
    }

    fn head(wave: u32) -> HeadNote {
        HeadNote { wave, start: 0, from: 0, d: 1, dj: 0, r: 0.5, gain: [1.0; 2], gain_end: [1.0; 2], fade: 0, release: 0, bus: 0, weight: 1, env_ratio: 1.0 }
    }

    /// A slice reading `waves`, each `(id, need)`, with an entry, an impulse \[128\]
    fn slice(waves: &[(u32, usize)], notes: u64) -> KeyPlan {
        let wv: Vec<Wave> = waves.iter().map(|&(id, need)| wave(id, need)).collect();
        let entries: Vec<Entry> = (0..waves.len() as u32).map(|i| Entry { wave: i, dj: 0, skip: 0, depth: waves[i as usize].0 }).collect();
        let g = Group {
            impulses: (0..waves.len() as u32).map(|i| Impulse { entry: i, frame: waves[i as usize].0 as u64, gain: [1.0; 2], weight: 1 }).collect(),
            stopped: Vec::new(),
            heads: (0..waves.len() as u32).map(head).collect(),
            lo: 0,
            hi: 1,
        };
        KeyPlan { waves: wv, entries, groups: vec![(0, g)], counts: PlanCounts { notes, cut: 0, released_on_head: 0, tails_cut: 0 } }
    }

    fn part(w: u64, plans: Vec<KeyPlan>, ends: bool) -> WindowPart {
        WindowPart { w, plans, ends }
    }

    /// The pool's budget cuts a key's windows into runs, and cuts a window that \[129\]
    #[test]
    fn a_window_over_a_pool_is_cut_by_waveform() {
        // [130]
        let big = vec![
            slice(&[(1, 100), (2, 100), (3, 50)], 7),
            slice(&[(2, 120), (4, 400)], 5),
            slice(&[(5, 60)], 2),
        ];
        let parts = vec![
            part(0, vec![slice(&[(1, 200)], 1)], true),
            part(1, big, true),
            part(2, vec![slice(&[(1, 150), (6, 100)], 1)], true),
            part(3, vec![slice(&[(7, 100)], 1)], true),
            part(4, vec![slice(&[(7, 100)], 1)], true),
        ];
        let runs = pool_runs(parts, 300);
        let flat: Vec<&WindowPart> = runs.iter().flatten().collect();
        assert_eq!(flat.iter().map(|p| p.w).collect::<Vec<_>>(), vec![0, 1, 1, 1, 2, 3, 4], "parts in window order");
        assert_eq!(flat.iter().map(|p| p.ends).collect::<Vec<_>>(), vec![true, false, false, true, true, true, true], "a window ends on its last part");
        assert_eq!(runs.iter().map(|r| r.len()).collect::<Vec<_>>(), vec![1, 1, 1, 1, 1, 2], "runs");
        let mut seen: FxHashMap<u32, usize> = FxHashMap::default();
        let mut notes = 0;
        for p in flat.iter().filter(|p| p.w == 1) {
            let (order, need) = part_needs(p);
            let total: usize = need.values().sum();
            assert!(total <= 300 || order.len() == 1, "a part of {total} frames with {} waves", order.len());
            for plan in &p.plans {
                notes += plan.counts.notes;
                let ids: Vec<u32> = plan.waves.iter().map(|w| w.key.smp_base).collect();
                for (i, e) in plan.entries.iter().enumerate() {
                    assert_eq!(e.depth, ids[e.wave as usize], "entry {i} left its wave");
                }
                for (_, g) in &plan.groups {
                    assert_eq!(g.impulses.len(), g.heads.len());
                    for (imp, h) in g.impulses.iter().zip(&g.heads) {
                        assert_eq!(imp.frame as u32, ids[plan.entries[imp.entry as usize].wave as usize], "an impulse left its entry");
                        assert_eq!(imp.frame as u32, ids[h.wave as usize], "a head left its wave");
                        *seen.entry(ids[h.wave as usize]).or_insert(0) += 1;
                    }
                }
            }
        }
        assert_eq!(notes, 14, "the notes counted once");
        let mut got: Vec<(u32, usize)> = seen.into_iter().collect();
        got.sort_unstable();
        assert_eq!(got, vec![(1, 1), (2, 2), (3, 1), (4, 1), (5, 1)], "every note read once");
    }

    /// A wave's slices never part: a slice's waves go where their class goes, \[131\]
    #[test]
    fn a_wave_read_by_two_slices_is_one_class() {
        let big = vec![slice(&[(1, 10), (2, 200)], 1), slice(&[(2, 250), (3, 200)], 1)];
        let runs = pool_runs(vec![part(0, big, true)], 260);
        let classes: Vec<Vec<u32>> = runs.iter().flatten().map(|p| {
            let (order, _) = part_needs(p);
            order.iter().map(|k| k.smp_base).collect()
        }).collect();
        assert_eq!(classes, vec![vec![1, 2], vec![3]]);
    }
}
