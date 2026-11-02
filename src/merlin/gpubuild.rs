// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sustain renders made on the device, many at once: the kernel builder. \[1\]

use crate::bank::{Bank, ModEnvParams, RegionParams};
use crate::config::Config;
use crate::gpu::{GpuShared, GpuSynth};
use crate::voice::SpawnCmd;
use anyhow::{bail, Result};
use std::sync::Arc;

use crate::backend::Backend;
use wgpu::util::DeviceExt;

/// The most bytes of rows one chunk of a batch lands in: its readback holds \[2\]
const CHUNK_BYTES: u64 = 64 << 20;

/// Frames in a page of `sustain.wgsl`: one workgroup's.
const PAGE: usize = 1024;

/// `sustain.wgsl`'s `SJob`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SJob {
    base: u32,
    len: u32,
    ls: u32,
    le: u32,
    looping: u32,
    step_hi: u32,
    step_lo: u32,
    n_ah: u32,
    attack_rate: f32,
    decay: f32,
    n_sus: u32,
    sustain: f32,
    mod_delay: u32,
    mod_inc: u32,
    volume: i32,
    _p: u32,
}

/// `sustain.wgsl`'s `SPage`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SPage {
    job: u32,
    out: u32,
    n0: u32,
    frames: u32,
    phase_hi: u32,
    phase_lo: u32,
}

/// A sustain for `sustain.wgsl`: `frames` frames of `cmd` through its own \[3\]
#[derive(Clone, Copy)]
pub struct DirectJob {
    pub cmd: SpawnCmd,
    pub params: RegionParams,
    pub frames: usize,
    pub at: u64,
}

/// Whether a sustain is a pure function of its frame, which is what \[4\]
pub fn direct_ok(cmd: &SpawnCmd, p: &RegionParams) -> bool {
    cmd.flags <= 7 && p.lfo_pitch == 0 && p.flags & crate::bank::RP_MOD_ENV == 0
}

/// `kernel::env_at`'s shape for `p`: frames of attack and hold, the first frame \[5\]
fn settle(p: &RegionParams, cfg: &Config) -> (u32, u32, f32, f32) {
    if p.attack_rate <= 0.0 {
        return (u32::MAX, 0, 0.0, 0.0);
    }
    let n_ah = (p.attack_end as f64 / p.attack_rate as f64).ceil().max(0.0) as u64;
    let exp = cfg.decay_curve == crate::config::EnvelopeCurve::Exponential;
    let (coef, target) = (p.decay_coef as f64, p.decay_target);
    let level = |k: u64| -> f32 {
        if !exp {
            (1.0 - coef * k as f64) as f32
        } else if coef <= 0.0 {
            0.0
        } else {
            coef.powf(k as f64) as f32
        }
    };
    // The first decay step at or below the target, if any is.
    let guess = if exp {
        (coef > 0.0 && coef < 1.0 && target > 0.0).then(|| (target as f64).ln() / coef.ln())
    } else {
        (coef > 0.0).then(|| (1.0 - target as f64) / coef)
    };
    let first = match guess {
        Some(g) => {
            let mut k = g.ceil().max(1.0) as u64;
            while k > 1 && level(k - 1) <= target {
                k -= 1;
            }
            while level(k) > target {
                k += 1;
            }
            Some(k)
        }
        None => (level(1) <= target).then_some(1),
    };
    let n_sus = first.map_or(u32::MAX, |k| (n_ah + k - 1).min(u32::MAX as u64) as u32);
    let held = if p.sustain <= cfg.env_floor { 0.0 } else { p.sustain };
    let decay = if exp { if coef > 0.0 { coef.log2() as f32 } else { 0.0 } } else { p.decay_coef };
    (n_ah.min(u32::MAX as u64) as u32, n_sus, held, decay)
}

/// A job's row, and its pages onto `pages`.
fn job_of(cfg: &Config, j: &DirectJob, index: u32, pages: &mut Vec<SPage>) -> SJob {
    let c = &j.cmd;
    let p = &j.params;
    let looping = c.flags & 1 != 0;
    let (n_ah, n_sus, sustain, decay) = settle(p, cfg);
    let phase = (c.phase_hi as u128) << 32 | c.phase_lo as u128;
    let step = (c.step_hi as u128) << 32 | c.step_lo as u128;
    let frames = j.frames.max(1);
    for n0 in (0..frames).step_by(PAGE) {
        let at = phase + n0 as u128 * step;
        let hi = (at >> 32) as u64;
        // [6]
        let hi = if looping && hi >= c.loop_end as u64 {
            c.loop_start as u64 + (hi - c.loop_start as u64) % (c.loop_end.saturating_sub(c.loop_start).max(1) as u64)
        } else {
            hi.min(0x7FFF_FFFF)
        };
        pages.push(SPage { job: index, out: (j.at + n0 as u64) as u32, n0: n0 as u32, frames: (frames - n0).min(PAGE) as u32, phase_hi: hi as u32, phase_lo: at as u32 });
    }
    SJob {
        base: c.smp_base,
        len: c.smp_len,
        ls: c.loop_start,
        le: c.loop_end,
        looping: looping as u32,
        step_hi: c.step_hi,
        step_lo: c.step_lo,
        n_ah,
        attack_rate: p.attack_rate,
        decay,
        n_sus,
        sustain,
        mod_delay: p.mod_lfo_delay(),
        mod_inc: p.mod_lfo_inc,
        volume: (p.flags as i32) >> 16,
        _p: 0,
    }
}

/// A pool that renders up to `rows` sustains a chunk.
pub struct GpuBuilder {
    synth: GpuSynth,
    cfg: Config,
    bank: Arc<Bank>,
    /// The sustain pass, once a job has asked for it.
    direct: Option<crate::gpu::SustainPass>,
    rows: u32,
    block: usize,
    /// Where `render_into` lands a chunk's rows before copying them out, and \[7\]
    scratch: Option<wgpu::Buffer>,
    scratch_bytes: u64,
}

/// A batch queued on the device and not yet read: one or more chunks, each its \[8\]
pub struct Batch {
    chunks: Vec<Chunk>,
    jobs: usize,
    at: Option<wgpu::SubmissionIndex>,
}

struct Chunk {
    readback: wgpu::Buffer,
    bytes: u64,
    /// The batch's job indices in this chunk, in row order, with their frames.
    rows: Vec<(usize, usize)>,
}

impl Batch {
    /// Host memory the batch's readbacks hold until it is read.
    pub fn bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.bytes).sum()
    }
}

impl GpuBuilder {
    /// A builder on `shared`'s device and sample pool, `rows` voices a chunk. \[9\]
    pub fn new(cfg: &Config, bank: Arc<Bank>, shared: &GpuShared, rows: u32) -> Result<Self> {
        let mut c = cfg.clone();
        c.max_voices = rows.max(1);
        c.profile = false;
        let synth = GpuSynth::solo_on_shared(&c, bank.clone(), shared, rows.max(1))?;
        Ok(Self { synth, cfg: c.clone(), bank, direct: None, rows: rows.max(1), block: c.block_frames as usize, scratch: None, scratch_bytes: CHUNK_BYTES })
    }

    /// Queue each `(command, frames)` to be rendered with no note-off for \[10\]
    pub fn begin(
        &mut self,
        params: &Arc<Vec<RegionParams>>,
        menv: &Arc<Vec<ModEnvParams>>,
        jobs: &[(SpawnCmd, usize)],
    ) -> Result<Batch> {
        let mut order: Vec<usize> = (0..jobs.len()).collect();
        order.sort_by(|&a, &b| jobs[b].1.cmp(&jobs[a].1).then(a.cmp(&b)));
        let block_bytes = (self.block * 4) as u64;
        let mut chunks = Vec::new();
        let mut at = 0usize;
        while at < order.len() {
            // The first job of a chunk is its longest.
            let blocks = jobs[order[at]].1.div_ceil(self.block).max(1);
            let fit = (CHUNK_BYTES / (blocks as u64 * block_bytes)).max(1) as usize;
            let n = (order.len() - at).min(self.rows as usize).min(fit);
            let rows: Vec<(usize, usize)> = order[at..at + n].iter().map(|&j| (j, jobs[j].1)).collect();
            at += n;

            self.synth.reset();
            self.synth.set_params_variant(0, params.clone(), menv.clone())?;
            let cmds: Vec<SpawnCmd> = rows
                .iter()
                .enumerate()
                .map(|(i, &(j, _))| SpawnCmd {
                    gate_slot: 0,
                    ordinal: 1,
                    start_rel: 0,
                    note_id_lo: i as u32,
                    note_id_hi: 0,
                    gain_l: 1.0,
                    gain_r: 0.0,
                    row_bias: 0,
                    variant: 0,
                    ..jobs[j].0
                })
                .collect();
            self.synth.spawn(&cmds)?;
            let row_bytes = n as u64 * block_bytes;
            let bytes = row_bytes * blocks as u64;
            let readback = self.synth.solo_readback(bytes);
            for b in 0..blocks {
                self.synth.solo_submit_to(&readback, b as u64 * row_bytes, row_bytes)?;
            }
            chunks.push(Chunk { readback, bytes, rows });
        }
        Ok(Batch { chunks, jobs: jobs.len(), at: self.synth.solo_submission() })
    }

    /// Wait for a batch and take its sustains, in the order `begin` was given \[11\]
    pub fn wait(&mut self, batch: Batch) -> Result<Vec<Vec<f32>>> {
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); batch.jobs];
        if batch.chunks.is_empty() {
            return Ok(out);
        }
        let bufs: Vec<(&wgpu::Buffer, u64)> = batch.chunks.iter().map(|c| (&c.readback, c.bytes)).collect();
        let reads = self.synth.solo_map(&bufs, batch.at.clone())?;
        for (chunk, raw) in batch.chunks.iter().zip(&reads) {
            let data: &[f32] = bytemuck::cast_slice(raw);
            let n = chunk.rows.len();
            let blocks = data.len() / (n * self.block).max(1);
            for (i, &(j, frames)) in chunk.rows.iter().enumerate() {
                let dst = &mut out[j];
                dst.reserve_exact(frames);
                for b in 0..blocks {
                    let take = frames.saturating_sub(dst.len()).min(self.block);
                    if take == 0 {
                        break;
                    }
                    let at = (b * n + i) * self.block;
                    dst.extend_from_slice(&data[at..at + take]);
                }
                if dst.len() != frames {
                    bail!("the kernel builder returned {} of {frames} frames for one row", dst.len());
                }
            }
        }
        Ok(out)
    }

    /// `jobs` into `dst` by `sustain.wgsl`, every frame of every job side by \[12\]
    pub fn direct_into(&mut self, jobs: &[DirectJob], dst: &wgpu::Buffer) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let (device, queue) = self.synth.solo_device();
        if self.direct.is_none() {
            self.direct = Some(self.synth.solo_sustain_pass(&self.cfg, &self.bank));
        }
        let pass = self.direct.as_ref().expect("made above");
        let mut pages: Vec<SPage> = Vec::with_capacity(jobs.iter().map(|j| j.frames.div_ceil(PAGE).max(1)).sum());
        let rows: Vec<SJob> = jobs.iter().enumerate().map(|(i, j)| job_of(&self.cfg, j, i as u32, &mut pages)).collect();
        let buffer = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::STORAGE })
        };
        let (job_buf, page_buf) = (buffer("sustain jobs", bytemuck::cast_slice(&rows)), buffer("sustain pages", bytemuck::cast_slice(&pages)));
        let mut entries: Vec<wgpu::BindGroupEntry> = [&pass.uniform, &pass.pool, &job_buf, &page_buf, dst]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        entries.extend(pass.pool_rest.iter().enumerate().map(|(k, b)| wgpu::BindGroupEntry { binding: 13 + k as u32, resource: b.as_entire_binding() }));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pass.layout, entries: &entries });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pass.pipeline);
            p.set_bind_group(0, &bg, &[]);
            let x = (pages.len() as u32).clamp(1, 32768);
            p.dispatch_workgroups(x, (pages.len() as u32).div_ceil(x), 1);
        }
        queue.submit([enc.finish()]);
        Ok(())
    }

    /// `jobs` by `direct_into`, each read back: what the tests hold against \[13\]
    pub fn direct(&mut self, jobs: &[(SpawnCmd, RegionParams, usize)]) -> Result<Vec<Vec<f32>>> {
        let (device, queue) = self.synth.solo_device();
        let total: usize = jobs.iter().map(|j| j.2).sum();
        let mut at = 0u64;
        let listed: Vec<DirectJob> = jobs
            .iter()
            .map(|&(cmd, params, frames)| {
                let j = DirectJob { cmd, params, frames, at };
                at += frames as u64;
                j
            })
            .collect();
        let size = total as u64 * 4;
        let dst = device.create_buffer(&wgpu::BufferDescriptor { label: Some("sustains"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        self.direct_into(&listed, &dst)?;
        let read = self.synth.solo_readback(size);
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&dst, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let data = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&data);
        Ok(listed.iter().map(|j| words[j.at as usize..j.at as usize + j.frames].to_vec()).collect())
    }

    /// Land up to `bytes` of rows at once in `render_into`: nothing is read \[14\]
    pub fn with_scratch(mut self, bytes: u64) -> Self {
        self.scratch_bytes = bytes.max(CHUNK_BYTES);
        self
    }

    /// Each `(command, frames)` rendered as [`GpuBuilder::begin`] renders it, \[15\]
    pub fn render_into(
        &mut self,
        params: &Arc<Vec<RegionParams>>,
        menv: &Arc<Vec<ModEnvParams>>,
        jobs: &[(SpawnCmd, usize)],
        dst: &wgpu::Buffer,
        at: &[u64],
    ) -> Result<()> {
        let mut order: Vec<usize> = (0..jobs.len()).collect();
        order.sort_by(|&a, &b| jobs[b].1.cmp(&jobs[a].1).then(a.cmp(&b)));
        let block_bytes = (self.block * 4) as u64;
        let (device, queue) = self.synth.solo_device();
        // [16]
        for chunk in order.chunks(self.rows as usize) {
            let n = chunk.len();
            let rows: Vec<(usize, usize)> = chunk.iter().map(|&j| (j, jobs[j].1)).collect();
            let blocks = rows[0].1.div_ceil(self.block).max(1);
            self.synth.reset();
            self.synth.set_params_variant(0, params.clone(), menv.clone())?;
            let cmds: Vec<SpawnCmd> = rows
                .iter()
                .enumerate()
                .map(|(i, &(j, _))| SpawnCmd {
                    gate_slot: 0,
                    ordinal: 1,
                    start_rel: 0,
                    note_id_lo: i as u32,
                    note_id_hi: 0,
                    gain_l: 1.0,
                    gain_r: 0.0,
                    row_bias: 0,
                    variant: 0,
                    ..jobs[j].0
                })
                .collect();
            self.synth.spawn(&cmds)?;
            let row_bytes = n as u64 * block_bytes;
            let per = (self.scratch_bytes / row_bytes).clamp(1, blocks as u64) as usize;
            let bytes = row_bytes * per as u64;
            if self.scratch.as_ref().is_none_or(|s| s.size() < bytes) {
                self.scratch = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("kernel builder rows"),
                    size: bytes.max(4),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let scratch = self.scratch.clone().expect("made above");
            for b0 in (0..blocks).step_by(per) {
                let b1 = (b0 + per).min(blocks);
                for b in b0..b1 {
                    self.synth.solo_submit_to(&scratch, (b - b0) as u64 * row_bytes, row_bytes)?;
                }
                // Block `b` of row `i` is at `((b - b0) * n + i) * block` frames.
                let mut enc = device.create_command_encoder(&Default::default());
                for (i, &(j, frames)) in rows.iter().enumerate() {
                    for b in b0..b1.min(frames.div_ceil(self.block)) {
                        let take = (frames - b * self.block).min(self.block);
                        enc.copy_buffer_to_buffer(
                            &scratch,
                            (((b - b0) * n + i) * self.block * 4) as u64,
                            dst,
                            (at[j] + (b * self.block) as u64) * 4,
                            (take * 4) as u64,
                        );
                    }
                }
                queue.submit([enc.finish()]);
            }
        }
        Ok(())
    }

    /// [`GpuBuilder::begin`] and [`GpuBuilder::wait`] together.
    pub fn render(
        &mut self,
        params: &Arc<Vec<RegionParams>>,
        menv: &Arc<Vec<ModEnvParams>>,
        jobs: &[(SpawnCmd, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        let batch = self.begin(params, menv, jobs)?;
        self.wait(batch)
    }
}
