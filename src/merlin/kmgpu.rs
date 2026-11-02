// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Key-major merlin on the device: a window of the song at a time, every key's \[1\]

use crate::merlin::fft::Fft;
use crate::merlin::head::Head;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// The four-step's row transform: merlin's largest single-workgroup FFT.
const N1: usize = 2048;
/// Signals one column-pass dispatch holds: it runs `signals * N1` transforms \[2\]
const SLICE: usize = 65535 / N1;
/// The most any one buffer of signals may take.
pub const BUFFER_BYTES: u64 = 64 << 20;
/// Frames a heads workgroup owns. `kmconv.wgsl`'s `PAGE`.
pub const PAGE: usize = 1024;
/// A (block, entry) pair with fewer impulses than this is added in the time \[3\]
pub const DIRECT_BELOW: usize = 256;
/// `Config::merlin_km_segment`'s default: half of the smallest transform.
pub const SEGMENT_FRAMES: usize = 32768;
/// Uploads waiting on the device before the host waits for it.
const STAGING_BYTES: u64 = 128 << 20;
/// Windows a bus goes on being filtered and mixed after the last one that \[4\]
const QUIET_WINDOWS: u8 = 2;

/// Timestamps read back at once.
const TIME_MARKS: u32 = 4096;

/// The device's own time per kind of work, with `--profile`, from timestamps \[5\]
pub struct DeviceTimes {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    next: std::sync::atomic::AtomicU32,
    pend: std::sync::Mutex<Vec<(usize, u32, u32)>>,
    pub secs: [f64; 8],
    period: f64,
}

/// Sustains the kernel builder renders a chunk, and the bytes of rows a \[6\]
const BUILD_ROWS: u32 = 4096;
const BUILD_BYTES: u64 = 128 << 20;

/// A params table for the kernel builder, its jobs' commands and frames, and \[7\]
pub type BuildTable = (std::sync::Arc<Vec<crate::bank::RegionParams>>, Vec<(crate::voice::SpawnCmd, usize)>, Vec<u64>);

/// The kernel builder, and what it renders with.
struct Builder {
    b: crate::merlin::gpubuild::GpuBuilder,
    menv: std::sync::Arc<Vec<crate::bank::ModEnvParams>>,
    _shared: crate::gpu::GpuShared,
}

/// One group's impulses, `(entry, frame from the window's start, [left, \[8\]
pub struct GroupIn {
    pub bus: u32,
    pub impulses: Vec<(u32, u64, [f32; 2])>,
    /// Impulses a CC120 cuts in their tails, always added in the time domain, \[9\]
    pub stopped: Vec<(u32, u64, [f32; 2], u32)>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Imp {
    at: u32,
    l: f32,
    r: f32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dimp {
    at: u32,
    entry: u32,
    l: f32,
    r: f32,
}

/// A `Dimp` a CC120 cuts, for `direct_stopped`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dstop {
    at: u32,
    entry: u32,
    l: f32,
    r: f32,
    /// Frames from `at` to where the CC120 starts it fading.
    stop: u32,
}

/// An impulse no CC120 stops.
const NO_STOP: u32 = u32::MAX;

/// What `KmGpu::pack` reads of the window.
#[derive(Clone, Copy)]
pub struct PackCfg {
    stride: usize,
    carry: usize,
    frames: usize,
    direct_below: usize,
    /// Kernels this long or shorter go through `direct` apart from the rest.
    segment: usize,
    /// Frames a stopped impulse takes to fade.
    fade: usize,
}

impl PackCfg {
    /// The same at windows of `frames`.
    pub fn with_frames(&self, frames: usize) -> Self {
        Self { stride: frames + self.carry, frames, ..*self }
    }
}

/// One key's part of a window as `KmGpu::pack` lays it out, for `put`.
pub struct Packed {
    /// The buses it writes.
    active: Vec<u32>,
    ops: Vec<Op>,
    /// What goes up, for `throttle`.
    bytes: u64,
}

/// One hand-over to the device, in `put`'s order.
enum Op {
    /// A subset of the entries' kernels, `(offset, length)` in the kernel \[10\]
    Kernels { n: usize, e: usize, desc: Vec<u32> },
    /// Dense pairs on the last kernels loaded.
    Chunk { n: usize, imps: Vec<Imp>, entries: Vec<u32>, mdesc: Vec<u32>, parity: [Vec<u32>; 2], blocks: usize, npairs: usize },
    /// The sparse impulses, added in the time domain.
    Direct { dimps: Vec<Dimp>, items: Vec<u32>, desc: Vec<u32>, stride: usize },
    /// The impulses a CC120 cuts, added the same way.
    Stopped { dimps: Vec<Dstop>, items: Vec<u32>, desc: Vec<u32>, stride: usize },
    Heads { heads: Vec<Head>, items: Vec<u32>, list: Vec<u32>, stride: usize },
}

/// A kernel as the device cuts it from the pool (`kmconv.wgsl`'s \[11\]
#[derive(Clone, Copy, Debug)]
pub struct KDesc {
    pub at: u32,
    pub len: u32,
    pub valid: u32,
    pub plain: u32,
    pub rel0: u32,
    pub log2r: f32,
    /// Frames from a stereo kernel's left to its right, zero for a mono one.
    pub right: u32,
}

impl KDesc {
    /// A kernel laid in the pool whole, read as it is.
    pub fn whole(at: u32, len: u32) -> Self {
        Self { at, len, valid: len, plain: len, rel0: 0, log2r: 0.0, right: 0 }
    }

    fn words(&self) -> [u32; KDESC_WORDS] {
        [self.at, self.len, self.valid, self.plain, self.rel0, self.log2r.to_bits(), self.right]
    }
}

/// Words of a `KDesc` as the device reads it.
const KDESC_WORDS: usize = 7;
/// A pair's entry in `mac` with this bit set has panned impulses on a stereo \[12\]
const SIDE: u32 = 1 << 31;

/// Kernels laid end to end onto the end of `pool`, and each one's descriptor, \[13\]
pub fn append_whole(pool: &mut Vec<f32>, kernels: &[&[f32]]) -> Vec<KDesc> {
    pool.reserve(kernels.iter().map(|k| k.len()).sum());
    let mut table = Vec::with_capacity(kernels.len());
    for k in kernels {
        table.push(KDesc::whole(pool.len() as u32, k.len() as u32));
        pool.extend_from_slice(k);
    }
    table
}

/// A sparse impulse: its frame from the window's start, its entry, its gain, \[14\]
type Sparse = (u64, u32, [f64; 2], u32);

/// One group's impulses sorted once by `(block, entry, offset, planned \[15\]
struct Prepped {
    /// The dense pairs, `(block, entry)`, in that order.
    pairs: Vec<(u64, u32)>,
    /// Their impulses, `(pair, offset in its block, gain)`, in pair order.
    dense: Vec<(u32, u32, [f64; 2])>,
    /// The sparse impulses, `(frame, entry, gain)`, in (frame, entry) order.
    sparse: Vec<Sparse>,
}

impl Prepped {
    fn new(g: &GroupIn, hop: usize, below: usize) -> Self {
        let hop = hop as u64;
        let mut keyed: Vec<(u64, u32, u32, u32)> =
            g.impulses.par_iter().enumerate().map(|(i, &(e, f, _))| (f / hop, e, (f % hop) as u32, i as u32)).collect();
        // [16]
        if keyed.len() > 1 << 16 {
            keyed.par_sort_unstable();
        } else {
            keyed.sort_unstable();
        }
        let mut p = Prepped { pairs: Vec::new(), dense: Vec::with_capacity(keyed.len()), sparse: Vec::new() };
        let mut few: Vec<(u64, u32, u32)> = Vec::new();
        let mut i = 0;
        while i < keyed.len() {
            let (b, e) = (keyed[i].0, keyed[i].1);
            let mut j = i;
            while j < keyed.len() && keyed[j].0 == b && keyed[j].1 == e {
                j += 1;
            }
            if j - i < below {
                few.extend(keyed[i..j].iter().map(|&(_, _, off, k)| (b * hop + off as u64, e, k)));
            } else {
                let pi = p.pairs.len() as u32;
                p.pairs.push((b, e));
                for &(_, _, off, k) in &keyed[i..j] {
                    let gn = g.impulses[k as usize].2;
                    match p.dense.last_mut() {
                        Some(m) if m.0 == pi && m.1 == off => {
                            m.2[0] += gn[0] as f64;
                            m.2[1] += gn[1] as f64;
                        }
                        _ => p.dense.push((pi, off, [gn[0] as f64, gn[1] as f64])),
                    }
                }
            }
            i = j;
        }
        few.sort_unstable();
        for (f, e, k) in few {
            let gn = g.impulses[k as usize].2;
            match p.sparse.last_mut() {
                Some(m) if m.0 == f && m.1 == e => {
                    m.2[0] += gn[0] as f64;
                    m.2[1] += gn[1] as f64;
                }
                _ => p.sparse.push((f, e, [gn[0] as f64, gn[1] as f64], NO_STOP)),
            }
        }
        p
    }
}

/// Frames in one chunk of the block-parallel recurrence. `kmconv.wgsl`'s \[17\]
const BQ_CHUNK: usize = 1024;

/// Each filter's powers of its state matrix `A = [[-a1, 1], [-a2, 0]]`, in \[18\]
fn powers(coefs: &[Option<[f32; 4]>]) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(coefs.len().max(1) * (BQ_CHUNK + 2));
    for c in coefs.iter().copied().chain(std::iter::repeat_n(None, usize::from(coefs.is_empty()))) {
        let Some([_, _, a1, a2]) = c else {
            out.extend(std::iter::repeat_n([0.0; 2], BQ_CHUNK + 2));
            continue;
        };
        let a = [[-(a1 as f64), 1.0], [-(a2 as f64), 0.0]];
        let mut m = [[1.0f64, 0.0], [0.0, 1.0]];
        for _ in 0..BQ_CHUNK {
            out.push([m[0][0] as f32, m[0][1] as f32]);
            m = [
                [a[0][0] * m[0][0] + a[0][1] * m[1][0], a[0][0] * m[0][1] + a[0][1] * m[1][1]],
                [a[1][0] * m[0][0] + a[1][1] * m[1][0], a[1][0] * m[0][1] + a[1][1] * m[1][1]],
            ];
        }
        out.push([m[0][0] as f32, m[0][1] as f32]);
        out.push([m[1][0] as f32, m[1][1] as f32]);
    }
    out
}

/// Chunks each filter's state takes to fall a billion times over, from its \[19\]
fn ring_chunks(coefs: &[Option<[f32; 4]>]) -> Vec<u32> {
    type M = [[f64; 2]; 2];
    let mul = |x: M, y: M| -> M {
        [
            [x[0][0] * y[0][0] + x[0][1] * y[1][0], x[0][0] * y[0][1] + x[0][1] * y[1][1]],
            [x[1][0] * y[0][0] + x[1][1] * y[1][0], x[1][0] * y[0][1] + x[1][1] * y[1][1]],
        ]
    };
    coefs
        .iter()
        .map(|c| {
            let Some([_, _, a1, a2]) = *c else { return 0 };
            let a = [[-(a1 as f64), 1.0], [-(a2 as f64), 0.0]];
            let step = (0..BQ_CHUNK).fold([[1.0, 0.0], [0.0, 1.0]], |m, _| mul(a, m));
            let mut m = step;
            for chunks in 1..=crate::merlin::bucket::RING_OUT as u32 / BQ_CHUNK as u32 {
                if m.iter().flatten().all(|v| v.abs() < 1e-9) {
                    return chunks;
                }
                m = mul(step, m);
            }
            u32::MAX
        })
        .collect()
}

/// One size's tables: the twiddles and the four transposes' uniforms.
struct Size {
    n2: usize,
    tw: wgpu::Buffer,
    /// `(rows, cols, mode)`: `(n2, N1, 0)`, `(N1, n2, 1)`, `(n2, N1, 2)`, \[20\]
    t: [wgpu::Buffer; 4],
}

/// A buffer that grows and keeps what it was given.
struct Grown {
    buf: Option<wgpu::Buffer>,
    cap: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl Grown {
    fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self { buf: None, cap: 0, usage, label }
    }

    fn at_least(&mut self, device: &wgpu::Device, bytes: u64) -> &wgpu::Buffer {
        let bytes = bytes.max(256).next_multiple_of(256);
        if self.cap < bytes {
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: bytes,
                usage: self.usage,
                mapped_at_creation: false,
            }));
            self.cap = bytes;
        }
        self.buf.as_ref().expect("made above")
    }

    fn get(&self) -> &wgpu::Buffer {
        self.buf.as_ref().expect("sized before use")
    }
}

struct Pipes {
    transpose: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    mac: wgpu::ComputePipeline,
    ola: wgpu::ComputePipeline,
    load: wgpu::ComputePipeline,
    heads: wgpu::ComputePipeline,
    direct: wgpu::ComputePipeline,
    direct_stopped: wgpu::ComputePipeline,
    bq_chunk: wgpu::ComputePipeline,
    bq_scan: wgpu::ComputePipeline,
    bq_fix: wgpu::ComputePipeline,
    mix: wgpu::ComputePipeline,
    carry_move: wgpu::ComputePipeline,
    carry_clear: wgpu::ComputePipeline,
    warp: wgpu::ComputePipeline,
    lt: wgpu::BindGroupLayout,
    ls: wgpu::BindGroupLayout,
    lm: wgpu::BindGroupLayout,
    lo: wgpu::BindGroupLayout,
    lk: wgpu::BindGroupLayout,
    lh: wgpu::BindGroupLayout,
    ld: wgpu::BindGroupLayout,
    lq: wgpu::BindGroupLayout,
    lx: wgpu::BindGroupLayout,
    lc: wgpu::BindGroupLayout,
    lw: wgpu::BindGroupLayout,
}

/// Where the device's time went, cumulative seconds, timed per call.
#[derive(Default, Clone, Copy, Debug)]
pub struct GpuProfile {
    pub kernels: f64,
    pub conv: f64,
    pub heads: f64,
    pub window: f64,
    pub direct: f64,
    /// Waiting for the device so its upload staging is given back.
    pub throttled: f64,
    pub transforms: u64,
    pub chunks: u64,
    pub pairs: u64,
    pub blocks: u64,
    pub head_pieces: u64,
    pub head_items: u64,
    pub direct_imps: u64,
    pub direct_items: u64,
}

/// The window being rendered.
struct Window {
    frames: usize,
    carry: usize,
    stride: usize,
    filters: usize,
}

pub struct KmGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    info: wgpu::AdapterInfo,
    has_ts: bool,
    /// The kernel builder on this device, which renders a key's sustains \[21\]
    builder: Option<Builder>,
    /// The queue's point at the last throttle, which the next one waits for.
    mark: Option<wgpu::SubmissionIndex>,
    /// The device's time by kind of work, when the device has timestamps.
    pub times: Option<DeviceTimes>,
    f1: Fft,
    f2: HashMap<usize, Fft>,
    sizes: HashMap<usize, Size>,
    p: Pipes,
    a: Grown,
    b: Grown,
    k: Grown,
    y: Grown,
    imps: Grown,
    mdesc: Grown,
    mentry: Grown,
    odesc: [Grown; 2],
    kdesc: Grown,
    heads: Grown,
    items: Grown,
    list: Grown,
    sus: Grown,
    dimp: Grown,
    ditems: Grown,
    ddesc: Grown,
    bus: Grown,
    /// The song's master, every key added in at its windows' frames, and \[22\]
    master: Grown,
    /// While a channel's bent notes render, what the windows add into in place \[23\]
    warped: Grown,
    warped_frames: usize,
    witems: Grown,
    song_frames: usize,
    /// `kmconv.wgsl`'s powers of each filter's state matrix, and the \[24\]
    qpow: Option<wgpu::Buffer>,
    qchunk: Grown,
    /// The buses the current key has written, and those as a list.
    active: Vec<bool>,
    /// Which pages of each bus hold anything, `[buses][stride / PAGE]`, the \[25\]
    cells: Vec<bool>,
    /// Chunks each bus's filter rings for past its last input (`ring_chunks`), \[26\]
    ring: Vec<u32>,
    ringing: Vec<u32>,
    /// A window's end's lists: the cells to filter, the cells a start state \[27\]
    qcells: Grown,
    fcells: Grown,
    franges: Grown,
    xitems: Grown,
    xbuses: Grown,
    cmoves: Grown,
    cclears: Grown,
    /// Windows since each bus was last written.
    quiet: Vec<u8>,
    alist: Grown,
    stage: Grown,
    state: Option<wgpu::Buffer>,
    coef: Option<wgpu::Buffer>,
    on: Option<wgpu::Buffer>,
    u_s: wgpu::Buffer,
    u_m: wgpu::Buffer,
    u_o: [wgpu::Buffer; 2],
    u_k: wgpu::Buffer,
    u_h: wgpu::Buffer,
    u_w: wgpu::Buffer,
    u_e: wgpu::Buffer,
    u_d: wgpu::Buffer,
    u_x: wgpu::Buffer,
    win: Option<Window>,
    /// `Config::merlin_km_direct_below`.
    direct_below: usize,
    /// `Config::merlin_km_segment`.
    segment: usize,
    /// Frames a stopped impulse takes to fade: `Config::steal_fade_frames`.
    fade: usize,
    pub prof: GpuProfile,
    /// Bytes handed to the queue since the device was last waited on.
    staged: u64,
}

/// One entry point's shader: everything before the first `//#section` line -- \[28\]
fn section(src: &str, name: &str) -> String {
    let head = src.split("//#section").next().unwrap_or("");
    let body = src
        .split("//#section ")
        .skip(1)
        .find_map(|s| s.strip_prefix(name).filter(|rest| rest.starts_with(['\n', '\r'])))
        .unwrap_or("");
    format!("{head}\n{body}")
}

fn slice(buf: &wgpu::Buffer, n: usize, from: usize, count: usize) -> wgpu::BufferBinding<'_> {
    let sig = (n * 8) as u64;
    wgpu::BufferBinding { buffer: buf, offset: from as u64 * sig, size: std::num::NonZeroU64::new(count as u64 * sig) }
}

/// Items in a 2-D grid of 256-wide workgroups.
fn grid(items: usize) -> (u32, u32) {
    let groups = items.div_ceil(256).max(1) as u32;
    let x = groups.min(32768);
    (x, groups.div_ceil(x))
}

impl KmGpu {
    pub fn new(cfg: &crate::config::Config) -> Result<Self> {
        let (device, queue, info, limits, has_ts) = crate::gpu::open_device_info(cfg)?;
        let f1 = Fft::new(&device, &limits, N1, 256)?;
        let src = include_str!("../../shaders/kmconv.wgsl");
        let layout = |label: &str, ro: &[bool]| crate::gpu::device::bind_layout(&device, label, ro);
        let lt = layout("kmconv::transpose", &[true, true, false, true]);
        let ls = layout("kmconv::scatter", &[true, true, false]);
        let lm = layout("kmconv::mac", &[true, true, true, true, true, false]);
        let lo = layout("kmconv::ola", &[true, true, true, false]);
        let lk = layout("kmconv::load", &[true, true, true, false]);
        let lh = layout("kmconv::heads", &[true, true, true, true, true, false]);
        let ld = layout("kmconv::direct", &[true, true, true, true, true, false]);
        let lq = layout("kmconv::biquad", &[true, true, true, false, false, true, false, true, true, true]);
        let lx = layout("kmconv::mix", &[true, true, true, false, true]);
        let lc = layout("kmconv::carry", &[true, false, true]);
        let lw = layout("kmconv::warp", &[true, true, true, false]);
        let pipe = |name: &str, l: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(section(src, name).into()),
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[l],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let p = Pipes {
            transpose: pipe("transpose", &lt),
            scatter: pipe("scatter", &ls),
            mac: pipe("mac", &lm),
            ola: pipe("ola", &lo),
            load: pipe("load_kernels", &lk),
            heads: pipe("heads", &lh),
            direct: pipe("direct", &ld),
            direct_stopped: pipe("direct_stopped", &ld),
            bq_chunk: pipe("bq_chunk", &lq),
            bq_scan: pipe("bq_scan", &lq),
            bq_fix: pipe("bq_fix", &lq),
            mix: pipe("mix", &lx),
            carry_move: pipe("carry_move", &lc),
            carry_clear: pipe("carry_clear", &lc),
            warp: pipe("warp", &lw),
            lt,
            ls,
            lm,
            lo,
            lk,
            lh,
            ld,
            lq,
            lx,
            lc,
            lw,
        };
        use wgpu::BufferUsages as U;
        let st = U::STORAGE | U::COPY_DST | U::COPY_SRC;
        let uni = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 48,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            u_s: uni("kmconv::u_scatter"),
            u_m: uni("kmconv::u_mac"),
            u_o: [uni("kmconv::u_ola_even"), uni("kmconv::u_ola_odd")],
            u_k: uni("kmconv::u_load"),
            u_h: uni("kmconv::u_heads"),
            u_w: uni("kmconv::u_window"),
            u_e: uni("kmconv::u_end"),
            u_d: uni("kmconv::u_direct"),
            u_x: uni("kmconv::u_warp"),
            a: Grown::new("kmconv::pairs", st),
            b: Grown::new("kmconv::scratch", st),
            k: Grown::new("kmconv::kernels", st),
            y: Grown::new("kmconv::blocks", st),
            imps: Grown::new("kmconv::impulses", st),
            mdesc: Grown::new("kmconv::mac_desc", st),
            mentry: Grown::new("kmconv::mac_entry", st),
            odesc: [Grown::new("kmconv::ola_even", st), Grown::new("kmconv::ola_odd", st)],
            kdesc: Grown::new("kmconv::kernel_desc", st),
            heads: Grown::new("kmconv::heads", st),
            items: Grown::new("kmconv::head_items", st),
            list: Grown::new("kmconv::head_list", st),
            sus: Grown::new("kmconv::sustains", st),
            dimp: Grown::new("kmconv::direct_impulses", st),
            ditems: Grown::new("kmconv::direct_items", st),
            ddesc: Grown::new("kmconv::direct_desc", st),
            bus: Grown::new("kmconv::buses", st),
            master: Grown::new("kmconv::master", st),
            song_frames: 0,
            warped: Grown::new("kmconv::warped", st),
            warped_frames: 0,
            witems: Grown::new("kmconv::warp_runs", st),
            active: Vec::new(),
            cells: Vec::new(),
            ring: Vec::new(),
            ringing: Vec::new(),
            qcells: Grown::new("kmconv::filter_cells", st),
            fcells: Grown::new("kmconv::ring_cells", st),
            franges: Grown::new("kmconv::ring_ranges", st),
            xitems: Grown::new("kmconv::mix_pages", st),
            xbuses: Grown::new("kmconv::mix_buses", st),
            cmoves: Grown::new("kmconv::carry_moves", st),
            cclears: Grown::new("kmconv::carry_clears", st),
            quiet: Vec::new(),
            qpow: None,
            qchunk: Grown::new("kmconv::chunk_states", st),
            alist: Grown::new("kmconv::active", st),
            stage: Grown::new("kmconv::stage", U::MAP_READ | U::COPY_DST),
            state: None,
            coef: None,
            on: None,
            f1,
            f2: HashMap::new(),
            sizes: HashMap::new(),
            p,
            device,
            queue,
            limits,
            info,
            has_ts,
            builder: None,
            mark: None,
            times: None,
            win: None,
            direct_below: cfg.merlin_km_direct_below as usize,
            segment: cfg.merlin_km_segment as usize,
            fade: cfg.steal_fade_frames.max(1) as usize,
            prof: GpuProfile::default(),
            staged: 0,
        })
    }

    /// The transform size for kernels up to `lmax` frames: a power of two at \[29\]
    pub fn size_for(lmax: usize) -> usize {
        (2 * lmax.max(1)).next_power_of_two().max(N1 * 32)
    }

    /// Signals one buffer holds at `n`.
    pub fn signals(n: usize) -> usize {
        ((BUFFER_BYTES / (n as u64 * 8)) as usize).max(2)
    }

    /// Frames a window can hold with `filters` buses inside `bytes`, `carry` \[30\]
    pub fn window_frames(bytes: u64, filters: usize, carry: usize) -> usize {
        let per = (bytes / (filters.max(1) as u64 * 8)) as usize;
        (per.saturating_sub(carry) / N1 * N1).max(carry.next_multiple_of(N1))
    }

    fn size(&mut self, n: usize) -> Result<()> {
        if self.sizes.contains_key(&n) {
            return Ok(());
        }
        let n2 = n / N1;
        if n2 < 16 || n2 * N1 != n || n2 > N1 {
            bail!("key-major transform of {n} points is outside what the four-step here builds");
        }
        if !self.f2.contains_key(&n2) {
            let wg = (n2 as u32 / 4).clamp(64, 256);
            self.f2.insert(n2, Fft::new(&self.device, &self.limits, n2, wg)?);
        }
        let tw: Vec<[f32; 2]> = (0..n)
            .map(|m| {
                let th = -2.0 * std::f64::consts::PI * m as f64 / n as f64;
                [th.cos() as f32, th.sin() as f32]
            })
            .collect();
        let d = &self.device;
        let tw = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::twiddles"),
            contents: bytemuck::cast_slice(&tw),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mk = |rows: usize, cols: usize, mode: u32| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("kmconv::u_transpose"),
                contents: bytemuck::cast_slice(&[rows as u32, cols as u32, n as u32, mode]),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let t = [mk(n2, N1, 0), mk(N1, n2, 1), mk(n2, N1, 2), mk(N1, n2, 0)];
        self.sizes.insert(n, Size { n2, tw, t });
        Ok(())
    }

    /// The four-step forward (or inverse) transform of `count` signals of `n` \[31\]
    fn four_step(&self, enc: &mut wgpu::CommandEncoder, data: &wgpu::Buffer, scratch: &wgpu::Buffer, n: usize, count: usize, inverse: bool) {
        let sz = &self.sizes[&n];
        let f2 = &self.f2[&sz.n2];
        let d = &self.device;
        let tb = |u: &wgpu::Buffer, s: wgpu::BufferBinding, t: wgpu::BufferBinding| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kmconv::transpose"),
                layout: &self.p.lt,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(s) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(t) },
                    wgpu::BindGroupEntry { binding: 3, resource: sz.tw.as_entire_binding() },
                ],
            })
        };
        let tpass = |enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup, rows: usize, cols: usize, g: usize| {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.transpose);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups((cols as u32).div_ceil(16), (rows as u32).div_ceil(16), g as u32);
        };
        let mut from = 0;
        while from < count {
            let c = (count - from).min(SLICE);
            let (ds, ss) = (slice(data, n, from, c), slice(scratch, n, from, c));
            if !inverse {
                tpass(enc, &tb(&sz.t[0], ds.clone(), ss.clone()), sz.n2, N1, c);
                f2.encode_exact(enc, &f2.bind_exact(d, ss.clone()), (c * N1) as u32, false);
                tpass(enc, &tb(&sz.t[1], ss, ds.clone()), N1, sz.n2, c);
                self.f1.encode_exact(enc, &self.f1.bind_exact(d, ds), (c * sz.n2) as u32, false);
            } else {
                self.f1.encode_exact(enc, &self.f1.bind_exact(d, ds.clone()), (c * sz.n2) as u32, true);
                tpass(enc, &tb(&sz.t[2], ds.clone(), ss.clone()), sz.n2, N1, c);
                f2.encode_exact(enc, &f2.bind_exact(d, ss.clone()), (c * N1) as u32, true);
                tpass(enc, &tb(&sz.t[3], ss, ds), N1, sz.n2, c);
            }
            from += c;
        }
    }

    /// Size the buses for `filters` filters with these coefficients (`None` \[32\]
    pub fn begin(&mut self, coefs: &[Option<[f32; 4]>], frames: usize, carry: usize) -> Result<()> {
        if carry > frames {
            bail!("a key-major window of {frames} frames is shorter than its {carry}-frame carry");
        }
        let filters = coefs.len().max(1);
        let stride = frames + carry;
        if (filters * stride) as u64 > u32::MAX as u64 {
            bail!("{filters} buses of {stride} frames pass what a u32 index reaches");
        }
        let d = self.device.clone();
        let bytes = (filters * stride * 8) as u64;
        self.bus.at_least(&d, bytes);
        let c: Vec<[f32; 4]> = (0..filters).map(|f| coefs.get(f).copied().flatten().unwrap_or([0.0; 4])).collect();
        let on: Vec<u32> = (0..filters).map(|f| coefs.get(f).copied().flatten().is_some() as u32).collect();
        use wgpu::BufferUsages as U;
        self.coef = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::coef"),
            contents: bytemuck::cast_slice(&c),
            usage: U::STORAGE,
        }));
        self.qpow = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::powers"),
            contents: bytemuck::cast_slice(&powers(coefs)),
            usage: U::STORAGE,
        }));
        self.on = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::on"),
            contents: bytemuck::cast_slice(&on),
            usage: U::STORAGE,
        }));
        self.state = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::state"),
            contents: bytemuck::cast_slice(&vec![[0f32; 4]; filters]),
            usage: U::STORAGE | U::COPY_SRC | U::COPY_DST,
        }));
        self.queue.write_buffer(&self.u_w, 0, bytemuck::cast_slice(&[frames as u32, stride as u32, filters as u32, carry as u32]));
        let mut enc = d.create_command_encoder(&Default::default());
        enc.clear_buffer(self.bus.get(), 0, Some(bytes));
        self.queue.submit([enc.finish()]);
        self.win = Some(Window { frames, carry, stride, filters });
        self.active = vec![false; filters];
        self.quiet = vec![0; filters];
        self.reset_cells(coefs, filters, stride);
        Ok(())
    }

    /// Nothing on any bus, at this stride, for these filters.
    fn reset_cells(&mut self, coefs: &[Option<[f32; 4]>], filters: usize, stride: usize) {
        self.cells = vec![false; filters * stride.div_ceil(PAGE)];
        self.ring = ring_chunks(coefs);
        self.ring.resize(filters, 0);
        self.ringing = vec![0; filters];
    }

    /// `[lo, hi)` of `bus` holds something now.
    fn touch(&mut self, bus: usize, lo: usize, hi: usize) {
        let Some(w) = self.win.as_ref() else { return };
        let pages = w.stride.div_ceil(PAGE);
        let row = bus * pages;
        if lo < hi && row + pages <= self.cells.len() {
            self.cells[row + lo / PAGE..=row + ((hi - 1) / PAGE).min(pages - 1)].fill(true);
        }
    }

    /// More filters than the buses were made for, between two windows: the \[33\]
    pub fn grow(&mut self, coefs: &[Option<[f32; 4]>]) -> Result<()> {
        let win = self.win.as_ref().context("growing the buses before the window began")?;
        let (old, stride, frames, carry) = (win.filters, win.stride, win.frames, win.carry);
        let filters = coefs.len();
        if filters <= old {
            return Ok(());
        }
        if (filters * stride) as u64 > u32::MAX as u64 {
            bail!("{filters} buses of {stride} frames pass what a u32 index reaches");
        }
        use wgpu::BufferUsages as U;
        let d = self.device.clone();
        let bytes = ((filters * stride * 8) as u64).next_multiple_of(256);
        if bytes > self.limits.max_storage_buffer_binding_size as u64 {
            bail!(
                "{filters} filters: {} MiB of buses of {stride} frames, past one binding's {} MiB",
                bytes >> 20,
                self.limits.max_storage_buffer_binding_size >> 20
            );
        }
        let old_state = self.state.take().context("no filter state")?;
        let in_place = bytes <= self.bus.cap;
        let grown = (!in_place).then(|| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kmconv::buses"),
                size: bytes,
                usage: U::STORAGE | U::COPY_DST | U::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let state = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kmconv::state"),
            size: (filters * 16) as u64,
            usage: U::STORAGE | U::COPY_SRC | U::COPY_DST,
            mapped_at_creation: false,
        });
        let c: Vec<[f32; 4]> = coefs.iter().map(|c| c.unwrap_or([0.0; 4])).collect();
        let on: Vec<u32> = coefs.iter().map(|c| c.is_some() as u32).collect();
        self.coef = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::coef"),
            contents: bytemuck::cast_slice(&c),
            usage: U::STORAGE,
        }));
        self.qpow = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::powers"),
            contents: bytemuck::cast_slice(&powers(coefs)),
            usage: U::STORAGE,
        }));
        self.on = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::on"),
            contents: bytemuck::cast_slice(&on),
            usage: U::STORAGE,
        }));
        let mut enc = d.create_command_encoder(&Default::default());
        if let Some(bus) = &grown {
            enc.clear_buffer(bus, 0, None);
            enc.copy_buffer_to_buffer(self.bus.get(), 0, bus, 0, (old * stride * 8) as u64);
        }
        enc.clear_buffer(&state, 0, None);
        enc.copy_buffer_to_buffer(&old_state, 0, &state, 0, (old * 16) as u64);
        self.queue.submit([enc.finish()]);
        if let Some(bus) = grown {
            self.bus.buf = Some(bus);
            self.bus.cap = bytes;
        }
        self.state = Some(state);
        self.queue.write_buffer(&self.u_w, 0, bytemuck::cast_slice(&[frames as u32, stride as u32, filters as u32, carry as u32]));
        self.win.as_mut().expect("checked above").filters = filters;
        self.active.resize(filters, false);
        self.quiet.resize(filters, 0);
        self.cells.resize(filters * stride.div_ceil(PAGE), false);
        self.ring = ring_chunks(coefs);
        self.ringing.resize(filters, 0);
        Ok(())
    }

    /// A new key whose buses are its own filters, `coefs` in the order its \[34\]
    pub fn start_key_with(&mut self, coefs: &[Option<[f32; 4]>], frames: usize) -> Result<()> {
        self.start_key_room(coefs, frames, coefs.len())
    }

    /// `start_key_with`, the buses' buffer holding `room` filters however few \[35\]
    pub fn start_key_room(&mut self, coefs: &[Option<[f32; 4]>], frames: usize, room: usize) -> Result<()> {
        self.start_key()?;
        let win = self.win.as_ref().context("a key before the buses were made")?;
        let carry = win.carry;
        if frames < carry {
            bail!("a key-major window of {frames} frames is shorter than its {carry}-frame carry");
        }
        let stride = frames + carry;
        // [36]
        let relaid = stride != win.stride;
        let one = [None];
        let coefs = if coefs.is_empty() { &one[..] } else { coefs };
        let filters = coefs.len();
        let bytes = (filters * stride * 8) as u64;
        if bytes > self.limits.max_storage_buffer_binding_size as u64 || (filters * stride) as u64 > u32::MAX as u64 {
            bail!(
                "a key uses {filters} filters: {} MiB of buses of {stride} frames, past one binding's {} MiB",
                bytes >> 20,
                self.limits.max_storage_buffer_binding_size >> 20
            );
        }
        let bytes = ((room * stride * 8) as u64).min(self.limits.max_storage_buffer_binding_size as u64).max(bytes);
        let d = self.device.clone();
        if self.bus.cap < bytes.max(256).next_multiple_of(256) {
            // The bigger buses only once the device is done with the smaller.
            self.bus.buf = None;
            self.bus.cap = 0;
            self.device.poll(wgpu::PollType::wait_indefinitely()).context("waiting to grow the buses")?;
            self.bus.at_least(&d, bytes);
            let mut enc = d.create_command_encoder(&Default::default());
            enc.clear_buffer(self.bus.get(), 0, None);
            self.queue.submit([enc.finish()]);
        } else if relaid {
            let mut enc = d.create_command_encoder(&Default::default());
            enc.clear_buffer(self.bus.get(), 0, None);
            self.queue.submit([enc.finish()]);
        }
        use wgpu::BufferUsages as U;
        let c: Vec<[f32; 4]> = coefs.iter().map(|c| c.unwrap_or([0.0; 4])).collect();
        let on: Vec<u32> = coefs.iter().map(|c| c.is_some() as u32).collect();
        self.coef = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("kmconv::coef"), contents: bytemuck::cast_slice(&c), usage: U::STORAGE }));
        self.qpow = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("kmconv::powers"), contents: bytemuck::cast_slice(&powers(coefs)), usage: U::STORAGE }));
        self.on = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("kmconv::on"), contents: bytemuck::cast_slice(&on), usage: U::STORAGE }));
        self.state = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kmconv::state"),
            contents: bytemuck::cast_slice(&vec![[0f32; 4]; filters]),
            usage: U::STORAGE | U::COPY_SRC | U::COPY_DST,
        }));
        self.queue.write_buffer(&self.u_w, 0, bytemuck::cast_slice(&[frames as u32, stride as u32, filters as u32, carry as u32]));
        let w = self.win.as_mut().expect("checked above");
        w.filters = filters;
        w.frames = frames;
        w.stride = stride;
        self.active = vec![false; filters];
        self.quiet = vec![0; filters];
        self.reset_cells(coefs, filters, stride);
        Ok(())
    }

    /// Count `bytes` handed to the queue, and once more than `STAGING_BYTES` \[37\]
    fn throttle(&mut self, bytes: u64) -> Result<()> {
        if self.times.as_ref().is_some_and(|g| g.next.load(std::sync::atomic::Ordering::Relaxed) > TIME_MARKS - 512) {
            self.time_flush();
        }
        self.staged += bytes;
        if self.staged > STAGING_BYTES {
            // [38]
            let t = std::time::Instant::now();
            let mark = self.queue.submit(std::iter::empty());
            if let Some(prev) = self.mark.replace(mark) {
                self.device.poll(wgpu::PollType::Wait { submission_index: Some(prev), timeout: None }).context("waiting for key-major's uploads")?;
            }
            self.staged = 0;
            self.prof.throttled += t.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// What `pack` needs of the window, once it has begun.
    pub fn pack_cfg(&self) -> Result<PackCfg> {
        let win = self.win.as_ref().context("a key before the window began")?;
        Ok(PackCfg { stride: win.stride, carry: win.carry, frames: win.frames, direct_below: self.direct_below, segment: self.segment, fade: self.fade })
    }

    /// One key's part of the window: its groups' convolutions onto their \[39\]
    pub fn key(&mut self, kernels: &[&[f32]], groups: &[GroupIn], sustains: &[&[f32]], heads: &[Head]) -> Result<()> {
        let c = self.pack_cfg()?;
        let mut pool: Vec<f32> = sustains.iter().flat_map(|s| s.iter().copied()).collect();
        let table = append_whole(&mut pool, kernels);
        self.set_pool(&pool)?;
        let p = Self::pack(&c, &table, groups, heads.to_vec())?;
        self.put(p)
    }

    /// Time the device's work by timestamps, if it was opened with them.
    pub fn time_device(&mut self) {
        if !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return;
        }
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("kmconv::times"), ty: wgpu::QueryType::Timestamp, count: TIME_MARKS });
        let mk = |u| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("kmconv::times"), size: TIME_MARKS as u64 * 8, usage: u, mapped_at_creation: false });
        let resolve = mk(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
        let read = mk(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        self.times = Some(DeviceTimes {
            set,
            resolve,
            read,
            next: std::sync::atomic::AtomicU32::new(0),
            pend: std::sync::Mutex::new(Vec::new()),
            secs: [0.0; 8],
            period: self.queue.get_timestamp_period() as f64,
        });
    }

    fn time_mark(&self, enc: &mut wgpu::CommandEncoder) -> Option<u32> {
        let g = self.times.as_ref()?;
        let i = g.next.load(std::sync::atomic::Ordering::Relaxed);
        if i >= TIME_MARKS {
            return None;
        }
        g.next.store(i + 1, std::sync::atomic::Ordering::Relaxed);
        enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: &g.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: None }),
        });
        Some(i)
    }

    fn time_close(&self, enc: &mut wgpu::CommandEncoder, a: Option<u32>, slot: usize) {
        if let Some(a) = a {
            if let Some(b) = self.time_mark(enc) {
                self.times.as_ref().expect("marked").pend.lock().unwrap().push((slot, a, b));
            }
        }
    }

    /// Read the marks back and add them up; between submissions only.
    pub fn time_flush(&mut self) {
        let Some(g) = self.times.as_mut() else { return };
        let n = g.next.load(std::sync::atomic::Ordering::Relaxed);
        if n == 0 {
            return;
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.resolve_query_set(&g.set, 0..n, &g.resolve, 0);
        enc.copy_buffer_to_buffer(&g.resolve, 0, &g.read, 0, n as u64 * 8);
        self.queue.submit([enc.finish()]);
        let sl = g.read.slice(0..n as u64 * 8);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        {
            let data = sl.get_mapped_range();
            let ts: &[u64] = bytemuck::cast_slice(&data);
            for &(slot, a, b) in g.pend.lock().unwrap().iter() {
                g.secs[slot] += ts[b as usize].saturating_sub(ts[a as usize]) as f64 * g.period * 1e-9;
            }
        }
        g.read.unmap();
        g.pend.lock().unwrap().clear();
        g.next.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Render the next keys' sustains here, on the device, with `bank`'s \[40\]
    pub fn attach_builder(&mut self, cfg: &crate::config::Config, bank: &std::sync::Arc<crate::bank::Bank>) -> Result<()> {
        let phase = crate::phase::PhaseBank::prepare(bank, &cfg.phase)?;
        let shared = crate::gpu::GpuShared::on_device(
            cfg,
            bank,
            &phase,
            self.device.clone(),
            self.queue.clone(),
            self.info.clone(),
            self.limits.clone(),
            self.has_ts,
        )?;
        let b = crate::merlin::gpubuild::GpuBuilder::new(cfg, bank.clone(), &shared, BUILD_ROWS)?.with_scratch(BUILD_BYTES);
        self.builder = Some(Builder { b, menv: std::sync::Arc::new(bank.menv.clone()), _shared: shared });
        Ok(())
    }

    /// The pool the next windows read, rendered here: job `j`'s sustain at \[41\]
    pub fn build_pool(
        &mut self,
        tables: &[BuildTable],
        total: usize,
        host: &[(u64, Vec<f32>)],
        direct: &[crate::merlin::gpubuild::DirectJob],
    ) -> Result<()> {
        if (total as u64) * 4 > self.limits.max_storage_buffer_binding_size as u64 {
            bail!("a pool of {} MiB passes one binding's {} MiB", total >> 18, self.limits.max_storage_buffer_binding_size >> 20);
        }
        let d = self.device.clone();
        // [42]
        let want = (total.max(1) * 4) as u64;
        if self.sus.cap < want.max(256).next_multiple_of(256) && self.sus.buf.is_some() {
            self.sus.buf = None;
            self.sus.cap = 0;
            self.device.poll(wgpu::PollType::wait_indefinitely()).context("waiting to grow the pool")?;
        }
        self.sus.at_least(&d, want);
        for (a, v) in host {
            self.queue.write_buffer(self.sus.get(), a * 4, bytemuck::cast_slice(v));
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        self.queue.submit([enc.finish()]);
        let b = self.builder.as_mut().context("no kernel builder on the device")?;
        b.b.direct_into(direct, self.sus.get())?;
        for (params, jobs, at) in tables {
            b.b.render_into(params, &b.menv, jobs, self.sus.get(), at)?;
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.time_close(&mut enc, tm, 6);
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// The pool the next windows read by offset: the sustains the heads' \[43\]
    pub fn set_pool(&mut self, pool: &[f32]) -> Result<()> {
        if (pool.len() as u64) * 4 > self.limits.max_storage_buffer_binding_size as u64 {
            bail!("a pool of {} MiB passes one binding's {} MiB", pool.len() >> 18, self.limits.max_storage_buffer_binding_size >> 20);
        }
        let d = self.device.clone();
        self.sus.at_least(&d, (pool.len().max(1) * 4) as u64);
        self.queue.write_buffer(self.sus.get(), 0, bytemuck::cast_slice(pool));
        self.throttle((pool.len() * 4) as u64)
    }

    /// `key`'s host work, on any thread: everything the device is handed, \[44\]
    pub fn pack(c: &PackCfg, kernels: &[KDesc], groups: &[GroupIn], heads: Vec<Head>) -> Result<Packed> {
        // The buses this writes, which the window's end then filters and mixes.
        let mut active: Vec<u32> = groups.iter().filter(|g| !g.impulses.is_empty() || !g.stopped.is_empty()).map(|g| g.bus).collect();
        active.extend(heads.chunk_by(|x, y| x.bus == y.bus).map(|run| run[0].bus));
        active.sort_unstable();
        active.dedup();
        let mut p = Packed { active, ops: Vec::new(), bytes: 0 };
        if groups.iter().any(|g| !g.impulses.is_empty() || !g.stopped.is_empty()) {
            Self::pack_convolve(c, kernels, groups, &mut p)?;
        }
        let head_bytes = std::mem::size_of_val(&heads[..]);
        if !heads.is_empty() {
            Self::pack_heads(c, heads, &mut p)?;
        }
        // [45]
        let imp_bytes: usize = groups.iter().map(|g| (g.impulses.len() + g.stopped.len()) * 16).sum();
        p.bytes = (imp_bytes + head_bytes) as u64;
        Ok(p)
    }

    /// What `pack` made, onto the device in its order.
    pub fn put(&mut self, p: Packed) -> Result<()> {
        for &bus in &p.active {
            if let Some(a) = self.active.get_mut(bus as usize) {
                *a = true;
                self.quiet[bus as usize] = 0;
            }
        }
        let stride = self.win.as_ref().map_or(1, |w| w.stride);
        for op in &p.ops {
            match op {
                Op::Heads { items, .. } | Op::Direct { items, .. } | Op::Stopped { items, .. } => {
                    for it in items.chunks_exact(4) {
                        self.touch(it[1] as usize, it[0] as usize, it[0] as usize + PAGE);
                    }
                }
                Op::Chunk { n, parity, .. } => {
                    for b in parity.iter().flat_map(|l| l.chunks_exact(2)) {
                        let (bus, at) = (b[1] as usize / stride, b[1] as usize % stride);
                        self.touch(bus, at, at + n);
                    }
                }
                Op::Kernels { .. } => {}
            }
        }
        for op in p.ops {
            let t = std::time::Instant::now();
            match op {
                Op::Kernels { n, e, desc } => {
                    self.size(n)?;
                    let cap = Self::signals(n);
                    let sig = (n * 8) as u64;
                    let d = self.device.clone();
                    self.a.at_least(&d, cap as u64 * sig);
                    self.b.at_least(&d, cap as u64 * sig);
                    self.y.at_least(&d, cap as u64 * sig);
                    self.load_kernels(&desc, e, n)?;
                    self.prof.conv += t.elapsed().as_secs_f64();
                }
                Op::Chunk { n, imps, entries, mdesc, parity, blocks, npairs } => {
                    self.chunk(&imps, &entries, &mdesc, &parity, blocks, npairs, n)?;
                    self.prof.conv += t.elapsed().as_secs_f64();
                }
                Op::Direct { dimps, items, desc, stride } => {
                    self.direct(false, &dimps, &items, &desc, stride)?;
                    self.prof.conv += t.elapsed().as_secs_f64();
                }
                Op::Stopped { dimps, items, desc, stride } => {
                    self.direct(true, &dimps, &items, &desc, stride)?;
                    self.prof.conv += t.elapsed().as_secs_f64();
                }
                Op::Heads { heads, items, list, stride } => {
                    self.heads(&heads, &items, &list, stride)?;
                    self.prof.heads += t.elapsed().as_secs_f64();
                }
            }
        }
        self.throttle(p.bytes)
    }

    /// The spectra of the kernels `desc` names in the kernel pool (offset and \[46\]
    fn load_kernels(&mut self, desc: &[u32], e: usize, n: usize) -> Result<()> {
        let t = std::time::Instant::now();
        let d = self.device.clone();
        let sig = (n * 8) as u64;
        self.k.at_least(&d, e as u64 * sig);
        self.b.at_least(&d, Self::signals(n).max(e) as u64 * sig);
        self.kdesc.at_least(&d, (desc.len().max(1) * 4) as u64);
        self.queue.write_buffer(self.kdesc.get(), 0, bytemuck::cast_slice(desc));
        self.queue.write_buffer(&self.u_k, 0, bytemuck::cast_slice(&[n as u32, e as u32, 0, 0]));
        let bg = crate::gpu::device::bind(&d, &self.p.lk, &[&self.u_k, self.sus.get(), self.kdesc.get(), self.k.get()]);
        let mut enc = d.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.load);
            p.set_bind_group(0, &bg, &[]);
            let (x, y) = grid(n);
            p.dispatch_workgroups(x, y, e as u32);
        }
        self.four_step(&mut enc, self.k.get(), self.b.get(), n, e, false);
        self.time_close(&mut enc, tm, 0);
        self.queue.submit([enc.finish()]);

        self.prof.transforms += e as u64;
        self.prof.kernels += t.elapsed().as_secs_f64();
        Ok(())
    }

    fn pack_convolve(c: &PackCfg, kernels: &[KDesc], groups: &[GroupIn], out: &mut Packed) -> Result<()> {
        let lmax = kernels.iter().map(|k| k.len as usize).max().unwrap_or(1).max(1);
        let n = Self::size_for(lmax);
        if n > c.carry {
            bail!("a {n}-point block runs past the window's {}-frame carry", c.carry);
        }
        let stride = c.stride;
        let hop = n - lmax + 1;
        let cap = Self::signals(n);
        // [47]
        let below = c.direct_below;
        let prepped: Vec<Prepped> = groups.par_iter().map(|g| Prepped::new(g, hop, below)).collect();
        // [48]
        for s0 in (0..kernels.len()).step_by(cap) {
            let s1 = (s0 + cap).min(kernels.len());
            let mut bases: Vec<usize> = Vec::new();
            let mut pairs: Vec<(usize, u64, u32)> = Vec::new();
            let mut merged: Vec<(usize, u32, [f64; 2])> = Vec::new();
            for (g, p) in groups.iter().zip(&prepped) {
                let gi = bases.len();
                let p0 = pairs.len();
                let mut to = vec![usize::MAX; p.pairs.len()];
                for (k, &(b, e)) in p.pairs.iter().enumerate() {
                    if (e as usize) >= s0 && (e as usize) < s1 {
                        to[k] = pairs.len();
                        pairs.push((gi, b, e - s0 as u32));
                    }
                }
                if pairs.len() == p0 {
                    continue;
                }
                bases.push(g.bus as usize * stride);
                merged.extend(p.dense.iter().filter(|m| to[m.0 as usize] != usize::MAX).map(|m| (to[m.0 as usize], m.1, m.2)));
            }
            if pairs.is_empty() {
                continue;
            }
            let sub = &kernels[s0..s1];
            let mut desc: Vec<u32> = Vec::with_capacity(KDESC_WORDS * sub.len());
            for k in sub {
                if k.len as usize > n {
                    bail!("a kernel of {} frames does not fit a {n}-point transform", k.len);
                }
                desc.extend_from_slice(&k.words());
            }
            out.ops.push(Op::Kernels { n, e: sub.len(), desc });
            Self::pack_chunks(sub, &bases, &pairs, &merged, n, hop, out)?;
        }
        // [49]
        for short in [true, false] {
            let lists: Vec<(u32, Vec<Sparse>)> = groups
                .iter()
                .zip(&prepped)
                .map(|(g, p)| (g.bus, p.sparse.iter().filter(|s| (kernels[s.1 as usize].len as usize <= c.segment) == short).copied().collect::<Vec<Sparse>>()))
                .filter(|(_, l)| !l.is_empty())
                .collect();
            if !lists.is_empty() {
                let sparse: Vec<(u32, &[Sparse])> = lists.iter().map(|(b, l)| (*b, &l[..])).collect();
                let (dimps, items, desc) = Self::pack_direct(kernels, &sparse, stride, c.fade, |at, entry, [l, r], _| Dimp { at, entry, l, r })?;
                out.ops.push(Op::Direct { dimps, items, desc, stride });
            }
        }
        let stopped: Vec<(u32, Vec<Sparse>)> = groups
            .iter()
            .filter(|g| !g.stopped.is_empty())
            .map(|g| {
                let mut l: Vec<Sparse> = g.stopped.iter().map(|&(e, f, gn, stop)| (f, e, [gn[0] as f64, gn[1] as f64], stop)).collect();
                l.sort_by_key(|s| (s.0, s.1, s.3));
                (g.bus, l)
            })
            .collect();
        if !stopped.is_empty() {
            let sparse: Vec<(u32, &[Sparse])> = stopped.iter().map(|(b, l)| (*b, &l[..])).collect();
            let (dimps, items, desc) = Self::pack_direct(kernels, &sparse, stride, c.fade, |at, entry, [l, r], stop| Dstop { at, entry, l, r, stop })?;
            out.ops.push(Op::Stopped { dimps, items, desc, stride });
        }
        Ok(())
    }

    /// The sparse pairs' impulses added onto their buses in the time domain, \[50\]
    fn pack_direct<T>(kernels: &[KDesc], sparse: &[(u32, &[Sparse])], stride: usize, fade: usize, rec: impl Fn(u32, u32, [f32; 2], u32) -> T) -> Result<(Vec<T>, Vec<u32>, Vec<u32>)> {
        // Only the kernels these impulses use go up, in id order.
        let mut row = vec![u32::MAX; kernels.len()];
        for (_, imps) in sparse {
            for i in imps.iter() {
                row[i.1 as usize] = 0;
            }
        }
        let mut desc: Vec<u32> = Vec::new();
        for (e, r) in row.iter_mut().enumerate() {
            if *r == 0 {
                *r = (desc.len() / KDESC_WORDS) as u32;
                desc.extend_from_slice(&kernels[e].words());
            }
        }
        let mut dimps: Vec<T> = Vec::new();
        let mut items: Vec<u32> = Vec::new();
        for &(bus, merged) in sparse {
            let g0 = dimps.len();
            let mut reach = 0u64;
            let mut lmax = 1u64;
            for &(f, e, gn, stop) in merged {
                // A stopped kernel is silent past its fade.
                let len = (kernels[e as usize].len as u64).min(stop as u64 + fade as u64);
                if f + len > stride as u64 {
                    bail!("a sparse impulse at frame {f} reaches past the bus's {stride} frames");
                }
                reach = reach.max(f + len);
                lmax = lmax.max(len);
                dimps.push(rec(f as u32, row[e as usize], [gn[0] as f32, gn[1] as f32], stop));
            }
            // [51]
            let (mut lo, mut hi) = (0, 0);
            let first = merged[0].0 as usize / PAGE;
            for page in first..=((reach.max(1) as usize - 1) / PAGE) {
                let (p0, p1) = ((page * PAGE) as u64, ((page + 1) * PAGE) as u64);
                while lo < merged.len() && merged[lo].0 + lmax <= p0 {
                    lo += 1;
                }
                hi = hi.max(lo);
                while hi < merged.len() && merged[hi].0 < p1 {
                    hi += 1;
                }
                if lo < hi {
                    items.extend_from_slice(&[p0 as u32, bus, (g0 + lo) as u32, (g0 + hi) as u32]);
                }
            }
        }
        Ok((dimps, items, desc))
    }

    fn direct<T: bytemuck::Pod>(&mut self, stopped: bool, dimps: &[T], items: &[u32], desc: &[u32], stride: usize) -> Result<()> {
        let t = std::time::Instant::now();
        let d = self.device.clone();
        self.dimp.at_least(&d, std::mem::size_of_val(dimps) as u64);
        self.ditems.at_least(&d, (items.len().max(4) * 4) as u64);
        self.ddesc.at_least(&d, (desc.len().max(KDESC_WORDS) * 4) as u64);
        self.queue.write_buffer(self.dimp.get(), 0, bytemuck::cast_slice(dimps));
        self.queue.write_buffer(self.ditems.get(), 0, bytemuck::cast_slice(items));
        self.queue.write_buffer(self.ddesc.get(), 0, bytemuck::cast_slice(desc));
        let count = items.len() / 4;
        self.queue.write_buffer(&self.u_d, 0, bytemuck::cast_slice(&[stride as u32, count as u32, self.fade as u32, 0]));
        let bg = crate::gpu::device::bind(
            &d,
            &self.p.ld,
            &[&self.u_d, self.dimp.get(), self.ditems.get(), self.sus.get(), self.ddesc.get(), self.bus.get()],
        );
        let mut enc = d.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(if stopped { &self.p.direct_stopped } else { &self.p.direct });
            p.set_bind_group(0, &bg, &[]);
            let x = (count as u32).clamp(1, 32768);
            p.dispatch_workgroups(x, (count as u32).div_ceil(x).max(1), 1);
        }
        self.time_close(&mut enc, tm, 1);
        self.queue.submit([enc.finish()]);

        self.prof.direct_imps += dimps.len() as u64;
        self.prof.direct_items += count as u64;
        self.prof.direct += t.elapsed().as_secs_f64();
        Ok(())
    }

    /// Dense pairs of one subset of the key's entries, `(group, block, entry)` \[52\]
    #[allow(clippy::too_many_arguments)]
    fn pack_chunks(kernels: &[KDesc], bases: &[usize], pairs: &[(usize, u64, u32)], merged: &[(usize, u32, [f64; 2])], n: usize, hop: usize, out: &mut Packed) -> Result<()> {
        let cap = Self::signals(n);
        let (mut first, mut mi) = (0, 0);
        while first < pairs.len() {
            let mut end = first;
            // (bus frame, first pair, end, parity)
            let mut blocks: Vec<(u64, u32, u32, usize)> = Vec::new();
            while end < pairs.len() {
                let (gi, b) = (pairs[end].0, pairs[end].1);
                let mut j = end;
                while j < pairs.len() && pairs[j].0 == gi && pairs[j].1 == b {
                    j += 1;
                }
                if j - first > cap {
                    if blocks.is_empty() {
                        bail!("one block fires {} entries and a buffer holds {cap}", j - first);
                    }
                    break;
                }
                let at = bases[gi] as u64 + b * hop as u64;
                blocks.push((at, (end - first) as u32, (j - first) as u32, (b % 2) as usize));
                end = j;
            }
            let mut imps: Vec<Imp> = Vec::new();
            // [53]
            let mut side = vec![false; end - first];
            while mi < merged.len() && merged[mi].0 < end {
                let (p, off, gn) = merged[mi];
                let (a, b) = if kernels[pairs[p].2 as usize].right != 0 {
                    side[p - first] |= gn[0] != gn[1];
                    ((gn[0] + gn[1]) * 0.5, (gn[0] - gn[1]) * 0.5)
                } else {
                    (gn[0], gn[1])
                };
                imps.push(Imp { at: ((p - first) * n) as u32 + off, l: a as f32, r: b as f32, _p: 0 });
                mi += 1;
            }
            let entries: Vec<u32> = pairs[first..end].iter().zip(&side).map(|(p, &s)| p.2 | if s { SIDE } else { 0 }).collect();
            let mdesc: Vec<u32> = blocks.iter().flat_map(|&(_, lo, hi, _)| [lo, hi]).collect();
            // [54]
            let mut parity: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
            for (i, &(at, _, _, par)) in blocks.iter().enumerate() {
                if at + n as u64 > u32::MAX as u64 {
                    bail!("a bus frame passed 2^32");
                }
                parity[par].extend_from_slice(&[i as u32, at as u32]);
            }
            out.ops.push(Op::Chunk { n, imps, entries, mdesc, parity, blocks: blocks.len(), npairs: end - first });
            first = end;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn chunk(&mut self, imps: &[Imp], entries: &[u32], mdesc: &[u32], parity: &[Vec<u32>; 2], blocks: usize, npairs: usize, n: usize) -> Result<()> {
        let d = self.device.clone();
        let sig = (n * 8) as u64;
        self.imps.at_least(&d, (imps.len().max(1) * 16) as u64);
        self.mentry.at_least(&d, (entries.len().max(1) * 4) as u64);
        self.mdesc.at_least(&d, (mdesc.len().max(1) * 4) as u64);
        self.queue.write_buffer(self.imps.get(), 0, bytemuck::cast_slice(imps));
        self.queue.write_buffer(self.mentry.get(), 0, bytemuck::cast_slice(entries));
        self.queue.write_buffer(self.mdesc.get(), 0, bytemuck::cast_slice(mdesc));
        self.queue.write_buffer(&self.u_s, 0, bytemuck::cast_slice(&[imps.len() as u32, 0, 0, 0]));
        self.queue.write_buffer(&self.u_m, 0, bytemuck::cast_slice(&[n as u32, blocks as u32, N1 as u32, 0]));
        for (par, list) in parity.iter().enumerate() {
            self.odesc[par].at_least(&d, (list.len().max(2) * 4) as u64);
            self.queue.write_buffer(self.odesc[par].get(), 0, bytemuck::cast_slice(list));
            self.queue.write_buffer(&self.u_o[par], 0, bytemuck::cast_slice(&[n as u32, (list.len() / 2) as u32, 0, 0]));
        }

        let mut enc = d.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        enc.clear_buffer(self.a.get(), 0, Some(npairs as u64 * sig));
        let bs = crate::gpu::device::bind(&d, &self.p.ls, &[&self.u_s, self.imps.get(), self.a.get()]);
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.scatter);
            p.set_bind_group(0, &bs, &[]);
            p.dispatch_workgroups((imps.len().div_ceil(256) as u32).clamp(1, 65535), 1, 1);
        }
        self.four_step(&mut enc, self.a.get(), self.b.get(), n, npairs, false);
        let bm = crate::gpu::device::bind(
            &d,
            &self.p.lm,
            &[&self.u_m, self.a.get(), self.k.get(), self.mdesc.get(), self.mentry.get(), self.y.get()],
        );
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.mac);
            p.set_bind_group(0, &bm, &[]);
            let (x, y) = grid(n);
            p.dispatch_workgroups(x, y, blocks as u32);
        }
        self.four_step(&mut enc, self.y.get(), self.b.get(), n, blocks, true);
        for (par, list) in parity.iter().enumerate() {
            let count = list.len() / 2;
            if count == 0 {
                continue;
            }
            let bo = crate::gpu::device::bind(&d, &self.p.lo, &[&self.u_o[par], self.y.get(), self.odesc[par].get(), self.bus.get()]);
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.ola);
            p.set_bind_group(0, &bo, &[]);
            let (x, y) = grid(n);
            p.dispatch_workgroups(x, y, count as u32);
        }
        self.time_close(&mut enc, tm, 2);
        self.queue.submit([enc.finish()]);

        self.prof.chunks += 1;
        self.prof.transforms += (npairs + blocks) as u64;
        self.prof.pairs += npairs as u64;
        self.prof.blocks += blocks as u64;
        Ok(())
    }

    /// A key's heads onto their buses: filed by (page, bus), in their order \[55\]
    fn pack_heads(c: &PackCfg, heads: Vec<Head>, out: &mut Packed) -> Result<()> {
        let (stride, frames) = (c.stride, c.frames);
        let (buses, pages) = (heads.iter().map(|h| h.bus as usize + 1).max().unwrap_or(1), frames.div_ceil(PAGE));
        let reach = |h: &Head| (h.out0 as usize / PAGE)..=((h.out0 as usize + h.len as usize - 1) / PAGE);
        let mut at = vec![0u32; pages * buses + 1];
        for h in heads.iter().filter(|h| h.len != 0) {
            let end = h.out0 as usize + h.len as usize;
            if end > frames {
                bail!("a head piece runs to frame {end} of a {frames}-frame window");
            }
            for page in reach(h) {
                at[page * buses + h.bus as usize + 1] += 1;
            }
        }
        for cell in 0..pages * buses {
            at[cell + 1] += at[cell];
        }
        let mut items: Vec<u32> = Vec::new();
        for page in 0..pages {
            for bus in 0..buses {
                let cell = page * buses + bus;
                if at[cell] < at[cell + 1] {
                    items.extend_from_slice(&[(page * PAGE) as u32, bus as u32, at[cell], at[cell + 1]]);
                }
            }
        }
        let mut list = vec![0u32; at[pages * buses] as usize];
        for (i, h) in heads.iter().enumerate().filter(|(_, h)| h.len != 0) {
            for page in reach(h) {
                let cell = page * buses + h.bus as usize;
                list[at[cell] as usize] = i as u32;
                at[cell] += 1;
            }
        }
        out.ops.push(Op::Heads { heads, items, list, stride });
        Ok(())
    }

    fn heads(&mut self, heads: &[Head], items: &[u32], list: &[u32], stride: usize) -> Result<()> {
        let d = self.device.clone();
        self.heads.at_least(&d, std::mem::size_of_val(heads) as u64);
        self.items.at_least(&d, (items.len().max(4) * 4) as u64);
        self.list.at_least(&d, (list.len().max(1) * 4) as u64);
        self.queue.write_buffer(self.heads.get(), 0, bytemuck::cast_slice(heads));
        self.queue.write_buffer(self.items.get(), 0, bytemuck::cast_slice(items));
        self.queue.write_buffer(self.list.get(), 0, bytemuck::cast_slice(list));
        let count = items.len() / 4;
        self.queue.write_buffer(&self.u_h, 0, bytemuck::cast_slice(&[stride as u32, count as u32, 0, 0]));
        let bg = crate::gpu::device::bind(
            &d,
            &self.p.lh,
            &[&self.u_h, self.heads.get(), self.items.get(), self.list.get(), self.sus.get(), self.bus.get()],
        );
        let mut enc = d.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.heads);
            p.set_bind_group(0, &bg, &[]);
            let x = (count as u32).clamp(1, 32768);
            p.dispatch_workgroups(x, (count as u32).div_ceil(x).max(1), 1);
        }
        self.time_close(&mut enc, tm, 3);
        self.queue.submit([enc.finish()]);

        self.prof.head_pieces += heads.len() as u64;
        self.prof.head_items += count as u64;
        Ok(())
    }

    /// A new key: the buses the last one wrote, and every filter's state, back \[56\]
    pub fn start_key(&mut self) -> Result<()> {
        let win = self.win.as_ref().context("a key before the buses were made")?;
        let stride = win.stride as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        for (f, _) in self.active.iter().enumerate().filter(|(_, &a)| a) {
            enc.clear_buffer(self.bus.get(), f as u64 * stride * 8, Some(stride * 8));
        }
        enc.clear_buffer(self.state.as_ref().context("no filter state")?, 0, None);
        self.time_close(&mut enc, tm, 5);
        self.queue.submit([enc.finish()]);
        self.active.iter_mut().for_each(|a| *a = false);
        self.cells.fill(false);
        self.ringing.fill(0);
        Ok(())
    }

    /// The master at `need` frames at least, grown by half again so a song \[57\]
    fn master_at_least(&mut self, need: usize) -> Result<()> {
        if need as u64 * 8 > self.limits.max_storage_buffer_binding_size as u64 {
            bail!("a song of {need} frames passes what one storage binding holds for key-major's master");
        }
        let d = self.device.clone();
        if need > self.song_frames {
            // Grown by half again at least, the old frames copied over.
            let frames_new = need.max(self.song_frames + self.song_frames / 2);
            let bytes = (frames_new as u64 * 8).next_multiple_of(256);
            let song = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kmconv::master"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut enc = d.create_command_encoder(&Default::default());
            let tm = self.time_mark(&mut enc);
            enc.clear_buffer(&song, 0, None);
            if let Some(old) = self.master.buf.take() {
                enc.copy_buffer_to_buffer(&old, 0, &song, 0, self.song_frames as u64 * 8);
            }
            self.time_close(&mut enc, tm, 4);
            self.queue.submit([enc.finish()]);
            self.master.buf = Some(song);
            self.master.cap = bytes;
            self.song_frames = frames_new;
        }
        Ok(())
    }

    /// The windows from here add into a master of their own, until `warp_end`.
    pub fn warp_begin(&mut self) {
        std::mem::swap(&mut self.master, &mut self.warped);
        std::mem::swap(&mut self.song_frames, &mut self.warped_frames);
    }

    /// Back to the song's master, and what the windows added since `warp_begin` \[58\]
    pub fn warp_end(&mut self, w: &crate::merlin::keyparse::Warp, lo: u64, hi: u64) -> Result<()> {
        self.warp_begin();
        let (f0, f1) = (w.frame(lo as f64).floor() as u64, w.frame(hi as f64).ceil() as u64);
        self.master_at_least(f1 as usize)?;
        let runs = w.runs(f0, f1, 256);
        let d = self.device.clone();
        self.witems.at_least(&d, std::mem::size_of_val(&runs[..]).max(20) as u64);
        self.queue.write_buffer(self.witems.get(), 0, bytemuck::cast_slice(&runs));
        self.queue.write_buffer(&self.u_x, 0, bytemuck::cast_slice(&[runs.len() as u32, 0, 0, 0]));
        let bg = crate::gpu::device::bind(&d, &self.p.lw, &[&self.u_x, self.witems.get(), self.warped.get(), self.master.get()]);
        let mut enc = d.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.warp);
            p.set_bind_group(0, &bg, &[]);
            // A workgroup a run, in rows as wide as a dispatch may be.
            let n = runs.len() as u32;
            let x = n.clamp(1, 32768);
            p.dispatch_workgroups(x, n.div_ceil(x).max(1), 1);
        }
        let end = (hi.min(self.warped_frames as u64)) * 8;
        enc.clear_buffer(self.warped.get(), lo * 8, Some(end.saturating_sub(lo * 8)));
        self.time_close(&mut enc, tm, 7);
        self.queue.submit([enc.finish()]);
        self.throttle((f1 - f0) * 8)
    }

    /// The window's end for the current key, the window starting at frame \[59\]
    pub fn end_window_into(&mut self, at: usize) -> Result<()> {
        let t = std::time::Instant::now();
        let win = self.win.as_ref().context("end of a window that never began")?;
        let (frames, stride, carry) = (win.frames, win.stride, win.carry);
        let list: Vec<u32> = self.active.iter().enumerate().filter(|(_, &a)| a).map(|(f, _)| f as u32).collect();
        self.master_at_least(at + frames)?;
        let d = self.device.clone();
        if list.is_empty() {
            return Ok(());
        }
        let chunks = frames / BQ_CHUNK;
        let (pages, kept) = (stride.div_ceil(PAGE), carry / PAGE);
        // [60]
        let (mut filter, mut ringing): (Vec<[u32; 2]>, Vec<[u32; 2]>) = (Vec::new(), Vec::new());
        let mut ranges: Vec<[u32; 2]> = Vec::with_capacity(list.len());
        let mut sounding = vec![false; list.len() * chunks];
        for (li, &f) in list.iter().enumerate() {
            let f = f as usize;
            let row = &self.cells[f * pages..(f + 1) * pages];
            let (ring, mut left) = (self.ring[f], self.ringing[f]);
            let lo = ringing.len() as u32;
            for page in 0..chunks {
                let cell = [li as u32, page as u32];
                if row[page] {
                    filter.push(cell);
                    left = ring;
                } else if left > 0 {
                    left -= 1;
                } else {
                    continue;
                }
                ringing.push(cell);
                sounding[li * chunks + page] = true;
            }
            self.ringing[f] = left;
            ranges.push([lo, ringing.len() as u32]);
        }
        // Each page's buses, ascending: the order a frame is summed in.
        let (mut items, mut buses): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        for page in 0..chunks {
            let lo = buses.len() as u32;
            buses.extend(list.iter().enumerate().filter(|(li, _)| sounding[li * chunks + page]).map(|(_, &f)| f));
            if buses.len() as u32 > lo {
                items.extend_from_slice(&[(page * PAGE) as u32, lo, buses.len() as u32]);
            }
        }
        // [61]
        let (mut moves, mut clears): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        for (li, &f) in list.iter().enumerate() {
            let row = &mut self.cells[f as usize * pages..(f as usize + 1) * pages];
            for page in 0..chunks {
                let from = page < kept && row[chunks + page];
                if from || sounding[li * chunks + page] {
                    moves.extend_from_slice(&[f, (page * PAGE) as u32, from as u32]);
                }
            }
            for page in 0..kept {
                if row[chunks + page] {
                    clears.extend_from_slice(&[f, (page * PAGE) as u32]);
                }
            }
            for page in 0..pages {
                row[page] = page < kept && row[chunks + page];
            }
        }
        let cells_bytes = (ringing.len() * BQ_CHUNK * 8) as u64;
        let upload = |q: &wgpu::Queue, g: &mut Grown, data: &[u8]| {
            g.at_least(&d, data.len() as u64);
            q.write_buffer(g.get(), 0, data);
        };
        upload(&self.queue, &mut self.alist, bytemuck::cast_slice(&list));
        upload(&self.queue, &mut self.qcells, bytemuck::cast_slice(&filter));
        upload(&self.queue, &mut self.fcells, bytemuck::cast_slice(&ringing));
        upload(&self.queue, &mut self.franges, bytemuck::cast_slice(&ranges));
        upload(&self.queue, &mut self.xitems, bytemuck::cast_slice(&items));
        upload(&self.queue, &mut self.xbuses, bytemuck::cast_slice(&buses));
        upload(&self.queue, &mut self.cmoves, bytemuck::cast_slice(&moves));
        upload(&self.queue, &mut self.cclears, bytemuck::cast_slice(&clears));
        let count = list.len() as u32;
        let (n_filter, n_ring, n_mix, n_move, n_clear) = (filter.len(), ringing.len(), items.len() / 3, moves.len() / 3, clears.len() / 2);
        self.queue.write_buffer(
            &self.u_e,
            0,
            bytemuck::cast_slice(&[frames as u32, stride as u32, count, carry as u32, at as u32, n_filter as u32, n_ring as u32, n_mix as u32, n_move as u32, n_clear as u32, 0, 0]),
        );
        let (coef, on, state) = (
            self.coef.as_ref().context("no coefficients")?,
            self.on.as_ref().context("no switches")?,
            self.state.as_ref().context("no state")?,
        );
        let states = (list.len() * chunks * 16) as u64;
        self.qchunk.at_least(&d, states);
        let qpow = self.qpow.as_ref().context("no filter powers")?;
        let bq = |cells: &wgpu::Buffer| {
            crate::gpu::device::bind(&d, &self.p.lq, &[&self.u_e, coef, on, state, self.bus.get(), self.alist.get(), self.qchunk.get(), qpow, cells, self.franges.get()])
        };
        let (bq_filter, bq_ring) = (bq(self.qcells.get()), bq(self.fcells.get()));
        let bx = crate::gpu::device::bind(&d, &self.p.lx, &[&self.u_e, self.bus.get(), self.xbuses.get(), self.master.get(), self.xitems.get()]);
        let bm = crate::gpu::device::bind(&d, &self.p.lc, &[&self.u_e, self.bus.get(), self.cmoves.get()]);
        let bc = crate::gpu::device::bind(&d, &self.p.lc, &[&self.u_e, self.bus.get(), self.cclears.get()]);
        let per_cell = |n: usize| {
            let x = (n as u32).clamp(1, 32768);
            (x, (n as u32).div_ceil(x).max(1))
        };
        let mut enc = d.create_command_encoder(&Default::default());
        let tm = self.time_mark(&mut enc);
        // Each state starts empty, and only a filtered cell leaves one.
        enc.clear_buffer(self.qchunk.get(), 0, Some(states));
        if n_filter > 0 {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.bq_chunk);
            p.set_bind_group(0, &bq_filter, &[]);
            p.dispatch_workgroups((n_filter as u32).div_ceil(64), 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.bq_scan);
            p.set_bind_group(0, &bq_ring, &[]);
            p.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        if n_ring > 0 {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.bq_fix);
            p.set_bind_group(0, &bq_ring, &[]);
            let (x, y) = per_cell(n_ring);
            p.dispatch_workgroups(x, y, 1);
        }
        if n_mix > 0 {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.mix);
            p.set_bind_group(0, &bx, &[]);
            let (x, y) = per_cell(n_mix);
            p.dispatch_workgroups(x, y, 1);
        }
        if n_move > 0 {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.carry_move);
            p.set_bind_group(0, &bm, &[]);
            let (x, y) = per_cell(n_move);
            p.dispatch_workgroups(x, y, 1);
        }
        if n_clear > 0 {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.p.carry_clear);
            p.set_bind_group(0, &bc, &[]);
            let (x, y) = per_cell(n_clear);
            p.dispatch_workgroups(x, y, 1);
        }
        self.time_close(&mut enc, tm, 7);
        self.queue.submit([enc.finish()]);
        self.prof.window += t.elapsed().as_secs_f64();
        // [62]
        for &f in &list {
            let q = &mut self.quiet[f as usize];
            *q += 1;
            if *q > QUIET_WINDOWS {
                self.active[f as usize] = false;
            }
        }
        // [63]
        self.throttle(cells_bytes)
    }

    /// The song's master, its first `frames`, back as interleaved stereo: \[64\]
    pub fn read_song(&mut self, frames: usize) -> Result<Vec<f32>> {
        let frames = frames.min(self.song_frames);
        if frames == 0 {
            return Ok(Vec::new());
        }
        let d = self.device.clone();
        let bytes = (frames * 8) as u64;
        self.stage.at_least(&d, bytes);
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(self.master.get(), 0, self.stage.get(), 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.stage.get().slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::PollType::wait_indefinitely()).context("waiting for key-major's master")?;
        rx.recv().context("the readback was dropped")??;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.stage.get().unmap();
        Ok(out)
    }
}

impl Packed {
    /// Host bytes it holds until `put`.
    pub fn held(&self) -> u64 {
        let v = |n: usize, w: usize| (n * w) as u64;
        self.ops
            .iter()
            .map(|o| match o {
                Op::Kernels { desc, .. } => v(desc.len(), 4),
                Op::Chunk { imps, entries, mdesc, parity, .. } => v(imps.len(), std::mem::size_of::<Imp>()) + v(entries.len() + mdesc.len() + parity[0].len() + parity[1].len(), 4),
                Op::Direct { dimps, items, desc, .. } => v(dimps.len(), std::mem::size_of::<Dimp>()) + v(items.len() + desc.len(), 4),
                Op::Stopped { dimps, items, desc, .. } => v(dimps.len(), std::mem::size_of::<Dstop>()) + v(items.len() + desc.len(), 4),
                Op::Heads { heads, items, list, .. } => v(heads.len(), std::mem::size_of::<Head>()) + v(items.len() + list.len(), 4),
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A biquad's state with nothing going in, `frames` on from each of the \[65\]
    fn left_after(c: [f32; 4], frames: usize) -> f64 {
        let (a1, a2) = (c[2] as f64, c[3] as f64);
        [(1.0f64, 0.0f64), (0.0, 1.0)]
            .iter()
            .map(|&(mut z1, mut z2)| {
                for _ in 0..frames {
                    let y = z1;
                    z1 = -a1 * y + z2;
                    z2 = -a2 * y;
                }
                z1.abs().max(z2.abs())
            })
            .fold(0.0, f64::max)
    }

    fn lowpass(fc: f32, q_cb: f32) -> [f32; 4] {
        let (b0, b1, a1, a2) = crate::bank::biquad_lowpass(fc, q_cb, 1.0);
        [b0, b1, a1, a2]
    }

    /// A filter's ring is long enough and not much more, from a low resonant \[66\]
    #[test]
    fn a_filters_ring_is_where_its_state_has_gone() {
        for (fc, q_cb) in [(20.0 / 48000.0, 0.0), (89.0 / 48000.0, 0.0), (1000.0 / 48000.0, 96.0), (1000.0 / 48000.0, 0.0), (0.4, 0.0)] {
            let c = lowpass(fc, q_cb);
            let ring = ring_chunks(&[Some(c)])[0];
            assert!(ring != u32::MAX, "{fc} at {q_cb} cB never settled");
            assert!(left_after(c, ring as usize * BQ_CHUNK) < 1e-8, "{fc} at {q_cb} cB after {ring} chunks");
            if ring > 1 {
                assert!(left_after(c, (ring as usize - 2) * BQ_CHUNK) > 1e-10, "{fc} at {q_cb} cB rang out well before {ring} chunks");
            }
        }
        assert_eq!(ring_chunks(&[None]), vec![0]);
        // Poles on the unit circle: a sine for ever.
        let w = 0.01f64;
        assert_eq!(ring_chunks(&[Some([0.0, 0.0, -(2.0 * w.cos()) as f32, 1.0])]), vec![u32::MAX]);
    }
}
