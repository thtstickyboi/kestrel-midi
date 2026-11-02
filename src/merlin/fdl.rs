// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The frequency-domain delay line. \[1\]

use crate::merlin::fft::{Fft, C32};
use crate::merlin::residency::{put_run, take_run};
use anyhow::{bail, Result};

/// One impulse: which row of the block's input it lands in, which frame, and \[2\]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Impulse {
    pub slot: u32,
    pub frame: u32,
    pub gain: f32,
    /// How many identical notes this impulse stands for (THE PLAN, Phase 1: \[3\]
    pub weight: u32,
}

impl Impulse {
    pub fn new(slot: u32, frame: u32, gain: f32) -> Self {
        Self::weighted(slot, frame, gain, 1)
    }

    pub fn weighted(slot: u32, frame: u32, gain: f32, weight: u32) -> Self {
        Self { slot, frame, gain, weight }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    fired: u32,
    lags: u32,
    bins: u32,
    slot0: u32,
    entries: u32,
    buses: u32,
    frames: u32,
    scale: f32,
    n_impulses: u32,
    _pad: [u32; 3],
}

/// Where [`Fdl::reserve_kernel`] put a kernel's pages: which cache buffer and \[4\]
#[derive(Clone, Copy, Debug)]
pub struct KernelSlot {
    buffer: usize,
    offset: u64,
    bytes: u64,
}

impl KernelSlot {
    /// Bytes the kernel occupies, which is how much a fill has to supply.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Cache buffers bound to the multiply-accumulate. \[5\]
pub const KBUFS: usize = 4;

pub struct Fdl {
    pub p: usize,
    pub lags: usize,
    pub entries: usize,
    pub buses: usize,
    wg: u32,
    fft: Fft,
    scatter: wgpu::ComputePipeline,
    to_real: wgpu::ComputePipeline,
    mac: wgpu::ComputePipeline,
    emit: wgpu::ComputePipeline,
    overlap: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,

    /// The kernel cache, packed. The buffers whose traffic matters. \[6\]
    pub kernels: [wgpu::Buffer; KBUFS],
    /// `[entries]` of `(offset in complex, partitions, buffer, _)`.
    kmeta: wgpu::Buffer,
    /// The same, host side, for the firing-list sort and the ordering check.
    meta: Vec<[u32; 4]>,
    /// Buses with rows this step, and the buffer they are staged in. The MAC's \[7\]
    active: Vec<u32>,
    active_buf: wgpu::Buffer,
    /// Free runs of pages in each cache buffer, ascending by start and \[8\]
    free: Vec<Vec<(u32, u32)>>,
    /// Pages per cache buffer.
    pages: usize,
    /// How many of the `KBUFS` slots are real; the rest are 8 bytes.
    bufs: usize,
    /// `[buses][K][bins]`, the accumulator.
    ring: wgpu::Buffer,
    /// `[buses][2P]`: emitted block, then carry.
    out: wgpu::Buffer,
    tin: wgpu::Buffer,
    data: wgpu::Buffer,
    spec: wgpu::Buffer,
    fired: wgpu::Buffer,
    bus_range: wgpu::Buffer,
    impulses: wgpu::Buffer,
    max_fired: usize,
    max_impulses: usize,
    /// Impulses this device binds at once, and how many steps have needed more \[9\]
    impulse_chunk: usize,
    impulse_chunks: u64,
    block: u64,
}

impl Fdl {
    /// `max_fired` and `max_impulses` size the per-block staging buffers. They \[10\]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        limits: &wgpu::Limits,
        p: usize,
        lags: usize,
        entries: usize,
        buses: usize,
        max_fired: usize,
        max_impulses: usize,
        pool_bytes: u64,
        chunk_cap: u64,
    ) -> Result<Self> {
        if lags == 0 || entries == 0 || buses == 0 {
            bail!("fdl needs at least one lag, entry and bus");
        }
        // [11]
        if max_fired == 0 {
            bail!("max_fired must be at least 1");
        }
        let fft = Fft::new(device, limits, p, 256)?;
        let bins = fft.spec_len();

        // [12]
        let ring_bytes =
            buses as u64 * lags as u64 * bins as u64 * std::mem::size_of::<C32>() as u64;
        let bind_cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if ring_bytes > bind_cap {
            bail!(
                "{buses} buses at {lags} partitions need {:.2} GiB of accumulator ring, and a \
                 storage binding holds {:.2} GiB. The ring is `buses * lags * bins * 8 B` and \
                 it is state, not a cache -- it cannot be evicted, because it holds the tail \
                 every entry has already scattered into it. Lower --merlin-filters (buses are \
                 twice that), or shorten --merlin-kernel-ms to cut the partition count.",
                ring_bytes as f64 / 1073741824.0,
                bind_cap as f64 / 1073741824.0,
            );
        }

        // [13]
        let cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let c32 = std::mem::size_of::<C32>() as u64;
        // [14]
        let rect = entries as u64 * lags as u64 * bins as u64 * c32;
        let one = lags as u64 * bins as u64 * c32;
        let want = if pool_bytes == 0 { rect } else { pool_bytes }
            .max(one)
            .min(cap * KBUFS as u64);
        if one > cap {
            bail!(
                "one kernel needs {:.2} GiB at {lags} partitions of {bins} bins, but a storage \
                 binding holds {:.2} GiB. Shorten --merlin-kernel-ms.",
                one as f64 / 1073741824.0,
                cap as f64 / 1073741824.0,
            );
        }
        // [15]
        let bufs = want.div_ceil(cap).max(1) as usize;
        let per = (want.div_ceil(bufs as u64).max(one) / c32) * c32;
        let stride = (per / c32) as usize;
        let wg = 256u32.min(limits.max_compute_invocations_per_workgroup);
        // [16]
        let impulse_chunk = (((if chunk_cap > 0 { chunk_cap.min(bind_cap) } else { bind_cap })
            / std::mem::size_of::<Impulse>() as u64) as usize
            / wg as usize
            * wg as usize)
            .max(wg as usize);

        let src = include_str!("../../shaders/fdl.wgsl").replace("{{FDL_WG}}", &wg.to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("merlin::fdl"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        // [17]
        let mut ro = vec![true; 6];
        ro.extend([false; 5]);
        ro.extend([true; KBUFS - 1]);
        ro.push(true); // fdl::active
        let layout = crate::gpu::device::bind_layout(device, "merlin::fdl", &ro);
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("merlin::fdl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("merlin::fdl::{entry}")),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let stor = wgpu::BufferUsages::STORAGE;
        let buf = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage: stor | extra,
                mapped_at_creation: false,
            })
        };

        Ok(Self {
            p,
            lags,
            entries,
            buses,
            wg,
            fft,
            scatter: pipe("scatter"),
            to_real: pipe("to_real"),
            mac: pipe("mac"),
            emit: pipe("emit"),
            overlap: pipe("overlap"),
            layout,
            uniform: buf("fdl::u", 48, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            kernels: std::array::from_fn(|i| {
                let bytes = if i < bufs { per } else { 8 };
                buf(&format!("fdl::kernels{i}"), bytes, wgpu::BufferUsages::COPY_DST)
            }),
            kmeta: buf("fdl::kmeta", (entries * 16) as u64, wgpu::BufferUsages::COPY_DST),
            meta: vec![[0; 4]; entries],
            active: Vec::with_capacity(buses),
            active_buf: buf("fdl::active", (buses * 4) as u64, wgpu::BufferUsages::COPY_DST),
            free: (0..bufs).map(|_| vec![(0u32, (stride / bins) as u32)]).collect(),
            pages: stride / bins,
            bufs,
            // COPY_SRC because `grow_buses` copies it into a wider one.
            ring: buf(
                "fdl::ring",
                (buses * lags * bins) as u64 * c32,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            out: buf(
                "fdl::out",
                (buses * 2 * p) as u64 * 4,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            tin: buf("fdl::tin", (max_fired * 2 * p) as u64 * 4, wgpu::BufferUsages::COPY_DST),
            data: buf("fdl::data", (max_fired.max(buses) * 2 * p) as u64 * 4, stor),
            spec: buf("fdl::spec", (max_fired.max(buses) * bins) as u64 * c32, stor),
            fired: buf("fdl::fired", max_fired as u64 * 4, wgpu::BufferUsages::COPY_DST),
            bus_range: buf("fdl::bus_range", (buses * 2) as u64 * 4, wgpu::BufferUsages::COPY_DST),
            impulses: buf(
                "fdl::impulses",
                (max_impulses.min(impulse_chunk) * std::mem::size_of::<Impulse>()) as u64,
                wgpu::BufferUsages::COPY_DST,
            ),
            max_fired,
            max_impulses: max_impulses.min(impulse_chunk),
            impulse_chunk,
            impulse_chunks: 0,
            block: 0,
        })
    }

    /// Steps whose impulses outgrew one storage binding, and the ceiling they \[18\]
    pub fn impulse_chunking(&self) -> (u64, usize) {
        (self.impulse_chunks, self.impulse_chunk)
    }

    /// The `[buses][2P]` output buffer, so a later pass can add into the block \[19\]
    pub fn out_buffer(&self) -> &wgpu::Buffer {
        &self.out
    }

    pub fn bins(&self) -> usize {
        self.fft.spec_len()
    }

    /// Install kernel `e`'s transformed partitions and record where they went. \[20\]
    pub fn upload_kernel(
        &mut self,
        queue: &wgpu::Queue,
        e: usize,
        spectra: &[C32],
    ) -> Result<bool> {
        let bins = self.bins();
        if spectra.is_empty() || !spectra.len().is_multiple_of(bins) {
            bail!("kernel {e} has {} complex, want a multiple of {bins}", spectra.len());
        }
        let depth = spectra.len() / bins;
        self.check_kernel(e, depth)?;
        if self.meta[e][1] != 0 {
            // [21]
            return Ok(true);
        }
        let Some(slot) = self.place_kernel(queue, e, depth) else {
            return Ok(false);
        };
        queue.write_buffer(&self.kernels[slot.buffer], slot.offset, bytemuck::cast_slice(spectra));
        Ok(true)
    }

    /// The first half of [`Fdl::upload_kernel`]: allocate `depth` pages for \[22\]
    pub fn reserve_kernel(
        &mut self,
        queue: &wgpu::Queue,
        e: usize,
        depth: usize,
    ) -> Result<Option<KernelSlot>> {
        self.check_kernel(e, depth)?;
        if self.meta[e][1] != 0 {
            bail!("kernel {e} is already resident, so there is nothing to reserve");
        }
        Ok(self.place_kernel(queue, e, depth))
    }

    fn check_kernel(&self, e: usize, depth: usize) -> Result<()> {
        if depth == 0 {
            bail!("kernel {e} has no partitions to place");
        }
        if depth > self.lags {
            bail!(
                "kernel {e} is {depth} partitions but the ring holds {}. Raise \
                 --merlin-kernel-ms.",
                self.lags
            );
        }
        if e >= self.entries {
            bail!("kernel index {e} beyond {} entries", self.entries);
        }
        Ok(())
    }

    fn place_kernel(&mut self, queue: &wgpu::Queue, e: usize, depth: usize) -> Option<KernelSlot> {
        let bins = self.bins();
        let (b, page) = self.take_pages(depth as u32)?;
        let off = page as usize * bins;
        self.meta[e] = [off as u32, depth as u32, b as u32, 0];
        // [23]
        queue.write_buffer(&self.kmeta, (e * 16) as u64, bytemuck::cast_slice(&self.meta[e]));
        let c32 = std::mem::size_of::<C32>() as u64;
        Some(KernelSlot {
            buffer: b,
            offset: off as u64 * c32,
            bytes: (depth * bins) as u64 * c32,
        })
    }

    /// The second half: record a copy of `slot.bytes()` bytes of device memory \[24\]
    pub fn copy_kernel(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        src_offset: u64,
        slot: &KernelSlot,
    ) {
        enc.copy_buffer_to_buffer(
            src,
            src_offset,
            &self.kernels[slot.buffer],
            slot.offset,
            slot.bytes,
        );
    }

    /// Give entry `e`'s pages back. Its `kmeta` row is left as it was: nothing \[25\]
    pub fn evict_kernel(&mut self, e: usize) {
        let m = self.meta[e];
        if m[1] == 0 {
            return;
        }
        self.put_pages(m[2] as usize, m[0] / self.bins() as u32, m[1]);
        self.meta[e] = [0; 4];
    }

    fn take_pages(&mut self, want: u32) -> Option<(usize, u32)> {
        take_run(&mut self.free, want)
    }

    fn put_pages(&mut self, b: usize, start: u32, len: u32) {
        put_run(&mut self.free[b], start, len);
    }

    /// Pages held, and the pool's capacity in pages.
    pub fn pages_used(&self) -> (usize, usize) {
        let free: usize =
            self.free.iter().flatten().map(|&(_, len)| len as usize).sum();
        (self.bufs * self.pages - free, self.bufs * self.pages)
    }

    /// Complex a kernel of `depth` partitions occupies, for the caller's budget \[26\]
    pub fn kernel_complex(&self, depth: usize) -> usize {
        depth * self.bins()
    }

    /// Partitions held for each entry, indexed by entry id. Zero until the \[27\]
    pub fn depths(&self) -> impl Fn(u32) -> u32 + '_ {
        move |e: u32| self.meta.get(e as usize).map_or(0, |m| m[1])
    }

    /// Make the per-step staging buffers big enough for this step. \[28\]
    fn grow(&mut self, device: &wgpu::Device, fired: usize, impulses: usize) {
        let stor = wgpu::BufferUsages::STORAGE;
        let buf = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage: stor | extra,
                mapped_at_creation: false,
            })
        };
        if fired > self.max_fired {
            // [29]
            let n = fired.next_power_of_two().max(self.max_fired * 2);
            let (p, bins) = (self.p, self.bins());
            let c32 = std::mem::size_of::<C32>() as u64;
            log::info!("merlin: firing buffers {} -> {n} rows", self.max_fired);
            self.tin = buf("fdl::tin", (n * 2 * p) as u64 * 4, wgpu::BufferUsages::COPY_DST);
            self.data = buf("fdl::data", (n.max(self.buses) * 2 * p) as u64 * 4, stor);
            self.spec = buf("fdl::spec", (n.max(self.buses) * bins) as u64 * c32, stor);
            self.fired = buf("fdl::fired", n as u64 * 4, wgpu::BufferUsages::COPY_DST);
            self.max_fired = n;
        }
        // [30]
        if impulses.min(self.impulse_chunk) > self.max_impulses {
            let n = impulses
                .next_power_of_two()
                .max(self.max_impulses * 2)
                .min(self.impulse_chunk);
            log::info!("merlin: impulse buffer {} -> {n}", self.max_impulses);
            self.impulses = buf(
                "fdl::impulses",
                (n * std::mem::size_of::<Impulse>()) as u64,
                wgpu::BufferUsages::COPY_DST,
            );
            self.max_impulses = n;
        }
    }

    /// Widen the bus dimension to `buses`, keeping everything already in it. \[31\]
    pub fn grow_buses(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, buses: usize) {
        if buses <= self.buses {
            return;
        }
        let stor = wgpu::BufferUsages::STORAGE;
        let buf = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage: stor | extra,
                mapped_at_creation: false,
            })
        };
        let (p, lags, bins) = (self.p, self.lags, self.bins());
        let c32 = std::mem::size_of::<C32>() as u64;
        let copy = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let ring = buf("fdl::ring", (buses * lags * bins) as u64 * c32, copy);
        let out = buf("fdl::out", (buses * 2 * p) as u64 * 4, copy);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("merlin::fdl::grow_buses"),
        });
        enc.copy_buffer_to_buffer(&self.ring, 0, &ring, 0, (self.buses * lags * bins) as u64 * c32);
        enc.copy_buffer_to_buffer(&self.out, 0, &out, 0, (self.buses * 2 * p) as u64 * 4);
        queue.submit([enc.finish()]);
        self.ring = ring;
        self.out = out;
        self.active_buf = buf("fdl::active", (buses * 4) as u64, wgpu::BufferUsages::COPY_DST);
        self.bus_range =
            buf("fdl::bus_range", (buses * 2) as u64 * 4, wgpu::BufferUsages::COPY_DST);
        // Sized `max(max_fired, buses)` rows, and scratch, so only the room.
        if buses > self.max_fired {
            self.data = buf("fdl::data", (buses * 2 * p) as u64 * 4, stor);
            self.spec = buf("fdl::spec", (buses * bins) as u64 * c32, stor);
        }
        self.buses = buses;
    }

    /// Complex currently allocated, and the pool's capacity.
    pub fn cache_used(&self) -> (usize, usize) {
        let (used, cap) = self.pages_used();
        (used * self.bins(), cap * self.bins())
    }

    /// Run one block. \[32\]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_block(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        fired: &[u32],
        bus_range: &[u32],
        impulses: &[Impulse],
        live: usize,
    ) -> Result<()> {
        self.grow(device, fired.len(), impulses.len());
        if bus_range.len() != self.buses * 2 {
            bail!("bus_range must be {} words", self.buses * 2);
        }
        // [33]
        let depth = |e: u32| self.meta.get(e as usize).map_or(0, |m| m[1]);
        for bus in 0..self.buses {
            let (lo, hi) = (bus_range[bus * 2] as usize, bus_range[bus * 2 + 1] as usize);
            if lo > hi || hi > fired.len() {
                bail!("bus {bus} range [{lo}, {hi}) is not inside a {}-entry list", fired.len());
            }
            let ordered = |a: u32, b: u32| depth(a) > depth(b) || (depth(a) == depth(b) && a < b);
            if fired[lo..hi].windows(2).any(|w| !ordered(w[0], w[1])) {
                bail!(
                    "bus {bus}'s firing range must run deepest kernel first, ascending by \
                     entry within a depth; the MAC's early exit and its determinism both \
                     depend on it"
                );
            }
        }

        // [34]
        let notes: u64 = impulses.iter().map(|i| i.weight as u64).sum();
        let scale = (i32::MAX as f32 / (notes.max(1) as f32)).min(1.0e6);
        let u = Uniforms {
            fired: fired.len() as u32,
            lags: self.lags as u32,
            bins: self.bins() as u32,
            slot0: (self.block % self.lags as u64) as u32,
            entries: self.entries as u32,
            buses: self.buses as u32,
            frames: self.p as u32,
            scale,
            n_impulses: impulses.len() as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        queue.write_buffer(&self.fired, 0, bytemuck::cast_slice(fired));
        queue.write_buffer(&self.bus_range, 0, bytemuck::cast_slice(bus_range));
        if !impulses.is_empty() && impulses.len() <= self.max_impulses {
            queue.write_buffer(&self.impulses, 0, bytemuck::cast_slice(impulses));
        }

        let mut bufs: Vec<&wgpu::Buffer> = vec![
            &self.uniform,
            &self.kernels[0],
            &self.kmeta,
            &self.fired,
            &self.bus_range,
            &self.impulses,
            &self.tin,
            &self.data,
            &self.spec,
            &self.ring,
            &self.out,
        ];
        bufs.extend(self.kernels[1..].iter());
        bufs.push(&self.active_buf);
        let bind = crate::gpu::device::bind(device, &self.layout, &bufs);
        // Separate bind groups per direction: see the note on `Fft::uniform`.
        let fwd_bind = self.fft.bind_real(device, &self.data, &self.spec, false);
        let inv_bind = self.fft.bind_real(device, &self.data, &self.spec, true);
        let n = fired.len() as u32;
        let ceil = |a: u32| a.div_ceil(self.wg);

        // [35]
        if impulses.len() <= self.max_impulses {
            self.pass(
                enc,
                &bind,
                &self.scatter,
                (ceil(impulses.len() as u32).min(65535), 1, 1),
            );
        } else {
            self.impulse_chunks += 1;
            log::info!(
                "merlin: {} impulses this step, past the {} one binding holds -- scattering \
                 in {} passes",
                impulses.len(),
                self.impulse_chunk,
                impulses.len().div_ceil(self.max_impulses),
            );
            for part in impulses.chunks(self.max_impulses) {
                queue.write_buffer(&self.impulses, 0, bytemuck::cast_slice(part));
                let mut cu = u;
                cu.n_impulses = part.len() as u32;
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&cu));
                let mut e = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("merlin::fdl::scatter_chunk"),
                });
                self.pass(&mut e, &bind, &self.scatter, (ceil(part.len() as u32).min(65535), 1, 1));
                queue.submit([e.finish()]);
            }
            // [36]
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        }
        self.pass(enc, &bind, &self.to_real, (ceil(n * 2 * self.p as u32).min(65535), 1, 1));
        if n > 0 {
            self.fft.encode_r2c(queue, enc, &fwd_bind, n);
        }
        // [37]
        self.active.clear();
        let mut deepest = 0u32;
        for bus in 0..live {
            let (lo, hi) = (bus_range[bus * 2], bus_range[bus * 2 + 1]);
            if hi > lo {
                self.active.push(bus as u32);
                deepest = deepest.max(depth(fired[lo as usize]));
            }
        }
        if !self.active.is_empty() {
            queue.write_buffer(&self.active_buf, 0, bytemuck::cast_slice(&self.active));
            self.pass(
                enc,
                &bind,
                &self.mac,
                (ceil(self.bins() as u32), deepest, self.active.len() as u32),
            );
        }
        let live = live as u32;
        self.pass(enc, &bind, &self.emit, (ceil(self.bins() as u32), live, 1));
        self.fft.encode_c2r(queue, enc, &inv_bind, live);
        self.pass(enc, &bind, &self.overlap, (ceil(self.p as u32), live, 1));

        self.block += 1;
        Ok(())
    }

    fn pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        bind: &wgpu::BindGroup,
        pipeline: &wgpu::ComputePipeline,
        grid: (u32, u32, u32),
    ) {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return;
        }
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        p.set_pipeline(pipeline);
        p.set_bind_group(0, bind, &[]);
        p.dispatch_workgroups(grid.0, grid.1, grid.2);
    }

    /// Copy out the block just rendered: `buses * P` frames, bus-major.
    pub fn read_block(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<f32>> {
        let bytes = (self.buses * 2 * self.p * 4) as u64;
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdl::read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.out, 0, &dst, 0, bytes);
        queue.submit([enc.finish()]);
        let slice = dst.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let all: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        dst.unmap();
        // Drop the carry half of each bus.
        Ok((0..self.buses)
            .flat_map(|b| all[b * 2 * self.p..b * 2 * self.p + self.p].to_vec())
            .collect())
    }
}
