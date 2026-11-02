// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Merlin M1': the batched complex FFT. \[1\]

use anyhow::{bail, Result};
use wgpu::util::DeviceExt;

/// A complex sample, matching WGSL's `vec2<f32>`.
pub type C32 = [f32; 2];

pub struct Fft {
    n: usize,
    forward: wgpu::ComputePipeline,
    inverse: wgpu::ComputePipeline,
    r2c: wgpu::ComputePipeline,
    c2r: wgpu::ComputePipeline,
    /// Three bindings: the complex entry points never touch `spec`, so they \[2\]
    layout: wgpu::BindGroupLayout,
    /// Four bindings, for the real entry points.
    real_layout: wgpu::BindGroupLayout,
    twiddles: wgpu::Buffer,
    /// One uniform per direction, and that is not tidiness. \[3\]
    uniform: [wgpu::Buffer; 2],
    /// A third uniform that is **written once, at creation, and never again**: \[4\]
    exact: wgpu::Buffer,
}

impl Fft {
    /// `n` must be a power of two of at least 4. `wg` is the workgroup size. \[5\]
    pub fn new(device: &wgpu::Device, limits: &wgpu::Limits, n: usize, wg: u32) -> Result<Self> {
        if n < 4 || !n.is_power_of_two() {
            bail!("fft length {n} must be a power of two of at least 4");
        }
        let need = (2 * n * std::mem::size_of::<C32>()) as u32;
        if need > limits.max_compute_workgroup_storage_size {
            bail!(
                "fft n={n} needs {need} B of workgroup storage, device allows {}. \
                 Halve the partition size.",
                limits.max_compute_workgroup_storage_size
            );
        }
        if wg > limits.max_compute_invocations_per_workgroup {
            bail!(
                "fft workgroup {wg} exceeds the device's {} invocations",
                limits.max_compute_invocations_per_workgroup
            );
        }

        let log2n = n.trailing_zeros();
        let src = include_str!("../../shaders/fft.wgsl")
            .replace("{{FFT_N}}", &n.to_string())
            .replace("{{FFT_2N}}", &(2 * n).to_string())
            .replace("{{FFT_LOG2N}}", &log2n.to_string())
            .replace("{{FFT_WG}}", &wg.to_string());

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("merlin::fft"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        let layout = crate::gpu::device::bind_layout(device, "merlin::fft", &[true, true, false]);
        let real_layout =
            crate::gpu::device::bind_layout(device, "merlin::fft::real", &[true, true, false, false]);
        let mk_pl = |l: &wgpu::BindGroupLayout, name: &str| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[l],
                push_constant_ranges: &[],
            })
        };
        let pl = mk_pl(&layout, "merlin::fft");
        let real_pl = mk_pl(&real_layout, "merlin::fft::real");
        let pipe = |entry: &str, l: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("merlin::fft::{entry}")),
                layout: Some(l),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // [6]
        let mut tw = Vec::with_capacity(2 * n);
        for k in 0..2 * n {
            let th = -std::f64::consts::PI * k as f64 / n as f64;
            tw.push([th.cos() as f32, th.sin() as f32]);
        }
        let twiddles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("merlin::fft::twiddles"),
            contents: bytemuck::cast_slice(&tw),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mk_u = |i: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if i == 0 { "merlin::fft::u_fwd" } else { "merlin::fft::u_inv" }),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniform = [mk_u(0), mk_u(1)];
        let exact = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("merlin::fft::u_exact"),
            contents: bytemuck::cast_slice(&[u32::MAX, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        Ok(Self {
            n,
            forward: pipe("forward", &pl),
            inverse: pipe("inverse", &pl),
            r2c: pipe("r2c", &real_pl),
            c2r: pipe("c2r", &real_pl),
            layout,
            real_layout,
            twiddles,
            uniform,
            exact,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Bind a `[batch][n]` complex buffer for transforming in place. \[7\]
    pub fn bind(
        &self,
        device: &wgpu::Device,
        data: &wgpu::Buffer,
        inverse: bool,
    ) -> wgpu::BindGroup {
        let u = &self.uniform[inverse as usize];
        crate::gpu::device::bind(device, &self.layout, &[u, &self.twiddles, data])
    }

    /// Record one batched transform. The caller owns the encoder so this can \[8\]
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        bind: &wgpu::BindGroup,
        batch: u32,
        inverse: bool,
    ) {
        queue.write_buffer(
            &self.uniform[inverse as usize],
            0,
            bytemuck::cast_slice(&[batch, 0, 0, 0]),
        );
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(if inverse { "fft::inverse" } else { "fft::forward" }),
            timestamp_writes: None,
        });
        p.set_pipeline(if inverse { &self.inverse } else { &self.forward });
        p.set_bind_group(0, bind, &[]);
        p.dispatch_workgroups(batch, 1, 1);
    }

    /// Bind a `[batch][n]` complex buffer -- or a slice of one -- for \[9\]
    pub fn bind_exact(&self, device: &wgpu::Device, data: wgpu::BufferBinding) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("merlin::fft::exact"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.exact.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.twiddles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(data) },
            ],
        })
    }

    /// Exactly `batch` complex transforms in place, with **no uniform write**: \[10\]
    pub fn encode_exact(&self, enc: &mut wgpu::CommandEncoder, bind: &wgpu::BindGroup, batch: u32, inverse: bool) {
        let pipeline = if inverse { &self.inverse } else { &self.forward };
        self.pass(enc, bind, batch, pipeline, if inverse { "fft::inverse_exact" } else { "fft::forward_exact" });
    }

    /// Bins a real transform produces: `n + 1`, not `n`. \[11\]
    pub fn spec_len(&self) -> usize {
        self.n + 1
    }

    /// Bind for the real entry points. `data` is `[batch][n]` complex, which \[12\]
    pub fn bind_real(
        &self,
        device: &wgpu::Device,
        data: &wgpu::Buffer,
        spec: &wgpu::Buffer,
        inverse: bool,
    ) -> wgpu::BindGroup {
        let u = &self.uniform[inverse as usize];
        crate::gpu::device::bind(device, &self.real_layout, &[u, &self.twiddles, data, spec])
    }

    /// `2n` reals in `data` -> `n+1` complex bins in `spec`. \[13\]
    pub fn encode_r2c(
        &self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        real_bind: &wgpu::BindGroup,
        batch: u32,
    ) {
        queue.write_buffer(&self.uniform[0], 0, bytemuck::cast_slice(&[batch, 0, 0, 0]));
        self.pass(enc, real_bind, batch, &self.r2c, "fft::r2c");
    }

    /// Bind for [`Fft::encode_r2c_exact`]: the same layout as \[14\]
    pub fn bind_real_exact(
        &self,
        device: &wgpu::Device,
        data: &wgpu::Buffer,
        spec: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        crate::gpu::device::bind(device, &self.real_layout, &[&self.exact, &self.twiddles, data, spec])
    }

    /// The forward real transform of exactly `batch` rows, with **no uniform \[15\]
    pub fn encode_r2c_exact(
        &self,
        enc: &mut wgpu::CommandEncoder,
        real_bind: &wgpu::BindGroup,
        batch: u32,
    ) {
        self.pass(enc, real_bind, batch, &self.r2c, "fft::r2c_exact");
    }

    /// `n+1` complex bins in `spec` -> `2n` reals in `data`. Also one dispatch.
    pub fn encode_c2r(
        &self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        real_bind: &wgpu::BindGroup,
        batch: u32,
    ) {
        queue.write_buffer(&self.uniform[1], 0, bytemuck::cast_slice(&[batch, 0, 0, 0]));
        self.pass(enc, real_bind, batch, &self.c2r, "fft::c2r");
    }

    fn pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        bind: &wgpu::BindGroup,
        batch: u32,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
    ) {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        p.set_pipeline(pipeline);
        p.set_bind_group(0, bind, &[]);
        p.dispatch_workgroups(batch, 1, 1);
    }
}

/// Upload, transform, read back. For tests and for the M1' timing harness -- \[16\]
pub fn run_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fft: &Fft,
    data: &mut [C32],
    inverse: bool,
) -> Result<()> {
    let n = fft.n();
    if !data.len().is_multiple_of(n) {
        bail!("data length {} is not a multiple of n={n}", data.len());
    }
    let batch = (data.len() / n) as u32;
    let bytes = std::mem::size_of_val(data) as u64;

    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("merlin::fft::data"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merlin::fft::read"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind = fft.bind(device, &buf, inverse);
    let mut enc = device.create_command_encoder(&Default::default());
    fft.encode(queue, &mut enc, &bind, batch, inverse);
    enc.copy_buffer_to_buffer(&buf, 0, &read, 0, bytes);
    queue.submit([enc.finish()]);

    let slice = read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    data.copy_from_slice(bytemuck::cast_slice(&slice.get_mapped_range()));
    read.unmap();
    Ok(())
}

/// Upload `2n * batch` reals, forward-transform, read back `(n+1) * batch` \[17\]
pub fn r2c_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fft: &Fft,
    reals: &[f32],
) -> Result<Vec<C32>> {
    let n = fft.n();
    if !reals.len().is_multiple_of(2 * n) {
        bail!("input length {} is not a multiple of 2n={}", reals.len(), 2 * n);
    }
    let batch = reals.len() / (2 * n);
    let spec_len = fft.spec_len() * batch;

    let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("merlin::fft::r2c::data"),
        contents: bytemuck::cast_slice(reals),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let spec = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merlin::fft::r2c::spec"),
        size: (spec_len * std::mem::size_of::<C32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    fft.encode_r2c(queue, &mut enc, &fft.bind_real(device, &data, &spec, false), batch as u32);
    Ok(bytemuck::cast_slice(&readback(device, queue, enc, &spec)?).to_vec())
}

/// The other direction: `(n+1) * batch` bins in, `2n * batch` reals out.
pub fn c2r_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fft: &Fft,
    bins: &[C32],
) -> Result<Vec<f32>> {
    let n = fft.n();
    if !bins.len().is_multiple_of(fft.spec_len()) {
        bail!("input length {} is not a multiple of n+1={}", bins.len(), fft.spec_len());
    }
    let batch = bins.len() / fft.spec_len();

    let data = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merlin::fft::c2r::data"),
        size: (batch * n * std::mem::size_of::<C32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let spec = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("merlin::fft::c2r::spec"),
        contents: bytemuck::cast_slice(bins),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    fft.encode_c2r(queue, &mut enc, &fft.bind_real(device, &data, &spec, true), batch as u32);
    Ok(bytemuck::cast_slice(&readback(device, queue, enc, &data)?).to_vec())
}

/// Persistent buffers for forward-transforming many rows in one dispatch and \[18\]
pub struct R2cBatch {
    n: usize,
    rows: usize,
    max_rows: usize,
    data: wgpu::Buffer,
    spec: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl R2cBatch {
    /// Rows the buffers start with: 2 MiB each at `n = 2048`.
    const START_ROWS: usize = 64;

    /// `max_rows` is clamped to what the device can dispatch in one dimension \[19\]
    pub fn new(device: &wgpu::Device, fft: &Fft, max_rows: usize) -> Self {
        let n = fft.n();
        let limits = device.limits();
        let row_bytes = (2 * n * 4).max(fft.spec_len() * std::mem::size_of::<C32>()) as u64;
        let by_binding = (limits.max_storage_buffer_binding_size as u64)
            .min(limits.max_buffer_size)
            / row_bytes;
        let max_rows = max_rows
            .min(limits.max_compute_workgroups_per_dimension as usize)
            .min(by_binding as usize)
            .max(1);
        let rows = Self::START_ROWS.min(max_rows);
        let (data, spec, bind) = Self::alloc(device, fft, rows);
        Self { n, rows, max_rows, data, spec, bind }
    }

    fn alloc(
        device: &wgpu::Device,
        fft: &Fft,
        rows: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        let n = fft.n();
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merlin::fft::batch::data"),
            size: (rows * 2 * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spec = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merlin::fft::batch::spec"),
            size: (rows * fft.spec_len() * std::mem::size_of::<C32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = fft.bind_real_exact(device, &data, &spec);
        (data, spec, bind)
    }

    /// The most rows one [`R2cBatch::transform`] call takes.
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Rows the buffers currently hold, which only ever grows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Row `i`'s spectrum is bins `i * (n+1)` onward of this buffer, after the \[20\]
    pub fn spectra(&self) -> &wgpu::Buffer {
        &self.spec
    }

    /// Stage `rows * 2n` reals through `fill` and record their forward \[21\]
    pub fn transform(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fft: &Fft,
        enc: &mut wgpu::CommandEncoder,
        rows: usize,
        fill: impl FnOnce(&mut [f32]) -> Result<()>,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if rows > self.max_rows {
            bail!("a batch of {rows} rows is past the {} this device takes at once", self.max_rows);
        }
        if rows > self.rows {
            // [22]
            let want = (self.rows * 2).max(rows).min(self.max_rows);
            log::info!(
                "merlin: kernel batch buffers {} -> {want} partitions ({:.0} MiB)",
                self.rows,
                (want * (2 * self.n * 4 + fft.spec_len() * std::mem::size_of::<C32>())) as f64
                    / 1048576.0,
            );
            let (data, spec, bind) = Self::alloc(device, fft, want);
            self.data = data;
            self.spec = spec;
            self.bind = bind;
            self.rows = want;
        }
        let len = rows * 2 * self.n;
        let Some(mut view) = wgpu::BufferSize::new(len as u64 * 4)
            .and_then(|size| queue.write_buffer_with(&self.data, 0, size))
        else {
            bail!("could not stage {len} reals for a batched transform");
        };
        match bytemuck::try_cast_slice_mut::<u8, f32>(&mut view[..]) {
            Ok(reals) => fill(reals)?,
            Err(_) => {
                // [23]
                let mut tmp = vec![0.0f32; len];
                fill(&mut tmp)?;
                view[..].copy_from_slice(bytemuck::cast_slice(&tmp));
            }
        }
        drop(view);
        fft.encode_r2c_exact(enc, &self.bind, rows as u32);
        Ok(())
    }
}

fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut enc: wgpu::CommandEncoder,
    src: &wgpu::Buffer,
) -> Result<Vec<u8>> {
    let bytes = src.size();
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merlin::fft::readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(src, 0, &dst, 0, bytes);
    queue.submit([enc.finish()]);

    let slice = dst.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let out = slice.get_mapped_range().to_vec();
    dst.unmap();
    Ok(out)
}
