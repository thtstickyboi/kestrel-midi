// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Adapter selection, device creation and bind-group boilerplate.

use crate::config::Config;
use anyhow::{bail, Result};

fn parse_backends(name: &str) -> Option<wgpu::Backends> {
    match name.to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => Some(wgpu::Backends::VULKAN),
        "dx12" | "d3d12" => Some(wgpu::Backends::DX12),
        "metal" => Some(wgpu::Backends::METAL),
        "gl" | "opengl" => Some(wgpu::Backends::GL),
        "all" => Some(wgpu::Backends::all()),
        _ => None,
    }
}

/// Discrete first, then integrated. Within a tier, prefer Vulkan over DX12 \[1\]
fn rank(i: &wgpu::AdapterInfo) -> (u8, u8) {
    let t = match i.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        _ => 3,
    };
    let b = match i.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 0,
        wgpu::Backend::Dx12 => 1,
        _ => 2,
    };
    (t, b)
}

/// Pick an adapter and open a device. \[2\]
pub fn create(
    cfg: &Config,
) -> Result<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo, wgpu::Limits, bool)> {
    let backends = match &cfg.gpu_backend {
        Some(b) => parse_backends(b)
            .ok_or_else(|| anyhow::anyhow!("unknown gpu backend {b:?}"))?,
        None => wgpu::Backends::all(),
    };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let mut candidates: Vec<wgpu::Adapter> = instance
        .enumerate_adapters(backends)
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect();

    if let Some(want) = &cfg.gpu_adapter {
        let want = want.to_ascii_lowercase();
        candidates.retain(|a| a.get_info().name.to_ascii_lowercase().contains(&want));
        if candidates.is_empty() {
            bail!("no gpu adapter matched {want:?}");
        }
    }

    candidates.sort_by_key(|a| rank(&a.get_info()));

    let adapter = candidates.into_iter().next().ok_or_else(|| {
        anyhow::anyhow!(
            "no usable gpu found. wgpu saw no non-software adapter; \
             install or enable a graphics driver, or render with --backend cpu"
        )
    })?;

    let info = adapter.get_info();
    let adapter_limits = adapter.limits();

    let has_timestamps = adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut features = wgpu::Features::empty();
    if has_timestamps && cfg.profile {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kestrel"),
        required_features: features,
        required_limits: adapter_limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))?;

    // [3]
    device.on_uncaptured_error(std::sync::Arc::new(|e| {
        log::error!("wgpu device error: {e}");
        panic!("wgpu device error: {e}");
    }));

    Ok((device, queue, info, adapter_limits, has_timestamps))
}

/// Compute-only bind group layout. `read_only[i]` says whether binding i is a \[4\]
pub fn bind_layout(
    device: &wgpu::Device,
    label: &str,
    read_only: &[bool],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
        .iter()
        .enumerate()
        .map(|(i, &ro)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if i == 0 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: ro },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        })
        .collect();

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

/// List every adapter wgpu can reach, for working out which device a render \[5\]
pub fn print_adapters() -> Result<()> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = a.get_info();
        let l = a.limits();
        println!(
            "{:?}  {}  [{:?}]  driver {} {}",
            info.backend, info.name, info.device_type, info.driver, info.driver_info
        );
        println!(
            "    storage buffer max {} MiB | workgroup storage {} B | \
             {} invocations/wg | {} workgroups/dim",
            l.max_storage_buffer_binding_size / 1048576,
            l.max_compute_workgroup_storage_size,
            l.max_compute_invocations_per_workgroup,
            l.max_compute_workgroups_per_dimension
        );
        // [6]
        let steal = Config::default().max_steal_percent;
        let binding = (l.max_storage_buffer_binding_size as u64).min(l.max_buffer_size);
        println!(
            "    max --max-voices {} at --steal-percent {} ({} pool slots)",
            crate::gpu::max_voices_for_binding(binding, steal),
            steal,
            binding / 96,
        );
        println!(
            "    subgroups {} | int64 {} | timestamps {}",
            a.features().contains(wgpu::Features::SUBGROUP),
            a.features().contains(wgpu::Features::SHADER_INT64),
            a.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        );
    }
    Ok(())
}

/// One adapter, as the guided renderer's environment check lists it.
#[derive(Debug, Clone)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    /// The most of one buffer the adapter will bind to a shader, which is \[7\]
    pub binding_bytes: u64,
    /// The largest `--max-voices` that binding takes at the configured \[8\]
    pub max_voices: u32,
}

impl AdapterSummary {
    /// Listed, never chosen. See `create`.
    pub fn is_software(&self) -> bool {
        self.device_type == wgpu::DeviceType::Cpu
    }
}

/// Every adapter wgpu can reach, best first, and the index of the one \[9\]
pub fn survey(cfg: &Config) -> Result<(Vec<AdapterSummary>, Option<usize>)> {
    let backends = match &cfg.gpu_backend {
        Some(b) => parse_backends(b)
            .ok_or_else(|| anyhow::anyhow!("unknown gpu backend {b:?}"))?,
        None => wgpu::Backends::all(),
    };
    let want = cfg.gpu_adapter.as_ref().map(|w| w.to_ascii_lowercase());

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let mut found: Vec<(wgpu::AdapterInfo, wgpu::Limits)> = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .map(|a| (a.get_info(), a.limits()))
        .collect();
    found.sort_by_key(|(i, _)| rank(i));

    let pick = found.iter().position(|(i, _)| {
        let named = match &want {
            Some(w) => i.name.to_ascii_lowercase().contains(w),
            None => true,
        };
        i.device_type != wgpu::DeviceType::Cpu
            && backends.contains(wgpu::Backends::from(i.backend))
            && named
    });

    let list = found
        .into_iter()
        .map(|(i, l)| {
            let binding = (l.max_storage_buffer_binding_size as u64).min(l.max_buffer_size);
            AdapterSummary {
                name: i.name,
                backend: i.backend,
                device_type: i.device_type,
                binding_bytes: binding,
                max_voices: crate::gpu::max_voices_for_binding(binding, cfg.max_steal_percent),
            }
        })
        .collect();
    Ok((list, pick))
}
