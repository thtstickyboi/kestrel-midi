// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Adapter selection, device creation and bind-group boilerplate.

use crate::config::Config;
use anyhow::{bail, Result};

/// Whether `name` is something `--gpu-backend` takes, so the command line can \[1\]
pub fn backend_known(name: &str) -> bool {
    parse_backends(name).is_some()
}

pub(crate) fn parse_backends(name: &str) -> Option<wgpu::Backends> {
    match name.to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => Some(wgpu::Backends::VULKAN),
        "dx12" | "d3d12" => Some(wgpu::Backends::DX12),
        "metal" => Some(wgpu::Backends::METAL),
        "gl" | "opengl" => Some(wgpu::Backends::GL),
        "all" => Some(wgpu::Backends::all()),
        _ => None,
    }
}

/// Which compiler turns the shaders into DX12 bytecode: DXC, statically linked \[2\]
fn dx12_compiler() -> wgpu::Dx12Compiler {
    if cfg!(all(windows, not(target_arch = "aarch64"))) {
        wgpu::Dx12Compiler::StaticDxc
    } else {
        wgpu::Dx12Compiler::Fxc
    }
}

/// Discrete first, then integrated. Within a tier, prefer Vulkan over DX12 \[3\]
fn rank(device_type: wgpu::DeviceType, backend: wgpu::Backend) -> (u8, u8) {
    let t = match device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        _ => 3,
    };
    let b = match backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 0,
        wgpu::Backend::Dx12 => 1,
        _ => 2,
    };
    (t, b)
}

/// Pick an adapter and open a device. \[4\]
pub fn create(
    cfg: &Config,
) -> Result<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo, wgpu::Limits, bool)> {
    let adapter = pick_adapter(cfg)?;
    let info = adapter.get_info();
    let adapter_limits = adapter.limits();
    // The driver is the first thing a crash report is read for.
    log::info!(
        "adapter: {} | {:?} {:?} | vendor {:#06x} device {:#06x} | driver {} {}",
        info.name,
        info.backend,
        info.device_type,
        info.vendor,
        info.device,
        info.driver,
        info.driver_info
    );

    // [5]
    log::info!("adapter limits: {}", limits_line(&adapter_limits));
    if let Some(m) = super::vram::sample_quick(info.vendor, info.device) {
        log::info!("adapter memory: {}", memory_line(&m));
    }

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
    }))
    .map_err(|e| {
        // [6]
        let others = other_ways(&info);
        anyhow::anyhow!(
            "{}",
            open_failure(
                &info.name,
                info.backend,
                &format!("{} {}", info.driver, info.driver_info),
                &e.to_string(),
                &others
            )
        )
    })?;
    finish_create(device, queue, info, adapter_limits, has_timestamps)
}

fn mib(bytes: u64) -> u64 {
    bytes >> 20
}

/// A size limit, in MiB: an adapter that has none says a number no memory could \[7\]
fn limit_mib(bytes: u64) -> String {
    if bytes >= 1 << 40 {
        "no limit".to_string()
    } else {
        format!("{} MiB", mib(bytes))
    }
}

/// The limits a render's buffers and passes are sized against, as one line.
fn limits_line(l: &wgpu::Limits) -> String {
    format!(
        "buffer {}, storage binding {}, {} storage buffers a stage, {} workgroup invocations, \
         {} workgroups a dimension",
        limit_mib(l.max_buffer_size),
        limit_mib(l.max_storage_buffer_binding_size as u64),
        l.max_storage_buffers_per_shader_stage,
        l.max_compute_invocations_per_workgroup,
        l.max_compute_workgroups_per_dimension
    )
}

/// What DXGI says about the adapter's memory: the dedicated amount, and the budget \[8\]
fn memory_line(m: &super::vram::GpuMemory) -> String {
    let some = |v: Option<u64>| v.map_or("unknown".to_string(), |b| format!("{} MiB", mib(b)));
    format!(
        "{} MiB dedicated, a budget of {} for this process, of which it has used {}",
        mib(m.dedicated_total),
        some(m.process_budget),
        some(m.process_used)
    )
}

/// The `--gpu-backend` word that opens an adapter the way its backend does.
fn backend_word(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::Metal => "metal",
        _ => "all",
    }
}

/// The other ways this machine offers to open a GPU, which a failure names: every \[9\]
fn other_ways(failed: &wgpu::AdapterInfo) -> Vec<String> {
    let mut found: Vec<String> = super::listing::list(wgpu::Backends::all())
        .adapters
        .into_iter()
        .filter(|i| i.device_type != wgpu::DeviceType::Cpu)
        .filter(|i| !(i.name == failed.name && i.backend == failed.backend))
        .map(|i| format!("{} on {:?} (--gpu-backend {})", i.name, i.backend, backend_word(i.backend)))
        .collect();
    found.sort();
    found.dedup();
    found
}

/// The error for a device that could not be opened at all.
fn open_failure(adapter: &str, backend: wgpu::Backend, driver: &str, error: &str, others: &[String]) -> String {
    let mut text = format!("could not open a device on {adapter} ({backend:?}, driver {}): {error}.", driver.trim());
    if error.to_ascii_lowercase().contains("lost") {
        text.push_str(
            " The driver reported the device lost while it was being opened, before any of this \
             render's own work, so it is the driver or the card and not the render's settings \
             (voices, block size). Updating the graphics driver and closing programs that use the \
             GPU can help.",
        );
    }
    if others.is_empty() {
        text.push_str(" --backend cpu renders on the processor, slowly.");
    } else {
        text.push_str(&format!(
            " This machine also offers: {}. --backend cpu renders on the processor, slowly.",
            others.join("; ")
        ));
    }
    text
}

/// The backend of the adapter a render with `cfg` lands on, from a listing taken \[10\]
fn listed_backend(cfg: &Config, backends: wgpu::Backends) -> Option<wgpu::Backends> {
    if !super::listing::isolated() {
        return None;
    }
    let mut found = super::listing::list(backends).adapters;
    found.sort_by_key(|i| rank(i.device_type, i.backend));
    let i = default_of(&found, backends, cfg.gpu_adapter.as_deref())?;
    log::debug!(
        "the listing ranks {} on {} first; this process opens {} alone",
        found[i].name,
        backend_word(found[i].backend),
        backend_word(found[i].backend)
    );
    Some(wgpu::Backends::from(found[i].backend))
}

/// The adapter a render lands on among `found`, sorted by `rank`: the first that \[11\]
fn default_of(found: &[super::listing::Listed], backends: wgpu::Backends, want: Option<&str>) -> Option<usize> {
    let want = want.map(str::to_ascii_lowercase);
    found.iter().position(|i| {
        let named = match &want {
            Some(w) => i.name.to_ascii_lowercase().contains(w),
            None => true,
        };
        i.device_type != wgpu::DeviceType::Cpu && backends.contains(wgpu::Backends::from(i.backend)) && named
    })
}

/// The adapter a render with `cfg` runs on, chosen as `create` chooses it, \[12\]
pub fn pick_adapter(cfg: &Config) -> Result<wgpu::Adapter> {
    let backends = match &cfg.gpu_backend {
        Some(b) => parse_backends(b)
            .ok_or_else(|| anyhow::anyhow!("unknown gpu backend {b:?}"))?,
        // [13]
        None => {
            let avoided = super::listing::avoided();
            if !avoided.is_empty() {
                log::warn!(
                    "leaving out {}, whose adapter listing did not finish in this session; \
                     --gpu-backend names a backend to use anyway",
                    backend_names(avoided)
                );
            }
            let all = wgpu::Backends::all() - avoided;
            listed_backend(cfg, all).unwrap_or(all)
        }
    };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        backend_options: wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: dx12_compiler(),
                ..Default::default()
            },
            ..Default::default()
        },
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

    candidates.sort_by_key(|a| {
        let i = a.get_info();
        rank(i.device_type, i.backend)
    });

    candidates.into_iter().next().ok_or_else(|| {
        anyhow::anyhow!(
            "no usable gpu found. wgpu saw no non-software adapter; \
             install or enable a graphics driver, or render with --backend cpu"
        )
    })
}

/// The rest of `create`, once the device is open.
fn finish_create(
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    adapter_limits: wgpu::Limits,
    has_timestamps: bool,
) -> Result<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo, wgpu::Limits, bool)> {
    // [14]
    device.on_uncaptured_error(std::sync::Arc::new(|e| {
        log::error!("wgpu device error: {e}");
        panic!("wgpu device error: {e}");
    }));

    // [15]
    *LOST.lock().unwrap_or_else(|p| p.into_inner()) = None;
    device.set_device_lost_callback(|reason, message| {
        // `Destroyed` is the device being dropped at the end of a render.
        if reason != wgpu::DeviceLostReason::Destroyed {
            // [16]
            log::error!(
                "gpu device lost: {message}. On Windows this is usually the graphics driver being \
                 reset because one piece of GPU work ran past its 2-second limit; fewer voices \
                 (--max-voices) or --block 1024 gives the GPU shorter pieces of work."
            );
            *LOST.lock().unwrap_or_else(|p| p.into_inner()) = Some(message);
        }
    });

    Ok((device, queue, info, adapter_limits, has_timestamps))
}

/// What the device-lost callback last said, if the device was lost.
static LOST: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// The error for a poll or a map that failed, which is how a lost device \[17\]
pub(super) fn lost(device: Option<&wgpu::Device>, detail: impl std::fmt::Display) -> anyhow::Error {
    let reason = LOST
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
        .map(|m| format!(": {m}"))
        .unwrap_or_default();
    // On DX12, Windows can say why: a timeout, a reset, or the driver.
    let dx12 = match device.and_then(crate::falconeye::winsys::dx12_removed_reason) {
        Some((code, text)) => {
            log::error!("DX12 device removed reason: {code:#010x}, {text}");
            format!(" DX12's reason: {text} ({code:#010x}).")
        }
        None => String::new(),
    };
    anyhow::anyhow!(
        "the GPU stopped responding and was lost{reason}.{dx12} On Windows this is \
         usually the graphics driver being reset because one piece of GPU work \
         ran past its 2-second limit, which a slower GPU can reach on a dense \
         passage. Kestrel already cuts a very large block into several \
         submissions; if this render stopped anyway, lower --max-voices, or \
         render again with --block 1024, which gives the GPU shorter pieces of \
         work. Closing other programs that use the GPU and updating the \
         graphics driver can also help. (wgpu: {detail})"
    )
}

/// Compute-only bind group layout. `read_only[i]` says whether binding i is a \[18\]
pub fn bind_layout(
    device: &wgpu::Device,
    label: &str,
    read_only: &[bool],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &layout_entries(read_only),
    })
}

/// `bind_layout`'s entries: binding 0 the uniforms, and a storage buffer at \[19\]
pub fn layout_entries(read_only: &[bool]) -> Vec<wgpu::BindGroupLayoutEntry> {
    read_only
        .iter()
        .enumerate()
        .map(|(i, &ro)| {
            if i == 0 {
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            } else {
                storage_entry(i as u32, ro)
            }
        })
        .collect()
}

/// A compute storage buffer at `binding`.
pub fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
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

/// The names of `backends`, as `--gpu-backend` spells them, for a message.
pub(crate) fn backend_names(backends: wgpu::Backends) -> String {
    [wgpu::Backend::Vulkan, wgpu::Backend::Metal, wgpu::Backend::Dx12, wgpu::Backend::Gl]
        .into_iter()
        .filter(|b| backends.contains(wgpu::Backends::from(*b)))
        .map(backend_word)
        .collect::<Vec<_>>()
        .join(", ")
}

/// List every adapter wgpu can reach on `backend` (`--gpu-backend`'s names; \[20\]
pub fn print_adapters(backend: Option<&str>) -> Result<()> {
    let backends = match backend {
        Some(b) => parse_backends(b).ok_or_else(|| anyhow::anyhow!("unknown gpu backend {b:?}"))?,
        None => wgpu::Backends::all(),
    };
    let listing = super::listing::list(backends);
    for a in &listing.adapters {
        println!(
            "{:?}  {}  [{:?}]  driver {} {}",
            a.backend, a.name, a.device_type, a.driver, a.driver_info
        );
        println!(
            "    storage buffer max {} MiB | workgroup storage {} B | \
             {} invocations/wg | {} workgroups/dim",
            a.max_storage_buffer_binding_size / 1048576,
            a.max_compute_workgroup_storage_size,
            a.max_compute_invocations_per_workgroup,
            a.max_compute_workgroups_per_dimension
        );
        // [21]
        let steal = Config::default().max_steal_percent;
        let binding = a.binding_bytes();
        println!(
            "    max --max-voices {} at --steal-percent {} ({} pool slots)",
            crate::gpu::max_voices_for_binding(binding, steal),
            steal,
            binding / 96,
        );
        // [22]
        match a.memory_bytes {
            Some(m) => println!(
                "    memory {} MiB for a render to plan on; the voices that fit in it depend on the soundfont's samples \
                 (about {} bytes a pool slot)",
                m >> 20,
                crate::gpu::bytes_per_slot(&Config::default()),
            ),
            None => println!("    memory: no figure on this system, so only the binding's limit above is known"),
        }
        println!("    subgroups {} | int64 {} | timestamps {}", a.subgroups, a.int64, a.timestamps);
    }
    for f in &listing.failed {
        println!(
            "{:?}  could not be listed: its listing process {}. Renders leave it out unless --gpu-backend {} names it.",
            f.backend,
            f.how,
            backend_word(f.backend)
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
    /// The most of one buffer the adapter will bind to a shader, which is \[23\]
    pub binding_bytes: u64,
    /// The largest `--max-voices` that binding takes at the configured \[24\]
    pub max_voices: u32,
    /// PCI ids, as `wgpu::AdapterInfo` has them.
    pub vendor: u32,
    pub device: u32,
    /// The memory a render can plan on, where it can be read: the card's own for a \[25\]
    pub memory_bytes: Option<u64>,
}

/// The memory a render can plan on for this adapter, for [`AdapterSummary::memory_bytes`]. \[26\]
pub fn memory_for(info: &wgpu::AdapterInfo) -> Option<u64> {
    /// What was read for an adapter, by its ids and whether it is integrated.
    type Known = Vec<((u32, u32, bool), Option<u64>)>;
    static KNOWN: std::sync::Mutex<Known> = std::sync::Mutex::new(Vec::new());
    let integrated = info.device_type == wgpu::DeviceType::IntegratedGpu;
    let key = (info.vendor, info.device, integrated);
    if let Some((_, v)) = KNOWN.lock().ok()?.iter().find(|(k, _)| *k == key) {
        return *v;
    }
    let figure = match info.device_type {
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::VirtualGpu => {
            super::vram::dedicated_total(info.vendor, info.device).filter(|&b| b > 0)
        }
        // [27]
        wgpu::DeviceType::IntegratedGpu if cfg!(windows) => super::vram::sample_quick(info.vendor, info.device)
            .and_then(|m| m.process_budget)
            .filter(|&b| b > 0),
        _ => None,
    };
    if let Ok(mut k) = KNOWN.lock() {
        k.push((key, figure));
    }
    figure
}

impl AdapterSummary {
    /// Listed, never chosen. See `create`.
    pub fn is_software(&self) -> bool {
        self.device_type == wgpu::DeviceType::Cpu
    }
}

/// What [`survey`] found.
#[derive(Debug, Clone)]
pub struct Survey {
    /// Every adapter wgpu can reach, best first.
    pub adapters: Vec<AdapterSummary>,
    /// The index of the one `create` would open -- `None` when nothing is eligible.
    pub default: Option<usize>,
    /// The backends whose listing crashed, hung or failed, and how. Renders that \[28\]
    pub failed: Vec<super::listing::Failed>,
}

/// Every adapter wgpu can reach, best first, and the one `create` would open \[29\]
pub fn survey(cfg: &Config) -> Result<Survey> {
    let backends = match &cfg.gpu_backend {
        Some(b) => parse_backends(b)
            .ok_or_else(|| anyhow::anyhow!("unknown gpu backend {b:?}"))?,
        None => wgpu::Backends::all(),
    };
    let listing = super::listing::list(wgpu::Backends::all());
    let mut found = listing.adapters;
    found.sort_by_key(|i| rank(i.device_type, i.backend));
    let default = default_of(&found, backends, cfg.gpu_adapter.as_deref());

    let adapters = found
        .into_iter()
        .map(|i| {
            let binding = i.binding_bytes();
            AdapterSummary {
                memory_bytes: i.memory_bytes,
                vendor: i.vendor,
                device: i.device,
                name: i.name,
                backend: i.backend,
                device_type: i.device_type,
                binding_bytes: binding,
                max_voices: crate::gpu::max_voices_for_config(binding, cfg),
            }
        })
        .collect();
    Ok(Survey { adapters, default, failed: listing.failed })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a user of an integrated GPU saw: "Parent device is lost", and nothing else.
    #[test]
    fn a_device_that_could_not_be_opened_says_which_adapter_why_and_what_else_to_try() {
        let others = vec!["Example(R) Integrated Graphics on Dx12 (--gpu-backend dx12)".to_string()];
        let text = open_failure(
            "Example(R) Integrated Graphics",
            wgpu::Backend::Vulkan,
            "Intel Corporation Intel driver ",
            "Parent device is lost",
            &others,
        );
        assert!(text.starts_with("could not open a device on Example(R) Integrated Graphics (Vulkan, driver Intel Corporation Intel driver): Parent device is lost."), "{text}");
        assert!(text.contains("while it was being opened") && text.contains("not the render's settings"), "{text}");
        assert!(text.contains("--gpu-backend dx12") && text.contains("--backend cpu"), "{text}");
        // Another failure is not called a lost device, and with nothing else to try says so.
        let text = open_failure("A", wgpu::Backend::Dx12, "d", "Requested limit is not available", &[]);
        assert!(!text.contains("lost") && !text.contains("also offers") && text.contains("--backend cpu"), "{text}");
    }

    /// A software adapter has no memory of its own to plan on, and an adapter that has a \[30\]
    #[test]
    fn a_survey_gives_memory_only_where_there_is_some_and_the_same_each_time() {
        let cfg = Config::default();
        let first = survey(&cfg).unwrap().adapters;
        let again = survey(&cfg).unwrap().adapters;
        assert_eq!(first.len(), again.len());
        for (a, b) in first.iter().zip(&again) {
            assert_eq!(a.memory_bytes, b.memory_bytes, "{}", a.name);
            if a.is_software() {
                assert_eq!(a.memory_bytes, None, "{}: a software adapter has no memory", a.name);
            }
            if let Some(m) = a.memory_bytes {
                // More than a few MiB, and under the largest card there is.
                assert!((32 << 20..=(1u64 << 40)).contains(&m), "{}: {m}", a.name);
            }
        }
    }

    fn listed(name: &str, backend: wgpu::Backend, device_type: wgpu::DeviceType) -> crate::gpu::listing::Listed {
        crate::gpu::listing::Listed {
            name: name.into(),
            backend,
            device_type,
            vendor: 0,
            device: 0,
            driver: String::new(),
            driver_info: String::new(),
            max_buffer_size: 2047 << 20,
            max_storage_buffer_binding_size: 2047 << 20,
            max_storage_buffers_per_shader_stage: 16,
            max_compute_workgroup_storage_size: 32768,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroups_per_dimension: 65535,
            int64_atomics: false,
            int64: false,
            subgroups: false,
            timestamps: false,
            memory_bytes: None,
        }
    }

    /// A user's adapters, as their machine report listed them (2026-10-23). A \[31\]
    #[test]
    fn a_render_opens_the_backend_of_the_adapter_it_lands_on() {
        use wgpu::{Backend, DeviceType};
        let mut found = vec![
            listed("AMD Radeon 780M Graphics", Backend::Vulkan, DeviceType::IntegratedGpu),
            listed("Example Discrete GPU", Backend::Vulkan, DeviceType::DiscreteGpu),
            listed("AMD Radeon 780M Graphics", Backend::Dx12, DeviceType::IntegratedGpu),
            listed("Example Discrete GPU", Backend::Dx12, DeviceType::DiscreteGpu),
            listed("Microsoft Basic Render Driver", Backend::Dx12, DeviceType::Cpu),
            listed("AMD Radeon 780M Graphics", Backend::Gl, DeviceType::Other),
        ];
        found.sort_by_key(|i| rank(i.device_type, i.backend));
        let pick = |backends, want| default_of(&found, backends, want).map(|i| (found[i].name.as_str(), found[i].backend));
        let all = wgpu::Backends::all();
        assert_eq!(pick(all, None), Some(("Example Discrete GPU", Backend::Vulkan)));
        assert_eq!(pick(all - wgpu::Backends::VULKAN, None), Some(("Example Discrete GPU", Backend::Dx12)));
        assert_eq!(pick(all, Some("780m")), Some(("AMD Radeon 780M Graphics", Backend::Vulkan)));
        assert_eq!(pick(wgpu::Backends::GL, None), Some(("AMD Radeon 780M Graphics", Backend::Gl)));
        // Never the software renderer, and nothing for a name nobody has.
        assert_eq!(pick(wgpu::Backends::DX12, Some("basic render")), None);
        assert_eq!(pick(all, Some("radeon rx")), None);
    }

    #[test]
    fn the_limits_and_memory_lines_are_in_mebibytes() {
        let limits = wgpu::Limits {
            max_buffer_size: 4 << 30,
            max_storage_buffer_binding_size: 2 << 30,
            ..wgpu::Limits::default()
        };
        let line = limits_line(&limits);
        assert!(line.contains("buffer 4096 MiB, storage binding 2048 MiB"), "{line}");
        // [32]
        for none in [1u64 << 52, u64::MAX] {
            let unlimited = wgpu::Limits { max_buffer_size: none, ..limits.clone() };
            assert!(limits_line(&unlimited).contains("buffer no limit, storage binding"), "{}", limits_line(&unlimited));
        }
        let m = super::super::vram::GpuMemory {
            dedicated_total: 128 << 20,
            process_budget: Some(3 << 30),
            process_used: None,
            ..Default::default()
        };
        assert_eq!(
            memory_line(&m),
            "128 MiB dedicated, a budget of 3072 MiB for this process, of which it has used unknown"
        );
    }
}
