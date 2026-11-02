// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The GPU adapters, listed one backend at a time, each in a process of its own. \[1\]

use crate::falconeye::winsys::{self, Process};
use anyhow::Result;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

/// How long one backend's listing may take before it is taken to be hung and \[2\]
pub const LIST_TIMEOUT: Duration = Duration::from_secs(60);

/// What a listing child prints last, once every adapter is out: a child that \[3\]
const DONE: &str = "done";

/// The order backends are listed and their adapters kept in, which is the order \[4\]
const ORDER: [wgpu::Backend; 4] = [wgpu::Backend::Vulkan, wgpu::Backend::Metal, wgpu::Backend::Dx12, wgpu::Backend::Gl];

/// One adapter, as much of it as a listing, `gpu-info` and the machine report \[5\]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Listed {
    pub name: String,
    #[serde(with = "backend_word")]
    pub backend: wgpu::Backend,
    #[serde(with = "kind_word")]
    pub device_type: wgpu::DeviceType,
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub driver_info: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub int64_atomics: bool,
    pub int64: bool,
    pub subgroups: bool,
    pub timestamps: bool,
    /// [`super::memory_for`], read where the adapter was listed.
    pub memory_bytes: Option<u64>,
}

impl Listed {
    fn read(a: &wgpu::Adapter) -> Listed {
        let i = a.get_info();
        let l = a.limits();
        let f = a.features();
        Listed {
            memory_bytes: super::memory_for(&i),
            name: i.name,
            backend: i.backend,
            device_type: i.device_type,
            vendor: i.vendor,
            device: i.device,
            driver: i.driver,
            driver_info: i.driver_info,
            max_buffer_size: l.max_buffer_size,
            max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
            max_storage_buffers_per_shader_stage: l.max_storage_buffers_per_shader_stage,
            max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
            int64_atomics: f.contains(wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS),
            int64: f.contains(wgpu::Features::SHADER_INT64),
            subgroups: f.contains(wgpu::Features::SUBGROUP),
            timestamps: f.contains(wgpu::Features::TIMESTAMP_QUERY),
        }
    }

    /// The most of one buffer the adapter binds to a shader, which caps the voice pool.
    pub fn binding_bytes(&self) -> u64 {
        (self.max_storage_buffer_binding_size as u64).min(self.max_buffer_size)
    }
}

/// A backend whose listing did not finish.
#[derive(Debug, Clone, PartialEq)]
pub struct Failed {
    pub backend: wgpu::Backend,
    /// How its process ended, to follow "its listing process": "crashed: an access \[6\]
    pub how: String,
}

/// Every adapter that was listed, and every backend that could not be.
#[derive(Debug, Clone, Default)]
pub struct Listing {
    pub adapters: Vec<Listed>,
    pub failed: Vec<Failed>,
}

/// One backend's listing: its adapters, or how its process ended.
type Outcome = Result<Vec<Listed>, String>;

/// The executable that has `list-adapters`, once a front end has said so.
static ISOLATE: Mutex<Option<PathBuf>> = Mutex::new(None);
/// Each backend's isolated listing, taken once a process and only when asked \[7\]
static LISTED: Mutex<Vec<(wgpu::Backend, Outcome)>> = Mutex::new(Vec::new());
/// The backends whose listing failed in this process, as `wgpu::Backends` bits.
static AVOID: AtomicU32 = AtomicU32::new(0);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// List adapters from now on in processes of their own, started from `exe`, \[8\]
pub fn isolate_with(exe: PathBuf) {
    *lock(&ISOLATE) = Some(exe);
}

/// Whether adapters are listed in processes of their own here.
pub fn isolated() -> bool {
    lock(&ISOLATE).is_some()
}

/// The backends whose listing crashed, hung or failed in this process. A render \[9\]
pub fn avoided() -> wgpu::Backends {
    wgpu::Backends::from_bits_truncate(AVOID.load(Ordering::SeqCst))
}

/// The adapters on `backends`: each backend in a process of its own once a front \[10\]
pub fn list(backends: wgpu::Backends) -> Listing {
    let exe = lock(&ISOLATE).clone();
    let Some(exe) = exe else {
        return Listing { adapters: list_here(backends), failed: Vec::new() };
    };
    let mut cached = lock(&LISTED);
    let missing: Vec<wgpu::Backend> =
        each(backends).into_iter().filter(|b| !cached.iter().any(|(c, _)| c == b)).collect();
    if !missing.is_empty() {
        let found = list_each(&exe, &missing, LIST_TIMEOUT);
        cached.extend(found);
    }
    assemble(each(backends).iter().filter_map(|b| cached.iter().find(|(c, _)| c == b)))
}

/// [`list`], isolated with `exe` when there is one, and taken afresh: for the \[11\]
pub fn list_via(exe: Option<&Path>, backends: wgpu::Backends) -> Listing {
    match exe {
        Some(exe) => list_isolated(exe, backends, LIST_TIMEOUT),
        None => Listing { adapters: list_here(backends), failed: Vec::new() },
    }
}

/// The backends this build can open on this platform.
fn compiled() -> wgpu::Backends {
    wgpu::Instance::enabled_backend_features()
        & (wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12 | wgpu::Backends::GL)
}

/// The adapters on `backends`, listed in this process. The instance is made for \[12\]
pub fn list_here(backends: wgpu::Backends) -> Vec<Listed> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() });
    instance.enumerate_adapters(backends).iter().map(Listed::read).collect()
}

/// The backends in `backends` this build has, in [`ORDER`].
fn each(backends: wgpu::Backends) -> Vec<wgpu::Backend> {
    let backends = backends & compiled();
    ORDER.into_iter().filter(|b| backends.contains(wgpu::Backends::from(*b))).collect()
}

/// Each of `backends` this build has in a child of `exe`, all at once, each \[13\]
pub fn list_isolated(exe: &Path, backends: wgpu::Backends, timeout: Duration) -> Listing {
    let found = list_each(exe, &each(backends), timeout);
    assemble(found.iter())
}

/// Each of `backends` in a child of `exe`, all at once. A backend whose listing \[14\]
fn list_each(exe: &Path, backends: &[wgpu::Backend], timeout: Duration) -> Vec<(wgpu::Backend, Outcome)> {
    let found: Vec<(wgpu::Backend, Outcome)> = std::thread::scope(|scope| {
        let running: Vec<_> = backends
            .iter()
            .map(|&b| (b, scope.spawn(move || list_one(exe, b, timeout))))
            .collect();
        running
            .into_iter()
            .map(|(b, h)| (b, h.join().unwrap_or_else(|_| Err("could not be read: the thread reading it panicked".into()))))
            .collect()
    });
    for (backend, outcome) in &found {
        let w = word(*backend);
        match outcome {
            Ok(adapters) => log::debug!("adapters on {w}, listed in a process of its own: {}", adapters.len()),
            Err(how) => {
                log::warn!("the adapter listing on {w} did not finish, and {w} is left out from here on: its listing process {how}");
                AVOID.fetch_or(wgpu::Backends::from(*backend).bits(), Ordering::SeqCst);
            }
        }
    }
    found
}

/// One listing from each backend's, in the order given.
fn assemble<'a>(outcomes: impl Iterator<Item = &'a (wgpu::Backend, Outcome)>) -> Listing {
    let mut listing = Listing::default();
    for (backend, outcome) in outcomes {
        match outcome {
            Ok(found) => listing.adapters.extend(found.iter().cloned()),
            Err(how) => listing.failed.push(Failed { backend: *backend, how: how.clone() }),
        }
    }
    listing
}

/// One backend's listing, in a child of `exe`: its adapters, or how the child \[15\]
fn list_one(exe: &Path, backend: wgpu::Backend, timeout: Duration) -> Result<Vec<Listed>, String> {
    let w = word(backend);
    // [16]
    let err_path = std::env::temp_dir().join(format!("kestrel-list-{}-{w}.stderr", std::process::id()));
    let mut child = std::fs::File::create(&err_path)
        .and_then(|err| {
            Command::new(exe)
                .args(["--force-cli", "list-adapters", "--backend", w])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(err)
                .spawn()
        })
        .map_err(|e| format!("could not be started: {e}"))?;
    // Opened at once, while it certainly lives, so a crash's module can be read.
    let process = Process::open(child.id());
    let (tx, rx) = mpsc::channel();
    let stdout = child.stdout.take().expect("stdout was piped");
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let deadline = Instant::now() + timeout;
    let mut found = Vec::new();
    let mut done = false;
    let mut hung = false;
    let mut crashed_in = None;
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) => {
                if let Some(rest) = line.strip_prefix("crash ") {
                    // [17]
                    let words: Vec<&str> = rest.split_whitespace().collect();
                    if let (Some(c), Some(p)) = (winsys::Crash::parse(&words), process.as_ref()) {
                        crashed_in = p.module_at(c.address);
                    }
                    Process::release(child.id());
                } else if line == DONE {
                    done = true;
                } else if let Ok(l) = serde_json::from_str::<Listed>(&line) {
                    found.push(l);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                hung = true;
                let _ = child.kill();
                break;
            }
            // Its end of the pipe closed: it has exited.
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child.wait();
    let tail = crate::falconeye::selftest::last_lines(&err_path, 4);
    let _ = std::fs::remove_file(&err_path);

    if done {
        if let Ok(s) = &status {
            if !s.success() {
                log::warn!(
                    "the adapter listing on {w} finished, then its process {}",
                    crate::falconeye::selftest::how_it_ended(s)
                );
            }
        }
        return Ok(found);
    }
    let mut how = if hung {
        format!("did not finish in {} s and was stopped", timeout.as_secs())
    } else {
        match &status {
            Ok(s) => crate::falconeye::selftest::how_it_ended(s),
            Err(e) => format!("could not be waited for: {e}"),
        }
    };
    if let Some((module, offset)) = crashed_in {
        how.push_str(&format!(", in {module}+{offset:#x}"));
    }
    if !tail.is_empty() {
        how.push_str(&format!(". Its last messages: {tail}"));
    }
    Err(how)
}

/// The child: `kestrel --force-cli list-adapters --backend <name>`, started by \[18\]
pub fn serve(backend: &str) -> Result<()> {
    let backends = super::device::parse_backends(backend)
        .ok_or_else(|| anyhow::anyhow!("unknown gpu backend {backend:?}"))?;
    // [19]
    winsys::install_crash_filter();
    winsys::arm_crash_filter_on_stdout();
    #[cfg(feature = "dev")]
    fail_on_purpose(backend);
    let found = list_here(backends);
    let mut out = std::io::stdout().lock();
    for l in found {
        writeln!(out, "{}", serde_json::to_string(&l)?)?;
        out.flush()?;
    }
    writeln!(out, "{DONE}")?;
    out.flush()?;
    Ok(())
}

/// Fail on purpose while listing a backend, `KESTREL_LIST_CRASH=<kind>@<backend>`, \[20\]
#[cfg(feature = "dev")]
fn fail_on_purpose(backend: &str) {
    let Ok(v) = std::env::var("KESTREL_LIST_CRASH") else { return };
    let Some((kind, at)) = v.split_once('@') else { return };
    if !at.eq_ignore_ascii_case(backend) {
        return;
    }
    eprintln!("[ERROR kestrel::gpu::listing] failing on purpose: {kind}");
    match kind {
        "segv" => winsys::access_violation(),
        "exit" => std::process::exit(3),
        "hang" => loop {
            std::thread::sleep(Duration::from_secs(1));
        },
        _ => {}
    }
}

/// The `--gpu-backend` spelling of a backend.
pub fn word(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Gl => "gl",
        _ => "other",
    }
}

fn backend_of(w: &str) -> wgpu::Backend {
    match w {
        "vulkan" => wgpu::Backend::Vulkan,
        "dx12" => wgpu::Backend::Dx12,
        "metal" => wgpu::Backend::Metal,
        "gl" => wgpu::Backend::Gl,
        _ => wgpu::Backend::Noop,
    }
}

fn kind_of(w: &str) -> wgpu::DeviceType {
    match w {
        "discrete" => wgpu::DeviceType::DiscreteGpu,
        "integrated" => wgpu::DeviceType::IntegratedGpu,
        "virtual" => wgpu::DeviceType::VirtualGpu,
        "cpu" => wgpu::DeviceType::Cpu,
        _ => wgpu::DeviceType::Other,
    }
}

fn kind_word(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "cpu",
        wgpu::DeviceType::Other => "other",
    }
}

mod backend_word {
    pub fn serialize<S: serde::Serializer>(b: &wgpu::Backend, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(super::word(*b))
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<wgpu::Backend, D::Error> {
        Ok(super::backend_of(&<String as serde::Deserialize>::deserialize(d)?))
    }
}

mod kind_word {
    pub fn serialize<S: serde::Serializer>(t: &wgpu::DeviceType, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(super::kind_word(*t))
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<wgpu::DeviceType, D::Error> {
        Ok(super::kind_of(&<String as serde::Deserialize>::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a child prints is what the lister reads back, backend and kind \[21\]
    #[test]
    fn a_listed_adapter_reads_back_as_it_was_printed() {
        for (backend, kind) in [
            (wgpu::Backend::Vulkan, wgpu::DeviceType::DiscreteGpu),
            (wgpu::Backend::Dx12, wgpu::DeviceType::IntegratedGpu),
            (wgpu::Backend::Gl, wgpu::DeviceType::Other),
            (wgpu::Backend::Metal, wgpu::DeviceType::Cpu),
            (wgpu::Backend::Dx12, wgpu::DeviceType::VirtualGpu),
        ] {
            let l = Listed {
                name: "Example Discrete GPU".into(),
                backend,
                device_type: kind,
                vendor: 0x10de,
                device: 0x25a2,
                driver: "NVIDIA".into(),
                driver_info: "566.36".into(),
                max_buffer_size: 1 << 52,
                max_storage_buffer_binding_size: u32::MAX,
                max_storage_buffers_per_shader_stage: 16,
                max_compute_workgroup_storage_size: 49152,
                max_compute_invocations_per_workgroup: 1024,
                max_compute_workgroups_per_dimension: 65535,
                int64_atomics: true,
                int64: true,
                subgroups: false,
                timestamps: true,
                memory_bytes: Some(4 << 30),
            };
            let line = serde_json::to_string(&l).unwrap();
            assert!(!line.contains('\n') && line.starts_with('{'), "{line}");
            assert_eq!(serde_json::from_str::<Listed>(&line).unwrap(), l);
        }
    }

    /// A listing is each backend's in the order asked for: its adapters, or \[22\]
    #[test]
    fn a_listing_keeps_each_backends_outcome_in_order() {
        let adapter = |backend| Listed {
            name: "A".into(),
            backend,
            device_type: wgpu::DeviceType::DiscreteGpu,
            vendor: 0,
            device: 0,
            driver: String::new(),
            driver_info: String::new(),
            max_buffer_size: 0,
            max_storage_buffer_binding_size: 0,
            max_storage_buffers_per_shader_stage: 0,
            max_compute_workgroup_storage_size: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroups_per_dimension: 0,
            int64_atomics: false,
            int64: false,
            subgroups: false,
            timestamps: false,
            memory_bytes: None,
        };
        let outcomes: Vec<(wgpu::Backend, Outcome)> = vec![
            (wgpu::Backend::Vulkan, Err("crashed".into())),
            (wgpu::Backend::Dx12, Ok(vec![adapter(wgpu::Backend::Dx12), adapter(wgpu::Backend::Dx12)])),
            (wgpu::Backend::Gl, Ok(vec![adapter(wgpu::Backend::Gl)])),
        ];
        let l = assemble(outcomes.iter());
        let backends: Vec<wgpu::Backend> = l.adapters.iter().map(|a| a.backend).collect();
        assert_eq!(backends, [wgpu::Backend::Dx12, wgpu::Backend::Dx12, wgpu::Backend::Gl]);
        assert_eq!(l.failed, [Failed { backend: wgpu::Backend::Vulkan, how: "crashed".into() }]);
        let gl_only = assemble(outcomes.iter().filter(|(b, _)| *b == wgpu::Backend::Gl));
        assert_eq!((gl_only.adapters.len(), gl_only.failed.len()), (1, 0));
    }

    /// A backend asked for by name is the only one listed: `--gpu-backend` is \[23\]
    #[test]
    fn only_the_backends_asked_for_are_listed() {
        for b in each(wgpu::Backends::all()) {
            assert_eq!(each(wgpu::Backends::from(b)), [b]);
        }
        // In wgpu's own order, Vulkan first.
        let all = each(wgpu::Backends::all());
        assert!(all.windows(2).all(|w| ORDER.iter().position(|b| *b == w[0]) < ORDER.iter().position(|b| *b == w[1])));
    }
}
