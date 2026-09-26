// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The machine report: one file a user can send, holding what it takes to \[1\]

use super::redact::Redactor;
use super::selftest;
use super::watch::{self, tool};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Shown before anything is collected, and at the top of the report.
pub const HEADER: &str = "\
This report is for tracking down a problem with Kestrel on your machine. It contains:
- your hardware: the CPU, memory, GPUs and their drivers, and how they are set up;
- Windows' version and graphics settings, and 30 days of graphics driver resets and
  Kestrel crashes from Windows' own logs;
- Kestrel's settings, and your recent render logs and crash reports, which name the
  MIDIs and soundfonts you rendered;
- a GPU self-test, if you ran one;
- if you allowed it as administrator, the list of Windows' own GPU crash dumps (listed,
  not copied) and Windows' short reports of GPU resets and Kestrel crashes.
Your PC's name and your Windows account's name are replaced with <pc> and <user>
everywhere in it, file paths included. It does not contain your files, the names of
other programs, network details, serial numbers or environment variables.
Send it privately to Kestrel's developer, not to a public issue.";

/// One part of the report: labelled values, and text that is kept as it is.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Section {
    pub name: String,
    pub items: Vec<(String, String)>,
    pub text: Option<String>,
}

impl Section {
    pub fn new(name: &str) -> Section {
        Section { name: name.into(), ..Default::default() }
    }

    pub fn item(&mut self, key: &str, value: impl Into<String>) {
        self.items.push((key.into(), value.into()));
    }
}

pub struct Options {
    /// Run the GPU self-test, which loads each GPU for up to a minute.
    pub self_test: bool,
    /// Sections only the front end can fill: its settings file.
    pub extra: Vec<Section>,
    /// Where to put the report, instead of `reports` beside the logs.
    pub out_dir: Option<PathBuf>,
    /// The administrator step, once the user has agreed to it: this \[2\]
    pub elevate_with: Option<PathBuf>,
}

/// Collect the report and write it. `say` hears each step as it starts. \[3\]
pub fn build(opts: Options, say: &mut dyn FnMut(&str)) -> Result<PathBuf> {
    let mut sections = Vec::new();
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();

    say("Kestrel");
    sections.push(kestrel());
    sections.extend(opts.extra);
    say("Windows");
    sections.push(windows());
    say("GPUs, on every backend");
    let adapters = adapters();
    sections.push(adapters.0);
    say("NVIDIA's own view (nvidia-smi)");
    sections.push(nvidia());
    say("Windows' graphics settings");
    sections.push(graphics_settings());
    say("30 days of driver resets and Kestrel crashes, from Windows' logs");
    sections.push(driver_history());
    say("Kestrel's render logs and reports");
    let (history, keep) = kestrel_history(&mut files);
    sections.push(history);
    if let Some(exe) = &opts.elevate_with {
        say("Windows' own GPU crash records (Windows asks for administrator permission)");
        sections.push(administrator_step(exe, &mut files));
    }

    let mut tests = Vec::new();
    if opts.self_test {
        say("GPU self-test");
        let dir = std::env::temp_dir().join(format!("kestrel-selftest-{}", std::process::id()));
        // [4]
        let level = log::max_level();
        log::set_max_level(log::LevelFilter::Warn);
        match selftest::prepare(&dir) {
            Ok(bank) => {
                for (name, backend, max_voices) in &adapters.1 {
                    say(&format!("  {name} ({backend})"));
                    let outcome = selftest::run(&bank, &dir, name, backend, *max_voices, &mut |l| say(&format!("    {l}")));
                    say(&format!("    {}", outcome.verdict));
                    sections.push(self_test_section(&outcome));
                    tests.push(outcome);
                }
            }
            Err(e) => {
                let mut s = Section::new("GPU self-test");
                s.item("error", format!("the test material could not be made: {e:#}"));
                sections.push(s);
            }
        }
        log::set_max_level(level);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Names: this machine's hidden, the MIDIs' in the logs kept.
    let mut redact = Redactor::from_env();
    for k in &keep {
        redact.keep(k);
    }
    let created = chrono::Local::now().format("%m-%d-%Y %H.%M.%S").to_string();
    let text = redact.apply(&render_text(&sections, &created));
    let json = redact.apply(&serde_json::to_string_pretty(&serde_json::json!({
        "kestrel_report": 1,
        "created": created,
        "sections": sections,
        "self_test": tests,
    }))?);

    let dir = match opts.out_dir {
        Some(d) => d,
        None => out_dir()?,
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(format!("kestrel-report-{created}.zip"));
    let mut zip = super::zip::ZipWriter::new(std::io::BufWriter::new(
        std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?,
    ));
    zip.add("report.txt", text.replace('\n', "\r\n").as_bytes())?;
    zip.add("report.json", json.as_bytes())?;
    for (name, mut data) in files {
        if name.ends_with(".dmp") {
            redact.scrub_bytes(&mut data);
        } else {
            data = redact.apply(&String::from_utf8_lossy(&data)).into_bytes();
        }
        // [5]
        let at = if name.contains('/') { name.clone() } else { format!("logs/{name}") };
        zip.add(&redact.apply(&at), &data)?;
    }
    zip.finish()?;
    Ok(path)
}

fn render_text(sections: &[Section], created: &str) -> String {
    let mut out = format!("Kestrel machine report, {created}\n\n{HEADER}\n");
    for s in sections {
        out.push_str(&format!("\n== {} ==\n", s.name));
        let width = s.items.iter().map(|(k, _)| k.chars().count()).max().unwrap_or(0);
        for (k, v) in &s.items {
            out.push_str(&format!("{k:<width$}  {v}\n"));
        }
        if let Some(t) = &s.text {
            out.push_str(t.trim_end());
            out.push('\n');
        }
    }
    out
}

/// `reports` beside the logs: beside the executable, or in local app data \[6\]
fn out_dir() -> Result<PathBuf> {
    for logs in super::renderlog::default_dirs() {
        let d = logs.with_file_name("reports");
        if std::fs::create_dir_all(&d).is_ok() && std::fs::metadata(&d).is_ok_and(|m| !m.permissions().readonly()) {
            return Ok(d);
        }
    }
    anyhow::bail!("no folder to write the report to")
}

fn kestrel() -> Section {
    let mut s = Section::new("Kestrel");
    s.item("version", env!("CARGO_PKG_VERSION"));
    s.item("build", if cfg!(feature = "dev") { "dev" } else { "release" });
    if let Ok(exe) = std::env::current_exe() {
        s.item("exe", exe.display().to_string());
        if let Ok(m) = std::fs::metadata(&exe) {
            s.item("exe size", format!("{} bytes", m.len()));
        }
    }
    s.item("anchor", format!("{:#x} (renderlog::anchor, for placing a backtrace)", super::renderlog::anchor()));
    s.item("platform", format!("{} {}", std::env::consts::OS, std::env::consts::ARCH));
    s.item("threads", std::thread::available_parallelism().map_or("?".into(), |n| n.to_string()));
    s
}

/// Windows' facts, read from the registry and kernel32 directly -- not \[7\]
#[cfg(windows)]
fn windows() -> Section {
    use super::winsys::{memory, power, reg_dword, reg_string};
    let mut s = Section::new("Windows");
    let cv = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion";
    let build = reg_string(cv, "CurrentBuild").unwrap_or_default();
    // Windows 11 still calls itself "Windows 10" here; its build says which.
    let mut product = reg_string(cv, "ProductName").unwrap_or_else(|| "Windows".into());
    if build.parse::<u32>().is_ok_and(|b| b >= 22000) {
        product = product.replacen("Windows 10", "Windows 11", 1);
    }
    s.item("Windows", format!(
        "{product} {}, build {build}.{}",
        reg_string(cv, "DisplayVersion").unwrap_or_default(),
        reg_dword(cv, "UBR").unwrap_or(0)
    ));
    if let Some(e) = reg_string(cv, "EditionID") {
        s.item("Edition", e);
    }
    let bios = r"HARDWARE\DESCRIPTION\System\BIOS";
    s.item("Machine", format!(
        "{} {}",
        reg_string(bios, "SystemManufacturer").unwrap_or_default(),
        reg_string(bios, "SystemProductName").unwrap_or_default()
    ));
    let cpu = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0";
    s.item("CPU", format!(
        "{}, {} threads, {} MHz base",
        reg_string(cpu, "ProcessorNameString").unwrap_or_else(|| "?".into()),
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        reg_dword(cpu, "~MHz").unwrap_or(0)
    ));
    if let Some(m) = memory() {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        s.item("RAM", format!("{:.1} GiB, {:.1} GiB free", gib(m.total), gib(m.available)));
        s.item("Commit limit", format!(
            "{:.1} GiB (RAM and page files), {:.1} GiB free",
            gib(m.commit_limit),
            gib(m.commit_available)
        ));
    }
    if let Some(p) = power() {
        let plugged = match p.plugged_in {
            Some(true) => "plugged in",
            Some(false) => "on battery",
            None => "power source unknown",
        };
        s.item("Power", match p.battery {
            Some(pct) => format!("{plugged}, battery {pct}%"),
            None => format!("{plugged}, no battery"),
        });
    }
    // The display class's driver entries: the name, version and date.
    let class = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
    for n in 0..16 {
        let key = format!(r"{class}\{n:04}");
        if let Some(desc) = reg_string(&key, "DriverDesc") {
            s.item("Display adapter", format!(
                "{desc}, driver {} of {}",
                reg_string(&key, "DriverVersion").unwrap_or_else(|| "?".into()),
                reg_string(&key, "DriverDate").unwrap_or_else(|| "?".into())
            ));
        }
    }
    match tool("powercfg", &["/getactivescheme"], Duration::from_secs(15)) {
        Ok(out) => s.item("Power plan", out.trim().rsplit_once("  ").map_or(out.trim(), |(_, name)| name).trim_matches(['(', ')'])),
        Err(e) => s.item("Power plan", e),
    }
    s
}

#[cfg(not(windows))]
fn windows() -> Section {
    let mut s = Section::new("Operating system");
    s.item("note", "only Windows is collected so far");
    s
}

/// Every adapter on every backend, and which of them the self-test runs on: \[8\]
fn adapters() -> (Section, Vec<(String, String, u32)>) {
    let mut s = Section::new("GPUs, as wgpu sees them on every backend");
    let mut text = String::new();
    let mut testable = Vec::new();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let i = a.get_info();
        let l = a.limits();
        let f = a.features();
        let binding = (l.max_storage_buffer_binding_size as u64).min(l.max_buffer_size);
        let voices = crate::gpu::max_voices_for_binding(binding, crate::config::Config::default().max_steal_percent);
        text.push_str(&format!(
            "{} ({:?}, {:?})\n  vendor {:#06x} device {:#06x}, driver {} {}\n  \
             largest binding {} MiB, largest buffer {}, so at most {} voices\n  \
             64-bit atomics {}, subgroups {}, timestamps {}\n",
            i.name,
            i.backend,
            i.device_type,
            i.vendor,
            i.device,
            i.driver,
            i.driver_info,
            binding >> 20,
            // Vulkan reports its largest buffer as effectively unbounded.
            if l.max_buffer_size >= 1 << 40 {
                "no practical limit".to_string()
            } else {
                format!("{} MiB", l.max_buffer_size >> 20)
            },
            voices,
            yes(f.contains(wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS)),
            yes(f.contains(wgpu::Features::SUBGROUP)),
            yes(f.contains(wgpu::Features::TIMESTAMP_QUERY)),
        ));
        if let Some(m) = crate::gpu::vram::sample(i.vendor, i.device) {
            let mib = |b: Option<u64>| b.map_or("?".into(), |b| (b >> 20).to_string());
            text.push_str(&format!(
                "  video memory {} MiB, {} MiB in use by everything, this process's budget {} MiB\n",
                m.dedicated_total >> 20,
                mib(m.dedicated_used),
                mib(m.process_budget)
            ));
        }
        let hardware = i.device_type != wgpu::DeviceType::Cpu;
        let backend = match i.backend {
            wgpu::Backend::Vulkan => Some("vulkan"),
            wgpu::Backend::Dx12 => Some("dx12"),
            wgpu::Backend::Metal => Some("metal"),
            _ => None,
        };
        if let (true, Some(b)) = (hardware, backend) {
            testable.push((i.name.clone(), b.to_string(), voices));
        }
    }
    s.text = Some(if text.is_empty() { "(none found)".into() } else { text });
    (s, testable)
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

/// `nvidia-smi`, where NVIDIA's driver is installed: the state now, the link, \[9\]
fn nvidia() -> Section {
    let mut s = Section::new("NVIDIA (nvidia-smi)");
    let query = "--query-gpu=name,driver_version,vbios_version,pstate,temperature.gpu,power.draw,\
                 power.limit,power.default_limit,clocks.gr,clocks.mem,clocks.max.graphics,clocks.max.memory,\
                 pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,\
                 utilization.gpu,memory.used,memory.total,clocks_event_reasons.active";
    match tool("nvidia-smi", &[query, "--format=csv"], Duration::from_secs(20)) {
        Ok(csv) => {
            let mut rows = csv.lines();
            let head: Vec<&str> = rows.next().unwrap_or("").split(", ").collect();
            for (n, row) in rows.enumerate() {
                for (k, v) in head.iter().zip(row.split(", ")) {
                    s.item(&format!("{n}: {k}"), v);
                }
            }
            if let Ok(table) = tool("nvidia-smi", &[], Duration::from_secs(20)) {
                s.item("programs using the GPU", format!("{} (names left out)", count_processes(&table)));
            }
        }
        Err(e) => s.item("note", format!("{e}; only NVIDIA's driver has it")),
    }
    s
}

/// The rows of `nvidia-smi`'s process table.
fn count_processes(table: &str) -> usize {
    table
        .lines()
        .skip_while(|l| !l.contains("Processes:"))
        .filter(|l| {
            let w: Vec<&str> = l.trim_matches(|c| c == '|' || c == ' ').split_whitespace().collect();
            w.len() >= 5 && w[0].parse::<u32>().is_ok() && matches!(w[4], "C" | "G" | "C+G")
        })
        .count()
}

/// Windows' timeout settings and GPU scheduling. Absent means Windows' \[10\]
#[cfg(windows)]
fn graphics_settings() -> Section {
    let mut s = Section::new("Windows' graphics settings");
    let key = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers";
    for name in ["TdrDelay", "TdrDdiDelay", "TdrLevel", "TdrLimitCount", "TdrLimitTime", "HwSchMode"] {
        s.item(name, match super::winsys::reg_dword(key, name) {
            Some(v) => v.to_string(),
            None => "not set (Windows' default)".into(),
        });
    }
    s
}

#[cfg(not(windows))]
fn graphics_settings() -> Section {
    Section::new("Graphics settings (Windows only)")
}

fn driver_history() -> Section {
    let mut s = Section::new("Driver resets and Kestrel crashes, last 30 days");
    s.text = Some(watch::windows_events(30 * 24 * 3600 * 1000));
    s
}

/// The render logs and FalconEye's reports: listed, and the recent ones added \[11\]
fn kestrel_history(files: &mut Vec<(String, Vec<u8>)>) -> (Section, Vec<String>) {
    let mut s = Section::new("Kestrel's render logs and reports");
    let mut found: Vec<(SystemTime, PathBuf)> = Vec::new();
    let mut dirs = super::renderlog::default_dirs();
    if let Some(d) = std::env::var_os(super::renderlog::DIR_VAR) {
        dirs.insert(0, PathBuf::from(d));
    }
    for d in &dirs {
        if let Ok(entries) = std::fs::read_dir(d) {
            for e in entries.flatten() {
                if let Some(t) = e.metadata().ok().and_then(|m| m.modified().ok()) {
                    found.push((t, e.path()));
                }
            }
        }
    }
    found.sort_by_key(|f| std::cmp::Reverse(f.0));
    let month = SystemTime::now() - Duration::from_secs(30 * 24 * 3600);
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut keep = Vec::new();
    let mut listed = String::new();
    let (mut logs, mut dumps) = (0, 0);
    for (t, p) in &found {
        let n = name(p);
        if let Some(m) = watch::midi_of(p) {
            keep.push(m);
        }
        let size = std::fs::metadata(p).map_or(0, |m| m.len());
        let is_log = n.ends_with(".log");
        let is_report = n.ends_with(" CRASH.txt") || n.ends_with(" HANG.txt");
        let is_dump = n.ends_with(".dmp");
        if !(is_log || is_report || is_dump) {
            continue;
        }
        if listed.lines().count() < 40 {
            listed.push_str(&format!("{n}  ({} KiB)\n", size.div_ceil(1024)));
        }
        let take = (is_log && logs < 10)
            || (is_report && *t > month)
            || (is_dump && dumps < 1 && *t > month && size <= 16 << 20);
        if take {
            if let Ok(data) = std::fs::read(p) {
                files.push((n, data));
                logs += is_log as usize;
                dumps += is_dump as usize;
            }
        }
    }
    s.item("folders", dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join("; "));
    s.item("added to this report", format!(
        "the newest {logs} logs, the crash and hang reports from the last 30 days, and {dumps} minidump"
    ));
    s.text = Some(if listed.is_empty() { "(no render logs yet)".into() } else { listed });
    (s, keep)
}

/// Run the administrator step and take in what it read: its listing as the \[12\]
fn administrator_step(exe: &Path, files: &mut Vec<(String, Vec<u8>)>) -> Section {
    use super::winsys::{run_elevated, Elevated};
    let mut s = Section::new("Windows' own GPU crash records (administrator)");
    let dir = std::env::temp_dir().join(format!("kestrel-system-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let args = format!("--force-cli falconeye-system --out \"{}\"", dir.display());
    match run_elevated(exe, &args, Duration::from_secs(120)) {
        Elevated::Exited(0) => {
            s.text = Some(std::fs::read_to_string(dir.join("system.txt")).unwrap_or_else(|e| format!("({e})")));
            if let Ok(entries) = std::fs::read_dir(dir.join("wer")) {
                for e in entries.flatten() {
                    if let Ok(data) = std::fs::read(e.path()) {
                        files.push((format!("wer/{}", e.file_name().to_string_lossy()), data));
                    }
                }
            }
        }
        Elevated::Exited(code) => s.item("error", format!("the administrator step ended with exit code {code:#x}")),
        Elevated::Declined => s.item("skipped", "permission was not given at Windows' prompt"),
        Elevated::TimedOut => s.item("error", "the administrator step did not finish within 2 minutes"),
        Elevated::Failed(e) => s.item("error", e),
    }
    let _ = std::fs::remove_dir_all(&dir);
    s
}

fn self_test_section(o: &selftest::Outcome) -> Section {
    let mut s = Section::new(&format!("GPU self-test: {} ({})", o.adapter, o.backend));
    let mut text = String::new();
    for st in &o.steps {
        text.push_str(&format!(
            "{:>10} voices ({} sounding): {:8.1} ms a block, {:8.1} ms spawning them, middle {:.1} ms{}\n",
            st.voices,
            st.live,
            st.worst_ms,
            st.spawn_ms,
            st.typical_ms,
            if st.timed_on_gpu { "" } else { " (timed on the host)" }
        ));
    }
    text.push_str(&format!("stopped: {}\n{}\n", o.stopped, o.verdict));
    s.text = Some(text);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smis_process_rows_are_counted() {
        let table = "\
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|=========================================================================================|
|    0   N/A  N/A            1234    C+G   ...\\Discord.exe                          N/A      |
|    0   N/A  N/A            5678      C   ...\\kestrel.exe                          N/A      |
+-----------------------------------------------------------------------------------------+";
        assert_eq!(count_processes(table), 2);
    }

    #[test]
    fn a_section_renders_as_aligned_lines() {
        let mut s = Section::new("Kestrel");
        s.item("version", "1.2.2");
        s.item("build", "dev");
        let t = render_text(&[s], "now");
        assert!(t.contains("== Kestrel ==\nversion  1.2.2\nbuild    dev\n"), "{t}");
        assert!(t.contains(HEADER));
    }
}
