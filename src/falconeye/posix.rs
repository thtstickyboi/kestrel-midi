// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What FalconEye reads on Linux and macOS, added in 1.2.3: how a render that \[1\]

#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

use super::report::Section;
use super::watch::tool;
use std::time::{Duration, SystemTime};

/// Lines of the system's records a report quotes, at most.
const RECORD_LINES: usize = 40;

// ---- Linux ----------------------------------------------------------------

/// The kernel log's lines that bear on a render with process id `pid`: its \[2\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn kernel_lines(text: &str, pid: u32) -> Vec<String> {
    let own = format!("[{pid}]:");
    let killed = format!("Killed process {pid} ");
    let named = format!("pid={pid},");
    let mut out: Vec<String> = text
        .lines()
        .filter_map(|l| {
            if l.contains(&own) || l.contains(&killed) || l.contains(&named) {
                Some(l.trim_end().to_string())
            } else if gpu_error(l) {
                Some(without_other_programs(l.trim_end()))
            } else {
                None
            }
        })
        .collect();
    let skip = out.len().saturating_sub(RECORD_LINES);
    out.drain(..skip);
    out
}

/// What stands in for a process that is not Kestrel.
const OTHER_PROGRAM: &str = "<other program>";
/// The process Kestrel runs as.
const OWN_PROGRAM: &str = "kestrel";

/// `line` with the name of every process in it that is not Kestrel blanked. \[3\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn without_other_programs(line: &str) -> String {
    let line = blank_after(line, "name=", &[','], "");
    let line = blank_after(&line, "process ", &[], " pid ");
    let line = blank_after(&line, "Process ", &[], " pid ");
    let line = blank_after(&line, "thread ", &[':'], ":");
    blank_before_pid(&line)
}

/// Every word that follows `prefix` -- up to a space or one of `stops` -- and is \[4\]
fn blank_after(line: &str, prefix: &str, stops: &[char], then: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = rest.find(prefix) {
        let (head, tail) = rest.split_at(at + prefix.len());
        out.push_str(head);
        let end = tail.find(|c: char| c.is_whitespace() || stops.contains(&c)).unwrap_or(tail.len());
        let (name, after) = tail.split_at(end);
        out.push_str(if !name.is_empty() && name != OWN_PROGRAM && after.starts_with(then) { OTHER_PROGRAM } else { name });
        rest = after;
    }
    out.push_str(rest);
    out
}

/// A process id in square brackets, `[12]`, takes the name before it with it: \[5\]
fn blank_before_pid(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 || !after[digits..].starts_with(']') {
            out.push_str(&rest[..=open]);
            rest = after;
            continue;
        }
        let head = &rest[..open];
        // `in otherprog [12]` has a space before the bracket, and only that form does.
        let before = head.trim_end_matches(' ');
        let spaced = before.len() < head.len() && {
            let start = before.rfind(|c: char| c.is_whitespace() || c == '[').map_or(0, |i| i + 1);
            start < before.len() && before[..start].ends_with(" in ")
        };
        let (keep, name, gap) = if spaced {
            let start = before.rfind(|c: char| c.is_whitespace() || c == '[').map_or(0, |i| i + 1);
            (&before[..start], &before[start..], " ")
        } else {
            let start = head.rfind(|c: char| c.is_whitespace() || c == '[').map_or(0, |i| i + 1);
            (&head[..start], &head[start..], "")
        };
        out.push_str(keep);
        out.push_str(if !name.is_empty() && name != OWN_PROGRAM { OTHER_PROGRAM } else { name });
        out.push_str(gap);
        out.push_str(&rest[open..open + 1 + digits + 1]);
        rest = &after[digits + 1..];
    }
    out.push_str(rest);
    out
}

/// A GPU driver's error in the kernel log: NVIDIA's Xid, or amdgpu, i915, \[6\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn gpu_error(l: &str) -> bool {
    const DRIVERS: [&str; 4] = ["amdgpu", "i915", "nouveau", " xe "];
    const TROUBLE: [&str; 5] = ["timeout", "reset", "fault", "hang", "HANG"];
    l.contains("NVRM: Xid")
        || l.contains("GPU HANG")
        || DRIVERS.iter().any(|d| l.contains(d)) && TROUBLE.iter().any(|t| l.contains(t))
}

/// How the kernel log says process `pid` died, if it says: the OOM killer, \[7\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn kernel_death(lines: &[String], pid: u32) -> Option<String> {
    let own = format!("[{pid}]:");
    if lines.iter().any(|l| l.contains(&format!("Killed process {pid} "))) {
        return Some(
            "was killed by Linux's out-of-memory killer: the machine ran out of RAM (the kernel log says so)"
                .into(),
        );
    }
    let line = lines.iter().find(|l| l.contains(&own) && (l.contains("segfault") || l.contains("trap")))?;
    // `kestrel[1234]: segfault at 10 ip ... error 6 in libfoo.so.1[7f00+2000]`
    let module = line
        .rsplit_once(" in ")
        .map(|(_, m)| m.split('[').next().unwrap_or(m).trim().to_string())
        .filter(|m| !m.is_empty());
    let kind = if line.contains("segfault") { "a segmentation fault" } else { "a CPU trap" };
    Some(match module {
        Some(m) => format!("crashed with {kind} in {m} (the kernel log says so)"),
        None => format!("crashed with {kind} (the kernel log says so)"),
    })
}

/// systemd-coredump's record of a crash, from `coredumpctl info <pid>`: the \[8\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn coredump(text: &str) -> Option<(String, String)> {
    let signal = text.lines().find_map(|l| l.trim().strip_prefix("Signal:").map(|s| s.trim().to_string()))?;
    let mut keep = Vec::new();
    let mut frames = 0;
    let mut in_stack = false;
    for line in text.lines() {
        let t = line.trim();
        if ["Signal:", "Timestamp:", "Executable:"].iter().any(|k| t.starts_with(k)) {
            keep.push(t.to_string());
        } else if t.starts_with("Stack trace of thread") {
            // The first thread's is the one that crashed.
            if frames > 0 {
                break;
            }
            in_stack = true;
            keep.push(t.to_string());
        } else if in_stack && t.starts_with('#') && frames < 16 {
            keep.push(format!("  {t}"));
            frames += 1;
        } else if in_stack && t.is_empty() && frames > 0 {
            break;
        }
    }
    Some((signal, keep.join("\n") + "\n"))
}

/// Whether the kernel hands core dumps to systemd, so `coredumpctl` will \[9\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn systemd_catches_cores(core_pattern: &str) -> bool {
    core_pattern.trim_start().starts_with('|') && core_pattern.contains("systemd-coredump")
}

/// `PRETTY_NAME` from `/etc/os-release`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn os_release_name(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix("PRETTY_NAME="))
        .map(|v| v.trim().trim_matches('"').to_string())
        .filter(|v| !v.is_empty())
}

/// The first `model name` in `/proc/cpuinfo`, or on ARM the `Hardware` or \[10\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn cpu_model(text: &str) -> Option<String> {
    for key in ["model name", "Hardware", "Model"] {
        let found = text.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            (k.trim() == key).then(|| v.trim().to_string())
        });
        if found.as_deref().is_some_and(|v| !v.is_empty()) {
            return found;
        }
    }
    None
}

/// A `/proc/meminfo` figure, in bytes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn meminfo(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.trim() != key {
            return None;
        }
        let kib: u64 = v.split_whitespace().next()?.parse().ok()?;
        Some(kib * 1024)
    })
}

/// The display adapters in `lspci -nn`: what follows each one's class.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn lspci_gpus(text: &str) -> Vec<String> {
    const CLASSES: [&str; 3] = ["VGA compatible controller", "3D controller", "Display controller"];
    text.lines()
        .filter(|l| CLASSES.iter().any(|c| l.contains(c)))
        .filter_map(|l| l.split_once("]: ").or_else(|| l.split_once(": ")).map(|(_, d)| d.trim().to_string()))
        .collect()
}

/// An NVIDIA card's total memory from `nvidia-smi \[11\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn nvidia_smi_total(csv: &str, vendor: u32, device: u32) -> Option<u64> {
    let want = (device << 16) | vendor;
    csv.lines().find_map(|l| {
        let (id, mib) = l.split_once(',')?;
        let id = u32::from_str_radix(id.trim().trim_start_matches("0x").trim_start_matches("0X"), 16).ok()?;
        (id == want).then(|| mib.trim().parse::<u64>().ok().map(|m| m << 20))?
    })
}

// ---- macOS ------------------------------------------------------------------

/// A crash report Apple wrote (`.ips`, macOS 12 on): a line of JSON about \[12\]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn ips(text: &str) -> Option<(u32, String, String)> {
    let (_, body) = text.split_once('\n')?;
    let v: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let pid = v.get("pid")?.as_u64()? as u32;
    let exception = v.get("exception");
    let field = |o: Option<&serde_json::Value>, k: &str| o.and_then(|e| e.get(k)).and_then(|x| x.as_str()).map(str::to_string);
    let kind = field(exception, "type").unwrap_or_else(|| "an exception".into());
    let mut what = kind.clone();
    if let Some(sig) = field(exception, "signal") {
        what.push_str(&format!(" ({sig})"));
    }
    if let Some(sub) = field(exception, "subtype") {
        what.push_str(&format!(", {sub}"));
    }
    let images: Vec<String> = v
        .get("usedImages")
        .and_then(|i| i.as_array())
        .map(|a| a.iter().map(|i| i.get("name").and_then(|n| n.as_str()).unwrap_or("?").to_string()).collect())
        .unwrap_or_default();
    let crashed = v.get("faultingThread").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
    let frames: Vec<String> = v
        .get("threads")
        .and_then(|t| t.get(crashed))
        .and_then(|t| t.get("frames"))
        .and_then(|f| f.as_array())
        .map(|a| {
            a.iter()
                .take(12)
                .enumerate()
                .map(|(n, f)| {
                    let image = f
                        .get("imageIndex")
                        .and_then(|i| i.as_u64())
                        .and_then(|i| images.get(i as usize))
                        .map_or("?", String::as_str);
                    let offset = f.get("imageOffset").and_then(|o| o.as_u64()).unwrap_or(0);
                    let symbol = f.get("symbol").and_then(|s| s.as_str()).map(|s| format!(" {s}")).unwrap_or_default();
                    format!("  #{n} {image}+{offset:#x}{symbol}")
                })
                .collect()
        })
        .unwrap_or_default();
    if let Some(top) = frames.first() {
        if let Some(image) = top.split_whitespace().nth(1).and_then(|f| f.split('+').next()) {
            what.push_str(&format!(" in {image}"));
        }
    }
    let mut excerpt = format!("Exception: {what}\n");
    if let Some(t) = v.get("termination") {
        let ns = field(Some(t), "namespace").unwrap_or_default();
        let ind = field(Some(t), "indicator").unwrap_or_default();
        if !ns.is_empty() || !ind.is_empty() {
            excerpt.push_str(&format!("Termination: {ns} {ind}\n"));
        }
    }
    if !frames.is_empty() {
        excerpt.push_str("Crashed thread:\n");
        excerpt.push_str(&frames.join("\n"));
        excerpt.push('\n');
    }
    Some((pid, what, excerpt))
}

/// The GPU's lines in `system_profiler SPDisplaysDataType`. Displays' serial \[13\]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn mac_gpus(text: &str) -> Vec<String> {
    const KEYS: [&str; 7] =
        ["Chipset Model:", "Type:", "Bus:", "VRAM", "Total Number of Cores:", "Vendor:", "Metal"];
    text.lines()
        .map(str::trim)
        .filter(|l| KEYS.iter().any(|k| l.starts_with(k)))
        .map(str::to_string)
        .collect()
}

/// Where the power comes from, and the battery's charge, from `pmset -g \[14\]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn pmset(text: &str) -> Option<String> {
    let mut lines = text.lines();
    let source = lines.next()?.split('\'').nth(1)?.to_string();
    let charge = lines.find_map(|l| l.split(['\t', ' ', ';']).find(|w| w.ends_with('%')).map(str::to_string));
    Some(match charge {
        Some(c) => format!("{source}, battery {c}"),
        None => format!("{source}, no battery"),
    })
}

// ---- the watcher's side ------------------------------------------------------

/// The Unix time `secs` ago, for the logs' `--since` and `--start`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn epoch_ago(secs: u64) -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
        .saturating_sub(secs)
}

/// How render `pid` died, when the system wrote it down, and the records to \[15\]
#[cfg(target_os = "linux")]
pub(crate) fn death(pid: u32, window: Duration) -> (Option<String>, String) {
    let mut records = String::new();
    let mut what = None;
    let core_pattern = std::fs::read_to_string("/proc/sys/kernel/core_pattern").unwrap_or_default();
    if systemd_catches_cores(&core_pattern) {
        // [16]
        let pid_s = pid.to_string();
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(20) {
            if let Ok(text) = tool("coredumpctl", &["--no-pager", "-q", "info", &pid_s], Duration::from_secs(15)) {
                if let Some((signal, excerpt)) = coredump(&text) {
                    what = Some(format!("died of signal {signal}, a crash (systemd-coredump recorded it)"));
                    records.push_str("-- systemd-coredump --\n");
                    records.push_str(&excerpt);
                    break;
                }
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    let kernel = kernel_log(window);
    let lines = match &kernel {
        Ok(text) => kernel_lines(text, pid),
        Err(_) => Vec::new(),
    };
    // The kernel's line names the module, which systemd's summary does not.
    if let Some(k) = kernel_death(&lines, pid) {
        what = Some(match what {
            Some(w) if k.contains(" in ") => format!("{w}; {k}"),
            Some(w) => w,
            None => k,
        });
    }
    records.push_str("-- The kernel log: this render, the out-of-memory killer, and GPU driver errors --\n");
    match kernel {
        Ok(_) if lines.is_empty() => records.push_str("(none)\n"),
        Ok(_) => {
            records.push_str(&lines.join("\n"));
            records.push('\n');
        }
        Err(e) => records.push_str(&format!("({e})\n")),
    }
    (what, records)
}

/// The kernel log from `window` ago: `journalctl -k`, or `dmesg` where there \[17\]
#[cfg(target_os = "linux")]
fn kernel_log(window: Duration) -> Result<String, String> {
    let since = format!("--since=@{}", epoch_ago(window.as_secs()));
    match tool("journalctl", &["-k", "-q", "--no-pager", "-o", "short-iso", &since], Duration::from_secs(20)) {
        Ok(text) if !text.trim().is_empty() => Ok(text),
        journal => match tool("dmesg", &["-T"], Duration::from_secs(15)) {
            Ok(text) if !text.trim().is_empty() => Ok(text),
            dmesg => Err(format!(
                "the kernel log could not be read: journalctl {}, dmesg {}",
                journal.err().unwrap_or_else(|| "said nothing".into()),
                dmesg.err().unwrap_or_else(|| "said nothing (it may need permission)".into())
            )),
        },
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn death(pid: u32, window: Duration) -> (Option<String>, String) {
    let mut records = String::new();
    let mut what = None;
    // ReportCrash writes the report a few seconds after the process dies.
    let t0 = std::time::Instant::now();
    while t0.elapsed() < Duration::from_secs(20) {
        if let Some((w, excerpt)) = mac_crash_report(pid, window) {
            what = Some(format!("died of {w} (Apple's crash report says so)"));
            records.push_str("-- Apple's crash report --\n");
            records.push_str(&excerpt);
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    records.push_str("-- The system log: this render's errors, and the GPU --\n");
    records.push_str(&mac_log(pid, window));
    (what, records)
}

/// The newest crash report about process `pid`, written in the last \[18\]
#[cfg(target_os = "macos")]
fn mac_crash_report(pid: u32, window: Duration) -> Option<(String, String)> {
    let dir = std::path::PathBuf::from(std::env::var_os("HOME")?).join("Library/Logs/DiagnosticReports");
    let recent = SystemTime::now().checked_sub(window)?;
    std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        let name = e.file_name().to_string_lossy().to_ascii_lowercase();
        if !name.starts_with("kestrel") || !name.ends_with(".ips") {
            return None;
        }
        if e.metadata().ok()?.modified().ok()? < recent {
            return None;
        }
        let (p, what, excerpt) = ips(&std::fs::read_to_string(e.path()).ok()?)?;
        (p == pid).then_some((what, excerpt))
    })
}

#[cfg(target_os = "macos")]
fn mac_log(pid: u32, window: Duration) -> String {
    let start = chrono::Local::now() - chrono::Duration::seconds(window.as_secs() as i64);
    let start = start.format("%Y-%m-%d %H:%M:%S").to_string();
    let predicate =
        format!("(processID == {pid} AND messageType >= 16) OR (process == \"kernel\" AND eventMessage CONTAINS[c] \"gpu\")");
    match tool("log", &["show", "--style", "compact", "--start", &start, "--predicate", &predicate], Duration::from_secs(40)) {
        Ok(text) => {
            let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with("Timestamp")).collect();
            if lines.is_empty() {
                "(none)\n".into()
            } else {
                lines[lines.len().saturating_sub(RECORD_LINES)..].join("\n") + "\n"
            }
        }
        Err(e) => format!("({e})\n"),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn death(_pid: u32, _window: Duration) -> (Option<String>, String) {
    (None, "(not collected on this platform)\n".into())
}

/// What the machine report quotes of the kernel log: every GPU driver error, with \[19\]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn history_lines(text: &str) -> Vec<String> {
    kernel_lines(text, u32::MAX)
        .into_iter()
        .chain(text.lines().filter(|l| l.contains("Killed process") && l.contains("(kestrel)")).map(str::to_string))
        .collect()
}

/// The GPU drivers' errors and Kestrel's crashes over the last `days`, for \[20\]
#[cfg(target_os = "linux")]
pub(crate) fn history(days: u64) -> String {
    let mut out = String::from("-- The kernel log: GPU driver errors and the out-of-memory killer --\n");
    match kernel_log(Duration::from_secs(days * 86_400)) {
        Ok(text) => {
            let lines = history_lines(&text);
            out.push_str(&if lines.is_empty() { "(none)\n".into() } else { lines.join("\n") + "\n" });
        }
        Err(e) => out.push_str(&format!("({e})\n")),
    }
    out.push_str("-- systemd-coredump: Kestrel's crashes --\n");
    let since = format!("--since=@{}", epoch_ago(days * 86_400));
    match tool("coredumpctl", &["--no-pager", "-q", "list", &since, "kestrel"], Duration::from_secs(20)) {
        Ok(text) if !text.trim().is_empty() => out.push_str(&text),
        Ok(_) => out.push_str("(none, or systemd does not catch core dumps here)\n"),
        Err(e) => out.push_str(&format!("({e})\n")),
    }
    out
}

#[cfg(target_os = "macos")]
pub(crate) fn history(days: u64) -> String {
    let mut out = String::from("-- Apple's crash reports for Kestrel --\n");
    let Some(home) = std::env::var_os("HOME") else {
        return out + "(no home folder)\n";
    };
    let dir = std::path::PathBuf::from(home).join("Library/Logs/DiagnosticReports");
    let recent = SystemTime::now().checked_sub(Duration::from_secs(days * 86_400)).unwrap_or(SystemTime::UNIX_EPOCH);
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().to_ascii_lowercase();
                    n.starts_with("kestrel") && n.ends_with(".ips")
                        && e.metadata().and_then(|m| m.modified()).is_ok_and(|t| t >= recent)
                })
                .filter_map(|e| {
                    let (_, what, _) = ips(&std::fs::read_to_string(e.path()).ok()?)?;
                    Some(format!("{}: {what}", e.file_name().to_string_lossy()))
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    out.push_str(&if found.is_empty() { "(none)\n".into() } else { found.join("\n") + "\n" });
    out
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn history(_days: u64) -> String {
    "(not collected on this platform)\n".into()
}

// ---- the machine report's section -------------------------------------------

#[cfg(target_os = "linux")]
pub(crate) fn os_section() -> Section {
    let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
    let mut s = Section::new("Linux");
    s.item("Distribution", os_release_name(&read("/etc/os-release")).unwrap_or_else(|| "?".into()));
    s.item("Kernel", read("/proc/sys/kernel/osrelease").trim().to_string());
    let dmi = |f: &str| read(&format!("/sys/devices/virtual/dmi/id/{f}")).trim().to_string();
    s.item("Machine", format!("{} {}", dmi("sys_vendor"), dmi("product_name")).trim().to_string());
    s.item("CPU", format!(
        "{}, {} threads",
        cpu_model(&read("/proc/cpuinfo")).unwrap_or_else(|| "?".into()),
        std::thread::available_parallelism().map_or(0, |n| n.get())
    ));
    let mem = read("/proc/meminfo");
    let gib = |b: Option<u64>| b.map_or("?".into(), |b| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64));
    s.item("RAM", format!("{}, {} available", gib(meminfo(&mem, "MemTotal")), gib(meminfo(&mem, "MemAvailable"))));
    s.item("Swap", format!("{}, {} free", gib(meminfo(&mem, "SwapTotal")), gib(meminfo(&mem, "SwapFree"))));
    match tool("lspci", &["-nn"], Duration::from_secs(15)) {
        Ok(text) => {
            for gpu in lspci_gpus(&text) {
                s.item("Display adapter", gpu);
            }
        }
        Err(e) => s.item("Display adapter", e),
    }
    if let Some(v) = read("/proc/driver/nvidia/version").lines().next() {
        s.item("NVIDIA driver", v.trim().to_string());
    }
    for (k, var) in [("Session", "XDG_SESSION_TYPE"), ("Desktop", "XDG_CURRENT_DESKTOP")] {
        if let Ok(v) = std::env::var(var) {
            s.item(k, v);
        }
    }
    let mut power = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/sys/class/power_supply") {
        for e in rd.flatten() {
            let p = e.path();
            let f = |n: &str| std::fs::read_to_string(p.join(n)).unwrap_or_default().trim().to_string();
            match f("type").as_str() {
                "Mains" => power.push(if f("online") == "1" { "plugged in".to_string() } else { "on battery".to_string() }),
                "Battery" => power.push(format!("battery {}%", f("capacity"))),
                _ => {}
            }
        }
    }
    s.item("Power", if power.is_empty() { "no battery or mains reported".into() } else { power.join(", ") });
    s
}

#[cfg(target_os = "macos")]
pub(crate) fn os_section() -> Section {
    let mut s = Section::new("macOS");
    let run = |p: &str, a: &[&str]| tool(p, a, Duration::from_secs(20)).map(|t| t.trim().to_string());
    s.item("macOS", format!(
        "{} ({})",
        run("sw_vers", &["-productVersion"]).unwrap_or_else(|e| e),
        run("sw_vers", &["-buildVersion"]).unwrap_or_default()
    ));
    s.item("Machine", run("sysctl", &["-n", "hw.model"]).unwrap_or_else(|e| e));
    s.item("CPU", format!(
        "{}, {} threads",
        run("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_else(|e| e),
        std::thread::available_parallelism().map_or(0, |n| n.get())
    ));
    if let Ok(bytes) = run("sysctl", &["-n", "hw.memsize"]) {
        if let Ok(b) = bytes.parse::<u64>() {
            s.item("RAM", format!("{:.1} GiB, shared with the GPU on Apple silicon", b as f64 / (1u64 << 30) as f64));
        }
    }
    match run("system_profiler", &["SPDisplaysDataType"]) {
        Ok(text) => {
            for l in mac_gpus(&text) {
                s.item("GPU", l);
            }
        }
        Err(e) => s.item("GPU", e),
    }
    if let Some(p) = run("pmset", &["-g", "batt"]).ok().as_deref().and_then(pmset) {
        s.item("Power", p);
    }
    s
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(crate) fn os_section() -> Section {
    let mut s = Section::new("Operating system");
    s.item("note", "not collected on this platform");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "\
2026-09-27T10:01:02+0000 host kernel: NVRM: Xid (PCI:0000:01:00): 79, pid=4242, name=kestrel, GPU has fallen off the bus.
2026-09-27T10:01:03+0000 host kernel: kestrel[4242]: segfault at 10 ip 00007f3a1c2d3e4f sp 00007ffd1234 error 6 in libnvidia-glcore.so.550.54.14[7f3a1a000000+2400000] likely on CPU 3
2026-09-27T10:01:04+0000 host kernel: usb 1-2: new high-speed USB device
2026-09-27T10:01:05+0000 host kernel: [drm] amdgpu kernel modesetting enabled.
2026-09-27T10:01:06+0000 host kernel: other[99]: segfault at 0 ip 0 sp 0 error 4 in other[400000+1000]
2026-09-27T10:01:07+0000 host kernel: amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=3
";

    #[test]
    fn the_kernel_log_gives_this_renders_crash_and_the_gpu_errors_only() {
        let lines = kernel_lines(KERNEL, 4242);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[0].contains("Xid") && lines[1].contains("kestrel[4242]") && lines[2].contains("ring gfx"));
        assert_eq!(
            kernel_death(&lines, 4242).unwrap(),
            "crashed with a segmentation fault in libnvidia-glcore.so.550.54.14 (the kernel log says so)"
        );
        assert!(kernel_death(&kernel_lines(KERNEL, 7), 7).is_none());
        let oom = "kernel: Out of memory: Killed process 4242 (kestrel) total-vm:40000000kB, anon-rss:30000000kB\n";
        assert!(kernel_death(&kernel_lines(oom, 4242), 4242).unwrap().contains("out-of-memory killer"));
    }

    /// The line in a Linux user's report (2026-10-04), exactly, with the machine's \[21\]
    const XID_OF_ANOTHER_PROGRAM: &str = "2026-10-04T14:16:55+03:00 <pc> kernel: NVRM: Xid (PCI:0000:01:00): 69, pid=57910, name=other-program, Class Error: channel 0x00000060, Class 0000902d, Offset 0000023c, Data 00000000, ErrorCode 00000004";

    /// The machine report says it contains no other program's name, and it quoted \[22\]
    #[test]
    fn the_report_does_not_name_a_program_that_is_not_kestrel() {
        let text = format!(
            "{XID_OF_ANOTHER_PROGRAM}\n\
             2026-10-04T14:20:00+03:00 <pc> kernel: NVRM: Xid (PCI:0000:01:00): 79, pid=4242, name=kestrel, GPU has fallen off the bus.\n\
             2026-10-04T14:21:00+03:00 <pc> kernel: amdgpu 0000:03:00.0: amdgpu: Process information: process otherprog pid 1234 thread otherprog:cs0 pid 1240\n\
             2026-10-04T14:21:01+03:00 <pc> kernel: amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=3\n\
             2026-10-04T14:22:00+03:00 <pc> kernel: i915 0000:00:02.0: [drm] GPU HANG: ecode 9:1:85dffffb, in otherprog [2310]\n\
             2026-10-04T14:22:01+03:00 <pc> kernel: i915 0000:00:02.0: [drm] otherprog[2310] context reset due to GPU hang\n\
             2026-10-04T14:23:00+03:00 <pc> kernel: nouveau 0000:01:00.0: fifo: channel 3 [otherprog[2310]] killed! (a fault)\n\
             2026-10-04T14:24:00+03:00 <pc> kernel: Out of memory: Killed process 4242 (kestrel) total-vm:1kB\n\
             2026-10-04T14:24:01+03:00 <pc> kernel: Out of memory: Killed process 77 (otherprog) total-vm:1kB\n"
        );
        let lines = history_lines(&text);
        let all = lines.join("\n");
        assert!(!all.contains("otherprog"), "another program's name is in the report:\n{all}");
        // [23]
        assert!(lines[0].contains("Xid (PCI:0000:01:00): 69, pid=57910, name=<other program>, Class Error"), "{}", lines[0]);
        assert!(all.contains("name=kestrel, GPU has fallen off the bus"));
        // [24]
        let amd = without_other_programs(
            "amdgpu 0000:03:00.0: amdgpu: Process information: process otherprog pid 1234 thread otherprog:cs0 pid 1240",
        );
        assert!(!amd.contains("otherprog") && amd.contains("Process information:"), "{amd}");
        assert!(amd.contains("process <other program> pid 1234 thread <other program>:cs0 pid 1240"), "{amd}");
        assert!(all.contains("ring gfx_0.0.0 timeout"));
        assert!(all.contains("in <other program> [2310]") && all.contains("<other program>[2310] context reset"), "{all}");
        assert!(all.contains("[<other program>[2310]] killed!"), "{all}");
        assert!(all.contains("Killed process 4242 (kestrel)"));
    }

    /// A render's own lines are not touched, and neither is what is not a name.
    #[test]
    fn a_renders_own_lines_and_the_brackets_that_are_not_pids_are_left_alone() {
        let own = without_other_programs("kernel: kestrel[4242]: segfault at 10 in libnvidia-glcore.so.550.54.14[7f3a1a000000+2400000]");
        assert!(own.contains("kestrel[4242]") && own.contains("[7f3a1a000000+2400000]"), "{own}");
        let same = "amdgpu 0000:03:00.0: [drm] *ERROR* [12345.678901] ring gfx timeout [drm]";
        assert_eq!(without_other_programs(same), same);
        // A GPU error that carries the render's pid is its own, whatever the name.
        let lines = kernel_lines("kernel: NVRM: Xid (PCI:0000:01:00): 79, pid=99, name=renamed-exe, fell off\n", 99);
        assert!(lines[0].contains("name=renamed-exe"), "{lines:?}");
    }

    #[test]
    fn coredumpctl_gives_the_signal_and_the_crashed_stack_and_not_the_machine() {
        let text = "\
           PID: 4242 (kestrel)
           UID: 1000 (someone)
        Signal: 11 (SEGV)
     Timestamp: Sat 2026-09-27 10:01:03 UTC (5s ago)
  Command Line: /home/someone/kestrel --force-cli render song.mid
    Executable: /home/someone/kestrel
      Hostname: someones-pc
       Boot ID: 0123456789abcdef
       Message: Process 4242 (kestrel) of user 1000 dumped core.

                Stack trace of thread 4250:
                #0  0x00007f3a1c2d3e4f n/a (libnvidia-glcore.so.550.54.14 + 0x2d3e4f)
                #1  0x000055d1a0b0c0d0 n/a (kestrel + 0x30c0d0)

                Stack trace of thread 4242:
                #0  0x00007f3a1b000000 n/a (libc.so.6 + 0x9a000)
";
        let (signal, excerpt) = coredump(text).unwrap();
        assert_eq!(signal, "11 (SEGV)");
        assert!(excerpt.contains("libnvidia-glcore") && excerpt.contains("kestrel + 0x30c0d0"), "{excerpt}");
        for gone in ["someones-pc", "Boot ID", "UID", "Command Line", "libc.so.6"] {
            assert!(!excerpt.contains(gone), "{gone} in {excerpt}");
        }
        assert!(coredump("No coredumps found.\n").is_none());
        assert!(systemd_catches_cores("|/usr/lib/systemd/systemd-coredump %P %u %g %s %t %c %h\n"));
        assert!(!systemd_catches_cores("|/usr/share/apport/apport -p%p -s%s -c%c -d%d -P%P -u%u -g%g -- %E\n"));
        assert!(!systemd_catches_cores("core\n"));
    }

    #[test]
    fn linux_system_files_read_as_the_report_wants_them() {
        assert_eq!(os_release_name("NAME=\"Ubuntu\"\nPRETTY_NAME=\"Ubuntu 24.04.1 LTS\"\n").unwrap(), "Ubuntu 24.04.1 LTS");
        assert_eq!(cpu_model("processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\n").unwrap(), "AMD Ryzen 9 7950X 16-Core Processor");
        assert_eq!(meminfo("MemTotal:       65536000 kB\nMemAvailable:   1024 kB\n", "MemTotal"), Some(65_536_000 * 1024));
        assert_eq!(meminfo("MemTotal: 1 kB\n", "SwapTotal"), None);
        let lspci = "00:02.0 VGA compatible controller [0300]: Intel Corporation Raptor Lake-S UHD Graphics [8086:a788] (rev 04)\n\
                     01:00.0 3D controller [0302]: NVIDIA Corporation GA102 [GeForce RTX 3090] [10de:2204] (rev a1)\n\
                     00:1f.3 Audio device [0403]: Intel Corporation Device [8086:7a50]\n";
        assert_eq!(
            lspci_gpus(lspci),
            ["Intel Corporation Raptor Lake-S UHD Graphics [8086:a788] (rev 04)", "NVIDIA Corporation GA102 [GeForce RTX 3090] [10de:2204] (rev a1)"]
        );
        let smi = "0x220410DE, 24576\n0x2D5910DE, 8151\n";
        assert_eq!(nvidia_smi_total(smi, 0x10de, 0x2d59), Some(8151 << 20));
        assert_eq!(nvidia_smi_total(smi, 0x10de, 0x1234), None);
    }

    #[test]
    fn an_apple_crash_report_gives_the_exception_and_the_crashed_frames() {
        let report = r#"{"app_name":"kestrel","timestamp":"2026-09-27 10:01:03.00 +0000","bug_type":"309","name":"kestrel"}
{
  "pid" : 4242,
  "procName" : "kestrel",
  "crashReporterKey" : "should-not-appear",
  "exception" : {"codes":"0x1, 0x10","type":"EXC_BAD_ACCESS","signal":"SIGSEGV","subtype":"KERN_INVALID_ADDRESS at 0x0000000000000010"},
  "termination" : {"namespace":"SIGNAL","indicator":"Segmentation fault: 11"},
  "faultingThread" : 1,
  "threads" : [
    {"frames":[{"imageOffset":100,"imageIndex":0}]},
    {"frames":[{"imageOffset":4096,"imageIndex":1,"symbol":"AGXMetalG14X::draw"},{"imageOffset":32,"imageIndex":0}]}
  ],
  "usedImages" : [{"name":"kestrel"},{"name":"AGXMetalG14X"}]
}"#;
        let (pid, what, excerpt) = ips(report).unwrap();
        assert_eq!(pid, 4242);
        assert_eq!(what, "EXC_BAD_ACCESS (SIGSEGV), KERN_INVALID_ADDRESS at 0x0000000000000010 in AGXMetalG14X");
        assert!(excerpt.contains("#0 AGXMetalG14X+0x1000 AGXMetalG14X::draw") && excerpt.contains("#1 kestrel+0x20"), "{excerpt}");
        assert!(excerpt.contains("Termination: SIGNAL Segmentation fault: 11"));
        assert!(!excerpt.contains("should-not-appear"));
        assert!(ips("not a report").is_none());
    }

    #[test]
    fn macos_tools_read_as_the_report_wants_them() {
        let sp = "Graphics/Displays:\n\n    Apple M2 Pro:\n\n      Chipset Model: Apple M2 Pro\n      Type: GPU\n      Bus: Built-In\n      \
                  Total Number of Cores: 19\n      Vendor: Apple (0x106b)\n      Metal Support: Metal 3\n      Displays:\n        Color LCD:\n          \
                  Display Serial Number: ABC123\n";
        assert_eq!(
            mac_gpus(sp),
            ["Chipset Model: Apple M2 Pro", "Type: GPU", "Bus: Built-In", "Total Number of Cores: 19", "Vendor: Apple (0x106b)", "Metal Support: Metal 3"]
        );
        let batt = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1234567)\t85%; charging; 1:02 remaining present: true\n";
        assert_eq!(pmset(batt).unwrap(), "AC Power, battery 85%");
        assert_eq!(pmset("Now drawing from 'AC Power'\n").unwrap(), "AC Power, no battery");
    }
}
