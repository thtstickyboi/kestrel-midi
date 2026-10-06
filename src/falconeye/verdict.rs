// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What a crash or hang report can say about itself. \[1\]

/// Windows' own words for a driver reset, in the System log.
const TDR_EVENT: &str = "Event ID: 4101";
/// `LiveKernelEvent` codes Windows files a GPU timeout under: 141 is the \[2\]
const TDR_CODES: [&str; 2] = ["P1: 141", "P1: 117"];

/// A paragraph for a report, when the log and the records show the graphics \[3\]
pub fn gpu_reset(log_tail: &str, records: &str) -> Option<String> {
    let lost = log_tail.contains("gpu device lost");
    let event = records.contains(TDR_EVENT);
    let watchdog = TDR_CODES.iter().any(|c| records.contains(c));
    if !(lost || event || watchdog) {
        return None;
    }

    let mut seen: Vec<&str> = Vec::new();
    if lost {
        seen.push("the render's log says the GPU device was lost");
    }
    if event {
        seen.push("Windows logged that the display driver stopped responding and recovered (Display, event 4101)");
    }
    if watchdog {
        seen.push("Windows filed a GPU watchdog report (a LiveKernelEvent, code 141 or 117)");
    }

    let mut out = format!(
        "The graphics driver was reset: {}. Windows does that when one piece of GPU work \
         runs past 2 seconds (its timeout detection), and the render's thread then finds its \
         device gone.",
        join(&seen)
    );
    if let Some(last) = last_progress(log_tail) {
        out.push_str(&format!(
            " The last progress line has {} voices live and a longest device wait of {} ms",
            thousands(last.voices),
            last.wait_ms.round() as u64
        ));
        out.push_str(if last.wait_ms >= 1000.0 {
            ", which is a good part of that 2 seconds."
        } else {
            "."
        });
    }
    if let Some(module) = faulting_module(records).filter(|m| is_graphics_driver(m)) {
        out.push_str(&format!(
            " The process itself ended inside {module}, the graphics driver, which is why the \
             render printed no message: a driver can abort the process once its device is lost."
        ));
    }
    out.push_str(
        "\nWhat helps: fewer voices (--max-voices; the wait grows with the voices live), \
         --block 1024 (shorter pieces of work, about 15% slower), closing other programs that \
         use the GPU, and a newer graphics driver. Kestrel cuts a block that covers a great many \
         voices into several submissions on its own; a log line saying a block \"goes up as\" \
         several submissions shows it was doing that.",
    );
    Some(out)
}

fn join(parts: &[&str]) -> String {
    match parts {
        [] => String::new(),
        [one] => (*one).into(),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

struct Progress {
    voices: u64,
    wait_ms: f64,
}

/// The newest `AT ... voices live ... longest device wait N ms` line.
fn last_progress(log_tail: &str) -> Option<Progress> {
    log_tail.lines().rev().find_map(|l| {
        let (_, after) = l.split_once("| ")?;
        let voices = after.split_once(" voices live")?.0.trim().parse().ok()?;
        let wait = l.split_once("longest device wait ")?.1;
        let wait_ms = wait.split_whitespace().next()?.parse().ok()?;
        Some(Progress { voices, wait_ms })
    })
}

/// `Faulting module name: nvoglv64.dll, version: ...` from an Application \[4\]
fn faulting_module(records: &str) -> Option<String> {
    let rest = records.split_once("Faulting module name: ")?.1;
    let name = rest.split(',').next()?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// A user-mode graphics driver, by the names the vendors give them.
fn is_graphics_driver(module: &str) -> bool {
    let m = module.to_ascii_lowercase();
    ["nvoglv", "nvwgf2", "nvd3dum", "nvlddmkm", "amdvlk", "atidxx", "atiu", "aticfx", "amdxc", "ig9icd", "igvk", "igd10", "igd12", "igdumd", "igxelp"]
        .iter()
        .any(|p| m.starts_with(p))
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    // Digits are ASCII, so every chunk is valid UTF-8.
    let groups: Vec<&str> = s.as_bytes().rchunks(3).rev().filter_map(|c| std::str::from_utf8(c).ok()).collect();
    groups.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tail of the 2026-10-03 report, shortened.
    const LOG: &str = "\
[  268.212] AT    130.90s audio, block 1534 | 15000000 voices live | last block: want 3560104 take 3560104 stolen 0 | 0 dropped in all | longest device wait 1402.1 ms | vram 7118 of 16107 MiB in use
[  270.438] AT    130.99s audio, block 1535 | 15000000 voices live | last block: want 4111816 take 3750000 stolen 0 | 361816 dropped in all | longest device wait 1732.5 ms | vram 7118 of 16107 MiB in use
[  270.660] ERROR kestrel::gpu::device: gpu device lost: Device is lost
";
    const RECORDS: &str = "\
-- System log: the graphics driver and the display --
Event[0]:
  Log Name: System
  Source: Display
  Event ID: 4101
Display driver nvlddmkm stopped responding and has successfully recovered.
-- Application log: kestrel.exe's errors and hangs, and GPU resets --
Event[0]:
  Event Name: LiveKernelEvent
Problem signature:
P1: 141
Event[3]:
  Source: Application Error
  Event ID: 1000
Faulting application name: kestrel.exe, version: 1.2.3.0, time stamp: 0x6ab8cae1
Faulting module name: nvoglv64.dll, version: 32.0.15.9636, time stamp: 0x69e98155
Exception code: 0xc0000409
";

    #[test]
    fn the_report_that_started_it_is_named_a_driver_reset() {
        let v = gpu_reset(LOG, RECORDS).expect("a reset");
        assert!(v.starts_with("The graphics driver was reset: the render's log says"), "{v}");
        assert!(v.contains("event 4101") && v.contains("watchdog report"), "{v}");
        assert!(v.contains("15,000,000 voices live") && v.contains("1733 ms"), "{v}");
        assert!(v.contains("inside nvoglv64.dll"), "{v}");
        assert!(v.contains("--max-voices") && v.contains("--block 1024"), "{v}");
        assert!(!v.contains("  "), "runs of spaces: {v}");
    }

    #[test]
    fn each_piece_of_evidence_is_enough_alone_and_is_the_only_one_named() {
        let only_log = gpu_reset(LOG, "").unwrap();
        assert!(only_log.contains("the render's log says") && !only_log.contains("4101"), "{only_log}");
        assert!(!only_log.contains("inside "), "no module was in the records: {only_log}");
        let only_event = gpu_reset("", RECORDS).unwrap();
        assert!(only_event.contains("event 4101") && !only_event.contains("log says"), "{only_event}");
        assert!(!only_event.contains("longest device wait"), "no progress line to quote: {only_event}");
    }

    #[test]
    fn a_crash_of_another_kind_is_not_called_a_reset() {
        assert!(gpu_reset("[ 1.0] AT  1.00s audio, block 3 | 10 voices live | longest device wait 3.0 ms\n", "").is_none());
        // [5]
        let stale = "Event Name: LiveKernelEvent\nProblem signature:\nP1: 1a8\nP1: 1b8\n";
        assert!(gpu_reset("", stale).is_none());
    }

    #[test]
    fn a_fault_in_something_else_is_not_blamed_on_the_driver() {
        let records = "Event ID: 4101\nFaulting module name: ntdll.dll, version: 10.0\n";
        let v = gpu_reset("", records).unwrap();
        assert!(!v.contains("inside "), "{v}");
        for dll in ["nvoglv64.dll", "NVWGF2UMX.dll", "amdvlk64.dll", "atidxx64.dll", "igvk64.dll"] {
            assert!(is_graphics_driver(dll), "{dll}");
        }
        assert!(!is_graphics_driver("kestrel.exe") && !is_graphics_driver("KERNELBASE.dll"));
    }

    #[test]
    fn numbers_are_grouped() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(14_810_232), "14,810,232");
    }
}
