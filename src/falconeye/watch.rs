// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! FalconEye's watcher: a second Kestrel process that watches a render from \[1\]

use crate::falconeye::redact::Redactor;
use crate::falconeye::winsys::{self, Process};
use anyhow::Result;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// No block finished for this long while rendering: hung. Windows resets a \[2\]
pub const HANG_SECS: u64 = 60;
/// No heartbeat for this long with the pipe still open: the whole process has \[3\]
pub const SILENT_SECS: u64 = 30;
/// Lines of the render log a report quotes.
const TAIL_LINES: usize = 60;

enum Msg {
    Line(String),
    Eof,
}

/// Watch the render `pid`, whose log is `log`, until it ends.
pub fn run(log: &Path, pid: u32) -> Result<()> {
    // [4]
    let process = Process::open(pid);
    let started = Instant::now();

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            match line {
                Ok(l) => {
                    if tx.send(Msg::Line(l)).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(Msg::Eof);
    });

    let mut rendering = false;
    let mut blocks = 0u64;
    let mut last_beat = Instant::now();
    let mut last_progress = Instant::now();
    let mut hang_reported = false;
    let mut silence_reported = false;
    let mut crash_dump: Option<PathBuf> = None;
    let mut hang_dump: Option<PathBuf> = None;

    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Msg::Line(l)) => {
                let mut words = l.split_whitespace();
                match words.next() {
                    Some("beat") => {
                        last_beat = Instant::now();
                        silence_reported = false;
                        rendering = words.next() == Some("1");
                        let b: u64 = words.next().and_then(|w| w.parse().ok()).unwrap_or(blocks);
                        if b != blocks || !rendering {
                            if hang_reported {
                                append(log, &format!(
                                    "progress resumed after {:.0} s without a block",
                                    last_progress.elapsed().as_secs_f64()
                                ));
                                hang_reported = false;
                            }
                            blocks = b;
                            last_progress = Instant::now();
                        }
                    }
                    // The render closed its log itself: nothing to report.
                    Some("end") => return Ok(()),
                    // Its crash filter caught a native crash and is waiting.
                    Some("crash") => {
                        let words: Vec<&str> = words.collect();
                        if let (Some(c), Some(p)) = (winsys::Crash::parse(&words), process.as_ref()) {
                            let dumped = dump(p, pid, log, "CRASH", Some(c));
                            Process::release(pid);
                            append(log, &format!(
                                "native crash: exception {:#010x} at {:#x}, thread {}; {}",
                                c.code, c.address, c.thread, dump_note(&dumped)
                            ));
                            crash_dump = dumped.ok();
                        }
                    }
                    _ => {}
                }
            }
            Ok(Msg::Eof) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }

        let stuck = rendering && !hang_reported && last_progress.elapsed() >= Duration::from_secs(HANG_SECS);
        let silent = !silence_reported && last_beat.elapsed() >= Duration::from_secs(SILENT_SECS);
        if stuck || silent {
            let what = if stuck {
                format!(
                    "is still running, but has finished no block for {} s (block {} was the last). \
                     The render thread is most likely waiting on the GPU",
                    last_progress.elapsed().as_secs(),
                    blocks
                )
            } else {
                format!(
                    "has sent nothing for {} s: the whole process has stopped, not only its render",
                    last_beat.elapsed().as_secs()
                )
            };
            // Every thread as it is now shows what each was waiting on.
            if hang_dump.is_none() {
                if let Some(p) = process.as_ref() {
                    let dumped = dump(p, pid, log, "HANG", None);
                    append(log, &format!("hang: {}", dump_note(&dumped)));
                    hang_dump = dumped.ok();
                }
            }
            let report = write_report(log, "HANG", &what, started, hang_dump.as_deref());
            append(log, &format!(
                "{}; {}",
                if stuck { format!("no block for {HANG_SECS} s") } else { format!("no heartbeat for {SILENT_SECS} s") },
                reported(&report)
            ));
            hang_reported |= stuck;
            silence_reported |= silent;
        }
    }

    // The pipe closed without `end`: the render is gone, or about to be.
    let code = process.as_ref().and_then(|p| {
        p.wait(Duration::from_secs(10));
        p.exit_code()
    });
    match code {
        Some(code) if !winsys::is_crash(code) => {
            append(log, &format!("the process {} (exit code {code:#x})", winsys::describe_exit(code)));
        }
        Some(code) => {
            // [5]
            std::thread::sleep(Duration::from_secs(3));
            let what = format!("{} (exit code {code:#010x})", winsys::describe_exit(code));
            let report = write_report(log, "CRASH", &what, started, crash_dump.as_deref());
            append(log, &format!("the process {what}; {}", reported(&report)));
        }
        None => {
            let what = "ended without closing its log, and its exit code could not be read";
            let report = write_report(log, "CRASH", what, started, crash_dump.as_deref());
            append(log, &format!("the process {what}; {}", reported(&report)));
        }
    }
    Ok(())
}

/// Write `<log> <kind>.dmp` of the render and overwrite this machine's names \[6\]
fn dump(process: &Process, pid: u32, log: &Path, kind: &str, crash: Option<winsys::Crash>) -> Result<PathBuf, String> {
    let stem = log.file_stem().ok_or("the log has no name")?.to_string_lossy().into_owned();
    let path = log.with_file_name(format!("{stem} {kind}.dmp"));
    process.dump(pid, &path, crash)?;
    let mut bytes = std::fs::read(&path).map_err(|e| format!("reading the dump back: {e}"))?;
    redactor(log).scrub_bytes(&mut bytes);
    std::fs::write(&path, bytes).map_err(|e| format!("writing the scrubbed dump: {e}"))?;
    Ok(path)
}

fn dump_note(dumped: &Result<PathBuf, String>) -> String {
    match dumped {
        Ok(p) => format!("minidump: {}", p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()),
        Err(e) => format!("no minidump ({e})"),
    }
}

fn reported(report: &Option<PathBuf>) -> String {
    match report {
        Some(p) => format!("report: {}", p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()),
        None => "the report could not be written".into(),
    }
}

/// The redactor for what the watcher writes: this machine's names, keeping \[7\]
fn redactor(log: &Path) -> Redactor {
    let mut r = Redactor::from_env();
    if let Some(midi) = midi_of(log) {
        r.keep(&midi);
    }
    r
}

/// The MIDI's name, from the log's: `<midi> <MM-DD-YYYY HH.MM.SS>.log`, or \[8\]
pub(crate) fn midi_of(log: &Path) -> Option<String> {
    let mut stem = log.file_stem()?.to_string_lossy().into_owned();
    if stem.ends_with(')') {
        if let Some(i) = stem.rfind(" (") {
            stem.truncate(i);
        }
    }
    let time = stem.rfind(' ')?;
    let date = stem[..time].rfind(' ')?;
    Some(stem[..date].to_string())
}

/// Add a line to the end of the render's log, which its own process no \[9\]
fn append(log: &Path, text: &str) {
    let line = format!(
        "[FalconEye {}] {}\n",
        chrono::Local::now().format("%H.%M.%S"),
        redactor(log).apply(text)
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(log) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// `<log> <kind>.txt` beside the log: what happened, the log's last lines, \[10\]
fn write_report(log: &Path, kind: &str, what: &str, started: Instant, dump: Option<&Path>) -> Option<PathBuf> {
    let stem = log.file_stem()?.to_string_lossy().into_owned();
    let path = log.with_file_name(format!("{stem} {kind}.txt"));
    let mut out = String::new();
    out.push_str(&format!(
        "Kestrel {} report, {}\n\
         Written by FalconEye, Kestrel's watcher: a second process that watches each render.\n\
         The PC's and the account's names are left out; the MIDI's, soundfonts' and tracks' are not.\n\n\
         The render {what}.\n\
         Its log: {}\n",
        kind.to_lowercase(),
        chrono::Local::now().format("%m-%d-%Y %H.%M.%S"),
        log.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
    ));
    if let Some(d) = dump {
        out.push_str(&format!(
            "A minidump is beside this report: {}. It holds every thread's stack and the memory \
             they point at, with the PC's and the account's names overwritten. It can be read \
             against this build's kestrel.pdb; send it if asked.\n",
            d.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()
        ));
    }
    out.push_str(&format!("\n== The last {TAIL_LINES} lines of the render log ==\n"));
    out.push_str(&tail(log, TAIL_LINES));
    // From the render's start, with a minute before it to spare.
    let window_ms = started.elapsed().as_millis() as u64 + 60_000;
    out.push_str("\n== Windows' records from the render's time ==\n");
    out.push_str(&windows_events(window_ms));
    out.push_str("\n== The GPU now ==\n");
    out.push_str(&gpu_now());
    let text = redactor(log).apply(&out);
    std::fs::write(&path, text).ok()?;
    crate::falconeye::renderlog::prune_reports(path.parent()?);
    Some(path)
}

fn tail(path: &Path, n: usize) -> String {
    match std::fs::read(path) {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.lines().collect();
            let mut s = lines[lines.len().saturating_sub(n)..].join("\n");
            s.push('\n');
            s
        }
        Err(e) => format!("(could not read it: {e})\n"),
    }
}

/// Run a tool for its output, giving up after `timeout`.
pub(crate) fn tool(program: &str, args: &[&str], timeout: Duration) -> Result<String, String> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("{program} could not be run: {e}"))?;
    let stdout = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(mut o) = stdout {
            let _ = std::io::Read::read_to_string(&mut o, &mut s);
        }
        s
    });
    let t0 = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if t0.elapsed() < timeout => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                return Err(format!("{program} did not answer within {} s", timeout.as_secs()));
            }
        }
    }
    reader.join().map_err(|_| format!("{program}'s output could not be read"))
}

/// Windows' own records from the last `window_ms`: GPU driver events from \[11\]
#[cfg(windows)]
pub(crate) fn windows_events(window_ms: u64) -> String {
    let when = format!("TimeCreated[timediff(@SystemTime) <= {window_ms}]");
    let system = format!(
        "*[System[Provider[@Name='Display' or @Name='nvlddmkm' or @Name='amdkmdag' or @Name='amdkmdap' \
         or @Name='igfx' or @Name='igfxn' or @Name='Microsoft-Windows-DxgKrnl'] and {when}]]"
    );
    let application = format!(
        "*[System[Provider[@Name='Application Error' or @Name='Application Hang' \
         or @Name='Windows Error Reporting'] and {when}]]"
    );
    let mut out = String::new();
    out.push_str("-- System log: the graphics driver and the display --\n");
    out.push_str(&events("System", &system, |_| true));
    out.push_str("-- Application log: kestrel.exe's errors and hangs, and GPU resets --\n");
    out.push_str(&events("Application", &application, |e| {
        let e = e.to_ascii_lowercase();
        e.contains("kestrel") || e.contains("livekernelevent")
    }));
    out
}

#[cfg(not(windows))]
pub(crate) fn windows_events(_window_ms: u64) -> String {
    "(not collected on this platform yet)\n".into()
}

#[cfg(windows)]
fn events(log: &str, query: &str, wanted: impl Fn(&str) -> bool) -> String {
    let q = format!("/q:{query}");
    match tool("wevtutil", &["qe", log, &q, "/f:text", "/rd:true", "/c:50"], Duration::from_secs(20)) {
        Ok(text) => {
            let kept: Vec<String> = split_events(&text).into_iter().filter(|e| wanted(e)).take(20).collect();
            if kept.is_empty() {
                "(none)\n".into()
            } else {
                kept.join("\n")
            }
        }
        Err(e) => format!("({e})\n"),
    }
}

/// `wevtutil /f:text` output, one string per event, each cut to its first 40 \[12\]
fn split_events(text: &str) -> Vec<String> {
    let mut events: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        if line.starts_with("Event[") {
            events.push(Vec::new());
        }
        if let Some(e) = events.last_mut() {
            if e.len() < 40 {
                e.push(line.trim_end());
            }
        }
    }
    events.into_iter().map(|e| e.join("\n") + "\n").collect()
}

/// The GPU's state, from `nvidia-smi` where there is one.
pub(crate) fn gpu_now() -> String {
    let query = "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,\
                 clocks.gr,clocks.mem,utilization.gpu,memory.used,memory.total,clocks_event_reasons.active";
    match tool("nvidia-smi", &[query, "--format=csv"], Duration::from_secs(15)) {
        Ok(text) if !text.trim().is_empty() => text,
        Ok(_) => "(nvidia-smi said nothing)\n".into(),
        Err(e) => format!("({e}; only NVIDIA's driver has it)\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_midi_name_is_read_back_from_the_log_name() {
        let p = Path::new(r"C:\k\logs\Song Title 09-26-2026 19.24.16.log");
        assert_eq!(midi_of(p).as_deref(), Some("Song Title"));
        let p = Path::new(r"C:\k\logs\Song (Remix) 09-26-2026 19.24.16 (2).log");
        assert_eq!(midi_of(p).as_deref(), Some("Song (Remix)"));
    }

    #[test]
    fn wevtutil_text_splits_into_events() {
        let text = "Event[0]:\n  Log Name: System\n  Event ID: 4101\n\nEvent[1]:\n  Event ID: 13\n";
        let e = split_events(text);
        assert_eq!(e.len(), 2);
        assert!(e[0].contains("4101") && e[1].contains("13"));
    }

    #[test]
    fn crashes_are_told_from_closes() {
        assert!(!winsys::is_crash(0xC000_013A));
        assert!(!winsys::is_crash(0));
        assert!(winsys::is_crash(0xC000_0005));
        assert!(winsys::is_crash(0xC000_0409));
        assert!(winsys::is_crash(1));
    }
}
