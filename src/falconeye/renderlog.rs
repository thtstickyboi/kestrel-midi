// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A log file for every render, written as it goes. \[1\]

use crate::falconeye::redact::Redactor;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Render logs kept in the folder; the oldest go when a new one opens.
pub const KEEP: usize = 100;

/// The environment variable that moves the folder, for tests and for anyone \[2\]
pub const DIR_VAR: &str = "KESTREL_LOG_DIR";

struct Settings {
    front_end: &'static str,
    dir: Option<PathBuf>,
}

static SETTINGS: Mutex<Option<Settings>> = Mutex::new(None);
/// The front end's argument list for the next render, as it would be typed.
static ARGS: Mutex<Option<Vec<String>>> = Mutex::new(None);
static SINK: Mutex<Option<Open>> = Mutex::new(None);
/// The last log opened, open or not: what a front end points the user at.
static LAST: Mutex<Option<PathBuf>> = Mutex::new(None);

/// The executable to start a watcher from, once a front end asks for one.
static WATCH_EXE: Mutex<Option<PathBuf>> = Mutex::new(None);
/// The open log's watcher.
static WATCH: Mutex<Option<Watch>> = Mutex::new(None);
/// What the heartbeat says: whether the render is in its block loop, and how \[3\]
static RENDERING: AtomicBool = AtomicBool::new(false);
static BLOCKS: AtomicU64 = AtomicU64::new(0);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Log every render from now on, into `logs/` beside the executable, or the \[4\]
pub fn enable(front_end: &'static str) {
    let dir = std::env::var_os(DIR_VAR).map(PathBuf::from);
    *lock(&SETTINGS) = Some(Settings { front_end, dir });
}

/// [`enable`], into `dir`.
pub fn enable_in(front_end: &'static str, dir: &Path) {
    *lock(&SETTINGS) = Some(Settings { front_end, dir: Some(dir.to_path_buf()) });
}

/// Start a watcher with every log from now on, as `exe --force-cli falconeye`. \[5\]
pub fn enable_watch(exe: PathBuf) {
    *lock(&WATCH_EXE) = Some(exe);
    // Native crashes, reported to the watcher, which writes their dump.
    crate::falconeye::winsys::install_crash_filter();
}

/// Where the render is, for the watcher's heartbeat: in the block loop or \[6\]
pub fn set_progress(rendering: bool, blocks: u64) {
    RENDERING.store(rendering, Ordering::Relaxed);
    BLOCKS.store(blocks, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    lock(&SETTINGS).is_some()
}

/// The arguments the next render was started with, as the user would type \[7\]
pub fn set_args<I, S>(args: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let args = args.into_iter().map(|a| a.as_ref().to_string_lossy().into_owned()).collect();
    *lock(&ARGS) = Some(args);
    // [8]
    *lock(&LAST) = None;
}

/// The last render log opened, if any.
pub fn last_path() -> Option<PathBuf> {
    lock(&LAST).clone()
}

/// A log record, from a front end's logger. Written only while a render's log \[9\]
pub fn record(level: log::Level, target: &str, args: &std::fmt::Arguments) {
    let wanted = if target.starts_with("kestrel") {
        level <= log::Level::Info
    } else {
        level <= log::Level::Warn
    };
    if !wanted {
        return;
    }
    if let Some(open) = lock(&SINK).as_mut() {
        open.line(level.as_str(), &format!("{target}: {args}"));
    }
}

/// Whether a logger should pass a record at this level and target on to \[10\]
pub fn wants(level: log::Level, target: &str) -> bool {
    if lock(&SETTINGS).is_none() {
        return false;
    }
    if target.starts_with("kestrel") {
        level <= log::Level::Info
    } else {
        level <= log::Level::Warn
    }
}

/// A line of the render's own: a breadcrumb, a phase, the outcome.
pub fn note(tag: &str, text: &str) {
    if let Some(open) = lock(&SINK).as_mut() {
        open.line(tag, text);
    }
}

/// Add a name to show as it is, once the render knows it: a track's.
pub fn keep_name(name: &str) {
    if let Some(open) = lock(&SINK).as_mut() {
        open.redact.keep(name);
    }
}

struct Open {
    file: File,
    redact: Redactor,
    started: Instant,
}

impl Open {
    fn line(&mut self, tag: &str, text: &str) {
        let t = self.started.elapsed().as_secs_f64();
        let text = self.redact.apply(text);
        let mut out = String::with_capacity(text.len() + 24);
        for (i, l) in text.lines().enumerate() {
            if i == 0 {
                out.push_str(&format!("[{t:9.3}] {tag:<5} {l}\n"));
            } else {
                out.push_str(&format!("{:18}{l}\n", ""));
            }
        }
        if text.is_empty() {
            out.push_str(&format!("[{t:9.3}] {tag}\n"));
        }
        // [11]
        let _ = self.file.write_all(out.as_bytes());
    }
}

/// An open render log. Dropping it closes the log, noting a panic if one is \[12\]
pub struct Guard {
    path: PathBuf,
}

impl Guard {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let panicking = std::thread::panicking();
        let mut sink = lock(&SINK);
        if let Some(open) = sink.as_mut() {
            if panicking {
                open.line("END", "the render ended in a panic; see PANIC above");
            } else {
                open.line("END", "log closed");
            }
        }
        *sink = None;
        drop(sink);
        // After END, so the watcher never finds a log still being written.
        if let Some(w) = lock(&WATCH).take() {
            w.finish(if panicking { "panic" } else { "clean" });
        }
    }
}

/// The pipe to a render's watcher, shared with the thread that beats.
struct Watch {
    stdin: Arc<Mutex<Option<std::process::ChildStdin>>>,
    stop: Arc<AtomicBool>,
}

impl Watch {
    fn start(exe: &Path, log: &Path) -> std::io::Result<(u32, Watch)> {
        let mut cmd = std::process::Command::new(exe);
        cmd.args(["--force-cli", "falconeye", "--pid", &std::process::id().to_string(), "--log"])
            .arg(log)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // [13]
            const DETACHED_PROCESS: u32 = 0x0000_0008;
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
        }
        let mut child = cmd.spawn()?;
        let pid = child.id();
        if let Some(pipe) = &child.stdin {
            crate::falconeye::winsys::arm_crash_filter(pipe);
        }
        let stdin = Arc::new(Mutex::new(child.stdin.take()));
        let stop = Arc::new(AtomicBool::new(false));
        RENDERING.store(false, Ordering::Relaxed);
        BLOCKS.store(0, Ordering::Relaxed);
        let (pipe, stopped) = (Arc::clone(&stdin), Arc::clone(&stop));
        // [14]
        std::thread::Builder::new().name("falconeye-heartbeat".into()).spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let beat = format!(
                    "beat {} {}\n",
                    RENDERING.load(Ordering::Relaxed) as u8,
                    BLOCKS.load(Ordering::Relaxed)
                );
                let mut p = lock(&pipe);
                let Some(w) = p.as_mut() else { break };
                if w.write_all(beat.as_bytes()).and_then(|_| w.flush()).is_err() {
                    break;
                }
                drop(p);
                for _ in 0..10 {
                    if stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        })?;
        Ok((pid, Watch { stdin, stop }))
    }

    /// Tell the watcher the log closed normally, and let it go.
    fn finish(self, how: &str) {
        // [15]
        crate::falconeye::winsys::disarm_crash_filter();
        self.stop.store(true, Ordering::Relaxed);
        if let Some(mut w) = lock(&self.stdin).take() {
            let _ = w.write_all(format!("end {how}\n").as_bytes());
            let _ = w.flush();
        }
    }
}

/// Writes one header line: a tag, and its text.
pub type HeaderLine<'a> = dyn FnMut(&str, &str) + 'a;

/// Open a log for the render `midi` names, if logging is on. `header` is \[16\]
pub fn open(midi: &Path, keep: &[&str], header: &dyn Fn(&mut HeaderLine)) -> Option<Guard> {
    let (front_end, dir) = {
        let s = lock(&SETTINGS);
        let s = s.as_ref()?;
        (s.front_end, s.dir.clone())
    };
    let stem = midi
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "render".into());
    let (path, file) = match create(dir.as_deref(), &stem) {
        Ok(p) => p,
        Err(e) => {
            log::warn!(target: "kestrel", "no render log: {e:#}");
            return None;
        }
    };
    let mut redact = Redactor::from_env();
    for k in keep {
        redact.keep(k);
    }
    let args = lock(&ARGS).take();
    let mut open = Open { file, redact, started: Instant::now() };
    open.line("LOG", &format!(
        "Kestrel {} ({}) render log, {}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(feature = "dev") { "dev" } else { "release" },
        chrono::Local::now().format("%m-%d-%Y %H.%M.%S"),
    ));
    open.line("LOG", &format!("front end: {front_end}; {} {}", std::env::consts::OS, std::env::consts::ARCH));
    if let Ok(exe) = std::env::current_exe() {
        open.line("LOG", &format!("exe: {}", exe.display()));
    }
    open.line("LOG", &format!(
        "anchor: {:#x} (kestrel::falconeye::renderlog::anchor, which places a backtrace's addresses against this build's .pdb)",
        anchor()
    ));
    open.line("LOG", "names: the PC's and the account's are hidden; the MIDI's, soundfonts' and tracks' are not");
    if let Some(args) = args {
        open.line("ARGS", &args.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" "));
    }
    header(&mut |tag, text| open.line(tag, text));
    let exe = lock(&WATCH_EXE).clone();
    if let Some(exe) = exe {
        match Watch::start(&exe, &path) {
            Ok((pid, watch)) => {
                open.line("EYE", &format!("FalconEye started, process {pid}"));
                *lock(&WATCH) = Some(watch);
            }
            Err(e) => open.line("EYE", &format!("no FalconEye watcher: {e}")),
        }
    }
    *lock(&SINK) = Some(open);
    *lock(&LAST) = Some(path.clone());
    Some(Guard { path })
}

/// Where this function was loaded. A backtrace's addresses are only useful \[17\]
#[inline(never)]
pub fn anchor() -> usize {
    anchor as fn() -> usize as usize
}

/// Quoted the way a Windows command line would need it.
fn quote(a: &str) -> String {
    if !a.is_empty() && !a.contains([' ', '\t', '"']) {
        a.to_string()
    } else {
        format!("\"{}\"", a.replace('"', "\\\""))
    }
}

/// Where logs go by default: `logs/` beside the executable, or, where that \[18\]
pub(crate) fn default_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        dirs.push(exe_dir.join("logs"));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        dirs.push(PathBuf::from(local).join("Kestrel").join("logs"));
    } else if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".local").join("state").join("kestrel").join("logs"));
    }
    dirs
}

fn create(dir: Option<&Path>, stem: &str) -> anyhow::Result<(PathBuf, File)> {
    let dirs = match dir {
        Some(d) => vec![d.to_path_buf()],
        None => default_dirs(),
    };
    let stamp = chrono::Local::now().format("%m-%d-%Y %H.%M.%S").to_string();
    let stem: String = stem.chars().take(100).collect();
    let mut last_err = anyhow::anyhow!("no folder to write logs to");
    for d in dirs {
        if let Err(e) = std::fs::create_dir_all(&d) {
            last_err = anyhow::anyhow!("creating {}: {e}", d.display());
            continue;
        }
        for n in 1..100 {
            let name = if n == 1 { format!("{stem} {stamp}.log") } else { format!("{stem} {stamp} ({n}).log") };
            let path = d.join(name);
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(f) => {
                    prune(&d, KEEP);
                    return Ok((path, f));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    last_err = anyhow::anyhow!("creating {}: {e}", path.display());
                    break;
                }
            }
        }
    }
    Err(last_err)
}

/// Delete all but the newest `keep` render logs in `dir`. Only `.log` files: \[19\]
pub fn prune(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut logs: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("log")))
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    if logs.len() <= keep {
        return;
    }
    logs.sort();
    for (_, p) in &logs[..logs.len() - keep] {
        let _ = std::fs::remove_file(p);
    }
}

/// Delete all but the newest watcher reports in `dir`: `KEEP` of the crash \[20\]
pub fn prune_reports(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut reports = Vec::new();
    let mut dumps = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(t) = e.metadata().ok().and_then(|m| m.modified().ok()) else { continue };
        if name.ends_with(" CRASH.txt") || name.ends_with(" HANG.txt") {
            reports.push((t, e.path()));
        } else if name.ends_with(".dmp") {
            dumps.push((t, e.path()));
        }
    }
    for (mut list, keep) in [(reports, KEEP), (dumps, 5)] {
        if list.len() > keep {
            list.sort();
            let extra = list.len() - keep;
            for (_, p) in &list[..extra] {
                let _ = std::fs::remove_file(p);
            }
        }
    }
}

/// Write a panic to the open log from the panic hook, with a backtrace, \[21\]
pub fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            write_panic(info);
            prev(info);
        }));
    });
}

fn write_panic(info: &std::panic::PanicHookInfo) {
    // A panic inside `record` holds the lock already; waiting would hang.
    let mut sink = match SINK.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    if let Some(open) = sink.as_mut() {
        let thread = std::thread::current();
        let bt = std::backtrace::Backtrace::force_capture();
        // [22]
        open.line("PANIC", &format!(
            "on thread {}: {info}\nbacktrace:\n{bt:#}",
            thread.name().unwrap_or("unnamed")
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kestrel_renderlog_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_log_is_named_after_the_midi_and_never_overwrites() {
        let d = tmp("names");
        let (a, _) = create(Some(&d), "Song Title").unwrap();
        let (b, _) = create(Some(&d), "Song Title").unwrap();
        let a = a.file_name().unwrap().to_string_lossy().into_owned();
        let b = b.file_name().unwrap().to_string_lossy().into_owned();
        assert!(a.starts_with("Song Title ") && a.ends_with(".log"), "{a}");
        assert_ne!(a, b);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn pruning_keeps_the_newest_logs_and_nothing_else_is_touched() {
        let d = tmp("prune");
        for i in 0..5 {
            std::fs::write(d.join(format!("{i}.log")), "x").unwrap();
            // Distinct modification times, oldest first.
            let t = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000 + i);
            File::options().write(true).open(d.join(format!("{i}.log"))).unwrap().set_modified(t).unwrap();
        }
        std::fs::write(d.join("report.txt"), "x").unwrap();
        prune(&d, 2);
        let mut left: Vec<String> =
            std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, ["3.log", "4.log", "report.txt"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_command_line_is_quoted_where_it_has_to_be() {
        assert_eq!(quote("--block"), "--block");
        assert_eq!(quote(r"C:\My Music\a.mid"), "\"C:\\My Music\\a.mid\"");
        assert_eq!(quote(""), "\"\"");
    }
}
