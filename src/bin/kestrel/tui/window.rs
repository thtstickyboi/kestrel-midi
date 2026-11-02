// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Windows of their own: how Extras and Merlin open one, and how a window \[1\]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// How a window that was asked to go answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Asked {
    /// It exited.
    Gone,
    /// It is rendering, and stays.
    Busy,
    /// No answer in time, so it stays.
    Silent,
}

/// What `hand_over` did with the window it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handover {
    /// The new window is open and the old one is gone, or there was none.
    Done,
    /// The new window is open, and the main window is rendering, so it stays.
    ParentBusy,
    /// The new window is open, and the main window did not answer.
    ParentSilent,
}

/// Open a new terminal window running this executable with `args`, titled \[2\]
pub fn open(title: &str, args: &[String]) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    spawn(title, &exe, args).map_err(|e| e.to_string())
}

#[cfg(windows)]
fn spawn(title: &str, exe: &Path, args: &[String]) -> std::io::Result<()> {
    use std::os::windows::process::CommandExt;
    let mut line = format!("/C start \"{title}\" \"{}\"", exe.display());
    for a in args {
        line.push(' ');
        line.push_str(a);
    }
    let status = std::process::Command::new("cmd").raw_arg(line).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!("start exited with {status}")))
    }
}

#[cfg(target_os = "macos")]
fn spawn(_title: &str, exe: &Path, args: &[String]) -> std::io::Result<()> {
    let mut quoted = format!("'{}'", exe.display().to_string().replace('\'', r"'\''"));
    for a in args {
        quoted.push(' ');
        quoted.push_str(a);
    }
    let script = format!(
        "tell application \"Terminal\" to do script \"{}\"",
        quoted.replace('\\', "\\\\").replace('"', "\\\"")
    );
    std::process::Command::new("osascript")
        .args(["-e", &script, "-e", "tell application \"Terminal\" to activate"])
        .spawn()
        .map(|_| ())
}

/// The terminal emulators to try, in order, each with what comes before the \[3\]
#[cfg_attr(not(all(unix, not(target_os = "macos"))), allow(dead_code))]
fn terminals(preferred: Option<String>) -> Vec<(String, Vec<&'static str>)> {
    let mut tried: Vec<(String, Vec<&'static str>)> = Vec::new();
    if let Some(t) = preferred.filter(|t| !t.is_empty()) {
        tried.push((t, vec!["-e"]));
    }
    for (term, lead) in [
        ("x-terminal-emulator", vec!["-e"]),
        ("gnome-terminal", vec!["--"]),
        ("konsole", vec!["-e"]),
        ("xfce4-terminal", vec!["-x"]),
        ("kitty", vec![]),
        ("alacritty", vec!["-e"]),
        ("wezterm", vec!["start", "--"]),
        ("foot", vec![]),
        ("xterm", vec!["-e"]),
    ] {
        tried.push((term.to_string(), lead));
    }
    tried
}

#[cfg(all(unix, not(target_os = "macos")))]
fn spawn(_title: &str, exe: &Path, args: &[String]) -> std::io::Result<()> {
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;
    for (term, lead) in terminals(std::env::var("TERMINAL").ok()) {
        let mut command = std::process::Command::new(&term);
        command
            .args(lead)
            .arg(exe)
            .args(args)
            // The new window has a terminal of its own, so it needs none of this one's.
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            // [4]
            .process_group(0);
        let Ok(mut child) = command.spawn() else { continue };
        // [5]
        let until = std::time::Instant::now() + Duration::from_millis(600);
        loop {
            match child.try_wait() {
                Ok(Some(status)) if !status.success() => break,
                Ok(Some(_)) | Err(_) => return Ok(()),
                Ok(None) if std::time::Instant::now() >= until => return Ok(()),
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no terminal emulator found"))
}

// ---- asking a window to go ------------------------------------------------

/// Renders and batches under way in this process. A window with any is not \[6\]
static BUSY: AtomicUsize = AtomicUsize::new(0);

/// Held while a render is on the screen.
pub struct Busy;

impl Busy {
    pub fn new() -> Self {
        BUSY.fetch_add(1, Ordering::SeqCst);
        Busy
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        BUSY.fetch_sub(1, Ordering::SeqCst);
    }
}

fn busy() -> bool {
    BUSY.load(Ordering::SeqCst) > 0
}

/// How often a window looks for a request to go, and how long a window that \[7\]
const POLL: Duration = Duration::from_millis(150);
const PATIENCE: Duration = Duration::from_secs(3);

fn request_file(dir: &Path, pid: u32) -> PathBuf {
    dir.join(format!("kestrel-quit-{pid}"))
}

fn refusal_file(dir: &Path, pid: u32) -> PathBuf {
    dir.join(format!("kestrel-quit-{pid}.refused"))
}

/// This process answers a request to go, for as long as it lives: on a thread \[8\]
pub fn serve_quit() {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    // A request left by a window that died is not for this one.
    let _ = std::fs::remove_file(request_file(&dir, pid));
    let _ = std::fs::remove_file(refusal_file(&dir, pid));
    let _ = std::thread::Builder::new().name("quit-watch".into()).spawn(move || loop {
        std::thread::sleep(POLL);
        if answer(&dir, pid, busy()) {
            std::process::exit(0);
        }
    });
}

/// One look for a request. True when there is one and this window is to go.
fn answer(dir: &Path, pid: u32, busy: bool) -> bool {
    let request = request_file(dir, pid);
    if !request.exists() {
        return false;
    }
    if busy {
        let _ = std::fs::rename(&request, refusal_file(dir, pid));
        return false;
    }
    let _ = std::fs::remove_file(&request);
    true
}

/// Ask the window of process `pid` to go and wait for what it says.
fn ask(dir: &Path, pid: u32, patience: Duration) -> Asked {
    let request = request_file(dir, pid);
    let _ = std::fs::remove_file(refusal_file(dir, pid));
    if std::fs::write(&request, b"").is_err() {
        return Asked::Silent;
    }
    let until = Instant::now() + patience;
    loop {
        if refusal_file(dir, pid).exists() {
            let _ = std::fs::remove_file(refusal_file(dir, pid));
            return Asked::Busy;
        }
        // The request is taken away by the window that is going.
        if !request.exists() {
            return Asked::Gone;
        }
        if Instant::now() >= until {
            let _ = std::fs::remove_file(&request);
            return Asked::Silent;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Open a window and see the one it was started from out: `parent`, when \[9\]
pub fn hand_over(title: &str, args: &[String], parent: Option<u32>) -> Result<Handover, String> {
    open(title, args)?;
    Ok(match parent.map(|pid| ask(&std::env::temp_dir(), pid, PATIENCE)) {
        None | Some(Asked::Gone) => Handover::Done,
        Some(Asked::Busy) => Handover::ParentBusy,
        Some(Asked::Silent) => Handover::ParentSilent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kestrel-window-test-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_sessions_terminal_is_tried_first_and_each_one_has_its_own_lead() {
        let plain = terminals(None);
        assert_eq!(plain[0], ("x-terminal-emulator".to_string(), vec!["-e"]));
        assert!(plain.iter().any(|(t, lead)| t == "kitty" && lead.is_empty()));
        assert!(plain.iter().any(|(t, lead)| t == "wezterm" && lead == &["start", "--"]));
        assert!(plain.iter().any(|(t, lead)| t == "gnome-terminal" && lead == &["--"]));
        let mine = terminals(Some("st".to_string()));
        assert_eq!(mine[0], ("st".to_string(), vec!["-e"]));
        assert_eq!(mine.len(), plain.len() + 1);
        // An empty TERMINAL says nothing.
        assert_eq!(terminals(Some(String::new())).len(), plain.len());
    }

    /// Plays the window being asked: looks until there is a request, as the \[10\]
    fn watcher(dir: PathBuf, pid: u32, busy: bool) -> std::thread::JoinHandle<bool> {
        std::thread::spawn(move || {
            for _ in 0..400 {
                if answer(&dir, pid, busy) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            false
        })
    }

    #[test]
    fn an_idle_window_that_is_asked_goes() {
        let d = dir("idle");
        let w = watcher(d.clone(), 4001, false);
        assert_eq!(ask(&d, 4001, Duration::from_secs(2)), Asked::Gone);
        assert!(w.join().unwrap(), "the watcher did not go");
        assert!(!request_file(&d, 4001).exists() && !refusal_file(&d, 4001).exists());
    }

    #[test]
    fn a_window_that_is_rendering_stays_and_says_so() {
        let d = dir("busy");
        let w = watcher(d.clone(), 4002, true);
        assert_eq!(ask(&d, 4002, Duration::from_secs(2)), Asked::Busy);
        // It never goes, however long it is asked.
        assert!(!w.join().unwrap());
        assert!(!request_file(&d, 4002).exists() && !refusal_file(&d, 4002).exists(), "nothing is left behind");
    }

    #[test]
    fn a_window_that_does_not_answer_is_left_and_the_request_is_taken_back() {
        let d = dir("silent");
        assert_eq!(ask(&d, 4003, Duration::from_millis(150)), Asked::Silent);
        assert!(!request_file(&d, 4003).exists(), "a request left behind would close the next window of that number");
    }

    /// A render holds the window, and a finished one lets it go: nesting counts.
    #[test]
    fn the_busy_guard_counts_the_renders_under_way() {
        let before = busy();
        let a = Busy::new();
        let b = Busy::new();
        assert!(busy());
        drop(a);
        assert!(busy(), "one render still under way");
        drop(b);
        assert_eq!(busy(), before);
    }

    #[test]
    fn a_request_for_another_window_is_not_answered() {
        let d = dir("other");
        std::fs::write(request_file(&d, 4004), b"").unwrap();
        assert!(!answer(&d, 4005, false));
        assert!(request_file(&d, 4004).exists());
        std::fs::remove_file(request_file(&d, 4004)).unwrap();
    }
}
