// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The GPU self-test: how long this GPU takes over one block, at voice counts \[1\]

use crate::bank::Bank;
use crate::config::Config;
use crate::driver::Driver;
use crate::Backend;
use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// The first step. Small, because an integrated GPU is slow: this laptop's \[2\]
pub const FIRST_VOICES: u32 = 4_096;
/// Stop once a block takes this long: the next step, twice the voices, then \[3\]
pub const STOP_MS: f64 = 250.0;
/// Windows' limit, and the share of it the verdict leaves as margin.
pub const TDR_MS: f64 = 2000.0;
const TARGET_MS: f64 = 1000.0;
/// Blocks rendered at each step: the one that spawns every voice, then the \[4\]
const BLOCKS: usize = 6;
/// The sample pool, in MiB.
const POOL_MB: usize = 128;
/// No step past this; a card that is fast enough here is fast enough.
const MAX_VOICES: u32 = 1 << 24;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Step {
    /// The pool's size, and how many voices were sounding.
    pub voices: u32,
    pub live: u64,
    /// The block that spawned them all, and the slowest and the middle of \[5\]
    pub spawn_ms: f64,
    pub worst_ms: f64,
    pub typical_ms: f64,
    pub timed_on_gpu: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Outcome {
    pub adapter: String,
    pub backend: String,
    pub steps: Vec<Step>,
    /// Why it stopped where it did.
    pub stopped: String,
    /// The answer, in a sentence or two.
    pub verdict: String,
    /// The same, as numbers: ms per block at the default 1,048,576 voices, \[6\]
    pub ms_at_default: Option<f64>,
    pub voices_for_1s: Option<u64>,
}

/// Write the self-test's soundfont into `dir`.
pub fn write_material(dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let sf = dir.join("selftest.sf2");
    crate::testkit::big_sf2(&sf, Config::default().sample_rate, POOL_MB)?;
    Ok(sf)
}

/// Load the soundfont [`write_material`] wrote.
fn load(sf: &Path) -> Result<Arc<Bank>> {
    Ok(Arc::new(crate::load_bank(sf, &Config::default())?))
}

/// Write the self-test's soundfont into `dir` and load it, for a test that runs \[7\]
pub fn prepare(dir: &Path) -> Result<Arc<Bank>> {
    load(&write_material(dir)?)
}

/// Where a report runs its self-tests: in this process, or in one of their own.
pub enum Runner {
    Here(Arc<Bank>),
    Child { exe: PathBuf, sf: PathBuf },
}

impl Runner {
    /// Make the test material in `dir`. With `isolate_with`, the executable that \[8\]
    pub fn new(dir: &Path, isolate_with: Option<PathBuf>) -> Result<Runner> {
        match isolate_with {
            Some(exe) => Ok(Runner::Child { exe, sf: write_material(dir)? }),
            None => Ok(Runner::Here(prepare(dir)?)),
        }
    }

    pub fn run(&self, dir: &Path, adapter: &str, backend: &str, max_voices: u32, say: &mut dyn FnMut(&str)) -> Outcome {
        match self {
            Runner::Here(bank) => run(bank, dir, adapter, backend, max_voices, say),
            Runner::Child { exe, sf } => run_isolated(exe, sf, adapter, backend, max_voices, STALL, say),
        }
    }
}

/// What the test process prints, a line each, for the report to read.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum Message {
    /// About to try this many voices. A crash is blamed on the last of these.
    Starting { voices: u32 },
    Step(Step),
    Done(Outcome),
}

/// What [`walk`] tells whoever is listening.
enum Heard<'a> {
    Starting(u32),
    Step(&'a Step),
}

fn step_line(s: &Step) -> String {
    format!(
        "{:>9} voices: {:7.1} ms a block ({:.1} ms spawning them){}",
        thousands(s.voices as u64),
        s.worst_ms,
        s.spawn_ms,
        if s.timed_on_gpu { "" } else { ", host-timed" }
    )
}

/// Run the self-test on one adapter and backend, as `gpu::survey` names \[9\]
pub fn run(
    bank: &Arc<Bank>,
    dir: &Path,
    adapter: &str,
    backend: &str,
    max_voices: u32,
    say: &mut dyn FnMut(&str),
) -> Outcome {
    walk(bank, dir, adapter, backend, max_voices, &mut |h| {
        if let Heard::Step(s) = h {
            say(&step_line(s));
        }
    })
}

/// The steps themselves, doubling the voices until a block is slow.
fn walk(
    bank: &Arc<Bank>,
    dir: &Path,
    adapter: &str,
    backend: &str,
    max_voices: u32,
    hear: &mut dyn FnMut(Heard),
) -> Outcome {
    let mut steps = Vec::new();
    let mut voices = FIRST_VOICES;
    let cap = max_voices.min(MAX_VOICES);
    let stopped = loop {
        if voices > cap {
            break format!("reached the most voices this adapter can hold, {cap}");
        }
        hear(Heard::Starting(voices));
        match step(bank, dir, adapter, backend, voices) {
            Ok(s) => {
                hear(Heard::Step(&s));
                let slow = s.worst_ms.max(s.spawn_ms);
                steps.push(s);
                if slow > STOP_MS {
                    break format!("a block took {slow:.0} ms; the next step could take twice that");
                }
            }
            Err(e) => break format!("{voices} voices could not be set up: {e:#}"),
        }
        voices = voices.saturating_mul(2);
    };
    // [10]
    let (verdict, ms_at_default, voices_for_1s) = if steps.is_empty() {
        (format!("The test could not run: {stopped}"), None, None)
    } else {
        verdict(&steps)
    };
    Outcome {
        adapter: adapter.to_string(),
        backend: backend.to_string(),
        steps,
        stopped,
        verdict,
        ms_at_default,
        voices_for_1s,
    }
}

/// The test process: `kestrel falconeye-selftest`, started by [`run_isolated`]. \[11\]
pub fn serve(sf: &Path, adapter: &str, backend: &str, max_voices: u32) -> Result<()> {
    // [12]
    log::set_max_level(log::LevelFilter::Warn);
    let bank = load(sf).with_context(|| format!("loading {}", sf.display()))?;
    let dir = sf.parent().unwrap_or(Path::new("."));
    let outcome = walk(&bank, dir, adapter, backend, max_voices, &mut |h| match h {
        Heard::Starting(voices) => send(&Message::Starting { voices }),
        Heard::Step(s) => send(&Message::Step(s.clone())),
    });
    send(&Message::Done(outcome));
    #[cfg(feature = "dev")]
    fail_on_purpose(0);
    Ok(())
}

fn send(m: &Message) {
    let mut out = std::io::stdout().lock();
    if let Ok(line) = serde_json::to_string(m) {
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}

/// How long the test process may go without a line before it is taken to be hung \[13\]
pub const STALL: Duration = Duration::from_secs(180);

/// [`run`], in a process of its own: `exe` is started as `--force-cli \[14\]
pub fn run_isolated(
    exe: &Path,
    sf: &Path,
    adapter: &str,
    backend: &str,
    max_voices: u32,
    stall: Duration,
    say: &mut dyn FnMut(&str),
) -> Outcome {
    let ended = |steps: Vec<Step>, stopped: String| {
        let (verdict, ms_at_default, voices_for_1s) = if steps.is_empty() {
            (format!("The test could not run: {stopped}"), None, None)
        } else {
            let (v, ms, for_1s) = verdict(&steps);
            (format!("The test stopped early: {stopped}. Before that: {v}"), ms, for_1s)
        };
        Outcome {
            adapter: adapter.to_string(),
            backend: backend.to_string(),
            steps,
            stopped,
            verdict,
            ms_at_default,
            voices_for_1s,
        }
    };

    // [15]
    let err_path = sf.with_extension("stderr");
    let spawned = std::fs::File::create(&err_path).map_err(anyhow::Error::from).and_then(|err| {
        Command::new(exe)
            .args(["--force-cli", "falconeye-selftest", "--adapter", adapter, "--backend", backend])
            .args(["--max-voices", &max_voices.to_string(), "--sf"])
            .arg(sf)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(err)
            .spawn()
            .map_err(anyhow::Error::from)
    });
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => return ended(Vec::new(), format!("the test process could not be started: {e:#}")),
    };
    let (tx, rx) = mpsc::channel();
    let stdout = child.stdout.take().expect("stdout was piped");
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let mut steps = Vec::new();
    let mut at: Option<u32> = None;
    let mut done = None;
    let mut hung = false;
    loop {
        match rx.recv_timeout(stall) {
            // Anything that is not a message, a banner say, is not for us.
            Ok(line) => match serde_json::from_str::<Message>(&line) {
                Ok(Message::Starting { voices }) => at = Some(voices),
                Ok(Message::Step(s)) => {
                    say(&step_line(&s));
                    steps.push(s);
                    at = None;
                }
                Ok(Message::Done(o)) => done = Some(o),
                Err(_) => {}
            },
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
    // [16]
    if let Some(mut o) = done {
        if let Ok(s) = &status {
            if !s.success() {
                o.stopped = format!("{}; the test process then {}", o.stopped, how_it_ended(s));
            }
        }
        return o;
    }

    let working = match (at, steps.is_empty()) {
        (Some(v), _) => format!("while it worked on {} voices", thousands(v as u64)),
        (None, true) => "before its first step".to_string(),
        (None, false) => "between steps".to_string(),
    };
    let how = if hung {
        format!("{working}, the test process said nothing for {} s and was ended", stall.as_secs())
    } else {
        match &status {
            Ok(s) => format!("{working}, the test process {}", how_it_ended(s)),
            Err(e) => format!("{working}, the test process could not be waited for: {e}"),
        }
    };
    let tail = last_lines(&err_path, 6);
    let stopped = if tail.is_empty() { how } else { format!("{how}. Its last messages: {tail}") };
    ended(steps, stopped)
}

/// What an exit means, for a process that was meant to exit with a zero.
fn how_it_ended(status: &std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            let name = match sig {
                4 => ", an illegal instruction",
                6 => ", an abort",
                7 => ", a bus error",
                9 => ", killed from outside: out of memory, or by hand",
                11 => ", a segmentation fault: a read or write of memory it did not own. Inside a graphics driver this is usually the driver's fault",
                _ => "",
            };
            return format!("was ended by signal {sig}{name}");
        }
    }
    match status.code() {
        Some(0) => "ended without giving its result".to_string(),
        Some(1) => "failed with an error (exit code 1)".to_string(),
        // Windows reports an NTSTATUS as the exit code.
        Some(c) => {
            let c = c as u32;
            format!("{} (exit code {c:#x})", super::winsys::describe_exit(c))
        }
        None => "ended in a way that gave no exit code".to_string(),
    }
}

/// The last `n` messages in a text file, each cut short, on one line. Not the \[17\]
fn last_lines(path: &Path, n: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<String> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with(char::is_whitespace) && !l.contains("wgpu_hal::vulkan::instance"))
        .map(|l| l.trim().chars().take(240).collect())
        .collect();
    lines[lines.len().saturating_sub(n)..].join(" | ")
}

/// Fail on purpose at one step, `KESTREL_SELFTEST_CRASH=<kind>@<voices>`, to \[18\]
#[cfg(feature = "dev")]
fn fail_on_purpose(voices: u32) {
    let Ok(v) = std::env::var("KESTREL_SELFTEST_CRASH") else { return };
    let Some((kind, at)) = v.split_once('@') else { return };
    if at.parse() != Ok(voices) {
        return;
    }
    eprintln!("[ERROR kestrel::falconeye::selftest] failing on purpose: {kind}");
    match kind {
        "segv" => super::winsys::access_violation(),
        "exit" => std::process::exit(3),
        "hang" => loop {
            std::thread::sleep(Duration::from_secs(1));
        },
        _ => {}
    }
}

fn step(bank: &Arc<Bank>, dir: &Path, adapter: &str, backend: &str, voices: u32) -> Result<Step> {
    #[cfg(feature = "dev")]
    fail_on_purpose(voices);
    let midi = dir.join(format!("selftest {voices}.mid"));
    if !midi.exists() {
        crate::testkit::simultaneous_midi(&midi, voices as usize, 10.0)?;
    }
    let cfg = Config {
        max_voices: voices,
        gpu_adapter: Some(adapter.to_string()),
        gpu_backend: Some(backend.to_string()),
        // Timestamps: the GPU's own time for each block, where it keeps them.
        profile: true,
        ..Config::default()
    };
    cfg.validate()?;
    let mut gpu = crate::gpu::GpuSynth::new(&cfg, bank.clone())?;
    let mut driver = Driver::open(&cfg, bank.clone(), &midi)?;
    let mut buf = vec![0.0f32; cfg.block_samples()];
    let mut times = Vec::with_capacity(BLOCKS);
    let mut on_gpu = true;
    let mut live = 0;
    for _ in 0..BLOCKS {
        let t0 = Instant::now();
        driver.next_block(&mut gpu, &mut buf)?;
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        let passes = gpu.timings();
        let device: f64 = passes.iter().map(|(_, ms)| ms).sum();
        if passes.is_empty() {
            on_gpu = false;
        }
        times.push(if passes.is_empty() { wall } else { device });
        live = live.max(gpu.stats().active_voices);
    }
    let spawn_ms = times[0];
    let mut rest = times[1..].to_vec();
    rest.sort_by(f64::total_cmp);
    Ok(Step {
        voices,
        live,
        spawn_ms,
        worst_ms: *rest.last().unwrap_or(&spawn_ms),
        typical_ms: rest.get(rest.len() / 2).copied().unwrap_or(spawn_ms),
        timed_on_gpu: on_gpu,
    })
}

/// The answer, from the biggest step that filled its pool at least halfway: \[19\]
fn verdict(steps: &[Step]) -> (String, Option<f64>, Option<u64>) {
    verdict_for(steps, cfg!(windows))
}

/// `verdict`, for Windows or not. **Only Windows has the 2 s limit**: it resets a \[20\]
fn verdict_for(steps: &[Step], windows: bool) -> (String, Option<f64>, Option<u64>) {
    let Some(s) = steps.iter().rev().find(|s| s.live * 2 >= s.voices as u64 && s.live > 0) else {
        return ("No step filled its pool, so there is nothing to scale from.".into(), None, None);
    };
    let per_voice = s.worst_ms.max(s.spawn_ms) / s.live as f64;
    let default = Config::default().max_voices as f64;
    let at_default = per_voice * default;
    let for_1s = (TARGET_MS / per_voice) as u64;
    let (voices, ms, limit) = (thousands(default as u64), thousands(at_default as u64), thousands(TDR_MS as u64));
    let text = if !windows {
        if at_default < TARGET_MS / 2.0 {
            format!(
                "Comfortable. At the default {voices} voices a block takes this GPU about {ms} ms. A \
                 block would reach 1 s at about {} voices.",
                thousands(round2(for_1s))
            )
        } else if at_default < TDR_MS * 0.75 {
            format!(
                "Slow. At the default {voices} voices a dense block takes this GPU about {ms} ms, and a \
                 screen it also draws may stutter for that long. Keep --max-voices under about {} to \
                 stay near a second a block, or render with --block 1024, which gives it a quarter of \
                 the work at a time.",
                thousands(round2(for_1s))
            )
        } else {
            format!(
                "Very slow. At the default {voices} voices a dense block takes this GPU about {ms} ms. \
                 That is long enough that some systems' GPU watchdogs end the render, and a screen the \
                 GPU also draws can freeze meanwhile. Render with --block 1024, and keep --max-voices \
                 under about {}.",
                thousands(round2(for_1s * 4))
            )
        }
    } else if at_default < TARGET_MS / 2.0 {
        format!(
            "Comfortable. At the default {voices} voices a block takes this GPU about {ms} ms, far \
             under the {limit} ms at which Windows resets it. A block would reach 1 s at about {} \
             voices.",
            thousands(round2(for_1s))
        )
    } else if at_default < TDR_MS * 0.75 {
        format!(
            "Close. At the default {voices} voices a dense block takes this GPU about {ms} ms, and \
             Windows resets it at {limit} ms. Keep --max-voices under about {}, or render with \
             --block 1024, which gives it a quarter of the work at a time.",
            thousands(round2(for_1s))
        )
    } else {
        format!(
            "At risk. At the default {voices} voices a dense block takes this GPU about {ms} ms, {} \
             the {limit} ms at which Windows resets it, which ends the render. Render with \
             --block 1024, and keep --max-voices under about {}.",
            if at_default > TDR_MS { "past" } else { "near" },
            thousands(round2(for_1s * 4))
        )
    };
    let text = format!(
        "{text} (Measured with every voice sounding at once; a real file is usually lighter. In a \
         per-track render, what counts is the voices of all the tracks rendering at once.)"
    );
    (text, Some(at_default), Some(for_1s))
}

/// Down to two significant figures: a voice count to suggest, not to quote.
fn round2(n: u64) -> u64 {
    let mut unit = 1;
    while n / unit >= 100 {
        unit *= 10;
    }
    n / unit * unit
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(voices: u32, ms: f64) -> Step {
        Step { voices, live: voices as u64, spawn_ms: ms, worst_ms: ms, typical_ms: ms, timed_on_gpu: true }
    }

    #[test]
    fn the_verdict_scales_from_the_last_full_step() {
        // About 110 ns a voice, the RTX 5060 Laptop's figure.
        let (text, ms, for_1s) = verdict_for(&[step(1 << 20, 115.0)], true);
        assert!(text.starts_with("Comfortable."), "{text}");
        assert!((ms.unwrap() - 115.0).abs() < 1.0);
        assert!((8_000_000..10_000_000).contains(&for_1s.unwrap()));
        // Twenty times slower: a million voices is 2.3 s.
        let (text, _, _) = verdict_for(&[step(1 << 16, 115.0 * 20.0 / 16.0)], true);
        assert!(text.starts_with("At risk.") && text.contains("--block 1024"), "{text}");
        let (text, _, _) = verdict_for(&[step(1 << 18, 250.0)], true);
        assert!(text.starts_with("Close."), "{text}");
    }

    /// Windows' 2 s limit is Windows'. A Linux user's report, 2026-10-04, ended "past \[21\]
    #[test]
    fn the_verdict_names_windows_only_on_windows() {
        let cases = [
            (step(1 << 20, 115.0), "Comfortable."),
            (step(1 << 18, 250.0), "Slow."),
            // [22]
            (step(1 << 16, 258.3), "Very slow."),
        ];
        for (s, start) in cases {
            let (win, ms_win, for_win) = verdict_for(std::slice::from_ref(&s), true);
            let (other, ms, for_1s) = verdict_for(std::slice::from_ref(&s), false);
            assert!(win.contains("Windows"), "{win}");
            assert!(other.starts_with(start), "{other}");
            for word in ["Windows", "2,000", "resets it"] {
                assert!(!other.contains(word), "{word} on another OS: {other}");
            }
            assert_eq!((ms, for_1s), (ms_win, for_win), "the numbers do not depend on the OS");
        }
        let (text, _, _) = verdict_for(&[step(1 << 16, 258.3)], false);
        assert!(text.contains("4,132 ms") && text.contains("--block 1024") && text.contains("--max-voices"), "{text}");
    }

    /// What the test process wrote to stderr before it died: the Vulkan loader's \[23\]
    #[test]
    fn the_last_messages_leave_out_the_loaders_instance_warnings() {
        let dir = std::env::temp_dir().join(format!("kestrel_last_lines_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("stderr.txt");
        let text = "[WARN  wgpu_hal::vulkan::instance] GENERAL [Loader Message (0x0)]\n\
                    \twindows_read_data_files_in_registry: Registry lookup failed\n\
                    [ERROR wgpu_core::device] Parent device is lost\n\
                    \n\
                    [WARN  wgpu_hal::vulkan::instance] \tobjects: (type: INSTANCE, hndl: 0x1)\n";
        std::fs::write(&p, text).unwrap();
        assert_eq!(last_lines(&p, 6), "[ERROR wgpu_core::device] Parent device is lost");
        assert_eq!(last_lines(&dir.join("not there"), 6), "");
        let many: String = (0..10).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&p, many).unwrap();
        assert_eq!(last_lines(&p, 3), "line 7 | line 8 | line 9");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn an_access_violation_is_named_with_its_code() {
        use std::os::windows::process::ExitStatusExt;
        let s = std::process::ExitStatus::from_raw(0xC000_0005);
        let t = how_it_ended(&s);
        assert!(t.starts_with("crashed: an access violation") && t.ends_with("(exit code 0xc0000005)"), "{t}");
        let t = how_it_ended(&std::process::ExitStatus::from_raw(1));
        assert!(t.contains("failed with an error"), "{t}");
    }

    #[test]
    fn a_step_whose_pool_did_not_fill_is_not_scaled_from() {
        let mut s = step(1 << 20, 50.0);
        s.live = 1000;
        let (text, ms, _) = verdict(&[step(1 << 16, 8.0), s]);
        assert!(ms.is_some() && text.starts_with("Comfortable."), "{text}");
        assert!((ms.unwrap() - 8.0 * 16.0).abs() < 1.0);
    }

    #[test]
    fn a_suggested_count_keeps_two_figures() {
        assert_eq!(round2(225_296), 220_000);
        assert_eq!(round2(10_692_762), 10_000_000);
        assert_eq!(round2(56_324), 56_000);
        assert_eq!(round2(99), 99);
    }

    #[test]
    fn thousands_are_grouped() {
        assert_eq!(thousands(1_048_576), "1,048,576");
        assert_eq!(thousands(16_384), "16,384");
        assert_eq!(thousands(999), "999");
    }
}
