// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The guided renderer: what `kestrel` runs when it is started with no
//! arguments, which is what double-clicking the executable does.
//!
//! A front end and nothing more. Every render it starts is an argument list
//! parsed through the same clap definition as `--force-cli render` and run
//! through the same `render::run`, so a render from here and the equivalent
//! command write byte-identical files. The ignored test at the bottom of this
//! file holds it to that.

mod capture;
/// Also read by `api`, which checks MIDIs, describes soundfonts and decides
/// when a soundfont needs loading again exactly as the guided renderer does.
pub(crate) mod checks;
pub mod extras;
mod progress;
pub(crate) mod style;

use kestrel::session::{self, Job, Plan, Summary};
use crate::{Cli, Cmd, RenderArgs};
use crate::settings::{self, Ring};
use crate::update::{self, Latest};
use anyhow::Result;
use checks::{FontProfile, MidiInfo, Verdict, VoiceAnswer};
use clap::Parser;
use crossterm::tty::IsTty;
use crossterm::{cursor, execute, queue, terminal};
use kestrel::bank::Bank;
use kestrel::config::{BackendKind, Config};
use kestrel::gpu::AdapterSummary;
use kestrel::tracks::{self as ktracks, SetupTracks, TrackKind, TrackList, TrackScan};
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use style::{b, c, s, Line, AMBER, DIM, ERR, OK, WARN};
use unicode_width::UnicodeWidthStr;

// ---- input ----------------------------------------------------------------

/// What a picker asks for. Each kind remembers the folder it was last answered
/// from for the rest of the session. The render flow's three -- MIDI,
/// soundfont and destination -- also keep it in `ktrl.ini`, so a render starts
/// where the last one's files were, even the last time Kestrel ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pick {
    Midi,
    Soundfont,
    Inspect,
    /// Only the null test picks WAVs, and it is in dev builds only.
    #[cfg_attr(not(feature = "dev"), allow(dead_code))]
    Wav,
    Folder,
}

impl Pick {
    const SAVED: [Pick; 3] = [Pick::Midi, Pick::Soundfont, Pick::Folder];

    /// Its line in `ktrl.ini`'s `[folders]`, for the kinds kept there.
    fn key(self) -> Option<&'static str> {
        match self {
            Pick::Midi => Some("midi"),
            Pick::Soundfont => Some("soundfont"),
            Pick::Folder => Some("output"),
            Pick::Inspect | Pick::Wav => None,
        }
    }
}

/// Everything the flow reads from the person at the keyboard. A trait so a
/// test can drive the whole flow with scripted answers.
pub trait Io {
    /// One typed line, or `None` once input has ended.
    fn line(&mut self) -> Option<String>;
    /// `None` when the picker was closed without choosing.
    fn pick_files(&mut self, kind: Pick, title: &str) -> Option<Vec<PathBuf>>;
    fn pick_folder(&mut self, title: &str, start: Option<&Path>) -> Option<PathBuf>;
}

/// The keyboard and the platform's own file pickers.
pub struct Native {
    dirs: HashMap<Pick, PathBuf>,
    /// Set once saving a folder has failed, so the warning is given once.
    unsaved: bool,
}

impl Native {
    /// Opening where `ktrl.ini` says each picker was last answered from.
    pub fn remembered() -> Self {
        let (saved, _) = settings::load();
        let dirs = Pick::SAVED
            .into_iter()
            .filter_map(|kind| {
                let dir = saved.folder(kind.key()?)?;
                Some((kind, dir.to_path_buf()))
            })
            .collect();
        Native { dirs, unsaved: false }
    }

    fn remember(&mut self, kind: Pick, dir: PathBuf) {
        if self.dirs.get(&kind) == Some(&dir) {
            return;
        }
        let saved = match kind.key() {
            Some(key) => settings::update(|s| s.set_folder(key, &dir)),
            None => Ok(()),
        };
        self.dirs.insert(kind, dir);
        if let Err(e) = saved {
            if !self.unsaved {
                self.unsaved = true;
                style::warn(format!(
                    "Couldn't save {} ({e:#}), so folders won't be remembered next time.",
                    settings::FILE
                ));
            }
        }
    }

    /// A remembered folder that still exists. One since deleted or on a drive
    /// no longer attached is passed over, and the picker opens wherever the
    /// system chooses.
    fn dir(&self, kind: Pick) -> Option<&Path> {
        self.dirs.get(&kind).map(PathBuf::as_path).filter(|d| d.is_dir())
    }
}

impl Io for Native {
    fn line(&mut self) -> Option<String> {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_string()),
        }
    }

    fn pick_files(&mut self, kind: Pick, title: &str) -> Option<Vec<PathBuf>> {
        let mut dialog = rfd::FileDialog::new().set_title(title);
        dialog = match kind {
            Pick::Midi => dialog.add_filter("MIDI files", &["mid", "midi"]),
            Pick::Soundfont => dialog.add_filter("SoundFonts", &["sf2", "sfz"]),
            Pick::Inspect => {
                dialog.add_filter("SoundFonts and MIDI files", &["sf2", "sfz", "mid", "midi"])
            }
            Pick::Wav => dialog.add_filter("WAV files", &["wav"]),
            Pick::Folder => dialog,
        };
        dialog = dialog.add_filter("All files", &["*"]);
        if let Some(dir) = self.dir(kind) {
            dialog = dialog.set_directory(dir);
        }
        let picked = match kind {
            Pick::Midi | Pick::Soundfont => dialog.pick_files()?,
            _ => vec![dialog.pick_file()?],
        };
        if let Some(dir) = picked.first().and_then(|p| p.parent()) {
            self.remember(kind, dir.to_path_buf());
        }
        Some(picked)
    }

    fn pick_folder(&mut self, title: &str, start: Option<&Path>) -> Option<PathBuf> {
        let mut dialog = rfd::FileDialog::new().set_title(title);
        if let Some(dir) = self.dir(Pick::Folder).or(start) {
            dialog = dialog.set_directory(dir);
        }
        let picked = dialog.pick_folder()?;
        self.remember(Pick::Folder, picked.clone());
        Some(picked)
    }
}

// ---- small pieces of screen ----------------------------------------------

pub fn prompt() {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(style::MARGIN.as_bytes());
    let _ = style::queue_line(&mut out, &[b("\u{25BA} ", AMBER)]);
    let _ = out.flush();
}

/// One numbered choice in a menu.
pub fn option(key: &str, label: &str, note: &str) {
    style::say(vec![
        s("  "),
        c(format!("[{key}]"), AMBER),
        s(format!(" {label:<26}")),
        c(note.to_string(), DIM),
    ]);
}

pub fn clear_screen() {
    if std::io::stdout().is_tty() {
        let _ = execute!(
            std::io::stdout(),
            terminal::Clear(terminal::ClearType::Purge),
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        );
    }
}

pub fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

fn thousands(n: u32) -> String {
    style::thousands(n as u64)
}

fn panel_width() -> usize {
    terminal::size().map_or(76, |(w, _)| (w as usize).saturating_sub(8).clamp(40, 76))
}

const SPIN: [&str; 4] = [
    "\u{25CF}\u{25CB}\u{25CB}",
    "\u{25CB}\u{25CF}\u{25CB}",
    "\u{25CB}\u{25CB}\u{25CF}",
    "\u{25CB}\u{25CF}\u{25CB}",
];

/// Run `work` on a thread and animate `label` until it is done. Loaders log
/// through `capture`, so nothing else writes to the line meanwhile.
fn spin<T: Send>(label: &str, work: impl FnOnce() -> T + Send) -> T {
    spin_status(label, &|| None, work)
}

/// `spin`, with `status` -- a percentage, say -- shown after the label.
fn spin_status<T: Send>(label: &str, status: &(dyn Fn() -> Option<String> + Sync), work: impl FnOnce() -> T + Send) -> T {
    let tty = std::io::stdout().is_tty();
    std::thread::scope(|scope| {
        let handle = scope.spawn(work);
        let t0 = Instant::now();
        let mut tick = 0usize;
        while tty && !handle.is_finished() {
            let secs = t0.elapsed().as_secs_f64();
            let mut line = vec![c(SPIN[tick % SPIN.len()], AMBER), s(format!(" {label}\u{2026}"))];
            if let Some(st) = status() {
                line.push(s(format!("  {st}")));
            }
            if secs >= 2.0 {
                line.push(c(format!("  {secs:.0} s"), DIM));
            }
            let mut out = std::io::stdout().lock();
            let _ = queue!(
                out,
                cursor::MoveToColumn(0),
                terminal::Clear(terminal::ClearType::CurrentLine)
            );
            let _ = out.write_all(style::MARGIN.as_bytes());
            let _ = style::queue_line(&mut out, &line);
            let _ = out.flush();
            drop(out);
            tick += 1;
            std::thread::sleep(Duration::from_millis(120));
        }
        if tty {
            let _ = execute!(
                std::io::stdout(),
                cursor::MoveToColumn(0),
                terminal::Clear(terminal::ClearType::CurrentLine)
            );
        }
        handle.join().unwrap_or_else(|p| std::panic::resume_unwind(p))
    })
}

// ---- environment ----------------------------------------------------------

struct Ffmpeg {
    version: String,
    source: &'static str,
    /// Extensions this build cannot encode.
    missing: Vec<&'static str>,
}

struct Env {
    adapters: Vec<AdapterSummary>,
    default_adapter: Option<usize>,
    gpu_error: Option<String>,
    ffmpeg: Result<Ffmpeg, String>,
    /// The update ring the check ran on, as `ktrl.ini` had it at startup.
    ring: Ring,
    /// `None` when the check is turned off.
    update: Option<Result<Latest, String>>,
}

impl Env {
    fn default_adapter(&self) -> Option<&AdapterSummary> {
        self.default_adapter.map(|i| &self.adapters[i])
    }
}

fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Gl => "OpenGL",
        _ => "other",
    }
}

/// The `--gpu-backend` spelling of a backend.
pub(crate) fn backend_flag(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Gl => "gl",
        _ => "all",
    }
}

pub(crate) fn kind_name(kind: wgpu::DeviceType) -> &'static str {
    match kind {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "software",
        _ => "other",
    }
}

fn install_hint() -> &'static str {
    if cfg!(windows) {
        "winget install Gyan.FFmpeg"
    } else if cfg!(target_os = "macos") {
        "brew install ffmpeg"
    } else {
        "sudo apt install ffmpeg"
    }
}

fn probe_ffmpeg() -> Result<Ffmpeg, String> {
    let f = kestrel::ffmpeg::find(None).map_err(|e| format!("{e:#}"))?;
    let missing = f
        .missing_encoders()
        .map_err(|e| format!("{e:#}"))?
        .iter()
        .map(|p| p.ext)
        .collect();
    Ok(Ffmpeg {
        version: f.version,
        source: f.source.describe(),
        missing,
    })
}

/// The numbered adapter list, as the environment check shows it.
fn adapter_lines(env: &Env) -> Vec<Line> {
    let name_w = env
        .adapters
        .iter()
        .map(|a| a.name.width())
        .max()
        .unwrap_or(0)
        .min(40);
    env.adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut line = vec![c(format!("[{}] ", i + 1), AMBER)];
            line.extend(style::fit(&[s(a.name.clone())], name_w));
            line.push(c(
                format!("  {:<7}{:<11}", backend_name(a.backend), kind_name(a.device_type)),
                DIM,
            ));
            if a.is_software() {
                line.push(c("never used", DIM));
            } else {
                line.push(s(format!("up to {} voices", thousands(a.max_voices))));
            }
            if Some(i) == env.default_adapter {
                line.push(b("  default", OK));
            }
            line
        })
        .collect()
}

fn check_environment() -> Env {
    style::heading("Checking the environment", "");
    let (saved, problems) = settings::load();
    let ring = saved.ring;
    // Started first, so the request runs behind the adapter survey rather than
    // after it, and is usually answered before anything waits on it.
    let update = (!update::opted_out()).then(|| std::thread::spawn(update::check));
    let cfg = Config::default();
    let survey = spin("Looking for GPU adapters", || kestrel::gpu::survey(&cfg));
    let _ = capture::problems();
    let (adapters, default_adapter, gpu_error) = match survey {
        Ok((list, pick)) => (list, pick, None),
        Err(e) => (Vec::new(), None, Some(format!("{e:#}"))),
    };
    let ffmpeg = spin("Looking for ffmpeg", probe_ffmpeg);
    let update = update.map(|check| match spin("Checking for updates", || check.join()) {
        Ok(Ok(latest)) => Ok(latest),
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(_) => Err("the check failed".into()),
    });
    let env = Env {
        adapters,
        default_adapter,
        gpu_error,
        ffmpeg,
        ring,
        update,
    };

    match (&env.gpu_error, env.default_adapter()) {
        (Some(e), _) => {
            style::status(ERR, vec![s("Couldn't list GPU adapters")]);
            style::detail(vec![c(e.clone(), ERR)]);
        }
        (None, Some(_)) => style::status(OK, vec![s("GPU adapters")]),
        (None, None) => {
            style::status(ERR, vec![s("No usable GPU adapter")]);
            style::detail(vec![c(
                "Kestrel renders on the GPU. --backend cpu at step 6 uses the far slower reference renderer.",
                DIM,
            )]);
        }
    }
    for line in adapter_lines(&env) {
        style::detail(line);
    }

    match &env.ffmpeg {
        Ok(f) => {
            let formats: Vec<&str> = FORMATS
                .iter()
                .filter(|fmt| format_available(&env, fmt.ext).is_ok())
                .map(|fmt| fmt.label)
                .collect();
            style::status(
                OK,
                vec![s("ffmpeg "), c(f.version.clone(), DIM), c(format!("  via {}", f.source), DIM)],
            );
            style::detail(vec![c(format!("writes {}", formats.join(", ")), DIM)]);
        }
        Err(_) => {
            style::status(WARN, vec![s("ffmpeg not found")]);
            style::detail(vec![c(
                format!("WAV only until it is installed: {}", install_hint()),
                DIM,
            )]);
        }
    }

    // An offline machine gets a dim line, not a warning: nothing about a render
    // depends on the answer.
    match &env.update {
        Some(Ok(latest)) if latest.announced(env.ring) => {
            style::status(
                AMBER,
                vec![
                    b(format!("Kestrel {} is out", latest.version), AMBER),
                    c(format!("  you have {}", update::CURRENT), DIM),
                ],
            );
            style::detail(vec![c(latest.url.clone(), DIM)]);
        }
        Some(Ok(_)) if env.ring == Ring::Slow => {
            style::status(OK, vec![s("Kestrel is up to date"), c("  Slow Ring", DIM)])
        }
        Some(Ok(_)) => style::status(OK, vec![s("Kestrel is up to date")]),
        Some(Err(e)) => {
            style::status(DIM, vec![c("Couldn't check for updates", DIM)]);
            style::detail(vec![c(e.clone(), DIM)]);
        }
        None => {}
    }
    for problem in problems {
        style::warn(format!("{}: {problem}", settings::FILE));
    }
    env
}

/// The environment in two lines, for the top of every screen after the first,
/// and a third while a newer release is out.
fn compact_env(env: &Env) {
    match env.default_adapter() {
        Some(a) => style::status(
            OK,
            vec![
                s(a.name.clone()),
                c(
                    format!(
                        "  {} \u{00B7} up to {} voices",
                        backend_name(a.backend),
                        thousands(a.max_voices)
                    ),
                    DIM,
                ),
            ],
        ),
        None => style::status(ERR, vec![s("No usable GPU adapter")]),
    }
    match &env.ffmpeg {
        Ok(f) => style::status(OK, vec![s("ffmpeg "), c(f.version.clone(), DIM)]),
        Err(_) => style::status(WARN, vec![s("ffmpeg not found"), c("  WAV only", DIM)]),
    }
    if let Some(Ok(latest)) = &env.update {
        if latest.announced(env.ring) {
            style::status(
                AMBER,
                vec![
                    b(format!("Kestrel {} is out", latest.version), AMBER),
                    c(format!("  {}", latest.url), DIM),
                ],
            );
        }
    }
}

// ---- formats --------------------------------------------------------------

struct Format {
    ext: &'static str,
    label: &'static str,
}

static FORMATS: [Format; 5] = [
    Format { ext: "wav", label: "WAV" },
    Format { ext: "flac", label: "FLAC" },
    Format { ext: "opus", label: "Opus" },
    Format { ext: "ogg", label: "OGG" },
    Format { ext: "mp3", label: "MP3" },
];

fn format_note(ext: &str) -> String {
    if ext == "wav" {
        return "32-bit float, uncompressed".into();
    }
    kestrel::ffmpeg::preset_for(ext)
        .map(|p| p.note.to_string())
        .unwrap_or_default()
}

fn format_available(env: &Env, ext: &str) -> Result<(), String> {
    if ext == "wav" {
        return Ok(());
    }
    match &env.ffmpeg {
        Err(_) => Err("needs ffmpeg, which wasn't found".into()),
        Ok(f) if f.missing.contains(&ext) => Err(format!(
            "isn't in this ffmpeg build, which has no {}",
            kestrel::ffmpeg::preset_for(ext).map_or("encoder for it", |p| p.encoder)
        )),
        Ok(_) => Ok(()),
    }
}

// ---- the flow -------------------------------------------------------------

/// `kestrel` with no arguments.
pub fn run() -> Result<()> {
    style::init();
    capture::install();
    capture::hold_panics();
    if std::io::stdout().is_tty() {
        let _ = execute!(std::io::stdout(), terminal::SetTitle("Kestrel"));
    }
    let mut io = Native::remembered();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| guided(&mut io)));
    let failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(format!("{e:#}")),
        Err(_) => Some(match capture::panic_message() {
            Some(m) => format!("Kestrel hit an internal error: {m}"),
            None => "Kestrel hit an internal error.".into(),
        }),
    };
    if let Some(message) = failure {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(std::io::stdout(), cursor::Show);
        style::blank();
        style::error(message);
        // A window opened by double-clicking closes the moment this returns,
        // and takes the message with it.
        style::say(vec![c("Press Enter to close.", DIM)]);
        let _ = io.line();
    }
    Ok(())
}

/// What `kestrel <anything>` prints when `--force-cli` is not among it.
pub fn notice(args: &[OsString]) {
    style::init();
    style::print_banner();
    style::say(vec![b("Kestrel is better experienced from the executable.", AMBER)]);
    style::blank();
    style::say(vec![
        s("Open it directly (double-click it, or run "),
        c("kestrel", AMBER),
        s(" with no arguments) for the"),
    ]);
    style::say(vec![s(
        "guided renderer: file pickers, a live progress screen, and the same engine.",
    )]);
    style::blank();
    style::say(vec![c(
        "To run this command from the terminal anyway, add --force-cli:",
        DIM,
    )]);
    let mut cmd = String::from("kestrel --force-cli");
    for a in args {
        cmd.push(' ');
        cmd.push_str(&quote(&a.to_string_lossy()));
    }
    style::say(vec![s("  "), c(cmd, AMBER)]);
    style::blank();
}

fn quote(arg: &str) -> String {
    let plain = !arg.is_empty()
        && !arg.contains(|ch: char| ch.is_whitespace() || ch == '"' || ch == '\'');
    if plain {
        arg.to_string()
    } else {
        format!("\"{}\"", arg.replace('"', "\\\""))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Choice {
    Render,
    PerTrack,
    Extras,
    Exit,
}

#[derive(PartialEq, Eq)]
enum Next {
    Menu,
    Exit,
}

/// A step's answer, or the way out of the flow it asked for.
enum Step<T> {
    Got(T),
    Menu,
    Exit,
}

macro_rules! step {
    ($e:expr) => {
        match $e {
            Step::Got(v) => v,
            Step::Menu => return Ok(Next::Menu),
            Step::Exit => return Ok(Next::Exit),
        }
    };
}

fn guided(io: &mut dyn Io) -> Result<()> {
    clear_screen();
    style::print_banner();
    let env = check_environment();
    loop {
        match main_menu(io) {
            Choice::Render => {
                if render_loop(io, &env)? == Next::Exit {
                    return Ok(());
                }
            }
            Choice::PerTrack => {
                if per_track_loop(io, &env)? == Next::Exit {
                    return Ok(());
                }
            }
            Choice::Extras => match extras::open_window() {
                Ok(()) => {
                    style::blank();
                    style::status(OK, vec![s("Extras opened in a new window.")]);
                    continue;
                }
                Err(e) => {
                    style::warn(format!(
                        "Couldn't open a new window ({e}), so Extras will run here."
                    ));
                    extras::menu(io);
                }
            },
            Choice::Exit => return Ok(()),
        }
        clear_screen();
        style::print_banner();
        compact_env(&env);
    }
}

fn main_menu(io: &mut dyn Io) -> Choice {
    style::heading("What would you like to do?", "");
    option("1", "Single / Multiple MIDIs", "render a MIDI to audio");
    option("2", "Per-Track Render", "each track alone: a file each, or one mix");
    // The null test is a dev-build Extras entry; say only what this build has.
    let extras = if cfg!(feature = "dev") {
        "GPU info, file info, null test, flag help, updates"
    } else {
        "GPU info, file info, flag help, updates"
    };
    option("3", "Extras", extras);
    option("4", "Exit", "");
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Choice::Exit;
        };
        match line.trim() {
            "1" => return Choice::Render,
            "2" => return Choice::PerTrack,
            "3" => return Choice::Extras,
            "4" | "0" | "q" | "Q" => return Choice::Exit,
            _ => style::error("Type 1, 2, 3 or 4."),
        }
    }
}

/// Renders, one after another, until the person stops asking for them.
fn render_loop(io: &mut dyn Io, env: &Env) -> Result<Next> {
    loop {
        clear_screen();
        style::print_banner();
        compact_env(env);

        let midi = step!(step_midi(io, (1, 6)));
        let fonts = step!(step_soundfonts(io, (2, 6)));
        let voices = step!(step_voices(io, env, (3, 6), None));
        let format = step!(step_format(io, env, (4, 6)));
        let folder = step!(step_folder(io, &midi.path, (5, 6), "the folder to write into"));
        let out = checks::output_path(&folder, &midi.path, format.ext, &checks::timestamp());
        let mut writes = vec![c("Writes ", DIM), b(file_name(&out), AMBER)];
        if out.file_stem() != midi.path.file_stem() {
            writes.push(c(
                "  (that name was taken, so this one carries the time)",
                DIM,
            ));
        }
        style::detail(writes);
        let ready = step!(step_flags(io, env, &midi, &fonts, voices, &out, (6, 6), &[]));
        // The render holds its own handle on this bank when it can use it,
        // and loads a new one when a typed flag changes how soundfonts load.
        // Either way this handle is done with. Kept, it held the step 2 bank
        // resident under the reloaded one: 1.9 GiB against 1.2 GiB for a
        // 757 MiB piano with `--volume 80` typed (2026-09-18).
        drop(fonts.bank);

        let labels = progress::Labels {
            midi: file_name(&midi.path),
            output: file_name(&out),
            format: format!("{} \u{00B7} {}", format.label, format_note(format.ext)),
            fonts: fonts.names.clone(),
            max_voices: ready.plan.cfg.max_voices,
            per_track: false,
        };
        let _ = capture::problems();
        let result = progress::run(&ready.job, ready.plan, ready.bank, &labels);
        let problems = capture::problems();
        // `output_path` only ever names a file that did not exist, so whatever
        // is there now came from this render and is safe to remove.
        let result = match result {
            Ok(summary) => {
                if summary.cancelled {
                    let _ = std::fs::remove_file(&out);
                }
                Ok(summary)
            }
            Err(e) => {
                let _ = std::fs::remove_file(&out);
                Err(format!("{e:#}"))
            }
        };

        clear_screen();
        style::print_banner();
        show_outcome(&out, ready.adapter.as_deref(), &ready.flags, &result, &problems, None);
        if what_next(io) == Next::Exit {
            return Ok(Next::Exit);
        }
    }
}

/// After a render: another, or out.
fn what_next(io: &mut dyn Io) -> Next {
    {
        style::heading("What next?", "");
        option("1", "Start another render", "");
        option("2", "Exit", "");
        loop {
            prompt();
            let Some(line) = io.line() else {
                return Next::Exit;
            };
            match line.trim() {
                "1" => return Next::Menu,
                "2" | "0" | "q" | "Q" => return Next::Exit,
                _ => style::error("Type 1 or 2."),
            }
        }
    }
}

/// Per-track renders, one after another, until the person stops asking.
/// Every track is rendered alone, as `--track` renders one, and written as a
/// stem or summed into one file; see `kestrel::stems`. The render is the
/// argument list the steps build, parsed by clap like any other.
fn per_track_loop(io: &mut dyn Io, env: &Env) -> Result<Next> {
    const OF: u8 = 8;
    loop {
        clear_screen();
        style::print_banner();
        compact_env(env);

        let (midi, scan) = loop {
            let midi = step!(step_midi(io, (1, OF)));
            if let Some(scan) = scan_tracks(&midi) {
                break (midi, scan);
            }
        };
        let fonts = step!(step_soundfonts(io, (2, OF)));
        let tracks = step!(step_tracks(io, &scan, (3, OF)));
        let voices = step!(step_voices(io, env, (4, OF), Some((tracks.named, &fonts.bank))));
        let merged = step!(step_output(io, (5, OF)));
        let format = step!(step_format(io, env, (6, OF)));
        let note = if merged { "the folder to write the file into" } else { "a folder named after the MIDI goes in here" };
        let folder = step!(step_folder(io, &midi.path, (7, OF), note));

        // Merged, a file, named as a normal render's is and never over one
        // already there. Stems, the folder the stems' own folder goes in.
        let stem_name = midi.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "stems".into());
        let (out, shown) = if merged {
            let out = checks::output_path(&folder, &midi.path, format.ext, &checks::timestamp());
            let shown = file_name(&out);
            (out, shown)
        } else {
            let into = folder.join(&stem_name);
            if std::fs::read_dir(&into).is_ok_and(|mut d| d.next().is_some()) {
                style::warn(format!(
                    "{} already has files in it. Stems of the same names will be replaced.",
                    into.display()
                ));
                if step!(confirm(io)) {
                    continue;
                }
            }
            (folder.clone(), format!("{stem_name}{}", std::path::MAIN_SEPARATOR))
        };
        style::detail(vec![c("Writes ", DIM), b(shown.clone(), AMBER)]);

        let mut base: Vec<OsString> = vec![
            format!("--tracks={}", tracks.spec).into(),
            format!("--setup-tracks={}", if tracks.setup == SetupTracks::Ignore { "ignore" } else { "apply" }).into(),
        ];
        if merged {
            base.push("--merge".into());
        } else {
            base.push(format!("--stem-format={}", format.ext).into());
        }
        let mut ready = step!(step_flags(io, env, &midi, &fonts, voices, &out, (8, OF), &base));
        drop(fonts.bank);

        // What the render will do, now that --min-velocity is known.
        let min = ready.plan.cfg.min_velocity;
        let kept = TrackList::parse(&tracks.spec)
            .and_then(|l| l.resolve(&scan))
            .map(|(named, _)| named.into_iter().filter(|&t| scan.tracks[t].notes_from(min) > 0).count())
            .unwrap_or(0);
        if kept == 0 {
            style::error(format!("No track has a note at velocity {min} or above, so there is nothing to render."));
            if what_next(io) == Next::Exit {
                return Ok(Next::Exit);
            }
            continue;
        }
        // As `stems::run` settles it: never more than the tracks.
        let threads = ready.job.stems.as_ref().map_or(1, |s| s.jobs).min(ktracks::max_jobs()).min(kept).max(1);
        let secs = ktracks::render_secs(&scan, ready.plan.cfg.sample_rate, ready.job.seconds);
        let float32 = ready.job.wav_format == kestrel::wav::SampleFormat::Float32;
        let (bytes, exact) = ktracks::output_bytes(if merged { 1 } else { kept }, secs, ready.plan.cfg.sample_rate, format.ext, float32);
        style::status(
            OK,
            vec![
                b(format!("{} track{}", style::thousands(kept as u64), plural(kept)), AMBER),
                s(format!(
                    " \u{00B7} {} voices each \u{00B7} {threads} thread{} \u{00B7} ",
                    thousands(ktracks::voices_each(ready.plan.cfg.max_voices, kept)),
                    plural(threads)
                )),
                s(if merged {
                    "one file".to_string()
                } else {
                    format!("{} files", style::thousands(kept as u64))
                }),
                c(format!("  {} {}", if exact { "about" } else { "at most" }, style::bytes(bytes)), DIM),
            ],
        );
        if let Some(stems) = ready.job.stems.as_mut() {
            stems.scanned = Some(scan.clone());
        }

        let labels = progress::Labels {
            midi: file_name(&midi.path),
            output: shown,
            format: format!("{} \u{00B7} {}", format.label, format_note(format.ext)),
            fonts: fonts.names.clone(),
            max_voices: ready.plan.cfg.max_voices,
            per_track: true,
        };
        let _ = capture::problems();
        let result = progress::run(&ready.job, ready.plan, ready.bank, &labels);
        let problems = capture::problems();
        let result = result.map_err(|e| format!("{e:#}"));

        clear_screen();
        style::print_banner();
        let at = if merged { out.clone() } else { out.join(&stem_name) };
        show_outcome(&at, ready.adapter.as_deref(), &ready.flags, &result, &problems, Some(PerTrack { tracks: kept, merged }));
        if what_next(io) == Next::Exit {
            return Ok(Next::Exit);
        }
    }
}

/// `[Enter]` goes on, `[C]` chooses again: true for choosing again.
fn confirm(io: &mut dyn Io) -> Step<bool> {
    style::say(vec![
        c("[Enter]", AMBER),
        c(" go ahead   ", DIM),
        c("[C]", AMBER),
        c(" start over   ", DIM),
        c("[0]", AMBER),
        c(" back to the menu", DIM),
    ]);
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        match line.trim().to_ascii_lowercase().as_str() {
            "" => return Step::Got(false),
            "c" => return Step::Got(true),
            "0" => return Step::Menu,
            _ => style::error("Press Enter to go ahead, C to start over, or 0 for the menu."),
        }
    }
}

/// Read every track of `midi`, and say what they hold. `None`, having said
/// why, when there is nothing to render in it.
fn scan_tracks(midi: &MidiInfo) -> Option<Arc<TrackScan>> {
    let progress = ktracks::ScanProgress::default();
    let status = || {
        let total = progress.bytes_total.load(std::sync::atomic::Ordering::Relaxed);
        let read = progress.bytes_read.load(std::sync::atomic::Ordering::Relaxed);
        (total > 0).then(|| format!("{:.0}%", read as f64 * 100.0 / total as f64))
    };
    let t0 = Instant::now();
    let label = format!("Reading {} track{}", style::thousands(midi.tracks as u64), plural(midi.tracks));
    let scanned = spin_status(&label, &status, || ktracks::scan(&midi.path, 0, Some(&progress)));
    let _ = capture::problems();
    let scan = match scanned {
        Ok(scan) => scan,
        Err(e) => {
            style::error(format!("{e:#}"));
            style::say(vec![c("Choose another.", DIM)]);
            return None;
        }
    };
    let count = |kind| scan.of_kind(kind).count();
    let (with, setup, empty) = (count(TrackKind::Notes), count(TrackKind::Setup), count(TrackKind::Empty));
    if with == 0 {
        style::status(ERR, vec![s("No track in this file has a note in it, so there is nothing to render.")]);
        style::say(vec![c("Choose another.", DIM)]);
        return None;
    }
    let notes = scan.notes();
    style::status(
        OK,
        vec![
            s(format!("{} tracks read", style::thousands(scan.tracks.len() as u64))),
            c(format!("  in {:.1} s", t0.elapsed().as_secs_f64()), DIM),
        ],
    );
    style::detail(vec![c(
        format!(
            "{} with notes \u{00B7} {} setup \u{00B7} {} empty \u{00B7} {} notes \u{00B7} {} long",
            style::thousands(with as u64),
            style::thousands(setup as u64),
            style::thousands(empty as u64),
            style::thousands(notes),
            style::audio_clock(scan.duration(48_000))
        ),
        DIM,
    )]);
    if let Some(t) = scan.busiest() {
        let info = &scan.tracks[t];
        style::detail(vec![
            c("busiest: ", DIM),
            s(format!("track {}", t + 1)),
            s(info.display_name().map(|n| format!(" ({n})")).unwrap_or_default()),
            c(
                format!(
                    ", {} notes, {:.1}%",
                    style::thousands(info.notes()),
                    info.notes() as f64 * 100.0 / notes.max(1) as f64
                ),
                DIM,
            ),
        ]);
    }
    Some(Arc::new(scan))
}

/// What step 3 chose.
struct Tracks {
    /// As `--tracks` takes it.
    spec: String,
    setup: SetupTracks,
    /// Tracks with notes it names.
    named: usize,
}

fn step_tracks(io: &mut dyn Io, scan: &TrackScan, at: (u8, u8)) -> Step<Tracks> {
    style::heading(&step_title(at, "Tracks"), "which to render, each alone");
    // No listing here: `kestrel --force-cli tracks` prints every track, and
    // the user found a list in this step redundant beside it (2026-09-24).
    let all = scan.of_kind(TrackKind::Notes).count();
    style::detail(vec![c("kestrel --force-cli tracks lists every track, with its notes and name", DIM)]);
    option("Enter", "every track with notes", &format!("{} track{}", style::thousands(all as u64), plural(all)));
    style::say(vec![c("  or type track numbers and ranges: 1-40, 57, 90-", DIM)]);
    let (spec, named) = loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        let typed = line.trim();
        let spec = if typed.is_empty() || typed.eq_ignore_ascii_case("enter") { "all" } else { typed };
        match TrackList::parse(spec).and_then(|l| l.resolve(scan)) {
            Ok((named, _)) if named.is_empty() => {
                style::error("None of those tracks has notes. Type others, or press Enter for all of them.")
            }
            Ok((named, without)) => {
                if !without.is_empty() {
                    style::warn(format!(
                        "{} of those {} no notes and {} skipped.",
                        style::thousands(without.len() as u64),
                        if without.len() == 1 { "has" } else { "have" },
                        if without.len() == 1 { "is" } else { "are" }
                    ));
                }
                style::status(
                    OK,
                    vec![b(style::thousands(named.len() as u64), AMBER), s(format!(" track{}", plural(named.len())))],
                );
                break (spec.to_string(), named.len());
            }
            Err(e) => style::error(format!("{e:#}")),
        }
    };

    let setups = scan.of_kind(TrackKind::Setup).count();
    let mut setup = SetupTracks::Apply;
    if setups > 0 {
        style::say(vec![c(
            format!(
                "{} setup track{}: controllers, programs and bends with no notes of {}.",
                style::thousands(setups as u64),
                plural(setups),
                if setups == 1 { "its own" } else { "their own" }
            ),
            DIM,
        )]);
        option("Enter", "apply to every track", "as a render of the whole file would");
        option("2", "ignore", "each track hears only its own");
        loop {
            prompt();
            let Some(line) = io.line() else {
                return Step::Exit;
            };
            match line.trim() {
                "" => break,
                "2" => {
                    setup = SetupTracks::Ignore;
                    break;
                }
                _ => style::error("Press Enter to apply them, or type 2 to ignore them."),
            }
        }
    }
    Step::Got(Tracks { spec, setup, named })
}

fn step_output(io: &mut dyn Io, at: (u8, u8)) -> Step<bool> {
    style::heading(&step_title(at, "Output"), "one file, or a file a track");
    option("1", "One merged file", "every track rendered alone, summed, limited once");
    option("2", "A file per track", "stems, in a folder named after the MIDI");
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        match line.trim() {
            "1" => {
                style::status(OK, vec![s("One merged file")]);
                return Step::Got(true);
            }
            "2" => {
                style::status(OK, vec![s("A file per track")]);
                return Step::Got(false);
            }
            _ => style::error("Type 1 or 2."),
        }
    }
}

/// A picker was closed without choosing. Choosing again is the likelier
/// intent, so it is what Enter does.
fn after_cancel(io: &mut dyn Io) -> Step<()> {
    style::say(vec![
        c("Nothing chosen.  ", DIM),
        c("[Enter]", AMBER),
        c(" choose again   ", DIM),
        c("[0]", AMBER),
        c(" back to the menu", DIM),
    ]);
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        match line.trim() {
            "" => return Step::Got(()),
            "0" => return Step::Menu,
            _ => style::error("Press Enter to choose again, or type 0 for the menu."),
        }
    }
}

/// Pick files from as many folders as it takes. The Windows picker selects
/// several files only inside one folder, and a General MIDI bank and a piano
/// rarely share one, so after each pick the person can add more from somewhere
/// else before going on (asked for 2026-09-12).
fn pick_many(io: &mut dyn Io, kind: Pick, title: &str) -> Step<Vec<PathBuf>> {
    const SHOWN: usize = 5;
    let mut chosen: Vec<PathBuf> = Vec::new();
    loop {
        style::say(vec![c("Opening the file picker\u{2026}", DIM)]);
        match io.pick_files(kind, title) {
            Some(files) if !files.is_empty() => {
                for f in files {
                    if !chosen.contains(&f) {
                        chosen.push(f);
                    }
                }
            }
            _ if chosen.is_empty() => match after_cancel(io) {
                Step::Got(()) => continue,
                Step::Menu => return Step::Menu,
                Step::Exit => return Step::Exit,
            },
            // Closing the picker after adding some keeps what was added.
            _ => return Step::Got(chosen),
        }
        for f in chosen.iter().take(SHOWN) {
            style::detail(vec![c(file_name(f), DIM)]);
        }
        if chosen.len() > SHOWN {
            style::detail(vec![c(
                format!("\u{2026} and {} more", style::thousands((chosen.len() - SHOWN) as u64)),
                DIM,
            )]);
        }
        style::say(vec![
            c(format!("{} selected.  ", chosen.len()), DIM),
            c("[Enter]", AMBER),
            c(" continue   ", DIM),
            c("[A]", AMBER),
            c(" add from another folder   ", DIM),
            c("[C]", AMBER),
            c(" start over", DIM),
        ]);
        loop {
            prompt();
            let Some(line) = io.line() else {
                return Step::Exit;
            };
            match line.trim().to_ascii_lowercase().as_str() {
                "" => return Step::Got(chosen),
                "a" => break,
                "c" => {
                    chosen.clear();
                    break;
                }
                _ => style::error("Press Enter to continue, A to add more, or C to start over."),
            }
        }
    }
}

/// A step's heading: where it is in the flow, and what it asks.
fn step_title(at: (u8, u8), name: &str) -> String {
    format!("Step {} of {} \u{00B7} {name}", at.0, at.1)
}

fn step_midi(io: &mut dyn Io, at: (u8, u8)) -> Step<MidiInfo> {
    style::heading(&step_title(at, "MIDI"), "the file to render");
    loop {
        let picked = match pick_many(io, Pick::Midi, "Kestrel \u{00B7} choose a MIDI file") {
            Step::Got(p) => p,
            Step::Menu => return Step::Menu,
            Step::Exit => return Step::Exit,
        };

        let verdicts = if picked.len() == 1 {
            vec![checks::check_midi(&picked[0])]
        } else {
            let label = format!("Checking {} files", style::thousands(picked.len() as u64));
            spin(&label, || checks::check_midis(&picked))
        };

        if verdicts.len() == 1 {
            match verdicts.into_iter().next() {
                Some(Verdict::Valid(info)) => {
                    style::status(OK, vec![s("Selected MIDI is valid")]);
                    style::detail(vec![
                        b(file_name(&info.path), AMBER),
                        c(
                            format!(
                                "  {} \u{00B7} format {} \u{00B7} {} track{} \u{00B7} {}",
                                style::bytes(info.size),
                                info.format,
                                style::thousands(info.tracks as u64),
                                plural(info.tracks),
                                checks::describe_division(info.division)
                            ),
                            DIM,
                        ),
                    ]);
                    for note in &info.notes {
                        style::detail(vec![c(note.clone(), WARN)]);
                    }
                    return Step::Got(info);
                }
                Some(Verdict::Invalid { path, reason }) => {
                    style::status(
                        ERR,
                        vec![s("Selected file can't be rendered: "), c(reason, ERR)],
                    );
                    style::detail(vec![c(file_name(&path), DIM)]);
                    style::say(vec![c("Choose another.", DIM)]);
                }
                None => {}
            }
            continue;
        }

        let total = verdicts.len();
        let valid = verdicts.iter().filter(|v| v.is_valid()).count();
        let bad: Vec<(PathBuf, String)> = verdicts
            .into_iter()
            .filter_map(|v| match v {
                Verdict::Invalid { path, reason } => Some((path, reason)),
                Verdict::Valid(_) => None,
            })
            .collect();
        let of = |n: usize| {
            format!(
                "{} of {}",
                style::thousands(n as u64),
                style::thousands(total as u64)
            )
        };
        if valid > 0 {
            style::status(OK, vec![s(format!("{} files are valid MIDIs", of(valid)))]);
        }
        if !bad.is_empty() {
            style::status(
                ERR,
                vec![s(format!("{} files reported as unable to be rendered", of(bad.len())))],
            );
            const SHOWN: usize = 8;
            let name_w = bad
                .iter()
                .take(SHOWN)
                .map(|(p, _)| file_name(p).width())
                .max()
                .unwrap_or(0)
                .min(36);
            for (path, reason) in bad.iter().take(SHOWN) {
                let mut line = style::fit(&[s(style::middle(&file_name(path), name_w))], name_w);
                line.push(c(format!("  {reason}"), DIM));
                style::detail(line);
            }
            if bad.len() > SHOWN {
                style::detail(vec![c(
                    format!(
                        "\u{2026} and {} more",
                        style::thousands((bad.len() - SHOWN) as u64)
                    ),
                    DIM,
                )]);
            }
        }
        style::warn("Multiple-MIDI loading isn't available yet. Choose a single MIDI.");
    }
}

struct Fonts {
    /// In layer order, base first.
    paths: Vec<PathBuf>,
    names: Vec<String>,
    bank: Arc<Bank>,
}

/// The loading settings of a render with no extra flags, got the way a render
/// gets them, so a bank built here is the bank the render would build.
fn load_config() -> Result<Config> {
    let argv: Vec<OsString> = ["kestrel", "render", "--soundfont=font.sf2", "--out=render.wav", "--", "song.mid"]
        .into_iter()
        .map(OsString::from)
        .collect();
    let args = parse_render(argv).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(args.to_config()?.0)
}

fn parse_render(argv: Vec<OsString>) -> std::result::Result<RenderArgs, clap::Error> {
    match Cli::try_parse_from(argv)?.cmd {
        Cmd::Render(args) => Ok(args),
        _ => unreachable!("the argument list names the render subcommand"),
    }
}

fn program_list(programs: &[u16]) -> String {
    match programs {
        [] => "no bank 0 programs".into(),
        [one] => format!("program {one} ({})", checks::gm_name(*one)),
        few if few.len() <= 3 => format!(
            "programs {}",
            few.iter()
                .map(|p| format!("{p} ({})", checks::gm_name(*p)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        many => format!("{} programs", many.len()),
    }
}

fn describe_font(name: &str, p: &FontProfile, secs: f64) {
    style::status(OK, vec![b(name.to_string(), AMBER)]);
    let kind = if p.is_general_midi() {
        vec![
            b("General MIDI bank", OK),
            c(
                format!(
                    "  {} programs \u{00B7} {} drum kit{}",
                    p.melodic_programs,
                    p.drum_kits,
                    plural(p.drum_kits)
                ),
                DIM,
            ),
        ]
    } else {
        let mut v = vec![b("Instrument", OK), c(format!("  {}", program_list(&p.programs)), DIM)];
        if p.drum_kits > 0 {
            v.push(c(format!(" \u{00B7} {} drum kit{}", p.drum_kits, plural(p.drum_kits)), DIM));
        }
        v
    };
    style::detail(kind);
    let rate = if p.pool_rate == 0 {
        "mixed rates".to_string()
    } else {
        format!("{} Hz", p.pool_rate)
    };
    style::detail(vec![c(
        format!(
            "{} preset{} \u{00B7} {} regions \u{00B7} {} samples \u{00B7} {} pool at {} \u{00B7} loaded in {:.1} s",
            style::thousands(p.presets as u64),
            plural(p.presets),
            style::thousands(p.regions as u64),
            style::thousands(p.samples as u64),
            style::bytes(p.pool_bytes),
            rate,
            secs
        ),
        DIM,
    )]);
}

fn describe_layering(layers: &[(PathBuf, Bank, FontProfile)], reordered: bool) {
    match layers {
        [(path, _, p)] => {
            let role = if p.is_general_midi() {
                " as the General MIDI bank"
            } else {
                " on every channel"
            };
            style::status(AMBER, vec![s("Using "), b(file_name(path), AMBER), s(role)]);
        }
        [(base, _, bp), (top, _, tp)] => {
            style::status(
                AMBER,
                vec![
                    s("Layering "),
                    b(file_name(top), AMBER),
                    s(" on top of "),
                    b(file_name(base), AMBER),
                ],
            );
            let shared: Vec<u16> = tp
                .programs
                .iter()
                .copied()
                .filter(|p| bp.programs.contains(p))
                .collect();
            if !shared.is_empty() {
                style::detail(vec![c(
                    format!("{} replaces {}", file_name(top), program_list(&shared)),
                    DIM,
                )]);
            }
            let note = if reordered {
                Some("The General MIDI bank goes underneath, whichever order they were picked in.")
            } else if !bp.is_general_midi() && !tp.is_general_midi() {
                Some("Neither is a General MIDI bank, so they layer in the order picked.")
            } else if bp.is_general_midi() && tp.is_general_midi() {
                Some("Both are General MIDI banks; the one picked second goes on top.")
            } else {
                None
            };
            if let Some(note) = note {
                style::detail(vec![c(note, DIM)]);
            }
        }
        _ => {}
    }
}

fn step_soundfonts(io: &mut dyn Io, at: (u8, u8)) -> Step<Fonts> {
    style::heading(
        &step_title(at, "Soundfonts"),
        "up to two: a General MIDI bank, a piano, or both",
    );
    let cfg = match load_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            style::error(format!("{e:#}"));
            return Step::Menu;
        }
    };
    'pick: loop {
        let title = "Kestrel \u{00B7} choose up to two soundfonts";
        let picked = match pick_many(io, Pick::Soundfont, title) {
            Step::Got(p) => p,
            Step::Menu => return Step::Menu,
            Step::Exit => return Step::Exit,
        };
        if picked.len() > 2 {
            style::warn(format!(
                "{} soundfonts selected. Kestrel layers at most two, a General MIDI bank \
                 and a piano. Choose again.",
                picked.len()
            ));
            continue;
        }

        let mut loaded: Vec<Option<(PathBuf, Bank, FontProfile)>> = Vec::new();
        for path in &picked {
            let name = file_name(path);
            let t0 = Instant::now();
            let result = spin(&format!("Loading {name}"), || kestrel::load_bank(path, &cfg));
            let problems = capture::problems();
            match result {
                Ok(bank) => {
                    let profile = FontProfile::of(&bank);
                    describe_font(&name, &profile, t0.elapsed().as_secs_f64());
                    for (_, p) in problems {
                        style::detail(vec![b("WARN ", WARN), c(p, WARN)]);
                    }
                    loaded.push(Some((path.clone(), bank, profile)));
                }
                Err(e) => {
                    style::status(ERR, vec![b(name, ERR), c(" couldn't be loaded", ERR)]);
                    style::detail(vec![c(format!("{e:#}"), ERR)]);
                    style::say(vec![c("Choose again.", DIM)]);
                    continue 'pick;
                }
            }
        }

        let profiles: Vec<FontProfile> = loaded
            .iter()
            .flatten()
            .map(|(_, _, p)| p.clone())
            .collect();
        let order = checks::layer_order(&profiles);
        let reordered = order.iter().enumerate().any(|(i, &j)| i != j);
        let mut layers: Vec<(PathBuf, Bank, FontProfile)> =
            order.iter().filter_map(|&i| loaded[i].take()).collect();
        describe_layering(&layers, reordered);

        let paths: Vec<PathBuf> = layers.iter().map(|(p, _, _)| p.clone()).collect();
        let names = paths.iter().map(|p| file_name(p)).collect();
        let (_, first, _) = layers.remove(0);
        let rest: Vec<Bank> = layers.into_iter().map(|(_, bank, _)| bank).collect();
        let stacked = if rest.is_empty() {
            Ok(first)
        } else {
            let rest_len = rest.len();
            spin("Layering", || {
                session::stack(first, rest_len, rest.into_iter().map(Ok), None, &cfg)
            })
        };
        let _ = capture::problems();
        match stacked {
            Ok(bank) => {
                return Step::Got(Fonts {
                    paths,
                    names,
                    bank: Arc::new(bank),
                })
            }
            Err(e) => {
                style::error(format!("{e:#}"));
                style::say(vec![c("Choose again.", DIM)]);
            }
        }
    }
}

/// For a per-track render, `per_track` is the tracks and the soundfont: the
/// total is shared between the tracks and defaults to `tracks::GUIDED_VOICES`.
/// Its device max is what that many tracks hold with as many on the device at
/// once as a batch takes, capped at `tracks::MAX_TOTAL_VOICES`. More may be
/// typed and runs fewer tracks at a time; a typed total past the cap is held
/// to it. As the user set it, 2026-09-24.
fn step_voices(io: &mut dyn Io, env: &Env, at: (u8, u8), per_track: Option<(usize, &kestrel::Bank)>) -> Step<u32> {
    match per_track {
        None => style::heading(&step_title(at, "Voices"), "how many may sound at once"),
        Some((n, _)) => style::heading(
            &step_title(at, "Voices"),
            &format!("the total, shared evenly between the {} track{}", style::thousands(n as u64), plural(n)),
        ),
    }
    let device = env.default_adapter();
    let cfg = Config::default();
    let ceiling = ktracks::MAX_TOTAL_VOICES;
    let max = match (device, per_track) {
        (Some(a), Some((n, bank))) => Some(ktracks::max_voices_at_once(&cfg, bank, a.binding_bytes, n)),
        (Some(a), None) => Some(a.max_voices),
        (None, _) => None,
    };
    let default = if per_track.is_some() { ktracks::GUIDED_VOICES } else { cfg.max_voices };
    option("Enter", "default", &thousands(default));
    match (device, max) {
        (Some(a), Some(m)) => option(
            "-1",
            "device max",
            &match per_track {
                Some((n, _)) if n > 1 => {
                    let lanes = n.min(kestrel::gpu::LANES_MAX);
                    format!(
                        "{}  {} a track, {} on {}",
                        thousands(m),
                        thousands(ktracks::voices_each(m, n)),
                        if lanes == n {
                            format!("all {} at once", style::thousands(n as u64))
                        } else {
                            format!("{lanes} at a time")
                        },
                        a.name
                    )
                }
                _ => format!("{}  on {}", thousands(m), a.name),
            },
        ),
        _ => style::detail(vec![c("No GPU was found, so there is no device max.", DIM)]),
    }
    // One track has no others to make room for: past -1 it is held down.
    if per_track.is_some_and(|(n, _)| n > 1) {
        match max {
            Some(m) if m < ceiling => style::detail(vec![c(
                format!("more is taken, up to {}, with fewer tracks on the GPU at a time", thousands(ceiling)),
                DIM,
            )]),
            Some(_) => {}
            None => style::detail(vec![c(format!("at most {}", thousands(ceiling)), DIM)]),
        }
    }
    style::say(vec![c(
        "  or type a number: 700000, 700,000 and 700k all work",
        DIM,
    )]);
    let got = |total: u32| {
        let mut line = vec![b(thousands(total), AMBER), s(" voices")];
        if let Some((n, bank)) = per_track {
            let each = ktracks::voices_each(total, n);
            let mut note = format!("  about {} a track, before --min-velocity", thousands(each));
            if let Some(a) = device {
                let alone = ktracks::max_voices_alone(&cfg, bank, a.binding_bytes);
                let k = ktracks::tracks_at_once(&cfg, bank, a.binding_bytes, n, total);
                if each > alone {
                    note.push_str(&format!("; held to {} a track, one at a time", thousands(alone)));
                } else if k < n.min(kestrel::gpu::LANES_MAX) {
                    note.push_str(&format!(", {} at a time", style::thousands(k as u64)));
                }
            }
            line.push(c(note, DIM));
        }
        style::status(OK, line);
        Step::Got(total)
    };
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        // A per-track total is held to the ceiling, not to the device max.
        let limit = if per_track.is_some() { Some(ceiling) } else { max };
        match checks::parse_voices(&line, limit) {
            VoiceAnswer::Default => match max {
                Some(m) if per_track.is_none() && default > m => style::warn(format!(
                    "The default of {} is over this device's max of {}. Type -1, or a smaller number.",
                    thousands(default),
                    thousands(m)
                )),
                _ => return got(default),
            },
            VoiceAnswer::DeviceMax => match (device, max) {
                (Some(_), Some(m)) => return got(m),
                _ => style::error("There is no device max without a GPU. Type a number."),
            },
            VoiceAnswer::Count(n) => return got(n),
            VoiceAnswer::TooMany(n) if per_track.is_some() => {
                style::warn(format!(
                    "{} is over the most a per-track render takes; {} it is.",
                    style::thousands(n),
                    thousands(ceiling)
                ));
                return got(ceiling);
            }
            VoiceAnswer::TooMany(n) => style::warn(match (device, max) {
                (Some(a), Some(m)) => format!(
                    "{} is over the device max of {} on {}. Type -1 for the max, or a smaller number.",
                    style::thousands(n),
                    thousands(m),
                    a.name
                ),
                _ => format!(
                    "{} is more than Kestrel can address; the most is {}.",
                    style::thousands(n),
                    thousands(u32::MAX)
                ),
            }),
            VoiceAnswer::Invalid => style::error(
                "That isn't a voice count. Press Enter for the default, -1 for the device max, or type a number.",
            ),
        }
    }
}

fn step_format(io: &mut dyn Io, env: &Env, at: (u8, u8)) -> Step<&'static Format> {
    style::heading(&step_title(at, "Format"), "");
    for (i, f) in FORMATS.iter().enumerate() {
        let note = match format_available(env, f.ext) {
            Ok(()) => c(format_note(f.ext), DIM),
            Err(why) => c(why, style::shade(DIM, 0.7)),
        };
        style::say(vec![
            s("  "),
            c(format!("[{}]", i + 1), AMBER),
            s(format!(" {:<6}", f.label)),
            note,
        ]);
    }
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        match line.trim().parse::<usize>() {
            Ok(n) if (1..=FORMATS.len()).contains(&n) => {
                let f = &FORMATS[n - 1];
                match format_available(env, f.ext) {
                    Ok(()) => {
                        style::status(
                            OK,
                            vec![b(f.label, AMBER), c(format!("  {}", format_note(f.ext)), DIM)],
                        );
                        return Step::Got(f);
                    }
                    Err(why) => style::warn(format!(
                        "{} {why}. Choose WAV, or install ffmpeg ({}) and start Kestrel again.",
                        f.label,
                        install_hint()
                    )),
                }
            }
            _ => style::error("Type a number from 1 to 5."),
        }
    }
}

fn step_folder(io: &mut dyn Io, midi: &Path, at: (u8, u8), note: &str) -> Step<PathBuf> {
    style::heading(&step_title(at, "Destination"), note);
    loop {
        style::say(vec![c("Opening the folder picker\u{2026}", DIM)]);
        match io.pick_folder("Kestrel \u{00B7} choose where to save the render", midi.parent()) {
            Some(dir) => {
                style::status(OK, vec![s(dir.display().to_string())]);
                return Step::Got(dir);
            }
            None => match after_cancel(io) {
                Step::Got(()) => continue,
                Step::Menu => return Step::Menu,
                Step::Exit => return Step::Exit,
            },
        }
    }
}

struct Ready {
    job: Job,
    plan: Plan,
    /// `None` when the typed flags change how the soundfont loads.
    bank: Option<Arc<Bank>>,
    adapter: Option<String>,
    /// Every flag the render runs with but its MIDI, soundfonts and output,
    /// the steps' own among them, as the done screen lists them.
    flags: Vec<String>,
}

fn joined(flag: &str, path: &Path) -> OsString {
    let mut arg = OsString::from(flag);
    arg.push(path.as_os_str());
    arg
}

/// clap's message without its usage block, which describes `kestrel render`
/// positionals the guided renderer fills in itself.
fn clap_message(e: &clap::Error) -> String {
    let text = e.render().to_string();
    let kept: Vec<&str> = text
        .lines()
        .take_while(|l| !l.trim_start().starts_with("Usage:"))
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect();
    kept.join("\n").trim_start_matches("error: ").to_string()
}

/// `base` is what earlier steps chose beyond the render's own: a per-track
/// render's `--tracks` and the rest.
#[allow(clippy::too_many_arguments)]
fn step_flags(
    io: &mut dyn Io,
    env: &Env,
    midi: &MidiInfo,
    fonts: &Fonts,
    voices: u32,
    out: &Path,
    at: (u8, u8),
    base: &[OsString],
) -> Step<Ready> {
    style::heading(&step_title(at, "Additional flags"), "optional");
    style::say(vec![c(
        "Press Enter to start rendering, or type render flags as you would on a command line.",
        DIM,
    )]);
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        let typed = line.trim();
        let extra = match checks::split_args(typed)
            .and_then(|tokens| checks::extra_flags(tokens, env.adapters.len(), !base.is_empty()))
        {
            Ok(x) => x,
            Err(e) => {
                style::error(e);
                continue;
            }
        };

        let mut argv: Vec<OsString> = vec!["kestrel".into(), "render".into()];
        for p in &fonts.paths {
            argv.push(joined("--soundfont=", p));
        }
        argv.push(joined("--out=", out));
        argv.extend(base.iter().cloned());
        if !extra.max_voices {
            argv.push(format!("--max-voices={voices}").into());
        }
        if let Some(i) = extra.adapter {
            let a = &env.adapters[i];
            if a.is_software() {
                style::error(format!(
                    "[{}] is a software renderer, which Kestrel never renders on.",
                    i + 1
                ));
                continue;
            }
            argv.push(format!("--gpu-backend={}", backend_flag(a.backend)).into());
            argv.push(format!("--gpu-adapter={}", a.name).into());
        }
        argv.extend(extra.args.iter().map(OsString::from));
        let flags: Vec<String> = argv[2..]
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .filter(|a| !a.starts_with("--soundfont=") && !a.starts_with("--out="))
            .collect();
        argv.push("--".into());
        argv.push(midi.path.clone().into_os_string());

        let args = match parse_render(argv) {
            Ok(a) => a,
            Err(e) => {
                style::error(clap_message(&e));
                continue;
            }
        };
        let planned = args
            .to_job()
            .and_then(|job| session::plan(&job).map(|plan| (job, plan)));
        let (job, plan) = match planned {
            Ok(p) => p,
            Err(e) => {
                let _ = capture::problems();
                style::error(format!("{e:#}"));
                continue;
            }
        };
        for (_, w) in capture::problems() {
            style::warn(w);
        }

        let mut adapter = None;
        if matches!(plan.kind, BackendKind::Gpu) {
            let cfg = &plan.cfg;
            match spin("Checking the adapter", || kestrel::gpu::survey(cfg)) {
                Ok((list, Some(i))) => {
                    let a = &list[i];
                    // Not a per-track total: that is shared between the
                    // tracks, and the render holds each track's share to what
                    // it can hold rather than refusing.
                    if job.stems.is_none() && cfg.max_voices > a.max_voices {
                        style::warn(format!(
                            "{} tops out at {} voices and this render asks for {}. Add \
                             --max-voices with a smaller number, or pick another adapter.",
                            a.name,
                            thousands(a.max_voices),
                            thousands(cfg.max_voices)
                        ));
                        continue;
                    }
                    adapter = Some(format!("{} ({})", a.name, backend_name(a.backend)));
                }
                Ok((_, None)) => {
                    style::error(
                        "No usable GPU adapter matches. Pick one with --adapter, or render with --backend cpu.",
                    );
                    continue;
                }
                Err(e) => {
                    style::error(format!("{e:#}"));
                    continue;
                }
            }
        }

        let bank = if extra.reload {
            style::status(
                AMBER,
                vec![s(
                    "These flags change how a soundfont loads, so it loads again when the render starts.",
                )],
            );
            None
        } else {
            Some(fonts.bank.clone())
        };
        match &adapter {
            Some(name) => style::status(OK, vec![s("Rendering on "), b(name.clone(), AMBER)]),
            None => style::status(
                WARN,
                vec![s("Rendering on the CPU reference renderer, which is far slower")],
            ),
        }
        return Step::Got(Ready {
            job,
            plan,
            bank,
            adapter,
            flags,
        });
    }
}

/// The done screen's "Flags" rows: `flags` as they would be typed, one after
/// another, wrapped at whole flags to `inner` columns under a 10-column label.
fn flag_lines(flags: &[String], inner: usize) -> Vec<Line> {
    let label = |t: &str| c(format!("{t:<10}"), DIM);
    let room = inner.saturating_sub(10).max(20);
    let mut rows: Vec<String> = Vec::new();
    let mut row = String::new();
    for flag in flags.iter().map(|f| quote(f)) {
        if !row.is_empty() && row.chars().count() + 1 + flag.chars().count() > room {
            rows.push(std::mem::take(&mut row));
        }
        if !row.is_empty() {
            row.push(' ');
        }
        row.push_str(&flag);
    }
    if !row.is_empty() {
        rows.push(row);
    }
    if rows.is_empty() {
        return vec![vec![label("Flags"), c("none", DIM)]];
    }
    rows.into_iter()
        .enumerate()
        .map(|(i, r)| vec![label(if i == 0 { "Flags" } else { "" }), s(style::middle(&r, room))])
        .collect()
}

/// A per-track render's outcome, for `show_outcome`: how many tracks, and
/// whether they were summed into `out` or written as stems into it.
struct PerTrack {
    tracks: usize,
    merged: bool,
}

/// `flags` is what the render ran with beyond its MIDI, soundfonts and
/// output; see `Ready::flags`.
fn show_outcome(
    out: &Path,
    adapter: Option<&str>,
    flags: &[String],
    result: &std::result::Result<Summary, String>,
    problems: &[(log::Level, String)],
    per_track: Option<PerTrack>,
) {
    let inner = panel_width();
    let label = |t: &str| c(format!("{t:<10}"), DIM);
    let lines = match result {
        Ok(sum) if sum.cancelled => style::panel(
            vec![b("Cancelled", WARN)],
            &[
                vec![s(format!(
                    "Stopped at {} of audio, {} in.",
                    style::audio_clock(sum.audio_secs),
                    style::clock(sum.wall_secs)
                ))],
                vec![c(
                    match &per_track {
                        None => "The partial file was removed.",
                        Some(p) if p.merged => "A mix missing tracks isn't the file asked for, so none was written.",
                        Some(_) => "The stems finished before the stop are in the folder; the rest are cut short.",
                    },
                    DIM,
                )],
            ],
            inner,
            None,
        ),
        Ok(sum) => {
            let speed = sum.audio_secs / sum.wall_secs.max(1e-9);
            let stems = per_track.as_ref().filter(|p| !p.merged);
            let mut body = vec![
                match (&per_track, stems) {
                    (Some(p), Some(_)) => vec![
                        label("Wrote"),
                        b(format!("{} stem{}", style::thousands(p.tracks as u64), plural(p.tracks)), AMBER),
                    ],
                    _ => vec![label("Wrote"), b(file_name(out), AMBER)],
                },
                vec![
                    label("Folder"),
                    // The end of a path is the part that says which folder.
                    s(style::middle(
                        &match stems {
                            Some(_) => out.display().to_string(),
                            None => out.parent().map(|p| p.display().to_string()).unwrap_or_default(),
                        },
                        inner - 10,
                    )),
                ],
                Line::new(),
                vec![
                    label("Audio"),
                    s(style::audio_clock(sum.audio_secs)),
                    c(if stems.is_some() { " over all stems  in " } else { "  in " }, DIM),
                    s(format!("{:.1} s", sum.wall_secs)),
                    c("  = ", DIM),
                    b(format!("{speed:.2}\u{00D7} realtime"), AMBER),
                ],
                vec![label("Size"), s(style::bytes(sum.bytes))],
                vec![
                    label("Notes"),
                    s(style::thousands(sum.notes)),
                    c("   voices spawned ", DIM),
                    s(style::thousands(sum.voices_spawned)),
                ],
                vec![
                    label("Voices"),
                    s(style::thousands(sum.peak_voices)),
                    c(" at peak   stolen ", DIM),
                    s(style::thousands(sum.stolen)),
                    c("   dropped ", DIM),
                    s(style::thousands(sum.dropped)),
                ],
                {
                    let mut peak = vec![
                        label(if stems.is_some() { "Peak" } else { "Mix peak" }),
                        s(format!("{:.3}", sum.peak_level)),
                        c(if stems.is_some() { "  loudest stem, before any limiter" } else { "  before the limiter" }, DIM),
                    ];
                    if sum.clipped > 0 {
                        peak.push(c(
                            format!(
                                "  \u{00B7} {} samples hard-clipped",
                                style::thousands(sum.clipped)
                            ),
                            WARN,
                        ));
                    }
                    peak
                },
            ];
            if let Some(p) = per_track.as_ref().filter(|p| p.merged) {
                body.push(vec![
                    label("Tracks"),
                    s(style::thousands(p.tracks as u64)),
                    c(" rendered alone and summed, then limited once", DIM),
                ]);
            }
            if let Some(a) = adapter {
                body.push(vec![label("Device"), s(a.to_string())]);
            }
            body.extend(flag_lines(flags, inner));
            style::panel(vec![b("Done", OK)], &body, inner, None)
        }
        Err(message) => {
            let body: Vec<Line> = style::wrap(message, inner)
                .into_iter()
                .map(|l| vec![c(l, ERR)])
                .collect();
            style::panel(vec![b("Render failed", ERR)], &body, inner, None)
        }
    };
    for line in lines {
        style::say(line);
    }
    for (level, message) in problems {
        // A failure's own text is already in the panel.
        if matches!(result, Err(m) if m.contains(message.as_str())) {
            continue;
        }
        if *level == log::Level::Error {
            style::error(message);
        } else {
            style::warn(message.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct Script {
        lines: VecDeque<&'static str>,
        files: VecDeque<Vec<PathBuf>>,
        folders: VecDeque<PathBuf>,
    }

    impl Io for Script {
        fn line(&mut self) -> Option<String> {
            self.lines.pop_front().map(String::from)
        }
        fn pick_files(&mut self, _: Pick, _: &str) -> Option<Vec<PathBuf>> {
            self.files.pop_front()
        }
        fn pick_folder(&mut self, _: &str, _: Option<&Path>) -> Option<PathBuf> {
            self.folders.pop_front()
        }
    }

    /// No adapters, no ffmpeg, no update check: what a test can run anywhere.
    fn bare_env() -> Env {
        Env {
            adapters: Vec::new(),
            default_adapter: None,
            gpu_error: None,
            ffmpeg: Err("not looked for".into()),
            ring: Ring::default(),
            update: None,
        }
    }

    /// Every file under `dir`, by name, with its bytes.
    fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let p = e.unwrap().path();
                (file_name(&p), std::fs::read(&p).unwrap())
            })
            .collect();
        v.sort();
        v
    }

    /// The guided renderer's promise, for its per-track renders: every step,
    /// the progress screen and the outcome, driven by script, write what the
    /// same command writes. Once merged, over every track; once as stems, over
    /// a typed list. On the CPU backend, so it runs anywhere, and with a
    /// setup track, so step 3 asks its second question.
    #[test]
    fn a_guided_per_track_render_is_byte_identical_to_the_same_command() {
        use kestrel::midi::MidiWriter;
        let dir = std::env::temp_dir().join("kestrel_guided_per_track");
        let _ = std::fs::remove_dir_all(&dir);
        let [merged, stems, cli_merged, cli_stems] =
            ["merged", "stems", "cli_merged", "cli_stems"].map(|d| dir.join(d));
        for d in [&merged, &stems, &cli_merged, &cli_stems] {
            std::fs::create_dir_all(d).unwrap();
        }
        let font = dir.join("rich.sf2");
        kestrel::testkit::rich_sf2(&font, 48_000).unwrap();
        let midi = dir.join("song.mid");
        let notes = |ch: u8, start: u64| -> Vec<(u64, Vec<u8>)> {
            (0..24u64)
                .flat_map(|i| {
                    let key = 48 + ((i * 7 + ch as u64) % 24) as u8;
                    let t = start + i * 120;
                    [(t, vec![0x90 | ch, key, 90]), (t + 200, vec![0x80 | ch, key, 0])]
                })
                .collect()
        };
        let mut w = MidiWriter::new(480);
        w.raw_track(vec![(0, vec![0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])]);
        w.raw_track(vec![(0, vec![0xB0, 7, 80]), (0, vec![0xB1, 10, 20])]);
        w.raw_track(notes(0, 0));
        w.raw_track(notes(1, 240));
        w.raw_track(notes(0, 480));
        w.save(&midi).unwrap();

        let mut io = Script {
            lines: [
                // merged: MIDI, soundfont, all tracks, setup applied, default
                // voices, one file, WAV, flags; then another render
                "", "", "", "", "", "1", "1", "--backend cpu --seconds 2", "1",
                // stems: tracks 3 to 4, setup applied, default voices, a file
                // each, WAV, flags; then exit
                "", "", "3-4", "", "", "2", "1", "--backend cpu --seconds 2", "2",
            ]
            .into(),
            files: [vec![midi.clone()], vec![font.clone()], vec![midi.clone()], vec![font.clone()]].into(),
            folders: [merged.clone(), stems.clone()].into(),
        };
        let next = per_track_loop(&mut io, &bare_env()).unwrap();
        assert!(next == Next::Exit && io.lines.is_empty(), "the flow stopped early: {:?}", io.lines);

        let cli = |out: &Path, extra: &[&str]| {
            let mut argv: Vec<OsString> = vec!["kestrel".into(), "render".into(), midi.clone().into_os_string()];
            argv.extend(["-s".into(), font.clone().into_os_string(), "-o".into(), out.as_os_str().to_owned()]);
            argv.extend(["--backend", "cpu", "--seconds", "2"].map(OsString::from));
            argv.extend(extra.iter().map(OsString::from));
            crate::render::render_cli(parse_render(argv).unwrap()).unwrap();
        };
        // Enter at the voices step is the per-track default, which the
        // command line does not share.
        cli(&cli_merged.join("song.wav"), &["--tracks", "all", "--merge", "--max-voices", "60000000"]);
        cli(&cli_stems, &["--tracks", "3-4", "--max-voices", "60000000"]);

        let (a, b) = (files(&merged), files(&cli_merged));
        assert_eq!(a.len(), 1);
        assert!(a[0].1.len() > 1000, "the merged render is empty");
        assert!(a == b, "the guided merge differs from the command's");
        let (a, b) = (files(&stems.join("song")), files(&cli_stems.join("song")));
        let names: Vec<&str> = a.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["3.wav", "4.wav"]);
        assert!(a == b, "the guided stems differ from the command's");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Step 4 of a per-track render, as the user set it (2026-09-24). Enter
    /// is 60M. -1 is what the tracks hold with as many on the device at once
    /// as a batch takes, and 2B where that is more. A typed total is taken as
    /// typed, past -1 as well, and held to 2B past that.
    #[test]
    fn a_per_track_voice_total_defaults_to_60m_and_its_max_is_the_tracks_share_of_the_card() {
        let dir = std::env::temp_dir().join("kestrel_guided_voices");
        std::fs::create_dir_all(&dir).unwrap();
        let font = dir.join("rich.sf2");
        kestrel::testkit::rich_sf2(&font, 48_000).unwrap();
        let bank = kestrel::load_bank(&font, &Config::default()).unwrap();
        let binding = 2047u64 << 20;
        let mut env = bare_env();
        env.adapters.push(AdapterSummary {
            name: "Test GPU".into(),
            backend: wgpu::Backend::Vulkan,
            device_type: wgpu::DeviceType::DiscreteGpu,
            binding_bytes: binding,
            max_voices: kestrel::gpu::max_voices_for_config(binding, &Config::default()),
        });
        env.default_adapter = Some(0);
        let cfg = Config::default();
        let fast = |n| ktracks::max_voices_at_once(&cfg, &bank, binding, n);
        let at_once = |n, total| ktracks::tracks_at_once(&cfg, &bank, binding, n, total);
        let answer = |env: &Env, lines: &[&'static str], tracks: usize| {
            let mut io = Script { lines: lines.iter().copied().collect(), files: [].into(), folders: [].into() };
            let got = match step_voices(&mut io, env, (4, 8), Some((tracks, &bank))) {
                Step::Got(n) => n,
                _ => panic!("step 4 ended without an answer"),
            };
            assert!(io.lines.is_empty(), "answers left over: {:?}", io.lines);
            got
        };
        let (guided, ceiling) = (ktracks::GUIDED_VOICES, ktracks::MAX_TOTAL_VOICES);

        // Thousands of tracks: -1 is what 256 of them at a time hold.
        assert!(guided < fast(6291) && fast(6291) < ceiling);
        assert_eq!(answer(&env, &[""], 6291), guided);
        assert_eq!(answer(&env, &["-1"], 6291), fast(6291));
        // Up to it all 256 lanes share the device, and one more voice a
        // track takes one off.
        let each = ktracks::voices_each(fast(6291), 6291);
        assert_eq!(at_once(6291, each * 6291), kestrel::gpu::LANES_MAX);
        assert!(at_once(6291, (each + 1) * 6291) < kestrel::gpu::LANES_MAX);
        // Past -1 is taken as typed, and past 2B held to 2B.
        assert_eq!(answer(&env, &["1000000000"], 6291), 1_000_000_000);
        assert_eq!(answer(&env, &["2000000001"], 6291), ceiling);
        assert_eq!(answer(&env, &["99999999999"], 6291), ceiling);

        // So many tracks that -1 would pass 2B: it is 2B.
        assert!(fast(100_000) == ceiling);
        assert_eq!(answer(&env, &["-1"], 100_000), ceiling);

        // Thirty tracks: -1 keeps all thirty on the device, and 60M, past it,
        // is taken with fewer at a time.
        assert!(fast(30) < guided);
        assert_eq!(answer(&env, &["-1"], 30), fast(30));
        assert_eq!(answer(&env, &[""], 30), guided);
        assert!((1..30).contains(&at_once(30, guided)), "{} at once", at_once(30, guided));

        // One track: Enter is still 60M, which the render holds to what one
        // track holds alone.
        assert_eq!(answer(&env, &[""], 1), guided);
        assert!(ktracks::max_voices_alone(&cfg, &bank, binding) < guided);

        // No GPU: no -1, and still held to 2B.
        let bare = bare_env();
        assert_eq!(answer(&bare, &["-1", ""], 30), guided);
        assert_eq!(answer(&bare, &["3000000000"], 30), ceiling);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The done screen's "Flags" rows: every flag, as it would be typed,
    /// wrapped at whole flags under the label.
    #[test]
    fn the_done_screen_lists_the_flags_a_render_ran_with() {
        let flags: Vec<String> = [
            "--tracks=all",
            "--setup-tracks=apply",
            "--merge",
            "--max-voices=60000000",
            "--gpu-adapter=NVIDIA GeForce RTX 5060 Laptop GPU",
            "--format",
            "pcm16",
        ]
        .map(String::from)
        .into();
        let lines: Vec<String> = flag_lines(&flags, 76).iter().map(|l| style::plain(l)).collect();
        assert!(lines[0].starts_with("Flags     --tracks=all --setup-tracks=apply"), "{lines:?}");
        assert!(lines.len() > 1 && lines[1..].iter().all(|l| l.starts_with(&" ".repeat(10))), "{lines:?}");
        let rows: Vec<&str> = lines.iter().map(|l| l[10..].trim_end()).collect();
        // Whole flags on each row, a flag with spaces quoted, none left out.
        assert!(rows.iter().any(|r| r.contains("\"--gpu-adapter=NVIDIA GeForce RTX 5060 Laptop GPU\"")), "{rows:?}");
        assert_eq!(rows.join(" ").split(' ').filter(|w| w.contains("--")).count(), 6, "{rows:?}");
        assert_eq!(style::plain(&flag_lines(&[], 60)[0]).trim_end(), "Flags     none");
    }

    #[test]
    fn a_quoted_command_can_be_pasted_back() {
        assert_eq!(quote("render"), "render");
        assert_eq!(quote("D:\\midis\\Song Title.mid"), "\"D:\\midis\\Song Title.mid\"");
        assert_eq!(quote(""), "\"\"");
    }

    /// The whole promise of the guided renderer: it is a front end, so what it
    /// writes is what the same command writes. Driven end to end -- menu,
    /// pickers, every prompt, the progress screen -- and compared byte for
    /// byte against `--force-cli render` on the same inputs.
    #[test]
    #[ignore = "renders on the GPU; needs tests/assets from gen-assets"]
    fn a_guided_render_is_byte_identical_to_the_same_command() {
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("assets");
        let midi = assets.join("scatter1000.mid");
        let font = assets.join("rich.sf2");
        assert!(midi.exists() && font.exists(), "run gen-assets first");

        let dir = std::env::temp_dir().join("kestrel_guided_parity");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut io = Script {
            // menu: render; MIDI picked, continue; soundfont picked, continue;
            // voices: default; format: WAV; flags: none; then exit.
            lines: ["1", "", "", "Enter", "1", "", "2"].into(),
            files: [vec![midi.clone()], vec![font.clone()]].into(),
            folders: [dir.clone()].into(),
        };
        guided(&mut io).unwrap();
        assert!(io.lines.is_empty(), "the flow stopped early: {:?}", io.lines);

        let guided_out = dir.join("scatter1000.wav");
        let cli_out = dir.join("cli.wav");
        let argv: Vec<OsString> = vec![
            "kestrel".into(),
            "render".into(),
            midi.into_os_string(),
            "-s".into(),
            font.into_os_string(),
            "-o".into(),
            cli_out.clone().into_os_string(),
        ];
        crate::render::render_cli(parse_render(argv).unwrap()).unwrap();

        let a = std::fs::read(&guided_out).unwrap();
        let b = std::fs::read(&cli_out).unwrap();
        assert!(a.len() > 44, "the guided render is empty");
        assert!(a == b, "guided and command-line renders differ");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
