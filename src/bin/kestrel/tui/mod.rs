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
mod style;

use kestrel::session::{self, Job, Plan, Summary};
use crate::{Cli, Cmd, RenderArgs};
use anyhow::Result;
use checks::{FontProfile, MidiInfo, Verdict, VoiceAnswer};
use clap::Parser;
use crossterm::tty::IsTty;
use crossterm::{cursor, execute, queue, terminal};
use kestrel::bank::Bank;
use kestrel::config::{BackendKind, Config};
use kestrel::gpu::AdapterSummary;
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
/// from for the rest of the session, so a second render starts where the first
/// one's files were.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pick {
    Midi,
    Soundfont,
    Inspect,
    Wav,
    Folder,
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
#[derive(Default)]
pub struct Native {
    dirs: HashMap<Pick, PathBuf>,
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
        if let Some(dir) = self.dirs.get(&kind) {
            dialog = dialog.set_directory(dir);
        }
        let picked = match kind {
            Pick::Midi | Pick::Soundfont => dialog.pick_files()?,
            _ => vec![dialog.pick_file()?],
        };
        if let Some(dir) = picked.first().and_then(|p| p.parent()) {
            self.dirs.insert(kind, dir.to_path_buf());
        }
        Some(picked)
    }

    fn pick_folder(&mut self, title: &str, start: Option<&Path>) -> Option<PathBuf> {
        let mut dialog = rfd::FileDialog::new().set_title(title);
        if let Some(dir) = self.dirs.get(&Pick::Folder).map(PathBuf::as_path).or(start) {
            dialog = dialog.set_directory(dir);
        }
        let picked = dialog.pick_folder()?;
        self.dirs.insert(Pick::Folder, picked.clone());
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
    let tty = std::io::stdout().is_tty();
    std::thread::scope(|scope| {
        let handle = scope.spawn(work);
        let t0 = Instant::now();
        let mut tick = 0usize;
        while tty && !handle.is_finished() {
            let secs = t0.elapsed().as_secs_f64();
            let mut line = vec![c(SPIN[tick % SPIN.len()], AMBER), s(format!(" {label}\u{2026}"))];
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
    let cfg = Config::default();
    let survey = spin("Looking for GPU adapters", || kestrel::gpu::survey(&cfg));
    let _ = capture::problems();
    let (adapters, default_adapter, gpu_error) = match survey {
        Ok((list, pick)) => (list, pick, None),
        Err(e) => (Vec::new(), None, Some(format!("{e:#}"))),
    };
    let ffmpeg = spin("Looking for ffmpeg", probe_ffmpeg);
    let env = Env {
        adapters,
        default_adapter,
        gpu_error,
        ffmpeg,
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
    env
}

/// The environment in two lines, for the top of every screen after the first.
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
    let mut io = Native::default();
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
                style::blank();
                style::status(
                    AMBER,
                    vec![b("Per-Track Render", AMBER), s(" is coming in a later release.")],
                );
                continue;
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
    option("2", "Per-Track Render", "coming soon");
    option("3", "Extras", "GPU info, file info, null test, flag help");
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

        let midi = step!(step_midi(io));
        let fonts = step!(step_soundfonts(io));
        let voices = step!(step_voices(io, env));
        let format = step!(step_format(io, env));
        let folder = step!(step_folder(io, &midi.path));
        let out = checks::output_path(&folder, &midi.path, format.ext, &checks::timestamp());
        let mut writes = vec![c("Writes ", DIM), b(file_name(&out), AMBER)];
        if out.file_stem() != midi.path.file_stem() {
            writes.push(c(
                "  (that name was taken, so this one carries the time)",
                DIM,
            ));
        }
        style::detail(writes);
        let ready = step!(step_flags(io, env, &midi, &fonts, voices, &out));

        let labels = progress::Labels {
            midi: file_name(&midi.path),
            output: file_name(&out),
            format: format!("{} \u{00B7} {}", format.label, format_note(format.ext)),
            fonts: fonts.names.clone(),
            max_voices: ready.plan.cfg.max_voices,
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
        show_outcome(&out, ready.adapter.as_deref(), &result, &problems);
        style::heading("What next?", "");
        option("1", "Start another render", "");
        option("2", "Exit", "");
        loop {
            prompt();
            let Some(line) = io.line() else {
                return Ok(Next::Exit);
            };
            match line.trim() {
                "1" => break,
                "2" | "0" | "q" | "Q" => return Ok(Next::Exit),
                _ => style::error("Type 1 or 2."),
            }
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

fn step_midi(io: &mut dyn Io) -> Step<MidiInfo> {
    style::heading("Step 1 of 6 \u{00B7} MIDI", "the file to render");
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

fn step_soundfonts(io: &mut dyn Io) -> Step<Fonts> {
    style::heading(
        "Step 2 of 6 \u{00B7} Soundfonts",
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

fn step_voices(io: &mut dyn Io, env: &Env) -> Step<u32> {
    style::heading("Step 3 of 6 \u{00B7} Voices", "how many may sound at once");
    let default = Config::default().max_voices;
    let device = env.default_adapter();
    option("Enter", "default", &thousands(default));
    match device {
        Some(a) => option(
            "-1",
            "device max",
            &format!("{}  on {}", thousands(a.max_voices), a.name),
        ),
        None => style::detail(vec![c("No GPU was found, so there is no device max.", DIM)]),
    }
    style::say(vec![c(
        "  or type a number: 700000, 700,000 and 700k all work",
        DIM,
    )]);
    let got = |n: u32| {
        style::status(OK, vec![b(thousands(n), AMBER), s(" voices")]);
        Step::Got(n)
    };
    loop {
        prompt();
        let Some(line) = io.line() else {
            return Step::Exit;
        };
        match checks::parse_voices(&line, device.map(|a| a.max_voices)) {
            VoiceAnswer::Default => match device {
                Some(a) if default > a.max_voices => style::warn(format!(
                    "The default of {} is over this device's max of {}. Type -1, or a smaller number.",
                    thousands(default),
                    thousands(a.max_voices)
                )),
                _ => return got(default),
            },
            VoiceAnswer::DeviceMax => match device {
                Some(a) => return got(a.max_voices),
                None => style::error("There is no device max without a GPU. Type a number."),
            },
            VoiceAnswer::Count(n) => return got(n),
            VoiceAnswer::TooMany(n) => style::warn(match device {
                Some(a) => format!(
                    "{} is over the device max of {} on {}. Type -1 for the max, or a smaller number.",
                    style::thousands(n),
                    thousands(a.max_voices),
                    a.name
                ),
                None => format!(
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

fn step_format(io: &mut dyn Io, env: &Env) -> Step<&'static Format> {
    style::heading("Step 4 of 6 \u{00B7} Format", "");
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

fn step_folder(io: &mut dyn Io, midi: &Path) -> Step<PathBuf> {
    style::heading("Step 5 of 6 \u{00B7} Destination", "the folder to write into");
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

fn step_flags(
    io: &mut dyn Io,
    env: &Env,
    midi: &MidiInfo,
    fonts: &Fonts,
    voices: u32,
    out: &Path,
) -> Step<Ready> {
    style::heading("Step 6 of 6 \u{00B7} Additional flags", "optional");
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
            .and_then(|tokens| checks::extra_flags(tokens, env.adapters.len()))
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
                    if cfg.max_voices > a.max_voices {
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
        });
    }
}

fn show_outcome(
    out: &Path,
    adapter: Option<&str>,
    result: &std::result::Result<Summary, String>,
    problems: &[(log::Level, String)],
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
                vec![c("The partial file was removed.", DIM)],
            ],
            inner,
            None,
        ),
        Ok(sum) => {
            let speed = sum.audio_secs / sum.wall_secs.max(1e-9);
            let mut body = vec![
                vec![label("Wrote"), b(file_name(out), AMBER)],
                vec![
                    label("Folder"),
                    s(out.parent().map(|p| p.display().to_string()).unwrap_or_default()),
                ],
                Line::new(),
                vec![
                    label("Audio"),
                    s(style::audio_clock(sum.audio_secs)),
                    c("  in ", DIM),
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
                        label("Mix peak"),
                        s(format!("{:.3}", sum.peak_level)),
                        c("  before the limiter", DIM),
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
            if let Some(a) = adapter {
                body.push(vec![label("Device"), s(a.to_string())]);
            }
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
