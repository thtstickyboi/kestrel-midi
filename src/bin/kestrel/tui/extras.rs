// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Extras: the diagnostic commands, and what `ktrl.ini` keeps, in a window of \[1\]

use super::style::{self, b, c, s, AMBER, DIM, ERR, OK, WARN};
use super::{prompt, Io, Native, Pick};
use crate::settings::{self, Ring};
use crate::update;
use clap::CommandFactory;
use std::path::Path;

/// The flag the new window is started with. Routed on only when it is the \[2\]
pub const FLAG: &str = "--extras-window";

/// Open Extras in a new terminal window running this executable.
pub fn open_window() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    spawn(&exe).map_err(|e| e.to_string())
}

#[cfg(windows)]
fn spawn(exe: &Path) -> std::io::Result<()> {
    use std::os::windows::process::CommandExt;
    // [3]
    let status = std::process::Command::new("cmd")
        .raw_arg(format!("/C start \"Kestrel Extras\" \"{}\" {FLAG}", exe.display()))
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!("start exited with {status}")))
    }
}

#[cfg(target_os = "macos")]
fn spawn(exe: &Path) -> std::io::Result<()> {
    let quoted = format!("'{}' {FLAG}", exe.display().to_string().replace('\'', r"'\''"));
    let script = format!(
        "tell application \"Terminal\" to do script \"{}\"",
        quoted.replace('\\', "\\\\").replace('"', "\\\"")
    );
    std::process::Command::new("osascript")
        .args(["-e", &script, "-e", "tell application \"Terminal\" to activate"])
        .spawn()
        .map(|_| ())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn spawn(exe: &Path) -> std::io::Result<()> {
    // [4]
    let mut tried: Vec<(String, Vec<&str>)> = Vec::new();
    if let Ok(t) = std::env::var("TERMINAL") {
        if !t.is_empty() {
            tried.push((t, vec!["-e"]));
        }
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
    for (term, lead) in tried {
        let started = std::process::Command::new(&term)
            .args(lead)
            .arg(exe)
            .arg(FLAG)
            .spawn();
        if started.is_ok() {
            return Ok(());
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "no terminal emulator found",
    ))
}

/// The Extras window's whole life: `kestrel --extras-window`.
pub fn run() -> anyhow::Result<()> {
    style::init();
    // [5]
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .try_init();
    if crossterm::tty::IsTty::is_tty(&std::io::stdout()) {
        let _ = crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::SetTitle("Kestrel \u{00B7} Extras")
        );
    }
    let mut io = Native::remembered();
    menu(&mut io);
    Ok(())
}

/// The Extras menu, until it is closed.
pub fn menu(io: &mut dyn Io) {
    loop {
        super::clear_screen();
        style::print_banner();
        style::heading("Extras", "diagnostics and settings");
        super::option("1", "GPU info", "every adapter wgpu can see, and its limits");
        super::option("2", "File info", "what the loader makes of a soundfont or MIDI");
        super::option("3", "Null test", "compare two WAV renders");
        super::option("4", "Flag help", "every flag, what it does, and how to use it");
        super::option(
            "5",
            "Update ring",
            &format!("which releases Kestrel tells you about; now {}", ring_label(settings::load().0.ring)),
        );
        super::option("6", "Remembered folders", "where the MIDI, soundfont and destination pickers open");
        super::option("7", "Close", "");
        let choice = loop {
            prompt();
            match io.line() {
                None => return,
                Some(l) => match l.trim() {
                    "1" | "2" | "3" | "4" | "5" | "6" => break l.trim().to_string(),
                    "7" | "0" | "q" | "Q" => return,
                    _ => style::error("Type a number from 1 to 7."),
                },
            }
        };
        style::blank();
        match choice.as_str() {
            "1" => gpu_info(),
            "2" => file_info(io),
            "3" => null_test(io),
            "4" => flag_help(),
            "5" => {
                if !update_ring(io) {
                    return;
                }
            }
            _ => folders(),
        }
        style::blank();
        style::say(vec![c("Press Enter to go back to Extras.", DIM)]);
        if io.line().is_none() {
            return;
        }
    }
}

fn ring_label(ring: Ring) -> &'static str {
    match ring {
        Ring::Fast => "Fast Ring",
        Ring::Slow => "Slow Ring",
    }
}

fn ring_note(ring: Ring) -> &'static str {
    match ring {
        Ring::Fast => "every release, fixes such as 1.1.1 included",
        Ring::Slow => "feature releases only, such as 1.2.0 or 2.0.0",
    }
}

/// Choose the update ring, by its number. `false` once input has ended.
fn update_ring(io: &mut dyn Io) -> bool {
    style::heading("Update ring", "which new releases Kestrel tells you about when it starts");
    style::say(vec![c("Kestrel only tells you about a release; it never downloads one.", DIM)]);
    let current = settings::load().0.ring;
    for ring in Ring::ALL {
        let mark = if ring == current { c(" \u{25CF} ", OK) } else { s("   ") };
        style::say(vec![
            mark,
            c(format!("[{}]", ring.number()), AMBER),
            s(format!(" {:<24}", ring_label(ring))),
            c(ring_note(ring), DIM),
        ]);
    }
    if update::opted_out() {
        style::detail(vec![c(
            format!("{} is set on this machine, so Kestrel doesn't check at all.", update::OPT_OUT),
            WARN,
        )]);
    }
    let ring = loop {
        prompt();
        let Some(line) = io.line() else { return false };
        match Ring::from_number(line.trim()) {
            Some(ring) => break ring,
            None => style::error("Type 1 or 2."),
        }
    };
    style::blank();
    match settings::update(|s| s.ring = ring) {
        Ok(()) => style::status(
            OK,
            vec![
                s(format!("Saved: {}.", ring_label(ring))),
                c("  Used from the next time Kestrel starts.", DIM),
            ],
        ),
        Err(e) => style::error(format!("Couldn't save {}: {e:#}", settings::FILE)),
    }
    true
}

fn folders() {
    style::heading("Remembered folders", "where the render's file pickers open");
    let saved = settings::load().0;
    for (key, what) in settings::FOLDERS {
        let dir = match saved.folder(key) {
            Some(d) => s(d.display().to_string()),
            None => c("not yet", DIM),
        };
        style::say(vec![s(format!("  {what:<26}")), dir]);
    }
    style::blank();
    let place = settings::path().map_or_else(|_| settings::FILE.into(), |p| p.display().to_string());
    style::say(vec![c("Filled in as you pick files, and kept in ", DIM), s(place)]);
    style::say(vec![c(format!("Delete {} to forget them all.", settings::FILE), DIM)]);
}

fn gpu_info() {
    style::heading("GPU info", "");
    if let Err(e) = kestrel::gpu::print_adapters() {
        style::error(format!("{e:#}"));
    }
}

fn file_info(io: &mut dyn Io) {
    style::heading("File info", "a soundfont or a MIDI file");
    style::say(vec![c("Opening the file picker\u{2026}", DIM)]);
    let Some(path) = io
        .pick_files(Pick::Inspect, "Kestrel \u{00B7} choose a soundfont or MIDI file")
        .and_then(|p| p.into_iter().next())
    else {
        style::say(vec![c("Nothing chosen.", DIM)]);
        return;
    };
    style::say(vec![c("Reading ", DIM), b(super::file_name(&path), AMBER), c("\u{2026}", DIM)]);
    style::blank();
    if let Err(e) = crate::info(path, 4096, 48000, 2, 1.0, None) {
        style::error(format!("{e:#}"));
    }
}

fn null_test(io: &mut dyn Io) {
    style::heading("Null test", "how far apart two renders are");
    style::say(vec![c("Choose the reference first, then the render to compare with it.", DIM)]);
    let Some(a) = io
        .pick_files(Pick::Wav, "Kestrel \u{00B7} choose the reference WAV")
        .and_then(|p| p.into_iter().next())
    else {
        style::say(vec![c("Nothing chosen.", DIM)]);
        return;
    };
    let Some(b_path) = io
        .pick_files(Pick::Wav, "Kestrel \u{00B7} choose the WAV to compare")
        .and_then(|p| p.into_iter().next())
    else {
        style::say(vec![c("Nothing chosen.", DIM)]);
        return;
    };
    style::say(vec![c("Reference  ", DIM), s(a.display().to_string())]);
    style::say(vec![c("Compared   ", DIM), s(b_path.display().to_string())]);
    style::blank();
    if let Err(e) = crate::null(a, b_path, -80.0) {
        style::say(vec![b("FAIL ", ERR), c(format!("{e:#}"), ERR)]);
    }
}

/// Every public subcommand's flags, straight from the clap definitions that \[6\]
pub fn flag_help() {
    let mut cmd = crate::Cli::command();
    cmd.build();
    style::heading("Flag help", "every flag, what it does, and how to use it");
    style::say(vec![c(
        "In the guided renderer, type render flags at step 6, the last one before rendering.",
        DIM,
    )]);
    style::say(vec![c("From a terminal, put --force-cli in front of the command:", DIM)]);
    style::say(vec![c(
        "  kestrel --force-cli render song.mid -s font.sf2 -o song.opus --max-voices 2000000",
        AMBER,
    )]);
    style::say(vec![
        c("Guided renderer only: ", DIM),
        b("--adapter <N>", AMBER),
        c(" renders on adapter N from the environment check's list.", DIM),
    ]);
    for name in ["render", "info", "null", "gpu-info", "ffmpeg-info", "get-ffmpeg"] {
        if let Some(sub) = cmd.find_subcommand_mut(name) {
            style::blank();
            style::say(vec![b(format!("kestrel --force-cli {name}"), AMBER)]);
            style::say(vec![c("\u{2500}".repeat(60), style::FRAME)]);
            println!("{}", sub.render_long_help().ansi());
        }
    }
}

