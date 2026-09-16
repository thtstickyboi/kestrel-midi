// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Extras: the diagnostic commands, in a window of their own. \[1\]

use super::style::{self, b, c, s, AMBER, DIM, ERR};
use super::{prompt, Io, Native, Pick};
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
    let mut io = Native::default();
    menu(&mut io);
    Ok(())
}

/// The Extras menu, until it is closed.
pub fn menu(io: &mut dyn Io) {
    loop {
        super::clear_screen();
        style::print_banner();
        style::heading("Extras", "diagnostics");
        super::option("1", "GPU info", "every adapter wgpu can see, and its limits");
        super::option("2", "File info", "what the loader makes of a soundfont or MIDI");
        super::option("3", "Null test", "compare two WAV renders");
        super::option("4", "Flag help", "every flag, what it does, and how to use it");
        super::option("5", "Close", "");
        let choice = loop {
            prompt();
            match io.line() {
                None => return,
                Some(l) => match l.trim() {
                    "1" | "2" | "3" | "4" => break l.trim().to_string(),
                    "5" | "0" | "q" | "Q" => return,
                    _ => style::error("Type a number from 1 to 5."),
                },
            }
        };
        style::blank();
        match choice.as_str() {
            "1" => gpu_info(),
            "2" => file_info(io),
            "3" => null_test(io),
            _ => flag_help(),
        }
        style::blank();
        style::say(vec![c("Press Enter to go back to Extras.", DIM)]);
        if io.line().is_none() {
            return;
        }
    }
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

