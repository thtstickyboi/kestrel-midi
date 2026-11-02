// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Extras: the diagnostic commands, and what `ktrl.ini` keeps, in a window of \[1\]

use super::style::{self, b, c, s, AMBER, DIM, OK, WARN};
use super::window::Handover;
#[cfg(feature = "dev")]
use super::style::ERR;
use super::{prompt, Io, Native, Pick};
use crate::settings::{self, Ring};
use crate::update;
use clap::CommandFactory;

/// The flag the new window is started with, and the one that follows it with \[2\]
pub const FLAG: &str = "--extras-window";
pub const PARENT: &str = "--parent";

/// Open Extras in a new terminal window running this executable, told which \[3\]
pub fn open_window() -> Result<(), String> {
    super::window::open("Kestrel Extras", &window_args(std::process::id()))
}

/// The command line of the Extras window opened from process `pid`.
pub fn window_args(pid: u32) -> Vec<String> {
    vec![FLAG.to_string(), PARENT.to_string(), pid.to_string()]
}

/// The Extras window's whole life: `kestrel --extras-window`, and the process \[4\]
pub fn run(parent: Option<u32>) -> anyhow::Result<()> {
    style::init();
    // [5]
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(crate::LOG_FILTER))
        .format_timestamp(None)
        .try_init();
    if crossterm::tty::IsTty::is_tty(&std::io::stdout()) {
        let _ = crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::SetTitle("Kestrel \u{00B7} Extras")
        );
    }
    let mut io = Native::remembered();
    ORIGIN.set(parent).ok();
    menu(&mut io);
    Ok(())
}

/// The process this window came from, set once by `run`; `None` for Extras run \[6\]
static ORIGIN: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();

/// The window Extras came from, for what is opened from here to see out.
pub fn origin() -> Option<u32> {
    ORIGIN.get().copied().flatten()
}

/// One entry of the Extras menu.
#[derive(Clone, Copy)]
enum Item {
    GpuInfo,
    MachineReport,
    FileInfo,
    /// A dev-build entry, like the `null` command it runs.
    #[cfg(feature = "dev")]
    NullTest,
    FlagHelp,
    UpdateRing,
    Folders,
    /// Merlin, in a window of its own.
    Merlin,
}

/// What the Extras menu offers, in the order shown and numbered from 1. Built \[7\]
fn items() -> Vec<Item> {
    let mut v = vec![Item::GpuInfo, Item::MachineReport, Item::FileInfo];
    #[cfg(feature = "dev")]
    v.push(Item::NullTest);
    v.extend([Item::FlagHelp, Item::UpdateRing, Item::Folders, Item::Merlin]);
    v
}

/// What the person is told before Merlin opens. The first sentence is the \[8\]
pub const MERLIN_DISCLAIMER: &str = "Merlin is a very intensive renderer. It may need capable hardware to run: a large \
    video card, a lot of memory, and tens to hundreds of gigabytes of free disk for a big file. It does not currently \
    guarantee that a render will finish, and any faults it may cause are strictly your responsibility.";

/// A person's answer to `[Y/n]`: Enter and yes are yes. `None` is not an answer.
fn disclaimer_answer(line: &str) -> Option<bool> {
    match line.trim().to_ascii_lowercase().as_str() {
        "" | "y" | "yes" => Some(true),
        "n" | "no" => Some(false),
        _ => None,
    }
}

/// Where Extras stands once Merlin has been asked for.
#[derive(Debug, PartialEq, Eq)]
enum Started {
    /// Merlin is open in a window of its own, and this one is to go.
    Opened,
    /// Not opened, or opened beside a window that stays: back to the menu.
    Stayed,
}

/// The disclaimer, and Merlin's window if it is agreed to. `open` is what opens \[9\]
fn merlin(io: &mut dyn Io, open: impl FnOnce() -> Result<Handover, String>) -> Started {
    style::heading("Merlin (Convolution)", "a very experimental and intensive backend");
    let (first, rest) = MERLIN_DISCLAIMER.split_once(". ").unwrap_or((MERLIN_DISCLAIMER, ""));
    let mut text = vec![format!("{first}."), rest.to_string()];
    text.retain(|t| t.len() > 1);
    for (i, para) in text.iter().enumerate() {
        for line in style::wrap(para, 76) {
            style::say(vec![if i == 0 { b(line, WARN) } else { c(line, WARN) }]);
        }
    }
    style::blank();
    style::say(vec![s("Continue? [Y/n]")]);
    let agreed = loop {
        prompt();
        let Some(line) = io.line() else { return Started::Stayed };
        match disclaimer_answer(&line) {
            Some(answer) => break answer,
            None => style::error("Type Y or n."),
        }
    };
    if !agreed {
        style::say(vec![c("Merlin was not opened.", DIM)]);
        return Started::Stayed;
    }
    style::say(vec![c("Opening Merlin\u{2026}", DIM)]);
    match open() {
        Ok(Handover::Done) => Started::Opened,
        Ok(Handover::ParentBusy) => {
            style::status(OK, vec![s("Merlin opened in a new window.")]);
            style::detail(vec![c("The Kestrel window Extras came from is rendering, so it was left open.", DIM)]);
            Started::Stayed
        }
        Ok(Handover::ParentSilent) => {
            style::status(OK, vec![s("Merlin opened in a new window.")]);
            style::detail(vec![c("The Kestrel window Extras came from did not answer, so it was left open. Close it yourself.", DIM)]);
            Started::Stayed
        }
        Err(e) => {
            style::error(format!("Couldn't open a new window ({e}). Merlin needs a window of its own."));
            Started::Stayed
        }
    }
}

/// Close this window, now: Merlin has its own.
fn leave() -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    std::process::exit(0)
}

/// The Extras menu, until it is closed.
pub fn menu(io: &mut dyn Io) {
    let items = items();
    let close = items.len() + 1;
    loop {
        super::clear_screen();
        style::print_banner();
        style::heading("Extras", "diagnostics and settings");
        for (i, item) in items.iter().enumerate() {
            let n = (i + 1).to_string();
            match item {
                Item::GpuInfo => {
                    super::option(&n, "GPU info", "every adapter wgpu can see, and its limits")
                }
                Item::MachineReport => super::option(
                    &n,
                    "Machine report",
                    "a file to send when Kestrel goes wrong on this machine",
                ),
                Item::FileInfo => {
                    super::option(&n, "File info", "what the loader makes of a soundfont or MIDI")
                }
                #[cfg(feature = "dev")]
                Item::NullTest => super::option(&n, "Null test", "compare two WAV renders"),
                Item::FlagHelp => {
                    super::option(&n, "Flag help", "every flag, what it does, and how to use it")
                }
                Item::UpdateRing => super::option(
                    &n,
                    "Update ring",
                    &format!(
                        "which releases Kestrel tells you about; now {}",
                        ring_label(settings::load().0.ring)
                    ),
                ),
                Item::Folders => super::option(
                    &n,
                    "Remembered folders",
                    "where the MIDI, soundfont and destination pickers open",
                ),
                Item::Merlin => {
                    super::option(&n, "Merlin (Convolution)", "a very experimental and intensive backend for");
                    // The note's second line, under the first's column.
                    style::say(vec![s(" ".repeat(32)), c("rendering MIDIs in a lossless manner", DIM)]);
                }
            }
        }
        super::option(&close.to_string(), "Close", "");
        let item = loop {
            prompt();
            let Some(l) = io.line() else {
                return;
            };
            let l = l.trim();
            if matches!(l, "0" | "q" | "Q") {
                return;
            }
            match l.parse::<usize>() {
                Ok(n) if n == close => return,
                Ok(n) if (1..close).contains(&n) => break items[n - 1],
                _ => style::error(format!("Type a number from 1 to {close}.")),
            }
        };
        style::blank();
        match item {
            Item::GpuInfo => gpu_info(),
            Item::MachineReport => machine_report(io),
            Item::FileInfo => file_info(io),
            #[cfg(feature = "dev")]
            Item::NullTest => null_test(io),
            Item::FlagHelp => flag_help(),
            Item::UpdateRing => {
                if !update_ring(io) {
                    return;
                }
            }
            Item::Folders => folders(),
            Item::Merlin => {
                if merlin(io, || super::merlin::open(origin())) == Started::Opened {
                    leave();
                }
            }
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
    if let Err(e) = kestrel::gpu::print_adapters(None) {
        style::error(format!("{e:#}"));
    }
}

/// FalconEye's machine report. What it holds is shown before anything is \[10\]
fn machine_report(io: &mut dyn Io) {
    use kestrel::falconeye::report;
    style::heading("Machine report", "a file to send when Kestrel goes wrong on this machine");
    for line in report::HEADER.lines() {
        style::say(vec![c(line, DIM)]);
    }
    style::blank();
    style::say(vec![s("Run the GPU self-test too? It loads each GPU for up to a minute. [Y/n]")]);
    prompt();
    let Some(answer) = io.line() else { return };
    let self_test = !matches!(answer.trim(), "n" | "N" | "no" | "No");
    // The administrator step is its own question, and no unless asked.
    let mut elevate_with = None;
    if cfg!(windows) {
        style::say(vec![s(
            "Also collect Windows' own records of GPU crashes? Windows will ask for administrator \
             permission; Kestrel only reads them, and copies no dump. [y/N]",
        )]);
        prompt();
        let Some(answer) = io.line() else { return };
        if matches!(answer.trim(), "y" | "Y" | "yes" | "Yes") {
            elevate_with = std::env::current_exe().ok();
        }
    }
    style::blank();
    let built = report::build(
        report::Options {
            self_test,
            extra: vec![settings::report_section()],
            out_dir: None,
            elevate_with,
            isolate_with: std::env::current_exe().ok(),
            gpu_backend: None,
        },
        &mut |line| style::say(vec![c(line, DIM)]),
    );
    style::blank();
    match built {
        Ok(path) => {
            style::say(vec![b("Saved: ", OK), s(path.display().to_string())]);
            style::say(vec![c("Send this file to Kestrel's developer, privately.", DIM)]);
            #[cfg(windows)]
            let _ = std::process::Command::new("explorer").arg("/select,").arg(&path).spawn();
        }
        Err(e) => style::error(format!("The report could not be written: {e:#}")),
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

#[cfg(feature = "dev")]
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

/// Every public subcommand's flags, straight from the clap definitions that \[11\]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::Pick;
    use std::collections::VecDeque;
    use std::path::{Path, PathBuf};

    struct Lines(VecDeque<&'static str>);

    impl Io for Lines {
        fn line(&mut self) -> Option<String> {
            self.0.pop_front().map(String::from)
        }
        fn pick_files(&mut self, _: Pick, _: &str) -> Option<Vec<PathBuf>> {
            None
        }
        fn pick_folder(&mut self, _: &str, _: Option<&Path>) -> Option<PathBuf> {
            None
        }
    }

    fn asked(lines: &[&'static str], result: Result<Handover, String>) -> (Started, usize) {
        let opened = std::cell::Cell::new(0);
        let started = merlin(&mut Lines(lines.iter().copied().collect()), || {
            opened.set(opened.get() + 1);
            result
        });
        (started, opened.get())
    }

    /// Merlin is the last choice, so Kestrel's own keep the numbers they had, \[12\]
    #[test]
    fn merlin_is_the_last_of_the_choices_and_the_others_keep_their_numbers() {
        let items = items();
        assert!(matches!(items.last(), Some(Item::Merlin)));
        assert!(matches!(items[0], Item::GpuInfo) && matches!(items[1], Item::MachineReport) && matches!(items[2], Item::FileInfo));
        assert_eq!(items.iter().filter(|i| matches!(i, Item::Merlin)).count(), 1);
        #[cfg(not(feature = "dev"))]
        assert!(matches!(items[items.len() - 2], Item::Folders), "Remembered folders was [7]");
    }

    /// Enter, Y and yes continue; n and no do not; anything else is asked again.
    #[test]
    fn the_disclaimer_is_answered_by_enter_y_or_n() {
        for yes in ["", "  ", "y", "Y", "yes", "YES", " y "] {
            assert_eq!(disclaimer_answer(yes), Some(true), "{yes:?}");
        }
        for no in ["n", "N", "no", "No", " n"] {
            assert_eq!(disclaimer_answer(no), Some(false), "{no:?}");
        }
        for neither in ["maybe", "1", "yy", "q"] {
            assert_eq!(disclaimer_answer(neither), None, "{neither:?}");
        }
    }

    /// It says what the user asked it to, in their words.
    #[test]
    fn the_disclaimer_says_what_was_asked() {
        for part in ["very intensive renderer", "capable hardware", "does not currently guarantee that a render will finish", "strictly your responsibility"] {
            assert!(MERLIN_DISCLAIMER.contains(part), "{part}");
        }
    }

    #[test]
    fn agreeing_opens_merlin_and_sends_this_window_away() {
        assert_eq!(asked(&["y"], Ok(Handover::Done)), (Started::Opened, 1));
        assert_eq!(asked(&[""], Ok(Handover::Done)), (Started::Opened, 1), "Enter is yes");
        // Asked again until it is an answer.
        assert_eq!(asked(&["what", "y"], Ok(Handover::Done)), (Started::Opened, 1));
    }

    #[test]
    fn declining_or_no_answer_opens_nothing() {
        assert_eq!(asked(&["n"], Ok(Handover::Done)), (Started::Stayed, 0));
        assert_eq!(asked(&["no"], Ok(Handover::Done)), (Started::Stayed, 0));
        assert_eq!(asked(&[], Ok(Handover::Done)), (Started::Stayed, 0), "input ended");
        assert_eq!(asked(&["what"], Ok(Handover::Done)), (Started::Stayed, 0), "input ended at a question");
    }

    /// A window that could not be opened, or a main window that stayed, leave \[13\]
    #[test]
    fn extras_stays_when_merlin_could_not_be_opened_or_the_main_window_stays() {
        assert_eq!(asked(&["y"], Err("no terminal emulator found".into())), (Started::Stayed, 1));
        assert_eq!(asked(&["y"], Ok(Handover::ParentBusy)), (Started::Stayed, 1));
        assert_eq!(asked(&["y"], Ok(Handover::ParentSilent)), (Started::Stayed, 1));
    }
}
