// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Merlin's window: the convolution renderer, on a screen of its own. \[1\]

use super::{capture, style, window, Native};
use anyhow::Result;

pub mod end;
pub mod flow;
pub mod progress;

/// The flag the window is started with. Routed on only when it is the whole \[2\]
pub const FLAG: &str = "--merlin-window";

/// The window's title, which the person sees in the tab.
pub const TITLE: &str = "Merlin";

/// Open Merlin in a window of its own and see the window it was chosen in out: \[3\]
pub fn open(parent: Option<u32>) -> Result<window::Handover, String> {
    window::hand_over(TITLE, &window_args(), parent)
}

/// The command line of Merlin's window.
pub fn window_args() -> Vec<String> {
    vec![FLAG.to_string()]
}

/// Back to Kestrel: its guided renderer in a window of its own. The caller \[4\]
#[allow(dead_code)] // Phase 5 of MERLIN-FRONT-END.md: Merlin's home offers it, and this goes.
pub fn back_to_kestrel() -> Result<(), String> {
    window::open("Kestrel", &[])
}

/// Merlin's window's whole life: `kestrel --merlin-window`.
pub fn run() -> Result<()> {
    style::init();
    style::set_theme(style::Theme::MERLIN);
    capture::install();
    capture::hold_panics();
    // After `hold_panics`, which replaces the hook: the log's chains to it.
    crate::start_falconeye("merlin");
    if crossterm::tty::IsTty::is_tty(&std::io::stdout()) {
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::SetTitle(TITLE));
    }
    let outcome = std::panic::catch_unwind(flow::run);
    let failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(format!("{e:#}")),
        Err(_) => Some(match capture::panic_message() {
            Some(m) => format!("Merlin hit an internal error: {m}"),
            None => "Merlin hit an internal error.".into(),
        }),
    };
    if let Some(message) = failure {
        flow::failed(&mut Native::remembered(), &message);
    }
    Ok(())
}
