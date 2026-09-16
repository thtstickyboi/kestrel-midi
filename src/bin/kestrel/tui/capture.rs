// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A logger that holds records for the screen instead of writing them. \[1\]

use log::{Level, LevelFilter, Log, Metadata, Record};
use std::sync::Mutex;

struct Capture {
    held: Mutex<Vec<(Level, String)>>,
}

static CAPTURE: Capture = Capture {
    held: Mutex::new(Vec::new()),
};

impl Log for Capture {
    fn enabled(&self, m: &Metadata) -> bool {
        // [2]
        if m.target().starts_with("kestrel") {
            m.level() <= Level::Info
        } else {
            m.level() <= Level::Error
        }
    }

    fn log(&self, r: &Record) {
        if self.enabled(r.metadata()) {
            if let Ok(mut held) = self.held.lock() {
                held.push((r.level(), r.args().to_string()));
            }
        }
    }

    fn flush(&self) {}
}

pub fn install() {
    if log::set_logger(&CAPTURE).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
}

/// The last panic and where it was raised, held for the same reason the \[3\]
static PANIC: Mutex<Option<String>> = Mutex::new(None);

/// Hold panics here instead of printing them. Whatever catches one shows the \[4\]
pub fn hold_panics() {
    std::panic::set_hook(Box::new(|info| {
        if let Ok(mut held) = PANIC.lock() {
            *held = Some(info.to_string());
        }
    }));
}

/// Take the last panic held, with where it was raised.
pub fn panic_message() -> Option<String> {
    PANIC.lock().ok().and_then(|mut held| held.take())
}

/// Take the warnings and errors held since the last call, and drop the rest.
pub fn problems() -> Vec<(Level, String)> {
    let held = match CAPTURE.held.lock() {
        Ok(mut h) => std::mem::take(&mut *h),
        Err(_) => return Vec::new(),
    };
    held.into_iter().filter(|(l, _)| *l <= Level::Warn).collect()
}
