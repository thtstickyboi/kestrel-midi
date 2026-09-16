// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `--progress json`: a render's live state as JSON lines on stdout, at a rate \[1\]

use crate::RenderArgs;
use anyhow::Result;
use kestrel::session::{self, Event, Monitor, Snapshot};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The wire format's version, sent in every `hello`. Adding a field leaves it \[2\]
pub const SCHEMA: u32 = 1;
const MIN_INTERVAL_MS: u64 = 10;
const MAX_INTERVAL_MS: u64 = 60_000;
/// How often the emitter looks for new events whatever the interval is, so a \[3\]
const EVENT_POLL: Duration = Duration::from_millis(100);
/// How long the end of a render waits for a reader that has stopped reading \[4\]
const FLUSH_GRACE: Duration = Duration::from_secs(5);

/// A requested interval as it will be applied: 0 turns periodic progress off, \[5\]
pub fn clamp_interval(ms: u64) -> u64 {
    if ms == 0 {
        0
    } else {
        ms.clamp(MIN_INTERVAL_MS, MAX_INTERVAL_MS)
    }
}

/// What one control line asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Control {
    pub interval_ms: Option<u64>,
    pub snapshot: bool,
    pub cancel: bool,
}

/// Read one control line. A line with any key this does not know, or a value \[6\]
pub fn parse_control(line: &str) -> std::result::Result<Control, String> {
    let value: Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or("a control message is a JSON object")?;
    let mut control = Control::default();
    for (key, v) in object {
        match key.as_str() {
            "interval_ms" => {
                control.interval_ms = Some(
                    v.as_u64()
                        .ok_or("interval_ms takes a whole, non-negative number of milliseconds")?,
                )
            }
            "snapshot" => control.snapshot = v.as_bool().ok_or("snapshot takes true or false")?,
            "cancel" => control.cancel = v.as_bool().ok_or("cancel takes true or false")?,
            other => return Err(format!("unknown control key {other:?}")),
        }
    }
    Ok(control)
}

/// What the control reader hands the emitter. `api` sends the same commands, \[7\]
#[derive(Debug)]
pub(crate) enum Command {
    Interval(u64),
    Snapshot,
    Cancelled,
    Error(String),
}

/// A `progress` line: the snapshot, and the figures derived from it, so a \[8\]
#[derive(Serialize)]
struct Progress<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(flatten)]
    snap: &'a Snapshot,
    progress: Option<f64>,
    speed: Option<f64>,
    eta_secs: Option<f64>,
}

fn progress_line(snap: &Snapshot) -> Progress<'_> {
    Progress {
        kind: "progress",
        snap,
        progress: snap.progress(),
        speed: snap.speed(),
        eta_secs: snap.eta_secs(),
    }
}

/// A `progress` line as a value, for a caller that adds to it before sending.
pub(crate) fn progress_value(snap: &Snapshot) -> Value {
    to_value(&progress_line(snap))
}

fn to_value(value: &impl Serialize) -> Value {
    serde_json::to_value(value).unwrap_or_else(|e| json!({"type": "control_error", "message": e.to_string()}))
}

fn write_line(out: &mut dyn Write, value: &Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *out, value).map_err(std::io::Error::other)?;
    out.write_all(b"\n")?;
    out.flush()
}

fn config_line(interval_ms: u64, cancel_requested: bool) -> Value {
    json!({"type": "config", "interval_ms": interval_ms, "cancel_requested": cancel_requested})
}

/// Read control lines until input ends, applying `cancel` straight to the \[9\]
fn read_controls(input: impl BufRead, monitor: &Monitor, tx: &Sender<Command>) {
    for line in input.lines() {
        let Ok(line) = line else { return };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut commands = Vec::new();
        match parse_control(line) {
            Ok(control) => {
                if control.cancel {
                    monitor.cancel();
                    commands.push(Command::Cancelled);
                }
                if let Some(ms) = control.interval_ms {
                    commands.push(Command::Interval(ms));
                }
                if control.snapshot {
                    commands.push(Command::Snapshot);
                }
            }
            Err(message) => commands.push(Command::Error(message)),
        }
        for command in commands {
            if tx.send(command).is_err() {
                return;
            }
        }
    }
}

/// The emitter's settings, as the reading program has left them.
struct Settings {
    /// 0 is no periodic progress.
    interval_ms: u64,
    /// A `snapshot` is owed.
    asked: bool,
    cancel_requested: bool,
}

impl Settings {
    /// Apply one command, returning the reply it calls for, if any.
    fn apply(&mut self, command: Command) -> Option<Value> {
        match command {
            Command::Interval(ms) => {
                self.interval_ms = clamp_interval(ms);
                Some(config_line(self.interval_ms, self.cancel_requested))
            }
            Command::Snapshot => {
                self.asked = true;
                None
            }
            Command::Cancelled => {
                self.cancel_requested = true;
                Some(config_line(self.interval_ms, true))
            }
            Command::Error(message) => Some(json!({"type": "control_error", "message": message})),
        }
    }
}

/// Write the monitor's events as they happen and its snapshot at the reading \[10\]
pub(crate) fn emit(
    monitor: &Monitor,
    rx: Receiver<Command>,
    interval_ms: u64,
    send: &mut dyn FnMut(&Value) -> std::io::Result<()>,
) {
    let mut settings = Settings {
        interval_ms,
        asked: false,
        cancel_requested: false,
    };
    let mut cursor = 0usize;
    let mut last_progress: Option<Instant> = None;
    let mut controls_open = true;

    loop {
        // Replies first, so a `config` comes before the progress it governs.
        let pending: Vec<Command> = rx.try_iter().collect();
        for command in pending {
            if let Some(reply) = settings.apply(command) {
                if send(&reply).is_err() {
                    return;
                }
            }
        }

        let (events, next) = monitor.events_since(cursor);
        cursor = next;
        let mut end = None;
        for event in events {
            if matches!(event, Event::Summary { .. } | Event::Failed { .. }) {
                end = Some(event);
            } else if send(&to_value(&event)).is_err() {
                return;
            }
        }

        let now = Instant::now();
        let period = Duration::from_millis(settings.interval_ms);
        let due = settings.interval_ms > 0
            && match last_progress {
                Some(t) => now.saturating_duration_since(t) >= period,
                None => true,
            };
        // [11]
        if due || settings.asked || end.is_some() {
            if send(&progress_value(&monitor.snapshot())).is_err() {
                return;
            }
            last_progress = Some(now);
            settings.asked = false;
        }
        if let Some(end) = end {
            let _ = send(&to_value(&end));
            return;
        }

        let until_due = match (settings.interval_ms, last_progress) {
            (0, _) => EVENT_POLL,
            (_, Some(t)) => (t + period).saturating_duration_since(Instant::now()),
            (_, None) => Duration::ZERO,
        };
        let wait = until_due.min(EVENT_POLL).max(Duration::from_millis(1));
        if !controls_open {
            std::thread::sleep(wait);
            continue;
        }
        // [12]
        match rx.recv_timeout(wait) {
            Ok(command) => {
                if let Some(reply) = settings.apply(command) {
                    if send(&reply).is_err() {
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            // [13]
            Err(RecvTimeoutError::Disconnected) => controls_open = false,
        }
    }
}

/// `kestrel --force-cli render ... --progress json`. \[14\]
pub fn run(args: &RenderArgs) -> Result<()> {
    let interval_ms = clamp_interval(args.progress_interval);
    let mut stdout = std::io::stdout();
    let hello = json!({
        "type": "hello",
        "schema": SCHEMA,
        "version": env!("CARGO_PKG_VERSION"),
        "midi": args.midi.to_string_lossy(),
        "out": args.out.to_string_lossy(),
        "interval_ms": interval_ms,
    });
    let _ = write_line(&mut stdout, &hello);
    let _ = write_line(&mut stdout, &config_line(interval_ms, false));

    let prepared = args
        .to_job()
        .and_then(|job| session::plan(&job).map(|plan| (job, plan)));
    let (job, plan) = match prepared {
        Ok(p) => p,
        Err(e) => {
            let line = json!({"type": "failed", "t": 0.0, "message": format!("{e:#}")});
            let _ = write_line(&mut stdout, &line);
            return Err(e);
        }
    };

    let monitor = Monitor::new();
    let (tx, rx) = mpsc::channel();
    {
        // [15]
        let monitor = Arc::clone(&monitor);
        std::thread::spawn(move || read_controls(std::io::stdin().lock(), &monitor, &tx));
    }
    let (done_tx, done_rx) = mpsc::channel::<()>();
    {
        let monitor = Arc::clone(&monitor);
        std::thread::spawn(move || {
            let mut stdout = std::io::stdout();
            emit(&monitor, rx, interval_ms, &mut |line| write_line(&mut stdout, line));
            let _ = done_tx.send(());
        });
    }

    let result = session::run_monitored(&job, plan, None, &monitor);
    // [16]
    let _ = done_rx.recv_timeout(FLUSH_GRACE);
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel::session::{Observer, Phase, Setup, Summary};

    #[test]
    fn control_lines_are_read_whole_or_refused_whole() {
        assert_eq!(
            parse_control(r#"{"interval_ms": 100}"#),
            Ok(Control {
                interval_ms: Some(100),
                ..Default::default()
            })
        );
        assert_eq!(
            parse_control(r#"{"interval_ms": 0, "snapshot": true, "cancel": true}"#),
            Ok(Control {
                interval_ms: Some(0),
                snapshot: true,
                cancel: true
            })
        );
        for bad in [
            "interval 100",
            "[1, 2]",
            r#"{"interval_ms": -5}"#,
            r#"{"interval_ms": 2.5}"#,
            r#"{"cancel": "yes"}"#,
            r#"{"interval_ms": 100, "intreval_ms": 50}"#,
        ] {
            assert!(parse_control(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn an_interval_is_zero_or_between_ten_ms_and_a_minute() {
        assert_eq!(clamp_interval(0), 0);
        assert_eq!(clamp_interval(1), 10);
        assert_eq!(clamp_interval(250), 250);
        assert_eq!(clamp_interval(10_000_000), 60_000);
    }

    fn finished_monitor() -> Arc<Monitor> {
        let monitor = Monitor::new();
        let mut obs = monitor.observer();
        obs.phase(Phase::LoadingSoundfont);
        obs.setup(&Setup {
            backend: "gpu",
            adapter: Some("test".into()),
            device_bytes: Some(1),
            vendor_id: None,
            device_id: None,
            tracks: 2,
            max_voices: 3,
            bytes_total: 4,
        });
        obs.phase(Phase::Rendering);
        monitor.finish(&Ok(Summary {
            bytes: 10,
            audio_secs: 1.0,
            wall_secs: 0.5,
            notes: 5,
            voices_spawned: 5,
            peak_voices: 5,
            stolen: 0,
            dropped: 0,
            peak_level: 0.25,
            clipped: 0,
            cancelled: false,
        }));
        monitor
    }

    fn lines(out: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(out)
            .lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}")))
            .collect()
    }

    /// With periodic progress off, a reader gets the events in order, one \[17\]
    #[test]
    fn the_summary_is_always_the_last_line() {
        let monitor = finished_monitor();
        let (_tx, rx) = mpsc::channel();
        let mut out = Vec::new();
        emit(&monitor, rx, 0, &mut |line| write_line(&mut out, line));
        let got = lines(&out);
        let types: Vec<&str> = got.iter().map(|v| v["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            ["phase", "setup", "phase", "phase", "progress", "summary"],
            "{got:?}"
        );
        assert_eq!(got[1]["max_voices"], 3);
        assert_eq!(got[3]["phase"], "finished");
        assert!(got[4].get("speed").is_some() && got[4].get("progress").is_some());
    }

    /// A control message is answered before the progress it changes, and a \[18\]
    #[test]
    fn control_replies_come_before_the_progress_they_govern() {
        let monitor = finished_monitor();
        let (tx, rx) = mpsc::channel();
        tx.send(Command::Interval(5)).unwrap();
        tx.send(Command::Error("unknown control key \"x\"".into())).unwrap();
        tx.send(Command::Cancelled).unwrap();
        let mut out = Vec::new();
        emit(&monitor, rx, 0, &mut |line| write_line(&mut out, line));
        let got = lines(&out);
        assert_eq!(got[0]["type"], "config");
        assert_eq!(got[0]["interval_ms"], 10, "5 ms is clamped, and the reply says so");
        assert_eq!(got[1]["type"], "control_error");
        assert_eq!(got[2]["cancel_requested"], true);
        assert_eq!(got.last().unwrap()["type"], "summary");
    }

    /// The control reader stops at the end of its input and applies cancel \[19\]
    #[test]
    fn cancel_is_applied_the_moment_it_is_read() {
        let monitor = Monitor::new();
        let (tx, rx) = mpsc::channel();
        let input = "{\"snapshot\": true}\n\n{\"cancel\": true}\nnot json\n";
        read_controls(input.as_bytes(), &monitor, &tx);
        assert!(monitor.is_cancelled());
        let got: Vec<Command> = rx.try_iter().collect();
        assert!(matches!(got[0], Command::Snapshot), "{got:?}");
        assert!(matches!(got[1], Command::Cancelled), "{got:?}");
        assert!(matches!(got[2], Command::Error(_)), "{got:?}");
    }
}
