// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `--progress json`: a render's live state as JSON lines on stdout, at a rate \[1\]

use crate::RenderArgs;
use anyhow::{anyhow, bail, Result};
use kestrel::batch::{self, BatchObserver};
use kestrel::session::{self, Event, Job, Monitor, MonitorObserver, Observer, Phase, Snapshot};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
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
    /// With `cancel`: keep none of the progress, which is what a stop that is not \[6\]
    pub discard: bool,
}

/// Read one control line. A line with any key this does not know, or a value \[7\]
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
            "discard" => control.discard = v.as_bool().ok_or("discard takes true or false")?,
            other => return Err(format!("unknown control key {other:?}")),
        }
    }
    Ok(control)
}

/// What the control reader hands the emitter. `api` sends the same commands, \[8\]
#[derive(Debug)]
pub(crate) enum Command {
    Interval(u64),
    Snapshot,
    Cancelled,
    Error(String),
}

/// A `progress` line: the snapshot, and the figures derived from it, so a \[9\]
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

/// Read control lines until input ends, applying `cancel` straight to the \[10\]
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
                    if control.discard {
                        monitor.cancel_discarding();
                    } else {
                        monitor.cancel();
                    }
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

/// Write the monitor's events as they happen and its snapshot at the reading \[11\]
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
        // [12]
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
        // [13]
        match rx.recv_timeout(wait) {
            Ok(command) => {
                if let Some(reply) = settings.apply(command) {
                    if send(&reply).is_err() {
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            // [14]
            Err(RecvTimeoutError::Disconnected) => controls_open = false,
        }
    }
}

/// `kestrel --force-cli render ... --progress json`. \[15\]
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

    // [16]
    let prepared = args
        .to_job()
        .map(|mut job| {
            args.with_checkpoints(&mut job, std::env::args_os());
            job
        })
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
        // [17]
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
    // [18]
    let _ = done_rx.recv_timeout(FLUSH_GRACE);
    result.map(|_| ())
}

// ---- A batch ---------------------------------------------------------------------

/// Where a line goes. `api` writes it to stdout with the request's id on it; the \[19\]
pub(crate) type Sink = Arc<dyn Fn(&Value) + Send + Sync>;

/// Put the request's `id` on a line, if it belongs to a request, and the batch job \[20\]
fn tagged(line: &Value, id: Option<&Value>, job: Option<usize>) -> Value {
    let mut line = line.clone();
    if let Value::Object(map) = &mut line {
        if let Some(id) = id {
            map.insert("id".into(), id.clone());
        }
        if let Some(job) = job {
            map.insert("job".into(), json!(job));
        }
    }
    line
}

/// The job on the stage: its monitor, and the thread writing its telemetry.
struct OnStage {
    index: usize,
    monitor: Arc<Monitor>,
    emitter: JoinHandle<()>,
}

/// Called when a job goes on the stage, with its monitor and the channel its \[21\]
pub(crate) type OnStageFn = Box<dyn FnMut(&Arc<Monitor>, Sender<Command>) -> Option<u64>>;

/// A batch's progress as JSON lines, for `api` and for `render ... --progress json`. \[22\]
pub(crate) struct BatchFeed {
    id: Option<Value>,
    send: Sink,
    cancel: Arc<AtomicBool>,
    interval_ms: u64,
    /// For each job, which set (from 1) it belongs to, and how many sets.
    set_of: Vec<usize>,
    sets: usize,
    stage: Option<OnStage>,
    obs: MonitorObserver,
    on_stage: OnStageFn,
}

impl BatchFeed {
    pub(crate) fn new(
        id: Option<Value>,
        send: Sink,
        cancel: Arc<AtomicBool>,
        interval_ms: u64,
        plan: &batch::BatchPlan,
        on_stage: OnStageFn,
    ) -> BatchFeed {
        let mut set_of = vec![0usize; plan.jobs.len()];
        for (set, group) in plan.groups().iter().enumerate() {
            for &j in group {
                set_of[j] = set + 1;
            }
        }
        BatchFeed {
            id,
            send,
            cancel,
            interval_ms,
            set_of,
            sets: plan.groups().len(),
            stage: None,
            obs: Monitor::new().observer(),
            on_stage,
        }
    }

    /// Put job `index` on the stage with a monitor of its own, unless it is.
    fn stage(&mut self, index: usize) {
        if self.stage.as_ref().is_some_and(|s| s.index == index) {
            return;
        }
        let monitor = Monitor::new();
        let (tx, rx) = mpsc::channel();
        // What `cancel`, `snapshot` and `set_interval` act on is this job.
        let period = (self.on_stage)(&monitor, tx).unwrap_or(self.interval_ms);
        let (m, id, send) = (Arc::clone(&monitor), self.id.clone(), Arc::clone(&self.send));
        let emitter = std::thread::spawn(move || {
            emit(&m, rx, period, &mut |line| {
                send(&tagged(line, id.as_ref(), Some(index)));
                Ok(())
            })
        });
        self.obs = monitor.observer();
        self.stage = Some(OnStage { index, monitor, emitter });
    }

    fn job_line(&self, state: &str, index: usize, midi_out: Option<(&std::path::Path, &std::path::Path)>, extra: Value) {
        let mut line = json!({
            "type": "batch_job",
            "job": index,
            "jobs": self.set_of.len(),
            "set": self.set_of[index],
            "sets": self.sets,
            "state": state,
        });
        if let Some((midi, out)) = midi_out {
            line["midi"] = json!(midi);
            line["out"] = json!(out);
        }
        if let (Value::Object(l), Value::Object(e)) = (&mut line, extra) {
            l.extend(e);
        }
        (self.send)(&tagged(&line, self.id.as_ref(), None));
    }
}

impl Observer for BatchFeed {
    fn phase(&mut self, phase: Phase) {
        self.obs.phase(phase);
    }
    fn setup(&mut self, setup: &session::Setup) {
        self.obs.setup(setup);
    }
    fn block(&mut self, tick: &session::Tick) {
        self.obs.block(tick);
    }
    fn tracks(&mut self, progress: &session::TrackProgress) {
        self.obs.tracks(progress);
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) || self.obs.cancelled()
    }
}

impl BatchObserver for BatchFeed {
    fn job(&mut self) -> &mut dyn Observer {
        self
    }

    fn loading(&mut self, set: usize, sets: usize, fonts: &[PathBuf], first_job: usize, jobs: usize) {
        let line = json!({
            "type": "batch_set",
            "set": set,
            "sets": sets,
            "soundfonts": fonts,
            "first_job": first_job,
            "jobs": jobs,
            "state": "loading",
        });
        (self.send)(&tagged(&line, self.id.as_ref(), None));
    }

    fn started(&mut self, index: usize, _total: usize, job: &Job) {
        // The line comes first, so a reader has the job before its telemetry.
        self.job_line("started", index, Some((&job.midi, &job.out)), json!({}));
        self.stage(index);
    }

    fn finished(&mut self, _total: usize, result: &batch::JobResult) {
        use batch::Outcome;
        // A job whose set would not load never started, and still ends.
        if self.stage.as_ref().is_none_or(|s| s.index != result.index) {
            self.job_line("started", result.index, Some((&result.midi, &result.out)), json!({}));
            self.stage(result.index);
        }
        let (state, ended, mut extra) = match &result.outcome {
            Outcome::Done(s) if !s.cancelled => ("done", Ok(s.clone()), json!({"summary": s})),
            Outcome::Done(s) => ("cancelled", Ok(s.clone()), json!({"summary": s})),
            Outcome::Failed(why) => ("failed", Err(anyhow!("{why}")), json!({"message": why})),
            Outcome::NotRun => return,
        };
        if let Some(stage) = self.stage.take() {
            // [23]
            stage.monitor.finish(&ended);
            let _ = stage.emitter.join();
        }
        extra["written"] = json!(result.written);
        self.job_line(state, result.index, None, extra);
    }
}

/// How a batch ended, as `api` answers a `batch` request with it and as the command \[24\]
pub(crate) fn batch_result(summary: &batch::BatchSummary) -> Value {
    let jobs: Vec<Value> = summary
        .results
        .iter()
        .map(|r| {
            use batch::Outcome;
            let (state, detail) = match &r.outcome {
                Outcome::Done(s) if !s.cancelled => ("done", json!({"summary": s})),
                Outcome::Done(s) => ("cancelled", json!({"summary": s})),
                Outcome::Failed(why) => ("failed", json!({"error": why})),
                Outcome::NotRun => ("not_run", json!({})),
            };
            let mut v = json!({"job": r.index, "midi": r.midi, "out": r.out, "state": state, "written": r.written});
            if let (Value::Object(m), Value::Object(d)) = (&mut v, detail) {
                m.extend(d);
            }
            v
        })
        .collect();
    json!({
        "jobs": jobs,
        "done": summary.done(),
        "failed": summary.failed(),
        "cancelled": summary.cancelled,
        "loads": summary.loads,
        "wall_secs": summary.wall_secs,
    })
}

/// Which job controls on stdin reach, and the interval they last asked for.
#[derive(Default)]
struct Target {
    monitor: Option<Arc<Monitor>>,
    tx: Option<Sender<Command>>,
    interval_ms: u64,
}

/// Read control lines for a batch until input ends. `cancel` stops the job on the \[25\]
fn read_batch_controls(input: impl BufRead, target: &Mutex<Target>, cancel: &AtomicBool, send: &Sink) {
    for line in input.lines() {
        let Ok(line) = line else { return };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_control(line) {
            Ok(control) => {
                let mut t = target.lock().unwrap_or_else(|p| p.into_inner());
                if control.cancel {
                    cancel.store(true, Ordering::Relaxed);
                    if let Some(m) = &t.monitor {
                        m.cancel();
                    }
                    if let Some(tx) = &t.tx {
                        let _ = tx.send(Command::Cancelled);
                    }
                }
                if let Some(ms) = control.interval_ms {
                    t.interval_ms = clamp_interval(ms);
                    match &t.tx {
                        Some(tx) => {
                            let _ = tx.send(Command::Interval(ms));
                        }
                        // No job on the stage yet: the reply is made here.
                        None => send(&config_line(t.interval_ms, cancel.load(Ordering::Relaxed))),
                    }
                }
                if control.snapshot {
                    if let Some(tx) = &t.tx {
                        let _ = tx.send(Command::Snapshot);
                    }
                }
            }
            Err(message) => send(&json!({"type": "control_error", "message": message})),
        }
    }
}

/// `kestrel --force-cli render a.mid b.mid ... -o DIR --progress json`. \[26\]
pub fn run_batch(args: &RenderArgs) -> Result<()> {
    let interval_ms = clamp_interval(args.progress_interval);
    let send: Sink = Arc::new(|line| {
        let _ = write_line(&mut std::io::stdout(), line);
    });
    let midis = args.midis();
    send(&json!({
        "type": "hello",
        "schema": SCHEMA,
        "version": env!("CARGO_PKG_VERSION"),
        "batch": true,
        "midi": midis[0].to_string_lossy(),
        "midis": midis.iter().map(|m| m.to_string_lossy()).collect::<Vec<_>>(),
        "jobs": midis.len(),
        "out": args.out.to_string_lossy(),
        "interval_ms": interval_ms,
    }));
    send(&config_line(interval_ms, false));

    let plan = match args.to_batch_jobs().and_then(batch::plan) {
        Ok(plan) => plan,
        Err(e) => {
            send(&json!({"type": "failed", "t": 0.0, "message": format!("{e:#}")}));
            return Err(e);
        }
    };

    let target = Arc::new(Mutex::new(Target { interval_ms, ..Default::default() }));
    let cancel = Arc::new(AtomicBool::new(false));
    {
        // [27]
        let (target, cancel, send) = (Arc::clone(&target), Arc::clone(&cancel), Arc::clone(&send));
        std::thread::spawn(move || read_batch_controls(std::io::stdin().lock(), &target, &cancel, &send));
    }
    let on_stage: OnStageFn = {
        let target = Arc::clone(&target);
        Box::new(move |monitor, tx| {
            let mut t = target.lock().unwrap_or_else(|p| p.into_inner());
            t.monitor = Some(Arc::clone(monitor));
            t.tx = Some(tx);
            Some(t.interval_ms)
        })
    };
    let mut feed = BatchFeed::new(None, Arc::clone(&send), Arc::clone(&cancel), interval_ms, &plan, on_stage);
    let summary = batch::run(plan, &mut feed);

    let mut last = batch_result(&summary);
    last["type"] = json!("batch_summary");
    send(&last);
    let total = summary.results.len();
    if summary.cancelled {
        bail!("the batch was stopped: {} of {total} jobs finished", summary.done());
    }
    if summary.failed() > 0 {
        bail!("{} of {total} jobs failed", summary.failed());
    }
    Ok(())
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
                cancel: true,
                ..Default::default()
            })
        );
        assert_eq!(
            parse_control(r#"{"cancel": true, "discard": true}"#),
            Ok(Control { cancel: true, discard: true, ..Default::default() })
        );
        for bad in [
            "interval 100",
            "[1, 2]",
            r#"{"interval_ms": -5}"#,
            r#"{"interval_ms": 2.5}"#,
            r#"{"cancel": "yes"}"#,
            r#"{"cancel": true, "discard": "yes"}"#,
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
            notes_skipped: 0,
            voices_spawned: 5,
            peak_voices: 5,
            stolen: 0,
            dropped: 0,
            peak_level: 0.25,
            clipped: 0,
            cancelled: false,
            checkpoint: None,
            discarded: false,
        }));
        monitor
    }

    fn lines(out: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(out)
            .lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}")))
            .collect()
    }

    /// With periodic progress off, a reader gets the events in order, one \[28\]
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

    /// A control message is answered before the progress it changes, and a \[29\]
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

    /// The control reader stops at the end of its input and applies cancel \[30\]
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
        // A plain cancel keeps the progress; the stop is the render's to save.
        assert!(!monitor.is_discarding());
    }

    /// `{"cancel": true, "discard": true}` is a stop that keeps none of its \[31\]
    #[test]
    fn a_cancel_can_ask_to_keep_no_progress() {
        let monitor = Monitor::new();
        let (tx, rx) = mpsc::channel();
        read_controls("{\"discard\": true}\n".as_bytes(), &monitor, &tx);
        assert!(!monitor.is_cancelled() && !monitor.is_discarding());
        assert_eq!(rx.try_iter().count(), 0);
        read_controls("{\"cancel\": true, \"discard\": true}\n".as_bytes(), &monitor, &tx);
        assert!(monitor.is_cancelled() && monitor.is_discarding());
        assert!(matches!(rx.try_iter().next(), Some(Command::Cancelled)));
    }
}
