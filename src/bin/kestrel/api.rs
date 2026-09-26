// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `kestrel --force-cli api`: Kestrel driven by another program -- a GUI, a \[1\]

use crate::feed::{self, Command as FeedCommand};
use crate::tui::checks::{self, FontProfile, Verdict};
use crate::{Cli, Cmd, RenderArgs};
use anyhow::{anyhow, bail, Context, Result};
use clap::{ArgAction, CommandFactory, Parser};
use kestrel::config::Config;
use kestrel::midi::{Division, Event as MidiEvent, MidiStream, TempoClock};
use kestrel::session::{self, Job, Monitor, Observer, Phase, Plan};
use kestrel::Bank;
use log::{Level, LevelFilter, Log, Metadata, Record};
use serde_json::{json, Map, Value};
use std::ffi::OsString;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The protocol's version, sent in `ready`. Adding a command or a field leaves \[2\]
pub const API: u32 = 1;

/// Every command, as `ready` lists them.
const COMMANDS: &[&str] = &[
    "adapters",
    "ffmpeg",
    "options",
    "inspect_midi",
    "scan_midi",
    "load_soundfonts",
    "unload",
    "render",
    "cancel",
    "set_interval",
    "snapshot",
    "status",
    "check_update",
    "shutdown",
];

/// Render arguments a request does not pass as options: the ones it names \[3\]
const NOT_OPTIONS: &[&str] = &[
    "soundfont",
    "sf-programs",
    "out",
    "progress",
    "progress-interval",
    "force-cli",
    "help",
];

/// Stand-ins for the MIDI and output when the render arguments are parsed only \[4\]
const PLACEHOLDER_MIDI: &str = "kestrel-api-placeholder.mid";
const PLACEHOLDER_OUT: &str = "kestrel-api-placeholder.wav";

// ---- Output ---------------------------------------------------------------

/// Write one line. Stdout's own lock keeps a line whole against every other \[5\]
fn send(line: &Value) {
    let mut out = std::io::stdout().lock();
    if serde_json::to_writer(&mut out, line).is_ok() {
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }
}

fn respond_ok(id: &Value, result: Value) {
    send(&json!({"type": "response", "id": id, "ok": true, "result": result}));
}

fn respond_err(id: &Value, error: impl std::fmt::Display) {
    send(&json!({"type": "response", "id": id, "ok": false, "error": error.to_string()}));
}

fn with_id(line: &Value, id: &Value) -> Value {
    let mut line = line.clone();
    if let Value::Object(map) = &mut line {
        map.insert("id".into(), id.clone());
    }
    line
}

// ---- Logging --------------------------------------------------------------

/// Log records as `log` lines, marked with the request running when they were \[6\]
struct ApiLog;

static LOGGER: ApiLog = ApiLog;
static LOG_ID: Mutex<Option<Value>> = Mutex::new(None);

impl ApiLog {
    /// What the protocol carries. As the guided renderer filters: Kestrel's \[7\]
    fn sends(&self, m: &Metadata) -> bool {
        if m.target().starts_with("kestrel") {
            m.level() <= Level::Info
        } else {
            m.level() <= Level::Error
        }
    }
}

impl Log for ApiLog {
    fn enabled(&self, m: &Metadata) -> bool {
        self.sends(m) || kestrel::falconeye::renderlog::wants(m.level(), m.target())
    }

    fn log(&self, r: &Record) {
        // The render log keeps its own filter, the same for every front end.
        kestrel::falconeye::renderlog::record(r.level(), r.target(), r.args());
        if !self.sends(r.metadata()) {
            return;
        }
        let level = match r.level() {
            Level::Error => "error",
            Level::Warn => "warn",
            Level::Info => "info",
            _ => "debug",
        };
        let mut line = json!({"type": "log", "level": level, "message": r.args().to_string()});
        if let Some(id) = LOG_ID.lock().ok().and_then(|held| held.clone()) {
            line["id"] = id;
        }
        send(&line);
    }

    fn flush(&self) {}
}

fn set_log_id(id: Option<Value>) {
    if let Ok(mut held) = LOG_ID.lock() {
        *held = id;
    }
}

// ---- Session state --------------------------------------------------------

/// What makes one loaded bank the bank another request would build: the same \[8\]
#[derive(Debug, Clone, PartialEq, Eq)]
struct LoadKey {
    soundfonts: Vec<PathBuf>,
    sf_programs: Option<String>,
    /// Option arguments not in `checks::LOAD_NEUTRAL`, sorted.
    flags: Vec<String>,
}

struct Loaded {
    key: LoadKey,
    bank: Arc<Bank>,
    info: Value,
}

/// The one long request that may run at a time. A GPU render wants the GPU \[9\]
struct Running {
    id: Value,
    cmd: &'static str,
    cancel: Arc<AtomicBool>,
    monitor: Option<Arc<Monitor>>,
    control: Option<Sender<FeedCommand>>,
}

struct State {
    running: Option<Running>,
    loaded: Option<Loaded>,
    /// Milliseconds between `progress` lines for the next render, and the \[10\]
    interval_ms: u64,
}

type Shared = Arc<Mutex<State>>;

fn lock(state: &Shared) -> MutexGuard<'_, State> {
    // [11]
    state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---- The loop -------------------------------------------------------------

/// `kestrel --force-cli api`.
pub fn run() -> Result<()> {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
    crate::start_falconeye("api");
    send(&json!({
        "type": "ready",
        "api": API,
        "version": env!("CARGO_PKG_VERSION"),
        // [12]
        "build": crate::BUILD,
        "commands": COMMANDS,
    }));

    let state: Shared = Arc::new(Mutex::new(State {
        running: None,
        loaded: None,
        interval_ms: 250,
    }));
    let mut worker: Option<JoinHandle<()>> = None;
    // [13]
    let mut quick: Vec<JoinHandle<()>> = Vec::new();

    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                respond_err(&Value::Null, format!("not JSON: {e}"));
                continue;
            }
        };
        let Some(req) = request.as_object() else {
            respond_err(&Value::Null, "a request is a JSON object");
            continue;
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        if id.is_null() {
            respond_err(&id, "every request needs an \"id\", which its response carries back");
            continue;
        }
        let Some(cmd) = req.get("cmd").and_then(Value::as_str) else {
            respond_err(&id, "missing \"cmd\"; the ready line lists the commands");
            continue;
        };

        match cmd {
            "shutdown" => {
                stop(&state, &mut worker, &mut quick);
                respond_ok(&id, json!({}));
                return Ok(());
            }
            "cancel" => cancel(&id, &state),
            "set_interval" => set_interval(&id, req, &state),
            "snapshot" => snapshot(&id, &state),
            "status" => status(&id, &state),
            "unload" => {
                let was = lock(&state).loaded.take().is_some();
                respond_ok(&id, json!({"unloaded": was}));
            }
            // [14]
            "adapters" | "ffmpeg" | "options" | "inspect_midi" | "check_update" => {
                let (cmd, id, req) = (cmd.to_string(), id.clone(), req.clone());
                quick.retain(|h| !h.is_finished());
                quick.push(std::thread::spawn(move || {
                    let result = match cmd.as_str() {
                        "adapters" => adapters(),
                        "ffmpeg" => ffmpeg(&req),
                        "options" => Ok(options()),
                        "check_update" => check_update(),
                        _ => inspect_midi(&req),
                    };
                    match result {
                        Ok(v) => respond_ok(&id, v),
                        Err(e) => respond_err(&id, format!("{e:#}")),
                    }
                }));
            }
            "load_soundfonts" | "scan_midi" | "render" => {
                start_long(cmd, &id, req, &state, &mut worker)
            }
            other => respond_err(&id, format!("unknown command {other:?}; the ready line lists the commands")),
        }
    }

    // [15]
    stop(&state, &mut worker, &mut quick);
    Ok(())
}

/// Cancel whatever is running, and wait for it and every quick request to \[16\]
fn stop(state: &Shared, worker: &mut Option<JoinHandle<()>>, quick: &mut Vec<JoinHandle<()>>) {
    if let Some(r) = &lock(state).running {
        r.cancel.store(true, Ordering::Relaxed);
        if let Some(m) = &r.monitor {
            m.cancel();
        }
    }
    if let Some(handle) = worker.take() {
        let _ = handle.join();
    }
    for handle in quick.drain(..) {
        let _ = handle.join();
    }
}

// ---- Long requests --------------------------------------------------------

enum Long {
    Load(Box<LoadJob>),
    Scan(ScanJob),
    Render(Box<RenderJob>),
}

struct LoadJob {
    key: LoadKey,
    cfg: Config,
}

struct ScanJob {
    path: PathBuf,
    rate: u32,
}

struct RenderJob {
    job: Job,
    plan: Plan,
    key: LoadKey,
    interval_ms: u64,
}

fn start_long(
    cmd: &str,
    id: &Value,
    req: &Map<String, Value>,
    state: &Shared,
    worker: &mut Option<JoinHandle<()>>,
) {
    if let Some(r) = &lock(state).running {
        respond_err(
            id,
            format!(
                "busy: {} (id {}) is still running; wait for its response, or cancel it",
                r.cmd, r.id
            ),
        );
        return;
    }
    // Whatever ran before has sent its response; this only collects the thread.
    if let Some(handle) = worker.take() {
        let _ = handle.join();
    }

    // [17]
    let (cmd, prepared): (&'static str, Result<Long>) = match cmd {
        "load_soundfonts" => (
            "load_soundfonts",
            prepare_load(req).map(|l| Long::Load(Box::new(l))),
        ),
        "scan_midi" => ("scan_midi", prepare_scan(req).map(Long::Scan)),
        _ => (
            "render",
            prepare_render(req, state).map(|r| Long::Render(Box::new(r))),
        ),
    };
    let long = match prepared {
        Ok(l) => l,
        Err(e) => {
            respond_err(id, format!("{e:#}"));
            return;
        }
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let (monitor, control, rx) = match &long {
        Long::Render(_) => {
            let (tx, rx) = mpsc::channel();
            (Some(Monitor::new()), Some(tx), Some(rx))
        }
        _ => (None, None, None),
    };
    lock(state).running = Some(Running {
        id: id.clone(),
        cmd,
        cancel: Arc::clone(&cancel),
        monitor: monitor.clone(),
        control,
    });
    set_log_id(Some(id.clone()));

    let (id, state) = (id.clone(), Arc::clone(state));
    *worker = Some(std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match long {
            Long::Load(job) => load(*job, &state),
            Long::Scan(job) => scan(job, &id, &cancel, &state),
            Long::Render(job) => render(
                *job,
                &id,
                &state,
                monitor.expect("a render has a monitor"),
                rx.expect("a render has a control channel"),
            ),
        }))
        .unwrap_or_else(|payload| {
            Err(anyhow!(
                "stopped on an internal error: {}",
                payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("no message")
            ))
        });
        // [18]
        lock(&state).running = None;
        set_log_id(None);
        match outcome {
            Ok(v) => respond_ok(&id, v),
            Err(e) => respond_err(&id, format!("{e:#}")),
        }
    }));
}

fn str_param(req: &Map<String, Value>, name: &str) -> Result<String> {
    match req.get(name) {
        Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        Some(_) => bail!("{name:?} is a non-empty string"),
        None => bail!("missing {name:?}"),
    }
}

fn soundfonts_param(req: &Map<String, Value>) -> Result<Option<Vec<PathBuf>>> {
    match req.get("soundfonts") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .context("\"soundfonts\" is a list of paths")
            })
            .collect::<Result<Vec<_>>>()
            .map(|v| (!v.is_empty()).then_some(v)),
        Some(_) => bail!("\"soundfonts\" is a list of paths, even for one"),
    }
}

fn programs_param(req: &Map<String, Value>) -> Result<Option<String>> {
    match req.get("sf_programs") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => bail!("\"sf_programs\" is a string, spelled as --sf-programs takes it: \"0,1\" or \"0-7\""),
    }
}

fn prepare_load(req: &Map<String, Value>) -> Result<LoadJob> {
    let soundfonts = soundfonts_param(req)?.context("missing \"soundfonts\"")?;
    let sf_programs = programs_param(req)?;
    let flags = option_flags(req.get("options"))?;
    let argv = render_argv(
        PLACEHOLDER_MIDI.as_ref(),
        &soundfonts,
        sf_programs.as_deref(),
        PLACEHOLDER_OUT.as_ref(),
        &flags,
    );
    let (cfg, _) = parse_render(argv)?.to_config()?;
    Ok(LoadJob {
        key: LoadKey {
            soundfonts,
            sf_programs,
            flags: load_flags(&flags),
        },
        cfg,
    })
}

fn prepare_scan(req: &Map<String, Value>) -> Result<ScanJob> {
    let rate = match req.get("rate") {
        None | Some(Value::Null) => Config::default().sample_rate,
        Some(v) => v
            .as_u64()
            .filter(|r| (8_000..=768_000).contains(r))
            .context("\"rate\" is a sample rate in Hz, 8000 to 768000")? as u32,
    };
    Ok(ScanJob {
        path: str_param(req, "path")?.into(),
        rate,
    })
}

fn prepare_render(req: &Map<String, Value>, state: &Shared) -> Result<RenderJob> {
    let midi = PathBuf::from(str_param(req, "midi")?);
    let out = PathBuf::from(str_param(req, "out")?);
    let flags = option_flags(req.get("options"))?;

    let (soundfonts, sf_programs) = match soundfonts_param(req)? {
        Some(fonts) => (fonts, programs_param(req)?),
        None => {
            let st = lock(state);
            let loaded = st.loaded.as_ref().context(
                "no soundfonts: pass \"soundfonts\", or load them first with load_soundfonts",
            )?;
            let programs = match programs_param(req)? {
                Some(p) => Some(p),
                None => loaded.key.sf_programs.clone(),
            };
            (loaded.key.soundfonts.clone(), programs)
        }
    };

    let argv = render_argv(&midi, &soundfonts, sf_programs.as_deref(), &out, &flags);
    kestrel::falconeye::renderlog::set_args(&argv);
    let job = parse_render(argv)?.to_job()?;
    let plan = session::plan(&job)?;
    let interval_ms = match req.get("progress_interval_ms") {
        None | Some(Value::Null) => lock(state).interval_ms,
        Some(v) => feed::clamp_interval(
            v.as_u64()
                .context("\"progress_interval_ms\" is a whole number of milliseconds")?,
        ),
    };
    Ok(RenderJob {
        job,
        plan,
        key: LoadKey {
            soundfonts,
            sf_programs,
            flags: load_flags(&flags),
        },
        interval_ms,
    })
}

/// Load, or keep what is loaded when it is already this.
fn load(job: LoadJob, state: &Shared) -> Result<Value> {
    if let Some(loaded) = &lock(state).loaded {
        if loaded.key == job.key {
            let mut info = loaded.info.clone();
            info["reused"] = json!(true);
            return Ok(info);
        }
    }
    let loaded = load_into(job.key, &job.cfg, state)?;
    Ok(loaded)
}

/// Load a bank and keep it as the session's, returning its description.
fn load_into(key: LoadKey, cfg: &Config, state: &Shared) -> Result<Value> {
    // The old bank goes first, so two large pools are never held at once.
    lock(state).loaded = None;
    let t0 = Instant::now();
    let bank = session::load_layered(&key.soundfonts, key.sf_programs.as_deref(), cfg)?;
    let secs = t0.elapsed().as_secs_f64();
    log::info!(target: "kestrel", "loaded {} in {:.2?}", bank.describe(), t0.elapsed());
    let mut info = bank_info(&bank, &key);
    info["load_secs"] = json!(secs);
    info["reused"] = json!(false);
    lock(state).loaded = Some(Loaded {
        key,
        bank: Arc::new(bank),
        info: info.clone(),
    });
    Ok(info)
}

fn bank_info(bank: &Bank, key: &LoadKey) -> Value {
    let profile = FontProfile::of(bank);
    let presets: Vec<Value> = bank
        .presets
        .iter()
        .map(|p| json!({"bank": p.bank, "program": p.program, "name": p.name}))
        .collect();
    json!({
        "soundfonts": key.soundfonts,
        "sf_programs": key.sf_programs,
        "name": bank.name,
        "presets": bank.presets.len(),
        "regions": bank.regions.len(),
        "samples": bank.samples.len(),
        "pool_bytes": bank.pool_bytes(),
        "pool_rate": bank.pool_rate,
        "uses_lfo": bank.uses_lfo,
        "uses_mod_env": bank.uses_mod_env,
        "general_midi": profile.is_general_midi(),
        "bank0_programs": profile.programs,
        "drum_kits": profile.drum_kits,
        "preset_list": presets,
    })
}

fn render(job: RenderJob, id: &Value, state: &Shared, monitor: Arc<Monitor>, rx: Receiver<FeedCommand>) -> Result<Value> {
    let RenderJob {
        job,
        plan,
        key,
        interval_ms,
    } = job;

    let emitter = {
        let (monitor, id) = (Arc::clone(&monitor), id.clone());
        std::thread::spawn(move || {
            feed::emit(&monitor, rx, interval_ms, &mut |line| {
                send(&with_id(line, &id));
                Ok(())
            })
        })
    };

    let cached = lock(state)
        .loaded
        .as_ref()
        .filter(|l| l.key == key)
        .map(|l| Arc::clone(&l.bank));
    let bank = match cached {
        Some(bank) => Ok(bank),
        None => {
            monitor.observer().phase(Phase::LoadingSoundfont);
            load_into(key.clone(), &plan.cfg, state).and_then(|_| {
                lock(state)
                    .loaded
                    .as_ref()
                    .map(|l| Arc::clone(&l.bank))
                    .context("the soundfont was loaded and then unloaded before the render began")
            })
        }
    };

    let out = job.out.clone();
    let result = match bank {
        Ok(bank) => session::run_monitored(&job, plan, Some(bank), &monitor),
        Err(e) => {
            let message = format!("{e:#}");
            monitor.finish(&Err(e));
            Err(anyhow!(message))
        }
    };
    let _ = emitter.join();
    let summary = result?;
    let mut value = serde_json::to_value(&summary)?;
    value["out"] = json!(out);
    Ok(value)
}

/// Read a MIDI to its end for what it holds, which only a full read can say.
fn scan(job: ScanJob, id: &Value, cancel: &AtomicBool, state: &Shared) -> Result<Value> {
    let interval = Duration::from_millis(lock(state).interval_ms);
    let mut s = MidiStream::open(&job.path)?;
    let bytes_total = s.bytes_total();
    let rate = job.rate as f64;
    let mut clock = TempoClock::new(s.division, job.rate);

    let (mut notes, mut note_offs, mut controllers, mut programs, mut bends, mut tempos, mut other) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut second, mut in_second, mut peak, mut peak_at) = (0u64, 0u64, 0u64, 0u64);
    let mut last_tick = 0u64;
    let mut read = 0u64;
    let mut last_report = Instant::now();
    let mut cancelled = false;

    while let Some((tick, event)) = s.next() {
        last_tick = tick;
        match event {
            MidiEvent::NoteOn { .. } => {
                notes += 1;
                let at = (clock.frame_at(tick) / rate) as u64;
                if at != second {
                    if in_second > peak {
                        (peak, peak_at) = (in_second, second);
                    }
                    (second, in_second) = (at, 0);
                }
                in_second += 1;
            }
            MidiEvent::NoteOff { .. } => note_offs += 1,
            MidiEvent::Cc { .. } => controllers += 1,
            MidiEvent::Program { .. } => programs += 1,
            MidiEvent::PitchBend { .. } => bends += 1,
            MidiEvent::Tempo(us) => {
                clock.set_tempo(tick, us);
                tempos += 1;
            }
            _ => other += 1,
        }
        read += 1;
        if read & 0xFFFF == 0 {
            if cancel.load(Ordering::Relaxed) {
                cancelled = true;
                break;
            }
            if !interval.is_zero() && last_report.elapsed() >= interval {
                let done = s.bytes_read();
                send(&json!({
                    "type": "scan_progress",
                    "id": id,
                    "bytes_read": done,
                    "bytes_total": bytes_total,
                    "progress": if bytes_total > 0 { done as f64 / bytes_total as f64 } else { 0.0 },
                    "notes": notes,
                }));
                last_report = Instant::now();
            }
        }
    }
    if in_second > peak {
        (peak, peak_at) = (in_second, second);
    }

    Ok(json!({
        "path": job.path,
        "cancelled": cancelled,
        "tracks": s.track_count,
        "format": s.format,
        "division": division_value(s.division),
        "bytes_total": bytes_total,
        "duration_secs": clock.frame_at(last_tick) / rate,
        "notes": notes,
        "note_offs": note_offs,
        "controllers": controllers,
        "program_changes": programs,
        "pitch_bends": bends,
        "tempo_changes": tempos,
        "other_events": other,
        "peak_notes_per_second": peak,
        "peak_second_at": peak_at,
    }))
}

// ---- Quick requests -------------------------------------------------------

fn adapters() -> Result<Value> {
    let (list, default) = kestrel::gpu::survey(&Config::default())?;
    let adapters: Vec<Value> = list
        .iter()
        .enumerate()
        .map(|(i, a)| {
            json!({
                "index": i,
                "name": a.name,
                "backend": crate::tui::backend_flag(a.backend),
                "type": crate::tui::kind_name(a.device_type),
                "max_voices": a.max_voices,
                "software": a.is_software(),
                "default": Some(i) == default,
            })
        })
        .collect();
    Ok(json!({"adapters": adapters}))
}

/// The guided renderer's update check, on the person's update ring from \[19\]
fn check_update() -> Result<Value> {
    if crate::update::opted_out() {
        bail!("update checks are turned off by {}", crate::update::OPT_OUT);
    }
    let ring = crate::settings::load().0.ring;
    let latest = crate::update::check().context("couldn't check for updates")?;
    Ok(json!({
        "current": crate::update::CURRENT,
        "latest": latest.version,
        "newer": latest.newer,
        "ring": ring.name(),
        "announce": latest.announced(ring),
        "url": latest.url,
    }))
}

fn ffmpeg(req: &Map<String, Value>) -> Result<Value> {
    let explicit = match req.get("path") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(PathBuf::from(s)),
        Some(_) => bail!("\"path\" is the path of an ffmpeg executable"),
    };
    let f = match kestrel::ffmpeg::find(explicit.as_deref()) {
        Ok(f) => f,
        // Not finding one is an answer, not a failed request.
        Err(e) => return Ok(json!({"found": false, "error": format!("{e:#}")})),
    };
    let missing = f.missing_encoders()?;
    let containers: Vec<Value> = kestrel::ffmpeg::PRESETS
        .iter()
        .map(|p| {
            json!({
                "ext": p.ext,
                "encoder": p.encoder,
                "available": !missing.iter().any(|m| m.ext == p.ext),
                "lossy": p.lossy,
                "preset": p.note,
            })
        })
        .collect();
    Ok(json!({
        "found": true,
        "path": f.path,
        "source": f.source.describe(),
        "version": f.version,
        "containers": containers,
    }))
}

fn inspect_midi(req: &Map<String, Value>) -> Result<Value> {
    let path = PathBuf::from(str_param(req, "path")?);
    Ok(match checks::check_midi(&path) {
        Verdict::Valid(m) => json!({
            "valid": true,
            "path": m.path,
            "size": m.size,
            "format": m.format,
            "tracks": m.tracks,
            "division": division_value(m.division),
            // [20]
            "warnings": m.notes,
        }),
        Verdict::Invalid { path, reason } => json!({
            "valid": false,
            "path": path,
            "reason": reason,
        }),
    })
}

fn division_value(d: Division) -> Value {
    match d {
        Division::Ppq(ppq) => json!({"ppq": ppq}),
        Division::Smpte {
            fps,
            ticks_per_frame,
        } => json!({"smpte_fps": fps, "ticks_per_frame": ticks_per_frame}),
    }
}

/// One render option, as clap defines it.
struct OptionSpec {
    long: String,
    switch: bool,
    default: Option<String>,
    values: Vec<String>,
    value_name: Option<String>,
    help: String,
}

/// Every option `render` takes that a request passes in `options`, read off \[21\]
fn option_specs() -> Vec<OptionSpec> {
    let cli = Cli::command();
    let render = cli
        .find_subcommand("render")
        .expect("the command line has a render subcommand");
    render
        .get_arguments()
        .filter_map(|a| {
            let long = a.get_long()?.to_string();
            if NOT_OPTIONS.contains(&long.as_str()) {
                return None;
            }
            Some(OptionSpec {
                switch: matches!(a.get_action(), ArgAction::SetTrue),
                default: a
                    .get_default_values()
                    .first()
                    .map(|v| v.to_string_lossy().into_owned()),
                values: a
                    .get_possible_values()
                    .iter()
                    .map(|v| v.get_name().to_string())
                    .collect(),
                value_name: a
                    .get_value_names()
                    .and_then(|names| names.first())
                    .map(|n| n.to_string()),
                help: a
                    .get_long_help()
                    .or_else(|| a.get_help())
                    .map(|h| h.to_string())
                    .unwrap_or_default(),
                long,
            })
        })
        .collect()
}

fn options() -> Value {
    let mut list: Vec<Value> = option_specs()
        .iter()
        .map(|o| {
            json!({
                "key": o.long.replace('-', "_"),
                "flag": format!("--{}", o.long),
                "kind": if o.switch { "switch" } else { "value" },
                "default": if o.switch { json!(false) } else { json!(o.default) },
                "values": o.values,
                "value_name": o.value_name,
                "reloads_soundfonts": !checks::LOAD_NEUTRAL.contains(&format!("--{}", o.long).as_str()),
                "help": o.help,
            })
        })
        .collect();
    list.push(json!({
        "key": "adapter",
        "flag": null,
        "kind": "value",
        "default": null,
        "values": [],
        "value_name": "INDEX",
        "reloads_soundfonts": false,
        "help": "Render on this adapter: an index from the adapters command. Sets --gpu-backend and --gpu-adapter to match it, and refuses a software adapter.",
    }));
    json!({"options": list})
}

// ---- Options to arguments -------------------------------------------------

/// Turn `options` into the arguments the command line would have been given. \[22\]
fn option_flags(options: Option<&Value>) -> Result<Vec<String>> {
    let map = match options {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Object(map)) => map,
        Some(_) => bail!("\"options\" is an object of option names to values"),
    };
    let specs = option_specs();
    let mut flags = Vec::new();
    for (key, value) in map {
        let name = key.trim_start_matches('-').replace('_', "-");
        if name == "adapter" {
            flags.extend(adapter_flags(value)?);
            continue;
        }
        let Some(spec) = specs.iter().find(|s| s.long == name) else {
            if NOT_OPTIONS.contains(&name.as_str()) || name == "midi" {
                bail!("{key:?} is not an option here: pass it as its own request field, or leave it to the API");
            }
            bail!("unknown option {key:?}; the options command lists every one");
        };
        if spec.switch {
            match value {
                Value::Bool(true) => flags.push(format!("--{name}")),
                Value::Bool(false) | Value::Null => {}
                _ => bail!("option {key:?} is a switch: true or false"),
            }
            continue;
        }
        let text = match value {
            Value::Null => continue,
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            // [23]
            Value::Number(n) => match n.as_f64() {
                Some(f) if !n.is_i64() && !n.is_u64() && f.fract() == 0.0 && f.abs() < 9.0e15 => {
                    format!("{}", f as i64)
                }
                _ => n.to_string(),
            },
            _ => bail!("option {key:?} takes a string or a number"),
        };
        flags.push(format!("--{name}={text}"));
    }
    Ok(flags)
}

fn adapter_flags(value: &Value) -> Result<Vec<String>> {
    let index = value
        .as_u64()
        .context("\"adapter\" is an index from the adapters command")? as usize;
    let (list, _) = kestrel::gpu::survey(&Config::default())?;
    let a = list
        .get(index)
        .with_context(|| format!("no adapter {index}; the adapters command lists {}", list.len()))?;
    if a.is_software() {
        bail!("adapter {index} is a software renderer, which Kestrel never renders on");
    }
    Ok(vec![
        format!("--gpu-backend={}", crate::tui::backend_flag(a.backend)),
        format!("--gpu-adapter={}", a.name),
    ])
}

/// The option arguments that decide how a soundfont loads, in a fixed order.
fn load_flags(flags: &[String]) -> Vec<String> {
    let mut kept: Vec<String> = flags
        .iter()
        .filter(|f| {
            let name = f.split('=').next().unwrap_or(f);
            !checks::LOAD_NEUTRAL.contains(&name)
        })
        .cloned()
        .collect();
    kept.sort();
    kept
}

fn render_argv(
    midi: &std::path::Path,
    soundfonts: &[PathBuf],
    sf_programs: Option<&str>,
    out: &std::path::Path,
    flags: &[String],
) -> Vec<OsString> {
    let joined = |flag: &str, value: &std::ffi::OsStr| {
        let mut arg = OsString::from(flag);
        arg.push(value);
        arg
    };
    let mut argv: Vec<OsString> = vec!["kestrel".into(), "render".into()];
    for font in soundfonts {
        argv.push(joined("--soundfont=", font.as_os_str()));
    }
    if let Some(p) = sf_programs {
        argv.push(format!("--sf-programs={p}").into());
    }
    argv.push(joined("--out=", out.as_os_str()));
    argv.extend(flags.iter().map(OsString::from));
    // After `--`, so a MIDI whose name starts with a dash is still a path.
    argv.push("--".into());
    argv.push(midi.as_os_str().to_owned());
    argv
}

fn parse_render(argv: Vec<OsString>) -> Result<RenderArgs> {
    match Cli::try_parse_from(argv) {
        Ok(cli) => match cli.cmd {
            Cmd::Render(args) => Ok(args),
            _ => unreachable!("the argument list names the render subcommand"),
        },
        Err(e) => {
            // [24]
            let text = e.render().to_string();
            let kept: Vec<&str> = text
                .lines()
                .take_while(|l| !l.trim_start().starts_with("Usage:"))
                .map(str::trim_end)
                .filter(|l| !l.is_empty())
                .collect();
            Err(anyhow!(kept.join("\n").trim_start_matches("error: ").to_string()))
        }
    }
}

// ---- Controls -------------------------------------------------------------

fn cancel(id: &Value, state: &Shared) {
    let st = lock(state);
    match &st.running {
        Some(r) if r.cmd == "load_soundfonts" => {
            respond_err(id, "a soundfont load cannot be stopped part way; it answers when it is done")
        }
        Some(r) => {
            r.cancel.store(true, Ordering::Relaxed);
            if let Some(m) = &r.monitor {
                m.cancel();
            }
            if let Some(c) = &r.control {
                let _ = c.send(FeedCommand::Cancelled);
            }
            respond_ok(id, json!({"cancelling": r.id, "cmd": r.cmd}));
        }
        None => respond_err(id, "nothing is running"),
    }
}

fn set_interval(id: &Value, req: &Map<String, Value>, state: &Shared) {
    let Some(ms) = req.get("interval_ms").and_then(Value::as_u64) else {
        respond_err(id, "\"interval_ms\" is a whole number of milliseconds; 0 stops periodic progress");
        return;
    };
    let ms = feed::clamp_interval(ms);
    let mut st = lock(state);
    st.interval_ms = ms;
    if let Some(c) = st.running.as_ref().and_then(|r| r.control.as_ref()) {
        let _ = c.send(FeedCommand::Interval(ms));
    }
    respond_ok(id, json!({"interval_ms": ms}));
}

fn snapshot(id: &Value, state: &Shared) {
    let monitor = lock(state)
        .running
        .as_ref()
        .and_then(|r| r.monitor.clone());
    match monitor {
        Some(m) => respond_ok(id, feed::progress_value(&m.snapshot())),
        None => respond_err(id, "no render is running"),
    }
}

fn status(id: &Value, state: &Shared) {
    let st = lock(state);
    respond_ok(
        id,
        json!({
            "running": st.running.as_ref().map(|r| json!({"id": r.id, "cmd": r.cmd})),
            "loaded": st.loaded.as_ref().map(|l| l.info.clone()),
            "interval_ms": st.interval_ms,
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_render_option_but_the_named_ones_is_offered() {
        let specs = option_specs();
        let names: Vec<&str> = specs.iter().map(|s| s.long.as_str()).collect();
        for expected in ["max-voices", "volume", "limiter", "note-grid", "ffmpeg", "seconds"] {
            assert!(names.contains(&expected), "{expected} missing from {names:?}");
        }
        // A developer option is offered exactly when the command line takes it.
        assert_eq!(names.contains(&"no-lfo"), cfg!(feature = "dev"), "{names:?}");
        for excluded in NOT_OPTIONS {
            assert!(!names.contains(excluded), "{excluded} offered");
        }
        let voices = specs.iter().find(|s| s.long == "max-voices").unwrap();
        assert_eq!(voices.default.as_deref(), Some("1048576"));
        assert!(!voices.switch);
        assert!(specs.iter().find(|s| s.long == "note-grid").unwrap().switch);
        let backend = specs.iter().find(|s| s.long == "backend").unwrap();
        assert_eq!(backend.values, ["cpu", "gpu"]);
    }

    #[test]
    fn options_become_the_arguments_the_command_line_takes() {
        let flags = option_flags(Some(&json!({
            "max_voices": 4194304,
            "--ceiling-db": -1.5,
            "note-grid": true,
            "nan_guard": false,
            "limiter": "omni",
            "volume": 80.0,
            "seconds": null,
        })))
        .unwrap();
        let mut sorted = flags.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            [
                "--ceiling-db=-1.5",
                "--limiter=omni",
                "--max-voices=4194304",
                "--note-grid",
                "--volume=80",
            ]
        );
        let args = parse_render(render_argv(
            "-dash.mid".as_ref(),
            &["a.sf2".into()],
            Some("0-7"),
            "o.wav".as_ref(),
            &flags,
        ))
        .unwrap();
        assert_eq!(args.max_voices, 4194304);
        assert_eq!(args.midi, PathBuf::from("-dash.mid"));
        assert_eq!(args.ceiling_db, Some(-1.5));
        assert!(args.note_grid && !args.nan_guard);
    }

    #[test]
    fn a_bad_option_is_refused_with_a_reason() {
        for (options, says) in [
            (json!({"max_voice": 1}), "unknown option"),
            (json!({"note_grid": "yes"}), "switch"),
            (json!({"out": "x.wav"}), "not an option here"),
            (json!({"limiter": [1]}), "string or a number"),
            (json!([1, 2]), "object"),
        ] {
            let e = option_flags(Some(&options)).unwrap_err().to_string();
            assert!(e.contains(says), "{options}: {e}");
        }
        let parsed = parse_render(render_argv(
            "a.mid".as_ref(),
            &["a.sf2".into()],
            None,
            "o.wav".as_ref(),
            &["--limiter=loud".into()],
        ));
        let Err(e) = parsed else {
            panic!("--limiter=loud parsed");
        };
        let e = e.to_string();
        assert!(e.contains("loud") && !e.contains("Usage"), "{e}");
    }

    #[test]
    fn only_options_a_loader_reads_decide_a_reload() {
        let flags: Vec<String> = ["--steal-percent=50", "--volume=50", "--max-voices=10", "--rate=44100"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(load_flags(&flags), ["--rate=44100"]);
    }

    #[test]
    fn phase_controls_reuse_the_raw_bank_and_reach_render_configuration() {
        let flags = option_flags(Some(&json!({"phase_mode": "analytic", "phase_seed": 42,
            "phase_continuous": true, "phase_preserve_attack_ms": 5.0}))).unwrap();
        assert!(load_flags(&flags).is_empty());
        let argv = render_argv(PLACEHOLDER_MIDI.as_ref(), &[PathBuf::from("bank.sf2")],
            None, PLACEHOLDER_OUT.as_ref(), &flags);
        let (cfg, _) = parse_render(argv).unwrap().to_config().unwrap();
        assert!(cfg.phase.active() && cfg.phase.continuous);
        assert_eq!(cfg.phase.seed, 42);
        assert_eq!(cfg.phase.preserve_attack_ms, 5.0);
    }
}
