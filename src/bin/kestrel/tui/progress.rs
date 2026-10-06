// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The progress screen: one reader of a `session::Monitor`. \[1\]

use super::style::{self, b, c, s, Line, AMBER, DIM, WARN};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::tty::IsTty;
use crossterm::{cursor, execute, queue, terminal};
use kestrel::bank::Bank;
use kestrel::session::{self, Job, Monitor, Observer as _, Phase, Plan, Snapshot, Summary};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Eight frames a second: smooth enough for the bar, slow enough that a \[2\]
const FRAME: Duration = Duration::from_millis(125);
/// How long a first Ctrl+C waits for the second. One keypress should not throw \[3\]
const CONFIRM: Duration = Duration::from_secs(3);
/// The widest the panel's content goes in a wide terminal: a normal render's, \[4\]
const WIDEST: usize = 76;
const WIDEST_PER_TRACK: usize = 100;
/// The footer after a first Ctrl+C, for a render that saves its progress. It sits \[5\]
const ARMED_SAVING: &str = "Ctrl+C: stop & save \u{00B7} Ctrl+D: discard";

/// The fixed text of the screen, settled before the render starts.
#[derive(Clone)]
pub struct Labels {
    pub midi: String,
    pub output: String,
    pub format: String,
    pub fonts: Vec<String>,
    pub max_voices: u32,
    /// A per-track render, which gets `track_frame`: its tracks render many at \[6\]
    pub per_track: bool,
    /// Set for each job of a batch: the frame then opens with a row saying \[7\]
    pub batch: Option<BatchRow>,
}

/// Recent speed and note rate, smoothed over a couple of seconds so the \[8\]
#[derive(Default)]
struct Rates {
    prev: Option<(f64, f64, u64)>,
    speed: Option<f64>,
    notes: f64,
}

impl Rates {
    fn update(&mut self, snap: &Snapshot) {
        if snap.render_secs <= 0.0 {
            return;
        }
        let Some((then, audio, notes)) = self.prev else {
            self.prev = Some((snap.wall_secs, snap.audio_done(), snap.notes));
            return;
        };
        let dt = snap.wall_secs - then;
        if dt < 0.5 {
            return;
        }
        let speed = (snap.audio_done() - audio) / dt;
        let rate = snap.notes.saturating_sub(notes) as f64 / dt;
        let k = 1.0 - (-dt / 2.0).exp();
        match self.speed {
            None => {
                self.speed = Some(speed);
                self.notes = rate;
            }
            Some(old) => {
                self.speed = Some(old + (speed - old) * k);
                self.notes += (rate - self.notes) * k;
            }
        }
        self.prev = Some((snap.wall_secs, snap.audio_done(), snap.notes));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Footer {
    Hint,
    Armed,
    /// Armed, for a render that saves its progress: a stop can keep it or not.
    ArmedSaving,
    Cancelling,
    Hidden,
}

/// What a key asks of a render that is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    /// Stop, and keep the progress for a resume.
    Keep,
    /// Stop, and keep none of it: no resume file.
    Discard,
}

/// The stop a key asks for, if it asks for one. Either takes a second press to \[9\]
fn stop_key(key: &crossterm::event::KeyEvent, saving: bool) -> Option<Stop> {
    if key.kind != KeyEventKind::Press {
        return None;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc => Some(Stop::Keep),
        KeyCode::Char(ch) if ctrl && ch.eq_ignore_ascii_case(&'c') => Some(Stop::Keep),
        KeyCode::Char(ch) if ctrl && saving && ch.eq_ignore_ascii_case(&'d') => Some(Stop::Discard),
        _ => None,
    }
}

fn spinner(tick: u64) -> &'static str {
    const FRAMES: [&str; 4] = [
        "\u{25CF}\u{25CB}\u{25CB}",
        "\u{25CB}\u{25CF}\u{25CB}",
        "\u{25CB}\u{25CB}\u{25CF}",
        "\u{25CB}\u{25CF}\u{25CB}",
    ];
    FRAMES[(tick % 4) as usize]
}

/// Two labelled values side by side, each column `col` wide.
fn pair(col: usize, l1: &str, v1: Line, l2: &str, v2: Line) -> Line {
    let mut left = vec![c(format!("{l1:<10}"), DIM)];
    left.extend(v1);
    let mut line = style::fit(&left, col);
    line.push(s(" "));
    line.push(c(format!("{l2:<10}"), DIM));
    line.extend(v2);
    line
}

fn dash() -> Line {
    vec![c("\u{2014}", DIM)]
}

/// Video and host memory, side by side.
fn memory(snap: &Snapshot, col: usize) -> Line {
    let mut vram = match (snap.device_bytes, snap.backend.as_deref()) {
        (Some(bytes), _) => vec![s(style::bytes(bytes))],
        (None, Some("cpu")) => vec![c("none, CPU backend", DIM)],
        _ => dash(),
    };
    // [10]
    if let Some(m) = snap.gpu_memory {
        if let Some(used) = m.dedicated_used {
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            vram.push(c(
                format!("  GPU {:.1}/{:.1} GiB", gib(used), gib(m.dedicated_total)),
                DIM,
            ));
        }
    }
    let ram = match (snap.host_rss_bytes, snap.host_rss_peak_bytes) {
        (Some(now), Some(peak)) => {
            vec![s(style::bytes(now)), c(format!("  peak {}", style::bytes(peak)), DIM)]
        }
        (Some(now), None) => vec![s(style::bytes(now))],
        _ => dash(),
    };
    pair(col, "VRAM", vram, "RAM", ram)
}

/// The spinner row a frame shows before rendering starts.
fn setting_up(phase: Phase, tick: u64, secs: f64) -> Line {
    let what = match phase {
        Phase::LoadingSoundfont => "Loading the soundfont",
        Phase::OpeningMidi => "Opening the MIDI",
        Phase::PreparingDevice => "Preparing the device",
        Phase::Failed => "Stopping",
        _ => "Starting",
    };
    vec![
        c(spinner(tick), AMBER),
        s(format!("  {what}\u{2026}")),
        c(format!("  {}", style::clock(secs)), DIM),
    ]
}

/// How a frame is drawn, as distinct from what it shows.
struct View {
    /// Frames drawn so far, for the spinner.
    tick: u64,
    /// Content width inside the panel.
    inner: usize,
    footer: Footer,
}

/// Lay out one frame. Pure, so a frame can be checked as text.
fn frame(labels: &Labels, snap: &Snapshot, rates: &Rates, view: &View) -> Vec<Line> {
    if labels.per_track {
        return track_frame(labels, snap, rates, view);
    }
    let View {
        tick,
        inner,
        footer,
    } = *view;
    let label = |t: &str| c(format!("{t:<10}"), DIM);
    let mut body: Vec<Line> = Vec::new();
    if let Some(row) = &labels.batch {
        let mut line = vec![
            label("Batch"),
            b(format!("job {} of {}", row.job, row.jobs), AMBER),
            c(format!("  \u{00B7}  set {} of {}", row.set, row.sets), DIM),
            c(format!("  \u{00B7}  {} done", row.done), DIM),
        ];
        if row.failed > 0 {
            line.push(c(format!(", {} failed", row.failed), WARN));
        }
        body.push(line);
    }
    body.extend([
        vec![label("MIDI"), s(style::middle(&labels.midi, inner - 10))],
        vec![label("Output"), s(labels.output.clone()), c(format!("  {}", labels.format), DIM)],
        vec![label("Fonts"), s(labels.fonts.join("  +  "))],
        Line::new(),
    ]);

    let rendering = snap.render_secs > 0.0;
    match snap.phase {
        Phase::Rendering | Phase::Finishing | Phase::Finished | Phase::Cancelled => {
            let finishing = matches!(snap.phase, Phase::Finishing | Phase::Finished);
            let frac = if finishing {
                1.0
            } else {
                snap.progress().unwrap_or(0.0)
            };
            let pct = format!("{:>5.1}%", frac * 100.0);
            let mut bar = style::bar(frac, inner.saturating_sub(pct.len() + 2));
            bar.push(s("  "));
            bar.push(b(pct, AMBER));
            body.push(bar);

            // [11]
            let middle = if finishing {
                vec![c(format!("Closing the file {}", spinner(tick)), AMBER)]
            } else if frac >= 1.0 {
                vec![c(format!("Rendering release tails {}", spinner(tick)), AMBER)]
            } else if let Some(eta) = snap.eta_secs() {
                vec![c("Remaining ", DIM), s(format!("~{}", style::clock(eta)))]
            } else {
                vec![c("Remaining ", DIM), c("estimating", DIM)]
            };
            let mut timing = vec![
                c("Elapsed ", DIM),
                s(style::clock(snap.render_secs)),
                s("    "),
            ];
            timing.extend(middle);
            timing.push(c("    Audio ", DIM));
            timing.push(s(style::audio_clock(snap.audio_secs)));
            body.push(timing);
        }
        phase => {
            body.push(setting_up(phase, tick, snap.phase_secs));
            body.push(Line::new());
        }
    }
    body.push(Line::new());

    let col = (inner - 1) / 2;
    let n = |v: u64| if rendering { vec![s(style::thousands(v))] } else { dash() };

    let speed = match snap.speed() {
        Some(overall) => {
            let mut v = vec![b(format!("{overall:.2}\u{00D7}"), AMBER), s(" realtime")];
            if let Some(recent) = rates.speed {
                v.push(c(format!("  now {recent:.1}\u{00D7}"), DIM));
            }
            v
        }
        None => dash(),
    };
    let mut notes = n(snap.notes);
    if rendering && rates.speed.is_some() {
        notes.push(c(format!("  {}/s", style::compact(rates.notes)), DIM));
    }
    body.push(pair(col, "Speed", speed, "Notes", notes));

    let mut voices = n(snap.voices);
    if rendering {
        voices.push(c(format!(" of {}", style::thousands(labels.max_voices as u64)), DIM));
    }
    body.push(pair(col, "Voices", voices, "Peak", n(snap.peak_voices)));
    body.push(pair(col, "Stolen", n(snap.stolen), "Dropped", n(snap.dropped)));

    body.push(memory(snap, col));

    let level = if rendering {
        vec![s(format!("{:.3}", snap.peak_level)), c("  before the limiter", DIM)]
    } else {
        dash()
    };
    let tracks = if snap.tracks > 0 {
        vec![s(style::thousands(snap.tracks as u64))]
    } else {
        dash()
    };
    body.push(pair(col, "Mix peak", level, "Tracks", tracks));

    let device = match (&snap.adapter, snap.backend.as_deref()) {
        (Some(name), _) => vec![s(name.clone())],
        (None, Some("cpu")) => vec![s("CPU reference renderer")],
        _ => dash(),
    };
    let mut row = vec![label("Device")];
    row.extend(device);
    body.push(row);

    let title = match snap.phase {
        Phase::Rendering => vec![b("Rendering", AMBER)],
        Phase::Finishing | Phase::Finished => vec![b("Finishing", AMBER)],
        _ => vec![b("Setting up", AMBER)],
    };
    let footer = match footer {
        Footer::Hint => Some(vec![c("Ctrl+C to cancel", DIM)]),
        Footer::Armed => Some(vec![b("press Ctrl+C again to cancel", WARN)]),
        Footer::ArmedSaving => Some(vec![b(ARMED_SAVING, WARN)]),
        Footer::Cancelling => Some(vec![b("stopping after this block\u{2026}", WARN)]),
        Footer::Hidden => None,
    };
    style::panel(title, &body, inner, footer)
}

/// A per-track render's frame: its tracks, many at once, rather than one \[12\]
fn track_frame(labels: &Labels, snap: &Snapshot, rates: &Rates, view: &View) -> Vec<Line> {
    let View {
        tick,
        inner,
        footer,
    } = *view;
    let label = |t: &str| c(format!("{t:<10}"), DIM);
    let mut body: Vec<Line> = vec![
        vec![label("MIDI"), s(style::middle(&labels.midi, inner - 10))],
        vec![label("Output"), s(labels.output.clone()), c(format!("  {}", labels.format), DIM)],
        vec![label("Fonts"), s(labels.fonts.join("  +  "))],
        Line::new(),
    ];

    let p = snap.per_track.as_ref();
    let active = matches!(snap.phase, Phase::Rendering | Phase::Finishing | Phase::Finished | Phase::Cancelled);
    match p.filter(|_| active) {
        Some(p) => {
            let finishing = matches!(snap.phase, Phase::Finishing | Phase::Finished);
            let frac = if finishing { 1.0 } else { snap.progress().unwrap_or(0.0) };
            let pct = format!("{:>5.1}%", frac * 100.0);
            let mut bar = style::bar(frac, inner.saturating_sub(pct.len() + 2));
            bar.push(s("  "));
            bar.push(b(pct, AMBER));
            body.push(bar);

            let middle = if finishing && p.merged {
                vec![c(format!("Limiting and writing the mix {}", spinner(tick)), AMBER)]
            } else if finishing {
                vec![c(format!("Closing the files {}", spinner(tick)), AMBER)]
            } else if frac >= 1.0 {
                vec![c(format!("Rendering release tails {}", spinner(tick)), AMBER)]
            } else if let Some(eta) = snap.eta_secs() {
                vec![c("Remaining ", DIM), s(format!("~{}", style::clock(eta)))]
            } else {
                vec![c("Remaining ", DIM), c("estimating", DIM)]
            };
            let mut timing = vec![c("Elapsed ", DIM), s(style::clock(snap.render_secs)), s("    ")];
            timing.extend(middle);
            timing.push(c("    Tracks ", DIM));
            timing.push(s(style::thousands(p.done as u64)));
            timing.push(c(format!(" of {}", style::thousands(p.total as u64)), DIM));
            body.push(timing);
        }
        None => {
            body.push(setting_up(snap.phase, tick, snap.phase_secs));
            body.push(Line::new());
        }
    }
    body.push(Line::new());

    let col = (inner - 1) / 2;
    let n = |v: u64| match p {
        Some(_) => vec![s(style::thousands(v))],
        None => dash(),
    };

    let speed = match snap.speed().filter(|_| p.is_some()) {
        Some(overall) => {
            let mut v = vec![b(format!("{overall:.2}\u{00D7}"), AMBER), s(" realtime")];
            if let Some(recent) = rates.speed {
                v.push(c(format!("  now {recent:.1}\u{00D7}"), DIM));
            }
            v
        }
        None => dash(),
    };
    let mut notes = n(snap.notes);
    if p.is_some() && rates.speed.is_some() {
        notes.push(c(format!("  {}/s", style::compact(rates.notes)), DIM));
    }
    body.push(pair(col, "Speed", speed, "Notes", notes));

    let (at_once, silent, stolen) = match p {
        Some(p) => (
            vec![
                s(style::thousands(p.running as u64)),
                c(" tracks, ", DIM),
                s(style::thousands(p.voices_each as u64)),
                c(" voices each", DIM),
            ],
            if p.blocks > 0 {
                vec![
                    s(format!("{:.0}%", p.silent_blocks as f64 * 100.0 / p.blocks as f64)),
                    c(" of blocks, skipped", DIM),
                ]
            } else {
                dash()
            },
            vec![
                s(style::thousands(snap.stolen)),
                c(format!(" in {} track{}", style::thousands(p.stole as u64), if p.stole == 1 { "" } else { "s" }), DIM),
            ],
        ),
        None => (dash(), dash(), dash()),
    };
    body.push(pair(col, "At once", at_once, "Silent", silent));
    body.push(pair(col, "Stolen", stolen, "Dropped", n(snap.dropped)));
    body.push(memory(snap, col));

    let level = match p {
        Some(_) => vec![s(format!("{:.3}", snap.peak_level)), c("  loudest track, before any limiter", DIM)],
        None => dash(),
    };
    let mut row = vec![label("Peak")];
    row.extend(level);
    body.push(row);

    let device = match (&snap.adapter, snap.backend.as_deref()) {
        (Some(name), _) => vec![s(name.clone())],
        (None, Some("cpu")) => vec![s("CPU reference renderer")],
        _ => dash(),
    };
    let mut row = vec![label("Device")];
    row.extend(device);
    body.push(row);

    // The busiest tracks still going, which are what the end waits on.
    let mut now = vec![label("Now")];
    match p.filter(|p| !p.now.is_empty()) {
        Some(p) => {
            for (i, t) in p.now.iter().enumerate() {
                if i > 0 {
                    now.push(c("  \u{00B7}  ", DIM));
                }
                now.push(s(format!("#{}", style::thousands(t.track as u64))));
                if let Some(name) = &t.name {
                    now.push(s(format!(" {}", style::middle(name, 16))));
                }
                now.push(c(format!(" {}", style::audio_clock(t.secs)), DIM));
            }
        }
        None => now.extend(dash()),
    }
    body.push(style::fit(&now, inner));

    let title = match snap.phase {
        Phase::Rendering => vec![b("Rendering tracks", AMBER)],
        Phase::Finishing | Phase::Finished => vec![b("Finishing", AMBER)],
        _ => vec![b("Setting up", AMBER)],
    };
    let footer = match footer {
        Footer::Hint => Some(vec![c("Ctrl+C to cancel", DIM)]),
        Footer::Armed => Some(vec![b("press Ctrl+C again to cancel", WARN)]),
        Footer::ArmedSaving => Some(vec![b(ARMED_SAVING, WARN)]),
        Footer::Cancelling => Some(vec![b("stopping the tracks in flight\u{2026}", WARN)]),
        Footer::Hidden => None,
    };
    style::panel(title, &body, inner, footer)
}

/// The terminal while the screen owns it: raw mode so Ctrl+C arrives as a key \[13\]
struct Screen {
    raw: bool,
    size: (u16, u16),
    top: u16,
    fresh: bool,
    rows: u16,
    /// The widest the panel's content goes.
    widest: usize,
}

impl Screen {
    /// `widest` is how wide the panel's content may grow in a wide terminal.
    fn open(interactive: bool, widest: usize) -> Self {
        let raw = interactive && terminal::enable_raw_mode().is_ok();
        let _ = execute!(std::io::stdout(), cursor::Hide);
        Screen {
            raw,
            size: terminal::size().unwrap_or((100, 30)),
            top: 0,
            fresh: true,
            rows: 0,
            widest,
        }
    }

    fn inner(&self) -> usize {
        (self.size.0 as usize).saturating_sub(8).clamp(40, self.widest)
    }

    fn draw(&mut self, lines: &[Line]) {
        let size = terminal::size().unwrap_or(self.size);
        if size != self.size {
            self.size = size;
            self.fresh = true;
        }
        let mut out = std::io::stdout().lock();
        let _ = queue!(out, terminal::BeginSynchronizedUpdate);
        if self.fresh {
            let _ = queue!(out, terminal::Clear(terminal::ClearType::All), cursor::MoveTo(0, 0));
            let room = self.size.1 >= style::BANNER_ROWS + lines.len() as u16 + 2;
            self.top = 1;
            if room {
                for (i, line) in style::banner().iter().enumerate() {
                    let _ = queue!(out, cursor::MoveTo(0, 1 + i as u16));
                    let _ = out.write_all(style::MARGIN.as_bytes());
                    let _ = style::queue_line(&mut out, line);
                }
                self.top = 1 + style::BANNER_ROWS;
            }
            self.fresh = false;
        }
        for (i, line) in lines.iter().enumerate() {
            let _ = queue!(out, cursor::MoveTo(0, self.top + i as u16));
            let _ = out.write_all(style::MARGIN.as_bytes());
            let _ = style::queue_line(&mut out, line);
            let _ = queue!(out, terminal::Clear(terminal::ClearType::UntilNewLine));
        }
        self.rows = lines.len() as u16;
        let _ = queue!(out, terminal::EndSynchronizedUpdate);
        let _ = out.flush();
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        if self.raw {
            let _ = terminal::disable_raw_mode();
        }
        let _ = execute!(
            std::io::stdout(),
            cursor::MoveTo(0, self.top + self.rows),
            cursor::Show
        );
        println!();
    }
}

/// Render with the progress screen up, and return how it ended.
pub fn run(job: &Job, plan: Plan, bank: Option<Arc<Bank>>, labels: &Labels) -> Result<Summary> {
    let monitor = Monitor::new();
    let interactive = std::io::stdin().is_tty() && std::io::stdout().is_tty();

    std::thread::scope(|scope| {
        let render_monitor = Arc::clone(&monitor);
        let worker =
            scope.spawn(move || session::run_monitored(job, plan, bank, &render_monitor));

        // [14]
        let mut screen = Screen::open(interactive, if labels.per_track { WIDEST_PER_TRACK } else { WIDEST });
        let mut rates = Rates::default();
        let mut armed: Option<Instant> = None;
        let mut tick = 0u64;
        // Whether a stop has progress to keep, and so a choice about it.
        let saving = job.checkpoint.is_some() || job.stems.as_ref().is_some_and(|s| s.resume.is_some());

        loop {
            let finished = worker.is_finished();
            let snap = monitor.snapshot();
            rates.update(&snap);
            let footer = if !interactive {
                Footer::Hidden
            } else if monitor.is_cancelled() {
                Footer::Cancelling
            } else if armed.is_some_and(|t| t.elapsed() < CONFIRM) {
                if saving {
                    Footer::ArmedSaving
                } else {
                    Footer::Armed
                }
            } else {
                Footer::Hint
            };
            let view = View {
                tick,
                inner: screen.inner(),
                footer,
            };
            screen.draw(&frame(labels, &snap, &rates, &view));
            if finished {
                break;
            }
            tick += 1;

            if !interactive {
                std::thread::sleep(FRAME);
                continue;
            }
            // [15]
            if !event::poll(FRAME).unwrap_or(false) {
                continue;
            }
            if let Ok(Event::Key(key)) = event::read() {
                let armed_now = armed.is_some_and(|t| t.elapsed() < CONFIRM);
                match stop_key(&key, saving) {
                    // [16]
                    Some(_) if !armed_now => armed = Some(Instant::now()),
                    Some(Stop::Keep) => monitor.cancel(),
                    Some(Stop::Discard) => monitor.cancel_discarding(),
                    None => {}
                }
            }
        }
        drop(screen);
        worker.join().unwrap_or_else(|p| std::panic::resume_unwind(p))
    })
}

/// Where a batch is, for the row a batch's frames add at the top.
#[derive(Clone, Debug, Default)]
pub struct BatchRow {
    /// The job on the screen, from 1, and how many there are.
    pub job: usize,
    pub jobs: usize,
    /// Its soundfont set, from 1, and how many sets the batch loads. Filled in \[17\]
    pub set: usize,
    pub sets: usize,
    /// Jobs finished, and of them how many failed. Live: filled in per frame.
    pub done: usize,
    pub failed: usize,
}

/// What `run_batch` shares between the render thread and the screen.
struct BatchShared {
    /// The monitor of the job on the screen: each job gets a fresh one, so its \[18\]
    monitor: std::sync::Mutex<Arc<Monitor>>,
    /// That job, from 0; `usize::MAX` before the first.
    current: std::sync::atomic::AtomicUsize,
    cancel: std::sync::atomic::AtomicBool,
    done: std::sync::atomic::AtomicUsize,
    failed: std::sync::atomic::AtomicUsize,
}

impl BatchShared {
    fn monitor(&self) -> Arc<Monitor> {
        Arc::clone(&self.monitor.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

/// The batch's observer: each job's own render reports to a monitor of its own, \[19\]
struct Watch {
    shared: Arc<BatchShared>,
    obs: session::MonitorObserver,
}

impl Watch {
    /// Put job `index` on the screen, on a monitor of its own, unless it \[20\]
    fn show(&mut self, index: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        if self.shared.current.load(Relaxed) == index {
            return;
        }
        let monitor = Monitor::new();
        self.obs = monitor.observer();
        *self.shared.monitor.lock().unwrap_or_else(|p| p.into_inner()) = monitor;
        self.shared.current.store(index, Relaxed);
    }
}

impl session::Observer for Watch {
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
        self.shared.cancel.load(std::sync::atomic::Ordering::Relaxed) || self.obs.cancelled()
    }
}

impl kestrel::batch::BatchObserver for Watch {
    fn job(&mut self) -> &mut dyn session::Observer {
        self
    }

    fn loading(&mut self, _set: usize, _sets: usize, _fonts: &[std::path::PathBuf], first_job: usize, _jobs: usize) {
        self.show(first_job);
        self.obs.phase(Phase::LoadingSoundfont);
    }

    fn started(&mut self, index: usize, _total: usize, _job: &Job) {
        self.show(index);
    }

    fn finished(&mut self, _total: usize, result: &kestrel::batch::JobResult) {
        use kestrel::batch::Outcome;
        use std::sync::atomic::Ordering::Relaxed;
        // [21]
        self.show(result.index);
        let ended = match &result.outcome {
            Outcome::Done(s) => Ok(s.clone()),
            Outcome::Failed(why) => Err(anyhow::anyhow!("{why}")),
            Outcome::NotRun => return,
        };
        if ended.is_err() {
            self.shared.failed.fetch_add(1, Relaxed);
        }
        self.shared.done.fetch_add(1, Relaxed);
        self.shared.monitor().finish(&ended);
    }
}

/// Run a batch with the progress screen up, one job at a time on the screen. \[22\]
pub fn run_batch(plan: kestrel::batch::BatchPlan, mut labels: Vec<Labels>) -> kestrel::batch::BatchSummary {
    use std::sync::atomic::Ordering::Relaxed;
    let interactive = std::io::stdin().is_tty() && std::io::stdout().is_tty();

    let sets = plan.groups().len();
    for (set, group) in plan.groups().iter().enumerate() {
        for &job in group {
            if let Some(row) = labels[job].batch.as_mut() {
                row.set = set + 1;
                row.sets = sets;
            }
        }
    }

    let first = Monitor::new();
    let shared = Arc::new(BatchShared {
        monitor: std::sync::Mutex::new(Arc::clone(&first)),
        current: std::sync::atomic::AtomicUsize::new(usize::MAX),
        cancel: std::sync::atomic::AtomicBool::new(false),
        done: std::sync::atomic::AtomicUsize::new(0),
        failed: std::sync::atomic::AtomicUsize::new(0),
    });

    std::thread::scope(|scope| {
        let mut watch = Watch { shared: Arc::clone(&shared), obs: first.observer() };
        let worker = scope.spawn(move || kestrel::batch::run(plan, &mut watch));

        let mut screen = Screen::open(interactive, WIDEST);
        let mut rates = Rates::default();
        let mut shown = usize::MAX;
        let mut armed: Option<Instant> = None;
        let mut tick = 0u64;

        loop {
            let finished = worker.is_finished();
            let at = shared.current.load(Relaxed);
            if at != shown {
                shown = at;
                rates = Rates::default();
            }
            let monitor = shared.monitor();
            let snap = monitor.snapshot();
            rates.update(&snap);
            let footer = if !interactive {
                Footer::Hidden
            } else if shared.cancel.load(Relaxed) {
                Footer::Cancelling
            } else if armed.is_some_and(|t| t.elapsed() < CONFIRM) {
                Footer::Armed
            } else {
                Footer::Hint
            };
            let view = View { tick, inner: screen.inner(), footer };
            let mut label = labels[at.min(labels.len() - 1)].clone();
            if let Some(row) = label.batch.as_mut() {
                row.done = shared.done.load(Relaxed);
                row.failed = shared.failed.load(Relaxed);
            }
            screen.draw(&frame(&label, &snap, &rates, &view));
            if finished {
                break;
            }
            tick += 1;

            if !interactive {
                std::thread::sleep(FRAME);
                continue;
            }
            if !event::poll(FRAME).unwrap_or(false) {
                continue;
            }
            if let Ok(Event::Key(key)) = event::read() {
                // A batch saves no progress, so there is one kind of stop.
                let armed_now = armed.is_some_and(|t| t.elapsed() < CONFIRM);
                if stop_key(&key, false).is_some() {
                    if armed_now {
                        shared.cancel.store(true, Relaxed);
                        monitor.cancel();
                    } else {
                        armed = Some(Instant::now());
                    }
                }
            }
        }
        drop(screen);
        worker.join().unwrap_or_else(|p| std::panic::resume_unwind(p))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Labels {
        Labels {
            midi: "東方 very long black MIDI title that will not fit the panel at all.mid".into(),
            output: "song.opus".into(),
            format: "Opus \u{00B7} libopus VBR 160 kbps".into(),
            fonts: vec!["gm.sf2".into(), "piano.sfz".into()],
            max_voices: 1 << 20,
            per_track: false,
            batch: None,
        }
    }

    fn per_track(snap: &mut Snapshot) {
        snap.per_track = matches!(snap.phase, Phase::Rendering | Phase::Finishing).then(|| {
            kestrel::session::TrackProgress {
                total: 50_000,
                done: 1_234,
                running: 256,
                stole: 12,
                blocks: 400_000,
                silent_blocks: 348_000,
                blocks_total: 1_000_000,
                span_blocks: 30_000,
                span_total: 100_000,
                notes: 50_000,
                notes_total: 100_000,
                length_secs: 240.0,
                voices_each: 18,
                merged: true,
                audio_secs: 102.1,
                now: vec![
                    kestrel::session::TrackNow { track: 19, name: Some("Arts".into()), secs: 83.0 },
                    kestrel::session::TrackNow { track: 12, name: None, secs: 45.5 },
                ],
                ..Default::default()
            }
        });
    }

    fn sample(phase: Phase) -> Snapshot {
        let rendering = matches!(phase, Phase::Rendering | Phase::Finishing);
        Snapshot {
            phase,
            phase_secs: 4.0,
            wall_secs: 14.0,
            render_secs: if rendering { 4.0 } else { 0.0 },
            audio_secs: 102.1,
            resumed_secs: 0.0,
            bytes_read: 523,
            bytes_total: 1000,
            blocks: 1200,
            notes: 84_995,
            voices: 800,
            max_voices: 1 << 20,
            peak_voices: 35_004,
            stolen: 0,
            dropped: 0,
            peak_level: 3.919,
            clipped: 0,
            backend: Some("gpu".into()),
            adapter: Some("NVIDIA GeForce RTX 5060 Laptop GPU (Vulkan)".into()),
            tracks: 13,
            device_bytes: Some(1_151_000_000),
            gpu_memory: Some(kestrel::gpu::vram::GpuMemory {
                dedicated_total: 8 << 30,
                dedicated_used: Some(3_650_000_000),
                shared_used: Some(80 << 20),
                process_used: Some(1_200_000_000),
                process_budget: Some(6 << 30),
            }),
            host_rss_bytes: Some(1_330_000_000),
            host_rss_peak_bytes: Some(1_900_000_000),
            per_track: None,
        }
    }

    const PHASES: [Phase; 5] = [
        Phase::Starting,
        Phase::LoadingSoundfont,
        Phase::PreparingDevice,
        Phase::Rendering,
        Phase::Finishing,
    ];

    /// Every phase lays out to the same rectangle, so the screen never leaves \[23\]
    #[test]
    fn every_phase_draws_the_same_rectangle() {
        let mut heights = Vec::new();
        for phase in PHASES {
            let view = View {
                tick: 3,
                inner: 70,
                footer: Footer::Hint,
            };
            let lines = frame(&labels(), &sample(phase), &Rates::default(), &view);
            let widths: Vec<usize> = lines.iter().map(|l| style::width(l)).collect();
            assert!(widths.iter().all(|&w| w == 74), "{phase:?}: {widths:?}");
            heights.push(lines.len());
            if phase == Phase::Rendering {
                let all: Vec<String> = lines.iter().map(|l| style::plain(l)).collect();
                let all = all.join("\n");
                assert!(all.contains(" 52.3%"), "{all}");
                assert!(all.contains("84,995"), "{all}");
                assert!(all.contains("of 1,048,576"), "{all}");
                assert!(all.contains("25.52\u{00D7} realtime"), "{all}");
                assert!(all.contains("1.1 GiB  GPU 3.4/8.0 GiB"), "{all}");
                assert!(all.contains("Ctrl+C to cancel"), "{all}");
            }
        }
        assert!(heights.iter().all(|&h| h == heights[0]), "{heights:?}");
    }

    /// The footer that offers both stops sits in the panel's bottom edge, which has \[24\]
    #[test]
    fn the_footer_that_offers_both_stops_fits_the_narrowest_panel() {
        for inner in [40, 70, WIDEST] {
            for footer in [Footer::Armed, Footer::ArmedSaving] {
                let view = View { tick: 3, inner, footer };
                let mut snap = sample(Phase::Rendering);
                let mut tracks = labels();
                for per in [false, true] {
                    tracks.per_track = per;
                    if per {
                        per_track(&mut snap);
                    }
                    let lines = frame(&tracks, &snap, &Rates::default(), &view);
                    let widths: Vec<usize> = lines.iter().map(|l| style::width(l)).collect();
                    assert!(widths.iter().all(|&w| w == inner + 4), "{inner} {per}: {widths:?}");
                    let text: Vec<String> = lines.iter().map(|l| style::plain(l)).collect();
                    let last = text.last().unwrap();
                    if footer == Footer::ArmedSaving {
                        assert!(last.contains("Ctrl+C: stop & save") && last.contains("Ctrl+D: discard"), "{last}");
                    } else {
                        assert!(last.contains("press Ctrl+C again to cancel"), "{last}");
                    }
                }
            }
        }
    }

    fn key(code: KeyCode, mods: KeyModifiers) -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::new(code, mods)
    }

    /// Which stop a key asks for: Ctrl+C and Esc keep the progress, Ctrl+D does not, \[25\]
    #[test]
    fn a_stop_key_says_whether_to_keep_the_progress() {
        let ctrl = KeyModifiers::CONTROL;
        let c = key(KeyCode::Char('c'), ctrl);
        let d = key(KeyCode::Char('d'), ctrl);
        let esc = key(KeyCode::Esc, KeyModifiers::NONE);
        for saving in [true, false] {
            assert_eq!(stop_key(&c, saving), Some(Stop::Keep));
            assert_eq!(stop_key(&esc, saving), Some(Stop::Keep));
            // Caps Lock, or Shift, makes the letter a capital: the same key.
            assert_eq!(stop_key(&key(KeyCode::Char('C'), ctrl | KeyModifiers::SHIFT), saving), Some(Stop::Keep));
            // A letter alone is not a stop, and an `x` is not either.
            assert_eq!(stop_key(&key(KeyCode::Char('d'), KeyModifiers::NONE), saving), None);
            assert_eq!(stop_key(&key(KeyCode::Char('D'), KeyModifiers::SHIFT), saving), None);
            assert_eq!(stop_key(&key(KeyCode::Char('x'), ctrl), saving), None);
        }
        assert_eq!(stop_key(&d, true), Some(Stop::Discard));
        assert_eq!(stop_key(&key(KeyCode::Char('D'), ctrl), true), Some(Stop::Discard));
        assert_eq!(stop_key(&d, false), None, "nothing to keep, so nothing to discard");
        // Only a press: a release or a repeat of the key is not another one.
        for k in [c, d] {
            let mut release = k;
            release.kind = KeyEventKind::Release;
            assert_eq!(stop_key(&release, true), None);
        }
    }

    /// And so does a per-track render's, whose rows differ from a normal \[26\]
    #[test]
    fn every_phase_of_a_per_track_render_draws_the_same_rectangle() {
        let labels = Labels { per_track: true, ..labels() };
        let mut heights = Vec::new();
        for phase in PHASES {
            let mut snap = sample(phase);
            per_track(&mut snap);
            let view = View { tick: 3, inner: 70, footer: Footer::Hint };
            let lines = frame(&labels, &snap, &Rates::default(), &view);
            let widths: Vec<usize> = lines.iter().map(|l| style::width(l)).collect();
            assert!(widths.iter().all(|&w| w == 74), "{phase:?}: {widths:?}");
            heights.push(lines.len());
            // The words, at the width a per-track render gets.
            let wide = View { inner: WIDEST_PER_TRACK, ..view };
            let all: Vec<String> = frame(&labels, &snap, &Rates::default(), &wide).iter().map(|l| style::plain(l)).collect();
            let all = all.join("\n");
            if phase == Phase::Rendering {
                // [27]
                assert!(all.contains(" 41.0%"), "{all}");
                // [28]
                assert!(all.contains("24.60\u{00D7} realtime"), "{all}");
                assert!(all.contains("Tracks 1,234 of 50,000"), "{all}");
                assert!(all.contains("256 tracks, 18 voices each"), "{all}");
                assert!(all.contains("87% of blocks, skipped"), "{all}");
                assert!(all.contains("in 12 tracks"), "{all}");
                assert!(all.contains("#19 Arts 01:23.0  \u{00B7}  #12 00:45.5"), "{all}");
            }
            if phase == Phase::Finishing {
                assert!(all.contains("Limiting and writing the mix"), "{all}");
            }
        }
        assert!(heights.iter().all(|&h| h == heights[0]), "{heights:?}");
    }

    /// At its own width a per-track render's rows fit whole, with figures the \[29\]
    #[test]
    fn a_per_track_render_fits_its_rows_at_its_width() {
        let labels = Labels { per_track: true, ..labels() };
        let mut snap = sample(Phase::Rendering);
        per_track(&mut snap);
        snap.notes = 270_178_131;
        snap.stolen = 118_181_196;
        snap.dropped = 409_590_826;
        if let Some(p) = snap.per_track.as_mut() {
            p.voices_each = 9_537;
            p.stole = 1_197;
        }
        let rates = Rates { prev: None, speed: Some(740.35), notes: 925_000.0 };
        let view = View { tick: 1, inner: WIDEST_PER_TRACK, footer: Footer::Hint };
        for line in frame(&labels, &snap, &rates, &view).iter().map(|l| style::plain(l)) {
            // The MIDI's name is shortened in the middle on purpose.
            if !line.contains("MIDI") {
                assert!(!line.contains('\u{2026}'), "cut: {line}");
            }
        }
    }

    /// Not a check: prints one frame per phase as plain text, to look at a \[30\]
    #[test]
    #[ignore = "prints frames to look at; asserts nothing"]
    fn preview_frames() {
        let rates = Rates {
            prev: None,
            speed: Some(25.9),
            notes: 21_213.0,
        };
        for per in [false, true] {
            let labels = Labels { per_track: per, ..labels() };
            for phase in [Phase::PreparingDevice, Phase::Rendering, Phase::Finishing] {
                let view = View {
                    tick: 1,
                    inner: if per { WIDEST_PER_TRACK } else { WIDEST },
                    footer: Footer::Hint,
                };
                let mut snap = sample(phase);
                if per {
                    per_track(&mut snap);
                }
                for line in frame(&labels, &snap, &rates, &view) {
                    println!("  {}", style::plain(&line));
                }
                println!();
            }
        }
    }
}
