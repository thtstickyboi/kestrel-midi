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
use kestrel::session::{self, Job, Monitor, Phase, Plan, Snapshot, Summary};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Eight frames a second: smooth enough for the bar, slow enough that a \[2\]
const FRAME: Duration = Duration::from_millis(125);
/// How long a first Ctrl+C waits for the second. One keypress should not throw \[3\]
const CONFIRM: Duration = Duration::from_secs(3);

/// The fixed text of the screen, settled before the render starts.
pub struct Labels {
    pub midi: String,
    pub output: String,
    pub format: String,
    pub fonts: Vec<String>,
    pub max_voices: u32,
}

/// Recent speed and note rate, smoothed over a couple of seconds so the \[4\]
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
            self.prev = Some((snap.wall_secs, snap.audio_secs, snap.notes));
            return;
        };
        let dt = snap.wall_secs - then;
        if dt < 0.5 {
            return;
        }
        let speed = (snap.audio_secs - audio) / dt;
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
        self.prev = Some((snap.wall_secs, snap.audio_secs, snap.notes));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Footer {
    Hint,
    Armed,
    Cancelling,
    Hidden,
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

            // [5]
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
            let what = match phase {
                Phase::LoadingSoundfont => "Loading the soundfont",
                Phase::OpeningMidi => "Opening the MIDI",
                Phase::PreparingDevice => "Preparing the device",
                Phase::Failed => "Stopping",
                _ => "Starting",
            };
            body.push(vec![
                c(spinner(tick), AMBER),
                s(format!("  {what}\u{2026}")),
                c(format!("  {}", style::clock(snap.phase_secs)), DIM),
            ]);
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

    let mut vram = match (snap.device_bytes, snap.backend.as_deref()) {
        (Some(bytes), _) => vec![s(style::bytes(bytes))],
        (None, Some("cpu")) => vec![c("none, CPU backend", DIM)],
        _ => dash(),
    };
    // [6]
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
    body.push(pair(col, "VRAM", vram, "RAM", ram));

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
        Footer::Cancelling => Some(vec![b("stopping after this block\u{2026}", WARN)]),
        Footer::Hidden => None,
    };
    style::panel(title, &body, inner, footer)
}

/// The terminal while the screen owns it: raw mode so Ctrl+C arrives as a key \[7\]
struct Screen {
    raw: bool,
    size: (u16, u16),
    top: u16,
    fresh: bool,
    rows: u16,
}

impl Screen {
    fn open(interactive: bool) -> Self {
        let raw = interactive && terminal::enable_raw_mode().is_ok();
        let _ = execute!(std::io::stdout(), cursor::Hide);
        Screen {
            raw,
            size: terminal::size().unwrap_or((100, 30)),
            top: 0,
            fresh: true,
            rows: 0,
        }
    }

    fn inner(&self) -> usize {
        (self.size.0 as usize).saturating_sub(8).clamp(40, 76)
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

        let mut screen = Screen::open(interactive);
        let mut rates = Rates::default();
        let mut armed: Option<Instant> = None;
        let mut tick = 0u64;

        loop {
            let finished = worker.is_finished();
            let snap = monitor.snapshot();
            rates.update(&snap);
            let footer = if !interactive {
                Footer::Hidden
            } else if monitor.is_cancelled() {
                Footer::Cancelling
            } else if armed.is_some_and(|t| t.elapsed() < CONFIRM) {
                Footer::Armed
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
            // [8]
            if !event::poll(FRAME).unwrap_or(false) {
                continue;
            }
            if let Ok(Event::Key(key)) = event::read() {
                let stop = key.kind == KeyEventKind::Press
                    && (key.code == KeyCode::Esc
                        || (key.code == KeyCode::Char('c')
                            && key.modifiers.contains(KeyModifiers::CONTROL)));
                if stop {
                    if armed.is_some_and(|t| t.elapsed() < CONFIRM) {
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
        }
    }

    fn sample(phase: Phase) -> Snapshot {
        let rendering = matches!(phase, Phase::Rendering | Phase::Finishing);
        Snapshot {
            phase,
            phase_secs: 4.0,
            wall_secs: 14.0,
            render_secs: if rendering { 4.0 } else { 0.0 },
            audio_secs: 102.1,
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
        }
    }

    const PHASES: [Phase; 5] = [
        Phase::Starting,
        Phase::LoadingSoundfont,
        Phase::PreparingDevice,
        Phase::Rendering,
        Phase::Finishing,
    ];

    /// Every phase lays out to the same rectangle, so the screen never leaves \[9\]
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

    /// Not a check: prints one frame per phase as plain text, to look at a \[10\]
    #[test]
    #[ignore = "prints frames to look at; asserts nothing"]
    fn preview_frames() {
        let rates = Rates {
            prev: None,
            speed: Some(25.9),
            notes: 21_213.0,
        };
        for phase in [Phase::PreparingDevice, Phase::Rendering, Phase::Finishing] {
            let view = View {
                tick: 1,
                inner: 76,
                footer: Footer::Hint,
            };
            for line in frame(&labels(), &sample(phase), &rates, &view) {
                println!("  {}", style::plain(&line));
            }
            println!();
        }
    }
}
