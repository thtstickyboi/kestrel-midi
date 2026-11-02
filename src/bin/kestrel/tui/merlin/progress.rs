// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Merlin's render screen: one reader of a `session::Monitor`, as Kestrel's is, \[1\]

use super::super::progress::{self as shared, memory_line, pair, run_screen, spinner, Footer, View};
use super::super::style::{self, b, c, s, Line, AMBER, DIM, WARN};
use anyhow::Result;
use kestrel::bank::Bank;
use kestrel::merlin::progress::Stage;
use kestrel::session::{Job, Phase, Plan, Snapshot, Summary};
use std::sync::Arc;

/// The fixed text of the screen, settled before the render starts.
#[derive(Clone)]
pub struct Labels {
    pub midi: String,
    pub output: String,
    pub format: String,
    pub fonts: Vec<String>,
    /// The adapter the render is on, with its backend.
    pub adapter: Option<String>,
    /// Its PCI vendor and device: the card's memory is watched by them, since a \[2\]
    pub adapter_ids: Option<(u32, u32)>,
}

/// Render with Merlin's progress screen up, and return how it ended.
pub fn run(job: &Job, plan: Plan, bank: Option<Arc<Bank>>, labels: &Labels) -> Result<Summary> {
    let watch = labels.adapter_ids.map(|(vendor, device)| kestrel::gpu::vram::VramWatch::start(vendor, device));
    run_screen(job, plan, bank, shared::MERLIN_WIDEST, &|snap, _rates, view| {
        let mut snap = snap.clone();
        if snap.gpu_memory.is_none() {
            snap.gpu_memory = watch.as_ref().and_then(|w| w.latest());
        }
        frame(labels, &snap, view)
    })
}

/// What the screen is showing, which the snapshot's phase and its merlin stage \[3\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shown {
    SettingUp,
    Arming,
    Rendering,
    Mastering,
    /// The file is being closed, or is.
    Closing,
    /// Stopped, or failed.
    Stopped,
}

fn shown(snap: &Snapshot) -> Shown {
    match snap.phase {
        Phase::Finishing | Phase::Finished => Shown::Closing,
        Phase::Cancelled | Phase::Failed => Shown::Stopped,
        _ => match snap.merlin.as_ref().map(|m| m.stage) {
            Some(Stage::Arming) => Shown::Arming,
            Some(Stage::Rendering) => Shown::Rendering,
            Some(Stage::Mastering) => Shown::Mastering,
            Some(Stage::Waiting) | None => Shown::SettingUp,
        },
    }
}

/// A count of notes: 1,234,567 reads "1.2 M", and 24,390,649,323 "24.39 G".
fn big(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2} G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1} M", n as f64 / 1e6)
    } else {
        style::thousands(n)
    }
}

/// A labelled bar with its percentage after it.
fn bar_row(label: &str, frac: f64, inner: usize) -> Line {
    let pct = format!("{:>5.1}%", frac.clamp(0.0, 1.0) * 100.0);
    let mut line = vec![c(format!("{label:<10}"), DIM)];
    line.extend(style::bar(frac, inner.saturating_sub(10 + pct.len() + 2)));
    line.push(s("  "));
    line.push(b(pct, AMBER));
    line
}

/// What the spinner row says before the first stage.
fn setting_up(phase: Phase, tick: u64, secs: f64) -> Line {
    let what = match phase {
        Phase::LoadingSoundfont => "Loading the soundfont",
        Phase::OpeningMidi => "Opening the MIDI",
        Phase::PreparingDevice => "Preparing the device",
        _ => "Starting",
    };
    vec![c(spinner(tick), AMBER), s(format!("  {what}\u{2026}")), c(format!("  {}", style::clock(secs)), DIM)]
}

/// Elapsed, and what is left when there is a figure to say it from.
fn timing(snap: &Snapshot, remaining: bool) -> Line {
    let mut line = vec![c("Elapsed ", DIM), s(style::clock(snap.render_secs))];
    if remaining {
        line.push(s("    "));
        line.push(c("Remaining ", DIM));
        match snap.eta_secs() {
            Some(eta) => line.push(s(format!("~{}", style::clock(eta)))),
            None => line.push(c("estimating", DIM)),
        }
    }
    line
}

/// Lay out one frame. Pure, so a frame can be checked as text.
fn frame(labels: &Labels, snap: &Snapshot, view: &View) -> Vec<Line> {
    let View { tick, inner, footer } = *view;
    let col = (inner - 1) / 2;
    let label = |t: &str| c(format!("{t:<10}"), DIM);
    let stage = shown(snap);
    let m = snap.merlin.clone().unwrap_or_default();
    let mut body: Vec<Line> = vec![
        vec![label("MIDI"), s(style::middle(&labels.midi, inner - 10))],
        vec![label("Output"), s(labels.output.clone()), c(format!("  {}", labels.format), DIM)],
        vec![label("Fonts"), s(labels.fonts.join("  +  "))],
        Line::new(),
    ];
    match stage {
        Shown::SettingUp => {
            body.push(setting_up(snap.phase, tick, snap.phase_secs));
            body.push(Line::new());
        }
        Shown::Arming => {
            body.push(bar_row("Reading", m.progress(snap.blocks).unwrap_or(0.0), inner));
            body.push(timing(snap, false));
            body.push(Line::new());
            body.push(pair(
                col,
                "Read",
                vec![s(style::bytes(m.pre_bytes)), c(format!(" of {}", style::bytes(m.pre_total)), DIM)],
                "Notes",
                vec![s(big(m.pre_notes)), c(" counted", DIM)],
            ));
        }
        Shown::Rendering => {
            body.push(bar_row("Overall", m.overall, inner));
            let (key, frac) = match m.key {
                Some(k) => (format!("Key {k:03}"), m.key_frac),
                None => ("Key".to_string(), 0.0),
            };
            body.push(bar_row(&key, frac, inner));
            body.push(timing(snap, true));
            body.push(Line::new());
            body.push(pair(
                col,
                "Keys",
                vec![s(format!("{} of {}", m.keys_done, m.keys_total))],
                "Notes",
                vec![s(big(m.notes_done)), c(format!(" of {}", big(m.notes_total)), DIM)],
            ));
            let busy = m.device_busy.map_or_else(|| vec![c("\u{2014}", DIM)], |f| vec![s(format!("{:.0}%", 100.0 * f))]);
            let store = m.store.map_or_else(|| vec![c("none", DIM)], |(bytes, _)| vec![s(style::bytes(bytes))]);
            body.push(pair(col, "GPU busy", busy, "Store", store));
        }
        Shown::Mastering => {
            body.push(bar_row("Writing", m.progress(snap.blocks).unwrap_or(0.0), inner));
            body.push(timing(snap, false));
            body.push(Line::new());
        }
        Shown::Closing => {
            body.push(vec![c(format!("Closing the file {}", spinner(tick)), AMBER)]);
            body.push(timing(snap, false));
            body.push(Line::new());
        }
        Shown::Stopped => {
            body.push(vec![c(format!("Stopping {}", spinner(tick)), WARN)]);
            body.push(timing(snap, false));
            body.push(Line::new());
        }
    }
    body.push(Line::new());
    body.push(memory_line(snap, col));
    if let Some(a) = &labels.adapter {
        body.push(vec![label("Device"), s(a.clone())]);
    }

    let title = match stage {
        Shown::SettingUp => vec![b("Setting up", AMBER)],
        Shown::Arming => vec![b("Arming\u{2026}", AMBER)],
        Shown::Rendering => vec![b("Rendering", AMBER)],
        Shown::Mastering => vec![b("Mastering", AMBER)],
        Shown::Closing => vec![b("Finishing", AMBER)],
        Shown::Stopped => vec![b("Stopping", WARN)],
    };
    // Never the choice Kestrel offers a render that saves its progress.
    let footer = match footer {
        Footer::Hint => Some(vec![c("Ctrl+C to cancel", DIM)]),
        Footer::Armed | Footer::ArmedSaving => Some(vec![b("press Ctrl+C again to cancel", WARN)]),
        Footer::Cancelling => Some(vec![b("stopping\u{2026}", WARN)]),
        Footer::Hidden => None,
    };
    style::panel(title, &body, inner, footer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel::merlin::progress::MerlinSnapshot;

    fn labels() -> Labels {
        Labels {
            midi: "song.mid".into(),
            output: "song.opus".into(),
            format: "Opus \u{00B7} 256 kbit/s".into(),
            fonts: vec!["piano.sfz".into()],
            adapter: Some("Test GPU (Vulkan)".into()),
            adapter_ids: None,
        }
    }

    fn at(stage: Stage, phase: Phase, tweak: impl FnOnce(&mut MerlinSnapshot, &mut Snapshot)) -> Snapshot {
        let mut m = MerlinSnapshot { stage, ..MerlinSnapshot::default() };
        let mut snap = Snapshot { phase, render_secs: 75.0, phase_secs: 4.0, ..Snapshot::default() };
        tweak(&mut m, &mut snap);
        snap.merlin = (stage != Stage::Waiting).then_some(m);
        snap
    }

    fn text(snap: &Snapshot, inner: usize, footer: Footer) -> Vec<String> {
        frame(&labels(), snap, &View { tick: 1, inner, footer }).iter().map(|l| style::plain(l)).collect()
    }

    fn arming() -> Snapshot {
        at(Stage::Arming, Phase::Arming, |m, _| {
            (m.pre_bytes, m.pre_total, m.pre_notes) = (62_000_000_000, 100_000_000_000, 8_900_000_000);
        })
    }

    fn rendering() -> Snapshot {
        at(Stage::Rendering, Phase::Rendering, |m, _| {
            (m.keys_total, m.keys_done) = (128, 70);
            (m.notes_total, m.notes_done) = (24_390_649_323, 7_620_000_000);
            (m.key, m.key_notes, m.key_frac, m.overall) = (Some(69), 190_000_000, 0.58, 0.312);
            m.device_busy = Some(0.78);
            m.store = Some((111_400_000_000, 42_792_527_518));
        })
    }

    /// Every stage is a rectangle, from the widest the panel goes to the \[4\]
    #[test]
    fn every_stage_is_a_rectangle_at_every_width() {
        let stages = [
            at(Stage::Waiting, Phase::LoadingSoundfont, |_, _| {}),
            arming(),
            rendering(),
            at(Stage::Mastering, Phase::Rendering, |m, s| {
                m.master_blocks = 200;
                s.blocks = 50;
            }),
            at(Stage::Mastering, Phase::Finishing, |_, _| {}),
            at(Stage::Rendering, Phase::Cancelled, |_, _| {}),
        ];
        for snap in &stages {
            for inner in [40, 60, 76] {
                for footer in [Footer::Hint, Footer::Armed, Footer::Cancelling, Footer::Hidden] {
                    let widths: Vec<usize> = frame(&labels(), snap, &View { tick: 1, inner, footer }).iter().map(|l| style::width(l)).collect();
                    assert!(widths.iter().all(|&w| w == widths[0]), "{:?} at {inner}: {widths:?}", shown(snap));
                }
            }
        }
    }

    /// The snapshot's phase and its merlin stage settle what is shown, and a \[5\]
    #[test]
    fn the_stage_shown_follows_the_phase_and_the_merlin_stage() {
        assert_eq!(shown(&at(Stage::Waiting, Phase::LoadingSoundfont, |_, _| {})), Shown::SettingUp);
        assert_eq!(shown(&Snapshot::default()), Shown::SettingUp, "nothing has begun");
        assert_eq!(shown(&arming()), Shown::Arming);
        assert_eq!(shown(&rendering()), Shown::Rendering);
        assert_eq!(shown(&at(Stage::Mastering, Phase::Rendering, |_, _| {})), Shown::Mastering);
        assert_eq!(shown(&at(Stage::Mastering, Phase::Finishing, |_, _| {})), Shown::Closing);
        assert_eq!(shown(&at(Stage::Mastering, Phase::Finished, |_, _| {})), Shown::Closing);
        assert_eq!(shown(&at(Stage::Arming, Phase::Cancelled, |_, _| {})), Shown::Stopped);
        assert_eq!(shown(&at(Stage::Rendering, Phase::Failed, |_, _| {})), Shown::Stopped);
    }

    /// Arming is named so, with a bar of the bytes read.
    #[test]
    fn arming_shows_the_pre_pass_as_a_bar_of_bytes() {
        let t = text(&arming(), 76, Footer::Hint).join("\n");
        assert!(t.contains("Arming\u{2026}"), "{t}");
        assert!(t.contains("Reading") && t.contains("62.0%"), "{t}");
        assert!(t.contains("57.7 GiB of 93.1 GiB") && t.contains("8.90 G counted"), "{t}");
        assert!(!t.contains("Overall") && !t.contains("Key "), "the keys are not on the screen yet: {t}");
        assert!(t.contains("Ctrl+C to cancel"));
    }

    /// Rendering has two bars, the whole and the key in hand, each its own figure.
    #[test]
    fn rendering_has_two_bars_the_whole_render_and_the_key_in_hand() {
        let rows = text(&rendering(), 76, Footer::Hint);
        let t = rows.join("\n");
        let overall = rows.iter().find(|r| r.contains("Overall")).expect("an overall bar");
        let key = rows.iter().find(|r| r.contains("Key 069")).expect("a key bar");
        assert!(overall.contains("31.2%") && key.contains("58.0%"), "{overall} / {key}");
        assert!(rows.iter().position(|r| r == overall) < rows.iter().position(|r| r == key), "the whole above the part");
        assert!(overall.contains('\u{2588}') && key.contains('\u{2588}'));
        assert!(t.contains("70 of 128") && t.contains("7.62 G") && t.contains("of 24.39 G"), "{t}");
        assert!(t.contains("GPU busy  78%") && t.contains("103.7 GiB"), "{t}");
        assert!(t.contains("Test GPU (Vulkan)"), "the adapter is named");
        // Not the numbers of a voice renderer, which a merlin render has none of.
        for gone in ["Voices", "Speed", "Stolen", "Dropped", "Mix peak", "realtime"] {
            assert!(!t.contains(gone), "{gone} on Merlin's screen: {t}");
        }
    }

    /// The estimate says nothing until it has something to say it from.
    #[test]
    fn the_remaining_time_is_estimating_until_there_is_a_figure() {
        let none = text(&rendering(), 76, Footer::Hint).join("\n");
        assert!(none.contains("Remaining estimating"), "{none}");
        let mut snap = rendering();
        let m = snap.merlin.as_mut().unwrap();
        (m.keys_done, m.overall, m.stage_secs) = (20, 0.25, 60.0);
        assert_eq!(snap.eta_secs(), Some(180.0));
        let some = text(&snap, 76, Footer::Hint).join("\n");
        assert!(some.contains("Remaining ~03:00") && some.contains("Elapsed 01:15"), "{some}");
    }

    /// Between keys there is no key to name, and the bar is empty, not a stale one.
    #[test]
    fn between_keys_the_key_bar_is_empty() {
        let mut snap = rendering();
        snap.merlin.as_mut().unwrap().key = None;
        let rows = text(&snap, 76, Footer::Hint);
        let key = rows.iter().find(|r| r.contains("Key       ")).expect("a key row");
        assert!(key.contains("0.0%") && !key.contains('\u{2588}'), "{key}");
    }

    #[test]
    fn mastering_is_a_bar_of_the_blocks_handed_out() {
        let snap = at(Stage::Mastering, Phase::Rendering, |m, s| {
            m.master_blocks = 200;
            s.blocks = 50;
        });
        let t = text(&snap, 76, Footer::Hint).join("\n");
        assert!(t.contains("Mastering") && t.contains("Writing") && t.contains("25.0%"), "{t}");
        let t = text(&at(Stage::Mastering, Phase::Finishing, |_, _| {}), 76, Footer::Hint).join("\n");
        assert!(t.contains("Finishing") && t.contains("Closing the file"), "{t}");
    }

    /// Ctrl+C takes a second press, says so, and never offers to save.
    #[test]
    fn the_footer_asks_twice_and_never_offers_to_save() {
        let snap = rendering();
        let last = |footer| text(&snap, 76, footer).pop().unwrap();
        assert!(last(Footer::Hint).contains("Ctrl+C to cancel"));
        assert!(last(Footer::Armed).contains("press Ctrl+C again to cancel"));
        assert!(last(Footer::ArmedSaving).contains("press Ctrl+C again to cancel"));
        assert!(last(Footer::Cancelling).contains("stopping"));
        for f in [Footer::Hint, Footer::Armed, Footer::ArmedSaving, Footer::Cancelling] {
            let l = last(f);
            assert!(!l.contains("save") && !l.contains("discard") && !l.contains("Ctrl+D"), "{l}");
        }
        assert!(!text(&snap, 76, Footer::Hidden).pop().unwrap().contains("Ctrl+C"));
    }

    /// Not a test: every stage of the screen, as text, to look at. \[6\]
    #[test]
    #[ignore]
    fn print_the_stages() {
        let mut eta = rendering();
        let m = eta.merlin.as_mut().unwrap();
        (m.keys_done, m.overall, m.stage_secs) = (70, 0.312, 2100.0);
        let stages = [
            ("setting up", at(Stage::Waiting, Phase::LoadingSoundfont, |_, _| {}), Footer::Hint),
            ("arming", arming(), Footer::Hint),
            ("rendering, estimating", rendering(), Footer::Hint),
            ("rendering, with an estimate, Ctrl+C pressed once", eta, Footer::Armed),
            (
                "mastering",
                at(Stage::Mastering, Phase::Rendering, |m, s| {
                    m.master_blocks = 200;
                    s.blocks = 50;
                }),
                Footer::Hint,
            ),
            ("cancelling", rendering(), Footer::Cancelling),
        ];
        for (name, snap, footer) in stages {
            println!("--- {name}");
            for row in text(&snap, 76, footer) {
                println!("  {row}");
            }
        }
    }

    #[test]
    fn counts_read_the_way_people_write_them() {
        assert_eq!(big(999), "999");
        assert_eq!(big(1_234_567), "1.2 M");
        assert_eq!(big(24_390_649_323), "24.39 G");
    }
}
