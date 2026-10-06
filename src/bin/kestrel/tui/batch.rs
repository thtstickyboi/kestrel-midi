// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The guided renderer's batch: several MIDIs, each with the soundfonts it is \[1\]

use super::assign::{step_sets, Assigned};
use super::*;
use kestrel::batch::{self, Entry, Outcome};

/// What `flow` returns for a step that turned the person away: the way out, as \[2\]
macro_rules! go {
    ($e:expr) => {
        match $e {
            Step::Got(v) => v,
            Step::Menu => return Ok(Next::Menu),
            Step::Exit => return Ok(Next::Exit),
        }
    };
}

/// The batch flow, from the soundfonts on: `midis` are picked, and `first` is \[3\]
pub(super) fn flow(
    io: &mut dyn Io,
    env: &Env,
    midis: Vec<MidiInfo>,
    first: Option<Fonts>,
) -> Result<Next> {
    let assigned = go!(step_sets(io, (2, 6), &midis, first));
    let voices = go!(step_voices(io, env, (3, 6), None, None));
    let format = go!(step_format(io, env, (4, 6)));
    let folder = go!(step_folder(io, &midis[0].path, (5, 6), "the folder to write the files into"));

    let Assigned { sets, rows, kept } = assigned;
    let ext = format.ext;
    let mut entries: Vec<Entry> = rows
        .iter()
        .map(|&(m, k)| Entry {
            midi: midis[m].path.clone(),
            soundfonts: sets[k].paths.clone(),
            sf_programs: None,
            out: None,
            seconds: None,
        })
        .collect();

    // [4]
    let named = batch::assign_outputs(&entries, Some(&folder), ext).unwrap_or_else(|_| {
        entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let stem = e.midi.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                folder.join(format!("{stem} ({}).{ext}", i + 1))
            })
            .collect()
    });
    // Never over a file that is there: the single flow does not either.
    let stamp = checks::timestamp();
    let outs: Vec<PathBuf> = named.iter().map(|o| checks::output_path(&folder, o, ext, &stamp)).collect();
    for (e, out) in entries.iter_mut().zip(&outs) {
        e.out = Some(out.clone());
    }

    const SHOWN: usize = 6;
    style::status(
        OK,
        vec![
            s("Writes "),
            b(format!("{} file{}", style::thousands(outs.len() as u64), plural(outs.len())), AMBER),
            s(format!(" from {} soundfont set{}", sets.len(), plural(sets.len()))),
            c(if sets.len() > 1 { ", one set at a time" } else { "" }, DIM),
        ],
    );
    for out in outs.iter().take(SHOWN) {
        style::detail(vec![c(file_name(out), DIM)]);
    }
    if outs.len() > SHOWN {
        style::detail(vec![c(format!("\u{2026} and {} more", style::thousands((outs.len() - SHOWN) as u64)), DIM)]);
    }

    // [5]
    let fonts = FontsRef {
        paths: &sets[rows[0].1].paths,
        bank: match &kept {
            Some((0, f)) if sets.len() == 1 => Some((&f.bank, f.budget)),
            _ => None,
        },
    };
    let mut ready = go!(step_flags(io, env, &midis[0], fonts, voices, &outs[0], (6, 6), &[]));

    let planned = batch::jobs_from(&ready.job, &entries, None, ext).and_then(batch::plan);
    let mut plan = match planned {
        Ok(p) => p,
        Err(e) => {
            let _ = capture::problems();
            style::error(format!("{e:#}"));
            return Ok(what_next(io));
        }
    };
    if let Some(bank) = ready.bank.take() {
        plan.preload(bank);
    }
    drop(kept);

    let format_line = format!("{} \u{00B7} {}", format.label, format_note(ext));
    let set_of: Vec<usize> = rows.iter().map(|&(_, k)| k).collect();
    let labels: Vec<progress::Labels> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| progress::Labels {
            midi: file_name(&e.midi),
            output: e.out.as_deref().map(file_name).unwrap_or_default(),
            format: format_line.clone(),
            fonts: sets[set_of[i]].names.clone(),
            max_voices: ready.plan.cfg.max_voices,
            per_track: false,
            batch: Some(progress::BatchRow { job: i + 1, jobs: entries.len(), ..Default::default() }),
        })
        .collect();

    let _ = capture::problems();
    let summary = progress::run_batch(plan, labels);
    let problems = capture::problems();

    // [6]
    for r in &summary.results {
        if let Some(w) = r.written.as_deref().filter(|w| *w != r.out) {
            let _ = std::fs::remove_file(w);
        }
    }

    clear_screen();
    style::print_banner();
    show_batch_outcome(&folder, sets.len(), ready.adapter.as_deref(), &ready.flags, &summary, &problems);
    Ok(what_next(io))
}

fn show_batch_outcome(
    folder: &Path,
    sets: usize,
    adapter: Option<&str>,
    flags: &[String],
    summary: &batch::BatchSummary,
    problems: &[(log::Level, String)],
) {
    let inner = panel_width();
    let label = |t: &str| c(format!("{t:<10}"), DIM);
    let total = summary.results.len();
    let done = summary.done();
    let failed = summary.failed();

    let (audio, bytes): (f64, u64) = summary
        .results
        .iter()
        .filter(|r| r.is_ok())
        .filter_map(|r| match &r.outcome {
            Outcome::Done(s) => Some((s.audio_secs, s.bytes)),
            _ => None,
        })
        .fold((0.0, 0), |(a, b), (x, y)| (a + x, b + y));

    let title = if summary.cancelled {
        b("Stopped", WARN)
    } else if failed > 0 {
        b("Done, with failures", WARN)
    } else {
        b("Done", OK)
    };

    let mut body: Vec<Line> = Vec::new();
    body.push(vec![
        label("Wrote"),
        b(format!("{} of {} file{}", style::thousands(done as u64), style::thousands(total as u64), plural(total)), AMBER),
    ]);
    body.push(vec![label("Folder"), s(style::middle(&folder.display().to_string(), inner - 10))]);
    if summary.cancelled {
        body.push(vec![c(
            "Files finished before the stop are in the folder; the one in progress was removed.",
            DIM,
        )]);
    }
    body.push(Line::new());
    body.push(vec![
        label("Audio"),
        s(style::audio_clock(audio)),
        c(" over all files  in ", DIM),
        s(format!("{:.1} s", summary.wall_secs)),
        c("  = ", DIM),
        b(format!("{:.2}\u{00D7} realtime", audio / summary.wall_secs.max(1e-9)), AMBER),
    ]);
    body.push(vec![label("Size"), s(style::bytes(bytes))]);
    body.push(vec![
        label("Sets"),
        s(format!("{sets} used")),
    ]);
    body.push(Line::new());

    const SHOWN: usize = 12;
    for r in summary.results.iter().take(SHOWN) {
        let name = file_name(&r.out);
        let room = inner.saturating_sub(4);
        let line = match &r.outcome {
            Outcome::Done(sum) if !sum.cancelled => vec![
                c("\u{2714} ", OK),
                s(style::middle(&name, room.saturating_sub(24))),
                c(format!("  {}  {} notes", style::audio_clock(sum.audio_secs), style::thousands(sum.notes)), DIM),
            ],
            Outcome::Done(_) => vec![c("\u{2013} ", WARN), s(style::middle(&name, room)), c("  stopped", WARN)],
            Outcome::Failed(why) => vec![
                c("\u{2718} ", ERR),
                s(style::middle(&name, room / 2)),
                c(format!("  {}", style::middle(why, room.saturating_sub(room / 2 + 2))), ERR),
            ],
            Outcome::NotRun => vec![c("\u{2013} ", DIM), s(style::middle(&name, room)), c("  not rendered", DIM)],
        };
        body.push(line);
    }
    if total > SHOWN {
        body.push(vec![c(format!("\u{2026} and {} more", style::thousands((total - SHOWN) as u64)), DIM)]);
    }
    body.push(Line::new());
    if let Some(a) = adapter {
        body.push(vec![label("Device"), s(a.to_string())]);
    }
    body.extend(flag_lines(flags, inner));
    if let Some(log) = kestrel::falconeye::renderlog::last_path() {
        body.push(vec![label("Log"), c(log.display().to_string(), DIM)]);
    }
    for line in style::panel(vec![title], &body, inner, None) {
        style::say(line);
    }
    for (level, message) in problems {
        if *level == log::Level::Error {
            style::error(message);
        } else {
            style::warn(message.clone());
        }
    }
}

#[cfg(test)]
mod flow_tests {
    use super::*;
    use std::collections::VecDeque;

    struct Script {
        lines: VecDeque<&'static str>,
        keys: VecDeque<Key>,
        files: VecDeque<Vec<PathBuf>>,
        folders: VecDeque<PathBuf>,
    }

    impl Io for Script {
        fn line(&mut self) -> Option<String> {
            self.lines.pop_front().map(String::from)
        }
        fn key(&mut self) -> Option<Key> {
            self.keys.pop_front()
        }
        fn pick_files(&mut self, _: Pick, _: &str) -> Option<Vec<PathBuf>> {
            self.files.pop_front()
        }
        fn pick_folder(&mut self, _: &str, _: Option<&Path>) -> Option<PathBuf> {
            self.folders.pop_front()
        }
    }

    /// No adapters, no ffmpeg, no update check: what a test can run anywhere.
    fn bare_env() -> Env {
        Env {
            adapters: Vec::new(),
            default_adapter: None,
            gpu_error: None,
            ffmpeg: Err("not looked for".into()),
            ring: Ring::default(),
            update: None,
        }
    }

    fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let p = e.unwrap().path();
                (file_name(&p), std::fs::read(&p).unwrap())
            })
            .collect();
        v.sort();
        v
    }

    struct Material {
        dir: PathBuf,
        simple: PathBuf,
        rich: PathBuf,
        midis: [PathBuf; 2],
    }

    fn material(name: &str) -> Material {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let simple = dir.join("simple.sf2");
        let rich = dir.join("rich.sf2");
        kestrel::testkit::simple_sf2(&simple, 48_000).unwrap();
        kestrel::testkit::rich_sf2(&rich, 48_000).unwrap();
        let a = dir.join("alpha.mid");
        let b = dir.join("beta.mid");
        kestrel::testkit::single_note_midi(&a, 60, 100, 0.4).unwrap();
        kestrel::testkit::scatter_midi(&b, 200, 0.6, 2, 40, 80).unwrap();
        Material { dir, simple, rich, midis: [a, b] }
    }

    /// The command line's render of `midi` on `fonts`, to `out`.
    fn cli(midi: &Path, fonts: &[&Path], out: &Path) {
        let mut argv: Vec<OsString> = vec!["kestrel".into(), "render".into(), midi.into()];
        for f in fonts {
            argv.extend(["-s".into(), f.into()]);
        }
        argv.extend(["-o".into(), out.into()]);
        argv.extend(["--backend", "cpu", "--seconds", "1"].map(OsString::from));
        crate::render::render_cli(parse_render(argv).unwrap()).unwrap();
    }

    const G: Key = Key::Char('g');

    /// The guided renderer's promise, for a batch: a table of rows, each given \[7\]
    #[test]
    fn a_guided_batch_is_byte_identical_to_the_same_commands() {
        let m = material("kestrel_guided_batch");
        let (out, alone) = (m.dir.join("out"), m.dir.join("alone"));
        std::fs::create_dir_all(&out).unwrap();
        std::fs::create_dir_all(&alone).unwrap();
        let mut io = Script {
            lines: [
                "",  // the picked MIDIs
                "2", // a different set for each
                "", "", // the two soundfont picks, confirmed
                "",  // voices, the default
                "1", // WAV
                "--backend cpu --seconds 1",
                "2", // exit after the outcome
            ]
            .into(),
            keys: [
                G,         // turned away: row 1 has nothing yet
                Key::Enter, // row 1: no sets yet, so straight to the picker
                Key::Down, Key::Enter, // row 2: a set exists, so a list is offered
                Key::Down, Key::Enter, // ... and "pick other soundfonts"
                G,
            ]
            .into(),
            files: [m.midis.to_vec(), vec![m.simple.clone()], vec![m.rich.clone()]].into(),
            folders: [out.clone()].into(),
        };
        let next = render_flow(&mut io, &bare_env()).unwrap();
        assert!(next == Next::Exit && io.lines.is_empty() && io.keys.is_empty(), "the flow stopped early: {:?} {:?}", io.lines, io.keys);

        cli(&m.midis[0], &[&m.simple], &alone.join("alpha.wav"));
        cli(&m.midis[1], &[&m.rich], &alone.join("beta.wav"));
        let (a, b) = (files(&out), files(&alone));
        let names: Vec<&str> = a.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["alpha.wav", "beta.wav"], "no .partial. file may be left behind");
        assert!(a[0].1.len() > 1000);
        assert!(a == b, "a guided batch differs from the commands it stands for");
        let _ = std::fs::remove_dir_all(&m.dir);
    }

    /// Space marks rows, and Enter gives all the marked ones one set: a table \[8\]
    #[test]
    fn marked_rows_share_a_set_and_one_midi_on_two_sets_is_a_row_added() {
        let m = material("kestrel_guided_batch_sets");
        let (marked, sets, alone) = (m.dir.join("marked"), m.dir.join("sets"), m.dir.join("alone"));
        for d in [&marked, &sets, &alone] {
            std::fs::create_dir_all(d).unwrap();
        }
        let mut io = Script {
            lines: [
                "", "2", "", // two MIDIs, a set for each, the pick confirmed
                "", "1", "--backend cpu --seconds 1", "1", // voices, WAV, flags, Home
                // then one MIDI: the picked file, then S for more sets
                "", "s", "", // ... and the second set's pick confirmed
                "", "1", "--backend cpu --seconds 1", "2",
            ]
            .into(),
            keys: [
                Key::Space, Key::Space, Key::Enter, G, // both rows, one set
                Key::Char('d'), Key::Enter, Key::Down, Key::Enter, G, // the MIDI again, on the other set
            ]
            .into(),
            files: [
                m.midis.to_vec(), vec![m.simple.clone()],
                vec![m.midis[1].clone()], vec![m.simple.clone()], vec![m.rich.clone()],
            ]
            .into(),
            folders: [marked.clone(), sets.clone()].into(),
        };
        // [9]
        let next = render_flow(&mut io, &bare_env()).unwrap();
        assert!(next == Next::Menu && io.keys.len() == 5, "the first batch did not end on Home: {:?} {:?}", io.lines, io.keys);
        let next = render_flow(&mut io, &bare_env()).unwrap();
        assert!(next == Next::Exit && io.lines.is_empty() && io.keys.is_empty(), "the flow stopped early: {:?} {:?}", io.lines, io.keys);

        cli(&m.midis[0], &[&m.simple], &alone.join("alpha.wav"));
        cli(&m.midis[1], &[&m.simple], &alone.join("beta.wav"));
        assert!(files(&marked) == files(&alone), "two rows on one set differ from the commands");

        let (a, b) = (m.dir.join("beta_simple.wav"), m.dir.join("beta_rich.wav"));
        cli(&m.midis[1], &[&m.simple], &a);
        cli(&m.midis[1], &[&m.rich], &b);
        let got = files(&sets);
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["beta (rich).wav", "beta (simple).wav"]);
        assert_eq!(got[0].1, std::fs::read(&b).unwrap());
        assert_eq!(got[1].1, std::fs::read(&a).unwrap());
        assert_ne!(got[0].1, got[1].1);
        let _ = std::fs::remove_dir_all(&m.dir);
    }

    struct Quiet;
    impl kestrel::session::Observer for Quiet {}

    /// "Resume a render": a per-track merge stopped part way is continued from \[10\]
    #[test]
    fn a_guided_resume_writes_what_an_uninterrupted_render_does() {
        let m = material("kestrel_guided_resume");
        let midi = m.dir.join("tracks.mid");
        {
            let note = |ch: u8, key: u8, at: u64, len: u64| {
                vec![(at, vec![0x90 | ch, key, 100]), (at + len, vec![0x80 | ch, key, 0])]
            };
            let mut w = kestrel::midi::MidiWriter::new(480);
            w.raw_track(vec![(0, vec![0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])]);
            w.raw_track((0..8u64).flat_map(|i| note(0, 50 + i as u8, i * 200, 150)).collect());
            w.raw_track((0..8u64).flat_map(|i| note(1, 60 + i as u8, i * 170, 120)).collect());
            w.save(&midi).unwrap();
        }
        let argv = |out: &Path| -> Vec<OsString> {
            let mut a: Vec<OsString> = vec!["kestrel".into(), "render".into()];
            a.push(joined("--soundfont=", &m.rich));
            a.push(joined("--out=", out));
            a.extend(["--backend=cpu", "--seconds=1.5", "--tracks=all", "--merge", "--"].map(OsString::from));
            a.push(midi.clone().into_os_string());
            a
        };
        let run = |out: &Path, stop: Option<u64>, restore: Option<std::sync::Arc<kestrel::resume::Checkpoint>>| {
            let args = parse_render(argv(out)).unwrap();
            let mut job = args.to_job().unwrap();
            args.with_checkpoints(&mut job, argv(out));
            let spec = job.stems.as_mut().and_then(|s| s.resume.as_mut()).expect("a per-track render has a spec");
            spec.stop_after_blocks = stop;
            spec.restore = restore;
            let plan = kestrel::session::plan(&job).unwrap();
            kestrel::session::run(&job, plan, None, &mut Quiet).unwrap()
        };

        let want = m.dir.join("want.wav");
        run(&want, None, None);
        let out = m.dir.join("got.wav");
        let stopped = run(&out, Some(9), None);
        assert!(stopped.cancelled && !out.exists());
        let ck = kestrel::resume::default_path(&out, &midi, true);
        assert!(ck.exists(), "the stop saved no checkpoint");

        let mut io = Script {
            lines: ["2"].into(), // exit after the outcome
            keys: VecDeque::new(),
            files: [vec![ck.clone()]].into(),
            folders: VecDeque::new(),
        };
        let next = resume_flow(&mut io, &bare_env()).unwrap();
        assert!(next == Next::Exit && io.lines.is_empty(), "the flow stopped early: {:?}", io.lines);
        assert!(std::fs::read(&out).unwrap() == std::fs::read(&want).unwrap(), "the resumed file differs");
        assert!(!ck.exists(), "the checkpoint outlived the render");
        // [11]
        let screen = last_outcome();
        assert!(screen.contains("--backend=cpu --seconds=1.5 --tracks=all --merge"), "{screen}");

        // A file that is not a checkpoint is said so, and asked for again.
        std::fs::write(m.dir.join("not.krsm"), b"plain text, nowhere near a checkpoint, but long enough").unwrap();
        let mut io = Script {
            lines: ["0"].into(), // back to the menu when asked again
            keys: VecDeque::new(),
            files: [vec![m.dir.join("not.krsm")], vec![]].into(),
            folders: VecDeque::new(),
        };
        assert!(resume_flow(&mut io, &bare_env()).unwrap() == Next::Menu);
        let _ = std::fs::remove_dir_all(&m.dir);
    }

    /// The same for a render of one file: it saves the render itself, the guided \[12\]
    #[test]
    fn a_guided_resume_of_a_single_render_writes_what_an_uninterrupted_render_does() {
        let m = material("kestrel_guided_resume_single");
        let midi = m.dir.join("one.mid");
        {
            let note = |ch: u8, key: u8, at: u64, len: u64| {
                vec![(at, vec![0x90 | ch, key, 100]), (at + len, vec![0x80 | ch, key, 0])]
            };
            let mut w = kestrel::midi::MidiWriter::new(480);
            w.raw_track(
                std::iter::once((0, vec![0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]))
                    .chain((0..8u64).flat_map(|i| note(0, 50 + i as u8, i * 200, 150)))
                    .chain((0..8u64).flat_map(|i| note(1, 60 + i as u8, i * 170, 120)))
                    .collect(),
            );
            w.save(&midi).unwrap();
        }
        let argv = |out: &Path| -> Vec<OsString> {
            let mut a: Vec<OsString> = vec!["kestrel".into(), "render".into()];
            a.push(joined("--soundfont=", &m.rich));
            a.push(joined("--out=", out));
            a.extend(["--backend=cpu", "--seconds=1.5", "--"].map(OsString::from));
            a.push(midi.clone().into_os_string());
            a
        };
        let run = |out: &Path, stop: Option<u64>, restore: Option<std::sync::Arc<kestrel::resume::Checkpoint>>| {
            let args = parse_render(argv(out)).unwrap();
            let mut job = args.to_job().unwrap();
            args.with_checkpoints(&mut job, argv(out));
            let spec = job.checkpoint.as_mut().expect("a render of one file has a spec");
            spec.stop_after_blocks = stop;
            spec.restore = restore;
            let plan = kestrel::session::plan(&job).unwrap();
            kestrel::session::run(&job, plan, None, &mut Quiet).unwrap()
        };

        let want = m.dir.join("want.wav");
        run(&want, None, None);
        let out = m.dir.join("got.wav");
        let stopped = run(&out, Some(5), None);
        assert!(stopped.cancelled && !out.exists());
        assert!(kestrel::resume::partial_of(&out).exists(), "the audio so far was not kept");
        let ck = kestrel::resume::default_path(&out, &midi, true);
        assert!(ck.exists(), "the stop saved no checkpoint");

        // What the stop panel says, at every width the screen can be.
        for inner in [40, 76] {
            let panel = outcome_panel(&out, None, &[], &Ok(stopped.clone()), None, Some(ck.clone()), inner)
                .iter()
                .map(|l| style::plain(l))
                .collect::<Vec<_>>()
                .join(" ");
            assert!(panel.contains("audio so far is kept") && panel.contains("Progress saved to"), "{panel}");
            assert!(panel.contains("continue") && !panel.contains("removed"), "{panel}");
        }

        let mut io = Script {
            lines: ["2"].into(), // exit after the outcome
            keys: VecDeque::new(),
            files: [vec![ck.clone()]].into(),
            folders: VecDeque::new(),
        };
        let next = resume_flow(&mut io, &bare_env()).unwrap();
        assert!(next == Next::Exit && io.lines.is_empty(), "the flow stopped early: {:?}", io.lines);
        assert!(std::fs::read(&out).unwrap() == std::fs::read(&want).unwrap(), "the resumed file differs");
        assert!(!ck.exists() && !kestrel::resume::partial_of(&out).exists(), "a whole render left its files");
        let screen = last_outcome();
        assert!(screen.contains("--backend=cpu --seconds=1.5") && !screen.contains("none"), "{screen}");
        let _ = std::fs::remove_dir_all(&m.dir);
    }
}
