// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Merlin's home and the steps of a render in it. \[1\]

use super::super::checks::{self, MidiInfo};
use super::super::style::{self, b, c, s, AMBER, DIM, WARN};
use super::super::{
    capture, check_environment, clear_screen, compact_env, file_name, format_note, option, pick_font_set, prompt,
    step_flags, step_folder, step_format, step_midis, step_title, Env, Fonts, Io, Native, Next, Step,
};
use anyhow::Result;
use kestrel::config::Config;
use kestrel::merlin::keystore::{self, Site};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// What a step gives, or the way out of the render it asked for.
macro_rules! step {
    ($e:expr) => {
        match $e {
            Step::Got(v) => v,
            Step::Menu => return Ok(Next::Menu),
            Step::Exit => return Ok(Next::Exit),
        }
    };
}

/// Steps in a render, as Kestrel's are numbered.
const OF: u8 = 6;

// ---- what Merlin will not take --------------------------------------------

/// A typed flag that means nothing to Merlin, or that it will not do, with why. \[2\]
pub fn refused(tokens: &[String]) -> Option<String> {
    for token in tokens {
        let name = token.split('=').next().unwrap_or(token);
        let why = match name {
            "--max-voices" | "--steal-percent" | "--steal" | "--admit" => "Merlin has no voice pool, so this means nothing to it",
            "--backend" => "this is Merlin; Kestrel's own renderer is on its menu",
            "--track" | "--tracks" | "--merge" | "--stem-format" | "--setup-tracks" | "--track-jobs" => {
                "Merlin renders a whole file; per-track rendering is in Kestrel"
            }
            "--checkpoint-every" | "--no-resume" => "a Merlin render cannot be resumed, so it saves no progress",
            "--31edo" => "Merlin does not render 31-EDO files yet; Kestrel does",
            "--min-velocity" => "Merlin does not skip quiet notes yet",
            "--merlin-time-major" => "the older Merlin is not this one",
            _ => continue,
        };
        return Some(format!("{name}: {why}"));
    }
    None
}

/// Why Merlin will not take this MIDI, if it will not. `limit` is Merlin's \[3\]
pub fn midi_refusal(info: &MidiInfo, limit: u64) -> Option<String> {
    if info.extended_keys {
        return Some(format!(
            "{} is written for the 31-EDO template (note keys over 127), which Merlin does not render yet. Kestrel does.",
            file_name(&info.path)
        ));
    }
    kestrel::session::merlin_takes_size(&info.path, info.size, limit).err().map(|e| format!("{e:#}"))
}

// ---- the disk --------------------------------------------------------------

/// What the key store asks of the disk for a file of `size` bytes, and what the \[4\]
fn disk_lines(size: u64, need: u64, floor: u64, temp: &Path, temp_free: Option<u64>) -> Vec<style::Line> {
    let label = |t: &str| c(format!("{t:<11}"), DIM);
    let mut lines = vec![
        vec![label("File"), s(style::bytes(size))],
        vec![label("Key store"), s(format!("about {}", style::bytes(need))), c("  filed beside the render and removed when it ends", DIM)],
    ];
    let free = match temp_free {
        Some(free) => vec![s(style::bytes(free)), c(format!(" free in {}", temp.display()), DIM)],
        None => vec![c("not known", DIM), c(format!(" for {}", temp.display()), DIM)],
    };
    lines.push([vec![label("Disk")], free].concat());
    if keystore::site(need, floor, temp_free, None) != Site::Temp {
        lines.push(vec![
            label(""),
            c(
                "Not enough there: the store goes in the folder you choose in step 5 if it has room, and the file is read the slow way if not.",
                WARN,
            ),
        ]);
    }
    lines
}

/// What step 5 says of where the key store will go, once the folder is chosen: \[5\]
fn site_line(site: Site, need: u64, folder: &Path) -> Option<style::Line> {
    match site {
        Site::Temp => None,
        Site::Output => Some(vec![
            c("The key store goes in ", DIM),
            s(folder.display().to_string()),
            c(format!(": the temporary folder has not the {} it wants.", style::bytes(need)), DIM),
        ]),
        Site::Neither => Some(vec![
            b("WARN ", WARN),
            c(
                format!(
                    "Neither the temporary folder nor this one has room for the key store (about {}). Merlin reads the file a group of keys \
                     at a time instead, which is much slower on a big file.",
                    style::bytes(need)
                ),
                WARN,
            ),
        ]),
    }
}

// ---- the steps -------------------------------------------------------------

/// One MIDI that Merlin takes.
fn step_merlin_midi(io: &mut dyn Io, at: (u8, u8), limit: u64) -> Step<MidiInfo> {
    loop {
        let mut midis = match step_midis(io, at, false) {
            Step::Got(m) => m,
            Step::Menu => return Step::Menu,
            Step::Exit => return Step::Exit,
        };
        let info = midis.remove(0);
        match midi_refusal(&info, limit) {
            None => {
                style::detail(vec![c(
                    format!(
                        "Merlin takes files under {:.0} GB and up to 25 billion notes; the notes are counted while it arms.",
                        limit as f64 / 1e9
                    ),
                    DIM,
                )]);
                return Step::Got(info);
            }
            Some(why) => {
                style::error(why);
                style::say(vec![c("Choose another.", DIM)]);
            }
        }
    }
}

/// One soundfont. Kestrel layers two; whether Merlin does has not been tried.
fn step_merlin_font(io: &mut dyn Io, at: (u8, u8)) -> Step<Fonts> {
    style::heading(&step_title(at, "Soundfont"), "one");
    loop {
        match pick_font_set(io, None) {
            Step::Got((fonts, _)) if fonts.paths.len() > 1 => {
                style::warn("Merlin renders one soundfont for now. Choose again.");
            }
            Step::Got((fonts, _)) => return Step::Got(fonts),
            Step::Menu => return Step::Menu,
            Step::Exit => return Step::Exit,
        }
    }
}

/// Merlin's third step, in place of the voices: what the render will ask of the \[6\]
fn step_before(io: &mut dyn Io, at: (u8, u8), midi: &MidiInfo, cfg: &Config) -> Step<()> {
    style::heading(&step_title(at, "Before you start"), "what this render asks of the disk");
    let temp = std::env::temp_dir();
    let need = keystore::store_bytes(midi.size);
    for line in disk_lines(midi.size, need, cfg.merlin_disk_floor, &temp, fs4::available_space(&temp).ok()) {
        style::say(line);
    }
    style::say(vec![c("[Enter]", AMBER), c(" go on   ", DIM), c("[0]", AMBER), c(" back to Merlin's home", DIM)]);
    loop {
        prompt();
        let Some(line) = io.line() else { return Step::Exit };
        match line.trim().to_ascii_lowercase().as_str() {
            "" => return Step::Got(()),
            "0" | "q" => return Step::Menu,
            _ => style::error("Press Enter to go on, or 0 for Merlin's home."),
        }
    }
}

/// The folder to write into, and where the key store will go from there.
fn step_merlin_folder(io: &mut dyn Io, midi: &MidiInfo, at: (u8, u8), cfg: &Config) -> Step<PathBuf> {
    loop {
        let folder = match step_folder(io, &midi.path, at, "the folder to write into") {
            Step::Got(f) => f,
            Step::Menu => return Step::Menu,
            Step::Exit => return Step::Exit,
        };
        let need = keystore::store_bytes(midi.size);
        let site = keystore::site(
            need,
            cfg.merlin_disk_floor,
            fs4::available_space(std::env::temp_dir()).ok(),
            fs4::available_space(&folder).ok(),
        );
        if let Some(line) = site_line(site, need, &folder) {
            style::say(line);
        }
        if site != Site::Neither {
            return Step::Got(folder);
        }
        style::say(vec![
            c("[Enter]", AMBER),
            c(" go on the slow way   ", DIM),
            c("[C]", AMBER),
            c(" choose another folder", DIM),
        ]);
        loop {
            prompt();
            let Some(line) = io.line() else { return Step::Exit };
            match line.trim().to_ascii_lowercase().as_str() {
                "" => return Step::Got(folder),
                "c" => break,
                _ => style::error("Press Enter to go on, or C to choose another folder."),
            }
        }
    }
}

// ---- home ------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Home {
    Start,
    Kestrel,
    Quit,
}

fn home_answer(line: &str) -> Option<Home> {
    match line.trim() {
        "1" => Some(Home::Start),
        "2" => Some(Home::Kestrel),
        "0" | "q" | "Q" => Some(Home::Quit),
        _ => None,
    }
}

/// After a render: Merlin's home, or out.
fn what_next(io: &mut dyn Io) -> Next {
    style::heading("What next?", "");
    option("1", "Home", "back to Merlin's home page");
    option("2", "Quit", "");
    loop {
        prompt();
        let Some(line) = io.line() else { return Next::Exit };
        match line.trim() {
            "1" => return Next::Menu,
            "2" | "0" | "q" | "Q" => return Next::Exit,
            _ => style::error("Type 1 or 2."),
        }
    }
}

/// One render, from the MIDI to the outcome, and then home or out.
fn render(io: &mut dyn Io, env: &Env) -> Result<Next> {
    if env.default_adapter().is_none() {
        style::error("Merlin renders on the GPU, and no usable adapter was found.");
        style::say(vec![c("Press Enter to go back.", DIM)]);
        let _ = io.line();
        return Ok(Next::Menu);
    }
    clear_screen();
    style::print_banner();
    compact_env(env);
    let cfg = Config::default();

    style::heading(&step_title((1, OF), "MIDI"), "the file to render");
    let midi = step!(step_merlin_midi(io, (1, OF), cfg.merlin_max_file_bytes));
    let fonts = step!(step_merlin_font(io, (2, OF)));
    step!(step_before(io, (3, OF), &midi, &cfg));
    let format = step!(step_format(io, env, (4, OF)));
    let folder = step!(step_merlin_folder(io, &midi, (5, OF), &cfg));
    let out = checks::output_path(&folder, &midi.path, format.ext, &checks::timestamp());
    style::detail(vec![c("Writes ", DIM), b(file_name(&out), AMBER)]);
    let base = [OsString::from("--backend=merlin")];
    let ready = step!(step_flags(io, env, &midi, (&fonts).into(), 0, &out, (OF, OF), &base, true));
    drop(fonts.bank);

    let labels = super::progress::Labels {
        midi: file_name(&midi.path),
        output: file_name(&out),
        format: format!("{} \u{00B7} {}", format.label, format_note(format.ext)),
        fonts: fonts.names.clone(),
        adapter: ready.adapter.clone(),
        adapter_ids: ready.adapter_ids,
    };
    let _ = capture::problems();
    let result = super::progress::run(&ready.job, ready.plan, ready.bank, &labels);
    let problems = capture::problems();
    // [7]
    let result = match result {
        Ok(summary) => {
            if summary.cancelled {
                let _ = std::fs::remove_file(&out);
            }
            Ok(summary)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&out);
            Err(format!("{e:#}"))
        }
    };
    clear_screen();
    style::print_banner();
    super::end::show(&out, ready.adapter.as_deref(), &ready.flags, &result, &problems);
    Ok(what_next(io))
}

/// Merlin's home, until Quit.
fn home(io: &mut dyn Io, env: &Env) -> Result<()> {
    loop {
        style::heading("What would you like to do?", "");
        option("1", "Start a Render", "render a MIDI with Merlin");
        option("2", "Back to Kestrel", "");
        let answer = loop {
            prompt();
            let Some(line) = io.line() else { return Ok(()) };
            match home_answer(&line) {
                Some(a) => break a,
                None => style::error("Type 1 or 2."),
            }
        };
        match answer {
            Home::Start => {
                if render(io, env)? == Next::Exit {
                    return Ok(());
                }
            }
            Home::Kestrel => match super::back_to_kestrel() {
                Ok(()) => return Ok(()),
                Err(e) => style::error(format!("Couldn't open a new window ({e}).")),
            },
            Home::Quit => return Ok(()),
        }
        clear_screen();
        style::print_banner();
        compact_env(env);
    }
}

/// The window's whole life, once its theme is set.
pub fn run() -> Result<()> {
    clear_screen();
    style::print_banner();
    let env = check_environment();
    let mut io = Native::remembered();
    home(&mut io, &env)
}

/// What the window says when it fails: in red, and held, since a window opened \[8\]
pub fn failed(io: &mut dyn Io, message: &str) {
    let _ = crossterm::terminal::disable_raw_mode();
    let _ = crossterm::execute!(std::io::stdout(), crossterm::cursor::Show);
    style::blank();
    style::error(message);
    style::say(vec![c("Press Enter to close.", DIM)]);
    let _ = io.line();
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel::midi::Division;

    fn info(size: u64, extended: bool) -> MidiInfo {
        MidiInfo {
            path: PathBuf::from("song.mid"),
            size,
            format: 1,
            tracks: 2,
            division: Division::Ppq(480),
            notes: Vec::new(),
            extended_keys: extended,
        }
    }

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    /// What means nothing to Merlin, or that it will not do, is refused by name, \[9\]
    #[test]
    fn the_flags_that_mean_nothing_to_merlin_are_refused_and_the_rest_are_not() {
        for bad in [
            "--max-voices 100",
            "--max-voices=100",
            "--steal-percent=10",
            "--backend gpu",
            "--tracks all",
            "--track 3",
            "--merge",
            "--checkpoint-every 5",
            "--no-resume",
            "--31edo",
            "--min-velocity 20",
            "--limiter off --max-voices 5",
        ] {
            let why = refused(&words(bad)).unwrap_or_else(|| panic!("{bad} was not refused"));
            assert!(why.starts_with("--") && why.contains(": "), "{why}");
        }
        for fine in ["", "--limiter balanced", "--volume 90", "--dc-blocker", "--format float32 --limiter off"] {
            assert_eq!(refused(&words(fine)), None, "{fine}");
        }
        // The flag that names what is refused, not a value that looks like one.
        assert_eq!(refused(&words("--volume --max-voices")).as_deref().map(|w| w.split(':').next().unwrap()), Some("--max-voices"));
    }

    /// 31-EDO files and files at or past the limit are turned back, by their \[10\]
    #[test]
    fn a_midi_merlin_will_not_take_is_turned_back_with_the_reason() {
        let limit = 200_000_000_000;
        assert_eq!(midi_refusal(&info(1_000_000, false), limit), None);
        assert_eq!(midi_refusal(&info(limit - 1, false), limit), None);
        let size = midi_refusal(&info(limit, false), limit).expect("at the limit");
        assert!(size.contains("takes files under 200.00 GB") && size.contains("0.00 GB past"), "{size}");
        let big = midi_refusal(&info(limit + 5_000_000_000, false), limit).unwrap();
        assert!(big.contains("5.00 GB past the limit"), "{big}");
        let edo = midi_refusal(&info(1_000, true), limit).unwrap();
        assert!(edo.contains("31-EDO") && edo.contains("Kestrel does"), "{edo}");
    }

    /// Home takes 1, 2 and the ways out, and nothing else.
    #[test]
    fn home_answers_to_one_two_and_the_ways_out() {
        assert_eq!(home_answer("1"), Some(Home::Start));
        assert_eq!(home_answer(" 2 "), Some(Home::Kestrel));
        for out in ["0", "q", "Q"] {
            assert_eq!(home_answer(out), Some(Home::Quit));
        }
        for no in ["", "3", "start", "11"] {
            assert_eq!(home_answer(no), None, "{no:?}");
        }
    }

    /// The disk lines say the file, the store and the room, and add the warning \[11\]
    #[test]
    fn the_disk_lines_warn_only_when_the_temporary_folder_is_short() {
        const GB: u64 = 1_000_000_000;
        let text = |free: Option<u64>| -> String {
            disk_lines(100 * GB, 65 * GB, 8 * GB, Path::new("C:/tmp"), free)
                .iter()
                .map(|l| style::plain(l))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let roomy = text(Some(200 * GB));
        assert!(roomy.contains("File") && roomy.contains("Key store") && roomy.contains("free in C:/tmp"), "{roomy}");
        assert!(!roomy.contains("Not enough"), "{roomy}");
        assert!(text(Some(10 * GB)).contains("Not enough there"));
        assert!(text(None).contains("not known") && !text(None).contains("Not enough"), "a folder that cannot say is taken at its word");
    }

    /// Step 5 says nothing when the store is in the temporary folder, where it \[12\]
    #[test]
    fn step_five_says_where_the_store_goes_only_when_it_is_not_the_usual_place() {
        let folder = Path::new("D:/renders");
        assert!(site_line(Site::Temp, 5, folder).is_none());
        let out = style::plain(&site_line(Site::Output, 5_000_000_000, folder).unwrap());
        assert!(out.contains("D:/renders") && out.contains("goes in"), "{out}");
        let none = style::plain(&site_line(Site::Neither, 5_000_000_000, folder).unwrap());
        assert!(none.starts_with("WARN") && none.contains("slow") || none.contains("much slower"), "{none}");
    }

    /// Typed answers, files and folders for a flow, in the order it asks.
    struct Script {
        lines: std::collections::VecDeque<&'static str>,
        files: std::collections::VecDeque<Vec<PathBuf>>,
        folders: std::collections::VecDeque<PathBuf>,
    }

    impl Io for Script {
        fn line(&mut self) -> Option<String> {
            self.lines.pop_front().map(String::from)
        }
        fn pick_files(&mut self, _: super::super::super::Pick, _: &str) -> Option<Vec<PathBuf>> {
            self.files.pop_front()
        }
        fn pick_folder(&mut self, _: &str, _: Option<&Path>) -> Option<PathBuf> {
            self.folders.pop_front()
        }
    }

    /// What the machine has, as the environment check would say it, without \[13\]
    fn this_machine() -> Env {
        let survey = kestrel::gpu::survey(&Config::default()).expect("an adapter to render on");
        Env {
            adapters: survey.adapters,
            default_adapter: survey.default,
            failed: survey.failed,
            gpu_error: None,
            ffmpeg: Err("not looked for".into()),
            ring: Default::default(),
            update: None,
        }
    }

    fn wavs(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "wav"))
            .map(|p| (file_name(&p), std::fs::read(&p).unwrap()))
            .collect();
        v.sort();
        v
    }

    /// The guided renderer's promise, for Merlin: every step and the render \[14\]
    #[test]
    fn a_guided_merlin_render_is_byte_identical_to_the_same_command_and_refuses_what_it_does_not_take() {
        use kestrel::midi::MidiWriter;
        let dir = std::env::temp_dir().join("kestrel_guided_merlin");
        let _ = std::fs::remove_dir_all(&dir);
        let [guided, refused_dir, cli] = ["guided", "refused", "cli"].map(|d| dir.join(d));
        for d in [&guided, &refused_dir, &cli] {
            std::fs::create_dir_all(d).unwrap();
        }
        let font = dir.join("rich.sf2");
        let other = dir.join("other.sf2");
        kestrel::testkit::rich_sf2(&font, 48_000).unwrap();
        kestrel::testkit::rich_sf2(&other, 48_000).unwrap();
        let midi = dir.join("song.mid");
        let mut w = MidiWriter::new(480);
        w.raw_track(vec![(0, vec![0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])]);
        for ch in 0..2u8 {
            let notes: Vec<(u64, Vec<u8>)> = (0..24u64)
                .flat_map(|i| {
                    let key = 48 + ((i * 7 + ch as u64) % 24) as u8;
                    let t = i * 120 + ch as u64 * 240;
                    [(t, vec![0x90 | ch, key, 90]), (t + 200, vec![0x80 | ch, key, 0])]
                })
                .collect();
            w.raw_track(notes);
        }
        w.save(&midi).unwrap();
        let env = this_machine();

        let mut io = Script {
            // [15]
            lines: ["", "", "", "", "1", "", "1"].into(),
            files: [vec![midi.clone()], vec![font.clone(), other.clone()], vec![font.clone()]].into(),
            folders: [guided.clone()].into(),
        };
        let next = render(&mut io, &env).unwrap();
        assert!(next == Next::Menu, "the render did not end on Home");
        assert!(io.lines.is_empty() && io.files.is_empty(), "the flow stopped early: {:?}", io.lines);

        let mut argv: Vec<OsString> = vec!["kestrel".into(), "render".into(), midi.clone().into_os_string()];
        argv.extend(["-s".into(), font.clone().into_os_string(), "-o".into(), cli.join("song.wav").into_os_string()]);
        argv.extend(["--backend", "merlin"].map(OsString::from));
        crate::render::render_cli(super::super::super::parse_render(argv).unwrap()).unwrap();
        let (a, b) = (wavs(&guided), wavs(&cli));
        assert_eq!(a.len(), 1);
        assert!(a[0].1.len() > 1000, "the guided render is empty");
        assert!(a == b, "the guided Merlin render differs from the command's");

        // [16]
        let mut io = Script {
            lines: ["", "", "", "1", "--max-voices 5"].into(),
            files: [vec![midi.clone()], vec![font.clone()]].into(),
            folders: [refused_dir.clone()].into(),
        };
        let next = render(&mut io, &env).unwrap();
        assert!(next == Next::Exit && io.lines.is_empty(), "the flow went on past a refused flag");
        assert!(wavs(&refused_dir).is_empty(), "a refused flag rendered anyway");

        // [17]
        let mut io = Script {
            lines: ["", "", "0"].into(),
            files: [vec![midi.clone()], vec![font.clone()]].into(),
            folders: [refused_dir.clone()].into(),
        };
        let next = render(&mut io, &env).unwrap();
        assert!(next == Next::Menu, "0 at the third step did not go home");
        assert_eq!(io.folders.len(), 1, "the folder was asked for after going home");
        assert!(wavs(&refused_dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The typed-flags step, in Merlin's window: the render it makes is \[18\]
    #[test]
    fn step_six_makes_merlins_render_with_no_voices_and_no_checkpoint_and_names_the_adapter() {
        use super::super::super::{FontsRef, Verdict};
        let dir = std::env::temp_dir().join("kestrel_guided_merlin_flags");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let font = dir.join("rich.sf2");
        kestrel::testkit::rich_sf2(&font, 48_000).unwrap();
        let midi = dir.join("song.mid");
        let mut w = kestrel::midi::MidiWriter::new(480);
        w.raw_track(vec![(0, vec![0x90, 60, 90]), (240, vec![0x80, 60, 0])]);
        w.save(&midi).unwrap();
        let Verdict::Valid(info) = checks::check_midi(&midi) else { panic!("the test's MIDI is not valid") };
        let env = this_machine();
        let paths = [font.clone()];
        let mut io = Script { lines: [""].into(), files: Default::default(), folders: Default::default() };
        let base = [OsString::from("--backend=merlin")];
        let out = dir.join("out.wav");
        let Step::Got(ready) = step_flags(&mut io, &env, &info, FontsRef { paths: &paths, bank: None }, 0, &out, (6, 6), &base, true) else {
            panic!("step 6 did not give a render")
        };
        assert!(matches!(ready.plan.kind, kestrel::config::BackendKind::Merlin));
        assert!(ready.plan.cfg.merlin_key_major && ready.plan.cfg.merlin_key_major_gpu && ready.plan.cfg.merlin_key_parse);
        assert!(ready.job.checkpoint.is_none(), "a merlin render was given a checkpoint");
        assert!(ready.adapter.is_some(), "the render does not say which adapter it is on");
        assert!(!ready.flags.iter().any(|f| f.starts_with("--max-voices")), "{:?}", ready.flags);
        assert!(ready.flags.iter().any(|f| f == "--backend=merlin"), "{:?}", ready.flags);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn what_next_is_home_or_quit() {
        let ask = |lines: &[&'static str]| what_next(&mut Script { lines: lines.iter().copied().collect(), files: Default::default(), folders: Default::default() });
        assert!(ask(&["1"]) == Next::Menu);
        assert!(ask(&["2"]) == Next::Exit);
        assert!(ask(&["q"]) == Next::Exit);
        assert!(ask(&["x", "3", "1"]) == Next::Menu, "asked again until it is an answer");
        assert!(ask(&[]) == Next::Exit, "input ended");
    }
}
