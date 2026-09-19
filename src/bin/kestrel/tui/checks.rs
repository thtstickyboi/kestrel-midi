// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The judgements the guided renderer makes about what it is handed: whether \[1\]

use kestrel::bank::Bank;
use kestrel::midi::{Division, SmfHeader};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

// ---- MIDI -----------------------------------------------------------------

/// A MIDI that will open, and what its header says.
#[derive(Debug)]
pub struct MidiInfo {
    pub path: PathBuf,
    pub size: u64,
    pub format: u16,
    pub tracks: usize,
    pub division: Division,
    /// Things that do not stop a render but are worth saying.
    pub notes: Vec<String>,
}

#[derive(Debug)]
pub enum Verdict {
    Valid(MidiInfo),
    Invalid { path: PathBuf, reason: String },
}

impl Verdict {
    pub fn is_valid(&self) -> bool {
        matches!(self, Verdict::Valid(_))
    }
}

/// What a file that is not a MIDI most likely is, from its first bytes. Black \[2\]
fn archive_kind(head: &[u8]) -> Option<&'static str> {
    const MAGIC: &[(&[u8], &str)] = &[
        (&[0xFD, b'7', b'z', b'X', b'Z', 0x00], "xz"),
        (&[b'7', b'z', 0xBC, 0xAF, 0x27, 0x1C], "7z"),
        (b"PK\x03\x04", "zip"),
        (&[0x1F, 0x8B], "gzip"),
        (&[0x28, 0xB5, 0x2F, 0xFD], "zstd"),
        (b"Rar!\x1A\x07", "RAR"),
        (b"BZh", "bzip2"),
    ];
    MAGIC
        .iter()
        .find(|(magic, _)| head.starts_with(magic))
        .map(|&(_, kind)| kind)
}

fn read_head(path: &Path, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut f = File::open(path)?;
    let mut got = 0;
    while got < buf.len() {
        match f.read(&mut buf[got..])? {
            0 => break,
            n => got += n,
        }
    }
    Ok(got)
}

/// The quick check: the header and the chunk list, never an event. \[3\]
pub fn check_midi(path: &Path) -> Verdict {
    let invalid = |reason: String| Verdict::Invalid {
        path: path.to_path_buf(),
        reason,
    };

    let size = match std::fs::metadata(path) {
        Ok(m) if m.is_file() => m.len(),
        Ok(_) => return invalid("not a file".into()),
        Err(e) => return invalid(format!("can't be read: {e}")),
    };
    if size == 0 {
        return invalid("the file is empty".into());
    }

    let mut head = [0u8; 16];
    let n = match read_head(path, &mut head) {
        Ok(n) => n,
        Err(e) => return invalid(format!("can't be read: {e}")),
    };
    let head = &head[..n];
    if let Some(kind) = archive_kind(head) {
        return invalid(format!(
            "a compressed {kind} archive; extract the MIDI from it first"
        ));
    }
    if head.len() >= 12 && &head[0..4] == b"RIFF" && &head[8..12] == b"RMID" {
        return invalid("an RMI file (MIDI wrapped in RIFF), which Kestrel does not read".into());
    }
    if !head.starts_with(b"MThd") {
        return invalid("not a MIDI file".into());
    }
    if size < 14 {
        return invalid(format!("too short to be a MIDI file ({size} bytes)"));
    }

    let h = match SmfHeader::read(path) {
        Ok(h) => h,
        Err(e) => return invalid(format!("{e:#}")),
    };
    if h.format > 2 {
        return invalid(format!("unknown MIDI format {}", h.format));
    }
    if let Division::Smpte {
        fps,
        ticks_per_frame,
    } = h.division
    {
        if fps == 0 || ticks_per_frame == 0 {
            return invalid("its timing division is zero, so no event has a time".into());
        }
    }
    if h.tracks.is_empty() {
        return invalid("no track data; the file is a header and nothing else".into());
    }
    if h.tracks.iter().all(|&(_, len)| len == 0) {
        return invalid("every track in it is empty".into());
    }

    let mut notes = Vec::new();
    if h.truncated_tracks > 0 {
        notes.push(format!(
            "{} track{} past the end of the file, so it is probably cut short; \
             what is there will render",
            h.truncated_tracks,
            if h.truncated_tracks == 1 { " runs" } else { "s run" }
        ));
    }
    if h.tracks.len() != h.declared_tracks as usize {
        notes.push(format!(
            "its header claims {} tracks and the file holds {}",
            h.declared_tracks,
            h.tracks.len()
        ));
    }
    if h.format == 2 {
        notes.push("format 2: its independent sequences will all play at once".into());
    }

    Verdict::Valid(MidiInfo {
        path: path.to_path_buf(),
        size,
        format: h.format,
        tracks: h.tracks.len(),
        division: h.division,
        notes,
    })
}

/// `check_midi` over a selection, on a few threads, in the order given.
pub fn check_midis(paths: &[PathBuf]) -> Vec<Verdict> {
    if paths.is_empty() {
        return Vec::new();
    }
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 8)
        .min(paths.len());
    let chunk = paths.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles: Vec<_> = paths
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(|p| check_midi(p)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a MIDI check panicked"))
            .collect()
    })
}

pub fn describe_division(d: Division) -> String {
    match d {
        Division::Ppq(n) => format!("{n} PPQ"),
        Division::Smpte {
            fps,
            ticks_per_frame,
        } => format!("SMPTE {fps} fps, {ticks_per_frame} ticks a frame"),
    }
}

// ---- Soundfonts -----------------------------------------------------------

/// What a loaded soundfont is, in the terms the flow reports it.
#[derive(Debug, Clone, Default)]
pub struct FontProfile {
    pub presets: usize,
    pub regions: usize,
    pub samples: usize,
    pub pool_bytes: u64,
    /// 0 when the pool is mixed-rate.
    pub pool_rate: u32,
    /// Distinct programs with a preset on bank 0.
    pub melodic_programs: usize,
    /// Presets on bank 128, which is where SoundFonts keep drum kits.
    pub drum_kits: usize,
    /// The bank-0 programs defined, lowest first.
    pub programs: Vec<u16>,
}

impl FontProfile {
    pub fn of(bank: &Bank) -> Self {
        let mut programs: Vec<u16> = bank
            .presets
            .iter()
            .filter(|p| p.bank == 0 && p.program < 128)
            .map(|p| p.program)
            .collect();
        programs.sort_unstable();
        programs.dedup();
        FontProfile {
            presets: bank.presets.len(),
            regions: bank.regions.len(),
            samples: bank.samples.len(),
            pool_bytes: bank.pool_bytes(),
            pool_rate: bank.pool_rate,
            melodic_programs: programs.len(),
            drum_kits: bank.presets.iter().filter(|p| p.bank == 128).count(),
            programs,
        }
    }

    /// A General MIDI bank, as opposed to an instrument. \[4\]
    pub fn is_general_midi(&self) -> bool {
        self.melodic_programs >= 96 || (self.melodic_programs >= 64 && self.drum_kits > 0)
    }
}

/// Which soundfont goes underneath. \[5\]
pub fn layer_order(profiles: &[FontProfile]) -> Vec<usize> {
    match profiles {
        [a, b] if !a.is_general_midi() && b.is_general_midi() => vec![1, 0],
        _ => (0..profiles.len()).collect(),
    }
}

pub fn gm_name(program: u16) -> &'static str {
    GM_PROGRAMS.get(program as usize).copied().unwrap_or("")
}

/// General MIDI Level 1 program names.
const GM_PROGRAMS: [&str; 128] = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
    "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ",
    "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica",
    "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)",
    "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass",
    "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1",
    "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello",
    "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone",
    "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1",
    "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
    "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina", "Lead 1 (square)",
    "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)",
    "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)",
    "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)",
    "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)",
    "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)",
    "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen",
    "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo",
    "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum",
    "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
];

// ---- Voices ---------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum VoiceAnswer {
    /// Enter, or the word "enter" in any case.
    Default,
    /// `-1`.
    DeviceMax,
    Count(u32),
    /// Over the device max, or over what a u32 holds.
    TooMany(u64),
    Invalid,
}

/// Read a typed voice count. \[6\]
pub fn parse_voices(input: &str, device_max: Option<u32>) -> VoiceAnswer {
    let t = input.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("enter") {
        return VoiceAnswer::Default;
    }
    if t == "-1" {
        return VoiceAnswer::DeviceMax;
    }
    let cleaned: String = t
        .chars()
        .filter(|ch| !matches!(ch, ',' | '_' | '\'' | ' '))
        .collect();
    let (digits, scale) = match cleaned.chars().last() {
        Some('k' | 'K') => (&cleaned[..cleaned.len() - 1], Some(1e3)),
        Some('m' | 'M') => (&cleaned[..cleaned.len() - 1], Some(1e6)),
        _ => (cleaned.as_str(), None),
    };
    let n: u64 = match scale {
        None => {
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return VoiceAnswer::Invalid;
            }
            // All digits and still unparseable means it overflowed.
            digits.parse().unwrap_or(u64::MAX)
        }
        Some(scale) => match digits.parse::<f64>() {
            Ok(v) if v.is_finite() && v >= 0.0 => (v * scale).round() as u64,
            _ => return VoiceAnswer::Invalid,
        },
    };
    if n == 0 {
        return VoiceAnswer::Invalid;
    }
    if n > u32::MAX as u64 || device_max.is_some_and(|m| n > m as u64) {
        return VoiceAnswer::TooMany(n);
    }
    VoiceAnswer::Count(n as u32)
}

// ---- Output ---------------------------------------------------------------

/// Local time as it goes into a file name. \[7\]
pub fn timestamp() -> String {
    chrono::Local::now().format("%m-%d-%Y %H.%M.%S").to_string()
}

/// Where a render of `midi` into `dir` goes: the MIDI's name with the \[8\]
pub fn output_path(dir: &Path, midi: &Path, ext: &str, stamp: &str) -> PathBuf {
    let stem = midi
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "render".to_string());
    let plain = dir.join(format!("{stem}.{ext}"));
    if !plain.exists() {
        return plain;
    }
    let stamped = dir.join(format!("{stem} ({stamp}).{ext}"));
    if !stamped.exists() {
        return stamped;
    }
    (2u32..)
        .map(|n| dir.join(format!("{stem} ({stamp}) ({n}).{ext}")))
        .find(|p| !p.exists())
        .expect("an unused name")
}

// ---- Typed flags ----------------------------------------------------------

/// Split a typed line into arguments: whitespace separates, and single or \[9\]
pub fn split_args(line: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut started = false;
    let mut quote: Option<char> = None;
    for ch in line.chars() {
        match quote {
            Some(q) if ch == q => quote = None,
            Some(_) => cur.push(ch),
            None if ch == '"' || ch == '\'' => {
                quote = Some(ch);
                started = true;
            }
            None if ch.is_whitespace() => {
                if started {
                    out.push(std::mem::take(&mut cur));
                    started = false;
                }
            }
            None => {
                cur.push(ch);
                started = true;
            }
        }
    }
    if quote.is_some() {
        return Err("a quote is never closed".into());
    }
    if started {
        out.push(cur);
    }
    Ok(out)
}

/// Typed flags, checked and ready to append to a render's arguments.
#[derive(Debug, Default, PartialEq)]
pub struct Extra {
    pub args: Vec<String>,
    /// `--adapter N`, as an index into the environment check's list.
    pub adapter: Option<usize>,
    /// `--max-voices` is among them, and overrides step 3.
    pub max_voices: bool,
    /// Some flag might change how a soundfont loads.
    pub reload: bool,
}

/// Flags no loader reads: they choose the device, the voice pool, the \[10\]
pub(crate) const LOAD_NEUTRAL: &[&str] = &[
    "--adapter",
    "--admit",
    "--backend",
    "--block-csv",
    "--ceiling-db",
    "--ffmpeg",
    "--format",
    "--gpu-adapter",
    "--gpu-backend",
    "--limiter",
    "--limiter-release-ms",
    "--limiter-sustain-ms",
    "--lookahead-ms",
    "--max-voices",
    "--min-velocity",
    "--nan-guard",
    "--no-limiter",
    "--no-sort",
    "--no-true-peak",
    "--profile",
    "--seconds",
    "--steal",
    "--steal-percent",
    "--unchecked-shaders",
];

/// Check typed flags against what the earlier steps already chose. \[11\]
pub fn extra_flags(tokens: Vec<String>, adapters: usize) -> Result<Extra, String> {
    let mut out = Extra::default();
    let mut it = tokens.into_iter();
    while let Some(tok) = it.next() {
        let name = match tok.split_once('=') {
            Some((n, _)) if n.starts_with("--") => n.to_string(),
            _ => tok.clone(),
        };
        let short = |f: char| name.starts_with(&format!("-{f}")) && !name.starts_with("--");

        if name == "--adapter" {
            let value = match tok.split_once('=') {
                Some((_, v)) => v.to_string(),
                None => it
                    .next()
                    .ok_or("--adapter needs a number from the adapter list")?,
            };
            let n: usize = value
                .trim()
                .parse()
                .map_err(|_| format!("--adapter takes a number from the list, not {value:?}"))?;
            if n == 0 || n > adapters {
                return Err(match adapters {
                    0 => "--adapter has nothing to choose from: no adapters were found".into(),
                    1 => format!("--adapter {n} is not in the list; the only adapter is [1]"),
                    _ => format!("--adapter {n} is not in the list; pick 1 to {adapters}"),
                });
            }
            out.adapter = Some(n - 1);
            continue;
        }
        if name == "--force-cli" {
            continue;
        }
        if name == "--soundfont" || short('s') {
            return Err("soundfonts were chosen in step 2".into());
        }
        if name == "--out" || short('o') {
            return Err("the output was chosen in steps 4 and 5".into());
        }
        if matches!(name.as_str(), "-h" | "--help" | "-V" | "--version") {
            return Err("type ? to list the flags".into());
        }
        if name == "--" {
            return Err("the MIDI was chosen in step 1".into());
        }
        if name == "--progress" || name == "--progress-interval" {
            return Err(
                "--progress is for another program reading Kestrel; the guided renderer \
                 has its own progress screen"
                    .into(),
            );
        }
        if name.starts_with("--") {
            if name == "--max-voices" {
                out.max_voices = true;
            }
            if !LOAD_NEUTRAL.contains(&name.as_str()) {
                out.reload = true;
            }
        }
        out.args.push(tok);
    }
    let picks_by_name = out
        .args
        .iter()
        .any(|a| a.starts_with("--gpu-adapter") || a.starts_with("--gpu-backend"));
    if out.adapter.is_some() && picks_by_name {
        return Err("use --adapter, or --gpu-adapter and --gpu-backend, not both".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel::midi::MidiWriter;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("kestrel_tui_checks").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn enter_in_any_spelling_is_the_default() {
        for typed in ["", "   ", "enter", "Enter", "ENTER", " eNtEr "] {
            assert_eq!(parse_voices(typed, Some(10)), VoiceAnswer::Default, "{typed:?}");
        }
    }

    #[test]
    fn voice_counts_read_the_way_people_type_them() {
        let max = Some(20_000_000);
        assert_eq!(parse_voices("-1", max), VoiceAnswer::DeviceMax);
        for typed in ["700000", "700,000", "700_000", "700 000", "700k", "700K", "0.7m"] {
            assert_eq!(parse_voices(typed, max), VoiceAnswer::Count(700_000), "{typed:?}");
        }
        assert_eq!(parse_voices("1.5M", max), VoiceAnswer::Count(1_500_000));
        assert_eq!(parse_voices("25000000", max), VoiceAnswer::TooMany(25_000_000));
        assert_eq!(
            parse_voices("99999999999999999999999", None),
            VoiceAnswer::TooMany(u64::MAX)
        );
        assert_eq!(parse_voices("5000000000", None), VoiceAnswer::TooMany(5_000_000_000));
        for bad in ["0", "-2", "lots", "1e6", "k", "12x", "-1k"] {
            assert_eq!(parse_voices(bad, max), VoiceAnswer::Invalid, "{bad:?}");
        }
    }

    #[test]
    fn an_existing_file_is_never_overwritten() {
        let dir = scratch("naming");
        let midi = Path::new("C:/somewhere/Song Title.mid");
        let first = output_path(&dir, midi, "opus", "09-12-2026 14.05.33");
        assert_eq!(first, dir.join("Song Title.opus"));

        std::fs::write(&first, b"x").unwrap();
        let second = output_path(&dir, midi, "opus", "09-12-2026 14.05.33");
        assert_eq!(second, dir.join("Song Title (09-12-2026 14.05.33).opus"));

        std::fs::write(&second, b"x").unwrap();
        let third = output_path(&dir, midi, "opus", "09-12-2026 14.05.33");
        assert_eq!(third, dir.join("Song Title (09-12-2026 14.05.33) (2).opus"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every character the timestamp puts in a name has to be legal on \[12\]
    #[test]
    fn the_timestamp_is_a_legal_windows_file_name() {
        let stamp = timestamp();
        assert!(!stamp.contains(['<', '>', ':', '"', '/', '\\', '|', '?', '*']), "{stamp}");
        assert_eq!(stamp.len(), "09-12-2026 14.05.33".len(), "{stamp}");
    }

    #[test]
    fn the_midi_check_tells_the_failures_apart() {
        let dir = scratch("verdicts");

        let good = dir.join("good.mid");
        let mut w = MidiWriter::new(480);
        w.tempo_track(500_000);
        w.track(vec![(0, [0x90, 60, 100], 3), (240, [0x80, 60, 0], 3)]);
        w.save(&good).unwrap();
        match check_midi(&good) {
            Verdict::Valid(info) => {
                assert_eq!((info.format, info.tracks), (1, 2));
                assert!(info.notes.is_empty(), "{:?}", info.notes);
            }
            v => panic!("{v:?}"),
        }

        let reason = |name: &str, bytes: &[u8]| -> String {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            match check_midi(&p) {
                Verdict::Invalid { reason, .. } => reason,
                v => panic!("{name} should be invalid, got {v:?}"),
            }
        };
        assert!(reason("text.mid", b"this is not a midi").contains("not a MIDI"));
        assert!(reason("empty.mid", b"").contains("empty"));
        assert!(reason("song.mid.xz", &[0xFD, b'7', b'z', b'X', b'Z', 0, 1, 2]).contains("xz"));
        assert!(reason("song.zip", b"PK\x03\x04rest").contains("zip"));
        assert!(reason("head.mid", b"MThd\0\0\0\x06\0\x01\0\x01\x01\xE0").contains("no track"));
        assert!(reason("stub.mid", b"MThd\0\0").contains("too short"));
        let zero_div = b"MThd\0\0\0\x06\0\x01\0\x01\0\0MTrk\0\0\0\x04\0\xFF\x2F\0";
        assert!(reason("zerodiv.mid", zero_div).contains("division"));
        assert!(reason("wrapped.rmi", b"RIFF\x10\0\0\0RMIDdata").contains("RMI"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_selection_keeps_its_order_across_threads() {
        let dir = scratch("many");
        let mut paths = Vec::new();
        for i in 0..40 {
            let p = dir.join(format!("{i:02}.mid"));
            if i % 3 == 0 {
                std::fs::write(&p, b"nope").unwrap();
            } else {
                let mut w = MidiWriter::new(480);
                w.track(vec![(0, [0x90, 60, 100], 3), (240, [0x80, 60, 0], 3)]);
                w.save(&p).unwrap();
            }
            paths.push(p);
        }
        let verdicts = check_midis(&paths);
        assert_eq!(verdicts.len(), 40);
        for (i, v) in verdicts.iter().enumerate() {
            assert_eq!(v.is_valid(), i % 3 != 0, "file {i}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn profile(programs: std::ops::Range<u16>, kits: usize) -> FontProfile {
        FontProfile {
            melodic_programs: programs.len(),
            drum_kits: kits,
            programs: programs.collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_general_midi_bank_goes_underneath_whichever_was_picked_first() {
        let gm = profile(0..128, 8);
        let piano = profile(0..1, 0);
        assert!(gm.is_general_midi());
        assert!(!piano.is_general_midi());
        assert!(profile(0..64, 1).is_general_midi(), "half the programs and a kit");
        assert!(!profile(0..64, 0).is_general_midi(), "half the programs, no kit");

        assert_eq!(layer_order(&[gm.clone(), piano.clone()]), vec![0, 1]);
        assert_eq!(layer_order(&[piano.clone(), gm.clone()]), vec![1, 0]);
        assert_eq!(layer_order(&[piano.clone(), piano.clone()]), vec![0, 1]);
        assert_eq!(layer_order(&[gm.clone(), gm]), vec![0, 1]);
        assert_eq!(layer_order(&[piano]), vec![0]);
        assert_eq!(gm_name(0), "Acoustic Grand Piano");
        assert_eq!(gm_name(127), "Gunshot");
    }

    #[test]
    fn typed_arguments_split_like_a_shell_for_the_cases_that_matter() {
        assert_eq!(
            split_args(r#"--gpu-adapter "Intel UHD" --block-csv C:\out\b.csv"#).unwrap(),
            vec!["--gpu-adapter", "Intel UHD", "--block-csv", r"C:\out\b.csv"]
        );
        assert_eq!(split_args("  ").unwrap(), Vec::<String>::new());
        assert_eq!(split_args("a '' b").unwrap(), vec!["a", "", "b"]);
        assert!(split_args("--gpu-adapter \"Intel").is_err());
    }

    fn flags(line: &str) -> Result<Extra, String> {
        extra_flags(split_args(line).unwrap(), 3)
    }

    #[test]
    fn typed_flags_cannot_collide_with_the_steps() {
        let x = flags("--adapter 2 --limiter omni").unwrap();
        assert_eq!(x.adapter, Some(1));
        assert_eq!(x.args, vec!["--limiter", "omni"]);
        assert!(!x.reload && !x.max_voices);

        assert_eq!(flags("--adapter=3").unwrap().adapter, Some(2));
        assert!(flags("--adapter 4").is_err());
        assert!(flags("--adapter 0").is_err());
        assert!(flags("--adapter two").is_err());
        assert!(flags("--adapter").is_err());
        assert!(flags("--adapter 1 --gpu-adapter Intel").is_err());

        for clash in [
            "-s x.sf2",
            "--soundfont x.sf2",
            "--soundfont=x.sf2",
            "-o a.wav",
            "--out=a.wav",
            "-- song.mid",
            "--help",
            "--progress json",
            "--progress-interval=100",
        ] {
            assert!(flags(clash).is_err(), "{clash}");
        }

        let v = flags("--max-voices 2000000").unwrap();
        assert!(v.max_voices && !v.reload);
    }

    #[test]
    fn only_flags_no_loader_reads_keep_the_loaded_soundfont() {
        for neutral in ["--seconds 10", "--profile", "--limiter omni", "--gpu-backend dx12", "--steal oldest"] {
            assert!(!flags(neutral).unwrap().reload, "{neutral}");
        }
        for loading in ["--rate 44100", "--no-filter", "--volume 50", "--sf-programs 0-7", "--pool-budget 512", "--decay-curve linear"] {
            assert!(flags(loading).unwrap().reload, "{loading}");
        }
    }
}
