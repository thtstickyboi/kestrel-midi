// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `ktrl.ini`, beside the executable: what Kestrel remembers between runs. \[1\]

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const FILE: &str = "ktrl.ini";

/// Which releases the update check announces. Chosen by number from Extras, \[2\]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ring {
    /// 1: every release, fix releases such as 1.1.1 included.
    #[default]
    Fast,
    /// 2: feature releases only -- a new minor or major version, such as 1.2.0 \[3\]
    Slow,
}

impl Ring {
    pub const ALL: [Ring; 2] = [Ring::Fast, Ring::Slow];

    /// For the API and for people.
    pub fn name(self) -> &'static str {
        match self {
            Ring::Fast => "fast",
            Ring::Slow => "slow",
        }
    }

    /// Its menu number, which is also what `ktrl.ini` holds.
    pub fn number(self) -> u8 {
        match self {
            Ring::Fast => 1,
            Ring::Slow => 2,
        }
    }

    pub fn from_number(s: &str) -> Option<Ring> {
        Ring::ALL.into_iter().find(|r| s == r.number().to_string())
    }
}

/// The folders remembered, as `(key, what it is)`, in the order the file \[4\]
pub const FOLDERS: &[(&str, &str)] = &[
    ("midi", "MIDI files"),
    ("soundfont", "Soundfonts"),
    ("output", "Where renders are saved"),
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Settings {
    pub ring: Ring,
    folders: BTreeMap<&'static str, PathBuf>,
}

impl Settings {
    pub fn folder(&self, key: &str) -> Option<&Path> {
        self.folders.get(key).map(PathBuf::as_path)
    }

    /// Remember `dir` for `key`. A path the file cannot hold -- one that is not \[5\]
    pub fn set_folder(&mut self, key: &str, dir: &Path) {
        let Some(&(key, _)) = FOLDERS.iter().find(|(k, _)| *k == key) else {
            return;
        };
        match dir.to_str() {
            Some(s) if !s.contains(['\n', '\r']) && !s.trim().is_empty() => {
                self.folders.insert(key, dir.to_path_buf());
            }
            _ => {}
        }
    }
}

/// `ktrl.ini` beside this executable.
pub fn path() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating the kestrel executable")?;
    let dir = exe.parent().context("the kestrel executable has no parent directory")?;
    Ok(dir.join(FILE))
}

/// The settings, and anything in the file that could not be used. A missing \[6\]
pub fn load() -> (Settings, Vec<String>) {
    match path() {
        Ok(p) => load_from(&p),
        Err(_) => (Settings::default(), Vec::new()),
    }
}

/// Change the settings and save them. Read afresh first, so a change made \[7\]
pub fn update(change: impl FnOnce(&mut Settings)) -> Result<()> {
    update_at(&path()?, change)
}

fn load_from(path: &Path) -> (Settings, Vec<String>) {
    match std::fs::read_to_string(path) {
        Ok(text) => parse(&text),
        Err(_) => (Settings::default(), Vec::new()),
    }
}

fn update_at(path: &Path, change: impl FnOnce(&mut Settings)) -> Result<()> {
    let (mut settings, _) = load_from(path);
    change(&mut settings);
    let mut text = render(&settings);
    if cfg!(windows) {
        text = text.replace('\n', "\r\n");
    }
    // [8]
    let tmp = path.with_extension("ini.tmp");
    std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

/// Lenient on purpose. Keys and sections it does not know are skipped without \[9\]
fn parse(text: &str) -> (Settings, Vec<String>) {
    let mut settings = Settings::default();
    let mut problems = Vec::new();
    let mut section = String::new();
    for line in text.trim_start_matches('\u{FEFF}').lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with([';', '#']) {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = name.trim().to_ascii_lowercase();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let (key, value) = (key.trim().to_ascii_lowercase(), value.trim());
        match (section.as_str(), key.as_str()) {
            ("updates", "ring") => match Ring::from_number(value) {
                Some(ring) => settings.ring = ring,
                None => problems.push(format!(
                    "ring = {value:?} isn't 1 or 2, so the Fast Ring is used; choose one in Extras"
                )),
            },
            ("folders", key) if !value.is_empty() => settings.set_folder(key, Path::new(value)),
            _ => {}
        }
    }
    (settings, problems)
}

fn render(settings: &Settings) -> String {
    let mut out = format!("[updates]\nring = {}\n\n[folders]\n", settings.ring.number());
    for (key, _) in FOLDERS {
        match settings.folder(key).and_then(Path::to_str) {
            Some(dir) => out.push_str(&format!("{key} = {dir}\n")),
            None => out.push_str(&format!("{key} =\n")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_every_default() {
        let (s, problems) = load_from(Path::new("this/does/not/exist/ktrl.ini"));
        assert_eq!(s, Settings::default());
        assert_eq!(s.ring, Ring::Fast);
        assert!(problems.is_empty());
    }

    #[test]
    fn what_is_written_reads_back() {
        let mut s = Settings { ring: Ring::Slow, ..Default::default() };
        s.set_folder("midi", Path::new(r"C:\Music\black MIDI\東方"));
        s.set_folder("output", Path::new("/home/someone/renders"));
        let (back, problems) = parse(&render(&s));
        assert_eq!(back, s);
        assert!(problems.is_empty());
        // And with the line endings Windows writes, and a BOM from an editor.
        let crlf = format!("\u{FEFF}{}", render(&s).replace('\n', "\r\n"));
        assert_eq!(parse(&crlf).0, s);
    }

    #[test]
    fn the_file_is_bare() {
        let mut s = Settings::default();
        s.set_folder("midi", Path::new(r"C:\Music"));
        assert_eq!(
            render(&s),
            "[updates]\nring = 1\n\n[folders]\nmidi = C:\\Music\nsoundfont =\noutput =\n"
        );
    }

    #[test]
    fn a_hand_edited_file_is_read_leniently() {
        let text = "
            # a comment of the other kind
            [Updates]
            Ring =  2
            [folders]
            midi   =   C:\\Music
            soundfont =
            not_a_folder = C:\\x
            [something_from_a_later_version]
            key = value
            a line with no equals sign
        ";
        let (s, problems) = parse(text);
        assert_eq!(s.ring, Ring::Slow);
        assert_eq!(s.folder("midi"), Some(Path::new(r"C:\Music")));
        assert_eq!(s.folder("soundfont"), None);
        assert_eq!(s.folder("not_a_folder"), None);
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn a_ring_it_cannot_read_is_reported_and_left_fast() {
        for bad in ["3", "0", "slow", "02"] {
            let (s, problems) = parse(&format!("[updates]\nring = {bad}\n"));
            assert_eq!(s.ring, Ring::Fast, "{bad}");
            assert_eq!(problems.len(), 1, "{bad}");
            assert!(problems[0].contains(bad), "{problems:?}");
        }
        assert_eq!(Ring::from_number("1"), Some(Ring::Fast));
        assert_eq!(Ring::from_number("2"), Some(Ring::Slow));
    }

    #[test]
    fn a_path_the_file_cannot_hold_is_not_remembered() {
        let mut s = Settings::default();
        s.set_folder("midi", Path::new("D:\\two\nlines"));
        s.set_folder("midi", Path::new("   "));
        assert_eq!(s.folder("midi"), None);
    }

    #[test]
    fn an_update_keeps_what_another_window_saved() {
        let dir = std::env::temp_dir().join(format!("kestrel_ktrl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join(FILE);
        let _ = std::fs::remove_file(&file);

        update_at(&file, |s| s.ring = Ring::Slow).unwrap();
        // Another process's change, made between this one's reads.
        update_at(&file, |s| s.set_folder("soundfont", Path::new(r"E:\fonts"))).unwrap();
        let (s, _) = load_from(&file);
        assert_eq!(s.ring, Ring::Slow);
        assert_eq!(s.folder("soundfont"), Some(Path::new(r"E:\fonts")));
        assert!(!file.with_extension("ini.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
