// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! SFZ loader. \[1\]

use crate::bank::*;
use crate::config::Config;
use crate::resample;
use crate::wav;
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Ignore the note-off gate entirely (SFZ `loop_mode=one_shot`).
pub const VF_ONE_SHOT: u32 = 1 << 2;

#[derive(Clone, Default)]
struct OpcodeSet(BTreeMap<String, String>);

impl OpcodeSet {
    fn merged(&self, other: &OpcodeSet) -> OpcodeSet {
        let mut m = self.0.clone();
        for (k, v) in &other.0 {
            m.insert(k.clone(), v.clone());
        }
        OpcodeSet(m)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(|s| s.as_str())
    }
    /// Collapse aliased opcodes to one canonical name, at the level that \[2\]
    fn canonicalise(&mut self) {
        if let Some(k) = self.0.remove("key") {
            for alias in ["lokey", "hikey", "pitch_keycenter"] {
                // [3]
                self.0.entry(alias.to_string()).or_insert_with(|| k.clone());
            }
        }
        for (from, to) in [
            ("loopmode", "loop_mode"),
            ("loopstart", "loop_start"),
            ("loopend", "loop_end"),
            ("pitch", "tune"),
        ] {
            if let Some(v) = self.0.remove(from) {
                self.0.entry(to.to_string()).or_insert(v);
            }
        }
    }
    fn f32(&self, k: &str) -> Option<f32> {
        self.get(k).and_then(|v| v.trim().parse::<f32>().ok())
    }
    fn i32(&self, k: &str) -> Option<i32> {
        self.get(k)
            .and_then(|v| v.trim().parse::<i32>().ok().or_else(|| v.trim().parse::<f32>().ok().map(|f| f as i32)))
    }
    /// Note names as well as numbers: `c4`, `a#3`, `60`.
    fn key(&self, k: &str) -> Option<i32> {
        let v = self.get(k)?.trim();
        if let Ok(n) = v.parse::<i32>() {
            return Some(n);
        }
        parse_note_name(v)
    }
}

fn parse_note_name(s: &str) -> Option<i32> {
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    let step = match b[0].to_ascii_lowercase() {
        b'c' => 0,
        b'd' => 2,
        b'e' => 4,
        b'f' => 5,
        b'g' => 7,
        b'a' => 9,
        b'b' => 11,
        _ => return None,
    };
    let mut i = 1;
    let mut acc = 0i32;
    while i < b.len() && (b[i] == b'#' || b[i] == b'b' || b[i] == b'-' && i == 1) {
        match b[i] {
            b'#' => acc += 1,
            b'b' => acc -= 1,
            _ => break,
        }
        i += 1;
    }
    let octave: i32 = s[i..].trim().parse().ok()?;
    // SFZ follows the convention where middle C (60) is c4.
    Some((octave + 1) * 12 + step + acc)
}

struct Parser {
    root: PathBuf,
    unknown: HashMap<String, u32>,
    depth: u32,
    /// `#define $NAME value`, longest name first. \[4\]
    defines: Vec<(String, String)>,
}

impl Parser {
    /// Record a define, or redefine one. \[5\]
    fn define(&mut self, name: String, value: String) {
        match self.defines.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => {
                self.defines.push((name, value));
                // [6]
                self.defines.sort_by_key(|d| std::cmp::Reverse(d.0.len()));
            }
        }
    }

    /// Textually replace every `$NAME` that has been defined by this point. \[7\]
    fn substitute(&self, line: &str) -> String {
        if self.defines.is_empty() || !line.contains('$') {
            return line.to_string();
        }
        let mut out = String::with_capacity(line.len());
        let mut rest = line;
        while let Some(i) = rest.find('$') {
            out.push_str(&rest[..i]);
            let tail = &rest[i..];
            match self.defines.iter().find(|(n, _)| tail.starts_with(n.as_str())) {
                Some((n, v)) => {
                    out.push_str(v);
                    rest = &tail[n.len()..];
                }
                None => {
                    out.push('$');
                    rest = &tail[1..];
                }
            }
        }
        out.push_str(rest);
        out
    }

    /// `default_path` is passed in rather than held on the parser because it \[8\]
    fn resolve(&self, default_path: &Path, rel: &str) -> PathBuf {
        // SFZ paths use backslashes on Windows-authored libraries.
        let rel = rel.replace('\\', "/");
        let joined = default_path.join(&rel);
        let cand = self.root.join(&joined);
        if cand.exists() {
            return cand;
        }
        // Case-insensitive fallback, which matters when a Windows-authored
        // library is rendered on a case-sensitive filesystem.
        let parent = cand.parent().unwrap_or(&self.root).to_path_buf();
        let name = cand.file_name().map(|n| n.to_string_lossy().to_lowercase());
        if let (Some(name), Ok(rd)) = (name, std::fs::read_dir(&parent)) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().to_lowercase() == name {
                    return e.path();
                }
            }
        }
        cand
    }

    fn parse_file(&mut self, path: &Path, out: &mut Vec<(String, OpcodeSet)>) -> Result<()> {
        if self.depth > 16 {
            bail!("#include nesting is too deep at {}", path.display());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;

        let mut current_header = String::from("global");
        let mut current = OpcodeSet::default();
        let mut started = false;

        for raw_line in text.lines() {
            let line = strip_comment(raw_line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Before substitution, or a redefinition would have its own name
            // replaced by the value it is about to be given.
            if let Some(rest) = line.strip_prefix("#define") {
                // `rest` must start at a word boundary, or this is some other
                // directive that merely begins the same way.
                if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                    let rest = rest.trim_start();
                    // The value runs to end of line and may contain spaces, so
                    // `find_value_end` is the wrong splitter here.
                    let name_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
                    let (name, value) = rest.split_at(name_end);
                    if name.len() > 1 && name.starts_with('$') {
                        // A define may be written in terms of an earlier one.
                        let value = self.substitute(value.trim());
                        self.define(name.to_string(), value);
                    } else {
                        *self.unknown.entry("`#define` with no `$name`".into()).or_default() += 1;
                    }
                    continue;
                }
            }
            // [9]
            let expanded = self.substitute(line);
            let line = expanded.trim();
            if let Some(rest) = line.strip_prefix("#include") {
                let inc = rest.trim().trim_matches('"').trim();
                // `default_path` governs `sample`, not `#include`.
                let p = self.resolve(Path::new(""), inc);
                if started {
                    out.push((current_header.clone(), std::mem::take(&mut current)));
                    started = false;
                }
                self.depth += 1;
                let r = self.parse_file(&p, out);
                self.depth -= 1;
                if let Err(e) = r {
                    log::warn!("{}", e);
                }
                continue;
            }
            // A line can hold several headers and opcodes.
            let mut rest = line;
            while !rest.is_empty() {
                rest = rest.trim_start();
                if let Some(stripped) = rest.strip_prefix('<') {
                    let Some(end) = stripped.find('>') else { break };
                    if started {
                        out.push((current_header.clone(), std::mem::take(&mut current)));
                    }
                    current_header = stripped[..end].trim().to_ascii_lowercase();
                    current = OpcodeSet::default();
                    started = true;
                    rest = &stripped[end + 1..];
                    continue;
                }
                let Some(eq) = rest.find('=') else { break };
                let key = rest[..eq].trim().to_ascii_lowercase();
                let after = &rest[eq + 1..];
                // [10]
                let value_end = find_value_end(after);
                let value = after[..value_end].trim().to_string();
                rest = &after[value_end..];
                if key.is_empty() {
                    continue;
                }
                current.0.insert(key, value);
                started = true;
            }
        }
        if started {
            out.push((current_header, current));
        }
        Ok(())
    }
}

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Find where a value ends: just before the last whitespace-separated token \[11\]
fn find_value_end(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut last_ws = None;
    while i < bytes.len() {
        if bytes[i] == b'=' {
            return match last_ws {
                Some(w) => w,
                None => i,
            };
        }
        if bytes[i] == b'<' {
            return last_ws.unwrap_or(i);
        }
        if bytes[i].is_ascii_whitespace() {
            // Remember the start of the token after this whitespace run.
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            last_ws = Some(i);
            i = j;
            continue;
        }
        i += 1;
    }
    s.len()
}

pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Bank> {
    let path = path.as_ref();
    let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();

    let mut parser = Parser {
        root: root.clone(),
        unknown: HashMap::new(),
        depth: 0,
        defines: Vec::new(),
    };

    let mut sections: Vec<(String, OpcodeSet)> = Vec::new();
    parser.parse_file(path, &mut sections)?;
    for (_, ops) in &mut sections {
        ops.canonicalise();
    }

    // [12]
    let mut default_path = PathBuf::new();
    let mut global = OpcodeSet::default();
    let mut master = OpcodeSet::default();
    let mut group = OpcodeSet::default();
    let mut region_sets: Vec<(PathBuf, OpcodeSet)> = Vec::new();

    for (h, ops) in &sections {
        match h.as_str() {
            "control" => {
                if let Some(dp) = ops.get("default_path") {
                    default_path = PathBuf::from(dp.replace('\\', "/"));
                }
            }
            "global" => {
                global = ops.clone();
                master = OpcodeSet::default();
                group = OpcodeSet::default();
            }
            // A `<master>` replaces the previous one. Merging it into `global`
            // instead let its opcodes outlive it and reach the regions of the
            // master that followed.
            "master" => {
                master = ops.clone();
                group = OpcodeSet::default();
            }
            "group" => group = ops.clone(),
            "region" => region_sets.push((
                default_path.clone(),
                global.merged(&master).merged(&group).merged(ops),
            )),
            "curve" | "effect" => {}
            // Closing tags. Not SFZ 1.0 -- the format scopes by the next
            // opening header, not by a matching close -- but generators emit
            // them and every other player tolerates them. Treated as the
            // scope end they are written to mean, so a `<region>` after a
            // `</group>` does not silently inherit the group that closed.
            // Ignoring them outright would work on well-formed files and
            // quietly mis-scope the malformed ones.
            "/group" => group = OpcodeSet::default(),
            "/master" => {
                master = OpcodeSet::default();
                group = OpcodeSet::default();
            }
            "/global" => {
                global = OpcodeSet::default();
                master = OpcodeSet::default();
                group = OpcodeSet::default();
            }
            "/region" | "/control" | "/curve" | "/effect" => {}
            other => {
                *parser.unknown.entry(format!("<{other}>")).or_default() += 1;
            }
        }
    }

    if region_sets.is_empty() {
        bail!("{}: no <region> sections", path.display());
    }

    // ---- load every referenced wav once -----------------------------------
    let mut pool: Vec<i16> = Vec::new();
    let mut samples: Vec<SampleInfo> = Vec::new();
    // (path, channel) -> sample index
    let mut sample_cache: HashMap<(PathBuf, usize), u32> = HashMap::new();
    let pool_rate = if cfg.resample_pool { cfg.sample_rate } else { 0 };

    let mut regions: Vec<Region> = Vec::new();
    let mut region_ids: Vec<u32> = Vec::new();
    // A sample that will not load fails once per *region*, and a library that
    // names one keymap from forty velocity groups has thousands of them. Warned
    // inline, the first and only informative line scrolls away. Count them and
    // report the distinct ones, the way unsupported opcodes already are.
    let mut sample_errors: HashMap<String, u32> = HashMap::new();
    let mut unresolved_defines = 0u32;

    for (default_path, ops) in &region_sets {
        let Some(sample_rel) = ops.get("sample") else {
            continue;
        };
        if ops.i32("end") == Some(-1) {
            continue; // conventional way to disable a region
        }
        let spath = parser.resolve(default_path, sample_rel);
        let channels = match load_sample_channels(
            &spath,
            &mut pool,
            &mut samples,
            &mut sample_cache,
            pool_rate,
        ) {
            Ok(c) => c,
            Err(e) => {
                *sample_errors.entry(format!("{e:#}")).or_default() += 1;
                if spath.to_string_lossy().contains('$') {
                    unresolved_defines += 1;
                }
                continue;
            }
        };

        for (ch, sidx) in channels.iter().enumerate() {
            note_unhandled(ops, &mut parser.unknown);
            let mut extra = Vec::new();
            let mut r =
                region_from_opcodes(ops, *sidx, &samples[*sidx as usize], &mut extra);
            for what in extra {
                *parser.unknown.entry(what).or_default() += 1;
            }
            if channels.len() == 2 {
                // [13]
                r.pan = if ch == 0 { -1.0 } else { 1.0 };
            }
            region_ids.push(regions.len() as u32);
            regions.push(r);
        }
    }

    if regions.is_empty() {
        bail!("{}: every region was unusable", path.display());
    }

    for (op, n) in &parser.unknown {
        match op.strip_prefix(UNIMPL_TAG) {
            // [14]
            Some(group) => log::warn!("sfz: {group} not implemented, {n} opcodes ignored"),
            // [15]
            None => log::warn!("sfz: ignored unsupported {op} ({n} times)"),
        }
    }
    report_sample_errors(sample_errors, unresolved_defines);

    let preset = Preset {
        bank: 0,
        program: 0,
        name: path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        regions: region_ids,
        key_index: Vec::new(),
        key_regions: Vec::new(),
    };

    let mut bank = Bank {
        uses_rand: false,
        uses_lfo: false,
        uses_lfo_volume: false,
        uses_lfo_pitch: false,
        uses_mod_env: false,
        pool,
        pool_rate,
        samples,
        regions,
        params: Vec::new(),
        menv: Vec::new(),
        menv_factors: Vec::new(),
        menv_factor_half: 0,
        menv_log2: Vec::new(),
        gain_table: Vec::new(),
        gain_volume: 0.0,
        delay_frames: Vec::new(),
        key_ok: Vec::new(),
        presets: vec![preset],
        index: Vec::new(),
        name: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
    };
    bank.build_params(cfg);
    bank.finish();
    Ok(bank)
}

/// How many distinct sample failures to name before summarising the rest.
const SAMPLE_ERROR_LINES: usize = 8;

/// Say what failed to load, once per distinct reason, with the region count.
fn report_sample_errors(errors: HashMap<String, u32>, unresolved_defines: u32) {
    if errors.is_empty() {
        return;
    }
    let dropped: u32 = errors.values().sum();
    let mut list: Vec<(String, u32)> = errors.into_iter().collect();
    // Worst first, then by name so two runs of one library agree.
    list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    log::warn!(
        "sfz: {dropped} regions dropped, {} distinct sample failures:",
        list.len()
    );
    for (msg, n) in list.iter().take(SAMPLE_ERROR_LINES) {
        log::warn!("  {msg} ({n} regions)");
    }
    if list.len() > SAMPLE_ERROR_LINES {
        log::warn!("  and {} more", list.len() - SAMPLE_ERROR_LINES);
    }
    if unresolved_defines > 0 {
        log::warn!(
            "sfz: {unresolved_defines} of those paths still contain `$`, so a \
             `#define` was used before it was written or its name is misspelt"
        );
    }
}

const POOL_GUARD: usize = 8;

fn load_sample_channels(
    spath: &Path,
    pool: &mut Vec<i16>,
    samples: &mut Vec<SampleInfo>,
    cache: &mut HashMap<(PathBuf, usize), u32>,
    pool_rate: u32,
) -> Result<Vec<u32>> {
    if let Some(&first) = cache.get(&(spath.to_path_buf(), 0)) {
        let mut out = vec![first];
        if let Some(&second) = cache.get(&(spath.to_path_buf(), 1)) {
            out.push(second);
        }
        return Ok(out);
    }

    let w = wav::read(spath)?;
    let nch = (w.channels as usize).clamp(1, 2);
    let mut out = Vec::with_capacity(nch);

    for ch in 0..nch {
        let raw = w.channel_i16(ch);
        let src_rate = w.sample_rate.max(1);
        let ratio = if pool_rate != 0 && pool_rate != src_rate {
            pool_rate as f64 / src_rate as f64
        } else {
            1.0
        };
        let data = if (ratio - 1.0).abs() > 1e-12 {
            resample::resample_i16(&raw, ratio)
        } else {
            raw
        };

        let len = data.len() as u32;
        // [16]
        let declared_loop = w.loop_points.is_some();
        let (mut ls, mut le) = w.loop_points.unwrap_or((0, len.saturating_sub(1)));
        ls = (ls as f64 * ratio).round() as u32;
        le = (le as f64 * ratio).round() as u32;

        let start = pool.len() as u32;
        pool.extend_from_slice(&data);
        pool.extend(std::iter::repeat_n(0i16, POOL_GUARD));

        let idx = samples.len() as u32;
        samples.push(SampleInfo {
            start,
            len,
            loop_start: ls.min(len.saturating_sub(1)),
            loop_end: le.min(len),
            declared_loop,
            rate: if pool_rate != 0 { pool_rate } else { src_rate },
            root_key: w.root_key.unwrap_or(60),
            correction_cents: w.fine_tune_cents,
            resample_ratio: ratio as f32,
            name: spath.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        });
        cache.insert((spath.to_path_buf(), ch), idx);
        out.push(idx);
    }
    Ok(out)
}

/// Every opcode this loader reads. Anything outside it is dropped, and being \[17\]
const KNOWN_OPCODES: &[&str] = &[
    "sample",
    "lokey",
    "hikey",
    "key",
    "lovel",
    "hivel",
    "pitch_keycenter",
    "tune",
    "pitch",
    "transpose",
    "volume",
    "amp_veltrack",
    "pan",
    "pitch_keytrack",
    "loop_mode",
    "loopmode",
    "loop_start",
    "loop_end",
    "offset",
    "end",
    "ampeg_delay",
    "ampeg_attack",
    "ampeg_hold",
    "ampeg_decay",
    "ampeg_sustain",
    "ampeg_release",
    "cutoff",
    "resonance",
    "fil_veltrack",
    "fil_type",
    "group",
    "off_by",
    "default_path",
    "lorand",
    "hirand",
    "xfin_lovel",
    "xfin_hivel",
    "xfout_lovel",
    "xfout_hivel",
    "amplfo_freq",
    "amplfo_depth",
    "amplfo_delay",
    "amplfo_fade",
    "pitchlfo_freq",
    "pitchlfo_depth",
    "pitchlfo_delay",
    "pitchlfo_fade",
    // [18]
    "fillfo_freq",
    "fillfo_delay",
    "fillfo_depth",
    "fillfo_fade",
];

/// Opcodes recognised as deliberately unimplemented, grouped by the feature \[19\]
const UNIMPL_TAG: char = '';

const UNIMPLEMENTED_GROUPS: &[(&str, &[&str])] = &[
    // [20]
    (
        "LFO controller modulation",
        &["amplfo_", "fillfo_", "pitchlfo_", "pitchlfo", "amplfo", "fillfo", "lfo"],
    ),
    // [21]
    ("effects", &["reverb_", "chorus_", "delay_", "send_effect", "send", "effect"]),
    // [22]
    (
        "envelope veltrack",
        &[
            "ampeg_veltrack",
            "ampeg_attack_veltrack",
            "ampeg_decay_veltrack",
            "ampeg_sustain_veltrack",
            "ampeg_release_veltrack",
            "ampeg_delay_veltrack",
            "ampeg_hold_veltrack",
            "ampeg_vel2",
            "gain_veltrack",
        ],
    ),
    // [23]
    ("voice masking", &["note_selfmask", "note_polyphony", "polyphony"]),
    // [24]
    ("crossfade curve", &["xf_velcurve", "xf_keycurve", "xf_cccurve"]),
    ("CC labelling", &["set_cc", "label_cc", "label_key"]),
];

/// Controller ranges: `loccN`/`hiccN`, which gate a region on a controller's \[25\]
const SILENT_CC_RANGES: &[&str] =
    &["locc", "hicc", "xfin_locc", "xfin_hicc", "xfout_locc", "xfout_hicc"];

fn silently_ignored(op: &str) -> bool {
    SILENT_CC_RANGES.iter().any(|p| {
        op.strip_prefix(p)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    })
}

fn note_unhandled(ops: &OpcodeSet, unknown: &mut HashMap<String, u32>) {
    for k in ops.0.keys() {
        if KNOWN_OPCODES.contains(&k.as_str()) || silently_ignored(k) {
            continue;
        }
        let group = UNIMPLEMENTED_GROUPS
            .iter()
            .find(|(_, pats)| pats.iter().any(|p| k.starts_with(*p)));
        let key = match group {
            Some((name, _)) => format!("{UNIMPL_TAG}{name}"),
            None => k.clone(),
        };
        *unknown.entry(key).or_default() += 1;
    }
}

fn region_from_opcodes(
    ops: &OpcodeSet,
    sample: u32,
    info: &SampleInfo,
    unhandled: &mut Vec<String>,
) -> Region {
    let mut r = Region {
        sample,
        ..Default::default()
    };

    // [26]
    if let Some(k) = ops.key("lokey") {
        r.key_lo = k.clamp(0, 127) as u8;
    }
    if let Some(k) = ops.key("hikey") {
        r.key_hi = k.clamp(0, 127) as u8;
    }
    if let Some(k) = ops.key("pitch_keycenter") {
        r.root_key_override = k.clamp(0, 127) as i16;
    }
    if r.root_key_override < 0 {
        r.root_key_override = info.root_key as i16;
    }
    if let Some(v) = ops.i32("lovel") {
        r.vel_lo = v.clamp(0, 127) as u8;
    }
    if let Some(v) = ops.i32("hivel") {
        r.vel_hi = v.clamp(0, 127) as u8;
    }

    let tune = ops.f32("tune").unwrap_or(0.0);
    r.fine_tune = tune.round().clamp(-32768.0, 32767.0) as i16;
    if let Some(t) = ops.i32("transpose") {
        r.coarse_tune = t.clamp(-127, 127) as i16;
    }
    if let Some(s) = ops.f32("pitch_keytrack") {
        r.scale_tuning = s.round().clamp(-1200.0, 1200.0) as i16;
    }

    if let Some(v) = ops.f32("volume") {
        r.attenuation_cb = -v * 10.0;
    }
    if let Some(p) = ops.f32("pan") {
        r.pan = (p / 100.0).clamp(-1.0, 1.0);
    }
    if let Some(v) = ops.f32("amp_veltrack") {
        r.amp_veltrack = v.clamp(-100.0, 100.0);
    }

    // [27]
    r.loop_mode = match ops.get("loop_mode").unwrap_or("") {
        "loop_continuous" => LoopMode::Continuous,
        "loop_sustain" => LoopMode::UntilRelease,
        "one_shot" => LoopMode::NoLoop,
        "no_loop" => LoopMode::NoLoop,
        "off" => LoopMode::NoLoop,
        "" => {
            if info.declared_loop {
                LoopMode::Continuous
            } else {
                LoopMode::NoLoop
            }
        }
        // [28]
        other => {
            unhandled.push(format!("loop_mode={other} (unknown, treated as no_loop)"));
            LoopMode::NoLoop
        }
    };

    // [29]
    if let Some(v) = ops.i32("xfin_lovel") {
        r.xfin_lo = v.clamp(0, 127) as u8;
    }
    if let Some(v) = ops.i32("xfin_hivel") {
        r.xfin_hi = v.clamp(0, 127) as u8;
    }
    if let Some(v) = ops.i32("xfout_lovel") {
        r.xfout_lo = v.clamp(0, 127) as u8;
    }
    if let Some(v) = ops.i32("xfout_hivel") {
        r.xfout_hi = v.clamp(0, 127) as u8;
    }
    // [30]
    if r.xfin_hi < r.xfin_lo {
        unhandled.push("xfin_hivel < xfin_lovel (ignored)".to_string());
        r.xfin_lo = 0;
        r.xfin_hi = 0;
    }
    if r.xfout_hi < r.xfout_lo {
        unhandled.push("xfout_hivel < xfout_lovel (ignored)".to_string());
        r.xfout_lo = 127;
        r.xfout_hi = 127;
    }

    // [31]
    if let Some(v) = ops.f32("lorand") {
        r.rand_lo = v.clamp(0.0, 1.0);
    }
    if let Some(v) = ops.f32("hirand") {
        r.rand_hi = v.clamp(0.0, 1.0);
    }
    if r.rand_hi < r.rand_lo {
        unhandled.push(format!(
            "lorand={} > hirand={} (empty range, ignored)",
            r.rand_lo, r.rand_hi
        ));
        // [32]
        r.rand_lo = 0.0;
        r.rand_hi = 1.0;
    }

    if let Some(o) = ops.i32("offset") {
        r.addr_start = o.max(0);
    }
    // [33]
    let to_source = |resampled: u32| {
        if info.resample_ratio > 0.0 {
            (resampled as f32 / info.resample_ratio).round() as i32
        } else {
            resampled as i32
        }
    };
    if let Some(e) = ops.i32("end") {
        if e > 0 {
            r.addr_end = e - to_source(info.len);
        }
    }
    if let Some(ls) = ops.i32("loop_start") {
        r.addr_loop_start = ls - to_source(info.loop_start);
    }
    if let Some(le) = ops.i32("loop_end") {
        r.addr_loop_end = le - to_source(info.loop_end);
    }

    // [34]
    if let Some(d) = ops.f32("amplfo_depth") {
        r.mod_lfo_to_volume = -10.0 * d;
        r.mod_lfo_hz = ops.f32("amplfo_freq").unwrap_or(0.0).max(0.0);
        r.mod_lfo_delay = ops.f32("amplfo_delay").unwrap_or(0.0).max(0.0);
    }
    if let Some(d) = ops.f32("pitchlfo_depth") {
        r.vib_lfo_to_pitch = d;
        r.vib_lfo_hz = ops.f32("pitchlfo_freq").unwrap_or(0.0).max(0.0);
        r.vib_lfo_delay = ops.f32("pitchlfo_delay").unwrap_or(0.0).max(0.0);
    }
    // [35]
    if ops.f32("fillfo_depth").is_some_and(|d| d != 0.0) {
        unhandled.push("fillfo_depth (the filter LFO is not implemented)".to_string());
    }
    for fade in ["amplfo_fade", "pitchlfo_fade", "fillfo_fade"] {
        if ops.f32(fade).is_some_and(|f| f > 0.0) {
            unhandled.push(format!("{fade} (LFO fade-in is not implemented)"));
        }
    }

    r.delay = ops.f32("ampeg_delay").unwrap_or(0.0).max(0.0);
    r.attack = ops.f32("ampeg_attack").unwrap_or(0.001).max(0.0);
    r.hold = ops.f32("ampeg_hold").unwrap_or(0.0).max(0.0);
    r.decay = ops.f32("ampeg_decay").unwrap_or(100.0).max(0.0);
    r.sustain = (ops.f32("ampeg_sustain").unwrap_or(100.0) / 100.0).clamp(0.0, 1.0);
    r.release = ops.f32("ampeg_release").unwrap_or(0.05).max(0.0);

    if let Some(hz) = ops.f32("cutoff") {
        if hz > 0.0 {
            r.filter_fc_cents = 1200.0 * (hz / 8.176).log2();
        }
    }
    if let Some(res) = ops.f32("resonance") {
        r.filter_q_cb = res * 10.0;
    }
    if let Some(vt) = ops.f32("fil_veltrack") {
        r.filter_veltrack_cents = vt.clamp(-9600.0, 9600.0);
    }
    // [36]
    if let Some(kind) = ops.get("fil_type") {
        if !kind.starts_with("lpf") {
            r.filter_fc_cents = 13500.0;
            r.filter_veltrack_cents = 0.0;
        }
    }
    // [37]
    let group_id = ops.i32("group").unwrap_or(0).clamp(0, 255);
    match ops.i32("off_by") {
        Some(off) if off.clamp(0, 255) == group_id && group_id > 0 => {
            r.exclusive_class = group_id as u8;
        }
        // [38]
        Some(_) => unhandled.push("off_by (cross-group, not implemented)".to_string()),
        None => {}
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_names() {
        assert_eq!(parse_note_name("c4"), Some(60));
        assert_eq!(parse_note_name("a4"), Some(69));
        assert_eq!(parse_note_name("c#4"), Some(61));
        assert_eq!(parse_note_name("c-1"), Some(0));
    }

    #[test]
    fn controller_ranges_are_ignored_quietly_and_nothing_else_is() {
        for op in ["locc1", "hicc64", "xfin_locc14", "xfin_hicc14", "xfout_locc20", "xfout_hicc127"] {
            assert!(silently_ignored(op), "{op}");
        }
        for op in ["locc", "xfin_lovel", "on_locc64", "hicc1x", "loccN", "xfin_locc", "xf_cccurve"] {
            assert!(!silently_ignored(op), "{op}");
        }
    }

    #[test]
    fn value_with_spaces_survives() {
        let s = "My Sample Name.wav lokey=30";
        let end = find_value_end(s);
        assert_eq!(s[..end].trim(), "My Sample Name.wav");
    }

    #[test]
    fn single_value_runs_to_end() {
        let s = "60";
        assert_eq!(find_value_end(s), 2);
    }

    fn parser() -> Parser {
        Parser {
            root: PathBuf::new(),
            unknown: HashMap::new(),
            depth: 0,
            defines: Vec::new(),
        }
    }

    /// Feed lines through the same `#define` reader `parse_file` uses.
    fn expand(lines: &[&str]) -> Vec<String> {
        let mut p = parser();
        let mut out = Vec::new();
        for line in lines {
            if let Some(rest) = line.strip_prefix("#define") {
                let rest = rest.trim_start();
                let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
                let (name, value) = rest.split_at(end);
                let value = p.substitute(value.trim());
                p.define(name.to_string(), value);
                continue;
            }
            out.push(p.substitute(line));
        }
        out
    }

    #[test]
    fn define_is_substituted_into_opcode_values() {
        let out = expand(&["#define $KL 64", "sample=WYV-$KL-64.wav"]);
        assert_eq!(out, ["sample=WYV-64-64.wav"]);
    }

    /// `$KEY` and `$KEYS` can both be defined. Replacing the shorter one first \[39\]
    #[test]
    fn longest_name_wins() {
        let out = expand(&["#define $KEY a", "#define $KEYS b", "sample=$KEYS/$KEY.wav"]);
        assert_eq!(out, ["sample=b/a.wav"]);
    }

    /// Redefinition takes effect from that point on, which is how a library \[40\]
    #[test]
    fn redefinition_applies_from_that_point() {
        let out = expand(&["#define $L 1", "a=$L", "#define $L 2", "b=$L"]);
        assert_eq!(out, ["a=1", "b=2"]);
    }

    #[test]
    fn value_runs_to_end_of_line() {
        let out = expand(&["#define $D My Samples/v1", "sample=$D/s.wav"]);
        assert_eq!(out, ["sample=My Samples/v1/s.wav"]);
    }

    #[test]
    fn a_define_may_use_an_earlier_one() {
        let out = expand(&["#define $R root", "#define $P $R/v1", "sample=$P/s.wav"]);
        assert_eq!(out, ["sample=root/v1/s.wav"]);
    }

    /// Left in place rather than blanked, so it survives into the resolved \[41\]
    #[test]
    fn an_undefined_name_survives_for_the_report() {
        let out = expand(&["#define $A a", "sample=$A-$NOPE.wav"]);
        assert_eq!(out, ["sample=a-$NOPE.wav"]);
    }
}
