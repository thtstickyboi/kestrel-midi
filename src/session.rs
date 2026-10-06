// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A whole render, and the live view of it that front ends read. \[1\]

use crate::backend::{Backend, BlockStats};
use crate::config::{BackendKind, Config};
use crate::limiter::LimiterMode;
use crate::falconeye::{observe, renderlog};
use crate::single;
use crate::{bank::Bank, cpu::CpuSynth, driver::Driver, gpu, load_bank, wav};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The log target every line here is written under. This code lived in the \[2\]
const TARGET: &str = "kestrel";

/// Parse `--sf-programs`: a comma-separated list of programs and `a-b` ranges.
fn parse_programs(spec: &str) -> Result<Vec<u16>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (u16, u16) = (a.trim().parse()?, b.trim().parse()?);
                if a > b {
                    bail!("--sf-programs range {part:?} runs backwards");
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse()?),
        }
    }
    if out.is_empty() {
        bail!("--sf-programs is empty");
    }
    if let Some(bad) = out.iter().find(|p| **p > 127) {
        bail!("--sf-programs {bad} is out of range; GM programs are 0-127");
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Load one or more soundfonts and layer them in order. \[3\]
pub fn load_layered(paths: &[PathBuf], programs: Option<&str>, cfg: &Config) -> Result<Bank> {
    let (first, rest) = paths.split_first().context("no soundfont given")?;
    let bank = load_bank(first, cfg)?;
    stack(bank, rest.len(), rest.iter().map(|p| load_bank(p, cfg)), programs, cfg)
}

/// Layer banks that are already loaded, or that `rest` loads as it is pulled. \[4\]
pub fn stack(
    mut bank: Bank,
    rest_len: usize,
    rest: impl Iterator<Item = Result<Bank>>,
    programs: Option<&str>,
    cfg: &Config,
) -> Result<Bank> {
    if rest_len == 0 {
        if let Some(spec) = programs {
            let progs = parse_programs(spec)?;
            bank.remap_to_programs(&progs)?;
            bank.build_params(cfg);
            bank.finish();
        }
        return Ok(bank);
    }
    log::info!(target: TARGET, "layer 1: {}", bank.describe());
    for (i, top) in rest.enumerate() {
        let mut top = top?;
        log::info!(target: TARGET, "layer {}: {}", i + 2, top.describe());
        // Only the last layer is remapped: it is the override.
        if i + 1 == rest_len {
            if let Some(spec) = programs {
                let progs = parse_programs(spec)?;
                top.remap_to_programs(&progs)?;
            }
        }
        // [5]
        crate::bank::check_pool_frames(
            (bank.pool.len() + top.pool.len()) as u64,
            bank.pool_rate,
            "these soundfonts together",
        )?;
        bank.merge(top);
    }
    // Everything derived is rebuilt from the merged whole.
    bank.build_params(cfg);
    bank.finish();
    Ok(bank)
}

/// What an output path asks Kestrel to write.
#[derive(Debug)]
pub enum Target {
    /// RIFF/WAVE, written directly.
    Wav,
    /// Encoded through an ffmpeg pipe.
    Encoded(&'static crate::ffmpeg::Preset),
}

/// Every extension Kestrel writes, for error messages and near-miss matching.
fn supported_exts() -> Vec<&'static str> {
    let mut v = vec!["wav"];
    v.extend(crate::ffmpeg::PRESETS.iter().map(|p| p.ext));
    v
}

/// Levenshtein distance, for "did you mean". Small and bounded -- both inputs \[6\]
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Decide what to write from the output path, and reject anything Kestrel \[7\]
pub fn resolve_output_target(path: &Path) -> Result<Target> {
    // [8]
    const UNSUPPORTED: &[&str] = &[
        "aac", "alac", "wma", "aiff", "aif", "aifc", "wv", "spx", "mka", "caf", "oga", "ac3",
        "amr", "ra", "au", "mp2", "mpc", "tta", "dsf",
    ];

    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    if ext == "wav" {
        return Ok(Target::Wav);
    }
    if let Some(p) = crate::ffmpeg::preset_for(&ext) {
        return Ok(Target::Encoded(p));
    }

    let supported = supported_exts().join(", .");
    if ext.is_empty() {
        bail!(
            "{}: the output path has no extension.\n\
             Supported: .{supported}",
            path.display()
        );
    }
    if UNSUPPORTED.contains(&ext.as_str()) {
        bail!(
            "{}: .{ext} is not one of the containers Kestrel writes.\n\
             Supported: .{supported}\n\
             For anything else, render to .wav and convert it.",
            path.display()
        );
    }

    // [9]
    let hint = supported_exts()
        .into_iter()
        .map(|s| (edit_distance(&ext, s), s))
        .filter(|(d, _)| *d <= 2)
        .min();
    match hint {
        Some((_, best)) => bail!(
            "{}: unrecognised output extension .{ext}. Did you mean .{best}?\n\
             Supported: .{supported}",
            path.display()
        ),
        None => bail!(
            "{}: unrecognised output extension .{ext}.\n\
             Supported: .{supported}",
            path.display()
        ),
    }
}

/// Where a render's blocks go: straight to a WAV, or through an encoder. \[10\]
pub(crate) enum Sink {
    Wav(wav::WavWriter),
    Encoded(Box<crate::ffmpeg::Encoder>),
}

impl Sink {
    /// The file a render writes, as `plan` settled it.
    pub(crate) fn create(
        path: &Path,
        cfg: &Config,
        encoder: Option<&(crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
        wav_format: wav::SampleFormat,
    ) -> Result<Self> {
        Ok(match encoder {
            Some((f, preset)) => Sink::Encoded(Box::new(f.encode_to(path, cfg.sample_rate, preset)?)),
            None => Sink::Wav(wav::WavWriter::create(path, cfg.sample_rate, 2, wav_format)?),
        })
    }

    pub(crate) fn write_block(&mut self, block: &[f32]) -> Result<()> {
        match self {
            Sink::Wav(w) => w.write_block(block),
            Sink::Encoded(e) => e.write_block(block),
        }
    }

    pub(crate) fn finish(self) -> Result<u64> {
        match self {
            Sink::Wav(w) => w.finish(),
            Sink::Encoded(e) => e.finish(),
        }
    }
}

/// Where a render is. The soundfont load and the device setup each take \[11\]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Nothing has happened yet.
    #[default]
    Starting,
    LoadingSoundfont,
    OpeningMidi,
    PreparingDevice,
    Rendering,
    /// The last block is written; the file is being closed, which for an \[12\]
    Finishing,
    /// The file is complete. This and the two after it are reported by a \[13\]
    Finished,
    /// Stopped early through [`Observer::cancelled`]. The file is closed and \[14\]
    Cancelled,
    /// Stopped by an error; [`Event::Failed`] carries it.
    Failed,
}

/// What is known once the render is set up, before its first block.
#[derive(Debug, Clone, Serialize)]
pub struct Setup {
    /// `"gpu"` or `"cpu"`.
    pub backend: &'static str,
    /// The adapter a GPU render landed on.
    pub adapter: Option<String>,
    /// Device buffers this render allocated, for a GPU render.
    pub device_bytes: Option<u64>,
    /// PCI vendor and device id of that adapter.
    pub vendor_id: Option<u32>,
    pub device_id: Option<u32>,
    pub tracks: u16,
    pub max_voices: u32,
    /// Track data in the MIDI: what [`Snapshot::progress`] divides by.
    pub bytes_total: u64,
}

/// The loop, as it stands after one block.
pub struct Tick<'a> {
    pub cfg: &'a Config,
    pub driver: &'a Driver,
    pub backend: &'a dyn Backend,
    pub stats: BlockStats,
    pub peak_voices: u64,
    /// When the first block started.
    pub start: Instant,
    /// This is the last block: the file ran out or `Job::seconds` was reached. \[15\]
    pub last: bool,
    /// Seconds of audio a resumed render had already rendered when this run \[16\]
    pub resumed_secs: f64,
}

/// The share of a per-track render's progress that is its notes read; the \[17\]
pub const PER_TRACK_NOTE_SHARE: f64 = 0.55;

/// A per-track render as it stands, which [`Observer::tracks`] is handed \[18\]
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrackProgress {
    /// Tracks the render has, has finished, and is rendering now.
    pub total: usize,
    pub done: usize,
    pub running: usize,
    /// Finished tracks that had to steal voices.
    pub stole: usize,
    /// Blocks rendered over every track; those of them that were silent and \[19\]
    pub blocks: u64,
    pub silent_blocks: u64,
    pub blocks_total: u64,
    /// Blocks rendered inside each track's notes -- from the block its first \[20\]
    pub span_blocks: u64,
    pub span_total: u64,
    /// Note-ons the tracks hold in all, after `--min-velocity`: where `notes` \[21\]
    pub notes_total: u64,
    /// Seconds of audio each track renders: the file's length, or \[22\]
    pub length_secs: f64,
    /// Voices each track may hold: the total, split evenly.
    pub voices_each: u32,
    /// One file, or a file a track.
    pub merged: bool,
    /// Audio rendered over every track, and the rest summed over them too.
    pub audio_secs: f64,
    pub notes: u64,
    pub stolen: u64,
    pub dropped: u64,
    /// The loudest track, before any limiter. Merged, the mix's own peak is \[23\]
    pub peak_level: f32,
    /// Microseconds the longest batch of tracks took, from going up to the \[24\]
    #[serde(skip)]
    pub longest_wait_us: u64,
    /// Up to three of the busiest tracks still rendering.
    pub now: Vec<TrackNow>,
}

/// One track in flight.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrackNow {
    /// Numbered from 1, as `kestrel tracks` lists them.
    pub track: usize,
    pub name: Option<String>,
    /// How far into the file it has rendered.
    pub secs: f64,
}

/// A front end's view of a render. Every method has a default, so an observer \[25\]
pub trait Observer {
    fn phase(&mut self, _phase: Phase) {}
    fn setup(&mut self, _setup: &Setup) {}
    /// Called after every block. Cheap counters only are in `Tick`; anything \[26\]
    fn block(&mut self, _tick: &Tick) {}
    /// A per-track render's progress, from the thread `run` was called on.
    fn tracks(&mut self, _progress: &TrackProgress) {}
    /// Polled during analytic preparation and once per block. True stops after \[27\]
    fn cancelled(&self) -> bool {
        false
    }
    /// With [`cancelled`](Observer::cancelled): the stop does not want its \[28\]
    fn discard_progress(&self) -> bool {
        false
    }
}

/// How a render ended.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Summary {
    pub bytes: u64,
    pub audio_secs: f64,
    pub wall_secs: f64,
    pub notes: u64,
    /// Note-ons `--min-velocity` skipped. Not in `notes`.
    pub notes_skipped: u64,
    pub voices_spawned: u64,
    pub peak_voices: u64,
    pub stolen: u64,
    pub dropped: u64,
    /// Before limiting, as the command line reports it.
    pub peak_level: f32,
    pub clipped: u64,
    /// Stopped by the observer rather than by the end of the file.
    pub cancelled: bool,
    /// Where a render that was stopped left its progress, the `.krsm` that \[29\]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<PathBuf>,
    /// A stop that was asked to keep none of its progress \[30\]
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub discarded: bool,
}

/// A render, described completely: what to read, what to write, and how. \[31\]
#[derive(Debug, Clone)]
pub struct Job {
    pub midi: PathBuf,
    /// Layered in order, each merged on top of the ones before it.
    pub soundfonts: Vec<PathBuf>,
    /// Programs of bank 0 the last soundfont takes over, spelled the way \[32\]
    pub sf_programs: Option<String>,
    /// The extension picks the container; see [`resolve_output_target`].
    pub out: PathBuf,
    /// An ffmpeg to encode with, instead of searching for one.
    pub ffmpeg: Option<PathBuf>,
    /// The sample format of a WAV. Encoded containers take float and refuse \[33\]
    pub wav_format: wav::SampleFormat,
    /// Brickwall ceiling in dBFS, when one was chosen. `None` keeps `cfg`'s \[34\]
    pub ceiling_db: Option<f64>,
    /// Stop after this many seconds of audio.
    pub seconds: Option<f64>,
    /// Write one CSV row per block here: the admission diagnostic.
    pub block_csv: Option<PathBuf>,
    /// Render this one track alone rather than the whole file: its own notes \[35\]
    pub track: Option<crate::tracks::TrackPick>,
    /// Render every track `stems.tracks` names, each alone and to a file of \[36\]
    pub stems: Option<crate::tracks::Stems>,
    /// Checkpoints of a render of one file or one track: where, how often, and \[37\]
    pub checkpoint: Option<crate::resume::Spec>,
    pub backend: BackendKind,
    pub cfg: Config,
}

/// Everything settled before the soundfont is touched.
pub struct Plan {
    pub cfg: Config,
    pub kind: BackendKind,
    pub(crate) encoder: Option<(crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
}

/// Validate a render and resolve its output, without loading anything. \[38\]
pub fn plan(job: &Job) -> Result<Plan> {
    let mut cfg = job.cfg.clone();
    if let Some(db) = job.ceiling_db {
        cfg.limiter_ceiling_db = db;
    }
    if job.track.is_some() && job.stems.is_some() {
        bail!("--track renders one track and --tracks renders stems; give one of them");
    }
    // [39]
    if (job.track.is_some() || job.stems.is_some())
        && cfg.phase != crate::phase::PhaseSettings::default()
    {
        log::warn!(
            target: TARGET,
            "--phase-* is ignored when rendering by track: analytic phase rotation is not \
             available per track yet, so the tracks render without it"
        );
        cfg.phase = crate::phase::PhaseSettings::default();
    }
    if job.stems.is_some() && job.block_csv.is_some() {
        bail!("--block-csv follows one render's blocks, and --tracks runs many at once");
    }
    if job.stems.is_some() && cfg.max_voices > crate::tracks::MAX_TOTAL_VOICES {
        log::warn!(
            target: TARGET,
            "--max-voices {} is over the most a --tracks render takes in all; {}",
            cfg.max_voices,
            crate::tracks::MAX_TOTAL_VOICES
        );
        cfg.max_voices = crate::tracks::MAX_TOTAL_VOICES;
    }
    // [40]
    let stems = job.stems.as_ref().filter(|s| !s.merge);
    if let Some(pick) = job.track {
        // [41]
        let h = crate::midi::SmfHeader::read(&job.midi)?;
        if pick.index >= h.tracks.len() {
            bail!(
                "{}: there is no track {}; the file has {}",
                job.midi.display(),
                pick.index + 1,
                h.tracks.len()
            );
        }
    }
    cfg.validate()?;
    let kind = job.backend;
    // [42]
    let target = match stems {
        Some(stems) => resolve_output_target(Path::new(&format!("stem.{}", stems.ext)))?,
        None => resolve_output_target(&job.out)?,
    };
    // [43]
    if stems.is_some()
        && matches!(target, Target::Wav)
        && job.wav_format == wav::SampleFormat::Float32
        && cfg.limiter_mode != LimiterMode::Off
    {
        log::info!(
            target: TARGET,
            "stems are 32-bit float WAV, so each is written without the limiter and they add \
             back up to the mix; --format pcm16 or an encoded --stem-format limits each one"
        );
        cfg.limiter_mode = LimiterMode::Off;
        cfg.clamp_output = false;
    }

    let encoder = match &target {
        Target::Wav => None,
        Target::Encoded(preset) => {
            // [44]
            if cfg.limiter_mode == LimiterMode::Off || !cfg.limiter {
                bail!(
                    "--limiter off cannot be used with .{}: encoding clamps anything \
                     above full scale, and Kestrel's raw mix routinely exceeds it \
                     (a recent merge peaked at 1088.9, +60.7 dBFS).\n\
                     Render to .wav for a raw mix, or drop --limiter off to encode.",
                    preset.ext
                );
            }
            // [45]
            if job.wav_format != wav::SampleFormat::Float32 {
                bail!(
                    "--format pcm16 applies to .wav output only; .{} is encoded from \
                     float and its sample format is the encoder's to choose.",
                    preset.ext
                );
            }
            // [46]
            if preset.lossy && job.ceiling_db.is_none() {
                cfg.limiter_ceiling_db = crate::ffmpeg::LOSSY_CEILING_DB;
            }
            let f = crate::ffmpeg::find(job.ffmpeg.as_deref())?;
            f.require(preset)?;
            log::info!(
                target: TARGET,
                "encoding .{} with {} via {} ({})",
                preset.ext,
                preset.note,
                f.path.display(),
                f.source.describe()
            );
            if preset.lossy {
                log::info!(
                    target: TARGET,
                    "limiter ceiling {:.2} dBFS{}",
                    cfg.limiter_ceiling_db,
                    if job.ceiling_db.is_some() {
                        " (yours)"
                    } else {
                        " (default for a lossy container: decoders overshoot, \
                         and 0 dBFS would clip on playback)"
                    }
                );
            }
            if preset.ext == "opus" && cfg.sample_rate != 48000 {
                log::warn!(
                    target: TARGET,
                    "Opus is always 48 kHz internally; this {} Hz render will be \
                     resampled by the encoder",
                    cfg.sample_rate
                );
            }
            Some((f, *preset))
        }
    };
    Ok(Plan { cfg, kind, encoder })
}

/// A soundfont that drives no LFO should not pay for the machinery. This turns \[47\]
pub(crate) fn fit_to_bank(cfg: &mut Config, bank: &Bank) {
    if !bank.uses_lfo {
        cfg.lfo_enabled = false;
    }
    if !bank.uses_mod_env {
        cfg.mod_env_enabled = false;
    }
}

/// Say what the soundfont holds once loaded: its sample pool is what goes to \[48\]
pub(crate) fn log_bank(bank: &Bank, took: Option<std::time::Duration>) {
    match took {
        Some(t) => log::info!(target: TARGET, "loaded {} in {t:.2?}", bank.describe()),
        None => log::info!(target: TARGET, "soundfont, loaded before the render: {}", bank.describe()),
    }
}

/// Render `args` as `plan` settled it. \[49\]
pub fn run(
    job: &Job,
    plan: Plan,
    preloaded: Option<Arc<Bank>>,
    obs: &mut dyn Observer,
) -> Result<Summary> {
    let log = observe::open_log(job, &plan);
    let result = match log {
        Some(_) => run_inner(job, plan, preloaded, &mut observe::Logged::new(obs)),
        None => run_inner(job, plan, preloaded, obs),
    };
    if log.is_some() {
        match &result {
            Ok(s) => renderlog::note("DONE", &format!("{s:?}")),
            Err(e) => renderlog::note("ERROR", &format!("{e:#}")),
        }
    }
    result
}

/// A file written for the 31-EDO template says so by a note key over 127, \[50\]
pub(crate) fn warn_extended_keys(cfg: &Config, midi: &Path) {
    if !cfg.edo31 && crate::midi::uses_extended_keys(midi).unwrap_or(false) {
        log::warn!(
            target: TARGET,
            "{} has note keys over 127: it is written for the 31-EDO template, and \
             without --31edo it plays as the wrong notes",
            midi.display()
        );
    }
}

fn run_inner(
    job: &Job,
    plan: Plan,
    preloaded: Option<Arc<Bank>>,
    obs: &mut dyn Observer,
) -> Result<Summary> {
    if job.stems.is_some() {
        return crate::stems::run(job, plan, preloaded, obs);
    }
    let Plan {
        mut cfg,
        kind,
        encoder,
    } = plan;

    // [51]
    let mut ckpt_spec = job.checkpoint.clone();
    if let Some(why) = single::unsupported(job, &cfg) {
        if ckpt_spec.as_ref().is_some_and(|s| s.restore.is_some()) {
            bail!("this render cannot be resumed: {why}");
        }
        if ckpt_spec.take().is_some() {
            log::warn!(target: TARGET, "this render cannot save its progress: {why}");
        }
    }
    let identity = match &ckpt_spec {
        Some(_) => Some(single::identity(job, &cfg)?),
        None => None,
    };
    let restore = ckpt_spec.as_ref().and_then(|s| s.restore.clone());
    if let (Some(cp), Some(id)) = (&restore, &identity) {
        cp.header.check(id)?;
    }
    let mut saver = match (&ckpt_spec, identity) {
        (Some(spec), Some(id)) => Some(single::Saver::new(spec, job, id, String::new(), 0)),
        _ => None,
    };

    obs.phase(Phase::LoadingSoundfont);
    let t0 = Instant::now();
    let loaded_here = preloaded.is_none();
    let bank = match preloaded {
        Some(bank) => bank,
        None => Arc::new(load_layered(&job.soundfonts,job.sf_programs.as_deref(), &cfg)?),
    };
    fit_to_bank(&mut cfg, &bank);
    log_bank(&bank, loaded_here.then(|| t0.elapsed()));

    let phase_started = Instant::now();
    let mut last_percent = usize::MAX;
    let phase = match crate::phase::PhaseBank::prepare_with(
        &bank, &cfg.phase, &|| obs.cancelled(), &mut |done, total| {
            let percent = done * 100 / total.max(1);
            if last_percent == usize::MAX || percent / 10 != last_percent / 10 {
                log::info!(target: TARGET, "analytic phase: {done}/{total} samples ({percent}%)");
                last_percent = percent;
            }
        },
    ) {
        Ok(phase) => phase,
        Err(e) if e.is::<crate::phase::PreparationCancelled>() => {
            return Ok(Summary { cancelled: true, wall_secs: t0.elapsed().as_secs_f64(), ..Default::default() });
        }
        Err(e) => return Err(e),
    };
    if cfg.phase.active() {
        log::info!(target: TARGET, "analytic phase: {} unique samples, {:.1} MiB cache, prepared in {:.2?}",
            phase.sample_count(), phase.cache_bytes() as f64 / 1048576.0, phase_started.elapsed());
    }
    obs.phase(Phase::OpeningMidi);
    warn_extended_keys(&cfg, &job.midi);
    // [52]
    if job.track.is_some() {
        cfg.max_block_candidates = crate::tracks::candidates_each(&cfg);
    }
    let mut driver = match job.track {
        None => {
            let d = Driver::open_prepared(&cfg, bank.clone(), &job.midi, phase.clone())?;
            log::info!(target: TARGET, "{} has {} tracks", job.midi.display(), d.track_count());
            d
        }
        Some(pick) => {
            let t0 = Instant::now();
            let scan = crate::tracks::scan(&job.midi, 0, None)?;
            let sel = scan.selection(pick.index, pick.setup)?;
            let t = &scan.tracks[pick.index];
            let setup = sel.tracks.len() - 1;
            log::info!(
                target: TARGET,
                "{} has {} tracks, scanned in {:.2?}; rendering track {}{}: {} notes, {}",
                job.midi.display(),
                scan.tracks.len(),
                t0.elapsed(),
                pick.index + 1,
                t.display_name().map(|n| format!(" ({n})")).unwrap_or_default(),
                t.notes(),
                match (pick.setup, setup) {
                    (crate::tracks::SetupTracks::Ignore, _) => "setup tracks ignored".to_string(),
                    (_, 0) => "no setup tracks".to_string(),
                    (_, 1) => "with the 1 setup track".to_string(),
                    (_, n) => format!("with the {n} setup tracks"),
                }
            );
            Driver::open_tracks_prepared(&cfg, bank.clone(), &job.midi, &sel, phase.clone())?
        }
    };

    // [53]
    let mut out = match (&saver, &restore) {
        (Some(s), Some(cp)) => {
            let single = cp.header.single.as_ref().context("the checkpoint is not of a render of one file")?;
            single::Output::resume(job, &cfg, encoder.as_ref(), &single.audio, &s.partial(&job.out), &s.pcm())?
        }
        (Some(s), None) => single::Output::create(job, &cfg, encoder.as_ref(), &s.partial(&job.out), &s.pcm())?,
        (None, _) => single::Output::plain(&job.out, &cfg, encoder.as_ref(), job.wav_format)?,
    };
    let mut block = vec![0.0f32; cfg.block_samples()];

    let max_frames = job
        .seconds
        .map(|s| (s * cfg.sample_rate as f64) as u64)
        .unwrap_or(u64::MAX);

    obs.phase(Phase::PreparingDevice);
    let (mut backend, adapter, device_bytes, ids): (Box<dyn Backend>, _, _, _) = match kind {
        BackendKind::Cpu => (Box::new(CpuSynth::new_prepared(&cfg, bank.clone(), phase.clone())), None, None, None),
        BackendKind::Gpu => {
            let g = gpu::GpuSynth::new_prepared(&cfg, bank.clone(), phase)?;
            let name = g.adapter_name().to_string();
            let bytes = g.vram_bytes();
            let ids = g.adapter_ids();
            (Box::new(g), Some(name), Some(bytes), Some(ids))
        }
    };
    log::info!(target: TARGET, "rendering with the {} backend", backend.name());
    // [54]
    let backend_id = match &adapter {
        Some(a) => format!("gpu: {a}"),
        None => "cpu".to_string(),
    };
    if let Some(s) = saver.as_mut() {
        s.set_run(backend_id.clone(), cfg.max_voices);
    }
    let mut peak_voices = 0u64;
    if let Some(cp) = &restore {
        if cp.header.backend != backend_id {
            bail!(
                "this checkpoint cannot be resumed: it was made on {} and this render runs on {backend_id}, \
                 and another backend or card can write different bytes for the same render",
                cp.header.backend
            );
        }
        if cp.header.voices_each != cfg.max_voices {
            bail!(
                "this checkpoint cannot be resumed: its render ran under {} voices and this render's limit is {}",
                cp.header.voices_each,
                cfg.max_voices
            );
        }
        let single = cp.header.single.as_ref().context("the checkpoint is not of a render of one file")?;
        let t0 = Instant::now();
        single::load(cp, &mut driver, backend.as_mut())
            .with_context(|| format!("resuming from {}", cp.path.display()))?;
        if driver.stats.blocks != single.blocks {
            bail!(
                "{} is damaged: it says {} blocks and holds {}",
                cp.path.display(),
                single.blocks,
                driver.stats.blocks
            );
        }
        peak_voices = single.peak_voices;
        log::info!(
            target: TARGET,
            "resuming {} at {:.2}s of audio ({} blocks); its state loaded in {:.2}s",
            cp.path.display(),
            driver.seconds_rendered(),
            single.blocks,
            t0.elapsed().as_secs_f64()
        );
    }
    obs.setup(&Setup {
        backend: backend.name(),
        adapter,
        device_bytes,
        vendor_id: ids.map(|(vendor, _)| vendor),
        device_id: ids.map(|(_, device)| device),
        tracks: driver.track_count(),
        max_voices: cfg.max_voices,
        bytes_total: driver.input_bytes().1,
    });

    let start = Instant::now();
    // Audio a resumed render had made before this run began: not this run's work.
    let resumed_secs = driver.seconds_rendered();
    obs.phase(Phase::Rendering);

    let mut csv = match &job.block_csv {
        Some(p) => {
            let mut f = std::io::BufWriter::new(std::fs::File::create(p)?);
            use std::io::Write;
            writeln!(f, "block,t,live,want,take,stolen,dropped,rms,peak,want_e,take_e")?;
            driver.measure_block_energy(true);
            Some(f)
        }
        None => None,
    };

    let mut cancelled = false;
    // [55]
    let mut discarded = false;
    loop {
        // [56]
        let done = driver.stats.blocks;
        let (more, held) = match saver.as_ref() {
            None => (driver.next_block(backend.as_mut(), &mut block)?, false),
            Some(s) => driver.next_block_or_hold(backend.as_mut(), &mut block, || s.hold(obs.cancelled(), done))?,
        };
        out.write_block(&block)?;

        let st = backend.stats();
        peak_voices = peak_voices.max(st.active_voices);

        if let Some(f) = csv.as_mut() {
            use std::io::Write;
            let n = block.len().max(1) as f64;
            let ss: f64 = block.iter().map(|v| *v as f64 * *v as f64).sum();
            let pk = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let d = &driver.stats;
            writeln!(
                f,
                "{},{:.6},{},{},{},{},{},{:.9},{:.9},{},{}",
                d.blocks,
                driver.seconds_rendered(),
                d.last_live,
                d.last_want,
                d.last_take,
                d.last_stolen,
                d.dropped,
                (ss / n).sqrt(),
                pk,
                d.last_want_energy,
                d.last_take_energy
            )?;
        }

        let last = !more || driver.stats.frames >= max_frames;
        obs.block(&Tick {
            cfg: &cfg,
            driver: &driver,
            backend: backend.as_ref(),
            stats: st,
            peak_voices,
            start,
            last,
            resumed_secs,
        });

        #[cfg(feature = "dev")]
        observe::crash_test(driver.stats.blocks, backend.as_mut());

        if last {
            break;
        }
        match saver.as_mut() {
            None => {
                if obs.cancelled() {
                    cancelled = true;
                    break;
                }
            }
            Some(s) => {
                // [57]
                if obs.cancelled() && obs.discard_progress() {
                    cancelled = true;
                    discarded = true;
                    break;
                }
                let stopping = obs.cancelled() || s.stop_hook(driver.stats.blocks);
                if held {
                    // [58]
                    match out.checkpoint() {
                        Ok(audio) => {
                            s.save_or_warn(&driver, backend.as_mut(), audio, peak_voices, stopping);
                        }
                        Err(e) => log::warn!(
                            target: TARGET,
                            "progress could not be saved to {}: {e:#}. The render goes on, and cannot be resumed from here",
                            s.path.display()
                        ),
                    }
                    if stopping {
                        cancelled = true;
                        break;
                    }
                }
                // [59]
            }
        }
    }

    obs.phase(Phase::Finishing);
    let bytes = out.finish()?;
    // [60]
    if let Some(s) = &saver {
        let partial = s.partial(&job.out);
        if cancelled {
            if encoder.is_some() || discarded {
                let _ = std::fs::remove_file(&partial);
            }
            if discarded {
                // [61]
                s.discard();
                log::warn!(
                    target: TARGET,
                    "stopped, and its progress was not kept: no resume file was written, and none is left"
                );
            }
        } else {
            std::fs::rename(&partial, &job.out)
                .with_context(|| format!("renaming {} to {}", partial.display(), job.out.display()))?;
            s.discard();
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let secs = driver.seconds_rendered();
    let st = backend.stats();
    if driver.stats.variant_states > 1 || driver.stats.variant_fallbacks > 0 {
        log::info!(
            target: TARGET,
            "sound controllers: {} distinct states, {} slots, {} rebuilds, {} approximated",
            driver.stats.variant_states,
            driver.stats.param_variants + 1,
            driver.stats.variant_rebuilds,
            driver.stats.variant_fallbacks
        );
    }
    log::info!(
        target: TARGET,
        "wrote {} ({:.1} MiB, {:.2}s audio) in {:.2}s = {:.2}x realtime",
        job.out.display(),
        bytes as f64 / 1048576.0,
        secs,
        wall,
        // Of what this run rendered: a resumed render's earlier audio was not.
        (secs - resumed_secs) / wall.max(1e-9)
    );
    log::info!(
        target: TARGET,
        "{} notes, {} voices spawned, peak {} concurrent, {} stolen, {} dropped, peak level {:.3}",
        driver.stats.notes,
        driver.stats.voices_spawned,
        peak_voices,
        st.stolen,
        // [62]
        driver.stats.dropped,
        driver.stats.peak
    );
    if cfg.min_velocity > 1 {
        log::info!(
            target: TARGET,
            "{} note-ons below velocity {} skipped, with their note-offs",
            driver.stats.notes_skipped,
            cfg.min_velocity
        );
    }
    if driver.stats.silent_blocks > 0 {
        log::info!(
            target: TARGET,
            "{} of {} blocks were silent and written without the device",
            driver.stats.silent_blocks,
            driver.stats.blocks
        );
    }
    if cfg.profile {
        let d = &driver.stats;
        let pct = |v: u64| {
            if d.us_total == 0 {
                0.0
            } else {
                v as f64 * 100.0 / d.us_total as f64
            }
        };
        let other = d
            .us_total
            .saturating_sub(d.us_drain + d.us_admit + d.us_spawn + d.us_render);
        log::info!(
            target: TARGET,
            "host/device split over {:.1}s inside next_block: \
             drain {:.1}s ({:.0}%)  admit {:.1}s ({:.0}%)  spawn {:.1}s ({:.0}%)  \
             render {:.1}s ({:.0}%)  other {:.1}s ({:.0}%)",
            d.us_total as f64 / 1e6,
            d.us_drain as f64 / 1e6,
            pct(d.us_drain),
            d.us_admit as f64 / 1e6,
            pct(d.us_admit),
            d.us_spawn as f64 / 1e6,
            pct(d.us_spawn),
            d.us_render as f64 / 1e6,
            pct(d.us_render),
            other as f64 / 1e6,
            pct(other),
        );
    }
    if !cfg.clamp_output {
        log::info!(
            target: TARGET,
            "final clamp is off: the limiter is off and the format is float32, so the file \
             holds the raw mix and may exceed +/-1.0. Its peak is {:.3}.",
            driver.output_peak()
        );
    }
    if driver.stats.clipped > 0 {
        log::warn!(
            target: TARGET,
            "{} samples were hard-clipped at full scale ({:.4}% of the render); each one \
             is a discontinuity. --limiter brickwall and --limiter omni both prevent them.",
            driver.stats.clipped,
            100.0 * driver.stats.clipped as f64
                / (driver.stats.frames.max(1) * cfg.channels as u64) as f64
        );
    }
    Ok(Summary {
        bytes,
        audio_secs: secs,
        wall_secs: wall,
        notes: driver.stats.notes,
        notes_skipped: driver.stats.notes_skipped,
        voices_spawned: driver.stats.voices_spawned,
        peak_voices,
        stolen: st.stolen,
        dropped: driver.stats.dropped,
        peak_level: driver.stats.peak,
        clipped: driver.stats.clipped,
        cancelled,
        checkpoint: saver.as_ref().filter(|_| cancelled).map(|s| s.path.clone()).filter(|p| p.exists()),
        discarded,
    })
}

/// Everything a front end can show about a render in progress, at one moment. \[63\]
#[derive(Debug, Clone, Default, Serialize)]
pub struct Snapshot {
    pub phase: Phase,
    /// Seconds in the current phase.
    pub phase_secs: f64,
    /// Seconds since the monitor was created.
    pub wall_secs: f64,
    /// Seconds since the first block started, or 0 before it has.
    pub render_secs: f64,
    /// Audio rendered so far.
    pub audio_secs: f64,
    /// Of `audio_secs`, what a resumed render had already made when this run \[64\]
    #[serde(skip)]
    pub resumed_secs: f64,
    /// Track data read so far, and in the whole file. A streamed MIDI has no \[65\]
    pub bytes_read: u64,
    pub bytes_total: u64,
    pub blocks: u64,
    /// Note-ons read so far.
    pub notes: u64,
    /// Voices sounding after the last block.
    pub voices: u64,
    pub max_voices: u32,
    pub peak_voices: u64,
    /// Voices killed to make room, cumulative.
    pub stolen: u64,
    /// Note-ons refused, cumulative.
    pub dropped: u64,
    /// Loudest sample so far, before the limiter.
    pub peak_level: f32,
    /// Samples the final clamp pulled back to full scale.
    pub clipped: u64,
    /// `"gpu"` or `"cpu"`, once the device is set up.
    pub backend: Option<String>,
    pub adapter: Option<String>,
    pub tracks: u16,
    /// Device buffers this render allocated. Not the whole GPU's usage; that \[66\]
    pub device_bytes: Option<u64>,
    /// The whole adapter's memory, every process on it counted, once the first \[67\]
    pub gpu_memory: Option<crate::gpu::vram::GpuMemory>,
    /// This process's resident memory now, and its highest since the monitor \[68\]
    pub host_rss_bytes: Option<u64>,
    pub host_rss_peak_bytes: Option<u64>,
    /// A per-track render's tracks, once it is rendering. Its sums are also \[69\]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_track: Option<TrackProgress>,
}

impl Snapshot {
    /// How far along, 0 to 1: the MIDI's track data read, once the file is \[70\]
    pub fn progress(&self) -> Option<f64> {
        if let Some(p) = &self.per_track {
            let spans = (p.span_total > 0).then(|| (p.span_blocks as f64 / p.span_total as f64).min(1.0));
            let notes = (p.notes_total > 0).then(|| (p.notes as f64 / p.notes_total as f64).min(1.0));
            return match (spans, notes) {
                (Some(s), Some(n)) => Some(PER_TRACK_NOTE_SHARE * n + (1.0 - PER_TRACK_NOTE_SHARE) * s),
                (Some(s), None) => Some(s),
                (None, _) => (p.blocks_total > 0).then(|| (p.blocks as f64 / p.blocks_total as f64).min(1.0)),
            };
        }
        (self.bytes_total > 0)
            .then(|| (self.bytes_read as f64 / self.bytes_total as f64).min(1.0))
    }

    /// Audio the render has made so far: `audio_secs`, or for a per-track \[71\]
    pub fn audio_done(&self) -> f64 {
        match &self.per_track {
            Some(p) => p.length_secs * self.progress().unwrap_or(0.0),
            None => self.audio_secs,
        }
    }

    /// Seconds of audio per second of rendering, over the render so far: the \[72\]
    pub fn speed(&self) -> Option<f64> {
        (self.render_secs > 0.2).then(|| (self.audio_done() - self.resumed_secs).max(0.0) / self.render_secs)
    }

    /// Seconds of rendering left, extrapolated from progress. `None` until \[73\]
    pub fn eta_secs(&self) -> Option<f64> {
        let p = self.progress()?;
        (p > 0.005 && p < 1.0 && self.render_secs > 1.0)
            .then(|| self.render_secs * (1.0 - p) / p)
    }
}

/// Something that happened once, as opposed to a counter that moves.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// `t` is seconds since the monitor was created.
    Phase { t: f64, phase: Phase },
    Setup {
        t: f64,
        #[serde(flatten)]
        setup: Setup,
    },
    /// The render ended normally or was cancelled. Always the last event of a \[74\]
    Summary {
        t: f64,
        #[serde(flatten)]
        summary: Summary,
    },
    /// The render stopped on an error. Always the last event of one that did.
    Failed { t: f64, message: String },
}

/// A render's live state, shared between the thread rendering it and any \[75\]
pub struct Monitor {
    created: Instant,
    inner: Mutex<Inner>,
    cancel: AtomicBool,
    /// With `cancel`: the stop is not to be resumed. Set before it.
    discard: AtomicBool,
    rss_peak: AtomicU64,
}

#[derive(Default)]
struct Inner {
    snap: Snapshot,
    phase_since: Option<Instant>,
    render_started: Option<Instant>,
    events: Vec<Event>,
    /// Started when the render reports its adapter, and stopped when the \[76\]
    vram: Option<crate::gpu::vram::VramWatch>,
}

impl Monitor {
    pub fn new() -> Arc<Monitor> {
        Arc::new(Monitor {
            created: Instant::now(),
            inner: Mutex::new(Inner::default()),
            cancel: AtomicBool::new(false),
            discard: AtomicBool::new(false),
            rss_peak: AtomicU64::new(0),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // [77]
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn secs(&self, at: Instant) -> f64 {
        at.saturating_duration_since(self.created).as_secs_f64()
    }

    /// The render as of now.
    pub fn snapshot(&self) -> Snapshot {
        let now = Instant::now();
        let mut snap = {
            let inner = self.lock();
            let since = |t: Option<Instant>| {
                t.map_or(0.0, |t| now.saturating_duration_since(t).as_secs_f64())
            };
            let mut snap = inner.snap.clone();
            snap.phase_secs = since(inner.phase_since);
            snap.render_secs = since(inner.render_started);
            snap.gpu_memory = inner.vram.as_ref().and_then(|watch| watch.latest());
            snap
        };
        snap.wall_secs = self.secs(now);
        if let Some(m) = memory_stats::memory_stats() {
            let rss = m.physical_mem as u64;
            snap.host_rss_bytes = Some(rss);
            snap.host_rss_peak_bytes =
                Some(self.rss_peak.fetch_max(rss, Ordering::Relaxed).max(rss));
        }
        snap
    }

    /// Every event from index `from` on, and the index to pass next time. \[78\]
    pub fn events_since(&self, from: usize) -> (Vec<Event>, usize) {
        let inner = self.lock();
        let from = from.min(inner.events.len());
        (inner.events[from..].to_vec(), inner.events.len())
    }

    /// Ask the render to stop after the block it is on. A render that saves \[79\]
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    /// [`cancel`](Monitor::cancel), and the progress is not wanted: see \[80\]
    pub fn cancel_discarding(&self) {
        self.discard.store(true, Ordering::Release);
        self.cancel.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    pub fn is_discarding(&self) -> bool {
        self.discard.load(Ordering::Acquire)
    }

    /// The observer that feeds this monitor, to hand to [`run`].
    pub fn observer(self: &Arc<Self>) -> MonitorObserver {
        MonitorObserver {
            monitor: Arc::clone(self),
            last_walk: None,
        }
    }

    /// Record how a render ended. [`run_monitored`] does this; a caller that \[81\]
    pub fn finish(&self, result: &Result<Summary>) {
        let now = Instant::now();
        let t = self.secs(now);
        let (phase, last) = match result {
            Ok(s) => (
                if s.cancelled {
                    Phase::Cancelled
                } else {
                    Phase::Finished
                },
                Event::Summary {
                    t,
                    summary: s.clone(),
                },
            ),
            Err(e) => (
                Phase::Failed,
                Event::Failed {
                    t,
                    message: format!("{e:#}"),
                },
            ),
        };
        let mut inner = self.lock();
        inner.snap.phase = phase;
        inner.phase_since = Some(now);
        inner.events.push(Event::Phase { t, phase });
        inner.events.push(last);
    }
}

/// How often the observer walks every track to count the bytes read. The \[82\]
const BYTES_EVERY: Duration = Duration::from_millis(50);

/// The [`Observer`] half of a [`Monitor`].
pub struct MonitorObserver {
    monitor: Arc<Monitor>,
    last_walk: Option<Instant>,
}

impl Observer for MonitorObserver {
    fn phase(&mut self, phase: Phase) {
        let now = Instant::now();
        let t = self.monitor.secs(now);
        let mut inner = self.monitor.lock();
        // [83]
        if inner.snap.phase == phase {
            return;
        }
        inner.snap.phase = phase;
        inner.phase_since = Some(now);
        inner.events.push(Event::Phase { t, phase });
    }

    fn setup(&mut self, setup: &Setup) {
        let t = self.monitor.secs(Instant::now());
        let mut inner = self.monitor.lock();
        let snap = &mut inner.snap;
        snap.backend = Some(setup.backend.to_string());
        snap.adapter = setup.adapter.clone();
        snap.device_bytes = setup.device_bytes;
        snap.tracks = setup.tracks;
        snap.max_voices = setup.max_voices;
        snap.bytes_total = setup.bytes_total;
        if let (Some(vendor), Some(device)) = (setup.vendor_id, setup.device_id) {
            inner.vram = Some(crate::gpu::vram::VramWatch::start(vendor, device));
        }
        inner.events.push(Event::Setup {
            t,
            setup: setup.clone(),
        });
    }

    fn block(&mut self, tick: &Tick) {
        let now = Instant::now();
        let walk = tick.last
            || match self.last_walk {
                Some(t) => now.saturating_duration_since(t) >= BYTES_EVERY,
                None => true,
            };
        let bytes = if walk {
            self.last_walk = Some(now);
            Some(tick.driver.input_bytes())
        } else {
            None
        };
        let mut inner = self.monitor.lock();
        if inner.render_started.is_none() {
            inner.render_started = Some(tick.start);
        }
        let d = &tick.driver.stats;
        let snap = &mut inner.snap;
        snap.audio_secs = tick.driver.seconds_rendered();
        snap.resumed_secs = tick.resumed_secs;
        snap.blocks = d.blocks;
        snap.notes = d.notes;
        snap.voices = tick.stats.active_voices;
        snap.peak_voices = tick.peak_voices;
        snap.stolen = tick.stats.stolen;
        snap.dropped = d.dropped;
        snap.peak_level = d.peak;
        snap.clipped = d.clipped;
        if let Some((read, total)) = bytes {
            snap.bytes_read = read;
            snap.bytes_total = total;
        }
    }

    fn tracks(&mut self, p: &TrackProgress) {
        let mut inner = self.monitor.lock();
        if inner.render_started.is_none() {
            inner.render_started = Some(Instant::now());
        }
        let snap = &mut inner.snap;
        snap.audio_secs = p.audio_secs;
        snap.blocks = p.blocks;
        snap.notes = p.notes;
        snap.stolen = p.stolen;
        snap.dropped = p.dropped;
        snap.peak_level = p.peak_level;
        snap.per_track = Some(p.clone());
    }

    fn cancelled(&self) -> bool {
        self.monitor.is_cancelled()
    }

    fn discard_progress(&self) -> bool {
        self.monitor.is_discarding()
    }
}

/// [`run`], reporting into `monitor`, which is marked finished however the \[84\]
pub fn run_monitored(
    job: &Job,
    plan: Plan,
    preloaded: Option<Arc<Bank>>,
    monitor: &Arc<Monitor>,
) -> Result<Summary> {
    let mut obs = monitor.observer();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(job, plan, preloaded, &mut obs)
    }))
    .unwrap_or_else(|payload| {
        Err(anyhow::anyhow!(
            "the render stopped on an internal error: {}",
            panic_text(&*payload)
        ))
    });
    monitor.finish(&result);
    result
}

/// The message a panic was raised with.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
}

#[cfg(test)]
mod tests {
    use super::{edit_distance, resolve_output_target, Target};
    use std::path::Path;

    fn target(p: &str) -> Result<Target, String> {
        resolve_output_target(Path::new(p)).map_err(|e| e.to_string())
    }

    /// The bug this guards: `-o mix.mp3` wrote a RIFF/WAVE file named `.mp3` \[85\]
    #[test]
    fn wav_and_every_preset_resolve_and_nothing_else_does() {
        assert!(matches!(target("out/mix.wav"), Ok(Target::Wav)));
        // Case comes from the user's shell, not from us.
        assert!(matches!(target("out/mix.WAV"), Ok(Target::Wav)));
        assert!(matches!(target("out/my.mix.wav"), Ok(Target::Wav)));

        for p in crate::ffmpeg::PRESETS {
            let path = format!("out/mix.{}", p.ext);
            match target(&path) {
                Ok(Target::Encoded(got)) => assert_eq!(got.ext, p.ext),
                other => panic!("{path} should encode, got {other:?}", other = other.is_ok()),
            }
        }

        for bad in ["out/mix.xyz", "out/mix.aiff", "out/mix", "out/mix.wma"] {
            assert!(target(bad).is_err(), "{bad} should be rejected");
        }
    }

    /// A typo should name the thing it probably meant. These four are the ones \[86\]
    #[test]
    fn a_near_miss_extension_suggests_the_real_one() {
        for (typo, want) in [
            ("ogf", ".ogg"),
            ("opis", ".opus"),
            ("mo3", ".mp3"),
            ("fkac", ".flac"),
            ("wavv", ".wav"),
        ] {
            let e = target(&format!("out/mix.{typo}")).unwrap_err();
            assert!(
                e.contains(&format!("Did you mean {want}")),
                "{typo}: {e}"
            );
        }
    }

    /// A real container Kestrel does not write is a considered request, not a \[87\]
    #[test]
    fn a_real_but_unsupported_container_is_not_treated_as_a_typo() {
        let e = target("out/mix.aiff").unwrap_err();
        assert!(e.contains("not one of the containers"), "{e}");
        assert!(!e.contains("Did you mean"), "{e}");
    }

    /// Far-off junk gets the list and no guess.
    #[test]
    fn an_unrelated_extension_gets_no_suggestion() {
        let e = target("out/mix.xyz").unwrap_err();
        assert!(!e.contains("Did you mean"), "{e}");
        assert!(e.contains("Supported"), "{e}");
    }

    #[test]
    fn edit_distance_is_a_metric_on_the_cases_that_matter() {
        assert_eq!(edit_distance("opus", "opus"), 0);
        assert_eq!(edit_distance("opis", "opus"), 1);
        // [88]
        assert_eq!(edit_distance("fkac", "flac"), 1);
        // [89]
        assert_eq!(edit_distance("flca", "flac"), 2);
        assert_eq!(edit_distance("", "wav"), 3);
    }

    fn summary(cancelled: bool) -> super::Summary {
        super::Summary {
            bytes: 1,
            audio_secs: 2.0,
            wall_secs: 1.0,
            notes: 3,
            notes_skipped: 0,
            voices_spawned: 4,
            peak_voices: 5,
            stolen: 0,
            dropped: 0,
            peak_level: 0.5,
            clipped: 0,
            cancelled,
            checkpoint: None,
            discarded: false,
        }
    }

    /// A reader that looks rarely still sees every phase, in order; a reader \[90\]
    #[test]
    fn a_monitor_keeps_every_event_for_every_reader() {
        use super::{Event, Monitor, Observer, Phase, Setup};
        let monitor = Monitor::new();
        let mut obs = monitor.observer();
        obs.phase(Phase::LoadingSoundfont);
        obs.phase(Phase::OpeningMidi);
        obs.setup(&Setup {
            backend: "gpu",
            adapter: Some("test adapter".into()),
            device_bytes: Some(10),
            vendor_id: None,
            device_id: None,
            tracks: 3,
            max_voices: 7,
            bytes_total: 100,
        });
        obs.phase(Phase::Rendering);

        let (first, next) = monitor.events_since(0);
        assert_eq!((first.len(), next), (4, 4));

        monitor.finish(&Ok(summary(false)));
        let (rest, end) = monitor.events_since(next);
        assert_eq!(end, 6);
        assert!(matches!(rest[0], Event::Phase { phase: Phase::Finished, .. }));
        assert!(matches!(rest[1], Event::Summary { .. }));
        assert_eq!(monitor.events_since(0).0.len(), 6);

        let snap = monitor.snapshot();
        assert_eq!(snap.phase, Phase::Finished);
        assert_eq!((snap.tracks, snap.max_voices, snap.bytes_total), (3, 7, 100));
        assert_eq!(snap.adapter.as_deref(), Some("test adapter"));
        assert!(snap.host_rss_bytes.is_some_and(|b| b > 0));
    }

    #[test]
    fn a_failed_render_ends_on_its_error() {
        use super::{Event, Monitor, Phase};
        let monitor = Monitor::new();
        monitor.finish(&Err(anyhow::anyhow!("device lost")));
        let (events, _) = monitor.events_since(0);
        assert!(matches!(events[0], Event::Phase { phase: Phase::Failed, .. }));
        assert!(matches!(&events[1], Event::Failed { message, .. } if message == "device lost"));
    }

    #[test]
    fn cancelling_a_monitor_reaches_the_render_through_its_observer() {
        use super::{Monitor, Observer};
        let monitor = Monitor::new();
        let obs = monitor.observer();
        assert!(!obs.cancelled());
        monitor.cancel();
        assert!(obs.cancelled());
    }

    /// Progress, speed and the estimate say nothing until they mean \[91\]
    #[test]
    fn derived_figures_wait_until_they_mean_something() {
        use super::Snapshot;
        let mut s = Snapshot::default();
        assert_eq!((s.progress(), s.speed(), s.eta_secs()), (None, None, None));
        s.bytes_total = 1000;
        s.bytes_read = 250;
        s.render_secs = 10.0;
        s.audio_secs = 250.0;
        assert_eq!(s.progress(), Some(0.25));
        assert_eq!(s.speed(), Some(25.0));
        assert_eq!(s.eta_secs(), Some(30.0));
        s.bytes_read = 1000;
        assert_eq!(s.eta_secs(), None);
    }

    /// A per-track render's progress is its notes read and its blocks inside \[92\]
    #[test]
    fn a_per_track_render_is_measured_by_its_notes_and_its_file() {
        use super::{Snapshot, TrackProgress, PER_TRACK_NOTE_SHARE};
        let mut s = Snapshot {
            render_secs: 60.0,
            audio_secs: 100_000.0,
            per_track: Some(TrackProgress {
                blocks: 800,
                silent_blocks: 700,
                blocks_total: 1000,
                span_blocks: 100,
                span_total: 400,
                notes: 500,
                notes_total: 1000,
                length_secs: 240.0,
                ..Default::default()
            }),
            ..Default::default()
        };
        let p = PER_TRACK_NOTE_SHARE * 0.5 + (1.0 - PER_TRACK_NOTE_SHARE) * 0.25;
        assert_eq!(s.progress(), Some(p));
        assert_eq!(s.audio_done(), 240.0 * p);
        assert_eq!(s.speed(), Some(240.0 * p / 60.0));
        assert_eq!(s.eta_secs(), Some(60.0 * (1.0 - p) / p));
        // Cut short by --seconds, it has no notes total: the spans alone.
        s.per_track.as_mut().unwrap().notes_total = 0;
        assert_eq!(s.progress(), Some(0.25));
        // With no spans either, as a report from before them had: blocks.
        s.per_track.as_mut().unwrap().span_total = 0;
        assert_eq!(s.progress(), Some(0.8));
    }

    /// The progress feed's wire format is this serialisation, so it is pinned: \[93\]
    #[test]
    fn events_serialise_with_a_type_tag_and_snake_case_names() {
        use super::{Event, Phase};
        let phase = Event::Phase {
            t: 1.5,
            phase: Phase::LoadingSoundfont,
        };
        assert_eq!(
            serde_json::to_string(&phase).unwrap(),
            r#"{"type":"phase","t":1.5,"phase":"loading_soundfont"}"#
        );
        let end = serde_json::to_string(&Event::Summary {
            t: 2.0,
            summary: summary(true),
        })
        .unwrap();
        assert!(end.starts_with(r#"{"type":"summary","t":2.0,"bytes":1,"#), "{end}");
        assert!(end.contains(r#""cancelled":true"#), "{end}");
    }
}
