// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Per-track rendering: every track a render names, each rendered alone \[1\]

use crate::backend::Backend;
use crate::bank::Bank;
use crate::config::{BackendKind, Config};
use crate::cpu::CpuSynth;
use crate::driver::Driver;
use crate::limiter::LimiterMode;
use crate::mix::Mix;
use crate::gpu::{GpuBatch, GpuShared, LaneBackend, LANES_MAX};
use crate::midi::TrackSelection;
use crate::phase::PhaseBank;
use crate::session::{fit_to_bank, load_layered, Job, Observer, Phase, Plan, Setup, Sink, Summary, TrackNow, TrackProgress};
use crate::tracks::{self, TrackScan};
use crate::wav;
use anyhow::{anyhow, bail, Context, Result};
use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TARGET: &str = "kestrel";

/// Video memory left spare when deciding how many jobs fit, for the desktop \[2\]
const VRAM_MARGIN: u64 = 512 << 20;

/// One stem to render.
struct Stem {
    track: usize,
    sel: TrackSelection,
    /// Its file; unused when merging.
    path: PathBuf,
    /// What the log calls it: the file's name, or the track when merging.
    label: String,
    /// The track's name, for the progress screen.
    name: Option<String>,
    /// Note-ons it keeps after `--min-velocity`: the order stems start in.
    notes: u64,
    /// The blocks its notes span, both ends counted: from the one its first \[3\]
    span: (u64, u64),
}

/// How one stem went.
#[derive(Debug, Default)]
struct Done {
    bytes: u64,
    audio_secs: f64,
    notes: u64,
    notes_skipped: u64,
    voices_spawned: u64,
    peak_voices: u64,
    stolen: u64,
    dropped: u64,
    peak_level: f32,
    clipped: u64,
    silent_blocks: u64,
    blocks: u64,
    cancelled: bool,
}

/// A job's renderer. A GPU job keeps one batch of lanes for every stem it \[4\]
enum JobBackend {
    Gpu(Box<GpuBatch>),
    Cpu,
}

/// What the thread `run` was called on reads to report progress: counters \[5\]
struct Progress {
    blocks: AtomicU64,
    silent: AtomicU64,
    /// Blocks rendered inside their track's notes; see `Stem::span`.
    span: AtomicU64,
    notes: AtomicU64,
    stolen: AtomicU64,
    dropped: AtomicU64,
    /// The loudest track's peak, as f32 bits: magnitudes are never negative, \[6\]
    peak: AtomicU32,
    stole: AtomicUsize,
    /// Per lane or job: one more than the rank of the track in it, 0 when \[7\]
    slots: Vec<(AtomicUsize, AtomicU64)>,
}

/// What a track has already added to `Progress`, so each block adds only \[8\]
#[derive(Default)]
struct Seen {
    notes: u64,
    stolen: u64,
    dropped: u64,
    silent: u64,
}

impl Progress {
    fn new(slots: usize) -> Self {
        Progress {
            blocks: AtomicU64::new(0),
            silent: AtomicU64::new(0),
            span: AtomicU64::new(0),
            notes: AtomicU64::new(0),
            stolen: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            peak: AtomicU32::new(0),
            stole: AtomicUsize::new(0),
            slots: (0..slots).map(|_| (AtomicUsize::new(0), AtomicU64::new(0))).collect(),
        }
    }

    fn enter(&self, slot: usize, rank: usize) {
        self.slots[slot].1.store(0, Ordering::Relaxed);
        self.slots[slot].0.store(rank + 1, Ordering::Relaxed);
    }

    fn leave(&self, slot: usize) {
        self.slots[slot].0.store(0, Ordering::Relaxed);
    }

    /// The block `driver` just finished for `stem`, `stolen` being its track's \[9\]
    fn block(&self, slot: usize, seen: &mut Seen, stem: &Stem, driver: &Driver, stolen: u64) {
        let d = &driver.stats;
        let add = |counter: &AtomicU64, now: u64, before: &mut u64| {
            counter.fetch_add(now.saturating_sub(*before), Ordering::Relaxed);
            *before = now;
        };
        self.blocks.fetch_add(1, Ordering::Relaxed);
        if (stem.span.0..=stem.span.1).contains(&d.blocks.saturating_sub(1)) {
            self.span.fetch_add(1, Ordering::Relaxed);
        }
        add(&self.notes, d.notes, &mut seen.notes);
        add(&self.stolen, stolen, &mut seen.stolen);
        add(&self.dropped, d.dropped, &mut seen.dropped);
        add(&self.silent, d.silent_blocks, &mut seen.silent);
        self.peak.fetch_max(d.peak.abs().to_bits(), Ordering::Relaxed);
        self.slots[slot].1.store(d.blocks, Ordering::Relaxed);
    }
}

/// What every job reads, and where it reports.
struct Ctx<'a> {
    cfg: &'a Config,
    bank: &'a Arc<Bank>,
    phase: &'a Arc<PhaseBank>,
    encoder: Option<&'a (crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
    job: &'a Job,
    stems: &'a [Stem],
    order: &'a [usize],
    max_frames: u64,
    next: &'a AtomicUsize,
    finished: &'a AtomicUsize,
    cancel: &'a AtomicBool,
    failure: &'a Mutex<Option<anyhow::Error>>,
    results: &'a Mutex<Vec<Done>>,
    /// Where every block goes when merging, in place of a file per stem.
    mix: Option<&'a Mix>,
    progress: &'a Progress,
}

impl<'a> Ctx<'a> {
    /// The next stem to start, busiest first, with its rank in that order, \[10\]
    fn take(&self) -> Option<(usize, &'a Stem)> {
        if self.cancel.load(Ordering::Relaxed) {
            return None;
        }
        let rank = self.next.fetch_add(1, Ordering::Relaxed);
        let &i = self.order.get(rank)?;
        Some((rank, &self.stems[i]))
    }

    fn done(&self, stem: &Stem, d: Done, t0: Instant) {
        if d.stolen > 0 {
            self.progress.stole.fetch_add(1, Ordering::Relaxed);
        }
        let n = self.finished.fetch_add(1, Ordering::Relaxed) + 1;
        log::info!(
            target: TARGET,
            "[{n}/{}] {}: {} notes, peak {} voices, {} stolen, {:.2}s in {:.2}s{}",
            self.stems.len(),
            stem.label,
            d.notes,
            d.peak_voices,
            d.stolen,
            d.audio_secs,
            t0.elapsed().as_secs_f64(),
            if d.cancelled { ", cancelled" } else { "" }
        );
        self.results.lock().unwrap().push(d);
    }

    fn fail(&self, e: anyhow::Error) {
        self.failure.lock().unwrap().get_or_insert(e);
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// A stem in a lane of a GPU job's batch.
struct Running<'a> {
    stem: &'a Stem,
    driver: Driver,
    /// The stem's file; `None` when merging.
    out: Option<Sink>,
    block: Vec<f32>,
    peak_voices: u64,
    t0: Instant,
    /// Its lane, which is its slot in `Progress`, and what it has reported.
    slot: usize,
    seen: Seen,
}

impl<'a> Running<'a> {
    fn start(ctx: &Ctx, stem: &'a Stem, rank: usize, slot: usize) -> Result<Self> {
        ctx.progress.enter(slot, rank);
        Ok(Running {
            stem,
            driver: Driver::open_tracks_prepared(ctx.cfg, ctx.bank.clone(), &ctx.job.midi, &stem.sel, ctx.phase.clone())?,
            out: match ctx.mix {
                Some(_) => None,
                None => Some(Sink::create(&stem.path, ctx.cfg, ctx.encoder, ctx.job.wav_format)?),
            },
            block: vec![0.0f32; ctx.cfg.block_samples()],
            peak_voices: 0,
            t0: Instant::now(),
            slot,
            seen: Seen::default(),
        })
    }

    fn finish(self, ctx: &Ctx, stolen: u64, cancelled: bool) -> Result<()> {
        ctx.progress.leave(self.slot);
        let bytes = match self.out {
            Some(out) => out.finish()?,
            None => 0,
        };
        let d = &self.driver.stats;
        let done = Done {
            bytes,
            audio_secs: self.driver.seconds_rendered(),
            notes: d.notes,
            notes_skipped: d.notes_skipped,
            voices_spawned: d.voices_spawned,
            peak_voices: self.peak_voices,
            stolen,
            dropped: d.dropped,
            peak_level: d.peak,
            clipped: d.clipped,
            silent_blocks: d.silent_blocks,
            blocks: d.blocks,
            cancelled,
        };
        ctx.done(self.stem, done, self.t0);
        Ok(())
    }
}

/// A GPU job: stems in every lane of `batch`, and a new stem into whichever \[11\]
fn gpu_job(ctx: &Ctx, batch: &mut GpuBatch, threads: usize) -> Result<()> {
    let mut lanes: Vec<Option<Running>> = (0..batch.lanes()).map(|_| None).collect();
    let threads = threads.clamp(1, lanes.len());
    // [12]
    let mut prof = [Duration::ZERO; 4];
    let silent = AtomicU64::new(0);
    let (mut sent, mut sounding) = (0u64, 0u64);
    let result = (|| -> Result<()> {
        loop {
            let mut t = Instant::now();
            let mut lap = |i: usize, t: &mut Instant| {
                prof[i] += t.elapsed();
                *t = Instant::now();
            };
            on_threads(threads, lanes.iter_mut().zip(batch.lanes_mut()), |(slot, mut lane)| {
                host_round(ctx, slot, &mut lane, &silent)
            })?;
            lap(0, &mut t);
            if lanes.iter().all(Option::is_none) {
                return Ok(());
            }
            batch.flush()?;
            sent += 1;
            sounding += lanes.iter().flatten().count() as u64;
            lap(1, &mut t);
            // [13]
            on_threads(threads, lanes.iter_mut().flatten(), |r| {
                r.driver.prepare_ahead().with_context(|| track(r.stem))
            })?;
            lap(2, &mut t);
            batch.wait()?;
            lap(3, &mut t);
        }
    })();
    if ctx.cfg.profile {
        if let Some(times) = batch.pass_times() {
            let line: Vec<String> = times.iter().map(|(n, ms)| format!("{n} {ms:.3}ms")).collect();
            log::info!(target: TARGET, "  batch passes, mean per batch: {}", line.join("  "));
        }
        let s = |d: Duration| d.as_secs_f64();
        log::info!(
            target: TARGET,
            "{} lanes on {threads} thread{}: {sent} batches ({} of them split) carrying {sounding} blocks ({:.1} a batch), {} silent blocks on the host | host {:.2}s  flush {:.2}s  prepare {:.2}s  wait {:.2}s",
            batch.lanes(),
            if threads == 1 { "" } else { "s" },
            batch.split_batches(),
            sounding as f64 / sent.max(1) as f64,
            silent.load(Ordering::Relaxed),
            s(prof[0]),
            s(prof[1]),
            s(prof[2]),
            s(prof[3])
        );
    }
    result
}

/// `f` on every item, on up to `threads` threads, each taking the next item \[14\]
fn on_threads<I, F>(threads: usize, items: I, f: F) -> Result<()>
where
    I: Iterator + Send,
    I::Item: Send,
    F: Fn(I::Item) -> Result<()> + Sync,
{
    let work = Mutex::new(items);
    let failed: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let worker = || loop {
        let next = work.lock().unwrap().next();
        let Some(item) = next else { break };
        if let Err(e) = f(item) {
            failed.lock().unwrap().get_or_insert(e);
            break;
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..threads {
            scope.spawn(worker);
        }
        worker();
    });
    match failed.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// One lane's host work for a round: finish the block the last batch rendered \[15\]
fn host_round<'a>(ctx: &Ctx<'a>, slot: &mut Option<Running<'a>>, lane: &mut LaneBackend, silent: &AtomicU64) -> Result<()> {
    // [16]
    if slot.is_some() {
        step_done(ctx, lane, slot)?;
    }
    loop {
        if slot.is_none() {
            let Some((rank, stem)) = ctx.take() else { return Ok(()) };
            lane.reset();
            *slot = Some(Running::start(ctx, stem, rank, lane.index()).with_context(|| track(stem))?);
        }
        let r = slot.as_mut().expect("filled above");
        r.driver.submit_block(lane).with_context(|| track(r.stem))?;
        if lane.submitted() {
            return Ok(());
        }
        // [17]
        r.driver.prepare_ahead().with_context(|| track(r.stem))?;
        silent.fetch_add(1, Ordering::Relaxed);
        step_done(ctx, lane, slot)?;
    }
}

fn track(stem: &Stem) -> String {
    format!("rendering track {}", stem.track + 1)
}

/// Finish the block the lane has in flight, write it, and end the stem if that \[18\]
fn step_done(ctx: &Ctx, lane: &mut LaneBackend, slot: &mut Option<Running>) -> Result<()> {
    let r = slot.as_mut().expect("a lane with a block in flight has a stem");
    let more = r.driver.finish_block(lane, &mut r.block).with_context(|| track(r.stem))?;
    emit(ctx, &mut r.out, &r.driver, &r.block).with_context(|| track(r.stem))?;
    let stats = lane.stats();
    ctx.progress.block(r.slot, &mut r.seen, r.stem, &r.driver, stats.stolen);
    r.peak_voices = r.peak_voices.max(stats.active_voices);
    let last = !more || r.driver.stats.frames >= ctx.max_frames;
    if last || ctx.cancel.load(Ordering::Relaxed) {
        let r = slot.take().expect("matched above");
        r.finish(ctx, stats.stolen, !last)?;
    }
    Ok(())
}

/// Where a finished block goes: the stem's own file, or the mix at the block's \[19\]
fn emit(ctx: &Ctx, out: &mut Option<Sink>, driver: &Driver, block: &[f32]) -> Result<()> {
    match out {
        Some(sink) => sink.write_block(block),
        None => ctx.mix.expect("a stem without a file is being merged").add(driver.stats.blocks - 1, block),
    }
}

/// A CPU job: one stem at a time on a fresh reference synth each.
fn cpu_job(ctx: &Ctx, slot: usize) {
    while let Some((rank, stem)) = ctx.take() {
        let t0 = Instant::now();
        ctx.progress.enter(slot, rank);
        let result = render_stem(ctx, stem, slot);
        ctx.progress.leave(slot);
        match result {
            Ok(d) => ctx.done(stem, d, t0),
            Err(e) => ctx.fail(e.context(format!("rendering track {}", stem.track + 1))),
        }
    }
}

pub(crate) fn run(job: &Job, plan: Plan, preloaded: Option<Arc<Bank>>, obs: &mut dyn Observer) -> Result<Summary> {
    let spec = job.stems.as_ref().expect("dispatched on Job::stems");
    let Plan { mut cfg, kind, encoder } = plan;
    let started = Instant::now();

    // [20]
    obs.phase(Phase::OpeningMidi);
    let t0 = Instant::now();
    let scan = match &spec.scanned {
        Some(scan) if scan.path == job.midi => {
            log::info!(target: TARGET, "{} has {} tracks, scanned already", job.midi.display(), scan.tracks.len());
            scan.clone()
        }
        _ => {
            let scan = Arc::new(tracks::scan(&job.midi, 0, None)?);
            log::info!(
                target: TARGET,
                "{} has {} tracks, scanned in {:.2?}",
                job.midi.display(),
                scan.tracks.len(),
                t0.elapsed()
            );
            scan
        }
    };
    let (named, without) = spec.tracks.resolve(&scan)?;
    if !without.is_empty() {
        log::warn!(
            target: TARGET,
            "{} named in --tracks {} no notes and {} skipped: {}",
            plural(without.len(), "track"),
            if without.len() == 1 { "has" } else { "have" },
            if without.len() == 1 { "is" } else { "are" },
            list(&without)
        );
    }
    // [21]
    let min = cfg.min_velocity;
    let (keep, emptied): (Vec<usize>, Vec<usize>) =
        named.into_iter().partition(|&t| scan.tracks[t].notes_from(min) > 0);
    if !emptied.is_empty() {
        log::info!(
            target: TARGET,
            "{} no notes at velocity {min} or above and {} skipped: {}",
            plural(emptied.len(), "track"),
            if emptied.len() == 1 { "has" } else { "have" },
            list(&emptied)
        );
    }
    if keep.is_empty() {
        bail!("{}: none of the tracks --tracks names has anything to render", job.midi.display());
    }

    let total_voices = cfg.max_voices;
    cfg.max_voices = tracks::voices_each(total_voices, keep.len());

    let folder_name = job
        .midi
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "stems".into());
    let folder = job.out.join(&folder_name);
    let merge = spec.merge;
    let mut stems = Vec::with_capacity(keep.len());
    for (&t, sel) in keep.iter().zip(scan.selections(&keep, spec.setup)?) {
        let info = &scan.tracks[t];
        let name = info.display_name();
        let file = tracks::stem_file_name(t, scan.tracks.len(), name.as_deref(), &spec.ext);
        stems.push(Stem {
            track: t,
            sel,
            label: match (merge, &name) {
                (false, _) => file.clone(),
                (true, Some(n)) => format!("track {} ({n})", t + 1),
                (true, None) => format!("track {}", t + 1),
            },
            path: folder.join(file),
            name,
            notes: info.notes_from(min),
            span: (0, 0),
        });
    }
    // [22]
    let ends: Vec<u64> = keep
        .iter()
        .flat_map(|&t| {
            let info = &scan.tracks[t];
            [info.first_note.unwrap_or(0), info.last_note.or(info.first_note).unwrap_or(0)]
        })
        .collect();
    let frames = scan.frames_at(&ends, cfg.sample_rate);
    for (stem, f) in stems.iter_mut().zip(frames.chunks(2)) {
        let block = |frame: f64| frame.max(0.0) as u64 / cfg.block_frames as u64;
        stem.span = (block(f[0]), block(f[1]).max(block(f[0])));
    }
    if !merge {
        check_space(&folder, &scan, stems.len(), &cfg, job, &spec.ext)?;
    }

    obs.phase(Phase::LoadingSoundfont);
    let t0 = Instant::now();
    let loaded_here = preloaded.is_none();
    let bank = match preloaded {
        Some(bank) => bank,
        None => Arc::new(load_layered(&job.soundfonts, job.sf_programs.as_deref(), &cfg)?),
    };
    fit_to_bank(&mut cfg, &bank);
    if loaded_here {
        log::info!(target: TARGET, "loaded {} in {:.2?}", bank.describe(), t0.elapsed());
    }
    // Baseline, `plan` saw to that, so this prepares nothing.
    let phase = PhaseBank::prepare(&bank, &cfg.phase)?;

    obs.phase(Phase::PreparingDevice);
    let asked = spec.jobs.max(1);
    let most = tracks::max_jobs();
    if asked > most {
        log::warn!(
            target: TARGET,
            "--track-jobs {asked} is more than this machine's cores less two; {most} at once"
        );
    }
    let wanted = asked.min(most).min(stems.len());
    let mut backends = Vec::with_capacity(wanted);
    let at_once;
    let mut gpu_threads = 1;
    let mut setup = Setup {
        backend: "cpu",
        adapter: None,
        device_bytes: None,
        vendor_id: None,
        device_id: None,
        tracks: scan.tracks.len().min(u16::MAX as usize) as u16,
        max_voices: cfg.max_voices,
        bytes_total: 0,
    };
    match kind {
        BackendKind::Cpu => {
            backends.extend((0..wanted).map(|_| JobBackend::Cpu));
            at_once = format!("{wanted} at once");
        }
        BackendKind::Gpu => {
            // [23]
            let mut probe = cfg.clone();
            probe.max_voices = 1;
            let shared = GpuShared::new(&probe, &bank, &phase)?;
            // [24]
            let cap = GpuBatch::max_voices_each(&cfg, &bank, shared.binding_bytes(), 1);
            if cfg.max_voices > cap && cap > 0 {
                log::warn!(
                    target: TARGET,
                    "{} voices each is more than one track can hold on {}; {} each",
                    cfg.max_voices,
                    shared.adapter_name(),
                    cap
                );
                cfg.max_voices = cap;
            }
            // [25]
            let mut lanes = stems.len().min(LANES_MAX);
            // And no more than fits in video memory beside the pool.
            let per_lane = GpuBatch::lane_bytes(&cfg, &bank).max(1);
            let (vendor, device) = shared.adapter_ids();
            let fit = crate::gpu::vram::sample(vendor, device)
                .and_then(|m| Some(m.process_budget?.saturating_sub(m.process_used?)))
                .map(|free| (free.saturating_sub(VRAM_MARGIN) / per_lane) as usize);
            if let Some(fit) = fit {
                if lanes > fit {
                    lanes = fit.max(1);
                    log::warn!(
                        target: TARGET,
                        "{lanes} stems at once: that is what fits in video memory at about {} a stem",
                        mib(per_lane)
                    );
                }
            }
            // [26]
            let bind = GpuBatch::lanes_that_bind(&cfg, &bank, shared.binding_bytes(), lanes);
            if bind > 0 && bind < lanes {
                lanes = bind;
                log::warn!(
                    target: TARGET,
                    "{lanes} stems at once: at {} voices each, that is as many as {} binds in one buffer",
                    cfg.max_voices,
                    shared.adapter_name()
                );
            }
            gpu_threads = wanted.min(lanes);
            backends.push(JobBackend::Gpu(Box::new(GpuBatch::new(&cfg, &bank, &shared, lanes)?)));
            setup = Setup {
                backend: "gpu",
                adapter: Some(shared.adapter_name().to_string()),
                device_bytes: Some(shared.bytes() + lanes as u64 * per_lane),
                vendor_id: Some(vendor),
                device_id: Some(device),
                max_voices: cfg.max_voices,
                ..setup
            };
            log::info!(
                target: TARGET,
                "gpu: {} | {} sample pool, {lanes} lanes at about {} a lane",
                shared.adapter_name(),
                mib(shared.bytes()),
                mib(per_lane)
            );
            at_once = format!(
                "{lanes} at once on {gpu_threads} thread{}",
                if gpu_threads == 1 { "" } else { "s" }
            );
        }
    }
    let jobs = backends.len();
    obs.setup(&setup);
    log::info!(
        target: TARGET,
        "rendering {} into {}, {at_once}, {} voices each ({} over {}), setup tracks {}",
        plural(stems.len(), if merge { "track" } else { "stem" }),
        if merge { job.out.display() } else { folder.display() },
        cfg.max_voices,
        total_voices,
        stems.len(),
        match spec.setup {
            tracks::SetupTracks::Apply => "applied",
            tracks::SetupTracks::Ignore => "ignored",
        }
    );
    // [27]
    let mut stem_cfg = cfg.clone();
    stem_cfg.max_block_candidates = tracks::candidates_each(&stem_cfg);
    // The tracks are already spread over threads; a thread each is enough.
    stem_cfg.materialise_threads = 1;
    let mix = if merge {
        stem_cfg.limiter_mode = LimiterMode::Off;
        stem_cfg.clamp_output = false;
        let secs = job.seconds.map_or(scan.duration(cfg.sample_rate), |s| s.min(scan.duration(cfg.sample_rate)));
        log::info!(
            target: TARGET,
            "the tracks are summed exactly in memory before the limiter: at most about {} for {:.1}s",
            mib((secs * cfg.sample_rate as f64) as u64 * 2 * 16),
            secs
        );
        Some(Mix::new(cfg.block_samples()))
    } else {
        std::fs::create_dir_all(&folder).with_context(|| format!("creating {}", folder.display()))?;
        None
    };

    // [28]
    let mut order: Vec<usize> = (0..stems.len()).collect();
    order.sort_by_key(|&i| (Reverse(stems[i].notes), stems[i].track));

    let max_frames = job.seconds.map(|s| (s * cfg.sample_rate as f64) as u64).unwrap_or(u64::MAX);
    let next = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let running = AtomicUsize::new(jobs);
    let cancel = AtomicBool::new(false);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let results: Mutex<Vec<Done>> = Mutex::new(Vec::with_capacity(stems.len()));
    let slots = backends
        .iter()
        .map(|b| match b {
            JobBackend::Gpu(batch) => batch.lanes(),
            JobBackend::Cpu => 1,
        })
        .sum();
    let progress = Progress::new(slots);
    // Every track runs to where the file ends, or to --seconds.
    let per_track = ((max_frames as f64).min(scan.duration(cfg.sample_rate) * cfg.sample_rate as f64)
        / cfg.block_frames as f64)
        .ceil() as u64;
    // [29]
    let span_total: u64 = stems
        .iter()
        .filter(|s| s.span.0 < per_track)
        .map(|s| s.span.1.min(per_track - 1) - s.span.0 + 1)
        .sum();
    let length_secs = (per_track * cfg.block_frames as u64) as f64 / cfg.sample_rate as f64;
    // Every note the tracks hold, unless --seconds stops them short of some.
    let cut = (max_frames as f64) < scan.duration(cfg.sample_rate) * cfg.sample_rate as f64;
    let notes_total: u64 = if cut { 0 } else { stems.iter().map(|s| s.notes).sum() };
    let report = || {
        let mut now: Vec<(usize, u64)> = progress
            .slots
            .iter()
            .filter_map(|(rank, blocks)| {
                let rank = rank.load(Ordering::Relaxed);
                (rank > 0).then(|| (rank - 1, blocks.load(Ordering::Relaxed)))
            })
            .collect();
        let running = now.len();
        now.sort_unstable();
        TrackProgress {
            total: stems.len(),
            done: finished.load(Ordering::Relaxed),
            running,
            stole: progress.stole.load(Ordering::Relaxed),
            blocks: progress.blocks.load(Ordering::Relaxed),
            silent_blocks: progress.silent.load(Ordering::Relaxed),
            blocks_total: per_track * stems.len() as u64,
            span_blocks: progress.span.load(Ordering::Relaxed),
            span_total,
            notes_total,
            length_secs,
            voices_each: cfg.max_voices,
            merged: merge,
            audio_secs: (progress.blocks.load(Ordering::Relaxed) * cfg.block_frames as u64) as f64 / cfg.sample_rate as f64,
            notes: progress.notes.load(Ordering::Relaxed),
            stolen: progress.stolen.load(Ordering::Relaxed),
            dropped: progress.dropped.load(Ordering::Relaxed),
            peak_level: f32::from_bits(progress.peak.load(Ordering::Relaxed)),
            now: now
                .into_iter()
                .take(3)
                .map(|(rank, blocks)| {
                    let stem = &stems[order[rank]];
                    TrackNow {
                        track: stem.track + 1,
                        name: stem.name.clone(),
                        secs: (blocks * cfg.block_frames as u64) as f64 / cfg.sample_rate as f64,
                    }
                })
                .collect(),
        }
    };
    let ctx = Ctx {
        cfg: &stem_cfg,
        bank: &bank,
        phase: &phase,
        encoder: encoder.as_ref(),
        job,
        stems: &stems,
        order: &order,
        max_frames,
        next: &next,
        finished: &finished,
        cancel: &cancel,
        failure: &failure,
        results: &results,
        mix: mix.as_ref(),
        progress: &progress,
    };

    obs.phase(Phase::Rendering);
    let render_started = Instant::now();
    std::thread::scope(|scope| {
        for (slot, backend) in backends.iter_mut().enumerate() {
            let (ctx, running) = (&ctx, &running);
            scope.spawn(move || {
                match backend {
                    JobBackend::Gpu(batch) => {
                        if let Err(e) = gpu_job(ctx, batch, gpu_threads) {
                            ctx.fail(e);
                        }
                    }
                    // CPU jobs each have one slot, in job order.
                    JobBackend::Cpu => cpu_job(ctx, slot),
                }
                running.fetch_sub(1, Ordering::Relaxed);
            });
        }
        // [30]
        while running.load(Ordering::Relaxed) > 0 {
            std::thread::sleep(Duration::from_millis(50));
            if obs.cancelled() {
                cancel.store(true, Ordering::Relaxed);
            }
            obs.tracks(&report());
        }
    });
    obs.tracks(&report());
    obs.phase(Phase::Finishing);
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }

    let done = results.into_inner().unwrap();
    let cancelled = done.iter().any(|d| d.cancelled) || done.len() < stems.len();
    let sum = |f: fn(&Done) -> u64| done.iter().map(f).sum::<u64>();
    let audio: f64 = done.iter().map(|d| d.audio_secs).sum();
    let mut summary = Summary {
        bytes: sum(|d| d.bytes),
        audio_secs: audio,
        wall_secs: 0.0,
        notes: sum(|d| d.notes),
        notes_skipped: sum(|d| d.notes_skipped),
        voices_spawned: sum(|d| d.voices_spawned),
        peak_voices: done.iter().map(|d| d.peak_voices).max().unwrap_or(0),
        stolen: sum(|d| d.stolen),
        dropped: sum(|d| d.dropped),
        peak_level: done.iter().map(|d| d.peak_level).fold(0.0, f32::max),
        clipped: sum(|d| d.clipped),
        cancelled,
    };
    match mix {
        // [31]
        Some(_) if cancelled => {
            log::warn!(target: TARGET, "stopped after {} of {} tracks; no merged file written", done.len(), stems.len());
            summary.bytes = 0;
            summary.audio_secs = 0.0;
        }
        Some(mix) => {
            let t0 = Instant::now();
            let w = mix.write(&cfg, &job.out, encoder.as_ref(), job.wav_format)?;
            let wall = render_started.elapsed().as_secs_f64();
            let secs = w.frames as f64 / cfg.sample_rate as f64;
            summary.bytes = w.bytes;
            summary.audio_secs = secs;
            summary.peak_level = w.peak;
            summary.clipped = w.clipped;
            log::info!(
                target: TARGET,
                "wrote {} ({}, {:.2}s audio) from {} in {:.2}s = {:.2}x realtime; the mix held {} and \
                 was limited and written in {:.2}s",
                job.out.display(),
                mib(w.bytes),
                secs,
                plural(stems.len(), "track"),
                wall,
                secs / wall.max(1e-9),
                mib(w.held),
                t0.elapsed().as_secs_f64()
            );
        }
        None => {
            let wall = render_started.elapsed().as_secs_f64();
            log::info!(
                target: TARGET,
                "wrote {} of {} to {} ({}) in {:.2}s: {:.2}s of audio, {:.2}x realtime over all of them",
                done.len(),
                plural(stems.len(), "stem"),
                folder.display(),
                mib(summary.bytes),
                wall,
                audio,
                audio / wall.max(1e-9)
            );
        }
    }
    summary.wall_secs = started.elapsed().as_secs_f64();
    log::info!(
        target: TARGET,
        "{} notes, {} voices spawned, peak {} concurrent in one track, {} stolen, {} dropped, \
         peak level {:.3}{}",
        summary.notes,
        summary.voices_spawned,
        summary.peak_voices,
        summary.stolen,
        summary.dropped,
        summary.peak_level,
        if merge { " in the mix" } else { "" }
    );
    let (silent, blocks) = (sum(|d| d.silent_blocks), sum(|d| d.blocks));
    if silent > 0 {
        log::info!(target: TARGET, "{silent} of {blocks} blocks were silent and written without the device");
    }
    if summary.clipped > 0 {
        log::warn!(
            target: TARGET,
            "{} samples were hard-clipped at full scale {}",
            summary.clipped,
            if merge { "in the mix" } else { "across the stems" }
        );
    }
    Ok(summary)
}

/// Render one stem, start to finish, on a fresh CPU reference synth.
fn render_stem(ctx: &Ctx, stem: &Stem, slot: usize) -> Result<Done> {
    let cfg = ctx.cfg;
    let mut driver = Driver::open_tracks_prepared(cfg, ctx.bank.clone(), &ctx.job.midi, &stem.sel, ctx.phase.clone())?;
    let mut cpu = CpuSynth::new_prepared(cfg, ctx.bank.clone(), ctx.phase.clone());
    let backend: &mut dyn Backend = &mut cpu;
    let mut out = match ctx.mix {
        Some(_) => None,
        None => Some(Sink::create(&stem.path, cfg, ctx.encoder, ctx.job.wav_format)?),
    };
    let mut block = vec![0.0f32; cfg.block_samples()];
    let mut peak_voices = 0u64;
    let mut cancelled = false;
    let mut seen = Seen::default();
    // [32]
    loop {
        let more = driver.next_block(backend, &mut block)?;
        emit(ctx, &mut out, &driver, &block)?;
        peak_voices = peak_voices.max(backend.stats().active_voices);
        ctx.progress.block(slot, &mut seen, stem, &driver, backend.stats().stolen);
        if !more || driver.stats.frames >= ctx.max_frames {
            break;
        }
        if ctx.cancel.load(Ordering::Relaxed) {
            cancelled = true;
            break;
        }
    }
    let bytes = match out {
        Some(out) => out.finish()?,
        None => 0,
    };
    let d = &driver.stats;
    Ok(Done {
        bytes,
        audio_secs: driver.seconds_rendered(),
        notes: d.notes,
        notes_skipped: d.notes_skipped,
        voices_spawned: d.voices_spawned,
        peak_voices,
        stolen: backend.stats().stolen,
        dropped: d.dropped,
        peak_level: d.peak,
        clipped: d.clipped,
        silent_blocks: d.silent_blocks,
        blocks: d.blocks,
        cancelled,
    })
}

/// Say how much the stems will take and refuse to start a WAV render that \[33\]
fn check_space(folder: &Path, scan: &TrackScan, stems: usize, cfg: &Config, job: &Job, ext: &str) -> Result<()> {
    let secs = tracks::render_secs(scan, cfg.sample_rate, job.seconds);
    let (need, exact) = tracks::output_bytes(stems, secs, cfg.sample_rate, ext, job.wav_format == wav::SampleFormat::Float32);
    let mut probe = folder;
    while !probe.exists() {
        match probe.parent() {
            Some(p) if !p.as_os_str().is_empty() => probe = p,
            _ => {
                probe = Path::new(".");
                break;
            }
        }
    }
    let free = fs4::available_space(probe)
        .map_err(|e| anyhow!("reading the free space at {}: {e}", probe.display()))?;
    log::info!(
        target: TARGET,
        "{} of {:.1}s each: {}{} needed, {} free on that drive",
        plural(stems, "stem"),
        secs,
        if exact { "about " } else { "at most " },
        gib(need),
        gib(free)
    );
    if need > free {
        if exact {
            bail!(
                "{} {ext} stems need about {} and {} has {} free. Free some space, choose \
                 fewer tracks with --tracks, or write a compressed --stem-format such as flac, \
                 in which silence costs almost nothing",
                stems,
                gib(need),
                probe.display(),
                gib(free)
            );
        }
        log::warn!(
            target: TARGET,
            "the stems may not fit: {} free against an upper bound of {}. {ext} is usually far \
             smaller than that bound, and a stem's silence smallest of all",
            gib(free),
            gib(need)
        );
    }
    Ok(())
}

fn plural(n: usize, what: &str) -> String {
    format!("{n} {what}{}", if n == 1 { "" } else { "s" })
}

/// 1-based track numbers, the first twenty of them.
fn list(tracks: &[usize]) -> String {
    let mut s: Vec<String> = tracks.iter().take(20).map(|t| (t + 1).to_string()).collect();
    if tracks.len() > 20 {
        s.push(format!("and {} more", tracks.len() - 20));
    }
    s.join(", ")
}

fn mib(b: u64) -> String {
    format!("{:.1} MiB", b as f64 / 1048576.0)
}

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
}
