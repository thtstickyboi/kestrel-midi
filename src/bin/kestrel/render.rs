// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `kestrel --force-cli render`: the arguments turned into a `session::Job`,
//! and the lines the command line has always printed while one renders. The
//! pipeline itself is `kestrel::session`; several renders at once are
//! `kestrel::batch`.

use crate::{Cli, Cmd, RenderArgs};
use anyhow::{bail, Result};
use clap::Parser;
use kestrel::batch::{self, BatchObserver};
use kestrel::session::{self, Job, Observer, Phase, Tick};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The target the progress lines are logged under, the same one
/// `kestrel::session` uses, so the whole render reads `[INFO  kestrel]`.
const TARGET: &str = "kestrel";

impl RenderArgs {
    /// The render these arguments describe, with everything clap cannot check
    /// -- the named enums, `Config::validate` -- already checked.
    pub fn to_job(&self) -> Result<Job> {
        let (cfg, backend) = self.to_config()?;
        let setup = kestrel::tracks::SetupTracks::parse(&self.setup_tracks)
            .expect("clap limits --setup-tracks to the two names SetupTracks parses");
        let stems = match &self.tracks {
            Some(spec) => Some(kestrel::tracks::Stems {
                tracks: kestrel::tracks::TrackList::parse(spec)?,
                setup,
                jobs: self.track_jobs,
                ext: self.stem_format.clone(),
                merge: self.merge,
                scanned: None,
                resume: None,
            }),
            None => None,
        };
        Ok(Job {
            midi: self.midi.clone(),
            soundfonts: self.soundfont.clone(),
            sf_programs: self.sf_programs.clone(),
            out: self.out.clone(),
            ffmpeg: self.ffmpeg.clone(),
            wav_format: kestrel::wav::SampleFormat::parse(&self.format)
                .expect("clap limits --format to the two names SampleFormat parses"),
            ceiling_db: self.ceiling_db,
            seconds: self.seconds,
            block_csv: self.dev_args().block_csv,
            track: self.track.map(|n| kestrel::tracks::TrackPick { index: n as usize - 1, setup }),
            stems,
            checkpoint: None,
            backend,
            cfg,
        })
    }

    /// Checkpoints for a render, as `--checkpoint-every` asks: of the tracks
    /// for a per-track render and of the render itself for one file or one
    /// track. `argv` is the command as it was typed: the checkpoint keeps it,
    /// and a resume parses it again through the same definition. A command that
    /// is not valid Unicode cannot be kept as text, so that render cannot be
    /// resumed, and is told so.
    pub fn with_checkpoints<I, S>(&self, job: &mut Job, argv: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        // `--no-resume`: none, so the audio goes straight to the output.
        if self.no_resume {
            return;
        }
        let argv: Result<Vec<String>, OsString> = argv.into_iter().map(|a| a.into().into_string()).collect();
        let Ok(argv) = argv else {
            log::warn!(
                target: TARGET,
                "this command line is not valid Unicode, so this render cannot save progress to be resumed"
            );
            return;
        };
        let mut spec = kestrel::resume::Spec::new(argv);
        // Minutes, any fraction of one; not a number that is no interval at all.
        // A year is the longest taken, which keeps the conversion from panicking.
        spec.every = (self.checkpoint_every.is_finite() && self.checkpoint_every > 0.0)
            .then(|| std::time::Duration::from_secs_f64(self.checkpoint_every.min(525_600.0) * 60.0));
        spec.stop_after_blocks = self.dev_args().stop_after_blocks;
        match job.stems.as_mut() {
            Some(stems) => stems.resume = Some(spec),
            None => job.checkpoint = Some(spec),
        }
    }

    /// What stops these arguments being one render, for the front ends that put
    /// the MIDI after `--` themselves and so know there is exactly one. A word
    /// that is neither a flag nor a flag's value, such as the "on" in
    /// `--dc-blocker on`, is taken as the MIDI by clap, which would leave the
    /// real MIDI as a second one; without this it would render a file called
    /// "on".
    pub fn not_a_single_render(&self) -> Option<String> {
        if !self.more_midi.is_empty() {
            return Some(format!(
                "{:?} is not a flag or a value for one, and was taken for a second MIDI",
                self.midi.to_string_lossy()
            ));
        }
        if self.out_format.is_some() {
            return Some("--out-format is for a batch of several MIDIs".into());
        }
        None
    }

    /// Every MIDI the command names, in the order given.
    pub(crate) fn midis(&self) -> Vec<PathBuf> {
        std::iter::once(self.midi.clone())
            .chain(self.more_midi.iter().cloned())
            .collect()
    }

    /// The jobs of a render that names several MIDIs: this one's flags and
    /// soundfonts for each, written into the folder `-o` names.
    pub(crate) fn to_batch_jobs(&self) -> Result<Vec<Job>> {
        let template = self.to_job()?;
        let entries: Vec<batch::Entry> = self
            .midis()
            .into_iter()
            .map(|midi| batch::Entry {
                midi,
                soundfonts: self.soundfont.clone(),
                sf_programs: self.sf_programs.clone(),
                out: None,
                seconds: None,
            })
            .collect();
        let ext = self.out_format.as_deref().unwrap_or("wav");
        batch::jobs_from(&template, &entries, Some(&self.out), ext)
    }
}

/// The command line's observer: a progress line a second, and the per-pass
/// timings under `--profile`, exactly as the loop printed them before it was
/// shared.
struct LogObserver {
    last_report: Instant,
}

impl Observer for LogObserver {
    fn phase(&mut self, phase: Phase) {
        if phase == Phase::Rendering {
            self.last_report = Instant::now();
        }
    }

    fn block(&mut self, t: &Tick) {
        if self.last_report.elapsed().as_secs_f64() > 1.0 {
            let secs = t.driver.seconds_rendered();
            let wall = t.start.elapsed().as_secs_f64();
            log::info!(
                target: TARGET,
                "{:>8.2}s rendered | {:>10} voices | {:>12} notes | {:.2}x realtime",
                secs,
                t.stats.active_voices,
                t.driver.stats.notes,
                // Of this run: a resumed render's earlier audio was not made in it.
                (secs - t.resumed_secs) / wall.max(1e-9)
            );
            if t.cfg.profile {
                let tm = t.backend.timings();
                if !tm.is_empty() {
                    let line: Vec<String> =
                        tm.iter().map(|(n, ms)| format!("{n} {ms:.3}ms")).collect();
                    log::info!(target: TARGET, "  passes: {}", line.join("  "));
                }
            }
            self.last_report = Instant::now();
        }
    }
}

/// `kestrel --force-cli render`, or the JSON feed when `--progress` asks for it.
pub fn render_cli(args: RenderArgs) -> Result<()> {
    if args.log {
        crate::start_falconeye("command line");
        kestrel::falconeye::renderlog::set_args(std::env::args_os());
    }
    if !args.more_midi.is_empty() {
        if args.progress.is_some() {
            return crate::feed::run_batch(&args);
        }
        return run_batch(args.to_batch_jobs()?);
    }
    if args.out_format.is_some() {
        bail!("--out-format is for several MIDIs; with one, the extension of -o picks the format");
    }
    if args.progress.is_some() {
        return crate::feed::run(&args);
    }
    let mut job = args.to_job()?;
    args.with_checkpoints(&mut job, std::env::args_os());
    let plan = session::plan(&job)?;
    let mut obs = LogObserver {
        last_report: Instant::now(),
    };
    session::run(&job, plan, None, &mut obs)?;
    Ok(())
}

/// A checkpoint turned back into the render it was made under, ready to
/// continue from it.
pub struct Resuming {
    pub args: RenderArgs,
    pub job: Job,
    /// What the checkpoint holds, for a front end that says how far it got.
    pub checkpoint: std::sync::Arc<kestrel::resume::Checkpoint>,
    /// The command it was made under, as it was typed.
    pub argv: Vec<OsString>,
}

/// Read `file`, parse the command it was made under again through the same
/// definition, and build the job that continues it. Everything that is wrong
/// with the file or the command is an error here, in words; whether the
/// MIDI, the soundfonts and the machine still fit is checked by the render.
pub fn resume_job(file: &Path) -> Result<Resuming> {
    let cp = kestrel::resume::Checkpoint::open(file)?;
    let argv: Vec<OsString> = cp.header.argv.iter().map(OsString::from).collect();
    let args = match Cli::try_parse_from(&argv) {
        Ok(Cli { cmd: Cmd::Render(a), .. }) => a,
        Ok(_) => bail!("{}: the command it was made under is not a render", file.display()),
        Err(e) => bail!(
            "{}: the command it was made under does not parse in this build: {}",
            file.display(),
            e.render()
        ),
    };
    // Which kind of render it was is in the checkpoint and in the command, and
    // they have to agree: the command is only a description of what to run.
    let single = cp.header.single.is_some();
    if single == args.tracks.is_some() {
        bail!(
            "{}: the checkpoint is of a render of {}, and the command it was made under {}",
            file.display(),
            if single { "one file" } else { "the tracks one by one" },
            if single { "has --tracks" } else { "has no --tracks" }
        );
    }
    let mut job = args.to_job()?;
    args.with_checkpoints(&mut job, argv.clone());
    let checkpoint = std::sync::Arc::new(cp);
    let spec = if single { job.checkpoint.as_mut() } else { job.stems.as_mut().and_then(|s| s.resume.as_mut()) };
    if let Some(spec) = spec {
        // Written back to the file it came from, so it goes on being the one.
        spec.path = Some(file.to_path_buf());
        spec.restore = Some(checkpoint.clone());
        // The command it was made under may carry the hook that stopped it,
        // and a resume goes on to the end.
        spec.stop_after_blocks = None;
    }
    Ok(Resuming { args, job, checkpoint, argv })
}

/// `kestrel --force-cli resume FILE.krsm`: the command the checkpoint was made
/// under, run again from where it stopped.
pub fn resume_cli(file: &Path) -> Result<()> {
    let Resuming { args, job, argv, .. } = resume_job(file)?;
    if args.log {
        crate::start_falconeye("command line");
        kestrel::falconeye::renderlog::set_args(&argv);
    }
    let plan = session::plan(&job)?;
    let mut obs = LogObserver {
        last_report: Instant::now(),
    };
    session::run(&job, plan, None, &mut obs)?;
    Ok(())
}

/// Stands in for the MIDI, soundfont and output while a batch file's `args` are
/// parsed, so it can be seen whether they tried to name their own.
const PLACEHOLDER: &str = "batch-file-placeholder";

/// `kestrel --force-cli batch FILE`.
pub fn batch_cli(file: &Path) -> Result<()> {
    let (shared, entries) = batch::File::read(file)?;
    let mut argv: Vec<OsString> = ["kestrel", "render"].into_iter().map(OsString::from).collect();
    argv.push(format!("--soundfont={PLACEHOLDER}.sf2").into());
    argv.push(format!("--out={PLACEHOLDER}.wav").into());
    argv.extend(shared.args.iter().map(OsString::from));
    argv.push("--".into());
    argv.push(format!("{PLACEHOLDER}.mid").into());
    let args = match Cli::try_parse_from(argv) {
        Ok(Cli { cmd: Cmd::Render(a), .. }) => a,
        Ok(_) => unreachable!("the argument list names the render subcommand"),
        Err(e) => bail!("{}: `args`: {}", file.display(), e.render()),
    };
    let named = |p: &Path| p.to_string_lossy().starts_with(PLACEHOLDER);
    if args.soundfont.len() != 1
        || !args.soundfont.iter().all(|p| named(p))
        || !named(&args.out)
        || !args.more_midi.is_empty()
        || !named(&args.midi)
    {
        bail!(
            "{}: `args` cannot name the MIDI, the soundfonts or the output: the jobs do",
            file.display()
        );
    }
    if args.tracks.is_some() || args.track.is_some() {
        bail!("{}: --track and --tracks do not combine with a batch", file.display());
    }
    if args.progress.is_some() {
        bail!("{}: --progress json is not available for a batch", file.display());
    }
    if args.log {
        crate::start_falconeye("command line");
        kestrel::falconeye::renderlog::set_args(std::env::args_os());
    }
    let template = args.to_job()?;
    let jobs = batch::jobs_from(&template, &entries, shared.out_dir.as_deref(), &shared.out_format)?;
    run_batch(jobs)
}

/// The command line's batch observer: each job reports as a render does, with
/// a line before and after.
struct BatchLog {
    inner: LogObserver,
}

impl BatchObserver for BatchLog {
    fn job(&mut self) -> &mut dyn Observer {
        &mut self.inner
    }

    fn started(&mut self, index: usize, total: usize, job: &Job) {
        log::info!(
            target: TARGET,
            "batch: job {} of {total}: {} -> {}",
            index + 1,
            job.midi.display(),
            job.out.display()
        );
    }

    fn finished(&mut self, total: usize, r: &batch::JobResult) {
        match &r.outcome {
            batch::Outcome::Done(s) if !s.cancelled => log::info!(
                target: TARGET,
                "batch: job {} of {total} done: {}",
                r.index + 1,
                r.out.display()
            ),
            batch::Outcome::Done(_) => log::warn!(
                target: TARGET,
                "batch: job {} of {total} stopped; what was rendered is in {}",
                r.index + 1,
                r.written.as_deref().map(|p| p.display().to_string()).unwrap_or_default()
            ),
            batch::Outcome::Failed(why) => log::error!(
                target: TARGET,
                "batch: job {} of {total} ({}) failed: {why}",
                r.index + 1,
                r.midi.display()
            ),
            batch::Outcome::NotRun => {}
        }
    }
}

/// Check and run a batch, then say how it went. Fails if any job did.
fn run_batch(jobs: Vec<Job>) -> Result<()> {
    let plan = batch::plan(jobs)?;
    let mut obs = BatchLog {
        inner: LogObserver {
            last_report: Instant::now(),
        },
    };
    let summary = batch::run(plan, &mut obs);
    let total = summary.results.len();
    log::info!(
        target: TARGET,
        "batch: {} of {total} done in {:.1}s, {} soundfont set{} loaded",
        summary.done(),
        summary.wall_secs,
        summary.loads,
        if summary.loads == 1 { "" } else { "s" }
    );
    for r in summary.results.iter().filter(|r| !r.is_ok()) {
        match &r.outcome {
            batch::Outcome::Failed(why) => {
                log::error!(target: TARGET, "  failed: {} ({why})", r.midi.display())
            }
            _ => log::warn!(target: TARGET, "  not finished: {}", r.midi.display()),
        }
    }
    if summary.cancelled {
        bail!("the batch was stopped: {} of {total} jobs finished", summary.done());
    }
    if summary.failed() > 0 {
        bail!("{} of {total} jobs failed", summary.failed());
    }
    Ok(())
}
