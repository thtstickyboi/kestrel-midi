// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Several renders in one go: N MIDIs make N files, each with the soundfont \[1\]

use crate::session::{self, Job, Observer, Summary};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The log target, the same one `session` writes under.
const TARGET: &str = "kestrel";

/// The batch file's format version. Bumped when a change would make an old \[2\]
pub const FILE_VERSION: u32 = 1;

/// One render in a batch, before it is a [`Job`].
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub midi: PathBuf,
    /// Layered in order, each on top of the ones before it. Never empty by the \[3\]
    pub soundfonts: Vec<PathBuf>,
    /// As `--sf-programs`: the programs of bank 0 the last soundfont takes over.
    pub sf_programs: Option<String>,
    /// Where this one goes, when the batch's folder and the default name are \[4\]
    pub out: Option<PathBuf>,
    /// Stop this one after this many seconds of audio.
    pub seconds: Option<f64>,
}

// ---- the batch file ------------------------------------------------------

/// `kestrel --force-cli batch jobs.json`, as it is on disk. \[5\]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct File {
    /// Must be [`FILE_VERSION`].
    pub version: u32,
    /// The folder jobs without an `out` of their own are written to.
    #[serde(default)]
    pub out: Option<PathBuf>,
    /// The container of those files, by extension. `wav` when absent.
    #[serde(default)]
    pub out_format: Option<String>,
    /// Flags every job shares, as they would be typed after `render`: \[6\]
    #[serde(default)]
    pub args: Vec<String>,
    /// The soundfonts of any job that names none of its own.
    #[serde(default)]
    pub soundfonts: Vec<PathBuf>,
    #[serde(default)]
    pub sf_programs: Option<String>,
    pub jobs: Vec<FileEntry>,
}

/// One `jobs` entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    pub midi: PathBuf,
    /// Replaces the file's `soundfonts` and its `sf_programs` together: the \[7\]
    #[serde(default)]
    pub soundfonts: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub sf_programs: Option<String>,
    #[serde(default)]
    pub out: Option<PathBuf>,
    #[serde(default)]
    pub seconds: Option<f64>,
}

/// What a batch file settles besides its jobs.
#[derive(Debug, Clone)]
pub struct Shared {
    pub out_dir: Option<PathBuf>,
    /// An extension with no dot.
    pub out_format: String,
    pub args: Vec<String>,
}

impl File {
    /// Read and check a batch file. Everything that can be wrong with the file \[8\]
    pub fn read(path: &Path) -> Result<(Shared, Vec<Entry>)> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let file: File = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a valid batch file", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new(""));
        file.resolve(base)
            .with_context(|| format!("{}", path.display()))
    }

    fn resolve(self, base: &Path) -> Result<(Shared, Vec<Entry>)> {
        if self.version != FILE_VERSION {
            bail!(
                "batch file version {} is not one this Kestrel reads; it reads version {}",
                self.version,
                FILE_VERSION
            );
        }
        if self.jobs.is_empty() {
            bail!("the batch file has no jobs");
        }
        let at = |p: &Path| -> PathBuf {
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                base.join(p)
            }
        };
        let shared_fonts: Vec<PathBuf> = self.soundfonts.iter().map(|p| at(p)).collect();
        let entries = entries_from(&shared_fonts, self.sf_programs.as_deref(), &self.jobs, &at)?;
        let out_format = self
            .out_format
            .as_deref()
            .unwrap_or("wav")
            .trim_start_matches('.')
            .to_ascii_lowercase();
        Ok((
            Shared {
                out_dir: self.out.as_ref().map(|p| at(p)),
                out_format,
                args: self.args,
            },
            entries,
        ))
    }
}

/// The entries for a list of jobs, as a batch file and the API's `batch` command \[9\]
pub fn entries_from(
    fonts: &[PathBuf],
    programs: Option<&str>,
    jobs: &[FileEntry],
    at: &dyn Fn(&Path) -> PathBuf,
) -> Result<Vec<Entry>> {
    let mut entries = Vec::with_capacity(jobs.len());
    for (i, job) in jobs.iter().enumerate() {
        let n = i + 1;
        let (soundfonts, sf_programs) = match &job.soundfonts {
            Some(own) => (own.iter().map(|p| at(p)).collect::<Vec<_>>(), job.sf_programs.clone()),
            None => (
                fonts.to_vec(),
                job.sf_programs.clone().or_else(|| programs.map(str::to_string)),
            ),
        };
        if soundfonts.is_empty() {
            bail!(
                "job {n} ({}) has no soundfonts, and none are given to fall back on",
                job.midi.display()
            );
        }
        if let Some(s) = job.seconds {
            if !(s.is_finite() && s > 0.0) {
                bail!("job {n} ({}): seconds must be above 0", job.midi.display());
            }
        }
        entries.push(Entry {
            midi: at(&job.midi),
            soundfonts,
            sf_programs,
            out: job.out.as_ref().map(|p| at(p)),
            seconds: job.seconds,
        });
    }
    Ok(entries)
}

// ---- names ---------------------------------------------------------------

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "render".into())
}

/// What a name is compared as: case-folded, since Windows and macOS would \[10\]
fn fold(p: &Path) -> String {
    p.to_string_lossy().to_lowercase()
}

/// Where each entry's file goes. \[11\]
pub fn assign_outputs(entries: &[Entry], dir: Option<&Path>, ext: &str) -> Result<Vec<PathBuf>> {
    let plain = |e: &Entry| dir.map(|d| d.join(format!("{}.{ext}", stem(&e.midi))));
    let mut count: HashMap<String, usize> = HashMap::new();
    for e in entries.iter().filter(|e| e.out.is_none()) {
        if let Some(p) = plain(e) {
            *count.entry(fold(&p)).or_default() += 1;
        }
    }
    let mut outs = Vec::with_capacity(entries.len());
    for (i, e) in entries.iter().enumerate() {
        let out = match &e.out {
            Some(o) => o.clone(),
            None => {
                let Some(d) = dir else {
                    bail!(
                        "job {} ({}) has no output: give it an `out`, or the batch a folder",
                        i + 1,
                        e.midi.display()
                    );
                };
                let p = plain(e).expect("a folder is given");
                if count[&fold(&p)] > 1 {
                    let fonts: Vec<String> = e.soundfonts.iter().map(|f| stem(f)).collect();
                    d.join(format!("{} ({}).{ext}", stem(&e.midi), fonts.join("+")))
                } else {
                    p
                }
            }
        };
        outs.push(out);
    }
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (i, out) in outs.iter().enumerate() {
        if let Some(first) = seen.insert(fold(out), i) {
            bail!(
                "jobs {} ({}) and {} ({}) would both write {}; give one of them an `out`",
                first + 1,
                entries[first].midi.display(),
                i + 1,
                entries[i].midi.display(),
                out.display()
            );
        }
        for e in entries {
            let input = e.midi.as_path();
            if fold(input) == fold(out) || e.soundfonts.iter().any(|f| fold(f) == fold(out)) {
                bail!("job {} would overwrite {}, which the batch reads", i + 1, out.display());
            }
        }
    }
    Ok(outs)
}

/// The jobs for `entries`: `template` with each entry's own MIDI, soundfonts, \[12\]
pub fn jobs_from(
    template: &Job,
    entries: &[Entry],
    dir: Option<&Path>,
    ext: &str,
) -> Result<Vec<Job>> {
    for (i, e) in entries.iter().enumerate() {
        if e.soundfonts.is_empty() {
            bail!("job {} ({}) has no soundfonts", i + 1, e.midi.display());
        }
    }
    let outs = assign_outputs(entries, dir, ext)?;
    Ok(entries
        .iter()
        .zip(outs)
        .map(|(e, out)| Job {
            midi: e.midi.clone(),
            soundfonts: e.soundfonts.clone(),
            sf_programs: e.sf_programs.clone(),
            out,
            seconds: e.seconds.or(template.seconds),
            ..template.clone()
        })
        .collect())
}

/// `song.wav` -> `song.partial.wav`.
fn partial_of(out: &Path) -> PathBuf {
    let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let renamed = match name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.partial.{ext}"),
        None => format!("{name}.partial"),
    };
    out.with_file_name(renamed)
}

// ---- plan and run --------------------------------------------------------

/// A batch, checked and ready.
pub struct BatchPlan {
    pub jobs: Vec<Job>,
    plans: Vec<session::Plan>,
    /// Indexes into `jobs`, one group per distinct soundfont set, in the order \[13\]
    groups: Vec<Vec<usize>>,
    /// A bank the caller already loaded for the one set every job uses.
    preloaded: Option<std::sync::Arc<crate::bank::Bank>>,
}

impl BatchPlan {
    /// Use `bank`, already loaded from this batch's soundfonts with its render \[14\]
    pub fn preload(&mut self, bank: std::sync::Arc<crate::bank::Bank>) -> bool {
        if self.groups.len() != 1 {
            return false;
        }
        self.preloaded = Some(bank);
        true
    }

    /// Job indexes by soundfont set, in the order they will run.
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }
}

/// How the files of a batch are scanned for a note key over 127, which is how \[15\]
fn extended_key_files(jobs: &[Job]) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    let once: Vec<(usize, &Path)> = jobs
        .iter()
        .enumerate()
        .filter(|(_, j)| seen.insert(j.midi.as_path()))
        .map(|(i, j)| (i, j.midi.as_path()))
        .collect();
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8).min(once.len().max(1));
    let chunk = once.len().div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = once
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .filter(|(_, p)| crate::midi::uses_extended_keys(p).unwrap_or(false))
                        .map(|&(i, _)| i)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    })
}

/// A batch does not render files written for the 31-EDO template, for now. \[16\]
fn refuse_31edo(jobs: &[Job]) -> Result<()> {
    if let Some(i) = jobs.iter().position(|j| j.cfg.edo31) {
        bail!(
            "job {} ({}): --31edo plays a file as the 31-EDO template writes it, and \
             in a batch it would play every file that way. A batch cannot render a 31-EDO file \
             yet: render it on its own",
            i + 1,
            jobs[i].midi.display()
        );
    }
    let wide = extended_key_files(jobs);
    if wide.is_empty() {
        return Ok(());
    }
    let mut named: Vec<String> = wide
        .iter()
        .take(3)
        .map(|&i| format!("job {} ({})", i + 1, jobs[i].midi.display()))
        .collect();
    if wide.len() > 3 {
        named.push(format!("and {} more", wide.len() - 3));
    }
    bail!(
        "{} {} note keys over 127: written for the 31-EDO template, which a batch cannot \
         render yet. Render such a file on its own, with --31edo",
        named.join(", "),
        if wide.len() == 1 { "has" } else { "have" }
    );
}

/// Check a batch without loading anything. \[17\]
pub fn plan(jobs: Vec<Job>) -> Result<BatchPlan> {
    if jobs.is_empty() {
        bail!("a batch needs at least one job");
    }
    refuse_31edo(&jobs)?;
    let mut plans = Vec::with_capacity(jobs.len());
    for (i, job) in jobs.iter().enumerate() {
        let n = i + 1;
        if job.track.is_some() || job.stems.is_some() {
            bail!(
                "job {n} ({}): --track and --tracks render one MIDI by itself and do not \
                 combine with a batch",
                job.midi.display()
            );
        }
        if job.block_csv.is_some() {
            bail!("--block-csv follows one render's blocks, and a batch renders several");
        }
        if !job.midi.is_file() {
            bail!("job {n}: the MIDI {} is not a file", job.midi.display());
        }
        for sf in &job.soundfonts {
            if !sf.is_file() {
                bail!("job {n} ({}): the soundfont {} is not a file", job.midi.display(), sf.display());
            }
        }
        plans.push(
            session::plan(job)
                .with_context(|| format!("job {n} ({})", job.midi.display()))?,
        );
    }
    // [18]
    let first = format!("{:?}", plans[0].cfg);
    if let Some(i) = plans.iter().position(|p| format!("{:?}", p.cfg) != first) {
        bail!(
            "job {} has render settings of its own; the jobs of a batch share theirs",
            i + 1
        );
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut at: HashMap<(&[PathBuf], Option<&str>), usize> = HashMap::new();
    let entries: Vec<(&[PathBuf], Option<&str>)> = jobs
        .iter()
        .map(|j| (j.soundfonts.as_slice(), j.sf_programs.as_deref()))
        .collect();
    for (i, key) in entries.iter().enumerate() {
        let g = *at.entry(*key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    drop(at);
    drop(entries);
    Ok(BatchPlan { jobs, plans, groups, preloaded: None })
}

/// How one job ended.
#[derive(Debug, Clone)]
pub enum Outcome {
    /// Rendered, or stopped partway by the observer (`Summary::cancelled`).
    Done(Summary),
    /// With the reason, as the error reads in full.
    Failed(String),
    /// The batch was stopped before it got to this one.
    NotRun,
}

/// One job's result.
#[derive(Debug, Clone)]
pub struct JobResult {
    pub index: usize,
    pub midi: PathBuf,
    /// The file the job was given. A job stopped partway is on disk as \[19\]
    pub out: PathBuf,
    /// Where the audio is, if any: `out` for a finished job, the `.partial.` \[20\]
    pub written: Option<PathBuf>,
    pub outcome: Outcome,
}

impl JobResult {
    pub fn is_ok(&self) -> bool {
        matches!(&self.outcome, Outcome::Done(s) if !s.cancelled)
    }
}

/// How a batch ended.
#[derive(Debug, Clone, Default)]
pub struct BatchSummary {
    /// In the order the jobs were listed.
    pub results: Vec<JobResult>,
    /// Soundfont sets loaded. One per distinct set that had a job to run.
    pub loads: usize,
    /// The observer stopped it.
    pub cancelled: bool,
    pub wall_secs: f64,
}

impl BatchSummary {
    pub fn done(&self) -> usize {
        self.results.iter().filter(|r| r.is_ok()).count()
    }
    pub fn failed(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.outcome, Outcome::Failed(_))).count()
    }
}

/// A front end's view of a batch. `job` is the observer each job's own render \[21\]
pub trait BatchObserver {
    fn job(&mut self) -> &mut dyn Observer;
    /// A soundfont set is about to load: the `set`th of `sets` (from 1), for \[22\]
    fn loading(&mut self, _set: usize, _sets: usize, _soundfonts: &[PathBuf], _first_job: usize, _jobs: usize) {}
    /// Job `index` (from 0) of `total` is about to start.
    fn started(&mut self, _index: usize, _total: usize, _job: &Job) {}
    fn finished(&mut self, _total: usize, _result: &JobResult) {}
}

/// Run a checked batch.
pub fn run(batch: BatchPlan, obs: &mut dyn BatchObserver) -> BatchSummary {
    let t0 = Instant::now();
    let BatchPlan { jobs, plans, groups, mut preloaded } = batch;
    let total = jobs.len();
    let mut plans: Vec<Option<session::Plan>> = plans.into_iter().map(Some).collect();
    let mut results: Vec<Option<JobResult>> = vec![None; total];
    let mut loads = 0usize;
    let mut cancelled = false;

    'groups: for (g, group) in groups.iter().enumerate() {
        if obs.job().cancelled() {
            cancelled = true;
            break;
        }
        let first = &jobs[group[0]];
        log::info!(
            target: TARGET,
            "batch: soundfont set {} of {}, for {} job{}",
            g + 1,
            groups.len(),
            group.len(),
            if group.len() == 1 { "" } else { "s" }
        );

        // [23]
        let loaded = match preloaded.take() {
            Some(bank) => {
                session::log_bank(&bank, None);
                Ok(bank)
            }
            None => {
                obs.loading(g + 1, groups.len(), &first.soundfonts, group[0], group.len());
                let cfg = plans[group[0]].as_ref().expect("a job is planned once").cfg.clone();
                let started = Instant::now();
                loads += 1;
                let loaded = session::load_layered(&first.soundfonts, first.sf_programs.as_deref(), &cfg);
                if let Ok(b) = &loaded {
                    session::log_bank(b, Some(started.elapsed()));
                }
                loaded.map(std::sync::Arc::new)
            }
        };
        let bank = match loaded {
            Ok(b) => b,
            Err(e) => {
                let why = format!("{e:#}");
                log::error!(target: TARGET, "batch: the soundfont set would not load: {why}");
                for &i in group {
                    let r = JobResult {
                        index: i,
                        midi: jobs[i].midi.clone(),
                        out: jobs[i].out.clone(),
                        written: None,
                        outcome: Outcome::Failed(why.clone()),
                    };
                    obs.finished(total, &r);
                    results[i] = Some(r);
                }
                continue;
            }
        };

        for &i in group {
            if obs.job().cancelled() {
                cancelled = true;
                break 'groups;
            }
            let job = &jobs[i];
            obs.started(i, total, job);
            let mut job_here = job.clone();
            job_here.out = partial_of(&job.out);
            if let Some(dir) = job.out.parent().filter(|d| !d.as_os_str().is_empty()) {
                if let Err(e) = std::fs::create_dir_all(dir) {
                    let r = JobResult {
                        index: i,
                        midi: job.midi.clone(),
                        out: job.out.clone(),
                        written: None,
                        outcome: Outcome::Failed(format!("creating {}: {e}", dir.display())),
                    };
                    obs.finished(total, &r);
                    results[i] = Some(r);
                    continue;
                }
            }
            let plan = plans[i].take().expect("a job is run once");
            let outcome = session::run(&job_here, plan, Some(bank.clone()), obs.job());
            let r = match outcome {
                Ok(s) if s.cancelled => {
                    cancelled = true;
                    JobResult {
                        index: i,
                        midi: job.midi.clone(),
                        out: job.out.clone(),
                        written: job_here.out.is_file().then(|| job_here.out.clone()),
                        outcome: Outcome::Done(s),
                    }
                }
                Ok(s) => match std::fs::rename(&job_here.out, &job.out) {
                    Ok(()) => JobResult {
                        index: i,
                        midi: job.midi.clone(),
                        out: job.out.clone(),
                        written: Some(job.out.clone()),
                        outcome: Outcome::Done(s),
                    },
                    Err(e) => JobResult {
                        index: i,
                        midi: job.midi.clone(),
                        out: job.out.clone(),
                        written: Some(job_here.out.clone()),
                        outcome: Outcome::Failed(format!(
                            "rendered, but {} could not be renamed to {}: {e}",
                            job_here.out.display(),
                            job.out.display()
                        )),
                    },
                },
                Err(e) => JobResult {
                    index: i,
                    midi: job.midi.clone(),
                    out: job.out.clone(),
                    written: job_here.out.is_file().then(|| job_here.out.clone()),
                    outcome: Outcome::Failed(format!("{e:#}")),
                },
            };
            obs.finished(total, &r);
            let stop = cancelled;
            results[i] = Some(r);
            if stop {
                break 'groups;
            }
        }
        // `bank` goes here, before the next set is read.
    }

    let results = results
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            r.unwrap_or_else(|| JobResult {
                index: i,
                midi: jobs[i].midi.clone(),
                out: jobs[i].out.clone(),
                written: None,
                outcome: Outcome::NotRun,
            })
        })
        .collect();
    BatchSummary { results, loads, cancelled, wall_secs: t0.elapsed().as_secs_f64() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(midi: &str, fonts: &[&str]) -> Entry {
        Entry {
            midi: midi.into(),
            soundfonts: fonts.iter().map(PathBuf::from).collect(),
            sf_programs: None,
            out: None,
            seconds: None,
        }
    }

    #[test]
    fn plain_names_when_nothing_collides() {
        let outs = assign_outputs(&[e("a/one.mid", &["x.sf2"]), e("b/two.mid", &["x.sf2"])], Some(Path::new("out")), "wav").unwrap();
        assert_eq!(outs, [Path::new("out/one.wav"), Path::new("out/two.wav")]);
    }

    #[test]
    fn colliding_names_both_take_their_soundfonts_whichever_is_first() {
        let a = e("a/song.mid", &["gm.sf2", "piano.sfz"]);
        let b = e("b/song.mid", &["gm.sf2"]);
        let fwd = assign_outputs(&[a.clone(), b.clone()], Some(Path::new("out")), "wav").unwrap();
        let rev = assign_outputs(&[b, a], Some(Path::new("out")), "wav").unwrap();
        assert_eq!(fwd[0], Path::new("out/song (gm+piano).wav"));
        assert_eq!(fwd[1], Path::new("out/song (gm).wav"));
        assert_eq!(fwd[0], rev[1]);
        assert_eq!(fwd[1], rev[0]);
    }

    #[test]
    fn one_midi_on_several_sets_is_named_by_set() {
        let outs = assign_outputs(
            &[e("song.mid", &["piano.sfz"]), e("song.mid", &["gm.sf2"])],
            Some(Path::new("out")),
            "flac",
        )
        .unwrap();
        assert_eq!(outs, [Path::new("out/song (piano).flac"), Path::new("out/song (gm).flac")]);
    }

    #[test]
    fn a_name_taken_after_that_is_refused_naming_both_jobs() {
        let err = assign_outputs(
            &[e("a/song.mid", &["x/gm.sf2"]), e("b/song.mid", &["y/gm.sf2"])],
            Some(Path::new("out")),
            "wav",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("jobs 1") && err.contains("and 2"), "{err}");
    }

    #[test]
    fn names_are_compared_without_case() {
        let err = assign_outputs(
            &[e("a/Song.mid", &["gm.sf2"]), e("b/song.mid", &["gm.sf2"])],
            Some(Path::new("out")),
            "wav",
        );
        // Both are `song (gm)` once folded, which still collides.
        assert!(err.is_err());
    }

    #[test]
    fn an_own_out_is_kept_and_may_not_collide() {
        let mut a = e("a.mid", &["x.sf2"]);
        a.out = Some("elsewhere/a.wav".into());
        let outs = assign_outputs(&[a.clone(), e("b.mid", &["x.sf2"])], Some(Path::new("out")), "wav").unwrap();
        assert_eq!(outs[0], Path::new("elsewhere/a.wav"));
        let mut b = e("b.mid", &["x.sf2"]);
        b.out = Some("elsewhere/a.wav".into());
        assert!(assign_outputs(&[a, b], None, "wav").is_err());
    }

    #[test]
    fn no_folder_and_no_out_is_refused() {
        assert!(assign_outputs(&[e("a.mid", &["x.sf2"])], None, "wav").is_err());
    }

    #[test]
    fn an_output_may_not_be_an_input() {
        let mut a = e("a.mid", &["x.sf2"]);
        a.out = Some("x.sf2".into());
        assert!(assign_outputs(&[a], None, "wav").is_err());
    }

    #[test]
    fn partial_keeps_the_extension_last() {
        assert_eq!(partial_of(Path::new("o/song (gm).wav")), Path::new("o/song (gm).partial.wav"));
        assert_eq!(partial_of(Path::new("song.opus")), Path::new("song.partial.opus"));
        assert_eq!(partial_of(Path::new("song")), Path::new("song.partial"));
    }

    fn parse(json: &str) -> Result<(Shared, Vec<Entry>)> {
        let f: File = serde_json::from_str(json)?;
        f.resolve(Path::new("base"))
    }

    #[test]
    fn a_file_inherits_soundfonts_and_a_job_replaces_them_with_their_programs() {
        let (shared, entries) = parse(
            r#"{"version":1,"out":"o","soundfonts":["gm.sf2"],"sf_programs":"0-7","jobs":[
                {"midi":"a.mid"},
                {"midi":"b.mid","soundfonts":["p.sfz"]},
                {"midi":"c.mid","sf_programs":"0"},
                {"midi":"/abs/d.mid","out":"/abs/d.wav","seconds":5}]}"#,
        )
        .unwrap();
        assert_eq!(shared.out_dir.as_deref(), Some(Path::new("base/o")));
        assert_eq!(shared.out_format, "wav");
        assert_eq!(entries[0].soundfonts, [Path::new("base/gm.sf2")]);
        assert_eq!(entries[0].sf_programs.as_deref(), Some("0-7"));
        assert_eq!(entries[1].soundfonts, [Path::new("base/p.sfz")]);
        assert_eq!(entries[1].sf_programs, None, "programs do not outlive a change of soundfonts");
        assert_eq!(entries[2].soundfonts, [Path::new("base/gm.sf2")]);
        assert_eq!(entries[2].sf_programs.as_deref(), Some("0"));
        assert_eq!(entries[3].midi, Path::new("/abs/d.mid"));
        assert_eq!(entries[3].seconds, Some(5.0));
    }

    #[test]
    fn a_file_with_a_mistake_is_refused_with_the_job_named() {
        // An unknown key.
        assert!(parse(r#"{"version":1,"jobs":[{"midi":"a.mid","soundfont":["x"]}]}"#).is_err());
        // The wrong version.
        assert!(parse(r#"{"version":2,"jobs":[{"midi":"a.mid"}]}"#).is_err());
        // No jobs.
        assert!(parse(r#"{"version":1,"jobs":[]}"#).is_err());
        // A job with no soundfonts to use.
        let err = parse(r#"{"version":1,"jobs":[{"midi":"a.mid"}]}"#).unwrap_err().to_string();
        assert!(err.contains("job 1") && err.contains("a.mid"), "{err}");
        // Seconds that mean nothing.
        assert!(parse(r#"{"version":1,"soundfonts":["x"],"jobs":[{"midi":"a.mid","seconds":0}]}"#).is_err());
    }
}
