// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `kestrel --force-cli render`: the arguments turned into a `session::Job`, \[1\]

use crate::RenderArgs;
use anyhow::Result;
use kestrel::session::{self, Job, Observer, Phase, Tick};
use std::time::Instant;

/// The target the progress lines are logged under, the same one \[2\]
const TARGET: &str = "kestrel";

impl RenderArgs {
    /// The render these arguments describe, with everything clap cannot check \[3\]
    pub fn to_job(&self) -> Result<Job> {
        let (cfg, backend) = self.to_config()?;
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
            backend,
            cfg,
        })
    }
}

/// The command line's observer: a progress line a second, and the per-pass \[4\]
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
                secs / wall.max(1e-9)
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
    if args.progress.is_some() {
        return crate::feed::run(&args);
    }
    let job = args.to_job()?;
    let plan = session::plan(&job)?;
    let mut obs = LogObserver {
        last_report: Instant::now(),
    };
    session::run(&job, plan, None, &mut obs)?;
    Ok(())
}
