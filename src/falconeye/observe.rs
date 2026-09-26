// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What `session::run` hands FalconEye: the render log's header, the \[1\]

use super::renderlog;
use crate::session::{Job, Observer, Phase, Plan, Setup, Tick, TrackProgress};
use std::path::Path;
use std::time::Instant;
#[cfg(feature = "dev")]
use crate::backend::Backend;
#[cfg(feature = "dev")]
use std::time::Duration;

#[cfg(feature = "dev")]
const TARGET: &str = "kestrel";

/// The render log's header: the files, what the render is, and the whole \[2\]
pub(crate) fn open_log(job: &Job, plan: &Plan) -> Option<renderlog::Guard> {
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut keep: Vec<String> = vec![name(&job.midi)];
    if let Some(stem) = job.midi.file_stem() {
        keep.push(stem.to_string_lossy().into_owned());
    }
    keep.extend(job.soundfonts.iter().map(|p| name(p)));
    let keep: Vec<&str> = keep.iter().map(String::as_str).collect();
    let size = |p: &Path| match std::fs::metadata(p) {
        Ok(m) => format!("{:.1} MiB", m.len() as f64 / 1048576.0),
        Err(e) => format!("unreadable: {e}"),
    };
    renderlog::open(&job.midi, &keep, &|w| {
        w("FILE", &format!("midi: {} ({})", job.midi.display(), size(&job.midi)));
        for (i, sf) in job.soundfonts.iter().enumerate() {
            w("FILE", &format!("soundfont {}: {} ({})", i + 1, sf.display(), size(sf)));
        }
        if let Some(p) = &job.sf_programs {
            w("FILE", &format!("programs the last soundfont takes: {p}"));
        }
        w("FILE", &format!(
            "output: {} ({})",
            job.out.display(),
            if plan.encoder.is_some() { "encoded with ffmpeg".to_string() } else { format!("{:?}", job.wav_format) }
        ));
        let what = match (&job.track, &job.stems) {
            (Some(t), _) => format!("track {} alone, setup tracks {:?}", t.index + 1, t.setup),
            (None, Some(st)) => format!(
                "per track: {:?}, setup tracks {:?}, {} jobs, {}",
                st.tracks, st.setup, st.jobs, if st.merge { "merged into one file" } else { "a file a track" }
            ),
            (None, None) => "the whole file".to_string(),
        };
        w("JOB", &format!("render: {what}; backend {:?}; seconds {:?}; ceiling {:?}", job.backend, job.seconds, job.ceiling_db));
        w("JOB", &format!("config: {:#?}", plan.cfg));
    })
}

/// Passes everything to the front end's observer, and writes the render log's \[3\]
pub(crate) struct Logged<'a> {
    inner: &'a mut dyn Observer,
    last: Instant,
    /// The longest the host waited on the device since the last breadcrumb.
    max_wait_us: u64,
    vram: Option<crate::gpu::vram::VramWatch>,
}

impl<'a> Logged<'a> {
    pub(crate) fn new(inner: &'a mut dyn Observer) -> Logged<'a> {
        Logged { inner, last: Instant::now(), max_wait_us: 0, vram: None }
    }

    fn vram_text(&self) -> String {
        const MIB: f64 = 1048576.0;
        let Some(m) = self.vram.as_ref().and_then(|v| v.latest()) else {
            return String::new();
        };
        let opt = |v: Option<u64>| v.map(|b| format!("{:.0}", b as f64 / MIB)).unwrap_or_else(|| "?".into());
        format!(
            " | vram {} of {:.0} MiB in use, this process {} of a {} MiB budget",
            opt(m.dedicated_used),
            m.dedicated_total as f64 / MIB,
            opt(m.process_used),
            opt(m.process_budget)
        )
    }
}

impl Observer for Logged<'_> {
    fn phase(&mut self, phase: Phase) {
        renderlog::note("PHASE", &format!("{phase:?}"));
        renderlog::set_progress(phase == Phase::Rendering, 0);
        self.inner.phase(phase);
    }

    fn setup(&mut self, setup: &Setup) {
        renderlog::note("SETUP", &format!("{setup:?}"));
        if let (Some(v), Some(d)) = (setup.vendor_id, setup.device_id) {
            self.vram = Some(crate::gpu::vram::VramWatch::start(v, d));
        }
        self.inner.setup(setup);
    }

    fn block(&mut self, t: &Tick) {
        renderlog::set_progress(true, t.driver.stats.blocks);
        self.max_wait_us = self.max_wait_us.max(t.driver.stats.last_wait_us);
        if t.last || self.last.elapsed().as_secs_f64() >= 1.0 {
            let d = &t.driver.stats;
            renderlog::note("AT", &format!(
                "{:.2}s audio, block {} | {} voices live | last block: want {} take {} stolen {} | {} dropped in all | longest device wait {:.1} ms{}",
                t.driver.seconds_rendered(),
                d.blocks,
                t.stats.active_voices,
                d.last_want,
                d.last_take,
                d.last_stolen,
                d.dropped,
                self.max_wait_us as f64 / 1000.0,
                self.vram_text(),
            ));
            self.max_wait_us = 0;
            self.last = Instant::now();
        }
        self.inner.block(t);
    }

    fn tracks(&mut self, p: &TrackProgress) {
        renderlog::set_progress(true, p.blocks);
        if self.last.elapsed().as_secs_f64() >= 1.0 {
            let now: Vec<String> = p
                .now
                .iter()
                .map(|n| {
                    if let Some(name) = &n.name {
                        renderlog::keep_name(name);
                    }
                    format!("track {}{} at {:.1}s", n.track, n.name.as_deref().map(|x| format!(" ({x})")).unwrap_or_default(), n.secs)
                })
                .collect();
            renderlog::note("AT", &format!(
                "tracks {} of {} done, {} running | {:.1}s audio over all tracks | {} of {} blocks | {} notes, {} stolen, {} dropped | now: {}{}",
                p.done,
                p.total,
                p.running,
                p.audio_secs,
                p.blocks,
                p.blocks_total,
                p.notes,
                p.stolen,
                p.dropped,
                if now.is_empty() { "-".to_string() } else { now.join(", ") },
                self.vram_text(),
            ));
            self.last = Instant::now();
        }
        self.inner.tracks(p);
    }

    fn cancelled(&self) -> bool {
        self.inner.cancelled()
    }
}

/// Fail on purpose partway through a render, to see what each kind of failure \[4\]
#[cfg(feature = "dev")]
pub(crate) fn crash_test(block: u64, backend: &mut dyn Backend) {
    static SPEC: std::sync::OnceLock<Option<(String, u64)>> = std::sync::OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        let v = std::env::var("KESTREL_CRASH_TEST").ok()?;
        let (kind, at) = v.split_once('@')?;
        Some((kind.to_string(), at.parse().ok()?))
    });
    let Some((kind, at)) = spec else { return };
    if block != *at {
        return;
    }
    log::warn!(target: TARGET, "crash test: {kind} at block {block}");
    match kind.as_str() {
        "panic" => panic!("crash test: a panic at block {block}"),
        "lost" => backend.lose_device(),
        "abort" => std::process::abort(),
        "exit" => std::process::exit(0),
        "segv" => super::winsys::access_violation(),
        // [5]
        "hang" => loop {
            std::thread::sleep(Duration::from_secs(1));
        },
        other => log::warn!(target: TARGET, "crash test: no such kind {other:?}"),
    }
}
