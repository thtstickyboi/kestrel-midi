// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Resuming a render of one file (`Job::checkpoint`): what `session::run_inner` \[1\]

use crate::backend::Backend;
use crate::config::Config;
use crate::driver::Driver;
use crate::resume::{self, Audio, AudioKind, Header, Identity, Single, Spec};
use crate::session::{Job, Sink};
use crate::snap::{Dec, Enc};
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const TARGET: &str = "kestrel";

/// Take a checkpoint's state into a driver and a backend that were just opened \[2\]
pub(crate) fn load(cp: &resume::Checkpoint, driver: &mut Driver, backend: &mut dyn Backend) -> Result<()> {
    let mut payload = cp.payload()?;
    let mut n = [0u8; 8];
    payload.read_exact(&mut n)?;
    let len = u64::from_le_bytes(n);
    if len > cp.payload_len() {
        bail!("the saved state is longer than the checkpoint that holds it");
    }
    let mut bytes = vec![0u8; len as usize];
    payload.read_exact(&mut bytes)?;
    let mut d = Dec::new(&bytes);
    driver.load_state(&mut d)?;
    d.finish()?;
    backend.load_state(&mut payload)?;
    let mut extra = [0u8; 1];
    if payload.read(&mut extra)? != 0 {
        bail!("there is more in the checkpoint than this build reads");
    }
    driver.reinstall_variants(backend)
}

/// Writes a single render's checkpoints. Built once the render is set up, \[3\]
pub(crate) struct Saver {
    /// The `.krsm`.
    pub path: PathBuf,
    every: Option<Duration>,
    last: Instant,
    argv: Vec<String>,
    backend: String,
    identity: Identity,
    voices: u32,
    stop_after_blocks: Option<u64>,
}

/// What `Saver::identity` is of: the render as it was asked for, before the \[4\]
pub(crate) fn identity(job: &Job, cfg: &Config) -> Result<Identity> {
    Ok(Identity {
        midi: resume::Fingerprint::of(&job.midi)?,
        soundfonts: job.soundfonts.iter().map(|p| resume::SoundfontPrint::of(p)).collect::<Result<_>>()?,
        merge: false,
        tracks: job.track.map(|t| vec![t.index]).unwrap_or_default(),
        block_samples: cfg.block_samples(),
        seconds: job.seconds,
        single: true,
    })
}

/// Why this render cannot save its progress, or `None` if it can.
pub(crate) fn unsupported(job: &Job, cfg: &Config) -> Option<&'static str> {
    if job.block_csv.is_some() {
        Some("--block-csv follows one run of blocks, which a resume would split")
    } else if cfg.phase.active() {
        Some("analytic phase rotation keeps per-render caches that a checkpoint does not hold yet")
    } else {
        None
    }
}

impl Saver {
    /// `identity` is what `identity` returned, and `backend` is `cpu` or \[5\]
    pub fn new(spec: &Spec, job: &Job, identity: Identity, backend: String, voices: u32) -> Saver {
        let path = spec.path.clone().unwrap_or_else(|| resume::default_path(&job.out, &job.midi, true));
        Saver {
            path,
            every: spec.every,
            last: Instant::now(),
            argv: spec.argv.clone(),
            backend,
            identity,
            voices,
            stop_after_blocks: spec.stop_after_blocks,
        }
    }

    /// What only the running render knows: where it runs and under how many \[6\]
    pub fn set_run(&mut self, backend: String, voices: u32) {
        self.backend = backend;
        self.voices = voices;
    }

    /// Where the audio being written goes until the render is whole.
    pub fn partial(&self, out: &Path) -> PathBuf {
        resume::partial_of(out)
    }

    /// Where the samples fed to an encoder are kept.
    pub fn pcm(&self) -> PathBuf {
        let mut p = self.path.as_os_str().to_owned();
        p.push(".pcm");
        PathBuf::from(p)
    }

    /// Whether to stop after the block that is on the device: a checkpoint is \[7\]
    pub fn hold(&self, cancelled: bool, done: u64) -> bool {
        cancelled || self.due() || self.stop_hook(done + 1)
    }

    /// The development hook that stops a render after a number of blocks.
    pub fn stop_hook(&self, blocks: u64) -> bool {
        self.stop_after_blocks.is_some_and(|n| blocks >= n)
    }

    fn due(&self) -> bool {
        self.every.is_some_and(|e| self.last.elapsed() >= e)
    }

    /// Write the checkpoint of the render as it stands: the driver, the \[8\]
    pub fn save(
        &mut self,
        driver: &Driver,
        backend: &mut dyn Backend,
        audio: Audio,
        peak_voices: u64,
    ) -> Result<u64> {
        let header = Header {
            format: resume::FORMAT,
            kestrel: env!("CARGO_PKG_VERSION").to_string(),
            build: resume::build_id().to_string(),
            argv: self.argv.clone(),
            backend: self.backend.clone(),
            midi: self.identity.midi.clone(),
            soundfonts: self.identity.soundfonts.clone(),
            merge: false,
            tracks: self.identity.tracks.clone(),
            voices_each: self.voices,
            block_samples: self.identity.block_samples,
            seconds: self.identity.seconds,
            finished: Vec::new(),
            in_flight: Vec::new(),
            mix_len: 0,
            single: Some(Single { blocks: driver.stats.blocks, peak_voices, audio }),
        };
        let mut w = resume::Writer::create(&self.path, &header)?;
        let written = (|| -> Result<()> {
            let mut e = Enc::new();
            driver.save_state(&mut e)?;
            let bytes = e.into_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(&bytes)?;
            backend.save_state(&mut w)
        })();
        match written {
            Ok(()) => {
                let size = w.commit()?;
                self.last = Instant::now();
                Ok(size)
            }
            Err(e) => {
                w.abandon();
                Err(e)
            }
        }
    }

    /// Save, and say so; a checkpoint that could not be written is a warning and \[9\]
    pub fn save_or_warn(
        &mut self,
        driver: &Driver,
        backend: &mut dyn Backend,
        audio: Audio,
        peak_voices: u64,
        stopping: bool,
    ) -> bool {
        let t0 = Instant::now();
        match self.save(driver, backend, audio, peak_voices) {
            Ok(bytes) if stopping => {
                log::warn!(
                    target: TARGET,
                    "progress saved to {} ({}, {:.2}s). Continue it with: kestrel --force-cli resume \"{}\"",
                    self.path.display(),
                    mib(bytes),
                    t0.elapsed().as_secs_f64(),
                    self.path.display()
                );
                true
            }
            Ok(bytes) => {
                log::info!(
                    target: TARGET,
                    "progress saved to {} ({}, the render held for {:.2}s)",
                    self.path.display(),
                    mib(bytes),
                    t0.elapsed().as_secs_f64()
                );
                true
            }
            Err(e) => {
                log::warn!(
                    target: TARGET,
                    "progress could not be saved to {}: {e:#}. The render goes on, and cannot be resumed from here",
                    self.path.display()
                );
                false
            }
        }
    }

    /// The render is whole: the checkpoint has done its job.
    pub fn discard(&self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.pcm());
    }
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / 1048576.0)
}

/// The float samples fed to an encoder, kept beside the checkpoint.
struct Tee {
    file: BufWriter<File>,
    bytes: u64,
    scratch: Vec<u8>,
}

impl Tee {
    fn create(path: &Path) -> Result<Tee> {
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        Ok(Tee { file: BufWriter::with_capacity(1 << 20, file), bytes: 0, scratch: Vec::new() })
    }

    /// Go on with the samples already kept, cut back to `bytes`.
    fn resume(path: &Path, bytes: u64) -> Result<Tee> {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        file.set_len(bytes)?;
        file.seek(SeekFrom::Start(bytes))?;
        Ok(Tee { file: BufWriter::with_capacity(1 << 20, file), bytes, scratch: Vec::new() })
    }

    fn write(&mut self, samples: &[f32]) -> Result<()> {
        self.scratch.clear();
        self.scratch.reserve(samples.len() * 4);
        for s in samples {
            self.scratch.extend_from_slice(&s.to_le_bytes());
        }
        self.file.write_all(&self.scratch)?;
        self.bytes += self.scratch.len() as u64;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_data()?;
        Ok(())
    }
}

/// Where a render's blocks go, and what a checkpoint needs of it: the `Sink` \[10\]
pub(crate) struct Output {
    sink: Sink,
    tee: Option<Tee>,
    /// The file the audio is in (a WAV) or the kept samples (an encoded stream).
    audio_path: PathBuf,
}

impl Output {
    /// The render as it has always been written: straight to its file, with \[11\]
    pub fn plain(
        out: &Path,
        cfg: &Config,
        encoder: Option<&(crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
        wav_format: crate::wav::SampleFormat,
    ) -> Result<Output> {
        Ok(Output { sink: Sink::create(out, cfg, encoder, wav_format)?, tee: None, audio_path: out.to_path_buf() })
    }

    /// Start writing `partial`, the name the audio has until the render is \[12\]
    pub fn create(
        job: &Job,
        cfg: &Config,
        encoder: Option<&(crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
        partial: &Path,
        pcm: &Path,
    ) -> Result<Output> {
        let sink = Sink::create(partial, cfg, encoder, job.wav_format)?;
        Ok(match encoder {
            Some(_) => Output { sink, tee: Some(Tee::create(pcm)?), audio_path: pcm.to_path_buf() },
            None => Output { sink, tee: None, audio_path: partial.to_path_buf() },
        })
    }

    /// Go on from the audio a checkpoint stands behind.
    pub fn resume(
        job: &Job,
        cfg: &Config,
        encoder: Option<&(crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
        audio: &Audio,
        partial: &Path,
        pcm: &Path,
    ) -> Result<Output> {
        match (audio.kind, encoder) {
            (AudioKind::Wav, None) => {
                let audio = Audio { path: partial.display().to_string(), ..audio.clone() };
                audio.check()?;
                let w = crate::wav::WavWriter::resume(partial, cfg.sample_rate, 2, job.wav_format, audio.bytes)?;
                Ok(Output { sink: Sink::Wav(w), tee: None, audio_path: partial.to_path_buf() })
            }
            (AudioKind::Pcm, Some((f, preset))) => {
                let audio = Audio { path: pcm.display().to_string(), ..audio.clone() };
                audio.check()?;
                let mut enc = f.encode_to(partial, cfg.sample_rate, preset)?;
                // [13]
                let t0 = Instant::now();
                let mut input = File::open(pcm).with_context(|| format!("opening {}", pcm.display()))?;
                let mut buf = vec![0u8; 1 << 20];
                let mut left = audio.bytes;
                while left > 0 {
                    let n = left.min(buf.len() as u64) as usize;
                    input.read_exact(&mut buf[..n])?;
                    enc.write_raw(&buf[..n])?;
                    left -= n as u64;
                }
                log::info!(
                    target: TARGET,
                    "fed the encoder the {} rendered so far, in {:.2}s",
                    mib(audio.bytes),
                    t0.elapsed().as_secs_f64()
                );
                Ok(Output {
                    sink: Sink::Encoded(Box::new(enc)),
                    tee: Some(Tee::resume(pcm, audio.bytes)?),
                    audio_path: pcm.to_path_buf(),
                })
            }
            (AudioKind::Wav, Some(_)) | (AudioKind::Pcm, None) => bail!(
                "this checkpoint was made for {} output and this render writes {}: the output format is different",
                if audio.kind == AudioKind::Wav { "WAV" } else { "an encoded" },
                if audio.kind == AudioKind::Wav { "an encoded format" } else { "WAV" }
            ),
        }
    }

    pub fn write_block(&mut self, block: &[f32]) -> Result<()> {
        self.sink.write_block(block)?;
        if let Some(t) = self.tee.as_mut() {
            t.write(block)?;
        }
        Ok(())
    }

    /// Everything written so far is on the disk, and this says how much and what \[14\]
    pub fn checkpoint(&mut self) -> Result<Audio> {
        let (kind, bytes) = match (&mut self.sink, self.tee.as_mut()) {
            (Sink::Wav(w), _) => {
                w.sync()?;
                (AudioKind::Wav, w.data_bytes())
            }
            (_, Some(t)) => {
                t.sync()?;
                (AudioKind::Pcm, t.bytes)
            }
            (Sink::Encoded(_), None) => bail!("an encoded render keeps its samples, and this one has none"),
        };
        let start = if kind == AudioKind::Wav { resume::WAV_HEADER } else { 0 };
        Ok(Audio {
            kind,
            path: self.audio_path.display().to_string(),
            bytes,
            tail: Audio::tail_of(&self.audio_path, start, bytes)?,
        })
    }

    /// Close the file. For an encoded stream that is the encoder finishing, \[15\]
    pub fn finish(self) -> Result<u64> {
        drop(self.tee);
        self.sink.finish()
    }
}
