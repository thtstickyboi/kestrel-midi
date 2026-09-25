// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Kestrel's executable. Started with no arguments -- which is what
//! double-clicking it does -- it opens the guided renderer in `tui`. The whole
//! command line is still here, behind `--force-cli`, and both front ends drive
//! the one render pipeline in `render`. Neither is a windowed GUI: the guided
//! renderer is a terminal UI that borrows the platform's file pickers.

mod api;
mod feed;
mod render;
mod settings;
mod tui;
mod update;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use kestrel::limiter::LimiterMode;
use kestrel::config::{
    AdmitRule, BackendKind, Config, EnvelopeCurve, Interpolation, StealRule,
};
use kestrel::{gpu, load_bank};
#[cfg(feature = "dev")]
use kestrel::{testkit, wav};
use std::ffi::OsString;
use std::path::PathBuf;

/// The version `--version` prints, marked when this is a dev build so a bug
/// report says which option set it came from. See the `dev` feature.
#[cfg(feature = "dev")]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (dev)");
#[cfg(not(feature = "dev"))]
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What the command line logs unless `RUST_LOG` says otherwise: Kestrel's own
/// information, and only warnings from the GPU stack. wgpu's DX12 backend logs
/// every shader it generates at info, which opened every DX12 render with
/// pages of HLSL.
pub(crate) const LOG_FILTER: &str = "info,wgpu_core=warn,wgpu_hal=warn,naga=warn";

/// Which option set this build has: `dev` with the developer options and
/// commands, `release` without them, as the downloadable zips are built.
pub(crate) const BUILD: &str = if cfg!(feature = "dev") { "dev" } else { "release" };

#[derive(Parser)]
#[command(name = "kestrel", version = VERSION, about = "GPU-accelerated SoundFont/SFZ renderer for black MIDI")]
struct Cli {
    /// Run from the command line. Given any other arguments without this,
    /// Kestrel points at the guided renderer instead of running them; with no
    /// arguments at all it opens the guided renderer.
    // Routed on before clap runs, in `route`. Declared so clap accepts it in
    // any position and lists it in help, not so anything reads it here.
    #[allow(dead_code)]
    #[arg(long = "force-cli", global = true)]
    force_cli: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)] // RenderArgs is big and there is one of it
enum Cmd {
    /// Render a MIDI file to WAV.
    Render(RenderArgs),
    /// Print what the loader made of a soundfont or MIDI file.
    Info {
        /// A soundfont (.sf2, .sfz) or a MIDI file (.mid, .midi).
        path: PathBuf,
        /// Frames per render block, for the per-block density report. Match
        /// what you intend to render with.
        #[arg(long, default_value_t = 4096)]
        block: u32,
        #[arg(long, default_value_t = 48000)]
        rate: u32,
        /// Voices each note-on spawns, for the memory projection. 2 is right
        /// for any stereo library, because a stereo sample becomes two
        /// hard-panned mono regions; a compact mono bank is 1, and a layered
        /// one is higher. The figures scale with it.
        ///
        /// Named apart from `render --layers`, which is a different quantity:
        /// that one caps voices per note-on, this one describes how many the
        /// soundfont actually spawns.
        #[arg(long = "sf-layers", default_value_t = 2)]
        sf_layers: u32,
        /// The soundfont's release time, in seconds. Sets the window the voice
        /// estimate counts over; a voice outlives its note-off by roughly this
        /// long, and in black MIDI that is nearly its whole life.
        #[arg(long = "sf-release", default_value_t = 1.0)]
        sf_release: f64,
        /// Read `--sf-layers` and `--sf-release` off this soundfont instead of
        /// taking them on trust. Both describe the soundfont, so neither is
        /// something the caller should have to know: layers is how many
        /// regions a note-on actually matches, and release is `ampeg_release`,
        /// which the loader has already parsed.
        #[arg(short = 's', long = "soundfont")]
        soundfont: Option<PathBuf>,
    },
    /// List a MIDI's tracks: how many notes each holds and at which
    /// velocities, its channels, when its notes start and stop, and which
    /// tracks carry only controllers. Every track is read on its own and in
    /// parallel, so on a file with many tracks this is far quicker than `info`.
    Tracks {
        /// A MIDI file (.mid, .midi).
        midi: PathBuf,
        /// Threads to read tracks on; 0 is one per core.
        #[arg(long, default_value_t = 0)]
        jobs: usize,
        /// Sample rate the times are worked out at.
        #[arg(long, default_value_t = 48000)]
        rate: u32,
        /// Also count the notes `render --min-velocity` would keep, per track.
        #[arg(long = "min-velocity", value_name = "VEL")]
        min_velocity: Option<u8>,
    },
    /// Compare two WAV files and report the null-test difference.
    #[cfg(feature = "dev")]
    Null {
        /// The reference render.
        a: PathBuf,
        /// The render to compare against it.
        b: PathBuf,
        /// Fail if the peak difference is above this many dB.
        #[arg(long, default_value_t = -80.0)]
        threshold: f64,
    },
    /// List the GPU adapters wgpu can see.
    GpuInfo,
    /// Report which ffmpeg encoded output would use, where it was found, and
    /// which containers this build can write. Exits non-zero if ffmpeg is
    /// missing or an encoder a preset needs is absent, so it works as a setup
    /// check and not only as something to read.
    FfmpegInfo {
        /// Use this ffmpeg instead of searching. Overrides the FFMPEG
        /// environment variable.
        #[arg(long = "ffmpeg", value_name = "PATH")]
        ffmpeg: Option<PathBuf>,
    },
    /// Let another program drive Kestrel: requests as JSON lines on stdin,
    /// responses, log lines and live render telemetry as JSON lines on
    /// stdout. For GUIs; the protocol is in API.md.
    ///
    /// A session loads MIDIs and soundfonts, lists adapters, ffmpeg and every
    /// render option with its default, runs renders with progress, pull-mode
    /// snapshots and cancel, and keeps a soundfont loaded between renders. It
    /// ends when stdin closes.
    Api,
    /// Ask GitHub whether a newer Kestrel has been released. Reads the latest
    /// release's version and nothing else: nothing is downloaded or installed.
    /// The guided renderer does the same check when it starts, unless the
    /// KESTREL_NO_UPDATE_CHECK environment variable is set.
    CheckUpdate,
    /// Download an ffmpeg into an `ffmpeg/` directory beside this executable.
    ///
    /// Deliberately a separate command and never part of a render: a flag on
    /// `render` would end up in scripts and fetch unattended. Nothing is
    /// installed system-wide and PATH is not modified, so undoing this is
    /// deleting that directory.
    GetFfmpeg {
        /// Print what would be downloaded, and where, then exit.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long = "yes")]
        yes: bool,
        /// The SHA-256 the archive must have, from the vendor's published
        /// checksum. Required, because no digest is pinned in this build; the
        /// first run prints the one it saw so it can be checked and passed back.
        #[arg(long = "accept-hash", value_name = "SHA256")]
        accept_hash: Option<String>,
        /// Install somewhere other than beside the executable.
        #[arg(long = "dir", value_name = "PATH")]
        dir: Option<PathBuf>,
    },
    /// Write synthetic soundfonts and MIDI files, for benchmarking against
    /// material you can reproduce exactly.
    ///
    /// A development command: hidden from help and from the guided renderer,
    /// and only in a dev build, for the tests and the benchmarks.
    #[cfg(feature = "dev")]
    #[command(hide = true)]
    GenAssets {
        dir: PathBuf,
        /// Also write a soundfont with a sample pool of this many MiB, for
        /// benchmarking against a pool that does not fit in cache.
        #[arg(long = "big-mb")]
        big_mb: Option<usize>,
        /// Also write a MIDI with this many sustained notes, for benchmarking
        /// a pool that stays full.
        #[arg(long = "sustained")]
        sustained: Option<usize>,
    },
}

#[derive(Args)]
#[command(group(clap::ArgGroup::new("per_track").args(["track", "tracks"])))]
struct RenderArgs {
    /// Input MIDI file.
    midi: PathBuf,
    /// Soundfont, .sf2 or .sfz. Repeat to layer: each one is merged on top of
    /// the ones before it, and its presets replace anything already at the same
    /// bank and program. A General MIDI .sf2 followed by a piano .sfz gives GM
    /// with that piano on program 0, which is the usual arrangement.
    #[arg(short = 's', long = "soundfont", required = true, num_args = 1..)]
    soundfont: Vec<PathBuf>,
    /// Programs of bank 0 the *last* `-s` should take over, comma separated,
    /// e.g. `0,1` for both grand pianos. Ranges with `-`, so `0-7` is the whole
    /// GM piano family. Without this it takes only the program it declares,
    /// which for an .sfz is program 0.
    #[arg(long = "sf-programs", value_name = "LIST")]
    sf_programs: Option<String>,
    /// Output file. The extension picks the container: .wav is written
    /// directly, and .opus/.mp3/.ogg/.flac/.m4a are encoded through ffmpeg at a
    /// high-quality preset chosen per container. Encoding needs ffmpeg on PATH;
    /// see `kestrel ffmpeg-info`.
    #[arg(short = 'o', long = "out")]
    out: PathBuf,
    /// ffmpeg to use for encoded output. Only needed when it is not on PATH or
    /// a specific build is wanted; overrides the FFMPEG environment variable.
    #[arg(long = "ffmpeg", value_name = "PATH")]
    ffmpeg: Option<PathBuf>,

    /// Which renderer: the GPU, or the single-threaded CPU reference the GPU
    /// is checked against. The CPU path exists for null tests and is far too
    /// slow for real material.
    #[arg(long, default_value = "gpu", value_parser = ["cpu", "gpu"])]
    backend: String,
    /// Output sample rate in Hz. Every sample in the soundfont is converted to
    /// it when the soundfont loads.
    #[arg(long, default_value_t = 48000)]
    rate: u32,
    /// Frames per render block. Admission and stealing are decided once a
    /// block, so this is also their granularity.
    #[arg(long, default_value_t = 4096)]
    block: u32,
    /// Most voices sounding at once. `gpu-info` prints the largest each
    /// adapter takes.
    #[arg(long = "max-voices", default_value_t = 1 << 20)]
    max_voices: u32,
    /// Sample interpolation: nearest, linear or cubic. Cubic reads twice as
    /// many samples a voice.
    #[arg(long, default_value = "linear")]
    interp: String,
    /// Sample phase policy. Analytic prepares a quadrature cache during loading.
    #[arg(long = "phase-mode", default_value = "baseline", value_parser = ["baseline", "analytic"])]
    phase_mode: String,
    /// Analytic rotation strength, from 0 (unchanged) to 1 (full range).
    #[arg(long = "phase-strength", default_value_t = 1.0)]
    phase_strength: f32,
    /// Deterministic analytic angle seed.
    #[arg(long = "phase-seed", default_value_t = 0)]
    phase_seed: u64,
    /// Number of cached analytic angles, 1..64. Ignored with --phase-continuous.
    #[arg(long = "phase-pool", default_value_t = 64)]
    phase_pool: u32,
    /// Assign continuous angles per MIDI tick/channel/key instead of a finite pool.
    #[arg(long = "phase-continuous")]
    phase_continuous: bool,
    /// Leave this source-sample prefix unchanged, then blend over 10 ms.
    #[arg(long = "phase-preserve-attack-ms", default_value_t = 0.0)]
    phase_preserve_attack_ms: f32,
    /// Maximum analytic quadrature cache in MiB; fails before allocating more.
    #[arg(long = "phase-cache-mib", default_value_t = 2048)]
    phase_cache_mib: u64,
    /// Maximum temporary analytic FFT workspace in MiB.
    #[arg(long = "phase-scratch-mib", default_value_t = 512)]
    phase_scratch_mib: u64,
    /// Ceiling on how much of the voice pool one block may steal, in percent.
    /// 100 lets a saturated block replace the entire pool, which pumps.
    #[arg(long = "steal-percent", default_value_t = 25)]
    steal_percent: u32,
    /// Skip every note-on quieter than this velocity, as if the file did not
    /// contain it: 10 skips velocities 1-9. Skipped notes are dropped on the
    /// host with their note-offs, before anything is built for them, so they
    /// cost next to nothing. 0 keeps every note.
    #[arg(
        long = "min-velocity",
        default_value_t = 0,
        value_parser = clap::value_parser!(u8).range(0..=127)
    )]
    min_velocity: u8,
    /// Hold every note to BASSMIDI's 4 ms envelope grid: a note-off takes
    /// effect at the next 4 ms step counted from its note-on, and a release
    /// shorter than that fades over one step. For MIDIs that rely on BASSMIDI
    /// sounding notes shorter than 4 ms, such as audio encoded as one-sample
    /// notes, which render silent without it. Costs about 12% of the render.
    #[arg(long = "note-grid")]
    note_grid: bool,
    /// Render only this track, numbered from 1 as `kestrel tracks` lists them:
    /// its own notes and controllers, timed by the whole file's tempo whichever
    /// track carries it, with every MIDI port folded onto the first. The file
    /// is scanned before the render starts. --phase-* does not apply.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    track: Option<u32>,
    /// Render these tracks as stems, each alone as --track renders one and
    /// each to a file of its own, in a folder named after the MIDI inside the
    /// directory -o names -- or, with --merge, summed into the one file -o
    /// names. `all` is every track with notes; otherwise track numbers and
    /// ranges, `1-40,57,90-`, a range with no end running to the last track.
    /// --max-voices is a total, split evenly between the tracks, and a track
    /// --min-velocity would empty is skipped.
    #[arg(long, value_name = "LIST", conflicts_with = "track")]
    tracks: Option<String>,
    /// With --track or --tracks: whether the file's setup tracks --
    /// controllers, programs and bends, no notes -- reach each track as they
    /// would in a render of the whole file (apply), or are left out (ignore).
    #[arg(
        long = "setup-tracks",
        default_value = "apply",
        value_parser = ["apply", "ignore"],
        requires = "per_track"
    )]
    setup_tracks: String,
    /// With --tracks: the format every stem is written in. 32-bit float WAV
    /// stems are written without the limiter, so they add back up to the mix;
    /// every other format limits each stem.
    #[arg(
        long = "stem-format",
        default_value = "wav",
        value_parser = ["wav", "flac", "opus", "ogg", "mp3", "m4a"],
        requires = "tracks",
        conflicts_with = "track"
    )]
    stem_format: String,
    /// With --tracks: one file instead of a folder of stems. Every track is
    /// still rendered alone; they are summed exactly, and --volume and the
    /// limiter apply to the sum, once, as in a normal render. -o is that file.
    #[arg(long, requires = "tracks", conflicts_with_all = ["stem_format", "track"])]
    merge: bool,
    /// With --tracks: how many threads do the host's share of the tracks.
    /// Held to this machine's cores less two.
    #[arg(long = "track-jobs", value_name = "N", default_value_t = kestrel::tracks::default_jobs(), requires = "tracks", conflicts_with = "track")]
    track_jobs: usize,
    /// WAV sample format. Encoded containers take float and refuse anything
    /// else.
    #[arg(long, default_value = "float32", value_parser = ["float32", "pcm16"])]
    format: String,
    /// Volume of the finished file as a percentage, from 0 to 200: 100 leaves
    /// it as it is, 50 is half (6 dB down, dense passages included), 0 is
    /// silent. Up to 100 it applies after the limiter. Above 100 the extra goes
    /// in ahead of the limiter, which still holds the file at the ceiling, so
    /// it makes the mix denser rather than louder.
    #[arg(
        long,
        value_name = "PERCENT",
        default_value_t = 100.0,
        value_parser = parse_volume,
        allow_negative_numbers = true
    )]
    volume: f32,
    /// High-pass the mix at 15 Hz ahead of the limiter. A very high voice count
    /// builds up a DC offset and a slow sub-audio swell that nobody hears but
    /// the limiter still spends its headroom on; this takes them out.
    #[arg(long = "dc-blocker")]
    dc_blocker: bool,
    /// Which limiter: brickwall (lookahead true-peak, the default), off, or
    /// omni (the OmniConverter port, deprecated for rendering).
    #[arg(long, default_value = "brickwall", value_parser = ["brickwall", "omni", "off"])]
    limiter: String,
    /// Brickwall ceiling in dBFS. Defaults to 0 (flat full scale) for .wav and
    /// .flac, and to -1 for the lossy containers, which need headroom because
    /// their decoders overshoot what was encoded. An explicit value always wins.
    #[arg(long = "ceiling-db", value_name = "DB", allow_negative_numbers = true)]
    ceiling_db: Option<f64>,
    /// Keep the sample pool at its source rates instead of converting it.
    #[arg(long = "no-resample-pool")]
    no_resample_pool: bool,
    /// Sample pool budget in MiB before automatic downsampling kicks in.
    #[arg(long = "pool-budget", default_value_t = 2048)]
    pool_budget: u64,
    /// Log per-pass timings.
    #[arg(long)]
    profile: bool,
    /// Stop after this many seconds of output.
    #[arg(long)]
    seconds: Option<f64>,
    /// Force a wgpu backend: vulkan, dx12, metal, gl.
    #[arg(long = "gpu-backend", value_parser = parse_gpu_backend)]
    gpu_backend: Option<String>,
    /// Substring match against the adapter name.
    #[arg(long = "gpu-adapter")]
    gpu_adapter: Option<String>,
    /// Check every block for NaN and Inf even in release builds.
    #[arg(long = "nan-guard")]
    nan_guard: bool,

    /// Report the render to another program instead of the terminal. `json`
    /// writes one JSON object per line on stdout -- the phases, the device,
    /// progress snapshots, and the result last -- and reads control messages
    /// from stdin: {"interval_ms": N}, {"snapshot": true}, {"cancel": true}.
    /// Log lines stay on stderr.
    #[arg(long = "progress", value_name = "FORMAT", value_parser = ["json"])]
    progress: Option<String>,
    /// Milliseconds between progress snapshots under --progress, held to
    /// 10..=60000. 0 sends them only when the reading program asks, and the
    /// reading program can change it while the render runs.
    #[arg(
        long = "progress-interval",
        value_name = "MS",
        default_value_t = 250,
        requires = "progress"
    )]
    progress_interval: u64,

    /// The developer options, on the command line only in a dev build. A
    /// release build renders with their defaults; see `dev_args`.
    #[cfg(feature = "dev")]
    #[command(flatten)]
    dev: DevArgs,
}

/// Options for measuring and debugging the engine rather than for rendering
/// with it: tuning the device passes, switching back to old behaviour for an
/// A/B comparison, the pumping diagnostics. On the command line only in a dev
/// build (`--features dev`); the downloadable zips leave them out, decided
/// 2026-09-19, and render with the defaults below, which is what the same
/// command renders in a dev build.
#[derive(Args, Clone)]
#[command(next_help_heading = "Developer options")]
struct DevArgs {
    /// Frames per workgroup reduction round, and the note-off gate resolution.
    #[arg(long = "reduce-tile", default_value_t = 4)]
    reduce_tile: u32,
    /// Frames between note-off gate checks, and the rate at which channel
    /// controllers and per-voice LFOs update. A multiple of --reduce-tile.
    #[arg(long = "gate-frames", default_value_t = 32)]
    gate_frames: u32,
    /// Invocations per render workgroup.
    #[arg(long = "workgroup", default_value_t = 256)]
    workgroup: u32,
    /// Upper bound on render workgroups; sizes the partial buffer.
    #[arg(long = "render-workgroups", default_value_t = 2048)]
    render_workgroups: u32,
    /// Most voices one note-on may spawn. Caps runaway presets; a stereo
    /// sample is two.
    #[arg(long, default_value_t = 16)]
    layers: u32,
    /// Shape of the envelope's decay stage: exponential or linear.
    #[arg(long = "decay-curve", default_value = "exponential")]
    decay_curve: String,
    /// Shape of the envelope's release stage: exponential or linear.
    #[arg(long = "release-curve", default_value = "exponential")]
    release_curve: String,
    /// Which voice goes when the pool is full: quietest, oldest, or drop-new,
    /// which refuses the new note instead.
    #[arg(long, default_value = "quietest")]
    steal: String,
    /// Which note-ons survive when one block has more of them than the pool
    /// has room for: loudest ranks by whether the note outlives the block and
    /// then by opening amplitude; even thins by position, ignoring both.
    #[arg(long = "admit", default_value = "loudest")]
    admit: String,
    /// Turn the soft limiter off. The same as --limiter off.
    #[arg(long = "no-limiter")]
    no_limiter: bool,
    /// Brickwall lookahead in ms. Also the render latency.
    #[arg(long = "lookahead-ms", default_value_t = 2.0)]
    lookahead_ms: f64,
    /// The --dc-blocker corner in Hz, a second-order Butterworth high-pass.
    #[arg(long = "dc-blocker-hz", default_value_t = 15.0)]
    dc_blocker_hz: f64,
    /// Brickwall release in ms.
    #[arg(long = "limiter-release-ms", default_value_t = 60.0)]
    limiter_release_ms: f64,
    /// Time constant of the brickwall's sustained stage, in ms. 0, the default,
    /// disables it: it measured worse on saturated black MIDI.
    #[arg(long = "limiter-sustain-ms", default_value_t = 0.0)]
    limiter_sustain_ms: f64,
    /// Detect only sample peaks, not inter-sample ones. Cheaper, lets
    /// inter-sample overshoot through.
    #[arg(long = "no-true-peak")]
    no_true_peak: bool,
    /// Turn the per-voice low-pass filter off.
    #[arg(long = "no-filter")]
    no_filter: bool,
    /// Step channel volume and expression at gate-tile boundaries instead of
    /// ramping across them. Restores pre-0.2.5 behaviour exactly, including
    /// the click an abrupt CC7/CC11 move puts in every sounding voice.
    #[arg(long = "no-gain-ramp")]
    no_gain_ramp: bool,
    /// Switch biquad coefficients instantly at gate-tile boundaries instead of
    /// ramping across them. Restores pre-0.2.5 behaviour exactly, including the
    /// filter ringing an abrupt CC71/CC74 move causes.
    #[arg(long = "no-filter-ramp")]
    no_filter_ramp: bool,
    /// Do not evaluate SF2 vibrato and tremolo LFOs. Restores pre-0.2.5
    /// behaviour, in which they were read from the file and discarded.
    #[arg(long = "no-lfo")]
    no_lfo: bool,
    /// Do not evaluate the SF2 modulation envelope. Restores pre-0.2.5
    /// behaviour, in which its generators were read and discarded.
    #[arg(long = "no-mod-env")]
    no_mod_env: bool,
    /// Do not re-sort the voice pool during compaction. Much slower at high
    /// voice counts; only useful for measuring what the sort buys.
    #[arg(long = "no-sort")]
    no_sort: bool,
    /// Send every block to the device, even one with no voice alive and none
    /// admitted. Such a block renders exact zeros either way; this is for
    /// checking that it does.
    #[arg(long = "no-skip-silence")]
    no_skip_silence: bool,
    /// Write one CSV row per block: the admission decision and the level it
    /// produced. This is the diagnostic for block-rate pumping -- the audio is
    /// downstream of these numbers, so read them rather than the waveform.
    #[arg(long = "block-csv", value_name = "PATH")]
    block_csv: Option<PathBuf>,
    /// Compile shaders without automatic bounds clamps. Faster, and unsafe if
    /// anything upstream miscounts.
    #[arg(long = "unchecked-shaders")]
    unchecked_shaders: bool,
}

impl DevArgs {
    /// Every developer option at its default, read by parsing an empty command
    /// line against the definition above rather than written out a second
    /// time, so a release build cannot drift from what a dev build renders.
    #[cfg_attr(all(feature = "dev", not(test)), allow(dead_code))]
    fn defaults() -> Self {
        use clap::FromArgMatches;
        let cmd = DevArgs::augment_args(clap::Command::new("dev"));
        DevArgs::from_arg_matches(&cmd.get_matches_from(["dev"]))
            .expect("every developer option has a default")
    }
}

/// `--volume`: a percentage from 0 to 200, with or without a trailing `%`.
///
/// A negative number is refused with what it would be as decibels, because a
/// decibel value is the likeliest thing someone typing one meant.
fn parse_volume(s: &str) -> std::result::Result<f32, String> {
    let v: f32 = s
        .trim()
        .trim_end_matches('%')
        .trim_end()
        .parse()
        .map_err(|_| format!("{s:?} is not a percentage; 100 leaves the mix as it is"))?;
    if v.is_nan() || v < 0.0 {
        let as_db = if v.is_finite() {
            let pct = 100.0 * 10f32.powf(v / 20.0);
            let pct = if pct >= 1.0 { format!("{pct:.1}") } else { format!("{pct:.3}") };
            format!("; {s} dB would be --volume {pct}")
        } else {
            String::new()
        };
        return Err(format!("volume is a percentage from 0 to 200, not {s}{as_db}"));
    }
    if v > 200.0 {
        return Err(format!("volume is at most 200, twice as loud, not {s}"));
    }
    Ok(v)
}

/// `--gpu-backend`: refused here if it is not a name the device layer knows,
/// with the names it could be, instead of after the soundfont has loaded.
fn parse_gpu_backend(s: &str) -> std::result::Result<String, String> {
    if gpu::backend_known(s) {
        Ok(s.to_string())
    } else {
        Err(format!("{s:?} is not a GPU backend; use vulkan, dx12, metal or gl"))
    }
}

impl RenderArgs {
    /// The developer options: as typed in a dev build, their defaults in a
    /// release build, which does not take them.
    fn dev_args(&self) -> DevArgs {
        #[cfg(feature = "dev")]
        return self.dev.clone();
        #[cfg(not(feature = "dev"))]
        return DevArgs::defaults();
    }

    fn to_config(&self) -> Result<(Config, BackendKind)> {
        let dev = self.dev_args();
        let mut cfg = Config {
            phase: kestrel::phase::PhaseSettings {
                mode: if self.phase_mode == "analytic" { kestrel::phase::PhaseMode::Analytic }
                    else { kestrel::phase::PhaseMode::Baseline },
                strength: self.phase_strength,
                seed: self.phase_seed,
                pool_size: self.phase_pool,
                continuous: self.phase_continuous,
                preserve_attack_ms: self.phase_preserve_attack_ms,
                cache_budget_bytes: self.phase_cache_mib.checked_mul(1 << 20)
                    .context("phase-cache-mib is too large")?,
                scratch_budget_bytes: self.phase_scratch_mib.checked_mul(1 << 20)
                    .context("phase-scratch-mib is too large")?,
            },
            sample_rate: self.rate,
            block_frames: self.block,
            reduce_tile: dev.reduce_tile,
            gate_frames: dev.gate_frames,
            workgroup_size: dev.workgroup,
            max_render_workgroups: dev.render_workgroups,
            max_voices: self.max_voices,
            max_layers: dev.layers,
            max_steal_percent: self.steal_percent,
            min_velocity: self.min_velocity,
            note_grid: self.note_grid,
            master_volume: self.volume / 100.0,
            dc_blocker: self.dc_blocker,
            dc_blocker_hz: dev.dc_blocker_hz,
            limiter: !dev.no_limiter,
            // `None` here means "not chosen yet". `render` fills it in from
            // the output container, which `to_config` cannot see.
            limiter_ceiling_db: self.ceiling_db.unwrap_or(0.0),
            limiter_lookahead_ms: dev.lookahead_ms,
            limiter_release_ms: dev.limiter_release_ms,
            limiter_sustain_ms: dev.limiter_sustain_ms,
            limiter_true_peak: !dev.no_true_peak,
            filter_enabled: !dev.no_filter,
            gain_ramp: !dev.no_gain_ramp,
            filter_ramp: !dev.no_filter_ramp,
            lfo_enabled: !dev.no_lfo,
            mod_env_enabled: !dev.no_mod_env,
            sort_voices: !dev.no_sort,
            skip_silence: !dev.no_skip_silence,
            resample_pool: !self.no_resample_pool,
            sample_pool_budget: self.pool_budget << 20,
            profile: self.profile,
            gpu_backend: self.gpu_backend.clone(),
            gpu_adapter: self.gpu_adapter.clone(),
            ..Default::default()
        };
        // Until 0.3.0 this was a linear gain, so a script written then passes
        // something like 0.5 and would now render 46 dB down without a word.
        if self.volume > 0.0 && self.volume <= 2.0 {
            log::warn!(
                "--volume {v} is {v}% of the mix ({:.1} dB). It is a percentage now; a value \
                 from the old linear scale needs multiplying by 100",
                20.0 * (self.volume / 100.0).log10(),
                v = self.volume
            );
        }
        if self.nan_guard {
            cfg.nan_guard = true;
        }
        cfg.unchecked_shaders = dev.unchecked_shaders;
        cfg.interpolation = Interpolation::parse(&self.interp)
            .with_context(|| format!("unknown interpolation {:?}", self.interp))?;
        cfg.decay_curve = EnvelopeCurve::parse(&dev.decay_curve)
            .with_context(|| format!("unknown decay curve {:?}", dev.decay_curve))?;
        cfg.release_curve = EnvelopeCurve::parse(&dev.release_curve)
            .with_context(|| format!("unknown release curve {:?}", dev.release_curve))?;
        cfg.limiter_mode = LimiterMode::parse(&self.limiter)
            .with_context(|| format!("unknown limiter {:?}", self.limiter))?;
        if dev.no_limiter {
            cfg.limiter_mode = LimiterMode::Off;
        }
        // The raw mix is only observable when nothing bounds it and the
        // container can hold it. `--limiter off --format float32` is that
        // pair, and it is the pair people reach for to measure what the synth
        // actually produced -- so the final clamp comes off for it and stays
        // on everywhere else. pcm16 has no representation for |x| > 1, so it
        // keeps the clamp whatever the limiter is doing.
        // `--no-limiter` and `--limiter off` are two spellings of the same
        // thing and both have to count, or the flag people actually type is
        // the one that keeps clipping.
        let unlimited = !cfg.limiter || cfg.limiter_mode == LimiterMode::Off;
        cfg.clamp_output = !(unlimited && self.format == "float32");

        cfg.steal_rule = StealRule::parse(&dev.steal)
            .with_context(|| format!("unknown steal rule {:?}", dev.steal))?;
        cfg.admit_rule = AdmitRule::parse(&dev.admit)
            .with_context(|| format!("unknown admit rule {:?}", dev.admit))?;
        cfg.validate()?;
        let kind = if self.backend == "cpu" {
            BackendKind::Cpu
        } else {
            BackendKind::Gpu
        };
        Ok((cfg, kind))
    }
}

/// Which front end a command line gets.
#[derive(Debug, PartialEq, Eq)]
enum Route {
    /// No arguments: the guided renderer.
    Guided,
    /// The Extras window the guided renderer opens.
    Extras,
    /// Arguments without `--force-cli`: a pointer to the guided renderer.
    Notice,
    /// `--force-cli`, or a request for help or the version.
    Cli,
}

fn route(args: &[OsString]) -> Route {
    let is = |a: &OsString, s: &str| a.to_str() == Some(s);
    if args.is_empty() {
        return Route::Guided;
    }
    if args.len() == 1 && is(&args[0], tui::extras::FLAG) {
        return Route::Extras;
    }
    if args.iter().any(|a| is(a, "--force-cli")) {
        return Route::Cli;
    }
    // Help and the version are how anyone finds `--force-cli` in the first
    // place, so they are never refused.
    let asks = |a: &OsString| matches!(a.to_str(), Some("-h" | "--help" | "-V" | "--version"));
    if args.iter().all(asks) || is(&args[0], "help") {
        return Route::Cli;
    }
    Route::Notice
}

fn main() -> Result<()> {
    let raw: Vec<OsString> = std::env::args_os().skip(1).collect();
    match route(&raw) {
        Route::Guided => return tui::run(),
        Route::Extras => return tui::extras::run(),
        Route::Notice => {
            tui::notice(&raw);
            std::process::exit(2);
        }
        Route::Cli => {}
    }

    let cli = Cli::parse();
    // The API writes its log records as JSON lines of its own, so stdout
    // carries nothing that is not a protocol line.
    if matches!(cli.cmd, Cmd::Api) {
        return api::run();
    }
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(LOG_FILTER))
        .format_timestamp(None)
        .init();

    match cli.cmd {
        Cmd::Api => unreachable!("handled above, before the logger"),
        Cmd::Render(args) => render::render_cli(args),
        Cmd::Info {
            path,
            block,
            rate,
            sf_layers,
            sf_release,
            soundfont,
        } => info(path, block, rate, sf_layers, sf_release, soundfont),
        Cmd::Tracks {
            midi,
            jobs,
            rate,
            min_velocity,
        } => tracks(midi, jobs, rate, min_velocity),
        #[cfg(feature = "dev")]
        Cmd::Null { a, b, threshold } => null(a, b, threshold),
        Cmd::GpuInfo => gpu::print_adapters(),
        Cmd::FfmpegInfo { ffmpeg } => ffmpeg_info(ffmpeg),
        Cmd::CheckUpdate => update::run_cli(),
        Cmd::GetFfmpeg {
            dry_run,
            yes,
            accept_hash,
            dir,
        } => get_ffmpeg(dry_run, yes, accept_hash, dir),
        #[cfg(feature = "dev")]
        Cmd::GenAssets {
            dir,
            big_mb,
            sustained,
        } => gen_assets(dir, big_mb, sustained),
    }
}

/// Fetch an ffmpeg, verify it, and install it beside this executable.
///
/// The order is the point: **the digest is checked before anything is
/// extracted, and extraction happens before anything is executed.** A download
/// that fails verification is deleted without ever being unpacked.
fn get_ffmpeg(
    dry_run: bool,
    yes: bool,
    accept_hash: Option<String>,
    dir: Option<PathBuf>,
) -> Result<()> {
    use kestrel::ffmpeg;

    let release = ffmpeg::release_for_host().with_context(|| {
        format!(
            "no ffmpeg build is listed for this platform ({} {}).\n\
             Install ffmpeg yourself and it will be found on PATH.",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let dest = match dir {
        Some(d) => d,
        None => ffmpeg::install_dir()?,
    };

    // Everything the operator needs in order to say no, before anything is
    // fetched. Licence included: this is a GPL binary being placed next to an
    // MPL program, which is a thing to be told rather than to discover.
    println!("platform  {}", release.platform);
    println!("url       {}", release.url);
    println!("origin    {}", release.origin);
    println!("licence   {}", release.license);
    println!("size      {}", release.size_hint);
    println!("install   {}", dest.join(ffmpeg::exe_name()).display());
    println!(
        "verify    {}",
        match (&accept_hash, release.sha256) {
            (Some(h), _) => format!("SHA-256 must equal {h}"),
            (None, Some(h)) => format!("SHA-256 must equal the pinned {h}"),
            (None, None) => "NO DIGEST GIVEN -- the download will be refused".to_string(),
        }
    );

    if dry_run {
        println!("\ndry run: nothing downloaded");
        return Ok(());
    }

    if !yes {
        use std::io::Write;
        print!("\nDownload and install this? [y/N] ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        // EOF means no controlling terminal, which is exactly when an
        // unattended fetch would be worst. Refusing is the safe reading.
        if std::io::stdin().read_line(&mut line).is_err() || !line.trim().eq_ignore_ascii_case("y")
        {
            bail!("cancelled");
        }
    }

    std::fs::create_dir_all(&dest).with_context(|| format!("creating {}", dest.display()))?;
    // Into the destination, not the system temp directory: a 40 MB archive
    // should fail on the volume it is going to live on, not after crossing one.
    let archive = dest.join("ffmpeg-download.part");

    println!("\ndownloading...");
    ffmpeg::download(release.url, &archive)?;
    let size = std::fs::metadata(&archive).map(|m| m.len()).unwrap_or(0);
    println!("got {:.1} MiB", size as f64 / 1048576.0);

    print!("verifying... ");
    use std::io::Write as _;
    std::io::stdout().flush().ok();
    let got = ffmpeg::sha256::hex_of_file(&archive)?;
    println!("{got}");

    if let Err(e) = ffmpeg::verify(release, &got, accept_hash.as_deref()) {
        // An unverified archive is not left lying around to be picked up by a
        // later run or a curious operator.
        let _ = std::fs::remove_file(&archive);
        return Err(e);
    }

    println!("extracting {}...", ffmpeg::exe_name());
    let installed = ffmpeg::extract_binary(&archive, &dest)?;
    let _ = std::fs::remove_file(&archive);

    // Only now is it run, and only to confirm it is what it claims to be and
    // can encode what the presets need.
    let f = ffmpeg::probe(&installed, ffmpeg::Source::BesideExe)?;
    println!("\ninstalled {}", installed.display());
    println!("version   {}", f.version);

    let missing = f.missing_encoders()?;
    if missing.is_empty() {
        println!("every container Kestrel encodes is available");
    } else {
        println!(
            "warning: this build cannot write {}",
            missing
                .iter()
                .map(|m| format!(".{}", m.ext))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("\nKestrel will now find this automatically; no flags needed.");
    Ok(())
}

/// Report the ffmpeg that encoded output would use.
///
/// Exits non-zero when there is none, or when the build is missing an encoder
/// a 0.3.0 preset needs, so it is usable as a setup check in a script rather
/// than only as something to read.
fn ffmpeg_info(explicit: Option<PathBuf>) -> Result<()> {
    let f = kestrel::ffmpeg::find(explicit.as_deref())?;
    println!("ffmpeg  {}", f.path.display());
    println!("found   via {}", f.source.describe());
    println!("version {}", f.version);

    // `FFMPEG` losing to `--ffmpeg` is correct, but silently is not: someone
    // debugging why their override "did nothing" should be told the other one
    // exists and is being ignored.
    if f.source == kestrel::ffmpeg::Source::Flag {
        if let Some(v) = std::env::var_os("FFMPEG") {
            if !v.is_empty() {
                println!(
                    "note    FFMPEG is also set ({}); --ffmpeg wins",
                    PathBuf::from(v).display()
                );
            }
        }
    }

    let missing = f.missing_encoders()?;
    println!();
    for p in kestrel::ffmpeg::PRESETS {
        let ok = !missing.iter().any(|m| m.ext == p.ext);
        println!(
            "  .{:<5} {:<12} {:<8} {}",
            p.ext,
            p.encoder,
            if ok { "present" } else { "MISSING" },
            p.note
        );
    }

    if !missing.is_empty() {
        println!();
        bail!(
            "this ffmpeg cannot write {}; it was built without {}",
            missing
                .iter()
                .map(|m| format!(".{}", m.ext))
                .collect::<Vec<_>>()
                .join(", "),
            missing
                .iter()
                .map(|m| m.encoder)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("\nevery container Kestrel encodes is available");
    Ok(())
}

/// `kestrel tracks`: the per-track scan, as a table.
fn tracks(midi: PathBuf, jobs: usize, rate: u32, min_velocity: Option<u8>) -> Result<()> {
    use kestrel::tracks::{self, TrackKind};
    use tui::style::{audio_clock, thousands};

    let t0 = std::time::Instant::now();
    let s = tracks::scan(&midi, jobs, None)?;
    let took = t0.elapsed().as_secs_f64();
    let threads = tracks::jobs_for(jobs, s.tracks.len());

    // Every tick the table shows, converted in one pass over the tempo map.
    let mut ticks = vec![s.end_tick()];
    for t in &s.tracks {
        ticks.push(t.first_note.unwrap_or(0));
        ticks.push(t.last_note.unwrap_or(0));
    }
    let secs: Vec<f64> = s.frames_at(&ticks, rate).iter().map(|f| f / rate as f64).collect();

    let name = midi
        .file_name()
        .map_or_else(|| midi.display().to_string(), |n| n.to_string_lossy().into_owned());
    let bytes: u64 = s.tracks.iter().map(|t| t.bytes).sum();
    println!("{name}");
    println!(
        "  format {}, {}, {} tracks, {} of track data",
        s.format,
        tui::checks::describe_division(s.division),
        s.tracks.len(),
        fmt_bytes(bytes)
    );
    let total = s.notes();
    println!(
        "  {} notes, {} tempo change{}, {} to the last event at {rate} Hz",
        thousands(total),
        thousands(s.tempo.len() as u64),
        if s.tempo.len() == 1 { "" } else { "s" },
        audio_clock(secs[0])
    );
    println!("  read in {took:.2} s on {threads} thread{}", if threads == 1 { "" } else { "s" });
    println!();

    let kept_head = min_velocity
        .map(|v| format!("  {:>13}", format!("vel >= {v}")))
        .unwrap_or_default();
    println!(
        "  {:>5}  {:<5}  {:>13}{kept_head}  {:<7}  {:<14}  {:>9}  {:>9}  {:>9}  name",
        "track", "kind", "notes", "vel", "channels", "controls", "first", "last"
    );
    let dash = |show: bool, v: String| if show { v } else { "-".to_string() };
    for (i, t) in s.tracks.iter().enumerate() {
        let kind = match t.kind() {
            TrackKind::Notes => "notes",
            TrackKind::Setup => "setup",
            TrackKind::Empty => "empty",
        };
        let has_notes = t.first_note.is_some();
        let kept = min_velocity
            .map(|v| format!("  {:>13}", dash(has_notes, thousands(t.notes_from(v)))))
            .unwrap_or_default();
        let vel = t.velocity_range().map_or("-".to_string(), |(lo, hi)| {
            if lo == hi {
                lo.to_string()
            } else {
                format!("{lo}-{hi}")
            }
        });
        println!(
            "  {:>5}  {:<5}  {:>13}{kept}  {:<7}  {:<14}  {:>9}  {:>9}  {:>9}  {}",
            i + 1,
            kind,
            dash(has_notes, thousands(t.notes())),
            vel,
            channel_list(t.channels),
            dash(t.controls > 0, thousands(t.controls)),
            dash(has_notes, audio_clock(secs[1 + 2 * i])),
            dash(has_notes, audio_clock(secs[2 + 2 * i])),
            t.display_name().unwrap_or_default()
        );
    }
    println!();

    let count = |kind| s.of_kind(kind).count();
    let plural = |n: usize, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
    println!(
        "  {} with notes, {}, {}",
        plural(count(TrackKind::Notes), "track"),
        plural(count(TrackKind::Setup), "setup track"),
        plural(count(TrackKind::Empty), "empty track")
    );
    if let Some(b) = s.busiest() {
        let n = s.tracks[b].notes();
        println!(
            "  busiest: track {}, {} notes ({:.1}%)",
            b + 1,
            thousands(n),
            n as f64 * 100.0 / total as f64
        );
    }
    let tempo_tracks: Vec<String> = s
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.tempos > 0)
        .map(|(i, _)| (i + 1).to_string())
        .collect();
    match tempo_tracks.len() {
        0 => println!("  no tempo changes: 120 bpm throughout"),
        1..=8 => println!("  tempo changes on track {}", tempo_tracks.join(", ")),
        n => println!("  tempo changes on {n} tracks"),
    }
    if let Some(v) = min_velocity {
        let kept: u64 = s.tracks.iter().map(|t| t.notes_from(v)).sum();
        let with = count(TrackKind::Notes);
        let left = s.tracks.iter().filter(|t| t.notes_from(v) > 0).count();
        println!(
            "  --min-velocity {v} keeps {} notes ({:.1}%) on {left} of {with} tracks; {} would render nothing",
            thousands(kept),
            if total > 0 { kept as f64 * 100.0 / total as f64 } else { 0.0 },
            plural(with - left, "track")
        );
    }
    Ok(())
}

/// Channels as ranges, 1-based: `1-16`, `1,3,10`.
fn channel_list(mask: u16) -> String {
    let mut out = Vec::new();
    let mut c = 0;
    while c < 16 {
        if mask & (1 << c) == 0 {
            c += 1;
            continue;
        }
        let start = c;
        while c < 16 && mask & (1 << c) != 0 {
            c += 1;
        }
        out.push(if c - start == 1 {
            format!("{}", start + 1)
        } else {
            format!("{}-{}", start + 1, c)
        });
    }
    if out.is_empty() {
        "-".into()
    } else {
        out.join(",")
    }
}

fn fmt_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}

/// Read the two soundfont-shaped figures `info` needs off the soundfont.
///
/// They were flags a caller had to supply, and there was no way to find the
/// right values short of rendering and dividing: layers is 2 for a stereo
/// library because a stereo sample becomes two hard-panned mono regions, 1 for
/// a compact mono bank, more for a layered one. Guessing low is the dangerous
/// direction -- it reports a file needing half the pool it does.
///
/// Layers is the median over a spread of keys and velocities rather than one
/// probe, because a library can be sparse at the extremes: a bottom octave
/// with no samples would read as 0 and a single velocity split as 1.
///
/// Release is a high percentile rather than the maximum. The estimate is about
/// how long the pool stays occupied, and one long region -- a pedal noise, a
/// release sample -- should not stand for the whole instrument.
fn derive_from_soundfont(sf: &std::path::Path, cfg: &Config) -> Result<(u32, f64, String)> {
    let bank = kestrel::load_bank(sf, cfg)?;
    let mut counts: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    // The seed advances across probes so a library using `lorand`/`hirand`
    // is sampled across its random slices rather than being asked the same
    // question repeatedly. The median below then reflects what a real render
    // sees; a fixed seed would report one slice's layer count as the whole
    // instrument's.
    let mut seed = 0u64;
    for key in (21u8..=108).step_by(3) {
        for vel in [8u8, 32, 64, 96, 127] {
            out.clear();
            bank.note_on(0, key, vel, seed, cfg, 255, &mut out);
            seed += 1;
            if !out.is_empty() {
                counts.push(out.len());
            }
        }
    }
    counts.sort_unstable();
    let layers = counts.get(counts.len() / 2).copied().unwrap_or(1).max(1) as u32;

    let mut rel: Vec<f32> = bank.regions.iter().map(|r| r.release).collect();
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let release = if rel.is_empty() {
        1.0
    } else {
        rel[(rel.len() * 9 / 10).min(rel.len() - 1)] as f64
    };
    let name = sf
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| sf.display().to_string());
    Ok((layers, release, name))
}

fn info(
    path: PathBuf,
    block: u32,
    rate: u32,
    layers: u32,
    release: f64,
    soundfont: Option<PathBuf>,
) -> Result<()> {
    let cfg = Config::default();
    let (layers, release, derived) = match &soundfont {
        Some(sf) => {
            let (l, r, what) = derive_from_soundfont(sf, &cfg)?;
            (l, r, Some(what))
        }
        None => (layers, release, None),
    };
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    if ext == "mid" || ext == "midi" || ext == "rmi" {
        let mut s = kestrel::midi::MidiStream::open(&path)?;
        println!("format {} division {:?}", s.format, s.division);
        println!("{} tracks", s.track_count);
        let mut counts = [0u64; 8];
        let mut cc_counts = [0u64; 128];
        let mut last_tick = 0u64;

        // Per-block density. This is the number that decides host memory: a
        // block is admitted as a whole, so every note-on inside it is held
        // until the block ends, whether or not the pool can take it.
        let mut clock = kestrel::midi::TempoClock::new(s.division, rate);
        let mut cur_block = 0u64;
        let mut in_block = 0u64;
        // One note-on count per block, kept so a sliding window over it can
        // estimate concurrency afterwards.
        let mut per_block: Vec<u64> = Vec::new();

        while let Some((tick, ev)) = s.next() {
            last_tick = tick;
            if let kestrel::midi::Event::Tempo(us) = ev {
                clock.set_tempo(tick, us);
            }
            let b = clock.frame_at(tick) as u64 / block as u64;
            if b != cur_block {
                while per_block.len() as u64 <= cur_block {
                    per_block.push(0);
                }
                per_block[cur_block as usize] = in_block;
                cur_block = b;
                in_block = 0;
            }
            if matches!(ev, kestrel::midi::Event::NoteOn { .. }) {
                in_block += 1;
            }

            let i = match ev {
                kestrel::midi::Event::NoteOn { .. } => 0,
                kestrel::midi::Event::NoteOff { .. } => 1,
                kestrel::midi::Event::Cc { num, .. } => {
                    cc_counts[num as usize & 127] += 1;
                    2
                }
                kestrel::midi::Event::Program { .. } => 3,
                kestrel::midi::Event::PitchBend { .. } => 4,
                kestrel::midi::Event::Tempo(_) => 5,
                kestrel::midi::Event::DrumPart { .. } | kestrel::midi::Event::ResetParts => 7,
                kestrel::midi::Event::Other => 6,
            };
            counts[i] += 1;
        }
        println!(
            "note-on {} note-off {} cc {} program {} bend {} tempo {} sysex {} other {}",
            counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[7],
            counts[6]
        );
        if counts[2] > 0 {
            use kestrel::driver::{cc_role, CcRole};
            println!("controllers used:");
            let mut missing = 0u64;
            for (num, n) in cc_counts.iter().enumerate() {
                if *n == 0 {
                    continue;
                }
                let (name, role) = cc_role(num as u8);
                let note = match role {
                    CcRole::Applied => "applied".to_string(),
                    CcRole::Inert(why) => format!("inert: {why}"),
                    CcRole::Missing(why) => {
                        missing += *n;
                        format!("MISSING: {why}")
                    }
                };
                println!("  cc{:<4}{:<20}{:>10}  {}", num, name, n, note);
            }
            println!(
                "  {} of {} controller events ({:.1}%) are unimplemented",
                missing,
                counts[2],
                missing as f64 * 100.0 / counts[2] as f64
            );
        }
        while per_block.len() as u64 <= cur_block {
            per_block.push(0);
        }
        per_block[cur_block as usize] = in_block;
        println!("last tick {last_tick}");

        let (peak_block, peak_at) = per_block
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i as u64))
            .max()
            .unwrap_or((0, 0));

        // Concurrency, as note-ons inside one release window.
        //
        // Counting notes between their note-on and note-off measures nothing
        // here: black MIDI notes are a tick long, so that peaks in the low
        // hundreds on a file that genuinely needs millions of voices. What
        // keeps a voice alive is its release tail, and since the notes are far
        // shorter than the release, "note-ons in the last `release` seconds" is
        // the estimate that tracks the real pool.
        let win = ((release * rate as f64) / block as f64).ceil().max(1.0) as usize;
        let mut sum: u64 = per_block.iter().take(win).sum();
        let mut peak_win = sum;
        let mut peak_win_at = 0usize;
        for i in win..per_block.len() {
            sum += per_block[i];
            sum -= per_block[i - win];
            if sum > peak_win {
                peak_win = sum;
                peak_win_at = i + 1 - win;
            }
        }

        // 24 B per candidate layer plus 12 B per note-on is what the driver
        // holds for a block while it waits to admit it. See the development notes. The
        // process peak runs above this -- the spawn list, the backend's copy
        // and the per-track read buffers are all on top, and on the reference
        // file that came to about 1.35x -- so treat it as a floor.
        let per_note = 24 * layers as u64 + 12;
        let bytes = peak_block * per_note;
        println!();
        println!(
            "at --block {block}, {rate} Hz, --sf-layers {layers}, --sf-release {release:.2}s"
        );
        match &derived {
            Some(name) => println!("  the last two read from {name}"),
            None => println!("  the last two assumed; pass -s <soundfont> to read them off it"),
        }
        println!(
            "  busiest block  {:>14} note-ons, at {:.2}s",
            peak_block,
            peak_at as f64 * block as f64 / rate as f64
        );
        println!("  host memory    {:>14} for that block, at least", fmt_bytes(bytes));
        println!(
            "  peak {:.2}s span {:>14} voices, at {:.2}s -- roughly what a pool",
            release,
            peak_win * layers as u64,
            peak_win_at as f64 * block as f64 / rate as f64
        );
        println!("                                must hold for nothing to be stolen");

        // Two GiB for one block is where this stops being incidental.
        const COMFORTABLE: u64 = 2 << 30;
        if bytes > COMFORTABLE {
            let mut narrower = block;
            while narrower > 128 && peak_block * per_note / (block / narrower) as u64 > COMFORTABLE
            {
                narrower /= 2;
            }
            println!();
            println!("  That is a lot of host RAM for one block. Note-ons per block scale");
            println!("  with --block, so --block {narrower} would need about {},", 
                fmt_bytes(bytes / (block / narrower) as u64));
            println!("  and 128 is the floor. Lowering --max-voices does not help: the block");
            println!("  is held in full before admission thins it.");
        }
        return Ok(());
    }

    let bank = load_bank(&path, &cfg)?;
    println!("{}", bank.describe());
    for p in &bank.presets {
        println!(
            "  bank {:>3} program {:>3}  {:<24} {} regions",
            p.bank,
            p.program,
            p.name,
            p.regions.len()
        );
    }
    Ok(())
}

#[cfg(feature = "dev")]
fn null(a: PathBuf, b: PathBuf, threshold: f64) -> Result<()> {
    let wa = wav::read(&a)?;
    let wb = wav::read(&b)?;
    if wa.channels != wb.channels {
        bail!("channel count differs: {} vs {}", wa.channels, wb.channels);
    }
    let n = wa.interleaved.len().min(wb.interleaved.len());
    if wa.interleaved.len() != wb.interleaved.len() {
        log::warn!(
            "lengths differ: {} vs {} samples, comparing the first {}",
            wa.interleaved.len(),
            wb.interleaved.len(),
            n
        );
    }

    let mut peak_diff = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut peak_ref = 0.0f64;
    let mut worst = 0usize;
    for i in 0..n {
        let d = (wa.interleaved[i] - wb.interleaved[i]).abs() as f64;
        if d > peak_diff {
            peak_diff = d;
            worst = i;
        }
        sum_sq += d * d;
        peak_ref = peak_ref.max(wa.interleaved[i].abs() as f64);
    }
    let rms = (sum_sq / n.max(1) as f64).sqrt();
    let db = |v: f64| if v <= 0.0 { -f64::INFINITY } else { 20.0 * v.log10() };

    let identical = wa.interleaved[..n] == wb.interleaved[..n];
    println!("samples compared : {n}");
    println!("bit-identical    : {identical}");
    println!("reference peak   : {:.6} ({:.2} dBFS)", peak_ref, db(peak_ref));
    println!("peak difference  : {:.9} ({:.2} dB)", peak_diff, db(peak_diff));
    println!("rms difference   : {:.9} ({:.2} dB)", rms, db(rms));
    println!("worst sample     : {worst}");

    if db(peak_diff) > threshold {
        bail!(
            "null test failed: peak difference {:.2} dB is above the {:.2} dB threshold",
            db(peak_diff),
            threshold
        );
    }
    println!("PASS (below {threshold:.1} dB)");
    Ok(())
}

#[cfg(feature = "dev")]
fn gen_assets(dir: PathBuf, big_mb: Option<usize>, sustained: Option<usize>) -> Result<()> {
    std::fs::create_dir_all(&dir)?;
    testkit::simple_sf2(dir.join("simple.sf2"), 48000)?;
    testkit::rich_sf2(dir.join("rich.sf2"), 48000)?;
    testkit::single_note_midi(dir.join("single.mid"), 69, 127, 1.0)?;
    testkit::scatter_midi(dir.join("scatter1000.mid"), 1000, 10.0, 4, 36, 84)?;
    testkit::scatter_midi(dir.join("scatter100k.mid"), 100_000, 30.0, 16, 21, 108)?;
    testkit::simultaneous_midi(dir.join("stress2m.mid"), 2_000_000, 4.0)?;
    if let Some(mb) = big_mb {
        let p = dir.join(format!("big{mb}.sf2"));
        testkit::big_sf2(&p, 48000, mb)?;
        println!("wrote {} ({} MiB sample pool)", p.display(), mb);
    }
    if let Some(n) = sustained {
        let p = dir.join(format!("sustained{n}.mid"));
        testkit::sustained_midi(&p, n, 10.0)?;
        println!("wrote {} ({n} sustained notes)", p.display());
    }
    println!("wrote test assets to {}", dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{route, Cli, DevArgs, Route};
    use clap::Parser;
    use std::ffi::OsString;

    /// Only a dev build takes the developer options and commands; a release
    /// build, the downloadable zips, refuses them as unknown.
    #[test]
    fn developer_options_exist_only_in_a_dev_build() {
        let dev = cfg!(feature = "dev");
        for flag in [
            &["--no-sort"][..],
            &["--block-csv", "b.csv"],
            &["--gate-frames", "64"],
            &["--steal", "oldest"],
            &["--layers", "4"],
            &["--no-limiter"],
        ] {
            assert_eq!(render_args(flag).is_ok(), dev, "{flag:?}");
        }
        for kept in [&["--steal-percent", "50"][..], &["--note-grid"], &["--limiter", "off"]] {
            assert!(render_args(kept).is_ok(), "{kept:?}");
        }
        for cmd in [&["null", "a.wav", "b.wav"][..], &["gen-assets", "dir"]] {
            let argv = [&["kestrel", "--force-cli"][..], cmd].concat();
            assert_eq!(Cli::try_parse_from(argv).is_ok(), dev, "{cmd:?}");
        }
    }

    /// A wrong `--gpu-backend` is refused as it is parsed, naming the ones it
    /// could be, not after the soundfont has loaded.
    #[test]
    fn an_unknown_gpu_backend_is_refused_with_the_names() {
        let Err(e) = render_args(&["--gpu-backend", "5"]) else {
            panic!("--gpu-backend 5 parsed");
        };
        let e = e.to_string();
        assert!(e.contains("vulkan, dx12, metal or gl"), "{e}");
        assert!(render_args(&["--gpu-backend", "DX12"]).is_ok());
        assert!(render_args(&["--gpu-backend=vulkan"]).is_ok());
    }

    /// A release build renders with what a dev build's command line defaults
    /// to, so the same command writes the same file in either.
    #[test]
    fn a_release_build_renders_with_the_dev_defaults() {
        let args = render_args(&[]).unwrap();
        let (typed, _) = args.to_config().unwrap();
        let d = DevArgs::defaults();
        let (dt, ht) = (d.reduce_tile, d.gate_frames);
        assert_eq!((typed.reduce_tile, typed.gate_frames), (dt, ht));
        assert_eq!(typed.max_layers, d.layers);
        assert_eq!(typed.workgroup_size, d.workgroup);
        assert_eq!(typed.max_render_workgroups, d.render_workgroups);
        assert_eq!(typed.limiter_release_ms, d.limiter_release_ms);
        assert_eq!(typed.limiter_lookahead_ms, d.lookahead_ms);
        assert_eq!(typed.dc_blocker_hz, d.dc_blocker_hz);
        assert_eq!(typed.dc_blocker_hz, kestrel::config::Config::default().dc_blocker_hz);
        assert!(!typed.dc_blocker, "the DC blocker is opt-in");
        assert!(typed.lfo_enabled && typed.mod_env_enabled && typed.sort_voices);
        assert!(typed.filter_enabled && typed.gain_ramp && typed.filter_ramp);
        assert!(typed.limiter && typed.limiter_true_peak && !typed.unchecked_shaders);
        assert_eq!(format!("{:?}", typed.steal_rule), "Quietest");
        assert_eq!(format!("{:?}", typed.admit_rule), "Loudest");
    }

    #[test]
    fn analytic_options_validate_only_applicable_controls() {
        let (cfg, _) = render_args(&["--phase-mode", "analytic", "--phase-seed", "42",
            "--phase-pool", "8", "--phase-preserve-attack-ms", "5"])
            .unwrap().to_config().unwrap();
        assert!(cfg.phase.active());
        assert_eq!(cfg.phase.seed, 42);
        assert_eq!(cfg.phase.pool_size, 8);
        assert_eq!(cfg.phase.preserve_attack_ms, 5.0);
        for flags in [
            &["--phase-mode", "analytic", "--phase-strength", "NaN"][..],
            &["--phase-mode", "analytic", "--phase-pool", "0"][..],
        ] { assert!(render_args(flags).unwrap().to_config().is_err()); }
        assert!(render_args(&["--phase-mode", "analytic", "--phase-continuous", "--phase-pool", "0"])
            .unwrap().to_config().is_ok());
        assert!(render_args(&["--phase-mode", "baseline", "--phase-strength", "NaN", "--phase-pool", "0"])
            .unwrap().to_config().is_ok());
    }

    fn r(args: &[&str]) -> Route {
        route(&args.iter().map(OsString::from).collect::<Vec<_>>())
    }

    /// A bare launch gets the guided renderer; the command line is behind
    /// `--force-cli` wherever it sits; and help is never refused, because it
    /// is how `--force-cli` gets found.
    #[test]
    fn a_command_line_goes_to_the_front_end_it_asked_for() {
        assert_eq!(r(&[]), Route::Guided);
        assert_eq!(r(&["--extras-window"]), Route::Extras);
        assert_eq!(r(&["render", "a.mid", "-s", "f.sf2", "-o", "a.wav"]), Route::Notice);
        assert_eq!(r(&["gpu-info"]), Route::Notice);
        assert_eq!(r(&["--force-cli", "render", "a.mid"]), Route::Cli);
        assert_eq!(r(&["render", "a.mid", "--force-cli"]), Route::Cli);
        assert_eq!(r(&["--help"]), Route::Cli);
        assert_eq!(r(&["-V"]), Route::Cli);
        assert_eq!(r(&["help", "render"]), Route::Cli);
        // The Extras flag is honoured only on its own.
        assert_eq!(r(&["--extras-window", "render"]), Route::Notice);
    }

    /// Declared to clap, so it parses on either side of the subcommand rather
    /// than reaching one as an unknown argument.
    #[test]
    fn force_cli_parses_before_or_after_the_subcommand() {
        use clap::Parser;
        for argv in [
            ["kestrel", "--force-cli", "gpu-info"],
            ["kestrel", "gpu-info", "--force-cli"],
        ] {
            assert!(super::Cli::try_parse_from(argv).is_ok(), "{argv:?}");
        }
    }

    fn render_args(extra: &[&str]) -> Result<super::RenderArgs, clap::Error> {
        use clap::Parser;
        let mut argv = vec!["kestrel", "render", "a.mid", "-s", "f.sf2", "-o", "a.wav"];
        argv.extend_from_slice(extra);
        match super::Cli::try_parse_from(argv)?.cmd {
            super::Cmd::Render(args) => Ok(args),
            _ => unreachable!("the argument list names the render subcommand"),
        }
    }

    /// `--merge` belongs to `--tracks`, and a stem format means nothing to it.
    /// `--stem-format` has a default, which must not count as naming it.
    #[test]
    fn merge_goes_with_tracks_and_not_with_a_stem_format() {
        let merged = |extra: &[&str]| render_args(extra).map(|a| a.to_job().unwrap().stems.map(|s| s.merge));
        assert_eq!(merged(&["--tracks", "all", "--merge"]).unwrap(), Some(true));
        assert_eq!(merged(&["--tracks", "all"]).unwrap(), Some(false));
        assert!(render_args(&["--merge"]).is_err(), "--merge without --tracks");
        assert!(render_args(&["--tracks", "all", "--merge", "--stem-format", "flac"]).is_err());
        // `--tracks` conflicts with `--track`, and clap waives a requirement
        // on an argument that conflicts with one present, so these three
        // need their own conflict with `--track` or they are ignored.
        assert!(render_args(&["--track", "2", "--merge"]).is_err());
        assert!(render_args(&["--track", "2", "--stem-format", "flac"]).is_err());
        assert!(render_args(&["--track", "2", "--track-jobs", "3"]).is_err());
        assert!(render_args(&["--track", "2"]).is_ok());
    }

    /// `--volume` is a percentage from 0 to 200, and 100 is exactly unity, so a
    /// render that never names it is the render it was before. `-15` was
    /// refused as an unknown `-1` until 2026-09-14; it is still refused, but
    /// the message says what it would be as decibels.
    #[test]
    fn volume_is_a_percentage_up_to_200() {
        let gain = |extra: &[&str]| {
            render_args(extra).map(|a| a.to_config().unwrap().0.master_volume)
        };
        assert_eq!(gain(&[]).unwrap(), 1.0);
        assert_eq!(gain(&["--volume", "50"]).unwrap(), 0.5);
        assert_eq!(gain(&["--volume", "200"]).unwrap(), 2.0);
        assert_eq!(gain(&["--volume", "0"]).unwrap(), 0.0);
        assert_eq!(gain(&["--volume=17.5%"]).unwrap(), 0.175);
        assert!(gain(&["--volume", "201"]).is_err());
        let err = render_args(&["--volume", "-15"])
            .err()
            .expect("-15 is refused")
            .to_string();
        assert!(err.contains("--volume 17.8"), "{err}");
    }

    /// The ceiling's own help describes negative values, and clap refused them
    /// with a space until 2026-09-14.
    #[test]
    fn a_negative_ceiling_parses_with_a_space() {
        assert_eq!(render_args(&["--ceiling-db", "-1"]).unwrap().ceiling_db, Some(-1.0));
    }
}
