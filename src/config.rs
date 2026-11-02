// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! All tunables live here. Nothing in this crate reads a magic constant that is \[1\]

use crate::limiter::{LevelerSettings, LimiterMode};
use anyhow::{bail, Result};

/// Sample interpolation quality used by both the CPU reference and the GPU path. \[2\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Interpolation {
    /// Truncate to the nearest lower sample. One fetch per voice per frame.
    Nearest = 0,
    /// Two-point linear. The default.
    Linear = 1,
    /// Four-point Catmull-Rom. Roughly 2x the sample-pool bandwidth.
    Cubic = 2,
}

impl Interpolation {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "nearest" | "none" | "0" => Some(Interpolation::Nearest),
            "linear" | "1" => Some(Interpolation::Linear),
            "cubic" | "hermite" | "2" => Some(Interpolation::Cubic),
            _ => None,
        }
    }

    /// Number of pool samples the interpolator touches per frame.
    pub fn taps(self) -> u32 {
        match self {
            Interpolation::Nearest => 1,
            Interpolation::Linear => 2,
            Interpolation::Cubic => 4,
        }
    }
}

/// Shape of the decay and release envelope segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EnvelopeCurve {
    /// Level falls by a constant number of dB per frame (multiplicative). \[3\]
    Exponential = 0,
    /// Level falls by a constant amount per frame (additive). Matches the \[4\]
    Linear = 1,
}

impl EnvelopeCurve {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "exp" | "exponential" | "db" => Some(EnvelopeCurve::Exponential),
            "lin" | "linear" => Some(EnvelopeCurve::Linear),
            _ => None,
        }
    }
}

/// What happens when a note-on arrives and the voice pool is full. \[5\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum StealRule {
    /// Kill the voices with the smallest note id, i.e. the ones that started \[6\]
    Oldest = 0,
    /// Refuse to start the new note instead of killing an old one.
    DropNew = 1,
    /// Kill the voices with the lowest envelope level, which are the ones \[7\]
    Quietest = 2,
}

/// Which of a saturated block's note-ons get admitted when they outnumber the \[8\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AdmitRule {
    /// Rank by `voice::admit_key`: opening amplitude on a logarithmic scale, \[9\]
    Loudest = 0,
    /// Thin the block evenly by event position, ignoring what each note is. \[10\]
    Even = 1,
}

impl Config {
    /// How many of `queued` note-ons a block can admit, given `live` voices \[11\]
    pub fn admit_take(&self, live: u32, queued: u32) -> u32 {
        let cap = self.max_voices;
        let want = queued.min(cap);
        if live + want <= cap {
            return want;
        }
        let need = match self.steal_rule {
            StealRule::DropNew => 0,
            _ => (live + want - cap).min(live).min(self.max_steal()),
        };
        want.min(cap - (live - need))
    }
}

impl AdmitRule {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "loudest" | "loud" | "rank" | "ranked" => Some(AdmitRule::Loudest),
            "even" | "spread" | "flat" => Some(AdmitRule::Even),
            _ => None,
        }
    }
}

impl StealRule {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "oldest" => Some(StealRule::Oldest),
            "quietest" | "quiet" | "level" => Some(StealRule::Quietest),
            "drop" | "dropnew" | "drop-new" => Some(StealRule::DropNew),
            _ => None,
        }
    }
}

/// Which backend renders the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Single-threaded scalar Rust. The ground truth for every test.
    Cpu,
    /// wgpu compute pipeline.
    Gpu,
    /// The convolution backend. Offline only, and see \[12\]
    Merlin,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Optional analytic sample rotation (`--phase-mode analytic`, contributed \[13\]
    pub phase: crate::phase::PhaseSettings,

    // [14]
    pub sample_rate: u32,
    /// Output channel count. Only 2 is implemented; kept here so the constant \[15\]
    pub channels: u32,

    // [16]
    pub block_frames: u32,
    /// Frames handled per workgroup reduction step inside the render shader. \[17\]
    pub reduce_tile: u32,
    /// Frames between note-off gate checks, and so the granularity of a \[18\]
    pub gate_frames: u32,
    /// Invocations per workgroup in the render pass. One voice per invocation.
    pub workgroup_size: u32,
    /// Upper bound on workgroups dispatched by the render pass. Voices beyond \[19\]
    pub max_render_workgroups: u32,
    /// The most voices one submission to the device renders, to start from. A \[20\]
    pub submit_voices: u32,
    /// Upper bound on the grid taken by every *per-voice* pass: spawn, steal \[21\]
    pub max_pool_workgroups: u32,

    // [22]
    pub max_voices: u32,
    /// Voices per (channel, key) note-on. SoundFont layers spawn one voice \[23\]
    pub max_layers: u32,
    pub steal_rule: StealRule,
    /// Which note-ons survive when a block oversubscribes the pool.
    pub admit_rule: AdmitRule,
    /// Note-ons quieter than this are not rendered at all. Each is skipped on \[24\]
    pub min_velocity: u8,
    /// Hold notes to BASSMIDI's 4 ms envelope grid: a note-off takes effect at \[25\]
    pub note_grid: bool,
    /// How a tempo change moves the clock: BASSMIDI's file player (the \[26\]
    pub tempo_timing: crate::midi::TempoTiming,
    /// Play the file as the 31-EDO template writes it: note keys read as \[27\]
    pub edo31: bool,
    /// A tuning that gives every key a pitch of its own: see `tuning`. `None` is \[28\]
    pub tuning: Option<std::sync::Arc<crate::tuning::Tuning>>,
    /// Apply MIDI Tuning Standard single-note tuning changes and bulk dumps, which \[29\]
    pub mts_notes: bool,
    /// Admission candidates one block may hold before they are thinned. \[30\]
    pub max_block_candidates: u32,
    /// Build a pool-less backend's (merlin's) spawn commands for the next \[31\]
    pub prebuild_voices: bool,
    /// Duplicate-note coalescing: note-ons that would build voices identical \[32\]
    pub coalesce: bool,
    /// How many note-ons the coalescing look-ahead reads past one whose release \[33\]
    pub coalesce_window: u32,
    /// Threads beside the drain's own that read the MIDI file for it \[34\]
    pub drain_threads: usize,
    /// Tracks one worker merges before a reading of the file gets another \[35\]
    pub drain_tracks_per_worker: usize,
    /// Threads that build a block's admitted voices, `Driver::materialise`. \[36\]
    pub materialise_threads: usize,
    /// The most spawn commands the driver builds at once for a backend that \[37\]
    pub spawn_chunk: usize,
    /// Ceiling on how much of the pool one block may steal, in percent. \[38\]
    pub max_steal_percent: u32,
    /// Frames a stolen voice fades over instead of being cut. \[39\]
    pub steal_fade_frames: u32,
    /// Re-sort the voice pool by (region, envelope stage, phase) during \[40\]
    pub sort_voices: bool,
    /// Skip the device for a block in which no voice is alive and none is \[41\]
    pub skip_silence: bool,

    // ---- dsp -------------------------------------------------------------
    pub interpolation: Interpolation,
    pub decay_curve: EnvelopeCurve,
    pub release_curve: EnvelopeCurve,
    /// Envelope level below which a releasing voice is considered dead. \[42\]
    pub env_floor: f32,
    /// Enable the per-voice low-pass filter. SoundFonts that leave the cutoff \[43\]
    pub filter_enabled: bool,
    /// `--volume` as a factor: 1 leaves the output as it is. \[44\]
    pub master_volume: f32,
    /// High-pass the mix ahead of the limiter, removing the sub-audio \[45\]
    pub dc_blocker: bool,
    /// The DC blocker's corner, in Hz: a second-order Butterworth high-pass.
    pub dc_blocker_hz: f64,
    /// How many copies of the params table the sound controllers CC71-CC75 may \[46\]
    pub max_param_variants: u32,
    /// Apply the soft limiter to the mixed output.
    pub limiter: bool,
    /// Which limiter runs. **A render from the command line, the guided \[47\]
    pub limiter_mode: LimiterMode,
    /// What `LimiterMode::Balanced` is tuned to. A preset with no flags: it \[48\]
    pub leveler: LevelerSettings,
    /// Brickwall ceiling in dBFS. 0.0 is flat full scale.
    pub limiter_ceiling_db: f64,
    /// Brickwall lookahead in milliseconds. This is also the render latency.
    pub limiter_lookahead_ms: f64,
    /// Brickwall release in milliseconds. Short keeps the material after a \[49\]
    pub limiter_release_ms: f64,
    /// Time constant of the brickwall's sustained stage, in milliseconds. \[50\]
    pub limiter_sustain_ms: f64,
    /// How long the brickwall keeps its gain, in milliseconds, after the \[51\]
    pub limiter_hold_ms: f64,
    /// Detect inter-sample peaks by 4x oversampling rather than looking at the \[52\]
    pub limiter_true_peak: bool,
    /// Limiter attack/release in seconds.
    pub limiter_attack: f32,
    pub limiter_release: f32,

    // [53]
    pub resample_pool: bool,
    /// Soft ceiling on sample-pool bytes on the device. When the pool does not \[54\]
    pub sample_pool_budget: u64,
    /// The most bytes of the sample pool one device buffer holds; 0, the \[55\]
    pub pool_part_bytes: u64,

    // [56]
    pub clamp_output: bool,

    /// Ramp channel gain across a gate tile instead of stepping it. \[57\]
    pub gain_ramp: bool,

    /// Ramp biquad coefficients across a gate tile instead of switching them. \[58\]
    pub filter_ramp: bool,

    /// Evaluate SF2 vibrato and tremolo LFOs. \[59\]
    pub lfo_enabled: bool,

    /// Evaluate the SF2 modulation envelope and its pitch and filter \[60\]
    pub mod_env_enabled: bool,

    /// Use Kahan compensation for the per-thread partial sums in the reduce \[61\]
    pub kahan_reduce: bool,
    /// The most render workgroups a block may have for the reduce pass to give \[62\]
    pub reduce_thin: u32,
    /// Check every output block for NaN/Inf. Always on in debug builds.
    pub nan_guard: bool,
    /// Compile the shaders without naga's automatic bounds clamps and loop \[63\]
    pub unchecked_shaders: bool,

    // [64]
    pub merlin_kernel_frames: u32,
    /// Release tail in seconds, for sizing the reference-point buckets.
    pub merlin_release_tail: f64,
    /// Reference points per release span. Ten is the development notes section 3b's \[65\]
    pub merlin_release_buckets: u32,
    /// Ceiling on distinct entries -- `(waveform, release reference)` pairs -- \[66\]
    pub merlin_max_entries: u32,
    /// Ceiling on distinct cached **waveforms**, which is not the same number \[67\]
    pub merlin_max_waves: u32,
    /// Kernel cache budget in MiB. **Zero derives it** from the entry ceiling \[68\]
    pub merlin_cache_mb: u32,
    /// **Host** budget for cached sustain renders, in MiB. \[69\]
    pub merlin_host_mb: u32,
    /// Render merlin's sustain waveforms -- what its kernels and head rows are \[70\]
    pub merlin_gpu_build: bool,
    /// Notes a waveform must be asked for before merlin builds it a kernel. \[71\]
    pub merlin_promote_after: u32,
    /// Count what the per-entry tail correction for CC7 would cost, and report \[72\]
    pub merlin_probe_tails: bool,
    /// Ceiling on impulses in one partition step.
    pub merlin_max_impulses: u32,
    /// Pretend one storage binding holds only this many KiB. **Zero means the \[73\]
    pub merlin_max_binding_kb: u32,
    /// Ceiling on firing *rows* in one partition step. \[74\]
    pub merlin_max_fired: u32,
    /// Distinct biquads merlin will hold, each owning two output buses. \[75\]
    pub merlin_max_filters: u32,
    /// Frames of the sustaining waveform cached per distinct waveform, for the \[76\]
    pub merlin_head_frames: u32,
    /// Round channel bend to this many cents before it reaches the cache key. \[77\]
    pub merlin_bend_cents: f64,
    /// Round each filter's cutoff to a grid of this many cents before it picks \[78\]
    pub merlin_filter_cents: f64,
    /// Key-major merlin's bucket pass in place of merlin: every note filed by \[79\]
    pub merlin_bucket_pass: bool,
    /// Key-major merlin in place of time-major merlin: every key rendered over \[80\]
    pub merlin_key_major: bool,
    /// Key-major on the device (`merlin::kmgpu`): the convolution, the heads, \[81\]
    pub merlin_key_major_gpu: bool,
    /// With `merlin_key_major_gpu`: the records come from merlin's own MIDI \[82\]
    pub merlin_key_parse: bool,
    /// On the device, a (block, entry) pair with fewer impulses than this is \[83\]
    pub merlin_km_direct_below: u32,
    /// Frames of a sustain segment on the device, rounded up to a partition. \[84\]
    pub merlin_km_segment: u32,
    /// Frames of sustains one pool on the device holds. Past it a key's windows \[85\]
    pub merlin_km_pool_frames: u32,
    /// Records of a key planned at once, past which the rest wait for the \[86\]
    pub merlin_km_plan_records: u32,
    /// The most of one key's records in one plan of the parsed render. A \[87\]
    pub merlin_km_slice_records: u32,
    /// The longest window of the per-key render, in frames. Zero is the most \[88\]
    pub merlin_km_window: u32,
    /// Note-ons of one key past which it comes from the parser a part at a \[89\]
    pub merlin_km_stream_notes: u32,
    /// Records of a streamed key handed to the render at a time.
    pub merlin_km_stream_part: u32,
    /// Records of a streamed key held in memory, past which those its windows \[90\]
    pub merlin_km_stream_held: u32,
    /// Controllers the key parser holds in memory, past which they go to disk \[91\]
    pub merlin_km_ctl_held: u32,
    /// Merlin refuses a MIDI file of this many bytes or more, before anything \[92\]
    pub merlin_max_file_bytes: u64,
    /// Bytes of prepared windows merlin's packing thread may have handed the \[93\]
    pub merlin_km_ahead_bytes: u64,
    /// For tests: the device's thread fails at its Nth window, to see what the \[94\]
    pub merlin_km_fail_device_at: u32,
    /// Events a worker of merlin's pre-pass reads before it tells the render \[95\]
    pub merlin_told_events: u32,
    /// Merlin refuses a file of more note-ons than this. The pre-pass counts \[96\]
    pub merlin_max_notes: u64,
    /// Bytes the key store leaves free on its disk. It stops there and the \[97\]
    pub merlin_disk_floor: u64,
    /// Notes a bend moves under follow it: each channel's are rendered unbent \[98\]
    pub merlin_km_warp: bool,
    /// The bucket store's memory, in MiB, past which its records go to a file. \[99\]
    pub merlin_bucket_mb: u32,
    /// Keep an order-free digest of every note merlin plans, copy by copy, and \[100\]
    pub merlin_digest: bool,

    // [101]
    pub profile: bool,
    /// Force a specific wgpu backend, e.g. "vulkan" or "dx12".
    pub gpu_backend: Option<String>,
    /// Substring match against the adapter name.
    pub gpu_adapter: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            phase: crate::phase::PhaseSettings::default(),
            sample_rate: 48_000,
            channels: 2,

            block_frames: 4096,
            // [102]
            reduce_tile: 4,
            gate_frames: 32,
            workgroup_size: 256,
            // [103]
            max_render_workgroups: 2048,
            submit_voices: 4_194_304,
            max_pool_workgroups: u32::MAX,

            max_voices: 1 << 20,
            max_layers: 16,
            steal_rule: StealRule::Quietest,
            admit_rule: AdmitRule::Loudest,
            min_velocity: 0,
            note_grid: false,
            tempo_timing: crate::midi::TempoTiming::Bassmidi,
            edo31: false,
            tuning: None,
            mts_notes: false,
            max_block_candidates: 1 << 27,
            prebuild_voices: true,
            coalesce: false,
            coalesce_window: 1 << 22,
            drain_threads: 4,
            drain_tracks_per_worker: 8192,
            materialise_threads: 4,
            spawn_chunk: 1 << 20,
            max_steal_percent: 25,
            steal_fade_frames: 96,
            sort_voices: true,
            skip_silence: true,

            interpolation: Interpolation::Linear,
            decay_curve: EnvelopeCurve::Exponential,
            release_curve: EnvelopeCurve::Exponential,
            env_floor: 1.0e-5, // -100 dB
            filter_enabled: true,
            master_volume: 1.0,
            dc_blocker: false,
            // [104]
            dc_blocker_hz: 15.0,
            max_param_variants: 32,
            limiter: true,
            limiter_mode: LimiterMode::Brickwall,
            leveler: LevelerSettings::default(),
            limiter_ceiling_db: 0.0,
            limiter_lookahead_ms: 2.0,
            limiter_release_ms: 60.0,
            limiter_sustain_ms: 0.0,
            limiter_hold_ms: 20.0,
            limiter_true_peak: true,
            limiter_attack: 0.01,
            limiter_release: 0.1,

            resample_pool: true,
            // [105]
            sample_pool_budget: 2 << 30,
            pool_part_bytes: 0,

            clamp_output: true,
            gain_ramp: true,
            filter_ramp: true,
            lfo_enabled: true,
            mod_env_enabled: true,

            kahan_reduce: false,
            reduce_thin: 64,
            nan_guard: cfg!(debug_assertions),
            unchecked_shaders: false,
            // [106]
            merlin_kernel_frames: 0,
            merlin_release_tail: 1.82,
            merlin_release_buckets: 10,
            // [107]
            merlin_max_entries: 1 << 20,
            // [108]
            merlin_max_waves: 768,
            merlin_cache_mb: 0,
            // [109]
            merlin_host_mb: 4096,
            merlin_gpu_build: false,
            // 256, from the measured use-count curve; see the field docs.
            merlin_promote_after: 256,
            merlin_probe_tails: false,
            // [110]
            merlin_max_impulses: 1 << 22,
            merlin_max_binding_kb: 0,
            merlin_max_fired: 16384,
            merlin_max_filters: 0,
            merlin_head_frames: 0,
            merlin_bend_cents: 0.0,
            merlin_filter_cents: 0.0,
            merlin_bucket_pass: false,
            merlin_key_major: false,
            merlin_key_major_gpu: false,
            merlin_key_parse: false,
            merlin_km_direct_below: crate::merlin::kmgpu::DIRECT_BELOW as u32,
            merlin_km_segment: crate::merlin::kmgpu::SEGMENT_FRAMES as u32,
            merlin_km_pool_frames: crate::merlin::keymajor::POOL_FRAMES as u32,
            merlin_km_plan_records: crate::merlin::keymajor::PLAN_RECORDS as u32,
            merlin_km_slice_records: crate::merlin::keymajor::SLICE_RECORDS as u32,
            merlin_km_window: 0,
            merlin_km_stream_notes: crate::merlin::keymajor::STREAM_NOTES,
            merlin_km_stream_part: crate::merlin::keymajor::STREAM_PART,
            merlin_km_stream_held: crate::merlin::keymajor::STREAM_HELD,
            merlin_km_ctl_held: crate::merlin::keyparse::CTL_HELD,
            merlin_max_file_bytes: 200_000_000_000,
            merlin_km_ahead_bytes: crate::merlin::keymajor::AHEAD_BYTES,
            merlin_km_fail_device_at: 0,
            merlin_told_events: crate::merlin::keyparse::TOLD_EVERY,
            merlin_max_notes: 25_000_000_000,
            merlin_disk_floor: crate::merlin::keystore::DISK_FLOOR,
            merlin_km_warp: false,
            merlin_bucket_mb: 0,
            merlin_digest: false,

            profile: false,
            gpu_backend: None,
            gpu_adapter: None,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        self.phase.validate()?;
        if self.coalesce && self.coalesce_window == 0 {
            bail!("--coalesce-window must be at least 1 note-on");
        }
        if self.edo31 && self.tuning.is_some() {
            bail!(
                "--31edo plays a file written for the 31-EDO template, which sets its own tuning, and \
                 a tuning laid over every key (`Config::tuning`) would move it again; use one of them"
            );
        }
        if self.channels != 2 {
            bail!("only stereo output is implemented (channels = {})", self.channels);
        }
        if self.sample_rate < 8_000 || self.sample_rate > 768_000 {
            bail!("sample_rate {} out of range", self.sample_rate);
        }
        if self.block_frames == 0 || !self.block_frames.is_multiple_of(self.reduce_tile) {
            bail!(
                "block_frames ({}) must be a non-zero multiple of reduce_tile ({})",
                self.block_frames,
                self.reduce_tile
            );
        }
        if !self.workgroup_size.is_power_of_two()
            || self.workgroup_size < 32
            || self.workgroup_size > 1024
        {
            bail!(
                "workgroup_size {} must be a power of two in [32, 1024]",
                self.workgroup_size
            );
        }
        if !self.reduce_tile.is_power_of_two() {
            bail!("reduce_tile {} must be a power of two", self.reduce_tile);
        }
        if self.gate_frames == 0
            || !self.gate_frames.is_multiple_of(self.reduce_tile)
            || !self.block_frames.is_multiple_of(self.gate_frames)
        {
            bail!(
                "gate_frames ({}) must be a multiple of reduce_tile ({}) and divide \
                 block_frames ({})",
                self.gate_frames,
                self.reduce_tile,
                self.block_frames
            );
        }
        // [111]
        let shared_bytes = self.workgroup_size as u64 * (self.reduce_tile as u64 * 2 + 1) * 4;
        if shared_bytes > 49152 {
            bail!(
                "workgroup_size {} with reduce_tile {} needs {} bytes of workgroup storage, \
                 over the 48 KiB a compute workgroup can have",
                self.workgroup_size,
                self.reduce_tile,
                shared_bytes
            );
        }
        // [112]
        if self.workgroup_size < self.reduce_tile * 2 {
            bail!(
                "workgroup_size {} must be at least twice reduce_tile {}",
                self.workgroup_size,
                self.reduce_tile
            );
        }
        if !(-24.0..=0.0).contains(&self.limiter_ceiling_db) {
            bail!(
                "limiter_ceiling_db {} must be in -24..=0",
                self.limiter_ceiling_db
            );
        }
        let lv = &self.leveler;
        if !(-24.0..=24.0).contains(&lv.input_gain_db)
            || !(-60.0..=0.0).contains(&lv.threshold_db)
            || !(1.0..=100.0).contains(&lv.ratio)
            || !(0.0..=24.0).contains(&lv.knee_db)
            || !(0.0..=120.0).contains(&lv.max_reduction_db)
            || !(1.0..=200.0).contains(&lv.detector_hp_hz)
            || lv.detector_hp_hz >= self.sample_rate as f64 / 10.0
            || ![lv.detector_attack_ms, lv.detector_release_ms, lv.gain_attack_ms, lv.gain_release_ms]
                .iter()
                .all(|ms| (1.0..=5000.0).contains(ms))
        {
            bail!("the leveler settings {lv:?} are out of range");
        }
        if self.submit_voices == 0 {
            bail!("submit_voices must be at least 1");
        }
        if !(0.0..=2.0).contains(&self.master_volume) {
            bail!("master_volume {} must be in 0..=2", self.master_volume);
        }
        // [113]
        if !(1.0..=200.0).contains(&self.dc_blocker_hz) || self.dc_blocker_hz >= self.sample_rate as f64 / 10.0 {
            bail!("dc_blocker_hz {} must be in 1..=200 and under a tenth of the rate", self.dc_blocker_hz);
        }
        if !(0.05..=100.0).contains(&self.limiter_lookahead_ms) {
            bail!(
                "limiter_lookahead_ms {} must be in 0.05..=100",
                self.limiter_lookahead_ms
            );
        }
        if !(0.0..=10000.0).contains(&self.limiter_sustain_ms) {
            bail!(
                "limiter_sustain_ms {} must be in 0..=10000",
                self.limiter_sustain_ms
            );
        }
        if !(0.0..=200.0).contains(&self.limiter_hold_ms) {
            bail!("limiter_hold_ms {} must be in 0..=200", self.limiter_hold_ms);
        }
        if !(0.1..=5000.0).contains(&self.limiter_release_ms) {
            bail!(
                "limiter_release_ms {} must be in 0.1..=5000",
                self.limiter_release_ms
            );
        }
        if self.max_voices == 0 {
            bail!("max_voices must be non-zero");
        }
        if self.steal_fade_frames == 0 || self.steal_fade_frames >= self.block_frames {
            bail!(
                "steal_fade_frames {} must be in 1..block_frames ({})",
                self.steal_fade_frames,
                self.block_frames
            );
        }
        if self.max_steal_percent == 0 || self.max_steal_percent > 100 {
            bail!(
                "max_steal_percent {} must be in 1..=100",
                self.max_steal_percent
            );
        }
        // [114]
        if self.max_layers > 255 {
            bail!("max_layers {} must be at most 255", self.max_layers);
        }
        if self.block_frames > 65535 {
            bail!("block_frames {} must be at most 65535", self.block_frames);
        }
        if self.max_layers == 0 {
            bail!("max_layers must be non-zero");
        }
        Ok(())
    }

    /// The brickwall ceiling as a linear amplitude.
    pub fn limiter_ceiling(&self) -> f64 {
        10f64.powf(self.limiter_ceiling_db / 20.0)
    }

    /// Frames in which a steal may be scheduled. The fade has to finish inside \[115\]
    pub fn steal_span(&self) -> u32 {
        self.block_frames.saturating_sub(self.steal_fade_frames).max(1)
    }

    /// Frames in one step of the envelope grid: 4 ms, 192 frames at 48 kHz. \[116\]
    pub fn env_step_frames(&self) -> u32 {
        (self.sample_rate / 250).clamp(1, crate::voice::GRID_MASK)
    }

    /// Voice slots to allocate. A stolen voice keeps sounding until its own \[117\]
    pub fn pool_slots(&self) -> u32 {
        self.max_voices.saturating_add(self.max_steal())
    }

    /// The most voices one block may steal. Integer arithmetic, and both \[118\]
    pub fn max_steal(&self) -> u32 {
        let n = self.max_voices as u64 * self.max_steal_percent as u64 / 100;
        (n as u32).max(1)
    }

    /// Bytes of one interleaved stereo output block.
    pub fn block_bytes(&self) -> u64 {
        self.block_frames as u64 * self.channels as u64 * 4
    }

    /// Samples (not frames) in one interleaved output block.
    pub fn block_samples(&self) -> usize {
        self.block_frames as usize * self.channels as usize
    }

    pub fn rate_f64(&self) -> f64 {
        self.sample_rate as f64
    }

    /// Frames of the sustaining waveform merlin caches per waveform, which is \[119\]
    pub fn merlin_head(&self) -> u32 {
        if self.merlin_head_frames == 0 {
            self.sample_rate * 10
        } else {
            self.merlin_head_frames
        }
    }
}
