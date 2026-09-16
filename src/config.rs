// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! All tunables live here. Nothing in this crate reads a magic constant that is \[1\]

use crate::limiter::LimiterMode;
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
}

#[derive(Debug, Clone)]
pub struct Config {
    // [12]
    pub sample_rate: u32,
    /// Output channel count. Only 2 is implemented; kept here so the constant \[13\]
    pub channels: u32,

    // [14]
    pub block_frames: u32,
    /// Frames handled per workgroup reduction step inside the render shader. \[15\]
    pub reduce_tile: u32,
    /// Frames between note-off gate checks, and so the granularity of a \[16\]
    pub gate_frames: u32,
    /// Invocations per workgroup in the render pass. One voice per invocation.
    pub workgroup_size: u32,
    /// Upper bound on workgroups dispatched by the render pass. Voices beyond \[17\]
    pub max_render_workgroups: u32,
    /// Upper bound on the grid taken by every *per-voice* pass: spawn, steal \[18\]
    pub max_pool_workgroups: u32,

    // [19]
    pub max_voices: u32,
    /// Voices per (channel, key) note-on. SoundFont layers spawn one voice \[20\]
    pub max_layers: u32,
    pub steal_rule: StealRule,
    /// Which note-ons survive when a block oversubscribes the pool.
    pub admit_rule: AdmitRule,
    /// Admission candidates one block may hold before they are thinned. \[21\]
    pub max_block_candidates: u32,
    /// Ceiling on how much of the pool one block may steal, in percent. \[22\]
    pub max_steal_percent: u32,
    /// Frames a stolen voice fades over instead of being cut. \[23\]
    pub steal_fade_frames: u32,
    /// Re-sort the voice pool by (region, envelope stage, phase) during \[24\]
    pub sort_voices: bool,

    // ---- dsp -------------------------------------------------------------
    pub interpolation: Interpolation,
    pub decay_curve: EnvelopeCurve,
    pub release_curve: EnvelopeCurve,
    /// Envelope level below which a releasing voice is considered dead. \[25\]
    pub env_floor: f32,
    /// Enable the per-voice low-pass filter. SoundFonts that leave the cutoff \[26\]
    pub filter_enabled: bool,
    /// Linear gain applied to the final mix before limiting.
    pub master_volume: f32,
    /// How many copies of the params table the sound controllers CC71-CC75 may \[27\]
    pub max_param_variants: u32,
    /// Apply the soft limiter to the mixed output.
    pub limiter: bool,
    /// Which limiter runs. `Brickwall` is the default: a lookahead true-peak \[28\]
    pub limiter_mode: LimiterMode,
    /// Brickwall ceiling in dBFS. 0.0 is flat full scale.
    pub limiter_ceiling_db: f64,
    /// Brickwall lookahead in milliseconds. This is also the render latency.
    pub limiter_lookahead_ms: f64,
    /// Brickwall release in milliseconds. Short keeps the material after a \[29\]
    pub limiter_release_ms: f64,
    /// Time constant of the brickwall's sustained stage, in milliseconds. \[30\]
    pub limiter_sustain_ms: f64,
    /// Detect inter-sample peaks by 4x oversampling rather than looking at the \[31\]
    pub limiter_true_peak: bool,
    /// Limiter attack/release in seconds.
    pub limiter_attack: f32,
    pub limiter_release: f32,

    // [32]
    pub resample_pool: bool,
    /// Soft ceiling on sample-pool bytes on the device. When the pool does not \[33\]
    pub sample_pool_budget: u64,

    // [34]
    pub clamp_output: bool,

    /// Ramp channel gain across a gate tile instead of stepping it. \[35\]
    pub gain_ramp: bool,

    /// Ramp biquad coefficients across a gate tile instead of switching them. \[36\]
    pub filter_ramp: bool,

    /// Evaluate SF2 vibrato and tremolo LFOs. \[37\]
    pub lfo_enabled: bool,

    /// Evaluate the SF2 modulation envelope and its pitch and filter \[38\]
    pub mod_env_enabled: bool,

    /// Use Kahan compensation for the per-thread partial sums in the reduce \[39\]
    pub kahan_reduce: bool,
    /// Check every output block for NaN/Inf. Always on in debug builds.
    pub nan_guard: bool,
    /// Compile the shaders without naga's automatic bounds clamps and loop \[40\]
    pub unchecked_shaders: bool,

    // [41]
    pub profile: bool,
    /// Force a specific wgpu backend, e.g. "vulkan" or "dx12".
    pub gpu_backend: Option<String>,
    /// Substring match against the adapter name.
    pub gpu_adapter: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            sample_rate: 48_000,
            channels: 2,

            block_frames: 4096,
            // [42]
            reduce_tile: 4,
            gate_frames: 32,
            workgroup_size: 256,
            // [43]
            max_render_workgroups: 2048,
            max_pool_workgroups: u32::MAX,

            max_voices: 1 << 20,
            max_layers: 16,
            steal_rule: StealRule::Quietest,
            admit_rule: AdmitRule::Loudest,
            max_block_candidates: 1 << 27,
            max_steal_percent: 25,
            steal_fade_frames: 96,
            sort_voices: true,

            interpolation: Interpolation::Linear,
            decay_curve: EnvelopeCurve::Exponential,
            release_curve: EnvelopeCurve::Exponential,
            env_floor: 1.0e-5, // -100 dB
            filter_enabled: true,
            master_volume: 1.0,
            max_param_variants: 32,
            limiter: true,
            limiter_mode: LimiterMode::Brickwall,
            limiter_ceiling_db: 0.0,
            limiter_lookahead_ms: 2.0,
            limiter_release_ms: 60.0,
            limiter_sustain_ms: 0.0,
            limiter_true_peak: true,
            limiter_attack: 0.01,
            limiter_release: 0.1,

            resample_pool: true,
            // [44]
            sample_pool_budget: 2 << 30,

            clamp_output: true,
            gain_ramp: true,
            filter_ramp: true,
            lfo_enabled: true,
            mod_env_enabled: true,

            kahan_reduce: false,
            nan_guard: cfg!(debug_assertions),
            unchecked_shaders: false,

            profile: false,
            gpu_backend: None,
            gpu_adapter: None,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
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
        // [45]
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
        // [46]
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
        // [47]
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

    /// Frames in which a steal may be scheduled. The fade has to finish inside \[48\]
    pub fn steal_span(&self) -> u32 {
        self.block_frames.saturating_sub(self.steal_fade_frames).max(1)
    }

    /// Voice slots to allocate. A stolen voice keeps sounding until its own \[49\]
    pub fn pool_slots(&self) -> u32 {
        self.max_voices.saturating_add(self.max_steal())
    }

    /// The most voices one block may steal. Integer arithmetic, and both \[50\]
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
}
