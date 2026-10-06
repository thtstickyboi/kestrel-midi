// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! wgpu compute backend. \[1\]

mod batch;
pub use batch::{GpuBatch, LaneBackend, LANES_MAX};
mod budget;
use budget::SubmitBudget;
mod device;
pub mod vram;

pub use device::{backend_known, memory_for, print_adapters, survey, AdapterSummary};

use crate::backend::{Backend, BlockStats};
use crate::bank::{Bank, ModEnvParams, RegionParams};
use crate::config::{AdmitRule, Config, EnvelopeCurve, StealRule};
use crate::snap::{Dec, Enc};
use crate::voice::{spawn_pick, SpawnCmd, BASE_CHANNELS, CHAN_FIELDS};
use anyhow::{bail, Context, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Mirrors `Uniforms` in `shaders/common.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    block_frames: u32,
    tiles: u32,
    capacity: u32,
    spawn_count: u32,

    render_workgroups: u32,
    interp: u32,
    exp_decay: u32,
    exp_release: u32,

    env_floor: f32,
    pool_words: u32,
    steal_k: u32,
    sort_bits: u32,

    sort_region_shift: u32,
    sort_stage_shift: u32,
    sort_phase_shift: u32,
    sort_phase_mask: u32,

    sort_dead_region: u32,
    /// Bit 0 bend, bit 1 gain, bit 2 params variant. One word rather than \[2\]
    chan_active: u32,
    params_per_variant: u32,
    steal_by_level: u32,

    /// Half-width of the modulation envelope's pitch-factor table, in \[3\]
    menv_factor_half: u32,
    /// Where the shared `log2` mantissa table starts in the same buffer.
    menv_log2_base: u32,
    /// Where portamento's tables start in the same buffer.
    glide_base: u32,
    /// Frame 0 of this block's position on the envelope grid. See \[4\]
    env_phase: u32,

    /// Channels in a controller row, and words of `[base, run]` meta ahead of \[5\]
    chan_count: u32,
    off_meta_words: u32,
    /// The first render workgroup of this submission, when a block's \[6\]
    render_wg_base: u32,
    /// Always 0. A `+0.0` for a shader to add that no compiler can see is one: it \[7\]
    zero: u32,
}

/// How the voice pool's sort key is packed into 32 bits. \[8\]
#[derive(Debug, Clone, Copy)]
struct SortKeyLayout {
    bits: u32,
    region_shift: u32,
    stage_shift: u32,
    phase_shift: u32,
    phase_mask: u32,
    dead_region: u32,
}

impl SortKeyLayout {
    fn plan(bank: &Bank) -> Self {
        let dead_region = bank.regions.len().max(1) as u32;
        let region_bits = 32 - dead_region.leading_zeros();
        let stage_bits = 3u32;

        // [9]
        let max_len = bank.samples.iter().map(|s| s.len).max().unwrap_or(1).max(1);
        let len_bits = 32 - max_len.leading_zeros();
        let phase_bits = 32u32
            .saturating_sub(region_bits + stage_bits)
            .min(len_bits);
        let phase_shift = len_bits.saturating_sub(phase_bits);
        let phase_mask = if phase_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << phase_bits) - 1
        };

        SortKeyLayout {
            bits: region_bits + stage_bits + phase_bits,
            region_shift: phase_bits + stage_bits,
            stage_shift: phase_bits,
            phase_shift,
            phase_mask,
            dead_region,
        }
    }
}

/// Slots in the device state buffer. Mirrors the `S_*` constants in \[10\]
const S_LIVE: usize = 0;
const S_STOLEN: usize = 8;
const S_DROPPED: usize = 9;
const STATE_SLOTS: usize = 16;

/// Largest grid one dispatch dimension may take. This is the D3D12 ceiling and \[11\]
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// u32 words the voice pool stores per slot. Mirrors `VOICE_FIELDS` in \[12\]
const VOICE_FIELDS: u64 = 26;

/// Words actually allocated per slot for this configuration. \[13\]
fn voice_fields(cfg: &Config) -> u64 {
    base_voice_fields(cfg) + if cfg.phase.active() { 3 } else { 0 }
}

fn base_voice_fields(cfg: &Config) -> u64 {
    if cfg.mod_env_enabled {
        VOICE_FIELDS
    } else if cfg.lfo_enabled {
        VOICE_FIELDS - 1
    } else {
        VOICE_FIELDS - 2
    }
}

/// The largest `max_voices` an adapter can take, given how much of one buffer \[14\]
pub fn max_voices_for_binding(binding_bytes: u64, steal_percent: u32) -> u32 {
    let slots = binding_bytes / (VOICE_FIELDS * 4);
    let v = slots * 100 / (100 + steal_percent.clamp(1, 100) as u64);
    v.min(u32::MAX as u64) as u32
}

/// `max_voices_for_binding` for the layout `cfg` actually allocates, which is \[15\]
pub fn max_voices_for_config(binding_bytes: u64, cfg: &Config) -> u32 {
    let slots = binding_bytes / (voice_fields(cfg) * 4);
    (slots * 100 / (100 + cfg.max_steal_percent.clamp(1, 100) as u64))
        .min(u32::MAX as u64) as u32
}

/// What a render's device buffers come to, in two parts, so a voice limit can be \[16\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceEstimate {
    /// Everything that does not grow with the voice limit: the sample pool, the \[17\]
    pub fixed: u64,
    /// What each pool slot costs: both voice pools, the scan and the sort.
    pub per_slot: u64,
}

impl DeviceEstimate {
    pub fn total(&self, slots: u32) -> u64 {
        self.fixed + self.per_slot * slots as u64
    }
}

/// The device buffers `cfg` and `bank` would allocate, as it would allocate them. \[18\]
pub fn device_estimate(cfg: &Config, bank: &Bank) -> DeviceEstimate {
    estimate_at(cfg, bank, cfg.pool_slots())
}

/// `device_estimate` for a pool of `capacity` slots, which only the partials and the \[19\]
fn estimate_at(cfg: &Config, bank: &Bank, capacity: u32) -> DeviceEstimate {
    let variants = cfg.max_param_variants.max(1) as u64;
    let params = bank.params.len().max(1) as u64 * variants * std::mem::size_of::<RegionParams>() as u64;
    let menv = bank.menv.len().max(1) as u64 * variants * std::mem::size_of::<ModEnvParams>() as u64;
    let tables = read_only_tables(cfg, bank).0.len() as u64 * 4;
    let nwg = cfg.max_render_workgroups.clamp(1, MAX_WORKGROUPS_PER_DIM).min(capacity.div_ceil(cfg.workgroup_size).max(1));
    let partials = cfg.block_frames as u64 * 2 * nwg as u64 * 4;
    let out = cfg.block_frames as u64 * 2 * 4;
    // [20]
    let gates = ((BASE_CHANNELS as u64 * 128 + 1) * 2 + 32768 * 2) * 4;
    let tiles = (cfg.block_frames / cfg.gate_frames) as u64;
    let chans = (tiles + 1) * BASE_CHANNELS as u64 * CHAN_FIELDS as u64 * 4;
    let cmds = 65536u32.min(capacity).max(1024) as u64 * std::mem::size_of::<SpawnCmd>() as u64;
    DeviceEstimate {
        fixed: bank.pool_bytes() + params + menv + tables + partials + out + gates + chans + cmds,
        per_slot: bytes_per_slot(cfg),
    }
}

/// What one pool slot costs for this configuration's voice layout.
pub fn bytes_per_slot(cfg: &Config) -> u64 {
    voice_fields(cfg) * 8 + 24
}

/// How much of a card's memory is left for a render to plan on: an eighth is kept for \[21\]
const MEMORY_RESERVE_DIVISOR: u64 = 8;

/// The most `--max-voices` whose device buffers fit in `memory` bytes with this bank, \[22\]
pub fn max_voices_in_memory(cfg: &Config, bank: &Bank, memory: u64) -> u32 {
    let usable = memory - memory / MEMORY_RESERVE_DIVISOR;
    let est = estimate_at(cfg, bank, u32::MAX);
    let Some(room) = usable.checked_sub(est.fixed) else { return 0 };
    let slots = room / est.per_slot;
    (slots * 100 / (100 + cfg.max_steal_percent.clamp(1, 100) as u64)).min(u32::MAX as u64) as u32
}

/// Output samples a `thin` workgroup reduces: its `@workgroup_size`, which \[23\]
pub(crate) const REDUCE_THIN_LANES: u32 = 64;

/// The most render workgroups a block may have for the reduce pass's `thin` \[24\]
pub(crate) fn reduce_thin_max(cfg: &Config) -> u32 {
    cfg.reduce_thin.min(cfg.workgroup_size).min(256)
}

fn substitute(src: &str, cfg: &Config, bank: &Bank) -> String {
    src.replace("{{WG}}", &cfg.workgroup_size.to_string())
        .replace("{{REDUCE_THIN}}", &reduce_thin_max(cfg).to_string())
        .replace("{{TILE}}", &cfg.reduce_tile.to_string())
        .replace("{{GATE_TILE}}", &cfg.gate_frames.to_string())
        .replace("{{STEAL_FADE}}", &cfg.steal_fade_frames.to_string())
        .replace("{{KAHAN}}", if cfg.kahan_reduce { "true" } else { "false" })
        .replace("{{GAIN_RAMP}}", if cfg.gain_ramp { "true" } else { "false" })
        .replace("{{FILTER_RAMP}}", if cfg.filter_ramp { "true" } else { "false" })
        .replace("{{USE_LFO}}", if cfg.lfo_enabled { "true" } else { "false" })
        // [25]
        .replace(
            "{{USE_LFO_VOLUME}}",
            if cfg.lfo_enabled && bank.uses_lfo_volume { "true" } else { "false" },
        )
        .replace(
            "{{USE_LFO_PITCH}}",
            if cfg.lfo_enabled && bank.uses_lfo_pitch { "true" } else { "false" },
        )
        .replace(
            "{{USE_MOD_ENV}}",
            if cfg.mod_env_enabled { "true" } else { "false" },
        )
        .replace("{{SAMPLE_RATE}}", &cfg.sample_rate.to_string())
        .replace("{{ENV_STEP}}", &cfg.env_step_frames().to_string())
        .replace("{{NOTE_GRID}}", if cfg.note_grid { "true" } else { "false" })
        .replace("{{VOICE_FIELDS}}", &voice_fields(cfg).to_string())
        .replace("{{ROTATION_BASE}}", &base_voice_fields(cfg).to_string())
        .replace("{{ANALYTIC}}", if cfg.phase.active() { "true" } else { "false" })
        .replace("{{PRESERVE_PHASE_ATTACK}}", if cfg.phase.preserve_attack_ms > 0.0 { "true" } else { "false" })
}

/// Write `bytes` into `buf` from `offset`, 64 MiB at a time, letting the device \[26\]
fn upload_in_pieces(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    // A multiple of wgpu's 4-byte copy alignment, so every piece is one.
    const PIECE: usize = 64 << 20;
    for (i, piece) in bytes.chunks(PIECE).enumerate() {
        queue.write_buffer(buf, offset + (i * PIECE) as u64, piece);
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| device::lost(Some(device), format_args!("device poll failed: {e:?}")))?;
    }
    Ok(())
}

fn shader_source(body: &str, cfg: &Config, bank: &Bank) -> String {
    let mut s = substitute(include_str!("../../shaders/common.wgsl"), cfg, bank);
    s.push('\n');
    s.push_str(&substitute(body, cfg, bank));
    s
}

/// The render pass with or without the channel controller path and the glide \[27\]
fn render_source(cfg: &Config, bank: &Bank, parts: PoolParts, chan: bool, glide: bool, split: bool) -> String {
    let analytic = cfg.phase.active();
    let [pool_decls, pool_fetch, pool_pairs] = pool_wgsl(parts);
    let body = include_str!("../../shaders/render.wgsl")
        .replace("{{POOL_PARTS}}", &pool_decls)
        .replace("{{POOL_FETCH}}", &pool_fetch)
        .replace("{{POOL_PAIRS}}", &pool_pairs)
        .replace("{{CHAN}}", if chan { "true" } else { "false" })
        .replace("{{GLIDE}}", if glide { "true" } else { "false" })
        .replace("{{WG_BASE}}", if split { "u.render_wg_base" } else { "0u" })
        .replace("{{PHASE_FUNCTIONS}}", if analytic { include_str!("../../shaders/phase.wgsl") } else { "" })
        .replace("{{PHASE_LOAD}}", if analytic {
            "let pm = voices[F_REGION * c + v] * 4u;
             rotation_meta = vec3<u32>(phase_data[pm], phase_data[pm + 1u], phase_data[pm + 2u]);
             rotation = vec3<f32>(bitcast<f32>(voices[F_ROT_C * c + v]),
                 bitcast<f32>(voices[F_ROT_S * c + v]), bitcast<f32>(voices[F_ROT_SCALE * c + v]));"
        } else { "" })
        .replace("{{PHASE_SAMPLE}}", if analytic {
            "let s = interpolate_rotation(smp_base, phase_hi, frac_of(phase_lo),
                looping, loop_start, loop_end, smp_len, rotation_meta, rotation);"
        } else {
            "let s = interpolate(smp_base, phase_hi, frac_of(phase_lo),
                looping, loop_start, loop_end, smp_len);"
        });
    shader_source(&body, cfg, bank)
}

/// Where the pool's second part is bound; the third and fourth follow it. \[28\]
const POOL_PART_BINDING: u32 = 13;

/// What a split pool adds to `render.wgsl`: `POOL_PARTS`, the declarations \[29\]
fn pool_wgsl(parts: PoolParts) -> [String; 3] {
    if parts.count <= 1 {
        return Default::default();
    }
    let each = parts.words_each as u64;
    let decls = (1..parts.count)
        .map(|k| format!("@group(0) @binding({}) var<storage, read> pool{k}: array<u32>;\n", POOL_PART_BINDING + k - 1))
        .collect();
    let mut fetch = format!("\n    if (w >= {each}u) {{");
    for k in 1..parts.count {
        let lo = k as u64 * each;
        if k + 1 < parts.count {
            fetch += &format!("\n        if (w < {}u) {{ return unpack2x16snorm(pool{k}[w - {lo}u]); }}", lo + each);
        } else {
            fetch += &format!("\n        return unpack2x16snorm(pool{k}[w - {lo}u]);");
        }
    }
    fetch += "\n    }";
    let pair = |name: &str, lo: u64| {
        let o = if lo == 0 { "w".to_string() } else { format!("w - {lo}u") };
        format!("{{ let o = {o}; return vec4<f32>(unpack2x16snorm({name}[o]), unpack2x16snorm({name}[o + 1u])); }}")
    };
    let mut pairs = format!("\n    if (w + 1u < u.pool_words) {{\n        if (w + 1u < {each}u) {}", pair("pool", 0));
    for k in 1..parts.count {
        let lo = k as u64 * each;
        let within = if k + 1 < parts.count { format!("w >= {lo}u && w + 1u < {}u", lo + each) } else { format!("w >= {lo}u") };
        pairs += &format!("\n        if ({within}) {}", pair(&format!("pool{k}"), lo));
    }
    pairs += "\n    }";
    [decls, fetch, pairs]
}

/// The render bind group, plus the analytic quadrature at binding 10 when \[30\]
fn bind_render(device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer], phase: Option<&wgpu::Buffer>, pool_rest: &[wgpu::Buffer]) -> wgpu::BindGroup {
    let mut buffers = buffers.to_vec();
    if let Some(phase) = phase { buffers.push(phase); }
    let mut entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    entries.extend(pool_rest.iter().enumerate().map(|(k, b)| wgpu::BindGroupEntry {
        binding: POOL_PART_BINDING + k as u32,
        resource: b.as_entire_binding(),
    }));
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

/// Compile one fully substituted module and one pipeline per entry point.
fn compile(
    device: &wgpu::Device,
    cfg: &Config,
    name: &str,
    src: String,
    layout: &wgpu::BindGroupLayout,
    entries: &[&str],
) -> Vec<wgpu::ComputePipeline> {
    let desc = wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    };
    let module = if cfg.unchecked_shaders {
        // [31]
        unsafe {
            device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
        }
    } else {
        device.create_shader_module(desc)
    };
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(name),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    entries
        .iter()
        .map(|e| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{name}::{e}")),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect()
}

struct Pipelines {
    spawn: wgpu::ComputePipeline,
    spawn_commit: wgpu::ComputePipeline,
    render: wgpu::ComputePipeline,
    /// The same pass compiled with the channel controller path in it. Selected \[32\]
    render_chan: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    /// The reduce for a block with few render workgroups: one thread a sample \[33\]
    reduce_thin: wgpu::ComputePipeline,
    scan_local: wgpu::ComputePipeline,
    scan_blocks: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    compact_commit: wgpu::ComputePipeline,
    mark_stolen: wgpu::ComputePipeline,
    note_stolen: wgpu::ComputePipeline,
    sel_clear: wgpu::ComputePipeline,
    sel_init: wgpu::ComputePipeline,
    sel_histogram: wgpu::ComputePipeline,
    sel_refine: wgpu::ComputePipeline,
    sort_init: wgpu::ComputePipeline,
    sort_advance: wgpu::ComputePipeline,
    sort_build_keys: wgpu::ComputePipeline,
    sort_scan_local: wgpu::ComputePipeline,
    sort_scan_blocks: wgpu::ComputePipeline,
    sort_split: wgpu::ComputePipeline,
    sort_gather: wgpu::ComputePipeline,
}

struct Layouts {
    spawn: wgpu::BindGroupLayout,
    render: wgpu::BindGroupLayout,
    reduce: wgpu::BindGroupLayout,
    compact: wgpu::BindGroupLayout,
    select: wgpu::BindGroupLayout,
    sort: wgpu::BindGroupLayout,
}

/// Bind groups for one parity of the voice double buffer.
struct Groups {
    spawn: wgpu::BindGroup,
    render: wgpu::BindGroup,
    compact: wgpu::BindGroup,
    select: wgpu::BindGroup,
    /// Indexed by which of the two (key, index) buffers currently holds the \[34\]
    sort: [wgpu::BindGroup; 2],
}

#[allow(dead_code)] // several buffers are only referenced through bind groups
pub struct GpuSynth {
    cfg: Config,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,

    pipelines: Pipelines,
    groups: [Groups; 2],
    reduce_group: wgpu::BindGroup,
    layouts: Layouts,

    uniform_buf: wgpu::Buffer,
    gates_buf: wgpu::Buffer,
    chan_buf: wgpu::Buffer,
    bend_active: bool,
    gain_active: bool,
    variant_active: bool,
    cut_active: bool,
    voices: [wgpu::Buffer; 2],
    partials_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    scan_buf: wgpu::Buffer,
    block_sums_buf: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    sort_keys_buf: wgpu::Buffer,
    pairs: [wgpu::Buffer; 2],
    sort_key: SortKeyLayout,
    cmds_buf: wgpu::Buffer,
    cmds_capacity: u32,
    /// Note-off runs the gates buffer can hold behind its meta header, two \[35\]
    off_runs_capacity: u64,
    /// Words of that meta header, `(slots + 1) * 2`. Grows with the MIDI ports \[36\]
    off_meta_words: u64,
    /// Channels in a controller row this block, read off the rows' length.
    chan_count: u32,
    /// The most of one buffer the adapter binds to a shader, which is as far \[37\]
    binding_cap: u64,
    pool_buf: wgpu::Buffer,
    /// The pool's other parts, when it has them; see `pool_parts`.
    pool_rest: Vec<wgpu::Buffer>,
    pool_parts: PoolParts,
    pool_words: u32,
    /// Analytic quadrature and per-region metadata (`PhaseBank::words`), bound \[38\]
    phase_buf: Option<wgpu::Buffer>,
    params_buf: wgpu::Buffer,
    params_per_variant: u32,
    menv_buf: wgpu::Buffer,
    // [39]
    menv_factor_buf: wgpu::Buffer,
    menv_per_variant: u32,
    menv_factor_half: u32,
    menv_log2_base: u32,
    glide_base: u32,
    /// Whether the driver says a voice may glide during the next block.
    glide_active: bool,
    /// The render pass with the glide path in it, plain and with the \[40\]
    render_glide: Option<[wgpu::ComputePipeline; 2]>,
    /// Their sources, fully substituted, kept so the compile can happen then.
    glide_sources: [String; 2],
    /// The render pass for a block that goes up in parts, the one that reads \[41\]
    render_split: [Option<wgpu::ComputePipeline>; 4],
    split_sources: [String; 4],

    readback_out: wgpu::Buffer,
    readback_state: wgpu::Buffer,

    /// Which of `voices` currently holds the live pool.
    parity: usize,
    /// Host mirror of the device live count, exact because every change to it \[42\]
    live: u32,
    /// Allocated voice slots, `Config::pool_slots()`. The SoA stride.
    slots: u32,
    /// The next block's position on the envelope grid: frames rendered so \[43\]
    env_phase: u32,
    /// `(steal_k, spawn_count)` of the block `spawn` planned and uploaded, \[44\]
    planned: (u32, u32),
    /// Reused buffer for the thinned spawn list, so a saturated block does not \[45\]
    spawn_scratch: Vec<SpawnCmd>,
    stolen: u64,
    dropped: u64,
    peak: f32,

    timing: Option<Timing>,
    last_timings: Vec<(&'static str, f64)>,
    vram_bytes: u64,
    /// Of `vram_bytes`, what lives in the `GpuShared` this was built on: the \[46\]
    shared_bytes: u64,
    pool_bytes: u64,
    voice_bytes: u64,
    partial_bytes: u64,
    /// Render workgroups the partials buffer holds; see \[47\]
    max_nwg: u32,
    /// The last block's submission, which is what `finish` waits for. Not \[48\]
    last_submission: Option<wgpu::SubmissionIndex>,
    /// How many voices one submission may cover, and what the last block did \[49\]
    budget: SubmitBudget,
    in_flight: Option<InFlight>,
    /// Blocks whose render pass went up as more than one submission.
    split_blocks: u64,
    /// Other renders are submitting to this device at the same time; see \[50\]
    concurrent: bool,
    /// PCI vendor and device id of the adapter, which is how `vram` finds it.
    adapter_ids: (u32, u32),
}

/// The block on the device, as `finish` needs it to tell the budget how long \[51\]
struct InFlight {
    /// When its first submission went in.
    at: std::time::Instant,
    voices: u32,
    parts: u32,
}

/// Everything GPU renders in one process can share: the device, and the sample \[52\]
#[derive(Clone)]
pub struct GpuShared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    has_timestamps: bool,
    /// The sample pool, in one buffer unless it is bigger than one binding; \[53\]
    pool_bufs: Vec<wgpu::Buffer>,
    pool_parts: PoolParts,
    /// Words in all of `pool_bufs` together, for the `pool_words` uniform.
    pool_words: u32,
    pool_bytes: u64,
    phase_buf: Option<wgpu::Buffer>,
    concurrent: bool,
}

impl GpuShared {
    /// Say that the renders built on this will run at the same time, so each \[54\]
    pub fn concurrent(mut self) -> Self {
        self.concurrent = true;
        self
    }

    /// Open the device and upload `bank`'s sample pool, after checking `cfg` \[55\]
    pub fn new(cfg: &Config, bank: &Bank, phase: &crate::phase::PhaseBank) -> Result<Self> {
        cfg.validate()?;
        let (device, queue, adapter_info, limits, has_timestamps) = device::create(cfg)?;
        let adapter_name = format!("{} ({:?})", adapter_info.name, adapter_info.backend);
        let binding_cap = check_limits(cfg, &limits, &adapter_name)?;

        // [56]
        let phase_buf = if cfg.phase.active() {
            if limits.max_storage_buffers_per_shader_stage < 10 {
                bail!("analytic phase requires 10 compute storage buffers; this adapter supports {}",
                    limits.max_storage_buffers_per_shader_stage);
            }
            let bytes = phase.cache_bytes().max(4);
            if bytes > binding_cap {
                bail!("analytic cache needs {:.1} MiB in one GPU binding; {} allows {:.1} MiB",
                    bytes as f64 / 1048576.0, adapter_name, binding_cap as f64 / 1048576.0);
            }
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("analytic quadrature"), size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            upload_in_pieces(&device, &queue, &buf, 0, bytemuck::cast_slice(&phase.words))?;
            Some(buf)
        } else { None };

        // [57]
        let pool = &bank.pool;
        let even = pool.len() / 2 * 2;
        let pool_words = pool.len().div_ceil(2).max(1) as u64;
        let parts = pool_parts(
            pool_words,
            bank.pool_rate,
            binding_cap,
            cfg.pool_part_bytes,
            limits.max_storage_buffers_per_shader_stage,
            &adapter_name,
        )?;
        if parts.count > 1 {
            // Whole MiB, rounded down, as `gpu-info` gives the binding.
            log::info!(
                "the sample pool is {:.1} MiB, so it goes up in {} buffers of at most {} MiB \
                 ({adapter_name} binds {} MiB of one){}",
                pool_words as f64 * 4.0 / 1048576.0,
                parts.count,
                (parts.words_each as u64 * 4) >> 20,
                binding_cap >> 20,
                if cfg.pool_part_bytes > 0 { ", held smaller by --pool-part-mib" } else { "" }
            );
        }
        let part_bytes = parts.words_each as u64 * 4;
        let bytes: &[u8] = bytemuck::cast_slice(&pool[..even]);
        // [58]
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let mut pool_bufs = Vec::with_capacity(parts.count as usize);
        let mut uploaded = Ok(());
        for k in 0..parts.count as u64 {
            let start = k * part_bytes;
            let size = part_bytes.min(pool_words * 4 - start);
            let label = if parts.count == 1 { "sample pool".to_string() } else { format!("sample pool, part {}", k + 1) };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // Both ends are whole words: a part is, and so is the even length.
            let (lo, hi) = ((start as usize).min(bytes.len()), ((start + size) as usize).min(bytes.len()));
            if uploaded.is_ok() {
                uploaded = upload_in_pieces(&device, &queue, &buf, 0, &bytes[lo..hi]);
            }
            pool_bufs.push(buf);
        }
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            bail!(
                "this soundfont's samples take {:.0} MiB of video memory, and {adapter_name} could not \
                 allocate them ({e}). Close other programs that use the GPU, or use a smaller soundfont.",
                pool_words as f64 * 4.0 / 1048576.0
            );
        }
        uploaded?;
        if even < pool.len() {
            // [59]
            let at = even as u64 * 2;
            let k = at / part_bytes;
            queue.write_buffer(&pool_bufs[k as usize], at - k * part_bytes, bytemuck::cast_slice(&[pool[even], 0]));
        }

        Ok(GpuShared {
            device,
            queue,
            adapter_info,
            limits,
            has_timestamps,
            pool_bufs,
            pool_parts: parts,
            pool_words: pool_words as u32,
            pool_bytes: pool.len() as u64 * 2,
            phase_buf,
            concurrent: false,
        })
    }

    pub fn adapter_name(&self) -> String {
        format!("{} ({:?})", self.adapter_info.name, self.adapter_info.backend)
    }

    /// PCI vendor and device id, for `vram::sample`.
    pub fn adapter_ids(&self) -> (u32, u32) {
        (self.adapter_info.vendor, self.adapter_info.device)
    }

    /// Device bytes held here, once, however many renders share them.
    pub fn bytes(&self) -> u64 {
        self.pool_bytes + self.phase_buf.as_ref().map_or(0, |b| b.size())
    }

    /// The most voices one render on this device can be given: the largest \[60\]
    pub fn max_voices(&self, cfg: &Config) -> u32 {
        max_voices_for_config(self.binding_bytes(), cfg)
    }

    /// The most of one buffer the adapter binds to a shader.
    pub fn binding_bytes(&self) -> u64 {
        (self.limits.max_storage_buffer_binding_size as u64).min(self.limits.max_buffer_size)
    }

    /// How the sample pool was laid out on the device.
    pub fn pool_parts(&self) -> PoolParts {
        self.pool_parts
    }

    /// The pool's first buffer, which the render pass binds at 1, and the \[61\]
    fn pool_split(&self) -> (&wgpu::Buffer, &[wgpu::Buffer]) {
        (&self.pool_bufs[0], &self.pool_bufs[1..])
    }
}

/// Read several readback buffers back with one device wait: `GpuSynth`'s and \[62\]
fn map_read(
device: &wgpu::Device,
bufs: &[&wgpu::Buffer],
concurrent: bool,
last_submission: Option<wgpu::SubmissionIndex>,
) -> Result<Vec<Vec<u8>>> {
    let whole: Vec<(&wgpu::Buffer, u64)> = bufs.iter().map(|b| (*b, b.size())).collect();
    map_read_prefix(device, &whole, concurrent, last_submission)
}

/// `map_read` of the first `len` bytes of each buffer only: a batch reads \[63\]
fn map_read_prefix(
device: &wgpu::Device,
bufs: &[(&wgpu::Buffer, u64)],
concurrent: bool,
last_submission: Option<wgpu::SubmissionIndex>,
) -> Result<Vec<Vec<u8>>> {
    let mut rxs = Vec::with_capacity(bufs.len());
    for &(buf, len) in bufs {
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..len).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rxs.push(rx);
    }
    let mut mapped: Vec<Option<Result<(), wgpu::BufferAsyncError>>> = vec![None; bufs.len()];
    if concurrent {
        // [64]
        let mut spins = 0u32;
        loop {
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| device::lost(Some(device), format_args!("device poll failed: {e:?}")))?;
            for (slot, rx) in mapped.iter_mut().zip(&rxs) {
                if slot.is_none() {
                    match rx.try_recv() {
                        Ok(r) => *slot = Some(r),
                        Err(std::sync::mpsc::TryRecvError::Empty) => {}
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            bail!("readback channel closed")
                        }
                    }
                }
            }
            if mapped.iter().all(Option::is_some) {
                break;
            }
            spins += 1;
            if spins < 64 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }
    } else {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: last_submission.clone(),
                timeout: None,
            })
            .map_err(|e| device::lost(Some(device), format_args!("device poll failed: {e:?}")))?;
        for (slot, rx) in mapped.iter_mut().zip(&rxs) {
            *slot = Some(rx.recv().context("readback channel closed")?);
        }
    }
    let mut out = Vec::with_capacity(bufs.len());
    for (&(buf, len), r) in bufs.iter().zip(mapped) {
        r.expect("every read was waited for")
            .map_err(|e| device::lost(Some(device), format_args!("buffer map failed: {e:?}")))?;
        // The view borrows the buffer, so it has to be gone before unmap.
        out.push(buf.slice(..len).get_mapped_range().to_vec());
        buf.unmap();
    }
    Ok(out)
}

/// How one block's spawn list fits the pool. Shared by `GpuSynth` and the \[65\]
struct SpawnPlan {
    steal_k: u32,
    spawn_count: u32,
    /// Note-ons refused this block, over the pool or over the steal bound.
    dropped: u64,
    /// The commands to upload are `scratch`'s, thinned from `pending`, rather \[66\]
    thinned: bool,
}

fn plan_spawns(cfg: &Config, live: u32, pending: &[SpawnCmd], scratch: &mut Vec<SpawnCmd>) -> SpawnPlan {
    let cap = cfg.max_voices;
    let mut dropped = 0u64;

    // [67]
    let want = (pending.len() as u32).min(cap);
    if want < pending.len() as u32 {
        dropped += pending.len() as u64 - want as u64;
    }

    let mut steal_k = 0u32;
    if live + want > cap {
        match cfg.steal_rule {
            // [68]
            StealRule::Oldest | StealRule::Quietest => {
                steal_k = (live + want - cap).min(live).min(cfg.max_steal())
            }
            StealRule::DropNew => {}
        }
    }
    let spawn_count = want.min(cap - (live - steal_k));
    dropped += (want - spawn_count) as u64;

    let total = pending.len();
    let take = spawn_count as usize;
    let thinned = spawn_count > 0 && take != total;
    if thinned {
        // [69]
        scratch.clear();
        scratch.reserve(take);
        match cfg.admit_rule {
            AdmitRule::Loudest => scratch.extend_from_slice(&pending[..take]),
            AdmitRule::Even => {
                for i in 0..take {
                    scratch.push(pending[spawn_pick(i, total, take)]);
                }
            }
        }
    }
    SpawnPlan { steal_k, spawn_count, dropped, thinned }
}

/// The read-only tables the render pass reads beside the params, as one \[70\]
fn read_only_tables(cfg: &Config, bank: &Bank) -> (Vec<u32>, u32, u32) {
    let mut t: Vec<u32> = if bank.menv_factors.is_empty() { vec![1 << 24] } else { bank.menv_factors.clone() };
    let log2_base = t.len() as u32;
    if bank.menv_log2.is_empty() {
        t.push(0);
    } else {
        t.extend_from_slice(&bank.menv_log2);
    }
    let glide_base = t.len() as u32;
    t.extend_from_slice(&crate::porta::tables(cfg.sample_rate));
    (t, log2_base, glide_base)
}

/// Workgroups for a grid-strided pass over `items`: one per `workgroup_size` \[71\]
fn dispatch_count(cfg: &Config, items: u32) -> u32 {
    let ceiling = cfg.max_pool_workgroups.clamp(1, MAX_WORKGROUPS_PER_DIM);
    items.div_ceil(cfg.workgroup_size).clamp(1, ceiling)
}

/// What a render's uniforms hold that does not change from block to block.
#[derive(Debug, Clone, Copy)]
struct Shape {
    slots: u32,
    pool_words: u32,
    sort_key: SortKeyLayout,
    params_per_variant: u32,
    menv_factor_half: u32,
    menv_log2_base: u32,
    glide_base: u32,
}

/// What a render's uniforms hold for one block.
#[derive(Debug, Clone, Copy, Default)]
struct BlockU {
    spawn_count: u32,
    steal_k: u32,
    nwg: u32,
    bend: bool,
    gain: bool,
    variant: bool,
    cut: bool,
    env_phase: u32,
    chan_count: u32,
    off_meta_words: u32,
}

/// One block's uniforms. Shared by `GpuSynth` and the batch, so the two cannot \[72\]
fn make_uniforms(cfg: &Config, s: &Shape, b: &BlockU) -> Uniforms {
    Uniforms {
        block_frames: cfg.block_frames,
        tiles: cfg.block_frames / cfg.reduce_tile,
        // [73]
        capacity: s.slots,
        spawn_count: b.spawn_count,
        render_workgroups: b.nwg,
        interp: cfg.interpolation as u32,
        exp_decay: (cfg.decay_curve == EnvelopeCurve::Exponential) as u32,
        exp_release: (cfg.release_curve == EnvelopeCurve::Exponential) as u32,
        env_floor: cfg.env_floor,
        pool_words: s.pool_words,
        steal_k: b.steal_k,
        sort_bits: s.sort_key.bits,
        sort_region_shift: s.sort_key.region_shift,
        sort_stage_shift: s.sort_key.stage_shift,
        sort_phase_shift: s.sort_key.phase_shift,
        sort_phase_mask: s.sort_key.phase_mask,
        sort_dead_region: s.sort_key.dead_region,
        chan_active: (b.bend as u32) | ((b.gain as u32) << 1) | ((b.variant as u32) << 2) | ((b.cut as u32) << 3),
        params_per_variant: s.params_per_variant,
        steal_by_level: (cfg.steal_rule == StealRule::Quietest) as u32,
        menv_factor_half: s.menv_factor_half,
        menv_log2_base: s.menv_log2_base,
        glide_base: s.glide_base,
        env_phase: b.env_phase,
        chan_count: b.chan_count,
        off_meta_words: b.off_meta_words,
        render_wg_base: 0,
        zero: 0,
    }
}

/// Where `Uniforms::render_wg_base` sits, for the write between the parts of \[74\]
const RENDER_WG_BASE: u64 = std::mem::offset_of!(Uniforms, render_wg_base) as u64;

/// Submissions a block's render pass goes up as: enough that none covers \[75\]
fn render_parts(voices: u32, nwg: u32, budget: u32) -> u32 {
    voices.div_ceil(budget.max(1)).clamp(1, nwg.max(1))
}

/// Render workgroups a render can ever dispatch, which sizes its partials \[76\]
fn max_render_workgroups(cfg: &Config) -> u32 {
    cfg.max_render_workgroups
        .clamp(1, MAX_WORKGROUPS_PER_DIM)
        .min(cfg.pool_slots().div_ceil(cfg.workgroup_size).max(1))
}

/// The most device buffers the sample pool is split across. \[77\]
pub const POOL_PARTS_MAX: u32 = 4;

/// Storage buffers the fullest render pass binds besides the pool's extra \[78\]
const RENDER_STORAGE_BUFFERS: u32 = 12;

/// The sample pool as it sits on the device: `count` buffers of `words_each` \[79\]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolParts {
    pub words_each: u32,
    pub count: u32,
}

/// Lay out a pool of `pool_words` words in buffers the adapter can bind, or \[80\]
fn pool_parts(
    pool_words: u64,
    pool_rate: u32,
    binding_cap: u64,
    part_bytes: u64,
    storage_buffers: u32,
    adapter_name: &str,
) -> Result<PoolParts> {
    let mut cap_words = (binding_cap / 4).min(u32::MAX as u64);
    if part_bytes > 0 {
        cap_words = cap_words.min((part_bytes / 4).max(1));
    }
    let pool_words = pool_words.max(1);
    let count = pool_words.div_ceil(cap_words);
    if count == 1 {
        return Ok(PoolParts { words_each: pool_words as u32, count: 1 });
    }
    const MIB: f64 = 1048576.0;
    let pool_bytes = pool_words * 4;
    let most = POOL_PARTS_MAX as u64 * cap_words * 4;
    if count > POOL_PARTS_MAX as u64 {
        let at = |rate: u32| pool_bytes as f64 * rate as f64 / pool_rate as f64;
        let fits = [44_100u32, 32_000, 24_000, 22_050, 16_000]
            .into_iter()
            .find(|&r| pool_rate != 0 && r < pool_rate && at(r) * 1.01 <= most as f64);
        let rate = if pool_rate == 0 { String::new() } else { format!(" at {pool_rate} Hz") };
        let advice = match fits {
            Some(r) => format!(
                " Kestrel keeps the samples at the output rate, so --rate {r} would bring them to about {:.0} MiB, \
                 at the cost of a lower-rate render.",
                at(r) / MIB
            ),
            None => " It needs a smaller soundfont.".to_string(),
        };
        bail!(
            "this soundfont's samples come to {:.0} MiB{rate}, and Kestrel can hold at most {} MiB of samples on \
             {adapter_name}: {POOL_PARTS_MAX} buffers of {} MiB, the most it binds to a shader.{advice}",
            pool_bytes as f64 / MIB,
            // Whole MiB, rounded down, as `gpu-info` gives the binding.
            most >> 20,
            (cap_words * 4) >> 20
        );
    }
    let count = count as u32;
    let need = RENDER_STORAGE_BUFFERS + count - 1;
    if need > storage_buffers {
        bail!(
            "this soundfont's samples come to {:.0} MiB, which Kestrel splits across {count} buffers on \
             {adapter_name}, and a render pass would then bind {need} storage buffers where it allows \
             {storage_buffers}. It needs a smaller soundfont on this GPU.",
            pool_bytes as f64 / MIB
        );
    }
    Ok(PoolParts { words_each: cap_words as u32, count })
}

/// The sample-pool budget until 1.2.3, and still the least `auto_pool_budget` \[81\]
pub const POOL_BUDGET_FLOOR: u64 = 2 << 30;

/// Video memory `pool_budget_for` keeps for everything but the pool: a \[82\]
const POOL_BUDGET_RESERVE: u64 = 3 << 29;

/// `--pool-budget` when it is not given, decided with the user 2026-09-27: \[83\]
pub fn auto_pool_budget(cfg: &Config) -> u64 {
    type Key = (Option<String>, Option<String>);
    static CACHE: std::sync::Mutex<Vec<(Key, u64)>> = std::sync::Mutex::new(Vec::new());
    let key = (cfg.gpu_backend.clone(), cfg.gpu_adapter.clone());
    let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, budget)) = cache.iter().find(|(k, _)| *k == key) {
        return *budget;
    }
    let budget = match device::pick_adapter(cfg) {
        Ok(adapter) => {
            let (info, limits) = (adapter.get_info(), adapter.limits());
            let binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
            let total = vram::dedicated_total(info.vendor, info.device);
            let budget = pool_budget_for(total, binding);
            log::debug!(
                "pool budget {} MiB, from {}'s {} of video memory",
                budget >> 20,
                info.name,
                total.map_or("unknown amount".to_string(), |t| format!("{} MiB", t >> 20))
            );
            budget
        }
        Err(_) => POOL_BUDGET_FLOOR,
    };
    cache.push((key, budget));
    budget
}

/// The pool budget for a card with `total` bytes of dedicated memory that \[84\]
fn pool_budget_for(total: Option<u64>, binding: u64) -> u64 {
    let ceiling = (POOL_PARTS_MAX as u64 * (binding / 4) * 4).clamp(POOL_BUDGET_FLOOR, 8 << 30);
    match total {
        Some(t) => t.saturating_sub((t / 4).max(POOL_BUDGET_RESERVE)).clamp(POOL_BUDGET_FLOOR, ceiling),
        None => POOL_BUDGET_FLOOR,
    }
}

/// Refuse a configuration the adapter cannot run, and return the most of one \[85\]
struct MemoryNote {
    text: String,
    /// The buffers need more than the process has left of its budget.
    short: bool,
}

/// Say what `needs` bytes of new buffers come to against `memory`'s budget, and \[86\]
fn memory_note(needs: u64, scaled: u64, max_voices: u32, memory: &vram::GpuMemory) -> Option<MemoryNote> {
    let (budget, used) = (memory.process_budget?, memory.process_used?);
    let free = budget.saturating_sub(used);
    let mib = |b: u64| b >> 20;
    let mut text = format!(
        "this render's buffers need about {} MiB more, and the OS lets this process use {} MiB of this \
         adapter's memory, {} MiB of it already in use",
        mib(needs),
        mib(budget),
        mib(used)
    );
    let short = needs > free;
    if short {
        let fixed = needs - scaled.min(needs);
        if free > fixed && scaled > 0 {
            let fits = (max_voices as u128 * (free - fixed) as u128 / scaled as u128) as u64;
            // Two significant digits, rounded down: a figure to try, not a promise.
            let digits = fits.max(1).ilog10().saturating_sub(1);
            let step = 10u64.pow(digits);
            text.push_str(&format!(
                ". That is more than it has left, so the render may stop with the device lost or run \
                 very slowly; --max-voices {} is about what fits",
                fits / step * step
            ));
        } else {
            text.push_str(". That is more than it has left even with no voices, so the render may stop with the device lost");
        }
    }
    Some(MemoryNote { text, short })
}

fn check_limits(cfg: &Config, limits: &wgpu::Limits, adapter_name: &str) -> Result<u64> {
    // [87]
    let need_shared = cfg.workgroup_size * (cfg.reduce_tile * 2 + 1) * 4;
    if need_shared > limits.max_compute_workgroup_storage_size {
        bail!(
            "the render pass needs {} bytes of workgroup storage but {} only offers {}; \
             lower --block or the reduce tile",
            need_shared,
            adapter_name,
            limits.max_compute_workgroup_storage_size
        );
    }

    // [88]
    let voice_pool_bytes = cfg.pool_slots() as u64 * voice_fields(cfg) * 4;
    let binding_cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    if voice_pool_bytes > binding_cap {
        let voice_cap = max_voices_for_config(binding_cap, cfg);
        bail!(
            "max_voices {} allocates {} pool slots, a {:.2} GiB voice buffer, and \
             {} binds at most {:.2} GiB of one buffer to a shader; cap \
             --max-voices at {} (the extra {} slots are the --steal-percent {} \
             fade headroom)",
            cfg.max_voices,
            cfg.pool_slots(),
            voice_pool_bytes as f64 / (1u64 << 30) as f64,
            adapter_name,
            binding_cap as f64 / (1u64 << 30) as f64,
            voice_cap,
            cfg.max_steal(),
            cfg.max_steal_percent
        );
    }
    if cfg.block_frames * 2 > MAX_WORKGROUPS_PER_DIM {
        bail!(
            "block_frames {} needs {} reduce workgroups, over the {} limit",
            cfg.block_frames,
            cfg.block_frames * 2,
            MAX_WORKGROUPS_PER_DIM
        );
    }
    Ok(binding_cap)
}

struct Timing {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
}

/// Pass groups that get their own timestamp pair.
const PASS_NAMES: [&str; 5] = ["steal", "spawn", "render", "reduce", "compact"];

const TIMESTAMP_COUNT: u32 = PASS_NAMES.len() as u32 * 2;

impl GpuSynth {
    pub fn new(cfg: &Config, bank: Arc<Bank>) -> Result<Self> {
        let phase = crate::phase::PhaseBank::prepare(&bank, &cfg.phase)?;
        Self::new_prepared(cfg, bank, phase)
    }

    pub(crate) fn new_prepared(cfg: &Config, bank: Arc<Bank>,
        phase: Arc<crate::phase::PhaseBank>) -> Result<Self> {
        cfg.validate()?;
        let shared = GpuShared::new(cfg, &bank, &phase)?;
        let s = Self::on_shared(cfg, bank, &shared)?;
        s.log_buffers();
        Ok(s)
    }

    /// A render's own buffers and pipelines, on a device and sample pool that \[89\]
    pub fn on_shared(cfg: &Config, bank: Arc<Bank>, shared: &GpuShared) -> Result<Self> {
        cfg.validate()?;
        let device = shared.device.clone();
        let queue = shared.queue.clone();
        let adapter_info = shared.adapter_info.clone();
        let has_timestamps = shared.has_timestamps;
        let adapter_name = shared.adapter_name();
        let binding_cap = check_limits(cfg, &shared.limits, &adapter_name)?;
        if shared.phase_buf.is_some() != cfg.phase.active() {
            bail!("the shared device was prepared for a different analytic phase setting");
        }

        // [90]
        let scan_workgroups = cfg.pool_slots().div_ceil(cfg.workgroup_size);

        // [91]
        let capacity = cfg.pool_slots();
        let tiles = cfg.block_frames / cfg.gate_frames;
        let nwg = max_render_workgroups(cfg);

        let (pool_first, pool_rest) = shared.pool_split();
        let (pool_buf, pool_rest) = (pool_first.clone(), pool_rest.to_vec());
        let (pool_parts, pool_words) = (shared.pool_parts, shared.pool_words);
        let phase_buf = shared.phase_buf.clone();

        // [92]
        let fallback;
        let params: &[RegionParams] = if bank.params.is_empty() {
            fallback = [RegionParams {
                mod_lfo_inc: 0,
                vib_lfo_inc: 0,
                lfo_delays: 0,
                lfo_pitch: 0,
                attack_rate: 1.0,
                attack_end: 1.0,
                decay_coef: 1.0,
                decay_target: 1.0,
                sustain: 1.0,
                release_coef: 0.0,
                b0: 1.0,
                b1: 0.0,
                a1: 0.0,
                a2: 0.0,
                flags: 0,
            }];
            &fallback
        } else {
            &bank.params
        };
        // [93]
        let params_per_variant = params.len() as u32;
        let variants = cfg.max_param_variants.max(1);
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("region params"),
            size: (params.len() * variants as usize * std::mem::size_of::<RegionParams>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        upload_in_pieces(&device, &queue, &params_buf, 0, bytemuck::cast_slice(params))?;

        // [94]
        let menv: Vec<ModEnvParams> = if bank.menv.is_empty() {
            vec![ModEnvParams::default()]
        } else {
            bank.menv.clone()
        };
        let menv_per_variant = menv.len() as u32;
        let menv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mod env params"),
            size: (menv.len() * variants as usize * std::mem::size_of::<ModEnvParams>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&menv_buf, 0, bytemuck::cast_slice(&menv));

        // [95]
        let (menv_tables, menv_log2_base, glide_base) = read_only_tables(cfg, &bank);
        let menv_factor_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mod env tables"),
            contents: bytemuck::cast_slice(&menv_tables),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ---- per-block data ----
        let storage = wgpu::BufferUsages::STORAGE;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let voice_bytes = capacity as u64 * voice_fields(cfg) * 4;
        let partial_bytes = cfg.block_frames as u64 * 2 * nwg as u64 * 4;
        let out_bytes = cfg.block_frames as u64 * 2 * 4;
        // [96]
        let off_meta_words = (BASE_CHANNELS as u64 * 128 + 1) * 2;
        let off_runs_capacity = 32768u64;
        let gates_bytes = (off_meta_words + off_runs_capacity * 2) * 4;

        // [97]
        let scaled = voice_bytes * 2 + capacity as u64 * 24; // the voice pools, the sort and the scan
        let needs = scaled
            + params_buf.size()
            + menv_buf.size()
            + menv_factor_buf.size()
            + partial_bytes
            + out_bytes
            + gates_bytes
            + (tiles as u64 + 1) * BASE_CHANNELS as u64 * CHAN_FIELDS as u64 * 4
            + 65536u32.min(capacity).max(1024) as u64 * std::mem::size_of::<SpawnCmd>() as u64;
        if let Some(m) = vram::sample_quick(adapter_info.vendor, adapter_info.device) {
            if let Some(note) = memory_note(needs, scaled, cfg.max_voices, &m) {
                if note.short {
                    log::warn!("{}", note.text);
                } else {
                    log::info!("{}", note.text);
                }
            }
        }

        let uniform_buf = mk(
            "uniforms",
            std::mem::size_of::<Uniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let gates_buf = mk("gates", gates_bytes, storage | wgpu::BufferUsages::COPY_DST);
        // [98]
        let chan_bytes = (tiles as u64 + 1) * BASE_CHANNELS as u64 * CHAN_FIELDS as u64 * 4;
        let chan_buf = mk("channels", chan_bytes, storage | wgpu::BufferUsages::COPY_DST);
        // [99]
        let voice_usage = storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let voices = [mk("voices a", voice_bytes, voice_usage), mk("voices b", voice_bytes, voice_usage)];
        let partials_buf = mk("partials", partial_bytes, storage);
        let out_buf = mk("out block", out_bytes, storage | wgpu::BufferUsages::COPY_SRC);
        let state_buf = mk(
            "state",
            STATE_SLOTS as u64 * 4,
            storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let scan_buf = mk("scan", capacity as u64 * 4, storage);
        let block_sums_buf = mk("block sums", scan_workgroups as u64 * 4, storage);
        let hist_buf = mk("select histogram", 256 * 4, storage);
        let sort_keys_buf = mk("sort keys", capacity as u64 * 4, storage);
        let sort_key = SortKeyLayout::plan(&bank);
        let pairs = [
            mk("sort pairs a", capacity as u64 * 8, storage),
            mk("sort pairs b", capacity as u64 * 8, storage),
        ];
        log::debug!(
            "sort key: {} bits, region<<{} stage<<{} phase>>{} mask {:#x}",
            sort_key.bits,
            sort_key.region_shift,
            sort_key.stage_shift,
            sort_key.phase_shift,
            sort_key.phase_mask
        );

        let cmds_capacity = 65536u32.min(capacity).max(1024);
        let cmds_buf = mk(
            "spawn commands",
            cmds_capacity as u64 * std::mem::size_of::<SpawnCmd>() as u64,
            storage | wgpu::BufferUsages::COPY_DST,
        );

        let readback_out = mk(
            "readback out",
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let readback_state = mk(
            "readback state",
            STATE_SLOTS as u64 * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let shared_bytes = shared.pool_bytes + phase_buf.as_ref().map_or(0, |b| b.size());
        let vram_bytes = shared_bytes
            + params_buf.size()
            + menv_buf.size()
            + menv_factor_buf.size()
            + voice_bytes * 2
            + partial_bytes
            + out_bytes
            + gates_bytes
            + chan_bytes
            + capacity as u64 * 8  // scan + sort keys
            + capacity as u64 * 16 // sort pairs, double buffered
            + cmds_capacity as u64 * std::mem::size_of::<SpawnCmd>() as u64;

        // ---- layouts, pipelines, bind groups ----
        let mut render_bindings = vec![false, true, true, true, false, false, true, true, true, true];
        if phase_buf.is_some() { render_bindings.push(true); }
        let mut render_entries = device::layout_entries(&render_bindings);
        render_entries.extend((0..pool_rest.len() as u32).map(|k| device::storage_entry(POOL_PART_BINDING + k, true)));
        let layouts = Layouts {
            spawn: device::bind_layout(&device, "spawn", &[false, true, false, false]),
            render: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render"),
                entries: &render_entries,
            }),
            reduce: device::bind_layout(&device, "reduce", &[false, true, false]),
            compact: device::bind_layout(
                &device,
                "compact",
                &[false, false, false, false, false, false, false],
            ),
            select: device::bind_layout(&device, "select", &[false, true, false, false]),
            sort: device::bind_layout(&device, "sort", &[false; 8]),
        };

        let pipelines = Self::build_pipelines(&device, cfg, &bank, pool_parts, &layouts)?;
        let glide_sources =
            [render_source(cfg, &bank, pool_parts, false, true, false), render_source(cfg, &bank, pool_parts, true, true, false)];
        let split_sources = std::array::from_fn(|i| render_source(cfg, &bank, pool_parts, i & 1 != 0, i & 2 != 0, true));

        let groups = [
            Groups {
                spawn: device::bind(
                    &device,
                    &layouts.spawn,
                    &[&uniform_buf, &cmds_buf, &voices[0], &state_buf],
                ),
                render: bind_render(
                    &device,
                    &layouts.render,
                    &[
                        &uniform_buf,
                        &pool_buf,
                        &params_buf,
                        &gates_buf,
                        &voices[0],
                        &partials_buf,
                        &state_buf,
                        &chan_buf,
                        &menv_buf,
                        &menv_factor_buf,
                    ],
                    phase_buf.as_ref(),
                    &pool_rest,
                ),
                compact: device::bind(
                    &device,
                    &layouts.compact,
                    &[
                        &uniform_buf,
                        &voices[0],
                        &voices[1],
                        &scan_buf,
                        &block_sums_buf,
                        &state_buf,
                        &sort_keys_buf,
                    ],
                ),
                select: device::bind(
                    &device,
                    &layouts.select,
                    &[&uniform_buf, &voices[0], &state_buf, &hist_buf],
                ),
                sort: [
                    device::bind(
                        &device,
                        &layouts.sort,
                        &[
                            &uniform_buf,
                            &pairs[0],
                            &pairs[1],
                            &scan_buf,
                            &block_sums_buf,
                            &state_buf,
                            &voices[0],
                            &voices[1],
                        ],
                    ),
                    device::bind(
                        &device,
                        &layouts.sort,
                        &[
                            &uniform_buf,
                            &pairs[1],
                            &pairs[0],
                            &scan_buf,
                            &block_sums_buf,
                            &state_buf,
                            &voices[0],
                            &voices[1],
                        ],
                    ),
                ],
            },
            Groups {
                spawn: device::bind(
                    &device,
                    &layouts.spawn,
                    &[&uniform_buf, &cmds_buf, &voices[1], &state_buf],
                ),
                render: bind_render(
                    &device,
                    &layouts.render,
                    &[
                        &uniform_buf,
                        &pool_buf,
                        &params_buf,
                        &gates_buf,
                        &voices[1],
                        &partials_buf,
                        &state_buf,
                        &chan_buf,
                        &menv_buf,
                        &menv_factor_buf,
                    ],
                    phase_buf.as_ref(),
                    &pool_rest,
                ),
                compact: device::bind(
                    &device,
                    &layouts.compact,
                    &[
                        &uniform_buf,
                        &voices[1],
                        &voices[0],
                        &scan_buf,
                        &block_sums_buf,
                        &state_buf,
                        &sort_keys_buf,
                    ],
                ),
                select: device::bind(
                    &device,
                    &layouts.select,
                    &[&uniform_buf, &voices[1], &state_buf, &hist_buf],
                ),
                sort: [
                    device::bind(
                        &device,
                        &layouts.sort,
                        &[
                            &uniform_buf,
                            &pairs[0],
                            &pairs[1],
                            &scan_buf,
                            &block_sums_buf,
                            &state_buf,
                            &voices[1],
                            &voices[0],
                        ],
                    ),
                    device::bind(
                        &device,
                        &layouts.sort,
                        &[
                            &uniform_buf,
                            &pairs[1],
                            &pairs[0],
                            &scan_buf,
                            &block_sums_buf,
                            &state_buf,
                            &voices[1],
                            &voices[0],
                        ],
                    ),
                ],
            },
        ];
        let reduce_group = device::bind(
            &device,
            &layouts.reduce,
            &[&uniform_buf, &partials_buf, &out_buf],
        );

        let timing = if cfg.profile && has_timestamps {
            let set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timings"),
                ty: wgpu::QueryType::Timestamp,
                count: TIMESTAMP_COUNT,
            });
            Some(Timing {
                set,
                resolve: mk(
                    "timestamp resolve",
                    TIMESTAMP_COUNT as u64 * 8,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                ),
                readback: mk(
                    "timestamp readback",
                    TIMESTAMP_COUNT as u64 * 8,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                ),
                period_ns: queue.get_timestamp_period(),
            })
        } else {
            if cfg.profile && !has_timestamps {
                log::warn!("--profile asked for pass timings but the adapter has no timestamp queries");
            }
            None
        };

        // The state buffer starts zeroed, which is exactly "no voices yet".
        queue.write_buffer(&state_buf, 0, bytemuck::cast_slice(&[0u32; STATE_SLOTS]));

        let s = GpuSynth {
            cfg: cfg.clone(),
            device,
            queue,
            adapter_name,
            pipelines,
            groups,
            reduce_group,
            layouts,
            uniform_buf,
            gates_buf,
            chan_buf,
            bend_active: false,
            gain_active: false,
            variant_active: false,
            cut_active: false,
            voices,
            partials_buf,
            out_buf,
            state_buf,
            scan_buf,
            block_sums_buf,
            hist_buf,
            sort_keys_buf,
            pairs,
            sort_key,
            cmds_buf,
            cmds_capacity,
            off_runs_capacity,
            off_meta_words,
            chan_count: BASE_CHANNELS as u32,
            binding_cap,
            slots: capacity,
            pool_buf,
            pool_rest,
            pool_parts,
            pool_words,
            phase_buf,
            params_buf,
            params_per_variant,
            menv_buf,
            menv_factor_buf,
            menv_per_variant,
            menv_factor_half: bank.menv_factor_half,
            menv_log2_base,
            glide_base,
            glide_active: false,
            render_glide: None,
            glide_sources,
            render_split: [None, None, None, None],
            split_sources,
            readback_out,
            readback_state,
            parity: 0,
            live: 0,
            env_phase: 0,
            planned: (0, 0),
            spawn_scratch: Vec::new(),
            stolen: 0,
            dropped: 0,
            peak: 0.0,
            timing,
            last_timings: Vec::new(),
            vram_bytes,
            shared_bytes,
            pool_bytes: shared.pool_bytes,
            voice_bytes,
            partial_bytes,
            max_nwg: nwg,
            last_submission: None,
            budget: SubmitBudget::new(cfg.submit_voices),
            in_flight: None,
            split_blocks: 0,
            concurrent: shared.concurrent,
            adapter_ids: (adapter_info.vendor, adapter_info.device),
        };
        s.write_uniforms(0, 0, s.render_workgroups(0));
        Ok(s)
    }

    /// The `gpu:` line a render has always opened with.
    fn log_buffers(&self) {
        log::info!(
            "gpu: {} | {:.1} MiB of device buffers ({:.1} MiB sample pool, \
             {:.1} MiB voice pool for {} voices, {:.1} MiB partials)",
            self.adapter_name,
            self.vram_bytes as f64 / 1048576.0,
            self.pool_bytes as f64 / 1048576.0,
            self.voice_bytes as f64 * 2.0 / 1048576.0,
            self.slots,
            self.partial_bytes as f64 / 1048576.0,
        );
    }

    /// Device bytes this render allocated for itself, apart from what it \[100\]
    pub fn own_bytes(&self) -> u64 {
        self.vram_bytes - self.shared_bytes
    }

    /// Make this a fresh backend again, for the next render on the same \[101\]
    pub fn reset(&mut self) {
        self.queue
            .write_buffer(&self.state_buf, 0, bytemuck::cast_slice(&[0u32; STATE_SLOTS]));
        self.parity = 0;
        self.live = 0;
        self.env_phase = 0;
        self.stolen = 0;
        self.dropped = 0;
        self.peak = 0.0;
        self.planned = (0, 0);
        self.bend_active = false;
        self.gain_active = false;
        self.variant_active = false;
        self.cut_active = false;
        self.glide_active = false;
        self.last_timings.clear();
        self.write_uniforms(0, 0, self.render_workgroups(0));
    }

    fn build_pipelines(
        device: &wgpu::Device,
        cfg: &Config,
        bank: &Bank,
        parts: PoolParts,
        layouts: &Layouts,
    ) -> Result<Pipelines> {
        let make = |name: &str, src: &str, layout: &wgpu::BindGroupLayout, entries: &[&str]| {
            compile(device, cfg, name, shader_source(src, cfg, bank), layout, entries)
        };

        let mut spawn = make(
            "spawn",
            include_str!("../../shaders/spawn.wgsl"),
            &layouts.spawn,
            &["main", "commit"],
        );
        let mut render_chan = compile(
            device,
            cfg,
            "render_chan",
            render_source(cfg, bank, parts, true, false, false),
            &layouts.render,
            &["main"],
        );
        let mut render = compile(
            device,
            cfg,
            "render",
            render_source(cfg, bank, parts, false, false, false),
            &layouts.render,
            &["main"],
        );
        let mut reduce = make(
            "reduce",
            include_str!("../../shaders/reduce.wgsl"),
            &layouts.reduce,
            &["main", "thin"],
        );
        let mut compact = make(
            "compact",
            include_str!("../../shaders/compact.wgsl"),
            &layouts.compact,
            &[
                "scan_local",
                "scan_blocks",
                "scatter",
                "commit",
                "mark_stolen",
                "note_stolen",
            ],
        );
        let mut select = make(
            "select",
            include_str!("../../shaders/select.wgsl"),
            &layouts.select,
            &["clear", "init", "histogram", "refine"],
        );
        let mut sort = make(
            "sort",
            include_str!("../../shaders/sort.wgsl"),
            &layouts.sort,
            &[
                "init",
                "advance_bit",
                "build_keys",
                "scan_local",
                "scan_blocks",
                "split",
                "gather",
            ],
        );

        Ok(Pipelines {
            spawn_commit: spawn.remove(1),
            spawn: spawn.remove(0),
            render: render.remove(0),
            render_chan: render_chan.remove(0),
            reduce_thin: reduce.remove(1),
            reduce: reduce.remove(0),
            note_stolen: compact.remove(5),
            mark_stolen: compact.remove(4),
            compact_commit: compact.remove(3),
            scatter: compact.remove(2),
            scan_blocks: compact.remove(1),
            scan_local: compact.remove(0),
            sel_refine: select.remove(3),
            sel_histogram: select.remove(2),
            sel_init: select.remove(1),
            sel_clear: select.remove(0),
            sort_gather: sort.remove(6),
            sort_split: sort.remove(5),
            sort_scan_blocks: sort.remove(4),
            sort_scan_local: sort.remove(3),
            sort_build_keys: sort.remove(2),
            sort_advance: sort.remove(1),
            sort_init: sort.remove(0),
        })
    }

    /// Workgroups the render pass will be dispatched with this block, given \[102\]
    fn render_workgroups(&self, voices: u32) -> u32 {
        voices.div_ceil(self.cfg.workgroup_size).clamp(1, self.max_nwg)
    }

    /// Compile the render pass that reads `u.render_wg_base`, for the plain or \[103\]
    fn ensure_split(&mut self, chan: bool, glide: bool) {
        let i = ((glide as usize) << 1) | chan as usize;
        if self.render_split[i].is_none() {
            let name = ["render_split", "render_chan_split", "render_glide_split", "render_chan_glide_split"][i];
            let t = std::time::Instant::now();
            let mut p = compile(
                &self.device,
                &self.cfg,
                name,
                self.split_sources[i].clone(),
                &self.layouts.render,
                &["main"],
            );
            self.render_split[i] = Some(p.remove(0));
            log::info!("gpu: compiled the render pass for blocks in parts ({name}) in {:.2} s", t.elapsed().as_secs_f64());
        }
    }

    fn write_uniforms(&self, spawn_count: u32, steal_k: u32, nwg: u32) {
        let shape = Shape {
            slots: self.slots,
            pool_words: self.pool_words(),
            sort_key: self.sort_key,
            params_per_variant: self.params_per_variant,
            menv_factor_half: self.menv_factor_half,
            menv_log2_base: self.menv_log2_base,
            glide_base: self.glide_base,
        };
        let block = BlockU {
            spawn_count,
            steal_k,
            nwg,
            bend: self.bend_active,
            gain: self.gain_active,
            variant: self.variant_active,
            cut: self.cut_active,
            env_phase: self.env_phase,
            chan_count: self.chan_count,
            off_meta_words: self.off_meta_words as u32,
        };
        let u = make_uniforms(&self.cfg, &shape, &block);
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    fn pool_words(&self) -> u32 {
        self.pool_words
    }

    fn grow_cmds(&mut self, needed: u32) {
        if needed <= self.cmds_capacity {
            return;
        }
        let new_cap = needed.next_power_of_two().min(self.cfg.max_voices.max(needed));
        self.cmds_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spawn commands"),
            size: new_cap as u64 * std::mem::size_of::<SpawnCmd>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.cmds_capacity = new_cap;
        // The spawn bind groups reference the old buffer, so rebuild them.
        for p in 0..2 {
            self.groups[p].spawn = device::bind(
                &self.device,
                &self.layouts.spawn,
                &[
                    &self.uniform_buf,
                    &self.cmds_buf,
                    &self.voices[p],
                    &self.state_buf,
                ],
            );
        }
        log::debug!("grew the spawn command buffer to {new_cap} entries");
    }

    /// Grow the gates buffer so it can carry a `meta_words` header and `runs` \[104\]
    fn grow_gates(&mut self, meta_words: u64, runs: usize) -> Result<()> {
        self.off_meta_words = meta_words;
        let runs = runs as u64;
        if meta_words + runs * 2 <= self.gates_buf.size() / 4 {
            return Ok(());
        }
        let new_cap = gates_capacity(runs, self.off_runs_capacity, meta_words, self.binding_cap)?;
        self.gates_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gates"),
            size: (meta_words + new_cap * 2) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.off_runs_capacity = new_cap;
        self.rebind_render();
        log::debug!("grew the gates buffer to {meta_words} meta words and {new_cap} note-off runs");
        Ok(())
    }

    /// Grow the channels buffer to hold `words` of controller rows: a port's \[105\]
    fn grow_channels(&mut self, words: usize) {
        let bytes = words as u64 * 4;
        if bytes <= self.chan_buf.size() {
            return;
        }
        self.chan_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("channels"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.rebind_render();
        log::debug!("grew the channels buffer to {words} words");
    }

    /// Rebuild both render bind groups, after the gates or the channels buffer \[106\]
    fn rebind_render(&mut self) {
        for p in 0..2 {
            self.groups[p].render = bind_render(
                &self.device,
                &self.layouts.render,
                &[
                    &self.uniform_buf,
                    &self.pool_buf,
                    &self.params_buf,
                    &self.gates_buf,
                    &self.voices[p],
                    &self.partials_buf,
                    &self.state_buf,
                    &self.chan_buf,
                    &self.menv_buf,
                    &self.menv_factor_buf,
                ],
                self.phase_buf.as_ref(),
                &self.pool_rest,
            );
        }
    }

    /// Workgroups to dispatch for `items` voices. Every entry point this feeds \[107\]
    fn dispatch_count(&self, items: u32) -> u32 {
        dispatch_count(&self.cfg, items)
    }

    /// Read several readback buffers back with **one** device wait. \[108\]
    fn map_read_many(&self, bufs: &[&wgpu::Buffer]) -> Result<Vec<Vec<u8>>> {
        map_read(&self.device, bufs, self.concurrent, self.last_submission.clone())
    }
}

/// How many note-off runs a regrown gates buffer should hold behind a \[109\]
fn gates_capacity(runs: u64, held: u64, meta_words: u64, binding_cap: u64) -> Result<u64> {
    let bytes = |cap: u64| (meta_words + cap * 2) * 4;
    if bytes(runs) > binding_cap {
        bail!(
            "one block publishes {runs} note-off runs, a {:.2} GiB gate table, and the \
             adapter binds at most {:.2} GiB of one buffer to a shader; lower --block",
            bytes(runs) as f64 / (1u64 << 30) as f64,
            binding_cap as f64 / (1u64 << 30) as f64
        );
    }
    let doubled = runs.max(held).next_power_of_two();
    Ok(if bytes(doubled) > binding_cap {
        (binding_cap / 4 - meta_words) / 2
    } else {
        doubled
    })
}

impl Backend for GpuSynth {
    #[cfg(feature = "dev")]
    fn lose_device(&mut self) {
        self.device.destroy();
    }

    fn set_params_variant(
        &mut self,
        index: u32,
        data: &[RegionParams],
        menv: &[ModEnvParams],
    ) -> Result<()> {
        let per = self.params_per_variant as usize;
        if data.len() != per {
            bail!(
                "params variant {index} has {} entries, expected {per}",
                data.len()
            );
        }
        if index >= self.cfg.max_param_variants.max(1) {
            bail!("params variant {index} is past the configured maximum");
        }
        let off = (index as usize * per * std::mem::size_of::<RegionParams>()) as u64;
        self.queue
            .write_buffer(&self.params_buf, off, bytemuck::cast_slice(data));

        // [110]
        let mper = self.menv_per_variant as usize;
        if menv.len() != mper {
            bail!(
                "mod env variant {index} has {} entries, expected {mper}",
                menv.len()
            );
        }
        let moff = (index as usize * mper * std::mem::size_of::<ModEnvParams>()) as u64;
        self.queue
            .write_buffer(&self.menv_buf, moff, bytemuck::cast_slice(menv));
        Ok(())
    }

    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, variant: bool, cut: bool) -> Result<()> {
        // One row past the last tile; see `voice::ChannelTable::row_bias`.
        let row_count = (self.cfg.block_frames / self.cfg.gate_frames) as usize + 1;
        self.chan_count = (rows.len() / (row_count * CHAN_FIELDS)) as u32;
        self.grow_channels(rows.len());
        // [111]
        self.queue
            .write_buffer(&self.chan_buf, 0, bytemuck::cast_slice(rows));
        self.bend_active = bend;
        self.gain_active = gain;
        self.variant_active = variant;
        self.cut_active = cut;
        Ok(())
    }

    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()> {
        self.grow_gates(meta.len() as u64, runs.len() / 2)?;
        self.queue
            .write_buffer(&self.gates_buf, 0, bytemuck::cast_slice(meta));
        if !runs.is_empty() {
            let off = meta.len() as u64 * 4;
            self.queue
                .write_buffer(&self.gates_buf, off, bytemuck::cast_slice(runs));
        }
        Ok(())
    }

    fn set_glide(&mut self, active: bool) -> Result<()> {
        if active && self.render_glide.is_none() {
            let [plain, chan] = &self.glide_sources;
            let mut a = compile(
                &self.device,
                &self.cfg,
                "render_glide",
                plain.clone(),
                &self.layouts.render,
                &["main"],
            );
            let mut b = compile(
                &self.device,
                &self.cfg,
                "render_chan_glide",
                chan.clone(),
                &self.layouts.render,
                &["main"],
            );
            self.render_glide = Some([a.remove(0), b.remove(0)]);
        }
        self.glide_active = active;
        Ok(())
    }

    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        // [112]
        let plan = plan_spawns(&self.cfg, self.live, cmds, &mut self.spawn_scratch);
        self.dropped += plan.dropped;
        if plan.spawn_count > 0 {
            self.grow_cmds(plan.spawn_count);
            let cmds = if plan.thinned { &self.spawn_scratch } else { cmds };
            self.queue
                .write_buffer(&self.cmds_buf, 0, bytemuck::cast_slice(&cmds[..plan.spawn_count as usize]));
        }
        self.planned = (plan.steal_k, plan.spawn_count);
        Ok(())
    }

    /// Queue the whole block: spawn, render, reduce, compact, and the copies \[113\]
    fn submit(&mut self) -> Result<()> {
        let (steal_k, spawn_count) = std::mem::take(&mut self.planned);
        // [114]
        let nwg = self.render_workgroups(self.live + spawn_count);
        self.write_uniforms(spawn_count, steal_k, nwg);
        // Written, so the next block's position can be taken now.
        self.env_phase = (self.env_phase + self.cfg.block_frames) % self.cfg.env_step_frames();

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("block") });

        let mut pass_index = 0usize;
        macro_rules! begin {
            ($enc:expr, $name:expr) => {{
                let ts = self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites {
                    query_set: &t.set,
                    beginning_of_pass_write_index: Some(pass_index as u32 * 2),
                    end_of_pass_write_index: Some(pass_index as u32 * 2 + 1),
                });
                #[allow(unused_assignments)]
                {
                    pass_index += 1;
                }
                $enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some($name),
                    timestamp_writes: ts,
                })
            }};
        }

        // [115]
        {
            let live_wgs = self.dispatch_count(self.live);
            let mut p = begin!(enc, "steal");
            if steal_k > 0 {
                p.set_bind_group(0, &self.groups[self.parity].select, &[]);
                p.set_pipeline(&self.pipelines.sel_init);
                p.dispatch_workgroups(1, 1, 1);
                for _ in 0..8 {
                    p.set_pipeline(&self.pipelines.sel_clear);
                    p.dispatch_workgroups(1, 1, 1);
                    p.set_pipeline(&self.pipelines.sel_histogram);
                    p.dispatch_workgroups(live_wgs, 1, 1);
                    p.set_pipeline(&self.pipelines.sel_refine);
                    p.dispatch_workgroups(1, 1, 1);
                }
                p.set_bind_group(0, &self.groups[self.parity].compact, &[]);
                p.set_pipeline(&self.pipelines.mark_stolen);
                p.dispatch_workgroups(live_wgs, 1, 1);
                p.set_pipeline(&self.pipelines.note_stolen);
                p.dispatch_workgroups(1, 1, 1);
            }
        }
        // [116]

        // ---- 2. spawn ----
        {
            let mut p = begin!(enc, "spawn");
            p.set_bind_group(0, &self.groups[self.parity].spawn, &[]);
            if spawn_count > 0 {
                p.set_pipeline(&self.pipelines.spawn);
                p.dispatch_workgroups(self.dispatch_count(spawn_count), 1, 1);
            }
            p.set_pipeline(&self.pipelines.spawn_commit);
            p.dispatch_workgroups(1, 1, 1);
        }
        self.live += spawn_count;

        // [117]
        let voices = self.live;
        let parts = render_parts(voices, nwg, self.budget.voices());
        if parts > 1 {
            self.split_blocks += 1;
            if self.split_blocks == 1 {
                log::info!(
                    "gpu: a block of {voices} voices goes up as {parts} submissions, over the {} voices \
                     one may cover; the output does not depend on it",
                    self.budget.voices()
                );
            }
        }
        let render_index = pass_index as u32;
        pass_index += 1;
        // [118]
        let chan = self.bend_active || self.gain_active || self.variant_active;
        let glide_on = self.render_glide.is_some() && self.glide_active;
        // [119]
        if parts > 1 {
            self.ensure_split(chan, glide_on);
        }
        let mut first_at: Option<std::time::Instant> = None;
        for part in 0..parts {
            let first = (nwg as u64 * part as u64 / parts as u64) as u32;
            let end = (nwg as u64 * (part as u64 + 1) / parts as u64) as u32;
            if part > 0 {
                let done = std::mem::replace(
                    &mut enc,
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("block") }),
                );
                first_at.get_or_insert_with(std::time::Instant::now);
                self.queue.submit(Some(done.finish()));
                // [120]
                self.queue.write_buffer(&self.uniform_buf, RENDER_WG_BASE, bytemuck::bytes_of(&first));
            }
            let ts = self.timing.as_ref().and_then(|t| {
                // [121]
                let begin = (part == 0).then_some(render_index * 2);
                let stop = (part + 1 == parts).then_some(render_index * 2 + 1);
                (begin.is_some() || stop.is_some()).then_some(wgpu::ComputePassTimestampWrites {
                    query_set: &t.set,
                    beginning_of_pass_write_index: begin,
                    end_of_pass_write_index: stop,
                })
            });
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render"),
                timestamp_writes: ts,
            });
            p.set_bind_group(0, &self.groups[self.parity].render, &[]);
            let glide = self.render_glide.as_ref().filter(|_| self.glide_active);
            p.set_pipeline(if parts > 1 {
                self.render_split[((glide_on as usize) << 1) | chan as usize]
                    .as_ref()
                    .expect("compiled by ensure_split above")
            } else {
                match (glide, chan) {
                    (Some(g), false) => &g[0],
                    (Some(g), true) => &g[1],
                    (None, true) => &self.pipelines.render_chan,
                    (None, false) => &self.pipelines.render,
                }
            });
            // [122]
            p.dispatch_workgroups(end - first, 1, 1);
        }

        // ---- 4. reduce ----
        {
            let mut p = begin!(enc, "reduce");
            p.set_bind_group(0, &self.reduce_group, &[]);
            // [123]
            if nwg <= reduce_thin_max(&self.cfg) {
                p.set_pipeline(&self.pipelines.reduce_thin);
                p.dispatch_workgroups((self.cfg.block_frames * 2).div_ceil(REDUCE_THIN_LANES), 1, 1);
            } else {
                p.set_pipeline(&self.pipelines.reduce);
                p.dispatch_workgroups(self.cfg.block_frames * 2, 1, 1);
            }
        }

        // ---- 5. compact, and re-sort in the same pass ----
        {
            // [124]
            let live_wgs = self.dispatch_count(self.live);
            let mut p = begin!(enc, "compact");

            // [125]
            p.set_bind_group(0, &self.groups[self.parity].compact, &[]);
            p.set_pipeline(&self.pipelines.scan_local);
            p.dispatch_workgroups(live_wgs, 1, 1);
            p.set_pipeline(&self.pipelines.scan_blocks);
            p.dispatch_workgroups(1, 1, 1);

            if self.cfg.sort_voices {
                // [126]
                let mut pair_parity = 0usize;
                p.set_bind_group(0, &self.groups[self.parity].sort[pair_parity], &[]);
                p.set_pipeline(&self.pipelines.sort_init);
                p.dispatch_workgroups(1, 1, 1);
                p.set_pipeline(&self.pipelines.sort_build_keys);
                p.dispatch_workgroups(live_wgs, 1, 1);

                for _ in 0..self.sort_key.bits {
                    p.set_bind_group(0, &self.groups[self.parity].sort[pair_parity], &[]);
                    p.set_pipeline(&self.pipelines.sort_scan_local);
                    p.dispatch_workgroups(live_wgs, 1, 1);
                    p.set_pipeline(&self.pipelines.sort_scan_blocks);
                    p.dispatch_workgroups(1, 1, 1);
                    p.set_pipeline(&self.pipelines.sort_split);
                    p.dispatch_workgroups(live_wgs, 1, 1);
                    p.set_pipeline(&self.pipelines.sort_advance);
                    p.dispatch_workgroups(1, 1, 1);
                    pair_parity ^= 1;
                }

                p.set_bind_group(0, &self.groups[self.parity].sort[pair_parity], &[]);
                p.set_pipeline(&self.pipelines.sort_gather);
                p.dispatch_workgroups(live_wgs, 1, 1);
            } else {
                p.set_pipeline(&self.pipelines.scatter);
                p.dispatch_workgroups(live_wgs, 1, 1);
            }

            p.set_bind_group(0, &self.groups[self.parity].compact, &[]);
            p.set_pipeline(&self.pipelines.compact_commit);
            p.dispatch_workgroups(1, 1, 1);
        }
        self.parity ^= 1;

        enc.copy_buffer_to_buffer(&self.out_buf, 0, &self.readback_out, 0, self.out_buf.size());
        enc.copy_buffer_to_buffer(
            &self.state_buf,
            0,
            &self.readback_state,
            0,
            self.state_buf.size(),
        );
        if let Some(t) = &self.timing {
            enc.resolve_query_set(&t.set, 0..TIMESTAMP_COUNT, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, t.resolve.size());
        }

        let at = *first_at.get_or_insert_with(std::time::Instant::now);
        self.last_submission = Some(self.queue.submit(Some(enc.finish())));
        self.in_flight = Some(InFlight { at, voices, parts });
        Ok(())
    }

    /// Wait for the queued block and take its audio and counters back. \[127\]
    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        // [128]
        let mut bufs: Vec<&wgpu::Buffer> = vec![&self.readback_out, &self.readback_state];
        let period_ns = self.timing.as_ref().map(|t| {
            bufs.push(&t.readback);
            t.period_ns
        });
        let waited = std::time::Instant::now();
        let reads = self.map_read_many(&bufs)?;
        // [129]
        if let Some(f) = self.in_flight.take() {
            let wait = waited.elapsed().as_secs_f64();
            if let Some((was, now)) = self.budget.observe(f.voices, f.parts, wait, f.at.elapsed().as_secs_f64()) {
                log::info!(
                    "gpu: submissions now cover at most {now} voices, from {was}: a block of {} voices \
                     in {} part(s) kept the host waiting {wait:.2} s",
                    f.voices,
                    f.parts
                );
            }
        }

        let samples: &[f32] = bytemuck::cast_slice(&reads[0]);
        out.copy_from_slice(&samples[..out.len()]);

        let state: &[u32] = bytemuck::cast_slice(&reads[1]);
        self.live = state[S_LIVE].min(self.cfg.max_voices);
        self.stolen = state[S_STOLEN] as u64;
        // [130]
        if state[S_DROPPED] != 0 {
            log::error!(
                "the spawn pass dropped {} voices it should never have been handed",
                state[S_DROPPED]
            );
            self.dropped += state[S_DROPPED] as u64;
            self.queue
                .write_buffer(&self.state_buf, S_DROPPED as u64 * 4, &0u32.to_le_bytes());
        }

        if let Some(period_ns) = period_ns {
            let ticks: &[u64] = bytemuck::cast_slice(&reads[2]);
            let mut v = Vec::new();
            for (i, name) in PASS_NAMES.iter().enumerate() {
                let a = ticks[i * 2];
                let b = ticks[i * 2 + 1];
                let ms = if b > a {
                    (b - a) as f64 * period_ns as f64 / 1.0e6
                } else {
                    0.0
                };
                v.push((*name, ms));
            }
            self.last_timings = v;
        }

        self.peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        Ok(())
    }

    /// With the pool empty nothing on the device carries into the next block: \[131\]
    fn skip_block(&mut self) -> Result<()> {
        debug_assert_eq!(self.live, 0, "a block with voices alive cannot be skipped");
        self.env_phase = (self.env_phase + self.cfg.block_frames) % self.cfg.env_step_frames();
        self.peak = 0.0;
        self.planned = (0, 0);
        Ok(())
    }

    fn stats(&self) -> BlockStats {
        BlockStats {
            active_voices: self.live as u64,
            stolen: self.stolen,
            dropped: self.dropped,
            peak: self.peak,
        }
    }

    fn name(&self) -> &'static str {
        "gpu"
    }

    fn timings(&self) -> Vec<(&'static str, f64)> {
        self.last_timings.clone()
    }

    /// The live voices and the counters beside them. \[132\]
    fn save_state(&mut self, w: &mut dyn std::io::Write) -> Result<()> {
        if self.in_flight.is_some() || self.planned != (0, 0) {
            bail!("a render can be saved only between blocks, with none submitted");
        }
        let fields = voice_fields(&self.cfg) as usize;
        let (slots, live) = (self.slots as usize, self.live as usize);
        let state = self.read_range(&self.state_buf, 0, STATE_SLOTS as u64 * 4)?;
        let mut e = Enc::new();
        e.u32(fields as u32);
        e.u32(self.slots);
        e.u32(self.live);
        e.u32(self.env_phase);
        e.u64(self.dropped);
        e.raw(&state);
        w.write_all(e.as_bytes())?;
        // [133]
        let piece = (SAVE_PIECE / 4) as usize;
        for f in 0..fields {
            let mut at = 0usize;
            while at < live {
                let n = piece.min(live - at);
                let bytes = self.read_range(
                    &self.voices[self.parity],
                    ((f * slots + at) * 4) as u64,
                    n as u64 * 4,
                )?;
                w.write_all(&bytes)?;
                at += n;
            }
        }
        Ok(())
    }

    fn load_state(&mut self, r: &mut dyn std::io::Read) -> Result<()> {
        if self.live != 0 || self.parity != 0 || self.env_phase != 0 || self.in_flight.is_some() {
            bail!("a saved state can be loaded only into a backend that has rendered nothing");
        }
        let mut head = [0u8; 4 * 4 + 8 + STATE_SLOTS * 4];
        r.read_exact(&mut head)?;
        let mut d = Dec::new(&head);
        let (fields, slots, live, env_phase) = (d.u32()?, d.u32()?, d.u32()?, d.u32()?);
        let dropped = d.u64()?;
        let state_bytes = d.raw(STATE_SLOTS * 4)?;
        if fields as u64 != voice_fields(&self.cfg) || slots != self.slots {
            bail!(
                "the saved voice pool is {fields} fields of {slots} slots and this one is {} of {}: \
                 the voice limit or the settings are different",
                voice_fields(&self.cfg),
                self.slots
            );
        }
        if live > slots || env_phase >= self.cfg.env_step_frames() {
            bail!("the saved voice count or envelope position is outside the pool");
        }
        let state = bytemuck::pod_collect_to_vec::<u8, u32>(state_bytes);
        if state[S_LIVE] != live {
            bail!("the saved live count and the saved state disagree");
        }
        let (fields, slots, live) = (fields as usize, slots as usize, live as usize);
        let piece = (SAVE_PIECE / 4) as usize;
        let mut buf = vec![0u8; piece.min(live) * 4];
        for f in 0..fields {
            let mut at = 0usize;
            while at < live {
                let n = piece.min(live - at);
                r.read_exact(&mut buf[..n * 4])?;
                upload_in_pieces(
                    &self.device,
                    &self.queue,
                    &self.voices[0],
                    ((f * slots + at) * 4) as u64,
                    &buf[..n * 4],
                )?;
                at += n;
            }
        }
        self.queue.write_buffer(&self.state_buf, 0, state_bytes);
        self.queue.submit(std::iter::empty());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| device::lost(Some(&self.device), format_args!("device poll failed: {e:?}")))?;
        self.parity = 0;
        self.live = live as u32;
        self.env_phase = env_phase;
        self.dropped = dropped;
        self.stolen = state[S_STOLEN] as u64;
        Ok(())
    }
}

/// Bytes of the voice pool read back or written at a time by `save_state` and \[134\]
const SAVE_PIECE: u64 = 64 << 20;

impl GpuSynth {
    /// `bytes` of `src` from `offset`, read back through a staging buffer made \[135\]
    fn read_range(&self, src: &wgpu::Buffer, offset: u64, bytes: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("save readback"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("save") });
        enc.copy_buffer_to_buffer(src, offset, &staging, 0, bytes);
        let at = self.queue.submit(Some(enc.finish()));
        let mut out = map_read_prefix(&self.device, &[(&staging, bytes)], self.concurrent, Some(at))?;
        Ok(out.remove(0))
    }
}

impl GpuSynth {
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }
    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }
    /// PCI vendor and device id of the adapter this renders on.
    pub fn adapter_ids(&self) -> (u32, u32) {
        self.adapter_ids
    }
}

#[cfg(test)]
mod tests {
    use super::{gates_capacity, pool_budget_for, pool_parts, pool_wgsl, PoolParts, POOL_BUDGET_FLOOR, POOL_PARTS_MAX};

    /// The reduce pass's two shapes (`shaders/reduce.wgsl`) on partials a render \[136\]
    #[test]
    fn the_two_reduce_shapes_add_any_partials_to_the_same_bits() {
        use super::{compile, device, shader_source, Uniforms};
        use crate::config::Config;
        use wgpu::util::DeviceExt;

        let base = Config { block_frames: 64, ..Config::default() };
        let Ok((device, queue, ..)) = device::create(&base) else {
            eprintln!("no GPU; skipped");
            return;
        };
        let dir = std::env::temp_dir().join("kestrel_reduce_kernel");
        std::fs::create_dir_all(&dir).unwrap();
        let sf = dir.join("sine.sf2");
        crate::testkit::simple_sf2(&sf, 48_000).unwrap();
        let bank = crate::load_bank(&sf, &base).unwrap();
        let layout = device::bind_layout(&device, "reduce", &[false, true, false]);
        let src = include_str!("../../shaders/reduce.wgsl");

        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 16) as u32
        };
        let value = |next: &mut dyn FnMut() -> u32| -> f32 {
            let r = next();
            match r % 16 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(next() & 0x807F_FFFF), // subnormal
                3 => f32::from_bits(next()),                // any bits at all
                _ => {
                    let scale = 10f32.powi((next() % 9) as i32 - 4);
                    ((next() % 20001) as f32 / 10000.0 - 1.0) * scale
                }
            }
        };

        let out_len = base.block_frames as usize * 2;
        let read = |buf: &wgpu::Buffer| -> Vec<u32> {
            let rb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: buf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &rb, 0, buf.size());
            queue.submit(Some(enc.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            rb.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let v = bytemuck::cast_slice::<u8, u32>(&rb.slice(..).get_mapped_range()).to_vec();
            v
        };

        for kahan in [false, true] {
            let thin_cfg = Config { kahan_reduce: kahan, ..base.clone() };
            let old_cfg = Config { reduce_thin: 0, ..thin_cfg.clone() };
            let mut thin = compile(&device, &thin_cfg, "thin", shader_source(src, &thin_cfg, &bank), &layout, &["thin"]);
            let mut old = compile(&device, &old_cfg, "old", shader_source(src, &old_cfg, &bank), &layout, &["main"]);
            let (thin, old) = (thin.remove(0), old.remove(0));
            for n in 1..=super::reduce_thin_max(&thin_cfg) {
                let partials: Vec<f32> = (0..out_len * n as usize).map(|_| value(&mut next)).collect();
                let u = Uniforms { block_frames: base.block_frames, render_workgroups: n, ..Default::default() };
                let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("uniforms"),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("partials"),
                    contents: bytemuck::cast_slice(&partials),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                // A word no sum is, so a sample nobody wrote cannot match one.
                let sentinel = vec![0xA5A5_A5A5u32; out_len];
                let make_out = || {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("out"),
                        contents: bytemuck::cast_slice(&sentinel),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    })
                };
                let (out_thin, out_old) = (make_out(), make_out());
                let mut enc = device.create_command_encoder(&Default::default());
                for (pipeline, out, groups) in [
                    (&thin, &out_thin, (out_len as u32).div_ceil(super::REDUCE_THIN_LANES)),
                    (&old, &out_old, out_len as u32),
                ] {
                    let group = device::bind(&device, &layout, &[&ubuf, &pbuf, out]);
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_bind_group(0, &group, &[]);
                    p.set_pipeline(pipeline);
                    p.dispatch_workgroups(groups, 1, 1);
                }
                queue.submit(Some(enc.finish()));
                let (a, b) = (read(&out_thin), read(&out_old));
                assert!(a.iter().all(|&w| w != 0xA5A5_A5A5), "kahan {kahan}, {n} partials: a sample was never written");
                if let Some(i) = (0..out_len).find(|&i| a[i] != b[i]) {
                    panic!(
                        "kahan {kahan}, {n} partials a sample, sample {i}: thread {:#010x} against workgroup {:#010x}",
                        a[i], b[i]
                    );
                }
            }
        }
    }

    /// The budget by card, at wgpu's 2,047 MiB binding: an 8 GB card keeps \[137\]
    #[test]
    fn the_pool_budget_is_three_quarters_of_the_card_between_2_gib_and_4_buffers() {
        let cap = (1u64 << 31) - 1;
        let four = POOL_PARTS_MAX as u64 * (cap / 4) * 4;
        assert_eq!(pool_budget_for(Some(8 * GIB), cap), 6 * GIB);
        assert_eq!(pool_budget_for(Some(12 * GIB), cap), four);
        assert_eq!(pool_budget_for(Some(16 * GIB), cap), four);
        assert_eq!(pool_budget_for(Some(6 * GIB), cap), 4 * GIB + GIB / 2);
        assert_eq!(pool_budget_for(Some(4 * GIB), cap), 2 * GIB + GIB / 2);
        assert_eq!(pool_budget_for(Some(2 * GIB), cap), POOL_BUDGET_FLOOR);
        assert_eq!(pool_budget_for(Some(128 << 20), cap), POOL_BUDGET_FLOOR);
        assert_eq!(pool_budget_for(None, cap), POOL_BUDGET_FLOOR);
        // [138]
        assert_eq!(pool_budget_for(Some(16 * GIB), 1023 << 20), 4 * (1023u64 << 20));
    }

    const GIB: u64 = 1 << 30;

    const CAP: u64 = (1 << 31) - 1;
    const MIB: u64 = 1 << 20;
    const STORAGE: u32 = 1 << 20;
    fn parts(bytes: u64, rate: u32) -> anyhow::Result<PoolParts> {
        pool_parts(bytes / 4, rate, CAP, 0, STORAGE, "Test GPU (Vulkan)")
    }

    /// The 1.2.3 report: a 2,610 MiB pool at 48 kHz against a 2,047 MiB \[139\]
    #[test]
    fn a_pool_past_one_binding_is_split_and_one_that_fits_is_not() {
        let whole = |p: PoolParts| p.words_each as u64 * p.count as u64;
        assert_eq!(parts(CAP / 4 * 4, 48_000).unwrap(), PoolParts { words_each: (CAP / 4) as u32, count: 1 });
        assert_eq!(parts(757 * MIB, 48_000).unwrap(), PoolParts { words_each: (757 * MIB / 4) as u32, count: 1 });
        let p = parts(2_736_636_000, 48_000).unwrap();
        assert_eq!(p, PoolParts { words_each: (CAP / 4) as u32, count: 2 });
        assert!(whole(p) >= 2_736_636_000 / 4);
        // Four parts is the most: 8 GiB less a few bytes.
        let p = parts(POOL_PARTS_MAX as u64 * (CAP / 4) * 4, 48_000).unwrap();
        assert_eq!(p.count, POOL_PARTS_MAX);
        // [140]
        let p = pool_parts(1000, 48_000, CAP, 1200, STORAGE, "Test GPU (Vulkan)").unwrap();
        assert_eq!(p, PoolParts { words_each: 300, count: 4 });
    }

    /// Past four parts it is refused, naming the rate that fits -- 44.1 kHz \[141\]
    #[test]
    fn a_pool_past_four_parts_is_refused_with_the_rate_that_fits() {
        let e = parts(9 << 30, 48_000).unwrap_err().to_string();
        for want in ["9216 MiB at 48000 Hz", "at most 8191 MiB", "4 buffers of 2047 MiB", "--rate 32000", "about 6144 MiB"] {
            assert!(e.contains(want), "no {want:?} in: {e}");
        }
        for (bytes, rate) in [(9 << 30, 0), (40 << 30, 48_000)] {
            let e = parts(bytes, rate).unwrap_err().to_string();
            assert!(!e.contains("--rate") && e.contains("smaller soundfont"), "{e}");
        }
        let e = pool_parts(5 << 28, 48_000, CAP, 0, 13, "Test GPU (Vulkan)").unwrap_err().to_string();
        assert!(e.contains("3 buffers") && e.contains("bind 14 storage buffers where it allows 13"), "{e}");
    }

    /// The WGSL a split adds, and that a pool in one buffer adds none, which \[142\]
    #[test]
    fn a_split_pool_adds_its_bindings_and_its_fetch_and_one_buffer_adds_nothing() {
        assert_eq!(pool_wgsl(PoolParts { words_each: 5, count: 1 }), [String::new(), String::new(), String::new()]);
        let [decls, fetch, pairs] = pool_wgsl(PoolParts { words_each: 100, count: 3 });
        assert_eq!(
            pairs,
            "\n    if (w + 1u < u.pool_words) {\
             \n        if (w + 1u < 100u) { let o = w; return vec4<f32>(unpack2x16snorm(pool[o]), unpack2x16snorm(pool[o + 1u])); }\
             \n        if (w >= 100u && w + 1u < 200u) { let o = w - 100u; return vec4<f32>(unpack2x16snorm(pool1[o]), unpack2x16snorm(pool1[o + 1u])); }\
             \n        if (w >= 200u) { let o = w - 200u; return vec4<f32>(unpack2x16snorm(pool2[o]), unpack2x16snorm(pool2[o + 1u])); }\
             \n    }"
        );
        assert_eq!(
            decls,
            "@group(0) @binding(13) var<storage, read> pool1: array<u32>;\n\
             @group(0) @binding(14) var<storage, read> pool2: array<u32>;\n"
        );
        assert_eq!(
            fetch,
            "\n    if (w >= 100u) {\
             \n        if (w < 200u) { return unpack2x16snorm(pool1[w - 100u]); }\
             \n        return unpack2x16snorm(pool2[w - 200u]);\
             \n    }"
        );
    }

    /// One port's meta header, and eight ports'.
    const META: u64 = (16 * 128 + 1) * 2;
    const META8: u64 = (128 * 128 + 1) * 2;

    #[test]
    fn the_gates_buffer_grows_by_doubling_and_never_past_one_binding() {
        assert_eq!(gates_capacity(40_000, 32768, META, 2 * GIB).unwrap(), 65536);
        // Doubling 10M runs would need 128 MiB and a bit; clamp to the binding.
        let clamped = gates_capacity(10_000_000, 32768, META, 128 << 20).unwrap();
        assert!((10_000_000..16_777_216).contains(&clamped), "{clamped}");
    }

    /// A file reaching a new port grows the header with no more runs than \[143\]
    #[test]
    fn a_grown_header_keeps_the_runs_the_buffer_already_held() {
        assert_eq!(gates_capacity(10, 32768, META8, 2 * GIB).unwrap(), 32768);
    }

    /// The count from the crash on 2026-09-13, as entries. It used to wrap the \[144\]
    #[test]
    fn too_many_runs_for_one_binding_is_an_error_that_names_the_sizes() {
        let e = gates_capacity(2_452_415_145, 32768, META, 2 * GIB).unwrap_err().to_string();
        assert!(e.contains("2452415145 note-off runs"), "{e}");
        assert!(e.contains("lower --block"), "{e}");
    }

    /// The lost-device message is one paragraph. Its first version left runs \[145\]
    #[test]
    fn the_lost_device_message_reads_as_one_paragraph() {
        let m = super::device::lost(None, "device poll failed: WrongSubmissionIndex(324, 323)").to_string();
        assert!(!m.contains("  "), "{m}");
        assert!(m.contains("--block 1024") && m.ends_with("(wgpu: device poll failed: WrongSubmissionIndex(324, 323))"), "{m}");
    }

    /// An integrated GPU with 5,000,000 voices (2026-10-05): the buffers asked for \[146\]
    #[test]
    fn buffers_past_the_memory_budget_are_said_so_with_the_voice_limit_that_fits() {
        use super::{memory_note, vram};
        let gib = |n: u64| n << 30;
        let m = vram::GpuMemory {
            dedicated_total: 128 << 20,
            process_budget: Some(gib(3)),
            process_used: Some(gib(1)),
            ..Default::default()
        };
        // 2 GiB left, 2,400 MiB wanted, 2,000 MiB of it for the voices.
        let n = memory_note(2400 << 20, 2000 << 20, 5_000_000, &m).unwrap();
        assert!(n.short);
        assert!(n.text.contains("2400 MiB") && n.text.contains("3072 MiB") && n.text.contains("1024 MiB"), "{}", n.text);
        // (2048 - 400) / 2000 of 5,000,000, to two figures, rounded down.
        assert!(n.text.contains("--max-voices 4100000"), "{}", n.text);
        // Room: a line for the log, and no warning.
        let n = memory_note(1000 << 20, 800 << 20, 5_000_000, &m).unwrap();
        assert!(!n.short && !n.text.contains("--max-voices"), "{}", n.text);
        // Not even the part that does not scale with the voices.
        let tight = vram::GpuMemory { process_used: Some(gib(3) - (100 << 20)), ..m };
        let n = memory_note(2400 << 20, 2000 << 20, 5_000_000, &tight).unwrap();
        assert!(n.short && n.text.contains("even with no voices"), "{}", n.text);
        // No budget known, no opinion.
        assert!(memory_note(1, 1, 1, &vram::GpuMemory::default()).is_none());
    }

    /// The guided renderer's device max is worked out from this estimate, so it has to \[147\]
    #[test]
    fn an_estimate_is_what_a_render_allocates() {
        use super::{device, device_estimate, GpuSynth};
        use crate::config::Config;
        use std::sync::Arc;
        let base = Config { max_voices: 20_000, ..Config::default() };
        if device::create(&base).is_err() {
            eprintln!("no GPU; skipped");
            return;
        }
        let dir = std::env::temp_dir().join("kestrel_device_estimate");
        std::fs::create_dir_all(&dir).unwrap();
        let (sine, rich) = (dir.join("sine.sf2"), dir.join("rich.sf2"));
        crate::testkit::simple_sf2(&sine, 48_000).unwrap();
        crate::testkit::rich_sf2(&rich, 48_000).unwrap();
        let cases = [
            Config { ..base.clone() },
            Config { mod_env_enabled: false, ..base.clone() },
            Config { mod_env_enabled: false, lfo_enabled: false, ..base.clone() },
            Config { block_frames: 1024, ..base.clone() },
            Config { max_voices: 1_000, max_steal_percent: 50, ..base.clone() },
            Config { max_voices: 300, ..base.clone() },
            Config { max_param_variants: 4, ..base.clone() },
        ];
        for font in [&sine, &rich] {
            for cfg in &cases {
                let bank = Arc::new(crate::load_bank(font, cfg).unwrap());
                let est = device_estimate(cfg, &bank);
                let synth = GpuSynth::new(cfg, bank).unwrap();
                assert_eq!(
                    est.total(cfg.pool_slots()),
                    synth.vram_bytes(),
                    "{}: {} voices, block {}, {} variants",
                    font.display(),
                    cfg.max_voices,
                    cfg.block_frames,
                    cfg.max_param_variants
                );
            }
        }
    }

    /// What fits: the usable memory is seven eighths of the card's, less what does not \[148\]
    #[test]
    fn the_voices_that_fit_in_memory_come_from_the_estimate() {
        use super::{device_estimate, max_voices_in_memory};
        use crate::config::Config;
        let dir = std::env::temp_dir().join("kestrel_memory_max");
        std::fs::create_dir_all(&dir).unwrap();
        let sf = dir.join("sine.sf2");
        crate::testkit::simple_sf2(&sf, 48_000).unwrap();
        let cfg = Config::default();
        let bank = crate::load_bank(&sf, &cfg).unwrap();
        let est = device_estimate(&Config { max_voices: u32::MAX / 4, ..cfg.clone() }, &bank);
        for gib in [1u64, 2, 8, 16] {
            let memory = gib << 30;
            let n = max_voices_in_memory(&cfg, &bank, memory);
            let probe = Config { max_voices: n, ..cfg.clone() };
            // [149]
            let usable = memory - memory / 8;
            assert!(est.fixed + est.per_slot * probe.pool_slots() as u64 <= usable, "{gib} GiB: {n}");
            let over = Config { max_voices: n + 2, ..cfg.clone() };
            assert!(est.fixed + est.per_slot * over.pool_slots() as u64 > usable, "{gib} GiB: {n} is not the most");
        }
        // More memory, more voices; a card the samples do not fit in, none.
        assert!(max_voices_in_memory(&cfg, &bank, 8 << 30) > max_voices_in_memory(&cfg, &bank, 2 << 30));
        assert_eq!(max_voices_in_memory(&cfg, &bank, 1000), 0);
        // A wider steal headroom leaves fewer voices of the same slots.
        let wide = Config { max_steal_percent: 100, ..cfg.clone() };
        assert!(max_voices_in_memory(&wide, &bank, 8 << 30) < max_voices_in_memory(&cfg, &bank, 8 << 30));
    }

    /// The reason the split costs a stock render nothing: its largest block is \[150\]
    #[test]
    fn a_block_at_stock_settings_is_never_split() {
        let cfg = crate::config::Config::default();
        let slots = cfg.max_voices as u64 * (100 + cfg.max_steal_percent as u64) / 100;
        assert!(slots < cfg.submit_voices as u64, "{slots} slots against {}", cfg.submit_voices);
        assert_eq!(super::render_parts(slots as u32, cfg.max_render_workgroups, cfg.submit_voices), 1);
    }

    #[test]
    fn a_huge_block_goes_up_in_parts_of_about_the_budget() {
        let budget = crate::config::Config::default().submit_voices;
        // 15M voices, as on the mid-range card that was reset (2026-10-03).
        let parts = super::render_parts(15_000_000, 2048, budget);
        assert_eq!(parts, 4);
        assert!(15_000_000u64.div_ceil(parts as u64) <= budget as u64);
        // Never more parts than workgroups, and never none.
        assert_eq!(super::render_parts(1_000_000, 4, 1), 4);
        assert_eq!(super::render_parts(0, 2048, budget), 1);
        // Every workgroup lands in exactly one part, in order.
        let (nwg, parts) = (2048u64, 8u64);
        let mut next = 0;
        for p in 0..parts {
            let (first, end) = (nwg * p / parts, nwg * (p + 1) / parts);
            assert_eq!(first, next);
            next = end;
        }
        assert_eq!(next, nwg);
    }
}
