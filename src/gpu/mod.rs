// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! wgpu compute backend. \[1\]

mod device;
pub mod vram;

pub use device::{print_adapters, survey, AdapterSummary};

use crate::backend::{Backend, BlockStats};
use crate::bank::{Bank, ModEnvParams, RegionParams};
use crate::config::{AdmitRule, Config, EnvelopeCurve, StealRule};
use crate::voice::{spawn_pick, SpawnCmd, BEND_CHANNELS, CHAN_FIELDS, GATE_SLOTS};
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
    _pad: u32,
}

/// How the voice pool's sort key is packed into 32 bits. \[4\]
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

        // [5]
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

/// Slots in the device state buffer. Mirrors the `S_*` constants in \[6\]
const S_LIVE: usize = 0;
const S_STOLEN: usize = 8;
const S_DROPPED: usize = 9;
const STATE_SLOTS: usize = 16;

/// Largest grid one dispatch dimension may take. This is the D3D12 ceiling and \[7\]
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// u32 words the voice pool stores per slot. Mirrors `VOICE_FIELDS` in \[8\]
const VOICE_FIELDS: u64 = 26;

/// Words actually allocated per slot for this configuration. \[9\]
fn voice_fields(cfg: &Config) -> u64 {
    if cfg.mod_env_enabled {
        VOICE_FIELDS
    } else if cfg.lfo_enabled {
        VOICE_FIELDS - 1
    } else {
        VOICE_FIELDS - 2
    }
}

/// The largest `max_voices` an adapter can take, given how much of one buffer \[10\]
pub fn max_voices_for_binding(binding_bytes: u64, steal_percent: u32) -> u32 {
    let slots = binding_bytes / (VOICE_FIELDS * 4);
    let v = slots * 100 / (100 + steal_percent.clamp(1, 100) as u64);
    v.min(u32::MAX as u64) as u32
}

fn substitute(src: &str, cfg: &Config, bank: &Bank) -> String {
    src.replace("{{WG}}", &cfg.workgroup_size.to_string())
        .replace("{{TILE}}", &cfg.reduce_tile.to_string())
        .replace("{{GATE_TILE}}", &cfg.gate_frames.to_string())
        .replace("{{STEAL_FADE}}", &cfg.steal_fade_frames.to_string())
        .replace("{{KAHAN}}", if cfg.kahan_reduce { "true" } else { "false" })
        .replace("{{GAIN_RAMP}}", if cfg.gain_ramp { "true" } else { "false" })
        .replace("{{FILTER_RAMP}}", if cfg.filter_ramp { "true" } else { "false" })
        .replace("{{USE_LFO}}", if cfg.lfo_enabled { "true" } else { "false" })
        // [11]
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
        .replace("{{VOICE_FIELDS}}", &voice_fields(cfg).to_string())
}

/// Write `bytes` into `buf` from `offset`, 64 MiB at a time, letting the device \[12\]
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
            .map_err(|e| anyhow::anyhow!("device poll failed: {e:?}"))?;
    }
    Ok(())
}

fn shader_source(body: &str, cfg: &Config, bank: &Bank) -> String {
    let mut s = substitute(include_str!("../../shaders/common.wgsl"), cfg, bank);
    s.push('\n');
    s.push_str(&substitute(body, cfg, bank));
    s
}

/// The render pass with or without the channel controller path and the glide \[13\]
fn render_source(cfg: &Config, bank: &Bank, chan: bool, glide: bool) -> String {
    let body = include_str!("../../shaders/render.wgsl")
        .replace("{{CHAN}}", if chan { "true" } else { "false" })
        .replace("{{GLIDE}}", if glide { "true" } else { "false" });
    shader_source(&body, cfg, bank)
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
        // [14]
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
    /// The same pass compiled with the channel controller path in it. Selected \[15\]
    render_chan: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
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
    /// Indexed by which of the two (key, index) buffers currently holds the \[16\]
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
    /// Note-off runs the gates buffer can hold behind its meta header, two \[17\]
    off_runs_capacity: u64,
    /// The most of one buffer the adapter binds to a shader, which is as far \[18\]
    binding_cap: u64,
    pool_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    params_per_variant: u32,
    menv_buf: wgpu::Buffer,
    // [19]
    menv_factor_buf: wgpu::Buffer,
    menv_per_variant: u32,
    menv_factor_half: u32,
    menv_log2_base: u32,
    glide_base: u32,
    /// Whether the driver says a voice may glide during the next block.
    glide_active: bool,
    /// The render pass with the glide path in it, plain and with the \[20\]
    render_glide: Option<[wgpu::ComputePipeline; 2]>,
    /// Their sources, fully substituted, kept so the compile can happen then.
    glide_sources: [String; 2],

    readback_out: wgpu::Buffer,
    readback_state: wgpu::Buffer,

    /// Which of `voices` currently holds the live pool.
    parity: usize,
    /// Host mirror of the device live count, exact because every change to it \[21\]
    live: u32,
    /// Allocated voice slots, `Config::pool_slots()`. The SoA stride.
    slots: u32,
    pending_spawns: Vec<SpawnCmd>,
    /// Reused buffer for the thinned spawn list, so a saturated block does not \[22\]
    spawn_scratch: Vec<SpawnCmd>,
    stolen: u64,
    dropped: u64,
    peak: f32,

    timing: Option<Timing>,
    last_timings: Vec<(&'static str, f64)>,
    vram_bytes: u64,
    /// PCI vendor and device id of the adapter, which is how `vram` finds it.
    adapter_ids: (u32, u32),
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
        cfg.validate()?;
        let (device, queue, adapter_info, limits, has_timestamps) = device::create(cfg)?;
        let adapter_name = format!("{} ({:?})", adapter_info.name, adapter_info.backend);

        // [23]
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
        // [24]
        let scan_workgroups = cfg.pool_slots().div_ceil(cfg.workgroup_size);

        // [25]
        let voice_pool_bytes = cfg.pool_slots() as u64 * voice_fields(cfg) * 4;
        let binding_cap =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if voice_pool_bytes > binding_cap {
            let voice_cap = max_voices_for_binding(binding_cap, cfg.max_steal_percent);
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

        // [26]
        let capacity = cfg.pool_slots();
        let tiles = cfg.block_frames / cfg.gate_frames;
        let nwg = cfg.max_render_workgroups.clamp(1, MAX_WORKGROUPS_PER_DIM);

        // [27]
        let pool = &bank.pool;
        let even = pool.len() / 2 * 2;
        let pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample pool"),
            size: (pool.len().div_ceil(2).max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        upload_in_pieces(&device, &queue, &pool_buf, 0, bytemuck::cast_slice(&pool[..even]))?;
        if even < pool.len() {
            // An odd last sample, padded out to a whole word.
            queue.write_buffer(&pool_buf, even as u64 * 2, bytemuck::cast_slice(&[pool[even], 0]));
        }

        // [28]
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
        // [29]
        let params_per_variant = params.len() as u32;
        let variants = cfg.max_param_variants.max(1);
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("region params"),
            size: (params.len() * variants as usize * std::mem::size_of::<RegionParams>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        upload_in_pieces(&device, &queue, &params_buf, 0, bytemuck::cast_slice(params))?;

        // [30]
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

        // [31]
        let mut menv_tables: Vec<u32> = if bank.menv_factors.is_empty() {
            vec![1 << 24]
        } else {
            bank.menv_factors.clone()
        };
        let menv_log2_base = menv_tables.len() as u32;
        if bank.menv_log2.is_empty() {
            menv_tables.push(0);
        } else {
            menv_tables.extend_from_slice(&bank.menv_log2);
        }
        // [32]
        let glide_base = menv_tables.len() as u32;
        menv_tables.extend_from_slice(&crate::porta::tables(cfg.sample_rate));
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
        // [33]
        let off_meta_words = (GATE_SLOTS as u64 + 1) * 2;
        let off_runs_capacity = 32768u64;
        let gates_bytes = (off_meta_words + off_runs_capacity * 2) * 4;

        let uniform_buf = mk(
            "uniforms",
            std::mem::size_of::<Uniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let gates_buf = mk("gates", gates_bytes, storage | wgpu::BufferUsages::COPY_DST);
        // One row past the last tile; see `voice::ChannelTable::row_bias`.
        let chan_bytes = (tiles as u64 + 1) * BEND_CHANNELS as u64 * CHAN_FIELDS as u64 * 4;
        let chan_buf = mk("channels", chan_bytes, storage | wgpu::BufferUsages::COPY_DST);
        let voices = [
            mk("voices a", voice_bytes, storage | wgpu::BufferUsages::COPY_DST),
            mk("voices b", voice_bytes, storage | wgpu::BufferUsages::COPY_DST),
        ];
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

        let vram_bytes = pool.len() as u64 * 2
            + params.len() as u64 * 48
            + voice_bytes * 2
            + partial_bytes
            + out_bytes
            + gates_bytes
            + chan_bytes
            + capacity as u64 * 8  // scan + sort keys
            + capacity as u64 * 16 // sort pairs, double buffered
            + cmds_capacity as u64 * 72;
        log::info!(
            "gpu: {} | {:.1} MiB of device buffers ({:.1} MiB sample pool, \
             {:.1} MiB voice pool for {} voices, {:.1} MiB partials)",
            adapter_name,
            vram_bytes as f64 / 1048576.0,
            pool.len() as f64 * 2.0 / 1048576.0,
            voice_bytes as f64 * 2.0 / 1048576.0,
            capacity,
            partial_bytes as f64 / 1048576.0,
        );

        // ---- layouts, pipelines, bind groups ----
        let layouts = Layouts {
            spawn: device::bind_layout(&device, "spawn", &[false, true, false, false]),
            render: device::bind_layout(
                &device,
                "render",
                &[false, true, true, true, false, false, true, true, true, true],
            ),
            reduce: device::bind_layout(&device, "reduce", &[false, true, false]),
            compact: device::bind_layout(
                &device,
                "compact",
                &[false, false, false, false, false, false, false],
            ),
            select: device::bind_layout(&device, "select", &[false, true, false, false]),
            sort: device::bind_layout(&device, "sort", &[false; 8]),
        };

        let pipelines = Self::build_pipelines(&device, cfg, &bank, &layouts)?;
        let glide_sources = [render_source(cfg, &bank, false, true), render_source(cfg, &bank, true, true)];

        let groups = [
            Groups {
                spawn: device::bind(
                    &device,
                    &layouts.spawn,
                    &[&uniform_buf, &cmds_buf, &voices[0], &state_buf],
                ),
                render: device::bind(
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
                render: device::bind(
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
            binding_cap,
            slots: capacity,
            pool_buf,
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
            readback_out,
            readback_state,
            parity: 0,
            live: 0,
            pending_spawns: Vec::new(),
            spawn_scratch: Vec::new(),
            stolen: 0,
            dropped: 0,
            peak: 0.0,
            timing,
            last_timings: Vec::new(),
            vram_bytes,
            adapter_ids: (adapter_info.vendor, adapter_info.device),
        };
        s.write_uniforms(0, 0, s.render_workgroups(0));
        Ok(s)
    }

    fn build_pipelines(
        device: &wgpu::Device,
        cfg: &Config,
        bank: &Bank,
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
            render_source(cfg, bank, true, false),
            &layouts.render,
            &["main"],
        );
        let mut render = compile(
            device,
            cfg,
            "render",
            render_source(cfg, bank, false, false),
            &layouts.render,
            &["main"],
        );
        let mut reduce = make(
            "reduce",
            include_str!("../../shaders/reduce.wgsl"),
            &layouts.reduce,
            &["main"],
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

    /// Workgroups the render pass will be dispatched with this block, given \[34\]
    fn render_workgroups(&self, voices: u32) -> u32 {
        let cap = self
            .cfg
            .max_render_workgroups
            .clamp(1, MAX_WORKGROUPS_PER_DIM);
        voices.div_ceil(self.cfg.workgroup_size).clamp(1, cap)
    }

    fn write_uniforms(&self, spawn_count: u32, steal_k: u32, nwg: u32) {
        let u = Uniforms {
            block_frames: self.cfg.block_frames,
            tiles: self.cfg.block_frames / self.cfg.reduce_tile,
            // [35]
            capacity: self.slots,
            spawn_count,
            render_workgroups: nwg,
            interp: self.cfg.interpolation as u32,
            exp_decay: (self.cfg.decay_curve == EnvelopeCurve::Exponential) as u32,
            exp_release: (self.cfg.release_curve == EnvelopeCurve::Exponential) as u32,
            env_floor: self.cfg.env_floor,
            pool_words: self.pool_words(),
            steal_k,
            sort_bits: self.sort_key.bits,
            sort_region_shift: self.sort_key.region_shift,
            sort_stage_shift: self.sort_key.stage_shift,
            sort_phase_shift: self.sort_key.phase_shift,
            sort_phase_mask: self.sort_key.phase_mask,
            sort_dead_region: self.sort_key.dead_region,
            chan_active: (self.bend_active as u32)
                | ((self.gain_active as u32) << 1)
                | ((self.variant_active as u32) << 2)
                | ((self.cut_active as u32) << 3),
            params_per_variant: self.params_per_variant,
            steal_by_level: (self.cfg.steal_rule == StealRule::Quietest) as u32,
            menv_factor_half: self.menv_factor_half,
            menv_log2_base: self.menv_log2_base,
            glide_base: self.glide_base,
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    fn pool_words(&self) -> u32 {
        (self.pool_buf.size() / 4) as u32
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

    /// Grow the gates buffer so it can carry `runs` note-off runs. \[36\]
    fn grow_gates(&mut self, runs: usize) -> Result<()> {
        let Some(new_cap) =
            gates_capacity(runs as u64, self.off_runs_capacity, self.binding_cap)?
        else {
            return Ok(());
        };
        let meta_words = (GATE_SLOTS as u64 + 1) * 2;
        self.gates_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gates"),
            size: (meta_words + new_cap * 2) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.off_runs_capacity = new_cap;
        // The render bind groups reference the old buffer, so rebuild them.
        for p in 0..2 {
            self.groups[p].render = device::bind(
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
            );
        }
        log::debug!("grew the gates buffer to {new_cap} note-off runs");
        Ok(())
    }

    /// Workgroups to dispatch for `items` voices. Every entry point this feeds \[37\]
    fn dispatch_count(&self, items: u32) -> u32 {
        let ceiling = self.cfg.max_pool_workgroups.clamp(1, MAX_WORKGROUPS_PER_DIM);
        items.div_ceil(self.cfg.workgroup_size).clamp(1, ceiling)
    }

    /// Read several readback buffers back with **one** device wait. \[38\]
    fn map_read_many(&self, bufs: &[&wgpu::Buffer]) -> Result<Vec<Vec<u8>>> {
        let mut rxs = Vec::with_capacity(bufs.len());
        for buf in bufs {
            let (tx, rx) = std::sync::mpsc::channel();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rxs.push(rx);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("device poll failed: {e:?}"))?;
        let mut out = Vec::with_capacity(bufs.len());
        for (buf, rx) in bufs.iter().zip(rxs) {
            rx.recv()
                .context("readback channel closed")?
                .map_err(|e| anyhow::anyhow!("buffer map failed: {e:?}"))?;
            // The view borrows the buffer, so it has to be gone before unmap.
            out.push(buf.slice(..).get_mapped_range().to_vec());
            buf.unmap();
        }
        Ok(out)
    }
}

/// How many note-off runs the gates buffer should hold to carry `runs`, or \[39\]
fn gates_capacity(runs: u64, current: u64, binding_cap: u64) -> Result<Option<u64>> {
    if runs <= current {
        return Ok(None);
    }
    let meta_words = (GATE_SLOTS as u64 + 1) * 2;
    let bytes = |cap: u64| (meta_words + cap * 2) * 4;
    if bytes(runs) > binding_cap {
        bail!(
            "one block publishes {runs} note-off runs, a {:.2} GiB gate table, and the \
             adapter binds at most {:.2} GiB of one buffer to a shader; lower --block",
            bytes(runs) as f64 / (1u64 << 30) as f64,
            binding_cap as f64 / (1u64 << 30) as f64
        );
    }
    let doubled = runs.next_power_of_two();
    Ok(Some(if bytes(doubled) > binding_cap {
        (binding_cap / 4 - meta_words) / 2
    } else {
        doubled
    }))
}

impl Backend for GpuSynth {
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

        // [40]
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
        // [41]
        self.queue
            .write_buffer(&self.chan_buf, 0, bytemuck::cast_slice(rows));
        self.bend_active = bend;
        self.gain_active = gain;
        self.variant_active = variant;
        self.cut_active = cut;
        Ok(())
    }

    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()> {
        self.grow_gates(runs.len() / 2)?;
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
        // Held until render, so the whole block goes in one command buffer.
        self.pending_spawns.clear();
        self.pending_spawns.extend_from_slice(cmds);
        Ok(())
    }

    /// Queue the whole block: spawn, render, reduce, compact, and the copies \[42\]
    fn submit(&mut self) -> Result<()> {
        let cap = self.cfg.max_voices;

        // [43]
        let want = self.pending_spawns.len() as u32;
        let want = want.min(cap);
        if want < self.pending_spawns.len() as u32 {
            self.dropped += self.pending_spawns.len() as u64 - want as u64;
        }

        let mut steal_k = 0u32;
        if self.live + want > cap {
            match self.cfg.steal_rule {
                // [44]
                StealRule::Oldest | StealRule::Quietest => {
                    steal_k = (self.live + want - cap)
                        .min(self.live)
                        .min(self.cfg.max_steal())
                }
                StealRule::DropNew => {}
            }
        }
        let spawn_count = want.min(cap - (self.live - steal_k));
        self.dropped += (want - spawn_count) as u64;

        if spawn_count > 0 {
            self.grow_cmds(spawn_count);
            let total = self.pending_spawns.len();
            let take = spawn_count as usize;
            if take == total {
                self.queue
                    .write_buffer(&self.cmds_buf, 0, bytemuck::cast_slice(&self.pending_spawns));
            } else {
                // [45]
                self.spawn_scratch.clear();
                self.spawn_scratch.reserve(take);
                match self.cfg.admit_rule {
                    AdmitRule::Loudest => self
                        .spawn_scratch
                        .extend_from_slice(&self.pending_spawns[..take]),
                    AdmitRule::Even => {
                        for i in 0..take {
                            self.spawn_scratch
                                .push(self.pending_spawns[spawn_pick(i, total, take)]);
                        }
                    }
                }
                self.queue
                    .write_buffer(&self.cmds_buf, 0, bytemuck::cast_slice(&self.spawn_scratch));
            }
        }
        // [46]
        let nwg = self.render_workgroups(self.live + spawn_count);
        self.write_uniforms(spawn_count, steal_k, nwg);

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

        // [47]
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
        // [48]

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

        // ---- 3. render ----
        {
            let mut p = begin!(enc, "render");
            p.set_bind_group(0, &self.groups[self.parity].render, &[]);
            // [49]
            let chan = self.bend_active || self.gain_active || self.variant_active;
            // [50]
            let glide = self.render_glide.as_ref().filter(|_| self.glide_active);
            p.set_pipeline(match (glide, chan) {
                (Some(g), false) => &g[0],
                (Some(g), true) => &g[1],
                (None, true) => &self.pipelines.render_chan,
                (None, false) => &self.pipelines.render,
            });
            // [51]
            p.dispatch_workgroups(nwg, 1, 1);
        }

        // ---- 4. reduce ----
        {
            let mut p = begin!(enc, "reduce");
            p.set_bind_group(0, &self.reduce_group, &[]);
            p.set_pipeline(&self.pipelines.reduce);
            p.dispatch_workgroups(self.cfg.block_frames * 2, 1, 1);
        }

        // ---- 5. compact, and re-sort in the same pass ----
        {
            // [52]
            let live_wgs = self.dispatch_count(self.live);
            let mut p = begin!(enc, "compact");

            // [53]
            p.set_bind_group(0, &self.groups[self.parity].compact, &[]);
            p.set_pipeline(&self.pipelines.scan_local);
            p.dispatch_workgroups(live_wgs, 1, 1);
            p.set_pipeline(&self.pipelines.scan_blocks);
            p.dispatch_workgroups(1, 1, 1);

            if self.cfg.sort_voices {
                // [54]
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

        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    /// Wait for the queued block and take its audio and counters back. \[55\]
    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        // [56]
        let mut bufs: Vec<&wgpu::Buffer> = vec![&self.readback_out, &self.readback_state];
        let period_ns = self.timing.as_ref().map(|t| {
            bufs.push(&t.readback);
            t.period_ns
        });
        let reads = self.map_read_many(&bufs)?;

        let samples: &[f32] = bytemuck::cast_slice(&reads[0]);
        out.copy_from_slice(&samples[..out.len()]);

        let state: &[u32] = bytemuck::cast_slice(&reads[1]);
        self.live = state[S_LIVE].min(self.cfg.max_voices);
        self.stolen = state[S_STOLEN] as u64;
        // [57]
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
        self.pending_spawns.clear();
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
    use super::gates_capacity;

    const GIB: u64 = 1 << 30;

    #[test]
    fn the_gates_buffer_grows_by_doubling_and_never_past_one_binding() {
        assert_eq!(gates_capacity(10, 32768, 2 * GIB).unwrap(), None);
        assert_eq!(gates_capacity(40_000, 32768, 2 * GIB).unwrap(), Some(65536));
        // Doubling 10M runs would need 128 MiB and a bit; clamp to the binding.
        let clamped = gates_capacity(10_000_000, 32768, 128 << 20).unwrap().unwrap();
        assert!((10_000_000..16_777_216).contains(&clamped), "{clamped}");
    }

    /// The count from the crash on 2026-09-13, as entries. It used to wrap the \[58\]
    #[test]
    fn too_many_runs_for_one_binding_is_an_error_that_names_the_sizes() {
        let e = gates_capacity(2_452_415_145, 32768, 2 * GIB).unwrap_err().to_string();
        assert!(e.contains("2452415145 note-off runs"), "{e}");
        assert!(e.contains("lower --block"), "{e}");
    }
}
