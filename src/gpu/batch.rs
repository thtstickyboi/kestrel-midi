// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Several renders in one dispatch: the stems runner's way past the device's \[1\]

use super::*;
use std::sync::Mutex;

/// The most one submission uploads into any one of the packed buffers. A \[2\]
const UPLOAD_MAX: u64 = 512 << 20;

/// Buffers every lane has a region of, as the shaders name them, with the \[3\]
const LANED: &[(&str, &str, &str)] = &[
    ("voices_out", "voices", "lb_voices_out"),
    ("voices", "voices", "lb_voices"),
    ("state", "state", "lb_state"),
    ("partials", "partials", "lb_partials"),
    ("out_block", "out_block", "lb_out"),
    ("cmds", "cmds", "lb_cmds"),
    ("gates", "gates", "lb_gates"),
    ("chan", "chan", "lb_chan"),
    ("scan", "scan", "lb_scan"),
    ("block_sums", "block_sums", "lb_block_sums"),
    ("sort_keys", "sort_keys", "lb_sort_keys"),
    ("pairs_in", "pairs_in", "lb_pairs"),
    ("pairs_out", "pairs_out", "lb_pairs"),
    ("hist", "hist", "lb_hist"),
];

/// Per-lane control words beside the uniforms. Bit 0 of `flags`: the lane is \[4\]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LaneU {
    pub u: Uniforms,
    /// flags, gates base, parity, spawn command base (in commands).
    pub ctl0: [u32; 4],
    /// channel rows base, then spare.
    pub ctl1: [u32; 4],
}

pub(super) const LANE_ACTIVE: u32 = 1;
pub(super) const LANE_STEALS: u32 = 8;

/// What a pass's entry point has to check before it does anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Plain,
    /// Runs only for lanes that steal this block, as the solo render runs the \[5\]
    Steal,
    /// The render pass: only for lanes that need this pipeline, and only its \[6\]
    Render(u32),
}

/// The fixed sizes of a lane's regions, in the element of each buffer.
#[derive(Debug, Clone, Copy)]
pub(super) struct Strides {
    /// One half of a lane's pool: `VOICE_FIELDS * capacity` words.
    pub voice_words: u32,
    pub partials: u32,
    pub out: u32,
    pub capacity: u32,
    pub block_sums: u32,
    /// Variant slots per lane in the slot table.
    pub variants: u32,
}

/// Rewrite one fully substituted solo shader into its batch form. \[7\]
fn laned(src: &str, entries: &[(&str, Role)], s: &Strides) -> Result<String> {
    let uniform_decl = "@group(0) @binding(0) var<uniform> u: Uniforms;";
    if !src.contains(uniform_decl) {
        bail!("batch rewrite: the uniform binding is not where it was");
    }
    let mut out = src.replace(
        uniform_decl,
        &format!(
            "struct LaneU {{ u: Uniforms, ctl0: vec4<u32>, ctl1: vec4<u32>, }}\n\
             @group(0) @binding(0) var<storage, read> lane_u: array<LaneU>;\n\
             @group(0) @binding(12) var<storage, read> lane_list: array<u32>;\n\
             var<private> kb_lane: u32;\n\
             var<private> lb_voices: u32;\n\
             var<private> lb_voices_out: u32;\n\
             var<private> lb_state: u32;\n\
             var<private> lb_partials: u32;\n\
             var<private> lb_out: u32;\n\
             var<private> lb_cmds: u32;\n\
             var<private> lb_gates: u32;\n\
             var<private> lb_chan: u32;\n\
             var<private> lb_scan: u32;\n\
             var<private> lb_block_sums: u32;\n\
             var<private> lb_sort_keys: u32;\n\
             var<private> lb_pairs: u32;\n\
             var<private> lb_hist: u32;\n\
             const LANE_VOICE_WORDS: u32 = {}u;\n\
             const LANE_PARTIALS: u32 = {}u;\n\
             const LANE_OUT: u32 = {}u;\n\
             const LANE_CAPACITY: u32 = {}u;\n\
             const LANE_BLOCK_SUMS: u32 = {}u;\n\
             const LANE_VARIANTS: u32 = {}u;\n",
            s.voice_words, s.partials, s.out, s.capacity, s.block_sums, s.variants,
        ),
    );

    // The second half of the pool is the same buffer: drop its binding.
    for decl in [
        "@group(0) @binding(2) var<storage, read_write> voices_out: array<u32>;",
        "@group(0) @binding(7) var<storage, read_write> voices_out: array<u32>;",
    ] {
        out = out.replace(decl, "");
    }

    // [8]
    if out.contains("var<storage, read> params: array<RegionParams>;") {
        out.push_str(
            "\n@group(0) @binding(11) var<storage, read> slots: array<u32>;\n\
             fn slot_index(i: u32) -> u32 {\n\
                 let per = u.params_per_variant;\n\
                 let v = i / per;\n\
                 return slots[kb_lane * LANE_VARIANTS + v] * per + (i - v * per);\n\
             }\n",
        );
        out = rewrite_index(&out, "params", |e| format!("slot_index({e})"));
        out = rewrite_index(&out, "menv", |e| format!("slot_index({e})"));
    }

    // [9]
    for &(name, buffer, base) in LANED {
        out = rewrite_index(&out, name, |e| format!("{base} + ({e})"));
        if name != buffer {
            out = rename_index(&out, name, &format!("{buffer}__alias"));
        }
    }
    for &(name, buffer, _) in LANED {
        if name != buffer {
            out = rename_index(&out, &format!("{buffer}__alias"), buffer);
        }
    }

    // Every `u.field` reads the lane's own uniforms in place.
    out = rewrite_uniform_reads(&out);

    for &(entry, role) in entries {
        out = add_prologue(&out, entry, role)?;
    }
    Ok(out)
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `u.field` becomes `lane_u[kb_lane].u.field`, for the identifier `u` only.
fn rewrite_uniform_reads(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + src.len() / 16);
    let mut rest = src;
    while let Some(at) = rest.find("u.") {
        let before = rest[..at].chars().next_back();
        out.push_str(&rest[..at]);
        if before.is_some_and(|c| is_ident(c) || c == '.') {
            out.push_str("u.");
        } else {
            out.push_str("lane_u[kb_lane].u.");
        }
        rest = &rest[at + 2..];
    }
    out.push_str(rest);
    out
}

/// Every `name[expr]` indexing the buffer `name` becomes `name[f(expr)]`. Not \[10\]
fn rewrite_index(src: &str, name: &str, f: impl Fn(&str) -> String) -> String {
    let pat = format!("{name}[");
    let mut out = String::with_capacity(src.len() + src.len() / 8);
    let mut rest = src;
    while let Some(at) = rest.find(&pat) {
        let before = rest[..at].chars().next_back();
        if before.is_some_and(|c| is_ident(c) || c == '.') {
            out.push_str(&rest[..at + pat.len()]);
            rest = &rest[at + pat.len()..];
            continue;
        }
        out.push_str(&rest[..at + pat.len()]);
        let inner = &rest[at + pat.len()..];
        let mut depth = 1usize;
        let mut end = 0usize;
        for (i, c) in inner.char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str(&f(&inner[..end]));
        rest = &inner[end..];
    }
    out.push_str(rest);
    out
}

/// `from[` becomes `to[`, for a name that is only an alias of another buffer.
fn rename_index(src: &str, from: &str, to: &str) -> String {
    let pat = format!("{from}[");
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(at) = rest.find(&pat) {
        let before = rest[..at].chars().next_back();
        out.push_str(&rest[..at]);
        if before.is_some_and(|c| is_ident(c) || c == '.') {
            out.push_str(&pat);
        } else {
            out.push_str(to);
            out.push('[');
        }
        rest = &rest[at + pat.len()..];
    }
    out.push_str(rest);
    out
}

/// Give `entry` a workgroup id if it has none, and open it with the lane's \[11\]
fn add_prologue(src: &str, entry: &str, role: Role) -> Result<String> {
    let head = format!("fn {entry}(");
    let at = src
        .match_indices(&head)
        .map(|(i, _)| i)
        .find(|&i| src[..i].trim_end().ends_with(')'))
        .with_context(|| format!("batch rewrite: entry point {entry} not found"))?;
    let params_start = at + head.len();
    let mut depth = 1usize;
    let mut params_end = params_start;
    for (i, c) in src[params_start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    params_end = params_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let params = &src[params_start..params_end];
    let (params, wgid) = match params.find("@builtin(workgroup_id)") {
        Some(p) => {
            let after = &params[p + "@builtin(workgroup_id)".len()..];
            let name = after.trim_start().split(':').next().unwrap_or("").trim().to_string();
            (params.to_string(), name)
        }
        None => {
            let trimmed = params.trim_end();
            let sep = if trimmed.is_empty() || trimmed.ends_with(',') { "" } else { "," };
            (
                format!("{trimmed}{sep}\n    @builtin(workgroup_id) lane_wgid: vec3<u32>,\n"),
                "lane_wgid".to_string(),
            )
        }
    };
    let brace = params_end
        + src[params_end..]
            .find('{')
            .with_context(|| format!("batch rewrite: no body for {entry}"))?;
    let check = match role {
        Role::Plain => String::new(),
        Role::Steal => "    if ((lane_ctl0.x & 8u) == 0u) { return; }\n".to_string(),
        Role::Render(v) => format!(
            "    if (((lane_ctl0.x >> 1u) & 3u) != {v}u || {wgid}.x >= lane_all.u.render_workgroups) {{ return; }}\n"
        ),
    };
    let prologue = format!(
        "\n    kb_lane = lane_list[{wgid}.y];\n\
         \x20   let lane_all = lane_u[kb_lane];\n\
         \x20   let lane_ctl0 = lane_all.ctl0;\n\
         \x20   if ((lane_ctl0.x & 1u) == 0u) {{ return; }}\n\
         {check}\
         \x20   lb_voices = kb_lane * (2u * LANE_VOICE_WORDS) + lane_ctl0.z * LANE_VOICE_WORDS;\n\
         \x20   lb_voices_out = kb_lane * (2u * LANE_VOICE_WORDS) + (1u - lane_ctl0.z) * LANE_VOICE_WORDS;\n\
         \x20   lb_state = kb_lane * {slots}u;\n\
         \x20   lb_partials = kb_lane * LANE_PARTIALS;\n\
         \x20   lb_out = kb_lane * LANE_OUT;\n\
         \x20   lb_cmds = lane_ctl0.w;\n\
         \x20   lb_gates = lane_ctl0.y;\n\
         \x20   lb_chan = lane_all.ctl1.x;\n\
         \x20   lb_scan = kb_lane * LANE_CAPACITY;\n\
         \x20   lb_block_sums = kb_lane * LANE_BLOCK_SUMS;\n\
         \x20   lb_sort_keys = kb_lane * LANE_CAPACITY;\n\
         \x20   lb_pairs = kb_lane * LANE_CAPACITY;\n\
         \x20   lb_hist = kb_lane * 256u;\n",
        slots = STATE_SLOTS
    );
    let mut out = String::with_capacity(src.len() + prologue.len() + 64);
    out.push_str(&src[..params_start]);
    out.push_str(&params);
    out.push_str(&src[params_end..=brace]);
    out.push_str(&prologue);
    out.push_str(&src[brace + 1..]);
    Ok(out)
}

/// Every pass's source in batch form, each with the entry points it compiles.
pub(super) struct BatchSources {
    pub spawn: String,
    pub render: [String; 4],
    pub reduce: String,
    pub compact: String,
    pub select: String,
    pub sort: String,
}

pub(super) fn batch_sources(cfg: &Config, bank: &Bank, s: &Strides) -> Result<BatchSources> {
    use Role::*;
    let src = |body: &str| shader_source(body, cfg, bank);
    let render = |chan: bool, glide: bool| {
        let v = chan as u32 | ((glide as u32) << 1);
        laned(&render_source(cfg, bank, chan, glide), &[("main", Render(v))], s)
    };
    Ok(BatchSources {
        spawn: laned(&src(include_str!("../../shaders/spawn.wgsl")), &[("main", Plain), ("commit", Plain)], s)?,
        render: [render(false, false)?, render(true, false)?, render(false, true)?, render(true, true)?],
        reduce: laned(&src(include_str!("../../shaders/reduce.wgsl")), &[("main", Plain)], s)?,
        compact: laned(
            &src(include_str!("../../shaders/compact.wgsl")),
            &[
                ("scan_local", Plain),
                ("scan_blocks", Plain),
                ("scatter", Plain),
                ("commit", Plain),
                ("mark_stolen", Steal),
                ("note_stolen", Steal),
            ],
            s,
        )?,
        select: laned(
            &src(include_str!("../../shaders/select.wgsl")),
            &[("clear", Steal), ("init", Steal), ("histogram", Steal), ("refine", Steal)],
            s,
        )?,
        sort: laned(
            &src(include_str!("../../shaders/sort.wgsl")),
            &[
                ("init", Plain),
                ("advance_bit", Plain),
                ("build_keys", Plain),
                ("scan_local", Plain),
                ("scan_blocks", Plain),
                ("split", Plain),
                ("gather", Plain),
            ],
            s,
        )?,
    })
}

/// A bind group layout whose binding 0 is the lanes' uniforms, read-only \[12\]
fn layout(device: &wgpu::Device, label: &str, entries: &[(u32, bool)]) -> wgpu::BindGroupLayout {
    let mut all = vec![(0u32, true), (12u32, true)];
    all.extend_from_slice(entries);
    let entries: Vec<wgpu::BindGroupLayoutEntry> = all
        .iter()
        .map(|&(binding, read_only)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries })
}

/// A bind group of whole buffers at the given bindings.
fn group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, buffers: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .map(|&(binding, b)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

pub(super) struct BatchLayouts {
    spawn: wgpu::BindGroupLayout,
    render: wgpu::BindGroupLayout,
    reduce: wgpu::BindGroupLayout,
    compact: wgpu::BindGroupLayout,
    select: wgpu::BindGroupLayout,
    sort: wgpu::BindGroupLayout,
}

impl BatchLayouts {
    /// The solo layouts with binding 0 as the lanes' uniforms, the second half \[13\]
    fn new(device: &wgpu::Device) -> Self {
        BatchLayouts {
            spawn: layout(device, "batch spawn", &[(1, true), (2, false), (3, false)]),
            render: layout(
                device,
                "batch render",
                &[
                    (1, true),
                    (2, true),
                    (3, true),
                    (4, false),
                    (5, false),
                    (6, true),
                    (7, true),
                    (8, true),
                    (9, true),
                    (11, true),
                ],
            ),
            reduce: layout(device, "batch reduce", &[(1, true), (2, false)]),
            compact: layout(device, "batch compact", &[(1, false), (3, false), (4, false), (5, false), (6, false)]),
            select: layout(device, "batch select", &[(1, true), (2, false), (3, false)]),
            sort: layout(device, "batch sort", &[(1, false), (2, false), (3, false), (4, false), (5, false), (6, false)]),
        }
    }
}

pub(super) struct BatchPipelines {
    spawn: wgpu::ComputePipeline,
    spawn_commit: wgpu::ComputePipeline,
    /// Indexed by `LaneU::ctl0` bits 1-2: plain, controllers, glide, both.
    render: Vec<wgpu::ComputePipeline>,
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

impl BatchPipelines {
    fn new(device: &wgpu::Device, cfg: &Config, src: BatchSources, l: &BatchLayouts) -> Self {
        let mut spawn = compile(device, cfg, "batch spawn", src.spawn, &l.spawn, &["main", "commit"]);
        let [r0, r1, r2, r3] = src.render;
        let render = [r0, r1, r2, r3]
            .into_iter()
            .enumerate()
            .map(|(i, s)| compile(device, cfg, &format!("batch render {i}"), s, &l.render, &["main"]).remove(0))
            .collect();
        let mut reduce = compile(device, cfg, "batch reduce", src.reduce, &l.reduce, &["main"]);
        let mut compact = compile(
            device,
            cfg,
            "batch compact",
            src.compact,
            &l.compact,
            &["scan_local", "scan_blocks", "scatter", "commit", "mark_stolen", "note_stolen"],
        );
        let mut select =
            compile(device, cfg, "batch select", src.select, &l.select, &["clear", "init", "histogram", "refine"]);
        let mut sort = compile(
            device,
            cfg,
            "batch sort",
            src.sort,
            &l.sort,
            &["init", "advance_bit", "build_keys", "scan_local", "scan_blocks", "split", "gather"],
        );
        BatchPipelines {
            spawn_commit: spawn.remove(1),
            spawn: spawn.remove(0),
            render,
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
        }
    }
}

// ---- the batch -------------------------------------------------------------

/// One lane's host state: what a solo `GpuSynth` keeps in its own fields, plus \[14\]
#[derive(Default)]
struct Lane {
    live: u32,
    parity: u32,
    env_phase: u32,
    stolen: u64,
    dropped: u64,
    peak: f32,
    pending: Vec<SpawnCmd>,
    scratch: Vec<SpawnCmd>,
    bend: bool,
    gain: bool,
    variant: bool,
    cut: bool,
    glide: bool,
    chan_count: u32,
    /// The note-off table, meta then runs, as it goes up.
    gates: Vec<u32>,
    off_meta_words: u32,
    rows: Vec<u32>,
    /// Params slot of each variant this lane has installed, 0 for none: \[15\]
    slots: Vec<u32>,
    /// `submit` has been called for the block the batch is gathering.
    submitted: bool,
    /// Planned when the batch goes up.
    spawn_count: u32,
    steal_k: u32,
    nwg: u32,
    live_before: u32,
    thinned: bool,
    /// The block the last batch rendered for this lane, until `finish`.
    out: Vec<f32>,
    delivered: bool,
}

/// What every lane's backend reaches besides its own `Lane`: the settings, the \[16\]
struct LaneHost {
    cfg: Config,
    device: wgpu::Device,
    queue: wgpu::Queue,
    binding_cap: u64,
    state_buf: wgpu::Buffer,
    slots_buf: wgpu::Buffer,
    params_per_variant: u32,
    menv_per_variant: u32,
    slots: Mutex<SlotPool>,
}

/// The params and modulation-envelope slots, and the buffers holding them. \[17\]
struct SlotPool {
    count: u32,
    free: Vec<u32>,
    params_buf: wgpu::Buffer,
    menv_buf: wgpu::Buffer,
    /// The buffers have been replaced since the bind groups were built.
    grown: bool,
}

/// Bind groups over the batch's buffers, rebuilt whenever one of them grows.
struct BatchGroups {
    spawn: wgpu::BindGroup,
    render: wgpu::BindGroup,
    reduce: wgpu::BindGroup,
    compact: wgpu::BindGroup,
    select: wgpu::BindGroup,
    sort: [wgpu::BindGroup; 2],
}

/// Up to `lanes` renders on one device, each advanced a block at a time and \[18\]
pub struct GpuBatch {
    cfg: Config,
    device: wgpu::Device,
    queue: wgpu::Queue,
    concurrent: bool,
    binding_cap: u64,
    lanes: Vec<Lane>,
    strides: Strides,
    shape: Shape,
    max_nwg: u32,
    layouts: BatchLayouts,
    pipes: BatchPipelines,
    lane_u_buf: wgpu::Buffer,
    /// The lanes in the batch being sent, which the grid's second axis \[19\]
    lane_list_buf: wgpu::Buffer,
    voices_buf: wgpu::Buffer,
    partials_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    scan_buf: wgpu::Buffer,
    block_sums_buf: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    sort_keys_buf: wgpu::Buffer,
    pairs: [wgpu::Buffer; 2],
    /// The per-block uploads -- spawn commands, note-off tables, channel \[20\]
    cmds_buf: wgpu::Buffer,
    /// Spawn commands `cmds_buf` holds.
    cmds_cap: u64,
    gates_buf: wgpu::Buffer,
    /// Words `gates_buf` holds.
    gates_cap: u64,
    chan_buf: wgpu::Buffer,
    /// Words `chan_buf` holds.
    chan_cap: u64,
    slots_buf: wgpu::Buffer,
    /// What the lanes' backends share, apart from the lanes themselves.
    host: LaneHost,
    pool_buf: wgpu::Buffer,
    menv_factor_buf: wgpu::Buffer,
    readback_out: wgpu::Buffer,
    readback_state: wgpu::Buffer,
    groups: Option<BatchGroups>,
    /// Lanes in the batch last submitted, waiting for `wait`.
    in_flight: Vec<usize>,
    last_submission: Option<wgpu::SubmissionIndex>,
    /// Under `--profile`: device timestamps around each stage, and what they \[21\]
    timing: Option<Timing>,
    pass_ms: [f64; 5],
    timed: u64,
    /// Batches that had to go up as more than one submission.
    split: u64,
    /// `UPLOAD_MAX`, lowered by tests to make a small batch split.
    upload_max: u64,
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4), usage, mapped_at_creation: false })
}

const STORAGE_RW: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);

/// Lanes one batch holds at most. Each lane is one track and a batch is one \[22\]
pub const LANES_MAX: usize = 256;

/// The batch's buffers that each have to fit in one binding, for `lanes` \[23\]
fn one_binding(cfg: &Config, bank: &Bank, lanes: u64) -> [(&'static str, u64); 4] {
    let capacity = cfg.pool_slots() as u64;
    let params = bank.params.len().max(1) as u64 * std::mem::size_of::<RegionParams>() as u64;
    [
        // Both halves of every lane's ping-pong pool.
        ("voice pool", lanes * 2 * voice_fields(cfg) * capacity * 4),
        ("partials", lanes * (cfg.block_frames * 2 * max_render_workgroups(cfg)) as u64 * 4),
        ("sort pairs", lanes * capacity * 8),
        // The bank's own table and a spare slot a lane.
        ("params", (1 + lanes) * params),
    ]
}

fn binds(cfg: &Config, bank: &Bank, lanes: u64, binding_bytes: u64) -> bool {
    one_binding(cfg, bank, lanes).iter().all(|&(_, bytes)| bytes <= binding_bytes)
}

impl GpuBatch {
    /// Device bytes one lane costs at `cfg`, before any lane has asked for a \[24\]
    pub fn lane_bytes(cfg: &Config, bank: &Bank) -> u64 {
        let capacity = cfg.pool_slots() as u64;
        let voice = voice_fields(cfg) * capacity * 4 * 2;
        let partials = cfg.block_frames as u64 * 2 * max_render_workgroups(cfg) as u64 * 4;
        let out = cfg.block_frames as u64 * 2 * 4 * 2;
        let sort = capacity * (4 + 4 + 16);
        let cmds = 1024u64.min(capacity).max(64) * std::mem::size_of::<SpawnCmd>() as u64;
        let gates = ((BASE_CHANNELS as u64 * 128 + 1) * 2 + 4096 * 2) * 4;
        let chan = (cfg.block_frames / cfg.gate_frames + 1) as u64 * BASE_CHANNELS as u64 * CHAN_FIELDS as u64 * 4;
        // Its spare params slot.
        let params = bank.params.len().max(1) as u64 * std::mem::size_of::<RegionParams>() as u64
            + bank.menv.len().max(1) as u64 * std::mem::size_of::<ModEnvParams>() as u64;
        voice + partials + out + sort + cmds + gates + chan + params + 4096
    }

    /// The most voices each of `lanes` lanes can hold in one batch at `cfg` \[25\]
    pub fn max_voices_each(cfg: &Config, bank: &Bank, binding_bytes: u64, lanes: usize) -> u32 {
        let lanes = lanes.max(1) as u64;
        let fits = |v: u32| binds(&Config { max_voices: v, ..cfg.clone() }, bank, lanes, binding_bytes);
        // [26]
        let (mut lo, mut hi) = (0u32, max_voices_for_config(binding_bytes, cfg).max(1));
        if fits(hi) {
            return hi;
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The most lanes, up to `lanes`, whose batch at `cfg` fits an adapter \[27\]
    pub fn lanes_that_bind(cfg: &Config, bank: &Bank, binding_bytes: u64, lanes: usize) -> usize {
        (1..=lanes).rev().find(|&l| binds(cfg, bank, l as u64, binding_bytes)).unwrap_or(0)
    }

    /// `lanes` renders at `cfg` on `shared`'s device and sample pool.
    pub fn new(cfg: &Config, bank: &Bank, shared: &GpuShared, lanes: usize) -> Result<Self> {
        cfg.validate()?;
        if cfg.phase.active() {
            bail!("analytic phase rotation cannot run in a batch");
        }
        let lanes = lanes.max(1);
        let binding_cap = check_limits(cfg, &shared.limits, &shared.adapter_name())?;
        let device = shared.device.clone();
        let queue = shared.queue.clone();
        let capacity = cfg.pool_slots();
        let strides = Strides {
            voice_words: (voice_fields(cfg) * capacity as u64) as u32,
            partials: cfg.block_frames * 2 * max_render_workgroups(cfg),
            out: cfg.block_frames * 2,
            capacity,
            block_sums: capacity.div_ceil(cfg.workgroup_size),
            variants: cfg.max_param_variants.max(1),
        };
        let l = lanes as u64;
        let [voices_bytes, partials_bytes, pairs_bytes, params_bytes] = one_binding(cfg, bank, l).map(|(what, bytes)| {
            if bytes > binding_cap {
                bail!(
                    "{lanes} lanes need {:.2} GiB of {what} in one binding and the adapter binds \
                     {:.2} GiB; use fewer lanes",
                    bytes as f64 / (1u64 << 30) as f64,
                    binding_cap as f64 / (1u64 << 30) as f64
                );
            }
            Ok(bytes)
        });
        let voices_buf = buffer(&device, "batch voices", voices_bytes?, STORAGE_RW);
        let partials_buf = buffer(&device, "batch partials", partials_bytes?, STORAGE_RW);
        let out_bytes = l * strides.out as u64 * 4;
        let out_buf = buffer(&device, "batch out", out_bytes, STORAGE_RW | wgpu::BufferUsages::COPY_SRC);
        let state_bytes = l * STATE_SLOTS as u64 * 4;
        let state_buf = buffer(&device, "batch state", state_bytes, STORAGE_RW | wgpu::BufferUsages::COPY_SRC);
        let scan_buf = buffer(&device, "batch scan", l * capacity as u64 * 4, STORAGE_RW);
        let block_sums_buf = buffer(&device, "batch block sums", l * strides.block_sums as u64 * 4, STORAGE_RW);
        let hist_buf = buffer(&device, "batch histogram", l * 256 * 4, STORAGE_RW);
        let sort_keys_buf = buffer(&device, "batch sort keys", l * capacity as u64 * 4, STORAGE_RW);
        let pairs = [
            buffer(&device, "batch pairs a", pairs_bytes?, STORAGE_RW),
            buffer(&device, "batch pairs b", l * capacity as u64 * 8, STORAGE_RW),
        ];
        // [28]
        let cmds_cap = l * 1024u64.min(capacity as u64).max(64);
        let cmds_buf = buffer(
            &device,
            "batch spawn commands",
            cmds_cap * std::mem::size_of::<SpawnCmd>() as u64,
            STORAGE_RW,
        );
        let gates_cap = l * (((BASE_CHANNELS as u64 * 128 + 1) * 2) + 4096 * 2);
        let gates_buf = buffer(&device, "batch gates", gates_cap * 4, STORAGE_RW);
        let chan_cap = l * ((cfg.block_frames / cfg.gate_frames + 1) as u64 * BASE_CHANNELS as u64 * CHAN_FIELDS as u64);
        let chan_buf = buffer(&device, "batch channels", chan_cap * 4, STORAGE_RW);
        let lane_list_buf = buffer(&device, "batch lane list", l * 4, STORAGE_RW);
        let lane_u_buf =
            buffer(&device, "batch lane uniforms", l * std::mem::size_of::<LaneU>() as u64, STORAGE_RW);

        // [29]
        let fallback = [RegionParams {
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
        let params: &[RegionParams] = if bank.params.is_empty() { &fallback } else { &bank.params };
        let menv: Vec<ModEnvParams> =
            if bank.menv.is_empty() { vec![ModEnvParams::default()] } else { bank.menv.clone() };
        let slot_count = 1 + lanes as u32;
        let params_buf = buffer(&device, "batch params", params_bytes?, STORAGE_RW | wgpu::BufferUsages::COPY_SRC);
        upload_in_pieces(&device, &queue, &params_buf, 0, bytemuck::cast_slice(params))?;
        let menv_buf = buffer(
            &device,
            "batch mod env",
            slot_count as u64 * menv.len() as u64 * std::mem::size_of::<ModEnvParams>() as u64,
            STORAGE_RW | wgpu::BufferUsages::COPY_SRC,
        );
        queue.write_buffer(&menv_buf, 0, bytemuck::cast_slice(&menv));
        let slots_buf = buffer(&device, "batch variant slots", l * strides.variants as u64 * 4, STORAGE_RW);

        let (tables, menv_log2_base, glide_base) = read_only_tables(cfg, bank);
        let menv_factor_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch mod env tables"),
            contents: bytemuck::cast_slice(&tables),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let readback = |label: &str, size: u64| {
            buffer(&device, label, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)
        };
        let readback_out = readback("batch readback out", out_bytes);
        let readback_state = readback("batch readback state", state_bytes);

        let shape = Shape {
            slots: capacity,
            pool_words: (shared.pool_buf.size() / 4) as u32,
            sort_key: SortKeyLayout::plan(bank),
            params_per_variant: params.len() as u32,
            menv_factor_half: bank.menv_factor_half,
            menv_log2_base,
            glide_base,
        };
        let layouts = BatchLayouts::new(&device);
        let pipes = BatchPipelines::new(&device, cfg, batch_sources(cfg, bank, &strides)?, &layouts);

        let mut b = GpuBatch {
            cfg: cfg.clone(),
            device,
            queue,
            concurrent: shared.concurrent,
            binding_cap,
            lanes: (0..lanes)
                .map(|_| Lane { slots: vec![0; strides.variants as usize], chan_count: BASE_CHANNELS as u32, ..Default::default() })
                .collect(),
            strides,
            shape,
            max_nwg: max_render_workgroups(cfg),
            layouts,
            pipes,
            lane_u_buf,
            lane_list_buf,
            voices_buf,
            partials_buf,
            out_buf,
            state_buf: state_buf.clone(),
            scan_buf,
            block_sums_buf,
            hist_buf,
            sort_keys_buf,
            pairs,
            cmds_buf,
            cmds_cap,
            gates_buf,
            gates_cap,
            chan_buf,
            chan_cap,
            slots_buf: slots_buf.clone(),
            host: LaneHost {
                cfg: cfg.clone(),
                device: shared.device.clone(),
                queue: shared.queue.clone(),
                binding_cap,
                state_buf,
                slots_buf,
                params_per_variant: params.len() as u32,
                menv_per_variant: menv.len() as u32,
                slots: Mutex::new(SlotPool {
                    count: slot_count,
                    free: (1..slot_count).rev().collect(),
                    params_buf,
                    menv_buf,
                    grown: false,
                }),
            },
            pool_buf: shared.pool_buf.clone(),
            menv_factor_buf,
            readback_out,
            readback_state,
            groups: None,
            in_flight: Vec::new(),
            last_submission: None,
            timing: None,
            pass_ms: [0.0; 5],
            timed: 0,
            split: 0,
            upload_max: UPLOAD_MAX,
        };
        if cfg.profile && shared.has_timestamps {
            let set = b.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("batch pass timings"),
                ty: wgpu::QueryType::Timestamp,
                count: TIMESTAMP_COUNT,
            });
            b.timing = Some(Timing {
                set,
                resolve: buffer(&b.device, "batch timestamp resolve", TIMESTAMP_COUNT as u64 * 8,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                readback: buffer(&b.device, "batch timestamp readback", TIMESTAMP_COUNT as u64 * 8,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
                period_ns: b.queue.get_timestamp_period(),
            });
        }
        b.groups = Some(b.build_groups());
        Ok(b)
    }

    pub fn lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Batches so far that went up as more than one submission.
    pub fn split_batches(&self) -> u64 {
        self.split
    }

    #[cfg(test)]
    fn set_upload_max(&mut self, bytes: u64) {
        self.upload_max = bytes;
    }

    /// Under `--profile`, the device time each stage took on average over the \[30\]
    pub fn pass_times(&self) -> Option<Vec<(&'static str, f64)>> {
        (self.timed > 0).then(|| {
            PASS_NAMES.iter().zip(self.pass_ms).map(|(n, ms)| (*n, ms / self.timed as f64)).collect()
        })
    }

    fn build_groups(&self) -> BatchGroups {
        let d = &self.device;
        let l = &self.layouts;
        let u = &self.lane_u_buf;
        let ll = &self.lane_list_buf;
        let slots = self.host.slots.lock().unwrap();
        BatchGroups {
            spawn: group(d, &l.spawn, &[(0, u), (12, ll), (1, &self.cmds_buf), (2, &self.voices_buf), (3, &self.state_buf)]),
            render: group(
                d,
                &l.render,
                &[
                    (0, u),
                    (12, ll),
                    (1, &self.pool_buf),
                    (2, &slots.params_buf),
                    (3, &self.gates_buf),
                    (4, &self.voices_buf),
                    (5, &self.partials_buf),
                    (6, &self.state_buf),
                    (7, &self.chan_buf),
                    (8, &slots.menv_buf),
                    (9, &self.menv_factor_buf),
                    (11, &self.slots_buf),
                ],
            ),
            reduce: group(d, &l.reduce, &[(0, u), (12, ll), (1, &self.partials_buf), (2, &self.out_buf)]),
            compact: group(
                d,
                &l.compact,
                &[
                    (0, u),
                    (12, ll),
                    (1, &self.voices_buf),
                    (3, &self.scan_buf),
                    (4, &self.block_sums_buf),
                    (5, &self.state_buf),
                    (6, &self.sort_keys_buf),
                ],
            ),
            select: group(d, &l.select, &[(0, u), (12, ll), (1, &self.voices_buf), (2, &self.state_buf), (3, &self.hist_buf)]),
            sort: [0usize, 1].map(|p| {
                group(
                    d,
                    &l.sort,
                    &[
                        (0, u),
                        (12, ll),
                        (1, &self.pairs[p]),
                        (2, &self.pairs[1 - p]),
                        (3, &self.scan_buf),
                        (4, &self.block_sums_buf),
                        (5, &self.state_buf),
                        (6, &self.voices_buf),
                    ],
                )
            }),
        }
    }

    /// Make `lane` a fresh render; see `LaneBackend::reset`.
    pub fn reset_lane(&mut self, lane: usize) {
        self.lane(lane).reset();
    }

    /// The `Backend` a driver renders `lane` through.
    pub fn lane(&mut self, lane: usize) -> LaneBackend<'_> {
        LaneBackend { lane: &mut self.lanes[lane], host: &self.host, index: lane }
    }

    /// Every lane's backend at once, to be driven from as many threads as \[31\]
    pub fn lanes_mut(&mut self) -> Vec<LaneBackend<'_>> {
        let host = &self.host;
        self.lanes.iter_mut().enumerate().map(|(index, lane)| LaneBackend { lane, host, index }).collect()
    }

    /// Whether any lane has submitted a block the batch has not sent yet.
    pub fn pending(&self) -> bool {
        self.lanes.iter().any(|l| l.submitted)
    }

    /// Whether `lane`'s driver handed its last block to the batch -- false for \[32\]
    pub fn lane_submitted(&self, lane: usize) -> bool {
        self.lanes[lane].submitted
    }

    /// Send every submitted lane's block to the device. Returns false when no \[33\]
    pub fn flush(&mut self) -> Result<bool> {
        if !self.in_flight.is_empty() {
            bail!("a batch is already on the device");
        }
        let active: Vec<usize> = (0..self.lanes.len()).filter(|&i| self.lanes[i].submitted).collect();
        if active.is_empty() {
            return Ok(false);
        }
        let cfg = self.cfg.clone();

        // Plan each lane's block exactly as a solo render plans its own.
        for &i in &active {
            let l = &mut self.lanes[i];
            let plan = plan_spawns(&cfg, l.live, &l.pending, &mut l.scratch);
            l.dropped += plan.dropped;
            l.steal_k = plan.steal_k;
            l.spawn_count = plan.spawn_count;
            l.thinned = plan.thinned;
            l.live_before = l.live;
            l.nwg = (l.live + plan.spawn_count).div_ceil(cfg.workgroup_size).clamp(1, self.max_nwg);
        }

        // Split into submissions whose packed uploads fit.
        let cmd_bytes = std::mem::size_of::<SpawnCmd>() as u64;
        // [34]
        let limit = self.upload_max.min(self.binding_cap);
        let sizes = |l: &Lane| (l.spawn_count as u64 * cmd_bytes, l.gates.len() as u64 * 4, l.rows.len() as u64 * 4);
        let mut parts: Vec<Vec<usize>> = vec![Vec::new()];
        let mut sum = (0u64, 0u64, 0u64);
        for &i in &active {
            let (c, g, r) = sizes(&self.lanes[i]);
            if c.max(g).max(r) > self.binding_cap {
                bail!(
                    "one track's block needs {:.2} GiB of uploads in one binding, and the adapter \
                     binds {:.2} GiB; lower --block",
                    c.max(g).max(r) as f64 / (1u64 << 30) as f64,
                    self.binding_cap as f64 / (1u64 << 30) as f64
                );
            }
            let next = (sum.0 + c, sum.1 + g, sum.2 + r);
            if next.0.max(next.1).max(next.2) > limit && !parts.last().unwrap().is_empty() {
                parts.push(Vec::new());
                sum = (c, g, r);
            } else {
                sum = next;
            }
            parts.last_mut().unwrap().push(i);
        }

        // [35]
        let need = |f: &dyn Fn(&Lane) -> u64| {
            parts.iter().map(|p| p.iter().map(|&i| f(&self.lanes[i])).sum::<u64>()).max().unwrap_or(0)
        };
        let need_cmds = need(&|l| l.spawn_count as u64);
        let need_gates = need(&|l| l.gates.len() as u64);
        let need_chan = need(&|l| l.rows.len() as u64);
        let grow = |need: u64, unit: u64| need.next_power_of_two().min(limit / unit).max(need);
        // A lane may have regrown the variant slots since the last batch.
        let mut regrouped = std::mem::take(&mut self.host.slots.get_mut().unwrap().grown);
        if need_cmds > self.cmds_cap {
            self.cmds_cap = grow(need_cmds, cmd_bytes);
            self.cmds_buf = buffer(&self.device, "batch spawn commands", self.cmds_cap * cmd_bytes, STORAGE_RW);
            regrouped = true;
        }
        if need_gates > self.gates_cap {
            self.gates_cap = grow(need_gates, 4);
            self.gates_buf = buffer(&self.device, "batch gates", self.gates_cap * 4, STORAGE_RW);
            regrouped = true;
        }
        if need_chan > self.chan_cap {
            self.chan_cap = grow(need_chan, 4);
            self.chan_buf = buffer(&self.device, "batch channels", self.chan_cap * 4, STORAGE_RW);
            regrouped = true;
        }
        if regrouped {
            self.groups = Some(self.build_groups());
        }

        let last = parts.len() - 1;
        for (n, part) in parts.iter().enumerate() {
            // [36]
            let mut lane_u = vec![LaneU::default(); self.lanes.len()];
            let (mut at_cmds, mut at_gates, mut at_chan) = (0u64, 0u64, 0u64);
            for &i in part {
                let l = &mut self.lanes[i];
                if l.spawn_count > 0 {
                    let cmds = if l.thinned { &l.scratch } else { &l.pending };
                    self.queue.write_buffer(&self.cmds_buf, at_cmds * cmd_bytes, bytemuck::cast_slice(&cmds[..l.spawn_count as usize]));
                }
                if !l.gates.is_empty() {
                    self.queue.write_buffer(&self.gates_buf, at_gates * 4, bytemuck::cast_slice(&l.gates));
                }
                if !l.rows.is_empty() {
                    self.queue.write_buffer(&self.chan_buf, at_chan * 4, bytemuck::cast_slice(&l.rows));
                }
                let block = BlockU {
                    spawn_count: l.spawn_count,
                    steal_k: l.steal_k,
                    nwg: l.nwg,
                    bend: l.bend,
                    gain: l.gain,
                    variant: l.variant,
                    cut: l.cut,
                    env_phase: l.env_phase,
                    chan_count: l.chan_count,
                    off_meta_words: l.off_meta_words,
                };
                // The same test `GpuSynth::submit` uses to pick its pipeline.
                let chan = l.bend || l.gain || l.variant;
                let flags = LANE_ACTIVE | ((chan as u32) << 1) | ((l.glide as u32) << 2) | if l.steal_k > 0 { LANE_STEALS } else { 0 };
                lane_u[i] = LaneU {
                    u: make_uniforms(&cfg, &self.shape, &block),
                    ctl0: [flags, at_gates as u32, l.parity, at_cmds as u32],
                    ctl1: [at_chan as u32, 0, 0, 0],
                };
                at_cmds += l.spawn_count as u64;
                at_gates += l.gates.len() as u64;
                at_chan += l.rows.len() as u64;
                // Written, so the next block's position can be taken now.
                l.env_phase = (l.env_phase + cfg.block_frames) % cfg.env_step_frames();
                l.live += l.spawn_count;
            }
            self.queue.write_buffer(&self.lane_u_buf, 0, bytemuck::cast_slice(&lane_u));
            self.queue.write_buffer(&self.lane_list_buf, 0, bytemuck::cast_slice(&part.iter().map(|&i| i as u32).collect::<Vec<u32>>()));
            let mut enc = self.encode(part);
            if n == last {
                // [37]
                let out_bytes = self.strides.out as u64 * 4;
                for (j, &i) in active.iter().enumerate() {
                    enc.copy_buffer_to_buffer(&self.out_buf, i as u64 * out_bytes, &self.readback_out, j as u64 * out_bytes, out_bytes);
                }
                enc.copy_buffer_to_buffer(&self.state_buf, 0, &self.readback_state, 0, self.state_buf.size());
                // [38]
                if let Some(t) = &self.timing {
                    enc.resolve_query_set(&t.set, 0..TIMESTAMP_COUNT, &t.resolve, 0);
                    enc.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, t.resolve.size());
                }
            }
            self.last_submission = Some(self.queue.submit(Some(enc.finish())));
        }
        if parts.len() > 1 {
            self.split += 1;
        }

        for &i in &active {
            let l = &mut self.lanes[i];
            l.submitted = false;
            l.pending.clear();
        }
        self.in_flight = active;
        Ok(true)
    }

    /// The five passes over the lanes in `part`, recorded.
    fn encode(&self, part: &[usize]) -> wgpu::CommandEncoder {
        let cfg = &self.cfg;
        let y = part.len() as u32;
        let grid = |f: &dyn Fn(&Lane) -> Option<u32>| part.iter().filter_map(|&i| f(&self.lanes[i])).max();
        let steal_x = grid(&|l| (l.steal_k > 0).then(|| dispatch_count(cfg, l.live_before)));
        let spawn_x = grid(&|l| (l.spawn_count > 0).then(|| dispatch_count(cfg, l.spawn_count)));
        let live_x = grid(&|l| Some(dispatch_count(cfg, l.live))).unwrap_or(1);
        let render_x: Vec<Option<u32>> = (0..4u32)
            .map(|v| {
                grid(&|l| {
                    let chan = l.bend || l.gain || l.variant;
                    ((chan as u32 | ((l.glide as u32) << 1)) == v).then_some(l.nwg)
                })
            })
            .collect();

        let p = &self.pipes;
        let g = self.groups.as_ref().expect("built in new");
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("batch"), timestamp_writes: self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }) });

            // ---- 1. steal, for the lanes that steal ----
            if let Some(x) = steal_x {
                pass.set_bind_group(0, &g.select, &[]);
                pass.set_pipeline(&p.sel_init);
                pass.dispatch_workgroups(1, y, 1);
                for _ in 0..8 {
                    pass.set_pipeline(&p.sel_clear);
                    pass.dispatch_workgroups(1, y, 1);
                    pass.set_pipeline(&p.sel_histogram);
                    pass.dispatch_workgroups(x, y, 1);
                    pass.set_pipeline(&p.sel_refine);
                    pass.dispatch_workgroups(1, y, 1);
                }
                pass.set_bind_group(0, &g.compact, &[]);
                pass.set_pipeline(&p.mark_stolen);
                pass.dispatch_workgroups(x, y, 1);
                pass.set_pipeline(&p.note_stolen);
                pass.dispatch_workgroups(1, y, 1);
            }

            drop(pass);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("batch"), timestamp_writes: self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(2), end_of_pass_write_index: Some(3) }) });
            // ---- 2. spawn ----
            pass.set_bind_group(0, &g.spawn, &[]);
            if let Some(x) = spawn_x {
                pass.set_pipeline(&p.spawn);
                pass.dispatch_workgroups(x, y, 1);
            }
            pass.set_pipeline(&p.spawn_commit);
            pass.dispatch_workgroups(1, y, 1);

            drop(pass);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("batch"), timestamp_writes: self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(4), end_of_pass_write_index: Some(5) }) });
            // ---- 3. render, once per pipeline some lane needs ----
            pass.set_bind_group(0, &g.render, &[]);
            for (v, x) in render_x.iter().enumerate() {
                if let Some(x) = *x {
                    pass.set_pipeline(&p.render[v]);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }

            drop(pass);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("batch"), timestamp_writes: self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(6), end_of_pass_write_index: Some(7) }) });
            // ---- 4. reduce ----
            pass.set_bind_group(0, &g.reduce, &[]);
            pass.set_pipeline(&p.reduce);
            pass.dispatch_workgroups(cfg.block_frames * 2, y, 1);

            drop(pass);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("batch"), timestamp_writes: self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(8), end_of_pass_write_index: Some(9) }) });
            // ---- 5. compact, and re-sort in the same pass ----
            pass.set_bind_group(0, &g.compact, &[]);
            pass.set_pipeline(&p.scan_local);
            pass.dispatch_workgroups(live_x, y, 1);
            pass.set_pipeline(&p.scan_blocks);
            pass.dispatch_workgroups(1, y, 1);
            if cfg.sort_voices {
                let mut pair = 0usize;
                pass.set_bind_group(0, &g.sort[pair], &[]);
                pass.set_pipeline(&p.sort_init);
                pass.dispatch_workgroups(1, y, 1);
                pass.set_pipeline(&p.sort_build_keys);
                pass.dispatch_workgroups(live_x, y, 1);
                for _ in 0..self.shape.sort_key.bits {
                    pass.set_bind_group(0, &g.sort[pair], &[]);
                    pass.set_pipeline(&p.sort_scan_local);
                    pass.dispatch_workgroups(live_x, y, 1);
                    pass.set_pipeline(&p.sort_scan_blocks);
                    pass.dispatch_workgroups(1, y, 1);
                    pass.set_pipeline(&p.sort_split);
                    pass.dispatch_workgroups(live_x, y, 1);
                    pass.set_pipeline(&p.sort_advance);
                    pass.dispatch_workgroups(1, y, 1);
                    pair ^= 1;
                }
                pass.set_bind_group(0, &g.sort[pair], &[]);
                pass.set_pipeline(&p.sort_gather);
                pass.dispatch_workgroups(live_x, y, 1);
            } else {
                pass.set_pipeline(&p.scatter);
                pass.dispatch_workgroups(live_x, y, 1);
            }
            pass.set_bind_group(0, &g.compact, &[]);
            pass.set_pipeline(&p.compact_commit);
            pass.dispatch_workgroups(1, y, 1);
        }
        enc
    }

    /// Wait for the batch `flush` sent and hand each of its lanes its block.
    pub fn wait(&mut self) -> Result<()> {
        if self.in_flight.is_empty() {
            return Ok(());
        }
        let out_bytes = self.in_flight.len() as u64 * self.strides.out as u64 * 4;
        let mut bufs = vec![(&self.readback_out, out_bytes), (&self.readback_state, self.readback_state.size())];
        if let Some(t) = &self.timing {
            bufs.push((&t.readback, t.readback.size()));
        }
        let reads = map_read_prefix(&self.device, &bufs, self.concurrent, self.last_submission.clone())?;
        if let Some(t) = &self.timing {
            let ticks: &[u64] = bytemuck::cast_slice(&reads[2]);
            for (i, ms) in self.pass_ms.iter_mut().enumerate() {
                let (a, b) = (ticks[i * 2], ticks[i * 2 + 1]);
                if b > a {
                    *ms += (b - a) as f64 * t.period_ns as f64 / 1.0e6;
                }
            }
            self.timed += 1;
        }
        let samples: &[f32] = bytemuck::cast_slice(&reads[0]);
        let state: &[u32] = bytemuck::cast_slice(&reads[1]);
        let n = self.cfg.block_samples();
        let stride = self.strides.out as usize;
        for (j, &i) in std::mem::take(&mut self.in_flight).iter().enumerate() {
            let s = &state[i * STATE_SLOTS..(i + 1) * STATE_SLOTS];
            let l = &mut self.lanes[i];
            l.out.clear();
            l.out.extend_from_slice(&samples[j * stride..j * stride + n]);
            l.live = s[S_LIVE].min(self.cfg.max_voices);
            l.stolen = s[S_STOLEN] as u64;
            if s[S_DROPPED] != 0 {
                log::error!(
                    "the spawn pass dropped {} voices it should never have been handed",
                    s[S_DROPPED]
                );
                l.dropped += s[S_DROPPED] as u64;
                self.queue.write_buffer(
                    &self.state_buf,
                    ((i * STATE_SLOTS + S_DROPPED) * 4) as u64,
                    &0u32.to_le_bytes(),
                );
            }
            l.peak = l.out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            l.parity ^= 1;
            l.delivered = true;
        }
        Ok(())
    }
}

impl LaneHost {
    /// Give `lane` -- lane `at` of the batch -- a params slot for its variant \[39\]
    fn install(&self, lane: &mut Lane, at: usize, index: u32, data: &[RegionParams], menv: &[ModEnvParams]) -> Result<()> {
        if data.len() != self.params_per_variant as usize {
            bail!("params variant {index} has {} entries, expected {}", data.len(), self.params_per_variant);
        }
        if menv.len() != self.menv_per_variant as usize {
            bail!("mod env variant {index} has {} entries, expected {}", menv.len(), self.menv_per_variant);
        }
        let variants = lane.slots.len();
        if index as usize >= variants {
            bail!("params variant {index} is past the configured maximum");
        }
        let mut pool = self.slots.lock().unwrap();
        let mut slot = lane.slots[index as usize];
        if slot == 0 {
            if pool.free.is_empty() {
                self.grow(&mut pool)?;
            }
            slot = pool.free.pop().expect("grow added some");
            lane.slots[index as usize] = slot;
            self.queue.write_buffer(&self.slots_buf, (at * variants * 4) as u64, bytemuck::cast_slice(&lane.slots));
        }
        let off = slot as u64 * self.params_per_variant as u64 * std::mem::size_of::<RegionParams>() as u64;
        self.queue.write_buffer(&pool.params_buf, off, bytemuck::cast_slice(data));
        let moff = slot as u64 * self.menv_per_variant as u64 * std::mem::size_of::<ModEnvParams>() as u64;
        self.queue.write_buffer(&pool.menv_buf, moff, bytemuck::cast_slice(menv));
        Ok(())
    }

    /// Double the variant slots, keeping every slot's contents. The next \[40\]
    fn grow(&self, pool: &mut SlotPool) -> Result<()> {
        let old = pool.count;
        let new = old * 2;
        let pbytes = new as u64 * self.params_per_variant as u64 * std::mem::size_of::<RegionParams>() as u64;
        if pbytes > self.binding_cap {
            bail!(
                "the lanes' sound-controller variants need {:.2} GiB of params in one binding, past \
                 the adapter's {:.2} GiB",
                pbytes as f64 / (1u64 << 30) as f64,
                self.binding_cap as f64 / (1u64 << 30) as f64
            );
        }
        let mbytes = new as u64 * self.menv_per_variant as u64 * std::mem::size_of::<ModEnvParams>() as u64;
        let usage = STORAGE_RW | wgpu::BufferUsages::COPY_SRC;
        let params = buffer(&self.device, "batch params", pbytes, usage);
        let menv = buffer(&self.device, "batch mod env", mbytes, usage);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grow slots") });
        enc.copy_buffer_to_buffer(&pool.params_buf, 0, &params, 0, pool.params_buf.size());
        enc.copy_buffer_to_buffer(&pool.menv_buf, 0, &menv, 0, pool.menv_buf.size());
        self.queue.submit(Some(enc.finish()));
        pool.params_buf = params;
        pool.menv_buf = menv;
        pool.count = new;
        pool.free.extend((old..new).rev());
        pool.grown = true;
        Ok(())
    }
}

/// A lane of a batch, as the `Backend` its driver renders through. Every call \[41\]
pub struct LaneBackend<'a> {
    lane: &'a mut Lane,
    host: &'a LaneHost,
    index: usize,
}

impl LaneBackend<'_> {
    /// Make this lane a fresh render: no voices, the counters at zero, the \[42\]
    pub fn reset(&mut self) {
        let variants = self.lane.slots.len();
        {
            let mut pool = self.host.slots.lock().unwrap();
            for s in self.lane.slots.iter_mut().filter(|s| **s != 0) {
                pool.free.push(*s);
            }
        }
        *self.lane = Lane { slots: vec![0; variants], chan_count: BASE_CHANNELS as u32, ..Default::default() };
        let q = &self.host.queue;
        q.write_buffer(&self.host.slots_buf, (self.index * variants * 4) as u64, bytemuck::cast_slice(&vec![0u32; variants]));
        q.write_buffer(&self.host.state_buf, (self.index * STATE_SLOTS * 4) as u64, bytemuck::cast_slice(&[0u32; STATE_SLOTS]));
    }

    /// Its place in the batch.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Whether the driver handed this lane's last block to the batch -- false \[43\]
    pub fn submitted(&self) -> bool {
        self.lane.submitted
    }
}

impl Backend for LaneBackend<'_> {
    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()> {
        let l = &mut *self.lane;
        l.gates.clear();
        l.gates.extend_from_slice(meta);
        l.gates.extend_from_slice(runs);
        l.off_meta_words = meta.len() as u32;
        Ok(())
    }

    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, variant: bool, cut: bool) -> Result<()> {
        let row_count = (self.host.cfg.block_frames / self.host.cfg.gate_frames) as usize + 1;
        let l = &mut *self.lane;
        l.rows.clear();
        l.rows.extend_from_slice(rows);
        l.chan_count = (rows.len() / (row_count * CHAN_FIELDS)) as u32;
        (l.bend, l.gain, l.variant, l.cut) = (bend, gain, variant, cut);
        Ok(())
    }

    fn set_params_variant(&mut self, index: u32, data: &[RegionParams], menv: &[ModEnvParams]) -> Result<()> {
        self.host.install(self.lane, self.index, index, data, menv)
    }

    fn set_glide(&mut self, active: bool) -> Result<()> {
        self.lane.glide = active;
        Ok(())
    }

    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        let l = &mut *self.lane;
        l.pending.clear();
        l.pending.extend_from_slice(cmds);
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        self.lane.submitted = true;
        Ok(())
    }

    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        let l = &mut *self.lane;
        if !l.delivered {
            bail!("lane {} finished a block the batch never rendered", self.index);
        }
        out.copy_from_slice(&l.out);
        l.delivered = false;
        Ok(())
    }

    fn skip_block(&mut self) -> Result<()> {
        let cfg = &self.host.cfg;
        let (frames, step) = (cfg.block_frames, cfg.env_step_frames());
        let l = &mut *self.lane;
        debug_assert_eq!(l.live, 0, "a block with voices alive cannot be skipped");
        l.env_phase = (l.env_phase + frames) % step;
        l.peak = 0.0;
        l.pending.clear();
        Ok(())
    }

    fn stats(&self) -> BlockStats {
        let l = &*self.lane;
        BlockStats { active_voices: l.live as u64, stolen: l.stolen, dropped: l.dropped, peak: l.peak }
    }

    fn name(&self) -> &'static str {
        "gpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every batch pass compiles and validates on the device, which is where a \[44\]
    #[test]
    fn the_batch_passes_compile() {
        let dir = std::env::temp_dir().join("kestrel_batch_compile");
        std::fs::create_dir_all(&dir).unwrap();
        let sf = dir.join("rich.sf2");
        crate::testkit::rich_sf2(&sf, 48_000).unwrap();
        let cfg = Config { max_voices: 512, ..Config::default() };
        let bank = crate::load_bank(&sf, &cfg).unwrap();
        let Ok((device, _queue, _info, _limits, _)) = device::create(&cfg) else {
            eprintln!("no GPU; skipped");
            return;
        };
        let s = Strides {
            voice_words: voice_fields(&cfg) as u32 * cfg.pool_slots(),
            partials: cfg.block_frames * 2 * max_render_workgroups(&cfg),
            out: cfg.block_frames * 2,
            capacity: cfg.pool_slots(),
            block_sums: cfg.pool_slots().div_ceil(cfg.workgroup_size),
            variants: cfg.max_param_variants.max(1),
        };
        let src = batch_sources(&cfg, &bank, &s).unwrap();
        let layouts = BatchLayouts::new(&device);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _p = BatchPipelines::new(&device, &cfg, src, &layouts);
        let err = pollster::block_on(device.pop_error_scope());
        assert!(err.is_none(), "{err:?}");
    }

    /// A render in a batch lane is the same bytes, block for block, as the \[45\]
    #[test]
    fn a_lane_renders_what_a_solo_render_does_block_for_block() {
        use crate::driver::Driver;
        use crate::midi::MidiWriter;
        let dir = std::env::temp_dir().join("kestrel_batch_lane");
        std::fs::create_dir_all(&dir).unwrap();
        let sf = dir.join("rich.sf2");
        crate::testkit::rich_sf2(&sf, 48_000).unwrap();
        let midi = |name: &str, seed: u64| {
            let mut w = MidiWriter::new(480);
            w.tempo_track(500_000);
            let mut ev: Vec<(u64, Vec<u8>)> = (0..400u64)
                .flat_map(|i| {
                    let t = i * 7 + (i * seed) % 5;
                    let ch = (i % 3) as u8;
                    let key = 36 + ((i * 7 + seed) % 48) as u8;
                    let vel = 20 + ((i * 13 + seed) % 100) as u8;
                    [(t, vec![0x90 | ch, key, vel]), (t + 60 + (i % 200), vec![0x80 | ch, key, 0])]
                })
                .collect();
            ev.extend((0..40u64).map(|i| (i * 60, vec![0xE0, (i * 3 % 128) as u8, (64 + i % 20) as u8])));
            ev.extend((0..12u64).map(|i| (100 + i * 200, vec![0xB1, 74, (20 + i * 9) as u8])));
            ev.push((0, vec![0xB2, 65, 127]));
            ev.push((0, vec![0xB2, 5, 40]));
            w.raw_track(ev);
            let p = dir.join(name);
            w.save(&p).unwrap();
            p
        };
        let (a_mid, b_mid) = (midi("a.mid", 3), midi("b.mid", 11));
        let base = Config { max_voices: 48, block_frames: 512, limiter: false, ..Config::default() };
        // [46]
        let configs = [
            ("every feature off", Config { filter_enabled: false, lfo_enabled: false, mod_env_enabled: false, ..base.clone() }, false),
            ("defaults", base.clone(), false),
            ("note grid", Config { note_grid: true, ..base.clone() }, false),
            ("split in two submissions", base.clone(), true),
        ];
        for (what, cfg, split) in configs {
            let bank = Arc::new(crate::load_bank(&sf, &cfg).unwrap());
            let Ok(mut solo) = GpuSynth::new(&cfg, bank.clone()) else {
                eprintln!("no GPU; skipped");
                return;
            };
            let phase = crate::phase::PhaseBank::prepare(&bank, &cfg.phase).unwrap();
            let shared = GpuShared::new(&cfg, &bank, &phase).unwrap();
            let mut batch = GpuBatch::new(&cfg, &bank, &shared, 2).unwrap();
            if split {
                batch.set_upload_max(1);
            }
            let mut d_solo = Driver::open(&cfg, bank.clone(), &a_mid).unwrap();
            let mut d_lane = Driver::open(&cfg, bank.clone(), &a_mid).unwrap();
            let mut d_other = Driver::open(&cfg, bank.clone(), &b_mid).unwrap();
            let n = cfg.block_samples();
            let (mut x, mut y, mut z) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            let mut other_done = false;
            let mut finished = false;
            for block in 0..600 {
                let more_solo = d_solo.next_block(&mut solo, &mut x).unwrap();
                d_lane.submit_block(&mut batch.lane(1)).unwrap();
                if !other_done {
                    d_other.submit_block(&mut batch.lane(0)).unwrap();
                }
                batch.flush().unwrap();
                d_lane.prepare_ahead().unwrap();
                if !other_done {
                    d_other.prepare_ahead().unwrap();
                }
                batch.wait().unwrap();
                let more_lane = d_lane.finish_block(&mut batch.lane(1), &mut y).unwrap();
                if !other_done {
                    other_done = !d_other.finish_block(&mut batch.lane(0), &mut z).unwrap();
                }
                let (s, l) = (solo.stats(), batch.lane(1).stats());
                if let Some(i) = (0..n).find(|&i| x[i].to_bits() != y[i].to_bits()) {
                    panic!(
                        "{what}: block {block}, sample {i}: solo {} lane {}; live {} against {}, stolen {} against {}",
                        x[i], y[i], s.active_voices, l.active_voices, s.stolen, l.stolen
                    );
                }
                assert_eq!((s.active_voices, s.stolen), (l.active_voices, l.stolen), "{what}: block {block}");
                assert_eq!(more_solo, more_lane, "{what}: block {block}");
                if !more_solo {
                    assert!(s.stolen > 0, "{what}: nothing was stolen, so stealing went untested");
                    finished = true;
                    break;
                }
            }
            assert!(finished, "{what}: the fixture did not finish");
            if split {
                assert!(batch.split_batches() > 0, "{what}: no batch was split, so splitting went untested");
            }
        }
    }

    #[test]
    fn an_index_gets_the_lane_base_and_nothing_else_does() {
        let src = "let a = voices[i * c + j]; let b = voices_out[f[k]]; x.voices[1]; myvoices[2];";
        let r = rewrite_index(src, "voices", |e| format!("lb + ({e})"));
        assert_eq!(r, "let a = voices[lb + (i * c + j)]; let b = voices_out[f[k]]; x.voices[1]; myvoices[2];");
        let r = rewrite_index(&r, "voices_out", |e| format!("lo + ({e})"));
        let r = rename_index(&r, "voices_out", "voices");
        assert_eq!(r, "let a = voices[lb + (i * c + j)]; let b = voices[lo + (f[k])]; x.voices[1]; myvoices[2];");
    }
}
