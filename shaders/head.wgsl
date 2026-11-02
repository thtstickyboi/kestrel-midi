// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

struct U {
    /// Heads to apply this step.
    count: u32,
    /// P, the partition length, and so the output block length.
    frames: u32,
    /// Buses in use. A head lands on the filter bus its note selected, because \[2\]
    buses: u32,
    /// Frames of `A[n]` a waveform may hold. Read by nothing in this shader \[3\]
    head_frames: u32,
    /// Fixed-point scale, chosen by the host from the head count so the \[4\]
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// One note's correction, clipped to one partition step.
struct Head {
    /// First frame of this note's waveform in the cache. \[5\]
    base: u32,
    /// First frame of the correction *within this step's output*.
    out0: u32,
    /// First frame of `A[n]` to read. Non-zero when the head began in an \[6\]
    src0: u32,
    /// How many frames to write here.
    len: u32,
    gain_l: f32,
    gain_r: f32,
    /// `f` at `src0`, already evaluated by the host.
    f0: f32,
    /// Per-frame ratio applied to `f` once past `Dj`.
    r: f32,
    /// Frames from `src0` until `f` starts moving, i.e. `Dj - src0` clamped.
    hold: u32,
    /// Left output bus; right is `bus + 1`.
    bus: u32,
    /// Frames from `src0` over which the delay line contributes nothing. \[7\]
    dead: u32,
    /// Frames this waveform holds. What `src` is bounded by, where a \[8\]
    cap: u32,
    /// Added to `f` each frame, after the multiply by `r`. \[9\]
    fade: f32,
    /// Per-frame gain increments, for a channel whose volume moves while the \[10\]
    dg_l: f32,
    dg_r: f32,
    /// The gain the delay line's impulse was actually fired at. \[11\]
    gi_l: f32,
    gi_r: f32,
    _pad0: u32,
    /// Identical notes this correction stands for (THE PLAN, Phase 1). Each \[12\]
    weight: u32,
}

@group(0) @binding(0) var<uniform> u: U;
/// The sustaining waveforms, unit gain and unpanned, packed end to end. Where \[13\]
@group(0) @binding(1) var<storage, read> cache: array<f32>;
@group(0) @binding(2) var<storage, read> heads: array<Head>;
/// `[buses][frames]` fixed-point accumulator.
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
/// `[buses][2P]` -- the FDL's output, whose first half this is added into.
@group(0) @binding(4) var<storage, read_write> outbuf: array<f32>;

// [14]

@compute @workgroup_size({{HEAD_WG}})
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngroups: vec3<u32>,
) {
    // [15]
    let stride = ngroups.x * {{HEAD_WG}}u;
    for (var i = gid.x; i < u.count; i = i + stride) {
        apply(heads[i]);
    }
}

fn apply(h: Head) {
    var f = h.f0;
    var gl = h.gain_l;
    var gr = h.gain_r;
    let base = h.base;
    for (var n = 0u; n < h.len; n = n + 1u) {
        let src = h.src0 + n;
        if src >= h.cap { break; }
        let out = h.out0 + n;
        if out >= u.frames { break; }

        // [16]
        let fe = select(f, 0.0, n < h.dead);
        // [17]
        let a = cache[base + src];
        atomicAdd(&acc[h.bus * u.frames + out], i32(round(a * (gl - fe * h.gi_l) * u.scale)) * i32(h.weight));
        atomicAdd(&acc[(h.bus + 1u) * u.frames + out], i32(round(a * (gr - fe * h.gi_r) * u.scale)) * i32(h.weight));
        gl = gl + h.dg_l;
        gr = gr + h.dg_r;

        // [18]
        if n + 1u >= h.hold {
            f = f * h.r + h.fade;
        }
    }
}


// [19]

@compute @workgroup_size({{HEAD_WG}})
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if f >= u.frames { return; }
    let bus = gid.y;
    if bus >= u.buses { return; }
    let raw = atomicExchange(&acc[bus * u.frames + f], 0);
    outbuf[bus * 2u * u.frames + f] = outbuf[bus * 2u * u.frames + f] + f32(raw) / u.scale;
}
