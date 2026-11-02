// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

struct U {
    /// Frames in this step, i.e. P.
    frames: u32,
    /// Buses actually in use: `filters * 2`. The rest are silent and skipped.
    buses: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: U;
/// `[filters]` of `(b0, b1, a1, a2)`. `b2 == b0`, as in `cpu.rs`.
@group(0) @binding(1) var<storage, read> coef: array<vec4<f32>>;
/// 1 when this filter is enabled, 0 to pass through.
@group(0) @binding(2) var<storage, read> enabled: array<u32>;
/// `[buses]` of `(z1, z2)`, carried across blocks.
@group(0) @binding(3) var<storage, read_write> state: array<vec2<f32>>;
/// `[buses][2P]`: the FDL's output, first half live.
@group(0) @binding(4) var<storage, read_write> outbuf: array<f32>;
/// `[2][P]` interleaved stereo result.
@group(0) @binding(5) var<storage, read_write> mixed: array<f32>;

// [2]

@compute @workgroup_size({{BUS_WG}})
fn biquad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bus = gid.x;
    if bus >= u.buses { return; }
    let filt = bus / 2u;
    if enabled[filt] == 0u { return; }

    let c = coef[filt];
    var z = state[bus];
    let base = bus * 2u * u.frames;
    for (var n = 0u; n < u.frames; n = n + 1u) {
        // [3]
        let x = outbuf[base + n];
        let y = c.x * x + z.x;
        z = vec2<f32>(c.y * x - c.z * y + z.y, c.x * x - c.w * y);
        outbuf[base + n] = y;
    }
    state[bus] = z;
}

// [4]

@compute @workgroup_size({{BUS_WG}})
fn mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if f >= u.frames { return; }
    var l = 0.0;
    var r = 0.0;
    // Ascending bus order, so the summation is a pure function of the input.
    for (var b = 0u; b < u.buses; b = b + 2u) {
        l = l + outbuf[b * 2u * u.frames + f];
        r = r + outbuf[(b + 1u) * 2u * u.frames + f];
    }
    mixed[f] = l;
    mixed[u.frames + f] = r;
}
