// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_block: array<f32>;

var<workgroup> red: array<f32, WG>;

const KAHAN: bool = {{KAHAN}};

// [2]
const THIN_MAX: u32 = {{REDUCE_THIN}}u;

// [3]
const THIN_LANES: u32 = 64u;

@compute @workgroup_size(64)
fn thin(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let j = wgid.x * THIN_LANES + tid;
    let nwg = u.render_workgroups;
    if (j >= u.block_frames * 2u || nwg > THIN_MAX) { return; }

    let base = j * nwg;
    // The smallest m with 2^m >= nwg.
    var m = 0u;
    while ((1u << m) < nwg) { m = m + 1u; }

    // [4]
    let zero = bitcast<f32>(u.zero);

    var stack: array<f32, 9>;
    let size = 1u << m;
    for (var k = 0u; k < size; k = k + 1u) {
        var t = 0u;
        if (m > 0u) { t = reverseBits(k) >> (32u - m); }
        var v = 0.0;
        if (t < nwg) { v = partials[base + t] + zero; }
        var level = 0u;
        var c = k;
        while ((c & 1u) == 1u) {
            v = stack[level] + v;
            level = level + 1u;
            c = c >> 1u;
        }
        stack[level] = v;
    }
    out_block[j] = stack[m];
}

@compute @workgroup_size({{WG}})
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let j = wgid.x;
    if (j >= u.block_frames * 2u) { return; }

    let nwg = u.render_workgroups;
    // `thin` has this block.
    if (nwg <= THIN_MAX) { return; }
    let base = j * nwg;

    var acc = 0.0;
    if (KAHAN) {
        var comp = 0.0;
        for (var w = tid; w < nwg; w = w + WG) {
            let y = partials[base + w] - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
        }
    } else {
        for (var w = tid; w < nwg; w = w + WG) {
            acc = acc + partials[base + w];
        }
    }
    red[tid] = acc;
    workgroupBarrier();

    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out_block[j] = red[0];
    }
}
