// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

struct U {
    /// Entries with impulses this block, summed over buses.
    fired: u32,
    /// K, the partition count.
    lags: u32,
    /// P + 1 bins from a 2P-point real transform.
    bins: u32,
    /// Which ring slot this block emits: `t % K`.
    slot0: u32,
    /// E, the resident entry count. `kmeta` is what indexes the cache now, so \[2\]
    entries: u32,
    buses: u32,
    /// P.
    frames: u32,
    /// Fixed-point scale for the scatter. The host picks it from the block's \[3\]
    scale: f32,
    /// Impulses **this block**, which is not `arrayLength(&impulses)`. \[4\]
    n_impulses: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Impulse {
    /// Row of the per-entry input buffer, i.e. index into the firing list.
    slot: u32,
    /// Frame within this block, 0 .. P.
    frame: u32,
    gain: f32,
    /// Identical notes this impulse stands for. Multiplies the *rounded* value, \[5\]
    weight: u32,
}

@group(0) @binding(0) var<uniform> u: U;
/// Kernel cache, packed. The buffers whose traffic sets the cost. \[6\]
@group(0) @binding(1) var<storage, read> h0: array<vec2<f32>>;
/// [E] of `(offset into its buffer in complex, partitions, which buffer, _)`. \[7\]
@group(0) @binding(2) var<storage, read> kmeta: array<vec4<u32>>;
/// [fired] entry ids, **deepest first within each bus**, ties broken by \[8\]
@group(0) @binding(3) var<storage, read> fired: array<u32>;
/// [buses * 2] half-open ranges into `fired`.
@group(0) @binding(4) var<storage, read> bus_range: array<u32>;
@group(0) @binding(5) var<storage, read> impulses: array<Impulse>;
/// [fired][2P] fixed-point accumulator for the impulse trains.
@group(0) @binding(6) var<storage, read_write> tin: array<atomic<i32>>;
/// [fired][2P] reals, shared with `fft.wgsl`'s `data`.
@group(0) @binding(7) var<storage, read_write> data: array<f32>;
/// [fired][bins] complex, shared with `fft.wgsl`'s `spec`.
@group(0) @binding(8) var<storage, read_write> spec: array<vec2<f32>>;
/// [buses][K][bins] frequency-domain accumulator.
@group(0) @binding(9) var<storage, read_write> ring: array<vec2<f32>>;
/// [buses][2P]: the emitted block in the first half, the carry in the second.
@group(0) @binding(10) var<storage, read_write> outbuf: array<f32>;
@group(0) @binding(11) var<storage, read> h1: array<vec2<f32>>;
@group(0) @binding(12) var<storage, read> h2: array<vec2<f32>>;
@group(0) @binding(13) var<storage, read> h3: array<vec2<f32>>;
/// Buses with at least one row this step, which is what the MAC's `z` walks.
@group(0) @binding(14) var<storage, read> live_bus: array<u32>;

const WG: u32 = {{FDL_WG}}u;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// One complex of the kernel cache. `b` is uniform across the workgroup.
fn kread(b: u32, i: u32) -> vec2<f32> {
    if b == 0u { return h0[i]; }
    if b == 1u { return h1[i]; }
    if b == 2u { return h2[i]; }
    return h3[i];
}

// [9]

@compute @workgroup_size({{FDL_WG}})
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngroups: vec3<u32>,
) {
    // [10]
    let stride = ngroups.x * {{FDL_WG}}u;
    for (var i = gid.x; i < u.n_impulses; i = i + stride) {
        let imp = impulses[i];
        if imp.frame >= u.frames { continue; }
        let v = i32(round(imp.gain * u.scale)) * i32(imp.weight);
        atomicAdd(&tin[imp.slot * (2u * u.frames) + imp.frame], v);
    }
}

// [11]

@compute @workgroup_size({{FDL_WG}})
fn to_real(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngroups: vec3<u32>,
) {
    // [12]
    let n = u.fired * 2u * u.frames;
    let stride = ngroups.x * {{FDL_WG}}u;
    for (var i = gid.x; i < n; i = i + stride) {
        let raw = atomicExchange(&tin[i], 0);
        data[i] = f32(raw) / u.scale;
    }
}

// [13]

@compute @workgroup_size({{FDL_WG}})
fn mac(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let bin = wid.x * WG + tid;
    if bin >= u.bins { return; }
    let k = wid.y;
    // [14]
    let bus = live_bus[wid.z];

    let lo = bus_range[bus * 2u];
    let hi = bus_range[bus * 2u + 1u];
    // Nothing on this bus reaches this lag: the deepest is first.
    if lo >= hi || k >= kmeta[fired[lo]].y { return; }

    var acc = vec2<f32>(0.0, 0.0);
    for (var i = lo; i < hi; i = i + 1u) {
        let m = kmeta[fired[i]];
        if k >= m.y { break; }
        acc = acc + cmul(kread(m.z, m.x + k * u.bins + bin), spec[i * u.bins + bin]);
    }

    let slot = (u.slot0 + k) % u.lags;
    let o = (bus * u.lags + slot) * u.bins + bin;
    ring[o] = ring[o] + acc;
}

// [15]

@compute @workgroup_size({{FDL_WG}})
fn emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bin = gid.x;
    if bin >= u.bins { return; }
    let bus = gid.y;
    if bus >= u.buses { return; }
    let o = (bus * u.lags + u.slot0) * u.bins + bin;
    spec[bus * u.bins + bin] = ring[o];
    ring[o] = vec2<f32>(0.0, 0.0);
}

// [16]

@compute @workgroup_size({{FDL_WG}})
fn overlap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if f >= u.frames { return; }
    let bus = gid.y;
    if bus >= u.buses { return; }
    let n = 2u * u.frames;
    let y0 = data[bus * n + f];
    let y1 = data[bus * n + u.frames + f];
    outbuf[bus * n + f] = y0 + outbuf[bus * n + u.frames + f];
    outbuf[bus * n + u.frames + f] = y1;
}
