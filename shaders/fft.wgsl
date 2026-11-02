// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

struct U {
    batch: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: U;
// [2]
@group(0) @binding(1) var<storage, read> tw: array<vec2<f32>>;
// [3]
@group(0) @binding(2) var<storage, read_write> data: array<vec2<f32>>;
// [batch][N+1] complex, the real spectrum. Only the real entry points bind it.
@group(0) @binding(3) var<storage, read_write> spec: array<vec2<f32>>;

const N: u32 = {{FFT_N}}u;
const LOG2N: u32 = {{FFT_LOG2N}}u;
const WG: u32 = {{FFT_WG}}u;

var<workgroup> buf: array<vec2<f32>, {{FFT_2N}}>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// [4]

fn load_complex(tid: u32, base: u32, inv: bool) {
    var i = tid;
    loop {
        if i >= N { break; }
        var z = data[base + i];
        if inv { z.y = -z.y; }
        buf[i] = z;
        i += WG;
    }
    workgroupBarrier();
}

// Runs the stages over `buf` and returns which half holds the result.
fn transform(tid: u32) -> u32 {
    // [5]
    var cur: u32 = 0u;
    var n: u32 = N;
    var s: u32 = 1u;
    var logs: u32 = 0u;

    if (LOG2N & 1u) == 1u {
        let m = n >> 1u;
        let src = cur * N;
        let dst = (1u - cur) * N;
        let step = (2u * N) / n;
        var idx = tid;
        loop {
            if idx >= (N >> 1u) { break; }
            let p = idx >> logs;
            let q = idx & (s - 1u);
            let a = buf[src + q + s * p];
            let b = buf[src + q + s * (p + m)];
            buf[dst + q + s * (2u * p)] = a + b;
            buf[dst + q + s * (2u * p + 1u)] = cmul(a - b, tw[p * step]);
            idx += WG;
        }
        workgroupBarrier();
        cur = 1u - cur;
        n = m;
        s = s << 1u;
        logs += 1u;
    }

    loop {
        if n <= 1u { break; }
        let m = n >> 2u;
        let src = cur * N;
        let dst = (1u - cur) * N;
        let step = (2u * N) / n;
        var idx = tid;
        loop {
            if idx >= (N >> 2u) { break; }
            let p = idx >> logs;
            let q = idx & (s - 1u);
            let a0 = buf[src + q + s * p];
            let a1 = buf[src + q + s * (p + m)];
            let a2 = buf[src + q + s * (p + 2u * m)];
            let a3 = buf[src + q + s * (p + 3u * m)];
            let t0 = a0 + a2;
            let t1 = a0 - a2;
            let t2 = a1 + a3;
            let t3 = a1 - a3;
            // [6]
            let jt3 = vec2<f32>(t3.y, -t3.x);
            buf[dst + q + s * (4u * p)] = t0 + t2;
            buf[dst + q + s * (4u * p + 1u)] = cmul(t1 + jt3, tw[p * step]);
            buf[dst + q + s * (4u * p + 2u)] = cmul(t0 - t2, tw[2u * p * step]);
            buf[dst + q + s * (4u * p + 3u)] = cmul(t1 - jt3, tw[3u * p * step]);
            idx += WG;
        }
        workgroupBarrier();
        cur = 1u - cur;
        n = m;
        s = s << 2u;
        logs += 2u;
    }
    return cur;
}

fn store_complex(tid: u32, base: u32, cur: u32, inv: bool) {
    let scale = select(1.0, 1.0 / f32(N), inv);
    var o = tid;
    loop {
        if o >= N { break; }
        var z = buf[cur * N + o];
        if inv { z.y = -z.y; }
        data[base + o] = z * scale;
        o += WG;
    }
}

// [7]

// [8]

@compute @workgroup_size({{FFT_WG}})
fn forward(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if wid.x >= u.batch { return; }
    let base = wid.x * N;
    load_complex(tid, base, false);
    store_complex(tid, base, transform(tid), false);
}

@compute @workgroup_size({{FFT_WG}})
fn inverse(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if wid.x >= u.batch { return; }
    let base = wid.x * N;
    load_complex(tid, base, true);
    store_complex(tid, base, transform(tid), true);
}

// [9]

@compute @workgroup_size({{FFT_WG}})
fn r2c(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if wid.x >= u.batch { return; }
    load_complex(tid, wid.x * N, false);
    let zb = transform(tid) * N;
    let sb = wid.x * (N + 1u);
    var k = tid;
    loop {
        if k > N { break; }
        // [10]
        let a = buf[zb + (k % N)];
        let b = buf[zb + ((N - k) % N)];
        let bc = vec2<f32>(b.x, -b.y);
        let e = (a + bc) * 0.5;
        let d = (a - bc) * 0.5;
        // O[k] = -i*d, so W^k * O[k] = -i * (W^k * d).
        let wd = cmul(tw[k], d);
        spec[sb + k] = e + vec2<f32>(wd.y, -wd.x);
        k += WG;
    }
}

@compute @workgroup_size({{FFT_WG}})
fn c2r(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if wid.x >= u.batch { return; }
    let sb = wid.x * (N + 1u);
    var k = tid;
    loop {
        if k >= N { break; }
        // [11]
        let xk = spec[sb + k];
        let xn = spec[sb + (N - k)];
        let xnc = vec2<f32>(xn.x, -xn.y);
        let e = (xk + xnc) * 0.5;
        let d = (xk - xnc) * 0.5;
        let w = tw[k];
        let o = cmul(vec2<f32>(w.x, -w.y), d);
        // [12]
        let z = e + vec2<f32>(-o.y, o.x);
        buf[k] = vec2<f32>(z.x, -z.y);
        k += WG;
    }
    workgroupBarrier();
    store_complex(tid, wid.x * N, transform(tid), true);
}
