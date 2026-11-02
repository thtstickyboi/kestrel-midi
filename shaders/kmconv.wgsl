// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

struct T {
    rows: u32,
    cols: u32,
    n: u32,
    /// 0 plain, 1 twiddle exp(-2 pi i r c / n), 2 its conjugate.
    mode: u32,
}

struct S {
    count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

struct M {
    n: u32,
    blocks: u32,
    /// The four-step transform's first factor, which says where a bin is.
    n1: u32,
    _p1: u32,
}

struct O {
    n: u32,
    /// Blocks in this pass.
    count: u32,
    _p0: u32,
    _p1: u32,
}

struct K {
    n: u32,
    entries: u32,
    _p0: u32,
    _p1: u32,
}

/// `merlin::head::Head`, field for field: one note's head correction clipped \[2\]
struct Hd {
    base: u32,
    out0: u32,
    src0: u32,
    len: u32,
    gain_l: f32,
    gain_r: f32,
    f0: f32,
    r: f32,
    hold: u32,
    bus: u32,
    dead: u32,
    cap: u32,
    fade: f32,
    dg_l: f32,
    dg_r: f32,
    gi_l: f32,
    gi_r: f32,
    split: u32,
    weight: u32,
}

struct Hu {
    /// Frames between one bus and the next in the bus buffer.
    stride: u32,
    items: u32,
    _p0: u32,
    _p1: u32,
}

struct Bu {
    /// Frames of this window: what the biquads run over and the mix sums.
    frames: u32,
    stride: u32,
    filters: u32,
    /// Frames past the window a block's tail reaches, which the next window \[3\]
    carry: u32,
}

/// A window's end, for one key: `count` of its buses, listed, the window's \[4\]
struct Eu {
    frames: u32,
    stride: u32,
    count: u32,
    carry: u32,
    at: u32,
    filtered: u32,
    ringing: u32,
    mixes: u32,
    moves: u32,
    clears: u32,
    _p0: u32,
    _p1: u32,
}

/// One impulse: its index in the pairs buffer (`pair * n + frame`) and its \[5\]
struct Imp {
    at: u32,
    l: f32,
    r: f32,
    _p: u32,
}

//#section transpose
// ---------------------------------------------------------------------------
// The four-step transform's transpose, with its twiddle from an f64 table.
//
// `[g][rows][cols] -> [g][cols][rows]`, `g` on the dispatch's z. The table is
// `exp(-2 pi i m / n)` for `m < n`, rounded once on the host, so the transform
// does not depend on the driver's `sin` and `cos`; `r * c < n` because the two
// indices run over the two factors of `n`.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> t: T;
@group(0) @binding(1) var<storage, read> t_src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> t_dst: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> t_tw: array<vec2<f32>>;

var<workgroup> tile: array<vec2<f32>, 272>;

@compute @workgroup_size(16, 16, 1)
fn transpose(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let base = wg.z * t.rows * t.cols;
    let r = wg.y * 16u + l.y;
    let c = wg.x * 16u + l.x;
    var v = vec2<f32>(0.0, 0.0);
    if (r < t.rows && c < t.cols) {
        v = t_src[base + r * t.cols + c];
        if (t.mode != 0u) {
            var w = t_tw[r * c];
            if (t.mode == 2u) {
                w.y = -w.y;
            }
            v = vec2<f32>(v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x);
        }
    }
    tile[l.y * 17u + l.x] = v;
    workgroupBarrier();
    let orow = wg.x * 16u + l.y;
    let ocol = wg.y * 16u + l.x;
    if (orow < t.cols && ocol < t.rows) {
        t_dst[base + orow * t.rows + ocol] = tile[l.x * 17u + l.y];
    }
}

//#section scatter
// ---------------------------------------------------------------------------
// A chunk's impulses into its zeroed pairs buffer. One invocation each,
// grid-strided; every target is distinct, so no atomic and no order.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> s: S;
@group(0) @binding(1) var<storage, read> s_imp: array<Imp>;
@group(0) @binding(2) var<storage, read_write> s_x: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let stride = nw.x * 256u;
    for (var i = gid.x; i < s.count; i = i + stride) {
        let p = s_imp[i];
        s_x[p.at] = vec2<f32>(p.l, p.r);
    }
}

//#section mac
// ---------------------------------------------------------------------------
// Each block's spectrum: the sum over its pairs of the pair's spectrum times
// its entry's kernel spectrum, in the pairs' order. One invocation per (bin,
// block); `y` is `[blocks][n]`.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> m: M;
@group(0) @binding(1) var<storage, read> m_x: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> m_k: array<vec2<f32>>;
/// `[blocks][2]`: each block's `[first pair, end)`.
@group(0) @binding(3) var<storage, read> m_desc: array<u32>;
/// `[pairs]`: each pair's entry, its row of `m_k`.
@group(0) @binding(4) var<storage, read> m_entry: array<u32>;
@group(0) @binding(5) var<storage, read_write> m_y: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// Where frequency `-f` is, for the bin stored at `bin`. The transform leaves \[6\]
fn mirror(bin: u32) -> u32 {
    let n2 = m.n / m.n1;
    let k2 = bin / m.n1;
    let k1 = bin % m.n1;
    let k1m = select(m.n1 - 1u - k1, (m.n1 - k1) % m.n1, k2 == 0u);
    return ((n2 - k2) % n2) * m.n1 + k1m;
}

@compute @workgroup_size(256, 1, 1)
fn mac(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let bin = gid.x + gid.y * nw.x * 256u;
    let b = gid.z;
    if (bin >= m.n || b >= m.blocks) {
        return;
    }
    let lo = m_desc[b * 2u];
    let hi = m_desc[b * 2u + 1u];
    var acc = vec2<f32>(0.0, 0.0);
    for (var p = lo; p < hi; p = p + 1u) {
        let x = m_x[p * m.n + bin];
        let e = m_entry[p];
        let row = e & 0x7FFFFFFFu;
        let k = m_k[row * m.n + bin];
        if (e == row) {
            acc = acc + cmul(x, k);
            continue;
        }
        // [7]
        let at = mirror(bin);
        let xm = m_x[p * m.n + at];
        let km = m_k[row * m.n + at];
        let mid = vec2<f32>(x.x + xm.x, x.y - xm.y) * 0.5;
        let side = vec2<f32>(x.y + xm.y, xm.x - x.x) * 0.5;
        acc = acc + cmul(mid, k) + cmul(side, vec2<f32>(km.x, -km.y));
    }
    m_y[b * m.n + bin] = acc;
}

//#section ola
// ---------------------------------------------------------------------------
// Overlap-add: blocks, back in time order, into the group's output. A pass
// holds the chunk's blocks of one parity: a block reaches `n` frames and
// starts `hop >= n / 2` after the one before, so only neighbours overlap and
// two blocks of one parity never write the same frame. The host runs even then
// odd, which fixes the order every frame is summed in.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> o: O;
@group(0) @binding(1) var<storage, read> o_y: array<vec2<f32>>;
/// `[count][2]`: the block's row of `o_y` and the frame it starts at.
@group(0) @binding(2) var<storage, read> o_desc: array<u32>;
@group(0) @binding(3) var<storage, read_write> o_out: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn ola(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let f = gid.x + gid.y * nw.x * 256u;
    let z = gid.z;
    if (f >= o.n || z >= o.count) {
        return;
    }
    let row = o_desc[z * 2u];
    let at = o_desc[z * 2u + 1u];
    o_out[at + f] = o_out[at + f] + o_y[row * o.n + f];
}

//#section load_kernels
// ---------------------------------------------------------------------------
// The kernels, cut from the pool, into `[entries][n]` complex with the rest of
// each row zero: what the forward transform takes.
//
// A kernel is `Kernel::segment` of a sustain in the pool: `len` frames from
// `at`, of which `valid` exist, the first `plain` as they are and each after
// scaled by `r^(rel0 + f - plain)`, `exp2` of that times `log2(r)` (the host's,
// in f64, rounded once). A kernel laid in the pool whole has `plain == len`.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> k: K;
@group(0) @binding(1) var<storage, read> k_real: array<f32>;
/// `[entries][7]`: `at, len, valid, plain, rel0, log2(r)` bits, and the frames \[8\]
@group(0) @binding(2) var<storage, read> k_desc: array<u32>;
@group(0) @binding(3) var<storage, read_write> k_out: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn load_kernels(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let f = gid.x + gid.y * nw.x * 256u;
    let e = gid.z;
    if (f >= k.n || e >= k.entries) {
        return;
    }
    let b = e * 7u;
    var v = vec2<f32>(0.0, 0.0);
    if (f < k_desc[b + 1u] && f < k_desc[b + 2u]) {
        v.x = k_real[k_desc[b] + f];
        let right = k_desc[b + 6u];
        if (right != 0u) {
            v.y = k_real[k_desc[b] + right + f];
        }
        let plain = k_desc[b + 3u];
        if (f >= plain) {
            v = v * exp2(f32(k_desc[b + 4u] + f - plain) * bitcast<f32>(k_desc[b + 5u]));
        }
    }
    k_out[e * k.n + f] = v;
}

//#section heads
// ---------------------------------------------------------------------------
// The head corrections, gathered. One workgroup per (page, bus) item, each
// invocation owning four frames of the page: it sums every head of the item
// that reaches them, in the item's order, and adds the sum to the bus. No two
// workgroups share a frame of a bus, so there is no atomic, and the order is
// the host's, so the sum is fixed -- and in f32, where `head.wgsl` rounds
// into a fixed point whose scale shrinks with a step's head count.
//
// `head.wgsl` walks a head a frame at a time with two recurrences, the gain
// ramp and `f`. A frame here is reached directly, so both are in closed form:
// the ramp is `gain + dg * j`, and `f` has had `m` affine steps
// `f = f * r + fade`, counted from where `hold` lets it move.
// ---------------------------------------------------------------------------

const PAGE: u32 = 1024u;
const HEAD_BATCH: u32 = 64u;
var<workgroup> h_shared: array<Hd, 64>;
var<workgroup> h_range: vec2<u32>;

@group(0) @binding(0) var<uniform> hu: Hu;
@group(0) @binding(1) var<storage, read> h_heads: array<Hd>;
/// `[items][4]`: the page's first frame, its bus, and its `[lo, hi)` in `h_list`.
@group(0) @binding(2) var<storage, read> h_items: array<u32>;
@group(0) @binding(3) var<storage, read> h_list: array<u32>;
@group(0) @binding(4) var<storage, read> h_sus: array<f32>;
@group(0) @binding(5) var<storage, read_write> h_bus: array<vec2<f32>>;

fn f_at(h: Hd, j: u32) -> f32 {
    // [9]
    let h1 = max(h.hold, 1u);
    var m = 0u;
    if (j >= h1) {
        m = j - h1 + 1u;
    }
    if (m == 0u) {
        return h.f0;
    }
    if (h.r == 1.0) {
        return h.f0 + h.fade * f32(m);
    }
    // [10]
    var rm = 0.0;
    if (h.r > 0.0) {
        rm = exp2(f32(m) * bitcast<f32>(h.split));
    }
    return h.f0 * rm + h.fade * (1.0 - rm) / (1.0 - h.r);
}

@compute @workgroup_size(256, 1, 1)
fn heads(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= hu.items) {
        return;
    }
    let page = h_items[item * 4u];
    let bus = h_items[item * 4u + 1u];
    // [11]
    if (lid == 0u) {
        h_range = vec2<u32>(h_items[item * 4u + 2u], h_items[item * 4u + 3u]);
    }
    let range = workgroupUniformLoad(&h_range);
    let lo = range.x;
    let hi = range.y;
    // [12]
    var acc: array<vec2<f32>, 4>;
    for (var base = lo; base < hi; base = base + HEAD_BATCH) {
        let cnt = min(HEAD_BATCH, hi - base);
        if (lid < cnt) {
            h_shared[lid] = h_heads[h_list[base + lid]];
        }
        workgroupBarrier();
        for (var k = 0u; k < PAGE / 256u; k = k + 1u) {
            let f = page + lid + k * 256u;
            var a_k = acc[k];
            for (var i = 0u; i < cnt; i = i + 1u) {
                let h = h_shared[i];
                // Wraps past `len` for a frame before the head.
                let j = f - h.out0;
                if (j >= h.len) {
                    continue;
                }
                let src = h.src0 + j;
                // [13]
                let cap = h.cap & 0x7FFFFFFFu;
                if (src >= cap) {
                    continue;
                }
                let at = h.base + src;
                let a = vec2<f32>(h_sus[at], h_sus[at + cap * (h.cap >> 31u)]);
                var fe = 0.0;
                if (j >= h.dead) {
                    fe = f_at(h, j);
                }
                let g = vec2<f32>(h.gain_l + h.dg_l * f32(j), h.gain_r + h.dg_r * f32(j));
                a_k = a_k + a * (g - fe * vec2<f32>(h.gi_l, h.gi_r)) * f32(h.weight);
            }
            acc[k] = a_k;
        }
        workgroupBarrier();
    }
    for (var k = 0u; k < PAGE / 256u; k = k + 1u) {
        let o = bus * hu.stride + page + lid + k * 256u;
        h_bus[o] = h_bus[o] + acc[k];
    }
}

//#section direct
// ---------------------------------------------------------------------------
// The sparse pairs, added in the time domain: a (block, entry) pair with a few
// impulses costs a whole forward transform on the FFT path, where adding its
// kernel at each impulse costs the kernel's length. One workgroup per (page,
// bus) item, each invocation owning four frames of the page, as the heads do:
// it walks the item's `[lo, hi)` of impulses, which are in start order and
// cover every impulse whose kernel can reach the page, sums the ones that do
// in that order, and adds the sum to the bus once. No atomic, and the order is
// the host's.
// ---------------------------------------------------------------------------

struct Du {
    /// Frames between one bus and the next, and the most a bus has.
    stride: u32,
    items: u32,
    _p0: u32,
    _p1: u32,
}

/// One impulse: its frame from the window's start, its kernel's row of \[14\]
struct Dimp {
    at: u32,
    entry: u32,
    l: f32,
    r: f32,
}

const DPAGE: u32 = 1024u;

@group(0) @binding(0) var<uniform> du: Du;
@group(0) @binding(1) var<storage, read> d_imp: array<Dimp>;
/// `[items][4]`: the page's first frame, its bus, and its `[lo, hi)` in `d_imp`.
@group(0) @binding(2) var<storage, read> d_items: array<u32>;
@group(0) @binding(3) var<storage, read> d_real: array<f32>;
/// `[entries][7]`: as `load_kernels`'s `k_desc`.
@group(0) @binding(4) var<storage, read> d_desc: array<u32>;
@group(0) @binding(5) var<storage, read_write> d_bus: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn direct(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= du.items) {
        return;
    }
    let page = d_items[item * 4u];
    let bus = d_items[item * 4u + 1u];
    let lo = d_items[item * 4u + 2u];
    let hi = d_items[item * 4u + 3u];
    var acc = array<vec2<f32>, 4>();
    for (var i = lo; i < hi; i = i + 1u) {
        let p = d_imp[i];
        let b = p.entry * 7u;
        let off = d_desc[b];
        let right = d_desc[b + 6u];
        let len = min(d_desc[b + 1u], d_desc[b + 2u]);
        let plain = d_desc[b + 3u];
        let rel0 = d_desc[b + 4u];
        let l2r = bitcast<f32>(d_desc[b + 5u]);
        let g = vec2<f32>(p.l, p.r);
        for (var k = 0u; k < DPAGE / 256u; k = k + 1u) {
            // Wraps past `len` for a frame before the impulse.
            let t = page + lid + k * 256u - p.at;
            if (t < len) {
                var v = vec2<f32>(d_real[off + t], d_real[off + right + t]);
                if (t >= plain) {
                    v = v * exp2(f32(rel0 + t - plain) * l2r);
                }
                acc[k] = acc[k] + v * g;
            }
        }
    }
    for (var k = 0u; k < DPAGE / 256u; k = k + 1u) {
        let f = page + lid + k * 256u;
        if (f < du.stride) {
            let o = bus * du.stride + f;
            d_bus[o] = d_bus[o] + acc[k];
        }
    }
}

//#section direct_stopped
// ---------------------------------------------------------------------------
// `direct` for impulses a CC120 cuts in their tails, which a transform cannot
// do. Apart from `direct` because that is every file's inner loop: the larger
// record and the check cost it 15-22% on files with nothing to cut.
// ---------------------------------------------------------------------------

struct Du {
    /// Frames between one bus and the next, and the most a bus has.
    stride: u32,
    items: u32,
    /// Frames a stopped impulse takes to fade.
    fade: u32,
    _p1: u32,
}

/// One impulse: its frame from the window's start, its kernel's row of \[15\]
struct Dstop {
    at: u32,
    entry: u32,
    l: f32,
    r: f32,
    stop: u32,
}

const DPAGE: u32 = 1024u;

@group(0) @binding(0) var<uniform> du: Du;
@group(0) @binding(1) var<storage, read> d_imp: array<Dstop>;
/// `[items][4]`: the page's first frame, its bus, and its `[lo, hi)` in `d_imp`.
@group(0) @binding(2) var<storage, read> d_items: array<u32>;
@group(0) @binding(3) var<storage, read> d_real: array<f32>;
/// `[entries][7]`: as `load_kernels`'s `k_desc`.
@group(0) @binding(4) var<storage, read> d_desc: array<u32>;
@group(0) @binding(5) var<storage, read_write> d_bus: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn direct_stopped(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= du.items) {
        return;
    }
    let page = d_items[item * 4u];
    let bus = d_items[item * 4u + 1u];
    let lo = d_items[item * 4u + 2u];
    let hi = d_items[item * 4u + 3u];
    var acc = array<vec2<f32>, 4>();
    for (var i = lo; i < hi; i = i + 1u) {
        let p = d_imp[i];
        let b = p.entry * 7u;
        let off = d_desc[b];
        let right = d_desc[b + 6u];
        let len = min(d_desc[b + 1u], d_desc[b + 2u]);
        let plain = d_desc[b + 3u];
        let rel0 = d_desc[b + 4u];
        let l2r = bitcast<f32>(d_desc[b + 5u]);
        let g = vec2<f32>(p.l, p.r);
        for (var k = 0u; k < DPAGE / 256u; k = k + 1u) {
            // Wraps past `len` for a frame before the impulse.
            let t = page + lid + k * 256u - p.at;
            if (t < len) {
                var v = vec2<f32>(d_real[off + t], d_real[off + right + t]);
                if (t >= plain) {
                    v = v * exp2(f32(rel0 + t - plain) * l2r);
                }
                if (t >= p.stop) {
                    v = v * max(1.0 - f32(t - p.stop) / f32(du.fade), 0.0);
                }
                acc[k] = acc[k] + v * g;
            }
        }
    }
    for (var k = 0u; k < DPAGE / 256u; k = k + 1u) {
        let f = page + lid + k * 256u;
        if (f < du.stride) {
            let o = bus * du.stride + f;
            d_bus[o] = d_bus[o] + acc[k];
        }
    }
}

//#section bq_chunk
// ---------------------------------------------------------------------------
// Each bus through its filter, `bus.wgsl`'s transposed direct form II with
// `b2 == b0`, its state carried from window to window -- in parallel over the
// window, where a biquad is serial in frames. True per-key runs each key's
// buses over the song, so a serial walk, about 90 ns a frame on one GPU thread,
// would be paid once a key.
//
// The recurrence is linear in its state `s = (z1, z2)`:
//
//     y = z1 + b0 x,   s' = A s + B x,   A = [[-a1, 1], [-a2, 0]]
//
// so a window is cut into chunks of `CHUNK` frames. `bq_chunk` filters each
// chunk from zero state, all at once, and keeps where each ended. `bq_scan`
// walks the chunks of a bus in order -- `frames / CHUNK` steps, not `frames`
// -- turning those ends into each chunk's true start, `start' = A^CHUNK start +
// end`. `bq_fix` adds what each chunk's start contributes, the first row of
// `A^k` against it, `k` frames in. The powers of `A` are the host's, in f64.
//
// **Cells, not buses.** A bus's window is mostly silence on a sparse file, so
// the host lists the (listed bus, chunk) cells that hold anything. `bq_chunk`
// takes the cells written in this window: any other starts and ends at zero.
// `bq_fix` takes those and the cells a start state still rings into, for the
// chunks `ring_chunks` gives each filter, past which it is a billionth of
// what it was. `bq_scan` walks those same cells, and a gap between two of them
// is a state that has rung out, so what crosses it is zero.
// ---------------------------------------------------------------------------

const CHUNK: u32 = 1024u;

@group(0) @binding(0) var<uniform> bq: Eu;
@group(0) @binding(1) var<storage, read> q_coef: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q_on: array<u32>;
/// `[filters]` of `(z1 left, z2 left, z1 right, z2 right)`.
@group(0) @binding(3) var<storage, read_write> q_state: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> q_bus: array<vec2<f32>>;
/// The buses the key has touched, ascending; only they can hold anything.
@group(0) @binding(5) var<storage, read> q_list: array<u32>;
/// `[listed bus][chunk]` states: where each chunk ended from zero, then, once \[16\]
@group(0) @binding(6) var<storage, read_write> q_chunk: array<vec4<f32>>;
/// The cells to filter: `(listed bus, chunk)`.
@group(0) @binding(8) var<storage, read> q_cells: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn bq_chunk(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= bq.filtered) {
        return;
    }
    let chunks = bq.frames / CHUNK;
    let cell = q_cells[gid.x];
    let f = q_list[cell.x];
    if (q_on[f] == 0u) {
        return;
    }
    let c = q_coef[f];
    var z1 = vec2<f32>(0.0, 0.0);
    var z2 = vec2<f32>(0.0, 0.0);
    let base = f * bq.stride + cell.y * CHUNK;
    for (var t = 0u; t < CHUNK; t = t + 1u) {
        let x = q_bus[base + t];
        let y = c.x * x + z1;
        z1 = c.y * x - c.z * y + z2;
        z2 = c.x * x - c.w * y;
        q_bus[base + t] = y;
    }
    q_chunk[cell.x * chunks + cell.y] = vec4<f32>(z1.x, z2.x, z1.y, z2.y);
}

//#section bq_scan
const CHUNK: u32 = 1024u;

@group(0) @binding(0) var<uniform> bq: Eu;
@group(0) @binding(1) var<storage, read> q_coef: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q_on: array<u32>;
@group(0) @binding(3) var<storage, read_write> q_state: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> q_bus: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> q_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> q_chunk: array<vec4<f32>>;
/// Per filter, `CHUNK + 2` pairs: the first row of `A^k` for each `k` below \[17\]
@group(0) @binding(7) var<storage, read> q_pow: array<vec2<f32>>;
/// The cells a state reaches, a bus's chunks ascending.
@group(0) @binding(8) var<storage, read> q_cells: array<vec2<u32>>;
/// Per listed bus, its `[lo, hi)` in `q_cells`.
@group(0) @binding(9) var<storage, read> q_ranges: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn bq_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= bq.count) {
        return;
    }
    let f = q_list[gid.x];
    if (q_on[f] == 0u) {
        return;
    }
    let chunks = bq.frames / CHUNK;
    let p0 = q_pow[f * (CHUNK + 2u) + CHUNK];
    let p1 = q_pow[f * (CHUNK + 2u) + CHUNK + 1u];
    let range = q_ranges[gid.x];
    var s = q_state[f];
    // The chunk `s` stands at.
    var next = 0u;
    for (var c = range.x; c < range.y; c = c + 1u) {
        let i = q_cells[c].y;
        if (i != next) {
            s = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        next = i + 1u;
        let at = gid.x * chunks + i;
        let e = q_chunk[at];
        q_chunk[at] = s;
        // Left is (x, y), right is (z, w).
        s = vec4<f32>(
            p0.x * s.x + p0.y * s.y + e.x,
            p1.x * s.x + p1.y * s.y + e.y,
            p0.x * s.z + p0.y * s.w + e.z,
            p1.x * s.z + p1.y * s.w + e.w,
        );
    }
    if (next != chunks) {
        s = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    q_state[f] = s;
}

//#section bq_fix
const CHUNK: u32 = 1024u;

@group(0) @binding(0) var<uniform> bq: Eu;
@group(0) @binding(1) var<storage, read> q_coef: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q_on: array<u32>;
@group(0) @binding(3) var<storage, read_write> q_state: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> q_bus: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> q_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> q_chunk: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> q_pow: array<vec2<f32>>;
/// The cells a state reaches: `(listed bus, chunk)`.
@group(0) @binding(8) var<storage, read> q_cells: array<vec2<u32>>;

@compute @workgroup_size(256, 1, 1)
fn bq_fix(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= bq.ringing) {
        return;
    }
    let cell = q_cells[item];
    let f = q_list[cell.x];
    if (q_on[f] == 0u) {
        return;
    }
    let s = q_chunk[cell.x * (bq.frames / CHUNK) + cell.y];
    let base = f * bq.stride + cell.y * CHUNK;
    for (var k = 0u; k < CHUNK / 256u; k = k + 1u) {
        let t = lid + k * 256u;
        let r = q_pow[f * (CHUNK + 2u) + t];
        q_bus[base + t] = q_bus[base + t] + vec2<f32>(r.x * s.x + r.y * s.y, r.x * s.z + r.y * s.w);
    }
}

//#section mix
// ---------------------------------------------------------------------------
// The key's buses, in bus order, added into the song's master at the window's
// frame: one add a frame, so the master sums keys in the order they come. A
// workgroup a page, each with the buses that hold anything there.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> xu: Eu;
@group(0) @binding(1) var<storage, read> x_bus: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> x_list: array<u32>;
@group(0) @binding(3) var<storage, read_write> x_out: array<vec2<f32>>;
/// `[pages][3]`: the page's first frame and its `[lo, hi)` in `x_list`.
@group(0) @binding(4) var<storage, read> x_items: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn mix(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= xu.mixes) {
        return;
    }
    let page = x_items[item * 3u];
    let lo = x_items[item * 3u + 1u];
    let hi = x_items[item * 3u + 2u];
    for (var k = 0u; k < 4u; k = k + 1u) {
        let t = page + lid + k * 256u;
        var acc = vec2<f32>(0.0, 0.0);
        for (var i = lo; i < hi; i = i + 1u) {
            acc = acc + x_bus[x_list[i] * xu.stride + t];
        }
        x_out[xu.at + t] = x_out[xu.at + t] + acc;
    }
}

//#section carry_move
// ---------------------------------------------------------------------------
// The next window starts with what this one's blocks left past its end: the
// pages of `[frames, frames + carry)` that hold anything move to `[0, carry)`,
// and the window's other pages that hold anything are cleared. `carry <=
// frames`, so the two never overlap. The pages it moved from are cleared by
// `carry_clear`, a pass later, so no invocation reads what another clears.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> cu: Eu;
@group(0) @binding(1) var<storage, read_write> c_bus: array<vec2<f32>>;
/// `[moves][3]`: the bus, the page's first frame, and whether it takes the \[18\]
@group(0) @binding(2) var<storage, read> c_items: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn carry_move(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= cu.moves) {
        return;
    }
    let o = c_items[item * 3u] * cu.stride + c_items[item * 3u + 1u];
    let from_carry = c_items[item * 3u + 2u] != 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let t = o + lid + k * 256u;
        if (from_carry) {
            c_bus[t] = c_bus[t + cu.frames];
        } else {
            c_bus[t] = vec2<f32>(0.0, 0.0);
        }
    }
}

//#section carry_clear
@group(0) @binding(0) var<uniform> cu: Eu;
@group(0) @binding(1) var<storage, read_write> c_bus: array<vec2<f32>>;
/// `[clears][2]`: the bus and the page's first frame from the window's end.
@group(0) @binding(2) var<storage, read> c_items: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn carry_clear(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= cu.clears) {
        return;
    }
    let o = c_items[item * 2u] * cu.stride + cu.frames + c_items[item * 2u + 1u];
    for (var k = 0u; k < 4u; k = k + 1u) {
        c_bus[o + lid + k * 256u] = vec2<f32>(0.0, 0.0);
    }
}

//#section warp
// ---------------------------------------------------------------------------
// A channel's notes a bend moves under, read back. They were rendered unbent, in
// `tau`, each key's into a master of its own; here the song's frames take them
// at the channel's clock, a run of one pitch to a workgroup, with a four-point
// interpolation. No two runs share a frame.
// ---------------------------------------------------------------------------

struct Wu {
    items: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

const W_RUN: u32 = 5u;

@group(0) @binding(0) var<uniform> wu: Wu;
/// `[items][5]`: the song's first frame, whole `tau`, its fraction, `tau` a \[19\]
@group(0) @binding(1) var<storage, read> w_items: array<u32>;
@group(0) @binding(2) var<storage, read> w_src: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> w_dst: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn warp(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= wu.items) {
        return;
    }
    let b = item * W_RUN;
    if (lid >= w_items[b + 4u]) {
        return;
    }
    let t = bitcast<f32>(w_items[b + 2u]) + bitcast<f32>(w_items[b + 3u]) * f32(lid);
    let whole = floor(t);
    let x = t - whole;
    let i = w_items[b + 1u] + u32(whole);
    let last = arrayLength(&w_src) - 1u;
    let a = w_src[max(i, 1u) - 1u];
    let p = w_src[min(i, last)];
    let q = w_src[min(i + 1u, last)];
    let d = w_src[min(i + 2u, last)];
    let y = p + 0.5 * x * (q - a + x * (2.0 * a - 5.0 * p + 4.0 * q - d + x * (3.0 * (p - q) + d - a)));
    let o = w_items[b] + lid;
    w_dst[o] = w_dst[o] + y;
}
