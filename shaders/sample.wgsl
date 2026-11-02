// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

// [2]
fn at_least(a: u32, b: u32) -> u32 {
    return 1u - ((a - b) >> 31u);
}

// 1 for a non-zero depth, else 0, the same way.
fn nonzero(v: i32) -> u32 {
    return (bitcast<u32>(v) | bitcast<u32>(-v)) >> 31u;
}

// [3]
fn tremolo(d: u32, inc: u32, volume: i32, age: u32) -> f32 {
    let on = f32(at_least(age, d) * nonzero(volume));
    let m = lfo_tri((age - d) * inc) * on;
    let x = -(m * f32(volume)) * (1.0 / 60.205999);
    return 1.0 + (exp2(x) - 1.0) * on;
}

// One 32-bit word of the pool holds two consecutive frames.
fn word_pair(w: u32) -> vec2<f32> {
    if (w >= u.pool_words) { return vec2<f32>(0.0, 0.0); }{{POOL_FETCH}}
    return unpack2x16snorm(pool[w]);
}

// [4]
fn word_pairs(w: u32) -> vec4<f32> {{{POOL_PAIRS}}
    return vec4<f32>(word_pair(w), word_pair(w + 1u));
}

fn fetch(base: u32, idx: u32) -> f32 {
    let i = base + idx;
    let p = word_pair(i >> 1u);
    if ((i & 1u) == 1u) { return p.y; }
    return p.x;
}

// [5]
fn next_index(idx: u32, looping: bool, ls: u32, le: u32, len: u32) -> u32 {
    let n = idx + 1u;
    if (looping) {
        if (n >= le) { return ls; }
        return n;
    }
    if (n >= len) { return len - 1u; }
    return n;
}

fn interpolate(
    base: u32, idx: u32, frac: f32,
    looping: bool, ls: u32, le: u32, len: u32
) -> f32 {
    if (u.interp == INTERP_NEAREST) {
        return fetch(base, idx);
    }
    if (u.interp == INTERP_LINEAR) {
        // [6]
        let i = base + idx;
        let w = i >> 1u;
        let q = word_pairs(w);
        let p0 = q.xy;
        let p1 = q.zw;
        let odd = (i & 1u) == 1u;
        let s0 = select(p0.x, p0.y, odd);
        var s1 = select(p0.y, p1.x, odd);
        // [7]
        let n = next_index(idx, looping, ls, le, len);
        if (n != idx + 1u) { s1 = fetch(base, n); }
        return s0 + (s1 - s0) * frac;
    }
    let im1 = neighbour_index(idx, -1, looping, ls, le, len);
    let i1 = neighbour_index(idx, 1, looping, ls, le, len);
    let i2 = neighbour_index(idx, 2, looping, ls, le, len);
    let sm1 = fetch(base, im1);
    let s0 = fetch(base, idx);
    let s1 = fetch(base, i1);
    let s2 = fetch(base, i2);
    let a = -0.5 * sm1 + 1.5 * s0 - 1.5 * s1 + 0.5 * s2;
    let b = sm1 - 2.5 * s0 + 2.0 * s1 - 0.5 * s2;
    let c = -0.5 * sm1 + 0.5 * s1;
    return ((a * frac + b) * frac + c) * frac + s0;
}
