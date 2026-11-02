// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pool: array<u32>;
{{POOL_PARTS}}

struct SJob {
    base: u32,
    len: u32,
    ls: u32,
    le: u32,
    looping: u32,
    step_hi: u32,
    step_lo: u32,
    /// Frames of attack and hold.
    n_ah: u32,
    attack_rate: f32,
    /// `log2(decay_coef)`, or the fall a frame for a linear decay.
    decay: f32,
    /// The first frame at `sustain`.
    n_sus: u32,
    sustain: f32,
    mod_delay: u32,
    mod_inc: u32,
    /// `modLfoToVolume` in centibels; zero has no tremolo.
    volume: i32,
    _p: u32,
}

struct SPage {
    job: u32,
    /// Its first frame's word in `s_out`.
    out: u32,
    /// Its first frame, and how many it has.
    n0: u32,
    frames: u32,
    phase_hi: u32,
    phase_lo: u32,
}

@group(0) @binding(2) var<storage, read> s_jobs: array<SJob>;
@group(0) @binding(3) var<storage, read> s_pages: array<SPage>;
@group(0) @binding(4) var<storage, read_write> s_out: array<f32>;

{{SAMPLE}}

/// `kernel::env_at`: the level after `n + 1` envelope steps.
fn level_at(j: SJob, n: u32) -> f32 {
    if (n >= j.n_sus) { return j.sustain; }
    let k = n + 1u;
    if (k < j.n_ah) { return min(f32(k) * j.attack_rate, 1.0); }
    if (k == j.n_ah) { return 1.0; }
    let d = f32(k - j.n_ah);
    if (u.exp_decay != 0u) { return exp2(d * j.decay); }
    return 1.0 - j.decay * d;
}

/// The position `k` frames into a page: `(hi, lo)` of its 32.32 phase, wrapped \[2\]
fn position(pg: SPage, j: SJob, k: u32, looping: bool) -> vec2<u32> {
    let a = k * (j.step_lo & 0xFFFFu);
    let m = k * (j.step_lo >> 16u);
    let low = a + (m << 16u);
    let c0 = select(0u, 1u, low < a);
    let lo = pg.phase_lo + low;
    let c1 = select(0u, 1u, lo < low);
    var hi = pg.phase_hi + k * j.step_hi + (m >> 16u) + c0 + c1;
    if (looping && hi >= j.le) {
        hi = j.ls + (hi - j.ls) % max(j.le - j.ls, 1u);
    }
    return vec2<u32>(hi, lo);
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nw.x;
    if (item >= arrayLength(&s_pages)) {
        return;
    }
    let pg = s_pages[item];
    let j = s_jobs[pg.job];
    let looping = j.looping != 0u;
    for (var r = 0u; r < 4u; r = r + 1u) {
        let k = lid + r * 256u;
        if (k >= pg.frames) {
            continue;
        }
        let n = pg.n0 + k;
        let at = position(pg, j, k, looping);
        var y = 0.0;
        if (looping || at.x < j.len) {
            let s = interpolate(j.base, at.x, frac_of(at.y), looping, j.ls, j.le, j.len);
            y = s * min(level_at(j, n), 1.0);
            if (j.volume != 0) {
                // [3]
                let t = n / GATE_TILE * GATE_TILE;
                let g0 = tremolo(j.mod_delay, j.mod_inc, j.volume, t);
                let g1 = tremolo(j.mod_delay, j.mod_inc, j.volume, t + GATE_TILE);
                y = y * (g0 + (g1 - g0) * f32(n - t) * INV_GATE_TILE);
            }
        }
        s_out[pg.out + k] = y;
    }
}
