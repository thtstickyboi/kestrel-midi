// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pool: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<RegionParams>;
@group(0) @binding(3) var<storage, read> gates: array<u32>;
@group(0) @binding(4) var<storage, read_write> voices: array<u32>;
// [2]
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read> state: array<u32>;
@group(0) @binding(7) var<storage, read> chan: array<u32>;
// [3]
@group(0) @binding(8) var<storage, read> menv: array<ModEnvParams>;
// [4]
@group(0) @binding(9) var<storage, read> menv_factors: array<u32>;

// [5]

// [6]
fn log2_exact(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let e = i32((bits >> 23u) & 0xFFu) - 127;
    let man = bits & 0x007FFFFFu;
    let i = man >> (23u - MOD_ENV_LOG2_BITS);
    let frac = (man >> (23u - MOD_ENV_LOG2_BITS - MOD_ENV_LOG2_FRAC_BITS))
        & ((1u << MOD_ENV_LOG2_FRAC_BITS) - 1u);
    let a = menv_factors[u.menv_log2_base + i];
    let b = menv_factors[u.menv_log2_base + i + 1u];
    let v = a + (((b - a) * frac) >> MOD_ENV_LOG2_FRAC_BITS);
    return f32(e) + f32(v) * (1.0 / 1073741824.0);
}

// [7]
fn mod_env_attack_level(x: f32) -> f32 {
    if (x <= 0.0) { return 0.0; }
    return clamp(1.0 + quantise_level(log2_exact(x) * MOD_ENV_ATTACK_SCALE), 0.0, 1.0);
}

// [8]
fn note_age(raw: u32) -> u32 {
    return select(raw, 0u, raw >= 0x80000000u);
}

// [9]
fn grid_pos(slot_word: u32, f: u32) -> u32 {
    return (u.env_phase + f + ENV_STEP - ((slot_word >> GRID_SHIFT) & GRID_MASK)) % ENV_STEP;
}

// [10]
fn release_start(slot_word: u32, off: u32) -> u32 {
    return off + (ENV_STEP - grid_pos(slot_word, off)) % ENV_STEP + 1u;
}

// [11]
fn fall_left(slot_word: u32, f: u32) -> u32 {
    return ENV_STEP - (grid_pos(slot_word, f) + ENV_STEP - 1u) % ENV_STEP;
}

// [12]
fn release_fall(p: RegionParams, level: f32, left: u32) -> f32 {
    if ((p.flags & RP_SHORT_RELEASE) != 0u) {
        return level / f32(left);
    }
    return select(p.release_coef, 0.0, u.exp_release != 0u);
}

// [13]
fn at_least(a: u32, b: u32) -> u32 {
    return 1u - ((a - b) >> 31u);
}

// 1 for a non-zero depth, else 0, the same way.
fn nonzero(v: i32) -> u32 {
    return (bitcast<u32>(v) | bitcast<u32>(-v)) >> 31u;
}

// [14]
fn lfo_tremolo(p: RegionParams, age: u32) -> f32 {
    let d = rp_mod_delay(p);
    let on = f32(at_least(age, d) * nonzero(bitcast<i32>(p.flags) >> 16u));
    let m = lfo_tri((age - d) * p.mod_lfo_inc) * on;
    let x = -(m * rp_mod_volume(p)) * (1.0 / 60.205999);
    return 1.0 + (exp2(x) - 1.0) * on;
}

fn mod_env_pre_release(p: ModEnvParams, age: u32) -> f32 {
    if (age < p.delay_frames) { return 0.0; }
    var a = age - p.delay_frames;
    if (a < p.attack_frames) {
        return mod_env_attack_level(f32(a) / f32(p.attack_frames));
    }
    a = a - p.attack_frames;
    if (a < p.hold_frames) { return 1.0; }
    a = a - p.hold_frames;
    return max(1.0 - quantise_level(f32(a) * p.decay_rate), p.sustain);
}

fn mod_env_level(p: ModEnvParams, age: u32, release_age: u32) -> f32 {
    // [15]
    let pre = mod_env_pre_release(p, min(age, release_age));
    if (age <= release_age) { return pre; }
    return max(pre - quantise_level(f32(age - release_age) * p.release_rate), 0.0);
}

// [16]
const CHAN_ENABLED: bool = {{CHAN}};

// [17]
const GLIDE: bool = {{GLIDE}};

// [18]
const GLIDE_FLAG_BITS: u32 = 7u;
const GLIDE_REM_SHIFT: u32 = 3u;
const GLIDE_RATE_SHIFT: u32 = 24u;
const GLIDE_UP_BIT: u32 = 0x80000000u;
const GLIDE_RATES: u32 = 128u;
const GLIDE_OCTAVE: u32 = 19200u;
const GLIDE_MID_OCTAVES: u32 = 11u;
const GLIDE_MID: u32 = GLIDE_MID_OCTAVES * GLIDE_OCTAVE;
const GLIDE_UP_MAX: u32 = 8u * GLIDE_OCTAVE - 1u;
const GLIDE_DOWN_MAX: u32 = GLIDE_MID;

// [19]
fn glide_factor(flags: u32, word: u32, f: u32) -> u32 {
    let r = flags >> GLIDE_REM_SHIFT;
    let rem = select(0u, r - f, r > f);
    let rq = menv_factors[u.glide_base + ((word >> GLIDE_RATE_SHIFT) & 0x7Fu)];
    // `(rq * rem) >> 24`, from the full 64-bit product.
    let p = mul32(rq, rem);
    let raw = (p.x >> 24u) | (p.y << 8u);
    let up = (word & GLIDE_UP_BIT) != 0u;
    let off = min(raw, select(GLIDE_DOWN_MAX, GLIDE_UP_MAX, up));
    let m = select(GLIDE_MID - off, GLIDE_MID + off, up);
    let oct = m / GLIDE_OCTAVE;
    let e = menv_factors[u.glide_base + GLIDE_RATES + (m - oct * GLIDE_OCTAVE)];
    let k = GLIDE_MID_OCTAVES + 6u;
    return select(e >> (k - oct), e << (oct - k), oct > k);
}

// The flags word after this block. Mirrors `porta::advance`.
fn glide_advance(flags: u32) -> u32 {
    if ((flags >> GLIDE_REM_SHIFT) > u.block_frames) {
        return flags - (u.block_frames << GLIDE_REM_SHIFT);
    }
    return flags & GLIDE_FLAG_BITS;
}

// [20]
const M: u32 = TILE * 2u;
// Threads cooperating on one lane during the first reduction level.
const PER_LANE: u32 = WG / M;

var<workgroup> sh: array<f32, WG * M>;
var<workgroup> sh2: array<f32, WG>;

// One 32-bit word of the pool holds two consecutive frames.
fn word_pair(w: u32) -> vec2<f32> {
    if (w >= u.pool_words) { return vec2<f32>(0.0, 0.0); }
    return unpack2x16snorm(pool[w]);
}

fn fetch(base: u32, idx: u32) -> f32 {
    let i = base + idx;
    let p = word_pair(i >> 1u);
    if ((i & 1u) == 1u) { return p.y; }
    return p.x;
}

// [21]
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
        // [22]
        let i = base + idx;
        let w = i >> 1u;
        let p0 = word_pair(w);
        let p1 = word_pair(w + 1u);
        let odd = (i & 1u) == 1u;
        let s0 = select(p0.x, p0.y, odd);
        var s1 = select(p0.y, p1.x, odd);
        // [23]
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

// [24]
{{PHASE_FUNCTIONS}}

// [25]
fn reduce_into_partials(tid: u32, wg: u32, nwg: u32, first_sample: u32) {
    let lane = tid / PER_LANE;
    let chunk = tid % PER_LANE;

    var acc = 0.0;
    for (var k = 0u; k < M; k = k + 1u) {
        acc = acc + sh[lane * WG + chunk + k * PER_LANE];
    }
    sh2[lane * PER_LANE + chunk] = acc;
    workgroupBarrier();

    if (tid < M) {
        var total = 0.0;
        for (var k = 0u; k < PER_LANE; k = k + 1u) {
            total = total + sh2[tid * PER_LANE + k];
        }
        let j = first_sample + tid;
        partials[j * nwg + wg] = partials[j * nwg + wg] + total;
    }
}

@compute @workgroup_size({{WG}})
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let wg = wgid.x;
    let nwg = u.render_workgroups;
    let live = state[S_LIVE];
    let samples = u.block_frames * 2u;

    // [26]
    for (var j = tid; j < samples; j = j + WG) {
        partials[j * nwg + wg] = 0.0;
    }
    workgroupBarrier();

    var batch = wg;
    loop {
        if (batch * WG >= live) { break; }
        let v = batch * WG + tid;
        let is_live = v < live;

        // ---- load voice state into registers ----
        var phase_hi = 0u;
        var phase_lo = 0u;
        var step_hi = 0u;
        var step_lo = 0u;
        var smp_base = 0u;
        var smp_len = 1u;
        var loop_start = 0u;
        var loop_end = 1u;
        var vflags = 0u;
        var stage = ENV_DEAD;
        var level = 0.0;
        var gain_l = 0.0;
        var gain_r = 0.0;
        // [27]
        var d_gain_l = 0.0;
        var d_gain_r = 0.0;
        var z1 = 0.0;
        var z2 = 0.0;
        var gate_slot = 0u;
        // [28]
        var slot_word = 0u;
        // [29]
        var glide_word = 0u;
        var ordinal = 0u;
        var start_rel = 0u;
        // Frame this voice was stolen at, plus one; zero if it was not.
        var stop_rel = 0u;
        // [30]
        var release_frame = NO_RELEASE;
        // [31]
        var rel_d = 0.0;
        var p: RegionParams;
        var use_filter = false;
        // [32]
        var cb0 = 1.0;
        var cb1 = 0.0;
        var ca1 = 0.0;
        var ca2 = 0.0;
        var db0 = 0.0;
        var db1 = 0.0;
        var da1 = 0.0;
        var da2 = 0.0;
        // [33]
        var filter_mix = 0.0;
        var d_filter_mix = 0.0;
        var params_base = 0u;
        var variant = 0u;
        // [34]
        var born_variant = 0u;
        // [35]
        var born_bias = 0u;

        var base_step_hi = 0u;
        var base_step_lo = 0u;
        // [36]
        var bent_hi = 0u;
        var bent_lo = 0u;
        var age0 = 0u;
        // [37]
        var mp: ModEnvParams;
        var use_menv = false;
        var menv_filter = false;
        var rel_age = NO_RELEASE;
        // [38]
        var lfo_factor = 1u << BEND_SHIFT;
        var base_gain_l = 0.0;
        var base_gain_r = 0.0;
        var rotation_meta = vec3<u32>(0u);
        var rotation = vec3<f32>(1.0, 0.0, 1.0);

        if (is_live) {
            let c = u.capacity;
            {{PHASE_LOAD}}
            phase_lo = voices[F_PHASE_LO * c + v];
            phase_hi = voices[F_PHASE_HI * c + v];
            // [39]
            base_step_lo = voices[F_STEP_LO * c + v];
            base_step_hi = voices[F_STEP_HI * c + v];
            step_lo = base_step_lo;
            step_hi = base_step_hi;
            bent_lo = base_step_lo;
            bent_hi = base_step_hi;
            // Only when the field exists; see the matching write in spawn.
            if (USE_LFO || USE_MOD_ENV) { age0 = voices[F_AGE * c + v]; }
            smp_base = voices[F_SMP_BASE * c + v];
            smp_len = voices[F_SMP_LEN * c + v];
            loop_start = voices[F_LOOP_START * c + v];
            loop_end = voices[F_LOOP_END * c + v];
            vflags = voices[F_FLAGS * c + v];
            stage = voices[F_ENV_STAGE * c + v];
            level = bitcast<f32>(voices[F_ENV_LEVEL * c + v]);
            base_gain_l = bitcast<f32>(voices[F_GAIN_L * c + v]);
            base_gain_r = bitcast<f32>(voices[F_GAIN_R * c + v]);
            gain_l = base_gain_l;
            gain_r = base_gain_r;
            z1 = bitcast<f32>(voices[F_FILT_Z1 * c + v]);
            z2 = bitcast<f32>(voices[F_FILT_Z2 * c + v]);
            slot_word = voices[F_GATE_SLOT * c + v];
            gate_slot = slot_word & SLOT_MASK;
            if (GLIDE) {
                glide_word = voices[F_NOTE_HI * c + v];
            }
            ordinal = voices[F_ORDINAL * c + v];
            start_rel = voices[F_START_REL * c + v];
            stop_rel = voices[F_STOP_REL * c + v];
            params_base = voices[F_PARAMS * c + v];
            born_variant = voices[F_BORN_VARIANT * c + v];
            born_bias = born_variant >> 16u;
            born_variant = born_variant & 0xFFFFu;
            // [40]
            let born_here = born_variant != 0u;
            if (stage < ENV_RELEASE) {
                let obase = gates[gate_slot * 2u];
                if (ordinal <= obase) {
                    // [41]
                    release_frame = 0u;
                    if (NOTE_GRID && !born_here) {
                        release_frame = fall_left(slot_word, 0u) % ENV_STEP;
                    }
                } else {
                    let first_run = gates[gate_slot * 2u + 1u];
                    let end_run = gates[(gate_slot + 1u) * 2u + 1u];
                    let j = ordinal - obase;
                    if (end_run > first_run && j <= gates[u.off_meta_words + (end_run - 1u) * 2u]) {
                        var lo_run = first_run;
                        var hi_run = end_run - 1u;
                        while (lo_run < hi_run) {
                            let mid_run = lo_run + (hi_run - lo_run) / 2u;
                            if (gates[u.off_meta_words + mid_run * 2u] >= j) {
                                hi_run = mid_run;
                            } else {
                                lo_run = mid_run + 1u;
                            }
                        }
                        let off = gates[u.off_meta_words + lo_run * 2u + 1u];
                        release_frame = off;
                        if (NOTE_GRID && !(born_here && off <= start_rel)) {
                            release_frame = release_start(slot_word, off);
                        }
                    }
                }
            }
            if (born_variant != 0u) { variant = born_variant - 1u; }
            p = params[params_base + variant * u.params_per_variant];
            if (NOTE_GRID && stage == ENV_RELEASE) {
                rel_d = release_fall(p, level, fall_left(slot_word, 0u));
            }
            use_filter = (p.flags & RP_FILTER) != 0u;
            cb0 = p.b0;
            cb1 = p.b1;
            ca1 = p.a1;
            ca2 = p.a2;
            filter_mix = select(0.0, 1.0, use_filter);
            if (USE_MOD_ENV) {
                rel_age = voices[F_REL_AGE * c + v];
                use_menv = (p.flags & RP_MOD_ENV) != 0u;
                if (use_menv) {
                    mp = menv[params_base + variant * u.params_per_variant];
                    menv_filter = mp.to_filter != 0.0;
                    // [42]
                    if (release_frame != NO_RELEASE) {
                        rel_age = note_age(age0 + release_frame);
                    }
                }
            }
        }

        let loop_enabled = (vflags & VF_LOOP) != 0u;
        let loop_until_release = (vflags & VF_LOOP_UNTIL_RELEASE) != 0u;
        // [43]
        let channel = gate_slot >> 7u;
        // [44]
        let born_gate = start_rel / GATE_TILE;

        for (var tile = 0u; tile < u.tiles; tile = tile + 1u) {
            // [45]
            if (is_live && (tile % TILES_PER_GATE) == 0u) {
                let gt = tile / TILES_PER_GATE;
                // [46]
                var tgt_l = base_gain_l;
                var tgt_r = base_gain_r;
                var snap = gt == 0u;
                // [47]
                if (CHAN_ENABLED && u.chan_active != 0u) {
                    let ci = (gt * u.chan_count + channel) * CHAN_FIELDS;
                    // [48]
                    let bias = select(
                        0u, born_bias, born_variant != 0u && gt <= born_gate
                    );
                    let gi = ((gt + bias) * u.chan_count + channel) * CHAN_FIELDS;
                    if ((u.chan_active & CHAN_ACTIVE_BEND) != 0u) {
                        let st = scale64(
                            base_step_hi, base_step_lo, chan[gi + CHAN_BEND]
                        );
                        bent_hi = st.x;
                        bent_lo = st.y;
                        step_hi = st.x;
                        step_lo = st.y;
                    }
                    if ((u.chan_active & CHAN_ACTIVE_GAIN) != 0u) {
                        tgt_l = base_gain_l * bitcast<f32>(chan[gi + CHAN_GAIN_L]);
                        tgt_r = base_gain_r * bitcast<f32>(chan[gi + CHAN_GAIN_R]);
                        // [49]
                        if (GAIN_RAMP && gt > born_gate) {
                            // [50]
                            d_gain_l = (tgt_l - gain_l) * INV_GATE_TILE;
                            d_gain_r = (tgt_r - gain_r) * INV_GATE_TILE;
                        } else {
                            gain_l = tgt_l;
                            gain_r = tgt_r;
                            d_gain_l = 0.0;
                            d_gain_r = 0.0;
                            snap = true;
                        }
                    }
                    // [51]
                    if ((u.chan_active & CHAN_ACTIVE_CUT) != 0u) {
                        let cut = chan[ci + CHAN_CUT];
                        if (cut != 0u && stop_rel == 0u) {
                            // [52]
                            let cap = u.capacity;
                            let id_lo = voices[F_NOTE_LO * cap + v];
                            let id_hi = voices[F_NOTE_HI * cap + v] & NOTE_HI_MASK;
                            if (less64(id_hi, id_lo,
                                       chan[ci + CHAN_CUT_ID_HI],
                                       chan[ci + CHAN_CUT_ID_LO])) {
                                stop_rel = cut;
                            }
                        }
                    }
                    // [53]
                    db0 = 0.0;
                    db1 = 0.0;
                    da1 = 0.0;
                    da2 = 0.0;
                    d_filter_mix = 0.0;
                    let want = select(0u, chan[ci + CHAN_VARIANT],
                                      (u.chan_active & CHAN_ACTIVE_VARIANT) != 0u);
                    // [54]
                    let born_here = born_variant != 0u && gt <= born_gate;
                    if (want != variant && !born_here) {
                        variant = want;
                        let was_filtering = use_filter;
                        p = params[params_base + variant * u.params_per_variant];
                        use_filter = (p.flags & RP_FILTER) != 0u;
                        // [55]
                        if (NOTE_GRID && stage == ENV_RELEASE) {
                            let sw = voices[F_GATE_SLOT * u.capacity + v];
                            rel_d = release_fall(p, level, fall_left(sw, tile * TILE));
                        }
                        // [56]
                        if (USE_MOD_ENV) {
                            use_menv = (p.flags & RP_MOD_ENV) != 0u;
                            if (use_menv) {
                                mp = menv[params_base
                                          + variant * u.params_per_variant];
                            }
                            menv_filter = use_menv && mp.to_filter != 0.0;
                        }
                        // [57]
                        if (FILTER_RAMP && gt > born_gate
                            && use_filter && was_filtering) {
                            db0 = (p.b0 - cb0) * INV_GATE_TILE;
                            db1 = (p.b1 - cb1) * INV_GATE_TILE;
                            da1 = (p.a1 - ca1) * INV_GATE_TILE;
                            da2 = (p.a2 - ca2) * INV_GATE_TILE;
                        } else if (FILTER_RAMP && gt > born_gate
                                   && use_filter != was_filtering) {
                            if (use_filter) {
                                // [58]
                                z1 = 0.0;
                                z2 = 0.0;
                                cb0 = p.b0;
                                cb1 = p.b1;
                                ca1 = p.a1;
                                ca2 = p.a2;
                                filter_mix = 0.0;
                                d_filter_mix = INV_GATE_TILE;
                            } else {
                                // [59]
                                filter_mix = 1.0;
                                d_filter_mix = -INV_GATE_TILE;
                            }
                            db0 = 0.0;
                            db1 = 0.0;
                            da1 = 0.0;
                            da2 = 0.0;
                        } else {
                            cb0 = p.b0;
                            cb1 = p.b1;
                            ca1 = p.a1;
                            ca2 = p.a2;
                            db0 = 0.0;
                            db1 = 0.0;
                            da1 = 0.0;
                            da2 = 0.0;
                            filter_mix = select(0.0, 1.0, use_filter);
                        }
                    }
                }
                // [60]
                if (GLIDE && vflags > GLIDE_FLAG_BITS) {
                    var from_hi = base_step_hi;
                    var from_lo = base_step_lo;
                    if (CHAN_ENABLED && (u.chan_active & CHAN_ACTIVE_BEND) != 0u) {
                        from_hi = bent_hi;
                        from_lo = bent_lo;
                    }
                    let f = max(gt * GATE_TILE, start_rel);
                    let st = scale64(from_hi, from_lo, glide_factor(vflags, glide_word, f));
                    bent_hi = st.x;
                    bent_lo = st.y;
                    step_hi = st.x;
                    step_lo = st.y;
                }
                // [61]
                if (USE_LFO_VOLUME) {
                    // [62]
                    let lp = params[params_base + variant * u.params_per_variant];
                    let now = lfo_tremolo(lp, note_age(age0 + tile * TILE));
                    let trem = lfo_tremolo(lp, note_age(age0 + (tile + TILES_PER_GATE) * TILE));
                    // [63]
                    gain_l = select(gain_l, tgt_l * now, snap);
                    gain_r = select(gain_r, tgt_r * now, snap);
                    d_gain_l = (tgt_l * trem - gain_l) * INV_GATE_TILE;
                    d_gain_r = (tgt_r * trem - gain_r) * INV_GATE_TILE;
                }
                if (USE_LFO_PITCH) {
                    let lp = params[params_base + variant * u.params_per_variant];
                    let age = note_age(age0 + tile * TILE);
                    let vp = bitcast<i32>(lp.lfo_pitch) >> 16u;
                    let mlp = bitcast<i32>(lp.lfo_pitch << 16u) >> 16u;
                    let on_vib = at_least(age, rp_vib_delay(lp)) * nonzero(vp);
                    let on_mod = at_least(age, rp_mod_delay(lp)) * nonzero(mlp);
                    let vib = lfo_tri((age - rp_vib_delay(lp)) * lp.vib_lfo_inc) * f32(on_vib);
                    let modl = lfo_tri((age - rp_mod_delay(lp)) * lp.mod_lfo_inc) * f32(on_mod);
                    let cents = f32(vp) * vib + f32(mlp) * modl;
                    // [64]
                    let one = 1u << BEND_SHIFT;
                    let fx = u32(exp2(cents * (1.0 / 1200.0)) * 16777216.0);
                    lfo_factor = one + (fx - one) * (on_vib | on_mod);
                    let st = scale64(bent_hi, bent_lo, lfo_factor);
                    step_hi = st.x;
                    step_lo = st.y;
                }
            }

            // [65]
            if (USE_MOD_ENV && is_live) {
                // [66]
                let age = note_age(age0 + tile * TILE);
                var menv_factor = 0u;
                if (USE_MOD_ENV && use_menv) {
                    let l = mod_env_level(mp, age, rel_age);
                    // [67]
                    if (mp.to_pitch != 0.0) {
                        let i = mod_env_pitch_index(mp.to_pitch * l, u.menv_factor_half);
                        menv_factor = menv_factors[i];
                    }
                    if (menv_filter) {
                        // [68]
                        let fc = cents_to_hz(mp.fc_cents + mp.to_filter * l);
                        let co = biquad_lowpass_pre(fc, mp.q_gain, mp.q_inv_2q, SAMPLE_RATE_F);
                        cb0 = co.x;
                        cb1 = co.y;
                        ca1 = co.z;
                        ca2 = co.w;
                        db0 = 0.0;
                        db1 = 0.0;
                        da1 = 0.0;
                        da2 = 0.0;
                    }
                }
                // [69]
                if (USE_LFO_PITCH) {
                    let st = scale64(bent_hi, bent_lo, lfo_factor);
                    step_hi = st.x;
                    step_lo = st.y;
                } else {
                    step_hi = bent_hi;
                    step_lo = bent_lo;
                }
                if (menv_factor != 0u) {
                    let st = scale64(step_hi, step_lo, menv_factor);
                    step_hi = st.x;
                    step_lo = st.y;
                }
            }

            let f0 = tile * TILE;
            for (var i = 0u; i < TILE; i = i + 1u) {
                var y = 0.0;
                let f = f0 + i;

                if (is_live && stage != ENV_DEAD && f >= start_rel) {
                    // [70]
                    if (stage < ENV_RELEASE && f >= release_frame) {
                        stage = ENV_RELEASE;
                        level = min(level, 1.0);
                        release_frame = NO_RELEASE;
                        if (NOTE_GRID) {
                            rel_d = release_fall(p, level, ENV_STEP);
                        }
                    }

                    let looping = loop_enabled
                        && !(loop_until_release && stage >= ENV_RELEASE);

                    if (!looping && phase_hi >= smp_len) {
                        stage = ENV_DEAD;
                    } else {
                        // [71]
                        if (stage == ENV_ATTACK) {
                            level = level + p.attack_rate;
                            if (level >= p.attack_end) {
                                stage = ENV_DECAY;
                                level = 1.0;
                            }
                        } else if (stage == ENV_DECAY) {
                            if (u.exp_decay != 0u) {
                                level = level * p.decay_coef;
                            } else {
                                level = level - p.decay_coef;
                            }
                            if (level <= p.decay_target) {
                                if (p.sustain <= u.env_floor) {
                                    stage = ENV_DEAD;
                                    level = 0.0;
                                } else {
                                    stage = ENV_SUSTAIN;
                                    level = p.sustain;
                                }
                            }
                        } else if (stage == ENV_RELEASE) {
                            if (NOTE_GRID) {
                                level = select(level * p.release_coef, level - rel_d, rel_d != 0.0);
                            } else if (u.exp_release != 0u) {
                                level = level * p.release_coef;
                            } else {
                                level = level - p.release_coef;
                            }
                            if (level <= u.env_floor) {
                                stage = ENV_DEAD;
                                level = 0.0;
                            }
                        }

                        {{PHASE_SAMPLE}}
                        let g = min(level, 1.0);
                        let x = s * g;

                        // Transposed direct form II. b2 == b0.
                        y = x;
                        // [72]
                        if (FILTER_RAMP && d_filter_mix != 0.0) {
                            // [73]
                            let fy = cb0 * x + z1;
                            z1 = cb1 * x - ca1 * fy + z2;
                            z2 = cb0 * x - ca2 * fy;
                            y = x + (fy - x) * filter_mix;
                            // [74]
                            filter_mix = filter_mix + d_filter_mix;
                        } else if (use_filter) {
                            y = cb0 * x + z1;
                            z1 = cb1 * x - ca1 * y + z2;
                            z2 = cb0 * x - ca2 * y;
                        }

                        // [75]
                        if (stop_rel != 0u && f + 1u >= stop_rel) {
                            let d = f + 1u - stop_rel;
                            if (d >= STEAL_FADE) {
                                y = 0.0;
                                stage = ENV_DEAD;
                            } else {
                                y = y * (1.0 - f32(d) / f32(STEAL_FADE));
                            }
                        }

                        let np = add64(phase_hi, phase_lo, step_hi, step_lo);
                        phase_hi = np.x;
                        phase_lo = np.y;
                        if (looping && phase_hi >= loop_end) {
                            // [76]
                            let span = max(loop_end - loop_start, 1u);
                            phase_hi = phase_hi - span;
                            if (phase_hi >= loop_end) {
                                phase_hi = loop_start + (phase_hi - loop_start) % span;
                            }
                        }
                    }
                }
                sh[(i * 2u) * WG + tid] = y * gain_l;
                sh[(i * 2u + 1u) * WG + tid] = y * gain_r;
                // [77]
                gain_l = gain_l + d_gain_l;
                gain_r = gain_r + d_gain_r;
                cb0 = cb0 + db0;
                cb1 = cb1 + db1;
                ca1 = ca1 + da1;
                ca2 = ca2 + da2;
            }

            // [78]
            workgroupBarrier();
            reduce_into_partials(tid, wg, nwg, f0 * 2u);
        }

        // ---- write voice state back ----
        if (is_live) {
            let c = u.capacity;
            voices[F_PHASE_LO * c + v] = phase_lo;
            voices[F_PHASE_HI * c + v] = phase_hi;
            voices[F_ENV_STAGE * c + v] = stage;
            voices[F_ENV_LEVEL * c + v] = bitcast<u32>(level);
            voices[F_FILT_Z1 * c + v] = bitcast<u32>(z1);
            voices[F_FILT_Z2 * c + v] = bitcast<u32>(z2);
            voices[F_START_REL * c + v] = 0u;
            voices[F_BORN_VARIANT * c + v] = 0u;
            voices[F_STOP_REL * c + v] = 0u;
            if (GLIDE && vflags > GLIDE_FLAG_BITS) {
                voices[F_FLAGS * c + v] = glide_advance(vflags);
            }
            if (USE_LFO || USE_MOD_ENV) {
                voices[F_AGE * c + v] = age0 + u.block_frames;
            }
            if (USE_MOD_ENV) { voices[F_REL_AGE * c + v] = rel_age; }
        }

        batch = batch + nwg;
    }
}
