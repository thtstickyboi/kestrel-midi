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
    // [8]
    let pre = mod_env_pre_release(p, min(age, release_age));
    if (age <= release_age) { return pre; }
    return max(pre - quantise_level(f32(age - release_age) * p.release_rate), 0.0);
}

// [9]
const CHAN_ENABLED: bool = {{CHAN}};

// [10]
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

// [11]
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
        // [12]
        let i = base + idx;
        let w = i >> 1u;
        let p0 = word_pair(w);
        let p1 = word_pair(w + 1u);
        let odd = (i & 1u) == 1u;
        let s0 = select(p0.x, p0.y, odd);
        var s1 = select(p0.y, p1.x, odd);
        // [13]
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

// [14]
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

    // [15]
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
        // [16]
        var d_gain_l = 0.0;
        var d_gain_r = 0.0;
        var z1 = 0.0;
        var z2 = 0.0;
        var gate_slot = 0u;
        var ordinal = 0u;
        var start_rel = 0u;
        // Frame this voice was stolen at, plus one; zero if it was not.
        var stop_rel = 0u;
        // [17]
        var release_frame = NO_RELEASE;
        var p: RegionParams;
        var use_filter = false;
        // [18]
        var cb0 = 1.0;
        var cb1 = 0.0;
        var ca1 = 0.0;
        var ca2 = 0.0;
        var db0 = 0.0;
        var db1 = 0.0;
        var da1 = 0.0;
        var da2 = 0.0;
        // [19]
        var filter_mix = 0.0;
        var d_filter_mix = 0.0;
        var params_base = 0u;
        var variant = 0u;
        // [20]
        var born_variant = 0u;
        // [21]
        var born_bias = 0u;

        var base_step_hi = 0u;
        var base_step_lo = 0u;
        // [22]
        var bent_hi = 0u;
        var bent_lo = 0u;
        var age0 = 0u;
        // [23]
        var mp: ModEnvParams;
        var use_menv = false;
        var menv_filter = false;
        var rel_age = NO_RELEASE;
        // Tremolo, as a linear gain. One for a voice with no volume LFO.
        var lfo_gain = 1.0;
        var base_gain_l = 0.0;
        var base_gain_r = 0.0;

        if (is_live) {
            let c = u.capacity;
            phase_lo = voices[F_PHASE_LO * c + v];
            phase_hi = voices[F_PHASE_HI * c + v];
            // [24]
            base_step_lo = voices[F_STEP_LO * c + v];
            base_step_hi = voices[F_STEP_HI * c + v];
            step_lo = base_step_lo;
            step_hi = base_step_hi;
            bent_lo = base_step_lo;
            bent_hi = base_step_hi;
            age0 = voices[F_AGE * c + v];
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
            gate_slot = voices[F_GATE_SLOT * c + v];
            ordinal = voices[F_ORDINAL * c + v];
            start_rel = voices[F_START_REL * c + v];
            stop_rel = voices[F_STOP_REL * c + v];
            params_base = voices[F_PARAMS * c + v];
            born_variant = voices[F_BORN_VARIANT * c + v];
            born_bias = born_variant >> 16u;
            born_variant = born_variant & 0xFFFFu;
            // [25]
            if (stage < ENV_RELEASE) {
                let obase = gates[gate_slot * 2u];
                if (ordinal <= obase) {
                    // Released before this block began.
                    release_frame = 0u;
                } else {
                    let olo = gates[gate_slot * 2u + 1u];
                    let ohi = gates[(gate_slot + 1u) * 2u + 1u];
                    let j = ordinal - obase;
                    if (j <= ohi - olo) {
                        release_frame = gates[OFF_META_WORDS + olo + j - 1u];
                    }
                }
            }
            if (born_variant != 0u) { variant = born_variant - 1u; }
            p = params[params_base + variant * u.params_per_variant];
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
                    // [26]
                    if (release_frame != NO_RELEASE) {
                        rel_age = age0 + release_frame;
                    }
                }
            }
        }

        let loop_enabled = (vflags & VF_LOOP) != 0u;
        let loop_until_release = (vflags & VF_LOOP_UNTIL_RELEASE) != 0u;
        // [27]
        let channel = gate_slot >> 7u;
        // [28]
        let born_gate = start_rel / GATE_TILE;

        for (var tile = 0u; tile < u.tiles; tile = tile + 1u) {
            // [29]
            if (is_live && (tile % TILES_PER_GATE) == 0u) {
                let gt = tile / TILES_PER_GATE;
                // [30]
                if (CHAN_ENABLED && u.chan_active != 0u) {
                    let ci = (gt * BEND_CHANNELS + channel) * CHAN_FIELDS;
                    // [31]
                    let bias = select(
                        0u, born_bias, born_variant != 0u && gt <= born_gate
                    );
                    let gi = ((gt + bias) * BEND_CHANNELS + channel) * CHAN_FIELDS;
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
                        let tgt_l = base_gain_l * bitcast<f32>(chan[gi + CHAN_GAIN_L]);
                        let tgt_r = base_gain_r * bitcast<f32>(chan[gi + CHAN_GAIN_R]);
                        // [32]
                        if (GAIN_RAMP && gt > born_gate) {
                            // [33]
                            d_gain_l = (tgt_l - gain_l) * INV_GATE_TILE;
                            d_gain_r = (tgt_r - gain_r) * INV_GATE_TILE;
                        } else {
                            gain_l = tgt_l;
                            gain_r = tgt_r;
                            d_gain_l = 0.0;
                            d_gain_r = 0.0;
                        }
                    }
                    // [34]
                    if ((u.chan_active & CHAN_ACTIVE_CUT) != 0u) {
                        let cut = chan[ci + CHAN_CUT];
                        if (cut != 0u && stop_rel == 0u) {
                            // [35]
                            let cap = u.capacity;
                            let id_lo = voices[F_NOTE_LO * cap + v];
                            let id_hi = voices[F_NOTE_HI * cap + v];
                            if (less64(id_hi, id_lo,
                                       chan[ci + CHAN_CUT_ID_HI],
                                       chan[ci + CHAN_CUT_ID_LO])) {
                                stop_rel = cut;
                            }
                        }
                    }
                    // [36]
                    db0 = 0.0;
                    db1 = 0.0;
                    da1 = 0.0;
                    da2 = 0.0;
                    d_filter_mix = 0.0;
                    let want = select(0u, chan[ci + CHAN_VARIANT],
                                      (u.chan_active & CHAN_ACTIVE_VARIANT) != 0u);
                    // [37]
                    let born_here = born_variant != 0u && gt <= born_gate;
                    if (want != variant && !born_here) {
                        variant = want;
                        let was_filtering = use_filter;
                        p = params[params_base + variant * u.params_per_variant];
                        use_filter = (p.flags & RP_FILTER) != 0u;
                        // [38]
                        if (USE_MOD_ENV) {
                            use_menv = (p.flags & RP_MOD_ENV) != 0u;
                            if (use_menv) {
                                mp = menv[params_base
                                          + variant * u.params_per_variant];
                            }
                            menv_filter = use_menv && mp.to_filter != 0.0;
                        }
                        // [39]
                        if (FILTER_RAMP && gt > born_gate
                            && use_filter && was_filtering) {
                            db0 = (p.b0 - cb0) * INV_GATE_TILE;
                            db1 = (p.b1 - cb1) * INV_GATE_TILE;
                            da1 = (p.a1 - ca1) * INV_GATE_TILE;
                            da2 = (p.a2 - ca2) * INV_GATE_TILE;
                        } else if (FILTER_RAMP && gt > born_gate
                                   && use_filter != was_filtering) {
                            if (use_filter) {
                                // [40]
                                z1 = 0.0;
                                z2 = 0.0;
                                cb0 = p.b0;
                                cb1 = p.b1;
                                ca1 = p.a1;
                                ca2 = p.a2;
                                filter_mix = 0.0;
                                d_filter_mix = INV_GATE_TILE;
                            } else {
                                // [41]
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
            }

            // [42]
            if ((USE_LFO || USE_MOD_ENV) && is_live) {
                // [43]
                let age = age0 + tile * TILE;
                var cents = 0.0;
                if (USE_LFO) {
                    let vp = rp_vib_pitch(p);
                    let mlp = rp_mod_pitch(p);
                    let mv = rp_mod_volume(p);
                    if (vp != 0.0 && age >= rp_vib_delay(p)) {
                        cents = cents + vp
                            * lfo_tri((age - rp_vib_delay(p)) * p.vib_lfo_inc);
                    }
                    var mod_lfo = 0.0;
                    if ((mlp != 0.0 || mv != 0.0) && age >= rp_mod_delay(p)) {
                        mod_lfo = lfo_tri((age - rp_mod_delay(p)) * p.mod_lfo_inc);
                        cents = cents + mlp * mod_lfo;
                    }
                    // [44]
                    if (mv != 0.0) {
                        lfo_gain = exp2(-(mod_lfo * mv) * (1.0 / 60.205999));
                    } else {
                        lfo_gain = 1.0;
                    }
                }
                var menv_factor = 0u;
                if (USE_MOD_ENV && use_menv) {
                    let l = mod_env_level(mp, age, rel_age);
                    // [45]
                    if (mp.to_pitch != 0.0) {
                        let i = mod_env_pitch_index(mp.to_pitch * l, u.menv_factor_half);
                        menv_factor = menv_factors[i];
                    }
                    if (menv_filter) {
                        // [46]
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
                if (cents != 0.0) {
                    // 8.24, the same fixed-point factor bend uses.
                    let factor = u32(exp2(cents * (1.0 / 1200.0)) * 16777216.0);
                    let st = scale64(bent_hi, bent_lo, factor);
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
                    // [47]
                    if (stage < ENV_RELEASE && f >= release_frame) {
                        stage = ENV_RELEASE;
                        level = min(level, 1.0);
                        release_frame = NO_RELEASE;
                    }

                    let looping = loop_enabled
                        && !(loop_until_release && stage >= ENV_RELEASE);

                    if (!looping && phase_hi >= smp_len) {
                        stage = ENV_DEAD;
                    } else {
                        // [48]
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
                            if (u.exp_release != 0u) {
                                level = level * p.release_coef;
                            } else {
                                level = level - p.release_coef;
                            }
                            if (level <= u.env_floor) {
                                stage = ENV_DEAD;
                                level = 0.0;
                            }
                        }

                        let s = interpolate(
                            smp_base, phase_hi, frac_of(phase_lo),
                            looping, loop_start, loop_end, smp_len
                        );
                        let g = min(level, 1.0);
                        let x = s * g;

                        // Transposed direct form II. b2 == b0.
                        y = x;
                        // [49]
                        if (FILTER_RAMP && d_filter_mix != 0.0) {
                            // [50]
                            let fy = cb0 * x + z1;
                            z1 = cb1 * x - ca1 * fy + z2;
                            z2 = cb0 * x - ca2 * fy;
                            y = x + (fy - x) * filter_mix;
                            // [51]
                            filter_mix = filter_mix + d_filter_mix;
                        } else if (use_filter) {
                            y = cb0 * x + z1;
                            z1 = cb1 * x - ca1 * y + z2;
                            z2 = cb0 * x - ca2 * y;
                        }

                        // [52]
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
                            // [53]
                            let span = max(loop_end - loop_start, 1u);
                            phase_hi = phase_hi - span;
                            if (phase_hi >= loop_end) {
                                phase_hi = loop_start + (phase_hi - loop_start) % span;
                            }
                        }
                    }
                }

                if (USE_LFO) { y = y * lfo_gain; }
                sh[(i * 2u) * WG + tid] = y * gain_l;
                sh[(i * 2u + 1u) * WG + tid] = y * gain_r;
                // [54]
                gain_l = gain_l + d_gain_l;
                gain_r = gain_r + d_gain_r;
                cb0 = cb0 + db0;
                cb1 = cb1 + db1;
                ca1 = ca1 + da1;
                ca2 = ca2 + da2;
            }

            // [55]
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
            if (USE_LFO || USE_MOD_ENV) {
                voices[F_AGE * c + v] = age0 + u.block_frames;
            }
            if (USE_MOD_ENV) { voices[F_REL_AGE * c + v] = rel_age; }
        }

        batch = batch + nwg;
    }
}
