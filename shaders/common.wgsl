// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

// [2]

const F_PHASE_LO: u32   = 0u;
const F_PHASE_HI: u32   = 1u;
const F_STEP_LO: u32    = 2u;
const F_STEP_HI: u32    = 3u;
const F_SMP_BASE: u32   = 4u;
const F_SMP_LEN: u32    = 5u;
const F_LOOP_START: u32 = 6u;
const F_LOOP_END: u32   = 7u;
const F_FLAGS: u32      = 8u;
const F_ENV_STAGE: u32  = 9u;
const F_ENV_LEVEL: u32  = 10u;
const F_GAIN_L: u32     = 11u;
const F_GAIN_R: u32     = 12u;
const F_FILT_Z1: u32    = 13u;
const F_FILT_Z2: u32    = 14u;
const F_PARAMS: u32     = 15u;
const F_REGION: u32     = 16u;
const F_GATE_SLOT: u32  = 17u;
const F_ORDINAL: u32    = 18u;
const F_START_REL: u32  = 19u;
const F_NOTE_LO: u32    = 20u;
// [3]
const F_NOTE_HI: u32    = 21u;
// [4]
const F_BORN_VARIANT: u32 = 22u;
// [5]
const F_STOP_REL: u32 = 23u;
// [6]
const F_AGE: u32 = 24u;
// [7]
const F_REL_AGE: u32 = 25u;
const VOICE_FIELDS: u32 = {{VOICE_FIELDS}}u;
// [8]
const ANALYTIC: bool = {{ANALYTIC}};
const F_ROT_C: u32 = {{ROTATION_BASE}}u;
const F_ROT_S: u32 = F_ROT_C + 1u;
const F_ROT_SCALE: u32 = F_ROT_C + 2u;

const ENV_ATTACK: u32  = 0u;
const ENV_DECAY: u32   = 1u;
const ENV_SUSTAIN: u32 = 2u;
const ENV_RELEASE: u32 = 3u;
const ENV_DEAD: u32    = 4u;

const VF_LOOP: u32 = 1u;
const VF_LOOP_UNTIL_RELEASE: u32 = 2u;

const RP_FILTER: u32 = 1u;
const RP_MOD_ENV: u32 = 2u;
// [9]
const RP_SHORT_RELEASE: u32 = 4u;

// [10]
const SLOT_MASK: u32 = 0x7FFFu;
// The part of F_NOTE_HI that is the note id. Mirrors `voice::NOTE_HI_MASK`.
const NOTE_HI_MASK: u32 = 0xFFFFu;

const INTERP_NEAREST: u32 = 0u;
const INTERP_LINEAR: u32 = 1u;
const INTERP_CUBIC: u32 = 2u;

// [11]

const S_LIVE: u32       = 0u;
const S_LIVE_NEW: u32   = 1u;
const S_PREFIX_HI: u32  = 2u;
const S_PREFIX_LO: u32  = 3u;
const S_K: u32          = 4u;
const S_BYTE: u32       = 5u;
const S_THRESH_HI: u32  = 6u;
const S_THRESH_LO: u32  = 7u;
const S_STOLEN: u32     = 8u;
const S_DROPPED: u32    = 9u;
const S_PEAK_BITS: u32  = 10u;
const S_SORT_BIT: u32   = 11u;
const S_TOTAL0: u32     = 12u;
const STATE_SLOTS: u32  = 16u;

// Substituted from Config so the compiler sees literals.
const WG: u32 = {{WG}}u;
// Frames a stolen voice fades over. See Config::steal_fade_frames.
const STEAL_FADE: u32 = {{STEAL_FADE}}u;
const TILE: u32 = {{TILE}}u;
// Frames between note-off gate checks. A multiple of TILE.
const GATE_TILE: u32 = {{GATE_TILE}}u;
const TILES_PER_GATE: u32 = GATE_TILE / TILE;
// [12]
const GAIN_RAMP: bool = {{GAIN_RAMP}};
const INV_GATE_TILE: f32 = 1.0 / f32(GATE_TILE);
// Ramp biquad coefficients across a gate tile. See `Config::filter_ramp`.
const FILTER_RAMP: bool = {{FILTER_RAMP}};
// A release frame no voice can reach, meaning "nothing due".
const NO_RELEASE: u32 = 0xFFFFFFFFu;
// [13]
const ENV_STEP: u32 = {{ENV_STEP}}u;
// [14]
const NOTE_GRID: bool = {{NOTE_GRID}};
// [15]
const GRID_SHIFT: u32 = 15u;
const GRID_MASK: u32 = 0x1FFFu;
// Whether this build evaluates SF2 LFOs at all. See `Config::lfo_enabled`.
const USE_LFO: bool = {{USE_LFO}};
// [16]
const USE_LFO_VOLUME: bool = {{USE_LFO_VOLUME}};
const USE_LFO_PITCH: bool = {{USE_LFO_PITCH}};
// [17]
const USE_MOD_ENV: bool = {{USE_MOD_ENV}};
// [18]
const SAMPLE_RATE_F: f32 = {{SAMPLE_RATE}}.0;
// [19]
const MOD_ENV_ATTACK_SCALE: f32 = 0.12542917;

struct RegionParams {
    attack_rate: f32,
    attack_end: f32,
    decay_coef: f32,
    decay_target: f32,
    sustain: f32,
    release_coef: f32,
    b0: f32,
    b1: f32,
    a1: f32,
    a2: f32,
    // [20]
    flags: u32,
    mod_lfo_inc: u32,
    vib_lfo_inc: u32,
    /// Start delays in frames: mod low half, vib high half.
    lfo_delays: u32,
    /// Peak pitch deviation in cents as two i16: mod low, vib high.
    lfo_pitch: u32,
};

// [21]
struct ModEnvParams {
    delay_frames: u32,
    attack_frames: u32,
    hold_frames: u32,
    decay_rate: f32,
    sustain: f32,
    release_rate: f32,
    to_pitch: f32,
    to_filter: f32,
    fc_cents: f32,
    q_gain: f32,
    q_inv_2q: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
};

// Unpackers, mirroring the accessors on `bank::RegionParams`.
fn rp_mod_delay(p: RegionParams) -> u32 { return p.lfo_delays & 0xFFFFu; }
fn rp_vib_delay(p: RegionParams) -> u32 { return p.lfo_delays >> 16u; }
fn rp_mod_pitch(p: RegionParams) -> f32 {
    return f32(bitcast<i32>(p.lfo_pitch << 16u) >> 16u);
}
fn rp_vib_pitch(p: RegionParams) -> f32 {
    return f32(bitcast<i32>(p.lfo_pitch) >> 16u);
}
fn rp_mod_volume(p: RegionParams) -> f32 {
    return f32(bitcast<i32>(p.flags) >> 16u);
}

/// SF2's LFO waveform: a triangle starting at zero and rising, in [-1, 1]. \[22\]
fn lfo_tri(phase: u32) -> f32 {
    let t = f32(phase + 0x40000000u) * (1.0 / 4294967296.0);
    return 1.0 - abs(4.0 * t - 2.0);
}

struct Uniforms {
    block_frames: u32,
    tiles: u32,
    capacity: u32,
    spawn_count: u32,

    render_workgroups: u32,
    interp: u32,
    exp_decay: u32,
    exp_release: u32,

    env_floor: f32,
    pool_words: u32,
    steal_k: u32,
    sort_bits: u32,

    sort_region_shift: u32,
    sort_stage_shift: u32,
    sort_phase_shift: u32,
    sort_phase_mask: u32,

    sort_dead_region: u32,
    chan_active: u32,
    params_per_variant: u32,
    /// Non-zero to steal by envelope level rather than by age.
    steal_by_level: u32,

    /// Half-width of the modulation envelope's pitch-factor table, in \[23\]
    menv_factor_half: u32,
    /// Where the shared `log2` mantissa table starts in `menv_factors`.
    menv_log2_base: u32,
    /// Where portamento's speeds, and then its exponent table, start in \[24\]
    glide_base: u32,
    /// Frame 0 of this block's position on the envelope grid: the frames \[25\]
    env_phase: u32,

    /// Channels in a controller row: sixteen for each MIDI port the file has \[26\]
    chan_count: u32,
    /// Words of `[base, run]` meta before the note-off runs in `gates`, which \[27\]
    off_meta_words: u32,
    _pad0: u32,
    _pad1: u32,
};

// Bits of Uniforms::chan_active.
const CHAN_ACTIVE_BEND: u32 = 1u;
const CHAN_ACTIVE_GAIN: u32 = 2u;
const CHAN_ACTIVE_VARIANT: u32 = 4u;
const CHAN_ACTIVE_CUT: u32 = 8u;

// [28]

// [29]
const CHAN_FIELDS: u32 = 8u;
const CHAN_BEND: u32 = 0u;
const CHAN_GAIN_L: u32 = 1u;
const CHAN_GAIN_R: u32 = 2u;
const CHAN_VARIANT: u32 = 3u;
const CHAN_CUT: u32 = 4u;
// [30]
const CHAN_CUT_ID_LO: u32 = 5u;
const CHAN_CUT_ID_HI: u32 = 6u;
// Fractional bits in a bend factor. Matches BEND_FRAC_BITS on the host.
const BEND_SHIFT: u32 = 24u;

// [31]
fn steal_key(hi: u32, lo: u32, level_bits: u32) -> vec2<u32> {
    if (u.steal_by_level == 0u) {
        return vec2<u32>(hi & NOTE_HI_MASK, lo);
    }
    let level = max(bitcast<f32>(level_bits), 0.0);
    let q = u32(min(level, 1.0) * 65535.0);
    return vec2<u32>((q << 16u) | (hi & 0xFFFFu), lo);
}

// [32]
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid1 = p10 + (p00 >> 16u);
    let mid2 = p01 + (mid1 & 0xFFFFu);
    let lo = (mid2 << 16u) | (p00 & 0xFFFFu);
    let hi = p11 + (mid1 >> 16u) + (mid2 >> 16u);
    return vec2<u32>(lo, hi);
}

// [33]
fn scale64(hi: u32, lo: u32, factor: u32) -> vec2<u32> {
    let pl = mul32(lo, factor);   // product bits 0..63
    let ph = mul32(hi, factor);   // product bits 32..95
    let b = pl.y + ph.x;          // bits 32..63
    var c = ph.y;                 // bits 64..95
    if (b < ph.x) { c = c + 1u; } // carry out of b
    if ((c >> BEND_SHIFT) != 0u) {
        return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    let r_lo = (pl.x >> BEND_SHIFT) | (b << (32u - BEND_SHIFT));
    let r_hi = (b >> BEND_SHIFT) | (c << (32u - BEND_SHIFT));
    return vec2<u32>(r_hi, r_lo);
}

fn add64(hi: u32, lo: u32, add_hi: u32, add_lo: u32) -> vec2<u32> {
    let nlo = lo + add_lo;
    var carry = 0u;
    if (nlo < lo) { carry = 1u; }
    return vec2<u32>(hi + add_hi + carry, nlo);
}

// True when a < b, treating each pair as one 64-bit value.
fn less64(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32) -> bool {
    if (a_hi != b_hi) { return a_hi < b_hi; }
    return a_lo < b_lo;
}

fn frac_of(lo: u32) -> f32 {
    return f32(lo) * (1.0 / 4294967296.0);
}

// [34]
fn neighbour_index(idx: u32, off: i32, looping: bool, ls: u32, le: u32, len: u32) -> u32 {
    let raw = i32(idx) + off;
    if (looping) {
        if (raw >= i32(le)) {
            let span = max(i32(le) - i32(ls), 1);
            return u32(i32(ls) + (raw - i32(ls)) % span);
        }
        if (raw < 0) { return 0u; }
        return u32(raw);
    }
    return u32(clamp(raw, 0, i32(len) - 1));
}

// [35]

// [36]
const MOD_ENV_LOG2_BITS: u32 = 10u;
const MOD_ENV_LOG2_FRAC_BITS: u32 = 8u;

// [37]
fn biquad_lowpass_pre(fc: f32, q_gain: f32, inv_2q: f32, sr: f32) -> vec4<f32> {
    let w0 = 6.2831855 * clamp(fc / sr, 1.0e-5, 0.49);
    let sin_w0 = sin(w0);
    let cos_w0 = cos(w0);
    let alpha = sin_w0 * inv_2q;

    let a0 = 1.0 + alpha;
    let b0 = q_gain * (1.0 - cos_w0) * 0.5 / a0;
    let b1 = q_gain * (1.0 - cos_w0) / a0;
    let a1 = -2.0 * cos_w0 / a0;
    let a2 = (1.0 - alpha) / a0;
    return vec4<f32>(b0, b1, a1, a2);
}

// [38]
const MOD_ENV_CENTS_STEPS: f32 = 64.0;

// [39]
fn mod_env_pitch_index(cents: f32, half: u32) -> u32 {
    // [40]
    let q = round(cents * MOD_ENV_CENTS_STEPS);
    let i = q + f32(half);
    return u32(clamp(i, 0.0, f32(half * 2u)));
}

// [41]
fn quantise_level(x: f32) -> f32 {
    let t = i32(clamp(x, -2.0, 2.0) * 4194304.0);
    return f32(t) * (1.0 / 4194304.0);
}

// Absolute cents to hertz. Mirrors `bank::cents_to_hz`.
fn cents_to_hz(cents: f32) -> f32 {
    return 8.176 * exp2(cents / 1200.0);
}
