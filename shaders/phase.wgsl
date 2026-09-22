// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Compiled into analytic render pipelines only. Coefficients were prepared on
// the host: no trigonometric functions or normalization in the sample loop.
@group(0) @binding(10) var<storage, read> phase_data: array<u32>;
const PRESERVE_PHASE_ATTACK: bool = {{PRESERVE_PHASE_ATTACK}};

fn rotation_tap(base: u32, idx: u32, phase_info: vec3<u32>, rotation: vec3<f32>) -> f32 {
    let original = fetch(base, idx);
    if (PRESERVE_PHASE_ATTACK && idx < phase_info.y) { return original; }
    let quadrature = bitcast<f32>(phase_data[phase_info.x + idx]);
    let changed = (rotation.x * original - rotation.y * quadrature) * rotation.z;
    if (!PRESERVE_PHASE_ATTACK || phase_info.z == 0u || idx - phase_info.y >= phase_info.z) {
        return changed;
    }
    let t = f32(idx - phase_info.y) / f32(phase_info.z);
    let blend = t * t * (3.0 - 2.0 * t);
    return original + (changed - original) * blend;
}

fn interpolate_rotation(base: u32, idx: u32, frac: f32,
    looping: bool, ls: u32, le: u32, len: u32,
    phase_info: vec3<u32>, rotation: vec3<f32>) -> f32 {
    let s0 = rotation_tap(base, idx, phase_info, rotation);
    if (u.interp == INTERP_NEAREST) { return s0; }
    let i1 = neighbour_index(idx, 1, looping, ls, le, len);
    let s1 = rotation_tap(base, i1, phase_info, rotation);
    if (u.interp == INTERP_LINEAR) { return s0 + (s1 - s0) * frac; }
    let im1 = neighbour_index(idx, -1, looping, ls, le, len);
    let i2 = neighbour_index(idx, 2, looping, ls, le, len);
    let sm1 = rotation_tap(base, im1, phase_info, rotation);
    let s2 = rotation_tap(base, i2, phase_info, rotation);
    let a = -0.5 * sm1 + 1.5 * s0 - 1.5 * s1 + 0.5 * s2;
    let b = sm1 - 2.5 * s0 + 2.0 * s1 - 0.5 * s2;
    let c = -0.5 * sm1 + 0.5 * s1;
    return ((a * frac + b) * frac + c) * frac + s0;
}
