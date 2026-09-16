// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// [1]

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> voices: array<u32>;
@group(0) @binding(2) var<storage, read_write> voices_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> scan: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(5) var<storage, read_write> state: array<u32>;
@group(0) @binding(6) var<storage, read_write> sort_keys: array<u32>;

var<workgroup> sh_scan: array<u32, WG>;

fn is_alive(i: u32) -> bool {
    return voices[F_ENV_STAGE * u.capacity + i] != ENV_DEAD;
}

@compute @workgroup_size({{WG}})
fn scan_local(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let live = state[S_LIVE];
    // [2]
    let blocks = (live + WG - 1u) / WG;
    var block = wgid.x;
    loop {
        if (block >= blocks) { break; }
        let i = block * WG + tid;
        var alive = 0u;
        if (i < live && is_alive(i)) { alive = 1u; }

        sh_scan[tid] = alive;
        workgroupBarrier();

        // Hillis-Steele inclusive scan. WG is a power of two.
        var offset = 1u;
        loop {
            if (offset >= WG) { break; }
            var v = 0u;
            if (tid >= offset) { v = sh_scan[tid - offset]; }
            workgroupBarrier();
            sh_scan[tid] = sh_scan[tid] + v;
            workgroupBarrier();
            offset = offset << 1u;
        }

        if (i < u.capacity) {
            scan[i] = sh_scan[tid] - alive; // exclusive
        }
        if (tid == WG - 1u) {
            block_sums[block] = sh_scan[tid];
        }
        // [3]
        workgroupBarrier();
        block = block + nwg.x;
    }
}

@compute @workgroup_size({{WG}})
fn scan_blocks(@builtin(local_invocation_index) tid: u32) {
    let blocks = (state[S_LIVE] + WG - 1u) / WG;
    var running = 0u;
    var chunk = 0u;
    loop {
        if (chunk * WG >= blocks) { break; }
        let i = chunk * WG + tid;
        var v = 0u;
        if (i < blocks) { v = block_sums[i]; }

        sh_scan[tid] = v;
        workgroupBarrier();
        var offset = 1u;
        loop {
            if (offset >= WG) { break; }
            var t = 0u;
            if (tid >= offset) { t = sh_scan[tid - offset]; }
            workgroupBarrier();
            sh_scan[tid] = sh_scan[tid] + t;
            workgroupBarrier();
            offset = offset << 1u;
        }

        let total = sh_scan[WG - 1u];
        if (i < blocks) {
            block_sums[i] = running + sh_scan[tid] - v; // exclusive, global
        }
        workgroupBarrier();
        running = running + total;
        chunk = chunk + 1u;
    }
    if (tid == 0u) {
        state[S_LIVE_NEW] = running;
    }
}

@compute @workgroup_size({{WG}})
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let live = state[S_LIVE];
    var i = gid.x;
    loop {
        if (i >= live) { break; }
        if (is_alive(i)) {
            let dst = block_sums[i / WG] + scan[i];
            if (dst < c) {
                for (var f = 0u; f < VOICE_FIELDS; f = f + 1u) {
                    voices_out[f * c + dst] = voices[f * c + i];
                }
                // [4]
                voices_out[F_START_REL * c + dst] = 0u;
                voices_out[F_BORN_VARIANT * c + dst] = 0u;
                voices_out[F_STOP_REL * c + dst] = 0u;
                // [5]
                sort_keys[dst] = (min(voices[F_REGION * c + i], 0x1FFFFFFFu) << 3u)
                    | min(voices[F_ENV_STAGE * c + i], 7u);
            }
        }
        i = i + stride;
    }
}

// [6]
@compute @workgroup_size({{WG}})
fn mark_stolen(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let t_hi = state[S_THRESH_HI];
    let t_lo = state[S_THRESH_LO];
    let live = state[S_LIVE];
    var i = gid.x;
    loop {
        if (i >= live) { break; }
        let lo = voices[F_NOTE_LO * c + i];
        var lvl = 0u;
        if (u.steal_by_level != 0u) { lvl = voices[F_ENV_LEVEL * c + i]; }
        let k = steal_key(voices[F_NOTE_HI * c + i], lo, lvl);
        if (!less64(t_hi, t_lo, k.x, k.y)) { // key <= threshold
            // [7]
            let span = u.block_frames - STEAL_FADE;
            voices[F_STOP_REL * c + i] = (lo % span) + 1u;
        }
        i = i + stride;
    }
}

// [8]
@compute @workgroup_size(1)
fn commit() {
    state[S_LIVE] = state[S_LIVE_NEW];
}

@compute @workgroup_size(1)
fn note_stolen() {
    state[S_STOLEN] = state[S_STOLEN] + u.steal_k;
}
