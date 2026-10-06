// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The GPU self-test: how long this GPU takes over one block, at voice counts \[1\]

use crate::bank::Bank;
use crate::config::Config;
use crate::driver::Driver;
use crate::Backend;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// The first step. Small, because an integrated GPU is slow: this laptop's \[2\]
pub const FIRST_VOICES: u32 = 4_096;
/// Stop once a block takes this long: the next step, twice the voices, then \[3\]
pub const STOP_MS: f64 = 250.0;
/// Windows' limit, and the share of it the verdict leaves as margin.
pub const TDR_MS: f64 = 2000.0;
const TARGET_MS: f64 = 1000.0;
/// Blocks rendered at each step: the one that spawns every voice, then the \[4\]
const BLOCKS: usize = 6;
/// The sample pool, in MiB.
const POOL_MB: usize = 128;
/// No step past this; a card that is fast enough here is fast enough.
const MAX_VOICES: u32 = 1 << 24;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Step {
    /// The pool's size, and how many voices were sounding.
    pub voices: u32,
    pub live: u64,
    /// The block that spawned them all, and the slowest and the middle of \[5\]
    pub spawn_ms: f64,
    pub worst_ms: f64,
    pub typical_ms: f64,
    pub timed_on_gpu: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Outcome {
    pub adapter: String,
    pub backend: String,
    pub steps: Vec<Step>,
    /// Why it stopped where it did.
    pub stopped: String,
    /// The answer, in a sentence or two.
    pub verdict: String,
    /// The same, as numbers: ms per block at the default 1,048,576 voices, \[6\]
    pub ms_at_default: Option<f64>,
    pub voices_for_1s: Option<u64>,
}

/// Write the self-test's soundfont into `dir` and load it. Shared by every \[7\]
pub fn prepare(dir: &Path) -> Result<Arc<Bank>> {
    std::fs::create_dir_all(dir)?;
    let sf = dir.join("selftest.sf2");
    let cfg = Config::default();
    crate::testkit::big_sf2(&sf, cfg.sample_rate, POOL_MB)?;
    Ok(Arc::new(crate::load_bank(&sf, &cfg)?))
}

/// Run the self-test on one adapter and backend, as `gpu::survey` names \[8\]
pub fn run(
    bank: &Arc<Bank>,
    dir: &Path,
    adapter: &str,
    backend: &str,
    max_voices: u32,
    say: &mut dyn FnMut(&str),
) -> Outcome {
    let mut steps = Vec::new();
    let mut voices = FIRST_VOICES;
    let cap = max_voices.min(MAX_VOICES);
    let stopped = loop {
        if voices > cap {
            break format!("reached the most voices this adapter can hold, {cap}");
        }
        match step(bank, dir, adapter, backend, voices) {
            Ok(s) => {
                say(&format!(
                    "{:>9} voices: {:7.1} ms a block ({:.1} ms spawning them){}",
                    thousands(s.voices as u64),
                    s.worst_ms,
                    s.spawn_ms,
                    if s.timed_on_gpu { "" } else { ", host-timed" }
                ));
                let slow = s.worst_ms.max(s.spawn_ms);
                steps.push(s);
                if slow > STOP_MS {
                    break format!("a block took {slow:.0} ms; the next step could take twice that");
                }
            }
            Err(e) => break format!("{voices} voices could not be set up: {e:#}"),
        }
        voices = voices.saturating_mul(2);
    };
    let (verdict, ms_at_default, voices_for_1s) = verdict(&steps);
    Outcome {
        adapter: adapter.to_string(),
        backend: backend.to_string(),
        steps,
        stopped,
        verdict,
        ms_at_default,
        voices_for_1s,
    }
}

fn step(bank: &Arc<Bank>, dir: &Path, adapter: &str, backend: &str, voices: u32) -> Result<Step> {
    let midi = dir.join(format!("selftest {voices}.mid"));
    if !midi.exists() {
        crate::testkit::simultaneous_midi(&midi, voices as usize, 10.0)?;
    }
    let cfg = Config {
        max_voices: voices,
        gpu_adapter: Some(adapter.to_string()),
        gpu_backend: Some(backend.to_string()),
        // Timestamps: the GPU's own time for each block, where it keeps them.
        profile: true,
        ..Config::default()
    };
    cfg.validate()?;
    let mut gpu = crate::gpu::GpuSynth::new(&cfg, bank.clone())?;
    let mut driver = Driver::open(&cfg, bank.clone(), &midi)?;
    let mut buf = vec![0.0f32; cfg.block_samples()];
    let mut times = Vec::with_capacity(BLOCKS);
    let mut on_gpu = true;
    let mut live = 0;
    for _ in 0..BLOCKS {
        let t0 = Instant::now();
        driver.next_block(&mut gpu, &mut buf)?;
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        let passes = gpu.timings();
        let device: f64 = passes.iter().map(|(_, ms)| ms).sum();
        if passes.is_empty() {
            on_gpu = false;
        }
        times.push(if passes.is_empty() { wall } else { device });
        live = live.max(gpu.stats().active_voices);
    }
    let spawn_ms = times[0];
    let mut rest = times[1..].to_vec();
    rest.sort_by(f64::total_cmp);
    Ok(Step {
        voices,
        live,
        spawn_ms,
        worst_ms: *rest.last().unwrap_or(&spawn_ms),
        typical_ms: rest.get(rest.len() / 2).copied().unwrap_or(spawn_ms),
        timed_on_gpu: on_gpu,
    })
}

/// The answer, from the biggest step that filled its pool at least halfway: \[9\]
fn verdict(steps: &[Step]) -> (String, Option<f64>, Option<u64>) {
    let Some(s) = steps.iter().rev().find(|s| s.live * 2 >= s.voices as u64 && s.live > 0) else {
        return ("No step filled its pool, so there is nothing to scale from.".into(), None, None);
    };
    let per_voice = s.worst_ms.max(s.spawn_ms) / s.live as f64;
    let default = Config::default().max_voices as f64;
    let at_default = per_voice * default;
    let for_1s = (TARGET_MS / per_voice) as u64;
    let (voices, ms, limit) = (thousands(default as u64), thousands(at_default as u64), thousands(TDR_MS as u64));
    let text = if at_default < TARGET_MS / 2.0 {
        format!(
            "Comfortable. At the default {voices} voices a block takes this GPU about {ms} ms, far \
             under the {limit} ms at which Windows resets it. A block would reach 1 s at about {} \
             voices.",
            thousands(round2(for_1s))
        )
    } else if at_default < TDR_MS * 0.75 {
        format!(
            "Close. At the default {voices} voices a dense block takes this GPU about {ms} ms, and \
             Windows resets it at {limit} ms. Keep --max-voices under about {}, or render with \
             --block 1024, which gives it a quarter of the work at a time.",
            thousands(round2(for_1s))
        )
    } else {
        format!(
            "At risk. At the default {voices} voices a dense block takes this GPU about {ms} ms, {} \
             the {limit} ms at which Windows resets it, which ends the render. Render with \
             --block 1024, and keep --max-voices under about {}.",
            if at_default > TDR_MS { "past" } else { "near" },
            thousands(round2(for_1s * 4))
        )
    };
    let text = format!(
        "{text} (Measured with every voice sounding at once; a real file is usually lighter. In a \
         per-track render, what counts is the voices of all the tracks rendering at once.)"
    );
    (text, Some(at_default), Some(for_1s))
}

/// Down to two significant figures: a voice count to suggest, not to quote.
fn round2(n: u64) -> u64 {
    let mut unit = 1;
    while n / unit >= 100 {
        unit *= 10;
    }
    n / unit * unit
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(voices: u32, ms: f64) -> Step {
        Step { voices, live: voices as u64, spawn_ms: ms, worst_ms: ms, typical_ms: ms, timed_on_gpu: true }
    }

    #[test]
    fn the_verdict_scales_from_the_last_full_step() {
        // About 110 ns a voice, the RTX 5060 Laptop's figure.
        let (text, ms, for_1s) = verdict(&[step(1 << 20, 115.0)]);
        assert!(text.starts_with("Comfortable."), "{text}");
        assert!((ms.unwrap() - 115.0).abs() < 1.0);
        assert!((8_000_000..10_000_000).contains(&for_1s.unwrap()));
        // Twenty times slower: a million voices is 2.3 s.
        let (text, _, _) = verdict(&[step(1 << 16, 115.0 * 20.0 / 16.0)]);
        assert!(text.starts_with("At risk.") && text.contains("--block 1024"), "{text}");
        let (text, _, _) = verdict(&[step(1 << 18, 250.0)]);
        assert!(text.starts_with("Close."), "{text}");
    }

    #[test]
    fn a_step_whose_pool_did_not_fill_is_not_scaled_from() {
        let mut s = step(1 << 20, 50.0);
        s.live = 1000;
        let (text, ms, _) = verdict(&[step(1 << 16, 8.0), s]);
        assert!(ms.is_some() && text.starts_with("Comfortable."), "{text}");
        assert!((ms.unwrap() - 8.0 * 16.0).abs() < 1.0);
    }

    #[test]
    fn a_suggested_count_keeps_two_figures() {
        assert_eq!(round2(225_296), 220_000);
        assert_eq!(round2(10_692_762), 10_000_000);
        assert_eq!(round2(56_324), 56_000);
        assert_eq!(round2(99), 99);
    }

    #[test]
    fn thousands_are_grouped() {
        assert_eq!(thousands(1_048_576), "1,048,576");
        assert_eq!(thousands(16_384), "16,384");
        assert_eq!(thousands(999), "999");
    }
}
