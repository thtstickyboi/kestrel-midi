// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Load-time analytic quadrature and immutable note coefficients. \[1\]

use crate::bank::{sample_geometry, Bank};
use anyhow::{ensure, Result};
use std::{collections::HashMap, f64::consts::PI, sync::Arc};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PhaseMode {
    #[default]
    Baseline,
    Analytic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhaseSettings {
    pub mode: PhaseMode,
    pub strength: f32,
    pub seed: u64,
    pub pool_size: u32,
    pub continuous: bool,
    pub preserve_attack_ms: f32,
    pub cache_budget_bytes: u64,
    pub scratch_budget_bytes: u64,
}

impl Default for PhaseSettings {
    fn default() -> Self {
        Self {
            mode: PhaseMode::Baseline,
            strength: 1.0,
            seed: 0,
            pool_size: 64,
            continuous: false,
            preserve_attack_ms: 0.0,
            cache_budget_bytes: 2 << 30,
            scratch_budget_bytes: 512 << 20,
        }
    }
}

impl PhaseSettings {
    pub fn active(&self) -> bool {
        self.mode == PhaseMode::Analytic && self.strength > 0.0
    }

    pub fn validate(&self) -> Result<()> {
        if self.mode == PhaseMode::Baseline {
            return Ok(());
        }
        ensure!(
            self.strength.is_finite() && (0.0..=1.0).contains(&self.strength),
            "phase strength must be finite and in 0..=1"
        );
        if !self.active() {
            return Ok(());
        }
        ensure!(
            self.continuous || (1..=64).contains(&self.pool_size),
            "phase pool must be in 1..=64"
        );
        ensure!(
            self.preserve_attack_ms.is_finite() && self.preserve_attack_ms >= 0.0,
            "phase preserve-attack-ms must be finite and nonnegative"
        );
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Coefficients {
    pub cosine: f32,
    pub sine: f32,
    pub scale: f32,
}

impl Default for Coefficients {
    fn default() -> Self {
        Self {
            cosine: 1.0,
            sine: 0.0,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Angle {
    cosine: f32,
    sine: f32,
    slot: u32,
}

#[derive(Debug)]
pub struct PreparationCancelled;
impl std::fmt::Display for PreparationCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("analytic phase preparation cancelled")
    }
}
impl std::error::Error for PreparationCancelled {}

fn check_cancel(cancel: &dyn Fn() -> bool) -> Result<()> {
    if cancel() {
        return Err(PreparationCancelled.into());
    }
    Ok(())
}

// [2]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SampleKey {
    base: u32,
    len: u32,
    ls: u32,
    le: u32,
    flags: u32,
    rate: u32,
}

struct Entry {
    energy: [f64; 3],
    scales: Vec<f32>,
}

/// One immutable preparation, shared by the MIDI driver and its backend. \[3\]
pub struct PhaseBank {
    pub(crate) words: Vec<u32>,
    pub(crate) settings: PhaseSettings,
    region_entries: Vec<Option<usize>>,
    entries: Vec<Entry>,
    angles: Vec<Angle>,
}

impl PhaseBank {
    pub fn prepare(bank: &Bank, settings: &PhaseSettings) -> Result<Arc<Self>> {
        Self::prepare_with(bank, settings, &|| false, &mut |_, _| {})
    }

    pub fn prepare_with(
        bank: &Bank,
        settings: &PhaseSettings,
        cancel: &dyn Fn() -> bool,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Arc<Self>> {
        settings.validate()?;
        check_cancel(cancel)?;
        let mut out = Self {
            words: Vec::new(),
            settings: settings.clone(),
            region_entries: Vec::new(),
            entries: Vec::new(),
            angles: Vec::new(),
        };
        if !settings.active() {
            return Ok(Arc::new(out));
        }

        let mut keys = Vec::new();
        let mut seen = HashMap::new();
        let mut total_words = (bank.regions.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("phase metadata size overflow"))?;
        let mut scratch = 0u64;
        for r in &bank.regions {
            check_cancel(cancel)?;
            let Some(s) = bank.samples.get(r.sample as usize).filter(|s| s.len != 0) else {
                out.region_entries.push(None);
                continue;
            };
            let (_, len, ls, le, flags) = sample_geometry(s, r);
            ensure!(
                s.start as u64 + len as u64 <= bank.pool.len() as u64,
                "analytic sample extends beyond the sample pool"
            );
            let key = SampleKey {
                base: s.start,
                len,
                ls,
                le,
                flags,
                rate: s.rate,
            };
            let id = match seen.get(&key) {
                Some(&id) => id,
                None => {
                    let id = keys.len();
                    keys.push(key);
                    seen.insert(key, id);
                    total_words += len as u64;
                    // [4]
                    let fft_len = fft_size(len as usize * 2)? as u64;
                    scratch = scratch.max(fft_len * 64 + len as u64 * 32);
                    id
                }
            };
            out.region_entries.push(Some(id));
        }
        ensure!(
            total_words <= u32::MAX as u64,
            "analytic cache exceeds 32-bit GPU addressing"
        );
        let pcm_bytes = total_words * 4;
        ensure!(
            pcm_bytes <= settings.cache_budget_bytes,
            "analytic cache needs {:.1} MiB; phase-cache-mib allows {:.1} MiB",
            pcm_bytes as f64 / 1048576.0,
            settings.cache_budget_bytes as f64 / 1048576.0
        );
        ensure!(
            scratch <= settings.scratch_budget_bytes,
            "analytic FFT scratch needs up to {:.1} MiB; phase-scratch-mib allows {:.1} MiB",
            scratch as f64 / 1048576.0,
            settings.scratch_budget_bytes as f64 / 1048576.0
        );

        if !settings.continuous {
            for slot in 0..settings.pool_size {
                let h = combine(combine(settings.seed, slot as u64), 0x414e474c45);
                out.angles.push(angle(h, settings.strength, slot));
            }
        }
        out.words.try_reserve_exact(total_words as usize)?;
        out.words.resize(bank.regions.len() * 4, 0);
        let mut bases = Vec::with_capacity(keys.len());
        progress(0, keys.len());
        for (id, key) in keys.iter().enumerate() {
            check_cancel(cancel)?;
            let original: Vec<f64> = bank.pool[key.base as usize..][..key.len as usize]
                .iter()
                .map(|&v| decode(v) as f64)
                .collect();
            let mut q = quadrature(&original, true, cancel)?;
            if key.flags != 0 {
                let periodic =
                    quadrature(&original[key.ls as usize..key.le as usize], false, cancel)?;
                install_loop(&mut q, &periodic, key.ls as usize, key.rate);
            }
            let mut energy = [0.0; 3];
            for (i, (&x, &v)) in original.iter().zip(&q).enumerate() {
                if i & 16383 == 0 {
                    check_cancel(cancel)?;
                }
                energy[0] += x * x;
                energy[1] += v as f64 * v as f64;
                energy[2] += x * v as f64;
            }
            bases.push(out.words.len() as u32);
            out.words.extend(q.iter().map(|v| v.to_bits()));
            let scales = out
                .angles
                .iter()
                .map(|a| normalization(energy, *a))
                .collect();
            out.entries.push(Entry { energy, scales });
            progress(id + 1, keys.len());
        }
        for (region, entry) in out.region_entries.iter().enumerate() {
            if let Some(id) = *entry {
                let k = keys[id];
                let hold = ((settings.preserve_attack_ms as f64 * 0.001 * k.rate as f64)
                    .round()
                    .min(k.len as f64)) as u32;
                let fade = if settings.preserve_attack_ms > 0.0 {
                    (k.len - hold).min((k.rate as f64 * 0.010).round() as u32)
                } else {
                    0
                };
                out.words[region * 4..region * 4 + 4].copy_from_slice(&[bases[id], hold, fade, 0]);
            }
        }
        check_cancel(cancel)?;
        Ok(Arc::new(out))
    }

    pub fn cache_bytes(&self) -> u64 {
        self.words.len() as u64 * 4
    }
    pub fn sample_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn angle(&self, tick: u64, channel: u8, key: u8) -> Angle {
        let h = combine(
            combine(
                combine(combine(self.settings.seed, tick), channel as u64),
                key as u64,
            ),
            0,
        );
        if self.settings.continuous {
            angle(h, self.settings.strength, u32::MAX)
        } else {
            self.angles[(h % self.angles.len() as u64) as usize]
        }
    }

    pub(crate) fn coefficients(&self, region: u32, angle: Angle) -> Coefficients {
        let Some(id) = self.region_entries[region as usize] else {
            return Coefficients::default();
        };
        let entry = &self.entries[id];
        let scale = if self.settings.continuous {
            normalization(entry.energy, angle)
        } else {
            entry.scales[angle.slot as usize]
        };
        Coefficients {
            cosine: angle.cosine,
            sine: angle.sine,
            scale,
        }
    }

    #[inline]
    pub(crate) fn apply(&self, original: f32, region: u32, index: u32, c: Coefficients) -> f32 {
        let m = &self.words[region as usize * 4..];
        if index < m[1] {
            return original;
        }
        let q = f32::from_bits(self.words[(m[0] + index) as usize]);
        let changed = (c.cosine * original - c.sine * q) * c.scale;
        if m[2] == 0 || index - m[1] >= m[2] {
            return changed;
        }
        let u = (index - m[1]) as f32 / m[2] as f32;
        original + (changed - original) * (u * u * (3.0 - 2.0 * u))
    }
}

fn normalization(e: [f64; 3], a: Angle) -> f32 {
    let (c, s) = (a.cosine as f64, a.sine as f64);
    let changed = c * c * e[0] + s * s * e[1] - 2.0 * c * s * e[2];
    if changed > 1e-30 {
        (e[0] / changed).sqrt() as f32
    } else {
        1.0
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn combine(seed: u64, x: u64) -> u64 {
    splitmix(seed ^ splitmix(x.wrapping_add(0x9e3779b97f4a7c15)))
}
fn angle(hash: u64, strength: f32, slot: u32) -> Angle {
    let unit = (splitmix(hash) >> 11) as f64 / 9007199254740992.0;
    let theta = (unit * 2.0 - 1.0) * PI * strength as f64;
    Angle {
        cosine: theta.cos() as f32,
        sine: theta.sin() as f32,
        slot,
    }
}

fn decode(v: i16) -> f32 {
    (v as f32 * (1.0 / 32767.0)).max(-1.0)
}

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}
impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
}
impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
}
impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
}
fn cis(t: f64) -> Complex {
    Complex {
        re: t.cos(),
        im: t.sin(),
    }
}
fn fft_size(n: usize) -> Result<usize> {
    n.checked_next_power_of_two()
        .ok_or_else(|| anyhow::anyhow!("analytic FFT size overflow"))
}

fn radix2(v: &mut [Complex], inverse: bool, cancel: &dyn Fn() -> bool) -> Result<()> {
    check_cancel(cancel)?;
    let n = v.len();
    let mut j = 0;
    for i in 1..n {
        if i & 16383 == 0 {
            check_cancel(cancel)?;
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            v.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = cis(if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        });
        for start in (0..n).step_by(len) {
            let mut factor = Complex { re: 1.0, im: 0.0 };
            for off in 0..len / 2 {
                if (start + off) & 16383 == 0 {
                    check_cancel(cancel)?;
                }
                let even = v[start + off];
                let odd = v[start + off + len / 2] * factor;
                v[start + off] = even + odd;
                v[start + off + len / 2] = even - odd;
                factor = factor * step;
            }
        }
        if len == n {
            break;
        }
        len *= 2;
    }
    if inverse {
        for x in v {
            x.re /= n as f64;
            x.im /= n as f64;
        }
    }
    Ok(())
}

// Bluestein preserves the exact loop length, including odd/prime lengths.
fn forward(mut input: Vec<Complex>, cancel: &dyn Fn() -> bool) -> Result<Vec<Complex>> {
    let n = input.len();
    if n.is_power_of_two() {
        radix2(&mut input, false, cancel)?;
        return Ok(input);
    }
    let m = fft_size(n * 2 - 1)?;
    let mut a = vec![Complex::default(); m];
    let mut b = vec![Complex::default(); m];
    for i in 0..n {
        if i & 16383 == 0 {
            check_cancel(cancel)?;
        }
        let theta = PI * ((i as f64 * i as f64) % (2.0 * n as f64)) / n as f64;
        a[i] = input[i] * cis(-theta);
        b[i] = cis(theta);
        if i != 0 {
            b[m - i] = b[i];
        }
    }
    radix2(&mut a, false, cancel)?;
    radix2(&mut b, false, cancel)?;
    for i in 0..m {
        a[i] = a[i] * b[i];
    }
    radix2(&mut a, true, cancel)?;
    a.truncate(n);
    for (i, x) in a.iter_mut().enumerate() {
        if i & 16383 == 0 {
            check_cancel(cancel)?;
        }
        let theta = PI * ((i as f64 * i as f64) % (2.0 * n as f64)) / n as f64;
        *x = *x * cis(-theta);
    }
    Ok(a)
}

fn quadrature(input: &[f64], padded: bool, cancel: &dyn Fn() -> bool) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let n = if padded {
        fft_size(input.len() * 2)?
    } else {
        input.len()
    };
    let mut v = vec![Complex::default(); n];
    for (x, &re) in v.iter_mut().zip(input) {
        x.re = re;
    }
    v = forward(v, cancel)?;
    for (i, x) in v.iter_mut().enumerate() {
        // [5]
        *x = if i == 0 || (n % 2 == 0 && i == n / 2) {
            Complex::default()
        } else if i < n.div_ceil(2) {
            Complex {
                re: x.im,
                im: -x.re,
            }
        } else {
            Complex {
                re: -x.im,
                im: x.re,
            }
        };
        x.im = -x.im;
    }
    v = forward(v, cancel)?;
    Ok(v[..input.len()]
        .iter()
        .map(|x| (x.re / n as f64) as f32)
        .collect())
}

fn install_loop(q: &mut [f32], periodic: &[f32], start: usize, rate: u32) {
    let fade = start
        .min(periodic.len() / 4)
        .min((rate as f64 * 0.010).round() as usize);
    for off in 0..fade {
        let u = (off + 1) as f64 / (fade + 1) as f64;
        let blend = u * u * (3.0 - 2.0 * u);
        q[start - fade + off] = (q[start - fade + off] as f64 * (1.0 - blend)
            + periodic[periodic.len() - fade + off] as f64 * blend)
            as f32;
    }
    q[start..start + periodic.len()].copy_from_slice(periodic);
}

#[cfg(test)]
mod tests;
