// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Per-track rendering, merged: every track's blocks summed into one mix as \[1\]

use crate::config::Config;
use crate::limiter::OutputStage;
use crate::session::Sink;
use anyhow::{anyhow, bail, Result};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

/// Fractional bits of the mix's fixed point.
const FRAC: i32 = 96;

/// Blocks added to the table at a time when a track reaches past its end, so \[2\]
const GROW: usize = 256;

/// `v` in units of 2^-96. Exact for every magnitude from 2^-73 to 2^30, \[3\]
pub(crate) fn to_fixed(v: f32) -> Result<i128> {
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    if exp == 0xFF {
        bail!("a track produced {v}, which the mix cannot hold");
    }
    if exp == 0 {
        // Zero, or subnormal: under 2^-126, a fraction of one unit.
        return Ok(0);
    }
    let m = ((bits & 0x7F_FFFF) | 0x80_0000) as i128;
    // v = m * 2^(exp - 150), which in units of 2^-96 is m * 2^(exp - 54).
    let shift = exp - 150 + FRAC;
    let mag = if shift > 102 {
        bail!("a track produced {v}, past the 2^30 the mix holds");
    } else if shift >= 0 {
        m << shift
    } else if shift > -24 {
        m >> -shift
    } else {
        0
    };
    Ok(if bits >> 31 != 0 { -mag } else { mag })
}

/// Back to f32, rounded once. `i128 as f32` rounds to nearest, and the scale \[4\]
pub(crate) fn from_fixed(x: i128) -> f32 {
    x as f32 * (2.0f32).powi(-FRAC)
}

/// The merged render's mix. Shared by every thread rendering a track.
pub(crate) struct Mix {
    block_samples: usize,
    /// One entry per block; empty until some track adds sound to it, and \[5\]
    blocks: RwLock<Vec<Mutex<Vec<i128>>>>,
    /// Blocks the longest track ran for, silent ones included: how long the \[6\]
    len: AtomicU64,
}

/// What writing the mix out came to.
pub(crate) struct Written {
    pub bytes: u64,
    pub frames: u64,
    /// Largest magnitude in the mix before the output stage, as a render's \[7\]
    pub peak: f32,
    pub clipped: u64,
    /// Bytes the mix held at its largest.
    pub held: u64,
}

impl Mix {
    pub(crate) fn new(block_samples: usize) -> Self {
        Mix { block_samples, blocks: RwLock::new(Vec::new()), len: AtomicU64::new(0) }
    }

    /// Add a track's block number `block`. A block of zeros only counts \[8\]
    pub(crate) fn add(&self, block: u64, samples: &[f32]) -> Result<()> {
        if samples.len() != self.block_samples {
            bail!("a block of {} samples, expected {}", samples.len(), self.block_samples);
        }
        self.len.fetch_max(block + 1, Ordering::Relaxed);
        if samples.iter().all(|&v| v == 0.0) {
            return Ok(());
        }
        let b = block as usize;
        if self.blocks.read().unwrap().len() <= b {
            let mut blocks = self.blocks.write().unwrap();
            let want = (b + 1).next_multiple_of(GROW);
            while blocks.len() < want {
                blocks.push(Mutex::new(Vec::new()));
            }
        }
        let blocks = self.blocks.read().unwrap();
        let mut acc = blocks[b].lock().unwrap();
        if acc.is_empty() {
            acc.resize(self.block_samples, 0);
        }
        for (a, &v) in acc.iter_mut().zip(samples) {
            *a = a.checked_add(to_fixed(v)?).ok_or_else(|| anyhow!("the merged mix overflowed at block {block}"))?;
        }
        Ok(())
    }

    /// Write the mix to `path` through the output stage `cfg` describes: \[9\]
    pub(crate) fn write(
        self,
        cfg: &Config,
        path: &Path,
        encoder: Option<&(crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
        wav_format: crate::wav::SampleFormat,
    ) -> Result<Written> {
        let len = self.len.load(Ordering::Relaxed);
        let blocks = self.blocks.into_inner().unwrap();
        let held = blocks.iter().map(|b| b.lock().unwrap().len() as u64 * 16).sum();
        let mut out = Sink::create(path, cfg, encoder, wav_format)?;
        let mut stage = OutputStage::new(cfg);
        let mut block = vec![0.0f32; self.block_samples];
        let (mut peak, mut clipped) = (0.0f32, 0u64);
        for b in 0..len {
            let acc = blocks.get(b as usize).map(|m| m.lock().unwrap());
            match acc.as_deref() {
                Some(acc) if !acc.is_empty() => {
                    for (o, &x) in block.iter_mut().zip(acc.iter()) {
                        *o = from_fixed(x);
                    }
                }
                _ => block.fill(0.0),
            }
            peak = block.iter().fold(peak, |m, v| m.max(v.abs()));
            clipped += stage.process(&mut block, b)?;
            out.write_block(&block)?;
        }
        let bytes = out.finish()?;
        Ok(Written { bytes, frames: len * (self.block_samples / 2) as u64, peak, clipped, held })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every f32 in the range the mix promises comes back as itself: all of \[10\]
    #[test]
    fn a_sample_goes_through_the_mix_unchanged() {
        for exp in (127 - 73)..=(127 + 29) {
            for frac in [0u32, 1, 0x2A_AAAA, 0x55_5555, 0x7F_FFFE, 0x7F_FFFF] {
                for sign in [0u32, 1] {
                    let v = f32::from_bits(sign << 31 | (exp as u32) << 23 | frac);
                    let x = to_fixed(v).unwrap();
                    assert_eq!(from_fixed(x).to_bits(), v.to_bits(), "{v:e}");
                    // And agrees with the slow conversion it replaces.
                    assert_eq!(x, (v as f64 * 2f64.powi(FRAC)) as i128, "{v:e}");
                }
            }
        }
        assert_eq!(to_fixed(0.0).unwrap(), 0);
        assert_eq!(to_fixed(-0.0).unwrap(), 0);
        assert_eq!(to_fixed(f32::MIN_POSITIVE / 2.0).unwrap(), 0);
        assert!(to_fixed(f32::NAN).is_err());
        assert!(to_fixed(f32::INFINITY).is_err());
        assert!(to_fixed(2.0f32.powi(30)).is_err());
        assert!(to_fixed(2.0f32.powi(30) - 64.0).is_ok());
    }

    /// The same blocks added in any order make the same mix, which float \[11\]
    #[test]
    fn the_order_blocks_arrive_in_does_not_matter() {
        let n = 8;
        let a = vec![1.0e8f32; n];
        let b = vec![-1.0e8f32; n];
        let c = vec![1.0f32; n];
        assert_ne!((a[0] + b[0]) + c[0], a[0] + (b[0] + c[0]), "the fixture sums the same either way");
        let sum = |order: &[&Vec<f32>]| {
            let mix = Mix::new(n);
            for s in order {
                mix.add(3, s).unwrap();
            }
            let blocks = mix.blocks.into_inner().unwrap();
            let out: Vec<f32> = blocks[3].lock().unwrap().iter().map(|&x| from_fixed(x)).collect();
            (out, mix.len.load(Ordering::Relaxed))
        };
        let (x, len) = sum(&[&a, &b, &c]);
        assert_eq!(len, 4);
        assert_eq!(x, vec![1.0; n]);
        assert_eq!(sum(&[&c, &a, &b]).0, x);
        assert_eq!(sum(&[&b, &c, &a]).0, x);
    }
}
