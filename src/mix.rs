// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Per-track rendering, merged: every track's blocks summed into one mix as \[1\]

use crate::config::Config;
use crate::limiter::OutputStage;
use crate::session::Sink;
use anyhow::{anyhow, bail, Result};
use std::io::{Read, Write};
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

    /// Blocks the mix runs for, silent ones included.
    pub(crate) fn len(&self) -> u64 {
        self.len.load(Ordering::Relaxed)
    }

    /// Write every block that holds sound -- its number, then its samples, each \[9\]
    pub(crate) fn save(&self, w: &mut impl Write) -> Result<()> {
        let blocks = self.blocks.read().unwrap();
        let held = blocks.iter().filter(|b| !b.lock().unwrap().is_empty()).count();
        w.write_all(&(held as u64).to_le_bytes())?;
        let mut buf = Vec::with_capacity(8 + self.block_samples * 16);
        for (i, block) in blocks.iter().enumerate() {
            let acc = block.lock().unwrap();
            if acc.is_empty() {
                continue;
            }
            buf.clear();
            buf.extend_from_slice(&(i as u64).to_le_bytes());
            for x in acc.iter() {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            w.write_all(&buf)?;
        }
        Ok(())
    }

    /// The mix `save` wrote, running for `len` blocks. Refused if a block is \[10\]
    pub(crate) fn load(block_samples: usize, len: u64, r: &mut impl Read) -> Result<Mix> {
        let mix = Mix::new(block_samples);
        let mut word = [0u8; 8];
        r.read_exact(&mut word)?;
        let held = u64::from_le_bytes(word);
        let mut raw = vec![0u8; block_samples * 16];
        {
            let mut blocks = mix.blocks.write().unwrap();
            for _ in 0..held {
                r.read_exact(&mut word)?;
                let at = u64::from_le_bytes(word);
                if at >= len {
                    bail!("the saved mix has a block at {at}, past its length of {len}");
                }
                r.read_exact(&mut raw)?;
                let want = (at as usize + 1).next_multiple_of(GROW);
                while blocks.len() < want {
                    blocks.push(Mutex::new(Vec::new()));
                }
                let acc = blocks[at as usize].get_mut().unwrap();
                acc.clear();
                acc.extend(raw.chunks_exact(16).map(|c| i128::from_le_bytes(c.try_into().expect("sixteen bytes"))));
            }
        }
        mix.len.store(len, Ordering::Relaxed);
        Ok(mix)
    }

    /// Write the mix to `path` through the output stage `cfg` describes: \[11\]
    pub(crate) fn write(
        &self,
        cfg: &Config,
        path: &Path,
        encoder: Option<&(crate::ffmpeg::Ffmpeg, &'static crate::ffmpeg::Preset)>,
        wav_format: crate::wav::SampleFormat,
    ) -> Result<Written> {
        let len = self.len.load(Ordering::Relaxed);
        let blocks = self.blocks.read().unwrap();
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

    /// Every f32 in the range the mix promises comes back as itself: all of \[12\]
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

    fn contents(mix: &Mix) -> (u64, Vec<Vec<i128>>) {
        let blocks = mix.blocks.read().unwrap();
        (mix.len(), blocks.iter().map(|b| b.lock().unwrap().clone()).filter(|b| !b.is_empty()).collect())
    }

    /// A mix saved and loaded is the same mix, and one that has had more added \[13\]
    #[test]
    fn a_saved_mix_loads_as_itself_and_goes_on_adding_exactly() {
        let n = 16;
        let tone = |k: f32| (0..n).map(|i| (i as f32 + k) * 0.0137 - 0.1).collect::<Vec<f32>>();
        let adds: Vec<(u64, Vec<f32>)> = vec![
            (0, tone(1.0)),
            (2, tone(2.0)),
            (2, tone(3.0)),
            (300, tone(4.0)),
            (301, vec![0.0; n]),
            (7, tone(5.0)),
            (300, tone(6.0)),
        ];
        let all = Mix::new(n);
        for (b, s) in &adds {
            all.add(*b, s).unwrap();
        }
        // Stop after four adds, save, load, and add the rest to the loaded one.
        let early = Mix::new(n);
        for (b, s) in &adds[..4] {
            early.add(*b, s).unwrap();
        }
        let mut bytes = Vec::new();
        early.save(&mut bytes).unwrap();
        let back = Mix::load(n, early.len(), &mut bytes.as_slice()).unwrap();
        assert_eq!(contents(&back), contents(&early), "what was saved is what came back");
        for (b, s) in &adds[4..] {
            back.add(*b, s).unwrap();
        }
        assert_eq!(contents(&back), contents(&all));
        // The silent block at the end still counts towards the length.
        assert_eq!(back.len(), 302);
        // Only blocks that hold sound were written: 8 bytes for the count, and 3 of them.
        assert_eq!(bytes.len(), 8 + 3 * (8 + n * 16));
    }

    #[test]
    fn a_saved_mix_that_does_not_fit_is_refused() {
        let n = 4;
        let m = Mix::new(n);
        m.add(5, &[0.5; 4]).unwrap();
        let mut bytes = Vec::new();
        m.save(&mut bytes).unwrap();
        // A length shorter than a block it holds, and data that stops early.
        assert!(Mix::load(n, 5, &mut bytes.as_slice()).is_err());
        assert!(Mix::load(n, 6, &mut &bytes[..bytes.len() - 1]).is_err());
        assert!(Mix::load(n, 6, &mut bytes.as_slice()).is_ok());
    }

    /// The same blocks added in any order make the same mix, which float \[14\]
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
