// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Portamento: CC5, CC65 and CC84, the way BASSMIDI does them. \[1\]

use crate::config::Config;

/// Speed of a glide, in semitones a second, for each value of CC5. BASSMIDI's, \[2\]
pub const RATE: [f64; 128] = [
    0.0, 327.0, 306.2, 288.0, 271.6, 256.7, 242.9, 230.2,
    218.7, 207.4, 197.0, 187.3, 178.2, 169.6, 161.5, 153.9,
    146.6, 139.8, 133.4, 127.2, 121.4, 115.9, 110.7, 105.8,
    101.0, 96.56, 92.31, 88.25, 84.39, 80.72, 77.22, 73.89,
    70.7, 67.67, 64.78, 62.01, 59.37, 56.86, 54.45, 52.16,
    49.96, 47.86, 45.86, 43.94, 42.11, 40.35, 38.67, 37.07,
    35.53, 34.06, 32.65, 31.3, 30.01, 28.77, 27.58, 26.45,
    25.36, 24.32, 23.31, 22.35, 21.43, 20.55, 19.7, 18.89,
    18.11, 17.37, 16.65, 15.96, 15.3, 14.67, 14.06, 13.47,
    12.91, 12.37, 11.86, 11.36, 10.88, 10.42, 9.979, 9.556,
    9.148, 8.757, 8.381, 8.02, 7.672, 7.338, 7.016, 6.708,
    6.411, 6.125, 5.85, 5.586, 5.332, 5.087, 4.852, 4.626,
    4.408, 4.199, 3.997, 3.803, 3.616, 3.436, 3.264, 3.097,
    2.937, 2.782, 2.634, 2.482, 2.353, 2.22, 2.093, 1.969,
    1.851, 1.737, 1.627, 1.521, 1.419, 1.321, 1.226, 1.135,
    1.047, 0.962, 0.881, 0.802, 0.726, 0.653, 0.583, 0.515,
];

/// How far ahead of a continuous line BASSMIDI's stepped glide sits on average.
pub const LEAD_SECONDS: f64 = 0.002;

/// Offsets are counted in sixteenths of a cent.
pub const CENT_STEPS: u32 = 16;
/// Steps in an octave, and entries in the exponent table.
pub const OCTAVE: u32 = 1200 * CENT_STEPS;
/// Steps in a semitone.
const SEMITONE: u64 = 100 * CENT_STEPS as u64;
/// Fractional bits of a per-frame speed.
const RATE_FRAC_BITS: u32 = 24;
/// Entries of the speed table ahead of the exponent table in `tables`.
pub const RATES: usize = 128;

/// Widest glide from above: just under eight octaves is what an 8.24 factor \[3\]
pub const MAX_UP: i32 = 95;

/// Bits of a voice's flags word that are flags. The rest is the glide.
pub const FLAG_BITS: u32 = 0x7;
/// Where the remaining glide starts in the flags word: frames until the glide \[4\]
pub const REM_SHIFT: u32 = 3;
pub const REM_MAX: u32 = u32::MAX >> REM_SHIFT;
/// The part of a gate slot word that is the slot.
pub const SLOT_MASK: u32 = 0xFFFF;
/// Where CC5 sits in a gate slot word, seven bits of it.
pub const RATE_SHIFT: u32 = 24;
/// Set in a gate slot word when the note glides down onto its own pitch, from \[5\]
pub const UP_BIT: u32 = 1 << 31;

/// The exponent is evaluated as `2^(m / OCTAVE - MID_OCTAVES)`, with `m` kept \[6\]
const MID_OCTAVES: u32 = 11;
const MID: u32 = MID_OCTAVES * OCTAVE;
/// Largest offsets either way, as guards: `spawn` already keeps a glide inside \[7\]
const UP_MAX_OFF: u32 = 8 * OCTAVE - 1;
const DOWN_MAX_OFF: u32 = MID;

/// The table both backends read: `RATES` per-frame speeds for this sample \[8\]
pub fn tables(sample_rate: u32) -> Vec<u32> {
    let mut t = Vec::with_capacity(RATES + OCTAVE as usize);
    for r in RATE {
        let q = r * SEMITONE as f64 * (1u64 << RATE_FRAC_BITS) as f64 / sample_rate as f64;
        t.push(q.round() as u32);
    }
    for i in 0..OCTAVE {
        let e = (i as f64 / OCTAVE as f64).exp2() * (1u64 << 30) as f64;
        t.push(e.round() as u32);
    }
    t
}

/// What the driver needs to start glides: the tables and the lead in frames.
pub struct Glide {
    pub tables: Vec<u32>,
    /// `LEAD_SECONDS` plus half a gate tile, because the factor is evaluated \[9\]
    pub lead: u32,
}

impl Glide {
    pub fn new(cfg: &Config) -> Self {
        Glide {
            tables: tables(cfg.sample_rate),
            lead: (LEAD_SECONDS * cfg.sample_rate as f64).round() as u32 + cfg.gate_frames / 2,
        }
    }

    /// Frames a glide of `semitones` at CC5 = `cc5` lasts, or zero for none.
    pub fn frames(&self, semitones: i32, cc5: u8) -> u64 {
        let rq = self.tables[cc5 as usize & 0x7F] as u64;
        if rq == 0 || semitones == 0 {
            return 0;
        }
        let d = semitones.min(MAX_UP).unsigned_abs() as u64;
        ((d * SEMITONE) << RATE_FRAC_BITS) / rq
    }

    /// The glide a voice starts with, as the bits to OR into its flags and its \[10\]
    pub fn spawn(&self, semitones: i32, cc5: u8, start_rel: u32) -> (u32, u32) {
        let t = self.frames(semitones, cc5);
        if t <= self.lead as u64 {
            return (0, 0);
        }
        let rem = (start_rel as u64 + t - self.lead as u64).min(REM_MAX as u64) as u32;
        let up = if semitones > 0 { UP_BIT } else { 0 };
        (rem << REM_SHIFT, ((cc5 as u32 & 0x7F) << RATE_SHIFT) | up)
    }
}

/// The 8.24 pitch factor a gliding voice holds for the gate tile evaluated at \[11\]
#[inline]
pub fn factor(flags: u32, slot: u32, f: u32, tab: &[u32]) -> u32 {
    let rem = (flags >> REM_SHIFT).saturating_sub(f);
    let rq = tab[((slot >> RATE_SHIFT) & 0x7F) as usize];
    let raw = ((rq as u64 * rem as u64) >> RATE_FRAC_BITS) as u32;
    let up = slot & UP_BIT != 0;
    let off = raw.min(if up { UP_MAX_OFF } else { DOWN_MAX_OFF });
    let m = if up { MID + off } else { MID - off };
    let oct = m / OCTAVE;
    let e = tab[RATES + (m - oct * OCTAVE) as usize];
    if oct > MID_OCTAVES + 6 {
        e << (oct - MID_OCTAVES - 6)
    } else {
        e >> (MID_OCTAVES + 6 - oct)
    }
}

/// The flags word after a block: the glide a block shorter, or gone once it \[12\]
#[inline]
pub fn advance(flags: u32, block_frames: u32) -> u32 {
    if flags >> REM_SHIFT > block_frames {
        flags - (block_frames << REM_SHIFT)
    } else {
        flags & FLAG_BITS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glide(sr: u32) -> Glide {
        Glide::new(&Config { sample_rate: sr, ..Config::default() })
    }

    fn cents(f: u32) -> f64 {
        (f as f64 / (1u64 << 24) as f64).log2() * 1200.0
    }

    #[test]
    fn a_landed_glide_is_exactly_unity() {
        let g = glide(48_000);
        for (d, cc5) in [(12, 15u8), (-12, 15), (95, 1), (-127, 127)] {
            let (fl, sl) = g.spawn(d, cc5, 100);
            assert!(fl > FLAG_BITS, "{d} at {cc5} should glide");
            let end = fl >> REM_SHIFT;
            assert_eq!(factor(fl, sl, end, &g.tables), 1 << 24);
            assert_eq!(factor(fl, sl, end + 5000, &g.tables), 1 << 24);
        }
    }

    #[test]
    fn a_glide_starts_where_it_should_and_moves_at_bassmidis_speed() {
        let sr = 48_000;
        let g = glide(sr);
        for (d, cc5) in [(13, 15u8), (-12, 64), (24, 64), (-5, 100), (4, 127)] {
            let start = 1000;
            let (fl, sl) = g.spawn(d, cc5, start);
            // [13]
            let rate = RATE[cc5 as usize];
            let want = d as f64 * 100.0 - d.signum() as f64 * rate * 100.0 * g.lead as f64 / sr as f64;
            let got = cents(factor(fl, sl, start, &g.tables));
            // [14]
            let frame = rate * 100.0 / sr as f64;
            assert!(
                (got - want).abs() < frame + 0.1,
                "{d} at CC5 {cc5}: starts {got:.2} cents, want {want:.2}"
            );
            // And it moves linearly at the measured speed.
            let later = start + sr / 50;
            let moved = got - cents(factor(fl, sl, later, &g.tables));
            let want = d.signum() as f64 * rate * 100.0 / 50.0;
            if (later as u64) < (fl >> REM_SHIFT) as u64 {
                assert!((moved - want).abs() < 0.2, "{d} at CC5 {cc5}: moved {moved:.2} cents in 20 ms, want {want:.2}");
            }
        }
    }

    #[test]
    fn a_glide_counts_down_a_block_at_a_time() {
        let g = glide(48_000);
        let (fl, sl) = g.spawn(-12, 64, 300);
        let one = advance(fl | 1, 4096);
        assert_eq!(one & FLAG_BITS, 1, "the loop flag must survive");
        // [15]
        for f in [0, 32, 1000, 4000] {
            assert_eq!(factor(one, sl, f, &g.tables), factor(fl, sl, f + 4096, &g.tables));
        }
        let mut w = fl;
        for _ in 0..1000 {
            w = advance(w, 4096);
        }
        assert_eq!(w, 0, "a glide must end");
    }

    #[test]
    fn no_speed_or_no_interval_is_no_glide() {
        let g = glide(48_000);
        assert_eq!(g.spawn(12, 0, 0), (0, 0));
        assert_eq!(g.spawn(0, 64, 0), (0, 0));
    }

    #[test]
    fn the_widest_glides_fit_at_every_sample_rate() {
        for sr in [8_000, 44_100, 48_000, 96_000, 768_000] {
            let g = glide(sr);
            for (d, cc5) in [(127, 1u8), (-127, 1), (127, 127), (-127, 127)] {
                let (fl, sl) = g.spawn(d, cc5, 0);
                assert!(fl >> REM_SHIFT < REM_MAX, "{sr} Hz, {d} at {cc5} was cut short");
                let f = factor(fl, sl, 0, &g.tables);
                let d = d.min(MAX_UP);
                let lead = RATE[cc5 as usize] * 100.0 * g.lead as f64 / sr as f64;
                let want = d as f64 * 100.0 - d.signum() as f64 * lead;
                let frame = RATE[cc5 as usize] * 100.0 / sr as f64;
                // [16]
                let lsb = (1.0 + 1.0 / f as f64).log2() * 1200.0;
                assert!(
                    (cents(f) - want).abs() < frame + lsb + 0.1,
                    "{sr} Hz, {d} at {cc5}: {:.1} cents, want {want:.1}",
                    cents(f)
                );
            }
        }
    }
}
