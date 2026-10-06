// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A tuning that gives every MIDI key a pitch of its own: `--edo N`, and Scala's \[1\]

use crate::mts::UNITS_PER_CENT;
use std::fmt;

/// Offsets are in 1/2048 of a cent, and a semitone is a hundred cents.
pub const UNITS_PER_SEMITONE: i64 = 100 * UNITS_PER_CENT as i64;

/// The most a key's pitch can be moved from where 12-tone puts it, in semitones: far \[2\]
const OFFSET_MAX_SEMITONES: f64 = 10_000.0;

/// The pitch, in 12-tone semitones above key 0, that a key is tuned to: A4 is key 69 \[3\]
pub fn semitones_of_hz(hz: f64) -> f64 {
    69.0 + 12.0 * (hz / 440.0).log2()
}

/// A key and the pitch it is tuned to, which a tuning is laid out around. The default \[4\]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reference {
    pub key: u8,
    /// The pitch in 12-tone semitones, `semitones_of_hz` of its frequency.
    pub semitones: f64,
}

impl Default for Reference {
    fn default() -> Reference {
        Reference { key: 60, semitones: 60.0 }
    }
}

impl Reference {
    /// `KEY:HZ`, as `--tuning-ref` takes it: `60:261.6256`, or a key and a frequency \[5\]
    pub fn parse(s: &str) -> Result<Reference, String> {
        let bad = || format!("{s:?} is not a reference; give a key and a frequency, KEY:HZ, such as 69:440");
        let (k, hz) = s.split_once(':').ok_or_else(bad)?;
        let key: u8 = k.trim().parse().map_err(|_| bad())?;
        if key > 127 {
            return Err(format!("the reference key is 0 to 127, not {key}"));
        }
        let hz: f64 = hz.trim().parse().map_err(|_| bad())?;
        if !(hz.is_finite() && (1.0..=40_000.0).contains(&hz)) {
            return Err(format!("the reference frequency is 1 to 40000 Hz, not {hz}"));
        }
        Ok(Reference { key, semitones: semitones_of_hz(hz) })
    }
}

/// Where each of the 128 keys plays.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuning {
    /// Each key's offset from 12-tone, in `UNITS_PER_CENT`.
    pub offsets: [i32; 128],
    /// Keys that do not sound: a keyboard map's `x`. Bit `k` is key `k`.
    pub silent: u128,
    /// What it was made from, for a message.
    pub name: String,
}

impl Tuning {
    /// `n` equal steps to the octave, one on each key: key `k` is `(k - ref) * 12 / n` \[6\]
    pub fn edo(n: u32, reference: Reference) -> Result<Tuning, String> {
        if !(1..=1200).contains(&n) {
            return Err(format!("--edo takes the steps to the octave, 1 to 1200, not {n}"));
        }
        let step = 12.0 / n as f64;
        Ok(Tuning::from_pitches(format!("{n}-EDO"), |k| {
            Some(reference.semitones + (k as f64 - reference.key as f64) * step)
        }))
    }

    /// A `.scl` and, optionally, a `.kbm`. `reference` (`--tuning-ref`) wins over the \[7\]
    pub fn scala(scl: &str, kbm: Option<&str>, reference: Option<Reference>) -> Result<Tuning, String> {
        let scale = Scale::parse(scl)?;
        let map = match kbm {
            Some(text) => KeyMap::parse(text)?,
            None => KeyMap::default(),
        };
        let cents = |key: u8| map.cents(&scale, key);
        // [8]
        let (ref_key, ref_semis) = match reference {
            Some(r) => (r.key, r.semitones),
            None => (map.ref_key, map.ref_semitones.unwrap_or(60.0)),
        };
        let Some(Some(ref_cents)) = cents(ref_key) else {
            return Err(format!(
                "the reference note, key {ref_key}, is not mapped to a degree of the scale, so there is no pitch to lay it out from"
            ));
        };
        let name = if scale.description.is_empty() { "a Scala scale".to_string() } else { scale.description.clone() };
        Ok(Tuning::from_pitches(name, |k| match cents(k) {
            Some(Some(c)) => Some(ref_semis + (c - ref_cents) / 100.0),
            // Outside the map's range: not retuned.
            None => Some(k as f64),
            // Mapped to `x`: no pitch, no sound.
            Some(None) => None,
        }))
    }

    /// The table from a pitch for each key, in 12-tone semitones; `None` is a key \[9\]
    fn from_pitches(name: String, pitch: impl Fn(u8) -> Option<f64>) -> Tuning {
        let mut t = Tuning { offsets: [0; 128], silent: 0, name };
        for k in 0..128u8 {
            match pitch(k) {
                Some(p) => {
                    let off = ((p - k as f64) * UNITS_PER_SEMITONE as f64).round();
                    t.offsets[k as usize] = off.clamp(-OFFSET_MAX_SEMITONES * UNITS_PER_SEMITONE as f64, OFFSET_MAX_SEMITONES * UNITS_PER_SEMITONE as f64) as i32;
                }
                None => t.silent |= 1u128 << k,
            }
        }
        t
    }

    /// Whether the table moves no key and silences none: 12-tone, which a render can \[10\]
    pub fn is_plain(&self) -> bool {
        self.silent == 0 && self.offsets.iter().all(|&o| o == 0)
    }

    /// The pitch of `key` in 12-tone semitones, for a test and a message.
    pub fn semitones(&self, key: u8) -> f64 {
        key as f64 + self.offsets[key as usize] as f64 / UNITS_PER_SEMITONE as f64
    }

    pub fn is_silent(&self, key: u8) -> bool {
        self.silent >> key & 1 != 0
    }
}

/// The key a pitch is nearest, and the cents left over: what a voice is built with. \[11\]
#[inline]
pub fn lookup(key: u8, offset: i32) -> (u8, f32) {
    let units = key as i64 * UNITS_PER_SEMITONE + offset as i64;
    let nearest = ((units + UNITS_PER_SEMITONE / 2).div_euclid(UNITS_PER_SEMITONE)).clamp(0, 127);
    (nearest as u8, ((units - nearest * UNITS_PER_SEMITONE) as f64 / UNITS_PER_CENT as f64) as f32)
}

/// A `.scl` file read.
struct Scale {
    description: String,
    /// Degree 0, which is 0 cents, then each pitch the file lists; the last is the \[12\]
    cents: Vec<f64>,
}

/// Why a line did not read, with its number.
struct Where<'a> {
    what: &'a str,
}

impl Where<'_> {
    fn at(&self, line: usize, msg: impl fmt::Display) -> String {
        format!("{} line {line}: {msg}", self.what)
    }
}

/// The lines of a Scala file that count: a `!` begins a comment, and the rest is \[13\]
fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().map(|(i, l)| (i + 1, l.trim_end_matches('\r'))).filter(|(_, l)| !l.trim_start().starts_with('!'))
}

/// The first word of a line: Scala puts pitches, counts and map entries first and \[14\]
fn first_word(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

impl Scale {
    fn parse(text: &str) -> Result<Scale, String> {
        let w = Where { what: "the scale file" };
        let mut rows = lines(text);
        // The description may be an empty line, so it is taken as it comes.
        let (_, description) = rows.next().ok_or("the scale file is empty")?;
        let description = description.trim().to_string();
        // After it, blank lines are not counted.
        let mut rows = rows.filter(|(_, l)| !l.trim().is_empty());
        let (n_line, count) = rows.next().ok_or_else(|| "the scale file stops after its description; the next line is the number of notes".to_string())?;
        let n: usize = first_word(count)
            .parse()
            .map_err(|_| w.at(n_line, format!("{:?} is not a number of notes", first_word(count))))?;
        if n == 0 {
            return Err(w.at(n_line, "a scale of no notes has no period; the last note is the octave, so a scale lists at least that"));
        }
        if n > 4096 {
            return Err(w.at(n_line, format!("{n} notes is more than Kestrel reads (4096)")));
        }
        let mut cents = vec![0.0];
        for _ in 0..n {
            let (i, l) = rows.next().ok_or_else(|| {
                format!("the scale file says {n} notes and has {}", cents.len() - 1)
            })?;
            cents.push(pitch_cents(first_word(l)).map_err(|e| w.at(i, e))?);
        }
        let period = *cents.last().unwrap();
        if period <= 0.0 {
            return Err(format!(
                "the scale file's last note is {period:.3} cents, and it is the period, so it has to be above 0"
            ));
        }
        Ok(Scale { description, cents })
    }

    fn degrees(&self) -> i64 {
        self.cents.len() as i64 - 1
    }

    fn period(&self) -> f64 {
        *self.cents.last().unwrap()
    }
}

/// One pitch of a scale in cents: a number with a `.` is cents, otherwise a ratio, `a/b` \[15\]
fn pitch_cents(word: &str) -> Result<f64, String> {
    let bad = || format!("{word:?} is not a pitch; give cents with a point (701.955) or a ratio (3/2)");
    if word.is_empty() {
        return Err("an empty line where a pitch is meant".into());
    }
    if word.contains('.') {
        let c: f64 = word.parse().map_err(|_| bad())?;
        return if c.is_finite() { Ok(c) } else { Err(bad()) };
    }
    let ratio = match word.split_once('/') {
        Some((a, b)) => {
            let (a, b): (f64, f64) = (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
            if b == 0.0 {
                return Err(format!("{word:?} divides by zero"));
            }
            a / b
        }
        None => word.parse::<f64>().map_err(|_| bad())?,
    };
    if !(ratio.is_finite() && ratio > 0.0) {
        return Err(format!("{word:?} is not a ratio above 0"));
    }
    Ok(1200.0 * ratio.log2())
}

/// A `.kbm` file read, or the default: one key to a degree from key 60.
struct KeyMap {
    first: u8,
    last: u8,
    middle: u8,
    ref_key: u8,
    /// The reference frequency as 12-tone semitones; `None` is "where 12-tone has it".
    ref_semitones: Option<f64>,
    /// The degree that is the formal octave, `None` for the scale's period.
    octave_degree: Option<usize>,
    /// Empty for a linear map: each key the next degree of the scale.
    entries: Vec<Option<usize>>,
}

impl Default for KeyMap {
    fn default() -> KeyMap {
        KeyMap { first: 0, last: 127, middle: 60, ref_key: 60, ref_semitones: None, octave_degree: None, entries: Vec::new() }
    }
}

impl KeyMap {
    fn parse(text: &str) -> Result<KeyMap, String> {
        let w = Where { what: "the keyboard map" };
        // The lines that count, as (line number, first word): the numbers come in a fixed order.
        let rows: Vec<(usize, String)> = lines(text)
            .filter(|(_, l)| !l.trim().is_empty())
            .map(|(i, l)| (i, first_word(l).to_string()))
            .collect();
        let row = |at: usize, what: &str| {
            rows.get(at).cloned().ok_or_else(|| format!("the keyboard map stops before {what}"))
        };
        let whole = |at: usize, what: &str, lo: i64, hi: i64| -> Result<i64, String> {
            let (i, v) = row(at, what)?;
            let n: i64 = v.parse().map_err(|_| w.at(i, format!("{v:?} is not {what}")))?;
            if !(lo..=hi).contains(&n) {
                return Err(w.at(i, format!("{what} is {lo} to {hi}, not {n}")));
            }
            Ok(n)
        };
        let size = whole(0, "a map size", 0, 128)? as usize;
        let first = whole(1, "the first note to retune", 0, 127)? as u8;
        let last = whole(2, "the last note to retune", 0, 127)? as u8;
        let middle = whole(3, "the middle note", 0, 127)? as u8;
        let ref_key = whole(4, "the reference note", 0, 127)? as u8;
        let (i, hz) = row(5, "the reference frequency")?;
        let hz: f64 = hz.parse().map_err(|_| w.at(i, format!("{hz:?} is not a frequency in Hz")))?;
        if !(hz.is_finite() && (1.0..=40_000.0).contains(&hz)) {
            return Err(w.at(i, format!("the reference frequency is 1 to 40000 Hz, not {hz}")));
        }
        let octave = whole(6, "the formal octave degree", 0, 4096)? as usize;
        let mut entries = Vec::with_capacity(size);
        for e in 0..size {
            let (i, v) = row(7 + e, "a mapping entry")?;
            if v.eq_ignore_ascii_case("x") {
                entries.push(None);
            } else {
                let d: usize = v
                    .parse()
                    .map_err(|_| w.at(i, format!("{v:?} is not a degree of the scale, or x for a key that does not sound")))?;
                entries.push(Some(d));
            }
        }
        if first > last {
            return Err(format!("the keyboard map retunes notes {first} to {last}, which is backwards"));
        }
        Ok(KeyMap {
            first,
            last,
            middle,
            ref_key,
            ref_semitones: Some(semitones_of_hz(hz)),
            octave_degree: Some(octave),
            entries,
        })
    }

    /// `key`'s pitch in cents above the scale's degree 0, as the map has it: `None` \[16\]
    fn cents(&self, scale: &Scale, key: u8) -> Option<Option<f64>> {
        if key < self.first || key > self.last {
            return None;
        }
        let n = key as i64 - self.middle as i64;
        let (period, degree) = if self.entries.is_empty() {
            // Linear: key after key through the scale, an octave of keys at a time.
            let d = scale.degrees();
            (n.div_euclid(d), Some(n.rem_euclid(d) as usize))
        } else {
            let size = self.entries.len() as i64;
            (n.div_euclid(size), self.entries[n.rem_euclid(size) as usize])
        };
        let octave = match self.octave_degree {
            Some(d) if d > 0 && (d as i64) <= scale.degrees() => scale.cents[d],
            _ => scale.period(),
        };
        Some(degree.map(|d| {
            // [17]
            let d = d as i64;
            let (extra, d) = (d.div_euclid(scale.degrees()), d.rem_euclid(scale.degrees()));
            (period + extra) as f64 * octave + scale.cents[d as usize]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semis(t: &Tuning, k: u8) -> f64 {
        t.semitones(k)
    }

    #[test]
    fn twelve_equal_steps_change_nothing() {
        let t = Tuning::edo(12, Reference::default()).unwrap();
        assert!(t.is_plain(), "{:?}", &t.offsets[..13]);
    }

    /// Every key one step: key 60 where it is, a key an octave of steps up is a \[18\]
    #[test]
    fn an_edo_puts_a_step_on_each_key() {
        let t = Tuning::edo(31, Reference::default()).unwrap();
        assert_eq!(t.offsets[60], 0);
        assert!((semis(&t, 91) - 72.0).abs() < 1e-4, "{}", semis(&t, 91));
        assert!((semis(&t, 29) - 48.0).abs() < 1e-4, "{}", semis(&t, 29));
        assert!((semis(&t, 61) - (60.0 + 12.0 / 31.0)).abs() < 1e-4);
        assert!((semis(&t, 0) - (60.0 - 60.0 * 12.0 / 31.0)).abs() < 1e-4);
        // Exact to the unit: no key is more than a unit from where the arithmetic puts it.
        for k in 0..128u8 {
            let want = 60.0 + (k as f64 - 60.0) * 12.0 / 31.0;
            assert!((semis(&t, k) - want).abs() < 1.0 / UNITS_PER_SEMITONE as f64, "key {k}");
        }
        assert_eq!(t.silent, 0);
    }

    #[test]
    fn the_reference_moves_where_the_tuning_is_laid_out() {
        // Key 69 at 432 Hz is 31.77 cents under where 12-tone has it.
        let r = Reference::parse("69:432").unwrap();
        let t = Tuning::edo(19, r).unwrap();
        assert!((semis(&t, 69) - semitones_of_hz(432.0)).abs() < 1e-5);
        assert!((semis(&t, 69) - 68.6824).abs() < 1e-3);
        assert!((semis(&t, 70) - semis(&t, 69) - 12.0 / 19.0).abs() < 1e-5);
        // The default is middle C where 12-tone has it, to the unit.
        let d = Reference::parse("60:261.6256").unwrap();
        assert!((d.semitones - 60.0).abs() < 1e-5);
        for bad in ["", "60", "60:", ":440", "x:440", "128:440", "60:0", "60:-1", "60:abc", "60:99999"] {
            assert!(Reference::parse(bad).is_err(), "{bad:?}");
        }
        assert!(Tuning::edo(0, r).is_err() && Tuning::edo(1201, r).is_err());
    }

    #[test]
    fn a_lookup_is_the_nearest_key_and_what_is_left_over() {
        // A key on its own pitch.
        assert_eq!(lookup(60, 0), (60, 0.0));
        // 31 steps in an octave: key 61 is a step up, 38.7 cents, and plays key 60's sample.
        let t = Tuning::edo(31, Reference::default()).unwrap();
        let (k, c) = lookup(61, t.offsets[61]);
        assert_eq!(k, 60);
        assert!((c as f64 - 1200.0 / 31.0).abs() < 1e-3, "{c}");
        // A step past half a semitone is the next key's sample, flat of it.
        let (k, c) = lookup(62, t.offsets[62]);
        assert_eq!(k, 61);
        assert!((c as f64 - (2.0 * 1200.0 / 31.0 - 100.0)).abs() < 1e-3, "{c}");
        // The pitch is the same either way: key * 100 + cents is what the key is tuned to.
        for k in 0..128u8 {
            let (nearest, cents) = lookup(k, t.offsets[k as usize]);
            let pitch = nearest as f64 * 100.0 + cents as f64;
            assert!((pitch - semis(&t, k) * 100.0).abs() < 1e-3, "key {k}");
            assert!((cents as f64).abs() <= 50.0 + 1e-3 || nearest == 0 || nearest == 127, "key {k}: {cents}");
        }
        // Past the keyboard the remainder carries what the key cannot.
        let (k, c) = lookup(127, 5 * UNITS_PER_SEMITONE as i32);
        assert_eq!(k, 127);
        assert!((c - 500.0).abs() < 1e-3, "{c}");
        let (k, c) = lookup(0, -3 * UNITS_PER_SEMITONE as i32);
        assert_eq!((k, c), (0, -300.0));
    }

    const FIFTH_COMMA: &str = "! meantone.scl\n! a comment\nQuarter-comma meantone\n 12\n!\n 76.04900\n 193.15686\n 310.26471\n 386.31371\n 503.42157\n 579.47057\n 696.57843\n 772.63137\n 889.73529\n 1006.84314\n 1082.89214\n 2/1\n";

    #[test]
    fn a_scale_in_cents_and_ratios_is_read() {
        let t = Tuning::scala(FIFTH_COMMA, None, None).unwrap();
        assert_eq!(t.name, "Quarter-comma meantone");
        // Linear from key 60, which is where it is: degree 7 is the 696.578 cent fifth.
        assert_eq!(t.offsets[60], 0);
        assert!((semis(&t, 67) - (60.0 + 6.9657843)).abs() < 1e-4, "{}", semis(&t, 67));
        // The octave is the ratio 2/1, 1200 cents: key 72 is an octave up, on the dot.
        assert!((semis(&t, 72) - 72.0).abs() < 1e-5);
        assert!((semis(&t, 48) - 48.0).abs() < 1e-5);
        // Ratios: 3/2 is 701.955 cents; an integer is a ratio over one.
        let t = Tuning::scala("just\n3\n9/8\n5/4\n2\n", None, None).unwrap();
        assert!((semis(&t, 61) - (60.0 + 2.0393)).abs() < 1e-3);
        assert!((semis(&t, 62) - (60.0 + 3.8631)).abs() < 1e-3);
        assert!((semis(&t, 63) - 72.0).abs() < 1e-5, "{}", semis(&t, 63));
    }

    #[test]
    fn what_follows_a_pitch_on_its_line_is_a_comment_and_a_description_may_be_empty() {
        let t = Tuning::scala("\n 2 \n 600.0 half an octave\n 1200. the octave\n", None, None).unwrap();
        assert_eq!(t.name, "a Scala scale");
        assert!((semis(&t, 61) - 66.0).abs() < 1e-5);
        assert!((semis(&t, 62) - 72.0).abs() < 1e-5);
        // Windows line ends.
        let crlf = "x\r\n1\r\n1200.0\r\n";
        assert!(!Tuning::scala(crlf, None, None).unwrap().is_plain());
    }

    #[test]
    fn a_keyboard_map_says_which_key_plays_which_degree_and_where_the_reference_is() {
        // [19]
        let scl = "five\n5\n 240.0\n 480.0\n 720.0\n 960.0\n 1200.0\n";
        let kbm = "! map\n4\n0\n127\n60\n60\n261.6256\n5\n0\n2\nx\n3\n";
        let t = Tuning::scala(scl, Some(kbm), None).unwrap();
        // Key 60 is degree 0 at the reference frequency: where 12-tone has it, to a hair.
        assert!(t.offsets[60].abs() < 20, "{}", t.offsets[60]);
        // Key 61 is degree 2, 480 cents up; key 62 does not sound; key 63 is degree 3.
        assert!((semis(&t, 61) - 64.8).abs() < 1e-3, "{}", semis(&t, 61));
        assert!(t.is_silent(62) && !t.is_silent(61) && t.is_silent(58) && t.is_silent(54));
        assert!((semis(&t, 63) - 67.2).abs() < 1e-3);
        // The next period starts on key 64, an octave of 5 degrees = 1200 cents up.
        assert!((semis(&t, 64) - 72.0).abs() < 1e-3, "{}", semis(&t, 64));
        // And downward: key 59 is the last entry of the period below, degree 3.
        assert!((semis(&t, 59) - 55.2).abs() < 1e-3, "{}", semis(&t, 59));
    }

    #[test]
    fn keys_outside_the_maps_range_are_not_retuned_and_the_reference_frequency_is_honoured() {
        let scl = "octave halves\n2\n 600.0\n 1200.0\n";
        // Retune 50 to 70 only; 440 Hz at key 69.
        let kbm = "0\n50\n70\n60\n69\n440.0\n2\n";
        let t = Tuning::scala(scl, Some(kbm), None).unwrap();
        assert_eq!(t.offsets[40], 0);
        assert_eq!(t.offsets[100], 0);
        // Key 69 is 440 Hz exactly: 9 steps from middle of a 2-degree linear scale.
        assert!((semis(&t, 69) - 69.0).abs() < 1e-5, "{}", semis(&t, 69));
        // ... and its neighbours are half-octaves, 6 semitones, apart.
        assert!((semis(&t, 70) - 75.0).abs() < 1e-5, "{}", semis(&t, 70));
        // --tuning-ref wins over the map's: key 69 at 432 Hz.
        let r = Reference::parse("69:432").unwrap();
        let t = Tuning::scala(scl, Some(kbm), Some(r)).unwrap();
        assert!((semis(&t, 69) - semitones_of_hz(432.0)).abs() < 1e-5);
    }

    /// The same 31 steps by three roads: --edo 31, a scale of 31 cents values, and a \[20\]
    #[test]
    fn a_scale_of_the_same_steps_is_the_same_table_as_the_edo() {
        let mut scl = String::from("31-EDO\n31\n");
        for i in 1..=31 {
            scl.push_str(&format!("{:.9}\n", 1200.0 * i as f64 / 31.0));
        }
        let from_scale = Tuning::scala(&scl, None, None).unwrap();
        let edo = Tuning::edo(31, Reference::default()).unwrap();
        for k in 0..128 {
            // Nine decimals of a cent in the file: the units are 1/2048 of a cent.
            assert!((from_scale.offsets[k] - edo.offsets[k]).abs() <= 1, "key {k}: {} against {}", from_scale.offsets[k], edo.offsets[k]);
        }
    }

    #[test]
    fn a_file_that_is_wrong_says_which_line() {
        let err = |scl: &str, kbm: Option<&str>| Tuning::scala(scl, kbm, None).unwrap_err();
        let e = err("x\n2\n 100.0\n banana\n", None);
        assert!(e.contains("scale file line 4") && e.contains("banana"), "{e}");
        let e = err("x\nmany\n", None);
        assert!(e.contains("line 2") && e.contains("not a number of notes"), "{e}");
        let e = err("x\n3\n 100.0\n", None);
        assert!(e.contains("says 3 notes and has 1"), "{e}");
        assert!(err("x\n0\n", None).contains("line 2"));
        assert!(err("", None).contains("empty"));
        assert!(err("x\n", None).contains("number of notes"));
        let e = err("x\n1\n 3/0\n", None);
        assert!(e.contains("line 3") && e.contains("zero"), "{e}");
        let e = err("x\n1\n -5/2\n", None);
        assert!(e.contains("line 3") && e.contains("above 0"), "{e}");
        let e = err("x\n1\n -100.0\n", None);
        assert!(e.contains("period") && e.contains("above 0"), "{e}");
        // A keyboard map: the line, and what it was meant to be.
        let ok = "x\n1\n1200.0\n";
        let e = err(ok, Some("0\n0\n200\n60\n60\n440\n1\n"));
        assert!(e.contains("keyboard map line 3") && e.contains("0 to 127"), "{e}");
        let e = err(ok, Some("0\n0\n127\n60\n60\nloud\n1\n"));
        assert!(e.contains("keyboard map line 6") && e.contains("frequency"), "{e}");
        let e = err(ok, Some("2\n0\n127\n60\n60\n440\n1\n0\n"));
        assert!(e.contains("stops before a mapping entry"), "{e}");
        let e = err(ok, Some("1\n0\n127\n60\n60\n440\n1\nzzz\n"));
        assert!(e.contains("line 8") && e.contains("zzz"), "{e}");
        let e = err(ok, Some("0\n100\n50\n60\n60\n440\n1\n"));
        assert!(e.contains("backwards"), "{e}");
        // A reference that the map leaves out has no pitch to lay the rest out from.
        let e = err("x\n1\n1200.0\n", Some("1\n0\n127\n60\n60\n440\n1\nx\n"));
        assert!(e.contains("reference note") && e.contains("not mapped"), "{e}");
    }
}
