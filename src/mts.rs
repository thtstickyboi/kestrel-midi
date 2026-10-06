// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! MIDI Tuning Standard scale/octave tuning: a SysEx that moves each of the \[1\]

/// What an offset is counted in: 1/2048 of a cent. The finest step either form \[2\]
pub const UNITS_PER_CENT: i32 = 2048;

/// One message: which channels it retunes, and by how much each pitch class, \[3\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleOctave {
    /// Bit `n` is channel `n`, counted from 0, on the port of the track that \[4\]
    pub channels: u16,
    pub units: [i32; 12],
}

/// A scale/octave tuning message, from the byte after its `F0`: the universal \[5\]
pub fn parse(p: &[u8]) -> Option<ScaleOctave> {
    if p.len() < 7 || !matches!(p[0], 0x7E | 0x7F) || p[2] != 0x08 {
        return None;
    }
    let two = match p[3] {
        0x08 => false,
        0x09 => true,
        _ => return None,
    };
    let need = 7 + if two { 24 } else { 12 };
    if p.len() < need || p[4..need].iter().any(|&b| b >= 0x80) {
        return None;
    }
    // [6]
    let channels = (p[6] as u16) | ((p[5] as u16) << 7) | ((p[4] as u16 & 3) << 14);
    let mut units = [0i32; 12];
    for (i, u) in units.iter_mut().enumerate() {
        *u = if two {
            let v = ((p[7 + 2 * i] as i32) << 7) | p[8 + 2 * i] as i32;
            (v - 8192) * 25
        } else {
            (p[7 + i] as i32 - 64) * UNITS_PER_CENT
        };
    }
    Some(ScaleOctave { channels, units })
}

/// Cents, for an offset in `UNITS_PER_CENT`. Exact in `f32`: an offset is at \[7\]
#[inline]
pub fn cents(units: i32) -> f32 {
    units as f32 / UNITS_PER_CENT as f32
}

/// Units in a semitone.
pub const UNITS_PER_SEMITONE: i32 = 100 * UNITS_PER_CENT;

/// Bytes of a SysEx the reader keeps for a key-tuning message: 127 changes of four \[8\]
pub const KEYS_MESSAGE_MAX: usize = 520;

/// What a single-note tuning change or a bulk tuning dump sets: the keys of one tuning \[9\]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyChanges {
    pub bank: u8,
    pub program: u8,
    /// Each key the message names, and how far its pitch is from where 12-tone puts \[10\]
    pub keys: Vec<(u8, i32)>,
}

/// Whether a message, from the byte after its `F0`, is the kind `parse_keys` reads \[11\]
pub fn is_key_tuning(p: &[u8]) -> bool {
    p.len() >= 4 && matches!(p[0], 0x7E | 0x7F) && p[2] == 0x08 && matches!(p[3], 0x01 | 0x02 | 0x04 | 0x07)
}

/// One target pitch from its three bytes -- the semitone, then 14 bits of the fraction \[12\]
fn target(key: u8, b: &[u8]) -> Option<i32> {
    if b.iter().any(|&x| x >= 0x80) || b == [0x7F, 0x7F, 0x7F] {
        return None;
    }
    let v = ((b[0] as i64) << 14) | ((b[1] as i64) << 7) | b[2] as i64;
    let units = (v * 25 + 1) / 2;
    Some((units - key as i64 * UNITS_PER_SEMITONE as i64) as i32)
}

/// A key-tuning message, from the byte after its `F0`. \[13\]
pub fn parse_keys(p: &[u8]) -> Option<KeyChanges> {
    if !is_key_tuning(p) {
        return None;
    }
    let (bank, program, count_at) = match p[3] {
        0x02 => (0, *p.get(4)?, 5),
        0x07 => (*p.get(4)?, *p.get(5)?, 6),
        0x01 | 0x04 => {
            let (bank, program, data) = if p[3] == 0x01 { (0, *p.get(4)?, 5) } else { (*p.get(4)?, *p.get(5)?, 6) };
            let at = data + 16;
            if p.len() < at + 384 || p[4..data].iter().any(|&b| b >= 0x80) || p[at..at + 384].iter().any(|&b| b >= 0x80) {
                return None;
            }
            let keys = (0..128u8).filter_map(|k| target(k, &p[at + 3 * k as usize..at + 3 * k as usize + 3]).map(|u| (k, u))).collect();
            return Some(KeyChanges { bank, program, keys });
        }
        _ => return None,
    };
    let count = *p.get(count_at)? as usize;
    let first = count_at + 1;
    if count > 127 || p.len() < first + 4 * count || p[4..first].iter().any(|&b| b >= 0x80) {
        return None;
    }
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let e = &p[first + 4 * i..first + 4 * i + 4];
        if e.iter().any(|&b| b >= 0x80) {
            return None;
        }
        if let Some(u) = target(e[0], &e[1..4]) {
            keys.push((e[0], u));
        }
    }
    Some(KeyChanges { bank, program, keys })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_byte(mask: [u8; 3], cents: [f64; 12]) -> Vec<u8> {
        let mut m = vec![0x7F, 0x7F, 0x08, 0x09, mask[0], mask[1], mask[2]];
        for c in cents {
            let v = (8192.0 + c * 8192.0 / 100.0).round() as i32;
            m.push((v >> 7) as u8);
            m.push((v & 0x7F) as u8);
        }
        m
    }

    #[test]
    fn a_two_byte_message_is_read_to_the_unit() {
        let mut c = [0.0; 12];
        c[0] = -38.71;
        c[7] = 99.0;
        c[11] = -100.0;
        let m = parse(&two_byte([0, 0, 1], c)).unwrap();
        assert_eq!(m.channels, 1);
        // -38.71 cents is 8192 - 3171.1, so 5021 as the 14-bit value.
        assert_eq!(m.units[0], (5021 - 8192) * 25);
        assert!((cents(m.units[0]) as f64 + 38.71).abs() < 0.007);
        assert_eq!(m.units[1], 0);
        assert!((cents(m.units[7]) as f64 - 99.0).abs() < 0.007);
        assert_eq!(cents(m.units[11]), -100.0);
    }

    #[test]
    fn a_one_byte_message_is_whole_cents() {
        let mut p = vec![0x7F, 0x7F, 0x08, 0x08, 0, 0, 0x01];
        p.extend([0x40, 0x00, 0x7F, 0x41, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40]);
        let m = parse(&p).unwrap();
        assert_eq!(cents(m.units[0]), 0.0);
        assert_eq!(cents(m.units[1]), -64.0);
        assert_eq!(cents(m.units[2]), 63.0);
        assert_eq!(cents(m.units[3]), 1.0);
        assert_eq!(cents(m.units[4]), -1.0);
    }

    #[test]
    fn the_channel_mask_is_read_in_the_order_the_bytes_come() {
        // [14]
        let zero = [0.0; 12];
        let lane = |m: [u8; 3]| -> Vec<u8> {
            let s = parse(&two_byte(m, zero)).unwrap();
            (0..16u8).filter(|c| s.channels >> c & 1 != 0).map(|c| c + 1).collect()
        };
        assert_eq!(lane([0x02, 0x22, 0x24]), [3, 6, 9, 13, 16]);
        assert_eq!(lane([0x01, 0x11, 0x12]), [2, 5, 8, 12, 15]);
        assert_eq!(lane([0x00, 0x48, 0x49]), [1, 4, 7, 11, 14]);
        assert_eq!(lane([0x03, 0x7F, 0x7F]), (1..=16).collect::<Vec<_>>());
    }

    #[test]
    fn the_non_real_time_header_and_any_device_are_taken() {
        let mut m = two_byte([0, 0, 1], [0.0; 12]);
        m[0] = 0x7E;
        m[1] = 0x10;
        assert!(parse(&m).is_some());
    }

    #[test]
    fn what_is_not_a_scale_octave_message_is_not_read() {
        let good = two_byte([0, 0, 1], [0.0; 12]);
        // The trailing F7 is not part of what is read, and may or may not be there.
        let mut with_end = good.clone();
        with_end.push(0xF7);
        assert_eq!(parse(&with_end), parse(&good));

        for cut in [0, 3, 6, 7, 20, good.len() - 1] {
            assert_eq!(parse(&good[..cut]), None, "cut at {cut}");
        }
        let mut other = good.clone();
        other[2] = 0x04; // not the tuning standard
        assert_eq!(parse(&other), None);
        let mut other = good.clone();
        other[3] = 0x02; // a single-note change
        assert_eq!(parse(&other), None);
        let mut other = good.clone();
        other[0] = 0x41; // a Roland message
        assert_eq!(parse(&other), None);
        let mut high = good;
        high[10] = 0x80; // a data byte with its top bit set
        assert_eq!(parse(&high), None);
    }

    /// A target pitch as the three bytes a message carries it in.
    fn bytes(semitones: f64) -> [u8; 3] {
        let v = (semitones * 16384.0).round() as u32;
        [(v >> 14) as u8, ((v >> 7) & 0x7F) as u8, (v & 0x7F) as u8]
    }

    fn single(header: u8, tt: u8, keys: &[(u8, f64)]) -> Vec<u8> {
        let mut m = vec![header, 0x7F, 0x08, 0x02, tt, keys.len() as u8];
        for &(k, s) in keys {
            m.push(k);
            m.extend(bytes(s));
        }
        m
    }

    #[test]
    fn a_single_note_change_is_read_to_the_unit() {
        // Key 60 to 62.5: two and a half semitones up, 250 cents, which is 512,000 units.
        let m = parse_keys(&single(0x7F, 0, &[(60, 62.5), (61, 61.0)])).unwrap();
        assert_eq!((m.bank, m.program), (0, 0));
        assert_eq!(m.keys, vec![(60, 250 * UNITS_PER_CENT), (61, 0)]);
        // The fraction is 14 bits of a semitone: a 16,384th is 12.5 units, rounded.
        let mut tiny = single(0x7F, 5, &[(60, 60.0)]);
        tiny[8] = 0x00;
        tiny[9] = 0x01; // 1/16384 of a semitone above key 60
        let m = parse_keys(&tiny).unwrap();
        assert_eq!(m.program, 5);
        assert_eq!(m.keys, vec![(60, 13)], "12.5 units, to the nearest");
        // Down is down: key 60 to 59.75 is -25 cents.
        let m = parse_keys(&single(0x7E, 0, &[(60, 59.75)])).unwrap();
        assert_eq!(m.keys, vec![(60, -25 * UNITS_PER_CENT)]);
        // A key can be sent to a pitch above or below its own by whole octaves.
        let m = parse_keys(&single(0x7F, 0, &[(0, 127.0), (127, 0.0)])).unwrap();
        assert_eq!(m.keys, vec![(0, 127 * UNITS_PER_SEMITONE), (127, -127 * UNITS_PER_SEMITONE)]);
    }

    #[test]
    fn a_change_with_a_bank_names_it_and_no_change_is_left_out() {
        let mut m = vec![0x7F, 0x7F, 0x08, 0x07, 3, 9, 2];
        m.extend([60, 62, 64, 0]);
        m.extend([61, 0x7F, 0x7F, 0x7F]);
        let k = parse_keys(&m).unwrap();
        assert_eq!((k.bank, k.program), (3, 9));
        // 62 and 64/128 of a semitone: 62.5, 250 cents over key 60.
        assert_eq!(k.keys, vec![(60, 250 * UNITS_PER_CENT)], "7F 7F 7F is 'no change'");
    }

    /// A bulk dump sets all 128 keys; the keys that are on their own pitch give offset zero.
    #[test]
    fn a_bulk_dump_sets_every_key_and_its_checksum_is_not_asked_for() {
        let mut m = vec![0x7E, 0x7F, 0x08, 0x01, 4];
        m.extend(b"a name, 16 byte"); // 15 characters
        m.push(b' ');
        for k in 0..128u8 {
            let semis = if k == 60 { 61.5 } else { k as f64 };
            m.extend(bytes(semis));
        }
        let mut with_sum = m.clone();
        with_sum.push(0x55); // wrong, and not read
        for msg in [&m, &with_sum] {
            let d = parse_keys(msg).unwrap();
            assert_eq!((d.bank, d.program), (0, 4));
            assert_eq!(d.keys.len(), 128);
            assert_eq!(d.keys[60], (60, 150 * UNITS_PER_CENT));
            assert!(d.keys.iter().filter(|&&(k, _)| k != 60).all(|&(_, u)| u == 0));
        }
        // The key-based dump with a bank.
        let mut k4 = vec![0x7E, 0x7F, 0x08, 0x04, 2, 7];
        k4.extend(std::iter::repeat_n(b'x', 16));
        k4.extend(m[21..].iter());
        let d = parse_keys(&k4).unwrap();
        assert_eq!((d.bank, d.program, d.keys.len()), (2, 7, 128));
        // Short of 128 keys it is not a dump.
        assert_eq!(parse_keys(&m[..m.len() - 1]), None);
    }

    #[test]
    fn what_is_not_a_key_tuning_message_is_not_read() {
        let good = single(0x7F, 0, &[(60, 62.5)]);
        assert!(is_key_tuning(&good) && parse_keys(&good).is_some());
        // A scale/octave message is not one, and nor is anything else.
        let scale = [0x7F, 0x7F, 0x08, 0x09, 0, 0, 1];
        assert!(!is_key_tuning(&scale) && parse_keys(&scale).is_none());
        let mut other = good.clone();
        other[0] = 0x41;
        assert_eq!(parse_keys(&other), None);
        other = good.clone();
        other[2] = 0x04;
        assert_eq!(parse_keys(&other), None);
        // A count that promises more than there is, and a count past 127.
        let mut short = good.clone();
        short[5] = 3;
        assert_eq!(parse_keys(&short), None);
        short[5] = 200;
        assert_eq!(parse_keys(&short), None);
        // A data byte with its top bit set, in the program and in an entry.
        let mut high = good.clone();
        high[4] = 0x80;
        assert_eq!(parse_keys(&high), None);
        high = good.clone();
        high[8] = 0x80;
        assert_eq!(parse_keys(&high), None);
        high = good;
        high[6] = 0x80;
        assert_eq!(parse_keys(&high), None);
        // A zero-change message is a message that sets nothing, not nothing.
        assert_eq!(parse_keys(&[0x7F, 0x7F, 0x08, 0x02, 0, 0]).map(|k| k.keys.len()), Some(0));
        // The trailing F7 is not part of what is read.
        let mut end = single(0x7F, 0, &[(60, 62.5)]);
        end.push(0xF7);
        assert_eq!(parse_keys(&end), parse_keys(&single(0x7F, 0, &[(60, 62.5)])));
    }
}
