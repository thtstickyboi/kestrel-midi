// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The 31-EDO template (`--31edo`): how a file written for it is \[1\]

/// The first channel of the triplet that channel `c` of a port is in, counted \[2\]
const fn triplet_start(c: u8) -> Option<u8> {
    match c {
        0..=2 => Some(0),
        3..=5 => Some(3),
        6..=8 => Some(6),
        9 => None,
        10..=12 => Some(10),
        _ => Some(13),
    }
}

/// `(key offset, lane)` for each of the 31 steps of an octave.
const STEPS: [(u8, u8); 31] = [
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 0), (2, 1), (3, 0),
    (3, 1), (4, 2), (4, 0), (4, 1), (5, 2), (5, 0), (5, 1), (6, 0),
    (6, 1), (7, 2), (7, 0), (7, 1), (8, 0), (8, 1), (9, 2), (9, 0),
    (9, 1), (10, 0), (10, 1), (11, 2), (11, 0), (11, 1), (12, 2),
];

/// The channel and key a note-on or note-off on `ch` with the extended key \[3\]
#[inline]
pub fn map_note(ch: u8, key: u8) -> (u8, u8) {
    let (port, c) = (ch & !15, ch & 15);
    let Some(start) = triplet_start(c) else {
        // Drums: not remapped, and the key is read as a plain MIDI key.
        return (ch, key & 0x7F);
    };
    let (offset, lane) = STEPS[(key % 31) as usize];
    let out = offset as u32 + 12 * (key / 31) as u32 + 24 * (c - start) as u32;
    (port | (start + lane), out.min(127) as u8)
}

/// The three channels a channel-wide message on `ch` is sent on, or `None` for \[4\]
#[inline]
pub fn triplet(ch: u8) -> Option<[u8; 3]> {
    let start = (ch & !15) | triplet_start(ch & 15)?;
    Some([start, start + 1, start + 2])
}

/// The three scale/octave messages, from the byte after the `F0`, as the \[5\]
pub const TUNING: [[u8; 31]; 3] = [
    [
        0x7F, 0x7F, 0x08, 0x09, 0x00, 0x48, 0x49, 0x40, 0x00, 0x31, 0x46, 0x3B, 0x6F, 0x2D, 0x35, 0x37,
        0x5E, 0x42, 0x08, 0x33, 0x4E, 0x3D, 0x77, 0x2F, 0x3D, 0x39, 0x67, 0x2B, 0x2D, 0x35, 0x56,
    ],
    [
        0x7F, 0x7F, 0x08, 0x09, 0x01, 0x11, 0x12, 0x58, 0x63, 0x4A, 0x29, 0x54, 0x52, 0x46, 0x18, 0x50,
        0x42, 0x5A, 0x6B, 0x4C, 0x31, 0x56, 0x5A, 0x48, 0x21, 0x52, 0x4A, 0x44, 0x10, 0x4E, 0x39,
    ],
    [
        0x7F, 0x7F, 0x08, 0x09, 0x02, 0x22, 0x24, 0x27, 0x1C, 0x18, 0x63, 0x23, 0x0C, 0x14, 0x52, 0x1E,
        0x7B, 0x29, 0x25, 0x1A, 0x6B, 0x25, 0x14, 0x16, 0x5A, 0x21, 0x04, 0x12, 0x4A, 0x1C, 0x73,
    ],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mts;

    /// The table and the rules of the account this was written from, spelled \[6\]
    #[test]
    fn a_step_goes_where_the_account_puts_it() {
        // (step, key offset, lane)
        let table = "0 0 0|1 0 1|2 1 0|3 1 1|4 2 2|5 2 0|6 2 1|7 3 0|8 3 1|9 4 2|10 4 0|11 4 1|\
                     12 5 2|13 5 0|14 5 1|15 6 0|16 6 1|17 7 2|18 7 0|19 7 1|20 8 0|21 8 1|\
                     22 9 2|23 9 0|24 9 1|25 10 0|26 10 1|27 11 2|28 11 0|29 11 1|30 12 2";
        for row in table.split('|') {
            let v: Vec<u32> = row.split(' ').map(|n| n.parse().unwrap()).collect();
            let (step, offset, lane) = (v[0], v[1], v[2]);
            assert_eq!(STEPS[step as usize], (offset as u8, lane as u8));
        }

        // [7]
        assert_eq!(map_note(0, 155), (0, 60));
        // [8]
        assert_eq!(map_note(1, 155), (0, 84));
        assert_eq!(map_note(2, 155), (0, 108));
        // Step 4 (key offset 2, lane 2) on the 4-6 triplet's second channel.
        assert_eq!(map_note(4, 4), (3 + 2, 2 + 24));
        // The 11-13 and 14-16 triplets, and a clamp past 127.
        assert_eq!(map_note(10, 31), (10, 12));
        assert_eq!(map_note(12, 255), (10, 127));
        assert_eq!(map_note(15, 30), (13 + 2, 12 + 48));
        // A second port repeats the lot a port higher.
        assert_eq!(map_note(16 + 4, 4), (16 + 3 + 2, 2 + 24));
    }

    #[test]
    fn the_drum_channel_is_not_remapped() {
        for port in 0..16u8 {
            let d = port * 16 + 9;
            assert_eq!(map_note(d, 36), (d, 36));
            assert_eq!(map_note(d, 200), (d, 200 & 0x7F));
            assert_eq!(triplet(d), None);
        }
    }

    #[test]
    fn every_key_lands_on_a_playable_note_of_its_own_triplet() {
        for ch in 0..=255u8 {
            if ch & 15 == 9 {
                continue;
            }
            let t = triplet(ch).unwrap();
            assert_eq!(t[1], t[0] + 1);
            assert_eq!(t[0] & !15, ch & !15, "a triplet stays on its port");
            assert!(t[0] <= ch && ch <= t[2]);
            for key in 0..=255u8 {
                let (oc, ok) = map_note(ch, key);
                assert!(t.contains(&oc), "channel {ch} key {key} went to {oc}");
                assert!(ok <= 127);
            }
        }
    }

    #[test]
    fn a_note_off_finds_its_note_on() {
        // [9]
        for key in 0..=255u8 {
            assert_eq!(map_note(3, key), map_note(3, key));
        }
        // [10]
        for n in 0..=193u8 {
            assert_eq!(map_note(1, n), map_note(0, n + 62), "key {n}");
        }
    }

    /// The offsets in the three messages are what the step table asks for: \[11\]
    #[test]
    fn the_tuning_is_what_the_step_table_says() {
        for (lane, msg) in TUNING.iter().enumerate() {
            let m = mts::parse(msg).expect("a scale/octave message");
            // The channels of the lane across the five triplets.
            let mut want = 0u16;
            for start in [0u8, 3, 6, 10, 13] {
                want |= 1 << (start + lane as u8);
            }
            assert_eq!(m.channels, want, "lane {lane}");
            for (step, &(offset, l)) in STEPS.iter().enumerate() {
                if l as usize != lane {
                    continue;
                }
                let cents = 1200.0 * step as f64 / 31.0 - 100.0 * offset as f64;
                // The pitch class is the key offset's, which for offset 12 is C again.
                let got = mts::cents(m.units[(offset % 12) as usize]) as f64;
                let unit = 100.0 / 8192.0;
                assert!(
                    got <= cents + 1e-9 && cents - got < unit,
                    "lane {lane} step {step}: message says {got}, table says {cents}"
                );
            }
        }
    }
}
