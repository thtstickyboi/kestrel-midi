// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Just enough ZIP to hand a user one file to send: entries stored as they \[1\]

use std::io::{self, Write};

pub struct ZipWriter<W: Write> {
    out: W,
    at: u64,
    entries: Vec<Entry>,
    time: (u16, u16),
}

struct Entry {
    name: String,
    crc: u32,
    size: u32,
    offset: u32,
}

const MAX: u64 = u32::MAX as u64;

impl<W: Write> ZipWriter<W> {
    pub fn new(out: W) -> ZipWriter<W> {
        ZipWriter { out, at: 0, entries: Vec::new(), time: dos_now() }
    }

    /// Add `data` as `name`, a path inside the archive with `/` between \[2\]
    pub fn add(&mut self, name: &str, data: &[u8]) -> io::Result<()> {
        if data.len() as u64 >= MAX || self.at >= MAX {
            return Err(io::Error::other("a machine report is limited to 4 GiB"));
        }
        let crc = crc32(data);
        let (time, date) = self.time;
        let mut h = Vec::with_capacity(30 + name.len());
        h.extend(0x0403_4b50u32.to_le_bytes());
        h.extend(20u16.to_le_bytes()); // version needed: 2.0
        h.extend(0x0800u16.to_le_bytes()); // the name is UTF-8
        h.extend(0u16.to_le_bytes()); // stored
        h.extend(time.to_le_bytes());
        h.extend(date.to_le_bytes());
        h.extend(crc.to_le_bytes());
        h.extend((data.len() as u32).to_le_bytes());
        h.extend((data.len() as u32).to_le_bytes());
        h.extend((name.len() as u16).to_le_bytes());
        h.extend(0u16.to_le_bytes());
        h.extend(name.as_bytes());
        self.out.write_all(&h)?;
        self.out.write_all(data)?;
        self.entries.push(Entry { name: name.to_string(), crc, size: data.len() as u32, offset: self.at as u32 });
        self.at += (h.len() + data.len()) as u64;
        Ok(())
    }

    /// Write the central directory and hand the output back.
    pub fn finish(mut self) -> io::Result<W> {
        let start = self.at;
        let (time, date) = self.time;
        let mut cd = Vec::new();
        for e in &self.entries {
            cd.extend(0x0201_4b50u32.to_le_bytes());
            cd.extend(20u16.to_le_bytes()); // made by 2.0
            cd.extend(20u16.to_le_bytes()); // needed 2.0
            cd.extend(0x0800u16.to_le_bytes());
            cd.extend(0u16.to_le_bytes());
            cd.extend(time.to_le_bytes());
            cd.extend(date.to_le_bytes());
            cd.extend(e.crc.to_le_bytes());
            cd.extend(e.size.to_le_bytes());
            cd.extend(e.size.to_le_bytes());
            cd.extend((e.name.len() as u16).to_le_bytes());
            cd.extend([0u8; 12]); // extra, comment, disk, internal and external attributes
            cd.extend(e.offset.to_le_bytes());
            cd.extend(e.name.as_bytes());
        }
        if start + cd.len() as u64 >= MAX || self.entries.len() >= u16::MAX as usize {
            return Err(io::Error::other("a machine report is limited to 4 GiB and 65,535 files"));
        }
        self.out.write_all(&cd)?;
        let mut end = Vec::with_capacity(22);
        end.extend(0x0605_4b50u32.to_le_bytes());
        end.extend([0u8; 4]); // this disk, the directory's disk
        end.extend((self.entries.len() as u16).to_le_bytes());
        end.extend((self.entries.len() as u16).to_le_bytes());
        end.extend((cd.len() as u32).to_le_bytes());
        end.extend((start as u32).to_le_bytes());
        end.extend(0u16.to_le_bytes());
        self.out.write_all(&end)?;
        self.out.flush()?;
        Ok(self.out)
    }
}

/// The local time as MS-DOS keeps it, which is what a ZIP entry carries.
fn dos_now() -> (u16, u16) {
    use chrono::{Datelike, Timelike};
    let t = chrono::Local::now();
    let time = ((t.hour() << 11) | (t.minute() << 5) | (t.second() / 2)) as u16;
    let date = (((t.year().clamp(1980, 2107) - 1980) as u32) << 9 | (t.month() << 5) | t.day()) as u16;
    (time, date)
}

/// CRC-32 as ZIP uses it: the reflected 0xEDB88320 polynomial.
pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *v = c;
        }
        t
    });
    !data.iter().fold(!0u32, |c, &b| table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    /// Read the archive back from its end, the way an unzipper does.
    #[test]
    fn an_archive_reads_back_through_its_central_directory() {
        let files: [(&str, &[u8]); 3] =
            [("report.txt", b"hello"), ("logs/a b.log", b"line\nline\n"), ("empty", b"")];
        let mut z = ZipWriter::new(Vec::new());
        for (n, d) in files {
            z.add(n, d).unwrap();
        }
        let zip = z.finish().unwrap();
        let u16_at = |i: usize| u16::from_le_bytes([zip[i], zip[i + 1]]) as usize;
        let u32_at = |i: usize| u32::from_le_bytes(zip[i..i + 4].try_into().unwrap()) as usize;
        let end = zip.len() - 22;
        assert_eq!(u32_at(end), 0x0605_4b50);
        assert_eq!(u16_at(end + 10), 3);
        let mut at = u32_at(end + 16);
        for (name, data) in files {
            assert_eq!(u32_at(at), 0x0201_4b50);
            let (crc, size, len, local) = (u32_at(at + 16), u32_at(at + 24), u16_at(at + 28), u32_at(at + 42));
            assert_eq!(&zip[at + 46..at + 46 + len], name.as_bytes());
            assert_eq!(u32_at(local), 0x0403_4b50);
            let body = local + 30 + u16_at(local + 26);
            assert_eq!(&zip[body..body + size], data);
            assert_eq!(crc as u32, crc32(data));
            at += 46 + len;
        }
    }
}
