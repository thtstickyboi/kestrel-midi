// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Minimal RIFF/WAVE reader and writer, plus a FLAC reading path. \[1\]

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit signed integer.
    Pcm16,
    /// 32-bit IEEE float.
    Float32,
}

impl SampleFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "pcm16" | "16" | "s16" | "int16" => Some(SampleFormat::Pcm16),
            "float32" | "f32" | "32" | "float" => Some(SampleFormat::Float32),
            _ => None,
        }
    }
    fn bits(self) -> u16 {
        match self {
            SampleFormat::Pcm16 => 16,
            SampleFormat::Float32 => 32,
        }
    }
    fn tag(self) -> u16 {
        match self {
            SampleFormat::Pcm16 => 1,     // WAVE_FORMAT_PCM
            SampleFormat::Float32 => 3,   // WAVE_FORMAT_IEEE_FLOAT
        }
    }
    fn bytes(self) -> u32 {
        u32::from(self.bits()) / 8
    }
}

// [2]

/// Streaming WAVE writer. Sizes are patched into the header on `finish`.
pub struct WavWriter {
    out: BufWriter<File>,
    format: SampleFormat,
    channels: u16,
    data_bytes: u64,
    finished: bool,
}

impl WavWriter {
    pub fn create(
        path: impl AsRef<Path>,
        sample_rate: u32,
        channels: u16,
        format: SampleFormat,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        let mut out = BufWriter::with_capacity(1 << 20, file);

        let block_align = channels as u32 * format.bytes();
        out.write_all(b"RIFF")?;
        out.write_all(&0u32.to_le_bytes())?; // patched
        out.write_all(b"WAVE")?;
        out.write_all(b"fmt ")?;
        out.write_all(&16u32.to_le_bytes())?;
        out.write_all(&format.tag().to_le_bytes())?;
        out.write_all(&channels.to_le_bytes())?;
        out.write_all(&sample_rate.to_le_bytes())?;
        out.write_all(&(sample_rate * block_align).to_le_bytes())?;
        out.write_all(&(block_align as u16).to_le_bytes())?;
        out.write_all(&format.bits().to_le_bytes())?;
        out.write_all(b"data")?;
        out.write_all(&0u32.to_le_bytes())?; // patched

        Ok(WavWriter {
            out,
            format,
            channels,
            data_bytes: 0,
            finished: false,
        })
    }

    /// Write one interleaved block of f32 samples.
    pub fn write_block(&mut self, samples: &[f32]) -> Result<()> {
        match self.format {
            SampleFormat::Float32 => {
                // bytemuck keeps this a single memcpy on little-endian hosts.
                let bytes: &[u8] = bytemuck::cast_slice(samples);
                self.out.write_all(bytes)?;
                self.data_bytes += bytes.len() as u64;
            }
            SampleFormat::Pcm16 => {
                let mut buf = Vec::with_capacity(samples.len() * 2);
                for &s in samples {
                    let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                self.out.write_all(&buf)?;
                self.data_bytes += buf.len() as u64;
            }
        }
        Ok(())
    }

    pub fn frames_written(&self) -> u64 {
        self.data_bytes / (self.channels as u64 * self.format.bytes() as u64)
    }

    pub fn finish(mut self) -> Result<u64> {
        self.finish_inner()?;
        Ok(self.data_bytes)
    }

    fn finish_inner(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        self.out.flush()?;

        let riff_size = 36u64 + self.data_bytes;
        if riff_size > u32::MAX as u64 {
            // [3]
            log::error!(
                "output exceeds 4 GiB ({} bytes); RIFF size fields saturated, \
                 use --format pcm16 or split the render",
                self.data_bytes
            );
        }
        let file = self.out.get_mut();
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&(riff_size.min(u32::MAX as u64) as u32).to_le_bytes())?;
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&(self.data_bytes.min(u32::MAX as u64) as u32).to_le_bytes())?;
        file.flush()?;
        Ok(())
    }
}

impl Drop for WavWriter {
    fn drop(&mut self) {
        if let Err(e) = self.finish_inner() {
            log::error!("failed to finalize wav header: {e}");
        }
    }
}

// [4]

#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    /// Deinterleaved to mono by taking the first channel if `channels > 1`. \[5\]
    pub interleaved: Vec<f32>,
    /// From the `smpl` chunk, if present: (start, end) in frames.
    pub loop_points: Option<(u32, u32)>,
    /// From the `smpl` chunk: MIDI note the sample was recorded at.
    pub root_key: Option<u8>,
    /// From the `smpl` chunk: pitch correction in cents.
    pub fine_tune_cents: f32,
}

impl WavData {
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.interleaved.len() / self.channels as usize
        }
    }

    /// Channel `ch` as a mono i16 buffer, which is the pool's storage format.
    pub fn channel_i16(&self, ch: usize) -> Vec<i16> {
        let nch = self.channels as usize;
        let ch = ch.min(nch.saturating_sub(1));
        self.interleaved
            .iter()
            .skip(ch)
            .step_by(nch.max(1))
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .collect()
    }
}

fn rd_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn rd_u16(r: &mut impl Read) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn rd_tag(r: &mut impl Read) -> Result<[u8; 4]> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// Read a sample file, dispatching on its contents rather than on its name. \[6\]
pub fn read(path: impl AsRef<Path>) -> Result<WavData> {
    let path = path.as_ref();
    let mut magic = [0u8; 4];
    {
        let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // [7]
        let _ = f.read_exact(&mut magic);
    }
    match &magic {
        b"fLaC" => read_flac(path),
        b"OggS" => bail!("{}: Ogg-compressed sample, which is not supported", path.display()),
        _ => read_riff(path),
    }
}

/// FLAC, whatever the file is called. \[8\]
fn read_flac(path: &Path) -> Result<WavData> {
    let mut reader = claxon::FlacReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let info = reader.streaminfo();
    if info.channels == 0 || info.bits_per_sample == 0 || info.bits_per_sample > 32 {
        bail!(
            "{}: FLAC stream declares {} channels at {} bits",
            path.display(),
            info.channels,
            info.bits_per_sample
        );
    }
    let nch = info.channels as usize;
    let scale = 1.0 / (1u64 << (info.bits_per_sample - 1)) as f32;

    let mut interleaved: Vec<f32> =
        Vec::with_capacity(info.samples.unwrap_or(0) as usize * nch);
    let mut buffer = Vec::with_capacity(info.max_block_size as usize * nch);
    let mut blocks = reader.blocks();
    loop {
        let block = blocks
            .read_next_or_eof(buffer)
            .with_context(|| format!("decoding {}", path.display()))?;
        let Some(block) = block else { break };
        let dur = block.duration() as usize;
        let base = interleaved.len();
        interleaved.resize(base + dur * nch, 0.0);
        for ch in 0..nch {
            for (i, &v) in block.channel(ch as u32).iter().enumerate() {
                interleaved[base + i * nch + ch] = v as f32 * scale;
            }
        }
        buffer = block.into_buffer();
    }

    // [9]
    let tag = |name: &str| -> Option<u32> { reader.get_tag(name).next()?.trim().parse().ok() };
    let loop_points = match (tag("LOOPSTART"), tag("LOOPLENGTH")) {
        (Some(start), Some(len)) => Some((start, start.saturating_add(len))),
        _ => None,
    };

    Ok(WavData {
        sample_rate: info.sample_rate,
        channels: info.channels.min(u16::MAX as u32) as u16,
        interleaved,
        loop_points,
        root_key: None,
        fine_tune_cents: 0.0,
    })
}

fn read_riff(path: &Path) -> Result<WavData> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 16, file);

    if &rd_tag(&mut r)? != b"RIFF" {
        bail!("{}: not a RIFF file", path.display());
    }
    let _riff_size = rd_u32(&mut r)?;
    if &rd_tag(&mut r)? != b"WAVE" {
        bail!("{}: not a WAVE file", path.display());
    }

    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits = 0u16;
    let mut tag = 0u16;
    let mut data: Option<Vec<u8>> = None;
    let mut loop_points = None;
    let mut root_key = None;
    let mut fine_tune_cents = 0.0f32;

    loop {
        let id = match rd_tag(&mut r) {
            Ok(t) => t,
            Err(_) => break,
        };
        let size = rd_u32(&mut r)? as u64;
        let padded = size + (size & 1);
        if size > file_len {
            bail!("{}: chunk {:?} claims {} bytes", path.display(), std::str::from_utf8(&id), size);
        }
        match &id {
            b"fmt " => {
                tag = rd_u16(&mut r)?;
                channels = rd_u16(&mut r)?;
                sample_rate = rd_u32(&mut r)?;
                let _byte_rate = rd_u32(&mut r)?;
                let _block_align = rd_u16(&mut r)?;
                bits = rd_u16(&mut r)?;
                if tag == 0xFFFE && size >= 40 {
                    // [10]
                    let _cb = rd_u16(&mut r)?;
                    let _valid_bits = rd_u16(&mut r)?;
                    let _mask = rd_u32(&mut r)?;
                    tag = rd_u16(&mut r)?;
                    r.seek_relative(padded as i64 - 26)?;
                } else {
                    r.seek_relative(padded as i64 - 16)?;
                }
            }
            b"data" => {
                let mut buf = vec![0u8; size as usize];
                r.read_exact(&mut buf)?;
                if padded > size {
                    r.seek_relative(1)?;
                }
                data = Some(buf);
            }
            b"smpl" => {
                let mut buf = vec![0u8; padded as usize];
                r.read_exact(&mut buf)?;
                if buf.len() >= 36 {
                    let midi_note = u32::from_le_bytes(buf[20..24].try_into().unwrap());
                    let pitch_frac = u32::from_le_bytes(buf[24..28].try_into().unwrap());
                    let num_loops = u32::from_le_bytes(buf[28..32].try_into().unwrap());
                    if midi_note < 128 {
                        root_key = Some(midi_note as u8);
                    }
                    // MIDIPitchFraction is a 0.32 fraction of a semitone.
                    fine_tune_cents = (pitch_frac as f64 / 4294967296.0 * 100.0) as f32;
                    if num_loops > 0 && buf.len() >= 36 + 24 {
                        let start = u32::from_le_bytes(buf[44..48].try_into().unwrap());
                        let end = u32::from_le_bytes(buf[48..52].try_into().unwrap());
                        loop_points = Some((start, end));
                    }
                }
            }
            _ => {
                r.seek_relative(padded as i64)?;
            }
        }
    }

    let data = data.with_context(|| format!("{}: no data chunk", path.display()))?;
    if channels == 0 {
        bail!("{}: no fmt chunk", path.display());
    }

    let interleaved = decode_samples(&data, tag, bits)
        .with_context(|| format!("{}: unsupported format tag {tag} / {bits} bits", path.display()))?;

    Ok(WavData {
        sample_rate,
        channels,
        interleaved,
        loop_points,
        root_key,
        fine_tune_cents,
    })
}

fn decode_samples(data: &[u8], tag: u16, bits: u16) -> Result<Vec<f32>> {
    let out = match (tag, bits) {
        (1, 8) => data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        (1, 24) => data
            .chunks_exact(3)
            .map(|c| {
                let v = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                v as f32 / 8_388_608.0
            })
            .collect(),
        (1, 32) => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (3, 64) => data
            .chunks_exact(8)
            .map(|c| {
                f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect(),
        _ => bail!("unsupported"),
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // [11]

    fn crc8(d: &[u8]) -> u8 {
        let mut c = 0u8;
        for &b in d {
            c ^= b;
            for _ in 0..8 {
                c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
            }
        }
        c
    }

    fn crc16(d: &[u8]) -> u16 {
        let mut c = 0u16;
        for &b in d {
            c ^= (b as u16) << 8;
            for _ in 0..8 {
                c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
            }
        }
        c
    }

    /// Block size the fixtures use. Deliberately smaller than any test's data, \[12\]
    const FIXTURE_BLOCK: usize = 32;

    /// One FLAC stream: STREAMINFO plus VERBATIM frames of `FIXTURE_BLOCK`.
    fn flac(rate: u32, channels: &[&[i16]]) -> Vec<u8> {
        let nch = channels.len();
        let n = channels[0].len();
        assert!((1..=8).contains(&nch) && (1..=65536).contains(&n));
        assert!(channels.iter().all(|c| c.len() == n));
        let last = if n.is_multiple_of(FIXTURE_BLOCK) { FIXTURE_BLOCK } else { n % FIXTURE_BLOCK };

        let mut out = b"fLaC".to_vec();
        out.push(0x80); // last metadata block, type 0 (STREAMINFO)
        out.extend_from_slice(&[0, 0, 34]);
        out.extend_from_slice(&(last.min(FIXTURE_BLOCK) as u16).to_be_bytes()); // min block
        out.extend_from_slice(&(FIXTURE_BLOCK as u16).to_be_bytes()); // max block
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // min/max frame size unknown
        let packed = ((rate as u64) << 44)
            | (((nch - 1) as u64) << 41)
            | ((16u64 - 1) << 36)
            | n as u64;
        out.extend_from_slice(&packed.to_be_bytes());
        out.extend_from_slice(&[0u8; 16]); // md5 unknown

        for (f, start) in (0..n).step_by(FIXTURE_BLOCK).enumerate() {
            let len = FIXTURE_BLOCK.min(n - start);
            // [13]
            let mut frame = vec![0xFF, 0xF8, 0x70, (((nch - 1) as u8) << 4) | 0x08];
            assert!(f < 128, "the fixture writes single-byte frame numbers");
            frame.push(f as u8); // UTF-8 coded frame number
            frame.extend_from_slice(&((len - 1) as u16).to_be_bytes());
            let crc = crc8(&frame);
            frame.push(crc);

            for ch in channels {
                frame.push(0x02); // VERBATIM, no wasted bits
                for &v in &ch[start..start + len] {
                    frame.extend_from_slice(&v.to_be_bytes());
                }
            }
            let crc = crc16(&frame);
            frame.extend_from_slice(&crc.to_be_bytes());
            out.extend_from_slice(&frame);
        }
        out
    }

    fn fixture(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("kestrel_wav_{name}"));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// The reported failure: FLAC samples under a library's own extension. \[14\]
    #[test]
    fn flac_is_read_whatever_the_extension_is() {
        let data: Vec<i16> = (0..64).map(|i| (i * 512 - 16384) as i16).collect();
        let p = fixture("ext.smp", &flac(44100, &[&data[..]]));

        let w = read(&p).expect("a FLAC file should load under any extension");
        assert_eq!(w.sample_rate, 44100);
        assert_eq!(w.channels, 1);
        assert_eq!(w.frames(), data.len());
        assert_eq!(w.channel_i16(0), data);
        std::fs::remove_file(&p).ok();
    }

    /// The transpose from claxon's planar blocks, which is this crate's own \[15\]
    #[test]
    fn flac_stereo_interleaves_in_channel_order() {
        let l: Vec<i16> = (0..48).map(|i| (i * 100) as i16).collect();
        let r: Vec<i16> = (0..48).map(|i| -((i * 100) as i16)).collect();
        let p = fixture("stereo.flac", &flac(48000, &[&l[..], &r[..]]));

        let w = read(&p).unwrap();
        assert_eq!(w.channels, 2);
        assert_eq!(w.frames(), 48);
        assert_eq!(w.channel_i16(0), l);
        assert_eq!(w.channel_i16(1), r);
        // And the scale, which is 2^-15 for 16-bit and not 2^-16.
        assert!((w.interleaved[2] - 100.0 / 32768.0).abs() < 1e-9);
        std::fs::remove_file(&p).ok();
    }

    /// A file that is neither is still reported against RIFF, so there is one \[16\]
    #[test]
    fn unknown_magic_still_reports_as_riff() {
        let p = fixture("junk.wav", b"NOPE\x00\x00\x00\x00");
        let e = read(&p).unwrap_err().to_string();
        assert!(e.contains("not a RIFF file"), "{e}");
        std::fs::remove_file(&p).ok();
    }
}
