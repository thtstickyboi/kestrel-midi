// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Bytes in and bytes out, for the state a single render saves when it stops: \[1\]

use anyhow::{bail, Result};

/// A snapshot being written.
#[derive(Default)]
pub struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    pub fn new() -> Self {
        Enc::default()
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn bool(&mut self, v: bool) {
        self.buf.push(v as u8);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i16(&mut self, v: i16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// The bits, so a NaN or a negative zero comes back as it was.
    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }

    pub fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }

    /// A count, for what follows it.
    pub fn len_of(&mut self, n: usize) {
        self.u64(n as u64);
    }

    pub fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        self.len_of(bytes.len());
        self.raw(bytes);
    }

    pub fn u8s(&mut self, v: &[u8]) {
        self.bytes(v);
    }

    pub fn bools(&mut self, v: &[bool]) {
        self.len_of(v.len());
        self.buf.extend(v.iter().map(|&b| b as u8));
    }

    pub fn u16s(&mut self, v: &[u16]) {
        self.len_of(v.len());
        for &x in v {
            self.u16(x);
        }
    }

    pub fn i16s(&mut self, v: &[i16]) {
        self.len_of(v.len());
        for &x in v {
            self.i16(x);
        }
    }

    pub fn u32s(&mut self, v: &[u32]) {
        self.len_of(v.len());
        self.buf.extend_from_slice(bytemuck::cast_slice(v));
    }

    pub fn i32s(&mut self, v: &[i32]) {
        self.len_of(v.len());
        self.buf.extend_from_slice(bytemuck::cast_slice(v));
    }

    pub fn u64s(&mut self, v: &[u64]) {
        self.len_of(v.len());
        self.buf.extend_from_slice(bytemuck::cast_slice(v));
    }

    pub fn f32s(&mut self, v: &[f32]) {
        self.len_of(v.len());
        self.buf.extend_from_slice(bytemuck::cast_slice(v));
    }

    pub fn f64s(&mut self, v: &[f64]) {
        self.len_of(v.len());
        self.buf.extend_from_slice(bytemuck::cast_slice(v));
    }

    /// Plain-old-data values, as the bytes they are in memory.
    pub fn pod<T: bytemuck::Pod>(&mut self, v: &[T]) {
        self.len_of(v.len());
        self.buf.extend_from_slice(bytemuck::cast_slice(v));
    }
}

/// A snapshot being read.
pub struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Dec { buf, pos: 0 }
    }

    pub fn left(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Every byte was read: a snapshot with some left over was made by \[2\]
    pub fn finish(self) -> Result<()> {
        if self.pos != self.buf.len() {
            bail!("{} bytes of the saved state were not read; it was made by a different build", self.left());
        }
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.left() < n {
            bail!("the saved state is cut short");
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => bail!("the saved state holds {b} where a yes or no belongs"),
        }
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("two bytes")))
    }

    pub fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().expect("two bytes")))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")))
    }

    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes")))
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// A count, which cannot be more than the bytes left could hold at \[3\]
    pub fn len_of(&mut self, each: usize) -> Result<usize> {
        let n = self.u64()?;
        let room = (self.left() / each.max(1)) as u64;
        if n > room {
            bail!("the saved state names {n} things and holds room for {room}");
        }
        Ok(n as usize)
    }

    /// A count that has to be `want`.
    pub fn len_is(&mut self, want: usize, what: &str, each: usize) -> Result<()> {
        let n = self.len_of(each)?;
        if n != want {
            bail!("the saved state has {n} {what} and this render has {want}");
        }
        Ok(())
    }

    pub fn raw(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }

    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.len_of(1)?;
        self.take(n)
    }

    pub fn u8s(&mut self) -> Result<Vec<u8>> {
        Ok(self.bytes()?.to_vec())
    }

    pub fn bools(&mut self) -> Result<Vec<bool>> {
        let n = self.len_of(1)?;
        self.take(n)?
            .iter()
            .map(|&b| match b {
                0 => Ok(false),
                1 => Ok(true),
                b => bail!("the saved state holds {b} where a yes or no belongs"),
            })
            .collect()
    }

    pub fn u16s(&mut self) -> Result<Vec<u16>> {
        let n = self.len_of(2)?;
        (0..n).map(|_| self.u16()).collect()
    }

    pub fn i16s(&mut self) -> Result<Vec<i16>> {
        let n = self.len_of(2)?;
        (0..n).map(|_| self.i16()).collect()
    }

    pub fn u32s(&mut self) -> Result<Vec<u32>> {
        self.pod::<u32>()
    }

    pub fn i32s(&mut self) -> Result<Vec<i32>> {
        self.pod::<i32>()
    }

    pub fn u64s(&mut self) -> Result<Vec<u64>> {
        self.pod::<u64>()
    }

    pub fn f32s(&mut self) -> Result<Vec<f32>> {
        self.pod::<f32>()
    }

    pub fn f64s(&mut self) -> Result<Vec<f64>> {
        self.pod::<f64>()
    }

    /// Plain-old-data values, copied out so they are aligned however the \[4\]
    pub fn pod<T: bytemuck::Pod>(&mut self) -> Result<Vec<T>> {
        let n = self.len_of(std::mem::size_of::<T>())?;
        let bytes = self.take(n * std::mem::size_of::<T>())?;
        let mut out = vec![T::zeroed(); n];
        bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(bytes);
        Ok(out)
    }

    /// Fill `into` from a stored array of exactly its length: a per-channel \[5\]
    pub fn fill_u8s(&mut self, into: &mut [u8], what: &str) -> Result<()> {
        self.len_is(into.len(), what, 1)?;
        into.copy_from_slice(self.take(into.len())?);
        Ok(())
    }

    pub fn fill_bools(&mut self, into: &mut [bool], what: &str) -> Result<()> {
        let v = self.bools()?;
        if v.len() != into.len() {
            bail!("the saved state has {} {what} and this render has {}", v.len(), into.len());
        }
        into.copy_from_slice(&v);
        Ok(())
    }

    pub fn fill_u16s(&mut self, into: &mut [u16], what: &str) -> Result<()> {
        let v = self.u16s()?;
        if v.len() != into.len() {
            bail!("the saved state has {} {what} and this render has {}", v.len(), into.len());
        }
        into.copy_from_slice(&v);
        Ok(())
    }

    pub fn fill_i16s(&mut self, into: &mut [i16], what: &str) -> Result<()> {
        let v = self.i16s()?;
        if v.len() != into.len() {
            bail!("the saved state has {} {what} and this render has {}", v.len(), into.len());
        }
        into.copy_from_slice(&v);
        Ok(())
    }

    pub fn fill_u32s(&mut self, into: &mut [u32], what: &str) -> Result<()> {
        let v = self.u32s()?;
        if v.len() != into.len() {
            bail!("the saved state has {} {what} and this render has {}", v.len(), into.len());
        }
        into.copy_from_slice(&v);
        Ok(())
    }

    pub fn fill_i32s(&mut self, into: &mut [i32], what: &str) -> Result<()> {
        let v = self.i32s()?;
        if v.len() != into.len() {
            bail!("the saved state has {} {what} and this render has {}", v.len(), into.len());
        }
        into.copy_from_slice(&v);
        Ok(())
    }

    pub fn fill_f64s(&mut self, into: &mut [f64], what: &str) -> Result<()> {
        let v = self.f64s()?;
        if v.len() != into.len() {
            bail!("the saved state has {} {what} and this render has {}", v.len(), into.len());
        }
        into.copy_from_slice(&v);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_written_reads_back_to_the_bit() {
        let mut e = Enc::new();
        e.u8(7);
        e.bool(true);
        e.u16(65_000);
        e.i16(-8192);
        e.u32(u32::MAX);
        e.i32(-204_800);
        e.u64(1 << 60);
        e.f32(-0.0);
        e.f64(f64::NAN);
        e.u32s(&[1, 2, 3]);
        e.i32s(&[-1, 0, i32::MIN]);
        e.f64s(&[0.1, 0.2]);
        e.bools(&[true, false, true]);
        e.bytes(b"hello");
        let bytes = e.into_bytes();
        let mut d = Dec::new(&bytes);
        assert_eq!(d.u8().unwrap(), 7);
        assert!(d.bool().unwrap());
        assert_eq!(d.u16().unwrap(), 65_000);
        assert_eq!(d.i16().unwrap(), -8192);
        assert_eq!(d.u32().unwrap(), u32::MAX);
        assert_eq!(d.i32().unwrap(), -204_800);
        assert_eq!(d.u64().unwrap(), 1 << 60);
        assert_eq!(d.f32().unwrap().to_bits(), (-0.0f32).to_bits());
        assert!(d.f64().unwrap().is_nan());
        assert_eq!(d.u32s().unwrap(), vec![1, 2, 3]);
        assert_eq!(d.i32s().unwrap(), vec![-1, 0, i32::MIN]);
        assert_eq!(d.f64s().unwrap(), vec![0.1, 0.2]);
        assert_eq!(d.bools().unwrap(), vec![true, false, true]);
        assert_eq!(d.bytes().unwrap(), b"hello");
        d.finish().unwrap();
    }

    #[test]
    fn a_cut_or_overlong_or_miscounted_snapshot_is_an_error_not_a_guess() {
        let mut e = Enc::new();
        e.u32s(&[1, 2, 3]);
        let bytes = e.into_bytes();
        assert!(Dec::new(&bytes[..bytes.len() - 1]).u32s().is_err());
        let mut d = Dec::new(&bytes);
        d.u32s().unwrap();
        assert!(Dec::new(&bytes).finish().is_err(), "unread bytes are refused");
        // A count far past the bytes that follow it must not allocate.
        let mut e = Enc::new();
        e.u64(u64::MAX / 2);
        assert!(Dec::new(&e.into_bytes()).u32s().is_err());
        // An array that is the wrong length for its table.
        let mut e = Enc::new();
        e.u16s(&[1, 2, 3]);
        let mut into = [0u16; 4];
        assert!(Dec::new(&e.into_bytes()).fill_u16s(&mut into, "channels").is_err());
        // Anything but 0 or 1 is not a bool.
        assert!(Dec::new(&[2]).bool().is_err());
    }
}
