// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What a per-key merlin render tells whoever watches it, and how it is \[1\]

use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// MIDI keys.
const KEYS: usize = 128;
/// Keys that have to be done, and the share of the note-ons, before the \[2\]
const KEYS_FOR_ETA: u32 = 3;
const SHARE_FOR_ETA: f64 = 0.05;

/// The error a stopped render returns, which the session turns into a \[3\]
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the render was cancelled")
    }
}

impl std::error::Error for Cancelled {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Nothing has begun.
    #[default]
    Waiting,
    Arming,
    Rendering,
    Mastering,
}

impl Stage {
    fn from_u8(n: u8) -> Self {
        match n {
            1 => Stage::Arming,
            2 => Stage::Rendering,
            3 => Stage::Mastering,
            _ => Stage::Waiting,
        }
    }
}

#[derive(Default)]
struct Keys {
    /// Note-ons on each key, and in all, from the pre-pass.
    notes: Vec<u64>,
    total: u64,
    /// Keys that have notes.
    count: u32,
    /// Keys whose last window has been through the device.
    done: u128,
    /// The key in hand and the last window of it the device has had.
    at: Option<(u8, u64)>,
    /// Windows the song takes, which each key runs over.
    windows: u64,
    /// The key store's bytes and events, when there is one.
    store: Option<(u64, u64)>,
    since: [Option<Instant>; 4],
}

/// Shared between the render's threads and a `Monitor`.
pub struct MerlinProgress {
    cancel: Arc<AtomicBool>,
    stage: AtomicU8,
    pre_bytes: AtomicU64,
    pre_total: AtomicU64,
    pre_notes: AtomicU64,
    device_busy_ns: AtomicU64,
    device_idle_ns: AtomicU64,
    master_blocks: AtomicU64,
    windows: AtomicU64,
    keys: Mutex<Keys>,
}

impl MerlinProgress {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cancel: Arc::new(AtomicBool::new(false)),
            stage: AtomicU8::new(0),
            pre_bytes: AtomicU64::new(0),
            pre_total: AtomicU64::new(0),
            pre_notes: AtomicU64::new(0),
            device_busy_ns: AtomicU64::new(0),
            device_idle_ns: AtomicU64::new(0),
            master_blocks: AtomicU64::new(0),
            windows: AtomicU64::new(0),
            keys: Mutex::new(Keys::default()),
        })
    }

    fn keys(&self) -> MutexGuard<'_, Keys> {
        self.keys.lock().unwrap_or_else(|p| p.into_inner())
    }

    // ---- stopping ---------------------------------------------------------

    /// Ask the render to stop. It does at its next check, which the pre-pass \[4\]
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    /// The flag itself, for the threads that take one.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// `Err(Cancelled)` once stopped.
    pub fn check(&self) -> anyhow::Result<()> {
        if self.is_cancelled() {
            Err(Cancelled.into())
        } else {
            Ok(())
        }
    }

    // ---- what the render reports -----------------------------------------

    fn enter(&self, stage: Stage) {
        self.keys().since[stage as usize] = Some(Instant::now());
        self.stage.store(stage as u8, Ordering::Release);
    }

    /// The pre-pass begins, over `total` bytes of tracks.
    pub fn begin_arming(&self, total: u64) {
        self.pre_total.store(total, Ordering::Relaxed);
        self.enter(Stage::Arming);
    }

    /// The pre-pass has read `bytes` more and counted `notes` more note-ons.
    pub fn add_pre(&self, bytes: u64, notes: u64) {
        self.pre_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.pre_notes.fetch_add(notes, Ordering::Relaxed);
    }

    /// The keys begin: each one's note-ons, the windows the song takes, and the \[5\]
    pub fn begin_rendering(&self, key_notes: &[u64], windows: u64, store: Option<(u64, u64)>) {
        {
            let mut k = self.keys();
            k.notes = key_notes.to_vec();
            k.total = key_notes.iter().sum();
            k.count = key_notes.iter().filter(|&&n| n > 0).count() as u32;
            k.windows = windows;
            k.store = store;
        }
        self.pre_bytes.store(self.pre_total.load(Ordering::Relaxed), Ordering::Relaxed);
        self.enter(Stage::Rendering);
    }

    /// The device has had window `index` of `key`.
    pub fn at(&self, key: u8, index: u64) {
        self.windows.fetch_add(1, Ordering::Relaxed);
        let mut k = self.keys();
        let index = match k.at {
            Some((was, i)) if was == key => i.max(index),
            _ => index,
        };
        k.at = Some((key, index));
    }

    /// The device has had all of `key`. Said again for a key's later passes \[6\]
    pub fn key_done(&self, key: u8) {
        let mut k = self.keys();
        k.done |= 1u128 << (key & 127);
        if matches!(k.at, Some((was, _)) if was == key) {
            k.at = None;
        }
    }

    /// The device thread's time busy and waiting for work, since its last say.
    pub fn add_device(&self, busy_ns: u64, idle_ns: u64) {
        self.device_busy_ns.fetch_add(busy_ns, Ordering::Relaxed);
        self.device_idle_ns.fetch_add(idle_ns, Ordering::Relaxed);
    }

    /// The master is whole and is being handed out in `blocks` blocks.
    pub fn begin_mastering(&self, blocks: u64) {
        self.master_blocks.store(blocks, Ordering::Relaxed);
        self.enter(Stage::Mastering);
    }

    // ---- what a reader sees ----------------------------------------------

    pub fn stage(&self) -> Stage {
        Stage::from_u8(self.stage.load(Ordering::Acquire))
    }

    pub fn snapshot(&self) -> MerlinSnapshot {
        let stage = self.stage();
        let now = Instant::now();
        let k = self.keys();
        let notes_of = |key: usize| k.notes.get(key).copied().unwrap_or(0);
        let mut done_notes = 0u64;
        let mut keys_done = 0u32;
        for key in 0..KEYS {
            if k.done >> key & 1 == 1 {
                done_notes += notes_of(key);
                keys_done += 1;
            }
        }
        let (key, key_notes, key_frac) = match k.at {
            Some((key, index)) if k.windows > 0 => {
                // The key has not had its last window until it is said done.
                (Some(key), notes_of(key as usize), ((index + 1) as f64 / k.windows as f64).min(0.99))
            }
            Some((key, _)) => (Some(key), notes_of(key as usize), 0.0),
            None => (None, 0, 0.0),
        };
        let overall = if k.total == 0 {
            0.0
        } else {
            ((done_notes as f64 + key_frac * key_notes as f64) / k.total as f64).min(1.0)
        };
        let busy = self.device_busy_ns.load(Ordering::Relaxed);
        let idle = self.device_idle_ns.load(Ordering::Relaxed);
        let since = |s: Stage| k.since[s as usize].map_or(0.0, |t| now.saturating_duration_since(t).as_secs_f64());
        MerlinSnapshot {
            stage,
            stage_secs: since(stage),
            pre_bytes: self.pre_bytes.load(Ordering::Relaxed),
            pre_total: self.pre_total.load(Ordering::Relaxed),
            pre_notes: self.pre_notes.load(Ordering::Relaxed),
            keys_total: k.count,
            keys_done,
            notes_total: k.total,
            notes_done: done_notes,
            key,
            key_notes,
            key_frac,
            overall,
            device_busy: (busy + idle > 0).then(|| busy as f64 / (busy + idle) as f64),
            store: k.store,
            master_blocks: self.master_blocks.load(Ordering::Relaxed),
            windows: self.windows.load(Ordering::Relaxed),
        }
    }
}

/// What a reader sees of a per-key merlin render, as of now.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MerlinSnapshot {
    pub stage: Stage,
    /// Seconds in this stage.
    pub stage_secs: f64,
    /// Arming: bytes of tracks read, in all, and note-ons counted so far.
    pub pre_bytes: u64,
    pub pre_total: u64,
    pub pre_notes: u64,
    /// Rendering: keys that have notes, keys done, note-ons in all and in the \[7\]
    pub keys_total: u32,
    pub keys_done: u32,
    pub notes_total: u64,
    pub notes_done: u64,
    /// The key being rendered, its note-ons, and how far through the song it \[8\]
    pub key: Option<u8>,
    pub key_notes: u64,
    pub key_frac: f64,
    /// Note-ons rendered over all of them, 0 to 1, the key in hand counted by \[9\]
    pub overall: f64,
    /// Of the time the device's thread has been running, how much it had work.
    pub device_busy: Option<f64>,
    /// The key store's bytes and events.
    pub store: Option<(u64, u64)>,
    /// Mastering: blocks the master is handed out in.
    pub master_blocks: u64,
    /// Windows the device has had, over every key.
    pub windows: u64,
}

impl MerlinSnapshot {
    /// The bar of the stage it is in. Mastering is of `blocks` handed out so far.
    pub fn progress(&self, blocks: u64) -> Option<f64> {
        match self.stage {
            Stage::Waiting => None,
            Stage::Arming => (self.pre_total > 0).then(|| (self.pre_bytes as f64 / self.pre_total as f64).min(1.0)),
            Stage::Rendering => Some(self.overall),
            Stage::Mastering => (self.master_blocks > 0).then(|| (blocks as f64 / self.master_blocks as f64).min(1.0)),
        }
    }

    /// Seconds of keys left, from the rate so far. `None` until three keys are \[10\]
    pub fn eta_secs(&self) -> Option<f64> {
        (self.stage == Stage::Rendering && self.keys_done >= KEYS_FOR_ETA.min(self.keys_total) && self.overall >= SHARE_FOR_ETA && self.overall < 1.0 && self.stage_secs > 1.0)
            .then(|| self.stage_secs * (1.0 - self.overall) / self.overall)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notes() -> Vec<u64> {
        let mut n = vec![0u64; KEYS];
        n[60] = 100;
        n[61] = 300;
        n[62] = 100;
        n[63] = 500;
        n
    }

    #[test]
    fn arming_is_the_bytes_read_and_it_ends_whole() {
        let p = MerlinProgress::new();
        assert_eq!(p.snapshot().progress(0), None);
        p.begin_arming(1000);
        p.add_pre(250, 40);
        let s = p.snapshot();
        assert_eq!((s.stage, s.progress(0), s.pre_notes), (Stage::Arming, Some(0.25), 40));
        p.begin_rendering(&notes(), 10, Some((7, 3)));
        assert_eq!(p.snapshot().pre_bytes, 1000, "the pre-pass is whole once the keys begin");
    }

    /// The overall bar is the note-ons of the keys done and the key in hand's \[11\]
    #[test]
    fn rendering_weights_keys_by_their_notes_and_the_key_in_hand_by_its_windows() {
        let p = MerlinProgress::new();
        p.begin_arming(1);
        p.begin_rendering(&notes(), 10, None);
        p.at(60, 4);
        let s = p.snapshot();
        assert_eq!((s.keys_total, s.keys_done, s.key), (4, 0, Some(60)));
        assert!((s.key_frac - 0.5).abs() < 1e-12);
        assert!((s.overall - 50.0 / 1000.0).abs() < 1e-12);
        p.key_done(60);
        p.key_done(60);
        p.at(63, 9);
        let s = p.snapshot();
        assert_eq!((s.keys_done, s.notes_done), (1, 100), "a key said done twice is done once");
        // Window 9 of 10 is the whole song, but a key is not done until it says.
        assert!((s.key_frac - 0.99).abs() < 1e-12);
        assert!((s.overall - (100.0 + 0.99 * 500.0) / 1000.0).abs() < 1e-12);
        p.at(63, 2);
        assert!((p.snapshot().key_frac - 0.99).abs() < 1e-12, "a window behind the last does not move the bar back");
        for k in [61, 62, 63] {
            p.key_done(k);
        }
        assert_eq!(p.snapshot().overall, 1.0);
    }

    #[test]
    fn the_estimate_waits_for_three_keys() {
        let p = MerlinProgress::new();
        p.begin_arming(1);
        p.begin_rendering(&notes(), 10, None);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        p.key_done(60);
        p.key_done(61);
        assert_eq!(p.snapshot().eta_secs(), None, "two keys");
        p.key_done(62);
        let s = p.snapshot();
        let eta = s.eta_secs().expect("three keys are done");
        assert!((eta - s.stage_secs * (1.0 - s.overall) / s.overall).abs() < 1e-9);
    }

    #[test]
    fn mastering_is_the_blocks_handed_out() {
        let p = MerlinProgress::new();
        p.begin_mastering(200);
        let s = p.snapshot();
        assert_eq!((s.stage, s.progress(50), s.progress(999)), (Stage::Mastering, Some(0.25), Some(1.0)));
    }

    #[test]
    fn cancelling_is_an_error_a_session_can_tell_from_a_failure() {
        let p = MerlinProgress::new();
        assert!(p.check().is_ok());
        let flag = p.cancel_flag();
        p.cancel();
        assert!(flag.load(Ordering::Relaxed), "the flag the threads hold is the one set");
        let e = p.check().unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some());
        assert!(e.context("while rendering").downcast_ref::<Cancelled>().is_some(), "through context");
    }

    #[test]
    fn the_device_is_busy_for_the_share_of_its_time_it_had_work() {
        let p = MerlinProgress::new();
        assert_eq!(p.snapshot().device_busy, None);
        p.add_device(300, 700);
        assert_eq!(p.snapshot().device_busy, Some(0.3));
    }
}
