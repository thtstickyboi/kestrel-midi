// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! How many voices one submission to the device covers. \[1\]

/// The host waited this long, per part, for the device: halve the budget. \[2\]
pub(crate) const SLOW_SECS: f64 = 0.4;
/// The device had a part back within this long of its submission, per part: \[3\]
pub(crate) const CALM_SECS: f64 = 0.15;
/// Blocks in a row that must be calm before the budget doubles.
pub(crate) const CALM_BLOCKS: u32 = 16;
/// The budget never goes below this: under it the submissions cost more than \[4\]
pub(crate) const FLOOR: u32 = 65_536;

#[derive(Debug, Clone)]
pub(crate) struct SubmitBudget {
    ceiling: u32,
    now: u32,
    calm: u32,
}

impl SubmitBudget {
    pub fn new(ceiling: u32) -> Self {
        let ceiling = ceiling.max(1);
        Self { ceiling, now: ceiling, calm: 0 }
    }

    /// Voices one submission may cover now.
    pub fn voices(&self) -> u32 {
        self.now
    }

    /// A block of `voices` went up as `parts` submissions. `wait` is how long \[5\]
    pub fn observe(&mut self, voices: u32, parts: u32, wait: f64, elapsed: f64) -> Option<(u32, u32)> {
        let before = self.now;
        let parts = parts.max(1) as f64;
        if wait / parts > SLOW_SECS {
            self.now = (self.now / 2).max(FLOOR.min(self.ceiling));
            self.calm = 0;
        } else if elapsed / parts < CALM_SECS && voices >= self.now && self.now < self.ceiling {
            // [6]
            self.calm += 1;
            if self.calm >= CALM_BLOCKS {
                self.now = self.now.saturating_mul(2).min(self.ceiling);
                self.calm = 0;
            }
        } else {
            self.calm = 0;
        }
        (self.now != before).then_some((before, self.now))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_starts_at_the_ceiling_and_a_fast_card_never_moves_it() {
        let mut b = SubmitBudget::new(2_097_152);
        for _ in 0..1000 {
            b.observe(1_000_000, 1, 0.05, 0.11);
        }
        assert_eq!(b.voices(), 2_097_152);
    }

    #[test]
    fn a_slow_part_halves_it_and_a_slow_card_settles_where_parts_are_short() {
        let mut b = SubmitBudget::new(2_097_152);
        // A card that takes 1 us a voice: 1.3M voices is 1.3 s in one part.
        let secs = |voices: u32, parts: u32| voices as f64 * 1e-6 / parts as f64;
        let mut parts = 1;
        for _ in 0..12 {
            let voices = 1_300_000u32;
            parts = voices.div_ceil(b.voices());
            let part = secs(voices, parts);
            b.observe(voices, parts, part * parts as f64, part * parts as f64);
        }
        assert!(b.voices() < 1_300_000, "still one part: {}", b.voices());
        assert!(secs(1_300_000, parts) <= SLOW_SECS, "parts of {} s", secs(1_300_000, parts));
        // And it stopped there rather than running down to the floor.
        assert!(b.voices() > FLOOR * 2, "{}", b.voices());
    }

    #[test]
    fn it_never_goes_below_the_floor() {
        let mut b = SubmitBudget::new(2_097_152);
        for _ in 0..100 {
            b.observe(5_000_000, 40, 40.0, 40.0);
        }
        assert_eq!(b.voices(), FLOOR);
        let mut small = SubmitBudget::new(1000);
        small.observe(1000, 1, 10.0, 10.0);
        assert_eq!(small.voices(), 1000, "a ceiling under the floor stays where it was set");
    }

    #[test]
    fn it_grows_back_only_after_a_run_of_full_short_parts() {
        let mut b = SubmitBudget::new(2_097_152);
        b.observe(2_000_000, 1, 1.0, 1.1);
        b.observe(2_000_000, 1, 1.0, 1.1);
        assert_eq!(b.voices(), 524_288);
        // Quiet blocks, however short, do not count: they did not fill a part.
        for _ in 0..200 {
            b.observe(10_000, 1, 0.001, 0.002);
        }
        assert_eq!(b.voices(), 524_288);
        // Full blocks that were short, fifteen of them, are not yet enough.
        for _ in 0..CALM_BLOCKS - 1 {
            b.observe(1_000_000, 2, 0.02, 0.04);
        }
        assert_eq!(b.voices(), 524_288);
        b.observe(1_000_000, 2, 0.02, 0.04);
        assert_eq!(b.voices(), 1_048_576);
        // A block in between that was not calm starts the run again.
        for _ in 0..CALM_BLOCKS - 1 {
            b.observe(2_000_000, 2, 0.02, 0.04);
        }
        b.observe(2_000_000, 2, 0.1, 0.5);
        for _ in 0..CALM_BLOCKS - 1 {
            b.observe(2_000_000, 2, 0.02, 0.04);
        }
        assert_eq!(b.voices(), 1_048_576);
        b.observe(2_000_000, 2, 0.02, 0.04);
        assert_eq!(b.voices(), 2_097_152);
        for _ in 0..100 {
            b.observe(5_000_000, 3, 0.02, 0.04);
        }
        assert_eq!(b.voices(), 2_097_152, "never above the ceiling");
    }

    #[test]
    fn a_host_bound_render_is_never_taken_for_a_slow_device() {
        // The host spent 0.9 s a block on its own work and waited 10 ms.
        let mut b = SubmitBudget::new(2_097_152);
        for _ in 0..100 {
            b.observe(1_000_000, 1, 0.01, 0.9);
        }
        assert_eq!(b.voices(), 2_097_152);
    }
}
