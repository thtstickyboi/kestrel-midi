// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Which cached things stay on the card, and which one goes when the room runs \[1\]

/// Eviction bookkeeping for one cache, over dense ids. \[2\]
#[derive(Default, Debug)]
pub struct Residency {
    /// Planned impulses not yet fired, per id. This is the whole of the future \[3\]
    pending: Vec<u32>,
    /// Step at which this id was last needed.
    used: Vec<u64>,
    /// Step at which this id was pinned. Equal to `now` means it is needed by \[4\]
    pinned: Vec<u64>,
    resident: Vec<bool>,
    /// Resident ids, so `victim` scans the live set rather than every id ever \[5\]
    live: Vec<u32>,
    now: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Steps in which something had to be evicted, and the worst step's count. \[6\]
    pub evicting_steps: u64,
    pub worst_step: u32,
    step_evictions: u32,
}

impl Residency {
    /// Make room for ids up to and including `id`.
    #[inline]
    pub fn reserve(&mut self, id: u32) {
        let n = id as usize + 1;
        if self.pending.len() < n {
            self.pending.resize(n, 0);
            // [7]
            self.used.resize(n, u64::MAX);
            self.pinned.resize(n, u64::MAX);
            self.resident.resize(n, false);
        }
    }

    /// A note has been planned against `id` and has not fired yet.
    #[inline]
    pub fn plan(&mut self, id: u32) {
        self.reserve(id);
        self.pending[id as usize] += 1;
    }

    /// One planned note against `id` has now been emitted.
    #[inline]
    pub fn unplan(&mut self, id: u32) {
        if let Some(p) = self.pending.get_mut(id as usize) {
            *p = p.saturating_sub(1);
        }
    }

    /// Start encoding step `step`. Ids pinned before this are candidates again.
    pub fn begin_step(&mut self, step: u64) {
        self.now = step;
        if self.step_evictions > 0 {
            self.evicting_steps += 1;
            self.worst_step = self.worst_step.max(self.step_evictions);
            self.step_evictions = 0;
        }
    }

    /// `id` is needed by the step being encoded: it cannot be evicted until the \[8\]
    #[inline]
    pub fn pin(&mut self, id: u32) {
        self.reserve(id);
        self.pinned[id as usize] = self.now;
        self.used[id as usize] = self.now;
    }

    /// Record a use of `id` **without** pinning it to the step being encoded. \[9\]
    #[inline]
    pub fn touch(&mut self, id: u32) {
        self.reserve(id);
        self.used[id as usize] = self.now;
    }

    #[inline]
    pub fn is_resident(&self, id: u32) -> bool {
        self.resident.get(id as usize).copied().unwrap_or(false)
    }

    /// Record that `id` is now held. The caller has already found it room.
    pub fn admit(&mut self, id: u32) {
        self.reserve(id);
        if !self.resident[id as usize] {
            self.resident[id as usize] = true;
            self.live.push(id);
        }
    }

    pub fn count_hit(&mut self) {
        self.hits += 1;
    }

    pub fn count_miss(&mut self) {
        self.misses += 1;
    }

    /// `n` uses that were not looked up because an earlier one in the same step \[10\]
    pub fn count_hits(&mut self, n: u64) {
        self.hits += n;
    }

    /// The id to throw away, or `None` when everything resident is pinned to \[11\]
    pub fn victim(&mut self) -> Option<u32> {
        let mut best: Option<(bool, u64, u32)> = None;
        for &id in &self.live {
            let i = id as usize;
            if self.pinned[i] == self.now {
                continue;
            }
            // [12]
            let key = (self.pending[i] > 0, self.used[i], id);
            if best.is_none() || Some(key) < best {
                best = Some(key);
            }
        }
        let id = best?.2;
        self.resident[id as usize] = false;
        if let Some(pos) = self.live.iter().position(|&x| x == id) {
            self.live.swap_remove(pos);
        }
        self.evictions += 1;
        self.step_evictions += 1;
        Some(id)
    }

    pub fn resident_count(&self) -> usize {
        self.live.len()
    }

    /// Hit rate over the whole render, as a fraction.
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            1.0
        } else {
            self.hits as f64 / n as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r() -> Residency {
        Residency::default()
    }

    #[test]
    fn a_pinned_id_is_never_the_victim() {
        let mut r = r();
        r.begin_step(1);
        for id in 0..3 {
            r.admit(id);
            r.pin(id);
        }
        assert_eq!(r.victim(), None, "everything is pinned to this step");
        r.begin_step(2);
        r.pin(2);
        // [13]
        assert_eq!(r.victim(), Some(0));
    }

    #[test]
    fn an_id_with_no_planned_use_goes_before_one_that_has_some() {
        let mut r = r();
        r.begin_step(1);
        // 0 is the *more* recently used, so LRU alone would keep 1.
        r.admit(1);
        r.pin(1);
        r.plan(1);
        r.begin_step(2);
        r.admit(0);
        r.pin(0);
        r.begin_step(3);
        assert_eq!(r.victim(), Some(0), "a pending use outranks being recent");
    }

    #[test]
    fn unplan_puts_an_id_back_in_the_first_class() {
        let mut r = r();
        r.begin_step(1);
        r.admit(0);
        r.pin(0);
        r.admit(1);
        r.pin(1);
        r.plan(1);
        r.begin_step(2);
        assert_eq!(r.victim(), Some(0));
        r.admit(0);
        r.unplan(1);
        // [14]
        assert_eq!(r.victim(), Some(0));
    }

    #[test]
    fn least_recently_used_decides_within_a_class() {
        let mut r = r();
        for (step, id) in [(1u64, 0u32), (2, 1), (3, 2)] {
            r.begin_step(step);
            r.admit(id);
            r.pin(id);
        }
        r.begin_step(4);
        assert_eq!(r.victim(), Some(0));
        assert_eq!(r.victim(), Some(1));
    }

    #[test]
    fn a_never_used_id_does_not_look_least_recently_used() {
        let mut r = r();
        r.begin_step(5);
        // [15]
        r.admit(7);
        r.admit(0);
        r.pin(0);
        r.begin_step(6);
        assert_eq!(r.victim(), Some(0), "the used one goes before the unused one");
    }

    #[test]
    fn eviction_is_reported_per_step() {
        let mut r = r();
        r.begin_step(1);
        r.admit(0);
        r.admit(1);
        r.victim();
        r.victim();
        r.begin_step(2);
        assert_eq!(r.evictions, 2);
        assert_eq!(r.evicting_steps, 1);
        assert_eq!(r.worst_step, 2);
    }
}

// [16]

/// First fit across the buffers, in order, returning `(buffer, first page)`. \[17\]
pub fn take_run(free: &mut [Vec<(u32, u32)>], want: u32) -> Option<(usize, u32)> {
    for (b, runs) in free.iter_mut().enumerate() {
        if let Some(i) = runs.iter().position(|&(_, len)| len >= want) {
            let (start, len) = runs[i];
            if len == want {
                runs.remove(i);
            } else {
                runs[i] = (start + want, len - want);
            }
            return Some((b, start));
        }
    }
    None
}

/// Give a run back, merging it with either neighbour. \[18\]
pub fn put_run(runs: &mut Vec<(u32, u32)>, start: u32, len: u32) {
    let i = runs.partition_point(|&(s, _)| s < start);
    runs.insert(i, (start, len));
    // [19]
    if i + 1 < runs.len() && runs[i].0 + runs[i].1 == runs[i + 1].0 {
        runs[i].1 += runs[i + 1].1;
        runs.remove(i + 1);
    }
    if i > 0 && runs[i - 1].0 + runs[i - 1].1 == runs[i].0 {
        runs[i - 1].1 += runs[i].1;
        runs.remove(i);
    }
}

#[cfg(test)]
mod alloc_tests {
    use super::*;

    #[test]
    fn a_run_is_taken_off_the_front_and_the_rest_stays_free() {
        let mut free = vec![vec![(0u32, 10u32)]];
        assert_eq!(take_run(&mut free, 3), Some((0, 0)));
        assert_eq!(free[0], vec![(3, 7)]);
        assert_eq!(take_run(&mut free, 7), Some((0, 3)));
        assert!(free[0].is_empty());
        assert_eq!(take_run(&mut free, 1), None);
    }

    #[test]
    fn a_run_never_straddles_two_buffers() {
        // [20]
        let mut free = vec![vec![(0u32, 4u32)], vec![(0u32, 10u32)]];
        assert_eq!(take_run(&mut free, 6), Some((1, 0)));
        assert_eq!(free[0], vec![(0, 4)]);
        assert_eq!(free[1], vec![(6, 4)]);
    }

    #[test]
    fn freed_runs_coalesce_with_both_neighbours() {
        let mut free = vec![vec![]];
        // [21]
        put_run(&mut free[0], 0, 4);
        put_run(&mut free[0], 8, 4);
        assert_eq!(free[0], vec![(0, 4), (8, 4)]);
        put_run(&mut free[0], 4, 4);
        assert_eq!(free[0], vec![(0, 12)], "the three runs must become one");
        assert_eq!(take_run(&mut free, 12), Some((0, 0)));
    }

    #[test]
    fn a_deep_kernel_can_be_made_out_of_several_short_evictions() {
        let mut free = vec![vec![(0u32, 12u32)]];
        let a = take_run(&mut free, 4).unwrap();
        let b = take_run(&mut free, 4).unwrap();
        let c = take_run(&mut free, 4).unwrap();
        assert!(take_run(&mut free, 1).is_none(), "the buffer is full");
        // [22]
        put_run(&mut free[0], c.1, 4);
        put_run(&mut free[0], a.1, 4);
        put_run(&mut free[0], b.1, 4);
        assert_eq!(take_run(&mut free, 12), Some((0, 0)));
    }

    #[test]
    fn a_hole_too_small_is_skipped_rather_than_split_across_it() {
        let mut free = vec![vec![(0u32, 2u32), (5, 6)]];
        assert_eq!(take_run(&mut free, 4), Some((0, 5)));
        assert_eq!(free[0], vec![(0, 2), (9, 2)]);
    }
}
