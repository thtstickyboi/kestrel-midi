// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The interface the driver renders through. \[1\]

use crate::voice::SpawnCmd;
use anyhow::Result;

#[derive(Debug, Clone, Copy, Default)]
pub struct BlockStats {
    pub active_voices: u64,
    /// Voices killed to make room, cumulative.
    pub stolen: u64,
    /// Note-ons that never became voices, cumulative.
    pub dropped: u64,
    /// Peak absolute sample in the last block, before limiting.
    pub peak: f32,
}

pub trait Backend {
    /// Publish this block's note-offs: `meta` is the interleaved per-slot \[2\]
    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()>;

    /// Publish this block's per-channel controller state. `rows` is \[3\]
    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, variant: bool, cut: bool)
        -> Result<()>;

    /// Install one copy of the region params table. Variant 0 is the \[4\]
    fn set_params_variant(
        &mut self,
        index: u32,
        data: &[crate::bank::RegionParams],
        menv: &[crate::bank::ModEnvParams],
    ) -> Result<()>;

    /// Add voices to the pool. May steal or drop according to the configured \[5\]
    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()>;

    /// Start rendering one block. Returns once the work is queued; the audio \[6\]
    fn submit(&mut self) -> Result<()>;

    /// Wait for the block started by `submit` and write it into `out`, \[7\]
    fn finish(&mut self, out: &mut [f32]) -> Result<()>;

    /// Render one block, start to finish. \[8\]
    fn render(&mut self, out: &mut [f32]) -> Result<()> {
        self.submit()?;
        self.finish(out)
    }

    fn stats(&self) -> BlockStats;

    fn name(&self) -> &'static str;

    /// Per-pass timings from the last block, for `--profile`. Empty when the \[9\]
    fn timings(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
}
