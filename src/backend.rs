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
    fn set_gates(&mut self, meta: &[u32], frames: &[u32]) -> Result<()>;

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

    /// Render one block into `out`, interleaved stereo, `block_frames * 2` \[6\]
    fn render(&mut self, out: &mut [f32]) -> Result<()>;

    fn stats(&self) -> BlockStats;

    fn name(&self) -> &'static str;

    /// Per-pass timings from the last block, for `--profile`. Empty when the \[7\]
    fn timings(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
}
