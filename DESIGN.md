# How Kestrel works

The [README](README.md) says how to use Kestrel. This is how it works inside, and the measurements behind its choices.

- [The render loop](#the-render-loop)
- [Per-track rendering](#per-track-rendering)
- [Memory](#memory)
- [Sizing a job with `info`](#sizing-a-job-with-info)
- [Admission and voice stealing](#admission-and-voice-stealing)
- [Limiting](#limiting)
- [Matching BASSMIDI](#matching-bassmidi)

## The render loop

One command buffer per audio block, five compute passes: **steal**, **spawn**, **render**, **reduce**, **compact**. The host parses MIDI and resolves presets, which is branchy table-lookup work that stays on the CPU permanently; the device does everything per-voice and per-sample. The host builds the next block while the device renders the current one, rather than waiting for it, and builds a block's admitted voices on several CPU cores.

**Renders are byte-identical.** Two runs of the same file with the same settings produce bitwise identical output. Nothing in the render path uses a floating-point atomic, because float atomics are non-deterministic in ordering: the mixdown is a fixed-order tree reduction into per-workgroup partial buffers no other workgroup touches, voice slots are assigned by index rather than from an atomic counter, and the pool sort is a stable radix sort. The only atomics in the codebase are integer histogram bins. Work spread over CPU threads writes each result to its own place, so the thread count never changes the output. (An encoded file is byte-identical for the same ffmpeg build; a different ffmpeg version may encode the same audio differently.)

**Phase is 32.32 fixed point**, not float. f32 loses fractional resolution past 2^24 samples, audible as detuning on long samples, and f64 costs real throughput on consumer NVIDIA parts. WGSL has no portable u64, so the device holds it as two u32 lanes.

**The voice pool is structure-of-arrays** inside a single storage buffer: field `f` of voice `i` at `f * capacity + i`. It is re-sorted by sample region every block during compaction so neighbouring invocations hit the same cache lines. That sort is what keeps the sample fetches cheap: measured through a whole render, the GPU's memory controller sits around 3% busy, so the time goes to compute and to keeping the device fed, not to memory bandwidth.

**Note-offs do not search the pool.** The pool is reordered every block, so a voice cannot be found by index. The lookup is inverted: the host publishes, per (channel, key), how many note-offs it has seen and on which frames, as runs of the same frame, and each voice finds its own release frame by binary search over them. However many note-offs a block carries, the table stays within about 67 MB at the defaults.

**A block where nothing sounds and nothing starts never reaches the device.** Its zeros are written on the host, and the limiter still runs over them, so the output is the same bytes.

## Per-track rendering

**Up to 256 tracks share one GPU dispatch.** Each track is a lane with its own region of every buffer, and the ordinary shaders are rewritten as the pipelines are built so each access adds its lane's base. Nothing a lane computes depends on the others, so a stem is byte for byte its track rendered alone. The sample pool and the read-only tables are shared. The host work for the lanes runs on `--track-jobs` threads while the batch is on the device. A thin track costs a lane a few milliseconds per sounding block whatever its voice count, so batching is what makes thousands of thin tracks fast.

**A merge is summed exactly**, in 128-bit fixed point in units of 2^-96, which holds every f32 from 2^-73 to 2^30 without rounding. So the merged file is the same bytes in any order the tracks finish in, and a one-track file merges to exactly the file a normal render writes.

**Each track holds its own voices and candidates.** The voice total is split evenly. Every lane's pool is a region of one buffer, and an adapter binds only so much of one buffer, so the more tracks share the device at once, the fewer voices each can hold; past that the render runs fewer tracks at a time rather than refusing. Each track's admission list is capped at 64 candidates per pool slot, between 2^16 and 2^19, where a normal render allows 2^27: with 256 tracks in flight at the full cap, the largest file tested held 25 GB of host memory three minutes in, and about 2 GB with the per-track cap.

**Progress** is 0.55 of the notes read plus 0.45 of the blocks rendered inside each track's notes. Every block of every track would count the silent ones, which cost almost nothing; notes alone run ahead early, because the busiest tracks render first, and note spans alone run behind. On two community merges the projected end was off by up to 23% and 22% after the first fifth of the render, where counting blocks was off by up to 209% and 150%.

## Memory

Kestrel requests whatever limits your adapter reports and checks them before allocating. If the render pass needs more workgroup storage than your card offers, or `--max-voices` implies a pool larger than your card will bind, it stops and names the flag to lower. It does not degrade quietly.

**VRAM is usually the binding constraint.** The startup line prints the breakdown:

```
gpu: NVIDIA GeForce RTX 5060 Laptop GPU (Vulkan) | 1148.6 MiB of device buffers
(757.1 MiB sample pool, 250.0 MiB voice pool for 1310720 voices, 64.0 MiB partials)
```

Three things allocate: the **voice pool**, sized by `--max-voices` and scaling linearly (250 MiB at the 1,048,576 default); the **partial buffers** used by the mixdown, 64 MiB at the defaults; and the **sample pool**, your soundfont resampled to one rate, which for a large multi-sampled piano runs to hundreds of megabytes. If the sample pool will not fit within `--pool-budget` (2 GB by default) Kestrel halves its sample rate until it does, on the grounds that a downsampled render beats no render.

The voice pool is one storage buffer, and every adapter caps how much of a single buffer a shader may bind. That is the ceiling on `--max-voices`: **16,519,104** on an RTX 5060 at the default `--steal-percent`, 8,259,552 on an Intel RaptorLake-S iGPU. `kestrel --force-cli gpu-info` prints the figure for *your* card, and so does the guided renderer's first screen. The cap is on the pool, not on the piece -- a file with more simultaneous notes than the pool holds still renders, with the excess stolen and dropped.

**Host RAM** is mostly the soundfont -- Kestrel keeps its own copy of the sample pool -- plus the note-ons of the block being admitted. Admission cannot choose until a block's last note-on is in, so the note-ons inside one block are held until it ends, and that scales with `--block`. It is capped: past **134,217,728 candidates** in one block (about 4.5 GiB) the block is thinned evenly across its length rather than held whole, so even a block carrying billions of note-ons fits in memory. Below the cap nothing is thinned. If a render still dies on a failed allocation rather than a GPU error, lower `--block` (128 is the floor). Lowering `--max-voices` does *not* help, because the block is considered in full before admission thins it.

## Sizing a job with `info`

`kestrel info` is the first thing to reach for when a render sounds wrong: it tells you what Kestrel *thinks* your file contains, which is often not what you think it contains. On a MIDI it also sizes the job before you start it:

```bash
kestrel --force-cli info huge.mid --block 512 --sf-layers 2 --sf-release 1.5
```

```
at --block 512, 48000 Hz, --sf-layers 2, --sf-release 1.50s
  the last two assumed; pass -s <soundfont> to read them off it
  busiest block        94289755 note-ons, at 286.66s
  host memory          5.27 GiB for that block, at least
  peak 1.50s span      386892142 voices, at 285.30s -- roughly what a pool
                                must hold for nothing to be stolen
```

| what `info` says | what to do |
|---|---|
| peak span under `--max-voices` | nothing. The defaults will not drop or steal a single voice |
| peak span above it | raise `--max-voices` towards it, up to the ceiling `gpu-info` prints |
| host memory in the gigabytes | lower `--block`. Not `--max-voices` -- that does not help |

Pass `-s <soundfont>` and `info` reads `--sf-layers` and `--sf-release` off it rather than taking them on trust. For anything that is not black MIDI the first row is the answer and there is nothing to tune. The host memory row does not yet know about the per-block candidate cap under *Memory*, so for a block past it the real figure is lower than the one printed.

## Admission and voice stealing

Stealing decides who dies. **Admission decides who never starts**, and on a saturated file that is the bigger number by an order of magnitude -- on a 44.7M-note file against a 32,767-voice pool, 83.7M note-ons were dropped against 5.7M voices stolen.

How much of that you face is a function of pool size, and the pool can be far larger than the default. On that same file, against a 22,528-region piano, measured at 1.1.0:

| `--max-voices` | dropped | stolen | realtime |
|---|---|---|---|
| 1,048,576 (default) | 12,077,835 | 49,667,330 | 1.79x |
| 4,194,304 | 511,768 | 33,624,100 | 0.94x |
| 13,421,568 | **0** | **0** | 0.88x |

At its busiest the file has 5,338,314 voices sounding at once. With 13,421,568 slots every note sounds, none refused and none cut short, for 4.3 GiB of device buffers. Admission stops mattering well before stealing does, and past what the file needs a bigger pool costs memory rather than time: 16,519,104 voices renders in the same 148 s.

**Admission** ranks an oversubscribed block by opening amplitude and keeps the loudest. Thinning the block evenly instead, which a dev build offers as `--admit even` for comparison, measurably loses at small pools -- at 32,767 voices, -10.065 dB RMS against `even`'s -11.225 at an identical peak, which is what "the long loud notes survived" looks like as a number -- and the gap narrows as the pool grows, to within 0.01 dB at the default. There is no duration term: ranking is purely by opening amplitude, so a loud note cut short and a loud note held look the same.

**Stealing** kills the voices with the lowest envelope level, ties broken by note id. Killing the earliest-started instead sounds backwards under saturation and is: the oldest voices are the mature, sounding ones. (A dev build keeps `--steal oldest` and `drop-new`, which refuses the incoming note, for comparison.)

**`--steal-percent`** (25) caps how much of the pool one block may replace. Unbounded, a block whose note-ons outnumber the pool replaces *every* voice, so no voice outlives the block it was born in and a saturated passage renders as a stream of 85 ms attack fragments instead of notes.

## Limiting

Black MIDI mixes clip constantly -- a saturated section can peak at hundreds of times full scale -- so what happens at the ceiling matters more than usual.

**`brickwall`** (default) is a lookahead true-peak limiter. It sees peaks before they arrive and cannot exceed its ceiling, so nothing downstream ever has to hard-clip. `--ceiling-db` sets the ceiling. Its 2 ms lookahead is also the render latency, which for an offline renderer costs nothing.

**`omni`** is a port of the realtime limiter OmniConverter ships, originally from Kiva. It is a feedback follower: it sees a peak only once the peak has passed, which is why it needs a third of a second of release, and that release audibly drags the level down after every loud moment. It does not bound its own output, so since 1.1.0 the brickwall runs behind it as a safety stage at the render's ceiling: wherever omni stays under the ceiling the output is omni's, delayed by the lookahead, and nothing clips.

**`off`** disables limiting. Your mix will clip. Refused for encoded output.

## Matching BASSMIDI

Where Kestrel and BASSMIDI disagree, BASSMIDI is taken as the reference, and these were measured against it directly:

- **Tempo changes.** BASSMIDI advances a whole output sample at a time, fires an event -- a tempo change included -- on the first sample whose tick position has reached it, and carries the ticks that ran past a tempo change into the new tempo. Kestrel does the same, and agrees with it to 0.000 ms across 57 sections of back-to-back tempo changes. At ordinary tempos a sample is a fraction of a tick and nothing can be heard; in bursts of changes in the thousands of bpm it adds up to tens of milliseconds.
- **Channel volume** powers on at 100, not 127.
- **LFOs.** SFZ's amplitude and pitch LFOs were measured against BASSMIDI's: a triangle starting at zero, delayed from the note's own start, at the depth, rate and direction it plays them. SF2's run on the same oscillator. Every LFO and modulation envelope counts from its note's own start -- before 1.1.0 they could run up to a block early. They update every 32 frames.
- **Short notes, with `--note-grid`.** BASSMIDI moves each note's envelope in 4 ms steps counted from its note-on, so a note-off only takes effect at the next step, and a release shorter than a step becomes a 4 ms fade. A note one sample long still sounds for 4 ms. With `--note-grid` Kestrel does the same, matched to the sample; without it a note ends on its own note-off, as before. It is off by default because almost no file notices and it costs about 12% of a render, but a file that encodes audio as one-sample notes renders silent without it.
- **Portamento.** CC65 turns it on, CC5 sets the speed and CC84 names the next note's starting key, as in BASSMIDI: a note glides in a straight line in pitch from the channel's last note, and CC5 = 0, the power-on value, means no glide. Kestrel's glides match BASSMIDI's speed to within 0.03%. Mono mode (CC126) is not implemented, so a note struck over a held one glides but does not cut it.
- **MIDI ports.** A track that names a port with the FF 21 meta event gets sixteen channels of its own: channel 6 on port B is not channel 6 on port A. As in BASSMIDI, a port belongs to its track and applies from where it appears, an FF 21 that does not carry exactly one byte is ignored, and channel 10 is drums on every port. **One deliberate difference:** Kestrel keeps sixteen ports, A to P, as Domino does, where BASSMIDI keeps eight. A file that plays on ports I to P renders those parts on channels of their own here, and merged onto A to H in BASSMIDI.
