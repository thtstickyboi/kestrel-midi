# Kestrel

'*experimental*'

A GPU-accelerated SoundFont/SFZ synthesizer for Black MIDI.

Kestrel renders a MIDI file to a WAV offline, doing the synthesis in compute
shaders instead of on the CPU. It is built for files where the note count is
absurd -- hundreds of millions of notes, with a hundred thousand to several
million voices sounding at once -- which is the point where conventional synths
stop being fast enough.

On an RTX 5060 Laptop it sustains **1,048,576 concurrent voices at 1.02x
realtime**.

It is a library and a command-line tool. There is no GUI, no realtime playback,
and no plugin build.

---

## Read this first: Kestrel is 100% vibecoded

**Every line of code in this repository was written by an AI.** Not
AI-assisted, not AI-reviewed -- AI-written, start to finish, from the shaders to
the SF2 parser to the limiter. I directed the work, made the design calls, ran
the renders and decided what was acceptable, but I did not hand-write this code
and I have not read every line of it.

- **No human has line-by-line reviewed this code.** Not me, not anyone. If you
  are the sort of person who reads a diff before running it, read this one.
- **It was developed against a test suite that is not in this repository:**
  null tests against a separate single-threaded CPU reference, byte-for-byte
  determinism checks, phase-accumulator precision tests, and envelope curves
  checked against analytically computed ones. The GPU path matches the CPU
  reference to better than -98 dB. That is real verification and it caught real
  bugs -- but you are taking my word for it, because the evidence is not here.
- **The corners are where it will break.** SF2 loading, the render loop and
  voice stealing under saturation are exercised constantly and are in decent
  shape. Unusual soundfonts, exotic SFZ opcodes and malformed MIDI are much
  less certain.
- **Check the output yourself.** `kestrel null` exists precisely so you can
  diff a render against a reference you trust.

I am not claiming this is production software. I am claiming it renders black
MIDI fast and the output sounded right to me. Please open an issue when it
breaks, because it will.

---

## Requirements

| | |
|---|---|
| **Rust** | 1.80 or newer, from [rustup.rs](https://rustup.rs) |
| **GPU** | Anything with a working Vulkan, DX12 or Metal driver |
| **VRAM** | Depends on the soundfont and `--max-voices`; see *Memory*. Tuned against 8 GB |
| **OS** | Windows, Linux, macOS. Only Windows has been *run* |

No system libraries, nothing to download separately. Every dependency comes
from crates.io and the shaders are compiled into the binary.

**Tested on NVIDIA and Intel, both on Windows via Vulkan.** Rendering the same
file on an RTX 5060 and on an Intel iGPU nulls at **-111 dB peak**, and each
device is byte-identical with itself across runs. WGSL is compiled by your
driver rather than by cargo, so two vendors' shader compilers agreeing is real
evidence the shaders are portable.

Linux and macOS **compile** clean for `x86_64-unknown-linux-gnu` and
`aarch64-apple-darwin`, which type-checks wgpu's Vulkan and Metal backends.
Neither has been linked or run. AMD and Apple silicon are unrun. If you are
first on one of those, please open an issue either way -- a report that it
simply worked is as useful as a crash.

Integrated GPUs are slower but usable: an Intel part held 2.95x realtime
against the 5060's 32.5x on the same 100k-note file, about what shared DDR5
against dedicated VRAM predicts for a bandwidth-bound workload. Kestrel prefers
a discrete GPU when both are present.

There is a **complete CPU backend** -- the reference implementation the GPU path
was built against. It is far slower and single-threaded. Use `--backend cpu` if
you have no usable GPU, or to check a suspicious render against something
simpler.

### Memory

Kestrel requests whatever limits your adapter reports and checks them before
allocating. If the render pass needs more workgroup storage than your card
offers, or `--max-voices` implies a pool larger than your card will bind, it
stops and names the flag to lower. It does not degrade quietly.

**VRAM is usually the binding constraint, not compute.** The startup line
prints the breakdown:

```
gpu: NVIDIA GeForce RTX 5060 Laptop GPU (Vulkan) | 1097.0 MiB of device buffers
(757.1 MiB sample pool, 240.0 MiB voice pool for 1310720 voices, 64.0 MiB partials)
```

Three things allocate: the **voice pool**, sized by `--max-voices` and scaling
linearly (240 MiB at the 1,048,576 default); the **partial buffers** used by the
mixdown, 64 MiB at the defaults; and the **sample pool**, your soundfont
resampled to one rate, which for a large multi-sampled piano runs to hundreds of
megabytes. If the sample pool will not fit within `--pool-budget` (2 GB by
default) Kestrel halves its sample rate until it does, on the grounds that a
downsampled render beats no render.

The voice pool is one storage buffer, and every adapter caps how much of a
single buffer a shader may bind. That is the ceiling on `--max-voices`:
**16,519,104** on an RTX 5060 at the default `--steal-percent`, 8,259,552 on an
Intel RaptorLake-S iGPU. `kestrel gpu-info` prints the figure for *your* card.
The cap is on the pool, not on the piece -- a file with more simultaneous notes
than the pool holds still renders, with the excess stolen and dropped.

**Host RAM is a separate constraint, and on very large files it is the binding
one.** A block is admitted as a whole, so every note-on inside it is held until
the block ends. That scales with `--block`: a 6.6 GB, 824M-note file has one
512-frame block carrying 94 million note-ons, about 4.8 GiB at `--block 512` and
roughly eight times that at the 4096 default. If a render dies with a failed
allocation rather than a GPU error, lower `--block` (128 is the floor).
Lowering `--max-voices` does *not* help, because the block is considered in full
before admission thins it.

## Building

```bash
git clone https://github.com/thtstickyboi/kestrel-midi.git
cd kestrel-midi && cargo build --release
```

Build in **release mode**. The debug build is not merely slower, it is unusable
for real files. The binary lands at `target/release/kestrel` and is
self-contained.

Check your GPU was found before anything else:

```bash
kestrel gpu-info
```

If that lists no adapter, your driver is the problem and `--backend cpu` is the
fallback.

## Running

```bash
kestrel render input.mid -s soundfont.sfz -o output.wav
```

`-s` takes a `.sf2` or a `.sfz`, and may be **repeated to layer** -- see
*General MIDI*. Output is 32-bit float WAV by default; `--format pcm16` for
16-bit.

An SFZ library's samples may be WAV or FLAC. Which one is decided by reading the
file rather than its extension, because libraries ship FLAC under names of their
own invention and dispatching on the name silently drops every region that uses
one.

### The flags that matter

| Flag | Default | What it does |
|---|---|---|
| `--backend` | `gpu` | `cpu` is the reference implementation: correct, slow, single-threaded. |
| `--max-voices N` | `1048576` | Ceiling on simultaneous voices. The single biggest lever on both VRAM and speed. The maximum is whatever your card will bind; `gpu-info` prints yours. |
| `--block N` | `4096` | Frames per render block. Also sets how many note-ons are held in host RAM at once; lower it if a huge file exhausts memory. |
| `--interp` | `linear` | `nearest`, `linear` or `cubic`. Cubic costs roughly double the bandwidth. |
| `--seconds N` | off | Stop after N seconds of output. Use this constantly while experimenting. |
| `--limiter` | `brickwall` | `brickwall`, `omni` or `off`. See *Limiting*. |
| `--steal` | `quietest` | `quietest`, `oldest` or `drop-new`. |
| `--admit` | `loudest` | `loudest` or `even`. Which note-ons survive when a block oversubscribes the pool. |
| `--steal-percent` | `25` | How much of the pool one block may replace. |
| `--volume X` | `1.0` | Pre-limiter gain. |
| `--format` | `float32` | `float32` or `pcm16`. |
| `--nan-guard` | off | Check every block for NaN and Inf. **Off in release builds**, so a clean exit is not by itself proof the WAV is finite. |
| `--profile` | off | Per-pass GPU timings, once per wall-clock second. |
| `--block-csv` | off | One CSV row per block: voices alive at admission, layers queued and admitted, stolen, dropped, output RMS and peak. Read these rather than the waveform when level moves at the block rate. |
| `--gpu-backend`, `--gpu-adapter` | off | Force a specific API or card on multi-GPU machines. |

`kestrel render --help` lists everything, including tuning knobs
(`--workgroup`, `--reduce-tile`, `--pool-budget`) best left alone unless you are
measuring.

### The other commands

```bash
kestrel gpu-info                  # adapters, limits, and each one's --max-voices ceiling
kestrel info file.sf2             # what the loader made of a soundfont
kestrel info file.mid             # ...or of a MIDI: note counts, CC usage, tempo, density
kestrel null a.wav b.wav          # peak difference between two renders, in dB
kestrel gen-assets dir/           # write synthetic soundfonts and MIDI
```

`kestrel info` is the first thing to reach for when a render sounds wrong: it
tells you what Kestrel *thinks* your file contains, which is often not what you
think it contains. On a MIDI it also sizes the job before you start it:

```bash
kestrel info huge.mid --block 512 --sf-layers 2 --sf-release 1.5
```

```
at --block 512, --sf-layers 2, 48000 Hz:
  busiest block        94289755 note-ons, at 286.66s
  host memory          5.27 GiB for that block, at least
  peak 1.50s span     386892142 voices, at 285.30s -- roughly what a pool
                                must hold for nothing to be stolen
```

| what `info` says | what to do |
|---|---|
| peak span under `--max-voices` | nothing. The defaults will not drop or steal a single voice |
| peak span above it | raise `--max-voices` towards it, up to the ceiling `gpu-info` prints |
| host memory in the gigabytes | lower `--block`. Not `--max-voices` -- that does not help |

Pass `-s <soundfont>` and `info` reads `--sf-layers` and `--sf-release` off it
rather than taking them on trust. For anything that is not black MIDI the first
row is the answer and there is nothing to tune.

## General MIDI

New in 0.2.5, and **rudimentary**. An ordinary GM file will play, and it will
play far closer to what you expect than 0.2.4 managed, but this is a first pass
and it is not guaranteed to render a file the way a GM synth would. Treat it as
usable and unfinished, not as a supported format. What is listed below is what
was built and checked; what is under *Where it is still wrong* is what is known
to be off, and there is very likely more that has not been found yet.

**Soundfonts layer.** `-s` is repeatable and merges in order, each one replacing
anything already at the same bank and program:

```bash
kestrel render gm.mid -s general-midi.sf2 -s piano.sfz --sf-programs 0-1 -o out.wav
```

`--sf-programs` places the *last* soundfont on the programs you name, ranges
included, so a one-preset SFZ piano can cover the whole piano family. Without it
a soundfont takes only the program it declares -- which for an `.sfz` is program
0, GM's Acoustic Grand, so the common case needs no flag.

Implemented: program change, bank select with the General MIDI fallback to bank
0, drum kits in bank 128 with their own fallback ladder, channel volume,
expression, pan, the sustain and sostenuto pedals, pitch bend, RPN 0/1/2 (bend
range, fine and coarse tuning), CC71-75, CC120, CC121 and CC123-127. SF2 vibrato
and tremolo LFOs are applied, as is the SF2 modulation envelope with both its
pitch and filter destinations.

### Where it is still wrong

Measured against a reference GM synth on one 17,000-note file. None of these
stops a render; all of them mean it will not match what you are used to.

- **Everything is about 2 dB loud**, consistently, and nobody knows why. It is a
  global gain difference rather than a balance error, so `--volume 0.78`
  level-matches it for comparison purposes. Unexplained.
- **Several channels come out noticeably brighter** than the reference -- 1.2 to
  1.4 times its spectral centroid. Ruled out as causes: the lowpass cutoff
  (measured exact at four settings), interpolation, and the envelopes. Cause
  unknown.
- **The sample pool is resampled to one rate at load**, which band-limits a
  soundfont whose samples are recorded lower. On a 22 kHz General MIDI set that
  removes everything above 11 kHz, where the reference has content there.
  `--no-resample-pool` keeps the samples at their own rate and closes it.
- **Only three SysEx messages are acted on**: the GS *use for rhythm part*, GS
  Reset and GM System On/Off. Master volume, part parameters and every bulk dump
  are ignored.
- **Effects are not implemented**, so CC91 reverb and CC93 chorus sends do
  nothing -- see *Known limitations*.
- **`modLfoToFilterFc` is not implemented.**
- **Rapid CC11 gating reaches digital silence** where the reference does not,
  because channel volume is not smoothed over as long a window.
- The note-boundary click under *Known limitations* hits GM presets hardest,
  because synth-lead and wavetable patches are exactly the ones with no attack
  or release of their own.

## Limiting

Black MIDI mixes clip constantly -- a saturated section can peak at hundreds of
times full scale -- so what happens at the ceiling matters more than usual.

**`brickwall`** (default) is a lookahead true-peak limiter. It sees peaks before
they arrive and cannot exceed its ceiling, so nothing downstream ever has to
hard-clip. `--ceiling-db`, `--lookahead-ms` and `--limiter-release-ms` control
it. The lookahead is also the render latency, which for an offline renderer
costs nothing.

**`omni`** is a port of the realtime limiter OmniConverter ships, originally
from Kiva. It is a feedback follower: it sees a peak only once the peak has
passed, which is why it needs a third of a second of release, and that release
audibly drags the level down after every loud moment. It also does not bound its
own output. **Keep it only for level-matching a render against BASS or XSynth.**

**`off`** disables limiting. Your mix will clip.

## Admission and voice stealing

Stealing decides who dies. **Admission decides who never starts**, and on a
saturated file that is the bigger number by an order of magnitude -- on a
44.7M-note file against a 32,767-voice pool, 83.7M note-ons were dropped against
5.7M voices stolen.

How much of that you face is a function of pool size, and the pool can be far
larger than the default. On that same file:

| `--max-voices` | dropped | stolen | realtime |
|---|---|---|---|
| 1,048,576 (default) | 18,682,538 | 57,113,653 | 1.43x |
| 13,421,568 | 0 | 2,732,530 | -- |
| 16,519,104 | **0** | **0** | 0.67x |

Its true peak is 14,035,344 simultaneous voices, and the bottom row is the first
pool that neither drops nor steals: every note sounding, none refused and none
cut short, for 5.3 GiB of VRAM and 2.1x the wall time. Admission stops mattering
well before stealing does, and an oversized pool costs memory rather than time.

**`--admit loudest`** (default) ranks an oversubscribed block by opening
amplitude and keeps the loudest; **`even`** thins the block evenly. At small
pools `loudest` measurably wins -- at 32,767 voices, -10.065 dB RMS against
`even`'s -11.225 at an identical peak, which is what "the long loud notes
survived" looks like as a number -- and the gap narrows as the pool grows, to
within 0.01 dB at the default. There is no duration term: ranking is purely by
opening amplitude, so a loud note cut short and a loud note held look the same.

**`--steal quietest`** (default) kills the voices with the lowest envelope
level, ties broken by note id. **`oldest`** kills the earliest-started, which
sounds backwards under saturation and is: the oldest voices are the mature,
sounding ones. **`drop-new`** refuses the incoming note instead.

**`--steal-percent`** (25) caps how much of the pool one block may replace.
Unbounded, a block whose note-ons outnumber the pool replaces *every* voice, so
no voice outlives the block it was born in and a saturated passage renders as a
stream of 85 ms attack fragments instead of notes.

## How it works

One command buffer per audio block, five compute passes: **steal**, **spawn**,
**render**, **reduce**, **compact**. The host parses MIDI and resolves presets,
which is branchy table-lookup work that stays on the CPU permanently; the device
does everything per-voice and per-sample.

**Renders are byte-identical.** Two runs of the same file with the same settings
produce bitwise identical WAVs. Nothing in the render path uses a floating-point
atomic, because float atomics are non-deterministic in ordering: the mixdown is
a fixed-order tree reduction into per-workgroup partial buffers no other
workgroup touches, voice slots are assigned by index rather than from an atomic
counter, and the pool sort is a stable radix sort. The only atomics in the
codebase are integer histogram bins.

**Phase is 32.32 fixed point**, not float. f32 loses fractional resolution past
2^24 samples, audible as detuning on long samples, and f64 costs real throughput
on consumer NVIDIA parts. WGSL has no portable u64, so the device holds it as
two u32 lanes.

**The voice pool is structure-of-arrays** inside a single storage buffer: field
`f` of voice `i` at `f * capacity + i`. It is re-sorted by sample region every
block during compaction so neighbouring invocations hit the same cache lines.
This workload is memory-bandwidth bound -- roughly 20 flops per voice per frame
against one scattered global read -- so the access pattern is worth far more
than the arithmetic.

**Note-offs do not search the pool.** The pool is reordered every block, so a
voice cannot be found by index. The lookup is inverted: the host publishes a
table of how many note-offs each (channel, key) has seen, and each voice
compares its own ordinal against it.

## Known limitations

- **No effects.** No reverb, no chorus. CC91, CC93, CC94 and CC95 are
  recognised and do nothing. `kestrel info` on a MIDI marks the unimplemented
  controllers `MISSING` and reports what share of the file's controller events
  fall in that bucket.
- **Note boundaries can click on presets with no attack or release.** Kestrel
  starts and stops a voice on its exact frame, where other synths apply a few
  milliseconds of fade. On a preset whose volume envelope is all defaults over a
  looped waveform -- a synth lead or a square-wave bass -- every note-on and
  note-off is then a step discontinuity, which reads as grit over the tone. It
  is the largest known audible gap and is next.
- **No realtime playback**, and **no host integration**: offline rendering to
  WAV only, no C ABI and no plugin build.
- **Notes can keep sounding after the music stops, on some soundfonts.**
  Reported on a 44.7M-note file, from roughly two thirds of the way in. It
  reproduces at every pool size and on one soundfont but not another; the
  suspicion is note-offs arriving while the sustain pedal is down being deferred
  and never released. Not root-caused. If you hear it, `--limiter off` and a
  drier soundfont will tell you quickly whether it is the same thing.
- **SFZ support is almost complete.** The common opcodes work, including
  `#include`, `#define`, velocity layers, `lorand`/`hirand`, velocity
  crossfade, `fil_veltrack`, `amp_veltrack`, `key`, `loopstart`/`loopend`,
  `pitch_keytrack` and `off_by`. Per-voice LFOs (`amplfo_*`, `fillfo_*`,
  `pitchlfo_*`) are recognised and reported, not applied -- note that an LFO
  with a rate and no `*_depth` opcode would do nothing anyway, which is the
  common case in piano libraries. `note_selfmask` and the per-stage envelope
  velocity-tracking opcodes are also read and dropped. Kestrel warns about every
  opcode it does not apply, so you will not hit this silently.
- **The SF2 modulation LFO's filter destination** (`modLfoToFilterFc`) is not
  implemented. Its pitch and volume destinations are.
- **NaN checking is off in release builds** unless you pass `--nan-guard`.

## Planned for the next release

**A de-click fade at note boundaries** is the priority -- see *Known
limitations*. It is measured and understood; what is left is calibrating the
length and deciding whether it clamps the envelope or sits beside it.

**SFZ per-voice LFOs** are the largest remaining gap in SFZ support. Effects, a
host C ABI and realtime playback remain out of scope.

## License

MIT -- see [LICENSE](LICENSE).

`src/limiter.rs` is a port of the realtime limiter OmniConverter ships, which
came originally from Kiva; the `--limiter omni` mode is named after it. The SF2
and SFZ loaders, and the RIFF/WAVE reader and writer, were written against the
published format specifications. FLAC samples are decoded by
[claxon](https://github.com/ruuda/claxon) (Apache-2.0), which is the one format
here not read by hand. Everything else was written for this project.
