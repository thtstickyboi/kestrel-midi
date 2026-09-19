# Kestrel

'*experimental*'

A GPU-accelerated SoundFont/SFZ synthesizer for Black MIDI.

Kestrel renders a MIDI file offline to WAV, FLAC, Opus, MP3, OGG or M4A, doing the synthesis in compute shaders instead of on the CPU. It is built for files where the note count is absurd -- hundreds of millions of notes, with a hundred thousand to several million voices sounding at once -- which is the point where conventional synths stop being fast enough.

On an RTX 5060 Laptop it sustains **1,048,576 concurrent voices at 0.93x realtime**, and renders a 44.7M-note file against a 22,528-region piano at 1.79x realtime.

Double-click it and a guided renderer walks you through a render in the terminal. Behind that it is a command-line tool, an API other programs' GUIs drive it through, and a Rust library. Kestrel itself has no windowed GUI, no realtime playback, and no plugin build.

---

## Read this first: Kestrel is 100% vibecoded

**Every line of code in this repository was written by an AI.** Not AI-assisted, not AI-reviewed -- AI-written, start to finish, from the shaders to the SF2 parser to the limiter. I directed the work, made the design calls, ran the renders and decided what was acceptable, but I did not hand-write this code and I have not read every line of it.

- **No human has line-by-line reviewed this code.** Not me, not anyone. If you are the sort of person who reads a diff before running it, read this one.
- **It was developed against a test suite that is not in this repository:** null tests against a separate single-threaded CPU reference, byte-for-byte determinism checks, phase-accumulator precision tests, envelope curves checked against analytically computed ones, and timing and LFO behaviour measured against BASSMIDI. The GPU path matches the CPU reference to better than -98 dB. That is real verification and it caught real bugs -- but you are taking my word for it, because the evidence is not here.
- **The corners are where it will break.** SF2 loading, the render loop and voice stealing under saturation are exercised constantly and are in decent shape. Unusual soundfonts, exotic SFZ opcodes and malformed MIDI are much less certain.
- **Check the output yourself.** `kestrel null` exists precisely so you can diff a render against a reference you trust.

I am not claiming this is production software. I am claiming it renders black MIDI fast and the output sounded right to me. Please open an issue when it breaks, because it will.

---

## Getting Kestrel

### Download

Ready-made builds are attached to each [release](https://github.com/thtstickyboi/kestrel-midi/releases):

| File | For | Status |
|---|---|---|
| `kestrel-<version>-windows-x64.zip` | Windows 10/11, 64-bit | Tested |
| `kestrel-<version>-macos-universal.dmg` | macOS 11 or newer, Apple silicon and Intel | Built, **never run** |
| `kestrel-<version>-linux-x64.tar.gz` | 64-bit Linux with glibc 2.35 or newer (Ubuntu 22.04+) | Built, **never run** |

`SHA256SUMS.txt` beside them holds each file's checksum.

- **Windows:** unzip, double-click `kestrel.exe`. The executable is not code-signed, so SmartScreen may warn about it: **More info -> Run anyway**.
- **macOS:** open the dmg and copy `kestrel` out of it. It is not signed or notarised, so macOS refuses it the first time: go to **System Settings -> Privacy & Security** and press **Open Anyway**, or run `xattr -dr com.apple.quarantine kestrel` in Terminal. Double-clicking it opens it in Terminal.
- **Linux:** `tar xzf` the archive and run `./kestrel` from a terminal. It needs a Vulkan driver. The file pickers go through the desktop portal (`xdg-desktop-portal`), which most desktops already run.

The macOS and Linux builds are compiled by GitHub Actions and nobody has run them. If you are first on one, please open an issue either way -- a report that it simply worked is as useful as a crash.

### Building from source

| | |
|---|---|
| **Rust** | 1.88 or newer, from [rustup.rs](https://rustup.rs) |

```bash
git clone https://github.com/thtstickyboi/kestrel-midi.git
cd kestrel-midi && cargo build --release
```

Build in **release mode**. The debug build is not merely slower, it is unusable for real files. The binary lands at `target/release/kestrel` and is self-contained: no system libraries, and the shaders are compiled into it. On Windows the build downloads Microsoft's DXC shader compiler once and links it in, for the DX12 backend.

That is the same build as the downloads. **Dev mode** adds the options for measuring and debugging the engine -- tuning the GPU passes, switching back to older behaviour for comparison, per-block diagnostics -- and the `null` command:

```bash
cargo build --release --features dev
```

A dev build says `(dev)` after its version and lists those options under *Developer options* in `render --help`. At the same settings it renders exactly what the release build does.

## Requirements

| | |
|---|---|
| **GPU** | Anything with a working Vulkan, DX12 or Metal driver |
| **VRAM** | Depends on the soundfont and `--max-voices`; see *Memory*. Tuned against 8 GB |
| **OS** | Windows, Linux, macOS. Only Windows has been *run* |
| **ffmpeg** | Only for output other than WAV; see *Output formats* |

**Tested on NVIDIA and Intel, both on Windows via Vulkan.** Rendering the same file on an RTX 5060 and on an Intel iGPU nulls at **-111 dB peak**, and each device is byte-identical with itself across runs. WGSL is compiled by your driver rather than by cargo, so two vendors' shader compilers agreeing is real evidence the shaders are portable. AMD and Apple silicon are unrun.

Integrated GPUs are slower but usable: an Intel part held 2.95x realtime against the 5060's 32.5x on the same 100k-note file. Kestrel prefers a discrete GPU when both are present.

There is a **complete CPU backend** -- the reference implementation the GPU path was built against. It is far slower and single-threaded. Use `--backend cpu` if you have no usable GPU, or to check a suspicious render against something simpler.

### Memory

Kestrel requests whatever limits your adapter reports and checks them before allocating. If the render pass needs more workgroup storage than your card offers, or `--max-voices` implies a pool larger than your card will bind, it stops and names the flag to lower. It does not degrade quietly.

**VRAM is usually the binding constraint.** The startup line prints the breakdown:

```
gpu: NVIDIA GeForce RTX 5060 Laptop GPU (Vulkan) | 1107.0 MiB of device buffers
(757.1 MiB sample pool, 250.0 MiB voice pool for 1310720 voices, 64.0 MiB partials)
```

Three things allocate: the **voice pool**, sized by `--max-voices` and scaling linearly (250 MiB at the 1,048,576 default); the **partial buffers** used by the mixdown, 64 MiB at the defaults; and the **sample pool**, your soundfont resampled to one rate, which for a large multi-sampled piano runs to hundreds of megabytes. If the sample pool will not fit within `--pool-budget` (2 GB by default) Kestrel halves its sample rate until it does, on the grounds that a downsampled render beats no render.

The voice pool is one storage buffer, and every adapter caps how much of a single buffer a shader may bind. That is the ceiling on `--max-voices`: **16,519,104** on an RTX 5060 at the default `--steal-percent`, 8,259,552 on an Intel RaptorLake-S iGPU. `kestrel --force-cli gpu-info` prints the figure for *your* card, and so does the guided renderer's first screen. The cap is on the pool, not on the piece -- a file with more simultaneous notes than the pool holds still renders, with the excess stolen and dropped.

**Host RAM** is mostly the soundfont -- Kestrel keeps its own copy of the sample pool -- plus the note-ons of the block being admitted. Admission cannot choose until a block's last note-on is in, so the note-ons inside one block are held until it ends, and that scales with `--block`. It is capped: past **134,217,728 candidates** in one block (about 4.5 GiB) the block is thinned evenly across its length rather than held whole, so even a block carrying billions of note-ons fits in memory. Below the cap nothing is thinned. If a render still dies on a failed allocation rather than a GPU error, lower `--block` (128 is the floor). Lowering `--max-voices` does *not* help, because the block is considered in full before admission thins it.

## Using Kestrel

### The guided renderer

Start `kestrel` with no arguments -- double-clicking it does exactly that -- and it walks you through a render:

1. **Environment check.** Every GPU Kestrel can use, the most voices each will hold, whether ffmpeg was found, and whether a newer Kestrel is out.
2. **Menu.** Render a MIDI; per-track render (coming soon); Extras; exit.
3. **MIDI and soundfonts**, through your system's own file pickers, from as many folders as you like. At most two soundfonts; if one of them is a General MIDI bank it goes underneath the other.
4. **Voices, format and destination folder.** A name that is already taken gets the date and time added rather than being overwritten.
5. **Additional flags**, one free-form line for anything the steps do not ask about -- `--limiter omni`, `--seconds 30`, `--volume 80`.
6. **Progress**, with speed, voices, RAM and VRAM, then a summary and the option to start another render.

**Extras** opens in a window of its own: GPU info, file info, the null test, help for every flag, the update ring, and the folders Kestrel remembers.

**Updates.** At startup the guided renderer asks GitHub for the latest release and tells you if there is a newer one. It never downloads or installs anything. In Extras, the **Fast Ring** (default) tells you about every release and the **Slow Ring** only about feature releases such as 1.2.0 or 2.0.0. Set `KESTREL_NO_UPDATE_CHECK=1` to turn the check off. The command line never checks unless you ask it to with `check-update`.

**`ktrl.ini`**, next to the executable, keeps the update ring and the folders the MIDI, soundfont and destination pickers last opened. Delete it to reset them. If Kestrel's folder is read-only, nothing is remembered.

A guided render is parsed through the same definition as the command line and runs through the same pipeline, so it writes a file byte-identical to the equivalent `--force-cli render`.

### The command line

**Every command-line use needs `--force-cli`.** Without it, any arguments print the command again with the flag added and exit with code 2 -- so an old script fails loudly rather than opening a menu. `--help` and `--version` work without it.

```bash
kestrel --force-cli render input.mid -s soundfont.sfz -o output.wav
```

`-s` takes a `.sf2` or a `.sfz`, and may be **repeated to layer** -- see *General MIDI*. WAV output is 32-bit float by default; `--format pcm16` for 16-bit.

An SFZ library's samples may be WAV or FLAC. Which one is decided by reading the file rather than its extension, because libraries ship FLAC under names of their own invention and dispatching on the name silently drops every region that uses one.

### Output formats

The extension of `-o` picks the container. No other flag is needed:

| Extension | Encoded as |
|---|---|
| `.wav` | written directly, 32-bit float or 16-bit PCM |
| `.flac` | FLAC level 8, 24-bit |
| `.opus` | libopus, 160 kbps VBR |
| `.mp3` | LAME V0 |
| `.ogg` | Vorbis, quality 8 |
| `.m4a` | ffmpeg's native AAC, 256 kbps |

Everything but `.wav` is piped through **ffmpeg**, which Kestrel looks for in this order: `--ffmpeg <path>`, the `FFMPEG` environment variable, beside the Kestrel executable (or in an `ffmpeg/` folder there), then `PATH`. A path you name that is not a working ffmpeg is an error, never a quiet fall-through to another one.

- **Lossy containers default to a -1 dBFS ceiling.** A lossy decoder does not reproduce peak level: a mix limited to exactly full scale decoded from Opus at +0.99 dBFS, and its clipped samples are heard as high-frequency clicks. WAV and FLAC keep 0 dBFS. `--ceiling-db` given explicitly always wins.
- **`--limiter off` and `--format pcm16` are refused for encoded output.** ffmpeg clamps anything over full scale without a word, and a raw black MIDI mix peaks hundreds of times over it. `--limiter off` to `.wav` still works, for stems.
- **FLAC of a float render is lossless relative to 24-bit**, not to the float mix.
- The extension is checked before the soundfont loads. A typo gets a suggestion (`.opis` -> *Did you mean .opus?*), and a real container Kestrel does not write (`.aiff`, `.wma`) is named as one.

**Getting ffmpeg.**

```bash
kestrel --force-cli ffmpeg-info       # which ffmpeg would be used, from where, and what it can encode
kestrel --force-cli get-ffmpeg        # download one into ffmpeg/ beside Kestrel
```

`ffmpeg-info` exits non-zero if ffmpeg or an encoder a format needs is missing, so it doubles as a setup check. `get-ffmpeg` installs nothing system-wide and never touches `PATH` -- undoing it is deleting that folder. It prints the URL, the origin and the licence (ffmpeg builds are GPL) before fetching, checks the archive's SHA-256 before extracting anything, and **fails closed**: no digest is pinned in the source, because the vendors' "latest" archives change with every upstream release, so the first run prints the digest it saw and refuses. Check it against the vendor's published checksum and run again with `--accept-hash <SHA256>`. `--dry-run` shows the plan without downloading. It has been run end to end on Windows; its Linux and Intel macOS sources have not been used, and on Apple silicon `brew install ffmpeg` is the way. `winget install Gyan.FFmpeg` and `sudo apt install ffmpeg` work too.

### The flags that matter

| Flag | Default | What it does |
|---|---|---|
| `--backend` | `gpu` | `cpu` is the reference implementation: correct, slow, single-threaded. |
| `--max-voices N` | `1048576` | Ceiling on simultaneous voices. The single biggest lever on both VRAM and speed. The maximum is whatever your card will bind; `gpu-info` prints yours. |
| `--block N` | `4096` | Frames per render block. Also sets how many note-ons are held in host RAM at once. |
| `--interp` | `linear` | `nearest`, `linear` or `cubic`. Cubic reads twice as many samples per voice. |
| `--seconds N` | off | Stop after N seconds of output. Use this constantly while experimenting. |
| `--volume P` | `100` | Volume as a percentage, 0 to 200, applied before the limiter. See below. |
| `--limiter` | `brickwall` | `brickwall`, `omni` or `off`. See *Limiting*. |
| `--ceiling-db` | `0`, or `-1` for lossy | The brickwall's ceiling in dBFS. |
| `--steal-percent` | `25` | How much of the pool one block may replace. |
| `--min-velocity N` | `0` | Skip every note quieter than velocity N, as if the file did not contain it. On black MIDI full of ghost notes this can make a render many times faster. |
| `--note-grid` | off | Hold notes to BASSMIDI's 4 ms envelope grid, so notes shorter than 4 ms still sound. Only for files that rely on it; see *Matching BASSMIDI*. |
| `--format` | `float32` | `float32` or `pcm16`, for WAV. |
| `--ffmpeg PATH` | found automatically | Use this ffmpeg for encoded output. |
| `--nan-guard` | off | Check every block for NaN and Inf. **Off by default**, so a clean exit is not by itself proof the WAV is finite. |
| `--profile` | off | Per-pass GPU timings and the host/device split, once per wall-clock second. |
| `--progress json` | off | Machine-readable progress for one render; see *Building a GUI on Kestrel*. |
| `--gpu-backend NAME` | automatic | Which graphics API: `vulkan`, `dx12`, `metal` or `gl`. Left alone, Kestrel prefers Vulkan, which is also the faster one on Windows. |
| `--gpu-adapter TEXT` | automatic | Which card, by any part of its name, ignoring case: `--gpu-adapter intel`, `--gpu-adapter 5060`. `gpu-info` lists the names. In the guided renderer, `--adapter N` picks card N from its numbered list instead. |

**`--volume` is a percentage** on the same linear scale as OmniConverter's: `100` changes nothing, `50` is half (-6 dB), `0` is silent, `200` the most. Before 1.1.0 it was a plain gain, so `--volume 0.5` in an old script now means half a percent -- values above 0 and up to 2 render with a warning for that reason. A dense mix sits far above full scale and the limiter holds it at the ceiling, so lowering the volume eases the limiting more than it quietens the file; `--ceiling-db` is what sets how loud the file can get.

`kestrel --force-cli render --help` lists everything. The engine's tuning and debugging options are in dev builds only; see *Building from source*.

### The other commands

```bash
kestrel --force-cli api                   # a session another program drives; see API.md
kestrel --force-cli gpu-info              # adapters, limits, and each one's --max-voices ceiling
kestrel --force-cli info file.sf2         # what the loader made of a soundfont
kestrel --force-cli info file.mid         # ...or of a MIDI: note counts, CC usage, tempo, density
kestrel --force-cli ffmpeg-info           # the ffmpeg encoded output would use
kestrel --force-cli get-ffmpeg            # fetch one; see Output formats
kestrel --force-cli check-update          # is a newer Kestrel out? downloads nothing
```

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

### Building a GUI on Kestrel

```bash
kestrel --force-cli api
```

starts a session another program drives, one JSON object per line each way over stdin and stdout, so a GUI in any language can control Kestrel without parsing its log text. A session lists GPUs, ffmpeg and every render option with its real default; checks and scans MIDIs; loads soundfonts and keeps them loaded between renders; and runs renders with live telemetry (phase, progress, notes, voices, stolen and dropped, speed, time left, RAM and VRAM), snapshots on request, and a cancel that leaves a properly closed file. A render asked for this way writes exactly the file the equivalent command writes. **[API.md](API.md)** is the protocol, with a Python example.

To watch a single render instead, add `--progress json` to a render command: the same telemetry on stdout, with `{"interval_ms": N}`, `{"snapshot": true}` and `{"cancel": true}` read from stdin.

From Rust, the same pipeline is `kestrel::session`: a `Job`, `plan` and `run`, and a `Monitor` that any number of readers take snapshots from at their own rate.

## General MIDI

**Rudimentary.** An ordinary GM file will play, but this is not guaranteed to render a file the way a GM synth would. Treat it as usable and unfinished, not as a supported format. What is listed below is what was built and checked; what is under *Where it is still wrong* is what is known to be off, and there is very likely more that has not been found yet.

**Soundfonts layer.** `-s` is repeatable and merges in order, each one replacing anything already at the same bank and program:

```bash
kestrel --force-cli render gm.mid -s general-midi.sf2 -s piano.sfz --sf-programs 0-1 -o out.wav
```

`--sf-programs` places the *last* soundfont on the programs you name, ranges included, so a one-preset SFZ piano can cover the whole piano family. Without it a soundfont takes only the program it declares -- which for an `.sfz` is program 0, GM's Acoustic Grand, so the common case needs no flag.

Implemented: program change, bank select with the General MIDI fallback to bank 0, drum kits in bank 128 with their own fallback ladder, channel volume (powering on at 100, as General MIDI specifies), expression, pan, the sustain and sostenuto pedals, pitch bend, portamento (CC5, CC65 and CC84), RPN 0/1/2 (bend range, fine and coarse tuning), CC71-75, CC120, CC121 and CC123-127. SF2 vibrato and tremolo LFOs are applied, as is the SF2 modulation envelope with both its pitch and filter destinations.

### Where it is still wrong

Measured against a reference GM synth. None of these stops a render; all of them mean it will not match what you are used to.

- **Several channels come out noticeably brighter** than the reference -- 1.2 to 1.4 times its spectral centroid. Ruled out as causes: the lowpass cutoff (measured exact at four settings), interpolation, and the envelopes. Cause unknown.
- **The sample pool is resampled to one rate at load**, which band-limits a soundfont whose samples are recorded lower. On a 22 kHz General MIDI set that removes everything above 11 kHz, where the reference has content there. `--no-resample-pool` keeps the samples at their own rate and closes it.
- **Only three SysEx messages are acted on**: the GS *use for rhythm part*, GS Reset and GM System On/Off. Master volume, part parameters and every bulk dump are ignored.
- **Effects are not implemented**, so CC91 reverb and CC93 chorus sends do nothing -- see *Known limitations*.
- **`modLfoToFilterFc` is not implemented.**
- **Rapid CC11 gating reaches digital silence** where the reference does not, because channel volume is not smoothed over as long a window.
- The note-boundary click under *Known limitations* hits GM presets hardest, because synth-lead and wavetable patches are exactly the ones with no attack or release of their own.

## Matching BASSMIDI

Where Kestrel and BASSMIDI disagree, BASSMIDI is taken as the reference, and these were measured against it directly:

- **Tempo changes.** BASSMIDI advances a whole output sample at a time, fires an event -- a tempo change included -- on the first sample whose tick position has reached it, and carries the ticks that ran past a tempo change into the new tempo. Kestrel does the same, and agrees with it to 0.000 ms across 57 sections of back-to-back tempo changes. At ordinary tempos a sample is a fraction of a tick and nothing can be heard; in bursts of changes in the thousands of bpm it adds up to tens of milliseconds.
- **Channel volume** powers on at 100, not 127.
- **LFOs.** SFZ's amplitude and pitch LFOs were measured against BASSMIDI's: a triangle starting at zero, delayed from the note's own start, at the depth, rate and direction it plays them. SF2's run on the same oscillator. Every LFO and modulation envelope counts from its note's own start -- before 1.1.0 they could run up to a block early. They update every 32 frames.
- **Short notes, with `--note-grid`.** BASSMIDI moves each note's envelope in 4 ms steps counted from its note-on, so a note-off only takes effect at the next step, and a release shorter than a step becomes a 4 ms fade. A note one sample long still sounds for 4 ms. With `--note-grid` Kestrel does the same, matched to the sample; without it a note ends on its own note-off, as before. It is off by default because almost no file notices and it costs about 12% of a render, but a file that encodes audio as one-sample notes renders silent without it.
- **Portamento.** CC65 turns it on, CC5 sets the speed and CC84 names the next note's starting key, as in BASSMIDI: a note glides in a straight line in pitch from the channel's last note, and CC5 = 0, the power-on value, means no glide. Kestrel's glides match BASSMIDI's speed to within 0.03%. Mono mode (CC126) is not implemented, so a note struck over a held one glides but does not cut it.

## Limiting

Black MIDI mixes clip constantly -- a saturated section can peak at hundreds of times full scale -- so what happens at the ceiling matters more than usual.

**`brickwall`** (default) is a lookahead true-peak limiter. It sees peaks before they arrive and cannot exceed its ceiling, so nothing downstream ever has to hard-clip. `--ceiling-db` sets the ceiling. Its 2 ms lookahead is also the render latency, which for an offline renderer costs nothing.

**`omni`** is a port of the realtime limiter OmniConverter ships, originally from Kiva. It is a feedback follower: it sees a peak only once the peak has passed, which is why it needs a third of a second of release, and that release audibly drags the level down after every loud moment. It does not bound its own output, so since 1.1.0 the brickwall runs behind it as a safety stage at the render's ceiling: wherever omni stays under the ceiling the output is omni's, delayed by the lookahead, and nothing clips. **Keep it only for level-matching a render against BASS or XSynth.**

**`off`** disables limiting. Your mix will clip. Refused for encoded output.

## Admission and voice stealing

Stealing decides who dies. **Admission decides who never starts**, and on a saturated file that is the bigger number by an order of magnitude -- on a 44.7M-note file against a 32,767-voice pool, 83.7M note-ons were dropped against 5.7M voices stolen.

How much of that you face is a function of pool size, and the pool can be far larger than the default. On that same file, against a 22,528-region piano:

| `--max-voices` | dropped | stolen | realtime |
|---|---|---|---|
| 1,048,576 (default) | 12,077,835 | 49,667,330 | 1.79x |
| 4,194,304 | 511,768 | 33,624,100 | 0.94x |
| 13,421,568 | **0** | **0** | 0.88x |

At its busiest the file has 5,338,314 voices sounding at once. With 13,421,568 slots every note sounds, none refused and none cut short, for 4.3 GiB of device buffers. Admission stops mattering well before stealing does, and past what the file needs a bigger pool costs memory rather than time: 16,519,104 voices renders in the same 148 s.

**Admission** ranks an oversubscribed block by opening amplitude and keeps the loudest. Thinning the block evenly instead, which a dev build offers as `--admit even` for comparison, measurably loses at small pools -- at 32,767 voices, -10.065 dB RMS against `even`'s -11.225 at an identical peak, which is what "the long loud notes survived" looks like as a number -- and the gap narrows as the pool grows, to within 0.01 dB at the default. There is no duration term: ranking is purely by opening amplitude, so a loud note cut short and a loud note held look the same.

**Stealing** kills the voices with the lowest envelope level, ties broken by note id. Killing the earliest-started instead sounds backwards under saturation and is: the oldest voices are the mature, sounding ones. (A dev build keeps `--steal oldest` and `drop-new`, which refuses the incoming note, for comparison.)

**`--steal-percent`** (25) caps how much of the pool one block may replace. Unbounded, a block whose note-ons outnumber the pool replaces *every* voice, so no voice outlives the block it was born in and a saturated passage renders as a stream of 85 ms attack fragments instead of notes.

## How it works

One command buffer per audio block, five compute passes: **steal**, **spawn**, **render**, **reduce**, **compact**. The host parses MIDI and resolves presets, which is branchy table-lookup work that stays on the CPU permanently; the device does everything per-voice and per-sample. The host builds the next block while the device renders the current one, rather than waiting for it.

**Renders are byte-identical.** Two runs of the same file with the same settings produce bitwise identical output. Nothing in the render path uses a floating-point atomic, because float atomics are non-deterministic in ordering: the mixdown is a fixed-order tree reduction into per-workgroup partial buffers no other workgroup touches, voice slots are assigned by index rather than from an atomic counter, and the pool sort is a stable radix sort. The only atomics in the codebase are integer histogram bins. (An encoded file is byte-identical for the same ffmpeg build; a different ffmpeg version may encode the same audio differently.)

**Phase is 32.32 fixed point**, not float. f32 loses fractional resolution past 2^24 samples, audible as detuning on long samples, and f64 costs real throughput on consumer NVIDIA parts. WGSL has no portable u64, so the device holds it as two u32 lanes.

**The voice pool is structure-of-arrays** inside a single storage buffer: field `f` of voice `i` at `f * capacity + i`. It is re-sorted by sample region every block during compaction so neighbouring invocations hit the same cache lines. That sort is what keeps the sample fetches cheap: measured through a whole render, the GPU's memory controller sits around 3% busy, so the time goes to compute and to keeping the device fed, not to memory bandwidth.

**Note-offs do not search the pool.** The pool is reordered every block, so a voice cannot be found by index. The lookup is inverted: the host publishes, per (channel, key), how many note-offs it has seen and on which frames, as runs of the same frame, and each voice finds its own release frame by binary search over them. However many note-offs a block carries, the table stays within about 67 MB at the defaults.

## Known limitations

- **No effects.** No reverb, no chorus. CC91, CC93, CC94 and CC95 are recognised and do nothing. `kestrel info` on a MIDI marks the unimplemented controllers `MISSING` and reports what share of the file's controller events fall in that bucket.
- **Note boundaries can click on presets with no attack or release.** Kestrel starts and stops a voice on its exact frame, where other synths apply a few milliseconds of fade. On a preset whose volume envelope is all defaults over a looped waveform -- a synth lead or a square-wave bass -- every note-on and note-off is then a step discontinuity, which reads as grit over the tone. It is the largest known audible gap.
- **No realtime playback**, and **no host integration**: offline rendering only, no C ABI and no plugin build.
- **A render once failed with `buffer map failed: BufferAsyncError`.** Not reproduced. If you see it, the voice count, the MIDI and the soundfont are what would pin it down.
- **On the densest files the host is the bottleneck.** MIDI reading and admission run on one CPU core, and a block carrying around a billion note-ons takes minutes of host work while the GPU waits.
- **SFZ support is almost complete.** The common opcodes work, including `#include`, `#define`, velocity layers, `lorand`/`hirand`, velocity crossfade, `fil_veltrack`, `amp_veltrack`, `key`, `loopstart`/`loopend`, `pitch_keytrack`, `off_by`, and the amplitude and pitch LFOs (`amplfo_*`, `pitchlfo_*`). Not applied: `fillfo_depth`, the LFO `*_fade` opcodes (BASSMIDI ramps the depth in over them; Kestrel applies it at once), LFOs driven by a controller, SFZ v2's `lfoN_*`, `note_selfmask`, and the per-stage envelope velocity-tracking opcodes. Kestrel warns about every opcode it does not apply, so you will not hit this silently.
- **The SF2 modulation LFO's filter destination** (`modLfoToFilterFc`) is not implemented. Its pitch and volume destinations are.
- **VRAM figures** in the progress screen and the JSON feed are Windows only.
- **NaN checking is off** unless you pass `--nan-guard`.

## Planned

**A de-click fade at note boundaries** -- see *Known limitations*. It is measured and understood; what is left is calibrating the length and deciding whether it clamps the envelope or sits beside it.

**A faster host.** The single-core host path first, then spreading it across cores, which is also what **per-track rendering** needs; the guided renderer already lists it as coming soon.

**A log file per render**, so a failure leaves its settings and its point of failure behind.

Effects, a host C ABI and realtime playback remain out of scope.

## License

Kestrel is licensed under the **Mozilla Public License 2.0** -- see [LICENSE](LICENSE).

Code that was not written for this project is listed in [THIRD-PARTY.md](THIRD-PARTY.md). In short: `src/limiter.rs` is a port of the realtime limiter OmniConverter ships, which came originally from Kiva, and the `--limiter omni` mode is named after it. FLAC samples are decoded by [claxon](https://github.com/ruuda/claxon) (Apache-2.0), which is the one format here not read by hand. The SF2 and SFZ loaders and the RIFF/WAVE reader and writer were written against the published format specifications. ffmpeg is never bundled; `get-ffmpeg` fetches it only when asked, and says it is GPL first.
