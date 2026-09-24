# Kestrel

'*experimental*'

A GPU-accelerated SoundFont/SFZ synthesizer for Black MIDI.

Kestrel renders a MIDI file offline to WAV, FLAC, Opus, MP3, OGG or M4A, doing the synthesis in compute shaders instead of on the CPU. It is built for files where the note count is absurd -- hundreds of millions of notes, with a hundred thousand to several million voices sounding at once -- which is the point where conventional synths stop being fast enough. It can also render every track of a MIDI on its own, as stems or summed into one file.

On an RTX 5060 Laptop it sustains **1,048,576 concurrent voices at 0.93x realtime**, and renders a 44.7M-note file against a 22,528-region piano at 1.87x realtime.

Double-click it and a guided renderer walks you through a render in the terminal. Behind that it is a command-line tool, an API other programs' GUIs drive it through, and a Rust library. Kestrel itself has no windowed GUI, no realtime playback, and no plugin build.

How it works inside, and the measurements behind it, are in **[DESIGN.md](https://github.com/thtstickyboi/kestrel-midi/blob/main/DESIGN.md)**.

---

## Read this first: Kestrel is 100% vibecoded

**Every line of code in this repository was written by an AI.** Not AI-assisted, not AI-reviewed -- AI-written, start to finish, from the shaders to the SF2 parser to the limiter. I directed the work, made the design calls, ran the renders and decided what was acceptable, but I did not hand-write this code and I have not read every line of it.

- **No human has line-by-line reviewed this code.** Not me, not anyone. If you are the sort of person who reads a diff before running it, read this one.
- **It was developed against a test suite that is not in this repository:** null tests against a separate single-threaded CPU reference, byte-for-byte determinism checks, phase-accumulator precision tests, envelope curves checked against analytically computed ones, and timing and LFO behaviour measured against BASSMIDI. The GPU path matches the CPU reference to better than -98 dB. That is real verification and it caught real bugs -- but you are taking my word for it, because the evidence is not here.
- **The corners are where it will break.** SF2 loading, the render loop and voice stealing under saturation are exercised constantly and are in decent shape. Unusual soundfonts, exotic SFZ opcodes and malformed MIDI are much less certain.
- **Check the output yourself.** `kestrel null`, in a dev build, exists precisely so you can diff a render against a reference you trust.

I am not claiming this is production software. I am claiming it renders black MIDI fast and the output sounded right to me. Please open an issue when it breaks, because it will.

---

## Getting Kestrel

Ready-made builds are attached to each [release](https://github.com/thtstickyboi/kestrel-midi/releases):

| File | For | Status |
|---|---|---|
| `kestrel-<version>-windows-x64.zip` | Windows 10/11, 64-bit | Tested |
| `kestrel-<version>-macos-universal.dmg` | macOS 11 or newer, Apple silicon and Intel | Built, **never run** |
| `kestrel-<version>-linux-x64.tar.gz` | 64-bit Linux with glibc 2.35 or newer (Ubuntu 22.04+) | Built, **never run** |

- **Windows:** unzip, double-click `kestrel.exe`. The executable is not code-signed, so SmartScreen may warn about it: **More info -> Run anyway**.
- **macOS:** open the dmg and copy `kestrel` out of it. It is not signed or notarised, so macOS refuses it the first time: go to **System Settings -> Privacy & Security** and press **Open Anyway**, or run `xattr -dr com.apple.quarantine kestrel` in Terminal.
- **Linux:** `tar xzf` the archive and run `./kestrel` from a terminal. It needs a Vulkan driver, and the file pickers go through the desktop portal (`xdg-desktop-portal`).

The macOS and Linux builds are compiled by GitHub Actions and nobody has run them. If you are first on one, please open an issue either way. `SHA256SUMS.txt` beside the downloads holds each file's checksum.

**Building from source** needs Rust 1.88 or newer from [rustup.rs](https://rustup.rs):

```bash
git clone https://github.com/thtstickyboi/kestrel-midi.git
cd kestrel-midi && cargo build --release
```

Build in **release mode**; the debug build is unusable for real files. That is the same build as the downloads. `cargo build --release --features dev` is **dev mode**, which adds the options for measuring and debugging the engine and the `null` command, and renders exactly what the release build does at the same settings.

## Requirements

| | |
|---|---|
| **GPU** | Anything with a working Vulkan, DX12 or Metal driver |
| **VRAM** | Depends on the soundfont and `--max-voices`. Tuned against 8 GB |
| **OS** | Windows, Linux, macOS. Only Windows has been *run* |
| **ffmpeg** | Only for output other than WAV |

Tested on NVIDIA and Intel GPUs on Windows. Integrated GPUs are slower but usable, and Kestrel prefers a discrete GPU when both are present. `--backend cpu` renders on the CPU reference instead, far slower, if you have no usable GPU.

**Memory.** VRAM is usually the limit: the voice pool (250 MiB at the default 1,048,576 voices) plus your soundfont's samples. The most voices your card holds is printed by `kestrel --force-cli gpu-info` and on the guided renderer's first screen: 16,519,104 on an RTX 5060. If a soundfont will not fit in `--pool-budget` (2 GB), it is downsampled until it does. Kestrel checks every limit before allocating and names the flag to lower rather than failing mid-render. The details are in DESIGN.md.

## Using Kestrel

### The guided renderer

Start `kestrel` with no arguments -- double-clicking it does exactly that -- and it walks you through a render:

1. **Environment check.** Every GPU Kestrel can use, the most voices each will hold, whether ffmpeg was found, and whether a newer Kestrel is out.
2. **Menu.** Render a MIDI; per-track render; Extras; exit.
3. **MIDI and soundfonts**, through your system's own file pickers. At most two soundfonts; a General MIDI bank goes underneath the other.
4. **Voices, format and destination folder.** A name already taken gets the date and time added rather than being overwritten.
5. **Additional flags**, one free-form line for anything the steps do not ask about -- `--seconds 30`, `--volume 80`, `--min-velocity 10`.
6. **Progress**, then a summary with the flags the render ran with, and the option to start another.

A guided render writes a file byte-identical to the equivalent `--force-cli render`. **Extras** has GPU info, file info, help for every flag, the update ring and the remembered folders.

**Updates.** At startup the guided renderer asks GitHub whether a newer release is out. It never downloads or installs anything. The **Fast Ring** (default) tells you about every release and the **Slow Ring** only about feature releases; set `KESTREL_NO_UPDATE_CHECK=1` to turn the check off. `ktrl.ini`, next to the executable, keeps the ring and the folders the pickers last opened.

### The command line

**Every command-line use needs `--force-cli`**; `--help` and `--version` work without it.

```bash
kestrel --force-cli render input.mid -s soundfont.sfz -o output.wav
```

`-s` takes a `.sf2` or a `.sfz` (with WAV or FLAC samples), and may be **repeated to layer** -- see *General MIDI*. WAV output is 32-bit float by default; `--format pcm16` for 16-bit.

**The extension of `-o` picks the format:** `.wav` is written directly; `.flac` (24-bit), `.opus`, `.mp3`, `.ogg` and `.m4a` go through **ffmpeg**, found through `--ffmpeg`, the `FFMPEG` environment variable, beside Kestrel, or `PATH`. `kestrel --force-cli ffmpeg-info` says which ffmpeg would be used, and `get-ffmpeg` downloads one into a folder beside Kestrel, showing its licence and checking its checksum first. Lossy formats default to a -1 dBFS ceiling, because a lossy decoder overshoots full scale.

### The flags that matter

| Flag | Default | What it does |
|---|---|---|
| `--max-voices N` | `1048576` | Ceiling on simultaneous voices. The biggest lever on both VRAM and speed. The maximum is whatever your card will bind; `gpu-info` prints yours. |
| `--seconds N` | off | Stop after N seconds of output. Use this constantly while experimenting. |
| `--tracks LIST` | off | Render each track on its own, as stems or, with `--merge`, summed into one file. See *Per-track rendering*. |
| `--min-velocity N` | `0` | Skip every note quieter than velocity N, as if the file did not contain it. On black MIDI full of ghost notes this can make a render many times faster. |
| `--volume P` | `100` | Volume as a percentage, 0 to 200, before the limiter. A dense mix sits far above full scale, so lowering it eases the limiting more than it quietens the file. |
| `--limiter` | `brickwall` | `brickwall` never exceeds its ceiling. `omni` is kept only for level-matching against OmniConverter, BASS or XSynth. `off` clips, and is refused for encoded output. |
| `--ceiling-db` | `0`, or `-1` for lossy | The limiter's ceiling in dBFS. |
| `--note-grid` | off | Hold notes to BASSMIDI's 4 ms envelope grid, so notes shorter than 4 ms still sound. Only for files that rely on it. |
| `--phase-mode` | `baseline` | `analytic` is experimental phase rotation: stacked copies of a note get different phases. Needs extra memory and minutes of preparation; see [its guide](https://github.com/thtstickyboi/kestrel-midi/blob/main/docs/analytic-phase-rotation.md). |
| `--interp` | `linear` | `nearest`, `linear` or `cubic`. Cubic reads twice as many samples per voice. |
| `--block N` | `4096` | Frames per render block. Also sets how many note-ons are held in host RAM at once. |
| `--steal-percent` | `25` | How much of the pool one block may replace. |
| `--format` | `float32` | `float32` or `pcm16`, for WAV. |
| `--backend` | `gpu` | `cpu` is the reference implementation: correct, slow, single-threaded. |
| `--gpu-backend NAME` | automatic | `vulkan`, `dx12`, `metal` or `gl`. Kestrel prefers Vulkan, which is also the faster one on Windows. |
| `--gpu-adapter TEXT` | automatic | Which card, by any part of its name: `--gpu-adapter intel`. In the guided renderer, `--adapter N` picks card N from its list. |
| `--nan-guard` | off | Check every block for NaN and Inf. |
| `--profile` | off | Per-pass GPU timings and the host/device split. |
| `--progress json` | off | Machine-readable progress for one render; see *Building a GUI on Kestrel*. |

`kestrel --force-cli render --help` lists everything.

### The other commands

```bash
kestrel --force-cli gpu-info              # adapters, limits, and each one's --max-voices ceiling
kestrel --force-cli info file.sf2         # what the loader made of a soundfont
kestrel --force-cli info file.mid         # ...or of a MIDI: note counts, controllers, tempo, density
kestrel --force-cli tracks file.mid       # what each track holds, for per-track rendering
kestrel --force-cli ffmpeg-info           # the ffmpeg encoded output would use
kestrel --force-cli get-ffmpeg            # fetch one
kestrel --force-cli check-update          # is a newer Kestrel out? downloads nothing
kestrel --force-cli api                   # a session another program drives; see API.md
```

`info` is the first thing to reach for when a render sounds wrong: it tells you what Kestrel *thinks* your file contains. On a MIDI it also estimates the voices and memory a render needs; DESIGN.md explains how to read it.

### Per-track rendering

Renders every track of a MIDI on its own, as if the file held nothing else, and writes each one to a file of its own, or sums them all into one. In the guided renderer it is menu item 2.

```bash
kestrel --force-cli tracks song.mid                  # what each track holds
kestrel --force-cli render song.mid -s piano.sfz --tracks all --merge -o song.flac
kestrel --force-cli render song.mid -s piano.sfz --tracks all -o stems --stem-format flac
kestrel --force-cli render song.mid -s piano.sfz --track 3 -o track3.wav
```

- **What a track hears:** its own notes and controllers, the file's tempo, and the file's *setup tracks* (controllers and no notes) unless you add `--setup-tracks ignore`. It does **not** hear other tracks' controllers. MIDI ports and analytic phase don't apply.
- **Which tracks:** `--tracks all` is every track with notes; otherwise list them as `kestrel tracks` numbers them, with ranges: `--tracks 1-40,57,90-`.
- **Voices are a total**, shared evenly: `--max-voices 60000000` over 50,000 tracks is 1,200 each, and the most is 2,000,000,000. The guided renderer starts at 60,000,000, and its `-1` gives the most your card takes with 256 tracks on it at once. More still renders, a few tracks at a time and slower.
- **Stems** go in a folder named after the MIDI, as `003 Piano.flac`, in the format `--stem-format` names. 32-bit float WAV stems skip the limiter so they add back up to the mix. Every stem runs the file's whole length, so WAV stems of a big file can reach hundreds of gigabytes; Kestrel checks the free space first. FLAC makes the silence nearly free.
- **One file** (`--merge`) sums the tracks exactly, then applies `--volume`, the limiter and the encode once. The mix is held in memory, about 92 MB per minute of audio, and a stopped merge writes no file.
- **Speed:** up to 256 tracks render together, and `--track-jobs N` sets the CPU threads that prepare them (8 by default). A silent stretch of a track never reaches the GPU. The progress screen's speed is of the file being made, so a big merge can honestly read below 1x realtime.

### Building a GUI on Kestrel

`kestrel --force-cli api` starts a session another program drives, one JSON object per line each way over stdin and stdout: list GPUs and options, scan MIDIs, keep soundfonts loaded between renders, and run renders with live telemetry and a clean cancel. **[API.md](API.md)** is the protocol, with a Python example. To watch a single render instead, add `--progress json` to a render command. From Rust, the same pipeline is `kestrel::session`.

## General MIDI

**Rudimentary.** An ordinary GM file will play, but not necessarily the way a GM synth would. Soundfonts layer: `-s` is repeatable and merges in order, and `--sf-programs` places the last one on the programs you name:

```bash
kestrel --force-cli render gm.mid -s general-midi.sf2 -s piano.sfz --sf-programs 0-1 -o out.wav
```

Implemented: program change, bank select with the GM fallback, drum kits in bank 128, channel volume, expression, pan, sustain and sostenuto, pitch bend, portamento, RPN 0/1/2, CC71-75, CC120, CC121, CC123-127, MIDI ports, SF2 vibrato and tremolo LFOs, and the SF2 modulation envelope. Where Kestrel and BASSMIDI disagree, BASSMIDI is the reference; DESIGN.md lists what was matched.

Known to differ from a reference GM synth:
- Several channels come out noticeably brighter. Cause unknown.
- The sample pool is resampled to one rate at load, which band-limits a soundfont recorded lower; `--no-resample-pool` avoids it.
- Only the GS rhythm-part, GS Reset and GM System On/Off SysEx messages are acted on.
- Rapid CC11 gating reaches digital silence where the reference does not.

## Known limitations

- **No effects.** No reverb or chorus; CC91 and CC93 do nothing.
- **Note boundaries can click on presets with no attack or release**, such as synth leads and square-wave basses, because Kestrel starts and stops a voice on its exact frame. It is the largest known audible gap.
- **A block too dense to hold whole can lose one side of a stereo piano.** Past the per-block candidate cap, the thinning keeps the same layer of every note-on, so with a stereo sample library one channel goes quiet in the densest moments. Fixing it is planned for 1.2.1.
- **A merge isn't the whole file.** It is the sum of the tracks as each sounds alone, so a file whose tracks drive each other's channels merges noticeably differently from a normal render. That's by design.
- **On the densest files the host is the bottleneck.** MIDI reading runs on one CPU core, and a block carrying around a billion note-ons takes minutes of host work while the GPU waits.
- **Analytic phase is experimental.** It prepares for a few minutes before every render and needs extra memory, and on a large piano `--phase-scratch-mib 1024`; the error names the flag.
- **SFZ support is almost complete.** Not applied: `fillfo_depth`, the LFO `*_fade` opcodes, controller-driven LFOs, SFZ v2's `lfoN_*`, `note_selfmask`, and the per-stage envelope velocity-tracking opcodes. Kestrel warns about every opcode it does not apply.
- **SF2's modulation LFO to filter** (`modLfoToFilterFc`) is not implemented.
- **VRAM figures** in the progress screen and the JSON feed are Windows only. **NaN checking is off** unless you pass `--nan-guard`.

## Planned

- **1.2.1:** the stereo thinning fix above; a high-pass on the mix before the limiter, so the inaudible low end of very dense mixes stops spending its headroom; and a faster per-track render.
- **A de-click fade at note boundaries.** Measured and understood; the fade length is what is left to settle.
- **A faster host**, with MIDI reading spread across cores.
- **A log file per render**, so a failure leaves its settings and its point of failure behind.

Effects, a host C ABI and realtime playback remain out of scope.

## License

Kestrel is licensed under the **Mozilla Public License 2.0** -- see [LICENSE](LICENSE). Code that was not written for this project, and every linked crate, is listed in [THIRD-PARTY.md](THIRD-PARTY.md): the `omni` limiter is a port of the one OmniConverter ships, originally from Kiva; analytic phase rotation is adapted from SYNCore by DixelU; FLAC samples are decoded by claxon. ffmpeg is never bundled.
