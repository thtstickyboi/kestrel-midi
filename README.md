# Kestrel

'*experimental*'

A GPU-accelerated SoundFont/SFZ synthesizer for Black MIDI.

Kestrel renders a MIDI file offline to WAV, FLAC, Opus, MP3, OGG or M4A, doing the synthesis in compute shaders instead of on the CPU. It is built for files where the note count is absurd -- hundreds of millions of notes, with a hundred thousand to several million voices sounding at once -- which is the point where conventional synths stop being fast enough. It can also render every track of a MIDI on its own, as stems or summed into one file, render several MIDIs in one command, pick a stopped render up where it left off, and play 31-EDO and other microtonal tunings.

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
| `kestrel-<version>-linux-x64.tar.gz` | 64-bit Linux with glibc 2.35 or newer (Ubuntu 22.04+) | Tested |

- **Windows:** unzip, double-click `kestrel.exe`. The executable is not code-signed, so SmartScreen may warn about it: **More info -> Run anyway**.
- **macOS:** open the dmg and copy `kestrel` out of it. It is not signed or notarised, so macOS refuses it the first time: go to **System Settings -> Privacy & Security** and press **Open Anyway**, or run `xattr -dr com.apple.quarantine kestrel` in Terminal.
- **Linux:** `tar xzf` the archive and run `./kestrel` from a terminal. It needs a Vulkan driver, and the file pickers go through the desktop portal (`xdg-desktop-portal`).

The macOS build is compiled by GitHub Actions and nobody has run it yet. If you are first, please open an issue either way. `SHA256SUMS.txt` beside the downloads holds each file's checksum, and from 1.2.2 each download also carries a GitHub attestation, a signed record that it was built from this repository by its release workflow: `gh attestation verify <file> --repo thtstickyboi/kestrel-midi` checks it.

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
| **OS** | Windows, Linux, macOS. macOS has not been *run* yet |
| **ffmpeg** | Only for output other than WAV |

Tested on NVIDIA and Intel GPUs on Windows, and on Linux. Integrated GPUs are slower but usable, and Kestrel prefers a discrete GPU when both are present. `--backend cpu` renders on the CPU reference instead, far slower, if you have no usable GPU.

**Memory.** VRAM is usually the limit: the voice pool (250 MiB at the default 1,048,576 voices) plus your soundfont's samples. The most voices your card holds is printed by `kestrel --force-cli gpu-info` and offered by the guided renderer, as the lower of what one GPU buffer binds (16,519,104 on an RTX 5060) and what the card's memory holds with your soundfont loaded. Only the binding is a hard limit; a count over what the memory holds is taken with a warning, since the driver may spill it into system memory, slowly. Soundfonts with up to 8 GB of samples load, if your card has the video memory for them. An SF2 whose samples pass `--pool-budget` is loaded at half its sample rate, and again until it fits. By default the budget is three quarters of your card's video memory, and never less than 2 GB. SFZ samples are never downsampled. Kestrel checks every limit before allocating and names the flag to lower rather than failing mid-render. The details are in DESIGN.md.

## Using Kestrel

### The guided renderer

Start `kestrel` with no arguments -- double-clicking it does exactly that -- and it walks you through a render:

1. **Environment check.** Every GPU Kestrel can use, the most voices each will hold, whether ffmpeg was found, and whether a newer Kestrel is out.
2. **Menu.** Render one MIDI or several; per-track render; resume a stopped render; Extras; exit.
3. **MIDI and soundfonts**, through your system's own file pickers. At most two soundfonts for a render; a General MIDI bank goes underneath the other. With several MIDIs you can give each its own soundfonts from a table in the terminal, driven by the arrow keys: see [docs/batch.md](docs/batch.md).
4. **Voices, format and destination folder.** A name already taken gets the date and time added rather than being overwritten. The voices step offers your card's most, which is the lower of what one GPU buffer binds and what its memory holds, and says which.
5. **Additional flags**, one free-form line for anything the steps do not ask about -- `--seconds 30`, `--volume 80`, `--min-velocity 10`.
6. **Progress**, then a summary with the flags the render ran with, and **Home** to go back to the menu.

A guided render writes a file byte-identical to the equivalent `--force-cli render`. On the progress screen, **Ctrl+C twice** (or Esc twice) stops the render and saves its progress so it can be resumed, and **Ctrl+C, then Ctrl+D** stops it and keeps nothing; see *Stopping and resuming*. **Extras** has GPU info, file info, help for every flag, the machine report, the update ring and the remembered folders.

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
| `--volume P` | `100` | Volume of the finished file as a percentage, 0 to 200. Up to 100 it applies after the limiter, so 50 is half as loud, dense passages included. Above 100 the extra goes in before the limiter, which still holds the ceiling, so the mix gets denser rather than louder. |
| `--dc-blocker` | off | Filter out everything below 15 Hz before the limiter. Very dense mixes build up an inaudible low-frequency rumble that the limiter spends its headroom on; with this on, the music gets that headroom back. |
| `--limiter` | `brickwall` | `brickwall` never exceeds its ceiling. `omni` is kept only for level-matching against OmniConverter, BASS or XSynth. `off` clips, and is refused for encoded output. |
| `--ceiling-db` | `0`, or `-1` for lossy | The limiter's ceiling in dBFS. |
| `--note-grid` | off | Hold notes to BASSMIDI's 4 ms envelope grid, so notes shorter than 4 ms still sound. Only for files that rely on it. |
| `--phase-mode` | `baseline` | `analytic` is experimental phase rotation: stacked copies of a note get different phases. Needs extra memory and minutes of preparation; see [its guide](https://github.com/thtstickyboi/kestrel-midi/blob/main/docs/analytic-phase-rotation.md). |
| `--interp` | `linear` | `nearest`, `linear` or `cubic`. Cubic reads twice as many samples per voice. |
| `--pool-budget MIB` | from your GPU | The most memory an SF2's samples may take. One that needs more loads at half its sample rate, and again until it fits. By default three quarters of your card's video memory, at least 2048; `--pool-budget 2048` is how 1.2.2 and earlier loaded. |
| `--block N` | `4096` | Frames per render block. Also sets how many note-ons are held in host RAM at once. |
| `--steal-percent` | `25` | How much of the pool one block may replace. |
| `--checkpoint-every MIN` | `10` | How often a render saves its progress, in minutes (fractions allowed). `0` saves only when it is stopped. See *Stopping and resuming*. |
| `--no-resume` | off | Keep no progress file at all: nothing saved as it goes or when it stops, and the audio goes straight to the output. |
| `--edo N` | off | Lay N equal steps over the octave, one on each key: `--edo 31`, 19, 22, 53. `--tuning-ref KEY:HZ` says where from; the default leaves middle C where it is. See *Microtonal tuning*. |
| `--scala FILE.scl` | off | Tune the keys from a Scala scale, with `--kbm FILE.kbm` for a keyboard map. |
| `--31edo` | off | Play a MIDI written for the 31-EDO template, which uses note keys over 127. |
| `--mts-notes` | off | Apply MIDI Tuning Standard single-note changes and bulk dumps. BASSMIDI ignores them. |
| `--format` | `float32` | `float32` or `pcm16`, for WAV. |
| `--backend` | `gpu` | `cpu` is the reference implementation: correct, slow, single-threaded. |
| `--gpu-backend NAME` | automatic | `vulkan`, `dx12`, `metal` or `gl`. Kestrel prefers Vulkan, which is also the faster one on Windows. |
| `--gpu-adapter TEXT` | automatic | Which card, by any part of its name: `--gpu-adapter intel`. In the guided renderer, `--adapter N` picks card N from its list. |
| `--nan-guard` | off | Check every block for NaN and Inf. |
| `--log` | off | Write a log of the render beside `kestrel.exe`, in `logs`. The guided renderer and the API always do; see *FalconEye*. |
| `--profile` | off | Per-pass GPU timings and the host/device split. |
| `--progress json` | off | Machine-readable progress for one render; see *Building a GUI on Kestrel*. |

`kestrel --force-cli render --help` lists everything.

### The other commands

```bash
kestrel --force-cli gpu-info              # adapters, limits, and each one's --max-voices ceiling
kestrel --force-cli info file.sf2         # what the loader made of a soundfont
kestrel --force-cli info file.mid         # ...or of a MIDI: note counts, controllers, tempo, density
kestrel --force-cli tracks file.mid       # what each track holds, for per-track rendering
kestrel --force-cli batch jobs.json       # several MIDIs, each with its own soundfonts; see docs/batch.md
kestrel --force-cli resume file.krsm      # pick a stopped render up where it left off
kestrel --force-cli ffmpeg-info           # the ffmpeg encoded output would use
kestrel --force-cli get-ffmpeg            # fetch one
kestrel --force-cli check-update          # is a newer Kestrel out? downloads nothing
kestrel --force-cli report                # a machine report to send with a bug report; see FalconEye
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

### Several MIDIs at once

New in 1.3.0. `kestrel --force-cli render a.mid b.mid c.mid -s gm.sf2 -o renders/` renders each MIDI to a file of its own, and `kestrel --force-cli batch jobs.json` gives each MIDI the soundfonts a JSON file names for it. The guided renderer takes several MIDIs too (menu item 1), and so does the API.

Jobs run one at a time, grouped by soundfont set so each set loads once, and **each file is byte-for-byte what that MIDI renders as on its own**. A job that fails does not stop the rest. The rules, the file format and the guided table are in **[docs/batch.md](docs/batch.md)**.

### Stopping and resuming

New in 1.3.0. A render that is stopped, fails, or loses power need not start over. It saves its progress as it goes (every 10 minutes, `--checkpoint-every`) and when it is stopped, to a `.krsm` file beside the output, and `kestrel --force-cli resume file.krsm` goes on from there. The guided renderer has **Resume a render** in its menu, and the API has `resume`. **The file a resumed render writes is byte-for-byte the one an uninterrupted render writes.**

- **A render of one file** saves the render itself: where it is in the MIDI, every channel's controllers, the voices that are sounding and the limiter. A resume loads the soundfont and goes on from the block it had come to, so the minutes already rendered are not rendered again. Until it is whole, the audio is in `name.partial.wav`, and a resume cuts it back to what the checkpoint says. An encoded format (`.flac`, `.opus`, ...) cannot be cut and continued, so it also keeps the samples it feeds the encoder in `<file>.krsm.pcm`.
- **A render with `--tracks`** saves its tracks. Those that were finished are not rendered again; tracks that were under way start again from their beginning, and a merge does not add what it already holds.
- **It is refused, and says why,** if anything the render depended on has changed: the build of Kestrel (a rebuild counts, and so does an update), the MIDI, a soundfont or an SFZ's samples, the tracks, the output format, the backend or the card, or the voice limit.
- **A checkpoint of a big render is big:** about 1.6 GiB at 16 million voices, written in a couple of seconds. To stop a render and keep none of it, press **Ctrl+C, then Ctrl+D** in the guided renderer (the API's `cancel` takes `"discard": true`), or start it with **`--no-resume`**, which saves nothing at all. Ctrl+D on a render that was itself resumed also deletes the checkpoint it was resumed from.
- **Ctrl+C on the plain command line** still ends the process with nothing saved, so only the periodic checkpoint protects a render there. A batch does not save its progress, and neither does a render with `--block-csv` or `--phase-*`.

### Microtonal tuning and 31-EDO

New in 1.3.0. Each of the 128 keys can have a pitch of its own. **The sample that plays is the one nearest the pitch, not the one under the key struck,** so a sampled piano does not stretch one sample over many semitones and jump at every region's edge. Drums are not tuned.

- **`--edo N`** lays N equal steps over the octave, one on each key (`--edo 31`, 19, 22, 53, any N from 1 to 1200). `--tuning-ref KEY:HZ` says where from; the default leaves middle C where it is, and `--edo 12` is no tuning at all, byte for byte.
- **`--scala FILE.scl`**, with **`--kbm FILE.kbm`** if there is one, reads the scale and keyboard map the microtonal archives publish: cents and ratios, a map's range, middle note and reference, and `x` for keys that do not sound. A file that is wrong is refused with its name and line.
- **`--31edo`** plays a MIDI written for the 31-EDO template, which carries note keys 0 to 255, one step of 31 each. Without the option such a file plays the wrong notes, here and in BASSMIDI, and says nothing; `kestrel info` tells you whether a file is one, a render warns, and the guided renderer adds the option itself for a file on its own.
- **MIDI Tuning Standard.** Scale/octave tuning messages are read always, since BASSMIDI reads them. Single-note changes and bulk dumps are applied only with **`--mts-notes`**, because BASSMIDI ignores them and a render without the flag should match it.

Not done: retuning a note that is already sounding, and a portamento glide that lands on a tuned pitch (it still glides in whole semitones). A batch cannot render a 31-EDO file, since the option would apply to every file in it.

### A very high voice count, and Windows' 2-second limit

Windows resets a graphics driver that spends more than 2 seconds on one piece of GPU work, which ends the render, and with NVIDIA's Vulkan driver the process with it before any message prints. At the default 1,048,576 voices that does not happen on a card Kestrel runs well on. The guided renderer's "device max" and `--max-voices` can go far past it, and **neither limit is a limit of speed**: a slower card can take over 2 seconds on one block at a count the step offers.

Kestrel keeps clear of it by cutting a block of millions of voices into several submissions, which changes no byte of the output and costs a render at the defaults nothing. If a render still stops this way (the render log says `gpu device lost`, and the crash report names the driver reset), lower `--max-voices`. The log's "longest device wait" says how close a render came, and the **GPU self-test** in Extras → Machine report says how many voices a block takes a second to reach on your card. `--block 1024` gives the GPU a quarter of the work at a time (about 15% slower, and not byte-identical to the default), and `--gpu-backend dx12` turns the abort into a readable message.

### Building a GUI on Kestrel

`kestrel --force-cli api` starts a session another program drives, one JSON object per line each way over stdin and stdout: list GPUs and options, scan MIDIs, keep soundfonts loaded between renders, and run renders, resumes and batches with live telemetry and a clean cancel. **[API.md](API.md)** is the protocol, with a Python example. To watch a single render instead, add `--progress json` to a render command. From Rust, the same pipeline is `kestrel::session`.

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

### FalconEye: logs and crash reports

FalconEye is Kestrel's error catching and logging, new in 1.2.2. Nothing it writes leaves your computer unless you send it, and your PC's and Windows account's names are left out of all of it.

- **Every render leaves a log** in `logs` beside `kestrel.exe` (guided renderer and API; `--log` on the command line). If a render fails, the guided renderer shows the log's path: that's the file to send with a bug report.
- **If a render crashes or hangs**, a second, windowless `kestrel.exe` that watched it writes `<log> CRASH.txt` or `HANG.txt` beside the log, with what the system recorded about it. On Windows, a crash inside a graphics driver also gets a small minidump, and the log names the file the crash happened in.
- **Extras → Machine report** (or `kestrel --force-cli report`) writes one zip with your hardware, drivers, graphics settings, recent logs and a GPU self-test that says how close your GPU runs to Windows' 2-second reset. It asks before the self-test, and separately before an optional step that needs administrator permission. Each GPU is tested in a process of its own, so a driver that crashes the test leaves a report that says so instead of no report. If a crash looks like a driver reset, the crash report says that at the top, with your Windows graphics timeout settings.

**[docs/falconeye.md](docs/falconeye.md)** is the transparency report: exactly what FalconEye writes, reads and hides, and what it does that antivirus might notice.

## Known limitations

- **No effects.** No reverb or chorus; CC91 and CC93 do nothing.
- **Note boundaries can click on presets with no attack or release**, such as synth leads and square-wave basses, because Kestrel starts and stops a voice on its exact frame. It is the largest known audible gap.
- **A merge isn't the whole file.** It is the sum of the tracks as each sounds alone, so a file whose tracks drive each other's channels merges noticeably differently from a normal render. That's by design.
- **On the densest files the host is the bottleneck.** MIDI reading runs on one CPU core, and a block carrying around a billion note-ons takes minutes of host work while the GPU waits.
- **Analytic phase is experimental.** It prepares for a few minutes before every render and needs extra memory, and on a large piano `--phase-scratch-mib 1024`; the error names the flag.
- **SFZ support is almost complete.** Not applied: `fillfo_depth`, the LFO `*_fade` opcodes, controller-driven LFOs, SFZ v2's `lfoN_*`, `note_selfmask`, and the per-stage envelope velocity-tracking opcodes. Kestrel warns about every opcode it does not apply.
- **SF2's modulation LFO to filter** (`modLfoToFilterFc`) is not implemented.
- **VRAM figures** in the progress screen and the JSON feed are Windows only. **NaN checking is off** unless you pass `--nan-guard`.
- **FalconEye on Linux and macOS is new, and a crash on purpose hasn't been tried on either.** The machine report has run on Linux. There, the watcher reads how a render ended from the system's own crash records and logs, not from the render itself. There are no minidumps, and some Linux distributions only let certain groups read the kernel log. macOS has never run it, and Kestrel has never been run on a Mac at all. If you use either, a machine report and your impressions would help.
- **Resume is exact, so it is strict.** A checkpoint is refused after an update or a rebuild of Kestrel, and after any change to the MIDI, the soundfonts or the settings. A resume of a per-track render renders the tracks that were under way again from their start.
- **Microtonal tuning doesn't retune a note that is sounding, and a portamento glide still moves in whole semitones.** A batch cannot render a 31-EDO file.- **A soundfont over 2 GB renders about 4% slower,** because its samples are split across GPU buffers. Past 8 GB of samples, it doesn't load.

## Planned

- **A faster per-track render**, for files with hundreds of tracks playing at once.
- **A de-click fade at note boundaries.** Measured and understood; the fade length is what is left to settle.
- **A faster host**, with MIDI reading spread across cores.

Effects, a host C ABI and realtime playback remain out of scope.

## License

Kestrel is licensed under the **Mozilla Public License 2.0** -- see [LICENSE](LICENSE). Code that was not written for this project, and every linked crate, is listed in [THIRD-PARTY.md](THIRD-PARTY.md): the `omni` limiter is a port of the one OmniConverter ships, originally from Kiva; analytic phase rotation is adapted from SYNCore by DixelU; FLAC samples are decoded by claxon. ffmpeg is never bundled.
