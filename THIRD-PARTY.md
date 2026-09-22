# Third-party code and licences

Kestrel is under the Mozilla Public License 2.0 (see [LICENSE](LICENSE)). Some of what it uses is not Kestrel's to license, and this file says what and whose. It covers three separate things, and they have different consequences:

1. **code carried in this repository** that came from somewhere else, which imposes obligations on anyone redistributing Kestrel;
2. **crates linked at build time**, which impose the usual notice obligations on a distributed binary;
3. **tools used while developing Kestrel** that are not in this repository and are not part of any release.

---

## 1. Code in this repository from elsewhere

### `src/phase.rs` and `shaders/phase.wgsl` -- analytic phase rotation

The analytic quadrature preparation, deterministic angle assignment, loop-body
handling, normalization, and attack protection are adapted from **SYNCore**,
`SAFSYN/phase.cpp` and `SAFSYN/phase.h`, revision `4cd9373`.
SYNCore releases this code under the Unlicense/public-domain dedication;
its original [license text](third_party/SYNCore-LICENSE.txt) is included.
The Rust/GPU integration and finite coefficient tables are Kestrel additions.

### `src/limiter.rs` -- the `omni` limiter

The `Limiter` type in `src/limiter.rs` (the `--limiter omni` path) is a port of `OmniConverter/Extensions/Audio/Limiter.cs`, which is itself from **Kiva** by **Arduano**.

- **Origin:** <https://github.com/arduano/Kiva>
- **Licence:** DON'T BE A DICK PUBLIC LICENSE, Version 1.1
- **Copyright:** (C) 2020 Arduano

DBAD is permissive and is not a copyleft licence, so it does not extend to the rest of Kestrel and does not conflict with the MPL. It asks, in substance, for credit and for not passing the work off as your own. This file is that credit; so is the comment at the top of `src/limiter.rs`.

**Scope, and why it is narrowing.** Only the `Limiter` type is derived from Kiva. `Brickwall`, which is the default and the only limiter recommended for rendering, is original work and shares no code with it. `--limiter omni` is kept for level-matching a render against BASSMIDI or XSynth and is deprecated for anything else. If it is ever removed, this entry goes with it.

---

## 2. Crates linked at build time

All are permissive and none are copyleft. Full texts are reproduced by `cargo about` or `cargo license` if a distribution needs them. Each licence is as the crate declares it, so `pollster`'s older slash syntax is left as written.

| crate | licence |
|---|---|
| `anyhow` | MIT OR Apache-2.0 |
| `bytemuck` | Zlib OR Apache-2.0 OR MIT |
| `chrono` | MIT OR Apache-2.0 |
| `claxon` | Apache-2.0 |
| `clap` | MIT OR Apache-2.0 |
| `crossterm` | MIT |
| `env_logger` | MIT OR Apache-2.0 |
| `log` | MIT OR Apache-2.0 |
| `memory-stats` | MIT OR Apache-2.0 |
| `pollster` | Apache-2.0/MIT |
| `rfd` | MIT |
| `serde` | MIT OR Apache-2.0 |
| `serde_json` | MIT OR Apache-2.0 |
| `unicode-width` | MIT OR Apache-2.0 |
| `wgpu` | MIT OR Apache-2.0 |
| `windows` *(Windows builds only)* | MIT OR Apache-2.0 |

`rfd` opens the native file pickers. On Linux it is built against the desktop portal rather than GTK, so it links no system library there either.

`claxon` is the FLAC decoder and is the one dependency here that is Apache-2.0 only rather than dual-licensed. It is also the only decoder in the project that is not hand-written; everything in `src/wav.rs` is Kestrel's own.

`windows` is used only by `src/gpu/vram.rs`, to read whole-GPU memory figures on Windows.

---

## 3. Development tools, not in this repository

### un4seen BASS and BASSMIDI

Kestrel's behaviour is measured against BASSMIDI during development, using un4seen's BASS audio library SDK. **Proprietary**, free for non-commercial use only; commercial use requires a licence from un4seen.

- <https://www.un4seen.com/>

It is not in this repository, nothing in the library or the `kestrel` binary links it, and no Kestrel release includes it.

### OmniConverter -- reference source

A copy of OmniConverter's source was consulted while porting behaviour, most directly the limiter above. It is not in this repository and is not built or linked.

---

## Not a dependency: ffmpeg

Kestrel encodes `.opus`, `.mp3`, `.ogg`, `.flac` and `.m4a` by **running ffmpeg as a separate process**. It does not link ffmpeg, contains none of its code, and does not ship it.

Distributed ffmpeg builds are typically **GPLv3**. Running an unmodified program at arm's length does not make Kestrel a derivative work of it, so this does not affect Kestrel's licensing. Two things would:

- **Bundling an ffmpeg binary in a Kestrel release.** Distributing a GPL binary alongside Kestrel brings GPL obligations for that binary, including a source offer. Kestrel's release archives do not include ffmpeg.
- `kestrel get-ffmpeg` deliberately **downloads at the user's explicit request** rather than bundling, and prints the licence of what it is about to fetch before fetching it. That keeps the obligation with the user's own copy, which is why it is a separate command and not part of `render`.

Rendered audio is not covered by Kestrel's licence in any case. The output contains the *soundfont's* sample data, so what governs a published render is the licence of the soundfont used, not this one.
