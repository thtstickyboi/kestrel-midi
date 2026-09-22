# Analytic phase rotation

Kestrel supports SYNCore-style analytic phase rotation. Enable it with
`--phase-mode analytic`; baseline rendering remains the default.

```powershell
kestrel --force-cli render song.mid -s piano.sf2 -o analytic.wav --phase-mode analytic --phase-seed 42
```

Quadrature samples are prepared once on the CPU before each render. The GPU
rotates and interpolates every playing voice. There are no per-sample
trigonometric calls, FFTs, or extra compute passes.

## Controls

| Flag | Default | Meaning |
| --- | --- | --- |
| `--phase-mode` | `baseline` | Choose `baseline` or `analytic`. |
| `--phase-strength` | `1` | Angle spread, from 0 to 1. Zero uses the baseline path. |
| `--phase-seed` | `0` | Deterministic angle assignment. |
| `--phase-pool` | `64` | Number of cached angles, from 1 to 64. |
| `--phase-continuous` | off | Assign continuous angles; ignores the pool size. |
| `--phase-preserve-attack-ms` | `0` | Keep this source-sample prefix unchanged, then blend into rotation over up to 10 ms. |
| `--phase-cache-mib` | `2048` | Maximum packed quadrature/metadata cache size. |
| `--phase-scratch-mib` | `512` | Maximum conservative FFT scratch estimate. |

All controls are available in regular and dev builds, in the guided renderer's
additional-flags field, and through API render options. For example:

```json
{"id": 3, "cmd": "render", "midi": "song.mid", "out": "analytic.wav",
 "options": {"phase_mode": "analytic", "phase_seed": 42, "phase_preserve_attack_ms": 5}}
```

Phase options do not invalidate the API's loaded soundfont. The first version
rebuilds the phase cache for each render, including renders using a preloaded
bank. The driver and backend share that preparation. Progress appears in logs
during `loading_soundfont`; cancellation is checked between samples and inside
FFT work. Cancellation during preparation creates no output file.

## Signal and coefficient caching

For a decoded sample `x` and its Hilbert quadrature `q`, each interpolation tap
is transformed as:

```text
rotated = (cosine * x - sine * q) * scale
```

The finite pool precomputes cosine/sine pairs once per render and normalization
scales once per effective sample and angle. At 64 angles, that is 512 bytes for
the pairs and 256 bytes of scales per sample. A voice receives three immutable
coefficients. It does not look up an angle table in the GPU sample loop.

Continuous mode computes the pair once per retained note and reuses it across
layers; normalization is sample-dependent. Neither mode caches a full waveform
per angle. The shader specializes away attack blending when it is disabled.

Angles depend on the seed, **original MIDI tick**, channel, and key. Same-tick
unisons and layers share an angle, including different velocities. Distinct
ticks remain distinct identities even when they round to the same output frame;
a finite pool can still assign them the same angle. Admission, thinning, delayed
attacks, voice compaction, and sorting retain that assignment.

SYNCore's sample preparation and normalization are preserved:

- Whole-sample quadrature uses zero padding to a power of two at least twice
  the source length. Exact-length periodic loops use radix-2 FFTs or Bluestein
  for arbitrary lengths, including prime lengths.
- The periodic loop body replaces the padded result. Its entry blend spans at
  most 10 ms, a quarter loop, and the available prefix.
- Effective source spans and loop bounds come from the same address calculation
  as voice spawning. Regions sharing that geometry share a quadrature.
- Whole-sample energy determines the angle-dependent scale. Stereo regions use
  a common angle with their own energy normalization.
- Protected attacks are measured in source indices. Each nearest, linear, or
  cubic interpolation tap gets its own wrapped/clamped index before rotation.

Rotation decorrelates different onset groups; it does not change note timing
or guarantee lower peaks. Keep the limiter enabled for listening comparisons.
Whole-sample energy normalization does not imply unchanged mix energy after
interpolation, attack blending, filtering, or repeated looping.

## Memory and device limits

Each unique effective sample adds four bytes per frame of quadrature beside
Kestrel's two-byte PCM. With one quadrature per stored sample, GPU sample data
therefore grows to approximately three times its original size. Different loop
or end-address variants can require additional quadratures. The host retains
the prepared cache too, plus temporary FFT storage during preparation.

Analytic voices add three scalar fields: 12 bytes per slot in each of the two
GPU voice buffers. At 1,048,576 voices and 25% stealing headroom, this adds
30 MiB. Spawn commands also carry the three coefficients. Baseline mode and
zero strength allocate no quadrature or extra GPU voice fields and retain the
original sample-fetch path.

The implementation checks cache and scratch budgets before transforming PCM.
The packed cache must also fit one GPU storage binding, and the adapter must
support ten compute storage buffers. Raising `--phase-cache-mib` cannot bypass
a device binding limit. Select a backend with a larger limit or explicitly
lower the render/sample-pool rate if needed; phase mode does not silently
downsample the bank.

## Implementation and validation

The source reference was `C:/Users/User/SAFC_v1/SYNCore` at `4cd9373`, especially
`SAFSYN/phase.cpp`, `phase.h`, and the interpolation path in `engine.cpp`.
Attribution and the original Unlicense text are in [THIRD-PARTY.md](../THIRD-PARTY.md).

Key implementation files:

- [src/phase.rs](../src/phase.rs): preparation, angle hashing, coefficients,
  normalization, budgets, and cancellation.
- [src/driver.rs](../src/driver.rs): source-tick identity and voice admission.
- [shaders/phase.wgsl](../shaders/phase.wgsl): GPU tap rotation/interpolation.
- [src/phase/tests.rs](../src/phase/tests.rs): analytic and integration checks.

Kestrel retains its signed-normalized PCM decoding (`/32767`, clamped at the
negative extreme). The independent SYNCore reference uses `/32768`; golden
values account for this difference. Whole renders from the two synthesizers
are not expected to be bit-identical because their schedulers and envelopes
also differ.

Validated on 2026-09-22:

- 140 automatic tests passed in both regular and dev builds (`cargo test
  --offline --all-targets`, with and without `--features dev`).
- Independent golden values generated by compiling SYNCore's C++ phase
  processor match the Rust preparation, finite/continuous coefficients, loop
  transition, and protected attack within the test tolerances.
- The explicit hardware test compares CPU and Vulkan output for nearest,
  linear, and cubic interpolation, all three voice layouts, finite/continuous
  angles, attack protection on/off, stereo samples, source-address offsets,
  delayed notes, and admission. Worst error was -131.5 dB relative to the peak; repeats were
  byte-identical on the Intel UHD Graphics 770.
- The original Kestrel `10190e5` executable, baseline mode, and analytic
  strength zero produced identical WAV SHA-256 hashes on the same fixture.
- Cache deduplication, budget rejection, FFT cancellation, preloaded-bank
  rendering, and cancellation before output creation are covered.
- A DX12 analytic render completed with finite output on the same GPU.

Run the GPU test explicitly:

```powershell
cargo test --offline --lib phase::tests::gpu_matches -- --ignored --nocapture
```

These checks establish numerical and integration behavior on this machine.
They are not a claim of identical output across GPU drivers or improved sound
for every MIDI. Timings and real-file audition results are recorded below.

## Real-file comparison

Tested the first 30 seconds of `The Creator.B6MCM.TTRb.mid` with
`sDetrimental Concert Grand Piano.sf2` on an Intel UHD Graphics 770, DX12 driver
32.0.101.7088. Both renders used 48 kHz stereo, linear interpolation, 4,096-frame
blocks, a 131,072-voice cap, and the default brickwall limiter. Analytic settings
were strength 1, seed 42, pool 64, and no attack protection. The FFT scratch
budget was explicitly raised to 1,024 MiB for this run.

| Measurement | Baseline | Analytic |
| --- | ---: | ---: |
| Audio duration (whole blocks) | 30.037 s | 30.037 s |
| Render time, excluding setup | 17.05 s | 20.38 s |
| Render speed | 1.76x realtime | 1.47x realtime |
| Notes processed | 43,940 | 43,940 |
| Voices spawned | 87,102 | 87,102 |
| Peak simultaneous voices | 23,436 | 23,436 |
| Stolen / dropped | 0 / 0 | 0 / 0 |
| Device buffers | 703.9 MiB | 1,894.8 MiB |
| Additional quadrature preparation | none | 81.86 s |
| Quadrature cache | none | 1,187.1 MiB, 174 samples |
| Peak before limiting | 4.017 | 4.299 |
| Final peak | 1.0 | 1.0 |
| Final RMS | 0.307540 | 0.309003 |

Every output sample was finite. The renders have matching length and scheduling
counts but different audio, as expected. Analytic render time was about 20%
higher in this one excerpt; preparation is an additional startup cost. The
complete MIDI has about 7.37 million notes and was not rendered end to end.

The 593.5 MiB source pool and its 1,187.1 MiB quadrature exceed the available
Vulkan adapter's 1,023 MiB **single-buffer** limit for the quadrature alone.
DX12 supports a 2,047 MiB binding here, so the comparison used DX12 for both
modes without changing the sample rate.

Local audition files produced during validation, with the limiter enabled:

- [Baseline excerpt](../out/phase-bench/creator-coherent.wav)
- [Analytic excerpt](../out/phase-bench/creator-analytic.wav)

Reproduce the analytic excerpt from the repository root:

```powershell
.\target\release\kestrel.exe --force-cli render "C:\Users\User\Downloads\The Creator.B6MCM.TTRb.mid" -s "C:\Users\User\Downloads\sDetrimental Concert Grand Piano.sf2" -o out/phase-bench/creator-analytic.wav --seconds 30 --max-voices 131072 --gpu-backend dx12 --phase-mode analytic --phase-seed 42 --phase-scratch-mib 1024 --nan-guard
```

Use `--phase-mode baseline` and a different output path for the baseline.

## Synthetic throughput

The generated `sustained16384.mid` / `simple.sf2` fixture ramps to 16,359
simultaneous voices over 10.07 seconds of audio. On the same UHD 770 using
Vulkan, three serial runs per mode gave these render times (48 kHz, linear,
32,768-voice cap, limiter off, NaN checking enabled):

| Mode | Median | Range |
| --- | ---: | ---: |
| Baseline | 5.03 s | 4.98-5.05 s |
| Analytic, 64 cached angles | 7.43 s | 7.29-7.57 s |
| Analytic, continuous | 7.34 s | 7.27-7.53 s |

All 16,384 notes spawned with no stealing or drops. Preparation of the single
sample took about 1.3 ms. This fixture shows about 48% extra render time for
finite analytic rotation; the continuous/finite difference is small relative
to run variation. It uses a tiny sample pool and does not predict throughput
for a large piano bank or other GPUs.

Specializing away disabled attack protection reduced this fixture's analytic
render time from an earlier median of 9.92 s to 7.43 s. Shader specialization
changes floating-point rounding, so this optimization is checked against the
CPU reference with a numerical tolerance. The before/after peak difference was
3.815e-6 (-108.37 dBFS) on this raw, over-range fixture. The original and
quadrature reads still add memory traffic; caching trigonometry alone does not
make rotation free.
