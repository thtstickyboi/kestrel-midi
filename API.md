# Kestrel API

For programs that drive Kestrel: GUIs, batch tools, scripts. One Kestrel process is one session. Your program talks to it over its standard input and output, one JSON object per line each way, so any language that can start a process and read lines can use it. There is no port, no library to link and nothing to install.

```
kestrel --force-cli api
```

A session can list GPUs and ffmpeg, describe MIDIs and soundfonts, list every render option with its default, keep soundfonts loaded between renders, run renders with live telemetry, and cancel them. A render asked for through the API writes exactly the file `kestrel --force-cli render` writes with the same settings, because it is parsed and run by the same code.

## The basics

- **Kestrel speaks first.** The first line is `ready`:
  ```json
  {"type": "ready", "api": 1, "version": "1.1.2", "build": "release", "commands": ["adapters", "ffmpeg", "..."]}
  ```
  `build` is `release` for the downloadable zips and `dev` for a build made with `--features dev`, which takes the developer options too (from 1.1.2; absent before).
- **A request** is an object with an `id` of your choosing (a number or a string) and a `cmd`:
  ```json
  {"id": 7, "cmd": "inspect_midi", "path": "C:/Music/song.mid"}
  ```
- **Every request gets exactly one response**, carrying its `id` back:
  ```json
  {"type": "response", "id": 7, "ok": true, "result": {"valid": true, "tracks": 17, "...": "..."}}
  {"type": "response", "id": 8, "ok": false, "error": "unknown option \"max_voice\"; the options command lists every one"}
  ```
- **Other lines** are logs and telemetry. Lines that belong to a request carry its `id`. Every line has a `type`; ignore types you don't know, and fields you don't use.
- **Read stdout all the time**, on a thread of its own. A program that stops reading eventually stalls Kestrel.
- **The session ends** on `shutdown`, or when you close Kestrel's stdin. Closing stdin cancels anything still running, so a GUI that quits never leaves the GPU rendering.
- Paths are ordinary strings. Forward slashes work on Windows too.

## Commands

**Quick commands** answer at once, and work while a render runs: `adapters`, `ffmpeg`, `options`, `inspect_midi`, `status`, `snapshot`, `set_interval`, `cancel`, `unload`, `shutdown`. `check_update` works while a render runs too, but asks GitHub, so it can take a few seconds.

**Long commands** run one at a time: `load_soundfonts`, `scan_midi`, `render`. Sending one while another is running gets an error that starts with `busy`; wait for the running one's response, or cancel it.

| Command | Fields | Result |
|---|---|---|
| `adapters` | | `adapters`: every GPU, each with `index`, `name`, `backend`, `type` (`discrete`, `integrated`, ...), `max_voices` (the largest `max_voices` it takes), `software`, and `default` (the one a render uses unless told otherwise) |
| `ffmpeg` | `path` (optional) | `found`. When found: `path`, `source` (where it was found), `version`, and `containers`, each with `ext`, `encoder`, `available`, `lossy`, `preset`. When not: `error` |
| `options` | | `options`: every render option; see [Render options](#render-options) |
| `inspect_midi` | `path` | Instant, reads only the header. `valid`, and then `size`, `format`, `tracks`, `division` (`{"ppq": 960}`, or `{"smpte_fps", "ticks_per_frame"}`) and `warnings` (worth showing, never fatal); or `reason` when it is not a usable MIDI |
| `scan_midi` | `path`, `rate` (optional) | Reads the whole file. `tracks`, `duration_secs`, `notes`, `note_offs`, `controllers`, `program_changes`, `pitch_bends`, `tempo_changes`, `peak_notes_per_second`, `peak_second_at`, `cancelled`. Sends `scan_progress` lines while it reads |
| `load_soundfonts` | `soundfonts` (a list), `sf_programs` (optional), `options` (optional) | Loads and keeps them. `name`, `presets`, `regions`, `samples`, `pool_bytes`, `pool_rate`, `uses_lfo`, `general_midi`, `bank0_programs`, `drum_kits`, `preset_list` (each `bank`, `program`, `name`), `load_secs`, `reused` |
| `unload` | | `unloaded`: frees the loaded soundfonts' memory |
| `render` | `midi`, `out`, `soundfonts` (optional), `sf_programs` (optional), `options` (optional), `progress_interval_ms` (optional) | When the render ends: the summary (see [Telemetry](#telemetry)) plus `out` |
| `cancel` | | Stops the running render or scan after its current block, or analytic preparation at its next cancellation check. `cancelling`: the id it stops. A raw soundfont load cannot be stopped part way |
| `snapshot` | | The running render's current `progress` line |
| `set_interval` | `interval_ms` | How often a render sends `progress`: 0 for never (use `snapshot` instead), otherwise held to 10–60,000 ms. Applies to the running render and later ones. Result: the `interval_ms` applied |
| `status` | | `running` (`id` and `cmd`, or null), `loaded` (what `load_soundfonts` returned, or null), `interval_ms` |
| `check_update` | | Asks GitHub for the latest Kestrel release; downloads nothing. `current` (this Kestrel's version), `latest`, `newer` (whether `latest` is newer than `current`), `ring` (the person's update ring: `fast` for every release, `slow` for feature releases such as 1.2.0 only), `announce` (whether their ring wants to hear about `latest`: show a notice when this is true) and `url` (the release page to send people to). An error when offline, when GitHub doesn't answer within about 5 seconds, or when `KESTREL_NO_UPDATE_CHECK` is set; show nothing in any of those cases |
| `shutdown` | | Cancels anything running, answers, and exits |

### Soundfonts

Soundfonts in a list are layered in order: each one replaces whatever the ones before it define at the same bank and program. Put a General MIDI bank first and an instrument after it. `sf_programs` places the last soundfont on those programs of bank 0, spelled as `--sf-programs` takes it: `"0,1"` or `"0-7"`.

A render with no `soundfonts` uses the loaded ones. **A loaded soundfont is reused** by every render that would load it the same way, which saves the load time on each render (seconds on a large library). A render that changes an option marked `reloads_soundfonts`, such as `volume` or `rate`, loads again, and that becomes the loaded set. A render that names different `soundfonts` does the same.

Analytic phase options, such as `"phase_mode": "analytic"` and `"phase_seed": 42`, reuse the loaded soundfont. Quadrature is prepared separately for each render, with log progress during `loading_soundfont` and cancellation checks inside preparation. Cancellation there creates no output file. See [analytic phase controls](docs/analytic-phase-rotation.md).

## Render options

`options` lists every option `render` takes, read from Kestrel's own command-line definition, so a GUI built from it has the real defaults and never drifts from the version it runs:

```json
{"key": "max_voices", "flag": "--max-voices", "kind": "value", "default": "1048576", "values": [],
 "value_name": "MAX_VOICES", "reloads_soundfonts": false, "help": "Most voices sounding at once. ..."}
{"key": "limiter", "flag": "--limiter", "kind": "value", "default": "brickwall", "values": ["brickwall", "omni", "off"], "...": "..."}
{"key": "note_grid", "flag": "--note-grid", "kind": "switch", "default": false, "...": "..."}
```

**The list depends on the build.** From 1.1.2 the release build leaves out the developer options (engine tuning, switches back to old behaviour, diagnostics such as `no_lfo`, `steal` or `block_csv`), and a render that names one gets `unknown option`. A dev build lists and takes them. Build a GUI from `options` rather than from a list of your own, and it works with either.

Pass options as an object under `options`, keyed by `key`:

```json
{"id": 3, "cmd": "render", "midi": "song.mid", "out": "song.flac",
 "options": {"max_voices": 4194304, "limiter": "omni", "volume": 80, "note_grid": true}}
```

- A `switch` takes `true` or `false`. A `value` takes a string or a number. `null` leaves an option at its default.
- Options are checked exactly as the command line checks them, with the same messages: a bad value, an unknown option or an unwritable output extension is an error response, before anything loads.
- `adapter` is extra: an `index` from `adapters`, which picks that GPU.
- The MIDI, the output, the soundfonts and the progress settings are fields of the request, not options.

## Telemetry

While it runs, a render sends these lines, each with its `id`:

| Type | When | Carries |
|---|---|---|
| `config` | after `set_interval` or `cancel` | `interval_ms`, `cancel_requested` |
| `phase` | each change | `phase`: `loading_soundfont`, `opening_midi`, `preparing_device`, `rendering`, `finishing`, then `finished`, `cancelled` or `failed`; `t`, seconds since the render began |
| `setup` | once the GPU is ready | `backend`, `adapter`, `device_bytes`, `tracks`, `max_voices`, `bytes_total` |
| `progress` | every `interval_ms`, and once at the end | see below |
| `log` | as they happen | `level` (`info`, `warn`, `error`), `message` |
| `summary` or `failed` | the end, after the last `progress` | the summary, or `message` |

Then the `response`. Log lines are written the moment they happen and the other lines within a tenth of a second, so a log line can arrive just ahead of the phase it belongs to.

**`progress`** carries: `phase`, `phase_secs`, `wall_secs`, `render_secs`, `audio_secs` (audio rendered so far), `notes` (note-ons read so far), `voices` (sounding now), `max_voices`, `peak_voices`, `stolen`, `dropped`, `peak_level` (before the limiter), `clipped`, `blocks`, `bytes_read`, `bytes_total`, `tracks`, `backend`, `adapter`, `device_bytes` (Kestrel's own GPU buffers), `gpu_memory` (the whole GPU's, Windows only: `dedicated_total`, `dedicated_used`, `shared_used`, `process_used`, `process_budget`), `host_rss_bytes`, `host_rss_peak_bytes`, and three figures worked out for you:

- `progress`: 0 to 1. It measures how much of the MIDI has been read, not audio time, because a MIDI's length isn't known until it has been read to the end, and render time follows the events anyway.
- `speed`: audio seconds per second, the "x realtime" figure.
- `eta_secs`: seconds left, once there is enough to estimate from.

Any of these can be `null` before it is known.

**The summary** has `bytes`, `audio_secs`, `wall_secs`, `notes`, `voices_spawned`, `peak_voices`, `stolen`, `dropped`, `peak_level`, `clipped` and `cancelled`. A cancelled render still ends normally: an opened file is closed properly and holds what was rendered, and `cancelled` is `true`. Cancellation during analytic preparation returns zero audio/bytes without creating a file.

`scan_midi` sends `scan_progress` lines instead, at the same interval, with `bytes_read`, `bytes_total`, `progress` and `notes`.

## Example: Python

```python
import json, queue, subprocess, threading

kestrel = subprocess.Popen(["kestrel", "--force-cli", "api"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
lines = queue.Queue()
threading.Thread(target=lambda: [lines.put(json.loads(l)) for l in kestrel.stdout],
                 daemon=True).start()
assert lines.get()["type"] == "ready"

next_id = 0
def call(cmd, **fields):
    """Send a request and wait for its response, showing progress meanwhile."""
    global next_id
    next_id += 1
    kestrel.stdin.write(json.dumps({"id": next_id, "cmd": cmd, **fields}) + "\n")
    kestrel.stdin.flush()
    while True:
        line = lines.get()
        if line["type"] == "progress":
            print(f"{line['progress'] or 0:6.1%}  {line['voices']:>9} voices  {line['notes']:>12} notes")
        elif line["type"] == "log" and line["level"] != "info":
            print(line["level"], line["message"])
        elif line["type"] == "response" and line["id"] == next_id:
            if not line["ok"]:
                raise RuntimeError(line["error"])
            return line["result"]

print(call("load_soundfonts", soundfonts=["piano.sfz"])["name"])
summary = call("render", midi="song.mid", out="song.opus", options={"max_voices": 4194304})
print(summary["notes"], "notes in", summary["wall_secs"], "s")
call("shutdown")
```

A GUI would read `lines` from its own event loop instead of blocking in `call`, and send `cancel` from a Stop button, which leaves a properly closed file rather than the broken one killing the process leaves.

## For a single render: `--progress json`

To run one render and only watch it, without a session, add `--progress json` to a normal render command. It sends the same telemetry lines on stdout (without `id`s, and beginning with `hello`), sends log lines to stderr as text, and takes `{"interval_ms": N}`, `{"snapshot": true}` and `{"cancel": true}` on stdin.

## Compatibility

`api` in `ready` is the protocol's version. Adding a command, a field or a line type does not change it; renaming or removing one, or changing what one means, does. Check it when you connect.
