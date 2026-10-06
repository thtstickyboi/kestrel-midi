# Kestrel API

For programs that drive Kestrel: GUIs, batch tools, scripts. One Kestrel process is one session. Your program talks to it over its standard input and output, one JSON object per line each way, so any language that can start a process and read lines can use it. There is no port, no library to link and nothing to install.

```
kestrel --force-cli api
```

A session can list GPUs and ffmpeg, describe MIDIs and soundfonts, list every render option with its default, keep soundfonts loaded between renders, run renders, or a batch of several MIDIs, with live telemetry, and cancel them. A render asked for through the API writes exactly the file `kestrel --force-cli render` writes with the same settings, because it is parsed and run by the same code.

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

**Long commands** run one at a time: `load_soundfonts`, `scan_midi`, `render`, `resume`, `batch`. Sending one while another is running gets an error that starts with `busy`; wait for the running one's response, or cancel it.

| Command | Fields | Result |
|---|---|---|
| `adapters` | | `adapters`: every GPU, each with `index`, `name`, `backend`, `type` (`discrete`, `integrated`, ...), `max_voices` (the largest `max_voices` it takes), `software`, and `default` (the one a render uses unless told otherwise) |
| `ffmpeg` | `path` (optional) | `found`. When found: `path`, `source` (where it was found), `version`, and `containers`, each with `ext`, `encoder`, `available`, `lossy`, `preset`. When not: `error` |
| `options` | | `options`: every render option; see [Render options](#render-options) |
| `inspect_midi` | `path` | Instant, reads only the header. `valid`, and then `size`, `format`, `tracks`, `division` (`{"ppq": 960}`, or `{"smpte_fps", "ticks_per_frame"}`) and `warnings` (worth showing, never fatal); or `reason` when it is not a usable MIDI |
| `scan_midi` | `path`, `rate` (optional) | Reads the whole file. `tracks`, `duration_secs`, `notes`, `note_offs`, `controllers`, `program_changes`, `pitch_bends`, `tempo_changes`, `peak_notes_per_second`, `peak_second_at`, `cancelled`. Sends `scan_progress` lines while it reads |
| `load_soundfonts` | `soundfonts` (a list), `sf_programs` (optional), `options` (optional) | Loads and keeps them. `name`, `presets`, `regions`, `samples`, `pool_bytes`, `pool_rate`, `uses_lfo`, `general_midi`, `bank0_programs`, `drum_kits`, `preset_list` (each `bank`, `program`, `name`), `load_secs`, `reused` |
| `unload` | | `unloaded`: frees the loaded soundfonts' memory |
| `render` | `midi`, `out`, `soundfonts` (optional), `sf_programs` (optional), `options` (optional), `progress_interval_ms` (optional) | When the render ends: the summary (see [Telemetry](#telemetry)) plus `out`. A render that was stopped also has `checkpoint`, the file `resume` takes. See [Resuming](#resuming-a-render) |
| `resume` | `file`, `progress_interval_ms` (optional) | Continues a render from its checkpoint (from 1.3.0). The command the render was started with is in the file, so nothing else is taken. Telemetry and the result are a render's, with `out`, and `checkpoint` again if it is stopped again |
| `batch` | `jobs`, `out` (optional), `out_format` (optional), `soundfonts` (optional), `sf_programs` (optional), `options` (optional), `progress_interval_ms` (optional) | Several MIDIs, each with its own soundfonts, one after another (from 1.3.0). When the batch ends: `jobs` (each with `job`, `midi`, `out`, `state`, `written`, and `summary` or `error`), `done`, `failed`, `cancelled`, `loads`, `wall_secs`. See [Batch](#batch) |
| `cancel` | `discard` (optional) | Stops the running render, batch or scan after its current block, or analytic preparation at its next cancellation check. `cancelling`: the id it stops. A raw soundfont load cannot be stopped part way. A `render` or `resume` that saves its progress saves it where it stops, so it can be resumed; with `"discard": true` it keeps none: no checkpoint is written, none from earlier is left, and neither is the audio kept beside it, so there is nothing to resume. The answer's `discard` says whether the stop was one (it is `false` for anything that has no progress to keep, a batch or a scan), and the render's `summary` then has `"discarded": true` and no `checkpoint`. A finished per-track stem stays. Anything but `true` or `false` is refused |
| `snapshot` | | The running render's current `progress` line |
| `set_interval` | `interval_ms` | How often a render sends `progress`: 0 for never (use `snapshot` instead), otherwise held to 10–60,000 ms. Applies to the running render and later ones. Result: the `interval_ms` applied |
| `status` | | `running` (`id` and `cmd`, or null), `loaded` (what `load_soundfonts` returned, or null), `interval_ms` |
| `check_update` | | Asks GitHub for the latest Kestrel release; downloads nothing. `current` (this Kestrel's version), `latest`, `newer` (whether `latest` is newer than `current`), `ring` (the person's update ring: `fast` for every release, `slow` for feature releases such as 1.2.0 only), `announce` (whether their ring wants to hear about `latest`: show a notice when this is true) and `url` (the release page to send people to). An error when offline, when GitHub doesn't answer within about 5 seconds, or when `KESTREL_NO_UPDATE_CHECK` is set; show nothing in any of those cases |
| `shutdown` | | Cancels anything running, answers, and exits |

### Soundfonts

Soundfonts in a list are layered in order: each one replaces whatever the ones before it define at the same bank and program. Put a General MIDI bank first and an instrument after it. `sf_programs` places the last soundfont on those programs of bank 0, spelled as `--sf-programs` takes it: `"0,1"` or `"0-7"`.

A render with no `soundfonts` uses the loaded ones. **A loaded soundfont is reused** by every render that would load it the same way, which saves the load time on each render (seconds on a large library). A render that changes an option marked `reloads_soundfonts`, such as `rate`, loads again, and that becomes the loaded set. (`volume` did too until 1.2.1; it is applied to the output now and never reloads.) A render that names different `soundfonts` does the same. From 1.2.3 there is one more case: without `pool_budget`, the budget is sized from the GPU, so a render whose `gpu_adapter` or `gpu_backend` picks a card with a different budget also loads again.

Analytic phase options, such as `"phase_mode": "analytic"` and `"phase_seed": 42`, reuse the loaded soundfont. Quadrature is prepared separately for each render, with log progress during `loading_soundfont` and cancellation checks inside preparation. Cancellation there creates no output file. See [analytic phase controls](docs/analytic-phase-rotation.md).

## Batch

`batch` renders several MIDIs, one after another, each with the soundfonts it is given: N MIDIs make N files. One MIDI on several sets is several jobs with the same `midi`. Every job is an ordinary render, so **each file is byte-identical to the same MIDI rendered by itself** with the same options. It is the same feature as `kestrel --force-cli render a.mid b.mid` and `kestrel --force-cli batch jobs.json`, and its rules are the ones in [docs/batch.md](docs/batch.md).

```json
{"id": 9, "cmd": "batch", "out": "C:/renders", "options": {"max_voices": 500000},
 "soundfonts": ["gm.sf2"],
 "jobs": [{"midi": "a.mid"},
          {"midi": "b.mid", "soundfonts": ["gm.sf2", "piano.sfz"], "sf_programs": "0,1"},
          {"midi": "c.mid", "out": "C:/special/c.flac", "seconds": 60}]}
```

- **`jobs`** is a non-empty list. Each job has `midi`, and optionally `soundfonts`, `sf_programs`, `out` and `seconds` (stop after that many seconds of audio). An unknown key is an error, naming it. A job's `soundfonts` replaces the batch's `soundfonts` **and** `sf_programs` together, because the programs belong to the last soundfont; a job that gives only `sf_programs` keeps the batch's soundfonts.
- **`soundfonts`** and **`sf_programs`** are what a job uses when it names none of its own. With neither, they are the soundfonts loaded with `load_soundfonts`.
- **`out`** is the folder for jobs with no `out` of their own, and `out_format` the container of those files by extension (`wav` when absent). A file is named after its MIDI, and after its soundfonts where two would share a name (`song (piano).wav`, `song (gm).wav`).
- **`options`** are shared by every job, checked as a render's are. `tracks`, `track` and `block_csv` are refused: they are per file.
- **Checked before anything starts:** a MIDI or soundfont that is not a file, two jobs that would write one file, a job with no soundfonts, a job with no output, a bad option. The answer is an error response naming the job. What is wrong *inside* a file, such as a MIDI that does not parse, fails that job alone.

**The order.** One job on the GPU at a time. Jobs are grouped by soundfont set, so each distinct set loads once and is freed before the next loads; the order the jobs run in is by set, and the `jobs` in the response follow the order you listed. A batch whose jobs all use the loaded soundfonts takes that bank and leaves it loaded. Any other batch unloads what was loaded first, so two large pools are never held at once, and leaves nothing loaded.

**Telemetry.** Each job sends what a render sends -- `phase`, `setup`, `progress`, `summary` or `failed` -- with the request's `id` and a `job` index (from 0). The batch adds:

| Type | When | Carries |
|---|---|---|
| `batch_set` | a soundfont set starts to load | `set` (from 1), `sets`, `soundfonts`, `first_job`, `jobs` (how many use it), `state`: `loading`. Not sent for a set that was already loaded |
| `batch_job` | a job starts, and again when it ends | `job`, `jobs`, `set`, `sets`, `state`: `started` (with `midi` and `out`), then `done`, `cancelled` or `failed`; the end carries `summary` or `message`, and `written`, where the audio is |

A job's lines run in order: `batch_job` `started`, its phases and progress, its `summary` (or `failed`), then `batch_job` with how it ended. A set that will not load fails every job that uses it, and no others: each of those gets its `started` and `failed` lines.

**Files in progress.** A job writes `name.partial.wav` (the extension stays last, because it picks the container) and takes its real name when it finishes, so a batch that is stopped never leaves something that looks complete.

**Control.** `cancel`, `snapshot` and `set_interval` act on the job running now. A cancel stops that job after its current block, leaves it as its `.partial.` file (`state` `cancelled`, `written` the path, since a cancelled render's file holds what was rendered), and starts none of the rest (`not_run`). A failed job does not stop the batch.

## Resuming a render

A `render` saves its progress as it goes, as often as the `checkpoint_every` option says (minutes, 10 unless told, 0 for only when it stops), and when it is stopped by `cancel`. The file is a `.krsm`: beside the output as `<file>.krsm`, or `stems.krsm` in the folder a render to stems goes in. A render that is cancelled answers with `cancelled: true` and `checkpoint`, its path (its `summary` telemetry line carries `checkpoint` too). A render that ends in an error leaves the last periodic checkpoint, and a per-track render saves what it had done as well.

`resume` takes that path and runs the same render again from there. **The file it writes is byte-for-byte the one an uninterrupted render writes** (an Opus file has a serial number in its header that ffmpeg draws at random each time, so for that format it is the audio that is the same). A finished render deletes its checkpoint.

**A render of one file** (from 1.3.0) saves the render itself, at a block boundary, and a resume goes on from the block it had come to rather than rendering the start again. Until it is whole, its audio is in `name.partial.<ext>` beside the output, and an encoded format also keeps the samples it feeds the encoder in `<file>.krsm.pcm`; both are the checkpoint's, and are removed with it. `out` is the real name only once the render is whole. A render with `block_csv` or an analytic `phase` option cannot save its progress, and renders without.

**A render with `tracks`** (one file per track, or `merge` for one mix) saves its tracks: those that were finished are not rendered again; those that were under way start again from their beginning, and a merge does not add what it already has.

It is **refused, with the reason in the error, and nothing starts**, if anything the render depended on has changed: this build of Kestrel (a rebuild counts), the MIDI, a soundfont or one of an SFZ's sample files, the tracks, the output format, the backend or the card, or the voice limit. A file that is not a checkpoint, or was cut off while it was written, is refused the same way. A render that has been resumed once can be stopped and resumed again. A batch does not save its progress.

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

Each render also writes a log file of its own, with the PC's and the account's names left out; see "FalconEye" in the README and [docs/falconeye.md](docs/falconeye.md). The `log` lines here are not affected.

Then the `response`. Log lines are written the moment they happen and the other lines within a tenth of a second, so a log line can arrive just ahead of the phase it belongs to.

**`progress`** carries: `phase`, `phase_secs`, `wall_secs`, `render_secs`, `audio_secs` (audio rendered so far), `notes` (note-ons read so far), `voices` (sounding now), `max_voices`, `peak_voices`, `stolen`, `dropped`, `peak_level` (before the limiter), `clipped`, `blocks`, `bytes_read`, `bytes_total`, `tracks`, `backend`, `adapter`, `device_bytes` (Kestrel's own GPU buffers), `gpu_memory` (the whole GPU's, Windows only: `dedicated_total`, `dedicated_used`, `shared_used`, `process_used`, `process_budget`), `host_rss_bytes`, `host_rss_peak_bytes`, and three figures worked out for you:

- `progress`: 0 to 1. It measures how much of the MIDI has been read, not audio time, because a MIDI's length isn't known until it has been read to the end, and render time follows the events anyway.
- `speed`: audio seconds per second, the "x realtime" figure.
- `eta_secs`: seconds left, once there is enough to estimate from.

Any of these can be `null` before it is known.

A per-track render -- the `tracks` option, with `merge` for one file or without it for a folder of stems -- renders many tracks at once, so its lines add `per_track` once it is rendering: `total`, `done` and `running` tracks, `stole` (finished tracks that stole voices), `blocks`, `silent_blocks`, `blocks_total`, `span_blocks`, `span_total`, `notes_total`, `length_secs`, `voices_each`, `merged`, and `now`, up to three of the busiest tracks still going, each with `track` (from 1), `name` and `secs` rendered. Its `progress` is 0.55 × `notes` out of `notes_total` + 0.45 × `span_blocks` out of `span_total`. `span_blocks` counts the blocks each track renders from its first note to its last. The time goes on the notes and on those blocks; the silent blocks outside them never reach the GPU. `notes_total` is 0 when `seconds` cuts the render short, and `progress` is then the spans alone. Its `speed` is of the file being made, `length_secs` times `progress` per second of rendering, which is the figure the summary ends on. The fields above are then summed over the tracks: `audio_secs` counts every track's audio, silence and all, and `peak_level` is the loudest track's. A normal render's lines have no `per_track`.

**The summary** has `bytes`, `audio_secs`, `wall_secs`, `notes`, `voices_spawned`, `peak_voices`, `stolen`, `dropped`, `peak_level`, `clipped` and `cancelled`. A cancelled render still ends normally: an opened file is closed properly and holds what was rendered, and `cancelled` is `true`. A stop that kept its progress names the `checkpoint`; one that was asked to discard it has `"discarded": true` instead. Cancellation during analytic preparation returns zero audio/bytes without creating a file.

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

To run one render and only watch it, without a session, add `--progress json` to a normal render command. It sends the same telemetry lines on stdout (without `id`s, and beginning with `hello`), sends log lines to stderr as text, and takes `{"interval_ms": N}`, `{"snapshot": true}` and `{"cancel": true}` on stdin. `{"cancel": true, "discard": true}` stops the render and keeps none of its progress, as `cancel`'s `discard` does; `discard` alone does nothing.

## Compatibility

`api` in `ready` is the protocol's version. Adding a command, a field or a line type does not change it; renaming or removing one, or changing what one means, does. Check it when you connect.

The `batch` command, and its `batch_set` and `batch_job` lines, were added in 1.3.0 without a change to `api`: a program written for 1.0 to 1.2 that ignores line types and fields it does not know is unaffected. So were the `resume` command, the `checkpoint` field of a stopped render, and the `checkpoint_every` option.
