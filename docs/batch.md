# Batch rendering

Several MIDIs in one command, each with the soundfonts it is given: N MIDIs
make N files. New in 1.3.0.

A batch changes nothing about the audio. Every job is an ordinary render, so
**each file is byte-identical to that MIDI rendered on its own** with the same
flags, on both backends.

What a batch adds is the order and the loading:

- **One job at a time.** The card is mostly busy on one render and its memory
  is the limit, so jobs do not overlap.
- **Grouped by soundfont set.** A set is the ordered list of soundfonts plus
  `--sf-programs`. Each distinct set loads once, its jobs run back to back,
  and it is freed before the next set loads, so two big sets are never in
  memory together. The log and the results still follow the order you listed
  the jobs in.
- **A job that fails does not stop the others.** It is logged with its
  reason, the rest run, and the exit code is non-zero if any failed. A set
  that will not load fails the jobs that use it and no others.
- **A file in progress is `name.partial.wav`** (the extension stays last,
  because it picks the container) and takes its real name when the job
  finishes. A batch that is stopped leaves the job in hand as `.partial.` and
  never starts the rest, so nothing on disk looks finished that is not.

## Several MIDIs, one set of soundfonts

```bash
kestrel --force-cli render a.mid b.mid c.mid -s gm.sf2 -s piano.sfz -o renders/
```

`-o` is a folder (created if it is missing) and each file is named after its
MIDI: `renders/a.wav`. `--out-format flac` (or opus, ogg, mp3, m4a) picks
another container. Every flag applies to every job.

## Different soundfonts per MIDI: a batch file

```bash
kestrel --force-cli batch jobs.json
```

```json
{
  "version": 1,
  "out": "renders",
  "out_format": "wav",
  "args": ["--max-voices", "500000", "--limiter", "brickwall"],
  "soundfonts": ["gm.sf2"],
  "jobs": [
    { "midi": "song_a.mid" },
    { "midi": "song_b.mid", "soundfonts": ["gm.sf2", "piano.sfz"], "sf_programs": "0,1" },
    { "midi": "song_c.mid", "out": "special/c.flac", "seconds": 60 }
  ]
}
```

| key | meaning |
|---|---|
| `version` | Must be `1`. |
| `out` | The folder for jobs with no `out` of their own. |
| `out_format` | The container of those files, by extension. `wav` if absent. |
| `args` | Flags every job shares, as typed after `render`. They go through the same definition a render's do. They cannot name the MIDI, the soundfonts or the output. |
| `soundfonts`, `sf_programs` | What a job uses when it names none of its own. |
| `jobs[].midi` | The MIDI. |
| `jobs[].soundfonts` | Replaces the file's `soundfonts` **and** its `sf_programs` together: the programs belong to the last soundfont, so they do not outlive a change of soundfonts. A job that gives only `sf_programs` keeps the file's soundfonts. |
| `jobs[].out` | Where this one goes, instead of `out`/`<midi name>.<ext>`. |
| `jobs[].seconds` | Stop this one after this many seconds of audio. |

**Relative paths are read against the folder the file is in**, not the
directory the command was run from, so a file means the same wherever it is
run. **Unknown keys are refused**, not ignored: a misspelt key that silently
did nothing would render the wrong thing for hours.

### One MIDI, several soundfont sets

Give the same MIDI as several jobs with different `soundfonts`. Files that
would share a name are all named by their soundfonts, whichever is listed
first:

```json
"jobs": [
  { "midi": "song.mid", "soundfonts": ["piano.sfz"] },
  { "midi": "song.mid", "soundfonts": ["gm.sf2"] }
]
```

writes `song (piano).wav` and `song (gm).wav`.

## The guided renderer

Menu item 1, **Single / Multiple MIDIs**, takes one MIDI or several in its file
picker. With several, step 2 asks:

- **[1] One set for all** is the usual soundfont step, once.
- **[2] A different set for each MIDI** is a table in the terminal, a row for
  each file to write, driven with the keys shown above it. Nothing is typed.
  The arrow keys move the cursor; **Space** marks a row and moves on; **Enter**
  gives the marked rows (or the one under the cursor) soundfonts, which offers a
  set already picked or opens the file picker; **A** marks every row; **D** adds
  the MIDI under the cursor again, below it, to give it another set; **X**
  removes a row (never a MIDI's last one); **C** clears; **G** starts, once every
  row has soundfonts, and otherwise moves to the first that has none; **Esc**
  goes back to the menu. Long lists scroll, with PageUp, PageDown, Home and End.

With one MIDI, pressing **S** on the soundfont step's confirm line opens the
same table with that MIDI's first set already in, to add it again on others.

The voices, format and destination are asked once for all of them, and the
flags line applies to every file. Each file is named after its MIDI, and after
its soundfonts where two would clash, and **never over a file already in the
folder** (a clash gets the time in brackets, as a single render's does). The
progress screen shows the job on the screen with a `Batch  job 3 of 6 · set 2
of 3 · 2 done` row; Ctrl+C twice stops the batch, keeps the files already
finished and removes the one in progress. The done screen lists every file,
with the reason for any that failed.

A soundfont is loaded once to be described when it is picked. When every file
uses one set, the render takes that load; with several sets each set loads
again when its jobs start, because holding them all would put every set in
memory at once.

## From another program

The API has a `batch` command (see [API.md](../API.md)), and a batch on the
command line takes `--progress json`: `render a.mid b.mid ... -o DIR --progress
json` reports each job's telemetry with a `job` index, and `batch_set`,
`batch_job` and `batch_summary` lines. It is the same feed the API's `batch`
sends.

## What is refused before anything renders

- an input that is not a file, with the job named;
- two jobs that would write the same file (compared without case), or a
  file the batch also reads;
- `--track` and `--tracks`, which render one MIDI by itself;
- `--block-csv`, `--progress json` in a batch *file*'s `args`, and a render's
  flags in `args` that name its own MIDI, soundfont or output;
- `--31edo`, or a MIDI that carries the note keys over 127 that file format
  uses: the option applies to every file of a batch, so one such file would
  move the others' notes. Render it on its own;
- a file with an unknown key, the wrong `version`, or a job with no soundfonts.

What is wrong *inside* a file -- a MIDI that does not parse, a soundfont that
will not load -- is only found by reading it, and fails that job alone.

## Not in 1.3.0

- Rendering jobs in parallel, or preparing the next set while one renders.
- Batches of per-track renders.
- Merging several MIDIs into one file.
