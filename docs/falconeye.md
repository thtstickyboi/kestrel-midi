# FalconEye transparency report

FalconEye is Kestrel's error catching and logging, new in 1.2.2. This page
says everything it does on your computer: what it writes, what it reads,
what it hides, and what it does that security software might notice.

**The short version:**
- **Nothing leaves your computer.** FalconEye has no network code. It never
  uploads, phones home, or sends anything anywhere. Its files stay on your
  disk until you delete them or choose to send one to someone.
- **It hides who you are.** Your PC's name, your Windows account name and
  your account's security ID are replaced with `<pc>`, `<user>` and `<sid>`
  in everything it writes, file paths included. MIDI, soundfont and track
  names are kept, because a report is about them.
- **It changes nothing on your system.** No settings, no startup entries,
  and no background service. Nothing runs once Kestrel closes, apart from
  the watcher finishing a report, which takes a few seconds.

## Why it exists

In 1.2.1, several people's renders stopped partway through. Their GPUs were
reset by Windows, or Kestrel closed without a word. Nobody could say when
it stopped, with what settings, or what Windows recorded, so nobody could
say why. FalconEye records that, so the next report has an answer in it.

## What it writes, and when

### A log for each render

- **When:** every render from the guided renderer or the API. On the command
  line, only with `--log`.
- **Where:** `logs\<MIDI name> <date time>.log` beside `kestrel.exe`. If that
  folder can't be written, `%LOCALAPPDATA%\Kestrel\logs` (on Linux and
  macOS, `~/.local/state/kestrel/logs`). `KESTREL_LOG_DIR` picks another
  folder. The newest 100 are kept; older ones are deleted.
- **What's in it:**
  - the Kestrel version, the settings you rendered with, and the MIDI and
    soundfont files with their sizes;
  - your GPU and its driver version;
  - Kestrel's own messages;
  - about once a second, where the render was: audio time, voices, the
    longest wait on the GPU, and video memory in use;
  - how it ended: the summary, the error, or the crash with its backtrace.
- **Each line is written the moment it happens,** so a log survives a render
  that dies.

There is no switch in 1.2.2 to stop the guided renderer or the API from
writing logs. They are small text files, and deleting the `logs` folder is
always safe.

### The watcher

Each logged render starts a second, windowless `kestrel.exe`: FalconEye's
watcher. It exists for the one case a render cannot report itself, when
the render is gone.

- **While the render runs,** the render tells the watcher once a second that
  it is alive and how far it has got. It uses a private pipe between the two
  processes, and nothing else hears it.
- **When the render ends normally** (finished, cancelled, or stopped by an
  error it could report), the watcher exits without writing anything.
- **When the render dies instead,** the watcher reads the render's exit
  code. For a crash, that is the Windows error it died of. It adds a line
  to the log saying how the render ended. A closed window or Ctrl+C is
  noted as that and nothing more.
- **For a crash, or a render that stopped finishing blocks for a minute,** it
  also writes `<log> CRASH.txt` or `<log> HANG.txt`. That holds:
  - the log's last lines;
  - Windows' own records from the render's time: graphics driver resets,
    and Windows' record of Kestrel's crash. They come from the System and
    Application event logs, read with `wevtutil`;
  - the GPU's state from `nvidia-smi`, on NVIDIA GPUs.
- **For a crash inside native code (a graphics driver, for example), or a
  hang,** it writes a minidump, `<log> CRASH.dmp` or `HANG.dmp`, a few
  hundred KB. It holds each Kestrel thread's stack and the memory those
  stacks point at, which can include fragments of what Kestrel was working
  on, such as file paths. Your names are overwritten in it before it is
  saved. It is a dump of Kestrel's own render process only, never of any
  other program. The newest 5 dumps and 100 reports are kept.

### The machine report

Only when you ask for it: **Extras → Machine report**, or
`kestrel --force-cli report`. It shows what it collects before collecting
anything, and writes one file, `reports\kestrel-report-<date time>.zip`.

**What it collects** (a fixed list, not "everything it can find"):
- Kestrel's version and settings file;
- Windows' edition and build, the CPU, RAM and page file, the power plan,
  and whether a laptop is plugged in;
- every GPU on every graphics API: its limits, features, video memory and
  driver;
- on NVIDIA GPUs, `nvidia-smi`'s clocks, power limits, PCIe link and
  throttle reasons, and how many programs are using the GPU (not which);
- Windows' GPU timeout settings (`TdrDelay`, `TdrLevel`) and GPU scheduling,
  read from the registry;
- 30 days of graphics driver resets and Kestrel crashes, from Windows' event
  logs;
- your newest 10 render logs and FalconEye's reports from the last 30 days,
  and the newest minidump from those 30 days if there is one and it is
  under 16 MB (with your names overwritten, as above);
- **a GPU self-test**, which you're asked about first. It renders one block
  with every voice sounding, starting at 4,096 voices and doubling, and
  stops before any block could take half a second. Then it tells you how
  close your GPU runs to Windows' 2-second limit.

**What it never collects:** your files, the names of other programs,
network details (IP or MAC addresses), serial numbers, or environment
variables. Kestrel's test suite builds a real report and fails if it finds
an IP or MAC address, an environment dump, an account ID or a hidden name
in it.

### The administrator step

A separate question in the machine report, and the answer is no unless you
say yes. With `report --system` on the command line, or "y" in Extras,
Windows' own permission prompt appears. If you allow it, a second Kestrel
running as administrator **only reads**:
- the list of Windows' dumps of GPU resets (`C:\Windows\LiveKernelReports`)
  and blue screens (`C:\Windows\Minidump`): names, sizes and dates. **The
  dumps themselves are never copied**, because they hold raw system memory
  that nothing can check for personal data;
- Windows Error Reporting's short text reports of GPU resets and Kestrel
  crashes from the last 30 days, with names hidden.

It changes nothing and never renders as administrator. If you'd like
Windows to keep a full dump of each Kestrel crash, the report says which
setting does it; Kestrel leaves that to you.

### On Linux and macOS

The render log works the same. The watcher runs, but it can't yet read how
a render ended or the system's logs. The machine report has no operating
system section there, and there is no administrator step. Everything else
on this page is as written.

## What security software might notice, and why it's there

Kestrel is not code-signed yet, so antivirus judges it by what it does.
These are the things FalconEye does that such software watches for. None
of them happens without a reason:

| What | Why | Limits |
|---|---|---|
| Starts a second copy of itself with no window | The watcher: it has to outlive a render that crashes | One per logged render; exits when the render does |
| Reads another process's exit code | That's how a crash is told apart from a closed window | Only its own render's |
| Writes a minidump of another process | The same way crash reporters in web browsers work: a dump is written from outside the crashed process | Only its own render's, only after a crash or a hang |
| Installs a crash filter | So a native crash can wait while its dump is written, then carry on to Windows' own handling | Changes no exit code and no Windows behaviour |
| Reads the registry | Windows' version, the CPU and machine model, display driver dates, GPU timeout settings | Read only; the machine report only |
| Runs `wevtutil`, `nvidia-smi` and `powercfg` | Windows' event logs, the NVIDIA GPU's state, the power plan | Read-only queries |
| Asks for administrator permission | The administrator step | Only if you say yes, and only reads |

**What it never does:** use PowerShell, connect to the network, read any
other program's memory, write outside its `logs` and `reports` folders
(and a temporary folder it deletes), change a setting, or start with
Windows.

If your antivirus flags Kestrel anyway, please report it to the antivirus
vendor as a false positive, and open an issue so others know.

## Checking a download

From 1.2.2, every download carries a GitHub attestation: a signed record
that the file was built from this repository by its release workflow. With
the GitHub CLI:

```bash
gh attestation verify kestrel-1.2.2-windows-x64.zip --repo thtstickyboi/kestrel-midi
```

## The code

All of FalconEye is in [`src/falconeye/`](../src/falconeye). The only file in
it that uses `unsafe` is `winsys.rs`, which holds the Windows calls listed
above; each `unsafe` block there says why it is sound. The engine calls
FalconEye in two places: to open a render's log, and to ask DX12 why a GPU
was lost.

## Cleaning up

Delete `logs` and `reports` beside `kestrel.exe` (or under
`%LOCALAPPDATA%\Kestrel`) whenever you like. Kestrel recreates them as
needed and leaves nothing anywhere else.
