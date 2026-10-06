// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Locating ffmpeg, and encoding a render through it. \[1\]

use anyhow::{bail, Context, Result};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where a resolved ffmpeg came from. Reported so that a surprising version \[2\]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// `--ffmpeg`.
    Flag,
    /// The `FFMPEG` environment variable.
    Env,
    /// Beside the Kestrel executable, or in `ffmpeg/` under it.
    BesideExe,
    /// Found on `PATH`.
    Path,
}

impl Source {
    pub fn describe(self) -> &'static str {
        match self {
            Source::Flag => "--ffmpeg",
            Source::Env => "the FFMPEG environment variable",
            Source::BesideExe => "beside the kestrel executable",
            Source::Path => "PATH",
        }
    }
}

/// A located ffmpeg that has been run at least once.
#[derive(Debug, Clone)]
pub struct Ffmpeg {
    pub path: PathBuf,
    pub source: Source,
    /// The version token, e.g. `N-122664-g3ab8b976c1-20260206` or `7.1`.
    pub version: String,
}

/// The platform's executable name.
pub fn exe_name() -> &'static str {
    if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    }
}

/// A container Kestrel can write, and the settings it writes it with. \[3\]
#[derive(Debug)]
pub struct Preset {
    /// Lower-case, no dot.
    pub ext: &'static str,
    /// Encoder that must be present in the ffmpeg build.
    pub encoder: &'static str,
    /// Arguments between the input and the output path.
    pub args: &'static [&'static str],
    /// Shown once per render, so the choice is never a silent one.
    pub note: &'static str,
    /// Whether the codec reconstructs something other than what it was given. \[4\]
    pub lossy: bool,
}

/// Default brickwall ceiling for a lossy container, in dBFS. \[5\]
pub const LOSSY_CEILING_DB: f64 = -1.0;

pub const PRESETS: &[Preset] = &[
    Preset {
        ext: "opus",
        encoder: "libopus",
        // [6]
        args: &["-c:a", "libopus", "-b:a", "160k"],
        note: "libopus VBR 160 kbps",
        lossy: true,
    },
    Preset {
        ext: "mp3",
        encoder: "libmp3lame",
        // [7]
        args: &["-c:a", "libmp3lame", "-q:a", "0"],
        note: "LAME V0 (~245 kbps VBR)",
        lossy: true,
    },
    Preset {
        ext: "ogg",
        encoder: "libvorbis",
        // 189 kbps measured.
        args: &["-c:a", "libvorbis", "-q:a", "8"],
        note: "libvorbis q8 (~190 kbps VBR)",
        lossy: true,
    },
    Preset {
        ext: "flac",
        encoder: "flac",
        // [8]
        args: &["-c:a", "flac", "-compression_level", "8"],
        note: "FLAC level 8, 24-bit (lossless relative to 24-bit, not to the f32 mix)",
        lossy: false,
    },
    Preset {
        ext: "m4a",
        encoder: "aac",
        // [9]
        args: &["-c:a", "aac", "-b:a", "256k"],
        note: "native AAC 256 kbps",
        lossy: true,
    },
];

pub fn preset_for(ext: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.ext == ext)
}

/// Find an ffmpeg without running it. Returns the first candidate by \[10\]
pub fn locate(explicit: Option<&Path>) -> Result<(PathBuf, Source)> {
    if let Some(p) = explicit {
        if !p.is_file() {
            bail!("--ffmpeg {}: no such file", p.display());
        }
        return Ok((p.to_path_buf(), Source::Flag));
    }

    if let Some(v) = std::env::var_os("FFMPEG") {
        // [11]
        if !v.is_empty() {
            let p = PathBuf::from(&v);
            if !p.is_file() {
                bail!(
                    "FFMPEG is set to {}, which is not a file.\n\
                     Unset it to search PATH instead.",
                    p.display()
                );
            }
            return Ok((p, Source::Env));
        }
    }

    // [12]
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [dir.join(exe_name()), dir.join("ffmpeg").join(exe_name())] {
                if cand.is_file() {
                    return Ok((cand, Source::BesideExe));
                }
            }
        }
    }

    if let Some(p) = search_path(exe_name()) {
        return Ok((p, Source::Path));
    }

    bail!(
        "ffmpeg not found.\n\
         Searched: --ffmpeg, the FFMPEG environment variable, beside the kestrel \
         executable, and PATH.\n\
         Install it with one of:\n    \
         winget install Gyan.FFmpeg        (Windows)\n    \
         brew install ffmpeg               (macOS)\n    \
         sudo apt install ffmpeg           (Debian/Ubuntu)\n\
         or point --ffmpeg at an existing copy."
    )
}

/// Walk `PATH` for `name`. `Command` would do this itself on exec, but the \[13\]
fn search_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|c| c.is_file())
}

/// Run `-version` and keep the version token. Doubles as the check that the \[14\]
pub fn probe(path: &Path, source: Source) -> Result<Ffmpeg> {
    let out = Command::new(path)
        .args(["-hide_banner", "-version"])
        .output()
        .with_context(|| format!("running {}", path.display()))?;

    if !out.status.success() {
        bail!(
            "{} exited with {} when asked for its version; it does not look \
             like a working ffmpeg",
            path.display(),
            out.status
        );
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let first = text.lines().next().unwrap_or_default();
    // [15]
    let version = first
        .split_whitespace()
        .skip_while(|w| *w != "version")
        .nth(1)
        .unwrap_or("unknown")
        .to_string();

    if !first.contains("ffmpeg version") {
        bail!(
            "{} ran, but its first line of -version output was {first:?}; \
             expected an ffmpeg banner",
            path.display()
        );
    }

    Ok(Ffmpeg {
        path: path.to_path_buf(),
        source,
        version,
    })
}

/// `locate` then `probe`, which is what every caller outside tests wants.
pub fn find(explicit: Option<&Path>) -> Result<Ffmpeg> {
    let (path, source) = locate(explicit)?;
    probe(&path, source)
}

impl Ffmpeg {
    /// Encoder names this build advertises. One `-encoders` call; the caller \[16\]
    pub fn encoders(&self) -> Result<Vec<String>> {
        let out = Command::new(&self.path)
            .args(["-hide_banner", "-encoders"])
            .output()
            .with_context(|| format!("running {} -encoders", self.path.display()))?;
        if !out.status.success() {
            bail!("{} -encoders exited with {}", self.path.display(), out.status);
        }
        let text = String::from_utf8_lossy(&out.stdout);
        Ok(text
            .lines()
            // [17]
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                let flags = it.next()?;
                let name = it.next()?;
                (flags.len() == 6 && flags.starts_with('A')).then(|| name.to_string())
            })
            .collect())
    }

    /// Which `PRESETS` this build cannot satisfy.
    pub fn missing_encoders(&self) -> Result<Vec<&'static Preset>> {
        let have = self.encoders()?;
        Ok(PRESETS
            .iter()
            .filter(|p| !have.iter().any(|h| h == p.encoder))
            .collect())
    }

    /// Check this build can satisfy one preset, before a render commits to it.
    pub fn require(&self, preset: &Preset) -> Result<()> {
        if self.encoders()?.iter().any(|h| h == preset.encoder) {
            return Ok(());
        }
        bail!(
            "{} was built without {}, so it cannot write .{}.\n\
             Install a fuller build, or render to .wav and encode separately.",
            self.path.display(),
            preset.encoder,
            preset.ext
        )
    }

    /// Start encoding to `out`. Raw `f32le` stereo goes in on stdin. \[18\]
    pub fn encode_to(&self, out: &Path, rate: u32, preset: &Preset) -> Result<Encoder> {
        Encoder::spawn(&self.path, out, rate, preset)
    }
}

/// A running ffmpeg being fed raw PCM on stdin. \[19\]
pub struct Encoder {
    child: std::process::Child,
    /// Taken in `finish`, which is what closes the pipe and tells ffmpeg to \[20\]
    stdin: Option<std::io::BufWriter<std::process::ChildStdin>>,
    /// Drained on a thread. ffmpeg blocks once a full stderr pipe stops being \[21\]
    stderr: Option<std::thread::JoinHandle<String>>,
    path: PathBuf,
    out: PathBuf,
    /// Reused interleaved byte buffer, so a block does not allocate.
    scratch: Vec<u8>,
    bytes_in: u64,
}

impl Encoder {
    fn spawn(ffmpeg: &Path, out: &Path, rate: u32, preset: &Preset) -> Result<Self> {
        use std::io::Read;
        use std::process::Stdio;

        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-hide_banner", "-loglevel", "error", "-y"])
            // [22]
            .args(["-f", "f32le", "-ar", &rate.to_string(), "-ac", "2", "-i", "-"])
            .args(preset.args)
            .arg(out)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("starting {}", ffmpeg.display()))?;

        let mut err = child.stderr.take().expect("stderr was piped");
        let stderr = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = err.read_to_string(&mut s);
            s
        });

        Ok(Encoder {
            stdin: Some(std::io::BufWriter::with_capacity(
                1 << 16,
                child.stdin.take().expect("stdin was piped"),
            )),
            child,
            stderr: Some(stderr),
            path: ffmpeg.to_path_buf(),
            out: out.to_path_buf(),
            scratch: Vec::new(),
            bytes_in: 0,
        })
    }

    /// Feed one interleaved stereo block.
    pub fn write_block(&mut self, samples: &[f32]) -> Result<()> {
        use std::io::Write;
        self.scratch.clear();
        self.scratch.reserve(samples.len() * 4);
        for s in samples {
            self.scratch.extend_from_slice(&s.to_le_bytes());
        }
        let w = self.stdin.as_mut().expect("write after finish");
        if let Err(e) = w.write_all(&self.scratch) {
            // [23]
            return Err(self.fail(Some(e)));
        }
        self.bytes_in += self.scratch.len() as u64;
        Ok(())
    }

    /// Collect ffmpeg's own complaint, which is the useful half of any failure \[24\]
    fn fail(&mut self, io: Option<std::io::Error>) -> anyhow::Error {
        let _ = self.child.kill();
        let err = self
            .stderr
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_default();
        let err = err.trim();
        let prefix = match io {
            Some(e) => format!("writing to {}: {e}", self.path.display()),
            None => format!("{} failed", self.path.display()),
        };
        if err.is_empty() {
            anyhow::anyhow!("{prefix}")
        } else {
            anyhow::anyhow!("{prefix}\nffmpeg said:\n{err}")
        }
    }

    /// Close the pipe, wait for the encode to finish, and return the size of \[25\]
    pub fn finish(mut self) -> Result<u64> {
        use std::io::Write;
        let mut w = self.stdin.take().expect("finish called twice");
        if let Err(e) = w.flush() {
            return Err(self.fail(Some(e)));
        }
        drop(w); // EOF; ffmpeg flushes its encoder and exits.

        let status = self
            .child
            .wait()
            .with_context(|| format!("waiting for {}", self.path.display()))?;
        if !status.success() {
            return Err(self.fail(None));
        }
        // [26]
        if let Some(h) = self.stderr.take() {
            if let Ok(s) = h.join() {
                let s = s.trim();
                if !s.is_empty() {
                    log::warn!("ffmpeg: {s}");
                }
            }
        }
        Ok(std::fs::metadata(&self.out).map(|m| m.len()).unwrap_or(0))
    }
}

/// True when `name` looks like a path rather than a bare command. Used to keep \[27\]
pub fn looks_like_path(name: &OsStr) -> bool {
    let s = name.to_string_lossy();
    s.contains('/') || s.contains('\\')
}


// ---------------------------------------------------------------------------
// get-ffmpeg
// ---------------------------------------------------------------------------

/// SHA-256, hand-written so that fetching ffmpeg costs no dependency.
///
/// The alternative was `sha2`, two crates for something this project can carry
/// in a hundred lines and pin against the FIPS-180-4 vectors. Same reasoning as
/// the hand-written decoders in `wav.rs`.
pub mod sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub struct Hasher {
        h: [u32; 8],
        buf: [u8; 64],
        len: usize,
        total: u64,
    }

    impl Default for Hasher {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Hasher {
        pub fn new() -> Self {
            Hasher {
                h: [
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                    0x1f83d9ab, 0x5be0cd19,
                ],
                buf: [0; 64],
                len: 0,
                total: 0,
            }
        }

        pub fn update(&mut self, mut data: &[u8]) {
            self.total = self.total.wrapping_add(data.len() as u64);
            while !data.is_empty() {
                let take = (64 - self.len).min(data.len());
                self.buf[self.len..self.len + take].copy_from_slice(&data[..take]);
                self.len += take;
                data = &data[take..];
                if self.len == 64 {
                    let block = self.buf;
                    self.compress(&block);
                    self.len = 0;
                }
            }
        }

        fn compress(&mut self, block: &[u8; 64]) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            for (i, v) in [a, b, c, d, e, f, g, h].into_iter().enumerate() {
                self.h[i] = self.h[i].wrapping_add(v);
            }
        }

        pub fn finish(mut self) -> String {
            // Captured before padding, which goes through `update` and would
            // otherwise be counted into the length field it is padding for.
            let bits = self.total.wrapping_mul(8);
            self.update(&[0x80]);
            while self.len != 56 {
                self.update(&[0]);
            }
            let at = self.len;
            self.buf[at..at + 8].copy_from_slice(&bits.to_be_bytes());
            let block = self.buf;
            self.compress(&block);
            self.h.iter().map(|w| format!("{w:08x}")).collect()
        }
    }

    pub fn hex_of_file(path: &std::path::Path) -> anyhow::Result<String> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut h = Hasher::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Ok(h.finish())
    }
}

/// One fetchable ffmpeg, per platform.
///
/// Deliberately an *essentials* build. It carries libopus, libmp3lame and
/// libvorbis, which with the native flac and aac encoders is every codec
/// `PRESETS` names; a full build is roughly twice the download for encoders
/// nothing here uses.
pub struct Release {
    pub platform: &'static str,
    pub url: &'static str,
    /// Pinned digest of the archive, checked before anything is extracted or \[28\]
    pub sha256: Option<&'static str>,
    pub size_hint: &'static str,
    pub license: &'static str,
    pub origin: &'static str,
}

/// **No pins are shipped, on purpose.** These URLs are rolling "latest release" \[29\]
pub const RELEASES: &[Release] = &[
    Release {
        platform: "windows-x86_64",
        url: "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        sha256: None,
        // [30]
        size_hint: "106 MiB download, 98 MiB installed (measured)",
        license: "GPLv3 (gyan.dev essentials build)",
        origin: "gyan.dev, linked from ffmpeg.org/download.html",
    },
    Release {
        platform: "linux-x86_64",
        url: "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        sha256: None,
        size_hint: "~80 MiB download, estimated -- unmeasured",
        license: "GPLv3 (John Van Sickle static build)",
        origin: "johnvansickle.com, linked from ffmpeg.org/download.html",
    },
    Release {
        platform: "macos-x86_64",
        url: "https://evermeet.cx/ffmpeg/getrelease/zip",
        sha256: None,
        size_hint: "~30 MiB download, estimated -- unmeasured",
        license: "GPLv3 (evermeet.cx build)",
        origin: "evermeet.cx, linked from ffmpeg.org/download.html",
    },
];

/// The release matching the host, if there is one.
pub fn release_for_host() -> Option<&'static Release> {
    let want = if cfg!(all(windows, target_arch = "x86_64")) {
        "windows-x86_64"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "linux-x86_64"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "macos-x86_64"
    } else {
        return None;
    };
    RELEASES.iter().find(|r| r.platform == want)
}

/// Where `get-ffmpeg` installs: an `ffmpeg/` directory beside the Kestrel \[31\]
pub fn install_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating the kestrel executable")?;
    let dir = exe
        .parent()
        .context("the kestrel executable has no parent directory")?;
    Ok(dir.join("ffmpeg"))
}

/// Compare a computed digest against what the operator or the build pinned. \[32\]
pub fn verify(release: &Release, got: &str, accept: Option<&str>) -> Result<()> {
    let want = accept.or(release.sha256);
    let Some(want) = want else {
        bail!(
            "downloaded archive has SHA-256\n    {got}\n\
             but this build pins no digest for {}, so it will not be trusted.\n\
             Check that against the checksum published at {}, and if it matches, \
             re-run with:\n    --accept-hash {got}",
            release.platform,
            release.origin
        );
    };
    let want = want.trim().to_ascii_lowercase();
    if want != got {
        bail!(
            "SHA-256 mismatch -- the download will not be used.\n  \
             expected {want}\n  got      {got}\n\
             Either the upstream release changed, or the file was tampered with \
             in transit. Nothing has been extracted or run."
        );
    }
    Ok(())
}

/// Run a helper and fail with its stderr rather than only an exit code.
fn run_tool(what: &str, cmd: &mut Command) -> Result<()> {
    let out = cmd
        .output()
        .with_context(|| format!("running {what}; is it installed and on PATH?"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim();
        bail!(
            "{what} failed ({}){}",
            out.status,
            if err.is_empty() {
                String::new()
            } else {
                format!(":\n{err}")
            }
        );
    }
    Ok(())
}

/// Download `url` to `dest` with the platform's own curl. \[33\]
pub fn download(url: &str, dest: &Path) -> Result<()> {
    run_tool(
        "curl",
        Command::new("curl").args([
            "-fSL",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--tlsv1.2",
            "--retry",
            "2",
            "-o",
            &dest.to_string_lossy(),
            url,
        ]),
    )
}

/// Pull just the ffmpeg binary out of `archive` into `into`. \[34\]
pub fn extract_binary(archive: &Path, into: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(into).with_context(|| format!("creating {}", into.display()))?;
    run_tool(
        "tar",
        Command::new("tar").args([
            "-xf",
            &archive.to_string_lossy(),
            "-C",
            &into.to_string_lossy(),
            // [35]
            &format!("*{}", exe_name()),
        ]),
    )?;

    // [36]
    let found = find_binary(into, 6)
        .with_context(|| format!("no {} found under {}", exe_name(), into.display()))?;
    let want = into.join(exe_name());
    if found != want {
        std::fs::rename(&found, &want)
            .with_context(|| format!("moving {} to {}", found.display(), want.display()))?;
        // [37]
        if let Some(mut p) = found.parent() {
            while p != into && std::fs::remove_dir(p).is_ok() {
                match p.parent() {
                    Some(up) => p = up,
                    None => break,
                }
            }
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = std::fs::metadata(&want)?.permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&want, perm)?;
    }
    Ok(want)
}

fn find_binary(dir: &Path, depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let mut dirs = Vec::new();
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_file() && p.file_name().map(|n| n == exe_name()).unwrap_or(false) {
            return Some(p);
        }
        if p.is_dir() {
            dirs.push(p);
        }
    }
    dirs.into_iter().find_map(|d| find_binary(&d, depth - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_path_that_does_not_exist_is_an_error_not_a_fallthrough() {
        let e = locate(Some(Path::new("definitely/not/here/ffmpeg.exe")))
            .unwrap_err()
            .to_string();
        assert!(e.contains("no such file"), "{e}");
        // The whole point: it must not have silently gone looking on PATH.
        assert!(!e.contains("Searched"), "{e}");
    }

    #[test]
    fn probing_something_that_is_not_ffmpeg_fails_clearly() {
        // Any file that exists but is not an executable ffmpeg.
        let not_ffmpeg = Path::new("Cargo.toml");
        assert!(not_ffmpeg.is_file(), "test fixture moved");
        assert!(probe(not_ffmpeg, Source::Flag).is_err());
    }

    #[test]
    fn source_labels_are_distinct() {
        let all = [Source::Flag, Source::Env, Source::BesideExe, Source::Path];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.describe(), b.describe());
            }
        }
    }

    /// FIPS-180-4 vectors, plus a multi-block case that catches a wrong length \[38\]
    #[test]
    fn sha256_matches_the_published_vectors() {
        fn hex(s: &[u8]) -> String {
            let mut h = sha256::Hasher::new();
            h.update(s);
            h.finish()
        }
        assert_eq!(
            hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // A million a s: three-plus blocks and a length that needs 64 bits.
        let mut h = sha256::Hasher::new();
        for _ in 0..1000 {
            h.update(&[b'a'; 1000]);
        }
        assert_eq!(
            h.finish(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        // Same input, awkward chunking: must not depend on call boundaries.
        let mut h = sha256::Hasher::new();
        for c in b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".chunks(7) {
            h.update(c);
        }
        assert_eq!(
            h.finish(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// Fails closed: no pin in the build and none supplied means refuse.
    #[test]
    fn an_unpinned_release_is_refused_and_names_the_digest() {
        let r = &RELEASES[0];
        assert!(r.sha256.is_none(), "shipping a pin changes this test");
        let e = verify(r, "abc123", None).unwrap_err().to_string();
        assert!(e.contains("abc123"), "{e}");
        assert!(e.contains("--accept-hash"), "{e}");
    }

    #[test]
    fn a_wrong_accepted_digest_is_refused() {
        let r = &RELEASES[0];
        let e = verify(r, "aaaa", Some("bbbb")).unwrap_err().to_string();
        assert!(e.contains("mismatch"), "{e}");
        assert!(e.contains("Nothing has been extracted or run"), "{e}");
    }

    #[test]
    fn a_matching_accepted_digest_passes_case_insensitively() {
        let r = &RELEASES[0];
        assert!(verify(r, "deadbeef", Some("DEADBEEF")).is_ok());
        assert!(verify(r, "deadbeef", Some("  deadbeef  ")).is_ok());
    }

    #[test]
    fn there_is_a_release_for_this_host() {
        assert!(release_for_host().is_some());
    }

    /// The wildcard extract plus relocate, which is the fiddly half of \[39\]
    #[test]
    fn extract_binary_finds_and_flattens_a_versioned_archive() {
        let base = std::env::temp_dir().join(format!("kestrel-xtest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let nested = base.join("src").join("ffmpeg-9.9-essentials_build").join("bin");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join(exe_name()), b"not really ffmpeg").unwrap();
        // [40]
        std::fs::write(nested.join("ffplay-decoy.txt"), b"x").unwrap();

        let archive = base.join("a.tar");
        let st = std::process::Command::new("tar")
            .args(["-cf", &archive.to_string_lossy(), "-C",
                   &base.join("src").to_string_lossy(), "."])
            .status()
            .expect("tar must be on PATH for this test");
        assert!(st.success());

        let into = base.join("into");
        let got = extract_binary(&archive, &into).unwrap();

        assert_eq!(got, into.join(exe_name()));
        assert_eq!(std::fs::read(&got).unwrap(), b"not really ffmpeg");
        assert!(!into.join("ffmpeg-9.9-essentials_build").exists(),
                "the version directory should have been cleaned up");
        assert!(!into.join("ffplay-decoy.txt").exists(),
                "only the binary should have been extracted");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The bug this guards, found 2026-09-02: `.opus` rendered at the 0 dBFS \[41\]
    #[test]
    fn every_lossy_container_is_flagged_and_flac_is_not() {
        assert!(!preset_for("flac").unwrap().lossy, "flac is lossless");
        for ext in ["opus", "mp3", "ogg", "m4a"] {
            assert!(preset_for(ext).unwrap().lossy, "{ext} is lossy");
        }
        // [42]
        assert!(
            (-2.0..=-0.5).contains(&LOSSY_CEILING_DB),
            "{LOSSY_CEILING_DB}"
        );
    }

    #[test]
    fn a_bare_command_name_is_not_a_path() {
        assert!(!looks_like_path(OsStr::new("ffmpeg")));
        assert!(looks_like_path(OsStr::new("./ffmpeg")));
        assert!(looks_like_path(OsStr::new(r"C:\tools\ffmpeg.exe")));
    }
}
