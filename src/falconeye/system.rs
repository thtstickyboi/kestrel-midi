// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The administrator step: what only an administrator can read about the \[1\]

use super::redact::Redactor;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Reports older than this are left out.
const DAYS: u64 = 30;
/// At most this many `.wer` texts.
const WER_MAX: usize = 20;

/// Collect into `out`: `system.txt`, and `wer/<report>.txt`. Run elevated.
pub fn collect(out: &Path) -> Result<()> {
    std::fs::create_dir_all(out.join("wer")).with_context(|| format!("creating {}", out.display()))?;
    let redact = Redactor::from_env();
    let root = std::env::var_os("SystemRoot").map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from);
    let data = std::env::var_os("ProgramData").map_or_else(|| PathBuf::from(r"C:\ProgramData"), PathBuf::from);
    let mut text = String::new();

    text.push_str("== LiveKernelReports: GPU resets (WATCHDOG) and other live kernel events ==\n");
    text.push_str("(dumps listed, not copied: they hold raw kernel memory)\n");
    text.push_str(&listing(&root.join("LiveKernelReports")));
    text.push_str("\n== Minidump: blue screens ==\n");
    text.push_str(&listing(&root.join("Minidump")));

    text.push_str("\n== Windows Error Reporting: GPU resets and kestrel.exe crashes, last 30 days ==\n");
    let since = SystemTime::now() - Duration::from_secs(DAYS * 24 * 3600);
    let mut reports: Vec<(SystemTime, PathBuf)> = Vec::new();
    for sub in ["ReportArchive", "ReportQueue"] {
        let dir = data.join(r"Microsoft\Windows\WER").join(sub);
        match std::fs::read_dir(&dir) {
            Ok(entries) => {
                for e in entries.flatten() {
                    let name = e.file_name().to_string_lossy().to_ascii_lowercase();
                    let wanted = name.contains("kestrel") || name.starts_with("kernel_") || name.contains("livekernelevent");
                    let when = e.metadata().ok().and_then(|m| m.modified().ok());
                    if let (true, Some(t)) = (wanted, when) {
                        if t >= since {
                            reports.push((t, e.path()));
                        }
                    }
                }
            }
            Err(e) => text.push_str(&format!("{}: {e}\n", dir.display())),
        }
    }
    reports.sort_by_key(|r| std::cmp::Reverse(r.0));
    let mut copied = 0;
    for (_, dir) in reports.iter().take(WER_MAX) {
        let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match std::fs::read(dir.join("Report.wer")) {
            Ok(bytes) => {
                let body = redact.apply(&decode_wer(&bytes));
                std::fs::write(out.join("wer").join(format!("{name}.txt")), body)?;
                text.push_str(&format!("{name}: copied\n"));
                copied += 1;
            }
            Err(e) => text.push_str(&format!("{name}: {e}\n")),
        }
    }
    if reports.is_empty() {
        text.push_str("(none)\n");
    } else if reports.len() > WER_MAX {
        text.push_str(&format!("({} older ones not read)\n", reports.len() - WER_MAX));
    }
    text.push_str(&format!("{copied} copied\n"));

    text.push_str(
        "\nNothing was changed. To have Windows keep a full dump of each kestrel.exe crash, set \
         LocalDumps for kestrel.exe under HKLM\\SOFTWARE\\Microsoft\\Windows\\Windows Error \
         Reporting\\LocalDumps; Kestrel does not do this for you.\n",
    );
    std::fs::write(out.join("system.txt"), redact.apply(&text))?;
    Ok(())
}

/// Every file in `dir` and one level of folders below it, newest first: \[2\]
fn listing(dir: &Path) -> String {
    let mut found: Vec<(SystemTime, String, u64)> = Vec::new();
    let mut walk = |d: &Path, prefix: &str| -> std::io::Result<Vec<PathBuf>> {
        let mut subdirs = Vec::new();
        for e in std::fs::read_dir(d)?.flatten() {
            let Ok(m) = e.metadata() else { continue };
            let name = format!("{prefix}{}", e.file_name().to_string_lossy());
            if m.is_dir() {
                subdirs.push(e.path());
            } else {
                found.push((m.modified().unwrap_or(SystemTime::UNIX_EPOCH), name, m.len()));
            }
        }
        Ok(subdirs)
    };
    let subdirs = match walk(dir, "") {
        Ok(s) => s,
        Err(e) => return format!("{}: {e}\n", dir.display()),
    };
    for s in subdirs {
        let prefix = format!("{}\\", s.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
        let _ = walk(&s, &prefix);
    }
    if found.is_empty() {
        return "(empty)\n".into();
    }
    found.sort_by_key(|f| std::cmp::Reverse(f.0));
    let mut out = String::new();
    for (t, name, size) in found.iter().take(50) {
        let when: chrono::DateTime<chrono::Local> = (*t).into();
        out.push_str(&format!("{}  {:>10} KiB  {name}\n", when.format("%Y-%m-%d %H:%M"), size.div_ceil(1024)));
    }
    if found.len() > 50 {
        out.push_str(&format!("({} older not listed)\n", found.len() - 50));
    }
    out
}

/// A `Report.wer` is UTF-16LE with a byte-order mark; read anything else as \[3\]
fn decode_wer(bytes: &[u8]) -> String {
    match bytes {
        [0xFF, 0xFE, rest @ ..] => {
            let units: Vec<u16> = rest.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            String::from_utf16_lossy(&units)
        }
        _ => String::from_utf8_lossy(bytes).into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wer_report_decodes_from_utf16() {
        let mut bytes = vec![0xFF, 0xFE];
        bytes.extend("EventType=LiveKernelEvent\r\n".encode_utf16().flat_map(u16::to_le_bytes));
        assert_eq!(decode_wer(&bytes), "EventType=LiveKernelEvent\r\n");
        assert_eq!(decode_wer(b"plain"), "plain");
    }

    #[test]
    fn a_listing_names_what_is_there_and_says_when_it_cannot() {
        let d = std::env::temp_dir().join(format!("kestrel_system_listing_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("WATCHDOG")).unwrap();
        std::fs::write(d.join("WATCHDOG").join("WATCHDOG-20260926-2029.dmp"), vec![0u8; 2048]).unwrap();
        let l = listing(&d);
        assert!(l.contains(r"WATCHDOG\WATCHDOG-20260926-2029.dmp") && l.contains("2 KiB"), "{l}");
        assert!(listing(&d.join("missing")).contains("missing"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
