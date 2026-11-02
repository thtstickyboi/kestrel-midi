// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What Merlin says when a render ends, in the three ways it can. \[1\]

use super::super::style::{self, b, c, s, Line, DIM, ERR, OK, WARN};
use super::super::{file_name, flag_lines, panel_width};
use kestrel::session::Summary;
use std::path::Path;

/// The panel for a render's end, `inner` columns wide. `problems` are what the \[2\]
pub fn panel(
    out: &Path,
    adapter: Option<&str>,
    flags: &[String],
    result: &Result<Summary, String>,
    problems: &[(log::Level, String)],
    inner: usize,
) -> Vec<Line> {
    let label = |t: &str| c(format!("{t:<10}"), DIM);
    match result {
        Ok(sum) if sum.cancelled => {
            let mut body = vec![vec![s(format!("Stopped {} in.", style::clock(sum.wall_secs)))]];
            body.extend(style::wrap("Nothing was kept: Merlin cannot resume a render, and the partial file was removed. Start it again from the beginning.", inner).into_iter().map(|l| vec![c(l, DIM)]));
            style::panel(vec![b("Cancelled", WARN)], &body, inner, None)
        }
        Ok(sum) => {
            let speed = sum.audio_secs / sum.wall_secs.max(1e-9);
            let mut body = vec![
                vec![label("Wrote"), b(file_name(out), super::super::style::AMBER)],
                vec![
                    label("Folder"),
                    // The end of a path is the part that says which folder.
                    s(style::middle(&out.parent().map(|p| p.display().to_string()).unwrap_or_default(), inner - 10)),
                ],
                Line::new(),
                vec![
                    label("Audio"),
                    s(style::audio_clock(sum.audio_secs)),
                    c("  in ", DIM),
                    s(duration(sum.wall_secs)),
                    c("  = ", DIM),
                    b(format!("{speed:.2}\u{00D7} realtime"), super::super::style::AMBER),
                ],
                vec![label("Size"), s(style::bytes(sum.bytes))],
                vec![label("Notes"), s(style::thousands(sum.notes))],
            ];
            if let Some(a) = adapter {
                body.push(vec![label("Device"), s(a.to_string())]);
            }
            body.extend(flag_lines(flags, inner));
            if let Some(log) = kestrel::falconeye::renderlog::last_path() {
                body.push(vec![label("Log"), c(log.display().to_string(), DIM)]);
            }
            // What the render had to say on its way, in the panel, where it is read.
            if !problems.is_empty() {
                body.push(Line::new());
                body.push(vec![b("Noted on the way", WARN)]);
                for (_, text) in problems {
                    body.extend(style::wrap(text, inner - 2).into_iter().map(|l| vec![c(format!("  {l}"), WARN)]));
                }
            }
            style::panel(vec![b("Done", OK)], &body, inner, None)
        }
        Err(message) => {
            let mut body: Vec<Line> = style::wrap(message, inner).into_iter().map(|l| vec![c(l, ERR)]).collect();
            body.push(Line::new());
            body.extend(style::wrap("Merlin cannot resume a render: start it again from the beginning.", inner).into_iter().map(|l| vec![c(l, DIM)]));
            if let Some(log) = kestrel::falconeye::renderlog::last_path() {
                body.push(Line::new());
                let text = format!("The render's log, to send with a report: {}", log.display());
                body.extend(style::wrap(&text, inner).into_iter().map(|l| vec![c(l, DIM)]));
            }
            style::panel(vec![b("Render failed", ERR)], &body, inner, None)
        }
    }
}

/// Seconds as `12.3 s` under a minute, then `m:ss`, then `h:mm:ss`: a render of \[3\]
fn duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.1} s")
    } else {
        style::clock(secs)
    }
}

/// What the render logged, less what a failure already says in its own text.
fn noted(result: &Result<Summary, String>, problems: &[(log::Level, String)]) -> Vec<(log::Level, String)> {
    problems.iter().filter(|(_, m)| !matches!(result, Err(e) if e.contains(m.as_str()))).cloned().collect()
}

/// Print the end of a render.
pub fn show(
    out: &Path,
    adapter: Option<&str>,
    flags: &[String],
    result: &Result<Summary, String>,
    problems: &[(log::Level, String)],
) {
    for line in panel(out, adapter, flags, result, &noted(result, problems), panel_width()) {
        style::say(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn text(result: &Result<Summary, String>, problems: &[(log::Level, String)], inner: usize) -> Vec<String> {
        panel(&PathBuf::from("D:/renders/song.opus"), Some("Test GPU (Vulkan)"), &["--limiter=balanced".into()], result, problems, inner)
            .iter()
            .map(|l| style::plain(l))
            .collect()
    }

    fn done() -> Summary {
        Summary { bytes: 92_799_020, audio_secs: 241.66, wall_secs: 1372.5, notes: 24_390_649_323, ..Summary::default() }
    }

    fn warn(m: &str) -> (log::Level, String) {
        (log::Level::Warn, m.to_string())
    }

    /// Every ending is a rectangle at every width, with or without warnings.
    #[test]
    fn every_ending_is_a_rectangle_at_every_width() {
        let long = "x".repeat(300);
        let endings: Vec<Result<Summary, String>> =
            vec![Ok(done()), Ok(Summary { cancelled: true, wall_secs: 75.0, ..Summary::default() }), Err(long.clone()), Err("short".into())];
        for e in &endings {
            for inner in [40, 60, 76] {
                for problems in [vec![], vec![warn(&long), warn("a second")]] {
                    let panel = panel(&PathBuf::from("song.opus"), Some("GPU"), &[], e, &problems, inner);
                    let w: Vec<usize> = panel.iter().map(|l| style::width(l)).collect();
                    assert!(w.iter().all(|&x| x == w[0]), "{w:?}");
                }
            }
        }
    }

    /// Done says what was written, how long the audio and the render took, and \[4\]
    #[test]
    fn done_says_what_was_written_and_has_no_voice_rows() {
        let t = text(&Ok(done()), &[], 76).join("\n");
        assert!(t.contains("Done") && t.contains("song.opus") && t.contains("D:/renders"), "{t}");
        assert!(t.contains("04:01.7") && t.contains("22:53") && t.contains("0.18\u{00D7} realtime"), "{t}");
        assert!(t.contains("88.5 MiB") && t.contains("24,390,649,323"), "{t}");
        assert!(t.contains("Test GPU (Vulkan)") && t.contains("--limiter=balanced"), "{t}");
        for gone in ["Voices", "stolen", "dropped", "Mix peak", "hard-clipped", "voices spawned"] {
            assert!(!t.contains(gone), "{gone} on Merlin's end screen: {t}");
        }
        // A short render is in seconds, a long one in minutes.
        assert!(text(&Ok(Summary { wall_secs: 12.34, ..done() }), &[], 76).join("\n").contains("12.3 s"));
    }

    /// What the render noted is in the panel, wrapped, under its own heading, \[5\]
    #[test]
    fn what_the_render_noted_is_in_the_panel() {
        let none = text(&Ok(done()), &[], 76).join("\n");
        assert!(!none.contains("Noted on the way"), "{none}");
        let t = text(
            &Ok(done()),
            &[warn("the key parser does not build these yet, and rendered without them: modulation"), warn("the key store was read the slow way")],
            60,
        )
        .join("\n");
        assert!(t.contains("Noted on the way") && t.contains("modulation") && t.contains("the slow way"), "{t}");
    }

    /// A failure's own words are not said twice; anything else the render \[6\]
    #[test]
    fn a_failure_is_not_told_twice() {
        let ps = vec![warn("the device failed"), warn("something else")];
        let failed: Result<Summary, String> = Err("render stopped: the device failed at window 3".into());
        assert_eq!(noted(&failed, &ps), vec![warn("something else")]);
        assert_eq!(noted(&Ok(done()), &ps), ps, "a render that finished is told all of it");
    }

    /// Stopped says so, says nothing was kept, and does not offer to resume.
    #[test]
    fn stopped_says_nothing_was_kept_and_offers_no_resume() {
        let t = text(&Ok(Summary { cancelled: true, wall_secs: 75.0, ..Summary::default() }), &[], 76).join("\n");
        assert!(t.contains("Cancelled") && t.contains("Stopped 01:15 in."), "{t}");
        assert!(t.contains("Nothing was kept") && t.contains("cannot resume") && t.contains("removed"), "{t}");
        for gone in ["Progress saved", "Resume a render", "Wrote"] {
            assert!(!t.contains(gone), "{gone}: {t}");
        }
    }

    /// Failed gives the message as it came, so a limit's words (which one, and \[7\]
    #[test]
    fn failed_gives_the_message_as_it_came() {
        let message = "song.mid is 200.00 GB, and merlin takes files under 200.00 GB: 0.00 GB past the limit. Nothing was rendered";
        let t = text(&Err(message.into()), &[], 200).join("\n");
        assert!(t.contains("Render failed") && t.contains(message), "{t}");
        assert!(t.contains("cannot resume a render"), "{t}");
        // Wrapped, not cut, in a narrow panel: every word is still there.
        let narrow = text(&Err(message.into()), &[], 40).join(" ");
        for word in message.split_whitespace() {
            assert!(narrow.contains(word), "{word} was cut off: {narrow}");
        }
    }
}
