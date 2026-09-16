// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Colour, layout primitives and the banner. \[1\]

use crossterm::queue;
use crossterm::style::{Attribute, Color, Print, SetAttribute, SetForegroundColor};
use std::io::{self, Write};
use std::sync::OnceLock;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub type Rgb = (u8, u8, u8);

/// Kestrel amber.
pub const AMBER: Rgb = (0xf5, 0xb7, 0x00);
/// The light end of the gradient.
pub const GOLD: Rgb = (0xff, 0xd8, 0x5c);
/// The deep end of the gradient.
pub const EMBER: Rgb = (0xe0, 0x78, 0x00);
/// Labels and secondary text: a warm grey that reads on dark and light \[2\]
pub const DIM: Rgb = (0x9a, 0x94, 0x86);
/// Box borders.
pub const FRAME: Rgb = (0x8c, 0x6e, 0x22);
/// The unfilled part of a bar.
pub const TRACK: Rgb = (0x4a, 0x45, 0x3a);
pub const OK: Rgb = (0x8f, 0xd1, 0x7a);
pub const WARN: Rgb = (0xff, 0xe0, 0x4d);
pub const ERR: Rgb = (0xff, 0x6b, 0x5b);

/// Left margin of everything the guided renderer prints.
pub const MARGIN: &str = "  ";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Span {
    pub text: String,
    pub fg: Option<Rgb>,
    pub bold: bool,
}

pub type Line = Vec<Span>;

/// Unstyled text.
pub fn s(text: impl Into<String>) -> Span {
    Span {
        text: text.into(),
        fg: None,
        bold: false,
    }
}

/// Coloured text.
pub fn c(text: impl Into<String>, fg: Rgb) -> Span {
    Span {
        text: text.into(),
        fg: Some(fg),
        bold: false,
    }
}

/// Bold coloured text.
pub fn b(text: impl Into<String>, fg: Rgb) -> Span {
    Span {
        text: text.into(),
        fg: Some(fg),
        bold: true,
    }
}

/// Columns a line occupies. Not its length in chars: a CJK title is two \[3\]
pub fn width(line: &[Span]) -> usize {
    line.iter().map(|s| s.text.width()).sum()
}

#[cfg(test)]
pub fn plain(line: &[Span]) -> String {
    line.iter().map(|s| s.text.as_str()).collect()
}

/// Truncate a line to `cols` columns, ending in an ellipsis if anything was \[4\]
pub fn fit(line: &[Span], cols: usize) -> Line {
    let mut out = Line::new();
    let mut used = 0usize;
    let total = width(line);
    let budget = if total > cols { cols.saturating_sub(1) } else { cols };
    'spans: for sp in line {
        let mut text = String::new();
        for ch in sp.text.chars() {
            let w = ch.width().unwrap_or(0);
            if used + w > budget {
                if !text.is_empty() {
                    out.push(Span { text, ..sp.clone() });
                }
                break 'spans;
            }
            used += w;
            text.push(ch);
        }
        out.push(Span { text, ..sp.clone() });
    }
    if total > cols {
        out.push(c("\u{2026}", DIM));
        used += 1;
    }
    if used < cols {
        out.push(s(" ".repeat(cols - used)));
    }
    out
}

/// Truncate a plain string from the middle, keeping both ends. File names are \[5\]
pub fn middle(text: &str, cols: usize) -> String {
    if text.width() <= cols || cols < 5 {
        return text.to_string();
    }
    let keep = cols - 1;
    let (head_cols, tail_cols) = (keep - keep / 2, keep / 2);
    let mut head = String::new();
    let mut w = 0;
    for ch in text.chars() {
        let cw = ch.width().unwrap_or(0);
        if w + cw > head_cols {
            break;
        }
        w += cw;
        head.push(ch);
    }
    let mut tail: Vec<char> = Vec::new();
    w = 0;
    for ch in text.chars().rev() {
        let cw = ch.width().unwrap_or(0);
        if w + cw > tail_cols {
            break;
        }
        w += cw;
        tail.push(ch);
    }
    tail.reverse();
    format!("{head}\u{2026}{}", tail.into_iter().collect::<String>())
}

fn colour_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // https://no-color.org: any non-empty value turns colour off.
    *ON.get_or_init(|| match std::env::var_os("NO_COLOR") {
        Some(v) => v.is_empty(),
        None => true,
    })
}

/// Turn on escape-sequence processing where the console has to be asked. \[6\]
pub fn init() {
    #[cfg(windows)]
    {
        let _ = crossterm::ansi_support::supports_ansi();
    }
}

/// Word-wrap plain text to `cols` columns, keeping its own line breaks. A word \[7\]
pub fn wrap(text: &str, cols: usize) -> Vec<String> {
    let cols = cols.max(8);
    let mut out = Vec::new();
    for para in text.lines() {
        let mut line = String::new();
        let mut used = 0usize;
        for word in para.split_whitespace() {
            let w = word.width();
            if used > 0 && used + 1 + w > cols {
                out.push(std::mem::take(&mut line));
                used = 0;
            }
            if w > cols {
                for ch in word.chars() {
                    let cw = ch.width().unwrap_or(0);
                    if used + cw > cols {
                        out.push(std::mem::take(&mut line));
                        used = 0;
                    }
                    line.push(ch);
                    used += cw;
                }
                continue;
            }
            if used > 0 {
                line.push(' ');
                used += 1;
            }
            line.push_str(word);
            used += w;
        }
        out.push(line);
    }
    out
}

/// Write one line's spans, without a newline.
pub fn queue_line(out: &mut impl Write, line: &[Span]) -> io::Result<()> {
    let colour = colour_enabled();
    for sp in line {
        let styled = colour && (sp.fg.is_some() || sp.bold);
        if colour {
            if let Some((r, g, b)) = sp.fg {
                queue!(out, SetForegroundColor(Color::Rgb { r, g, b }))?;
            }
            if sp.bold {
                queue!(out, SetAttribute(Attribute::Bold))?;
            }
        }
        queue!(out, Print(&sp.text))?;
        if styled {
            queue!(out, SetAttribute(Attribute::Reset))?;
        }
    }
    Ok(())
}

/// Print a line with the margin in front of it.
pub fn say(line: Line) {
    let mut out = io::stdout().lock();
    let _ = out.write_all(MARGIN.as_bytes());
    let _ = queue_line(&mut out, &line);
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

pub fn blank() {
    println!();
}

/// A section heading: a bold amber title and a dim note after it.
pub fn heading(title: &str, note: &str) {
    blank();
    let mut line = vec![b(title, AMBER)];
    if !note.is_empty() {
        line.push(c(format!("  {note}"), DIM));
    }
    say(line);
}

/// A status line: a coloured bullet, then the text.
pub fn status(colour: Rgb, text: Line) {
    let mut line = vec![c("\u{25CF} ", colour)];
    line.extend(text);
    say(line);
}

/// A yellow WARN line.
pub fn warn(text: impl Into<String>) {
    say(vec![b("WARN", WARN), c(format!("  {}", text.into()), WARN)]);
}

/// A red error line. Multi-line messages keep their own lines, indented.
pub fn error(text: impl AsRef<str>) {
    for (i, part) in text.as_ref().lines().enumerate() {
        if i == 0 {
            say(vec![b("ERROR", ERR), c(format!(" {part}"), ERR)]);
        } else {
            say(vec![c(format!("      {part}"), ERR)]);
        }
    }
}

/// An indented detail line under a status line.
pub fn detail(text: Line) {
    let mut line = vec![s("    ")];
    line.extend(text);
    say(line);
}

pub fn lerp(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    (m(a.0, b.0), m(a.1, b.1), m(a.2, b.2))
}

/// Gold through amber to ember, over `t` in 0..=1.
pub fn gradient(t: f32) -> Rgb {
    if t < 0.5 {
        lerp(GOLD, AMBER, t * 2.0)
    } else {
        lerp(AMBER, EMBER, (t - 0.5) * 2.0)
    }
}

pub fn shade(c: Rgb, k: f32) -> Rgb {
    lerp((0, 0, 0), c, k)
}

const BANNER: [&str; 6] = [
    "██╗  ██╗███████╗███████╗████████╗██████╗ ███████╗██╗     ",
    "██║ ██╔╝██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║     ",
    "█████╔╝ █████╗  ███████╗   ██║   ██████╔╝█████╗  ██║     ",
    "██╔═██╗ ██╔══╝  ╚════██║   ██║   ██╔══██╗██╔══╝  ██║     ",
    "██║  ██╗███████╗███████║   ██║   ██║  ██║███████╗███████╗",
    "╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝",
];

/// Rows the banner takes, tagline and the blank line under it included.
pub const BANNER_ROWS: u16 = 8;

/// The KESTREL banner. The gradient runs diagonally, and the outline glyphs \[8\]
pub fn banner() -> Vec<Line> {
    let cols = BANNER[0].chars().count() as f32;
    let span = cols + BANNER.len() as f32 * 2.0;
    let mut lines = Vec::new();
    for (row, text) in BANNER.iter().enumerate() {
        let mut line = Line::new();
        for (col, ch) in text.chars().enumerate() {
            let base = gradient((col as f32 + row as f32 * 2.0) / span);
            let colour = if ch == '█' { base } else { shade(base, 0.55) };
            match line.last_mut() {
                Some(last) if last.fg == Some(colour) => last.text.push(ch),
                _ => line.push(c(ch.to_string(), colour)),
            }
        }
        lines.push(line);
    }
    lines.push(vec![
        c("GPU-accelerated MIDI renderer", DIM),
        c("  \u{00B7}  ", FRAME),
        c(format!("v{}", env!("CARGO_PKG_VERSION")), AMBER),
    ]);
    lines.push(Line::new());
    lines
}

/// The banner at the top of a screen. Its trailing blank row is dropped here, \[9\]
pub fn print_banner() {
    blank();
    let mut lines = banner();
    lines.pop();
    for line in lines {
        say(line);
    }
}

/// A box around `body`, `inner` columns wide, with a title in the top edge and \[10\]
pub fn panel(title: Line, body: &[Line], inner: usize, footer: Option<Line>) -> Vec<Line> {
    let edge = |t: &str| c(t, FRAME);
    let mut out = Vec::with_capacity(body.len() + 2);

    let title_w = width(&title);
    let mut top = vec![edge("\u{250C}\u{2500} ")];
    top.extend(title);
    top.push(edge(&format!(
        " {}\u{2510}",
        "\u{2500}".repeat((inner + 1).saturating_sub(title_w + 2))
    )));
    out.push(top);

    for line in body {
        let mut row = vec![edge("\u{2502} ")];
        row.extend(fit(line, inner));
        row.push(edge(" \u{2502}"));
        out.push(row);
    }

    let mut bottom = vec![edge("\u{2514}")];
    match footer {
        Some(note) => {
            let w = width(&note);
            bottom.push(edge(&"\u{2500}".repeat((inner + 2).saturating_sub(w + 3))));
            bottom.push(s(" "));
            bottom.extend(note);
            bottom.push(edge(" \u{2500}\u{2518}"));
        }
        None => bottom.push(edge(&format!("{}\u{2518}", "\u{2500}".repeat(inner + 2)))),
    }
    out.push(bottom);
    out
}

/// A bar `cells` wide filled to `frac`, at half-cell resolution, coloured along \[11\]
pub fn bar(frac: f64, cells: usize) -> Line {
    let halves = (frac.clamp(0.0, 1.0) * (cells * 2) as f64).round() as usize;
    let mut line = Line::new();
    for i in 0..cells {
        let fill = halves.saturating_sub(i * 2).min(2);
        let colour = gradient(i as f32 / cells.max(2) as f32);
        match fill {
            2 => line.push(c("\u{2588}", colour)),
            1 => line.push(c("\u{258C}", colour)),
            _ => line.push(c("\u{2591}", TRACK)),
        }
    }
    line
}

/// 1048576 -> "1,048,576".
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Binary units, one decimal.
pub fn bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}

/// Seconds as `m:ss`, or `h:mm:ss` past an hour.
pub fn clock(secs: f64) -> String {
    let t = secs.max(0.0).round() as u64;
    let (h, m, s) = (t / 3600, (t / 60) % 60, t % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Seconds of audio as `mm:ss.t`, or `h:mm:ss.t`.
pub fn audio_clock(secs: f64) -> String {
    let tenths = (secs.max(0.0) * 10.0).round() as u64;
    let (whole, t) = (tenths / 10, tenths % 10);
    let (h, m, s) = (whole / 3600, (whole / 60) % 60, whole % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}.{t}")
    } else {
        format!("{m:02}:{s:02}.{t}")
    }
}

/// A rate with a k or M suffix once it is large: 21,213 -> "21.2k".
pub fn compact(v: f64) -> String {
    if v >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1e4 {
        format!("{:.1}k", v / 1e3)
    } else {
        format!("{v:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_banner_rows_line_up() {
        let w: Vec<usize> = BANNER.iter().map(|r| r.chars().count()).collect();
        assert!(w.iter().all(|&x| x == w[0]), "{w:?}");
    }

    #[test]
    fn fit_pads_and_truncates_by_display_width() {
        assert_eq!(plain(&fit(&[s("abc")], 5)), "abc  ");
        assert_eq!(plain(&fit(&[s("abcdef")], 4)), "abc\u{2026}");
        // [12]
        let jp = fit(&[s("東方紅魔郷")], 8);
        assert_eq!(width(&jp), 8);
        assert_eq!(plain(&jp), "東方紅\u{2026} ");
    }

    #[test]
    fn a_panel_is_rectangular() {
        let p = panel(
            vec![s("Title")],
            &[vec![s("short")], vec![s("a much longer line than the box")]],
            20,
            Some(vec![s("note")]),
        );
        let widths: Vec<usize> = p.iter().map(|l| width(l)).collect();
        assert!(widths.iter().all(|&w| w == widths[0]), "{widths:?}");
    }

    #[test]
    fn numbers_read_the_way_people_write_them() {
        assert_eq!(thousands(1_048_576), "1,048,576");
        assert_eq!(thousands(999), "999");
        assert_eq!(bytes(3_900_000), "3.7 MiB");
        assert_eq!(clock(195.3), "03:15");
        assert_eq!(clock(3725.0), "1:02:05");
        assert_eq!(audio_clock(195.33), "03:15.3");
        assert_eq!(compact(21_213.0), "21.2k");
    }

    #[test]
    fn a_bar_fills_by_halves() {
        assert_eq!(plain(&bar(0.0, 4)), "\u{2591}".repeat(4));
        assert_eq!(plain(&bar(1.0, 4)), "\u{2588}".repeat(4));
        assert_eq!(plain(&bar(0.25 + 0.125, 4)), "\u{2588}\u{258C}\u{2591}\u{2591}");
    }
}
