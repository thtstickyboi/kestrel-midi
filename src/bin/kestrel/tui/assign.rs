// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Which soundfonts each file of a batch gets: a table driven with the arrow \[1\]

use super::*;
use crossterm::{cursor, queue, terminal};

/// What a step passes up when it is turned away: the same as `step!`, for a \[2\]
macro_rules! pass {
    ($e:expr) => {
        match $e {
            Step::Got(v) => v,
            Step::Menu => return Step::Menu,
            Step::Exit => return Step::Exit,
        }
    };
}

/// A soundfont set as it was picked, described once when it was loaded.
pub(super) struct SetPick {
    pub paths: Vec<PathBuf>,
    pub names: Vec<String>,
}

impl SetPick {
    fn label(&self) -> String {
        self.names.join(" + ")
    }
}

/// One file to write: a MIDI, and the set it gets once there is one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Row {
    pub midi: usize,
    pub set: Option<usize>,
}

/// The soundfont step's answer.
pub(super) struct Assigned {
    pub sets: Vec<SetPick>,
    /// Each file to write, in the order shown: its MIDI and its set.
    pub rows: Vec<(usize, usize)>,
    /// The bank loaded from the set picked last, and which set that is. When \[3\]
    pub kept: Option<(usize, Fonts)>,
}

// ---- the table's state ----------------------------------------------------------

/// What the person's key asks the screen to do next.
#[derive(Debug, PartialEq, Eq)]
enum Act {
    /// Nothing outside the table: redraw.
    Stay,
    /// Give these rows soundfonts.
    Choose(Vec<usize>),
    /// Every row has soundfonts: start.
    Go,
    /// Back to the menu.
    Back,
    /// Ctrl+C.
    Quit,
}

struct Table {
    /// How many MIDIs there are: `clear` goes back to one row each.
    midis: usize,
    rows: Vec<Row>,
    marked: Vec<bool>,
    cursor: usize,
    /// The first row on the screen, when there are more than fit.
    top: usize,
    /// Said under the table until the next key.
    message: Option<String>,
}

impl Table {
    fn new(midis: usize, rows: Vec<Row>) -> Self {
        let marked = vec![false; rows.len()];
        Table { midis, rows, marked, cursor: 0, top: 0, message: None }
    }

    /// Keep the cursor on the screen.
    fn scroll(&mut self, view: usize) {
        let view = view.max(1);
        if self.cursor < self.top {
            self.top = self.cursor;
        } else if self.cursor >= self.top + view {
            self.top = self.cursor + 1 - view;
        }
        self.top = self.top.min(self.rows.len().saturating_sub(view));
    }

    /// Give `targets` the set `k`, and unmark everything.
    fn assign(&mut self, targets: &[usize], k: usize) {
        for &i in targets {
            self.rows[i].set = Some(k);
        }
        self.marked.fill(false);
    }

    fn handle(&mut self, key: Key, view: usize) -> Act {
        self.message = None;
        let last = self.rows.len() - 1;
        match key {
            Key::Up => self.cursor = self.cursor.saturating_sub(1),
            Key::Down => self.cursor = (self.cursor + 1).min(last),
            Key::PageUp => self.cursor = self.cursor.saturating_sub(view.max(1)),
            Key::PageDown => self.cursor = (self.cursor + view.max(1)).min(last),
            Key::Home => self.cursor = 0,
            Key::End => self.cursor = last,
            // Mark and move on, so a run of rows is marked by holding the key.
            Key::Space => {
                self.marked[self.cursor] = !self.marked[self.cursor];
                self.cursor = (self.cursor + 1).min(last);
            }
            Key::Char('a') => {
                let all = self.marked.iter().all(|&m| m);
                self.marked.fill(!all);
            }
            Key::Enter => {
                let mut targets: Vec<usize> = (0..self.rows.len()).filter(|&i| self.marked[i]).collect();
                if targets.is_empty() {
                    targets.push(self.cursor);
                }
                return Act::Choose(targets);
            }
            // The MIDI under the cursor again, below it, to give it another set.
            Key::Char('d') => {
                let midi = self.rows[self.cursor].midi;
                self.rows.insert(self.cursor + 1, Row { midi, set: None });
                self.marked.insert(self.cursor + 1, false);
                self.cursor += 1;
            }
            Key::Char('x') => {
                let midi = self.rows[self.cursor].midi;
                if self.rows.iter().filter(|r| r.midi == midi).count() > 1 {
                    self.rows.remove(self.cursor);
                    self.marked.remove(self.cursor);
                    self.cursor = self.cursor.min(self.rows.len() - 1);
                } else {
                    self.message = Some("That is this MIDI's only row, so it stays.".into());
                }
            }
            Key::Char('c') => {
                self.rows = (0..self.midis).map(|midi| Row { midi, set: None }).collect();
                self.marked = vec![false; self.rows.len()];
                self.cursor = 0;
                self.top = 0;
            }
            Key::Char('g') => match self.rows.iter().position(|r| r.set.is_none()) {
                Some(i) => {
                    self.cursor = i;
                    self.message = Some(format!("Row {} has no soundfonts yet.", i + 1));
                }
                None => return Act::Go,
            },
            Key::Esc => return Act::Back,
            Key::CtrlC => return Act::Quit,
            _ => {}
        }
        Act::Stay
    }
}

// ---- drawing -------------------------------------------------------------------

/// The keys, as two short lines.
fn legend(cols: usize) -> Vec<Line> {
    let key = |k: &str, what: &str| [c(k.to_string(), AMBER), c(format!(" {what}   "), DIM)];
    let line = |parts: Vec<[style::Span; 2]>| style::fit(&parts.into_iter().flatten().collect::<Line>(), cols);
    vec![
        line(vec![
            key("\u{2191}\u{2193}", "move"),
            key("Space", "mark"),
            key("A", "mark all"),
            key("Enter", "choose soundfonts"),
        ]),
        line(vec![
            key("D", "add this MIDI again"),
            key("X", "remove a row"),
            key("C", "clear"),
            key("G", "start"),
            key("Esc", "back"),
        ]),
    ]
}

/// One screen of the table: the keys, the rows from `top`, and a line under.
fn frame(t: &Table, midis: &[MidiInfo], sets: &[SetPick], cols: usize, view: usize) -> Vec<Line> {
    let mut out = legend(cols);
    out.push(Line::new());
    let name_w = (cols.saturating_sub(16) / 2).clamp(14, 40);
    let end = (t.top + view).min(t.rows.len());
    for i in t.top..end {
        let row = &t.rows[i];
        let here = i == t.cursor;
        let mut line = vec![
            if here { b("\u{25B6} ", AMBER) } else { s("  ") },
            if t.marked[i] { b("[x]", AMBER) } else { c("[ ]", DIM) },
            c(format!(" {:>3}  ", i + 1), DIM),
        ];
        let name = style::middle(&file_name(&midis[row.midi].path), name_w);
        line.extend(style::fit(&[if here { b(name, AMBER) } else { s(name) }], name_w));
        line.push(s("  "));
        let room = cols.saturating_sub(name_w + 16);
        match row.set {
            Some(k) => line.extend(style::fit(&[s(sets[k].label())], room)),
            None => line.push(c("no soundfonts yet", WARN)),
        }
        out.push(line);
    }
    let bare = t.rows.iter().filter(|r| r.set.is_none()).count();
    let mut under = Vec::new();
    if t.rows.len() > view {
        under.push(c(format!("rows {}-{} of {}   ", t.top + 1, end, t.rows.len()), DIM));
    }
    match &t.message {
        Some(m) => under.push(b(m.clone(), WARN)),
        None if bare == 0 => under.push(c("Every row has soundfonts. Press G to start.", OK)),
        None => under.push(c(
            format!("{} of {} rows still need soundfonts", bare, t.rows.len()),
            DIM,
        )),
    }
    out.push(under);
    out
}

/// Draw `lines` where the last `drawn` were, and say how many there are now.
fn paint(lines: &[Line], drawn: usize) -> usize {
    let mut out = std::io::stdout().lock();
    if drawn > 0 {
        let _ = queue!(
            out,
            cursor::MoveUp(drawn.min(u16::MAX as usize) as u16),
            cursor::MoveToColumn(0),
            terminal::Clear(terminal::ClearType::FromCursorDown)
        );
    }
    for line in lines {
        let _ = out.write_all(style::MARGIN.as_bytes());
        let _ = style::queue_line(&mut out, line);
        let _ = out.write_all(b"\r\n");
    }
    let _ = out.flush();
    lines.len()
}

fn screen() -> (usize, usize) {
    terminal::size().map_or((100, 30), |(w, h)| (w as usize, h as usize))
}

/// Rows that fit with the keys and a line under, however tall the terminal is.
fn view_rows(height: usize) -> usize {
    height.saturating_sub(9).clamp(4, 30)
}

// ---- choosing --------------------------------------------------------------------

/// A short list to choose from with the arrow keys: the index chosen, or `None` \[4\]
fn pick_item(io: &mut dyn Io, title: &str, items: &[String]) -> Step<Option<usize>> {
    style::say(vec![b(title.to_string(), AMBER)]);
    let mut at = 0usize;
    let mut drawn = 0usize;
    io.raw(true);
    loop {
        let (cols, _) = screen();
        let mut lines: Vec<Line> = items
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let here = i == at;
                vec![
                    if here { b("\u{25B6} ", AMBER) } else { s("  ") },
                    c(format!("{} ", i + 1), DIM),
                    if here { b(name.clone(), AMBER) } else { s(name.clone()) },
                ]
            })
            .collect();
        lines.push(style::fit(
            &[c("\u{2191}\u{2193}", AMBER), c(" move   ", DIM), c("Enter", AMBER), c(" choose   ", DIM), c("Esc", AMBER), c(" back", DIM)],
            cols.saturating_sub(2),
        ));
        drawn = paint(&lines, drawn);
        let Some(key) = io.key() else {
            io.raw(false);
            return Step::Exit;
        };
        match key {
            Key::Up => at = at.saturating_sub(1),
            Key::Down => at = (at + 1).min(items.len() - 1),
            Key::Home => at = 0,
            Key::End => at = items.len() - 1,
            Key::Char(d @ '1'..='9') if (d as usize - '1' as usize) < items.len() => {
                io.raw(false);
                return Step::Got(Some(d as usize - '1' as usize));
            }
            Key::Enter => {
                io.raw(false);
                return Step::Got(Some(at));
            }
            Key::Esc => {
                io.raw(false);
                return Step::Got(None);
            }
            Key::CtrlC => {
                io.raw(false);
                return Step::Exit;
            }
            _ => {}
        }
    }
}

/// The set for some rows: one already picked, or new soundfonts from the picker. \[5\]
fn choose_set(io: &mut dyn Io, sets: &mut Vec<SetPick>, kept: &mut Option<(usize, Fonts)>) -> Step<Option<usize>> {
    if !sets.is_empty() {
        let mut items: Vec<String> = sets.iter().map(SetPick::label).collect();
        items.push("Pick other soundfonts\u{2026}".into());
        match pass!(pick_item(io, "Which soundfonts?", &items)) {
            Some(i) if i < sets.len() => return Step::Got(Some(i)),
            Some(_) => {}
            None => return Step::Got(None),
        }
    }
    style::say(vec![c("Up to two: a General MIDI bank, a piano, or both.", DIM)]);
    let (fonts, _) = pass!(pick_font_set(io, None));
    let k = match sets.iter().position(|p| p.paths == fonts.paths) {
        Some(k) => k,
        None => {
            sets.push(SetPick { paths: fonts.paths.clone(), names: fonts.names.clone() });
            sets.len() - 1
        }
    };
    // [6]
    *kept = Some((k, fonts));
    Step::Got(Some(k))
}

/// Which soundfonts each MIDI gets. With several MIDIs, one set for all of them \[7\]
pub(super) fn step_sets(
    io: &mut dyn Io,
    at: (u8, u8),
    midis: &[MidiInfo],
    first: Option<Fonts>,
) -> Step<Assigned> {
    let mut sets: Vec<SetPick> = Vec::new();
    let mut rows: Vec<Row> = (0..midis.len()).map(|midi| Row { midi, set: None }).collect();
    let mut kept: Option<(usize, Fonts)> = None;
    let hint;

    match first {
        Some(f) => {
            style::status(AMBER, vec![s("This MIDI on more than one soundfont set")]);
            sets.push(SetPick { paths: f.paths.clone(), names: f.names.clone() });
            rows[0].set = Some(0);
            kept = Some((0, f));
            hint = "Press D to add this MIDI again, then Enter to give the new row its soundfonts.";
        }
        None => {
            style::heading(&step_title(at, "Soundfonts"), "one set for all, or a set for each MIDI");
            option("1", &format!("One set for all {} MIDIs", style::thousands(midis.len() as u64)), "");
            option("2", "A different set for each MIDI", "");
            loop {
                prompt();
                let Some(line) = io.line() else {
                    return Step::Exit;
                };
                match line.trim() {
                    "1" => {
                        style::say(vec![c("Up to two: a General MIDI bank, a piano, or both.", DIM)]);
                        let (fonts, _) = pass!(pick_font_set(io, None));
                        let set = SetPick { paths: fonts.paths.clone(), names: fonts.names.clone() };
                        return Step::Got(Assigned {
                            sets: vec![set],
                            rows: (0..midis.len()).map(|m| (m, 0)).collect(),
                            kept: Some((0, fonts)),
                        });
                    }
                    "2" => break,
                    "0" => return Step::Menu,
                    _ => style::error("Type 1 or 2."),
                }
            }
            hint = "Press Enter to give a row soundfonts, or Space to mark several rows first.";
        }
    }

    let mut table = Table::new(midis.len(), rows);
    table.message = Some(hint.into());
    let mut drawn = 0usize;
    io.raw(true);
    loop {
        let (cols, height) = screen();
        let view = view_rows(height).min(table.rows.len());
        table.scroll(view);
        drawn = paint(&frame(&table, midis, &sets, cols.saturating_sub(2), view), drawn);
        let Some(key) = io.key() else {
            io.raw(false);
            return Step::Exit;
        };
        match table.handle(key, view) {
            Act::Stay => {}
            Act::Back => {
                io.raw(false);
                return Step::Menu;
            }
            Act::Quit => {
                io.raw(false);
                return Step::Exit;
            }
            Act::Go => {
                io.raw(false);
                break;
            }
            Act::Choose(targets) => {
                io.raw(false);
                let chosen = match choose_set(io, &mut sets, &mut kept) {
                    Step::Got(k) => k,
                    Step::Menu => return Step::Menu,
                    Step::Exit => return Step::Exit,
                };
                if let Some(k) = chosen {
                    table.assign(&targets, k);
                }
                // [8]
                drawn = 0;
                io.raw(true);
            }
        }
    }

    // [9]
    let rows = table.rows;
    let mut order: Vec<usize> = Vec::new();
    for row in &rows {
        let k = row.set.expect("every row has a set");
        if !order.contains(&k) {
            order.push(k);
        }
    }
    let remap = |k: usize| order.iter().position(|&o| o == k).expect("a used set");
    let kept = kept.and_then(|(k, f)| order.contains(&k).then(|| (remap(k), f)));
    let mut sets: Vec<Option<SetPick>> = sets.into_iter().map(Some).collect();
    let sets: Vec<SetPick> = order.iter().map(|&k| sets[k].take().expect("a set is taken once")).collect();
    let rows: Vec<(usize, usize)> = rows.iter().map(|r| (r.midi, remap(r.set.expect("set")))).collect();
    Step::Got(Assigned { sets, rows, kept })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(rows: usize) -> Table {
        Table::new(rows, (0..rows).map(|midi| Row { midi, set: None }).collect())
    }

    fn press(t: &mut Table, keys: &[Key]) -> Act {
        let mut last = Act::Stay;
        for &k in keys {
            last = t.handle(k, 5);
        }
        last
    }

    #[test]
    fn enter_chooses_for_the_marked_rows_or_else_the_one_under_the_cursor() {
        let mut t = table(4);
        assert_eq!(press(&mut t, &[Key::Down, Key::Enter]), Act::Choose(vec![1]));
        // Space marks and moves on, so three presses mark three rows in a run.
        let mut t = table(4);
        assert_eq!(press(&mut t, &[Key::Space, Key::Space, Key::Down, Key::Space, Key::Enter]), Act::Choose(vec![0, 1, 3]));
        // A, and again to take them back.
        let mut t = table(3);
        assert_eq!(press(&mut t, &[Key::Char('a'), Key::Enter]), Act::Choose(vec![0, 1, 2]));
        let mut t = table(3);
        assert_eq!(press(&mut t, &[Key::Char('a'), Key::Char('a'), Key::Down, Key::Enter]), Act::Choose(vec![1]));
    }

    #[test]
    fn assigning_gives_the_rows_their_set_and_clears_the_marks() {
        let mut t = table(3);
        press(&mut t, &[Key::Space, Key::Down, Key::Down, Key::Space]);
        t.assign(&[0, 2], 7);
        assert_eq!(t.rows.iter().map(|r| r.set).collect::<Vec<_>>(), [Some(7), None, Some(7)]);
        assert!(t.marked.iter().all(|&m| !m));
    }

    #[test]
    fn g_starts_only_when_every_row_has_soundfonts_and_shows_the_first_that_does_not() {
        let mut t = table(3);
        t.assign(&[0, 2], 0);
        t.cursor = 2;
        assert_eq!(t.handle(Key::Char('g'), 5), Act::Stay);
        assert_eq!(t.cursor, 1, "the cursor goes to the row that is missing");
        assert!(t.message.as_deref().is_some_and(|m| m.contains("Row 2")));
        t.assign(&[1], 0);
        assert_eq!(t.handle(Key::Char('g'), 5), Act::Go);
    }

    #[test]
    fn d_adds_the_midi_again_and_x_removes_a_row_but_never_a_midis_last() {
        let mut t = table(2);
        press(&mut t, &[Key::Char('d')]);
        assert_eq!(t.rows.iter().map(|r| r.midi).collect::<Vec<_>>(), [0, 0, 1]);
        assert_eq!(t.cursor, 1, "the cursor is on the new row");
        press(&mut t, &[Key::Char('x')]);
        assert_eq!(t.rows.len(), 2);
        // Row 1 is its MIDI's only one again.
        press(&mut t, &[Key::Home, Key::Char('x')]);
        assert_eq!(t.rows.len(), 2);
        assert!(t.message.is_some());
        // The keys the table has no use for do nothing.
        assert_eq!(press(&mut t, &[Key::Char('z'), Key::Redraw]), Act::Stay);
    }

    #[test]
    fn c_clears_to_one_row_a_midi_and_the_way_out_keys_leave() {
        let mut t = table(2);
        press(&mut t, &[Key::Char('d'), Key::Char('d')]);
        t.assign(&[0], 1);
        press(&mut t, &[Key::Char('c')]);
        assert_eq!(t.rows, vec![Row { midi: 0, set: None }, Row { midi: 1, set: None }]);
        assert_eq!((t.cursor, t.marked.len()), (0, 2));
        assert_eq!(t.handle(Key::Esc, 5), Act::Back);
        assert_eq!(t.handle(Key::CtrlC, 5), Act::Quit);
    }

    #[test]
    fn the_cursor_stays_on_the_screen_in_a_long_table() {
        let mut t = table(40);
        t.handle(Key::End, 5);
        t.scroll(5);
        assert_eq!((t.cursor, t.top), (39, 35));
        t.handle(Key::PageUp, 5);
        t.scroll(5);
        assert_eq!((t.cursor, t.top), (34, 34));
        t.handle(Key::Home, 5);
        t.scroll(5);
        assert_eq!(t.top, 0);
    }

    fn info(name: &str) -> MidiInfo {
        MidiInfo {
            path: PathBuf::from(name),
            size: 1,
            format: 1,
            tracks: 1,
            division: kestrel::midi::Division::Ppq(480),
            notes: Vec::new(),
            extended_keys: false,
        }
    }

    /// The screen reads as what it is: the keys, one line a row with the cursor \[10\]
    #[test]
    fn a_frame_shows_the_keys_the_rows_and_what_is_left() {
        let midis = [info("song one.mid"), info("two.mid")];
        let sets = [SetPick { paths: vec!["gm.sf2".into()], names: vec!["gm.sf2".into(), "piano.sfz".into()] }];
        let mut t = table(2);
        t.assign(&[0], 0);
        t.marked[1] = true;
        let text: Vec<String> = frame(&t, &midis, &sets, 90, 5).iter().map(|l| style::plain(l)).collect();
        assert!(text[0].contains("move") && text[0].contains("Space mark") && text[0].contains("Enter choose soundfonts"), "{text:?}");
        assert!(text[1].contains("D add this MIDI again") && text[1].contains("G start"), "{text:?}");
        assert!(text[3].starts_with("\u{25B6} [ ]") && text[3].contains("song one.mid") && text[3].contains("gm.sf2 + piano.sfz"), "{text:?}");
        assert!(text[4].starts_with("  [x]") && text[4].contains("no soundfonts yet"), "{text:?}");
        assert!(text[5].contains("1 of 2 rows still need soundfonts"), "{text:?}");
        // Every line fits the width it was given.
        assert!(frame(&t, &midis, &sets, 90, 5).iter().all(|l| style::width(l) <= 90));
    }
}

#[cfg(test)]
mod preview {
    use super::*;

    /// Not a check: prints the table as plain text, to look at a layout change \[11\]
    #[test]
    #[ignore = "prints a frame to look at; asserts nothing"]
    fn preview_table() {
        let names = [
            "[abc] Artist - Song One.mid",
            "Song Two AUDIO-ONLY.mid",
            "A friend's submission (audio).mid",
            "Song Three.mid",
            "(AUDIO VER.) [Black MIDI] Animation Meme - A Birthday Special.mid",
            "Song Four Audio.mid",
        ];
        let midis: Vec<MidiInfo> = names
            .iter()
            .map(|n| MidiInfo {
                path: PathBuf::from(n),
                size: 1,
                format: 1,
                tracks: 1,
                division: kestrel::midi::Division::Ppq(480),
                notes: Vec::new(),
                extended_keys: false,
            })
            .collect();
        let sets = [
            SetPick { paths: vec!["a".into()], names: vec!["Concert Grand (realistic).sfz".into()] },
            SetPick { paths: vec!["b".into()], names: vec!["Studio Grand BOOSTED REALISTIC Ext.sfz".into()] },
        ];
        let mut t = Table::new(6, (0..6).map(|midi| Row { midi, set: None }).collect());
        t.assign(&[0, 3], 0);
        t.assign(&[1], 1);
        t.cursor = 2;
        t.marked[4] = true;
        t.marked[5] = true;
        for cols in [100, 80] {
            println!("--- {cols} columns");
            for line in frame(&t, &midis, &sets, cols - 2, 6) {
                println!("  {}", style::plain(&line));
            }
        }
    }
}
