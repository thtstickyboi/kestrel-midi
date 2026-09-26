// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Hiding whose machine a log or a report came from, while keeping what it \[1\]

/// What a hidden account name becomes.
pub const USER: &str = "<user>";
/// What a hidden PC or domain name becomes.
pub const PC: &str = "<pc>";

/// Shorter names are matched only as a whole path segment.
const WORD_MIN: usize = 3;

#[derive(Debug, Clone, Default)]
pub struct Redactor {
    /// Account names: the login name, and the profile folder's name where it \[2\]
    users: Vec<String>,
    /// The PC's name, and the domain's where it differs.
    pcs: Vec<String>,
    /// Names shown as they are, longest first.
    keep: Vec<String>,
}

impl Redactor {
    /// Hide these names.
    pub fn new(users: &[&str], pcs: &[&str]) -> Redactor {
        let mut r = Redactor::default();
        for u in users {
            push_name(&mut r.users, u);
        }
        for p in pcs {
            push_name(&mut r.pcs, p);
        }
        r
    }

    /// Hide this machine's names, read from the environment. On Windows, \[3\]
    pub fn from_env() -> Redactor {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let last = |k: &str| {
            var(k).and_then(|p| {
                std::path::Path::new(&p)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
            })
        };
        let users: Vec<String> = [var("USERNAME"), var("USER"), last("USERPROFILE"), last("HOME")]
            .into_iter()
            .flatten()
            .collect();
        let host = std::fs::read_to_string("/etc/hostname")
            .ok()
            .map(|h| h.trim().to_string());
        let pcs: Vec<String> = [var("COMPUTERNAME"), var("USERDOMAIN"), var("HOSTNAME"), host]
            .into_iter()
            .flatten()
            .collect();
        let users: Vec<&str> = users.iter().map(String::as_str).collect();
        let pcs: Vec<&str> = pcs.iter().map(String::as_str).collect();
        Redactor::new(&users, &pcs)
    }

    /// Show `name` as it is wherever it appears: a MIDI or soundfont file \[4\]
    pub fn keep(&mut self, name: &str) {
        if name.is_empty() || self.keep.iter().any(|k| k == name) {
            return;
        }
        self.keep.push(name.to_string());
        // Longest first, so a name that contains another is kept whole.
        self.keep.sort_by_key(|k| std::cmp::Reverse(k.len()));
    }

    /// Whether there is anything to hide.
    pub fn is_empty(&self) -> bool {
        self.users.is_empty() && self.pcs.is_empty()
    }

    /// Overwrite the hidden names in binary data, such as a minidump, with \[5\]
    pub fn scrub_bytes(&self, bytes: &mut [u8]) -> usize {
        let mut n = 0;
        for name in self.users.iter().chain(&self.pcs) {
            if name.chars().count() < WORD_MIN {
                continue;
            }
            n += overwrite(bytes, name.as_bytes(), 1);
            let wide: Vec<u8> = name.encode_utf16().flat_map(u16::to_le_bytes).collect();
            n += overwrite(bytes, &wide, 2);
        }
        n
    }

    /// `text` with the names hidden.
    pub fn apply(&self, text: &str) -> String {
        if self.is_empty() {
            return hide_sids(text);
        }
        // [6]
        let mut s = text.to_string();
        let masked: Vec<usize> = (0..self.keep.len())
            .filter(|&i| s.contains(self.keep[i].as_str()))
            .collect();
        for &i in &masked {
            s = s.replace(self.keep[i].as_str(), &marker(i));
        }
        // PC names first: `DESKTOP-1\Fule` has both.
        for p in &self.pcs {
            s = replace_name(&s, p, PC);
        }
        for u in &self.users {
            s = replace_name(&s, u, USER);
        }
        s = hide_sids(&s);
        for &i in &masked {
            s = s.replace(&marker(i), self.keep[i].as_str());
        }
        s
    }
}

/// Every match of `pat` in `bytes`, ignoring ASCII case, overwritten unit by \[7\]
fn overwrite(bytes: &mut [u8], pat: &[u8], unit: usize) -> usize {
    if pat.is_empty() || bytes.len() < pat.len() {
        return 0;
    }
    let (mut count, mut i) = (0, 0);
    while i + pat.len() <= bytes.len() {
        if bytes[i..i + pat.len()].eq_ignore_ascii_case(pat) {
            for j in (i..i + pat.len()).step_by(unit) {
                bytes[j] = b'X';
                bytes[j + 1..j + unit].fill(0);
            }
            count += 1;
            i += pat.len();
        } else {
            i += 1;
        }
    }
    count
}

/// What an account's security ID becomes.
pub const SID: &str = "<sid>";

/// `text` with every account SID hidden. Not a name, but it names one \[8\]
fn hide_sids(text: &str) -> String {
    const PREFIX: &str = "S-1-5-21-";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find(PREFIX) {
        let after = &rest[i + PREFIX.len()..];
        let len = after.find(|c: char| !(c.is_ascii_digit() || c == '-')).unwrap_or(after.len());
        out.push_str(&rest[..i]);
        out.push_str(SID);
        rest = &after[len..];
    }
    out.push_str(rest);
    out
}

fn push_name(list: &mut Vec<String>, name: &str) {
    let name = name.trim();
    if !name.is_empty() && !list.iter().any(|n| n.eq_ignore_ascii_case(name)) {
        list.push(name.to_string());
    }
}

fn marker(i: usize) -> String {
    format!("\u{E000}{i}\u{E001}")
}

/// Letters that continue a name: a match must not have one on either side. \[9\]
fn word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

fn separator(c: char) -> bool {
    c == '\\' || c == '/'
}

/// Where `name` matches `text` from byte `at`, ignoring case, and the byte \[10\]
fn match_at(text: &str, at: usize, name: &str) -> Option<usize> {
    let mut t = text[at..].char_indices();
    for n in name.chars() {
        let (_, c) = t.next()?;
        if !c.to_lowercase().eq(n.to_lowercase()) {
            return None;
        }
    }
    Some(t.next().map_or(text.len(), |(i, _)| at + i))
}

/// `text` with every whole-word match of `name` replaced by `with`. A short \[11\]
fn replace_name(text: &str, name: &str, with: &str) -> String {
    let short = name.chars().count() < WORD_MIN;
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut prev: Option<char> = None;
    for (at, c) in text.char_indices() {
        if at >= copied {
            if let Some(end) = match_at(text, at, name) {
                let next = text[end..].chars().next();
                // [12]
                let placeholder = prev == Some('<') && next == Some('>');
                let fits = !placeholder && if short {
                    prev.is_some_and(separator)
                        && next.is_none_or(|n| separator(n) || !word_char(n) && n != '.')
                } else {
                    !prev.is_some_and(word_char) && !next.is_some_and(word_char)
                };
                if fits {
                    out.push_str(&text[copied..at]);
                    out.push_str(with);
                    copied = end;
                }
            }
        }
        prev = Some(c);
    }
    out.push_str(&text[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r() -> Redactor {
        Redactor::new(&["Fule"], &["DESKTOP-7Q2K1"])
    }

    #[test]
    fn an_account_name_goes_from_paths_and_from_text() {
        let r = r();
        assert_eq!(r.apply(r"C:\Users\Fule\Music\out.wav"), r"C:\Users\<user>\Music\out.wav");
        assert_eq!(r.apply("c:/users/FULE/x"), "c:/users/<user>/x");
        assert_eq!(r.apply(r"D:\Fule's midis\a.mid"), r"D:\<user>'s midis\a.mid");
        assert_eq!(r.apply(r"C:\Users\Fule"), r"C:\Users\<user>");
        assert_eq!(r.apply("\"C:\\Users\\Fule\" and more"), "\"C:\\Users\\<user>\" and more");
    }

    #[test]
    fn a_name_inside_another_word_is_left_alone() {
        let r = r();
        assert_eq!(r.apply("Fuller Fule2 xFule"), "Fuller Fule2 xFule");
        assert_eq!(r.apply("DESKTOP-7Q2K12 is another PC"), "DESKTOP-7Q2K12 is another PC");
    }

    #[test]
    fn the_pc_name_goes_and_so_does_domain_backslash_user() {
        let r = r();
        assert_eq!(r.apply(r"DESKTOP-7Q2K1\Fule logged on"), r"<pc>\<user> logged on");
        assert_eq!(r.apply("host desktop-7q2k1."), "host <pc>.");
    }

    #[test]
    fn midi_soundfont_and_track_names_are_kept_whole() {
        let mut r = r();
        r.keep("Fule remix.mid");
        r.keep("Fule's Piano.sf2");
        r.keep("Fule");
        let line = r"C:\Users\Fule\Fule remix.mid with C:\sf\Fule's Piano.sf2";
        // "Fule" itself was kept, as a track name would be, so nothing goes.
        assert_eq!(r.apply(line), line);
        let mut r = Redactor::new(&["Fule"], &[]);
        r.keep("Fule remix.mid");
        assert_eq!(r.apply(r"C:\Users\Fule\Fule remix.mid"), r"C:\Users\<user>\Fule remix.mid");
    }

    #[test]
    fn a_short_name_only_goes_as_a_whole_path_segment() {
        let r = Redactor::new(&["al"], &[]);
        assert_eq!(r.apply(r"C:\Users\al\x.mid"), r"C:\Users\<user>\x.mid");
        assert_eq!(r.apply(r"C:\Users\al"), r"C:\Users\<user>");
        assert_eq!(r.apply("al is a total of al"), "al is a total of al");
        assert_eq!(r.apply(r"C:\x\al.mid"), r"C:\x\al.mid");
    }

    #[test]
    fn hiding_twice_changes_nothing_more() {
        let r = Redactor::new(&["user"], &["pc"]);
        let once = r.apply(r"C:\Users\user\x and user");
        assert_eq!(once, r"C:\Users\<user>\x and <user>");
        assert_eq!(r.apply(&once), once);
    }

    #[test]
    fn nothing_to_hide_changes_nothing() {
        let r = Redactor::new(&["", "  "], &[]);
        assert!(r.is_empty());
        assert_eq!(r.apply(r"C:\Users\Fule"), r"C:\Users\Fule");
    }

    #[test]
    fn an_account_sid_is_hidden_and_a_well_known_one_is_not() {
        let r = r();
        assert_eq!(
            r.apply("User: S-1-5-21-2308864636-1727320971-537463756-1001\nSystem: S-1-5-18"),
            "User: <sid>\nSystem: S-1-5-18"
        );
    }

    #[test]
    fn names_in_binary_data_are_overwritten_in_place() {
        let r = r();
        let mut data: Vec<u8> = b"\x01C:\\Users\\fule\\k.exe\x00".to_vec();
        data.extend("C:\\Users\\Fule\\".encode_utf16().flat_map(u16::to_le_bytes));
        data.extend(b"DESKTOP-7Q2K1 Fuller");
        let len = data.len();
        assert_eq!(r.scrub_bytes(&mut data), 3);
        assert_eq!(data.len(), len);
        let text = String::from_utf8_lossy(&data);
        assert!(text.contains(r"C:\Users\XXXX\k.exe"), "{text}");
        assert!(text.contains("XXXXXXXXXXXXX Fuller"), "{text}");
        let wide: Vec<u8> = "\\XXXX\\".encode_utf16().flat_map(u16::to_le_bytes).collect();
        assert!(data.windows(wide.len()).any(|w| w == wide.as_slice()));
    }

    #[test]
    fn text_that_is_not_ascii_survives() {
        let r = Redactor::new(&["Fule"], &[]);
        assert_eq!(r.apply("東方 Fule ♪ İstanbul"), "東方 <user> ♪ İstanbul");
    }
}
