// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Whether a newer Kestrel has been released: one request to GitHub's releases \[1\]

use crate::settings::Ring;
use anyhow::{bail, Context, Result};
use std::process::{Command, Stdio};

/// The latest published release. Drafts and pre-releases are not "latest", so \[2\]
const LATEST: &str = "https://api.github.com/repos/thtstickyboi/kestrel-midi/releases/latest";

/// Where a person is sent when the API's answer does not carry its own page.
pub const RELEASES: &str = "https://github.com/thtstickyboi/kestrel-midi/releases/latest";

/// Set to anything but `0` or empty, and nothing is fetched.
pub const OPT_OUT: &str = "KESTREL_NO_UPDATE_CHECK";

/// This build's version.
pub const CURRENT: &str = env!("CARGO_PKG_VERSION");

/// What the check found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Latest {
    /// The latest release's version, without the tag's `v`.
    pub version: String,
    /// Its release page.
    pub url: String,
    /// Whether it is newer than this build. A development build ahead of the \[3\]
    pub newer: bool,
    /// Whether it is newer in its major or minor number: a feature release, \[4\]
    pub newer_feature: bool,
}

impl Latest {
    /// Whether someone on `ring` should be told about it.
    pub fn announced(&self, ring: Ring) -> bool {
        match ring {
            Ring::Fast => self.newer,
            Ring::Slow => self.newer_feature,
        }
    }
}

/// Whether `KESTREL_NO_UPDATE_CHECK` is set.
pub fn opted_out() -> bool {
    std::env::var_os(OPT_OUT).is_some_and(|v| !v.is_empty() && v != "0")
}

/// Ask GitHub. Bounded at a few seconds, so an offline machine or a filtered \[5\]
pub fn check() -> Result<Latest> {
    let out = Command::new("curl")
        .args([
            "-fsSL",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--tlsv1.2",
            "--connect-timeout",
            "3",
            "--max-time",
            "5",
            "-A",
            concat!("kestrel/", env!("CARGO_PKG_VERSION")),
            "-H",
            "Accept: application/vnd.github+json",
            LATEST,
        ])
        .stdin(Stdio::null())
        .output()
        .context("couldn't run curl")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim().trim_start_matches("curl: ");
        // [6]
        let err = match err.split_once(") ") {
            Some((code, rest)) if code.starts_with('(') => rest,
            _ => err,
        };
        if err.is_empty() {
            bail!("GitHub didn't answer");
        }
        bail!("{err}");
    }
    parse(&out.stdout, CURRENT)
}

/// Read the API's answer and compare its tag against `current`.
fn parse(body: &[u8], current: &str) -> Result<Latest> {
    let v: serde_json::Value =
        serde_json::from_slice(body).context("GitHub's answer wasn't JSON")?;
    let tag = v["tag_name"].as_str().context("GitHub's answer had no release tag")?;
    let latest = numbers(tag).with_context(|| format!("unrecognised release tag {tag:?}"))?;
    let ours = numbers(current).with_context(|| format!("unrecognised version {current:?}"))?;
    Ok(Latest {
        version: tag.trim_start_matches(['v', 'V']).to_string(),
        url: v["html_url"].as_str().unwrap_or(RELEASES).to_string(),
        newer: latest > ours,
        newer_feature: (latest.0, latest.1) > (ours.0, ours.1),
    })
}

/// `v1.2.3` as `(1, 2, 3)`. A missing part is 0, and a pre-release or build \[7\]
fn numbers(tag: &str) -> Option<(u64, u64, u64)> {
    let core = tag.trim_start_matches(['v', 'V']);
    let core = core.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let mut next = || -> Option<u64> {
        match parts.next() {
            None => Some(0),
            Some(p) => p.parse().ok(),
        }
    };
    let n = (next()?, next()?, next()?);
    if parts.next().is_some() {
        return None;
    }
    Some(n)
}

/// `kestrel --force-cli check-update`. Asked for explicitly, so it checks \[8\]
pub fn run_cli() -> Result<()> {
    let latest = check().context("couldn't check for updates")?;
    if latest.newer {
        println!("Kestrel {} is out (this is {CURRENT}).", latest.version);
        println!("Download it from {}", latest.url);
        if !latest.newer_feature && crate::settings::load().0.ring == Ring::Slow {
            println!("It's a fix release, which the Slow Ring doesn't announce.");
        }
    } else {
        println!("Kestrel {CURRENT} is up to date (latest release {}).", latest.version);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(tag: &str) -> Vec<u8> {
        format!(
            r#"{{"tag_name": "{tag}", "html_url": "https://github.com/x/y/releases/tag/{tag}", "draft": false}}"#
        )
        .into_bytes()
    }

    #[test]
    fn a_later_release_is_newer() {
        let got = parse(&answer("v1.1.1"), "1.1.0").unwrap();
        assert!(got.newer);
        assert_eq!(got.version, "1.1.1");
        assert_eq!(got.url, "https://github.com/x/y/releases/tag/v1.1.1");
        assert!(parse(&answer("v1.2.0"), "1.1.9").unwrap().newer);
        assert!(parse(&answer("v2.0.0"), "1.10.0").unwrap().newer);
    }

    #[test]
    fn the_same_or_an_older_release_is_not() {
        assert!(!parse(&answer("v1.1.0"), "1.1.0").unwrap().newer);
        // A development build ahead of the latest release.
        assert!(!parse(&answer("v1.1.0"), "1.1.1").unwrap().newer);
        // Compared as numbers, not text.
        assert!(!parse(&answer("v1.9.0"), "1.10.0").unwrap().newer);
    }

    #[test]
    fn the_slow_ring_hears_only_of_feature_releases() {
        let heard = |tag: &str, ours: &str, ring| parse(&answer(tag), ours).unwrap().announced(ring);
        // A fix release: fast hears of it, slow does not.
        assert!(heard("v1.1.1", "1.1.0", Ring::Fast));
        assert!(!heard("v1.1.1", "1.1.0", Ring::Slow));
        // Minor and major releases: both hear, from any fix of the version before.
        for (tag, ours) in [("v1.2.0", "1.1.0"), ("v1.3.4", "1.1.7"), ("v2.0.0", "1.9.9"), ("v2.4.0", "2.3.1")] {
            assert!(heard(tag, ours, Ring::Fast), "{tag} over {ours}");
            assert!(heard(tag, ours, Ring::Slow), "{tag} over {ours}");
        }
        // Once on the feature release, its fixes are the slow ring's to skip.
        assert!(!heard("v1.2.3", "1.2.0", Ring::Slow));
        // Nobody hears of an old release.
        assert!(!heard("v1.1.0", "1.2.0", Ring::Fast));
        assert!(!heard("v1.1.0", "1.2.0", Ring::Slow));
    }

    #[test]
    fn tags_are_read_leniently_but_not_blindly() {
        assert_eq!(numbers("v1.1"), Some((1, 1, 0)));
        assert_eq!(numbers("1.1.0"), Some((1, 1, 0)));
        assert_eq!(numbers("v1.2.0-rc1"), Some((1, 2, 0)));
        assert!(!parse(&answer("v1.1.0-rc1"), "1.1.0").unwrap().newer);
        assert_eq!(numbers("nightly"), None);
        assert_eq!(numbers("v1.2.3.4"), None);
        assert!(parse(&answer("latest"), "1.1.0").is_err());
    }

    #[test]
    fn an_answer_without_a_page_points_at_the_releases() {
        let got = parse(br#"{"tag_name": "v9.0.0"}"#, "1.1.0").unwrap();
        assert_eq!(got.url, RELEASES);
        assert!(parse(b"<html>rate limited</html>", "1.1.0").is_err());
        assert!(parse(br#"{"message": "Not Found"}"#, "1.1.0").is_err());
    }
}
