// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! **FalconEye**: Kestrel's error catching and logging, kept apart from the \[1\]

pub(crate) mod observe;
pub mod redact;
pub mod renderlog;
pub mod report;
pub mod selftest;
pub mod system;
pub mod watch;
pub mod winsys;
pub mod zip;
