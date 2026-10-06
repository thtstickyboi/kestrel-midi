// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Windows version information for kestrel.exe, added in 1.2.3: the version
//! comes from `Cargo.toml`, so every release carries its own. Without it
//! Windows' crash records gave the exe as version 0.0.0.0, and an exe with no
//! version information is one more thing antivirus heuristics count against
//! it. Nothing is done on other platforms.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    #[cfg(windows)]
    windows_version_info();
}

#[cfg(windows)]
fn windows_version_info() {
    // The host is Windows; so must the target be.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let mut res = winresource::WindowsResource::new();
    res.set("ProductName", "Kestrel")
        .set("FileDescription", "Kestrel, a GPU renderer for black MIDI")
        .set("InternalName", "kestrel")
        .set("OriginalFilename", "kestrel.exe");
    // A machine without the resource compiler still builds, only without
    // the version information, and says so.
    if let Err(e) = res.compile() {
        println!("cargo:warning=kestrel.exe gets no version information: {e}");
    }
}
