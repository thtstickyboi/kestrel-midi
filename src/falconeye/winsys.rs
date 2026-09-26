// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The raw system calls the watcher needs to tell one way of dying from \[1\]

use std::time::Duration;

/// Another process, held open so that its exit code can still be read after \[2\]
pub struct Process {
    #[cfg(windows)]
    handle: windows::Win32::Foundation::HANDLE,
}

// [3]
#[cfg(windows)]
unsafe impl Send for Process {}

#[cfg(windows)]
impl Process {
    /// Open process `pid` to wait on it and read its exit code, and to read \[4\]
    pub fn open(pid: u32) -> Option<Process> {
        use windows::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_VM_READ,
        };
        // [5]
        let handle = unsafe {
            OpenProcess(PROCESS_SYNCHRONIZE | PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, false, pid)
        }
        .ok()?;
        Some(Process { handle })
    }

    /// Wait up to `timeout` for the process to end. True once it has.
    pub fn wait(&self, timeout: Duration) -> bool {
        use windows::Win32::Foundation::WAIT_OBJECT_0;
        use windows::Win32::System::Threading::WaitForSingleObject;
        let ms = timeout.as_millis().min(u32::MAX as u128 - 1) as u32;
        // [6]
        unsafe { WaitForSingleObject(self.handle, ms) == WAIT_OBJECT_0 }
    }

    /// The exit code, once the process has ended: for a crash, the Windows \[7\]
    pub fn exit_code(&self) -> Option<u32> {
        use windows::Win32::System::Threading::GetExitCodeProcess;
        // [8]
        if !self.wait(Duration::ZERO) {
            return None;
        }
        let mut code = 0u32;
        // [9]
        unsafe { GetExitCodeProcess(self.handle, &mut code) }.ok()?;
        Some(code)
    }
}

#[cfg(windows)]
impl Drop for Process {
    fn drop(&mut self) {
        // SAFETY: the handle came from OpenProcess and is closed only here.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

#[cfg(not(windows))]
impl Process {
    pub fn open(_pid: u32) -> Option<Process> {
        None
    }
    pub fn wait(&self, _timeout: Duration) -> bool {
        false
    }
    pub fn exit_code(&self) -> Option<u32> {
        None
    }
}

// [10]

/// The dump the watcher writes. The threads, their stacks and the memory \[11\]
#[cfg(windows)]
const DUMP_TYPE: i32 = 0x1000 | 0x20 | 0x40; // WithThreadInfo | WithUnloadedModules | WithIndirectlyReferencedMemory

#[cfg(windows)]
mod filter {
    use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::Diagnostics::Debug::{
        SetUnhandledExceptionFilter, EXCEPTION_POINTERS, LPTOP_LEVEL_EXCEPTION_FILTER,
    };

    /// The watcher's stdin, as a raw handle; 0 while no render is watched.
    pub(super) static PIPE: AtomicIsize = AtomicIsize::new(0);
    /// The event the watcher sets once the dump is written.
    pub(super) static DUMPED: AtomicIsize = AtomicIsize::new(0);
    /// The filter that was installed before ours, called after it.
    static PREVIOUS: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn install() {
        // [12]
        let previous = unsafe { SetUnhandledExceptionFilter(Some(on_crash)) };
        if let Some(f) = previous {
            PREVIOUS.store(f as usize, Ordering::SeqCst);
        }
    }

    /// Called by Windows on the faulting thread, for an exception nothing \[13\]
    unsafe extern "system" fn on_crash(info: *const EXCEPTION_POINTERS) -> i32 {
        // Once only: a second fault while telling the watcher must not loop.
        let pipe = PIPE.swap(0, Ordering::SeqCst);
        if pipe != 0 && !info.is_null() {
            // [14]
            let (code, address) = unsafe {
                let rec = (*info).ExceptionRecord;
                if rec.is_null() {
                    (0, 0)
                } else {
                    ((*rec).ExceptionCode.0 as u32, (*rec).ExceptionAddress as usize)
                }
            };
            // SAFETY: GetCurrentThreadId has no preconditions.
            let thread = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
            let mut line = [0u8; 96];
            let n = super::crash_line(&mut line, code, address, thread, info as usize);
            // [15]
            unsafe {
                let _ = windows::Win32::Storage::FileSystem::WriteFile(
                    HANDLE(pipe as *mut _),
                    Some(&line[..n]),
                    None,
                    None,
                );
            }
            let event = DUMPED.load(Ordering::SeqCst);
            if event != 0 {
                // [16]
                unsafe {
                    windows::Win32::System::Threading::WaitForSingleObject(HANDLE(event as *mut _), 30_000);
                }
            }
        }
        let previous = PREVIOUS.load(Ordering::SeqCst);
        if previous != 0 {
            // [17]
            let f: LPTOP_LEVEL_EXCEPTION_FILTER = unsafe { std::mem::transmute(previous) };
            if let Some(f) = f {
                // [18]
                return unsafe { f(info) };
            }
        }
        0 // EXCEPTION_CONTINUE_SEARCH: Windows' own handling goes on.
    }
}

/// `crash <code> <address> <thread> <pointers>\n`, in hex but the thread, \[19\]
fn crash_line(buf: &mut [u8; 96], code: u32, address: usize, thread: u32, pointers: usize) -> usize {
    fn hex(buf: &mut [u8], at: &mut usize, v: u64, digits: u32) {
        for i in (0..digits).rev() {
            buf[*at] = b"0123456789abcdef"[((v >> (i * 4)) & 0xf) as usize];
            *at += 1;
        }
    }
    fn text(buf: &mut [u8], at: &mut usize, s: &[u8]) {
        buf[*at..*at + s.len()].copy_from_slice(s);
        *at += s.len();
    }
    let mut at = 0;
    text(buf, &mut at, b"crash ");
    hex(buf, &mut at, code as u64, 8);
    text(buf, &mut at, b" ");
    hex(buf, &mut at, address as u64, 16);
    text(buf, &mut at, b" ");
    // The thread in decimal, as the watcher parses it.
    let mut digits = [0u8; 10];
    let (mut n, mut t) = (0, thread);
    loop {
        digits[n] = b'0' + (t % 10) as u8;
        n += 1;
        t /= 10;
        if t == 0 {
            break;
        }
    }
    for i in (0..n).rev() {
        text(buf, &mut at, &digits[i..i + 1]);
    }
    text(buf, &mut at, b" ");
    hex(buf, &mut at, pointers as u64, 16);
    text(buf, &mut at, b"\n");
    at
}

/// Catch native crashes from now on. Harmless while no render is watched: \[20\]
pub fn install_crash_filter() {
    #[cfg(windows)]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(filter::install);
    }
}

/// A render has a watcher on `pipe`: tell it about a native crash.
#[cfg(windows)]
pub fn arm_crash_filter(pipe: &std::process::ChildStdin) {
    use std::os::windows::io::AsRawHandle;
    use std::sync::atomic::Ordering;
    use windows::core::HSTRING;
    use windows::Win32::System::Threading::CreateEventW;
    if filter::DUMPED.load(Ordering::SeqCst) == 0 {
        let name = HSTRING::from(dump_event_name(std::process::id()));
        // [21]
        if let Ok(event) = unsafe { CreateEventW(None, true, false, &name) } {
            filter::DUMPED.store(event.0 as isize, Ordering::SeqCst);
        }
    }
    filter::PIPE.store(pipe.as_raw_handle() as isize, Ordering::SeqCst);
}

#[cfg(not(windows))]
pub fn arm_crash_filter(_pipe: &std::process::ChildStdin) {}

/// The render is no longer watched; call before its pipe closes.
pub fn disarm_crash_filter() {
    #[cfg(windows)]
    filter::PIPE.store(0, std::sync::atomic::Ordering::SeqCst);
}

fn dump_event_name(pid: u32) -> String {
    format!("Local\\falconeye-dump-{pid}")
}

/// A crash as the filter reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crash {
    pub code: u32,
    pub address: u64,
    pub thread: u32,
    pub pointers: u64,
}

impl Crash {
    /// The words after `crash` in the filter's line.
    pub fn parse(words: &[&str]) -> Option<Crash> {
        let [code, address, thread, pointers] = words else { return None };
        Some(Crash {
            code: u32::from_str_radix(code, 16).ok()?,
            address: u64::from_str_radix(address, 16).ok()?,
            thread: thread.parse().ok()?,
            pointers: u64::from_str_radix(pointers, 16).ok()?,
        })
    }
}

#[cfg(windows)]
impl Process {
    /// Write a minidump of the process to `path`: at the crash the filter \[22\]
    pub fn dump(&self, pid: u32, path: &std::path::Path, crash: Option<Crash>) -> Result<(), String> {
        use std::os::windows::io::AsRawHandle;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::Diagnostics::Debug::{
            MiniDumpWriteDump, EXCEPTION_POINTERS, MINIDUMP_EXCEPTION_INFORMATION, MINIDUMP_TYPE,
        };
        let file = std::fs::File::create(path).map_err(|e| format!("creating the dump: {e}"))?;
        let exception = crash.map(|c| MINIDUMP_EXCEPTION_INFORMATION {
            ThreadId: c.thread,
            ExceptionPointers: c.pointers as usize as *mut EXCEPTION_POINTERS,
            // The pointers are the crashed process's, not ours.
            ClientPointers: true.into(),
        });
        // [23]
        let written = unsafe {
            MiniDumpWriteDump(
                self.handle,
                pid,
                HANDLE(file.as_raw_handle()),
                MINIDUMP_TYPE(DUMP_TYPE),
                exception.as_ref().map(|e| e as *const _),
                None,
                None,
            )
        };
        drop(file);
        written.map_err(|e| format!("MiniDumpWriteDump: {e}"))
    }

    /// Let the render's crash filter go on, once its dump is written.
    pub fn release(pid: u32) {
        use windows::core::HSTRING;
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::System::Threading::{OpenEventW, SetEvent, EVENT_MODIFY_STATE};
        let name = HSTRING::from(dump_event_name(pid));
        // [24]
        unsafe {
            if let Ok(event) = OpenEventW(EVENT_MODIFY_STATE, false, &name) {
                let _ = SetEvent(event);
                let _ = CloseHandle(event);
            }
        }
    }
}

#[cfg(not(windows))]
impl Process {
    pub fn dump(&self, _pid: u32, _path: &std::path::Path, _crash: Option<Crash>) -> Result<(), String> {
        Err("minidumps are Windows only".into())
    }
    pub fn release(_pid: u32) {}
}

// [25]

/// A string value under `HKEY_LOCAL_MACHINE\key`.
#[cfg(windows)]
pub fn reg_string(key: &str, name: &str) -> Option<String> {
    use windows::core::HSTRING;
    use windows::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};
    let (key, name) = (HSTRING::from(key), HSTRING::from(name));
    let mut bytes = 0u32;
    // SAFETY: a size query: no buffer, and `bytes` is a live local.
    let first = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, &key, &name, RRF_RT_REG_SZ, None, None, Some(&mut bytes)) };
    if first.0 != 0 || bytes == 0 {
        return None;
    }
    let mut buf = vec![0u16; (bytes as usize).div_ceil(2)];
    // [26]
    let second = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            &key,
            &name,
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr().cast()),
            Some(&mut bytes),
        )
    };
    if second.0 != 0 {
        return None;
    }
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    Some(String::from_utf16_lossy(&buf[..len]).trim().to_string())
}

/// A number value under `HKEY_LOCAL_MACHINE\key`.
#[cfg(windows)]
pub fn reg_dword(key: &str, name: &str) -> Option<u32> {
    use windows::core::HSTRING;
    use windows::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD};
    let (key, name) = (HSTRING::from(key), HSTRING::from(name));
    let mut value = 0u32;
    let mut bytes = 4u32;
    // SAFETY: `value` is a live u32 and the call is told it has 4 bytes.
    let r = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            &key,
            &name,
            RRF_RT_REG_DWORD,
            None,
            Some((&mut value as *mut u32).cast()),
            Some(&mut bytes),
        )
    };
    (r.0 == 0).then_some(value)
}

#[cfg(not(windows))]
pub fn reg_string(_key: &str, _name: &str) -> Option<String> {
    None
}

#[cfg(not(windows))]
pub fn reg_dword(_key: &str, _name: &str) -> Option<u32> {
    None
}

/// Physical memory and the commit limit (memory plus page files), in bytes.
#[derive(Debug, Clone, Copy)]
pub struct Memory {
    pub total: u64,
    pub available: u64,
    pub commit_limit: u64,
    pub commit_available: u64,
}

#[cfg(windows)]
pub fn memory() -> Option<Memory> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    // SAFETY: `m` is a live MEMORYSTATUSEX with its length set, as required.
    unsafe { GlobalMemoryStatusEx(&mut m) }.ok()?;
    Some(Memory {
        total: m.ullTotalPhys,
        available: m.ullAvailPhys,
        commit_limit: m.ullTotalPageFile,
        commit_available: m.ullAvailPageFile,
    })
}

#[cfg(not(windows))]
pub fn memory() -> Option<Memory> {
    None
}

/// Plugged in or not, and the battery's charge; `battery` is `None` with no \[27\]
#[derive(Debug, Clone, Copy)]
pub struct Power {
    pub plugged_in: Option<bool>,
    pub battery: Option<u8>,
}

#[cfg(windows)]
pub fn power() -> Option<Power> {
    use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
    let mut p = SYSTEM_POWER_STATUS::default();
    // SAFETY: `p` is a live SYSTEM_POWER_STATUS the call fills in.
    unsafe { GetSystemPowerStatus(&mut p) }.ok()?;
    Some(Power {
        plugged_in: match p.ACLineStatus {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        },
        // 128: no battery; 255: unknown.
        battery: (p.BatteryFlag & 128 == 0 && p.BatteryLifePercent <= 100).then_some(p.BatteryLifePercent),
    })
}

#[cfg(not(windows))]
pub fn power() -> Option<Power> {
    None
}

// ---- The administrator step ------------------------------------------------

/// How an elevated run went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Elevated {
    Exited(u32),
    /// The user said no at Windows' permission prompt.
    Declined,
    TimedOut,
    Failed(String),
}

/// Run `exe args` as administrator, through Windows' own permission prompt \[28\]
#[cfg(windows)]
pub fn run_elevated(exe: &std::path::Path, args: &str, timeout: Duration) -> Elevated {
    use windows::core::{HSTRING, PCWSTR};
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{GetExitCodeProcess, WaitForSingleObject};
    use windows::Win32::UI::Shell::{ShellExecuteExW, SEE_MASK_NOASYNC, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW};
    let (verb, file, params) = (HSTRING::from("runas"), HSTRING::from(exe.as_os_str()), HSTRING::from(args));
    let mut info = SHELLEXECUTEINFOW {
        cbSize: std::mem::size_of::<SHELLEXECUTEINFOW>() as u32,
        fMask: SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC,
        lpVerb: PCWSTR(verb.as_ptr()),
        lpFile: PCWSTR(file.as_ptr()),
        lpParameters: PCWSTR(params.as_ptr()),
        nShow: 0, // SW_HIDE: the elevated part has nothing to show
        ..Default::default()
    };
    // [29]
    if let Err(e) = unsafe { ShellExecuteExW(&mut info) } {
        // HRESULT_FROM_WIN32(ERROR_CANCELLED): the prompt was refused.
        return if e.code().0 as u32 == 0x8007_04C7 { Elevated::Declined } else { Elevated::Failed(e.to_string()) };
    }
    if info.hProcess.is_invalid() {
        return Elevated::Failed("no process was started".into());
    }
    let ms = timeout.as_millis().min(u32::MAX as u128 - 1) as u32;
    // [30]
    unsafe {
        let done = WaitForSingleObject(info.hProcess, ms) == windows::Win32::Foundation::WAIT_OBJECT_0;
        let mut code = 0u32;
        let read = GetExitCodeProcess(info.hProcess, &mut code).is_ok();
        let _ = CloseHandle(info.hProcess);
        match (done, read) {
            (true, true) => Elevated::Exited(code),
            (true, false) => Elevated::Failed("its exit code could not be read".into()),
            (false, _) => Elevated::TimedOut,
        }
    }
}

#[cfg(not(windows))]
pub fn run_elevated(_exe: &std::path::Path, _args: &str, _timeout: Duration) -> Elevated {
    Elevated::Failed("the administrator step is Windows only".into())
}

// ---- DX12's reason for a lost device --------------------------------------

/// Why DX12 removed `device`, when it is a DX12 device: the one thing that \[31\]
#[cfg(windows)]
pub fn dx12_removed_reason(device: &wgpu::Device) -> Option<(u32, &'static str)> {
    // [32]
    let code = unsafe {
        let hal = device.as_hal::<wgpu::hal::api::Dx12>()?;
        match hal.raw_device().GetDeviceRemovedReason() {
            Ok(()) => 0,
            Err(e) => e.code().0 as u32,
        }
    };
    Some((code, describe_removed(code)))
}

#[cfg(not(windows))]
pub fn dx12_removed_reason(_device: &wgpu::Device) -> Option<(u32, &'static str)> {
    None
}

/// What DX12's removed reason means.
pub fn describe_removed(code: u32) -> &'static str {
    match code {
        0 => "none: DX12 does not see the device as removed, so this was not a driver reset",
        0x887A_0006 => "DXGI_ERROR_DEVICE_HUNG: the GPU took too long on one piece of work and Windows reset it, which is the 2-second limit",
        0x887A_0007 => "DXGI_ERROR_DEVICE_RESET: the GPU was reset, by a badly formed command or by another program",
        0x887A_0005 => "DXGI_ERROR_DEVICE_REMOVED: the GPU went away, as it does when the driver is updated or restarts",
        0x887A_0020 => "DXGI_ERROR_DRIVER_INTERNAL_ERROR: the graphics driver failed internally",
        0x887A_0001 => "DXGI_ERROR_INVALID_CALL: the device was given a call it could not take",
        0x8007_000E => "E_OUTOFMEMORY: the device ran out of memory",
        _ => "a reason Kestrel does not recognise",
    }
}

/// Crash with an access violation, for `KESTREL_CRASH_TEST=segv`: the native \[33\]
#[cfg(feature = "dev")]
pub fn access_violation() -> ! {
    // [34]
    unsafe { (0x10 as *mut u8).write_volatile(1) };
    unreachable!("the write above faults")
}

/// What an exit code means, for the codes a render dies with.
pub fn describe_exit(code: u32) -> &'static str {
    match code {
        0 => "exited normally, but without finishing its log",
        1 => "was ended from outside, as Task Manager's End task and taskkill /F do, or exited with an error before finishing its log",
        0xFFFF_FFFF => "was ended from outside, as PowerShell's Stop-Process does",
        101 => "stopped on a panic in Kestrel",
        0xC000_013A => "was closed: the window was closed or Ctrl+C was pressed",
        0xC000_0005 => "crashed: an access violation (STATUS_ACCESS_VIOLATION), a read or write of memory it did not own. Inside a graphics driver this is usually the driver's fault",
        0xC000_0409 => "aborted (STATUS_STACK_BUFFER_OVERRUN, which is also how an abort or a fail-fast ends a process)",
        0xC000_00FD => "crashed: a stack overflow",
        0xC000_0374 => "crashed: the heap was corrupted",
        0xC000_001D => "crashed: an illegal instruction",
        0x8000_0003 => "crashed: a breakpoint was hit outside a debugger",
        0xC000_0017 | 0xC000_012D => "ran out of memory",
        0xC000_0142 => "failed to start: a DLL could not initialise",
        _ => "ended with an exit code Kestrel does not recognise",
    }
}

/// Whether an exit code is a crash, as opposed to a normal exit or a close.
pub fn is_crash(code: u32) -> bool {
    !matches!(code, 0 | 101 | 0xC000_013A)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_filters_line_reads_back_as_the_crash_it_describes() {
        let mut buf = [0u8; 96];
        let n = crash_line(&mut buf, 0xC000_0005, 0x7ff6_1234_5678, 30092, 0x0000_00ab_cdef_0010);
        let line = std::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(line, "crash c0000005 00007ff612345678 30092 000000abcdef0010\n");
        let words: Vec<&str> = line.split_whitespace().skip(1).collect();
        assert_eq!(
            Crash::parse(&words),
            Some(Crash { code: 0xC000_0005, address: 0x7ff6_1234_5678, thread: 30092, pointers: 0xab_cdef_0010 })
        );
        let n = crash_line(&mut buf, u32::MAX, usize::MAX, u32::MAX, usize::MAX);
        assert!(n <= buf.len());
    }
}
