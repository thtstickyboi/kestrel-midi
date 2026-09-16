// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! How much video memory an adapter has, and how much of it is in use -- by \[1\]

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// One reading of an adapter's memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct GpuMemory {
    /// Dedicated video memory the adapter has.
    pub dedicated_total: u64,
    /// Dedicated video memory in use by every process on the adapter: the \[2\]
    pub dedicated_used: Option<u64>,
    /// System memory the adapter has mapped, for every process. Most of what \[3\]
    pub shared_used: Option<u64>,
    /// Dedicated memory this process is using.
    pub process_used: Option<u64>,
    /// How much dedicated memory the OS lets this process use before it \[4\]
    pub process_budget: Option<u64>,
}

/// How often a `VramWatch` takes a reading.
const EVERY: Duration = Duration::from_secs(1);
/// How often its thread checks whether the watch has been dropped.
const STOP_CHECK: Duration = Duration::from_millis(100);

/// Reads an adapter's memory on a thread of its own, once a second, so asking \[5\]
pub struct VramWatch {
    latest: Arc<Mutex<Option<GpuMemory>>>,
    stop: Arc<AtomicBool>,
}

impl VramWatch {
    /// Start watching the adapter with these PCI vendor and device ids, the \[6\]
    pub fn start(vendor: u32, device: u32) -> VramWatch {
        let latest = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let (slot, stopped) = (Arc::clone(&latest), Arc::clone(&stop));
        // [7]
        let _ = std::thread::Builder::new()
            .name("kestrel-vram".into())
            .spawn(move || {
                let Some(mut probe) = Probe::open(vendor, device) else {
                    return;
                };
                while !stopped.load(Ordering::Relaxed) {
                    let reading = probe.read();
                    if let Ok(mut s) = slot.lock() {
                        *s = Some(reading);
                    }
                    let mut waited = Duration::ZERO;
                    while waited < EVERY && !stopped.load(Ordering::Relaxed) {
                        std::thread::sleep(STOP_CHECK);
                        waited += STOP_CHECK;
                    }
                }
            });
        VramWatch { latest, stop }
    }

    pub fn latest(&self) -> Option<GpuMemory> {
        self.latest.lock().ok().and_then(|s| *s)
    }
}

impl Drop for VramWatch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// One reading, taken now. Slow the first time in a process; read it through \[8\]
pub fn sample(vendor: u32, device: u32) -> Option<GpuMemory> {
    Probe::open(vendor, device).map(|mut probe| probe.read())
}

/// The instance-name prefix the performance counters give an adapter, \[9\]
#[cfg_attr(not(windows), allow(dead_code))]
fn counter_prefix(luid_high: i32, luid_low: u32) -> String {
    format!("luid_0x{:08x}_0x{:08x}_phys_", luid_high as u32, luid_low)
}

#[cfg(not(windows))]
struct Probe;

#[cfg(not(windows))]
impl Probe {
    fn open(_vendor: u32, _device: u32) -> Option<Probe> {
        None
    }

    fn read(&mut self) -> GpuMemory {
        GpuMemory::default()
    }
}

#[cfg(windows)]
use windows_probe::Probe;

#[cfg(windows)]
mod windows_probe {
    use super::{counter_prefix, GpuMemory};
    use windows::core::{w, Interface, PCWSTR};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    use windows::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
        PdhOpenQueryW, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE,
        PDH_MORE_DATA,
    };

    pub(super) struct Probe {
        /// For this process's usage and budget. Absent before Windows 10.
        adapter: Option<IDXGIAdapter3>,
        dedicated_total: u64,
        prefix: String,
        counters: Option<Counters>,
    }

    impl Probe {
        pub(super) fn open(vendor: u32, device: u32) -> Option<Probe> {
            // [10]
            let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
            let mut index = 0;
            loop {
                // [11]
                let adapter = unsafe { factory.EnumAdapters1(index) }.ok()?;
                index += 1;
                // SAFETY: fills a plain-data struct from a live adapter.
                let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
                    continue;
                };
                if desc.VendorId != vendor || desc.DeviceId != device {
                    continue;
                }
                return Some(Probe {
                    adapter: adapter.cast::<IDXGIAdapter3>().ok(),
                    dedicated_total: desc.DedicatedVideoMemory as u64,
                    prefix: counter_prefix(desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart),
                    counters: Counters::open(),
                });
            }
        }

        pub(super) fn read(&mut self) -> GpuMemory {
            let mut reading = GpuMemory {
                dedicated_total: self.dedicated_total,
                ..Default::default()
            };
            if let Some(adapter) = &self.adapter {
                let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                // [12]
                let asked = unsafe {
                    adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
                };
                if asked.is_ok() {
                    reading.process_used = Some(info.CurrentUsage);
                    reading.process_budget = Some(info.Budget);
                }
            }
            if let Some(counters) = &self.counters {
                if counters.collect() {
                    reading.dedicated_used = counters.sum(counters.dedicated, &self.prefix);
                    reading.shared_used = counters.sum(counters.shared, &self.prefix);
                }
            }
            reading
        }
    }

    /// The two "GPU Adapter Memory" counters, over every adapter instance.
    struct Counters {
        query: isize,
        dedicated: isize,
        shared: isize,
    }

    impl Counters {
        fn open() -> Option<Counters> {
            let mut query = 0isize;
            // [13]
            if unsafe { PdhOpenQueryW(PCWSTR::null(), 0, &mut query) } != 0 {
                return None;
            }
            let (mut dedicated, mut shared) = (0isize, 0isize);
            // [14]
            let added = unsafe {
                PdhAddEnglishCounterW(query, w!("\\GPU Adapter Memory(*)\\Dedicated Usage"), 0, &mut dedicated) == 0
                    && PdhAddEnglishCounterW(query, w!("\\GPU Adapter Memory(*)\\Shared Usage"), 0, &mut shared) == 0
            };
            if !added {
                // SAFETY: closes the query opened above, once.
                unsafe { PdhCloseQuery(query) };
                return None;
            }
            Some(Counters {
                query,
                dedicated,
                shared,
            })
        }

        fn collect(&self) -> bool {
            // SAFETY: the query stays open for as long as `self` lives.
            unsafe { PdhCollectQueryData(self.query) == 0 }
        }

        /// A counter's value summed over this adapter's instances, or `None` \[15\]
        fn sum(&self, counter: isize, prefix: &str) -> Option<u64> {
            let (mut bytes, mut count) = (0u32, 0u32);
            // [16]
            let status = unsafe {
                PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut bytes, &mut count, None)
            };
            if status != PDH_MORE_DATA {
                return None;
            }
            // [17]
            let mut buf = vec![0u64; (bytes as usize).div_ceil(8)];
            let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
            // [18]
            let status = unsafe {
                PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut bytes, &mut count, Some(items))
            };
            if status != 0 {
                return None;
            }
            let mut total = 0u64;
            let mut found = false;
            for i in 0..count as usize {
                // [19]
                let (name, valid, value) = unsafe {
                    let item = &*items.add(i);
                    (
                        item.szName.to_string().unwrap_or_default(),
                        item.FmtValue.CStatus == PDH_CSTATUS_VALID_DATA,
                        item.FmtValue.Anonymous.largeValue,
                    )
                };
                if valid && name.to_ascii_lowercase().starts_with(prefix) {
                    total += value.max(0) as u64;
                    found = true;
                }
            }
            found.then_some(total)
        }
    }

    impl Drop for Counters {
        fn drop(&mut self) {
            // SAFETY: the query was opened in `open` and is closed only here.
            unsafe { PdhCloseQuery(self.query) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The instance names as Windows writes them, taken from `Get-Counter \[20\]
    #[test]
    fn the_counter_prefix_matches_the_instance_names_windows_uses() {
        assert_eq!(counter_prefix(0, 0x0001_1d6f), "luid_0x00000000_0x00011d6f_phys_");
        assert!("luid_0x00000000_0x00011d6f_phys_0".starts_with(&counter_prefix(0, 0x11d6f)));
        assert_eq!(counter_prefix(-1, 1), "luid_0xffffffff_0x00000001_phys_");
    }

    /// The adapter wgpu would render on reports a total, and a usage no \[21\]
    #[test]
    #[cfg(windows)]
    fn the_render_adapter_reports_its_memory() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .map(|a| a.get_info())
            .find(|i| i.device_type == wgpu::DeviceType::DiscreteGpu)
        else {
            return;
        };
        let m = sample(adapter.vendor, adapter.device).expect("DXGI knows the adapter wgpu found");
        assert!(m.dedicated_total > 0, "{m:?}");
        let used = m.dedicated_used.expect("the performance counters report it");
        assert!(used <= m.dedicated_total, "{m:?}");
        assert!(m.process_budget.is_some_and(|b| b > 0), "{m:?}");
    }
}
