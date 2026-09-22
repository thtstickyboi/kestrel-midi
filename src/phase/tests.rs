use super::*;
use crate::{
    backend::Backend,
    config::{Config, Interpolation},
    cpu::CpuSynth,
    driver::Driver,
    gpu::GpuSynth,
    midi::MidiWriter,
    testkit::*,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn settings() -> PhaseSettings {
    PhaseSettings {
        mode: PhaseMode::Analytic,
        seed: 42,
        ..Default::default()
    }
}

#[test]
fn periodic_quadrature_has_the_correct_sign_at_even_odd_and_prime_lengths() {
    for n in [2, 16, 31, 64, 97, 255] {
        let x: Vec<_> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).cos())
            .collect();
        let q = quadrature(&x, false, &|| false).unwrap();
        for (i, &value) in q.iter().enumerate() {
            let expected = if n == 2 {
                0.0
            } else {
                (2.0 * PI * i as f64 / n as f64).sin()
            };
            assert!(
                (value as f64 - expected).abs() < 2e-6,
                "n={n}, i={i}: {value} vs {expected}"
            );
        }
    }
}

#[test]
fn dc_nyquist_and_empty_inputs_have_zero_quadrature() {
    for x in [
        vec![],
        vec![0.7],
        vec![0.7; 31],
        vec![0.7; 32],
        (0..32)
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.7 })
            .collect(),
    ] {
        assert!(quadrature(&x, false, &|| false)
            .unwrap()
            .iter()
            .all(|x| x.abs() < 1e-6));
    }
}

#[test]
fn padded_impulse_matches_the_discrete_hilbert_kernel() {
    let mut x = vec![0.0; 29];
    x[0] = 1.0;
    let q = quadrature(&x, true, &|| false).unwrap();
    for (i, &v) in q.iter().enumerate() {
        let expected = if i % 2 == 0 {
            0.0
        } else {
            2.0 / (64.0 * (PI * i as f64 / 64.0).tan())
        };
        assert!((v as f64 - expected).abs() < 1e-6);
    }
}

#[test]
fn fft_cancellation_is_checked_inside_the_transform() {
    let checks = AtomicUsize::new(0);
    let cancel = || checks.fetch_add(1, Ordering::Relaxed) > 8;
    let err = quadrature(&vec![0.1; 100_000], true, &cancel).unwrap_err();
    assert!(err.is::<PreparationCancelled>());
}

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    dir: PathBuf,
    bank: Arc<Bank>,
}
impl Fixture {
    fn new(cfg: &Config, delayed: bool) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("out/phase-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut sample = TestSample::new(
            "prime-loop",
            (0..503)
                .map(|i| {
                    (8000.0 * (2.0 * PI * i as f64 / 97.0).cos()
                        + 3000.0 * (6.0 * PI * i as f64 / 97.0).sin()) as i16
                })
                .collect(),
            8000,
            60,
        );
        sample.loop_start = 97;
        sample.loop_end = 388;
        let zone = || {
            let z = TestZone::new(0).gen(54, 3).gen(34, -12000).gen(38, -6000);
            if delayed {
                z.gen(33, -4800)
            } else {
                z
            }
        };
        let sf = Sf2Builder {
            samples: vec![sample],
            instruments: vec![TestInstrument {
                name: "test".into(),
                zones: vec![zone(), zone().gen(17, 250)],
            }],
            presets: vec![TestPreset {
                name: "test".into(),
                bank: 0,
                program: 0,
                instrument: 0,
                gens: vec![],
            }],
        };
        sf.write(dir.join("bank.sf2")).unwrap();
        let bank = Arc::new(crate::load_bank(dir.join("bank.sf2"), cfg).unwrap());
        Self { dir, bank }
    }

    fn distinct_stereo(mut self, cfg: &Config) -> Self {
        let bank = Arc::get_mut(&mut self.bank).unwrap();
        let mut right = bank.samples[0].clone();
        right.start = bank.pool.len() as u32;
        for i in 0..right.len {
            bank.pool
                .push((6000.0 * (4.0 * PI * i as f64 / 97.0).sin()) as i16);
        }
        bank.pool.extend([0; 64]);
        bank.samples.push(right);
        bank.regions[0].pan = -1.0;
        bank.regions[1].sample = 1;
        bank.regions[1].pan = 1.0;
        bank.regions[1].addr_start = 3;
        bank.regions[1].addr_end = -7;
        bank.regions[1].addr_loop_start = 2;
        bank.regions[1].addr_loop_end = -3;
        bank.build_params(cfg);
        self
    }

    fn midi(&self, dense: bool) -> PathBuf {
        let mut m = MidiWriter::new(32767);
        m.tempo_track(500_000);
        let mut events = vec![];
        for i in 0..if dense { 600 } else { 18 } {
            // Different source ticks can land in one output frame.
            let t = if dense {
                i / 2 + (i / 200) * 5000
            } else {
                i * 97
            };
            let key = 60 + (i % 7) as u8;
            events.push((t, [0x90, key, 75 + (i % 50) as u8], 3));
            events.push((t + 18000, [0x80, key, 0], 3));
        }
        events.push((1100, [0xE0, 0, 68], 3));
        events.push((1300, [0xB0, 11, 90], 3));
        m.track(events);
        let path = self.dir.join("notes.mid");
        m.save(&path).unwrap();
        path
    }
}

fn config() -> Config {
    Config {
        sample_rate: 8000,
        block_frames: 128,
        max_voices: 32,
        max_block_candidates: 64,
        max_render_workgroups: 2,
        phase: settings(),
        limiter: false,
        clamp_output: false,
        lfo_enabled: false,
        mod_env_enabled: false,
        filter_enabled: false,
        ..Default::default()
    }
}

fn render(cfg: &Config, fixture: &Fixture, midi: &std::path::Path, gpu: bool) -> (Vec<f32>, u64) {
    let phase = PhaseBank::prepare(&fixture.bank, &cfg.phase).unwrap();
    let mut driver = Driver::open_prepared(cfg, fixture.bank.clone(), midi, phase.clone()).unwrap();
    let mut backend: Box<dyn Backend> = if gpu {
        let g = GpuSynth::new_prepared(cfg, fixture.bank.clone(), phase).unwrap();
        eprintln!("phase GPU test: {}", g.adapter_name());
        Box::new(g)
    } else {
        Box::new(CpuSynth::new_prepared(cfg, fixture.bank.clone(), phase))
    };
    let mut block = vec![0.0; cfg.block_samples()];
    let mut audio = vec![];
    for _ in 0..100 {
        let more = driver.next_block(&mut *backend, &mut block).unwrap();
        audio.extend_from_slice(&block);
        if !more {
            return (audio, driver.stats.dropped);
        }
    }
    panic!("fixture did not finish");
}

#[test]
fn cache_deduplicates_layers_but_distinguishes_effective_loops() {
    let cfg = config();
    let mut fixture = Fixture::new(&cfg, false);
    let one = PhaseBank::prepare(&fixture.bank, &cfg.phase).unwrap();
    assert_eq!(one.sample_count(), 1);
    Arc::get_mut(&mut fixture.bank).unwrap().regions[1].addr_loop_end = -1;
    let two = PhaseBank::prepare(&fixture.bank, &cfg.phase).unwrap();
    assert_eq!(two.sample_count(), 2);
    assert!(two.cache_bytes() > one.cache_bytes());
}

#[test]
fn budgets_and_cancellation_fail_before_publishing_a_cache() {
    let cfg = config();
    let fixture = Fixture::new(&cfg, false);
    let mut s = cfg.phase.clone();
    s.cache_budget_bytes = 1;
    assert!(PhaseBank::prepare(&fixture.bank, &s)
        .err()
        .unwrap()
        .to_string()
        .contains("phase-cache-mib"));
    s.cache_budget_bytes = 1 << 20;
    s.scratch_budget_bytes = 1;
    assert!(PhaseBank::prepare(&fixture.bank, &s)
        .err()
        .unwrap()
        .to_string()
        .contains("phase-scratch-mib"));
    assert!(
        PhaseBank::prepare_with(&fixture.bank, &cfg.phase, &|| true, &mut |_, _| {})
            .err()
            .unwrap()
            .is::<PreparationCancelled>()
    );
    s.strength = 0.0;
    assert_eq!(
        PhaseBank::prepare(&fixture.bank, &s).unwrap().cache_bytes(),
        0
    );
}

#[test]
fn angles_share_tick_channel_key_and_finite_scales_match_continuous_formula() {
    let cfg = config();
    let fixture = Fixture::new(&cfg, false);
    let p = PhaseBank::prepare(&fixture.bank, &cfg.phase).unwrap();
    for tick in 0..128 {
        let a = p.angle(tick, 3, 60);
        let c = p.coefficients(0, a);
        assert_eq!(c, p.coefficients(1, a));
        assert_eq!(c.scale, normalization(p.entries[0].energy, a));
    }
    let mut s = cfg.phase.clone();
    s.continuous = true;
    s.pool_size = 0; // ignored in continuous mode
    let p = PhaseBank::prepare(&fixture.bank, &s).unwrap();
    assert_ne!(p.angle(42, 0, 60).sine, p.angle(43, 0, 60).sine);
    assert_ne!(p.angle(42, 0, 60).sine, p.angle(42, 1, 60).sine);
}

#[test]
fn attack_prefix_is_unchanged_and_loop_body_is_periodic() {
    let mut cfg = config();
    cfg.phase.preserve_attack_ms = 2.0;
    let fixture = Fixture::new(&cfg, false);
    let p = PhaseBank::prepare(&fixture.bank, &cfg.phase).unwrap();
    let c = p.coefficients(0, p.angle(42, 0, 60));
    let s = &fixture.bank.samples[0];
    for i in 0..16 {
        let original = decode(fixture.bank.pool[(s.start + i) as usize]);
        assert_eq!(original, p.apply(original, 0, i, c));
    }
    let base = p.words[0] as usize;
    for i in 97..194 {
        assert!(
            (f32::from_bits(p.words[base + i]) - f32::from_bits(p.words[base + i + 97])).abs()
                < 1e-6
        );
    }
}

#[test]
fn matches_syncore_phase_processor_reference_values() {
    // Generated by compiling SYNCore 4cd9373 SAFSYN/phase.cpp directly with
    // MSVC /O2, using PCM[i] = (i*997)%20001-10000, 97 frames at 8 kHz,
    // sustain loop [17,78), tick 1234567890123, channel 3, key 60, seed 42.
    // Quadrature/output scale by 32768/32767 for Kestrel's PCM convention.
    let mut cfg = config();
    cfg.phase.preserve_attack_ms = 2.0;
    let mut fixture = Fixture::new(&cfg, false);
    let bank = Arc::get_mut(&mut fixture.bank).unwrap();
    let s = &mut bank.samples[0];
    s.len = 97;
    s.loop_start = 17;
    s.loop_end = 78;
    for i in 0..97 {
        bank.pool[s.start as usize + i] = ((i * 997) % 20001) as i16 - 10000;
    }
    let indices = [0, 15, 16, 17, 30, 77, 78, 96];
    let quadrature_ref = [
        0.238169342,
        -0.104636818,
        0.00716878148,
        -0.0217163824,
        -0.126701802,
        0.00781021919,
        0.0842360705,
        0.0843828171,
    ];
    for (continuous, cosine, sine, scale, samples) in [
        (
            false,
            0.927942276,
            0.372723877,
            1.00636792,
            [
                -0.305175781,
                0.1512146,
                0.181640625,
                0.212063923,
                0.00109480089,
                0.192256421,
                0.195742071,
                0.131019831,
            ],
        ),
        (
            true,
            0.925170124,
            -0.379552722,
            1.00040746,
            [
                -0.305175781,
                0.1512146,
                0.181640625,
                0.212055475,
                -0.00666471478,
                0.195841521,
                0.249402478,
                0.19326584,
            ],
        ),
    ] {
        cfg.phase.continuous = continuous;
        let p = PhaseBank::prepare(&fixture.bank, &cfg.phase).unwrap();
        let c = p.coefficients(0, p.angle(1234567890123, 3, 60));
        assert!((c.cosine as f64 - cosine).abs() < 1e-8);
        assert!((c.sine as f64 - sine).abs() < 1e-8);
        assert!((c.scale as f64 - scale).abs() < 2e-6);
        for (j, &i) in indices.iter().enumerate() {
            let q = f32::from_bits(p.words[p.words[0] as usize + i]) as f64;
            assert!((q - quadrature_ref[j] * 32768.0 / 32767.0).abs() < 2e-6);
            let original = decode(fixture.bank.pool[fixture.bank.samples[0].start as usize + i]);
            let y = p.apply(original, 0, i as u32, c) as f64;
            assert!((y - samples[j] * 32768.0 / 32767.0).abs() < 2e-6);
        }
    }
}

struct CapturingCpu {
    cpu: CpuSynth,
    cmds: Vec<crate::voice::SpawnCmd>,
}
impl Backend for CapturingCpu {
    fn set_gates(&mut self, meta: &[u32], runs: &[u32]) -> Result<()> {
        self.cpu.set_gates(meta, runs)
    }
    fn set_channels(
        &mut self,
        rows: &[u32],
        bend: bool,
        gain: bool,
        variant: bool,
        cut: bool,
    ) -> Result<()> {
        self.cpu.set_channels(rows, bend, gain, variant, cut)
    }
    fn set_params_variant(
        &mut self,
        index: u32,
        data: &[crate::bank::RegionParams],
        menv: &[crate::bank::ModEnvParams],
    ) -> Result<()> {
        self.cpu.set_params_variant(index, data, menv)
    }
    fn spawn(&mut self, cmds: &[crate::voice::SpawnCmd]) -> Result<()> {
        self.cmds.extend_from_slice(cmds);
        self.cpu.spawn(cmds)
    }
    fn submit(&mut self) -> Result<()> {
        self.cpu.submit()
    }
    fn finish(&mut self, out: &mut [f32]) -> Result<()> {
        self.cpu.finish(out)
    }
    fn stats(&self) -> crate::backend::BlockStats {
        self.cpu.stats()
    }
    fn name(&self) -> &'static str {
        "capture"
    }
}

#[test]
fn source_ticks_survive_frame_quantization_thinning_layers_and_deferred_spawns() {
    for delayed in [false, true] {
        for dense in [false, true] {
            let mut cfg = config();
            cfg.phase.continuous = true;
            let fixture = Fixture::new(&cfg, delayed);
            let midi = if dense {
                fixture.midi(true)
            } else {
                let mut writer = MidiWriter::new(32767);
                writer.tempo_track(500000);
                writer.track(vec![
                    (42, [0x90, 60, 100], 3),
                    (42, [0x90, 60, 100], 3),
                    (43, [0x90, 60, 100], 3),
                    (18000, [0x80, 60, 0], 3),
                    (18000, [0x80, 60, 0], 3),
                    (18000, [0x80, 60, 0], 3),
                ]);
                let path = fixture.dir.join("ticks.mid");
                writer.save(&path).unwrap();
                path
            };
            let p = PhaseBank::prepare(&fixture.bank, &cfg.phase).unwrap();
            let mut driver =
                Driver::open_prepared(&cfg, fixture.bank.clone(), midi, p.clone()).unwrap();
            let mut capture = CapturingCpu {
                cpu: CpuSynth::new_prepared(&cfg, fixture.bank.clone(), p.clone()),
                cmds: vec![],
            };
            let mut block = vec![0.0; cfg.block_samples()];
            for _ in 0..100 {
                if !driver.next_block(&mut capture, &mut block).unwrap() {
                    break;
                }
            }
            assert!(!capture.cmds.is_empty());
            if dense {
                assert!(driver.stats.dropped > 0);
            } else {
                assert_eq!(capture.cmds.len(), 6);
            }
            for cmd in &capture.cmds {
                let key = (cmd.gate_slot & 127) as u8;
                let tick = if dense {
                    let i = (cmd.ordinal as u64 - 1) * 7 + (key - 60) as u64;
                    i / 2 + (i / 200) * 5000
                } else if cmd.ordinal < 3 {
                    42
                } else {
                    43
                };
                assert_eq!(
                    cmd.rotation,
                    p.coefficients(cmd.region, p.angle(tick, 0, key)),
                    "delayed={delayed}, dense={dense}, ordinal={}",
                    cmd.ordinal
                );
            }
        }
    }
}

#[test]
fn zero_strength_is_bit_identical_to_baseline_through_admission_and_delays() {
    for delayed in [false, true] {
        let mut cfg = config();
        cfg.phase.mode = PhaseMode::Baseline;
        let fixture = Fixture::new(&cfg, delayed);
        let midi = fixture.midi(true);
        let baseline = render(&cfg, &fixture, &midi, false);
        cfg.phase.mode = PhaseMode::Analytic;
        cfg.phase.strength = 0.0;
        assert_eq!(baseline, render(&cfg, &fixture, &midi, false));
        cfg.phase.strength = 1.0;
        let analytic = render(&cfg, &fixture, &midi, false);
        assert_eq!(baseline.1, analytic.1);
        assert!(
            baseline.0.iter().any(|x| x.abs() > 1e-4),
            "silent fixture, delayed={delayed}"
        );
        assert!(
            baseline.0 != analytic.0,
            "analytic had no effect, delayed={delayed}"
        );
        assert_eq!(analytic, render(&cfg, &fixture, &midi, false));
    }
}

#[test]
#[ignore = "requires a hardware GPU; run explicitly with --ignored --nocapture"]
fn gpu_matches_cpu_and_repeats_across_interpolation_and_voice_layouts() {
    for preserve_attack_ms in [0.0, 2.0] {
        for (interp, lfo, menv) in [
            (Interpolation::Nearest, false, false),
            (Interpolation::Linear, true, false),
            (Interpolation::Cubic, true, true),
        ] {
            for continuous in [false, true] {
                let mut cfg = config();
                cfg.interpolation = interp;
                cfg.lfo_enabled = lfo;
                cfg.mod_env_enabled = menv;
                cfg.phase.continuous = continuous;
                cfg.phase.preserve_attack_ms = preserve_attack_ms;
                let fixture = Fixture::new(&cfg, true).distinct_stereo(&cfg);
                let midi = fixture.midi(true);
                let cpu = render(&cfg, &fixture, &midi, false);
                let gpu = render(&cfg, &fixture, &midi, true);
                assert_eq!(cpu.1, gpu.1);
                assert_eq!(cpu.0.len(), gpu.0.len());
                let peak = cpu.0.iter().fold(0.0f32, |p, x| p.max(x.abs())).max(1e-9);
                let error = cpu
                    .0
                    .iter()
                    .zip(&gpu.0)
                    .fold(0.0f32, |p, (a, b)| p.max((a - b).abs()));
                eprintln!("{interp:?}, attack={preserve_attack_ms}, continuous={continuous}, peak={peak}, error={error} ({:.1} dB relative)",
                20.0*(error/peak).max(1e-20).log10());
                assert!(error / peak < 2e-5);
                assert_eq!(gpu, render(&cfg, &fixture, &midi, true));
            }
        }
    }
}

#[test]
fn sessions_prepare_preloaded_banks_and_cancel_before_creating_output() {
    use crate::session::{self, Job, Observer};
    struct Quiet;
    impl Observer for Quiet {}
    struct Cancelled;
    impl Observer for Cancelled {
        fn cancelled(&self) -> bool {
            true
        }
    }

    let mut cfg = config();
    cfg.phase.mode = PhaseMode::Baseline;
    let fixture = Fixture::new(&cfg, false);
    let mut job = Job {
        midi: fixture.midi(false),
        soundfonts: vec![], // Only the preloaded bank can satisfy this job.
        sf_programs: None,
        out: fixture.dir.join("session-baseline.wav"),
        ffmpeg: None,
        wav_format: crate::wav::SampleFormat::Float32,
        ceiling_db: None,
        seconds: None,
        block_csv: None,
        backend: crate::config::BackendKind::Cpu,
        cfg,
    };
    let baseline = session::run(
        &job,
        session::plan(&job).unwrap(),
        Some(fixture.bank.clone()),
        &mut Quiet,
    )
    .unwrap();
    let original = crate::wav::read(&job.out).unwrap();
    job.cfg.phase.mode = PhaseMode::Analytic;
    job.out = fixture.dir.join("session-analytic.wav");
    let analytic = session::run(
        &job,
        session::plan(&job).unwrap(),
        Some(fixture.bank.clone()),
        &mut Quiet,
    )
    .unwrap();
    let rotated = crate::wav::read(&job.out).unwrap();
    assert_eq!(baseline.notes, analytic.notes);
    assert_eq!(baseline.voices_spawned, analytic.voices_spawned);
    assert_eq!(original.interleaved.len(), rotated.interleaved.len());
    assert_ne!(original.interleaved, rotated.interleaved);

    job.out = fixture.dir.join("cancelled.wav");
    let stopped = session::run(
        &job,
        session::plan(&job).unwrap(),
        Some(fixture.bank.clone()),
        &mut Cancelled,
    )
    .unwrap();
    assert!(stopped.cancelled);
    assert_eq!(stopped.bytes, 0);
    assert!(!job.out.exists());
}
