//! Realistic-content benchmark harness.
//!
//! Downloads a public-domain source video on first run (Big Buck Bunny from
//! Google's `gtv-videos-bucket` — CC-BY, ~158 MB, but `ffmpeg` uses HTTP
//! range requests to fetch only the needed bytes), transcodes several
//! fixtures with production-grade x265 settings (preset medium/slow, full
//! filters on, CRF rate control), caches them in `target/bench-fixtures/`,
//! then times `rust_h265` and `ffmpeg` on each.
//!
//! Usage:
//!   cargo run --release --example bench_realworld -- [--only NAME] \
//!     [--source URL_OR_PATH] [--only-generate] [--no-generate]
//!
//! `--only NAME`        run just the one fixture (by name, see FIXTURES).
//! `--source`           override the source URL (e.g. point at a local .mp4).
//! `--only-generate`    transcode fixtures, don't run the benchmark.
//! `--no-generate`      don't fetch/transcode; fail if a fixture is missing.
//!
//! This uses `std::process::Command` to shell out to ffmpeg/ffprobe/x265 —
//! the orchestration, caching, timing, and result-table logic is in Rust.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Default source clip: Big Buck Bunny 1080p H.264, CC-BY Blender Foundation.
/// Hosted on Blender's own mirror (supports HTTP range requests, so `ffmpeg
/// -ss` only fetches the bytes it needs).
const DEFAULT_SOURCE: &str =
    "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov";

struct Fixture {
    /// Short name used for the cached filename and the result-table label.
    name: &'static str,
    /// Start time within the source clip (seconds).
    start_sec: u32,
    /// Duration of the extracted segment (seconds).
    duration_sec: u32,
    /// Output resolution. Source is scaled to this size via the ffmpeg
    /// `scale` video filter.
    width: u32,
    height: u32,
    /// x265 preset — governs encoder speed/quality trade-off.
    x265_preset: &'static str,
    /// Rate-control mode: "crf" (quality target) or "qp" (flat QP).
    rate_mode: &'static str,
    rate_value: u32,
    /// Additional x265-params string (colon-separated `key=value`).
    /// Useful for `no-sao=1:no-deblock=1:...` etc.
    x265_params: &'static str,
    /// Pixel format for ffmpeg output. "yuv420p" for 8-bit, "yuv420p10le" for 10-bit.
    pix_fmt: &'static str,
}

const FIXTURES: &[Fixture] = &[
    // "Safe" presets that exercise our decoder within its supported feature
    // set: SAO on, deblock on, AQ on, B-frames up to 1, flat QP.
    Fixture {
        name: "bbb_1080p_5s_safe",
        start_sec: 60,
        duration_sec: 5,
        width: 1920,
        height: 1080,
        x265_preset: "ultrafast",
        rate_mode: "qp",
        rate_value: 26,
        x265_params: "bframes=1:ref=4:no-wpp=1:no-cutree=1",
        pix_fmt: "yuv420p",
    },
    Fixture {
        name: "bbb_720p_10s_safe",
        start_sec: 60,
        duration_sec: 10,
        width: 1280,
        height: 720,
        x265_preset: "ultrafast",
        rate_mode: "qp",
        rate_value: 26,
        x265_params: "bframes=1:ref=4:no-wpp=1:no-cutree=1",
        pix_fmt: "yuv420p",
    },
    // "Full" presets that model real streaming workloads.
    Fixture {
        name: "bbb_1080p_5s_medium",
        start_sec: 60,
        duration_sec: 5,
        width: 1920,
        height: 1080,
        x265_preset: "medium",
        rate_mode: "crf",
        rate_value: 23,
        x265_params: "",
        pix_fmt: "yuv420p",
    },
    Fixture {
        name: "bbb_1080p_5s_slow",
        start_sec: 60,
        duration_sec: 5,
        width: 1920,
        height: 1080,
        x265_preset: "slow",
        rate_mode: "crf",
        rate_value: 23,
        x265_params: "",
        pix_fmt: "yuv420p",
    },
    // 10-bit (Main 10 profile) — models HDR content decode path.
    Fixture {
        name: "bbb_1080p_5s_10bit_safe",
        start_sec: 60,
        duration_sec: 5,
        width: 1920,
        height: 1080,
        x265_preset: "ultrafast",
        rate_mode: "qp",
        rate_value: 26,
        x265_params: "bframes=1:ref=4:no-wpp=1:no-cutree=1",
        pix_fmt: "yuv420p10le",
    },
    Fixture {
        name: "bbb_1080p_5s_10bit_medium",
        start_sec: 60,
        duration_sec: 5,
        width: 1920,
        height: 1080,
        x265_preset: "medium",
        rate_mode: "crf",
        rate_value: 23,
        x265_params: "",
        pix_fmt: "yuv420p10le",
    },
];

// ------------------------- orchestration --------------------------

fn cache_dir() -> PathBuf {
    let d = PathBuf::from("target/bench-fixtures");
    std::fs::create_dir_all(&d).expect("create cache dir");
    d
}

fn ensure_fixture(f: &Fixture, source: &str) -> std::io::Result<PathBuf> {
    let path = cache_dir().join(format!("{}.h265", f.name));
    if path.exists() {
        return Ok(path);
    }
    eprintln!(
        "  transcoding {} (source={}, start={}s, dur={}s, {}x{}, preset={})",
        f.name, source, f.start_sec, f.duration_sec, f.width, f.height, f.x265_preset
    );
    let tmp = path.with_extension("h265.partial");
    // `-ss` BEFORE `-i` does fast-seek on the source (uses HTTP range
    // requests when source is a URL), which avoids downloading the whole
    // file just to take a 5-second window from the middle.
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "warning",
        "-ss",
        &f.start_sec.to_string(),
        "-i",
        source,
        "-t",
        &f.duration_sec.to_string(),
        "-an", // drop audio
        "-vf",
        &format!("scale={}:{}", f.width, f.height),
        "-pix_fmt",
        f.pix_fmt,
        "-c:v",
        "libx265",
        "-preset",
        f.x265_preset,
    ]);
    match f.rate_mode {
        "crf" => cmd.args(["-crf", &f.rate_value.to_string()]),
        "qp" => cmd.args(["-qp", &f.rate_value.to_string()]),
        m => panic!("unknown rate_mode: {m}"),
    };
    if !f.x265_params.is_empty() {
        cmd.args(["-x265-params", f.x265_params]);
    }
    cmd.args(["-f", "hevc", "-y", tmp.to_str().unwrap()]);
    let status = cmd.status()?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(std::io::Error::other(format!(
            "transcode failed for {} (exit {:?})",
            f.name,
            status.code()
        )));
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

fn probe_frame_count(path: &Path) -> u32 {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            path.to_str().unwrap(),
        ])
        .output()
        .expect("ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

// ------------------------- timing --------------------------

struct OursTiming {
    frames: u32,
    best: Duration,
    /// Set when the decoder reported an error partway through. The reported
    /// frame count / timing then covers only what was successfully decoded
    /// before the error.
    error: Option<String>,
}

fn run_ours(path: &Path, warmup: u32, repeat: u32) -> OursTiming {
    let data = std::fs::read(path).expect("read fixture");
    let nals = rust_h265::nal::parse_annex_b(&data);
    let run = || -> (u32, Duration, Option<String>) {
        let mut dec = rust_h265::decoder::Decoder::new();
        let t = Instant::now();
        let mut n = 0u32;
        let mut err = None;
        for nal in &nals {
            match dec.decode_nal(nal) {
                Ok(Some(_)) => n += 1,
                Ok(None) => {}
                Err(e) => {
                    err = Some(format!("{e:?}"));
                    break;
                }
            }
        }
        if err.is_none() {
            while dec.flush().is_some() {
                n += 1;
            }
        }
        (n, t.elapsed(), err)
    };
    for _ in 0..warmup {
        let _ = run();
    }
    let mut best = Duration::from_secs(u64::MAX / 2);
    let mut frames = 0u32;
    let mut error = None;
    for _ in 0..repeat {
        let (n, d, e) = run();
        frames = n;
        error = e;
        if d < best {
            best = d;
        }
    }
    OursTiming {
        frames,
        best,
        error,
    }
}

/// Read `rtime=<secs>` out of ffmpeg's `-benchmark` stderr line.
fn parse_rtime(stderr: &str) -> Option<Duration> {
    for line in stderr.lines() {
        if let Some(after) = line.strip_prefix("bench:") {
            if let Some(rt) = after.split("rtime=").nth(1) {
                let t: f64 = rt.trim_end_matches('s').trim().parse().ok()?;
                return Some(Duration::from_secs_f64(t));
            }
        }
    }
    None
}

fn run_ffmpeg(path: &Path, threads: u32, warmup: u32, repeat: u32) -> Duration {
    let run = || -> Duration {
        let out = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-nostdin",
                "-threads",
                &threads.to_string(),
                "-i",
                path.to_str().unwrap(),
                "-f",
                "null",
                "-",
                "-benchmark",
            ])
            .output()
            .expect("ffmpeg");
        let stderr = String::from_utf8_lossy(&out.stderr);
        parse_rtime(&stderr).expect("ffmpeg -benchmark rtime")
    };
    for _ in 0..warmup {
        let _ = run();
    }
    let mut best = Duration::from_secs(u64::MAX / 2);
    for _ in 0..repeat {
        let d = run();
        if d < best {
            best = d;
        }
    }
    best
}

fn mpixps(frames: u32, w: u32, h: u32, dur: Duration) -> f64 {
    let secs = dur.as_secs_f64();
    if secs == 0.0 {
        return 0.0;
    }
    (frames as f64 * w as f64 * h as f64) / secs / 1e6
}

// ------------------------- main --------------------------

fn main() -> std::io::Result<()> {
    let mut source = DEFAULT_SOURCE.to_string();
    let mut only: Option<String> = None;
    let mut only_generate = false;
    let mut no_generate = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                source = args[i + 1].clone();
                i += 2;
            }
            "--only" => {
                only = Some(args[i + 1].clone());
                i += 2;
            }
            "--only-generate" => {
                only_generate = true;
                i += 1;
            }
            "--no-generate" => {
                no_generate = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!(
                    "usage: bench_realworld [--only NAME] [--source URL_OR_PATH] [--only-generate] [--no-generate]"
                );
                println!("\nFixtures:");
                for f in FIXTURES {
                    println!(
                        "  {:<28}  {}x{}  {}s @ preset={}",
                        f.name, f.width, f.height, f.duration_sec, f.x265_preset
                    );
                }
                return Ok(());
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    // Phase 1: ensure all fixtures exist.
    eprintln!("bench_realworld: checking/generating fixtures");
    let mut paths: Vec<(&Fixture, PathBuf)> = Vec::new();
    for f in FIXTURES {
        if let Some(n) = &only {
            if n != f.name {
                continue;
            }
        }
        let p = cache_dir().join(format!("{}.h265", f.name));
        if !p.exists() {
            if no_generate {
                eprintln!("  SKIP {} (missing and --no-generate set)", f.name);
                continue;
            }
            let p = ensure_fixture(f, &source)?;
            paths.push((f, p));
        } else {
            paths.push((f, p));
        }
    }

    if only_generate {
        eprintln!("--only-generate: done");
        return Ok(());
    }

    // Phase 2: benchmark.
    println!();
    println!(
        "| {:<28} | {:>9} | {:>6} | {:>16} | {:>16} | {:>16} |",
        "fixture", "res", "frames", "ours", "ff-t1", "ff-tN"
    );
    println!(
        "|{:-<30}|{:-<11}|{:-<8}|{:-<18}|{:-<18}|{:-<18}|",
        "", "", "", "", "", ""
    );

    // Raw table for later analysis / pasting into BENCHMARK.md.
    let raw_path = Path::new("target/bench-fixtures/results.tsv");
    let mut raw = String::from(
        "fixture\tw\th\tframes\tours_ms\tours_mpxs\tff_t1_ms\tff_t1_mpxs\tff_tN_ms\tff_tN_mpxs\n",
    );

    for (f, path) in &paths {
        let frames = probe_frame_count(path);
        // Aim for ≥ 0.3s of decode work for stable measurement.
        let (our_warmup, our_repeat, ff_repeat) = if f.width >= 1920 {
            (1, 6, 8)
        } else if f.width >= 1280 {
            (1, 12, 12)
        } else {
            (2, 30, 20)
        };
        let ours = run_ours(path, our_warmup, our_repeat);
        let ff1 = run_ffmpeg(path, 1, 1, ff_repeat);
        let ffn = run_ffmpeg(path, 0, 1, ff_repeat);
        let mpx_1 = mpixps(frames, f.width, f.height, ff1);
        let mpx_n = mpixps(frames, f.width, f.height, ffn);
        let ff1_ms = ff1.as_secs_f64() * 1000.0;
        let ffn_ms = ffn.as_secs_f64() * 1000.0;
        let (ours_col, ours_ms_str, ours_mpxs_str, ours_ok) = match &ours.error {
            None => {
                let mpx = mpixps(ours.frames, f.width, f.height, ours.best);
                let ms = ours.best.as_secs_f64() * 1000.0;
                (
                    format!("{ms:>6.1} ms {mpx:>6.0}"),
                    format!("{ms:.1}"),
                    format!("{mpx:.1}"),
                    true,
                )
            }
            Some(_) => ("ERR (see below)".into(), "N/A".into(), "N/A".into(), false),
        };
        println!(
            "| {:<28} | {:>4}x{:<4} | {:>6} | {:^16} | {:>6.1} ms {:>6.0} | {:>6.1} ms {:>6.0} |",
            f.name, f.width, f.height, frames, ours_col, ff1_ms, mpx_1, ffn_ms, mpx_n
        );
        raw.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\n",
            f.name,
            f.width,
            f.height,
            frames,
            ours_ms_str,
            ours_mpxs_str,
            ff1_ms,
            mpx_1,
            ffn_ms,
            mpx_n
        ));
        if !ours_ok {
            if let Some(e) = &ours.error {
                eprintln!("    ours failed: {e}");
            }
        }
    }
    std::fs::write(raw_path, raw)?;
    eprintln!("\nraw TSV: {}", raw_path.display());
    Ok(())
}
