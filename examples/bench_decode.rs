//! Simple decode benchmark for HEVC bitstreams.
//!
//! Usage:
//!   cargo run --release --example bench_decode -- <file.h265> [--repeat N] [--warmup K]
//!
//! With `--repeat N`, the full file is decoded N times and only the best
//! (fastest) run is reported — the standard trick for microbenchmarking
//! on an unloaded system. `--warmup K` runs K untimed decodes first.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path: Option<String> = None;
    let mut repeat: u32 = 1;
    let mut warmup: u32 = 0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repeat" => {
                repeat = args[i + 1].parse().expect("--repeat NUMBER");
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse().expect("--warmup NUMBER");
                i += 2;
            }
            s => {
                path = Some(s.to_string());
                i += 1;
            }
        }
    }
    let path = path.expect("usage: bench_decode <file.h265> [--repeat N] [--warmup K]");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let nals = rust_h265::nal::parse_annex_b(&data);

    let run_once = || -> (u64, std::time::Duration) {
        let mut decoder = rust_h265::decoder::Decoder::new();
        let start = std::time::Instant::now();
        let mut frames = 0u64;
        for nal in &nals {
            match decoder.decode_nal(nal) {
                Ok(Some(_frame)) => frames += 1,
                Ok(None) => {}
                Err(e) => {
                    eprintln!("decode error after {frames} frames: {e:?}");
                    break;
                }
            }
        }
        while let Some(_frame) = decoder.flush() {
            frames += 1;
        }
        (frames, start.elapsed())
    };

    for _ in 0..warmup {
        let _ = run_once();
    }

    let mut best = std::time::Duration::from_secs(u64::MAX / 2);
    let mut frames = 0u64;
    let mut total = std::time::Duration::ZERO;
    for _ in 0..repeat {
        let (f, d) = run_once();
        frames = f;
        total += d;
        if d < best {
            best = d;
        }
    }
    let avg = total / repeat;
    if repeat > 1 {
        println!(
            "{} frames × {} runs: best {:.3}s ({:.1} fps), avg {:.3}s ({:.1} fps)",
            frames,
            repeat,
            best.as_secs_f64(),
            frames as f64 / best.as_secs_f64(),
            avg.as_secs_f64(),
            frames as f64 / avg.as_secs_f64()
        );
    } else {
        println!(
            "{} frames in {:.3}s = {:.1} fps",
            frames,
            best.as_secs_f64(),
            frames as f64 / best.as_secs_f64()
        );
    }
}
