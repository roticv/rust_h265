//! Simple decode benchmark for HEVC bitstreams.
//!
//! Usage:
//!   cargo run --release --example bench_decode -- <file.h265>
//!
//! Prints the number of decoded frames, wall-clock time, and throughput in FPS.

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_decode <file.h265>");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let nals = rust_h265::nal::parse_annex_b(&data);

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
    // Flush remaining buffered frames.
    while let Some(_frame) = decoder.flush() {
        frames += 1;
    }

    let elapsed = start.elapsed();
    println!(
        "{} frames in {:.2}s = {:.1} fps",
        frames,
        elapsed.as_secs_f64(),
        frames as f64 / elapsed.as_secs_f64()
    );
}
