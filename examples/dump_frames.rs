//! Dump decoded frames to a raw YUV420 file in display order.
//!
//! Usage:
//!   cargo run --release --example dump_frames -- <in.h265> <out.yuv>
//!
//! Intended for byte-exact verification against FFmpeg's HEVC decoder:
//!   ffmpeg -i <in.h265> -f rawvideo -pix_fmt yuv420p ref.yuv      # 8-bit
//!   ffmpeg -i <in.h265> -f rawvideo -pix_fmt yuv420p10le ref.yuv  # 10-bit
//!   cmp ref.yuv out.yuv

use rust_h265::PixelData;
use std::io::Write;

/// Write a pixel plane to the output. 8-bit writes raw bytes; 10-bit writes
/// each sample as two little-endian bytes (matching FFmpeg's yuv420p10le).
fn write_plane(out: &mut impl Write, plane: &PixelData) {
    match plane {
        PixelData::U8(data) => out.write_all(data).unwrap(),
        PixelData::U16(data) => {
            for &sample in data {
                out.write_all(&sample.to_le_bytes()).unwrap();
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: dump_frames <in.h265> <out.yuv>");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).unwrap_or_else(|e| panic!("read {}: {e}", args[1]));
    let out = std::fs::File::create(&args[2]).unwrap_or_else(|e| panic!("create {}: {e}", args[2]));
    let mut out = std::io::BufWriter::new(out);

    let mut dec = rust_h265::decoder::Decoder::new();
    let mut frames: Vec<rust_h265::decoder::Frame> = Vec::new();
    for nal in rust_h265::nal::parse_annex_b(&data) {
        match dec.decode_nal(&nal) {
            Ok(Some(f)) => frames.push(f),
            Ok(None) => {}
            Err(e) => {
                eprintln!("decode error after {} frames: {e:?}", frames.len());
                std::process::exit(1);
            }
        }
    }
    while let Some(f) = dec.flush() {
        frames.push(f);
    }

    // Re-sort by POC for display order.
    frames.sort_by_key(|f| f.pic_order_cnt);

    for f in &frames {
        write_plane(&mut out, &f.y);
        write_plane(&mut out, &f.u);
        write_plane(&mut out, &f.v);
    }
    eprintln!(
        "wrote {} frames ({}x{}) to {}",
        frames.len(),
        frames.first().map(|f| f.width).unwrap_or(0),
        frames.first().map(|f| f.height).unwrap_or(0),
        args[2]
    );
}
