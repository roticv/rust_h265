//! Dump decoded frames to a raw YUV file in display order.
//!
//! By default the decoder's display-order mode (`Decoder::with_display_order`,
//! the spec C.5.2 output process) decides which frames are output and in
//! what order. With `--decode-order` the decoder returns every picture in
//! decoding order and this tool sorts them by POC within each coded video
//! sequence (CVS) instead; that keeps pictures the output process would
//! discard.
//!
//! Usage:
//!   cargo run --release --example dump_frames -- <in.h265> <out.yuv> [--decode-order] [--threads N]
//!
//! `--threads N` decodes with `Decoder::with_threads(N)`; the output must
//! not depend on it.
//!
//! Intended for byte-exact verification against FFmpeg's HEVC decoder:
//!   ffmpeg -i <in.h265> -f rawvideo -pix_fmt yuv420p ref.yuv      # 8-bit
//!   ffmpeg -i <in.h265> -f rawvideo -pix_fmt yuv420p10le ref.yuv  # 10-bit
//!   cmp ref.yuv out.yuv

use rust_h265::{Frame, NalUnitType, PixelData};
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
    let usage = || -> ! {
        eprintln!("usage: dump_frames <in.h265> <out.yuv> [--decode-order] [--threads N]");
        std::process::exit(2);
    };
    if args.len() < 3 {
        usage();
    }
    let (mut decode_order, mut threads) = (false, 1usize);
    let mut flags = args[3..].iter();
    while let Some(flag) = flags.next() {
        match flag.as_str() {
            "--decode-order" => decode_order = true,
            "--threads" => {
                threads = flags
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            _ => usage(),
        }
    }
    let data = std::fs::read(&args[1]).unwrap_or_else(|e| panic!("read {}: {e}", args[1]));
    let out = std::fs::File::create(&args[2]).unwrap_or_else(|e| panic!("create {}: {e}", args[2]));
    let mut out = std::io::BufWriter::new(out);

    if !decode_order {
        let frames = decode_display_order(&data, threads);
        write_frames(&mut out, &frames, &args[2]);
        return;
    }

    let mut dec = rust_h265::decoder::Decoder::new().with_threads(threads);
    // (coded video sequence index, frame). The decoder returns a frame while
    // decoding a slice of that same picture, so the sequence a frame belongs
    // to is the one current when it is returned.
    let mut frames: Vec<(u32, rust_h265::decoder::Frame)> = Vec::new();
    let mut cvs = 0u32;
    let mut seen_picture = false;
    let mut after_eos = false;
    for nal in rust_h265::nal::parse_annex_b(&data) {
        let t = nal.nal_unit_type;
        if t == NalUnitType::Eos {
            after_eos = true;
        }
        // first_slice_segment_in_pic_flag is the first bit of the slice header.
        let first_slice = t.is_vcl() && nal.rbsp.first().is_some_and(|b| b & 0x80 != 0);
        if first_slice {
            // A CVS starts at an IDR or BLA picture, or at a CRA that is the
            // first picture or follows an end of sequence (spec 3.29, 8.1.3).
            let starts_cvs = t.is_irap() && (!t.is_cra() || !seen_picture || after_eos);
            if starts_cvs && seen_picture {
                cvs += 1;
            }
            seen_picture = true;
            after_eos = false;
        }
        match dec.decode_nal(&nal) {
            Ok(Some(f)) => frames.push((cvs, f)),
            Ok(None) => {}
            Err(e) => {
                eprintln!("decode error after {} frames: {e:?}", frames.len());
                std::process::exit(1);
            }
        }
    }
    while let Some(f) = dec.flush() {
        frames.push((cvs, f));
    }
    exit_on_late_error(&mut dec, frames.len());

    // Display order: by sequence, then by POC within the sequence.
    frames.sort_by_key(|(cvs, f)| (*cvs, f.pic_order_cnt));
    let frames: Vec<_> = frames.into_iter().map(|(_, f)| f).collect();
    write_frames(&mut out, &frames, &args[2]);
}

/// Decode with the decoder's own output process.
fn decode_display_order(data: &[u8], threads: usize) -> Vec<Frame> {
    let mut dec = rust_h265::decoder::Decoder::new()
        .with_display_order()
        .with_threads(threads);
    let mut frames = Vec::new();
    for nal in rust_h265::nal::parse_annex_b(data) {
        if let Err(e) = dec.decode_nal(&nal) {
            eprintln!("decode error after {} frames: {e:?}", frames.len());
            std::process::exit(1);
        }
        while let Some(f) = dec.next_output() {
            frames.push(f);
        }
    }
    while let Some(f) = dec.flush() {
        frames.push(f);
    }
    exit_on_late_error(&mut dec, frames.len());
    frames
}

/// With `--threads` above 1 an error in a picture's slice data can surface
/// only after the last `decode_nal` (it is found on the picture's thread);
/// treat it like one `decode_nal` returned.
fn exit_on_late_error(dec: &mut rust_h265::decoder::Decoder, frames: usize) {
    if let Some(e) = dec.next_error() {
        eprintln!("decode error after {frames} frames: {e:?}");
        std::process::exit(1);
    }
}

fn write_frames(out: &mut impl Write, frames: &[Frame], path: &str) {
    for f in frames {
        write_plane(out, &f.y);
        write_plane(out, &f.u);
        write_plane(out, &f.v);
    }
    eprintln!(
        "wrote {} frames ({}x{}) to {}",
        frames.len(),
        frames.first().map(|f| f.width).unwrap_or(0),
        frames.first().map(|f| f.height).unwrap_or(0),
        path
    );
}
