//! Dump decoded frames to a raw YUV file in display order.
//!
//! Display order is POC order within each coded video sequence (CVS), with
//! sequences in bitstream order. POC restarts at every CVS, so sorting the
//! whole stream by POC would interleave frames of different sequences.
//!
//! Usage:
//!   cargo run --release --example dump_frames -- <in.h265> <out.yuv>
//!
//! Intended for byte-exact verification against FFmpeg's HEVC decoder:
//!   ffmpeg -i <in.h265> -f rawvideo -pix_fmt yuv420p ref.yuv      # 8-bit
//!   ffmpeg -i <in.h265> -f rawvideo -pix_fmt yuv420p10le ref.yuv  # 10-bit
//!   cmp ref.yuv out.yuv

use rust_h265::{NalUnitType, PixelData};
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

    // Display order: by sequence, then by POC within the sequence.
    frames.sort_by_key(|(cvs, f)| (*cvs, f.pic_order_cnt));
    let frames: Vec<_> = frames.into_iter().map(|(_, f)| f).collect();

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
