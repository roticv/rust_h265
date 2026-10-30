//! Fuzz multithreaded decoding against single-threaded decoding.
//!
//! Decodes the input as an Annex B bitstream with one thread and with three
//! (frame threading, row filters, reference waits, wavefront rows, the
//! shared picture tables), carrying on through errors, and requires the
//! same frames (every sample) and the same errors. Finds crashes in the
//! threaded paths and any divergence between them and sequential decoding,
//! which on a corrupt stream is the likeliest sign of a protocol bug.
//!
//! Run: `cargo +nightly fuzz run decode_threaded`

#![no_main]

use libfuzzer_sys::fuzz_target;
use rust_h265::{Decoder, PixelData};

fn decode(nals: &[rust_h265::NalUnit], threads: usize) -> (Vec<String>, Vec<(i32, Vec<u8>)>) {
    let mut decoder = Decoder::new().with_threads(threads);
    let (mut errors, mut frames) = (Vec::new(), Vec::new());
    let mut take = |f: rust_h265::Frame| {
        let mut bytes = Vec::new();
        for plane in [&f.y, &f.u, &f.v] {
            match plane {
                PixelData::U8(p) => bytes.extend_from_slice(p),
                PixelData::U16(p) => bytes.extend(p.iter().flat_map(|s| s.to_le_bytes())),
            }
        }
        frames.push((f.pic_order_cnt, bytes));
    };
    for nal in nals {
        match decoder.decode_nal(nal) {
            Ok(Some(f)) => take(f),
            Ok(None) => {}
            Err(e) => errors.push(format!("{e:?}")),
        }
    }
    while let Some(f) = decoder.flush() {
        take(f);
    }
    while let Some(e) = decoder.next_error() {
        errors.push(format!("{e:?}"));
    }
    // Frame threading reports slice-data errors on later calls.
    errors.sort();
    (errors, frames)
}

fuzz_target!(|data: &[u8]| {
    let nals = rust_h265::parse_annex_b(data);
    let single = decode(&nals, 1);
    let threaded = decode(&nals, 3);
    assert!(single == threaded, "threaded decoding differs from single-threaded");
});
