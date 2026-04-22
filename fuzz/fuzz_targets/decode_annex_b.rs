//! Fuzz the full Annex B decode pipeline.
//!
//! Feeds arbitrary bytes as an Annex B bitstream to the decoder, exercising
//! NAL parsing, parameter set decode, CABAC, CU/TU tree, MC, intra pred,
//! deblocking, SAO, and DPB management. The fuzzer looks for panics and
//! memory safety violations — decode errors are expected and silently caught.
//!
//! Run: `cargo fuzz run decode_annex_b`
//! Seed corpus: `fuzz/corpus/decode_annex_b/` (copy test fixtures there)

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let nals = rust_h265::parse_annex_b(data);
    let mut decoder = rust_h265::Decoder::new();
    for nal in &nals {
        // Errors are expected on malformed input — just don't panic.
        match decoder.decode_nal(nal) {
            Ok(_) => {}
            Err(_) => return,
        }
    }
    // Flush any remaining frames.
    while decoder.flush().is_some() {}
});
