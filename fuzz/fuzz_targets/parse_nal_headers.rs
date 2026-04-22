//! Fuzz just the NAL parsing layer (Annex B start-code detection, EPB removal,
//! NAL header parsing). Lightweight — doesn't invoke the decoder, so runs much
//! faster and finds issues in the parser without needing valid parameter sets.
//!
//! Run: `cargo fuzz run parse_nal_headers`

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let nals = rust_h265::parse_annex_b(data);
    for nal in &nals {
        // Just access the parsed fields to ensure no panics.
        let _ = nal.nal_unit_type;
        let _ = nal.nuh_layer_id;
        let _ = nal.temporal_id;
        let _ = nal.rbsp.len();
    }
});
