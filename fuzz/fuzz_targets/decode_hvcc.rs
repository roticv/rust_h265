//! Fuzz the HVCC (length-prefixed) decode pipeline.
//!
//! Feeds arbitrary bytes as HVCC-format NAL units to the decoder, exercising
//! the `parse_hvcc` path used for MP4/MKV container input. Tests all
//! length_size values (1-4 bytes) based on the first byte of fuzz input.
//!
//! Run: `cargo fuzz run decode_hvcc`

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Use first byte to select length_size (1-4).
    let length_size = (data[0] % 4) + 1;
    let payload = &data[1..];

    let nals = rust_h265::parse_hvcc(payload, length_size);
    let mut decoder = rust_h265::Decoder::new();
    for nal in &nals {
        match decoder.decode_nal(nal) {
            Ok(_) => {}
            Err(_) => return,
        }
    }
    while decoder.flush().is_some() {}
});
