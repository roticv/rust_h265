//! Fuzz with a single synthetic NAL unit (no start-code parsing).
//!
//! Wraps the fuzz input in a minimal Annex B frame (start code + 2-byte NAL
//! header) and feeds it to the decoder. This gives the fuzzer more direct
//! control over the NAL payload bytes, which is useful for finding bugs in
//! parameter set parsing and slice header decode that `decode_annex_b` might
//! miss due to the start-code search overhead.
//!
//! Run: `cargo fuzz run decode_single_nal`

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    // First byte selects the NAL unit type (0-63) and temporal_id.
    // Remaining bytes are the NAL payload.
    let nal_type = data[0] & 0x3f;
    let temporal_id = (data[1] & 0x07).max(1); // nuh_temporal_id_plus1 >= 1

    // Build a minimal Annex B NAL: start code + 2-byte header + payload.
    let mut annex_b = Vec::with_capacity(4 + 2 + data.len() - 2);
    annex_b.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    // NAL header: forbidden_zero_bit(0) | nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)
    let byte0 = (nal_type << 1) & 0x7e; // forbidden=0, type in bits 6..1
    let byte1 = temporal_id & 0x07; // layer_id=0, tid in bits 2..0
    annex_b.push(byte0);
    annex_b.push(byte1);
    annex_b.extend_from_slice(&data[2..]);

    let nals = rust_h265::parse_annex_b(&annex_b);
    let mut decoder = rust_h265::Decoder::new();
    for nal in &nals {
        match decoder.decode_nal(nal) {
            Ok(_) => {}
            Err(_) => return,
        }
    }
    while decoder.flush().is_some() {}
});
