//! Fuzz the slice data of valid streams.
//!
//! Takes one of the in-tree fixtures and corrupts only the bytes after its
//! slice headers, keeping the parameter sets, slice headers and NAL framing
//! intact. Arbitrary-byte targets (`decode_annex_b`) spend most inputs on
//! parameter sets and headers that fail to parse; here every input reaches
//! CABAC, the CU / PU / TU syntax, residual coding, prediction and the loop
//! filters, including in inter pictures whose references decoded cleanly
//! (where overflows such as an unbounded mvd Exp-Golomb prefix hide).
//!
//! Input layout:
//! - byte 0: fixture index (modulo the number of fixtures);
//! - then 4-byte edits `[p0, p1, p2, x]`: position `p` (24-bit LE, modulo
//!   the total slice-data length of the fixture) and an XOR mask `x`; a
//!   mask of 0 truncates that slice NAL at the position instead.
//!
//! Decoding carries on through errors, so later pictures still decode on
//! top of damaged references. The fuzzer looks for panics (overflow checks
//! are on in cargo-fuzz builds) and memory errors.
//!
//! Run: `cargo +nightly fuzz run slice_data`

#![no_main]

use std::borrow::Cow;

use libfuzzer_sys::fuzz_target;
use rust_h265::{Decoder, NalUnit};

macro_rules! fixtures {
    ($($name:literal),* $(,)?) => {
        [$(include_bytes!(concat!("../../testdata/", $name)).as_slice()),*]
    };
}

/// Small fixtures covering the coding tools: inter P / B, weighted
/// prediction, 10 / 12-bit, 4:4:4, PCM, transform skip, lossless, sign
/// hiding, scaling lists, AMP / 4x4 inter, delta QP groups, tiles, WPP,
/// dependent slices, multiple slices, CRA / RASL and the RExt tools.
static FIXTURES: [&[u8]; 28] = fixtures![
    "inter_b.h265",
    "tu_inter4x4_motion.h265",
    "inter420_100x68.h265",
    "inter444_b_96x96.h265",
    "inter444_10bit_96x96.h265",
    "10bit_128x128.h265",
    "bframes3_128x128.h265",
    "motion_320x240.h265",
    "aq_p_320x240.h265",
    "pcm_hm_64x64.h265",
    "pcm444_10bit_hm_64x64.h265",
    "transquant_bypass_64x64.h265",
    "tskip_128x128.h265",
    "signhide_scaling_320x240.h265",
    "constrained_intra_128x128.h265",
    "delta_qp_groups_hm_416x240.h265",
    "pmerge_level3_hm_416x240.h265",
    "tiles_hm_lf_across_832x256.h265",
    "ctu64_wpp.h265",
    "dep_slices.h265",
    "multi_slice_sao_deblock_256x256.h265",
    "seek_to_cra.h265",
    "rext_ccp_hm_444_64x64.h265",
    "rext_implicit_rdpcm_hm_444_64x64.h265",
    "rext_hp_wp_hm_444_96x64.h265",
    "rext_ts32_12bit_hm_444_12bit_96x64.h265",
    "rext_chroma_qp_adj_dep_hm_444_96x64.h265",
    "rext_lossless_rot_rdpcm_hm_444_96x64.h265",
];

fuzz_target!(|data: &[u8]| {
    let Some((&pick, edits)) = data.split_first() else {
        return;
    };
    let fixture = FIXTURES[pick as usize % FIXTURES.len()];
    let mut nals = rust_h265::parse_annex_b(fixture);
    let offsets = rust_h265::fuzzing::slice_data_offsets(&nals);

    // (NAL index, slice data start) for every slice, and their total length.
    let slices: Vec<(usize, usize)> = offsets
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.filter(|&o| o < nals[i].rbsp.len()).map(|o| (i, o)))
        .collect();
    let total: usize = slices.iter().map(|&(i, o)| nals[i].rbsp.len() - o).sum();
    if total == 0 {
        return;
    }

    let mut truncate: Vec<Option<usize>> = vec![None; nals.len()];
    for edit in edits.chunks_exact(4) {
        let mut pos =
            (edit[0] as usize | (edit[1] as usize) << 8 | (edit[2] as usize) << 16) % total;
        let &(i, start) = slices
            .iter()
            .find(|&&(i, o)| {
                let len = nals[i].rbsp.len() - o;
                if pos < len {
                    true
                } else {
                    pos -= len;
                    false
                }
            })
            .expect("position within the slice data");
        let at = start + pos;
        if edit[3] == 0 {
            truncate[i] = Some(truncate[i].map_or(at, |t| t.min(at)));
        } else {
            nals[i].rbsp.to_mut()[at] ^= edit[3];
        }
    }
    for (nal, t) in nals.iter_mut().zip(truncate) {
        if let Some(t) = t {
            let mut bytes = nal.rbsp.to_vec();
            bytes.truncate(t);
            nal.rbsp = Cow::Owned(bytes);
        }
    }

    decode(&nals);
});

fn decode(nals: &[NalUnit<'_>]) {
    let mut decoder = Decoder::new();
    for nal in nals {
        // Carry on through errors: later pictures predict from the damaged
        // ones, which is where many overflows show up.
        let _ = decoder.decode_nal(nal);
    }
    while decoder.flush().is_some() {}
    while decoder.next_error().is_some() {}
}
