//! HEVC slice data parsing — coding tree, coding unit, intra mode signaling.
//!
//! Phase 2c-1 scope: recursive `coding_quadtree` / `coding_unit` decoding for
//! the I-slice intra path, **stopping after intra prediction mode signaling**.
//! Transform tree, residual coding, intra sample generation, and inverse
//! transform are not yet implemented — those are Phase 2c-2 / 2c-3 / 2c-4.
//!
//! The recursion structure mirrors FFmpeg `libavcodec/hevc/hevcdec.c`
//! (`hls_coding_quadtree`, `hls_coding_unit`, `intra_prediction_unit`,
//! `luma_intra_pred_mode`) so that decoded values match byte-for-byte.

use crate::cabac::{CabacContexts, CabacReader};
use crate::cabac_tables::ctx;
use crate::error::DecodeError;
use crate::pps::Pps;
use crate::sps::Sps;

/// HEVC luma intra prediction mode constants (spec table 8-1).
pub const INTRA_PLANAR: u8 = 0;
pub const INTRA_DC: u8 = 1;
pub const INTRA_ANGULAR_10: u8 = 10;
pub const INTRA_ANGULAR_26: u8 = 26;
/// Used by the chroma DM-substitution rule when the mapped chroma mode
/// equals the luma mode (spec table 8-3).
pub const INTRA_ANGULAR_34: u8 = 34;

/// HEVC partition mode (spec table 7-10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartMode {
    Part2Nx2N,
    PartNxN,
    Part2NxN,
    PartNx2N,
    Part2NxnU,
    Part2NxnD,
    PartnLx2N,
    PartnRx2N,
}

/// Per-picture mutable state needed during slice decode.
///
/// `tab_ct_depth` is per-min-CB and used for `split_cu_flag` neighbor
/// context derivation. `tab_ipm` is per-min-PU (4×4 in HEVC base profile)
/// and stores the decoded luma intra prediction mode for downstream use
/// (intra prediction in Phase 2c-5, deblocking in Phase 3+).
pub struct PictureState {
    pub width: u32,
    pub height: u32,
    pub log2_min_cb_size: u8,
    pub log2_min_pu_size: u8,
    pub log2_ctb_size: u8,
    pub min_cb_width: usize,
    pub min_pu_width: usize,
    pub tab_ct_depth: Vec<u8>,
    pub tab_ipm: Vec<u8>,

    /// Phase 2c-1 sentinel: most recently decoded luma intra mode of the
    /// most recently decoded CU. Will go away once full CU decode is wired up.
    pub last_luma_pred_mode: u8,
    /// Phase 2c-1 sentinel: most recently decoded chroma intra mode (mapped
    /// to the luma mode space, not the raw `chroma_mode_idx`).
    pub last_chroma_pred_mode: u8,
    /// Number of CUs visited during decode (sentinel for tests).
    pub cu_count: u32,
}

impl PictureState {
    pub fn new(sps: &Sps) -> Self {
        let log2_min_cb_size = sps.min_cb_log2_size_y;
        // HEVC base profile pins min PU size to 4×4 (spec 7.4.3.2.1).
        let log2_min_pu_size = 2u8;
        let log2_ctb_size = sps.ctb_log2_size_y;
        let w = sps.pic_width_in_luma_samples;
        let h = sps.pic_height_in_luma_samples;
        let min_cb_width = (w >> log2_min_cb_size) as usize;
        let min_cb_height = (h >> log2_min_cb_size) as usize;
        let min_pu_width = (w >> log2_min_pu_size) as usize;
        let min_pu_height = (h >> log2_min_pu_size) as usize;
        Self {
            width: w,
            height: h,
            log2_min_cb_size,
            log2_min_pu_size,
            log2_ctb_size,
            min_cb_width,
            min_pu_width,
            tab_ct_depth: vec![0u8; min_cb_width * min_cb_height],
            // Default IPM is INTRA_DC (matches FFmpeg
            // `intra_prediction_unit_default_value`).
            tab_ipm: vec![INTRA_DC; min_pu_width * min_pu_height],
            last_luma_pred_mode: 0,
            last_chroma_pred_mode: 0,
            cu_count: 0,
        }
    }
}

/// Recursive coding tree decode (HEVC spec 7.3.8.4).
///
/// Phase 2c-1 stops after CU intra mode signaling — it does **not** decode
/// transform_tree, so the CABAC stream position will not match the end of
/// slice once this returns. Use `cu_count` and `last_luma_pred_mode` to
/// observe what was decoded.
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_quadtree(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    sps: &Sps,
    pps: &Pps,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    cb_depth: u8,
) -> Result<(), DecodeError> {
    let cb_size = 1u32 << log2_cb_size;

    // Implicit-no-split when we're at min CB size or when the CU would
    // overflow the picture (spec 7.3.8.4).
    let split_cu = if x0 + cb_size <= state.width
        && y0 + cb_size <= state.height
        && log2_cb_size > state.log2_min_cb_size
    {
        decode_split_cu_flag(cabac, contexts, state, x0, y0, cb_depth)? != 0
    } else {
        log2_cb_size > state.log2_min_cb_size
    };

    if split_cu {
        let cb_size_split = cb_size >> 1;
        let x1 = x0 + cb_size_split;
        let y1 = y0 + cb_size_split;
        decode_coding_quadtree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            x0,
            y0,
            log2_cb_size - 1,
            cb_depth + 1,
        )?;
        if x1 < state.width {
            decode_coding_quadtree(
                cabac,
                contexts,
                state,
                sps,
                pps,
                x1,
                y0,
                log2_cb_size - 1,
                cb_depth + 1,
            )?;
        }
        if y1 < state.height {
            decode_coding_quadtree(
                cabac,
                contexts,
                state,
                sps,
                pps,
                x0,
                y1,
                log2_cb_size - 1,
                cb_depth + 1,
            )?;
        }
        if x1 < state.width && y1 < state.height {
            decode_coding_quadtree(
                cabac,
                contexts,
                state,
                sps,
                pps,
                x1,
                y1,
                log2_cb_size - 1,
                cb_depth + 1,
            )?;
        }
    } else {
        decode_coding_unit(cabac, contexts, state, sps, pps, x0, y0, log2_cb_size)?;
    }

    set_ct_depth(state, x0, y0, log2_cb_size, cb_depth);
    Ok(())
}

/// `split_cu_flag` neighbor context derivation (HEVC spec 9.3.4.2.2).
///
/// `inc = (depth_left > cb_depth) + (depth_top > cb_depth)`. Neighbor depths
/// come from `tab_ct_depth`. For now we treat anything outside the picture
/// as "no neighbor" (depth 0); when multi-slice / multi-CTU support lands,
/// we'll need to also track the per-CTU `ctb_left/up_flag`.
fn decode_split_cu_flag(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &PictureState,
    x0: u32,
    y0: u32,
    cb_depth: u8,
) -> Result<u32, DecodeError> {
    let x_cb = (x0 >> state.log2_min_cb_size) as usize;
    let y_cb = (y0 >> state.log2_min_cb_size) as usize;

    let depth_left = if x_cb > 0 {
        state.tab_ct_depth[y_cb * state.min_cb_width + x_cb - 1]
    } else {
        0
    };
    let depth_top = if y_cb > 0 {
        state.tab_ct_depth[(y_cb - 1) * state.min_cb_width + x_cb]
    } else {
        0
    };

    let mut inc = 0usize;
    if depth_left > cb_depth {
        inc += 1;
    }
    if depth_top > cb_depth {
        inc += 1;
    }

    Ok(cabac.decode_bin(&mut contexts.state[ctx::SPLIT_CODING_UNIT_FLAG + inc]))
}

fn set_ct_depth(state: &mut PictureState, x0: u32, y0: u32, log2_cb_size: u8, cb_depth: u8) {
    let length = ((1u32 << log2_cb_size) >> state.log2_min_cb_size) as usize;
    let x_cb = (x0 >> state.log2_min_cb_size) as usize;
    let y_cb = (y0 >> state.log2_min_cb_size) as usize;
    for j in 0..length {
        let row = (y_cb + j) * state.min_cb_width;
        for i in 0..length {
            state.tab_ct_depth[row + x_cb + i] = cb_depth;
        }
    }
}

/// `coding_unit` decode for the I-slice intra path (spec 7.3.8.5).
///
/// Stops after `intra_chroma_pred_mode` — does NOT decode `transform_tree`.
#[allow(clippy::too_many_arguments)]
fn decode_coding_unit(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    sps: &Sps,
    pps: &Pps,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
) -> Result<(), DecodeError> {
    if pps.transquant_bypass_enabled_flag {
        // FFmpeg would read cu_transquant_bypass_flag here; we already
        // rejected this PPS feature in pps.rs.
        return Err(DecodeError::Unsupported(
            "cu_transquant_bypass not supported",
        ));
    }

    // I-slice: no skip_flag, no pred_mode_flag — pred_mode is always intra.
    // part_mode: only at min CB size do we even decode it (intra has only
    // PART_2Nx2N elsewhere).
    let part_mode = if log2_cb_size == state.log2_min_cb_size {
        let bit0 = cabac.decode_bin(&mut contexts.state[ctx::PART_MODE]);
        if bit0 != 0 {
            PartMode::Part2Nx2N
        } else {
            // For intra at min CB size, the only other allowed value is NxN.
            PartMode::PartNxN
        }
    } else {
        PartMode::Part2Nx2N
    };

    if sps.pcm_enabled_flag {
        // PCM is gated by SPS — we already rejected this in sps.rs, so
        // hitting this branch means a programming error.
        return Err(DecodeError::Unsupported("pcm_enabled_flag not supported"));
    }

    decode_intra_mode_signaling(cabac, contexts, state, x0, y0, log2_cb_size, part_mode)?;

    state.cu_count += 1;
    // Phase 2c-1 stop point: do NOT proceed into transform_tree.
    Ok(())
}

/// Decode all intra prediction modes for a CU's PUs and write them into
/// `tab_ipm`. Mirrors FFmpeg `intra_prediction_unit` for chroma_format_idc=1.
fn decode_intra_mode_signaling(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    part_mode: PartMode,
) -> Result<(), DecodeError> {
    let split = part_mode == PartMode::PartNxN;
    let cb_size = 1u32 << log2_cb_size;
    let pb_size = if split { cb_size >> 1 } else { cb_size };
    let side = if split { 2usize } else { 1 };
    let n_pus = side * side;

    // 1) prev_intra_luma_pred_flag for each PU.
    let mut prev_flag = [false; 4];
    for slot in prev_flag.iter_mut().take(n_pus) {
        *slot = cabac.decode_bin(&mut contexts.state[ctx::PREV_INTRA_LUMA_PRED_FLAG]) != 0;
    }

    // 2) Either mpm_idx (truncated unary, bypass) or rem_intra_luma_pred_mode
    //    (5 bypass bits) for each PU, then derive the actual luma mode.
    let mut intra_pred_mode = [0u8; 4];
    for k in 0..n_pus {
        let mpm_idx;
        let rem;
        if prev_flag[k] {
            // mpm_idx: unary, max value 2 (so up to 2 bypass bits).
            let mut i = 0u8;
            while i < 2 && cabac.decode_bypass() != 0 {
                i += 1;
            }
            mpm_idx = i;
            rem = 0;
        } else {
            mpm_idx = 0;
            rem = cabac.decode_bypass_bits(5) as u8;
        }

        let pj = (k % side) as u32;
        let pi = (k / side) as u32;
        let pu_x = x0 + pb_size * pj;
        let pu_y = y0 + pb_size * pi;

        intra_pred_mode[k] =
            compute_luma_intra_pred_mode(state, pu_x, pu_y, prev_flag[k], mpm_idx, rem);

        // Write the mode into tab_ipm so subsequent PUs (and CUs) can read it
        // for their own MPM derivation.
        write_intra_pred_mode(state, pu_x, pu_y, pb_size, intra_pred_mode[k]);
    }

    // 3) intra_chroma_pred_mode (single value for the whole CU at 4:2:0).
    let chroma_mode_idx = decode_intra_chroma_pred_mode(cabac, contexts);
    let chroma_pred_mode = if chroma_mode_idx == 4 {
        // DM mode: chroma uses luma mode.
        intra_pred_mode[0]
    } else {
        // Spec table 8-3: chroma_mode_idx → luma-mode space.
        const TABLE: [u8; 4] = [INTRA_PLANAR, INTRA_ANGULAR_26, INTRA_ANGULAR_10, INTRA_DC];
        let mapped = TABLE[chroma_mode_idx as usize];
        if intra_pred_mode[0] == mapped {
            INTRA_ANGULAR_34
        } else {
            mapped
        }
    };

    state.last_luma_pred_mode = intra_pred_mode[0];
    state.last_chroma_pred_mode = chroma_pred_mode;
    Ok(())
}

fn write_intra_pred_mode(state: &mut PictureState, x0: u32, y0: u32, pu_size: u32, mode: u8) {
    let size_in_pus = (pu_size >> state.log2_min_pu_size).max(1) as usize;
    let x_pu = (x0 >> state.log2_min_pu_size) as usize;
    let y_pu = (y0 >> state.log2_min_pu_size) as usize;
    for j in 0..size_in_pus {
        let row = (y_pu + j) * state.min_pu_width;
        for i in 0..size_in_pus {
            state.tab_ipm[row + x_pu + i] = mode;
        }
    }
}

/// `intra_chroma_pred_mode` decode (FFmpeg `ff_hevc_intra_chroma_pred_mode_decode`).
/// Returns 4 for "DM" (chroma uses luma mode), or 0..3 for the chroma table
/// index (spec table 8-3).
fn decode_intra_chroma_pred_mode(cabac: &mut CabacReader, contexts: &mut CabacContexts) -> u8 {
    if cabac.decode_bin(&mut contexts.state[ctx::INTRA_CHROMA_PRED_MODE]) == 0 {
        return 4;
    }
    let hi = cabac.decode_bypass();
    let lo = cabac.decode_bypass();
    ((hi << 1) | lo) as u8
}

/// Luma intra mode derivation with the 3-entry MPM list (HEVC spec 8.4.2).
/// Mirrors FFmpeg `luma_intra_pred_mode`.
fn compute_luma_intra_pred_mode(
    state: &PictureState,
    x0: u32,
    y0: u32,
    prev_intra_luma_pred_flag: bool,
    mpm_idx: u8,
    rem_intra_luma_pred_mode: u8,
) -> u8 {
    let log2_ctb = state.log2_ctb_size as u32;
    let x_pu = (x0 >> state.log2_min_pu_size) as usize;
    let y_pu = (y0 >> state.log2_min_pu_size) as usize;
    let y_ctb = (y0 >> log2_ctb) << log2_ctb;

    // Within a single-CTU picture, "neighbor exists" reduces to "x_pu/y_pu > 0".
    let mut cand_up = if y_pu > 0 {
        state.tab_ipm[(y_pu - 1) * state.min_pu_width + x_pu]
    } else {
        INTRA_DC
    };
    let cand_left = if x_pu > 0 {
        state.tab_ipm[y_pu * state.min_pu_width + x_pu - 1]
    } else {
        INTRA_DC
    };

    // Intra mode prediction does not cross vertical CTB boundaries
    // (FFmpeg comment / spec 8.4.2).
    if (y0 as i64 - 1) < y_ctb as i64 {
        cand_up = INTRA_DC;
    }

    let mut candidate = [0u8; 3];
    if cand_left == cand_up {
        if cand_left < 2 {
            candidate[0] = INTRA_PLANAR;
            candidate[1] = INTRA_DC;
            candidate[2] = INTRA_ANGULAR_26;
        } else {
            // Both neighbors are the same angular mode → derive ±1 angular.
            candidate[0] = cand_left;
            candidate[1] = 2 + (((cand_left as i32) - 2 - 1 + 32) & 31) as u8;
            candidate[2] = 2 + (((cand_left as i32) - 2 + 1) & 31) as u8;
        }
    } else {
        candidate[0] = cand_left;
        candidate[1] = cand_up;
        if candidate[0] != INTRA_PLANAR && candidate[1] != INTRA_PLANAR {
            candidate[2] = INTRA_PLANAR;
        } else if candidate[0] != INTRA_DC && candidate[1] != INTRA_DC {
            candidate[2] = INTRA_DC;
        } else {
            candidate[2] = INTRA_ANGULAR_26;
        }
    }

    if prev_intra_luma_pred_flag {
        candidate[mpm_idx as usize]
    } else {
        // Sort the candidate list ascending, then add `rem_intra_luma_pred_mode`
        // and bump it past every candidate it equals or exceeds.
        if candidate[0] > candidate[1] {
            candidate.swap(0, 1);
        }
        if candidate[0] > candidate[2] {
            candidate.swap(0, 2);
        }
        if candidate[1] > candidate[2] {
            candidate.swap(1, 2);
        }
        let mut mode = rem_intra_luma_pred_mode;
        for c in &candidate {
            if mode >= *c {
                mode += 1;
            }
        }
        mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacContexts;
    use crate::nal::{parse_annex_b, NalUnitType};
    use crate::pps::parse_pps;
    use crate::slice::{parse_slice_segment_header, SliceType};
    use crate::sps::parse_sps;

    /// End-to-end Phase 2c-1 test: parse `testdata/tiny_intra.h265`, run the
    /// CU tree decoder up to (but not including) `transform_tree`, and assert
    /// the resulting intra prediction modes.
    ///
    /// For the all-gray 16×16 fixture we **expect** x265 to pick the cheapest
    /// possible mode (PLANAR via mpm_idx=0) for the only CU and DM (= same
    /// as luma) for chroma. If x265 ever changes its mind, the assertion
    /// values would need to be regenerated against an FFmpeg trace.
    #[test]
    fn test_decode_tiny_intra_cu_tree_phase2c1() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tiny_intra.h265");
        let data = std::fs::read(path).expect("read fixture");
        let nals = parse_annex_b(&data);

        let sps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Sps)
            .expect("SPS NAL");
        let pps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Pps)
            .expect("PPS NAL");
        let slice_nal = nals
            .iter()
            .find(|n| n.nal_unit_type.is_idr())
            .expect("IDR slice NAL");

        let sps = parse_sps(&sps_nal.rbsp).expect("parse SPS");
        let pps = parse_pps(&pps_nal.rbsp).expect("parse PPS");
        let sh = parse_slice_segment_header(&slice_nal.rbsp, slice_nal.nal_unit_type, &sps, &pps)
            .expect("parse slice header");

        assert_eq!(sh.slice_type, SliceType::I);
        assert_eq!(sh.slice_qp_y, 25);

        // CABAC contexts initialized from the slice's effective QP.
        let mut contexts = CabacContexts::init(sh.slice_qp_y, sh.slice_type, false);

        // CABAC bytestream begins immediately after the slice header
        // (header is byte-aligned for our fixture).
        let cabac_byte_offset = sh.header_size_bits / 8;
        assert_eq!(cabac_byte_offset, 2, "fixture slice header is exactly 2 bytes");
        let mut cabac = CabacReader::new(&slice_nal.rbsp, cabac_byte_offset);

        let mut state = PictureState::new(&sps);
        // The single CTU is at (0, 0) with log2_cb_size = ctb_log2_size_y = 4.
        decode_coding_quadtree(
            &mut cabac,
            &mut contexts,
            &mut state,
            &sps,
            &pps,
            0,
            0,
            sps.ctb_log2_size_y,
            0,
        )
        .expect("decode coding tree");

        // For the flat-gray 16x16 frame, x265 with --no-signhide should pick
        // the cheapest available intra mode. With no neighbors, the MPM list
        // is [PLANAR, DC, ANGULAR_26], and PLANAR via mpm_idx=0 is cheapest.
        assert!(
            state.cu_count >= 1,
            "expected at least one CU; got {}",
            state.cu_count
        );
        assert!(
            state.last_luma_pred_mode <= 34,
            "luma intra mode {} out of range",
            state.last_luma_pred_mode
        );
        assert!(
            state.last_chroma_pred_mode <= 34,
            "chroma intra mode {} out of range",
            state.last_chroma_pred_mode
        );

        // Hypothesis: PLANAR luma + DM chroma. If this assertion ever
        // breaks because x265 picked a different mode, regenerate the
        // expected value from an FFmpeg trace.
        assert_eq!(
            state.last_luma_pred_mode,
            INTRA_PLANAR,
            "expected PLANAR luma intra for flat-gray fixture, got {}",
            state.last_luma_pred_mode
        );
        assert_eq!(
            state.last_chroma_pred_mode,
            INTRA_PLANAR,
            "expected DM chroma (= PLANAR) for flat-gray fixture, got {}",
            state.last_chroma_pred_mode
        );
    }
}
