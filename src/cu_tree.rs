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

    /// Phase 2c-2 sentinels — most recent values from `transform_tree`.
    pub last_split_transform_flag: bool,
    pub last_cbf_luma: bool,
    pub last_cbf_cb: bool,
    pub last_cbf_cr: bool,
    /// Signed CU QP delta as decoded for the most recent TU. 0 if not coded.
    pub last_cu_qp_delta: i32,
    /// Effective QP after applying `last_cu_qp_delta` to `slice_qp_y`.
    pub last_qp_y: i32,
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
            last_split_transform_flag: false,
            last_cbf_luma: false,
            last_cbf_cb: false,
            last_cbf_cr: false,
            last_cu_qp_delta: 0,
            last_qp_y: 0,
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

    // Phase 2c-2: descend into transform_tree (no residual_coding yet).
    // For intra at PART_2Nx2N, intra_split is false → max_trafo_depth =
    // sps.max_transform_hierarchy_depth_intra. The PART_NxN case adds 1 to
    // max_trafo_depth and sets intra_split, but our fixture doesn't hit it.
    let intra_split = part_mode == PartMode::PartNxN;
    let max_trafo_depth =
        sps.max_transform_hierarchy_depth_intra + if intra_split { 1 } else { 0 };

    decode_transform_tree(
        cabac,
        contexts,
        state,
        sps,
        pps,
        x0,
        y0,
        log2_cb_size,
        log2_cb_size,
        0,
        max_trafo_depth,
        intra_split,
        TransformTreeCbf::default(),
    )?;

    state.cu_count += 1;
    Ok(())
}

/// Inherited cbf state passed down through `decode_transform_tree` recursion.
/// Once a parent has `cbf_cb = 0`, the child does not re-decode it (spec
/// 7.3.8.10 / FFmpeg `cbf_cb[]` propagation).
#[derive(Debug, Clone, Copy, Default)]
struct TransformTreeCbf {
    cbf_cb: bool,
    cbf_cr: bool,
    /// "is_cu_qp_delta_coded" sentinel — set to `true` once the first TU in
    /// the CU has decoded `cu_qp_delta`. Subsequent TUs skip the read.
    cu_qp_delta_coded: bool,
}

/// Recursive transform tree decode (HEVC spec 7.3.8.10).
///
/// Phase 2c-2 implements the structural part — `split_transform_flag` with
/// gating, chroma `cbf_cb`/`cbf_cr` decode, recursion, and the leaf
/// `decode_transform_unit`. The leaf does NOT yet call `residual_coding`,
/// so the test must select inputs where the CABAC stream stops at a usable
/// point (i.e. just after `cu_qp_delta` for the first non-empty TU).
///
/// `log2_cb_size` is currently only forwarded into the recursion; it'll be
/// consumed by Phase 2c-3 (residual_coding scan size).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::only_used_in_recursion)]
fn decode_transform_tree(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    sps: &Sps,
    pps: &Pps,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    log2_trafo_size: u8,
    trafo_depth: u8,
    max_trafo_depth: u32,
    intra_split: bool,
    parent_cbf: TransformTreeCbf,
) -> Result<TransformTreeCbf, DecodeError> {
    // 1) Decide split_transform_flag (FFmpeg `hls_transform_tree` lines 1566-1580).
    let split_transform_flag = if log2_trafo_size <= sps.max_tb_log2_size_y
        && log2_trafo_size > sps.min_tb_log2_size_y
        && (trafo_depth as u32) < max_trafo_depth
        && !(intra_split && trafo_depth == 0)
    {
        decode_split_transform_flag(cabac, contexts, log2_trafo_size) != 0
    } else {
        // Implicit split: oversized TU, intra_split forcing depth-1, or inter
        // split (which we don't hit for I-slices).
        log2_trafo_size > sps.max_tb_log2_size_y || (intra_split && trafo_depth == 0)
    };

    state.last_split_transform_flag = split_transform_flag;

    // 2) Chroma cbf decode. For 4:2:0 (chroma_format_idc==1) we only do this
    //    when log2_trafo_size > 2 (chroma TUs would otherwise be 2×2, which is
    //    illegal). The flag is decoded when either (a) we're at the root of
    //    the transform tree, or (b) the parent already had a non-zero cbf.
    let mut cbf_cb = parent_cbf.cbf_cb;
    let mut cbf_cr = parent_cbf.cbf_cr;
    if sps.chroma_format_idc == 1 && log2_trafo_size > 2 {
        if trafo_depth == 0 || parent_cbf.cbf_cb {
            cbf_cb = decode_cbf_cb_cr(cabac, contexts, trafo_depth) != 0;
        }
        if trafo_depth == 0 || parent_cbf.cbf_cr {
            cbf_cr = decode_cbf_cb_cr(cabac, contexts, trafo_depth) != 0;
        }
    }
    state.last_cbf_cb = cbf_cb;
    state.last_cbf_cr = cbf_cr;

    let inherited = TransformTreeCbf {
        cbf_cb,
        cbf_cr,
        cu_qp_delta_coded: parent_cbf.cu_qp_delta_coded,
    };

    if split_transform_flag {
        let trafo_size_split = 1u32 << (log2_trafo_size - 1);
        let x1 = x0 + trafo_size_split;
        let y1 = y0 + trafo_size_split;
        let mut child_cbf = inherited;
        child_cbf = decode_transform_tree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            x0,
            y0,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            child_cbf,
        )?;
        child_cbf = decode_transform_tree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            x1,
            y0,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            child_cbf,
        )?;
        child_cbf = decode_transform_tree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            x0,
            y1,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            child_cbf,
        )?;
        let final_cbf = decode_transform_tree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            x1,
            y1,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            child_cbf,
        )?;
        Ok(final_cbf)
    } else {
        decode_transform_unit(
            cabac,
            contexts,
            state,
            sps,
            pps,
            x0,
            y0,
            log2_trafo_size,
            trafo_depth,
            inherited,
        )
    }
}

/// Decode `split_transform_flag` (HEVC spec 9.3.4.2.5).
/// Context offset: `SPLIT_TRANSFORM_FLAG + (5 - log2_trafo_size)`.
fn decode_split_transform_flag(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    log2_trafo_size: u8,
) -> u32 {
    let inc = 5usize - log2_trafo_size as usize;
    cabac.decode_bin(&mut contexts.state[ctx::SPLIT_TRANSFORM_FLAG + inc])
}

/// Decode `cbf_cb` or `cbf_cr` — same context structure (HEVC spec 9.3.4.2.6).
/// Context offset: `CBF_CB_CR + trafo_depth`.
fn decode_cbf_cb_cr(cabac: &mut CabacReader, contexts: &mut CabacContexts, trafo_depth: u8) -> u32 {
    cabac.decode_bin(&mut contexts.state[ctx::CBF_CB_CR + trafo_depth as usize])
}

/// Decode `cbf_luma` (HEVC spec 9.3.4.2.6).
/// Context offset: `CBF_LUMA + (trafo_depth == 0 ? 1 : 0)`.
/// (FFmpeg expresses this as `CBF_LUMA + !trafo_depth`.)
fn decode_cbf_luma(cabac: &mut CabacReader, contexts: &mut CabacContexts, trafo_depth: u8) -> u32 {
    let inc = if trafo_depth == 0 { 1 } else { 0 };
    cabac.decode_bin(&mut contexts.state[ctx::CBF_LUMA + inc])
}

/// Decode `cu_qp_delta_abs` (HEVC spec 9.3.4.2.7) — truncated unary prefix
/// (max value 5) followed by an Exp-Golomb-0 suffix when the prefix is at
/// its max.
fn decode_cu_qp_delta_abs(cabac: &mut CabacReader, contexts: &mut CabacContexts) -> u32 {
    let mut prefix = 0u32;
    let mut inc = 0usize;
    while prefix < 5 && cabac.decode_bin(&mut contexts.state[ctx::CU_QP_DELTA + inc]) != 0 {
        prefix += 1;
        inc = 1;
    }
    if prefix < 5 {
        return prefix;
    }
    // EG-0 suffix: read bypass bits until a 0, then `k` more.
    let mut suffix = 0u32;
    let mut k = 0u32;
    while k < 7 && cabac.decode_bypass() != 0 {
        suffix += 1 << k;
        k += 1;
    }
    while k > 0 {
        k -= 1;
        suffix += cabac.decode_bypass() << k;
    }
    prefix + suffix
}

/// Decode `cu_qp_delta_sign_flag` (bypass).
fn decode_cu_qp_delta_sign_flag(cabac: &mut CabacReader) -> u32 {
    cabac.decode_bypass()
}

/// `transform_unit` decode (spec 7.3.8.11).
///
/// Phase 2c-2 stops just before residual_coding. We do decode `cu_qp_delta`
/// when applicable so the CABAC stream is at the correct position for the
/// first residual_coding call (Phase 2c-3).
#[allow(clippy::too_many_arguments)]
fn decode_transform_unit(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    _sps: &Sps,
    pps: &Pps,
    _x0: u32,
    _y0: u32,
    log2_trafo_size: u8,
    trafo_depth: u8,
    inherited: TransformTreeCbf,
) -> Result<TransformTreeCbf, DecodeError> {
    // FFmpeg gates cbf_luma decoding behind:
    //   pred_mode == INTRA || trafo_depth != 0 || any chroma cbf set
    // For our I-slice intra path, the first clause is always true.
    let cbf_luma = decode_cbf_luma(cabac, contexts, trafo_depth) != 0;
    state.last_cbf_luma = cbf_luma;

    let mut new_cbf = inherited;

    if cbf_luma || inherited.cbf_cb || inherited.cbf_cr {
        // cu_qp_delta is decoded once per CU, the first time we see a TU
        // with a non-zero CBF.
        if pps.cu_qp_delta_enabled_flag && !inherited.cu_qp_delta_coded {
            let abs = decode_cu_qp_delta_abs(cabac, contexts) as i32;
            let signed = if abs != 0 {
                let sign = decode_cu_qp_delta_sign_flag(cabac);
                if sign != 0 {
                    -abs
                } else {
                    abs
                }
            } else {
                0
            };
            state.last_cu_qp_delta = signed;
            // Spec 7.4.7.10: cu_qp_delta_val ∈ [-(26 + QpBdOffsetY/2),
            // 25 + QpBdOffsetY/2]. For 8-bit, that's [-26, 25].
            if !(-26..=25).contains(&signed) {
                return Err(DecodeError::InvalidSyntax("cu_qp_delta out of range"));
            }
            new_cbf.cu_qp_delta_coded = true;
        }

        // Phase 2c-2 stops here. residual_coding (luma + chroma) is the
        // next thing to land in Phase 2c-3.
        // The pre-residual CABAC bin sequence ends right after cu_qp_delta.
        let _ = log2_trafo_size; // silence unused warning until 2c-3
    }

    Ok(new_cbf)
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

    /// End-to-end Phase 2c-1/2c-2 test: parse `testdata/tiny_intra.h265`,
    /// run the CU tree decoder up through `cu_qp_delta` (just before the
    /// first residual_coding call), and assert the decoded intra modes,
    /// `cbf_*` flags, and `cu_qp_delta`.
    ///
    /// For the all-gray 16×16 fixture we **expect** x265 to pick PLANAR for
    /// luma and DM for chroma. The cbf_* flags will reveal whether x265
    /// produced any non-zero residual coefficients (the reference YUV being
    /// 0x7E vs the prediction's 0x80 strongly suggests yes for luma).
    #[test]
    fn test_decode_tiny_intra_cu_tree_phase2c2() {
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

        // Phase 2c-2 sentinels: with `max_transform_hierarchy_depth_intra=0`
        // and intra_split=false, we expect no split_transform_flag bin to
        // be decoded — implicit no-split. The TU is the full 16×16 CU.
        assert!(
            !state.last_split_transform_flag,
            "expected implicit no-split for 16x16 CU at depth 0"
        );

        // The flat-gray fixture's reference YUV is 0x7E (=126), but PLANAR
        // prediction with no neighbors gives 0x80 (=128). So luma residual
        // must be non-zero → cbf_luma should be 1. Chroma stays at 0x80 in
        // both prediction and reference, so cbf_cb and cbf_cr should be 0.
        assert!(
            state.last_cbf_luma,
            "expected cbf_luma=1 for non-trivial luma residual"
        );
        assert!(
            !state.last_cbf_cb,
            "expected cbf_cb=0 for chroma matching prediction"
        );
        assert!(
            !state.last_cbf_cr,
            "expected cbf_cr=0 for chroma matching prediction"
        );

        // cu_qp_delta is signaled (cu_qp_delta_enabled_flag=1 in PPS) once
        // the first non-zero CBF appears. x265's CRF rate control on this
        // fixture applies an adaptive QP — encoder log reports
        // "Avg QP:20.00", and slice_qp_y is 25, so the per-CU delta is -5.
        // (Decoding -5 successfully also exercises the truncated-unary
        // prefix at its max value followed by the EG-0 suffix path of
        // `cu_qp_delta_abs`.)
        assert_eq!(
            state.last_cu_qp_delta, -5,
            "expected cu_qp_delta=-5 for x265 CRF AQ on flat fixture"
        );
    }
}
