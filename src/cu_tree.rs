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
use crate::intra_pred::{
    ReferenceAvailability, add_residual, build_reference_samples, filter_reference_samples,
    predict_angular, predict_dc, predict_planar,
};
use crate::inverse_transform::apply_inverse_transform;
use crate::pps::Pps;
use crate::residual_coding::{ResidualBlock, ResidualPlane, ScanOrder, decode_residual_coding};
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
/// and stores the decoded luma intra prediction mode for downstream use.
/// `y_plane`/`u_plane`/`v_plane` are the reconstructed picture planes that
/// `decode_transform_unit` writes prediction + residual into.
pub struct PictureState {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub log2_min_cb_size: u8,
    pub log2_min_pu_size: u8,
    pub log2_ctb_size: u8,
    pub min_cb_width: usize,
    pub min_pu_width: usize,
    pub tab_ct_depth: Vec<u8>,
    pub tab_ipm: Vec<u8>,
    pub y_plane: Vec<u8>,
    pub u_plane: Vec<u8>,
    pub v_plane: Vec<u8>,
    pub y_stride: usize,
    pub uv_stride: usize,

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

    /// Phase 2c-3 sentinel — most recently decoded luma residual block.
    pub last_luma_residual: Option<ResidualBlock>,

    /// Phase 3b-1 deblocking: per-min-CB QP for deblock filter strength.
    pub tab_qp_y: Vec<u8>,
    /// Phase 3b-1 deblocking: per-4×4 boundary strength for vertical edges.
    /// Indexed by `(y/4) * (width/4) + (x/4)` (matches FFmpeg's `vertical_bs`).
    pub bs_vertical: Vec<u8>,
    /// Phase 3b-1 deblocking: per-4×4 boundary strength for horizontal edges.
    pub bs_horizontal: Vec<u8>,
    /// Phase 3b-2 SAO: per-CTB SAO parameters, indexed by CTB raster address.
    pub sao_params: Vec<crate::sao::SaoParams>,
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
        let y_stride = w as usize;
        let uv_stride = (w / 2) as usize;
        Self {
            width: w,
            height: h,
            bit_depth: sps.bit_depth_luma,
            log2_min_cb_size,
            log2_min_pu_size,
            log2_ctb_size,
            min_cb_width,
            min_pu_width,
            tab_ct_depth: vec![0u8; min_cb_width * min_cb_height],
            // Default IPM is INTRA_DC (matches FFmpeg
            // `intra_prediction_unit_default_value`).
            tab_ipm: vec![INTRA_DC; min_pu_width * min_pu_height],
            y_plane: vec![0u8; (w * h) as usize],
            u_plane: vec![0u8; ((w / 2) * (h / 2)) as usize],
            v_plane: vec![0u8; ((w / 2) * (h / 2)) as usize],
            y_stride,
            uv_stride,
            last_luma_pred_mode: 0,
            last_chroma_pred_mode: 0,
            cu_count: 0,
            last_split_transform_flag: false,
            last_cbf_luma: false,
            last_cbf_cb: false,
            last_cbf_cr: false,
            last_cu_qp_delta: 0,
            last_qp_y: 0,
            last_luma_residual: None,
            tab_qp_y: vec![0u8; min_cb_width * min_cb_height],
            bs_vertical: vec![0u8; ((w / 4) * (h / 4)) as usize],
            bs_horizontal: vec![0u8; ((w / 4) * (h / 4)) as usize],
            sao_params: {
                let ctb_size = 1u32 << log2_ctb_size;
                let pw = w.div_ceil(ctb_size) as usize;
                let ph = h.div_ceil(ctb_size) as usize;
                vec![crate::sao::SaoParams::default(); pw * ph]
            },
        }
    }
}

/// Recursive coding tree decode (HEVC spec 7.3.8.4).
///
/// `slice_qp_y` is needed to compute the per-CU effective QP for dequant
/// (`qp_y = slice_qp_y + cu_qp_delta`).
///
/// Returns `Ok(true)` if there is more data in the slice (= `end_of_slice_flag`
/// was 0 at the CTB boundary, OR we haven't reached one yet), or `Ok(false)`
/// if we've consumed the slice's terminate bin and the slice is done.
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_quadtree(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    sps: &Sps,
    pps: &Pps,
    slice_qp_y: i32,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    cb_depth: u8,
) -> Result<bool, DecodeError> {
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

    let more_data;
    if split_cu {
        let cb_size_split = cb_size >> 1;
        let x1 = x0 + cb_size_split;
        let y1 = y0 + cb_size_split;
        let mut md = decode_coding_quadtree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            slice_qp_y,
            x0,
            y0,
            log2_cb_size - 1,
            cb_depth + 1,
        )?;
        if md && x1 < state.width {
            md = decode_coding_quadtree(
                cabac,
                contexts,
                state,
                sps,
                pps,
                slice_qp_y,
                x1,
                y0,
                log2_cb_size - 1,
                cb_depth + 1,
            )?;
        }
        if md && y1 < state.height {
            md = decode_coding_quadtree(
                cabac,
                contexts,
                state,
                sps,
                pps,
                slice_qp_y,
                x0,
                y1,
                log2_cb_size - 1,
                cb_depth + 1,
            )?;
        }
        if md && x1 < state.width && y1 < state.height {
            md = decode_coding_quadtree(
                cabac,
                contexts,
                state,
                sps,
                pps,
                slice_qp_y,
                x1,
                y1,
                log2_cb_size - 1,
                cb_depth + 1,
            )?;
        }
        more_data = md;
    } else {
        decode_coding_unit(
            cabac,
            contexts,
            state,
            sps,
            pps,
            slice_qp_y,
            x0,
            y0,
            log2_cb_size,
        )?;

        // After a leaf CU, decode end_of_slice_flag if we're at a CTB
        // boundary (or picture edge). Spec 7.3.8.5.
        let ctb_size = 1u32 << state.log2_ctb_size;
        let at_ctb_x_edge =
            (x0 + cb_size).is_multiple_of(ctb_size) || (x0 + cb_size >= state.width);
        let at_ctb_y_edge =
            (y0 + cb_size).is_multiple_of(ctb_size) || (y0 + cb_size >= state.height);
        if at_ctb_x_edge && at_ctb_y_edge {
            let end_of_slice = cabac.decode_terminate();
            more_data = end_of_slice == 0;
        } else {
            more_data = true;
        }
    }

    set_ct_depth(state, x0, y0, log2_cb_size, cb_depth);
    Ok(more_data)
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
#[allow(clippy::too_many_arguments)]
fn decode_coding_unit(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    sps: &Sps,
    pps: &Pps,
    slice_qp_y: i32,
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

    // PCM decode gate (spec 7.3.8.5 / FFmpeg `hls_coding_unit`). `pcm_flag`
    // is signaled as a terminate bin only when:
    //   - part_mode == PART_2Nx2N
    //   - sps.pcm_enabled_flag
    //   - log2_min_pcm_cb_size <= log2_cb_size <= log2_max_pcm_cb_size
    let pcm_allowed = sps.pcm_enabled_flag
        && part_mode == PartMode::Part2Nx2N
        && log2_cb_size >= sps.log2_min_pcm_cb_size
        && log2_cb_size <= sps.log2_max_pcm_cb_size;
    if pcm_allowed && cabac.decode_terminate() != 0 {
        // PCM block: skip intra prediction signaling, prediction, residual,
        // and transform entirely. The raw PCM bytes follow `pcm_flag` at the
        // next byte boundary; after consuming them CABAC is reinitialized.
        decode_pcm_block(cabac, state, sps, x0, y0, log2_cb_size)?;
        state.cu_count += 1;
        return Ok(());
    }

    decode_intra_mode_signaling(cabac, contexts, state, x0, y0, log2_cb_size, part_mode)?;

    // Phase 2c-2: descend into transform_tree (no residual_coding yet).
    // For intra at PART_2Nx2N, intra_split is false → max_trafo_depth =
    // sps.max_transform_hierarchy_depth_intra. The PART_NxN case adds 1 to
    // max_trafo_depth and sets intra_split, but our fixture doesn't hit it.
    let intra_split = part_mode == PartMode::PartNxN;
    let max_trafo_depth = sps.max_transform_hierarchy_depth_intra + if intra_split { 1 } else { 0 };

    decode_transform_tree(
        cabac,
        contexts,
        state,
        sps,
        pps,
        slice_qp_y,
        x0,
        y0,
        x0,
        y0,
        log2_cb_size,
        log2_cb_size,
        0,
        max_trafo_depth,
        intra_split,
        0,
        TransformTreeCbf::default(),
    )?;

    state.cu_count += 1;
    Ok(())
}

/// Decode a PCM (raw pixel) CU (HEVC spec 7.3.8.6 / FFmpeg `hls_pcm_sample`).
///
/// The PCM sample payload is bit-packed in the order Y, then Cb, then Cr.
/// Total length in bits is
///
/// ```text
///     cb_size * cb_size * pcm_bit_depth
///   + 2 * (cb_size/2) * (cb_size/2) * pcm_bit_depth_chroma
/// ```
///
/// and the CABAC engine is reinitialized at the next byte boundary afterwards.
///
/// Samples are stored scaled by `1 << (BitDepth - PcmBitDepth)` — i.e. the
/// reconstructed picture's bit depth may be larger than the PCM sample bit
/// depth, in which case PCM samples get left-shifted to match. We only
/// support 8-bit reconstruction today so the shift is in [0, 7].
fn decode_pcm_block(
    cabac: &mut CabacReader,
    state: &mut PictureState,
    sps: &Sps,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
) -> Result<(), DecodeError> {
    let cb_size = 1usize << log2_cb_size;
    let pcm_bd_luma: u8 = sps.pcm_sample_bit_depth_luma;
    let pcm_bd_chroma: u8 = sps.pcm_sample_bit_depth_chroma;
    // `pcm_bit_depth` is guaranteed <= `bit_depth_luma/chroma` by sps.rs, so
    // these shifts are in [0, 7] for 8-bit reconstruction.
    let luma_shift: u8 = sps.bit_depth_luma - pcm_bd_luma;
    let chroma_shift: u8 = sps.bit_depth_chroma - pcm_bd_chroma;

    // Byte offset where the raw PCM bytes live.
    let pcm_start = cabac.pcm_byte_position();

    // Total payload length in bits (spec 7.3.8.6, 4:2:0 only).
    let cb_chroma = cb_size / 2;
    let length_bits = cb_size * cb_size * pcm_bd_luma as usize
        + 2 * cb_chroma * cb_chroma * pcm_bd_chroma as usize;
    let length_bytes = length_bits.div_ceil(8);

    // Bounds check against the RBSP buffer.
    if pcm_start + length_bytes > cabac.rbsp().len() {
        return Err(DecodeError::UnexpectedEof);
    }
    // Borrow once so we don't alias `cabac` across the read loop.
    let pcm_bytes: Vec<u8> = cabac.rbsp()[pcm_start..pcm_start + length_bytes].to_vec();

    let mut reader = PcmBitReader::new(&pcm_bytes);

    // Luma plane write.
    {
        let stride = state.y_stride;
        let dst_off = (y0 as usize) * stride + (x0 as usize);
        for j in 0..cb_size {
            for i in 0..cb_size {
                let sample = reader.read_bits(pcm_bd_luma) as u8;
                state.y_plane[dst_off + j * stride + i] = sample << luma_shift;
            }
        }
    }

    // Chroma planes (Cb, Cr). For 4:2:0 both planes are `cb_size/2` in each
    // dimension and share the same bit depth.
    {
        let stride = state.uv_stride;
        let x_c = (x0 as usize) >> 1;
        let y_c = (y0 as usize) >> 1;
        let dst_off = y_c * stride + x_c;
        for plane_idx in 0..2 {
            let plane = if plane_idx == 0 {
                &mut state.u_plane
            } else {
                &mut state.v_plane
            };
            for j in 0..cb_chroma {
                for i in 0..cb_chroma {
                    let sample = reader.read_bits(pcm_bd_chroma) as u8;
                    plane[dst_off + j * stride + i] = sample << chroma_shift;
                }
            }
        }
    }

    // Record the luma mode for subsequent CUs' MPM derivation. PCM CUs
    // contribute `INTRA_DC` to the IPM table (spec 8.4.2, same as
    // "not available" — handled by `intra_prediction_unit_default_value`
    // in FFmpeg).
    let pb_size = cb_size as u32;
    write_intra_pred_mode(state, x0, y0, pb_size, INTRA_DC);
    state.last_luma_pred_mode = INTRA_DC;
    state.last_chroma_pred_mode = INTRA_DC;

    // Reinit CABAC at the byte boundary immediately after the PCM payload.
    cabac.reinit_at(pcm_start + length_bytes);

    Ok(())
}

/// Minimal bit-packed reader for PCM samples (MSB-first within each byte).
struct PcmBitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> PcmBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn read_bits(&mut self, n: u8) -> u32 {
        // Fast path for 8-bit byte-aligned reads (the common case).
        if n == 8 && self.bit_pos.is_multiple_of(8) {
            let b = self.data[self.bit_pos / 8] as u32;
            self.bit_pos += 8;
            return b;
        }
        let mut val: u32 = 0;
        for _ in 0..n {
            let byte = self.data[self.bit_pos / 8] as u32;
            let bit = (byte >> (7 - (self.bit_pos & 7))) & 1;
            val = (val << 1) | bit;
            self.bit_pos += 1;
        }
        val
    }
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
#[allow(clippy::too_many_arguments)]
#[allow(clippy::only_used_in_recursion)]
fn decode_transform_tree(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    sps: &Sps,
    pps: &Pps,
    slice_qp_y: i32,
    x0: u32,
    y0: u32,
    x_base: u32,
    y_base: u32,
    log2_cb_size: u8,
    log2_trafo_size: u8,
    trafo_depth: u8,
    max_trafo_depth: u32,
    intra_split: bool,
    blk_idx: u8,
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
            slice_qp_y,
            x0,
            y0,
            x0,
            y0,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            0,
            child_cbf,
        )?;
        child_cbf = decode_transform_tree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            slice_qp_y,
            x1,
            y0,
            x0,
            y0,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            1,
            child_cbf,
        )?;
        child_cbf = decode_transform_tree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            slice_qp_y,
            x0,
            y1,
            x0,
            y0,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            2,
            child_cbf,
        )?;
        let final_cbf = decode_transform_tree(
            cabac,
            contexts,
            state,
            sps,
            pps,
            slice_qp_y,
            x1,
            y1,
            x0,
            y0,
            log2_cb_size,
            log2_trafo_size - 1,
            trafo_depth + 1,
            max_trafo_depth,
            intra_split,
            3,
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
            slice_qp_y,
            x0,
            y0,
            x_base,
            y_base,
            log2_trafo_size,
            trafo_depth,
            blk_idx,
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

/// `transform_unit` decode (spec 7.3.8.11). Decodes the cbf flags,
/// `cu_qp_delta`, and (if any cbf is set) the per-plane residual_coding.
/// Also performs intra prediction and reconstruction (residual + prediction
/// → clipped pixels) into the picture's frame planes.
#[allow(clippy::too_many_arguments)]
fn decode_transform_unit(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    state: &mut PictureState,
    sps: &Sps,
    pps: &Pps,
    slice_qp_y: i32,
    x0: u32,
    y0: u32,
    x_base: u32,
    y_base: u32,
    log2_trafo_size: u8,
    trafo_depth: u8,
    blk_idx: u8,
    inherited: TransformTreeCbf,
) -> Result<TransformTreeCbf, DecodeError> {
    // ---- Step 1: luma intra prediction (always for the I-slice intra path).
    let luma_mode = state.last_luma_pred_mode;
    predict_intra_luma(state, sps, x0, y0, log2_trafo_size, luma_mode)?;

    // ---- Step 2: cbf_luma decode.
    // FFmpeg gates cbf_luma decoding behind:
    //   pred_mode == INTRA || trafo_depth != 0 || any chroma cbf set
    // For our I-slice intra path, the first clause is always true.
    let cbf_luma = decode_cbf_luma(cabac, contexts, trafo_depth) != 0;
    state.last_cbf_luma = cbf_luma;

    let mut new_cbf = inherited;
    // For 4:2:0, chroma is handled at log2_trafo_size > 2. When log2_trafo_size
    // == 2 (4x4 luma TUs), chroma is deferred to blk_idx==3 where it's handled
    // at the parent TU size (xBase, yBase, log2_trafo_size == parent's log2-1).
    let do_chroma_inline = sps.chroma_format_idc == 1 && log2_trafo_size > 2;
    let do_chroma_deferred = sps.chroma_format_idc == 1 && log2_trafo_size == 2 && blk_idx == 3;

    if cbf_luma || inherited.cbf_cb || inherited.cbf_cr {
        // cu_qp_delta is decoded once per CU, the first time we see a TU
        // with a non-zero CBF.
        if pps.cu_qp_delta_enabled_flag && !inherited.cu_qp_delta_coded {
            let abs = decode_cu_qp_delta_abs(cabac, contexts) as i32;
            let signed = if abs != 0 {
                let sign = decode_cu_qp_delta_sign_flag(cabac);
                if sign != 0 { -abs } else { abs }
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

        // Effective per-CU QP for dequant.
        let qp_y = slice_qp_y + state.last_cu_qp_delta;
        state.last_qp_y = qp_y;

        // ---- Step 3: luma residual_coding + IDCT + reconstruction.
        if cbf_luma {
            let scan_idx = pick_scan_order(log2_trafo_size, state.last_luma_pred_mode);
            let block = decode_residual_coding(
                cabac,
                contexts,
                sps,
                pps,
                log2_trafo_size,
                ResidualPlane::Luma,
                qp_y,
                scan_idx,
                true, // is_intra (I-slice path)
            )?;
            apply_residual_to_luma(state, x0, y0, log2_trafo_size, &block);
            state.last_luma_residual = Some(block);
        }

        // ---- Step 4: chroma intra prediction + (optional) residual.
        if do_chroma_inline {
            let chroma_mode = state.last_chroma_pred_mode;
            predict_intra_chroma(state, sps, x0, y0, log2_trafo_size - 1, chroma_mode)?;
            if inherited.cbf_cb || inherited.cbf_cr {
                return Err(DecodeError::Unsupported(
                    "chroma residual_coding not yet implemented",
                ));
            }
        } else if do_chroma_deferred {
            // For 4:2:0 with 4x4 luma TUs, chroma prediction happens at blk_idx==3
            // using the parent TU coordinates (xBase, yBase) at log2_trafo_size.
            let chroma_mode = state.last_chroma_pred_mode;
            predict_intra_chroma(state, sps, x_base, y_base, log2_trafo_size, chroma_mode)?;
            if inherited.cbf_cb || inherited.cbf_cr {
                return Err(DecodeError::Unsupported(
                    "chroma residual_coding not yet implemented",
                ));
            }
        }
    } else if do_chroma_inline {
        // Intra CU with no CBFs at all — still need chroma prediction.
        let chroma_mode = state.last_chroma_pred_mode;
        predict_intra_chroma(state, sps, x0, y0, log2_trafo_size - 1, chroma_mode)?;
    } else if do_chroma_deferred {
        let chroma_mode = state.last_chroma_pred_mode;
        predict_intra_chroma(state, sps, x_base, y_base, log2_trafo_size, chroma_mode)?;
    }

    // ---- Step 5: deblocking bookkeeping (Phase 3b-1).
    //
    // For the I-slice intra path, every internal TU edge gets bS = 2.
    // We mark the top and left edges of this TU on the per-4×4 BS grids.
    // The picture's outer borders (x0 == 0, y0 == 0) are skipped because
    // there's nothing to filter against. We also write the per-min-CB QP
    // so the deblock pass can read the right tc/beta indices.
    let qp_y = if cbf_luma || inherited.cbf_cb || inherited.cbf_cr {
        state.last_qp_y
    } else {
        // No CBFs → no cu_qp_delta this CU; the running qp from previous TUs
        // (or slice_qp_y if first TU) still applies.
        slice_qp_y + state.last_cu_qp_delta
    };
    write_qp_y_table(state, x0, y0, log2_trafo_size, qp_y);
    mark_intra_tu_boundaries(state, x0, y0, log2_trafo_size);

    Ok(new_cbf)
}

/// Write `qp_y` into the per-min-CB QP table for all min-CB positions
/// covered by the TU at `(x0, y0)` of size `1 << log2_size`. Used by
/// the deblock pass to look up tc/β.
fn write_qp_y_table(state: &mut PictureState, x0: u32, y0: u32, log2_size: u8, qp_y: i32) {
    let length = ((1u32 << log2_size) >> state.log2_min_cb_size).max(1) as usize;
    let x_cb = (x0 >> state.log2_min_cb_size) as usize;
    let y_cb = (y0 >> state.log2_min_cb_size) as usize;
    let v = qp_y.clamp(0, 51) as u8;
    for j in 0..length {
        let row = (y_cb + j) * state.min_cb_width;
        for i in 0..length {
            state.tab_qp_y[row + x_cb + i] = v;
        }
    }
}

/// Mark the top and left edges of an intra TU at `(x0, y0)` of size
/// `1 << log2_size` with boundary strength 2 in the per-4×4 BS grid.
/// Skips picture borders.
fn mark_intra_tu_boundaries(state: &mut PictureState, x0: u32, y0: u32, log2_size: u8) {
    let size = 1u32 << log2_size;
    let pic_w = state.width as usize;
    let bs_w = pic_w >> 2; // entries per row in the BS grid

    // Top edge: only mark if y0 > 0 (there's a TU above to deblock against).
    if y0 > 0 {
        let yy = (y0 >> 2) as usize;
        let xx_start = (x0 >> 2) as usize;
        let xx_end = ((x0 + size) >> 2) as usize;
        for xx in xx_start..xx_end {
            state.bs_horizontal[yy * bs_w + xx] = 2;
        }
    }
    // Left edge: only mark if x0 > 0.
    if x0 > 0 {
        let xx = (x0 >> 2) as usize;
        let yy_start = (y0 >> 2) as usize;
        let yy_end = ((y0 + size) >> 2) as usize;
        for yy in yy_start..yy_end {
            state.bs_vertical[yy * bs_w + xx] = 2;
        }
    }
}

/// Build the reference samples and call PLANAR/DC/angular for a luma TU.
/// Writes the prediction into `state.y_plane` at `(x0, y0)`.
fn predict_intra_luma(
    state: &mut PictureState,
    sps: &Sps,
    x0: u32,
    y0: u32,
    log2_size: u8,
    mode: u8,
) -> Result<(), DecodeError> {
    let size = 1usize << log2_size;
    let pic_w = state.width as usize;
    let pic_h = state.height as usize;
    let avail = compute_luma_avail(state, x0, y0, size as u32);
    let (mut top, mut left) = build_reference_samples(
        &state.y_plane,
        state.y_stride,
        pic_w,
        pic_h,
        x0 as usize,
        y0 as usize,
        log2_size,
        state.bit_depth,
        avail,
    );

    // Reference sample filtering for angular modes (not needed for PLANAR/DC).
    if (2..=34).contains(&mode) {
        filter_reference_samples(
            &mut top,
            &mut left,
            log2_size,
            mode,
            sps.strong_intra_smoothing_enabled_flag,
            0, // c_idx = 0 (luma)
            sps.chroma_format_idc,
        );
    }

    let dst_stride = state.y_stride;
    let dst_offset = (y0 as usize) * dst_stride + (x0 as usize);
    let dst = &mut state.y_plane[dst_offset..dst_offset + (size - 1) * dst_stride + size];

    match mode {
        0 => predict_planar(dst, dst_stride, &top, &left, log2_size),
        1 => predict_dc(dst, dst_stride, &top, &left, log2_size, true),
        2..=34 => predict_angular(dst, dst_stride, &top, &left, log2_size, mode, 0),
        _ => {
            return Err(DecodeError::Unsupported("invalid intra prediction mode"));
        }
    }
    Ok(())
}

/// Decide which reference-sample directions are available for a luma TU at
/// `(x0, y0)` of size `size`. For Phase 3a-1 we use a simple raster-scan
/// availability rule: a neighbor is available iff its bottom-right pixel
/// has a strictly smaller raster index than `(x0, y0)` AND lies within the
/// picture. This works for single-slice intra-only pictures with raster
/// CTU order — multi-slice / tiles / WPP will need a more elaborate check.
fn compute_luma_avail(state: &PictureState, x0: u32, y0: u32, size: u32) -> ReferenceAvailability {
    let pic_w = state.width;
    let pic_h = state.height;

    // Raster index of the current TU's top-left.
    let cur_idx = (y0 as u64) * (pic_w as u64) + (x0 as u64);

    let pixel_decoded = |x: u32, y: u32| -> bool {
        if x >= pic_w || y >= pic_h {
            return false;
        }
        ((y as u64) * (pic_w as u64) + (x as u64)) < cur_idx
    };

    // Up-left: pixel at (x0 - 1, y0 - 1)
    let up_left = x0 > 0 && y0 > 0 && pixel_decoded(x0 - 1, y0 - 1);

    // Up row exists iff the row above is decoded for x in [x0..x0+size).
    // For raster scan in a single slice, that's true iff y0 > 0.
    let up = y0 > 0 && pixel_decoded(x0, y0 - 1);

    // Up-right: pixels at (x0 + size .. x0 + 2*size, y0 - 1).
    // Strict: any of them must be decoded. Simplification: require the
    // FIRST one to be decoded (raster scan means later columns weren't yet
    // decoded at the same y).
    let up_right = y0 > 0 && pixel_decoded(x0 + size, y0 - 1);

    // Left column exists iff the column to the left is decoded.
    let left = x0 > 0 && pixel_decoded(x0 - 1, y0);

    // Bottom-left: pixels at (x0 - 1, y0 + size .. y0 + 2*size). For raster
    // scan these are NEVER decoded yet (they're in a row strictly below us).
    // Be conservative and report unavailable.
    let _ = pixel_decoded; // silence unused warning if we add more
    let bottom_left = false;

    ReferenceAvailability {
        up_left,
        up,
        up_right,
        left,
        bottom_left,
    }
}

/// Same as `predict_intra_luma` but for one chroma plane (Cb and Cr both
/// use the same logic — different planes, same prediction). The chroma
/// position `(x0, y0)` here is in **luma sample coordinates**; we right-shift
/// by `hshift = vshift = 1` for 4:2:0.
fn predict_intra_chroma(
    state: &mut PictureState,
    sps: &Sps,
    x0_luma: u32,
    y0_luma: u32,
    log2_size: u8,
    mode: u8,
) -> Result<(), DecodeError> {
    let size = 1usize << log2_size;
    let pic_w_c = (state.width / 2) as usize;
    let pic_h_c = (state.height / 2) as usize;
    let x_c = (x0_luma >> 1) as usize;
    let y_c = (y0_luma >> 1) as usize;
    let avail = compute_chroma_avail(state, x0_luma, y0_luma, (size as u32) * 2);
    let dst_stride = state.uv_stride;

    for plane_idx in 0..2 {
        let c_idx = (plane_idx + 1) as u8; // 1 = Cb, 2 = Cr
        let (mut top, mut left) = {
            let src_plane = if plane_idx == 0 {
                &state.u_plane
            } else {
                &state.v_plane
            };
            build_reference_samples(
                src_plane,
                state.uv_stride,
                pic_w_c,
                pic_h_c,
                x_c,
                y_c,
                log2_size,
                state.bit_depth,
                avail,
            )
        };

        // Reference sample filtering for angular chroma modes.
        if (2..=34).contains(&mode) {
            filter_reference_samples(
                &mut top,
                &mut left,
                log2_size,
                mode,
                sps.strong_intra_smoothing_enabled_flag,
                c_idx,
                sps.chroma_format_idc,
            );
        }

        let plane = if plane_idx == 0 {
            &mut state.u_plane
        } else {
            &mut state.v_plane
        };
        let dst_offset = y_c * dst_stride + x_c;
        let dst = &mut plane[dst_offset..dst_offset + (size - 1) * dst_stride + size];
        match mode {
            0 => predict_planar(dst, dst_stride, &top, &left, log2_size),
            1 => predict_dc(dst, dst_stride, &top, &left, log2_size, false),
            2..=34 => predict_angular(dst, dst_stride, &top, &left, log2_size, mode, c_idx),
            _ => {
                return Err(DecodeError::Unsupported("invalid intra prediction mode"));
            }
        }
    }
    Ok(())
}

/// Chroma availability mirrors luma availability — derived from the
/// luma-coordinate position. For 4:2:0 the chroma TU's neighbors are
/// available iff the corresponding luma neighbors were decoded.
fn compute_chroma_avail(
    state: &PictureState,
    x0_luma: u32,
    y0_luma: u32,
    luma_size: u32,
) -> ReferenceAvailability {
    compute_luma_avail(state, x0_luma, y0_luma, luma_size)
}

/// Apply the inverse transform to a luma residual block and add it to the
/// already-predicted luma plane at `(x0, y0)`, with clipping.
///
/// HEVC uses the **4×4 DST** (`transform_4x4_luma`) for intra luma 4×4 TUs;
/// every other size and chroma uses the regular DCT. Since this function is
/// only called from the I-slice intra path, `pred_mode == INTRA` is always
/// true here.
fn apply_residual_to_luma(
    state: &mut PictureState,
    x0: u32,
    y0: u32,
    log2_size: u8,
    block: &ResidualBlock,
) {
    let size = 1usize << log2_size;
    let mut residual_pixels = block.coeffs.clone();
    let is_luma_intra_4x4 = log2_size == 2;
    apply_inverse_transform(
        &mut residual_pixels,
        log2_size,
        block.last_sig_x,
        block.last_sig_y,
        state.bit_depth as u32,
        is_luma_intra_4x4,
    );
    let dst_stride = state.y_stride;
    let dst_offset = (y0 as usize) * dst_stride + (x0 as usize);
    let dst = &mut state.y_plane[dst_offset..dst_offset + (size - 1) * dst_stride + size];
    add_residual(dst, dst_stride, &residual_pixels, log2_size);
}

/// Pick `scan_idx` for residual_coding (spec 7.4.9.11). For intra TUs at
/// 4×4 / 8×8, certain mode ranges select horizontal or vertical scans. All
/// other cases use diagonal.
fn pick_scan_order(log2_trafo_size: u8, intra_pred_mode: u8) -> ScanOrder {
    if log2_trafo_size > 3 {
        return ScanOrder::Diag;
    }
    // log2_trafo_size in {2, 3}: intra TU. Spec ranges per intra_pred_mode.
    if (6..=14).contains(&intra_pred_mode) {
        ScanOrder::Vert
    } else if (22..=30).contains(&intra_pred_mode) {
        ScanOrder::Horiz
    } else {
        ScanOrder::Diag
    }
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
    use crate::nal::{NalUnitType, parse_annex_b};
    use crate::pps::parse_pps;
    use crate::slice::{SliceType, parse_slice_segment_header};
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
        assert_eq!(
            cabac_byte_offset, 2,
            "fixture slice header is exactly 2 bytes"
        );
        let mut cabac = CabacReader::new(&slice_nal.rbsp, cabac_byte_offset);

        let mut state = PictureState::new(&sps);
        // The single CTU is at (0, 0) with log2_cb_size = ctb_log2_size_y = 4.
        decode_coding_quadtree(
            &mut cabac,
            &mut contexts,
            &mut state,
            &sps,
            &pps,
            sh.slice_qp_y,
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
            state.last_luma_pred_mode, INTRA_PLANAR,
            "expected PLANAR luma intra for flat-gray fixture, got {}",
            state.last_luma_pred_mode
        );
        assert_eq!(
            state.last_chroma_pred_mode, INTRA_PLANAR,
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

        // Effective per-CU QP for dequant.
        assert_eq!(state.last_qp_y, 20);

        // Phase 2c-3: residual_coding decoded.
        let resid = state
            .last_luma_residual
            .as_ref()
            .expect("luma residual block must be present when cbf_luma=1");
        assert_eq!(resid.log2_size, 4, "16x16 luma TU");

        // For our flat fixture the residual is uniform -2 per pixel, which
        // forward DCT concentrates entirely in the DC coefficient. We
        // expect a single non-zero coefficient at (0, 0).
        assert_eq!(resid.last_sig_x, 0, "last_sig_x");
        assert_eq!(resid.last_sig_y, 0, "last_sig_y");
        let nonzero = resid.coeffs.iter().filter(|&&c| c != 0).count();
        assert_eq!(nonzero, 1, "expected single DC coefficient");

        // Hand-derived expected dequant value:
        //   level   = -5 (x265's actual encoded value for this fixture)
        //   scale   = level_scale[20%6=2] << (20/6=3) = 51 << 3 = 408
        //   shift   = bit_depth + log2_trafo_size - 5 = 8 + 4 - 5 = 7
        //   add     = 1 << 6 = 64
        //   scale_m = 16  (no scaling list)
        //   dequant = (-5 * 408 * 16 + 64) >> 7 = -32576 >> 7 = -255
        // Then idct_dc:
        //   shift = 14 - 8 = 6, add = 32
        //   ((-255 + 1) >> 1 + 32) >> 6 = (-127 + 32) >> 6 = -95 >> 6 = -2 ✓
        assert_eq!(resid.coeffs[0], -255, "dequantized DC coefficient");

        // The CABAC stream should now be at the end of slice. The terminate
        // bin returns 1 when we're done.
        assert_eq!(
            cabac.decode_terminate(),
            1,
            "CABAC must be at end of slice after residual_coding"
        );

        // Phase 2c-4: apply inverse transform. For this DC-only 16x16 block
        // the result should be -2 at every pixel (matching the encoder's
        // residual = ref_yuv 0x7E - prediction 0x80).
        let mut residual_pixels = state.last_luma_residual.as_ref().unwrap().coeffs.clone();
        crate::inverse_transform::apply_inverse_transform(
            &mut residual_pixels,
            4,
            resid.last_sig_x,
            resid.last_sig_y,
            8,
            false,
        );
        assert!(
            residual_pixels.iter().all(|&p| p == -2),
            "expected all-(-2) residual after IDCT, got: first 4 = {:?}",
            &residual_pixels[..4]
        );
    }

    /// PCM bit reader: 8-bit byte-aligned reads return the raw bytes.
    #[test]
    fn test_pcm_bit_reader_byte_aligned() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut r = PcmBitReader::new(&data);
        assert_eq!(r.read_bits(8), 0x12);
        assert_eq!(r.read_bits(8), 0x34);
        assert_eq!(r.read_bits(8), 0x56);
        assert_eq!(r.read_bits(8), 0x78);
    }

    /// PCM bit reader: sub-byte reads pack MSB-first.
    #[test]
    fn test_pcm_bit_reader_bit_packed() {
        // 0b1010_1100 0b0011_1001 = read four 4-bit samples: A, C, 3, 9
        let data = [0xAC, 0x39];
        let mut r = PcmBitReader::new(&data);
        assert_eq!(r.read_bits(4), 0xA);
        assert_eq!(r.read_bits(4), 0xC);
        assert_eq!(r.read_bits(4), 0x3);
        assert_eq!(r.read_bits(4), 0x9);
    }

    /// `PcmBitReader` on a non-trivial alignment: read a 5-bit sample then
    /// a 3-bit sample, spanning the first byte's boundary.
    #[test]
    fn test_pcm_bit_reader_unaligned() {
        // 0b1_0110_101 | 0b_1011_0001 ...
        // First read 5 bits (MSB first): 0b10110 = 0x16
        // Then 3 bits: 0b101 = 0x5
        let data = [0b1011_0101, 0b1011_0001];
        let mut r = PcmBitReader::new(&data);
        assert_eq!(r.read_bits(5), 0b10110);
        assert_eq!(r.read_bits(3), 0b101);
        assert_eq!(r.read_bits(8), 0b1011_0001);
    }
}
