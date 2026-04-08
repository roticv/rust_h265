//! HEVC residual coding decode (spec 7.3.8.11 → `residual_coding`).
//!
//! Phase 2c-3 scope: decode the CABAC `residual_coding` syntax for an
//! intra-only TU and produce a buffer of dequantized coefficients (no
//! inverse transform yet — that's Phase 2c-4).
//!
//! Mirrors FFmpeg `libavcodec/hevc/cabac.c` `ff_hevc_hls_residual_coding`
//! line by line for the supported subset:
//!   - 8-bit luma/chroma 4:2:0
//!   - No `transform_skip_flag` (PPS rejects it)
//!   - No `cu_transquant_bypass_flag`
//!   - No `scaling_list_enabled_flag` (SPS rejects it)
//!   - No `persistent_rice_adaptation_enabled` (range extension)
//!   - No `sign_data_hiding` (slice header / x265 `--no-signhide`)
//!   - No `explicit_rdpcm` (range extension)
//!
//! Anything outside that subset returns `Unsupported`.

use crate::cabac::{CabacContexts, CabacReader};
use crate::cabac_tables::ctx;
use crate::error::DecodeError;
use crate::pps::Pps;
use crate::sps::Sps;

/// Component index passed into `decode_residual_coding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualPlane {
    Luma,
    Cb,
    Cr,
}

impl ResidualPlane {
    fn c_idx(self) -> usize {
        match self {
            ResidualPlane::Luma => 0,
            ResidualPlane::Cb => 1,
            ResidualPlane::Cr => 2,
        }
    }
}

/// Coefficient scan order (HEVC spec 7.3.8.11). For non-`SCAN_DIAG` cases,
/// only certain block sizes are allowed (intra 4×4 / 8×8 luma).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOrder {
    Diag,
    Horiz,
    Vert,
}

// ---- Scan tables (HEVC spec 6.5, FFmpeg `data.c`) ----------------------------

/// 4×4 diagonal scan order: x coordinates.
#[rustfmt::skip]
const DIAG_SCAN_4X4_X: [u8; 16] = [
    0, 0, 1, 0,
    1, 2, 0, 1,
    2, 3, 1, 2,
    3, 2, 3, 3,
];

/// 4×4 diagonal scan order: y coordinates.
#[rustfmt::skip]
const DIAG_SCAN_4X4_Y: [u8; 16] = [
    0, 1, 0, 2,
    1, 0, 3, 2,
    1, 0, 3, 2,
    1, 3, 2, 3,
];

/// 4×4 diagonal scan inverse: `(y, x) → scan_pos`.
#[rustfmt::skip]
const DIAG_SCAN_4X4_INV: [[u8; 4]; 4] = [
    [ 0,  2,  5,  9],
    [ 1,  4,  8, 12],
    [ 3,  7, 11, 14],
    [ 6, 10, 13, 15],
];

/// 2×2 diagonal scan order: x coordinates.
const DIAG_SCAN_2X2_X: [u8; 4] = [0, 0, 1, 1];
/// 2×2 diagonal scan order: y coordinates.
const DIAG_SCAN_2X2_Y: [u8; 4] = [0, 1, 0, 1];
/// 2×2 diagonal scan inverse: `(y, x) → scan_pos`.
const DIAG_SCAN_2X2_INV: [[u8; 2]; 2] = [[0, 2], [1, 3]];

/// 8×8 diagonal scan inverse: `(y, x) → scan_pos`.
#[rustfmt::skip]
const DIAG_SCAN_8X8_INV: [[u8; 8]; 8] = [
    [ 0,  2,  5,  9, 14, 20, 27, 35],
    [ 1,  4,  8, 13, 19, 26, 34, 42],
    [ 3,  7, 12, 18, 25, 33, 41, 48],
    [ 6, 11, 17, 24, 32, 40, 47, 53],
    [10, 16, 23, 31, 39, 46, 52, 57],
    [15, 22, 30, 38, 45, 51, 56, 60],
    [21, 29, 37, 44, 50, 55, 59, 62],
    [28, 36, 43, 49, 54, 58, 61, 63],
];

/// 8×8 diagonal scan: x coordinates (used as the sub-block scan for 32×32).
#[rustfmt::skip]
const DIAG_SCAN_8X8_X: [u8; 64] = [
    0, 0, 1, 0,
    1, 2, 0, 1,
    2, 3, 0, 1,
    2, 3, 4, 0,
    1, 2, 3, 4,
    5, 0, 1, 2,
    3, 4, 5, 6,
    0, 1, 2, 3,
    4, 5, 6, 7,
    1, 2, 3, 4,
    5, 6, 7, 2,
    3, 4, 5, 6,
    7, 3, 4, 5,
    6, 7, 4, 5,
    6, 7, 5, 6,
    7, 6, 7, 7,
];

/// 8×8 diagonal scan: y coordinates.
#[rustfmt::skip]
const DIAG_SCAN_8X8_Y: [u8; 64] = [
    0, 1, 0, 2,
    1, 0, 3, 2,
    1, 0, 4, 3,
    2, 1, 0, 5,
    4, 3, 2, 1,
    0, 6, 5, 4,
    3, 2, 1, 0,
    7, 6, 5, 4,
    3, 2, 1, 0,
    7, 6, 5, 4,
    3, 2, 1, 7,
    6, 5, 4, 3,
    2, 7, 6, 5,
    4, 3, 7, 6,
    5, 4, 7, 6,
    5, 7, 6, 7,
];

/// `level_scale[i]` from HEVC spec 8.6.3 (the dequant per-rem6 multiplier).
const LEVEL_SCALE: [u32; 6] = [40, 45, 51, 57, 64, 72];

/// Maximum bin count for `coeff_abs_level_remaining` Exp-Golomb prefix.
const CABAC_MAX_BIN: u32 = 31;

// ---- Syntax-element decoders (mirror FFmpeg) ---------------------------------

/// Decode `last_significant_coeff_x_prefix` and `last_significant_coeff_y_prefix`
/// (HEVC spec 9.3.4.2.4 / FFmpeg `last_significant_coeff_xy_prefix_decode`).
fn decode_last_significant_coeff_xy_prefix(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    c_idx: usize,
    log2_size: u8,
) -> (u32, u32) {
    let max = (log2_size as u32 * 2) - 1;
    let (ctx_offset, ctx_shift) = if c_idx == 0 {
        let off = 3 * (log2_size as u32 - 2) + ((log2_size as u32 - 1) >> 2);
        let shift = (log2_size as u32 + 1) >> 2;
        (off, shift)
    } else {
        (15, log2_size as u32 - 2)
    };

    let mut x_prefix = 0u32;
    while x_prefix < max {
        let inc = (x_prefix >> ctx_shift) + ctx_offset;
        let bit = cabac.decode_bin(
            &mut contexts.state[ctx::LAST_SIGNIFICANT_COEFF_X_PREFIX + inc as usize],
        );
        if bit == 0 {
            break;
        }
        x_prefix += 1;
    }

    let mut y_prefix = 0u32;
    while y_prefix < max {
        let inc = (y_prefix >> ctx_shift) + ctx_offset;
        let bit = cabac.decode_bin(
            &mut contexts.state[ctx::LAST_SIGNIFICANT_COEFF_Y_PREFIX + inc as usize],
        );
        if bit == 0 {
            break;
        }
        y_prefix += 1;
    }

    (x_prefix, y_prefix)
}

/// Decode `last_significant_coeff_*_suffix` (FLC bypass — `(prefix>>1)-1` bits).
fn decode_last_significant_coeff_suffix(cabac: &mut CabacReader, prefix: u32) -> u32 {
    let length = (prefix >> 1) - 1;
    let mut value = cabac.decode_bypass();
    for _ in 1..length {
        value = (value << 1) | cabac.decode_bypass();
    }
    value
}

/// Compute `last_significant_coeff_x` (or y) from prefix + suffix
/// (HEVC spec 7.4.9.11 eq. 7-66 / 7-67).
fn last_significant_coeff_value(prefix: u32, cabac: &mut CabacReader) -> u32 {
    if prefix > 3 {
        let suffix = decode_last_significant_coeff_suffix(cabac, prefix);
        (1 << ((prefix >> 1) - 1)) * (2 + (prefix & 1)) + suffix
    } else {
        prefix
    }
}

/// Decode `coded_sub_block_flag` (HEVC spec 9.3.4.2.6).
fn decode_coded_sub_block_flag(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    c_idx: usize,
    ctx_cg: u32,
) -> u32 {
    // FFmpeg: inc = min(ctx_cg, 1) + (c_idx > 0 ? 2 : 0)
    let inc = ctx_cg.min(1) as usize + if c_idx > 0 { 2 } else { 0 };
    cabac.decode_bin(&mut contexts.state[ctx::SIGNIFICANT_COEFF_GROUP_FLAG + inc])
}

/// Decode `sig_coeff_flag` for `(x_c, y_c)` within the sub-block, using the
/// pre-computed `ctx_idx_map` and `scf_offset` (HEVC spec 9.3.4.2.5).
fn decode_sig_coeff_flag(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    x_c: u8,
    y_c: u8,
    scf_offset: usize,
    ctx_idx_map: &[u8; 16],
) -> u32 {
    let inc = ctx_idx_map[((y_c as usize) << 2) + x_c as usize] as usize + scf_offset;
    cabac.decode_bin(&mut contexts.state[ctx::SIGNIFICANT_COEFF_FLAG + inc])
}

/// Decode `sig_coeff_flag` at the (0, 0) position of a sub-block — uses just
/// the per-component DC context, no map.
fn decode_sig_coeff_flag_dc(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    scf_offset: usize,
) -> u32 {
    cabac.decode_bin(&mut contexts.state[ctx::SIGNIFICANT_COEFF_FLAG + scf_offset])
}

/// Decode `coeff_abs_level_greater1_flag`.
fn decode_coeff_abs_level_greater1_flag(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    c_idx: usize,
    inc: u32,
) -> u32 {
    let inc = inc as usize + if c_idx > 0 { 16 } else { 0 };
    cabac.decode_bin(&mut contexts.state[ctx::COEFF_ABS_LEVEL_GREATER1_FLAG + inc])
}

/// Decode `coeff_abs_level_greater2_flag`.
fn decode_coeff_abs_level_greater2_flag(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    c_idx: usize,
    inc: u32,
) -> u32 {
    let inc = inc as usize + if c_idx > 0 { 4 } else { 0 };
    cabac.decode_bin(&mut contexts.state[ctx::COEFF_ABS_LEVEL_GREATER2_FLAG + inc])
}

/// Decode `coeff_abs_level_remaining` (Rice + Exp-Golomb), HEVC spec 9.3.4.2.8.
fn decode_coeff_abs_level_remaining(cabac: &mut CabacReader, c_rice_param: u32) -> u32 {
    let mut prefix = 0u32;
    while prefix < CABAC_MAX_BIN && cabac.decode_bypass() != 0 {
        prefix += 1;
    }
    if prefix < 3 {
        let mut suffix = 0u32;
        for _ in 0..c_rice_param {
            suffix = (suffix << 1) | cabac.decode_bypass();
        }
        (prefix << c_rice_param) + suffix
    } else {
        let prefix_minus3 = prefix - 3;
        let mut suffix = 0u32;
        for _ in 0..(prefix_minus3 + c_rice_param) {
            suffix = (suffix << 1) | cabac.decode_bypass();
        }
        (((1u32 << prefix_minus3) + 3 - 1) << c_rice_param) + suffix
    }
}

/// Decode `nb` bypass-coded `coeff_sign_flag` bits.
fn decode_coeff_sign_flag(cabac: &mut CabacReader, nb: u8) -> u32 {
    let mut ret = 0u32;
    for _ in 0..nb {
        ret = (ret << 1) | cabac.decode_bypass();
    }
    ret
}

// ---- Significance flag context map (HEVC spec table 9-19) -------------------

/// `ctx_idx_map` lookup table from FFmpeg's residual_coding implementation.
/// Indexed by `(prev_sig_group_pattern, y_c, x_c)`.
#[rustfmt::skip]
const SIG_CTX_IDX_MAP: [u8; 5 * 16] = [
    // log2_trafo_size == 2 (4×4 TU)
    0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8,
    // prev_sig_group_pattern == 0 (no neighbors)
    1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    // prev_sig_group_pattern == 1 (right neighbor)
    2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    // prev_sig_group_pattern == 2 (bottom neighbor)
    2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0,
    // default (prev_sig_group_pattern == 3)
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

// ---- Main residual_coding entry point ---------------------------------------

/// Result of decoding a single TU's residual_coding: dequantized 16-bit
/// coefficients (in raster order), the size of the block, and the position
/// of the last significant coefficient (used by Phase 2c-4 to limit IDCT
/// scan).
pub struct ResidualBlock {
    pub coeffs: Vec<i16>,
    pub log2_size: u8,
    pub last_sig_x: u32,
    pub last_sig_y: u32,
}

/// Compute the dequantization scale parameters for a TU.
///
/// Returns `(shift, add, scale)` for the formula
/// `dequant = (level * scale * scale_m + add) >> shift`. `scale_m` is 16
/// (no scaling lists) for our subset.
fn compute_dequant_scale(qp: i32, log2_trafo_size: u8, bit_depth: u8) -> (u32, u32, u32) {
    let shift = (bit_depth as u32 + log2_trafo_size as u32) - 5;
    let add = 1u32 << (shift - 1);
    let qp = qp as usize;
    let scale = LEVEL_SCALE[qp % 6] << (qp / 6);
    (shift, add, scale)
}

/// Map a luma QP to the chroma QP via spec table 8-9 (chroma_format_idc=1).
#[allow(dead_code)]
fn chroma_qp(qp_y: i32, offset: i32) -> i32 {
    let qp_i = (qp_y + offset).clamp(0, 57);
    if qp_i < 30 {
        qp_i
    } else if qp_i > 43 {
        qp_i - 6
    } else {
        const QP_C: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
        QP_C[(qp_i - 30) as usize]
    }
}

/// Decode `residual_coding` for a single TU plane (HEVC spec 7.3.8.11).
///
/// Returns the 1D coefficient buffer in raster order along with the last
/// significant coefficient position. Coefficients are 16-bit signed
/// dequantized values; the inverse transform happens in Phase 2c-4.
#[allow(clippy::too_many_arguments)]
pub fn decode_residual_coding(
    cabac: &mut CabacReader,
    contexts: &mut CabacContexts,
    sps: &Sps,
    pps: &Pps,
    log2_trafo_size: u8,
    plane: ResidualPlane,
    qp: i32,
    scan_idx: ScanOrder,
) -> Result<ResidualBlock, DecodeError> {
    if pps.transform_skip_enabled_flag {
        return Err(DecodeError::Unsupported(
            "transform_skip_flag in residual_coding not supported",
        ));
    }
    if sps.scaling_list_enabled_flag {
        return Err(DecodeError::Unsupported(
            "scaling_list_enabled in residual_coding not supported",
        ));
    }
    if pps.sign_data_hiding_enabled_flag {
        return Err(DecodeError::Unsupported(
            "sign_data_hiding in residual_coding not supported",
        ));
    }
    if scan_idx != ScanOrder::Diag {
        return Err(DecodeError::Unsupported(
            "non-diagonal scan in residual_coding not yet supported",
        ));
    }

    let trafo_size = 1usize << log2_trafo_size;
    let c_idx = plane.c_idx();
    let mut coeffs = vec![0i16; trafo_size * trafo_size];

    // ---- Last significant coefficient position ----
    let (last_x_prefix, last_y_prefix) =
        decode_last_significant_coeff_xy_prefix(cabac, contexts, c_idx, log2_trafo_size);
    let mut last_sig_x = last_significant_coeff_value(last_x_prefix, cabac);
    let mut last_sig_y = last_significant_coeff_value(last_y_prefix, cabac);
    if scan_idx == ScanOrder::Vert {
        std::mem::swap(&mut last_sig_x, &mut last_sig_y);
    }

    // ---- Sub-block (CG) selection ----
    let x_cg_last = (last_sig_x >> 2) as usize;
    let y_cg_last = (last_sig_y >> 2) as usize;

    // Pick scan tables and compute the linear scan position of the last sig.
    let (scan_x_off, scan_y_off, scan_x_cg, scan_y_cg);
    let mut num_coeff: u32;

    {
        let last_x_c = (last_sig_x & 3) as usize;
        let last_y_c = (last_sig_y & 3) as usize;
        scan_x_off = &DIAG_SCAN_4X4_X[..];
        scan_y_off = &DIAG_SCAN_4X4_Y[..];
        num_coeff = DIAG_SCAN_4X4_INV[last_y_c][last_x_c] as u32;
        match trafo_size {
            4 => {
                // 1×1 sub-block scan
                scan_x_cg = &SCAN_1X1[..];
                scan_y_cg = &SCAN_1X1[..];
            }
            8 => {
                num_coeff += (DIAG_SCAN_2X2_INV[y_cg_last][x_cg_last] as u32) << 4;
                scan_x_cg = &DIAG_SCAN_2X2_X[..];
                scan_y_cg = &DIAG_SCAN_2X2_Y[..];
            }
            16 => {
                num_coeff += (DIAG_SCAN_4X4_INV[y_cg_last][x_cg_last] as u32) << 4;
                scan_x_cg = &DIAG_SCAN_4X4_X[..];
                scan_y_cg = &DIAG_SCAN_4X4_Y[..];
            }
            32 => {
                num_coeff += (DIAG_SCAN_8X8_INV[y_cg_last][x_cg_last] as u32) << 4;
                scan_x_cg = &DIAG_SCAN_8X8_X[..];
                scan_y_cg = &DIAG_SCAN_8X8_Y[..];
            }
            _ => {
                return Err(DecodeError::InvalidSyntax("invalid trafo_size"));
            }
        }
    }
    num_coeff += 1;
    let num_last_subset = ((num_coeff - 1) >> 4) as usize;

    let (shift, add, scale) = compute_dequant_scale(qp, log2_trafo_size, sps.bit_depth_luma);
    let scale_m = 16u32; // no scaling list
    let _ = scale_m; // (kept symbolic for now; spec multiplies it through)

    // 8×8 grid of CG flags. The largest TU is 32×32 → 8×8 sub-blocks.
    let mut significant_coeff_group_flag = [[false; 8]; 8];
    let mut greater1_ctx: u32 = 1;

    // ---- Reverse scan from last sub-block to (0, 0) ----
    for i in (0..=num_last_subset).rev() {
        let offset = i << 4;
        let x_cg = scan_x_cg[i] as usize;
        let y_cg = scan_y_cg[i] as usize;
        let mut implicit_non_zero_coeff = false;

        if i < num_last_subset && i > 0 {
            // Decode coded_sub_block_flag with right/below neighbor context.
            let mut ctx_cg = 0u32;
            if x_cg < (1 << (log2_trafo_size - 2)) - 1
                && significant_coeff_group_flag[y_cg][x_cg + 1]
            {
                ctx_cg += 1;
            }
            if y_cg < (1 << (log2_trafo_size - 2)) - 1
                && significant_coeff_group_flag[y_cg + 1][x_cg]
            {
                ctx_cg += 1;
            }
            significant_coeff_group_flag[y_cg][x_cg] =
                decode_coded_sub_block_flag(cabac, contexts, c_idx, ctx_cg) != 0;
            implicit_non_zero_coeff = true;
        } else {
            // First and last subsets are implicitly non-zero (last because it
            // contains the last significant coeff, first because the DC
            // sub-block always exists if there are any non-zero coefficients
            // in the TU).
            significant_coeff_group_flag[y_cg][x_cg] =
                (x_cg == x_cg_last && y_cg == y_cg_last) || (x_cg == 0 && y_cg == 0);
        }

        // Compute prev_sig pattern (right + below neighbor sub-blocks).
        let mut prev_sig = 0u32;
        if x_cg < (((1 << log2_trafo_size) - 1) >> 2)
            && significant_coeff_group_flag[y_cg][x_cg + 1]
        {
            prev_sig |= 1;
        }
        if y_cg < (((1 << log2_trafo_size) - 1) >> 2)
            && significant_coeff_group_flag[y_cg + 1][x_cg]
        {
            prev_sig |= 2;
        }

        let last_scan_pos = (num_coeff as i32) - (offset as i32) - 1;
        let mut significant_coeff_flag_idx: [u8; 16] = [0; 16];
        let mut nb_significant_coeff_flag = 0u8;
        let n_end_initial: i32 = if i == num_last_subset {
            // The last significant coefficient is implicitly significant.
            significant_coeff_flag_idx[0] = last_scan_pos as u8;
            nb_significant_coeff_flag = 1;
            last_scan_pos - 1
        } else {
            15
        };

        if significant_coeff_group_flag[y_cg][x_cg] && n_end_initial >= 0 {
            // ---- Compute scf_offset and ctx_idx_map_p ----
            let ctx_idx_map_p: &[u8; 16];
            let mut scf_offset: usize = 0;
            if c_idx != 0 {
                scf_offset = 27;
            }
            if log2_trafo_size == 2 {
                ctx_idx_map_p = (&SIG_CTX_IDX_MAP[0..16]).try_into().unwrap();
            } else {
                let map_idx = ((prev_sig + 1) << 4) as usize;
                ctx_idx_map_p = (&SIG_CTX_IDX_MAP[map_idx..map_idx + 16])
                    .try_into()
                    .unwrap();
                if c_idx == 0 {
                    if x_cg > 0 || y_cg > 0 {
                        scf_offset += 3;
                    }
                    if log2_trafo_size == 3 {
                        scf_offset += if scan_idx == ScanOrder::Diag { 9 } else { 15 };
                    } else {
                        scf_offset += 21;
                    }
                } else if log2_trafo_size == 3 {
                    scf_offset += 9;
                } else {
                    scf_offset += 12;
                }
            }

            // Iterate scan positions n_end_initial..=1, decoding sig_coeff_flag.
            let mut n = n_end_initial;
            while n > 0 {
                let x_c = scan_x_off[n as usize];
                let y_c = scan_y_off[n as usize];
                if decode_sig_coeff_flag(cabac, contexts, x_c, y_c, scf_offset, ctx_idx_map_p)
                    != 0
                {
                    significant_coeff_flag_idx[nb_significant_coeff_flag as usize] = n as u8;
                    nb_significant_coeff_flag += 1;
                    implicit_non_zero_coeff = false;
                }
                n -= 1;
            }

            // Position 0 (DC of the sub-block).
            if !implicit_non_zero_coeff {
                // Re-derive scf_offset for position 0.
                let scf_offset_0 = if i == 0 {
                    if c_idx == 0 {
                        0
                    } else {
                        27
                    }
                } else {
                    2 + scf_offset
                };
                if decode_sig_coeff_flag_dc(cabac, contexts, scf_offset_0) != 0 {
                    significant_coeff_flag_idx[nb_significant_coeff_flag as usize] = 0;
                    nb_significant_coeff_flag += 1;
                }
            } else {
                significant_coeff_flag_idx[nb_significant_coeff_flag as usize] = 0;
                nb_significant_coeff_flag += 1;
            }
        }

        let n_end = nb_significant_coeff_flag;
        if n_end == 0 {
            continue;
        }

        // ---- Coefficient levels ----
        let mut ctx_set = if i > 0 && c_idx == 0 { 2u32 } else { 0 };
        if i != num_last_subset && greater1_ctx == 0 {
            ctx_set += 1;
        }
        greater1_ctx = 1;

        let mut coeff_abs_level_greater1_flag = [0u8; 8];
        let mut first_greater1_idx: i32 = -1;
        let g1_count = (n_end as usize).min(8);
        for (m, slot) in coeff_abs_level_greater1_flag
            .iter_mut()
            .enumerate()
            .take(g1_count)
        {
            let inc = (ctx_set << 2) + greater1_ctx;
            let bit = decode_coeff_abs_level_greater1_flag(cabac, contexts, c_idx, inc);
            *slot = bit as u8;
            if bit != 0 {
                greater1_ctx = 0;
                if first_greater1_idx == -1 {
                    first_greater1_idx = m as i32;
                }
            } else if greater1_ctx > 0 && greater1_ctx < 3 {
                greater1_ctx += 1;
            }
        }

        // greater2 only applies to the first level > 1.
        if first_greater1_idx != -1 {
            let greater2 =
                decode_coeff_abs_level_greater2_flag(cabac, contexts, c_idx, ctx_set) as u8;
            coeff_abs_level_greater1_flag[first_greater1_idx as usize] += greater2;
        }

        // Sign flags (bypass; no sign hiding in our subset).
        let coeff_sign_flag = decode_coeff_sign_flag(cabac, n_end);
        let mut sign_bits = coeff_sign_flag << (16 - n_end);

        // Levels in reverse scan order, dequantize, and store.
        let mut c_rice_param = 0u32;
        for m in 0..n_end as usize {
            let n = significant_coeff_flag_idx[m] as usize;
            let x_c = (x_cg << 2) + scan_x_off[n] as usize;
            let y_c = (y_cg << 2) + scan_y_off[n] as usize;
            let mut trans_coeff_level: i64;

            if m < 8 {
                trans_coeff_level = 1 + coeff_abs_level_greater1_flag[m] as i64;
                let needed_for_remaining = if m as i32 == first_greater1_idx {
                    3
                } else {
                    2
                };
                if trans_coeff_level == needed_for_remaining {
                    let last = decode_coeff_abs_level_remaining(cabac, c_rice_param) as i64;
                    trans_coeff_level += last;
                    if trans_coeff_level > (3 << c_rice_param) as i64 {
                        c_rice_param = (c_rice_param + 1).min(4);
                    }
                }
            } else {
                let last = decode_coeff_abs_level_remaining(cabac, c_rice_param) as i64;
                trans_coeff_level = 1 + last;
                if trans_coeff_level > (3 << c_rice_param) as i64 {
                    c_rice_param = (c_rice_param + 1).min(4);
                }
            }

            if (sign_bits >> 15) != 0 {
                trans_coeff_level = -trans_coeff_level;
            }
            sign_bits <<= 1;

            // Dequantize: clamp to int16.
            let dq = (trans_coeff_level * scale as i64 * scale_m as i64 + add as i64) >> shift;
            let dq = dq.clamp(-32768, 32767) as i16;
            coeffs[y_c * trafo_size + x_c] = dq;
        }
    }

    Ok(ResidualBlock {
        coeffs,
        log2_size: log2_trafo_size,
        last_sig_x,
        last_sig_y,
    })
}

/// 1×1 scan order (used as the sub-block scan for 4×4 TUs).
const SCAN_1X1: [u8; 1] = [0];
