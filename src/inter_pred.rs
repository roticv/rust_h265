//! Phase 3d-6: Inter prediction — motion compensation with sub-pixel interpolation.
//!
//! Implements HEVC spec 8.5.3.2 (luma and chroma sample interpolation).
//! - Luma uses an 8-tap filter at quarter-pel resolution (4 positions).
//! - Chroma uses a 4-tap filter at eighth-pel resolution (8 positions).
//! - Boundary samples are handled by clamping reference coordinates.
//! - Bi-prediction averages L0 and L1 predictions: `(L0 + L1 + 1) >> 1`.
//!
//! Low-level MC functions (`mc_luma`, `mc_chroma`, `mc_luma_i32`, `mc_chroma_i32`,
//! `ref_sample`) are generic over `P: Pixel` for multi-bit-depth support.

use std::rc::Rc;

use crate::cu_tree::PictureState;
use crate::dpb::DecodedPicture;
use crate::pixel::Pixel;

/// HEVC 8-tap luma interpolation filter coefficients (spec table 8-12).
/// Index 0 = integer position (identity / copy), 1..3 = quarter-pel.
const QPEL_FILTER: [[i16; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// HEVC 4-tap chroma interpolation filter coefficients (spec table 8-13).
/// Index 0 = integer position (identity / copy), 1..7 = eighth-pel.
const EPEL_FILTER: [[i16; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Number of extra reference rows/columns needed on each side for luma.
const QPEL_EXTRA_BEFORE: i32 = 3;
const QPEL_EXTRA_AFTER: i32 = 4;

/// Number of extra reference rows/columns needed on each side for chroma.
const EPEL_EXTRA_BEFORE: i32 = 1;
#[allow(dead_code)]
const EPEL_EXTRA_AFTER: i32 = 2;

/// Maximum PB size (used for intermediate buffer stride).
const MAX_PB_SIZE: usize = 64;

/// Max rows in the 2D separable luma filter intermediate buffer:
/// n_pb_h (max 64) + QPEL_EXTRA_BEFORE (3) + QPEL_EXTRA_AFTER (4) = 71.
const QPEL_TMP_ROWS: usize = MAX_PB_SIZE + 3 + 4;

/// Max rows in the 2D separable chroma filter intermediate buffer:
/// n_pb_h_c (max 64 for 4:4:4) + EPEL_EXTRA_BEFORE (1) + EPEL_EXTRA_AFTER (2).
const EPEL_TMP_ROWS: usize = MAX_PB_SIZE + 1 + 2;

/// Max luma PU area (64×64).
const MAX_PB_LUMA: usize = MAX_PB_SIZE * MAX_PB_SIZE;

/// Max chroma PU area: 64×64 for 4:4:4 (32×32 would suffice for 4:2:0).
const MAX_PB_CHROMA: usize = MAX_PB_SIZE * MAX_PB_SIZE;

/// Length of the separable-filter intermediate buffer: enough rows for a
/// 64-row block plus the 8-tap luma filter margin (the 4-tap chroma margin
/// is smaller), at a fixed `MAX_PB_SIZE` stride.
pub(crate) const MC_TMP_LEN: usize = QPEL_TMP_ROWS * MAX_PB_SIZE;

/// Reusable motion compensation scratch space. `PictureState` owns one and
/// lends it to every prediction block, so no per-block buffer is
/// zero-filled. That zero-filling used to dominate the cost of small
/// blocks: a bi-predicted block cleared six 16 KB stack arrays plus a filter
/// temporary on every call. Every region is fully written before it is
/// read, so stale contents left by earlier blocks are never observed.
pub struct McScratch<P: Pixel> {
    /// L0 luma intermediate (bi-pred), or the weighted uni-pred luma.
    luma0: Vec<i32>,
    /// L1 luma intermediate (bi-pred).
    luma1: Vec<i32>,
    /// L0 Cb / Cr intermediates (bi-pred), or the weighted uni-pred chroma.
    cb0: Vec<i32>,
    cr0: Vec<i32>,
    /// L1 Cb / Cr intermediates (bi-pred).
    cb1: Vec<i32>,
    cr1: Vec<i32>,
    /// Separable 2-D filter intermediate, shared by luma and chroma.
    tmp: Vec<i32>,
    /// Padded copy of the reference window for blocks near the picture
    /// edge (see `ref_window`).
    edge: Vec<P>,
}

impl<P: Pixel> Default for McScratch<P> {
    fn default() -> Self {
        Self {
            luma0: Vec::new(),
            luma1: Vec::new(),
            cb0: Vec::new(),
            cr0: Vec::new(),
            cb1: Vec::new(),
            cr1: Vec::new(),
            tmp: Vec::new(),
            edge: Vec::new(),
        }
    }
}

impl<P: Pixel> McScratch<P> {
    /// Allocate buffers sized for the largest prediction block (64×64, and
    /// 64×64 chroma for 4:4:4). `Default` gives empty buffers instead, which
    /// is only used as the placeholder while the scratch is lent out.
    pub fn new() -> Self {
        Self {
            luma0: vec![0; MAX_PB_LUMA],
            luma1: vec![0; MAX_PB_LUMA],
            cb0: vec![0; MAX_PB_CHROMA],
            cr0: vec![0; MAX_PB_CHROMA],
            cb1: vec![0; MAX_PB_CHROMA],
            cr1: vec![0; MAX_PB_CHROMA],
            tmp: vec![0; MC_TMP_LEN],
            edge: vec![P::zero(); MC_EDGE_LEN],
        }
    }
}

/// Chroma motion vector component (spec 8.5.3.2.10): `mvLX * 2 / SubWidthC`
/// (or `SubHeightC`), in units of 1/8 chroma sample. `shift` is
/// `log2(SubWidthC)` — 1 for 4:2:0 (chroma MV equals the luma MV, now in
/// eighth-chroma-sample units) or 0 for 4:4:4 (twice the luma MV, so only
/// even eighth-sample fractions occur). Widened to i32 since `mv * 2` can
/// exceed the i16 range.
#[inline]
fn chroma_mv(mv: i16, shift: u32) -> i32 {
    (mv as i32 * 2) >> shift
}

/// Fetch a reference sample with boundary clamping (spec 8.5.3.2.1).
/// Generic over `P: Pixel` for multi-bit-depth support.
#[inline]
fn ref_sample<P: Pixel>(plane: &[P], stride: usize, x: i32, y: i32, w: i32, h: i32) -> i32 {
    if w <= 0 || h <= 0 {
        return 0;
    }
    let cx = x.clamp(0, w - 1) as usize;
    let cy = y.clamp(0, h - 1) as usize;
    let idx = cy * stride + cx;
    if idx >= plane.len() {
        return 0;
    }
    plane[idx].to_i32()
}

/// Row stride of the padded edge buffer: a 64-wide block plus the 7-sample
/// luma filter margin.
pub(crate) const MC_EDGE_STRIDE: usize = MAX_PB_SIZE + 7;
/// Length of the padded edge buffer: 71 rows of `MC_EDGE_STRIDE`.
pub(crate) const MC_EDGE_LEN: usize = MC_EDGE_STRIDE * (MAX_PB_SIZE + 7);

/// Locate the reference samples a block's filter needs: the `ww × wh`
/// window whose top-left is `(wx0, wy0)` in the reference plane.
///
/// Returns `(source, stride, origin)` such that reference sample
/// `(wx0 + dx, wy0 + dy)` is `source[origin + dy * stride + dx]`.
///
/// When the window lies entirely inside the picture (the common case) the
/// source is the reference plane itself, so the filter loops read memory
/// directly with no per-tap clamping. Otherwise the window is copied into
/// `edge` through `ref_sample`, which applies the spec 8.5.3.2.1 boundary
/// clamp, and the filters read the padded copy. Either way every tap sees
/// exactly the value `ref_sample` would return.
#[allow(clippy::too_many_arguments)]
#[inline]
fn ref_window<'a, P: Pixel>(
    ref_plane: &'a [P],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    wx0: i32,
    wy0: i32,
    ww: usize,
    wh: usize,
    edge: &'a mut [P],
    bit_depth: u8,
) -> (&'a [P], usize, usize) {
    let inside = ww > 0
        && wh > 0
        && wx0 >= 0
        && wy0 >= 0
        && wx0 as i64 + ww as i64 <= ref_w as i64
        && wy0 as i64 + wh as i64 <= ref_h as i64
        && (wy0 as usize + wh - 1) * ref_stride + wx0 as usize + ww - 1 < ref_plane.len();
    if inside {
        return (
            ref_plane,
            ref_stride,
            wy0 as usize * ref_stride + wx0 as usize,
        );
    }
    debug_assert!(ww <= MC_EDGE_STRIDE && wh * MC_EDGE_STRIDE <= edge.len());
    for r in 0..wh {
        for c in 0..ww {
            // `ref_sample` returns an in-range sample (or 0), so this
            // conversion never actually clamps.
            edge[r * MC_EDGE_STRIDE + c] = P::from_i32_clamped(
                ref_sample(
                    ref_plane,
                    ref_stride,
                    wx0 + c as i32,
                    wy0 + r as i32,
                    ref_w,
                    ref_h,
                ),
                bit_depth,
            );
        }
    }
    (edge, MC_EDGE_STRIDE, 0)
}

/// Uni-directional luma motion compensation (8-tap filter).
///
/// Writes `nPbW * nPbH` prediction samples into `dst` at stride `dst_stride`.
/// `x0, y0` are the PU's top-left luma coordinates in the current picture.
/// `mv` is the quarter-pel motion vector. `bit_depth` is the luma bit depth.
///
/// Generic over `P: Pixel` for multi-bit-depth support.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    ref_plane: &[P],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0: i32,
    y0: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    mv_x: i16,
    mv_y: i16,
    bit_depth: u8,
    tmp: &mut [i32],
    edge: &mut [P],
) {
    let x_frac = (mv_x & 3) as usize;
    let y_frac = (mv_y & 3) as usize;
    // Integer part of the reference position.
    let x_int = x0 + (mv_x >> 2) as i32;
    let y_int = y0 + (mv_y >> 2) as i32;
    // Filter window: taps reach 3 sample(s) before and 4 after on each axis
    // with a fractional MV. Interior blocks read the reference plane
    // directly; edge blocks read a padded copy (see `ref_window`).
    let (bx, ax) = if x_frac != 0 { (3, 4) } else { (0, 0) };
    let (by, ay) = if y_frac != 0 { (3, 4) } else { (0, 0) };
    let (wx0, wy0) = (x_int - bx, y_int - by);
    let (src, src_stride, org) = ref_window(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        wx0,
        wy0,
        n_pb_w + (bx + ax) as usize,
        n_pb_h + (by + ay) as usize,
        edge,
        bit_depth,
    );
    let at = |x: i32, y: i32| -> i32 {
        src[org + (y - wy0) as usize * src_stride + (x - wx0) as usize].to_i32()
    };

    let shift = crate::pixel::mc_shift(bit_depth) as i32;
    let offset = crate::pixel::mc_offset(bit_depth);
    // Sub-pel filter output shift: bit_depth - 8 (8-bit: 0, 10-bit: 2).
    // Applied to single-pass filter output before the final (offset + shift).
    // For the 2D case, applied in the horizontal pass.
    let shift1 = (bit_depth as i32 - 8).max(0);

    if x_frac == 0 && y_frac == 0 {
        // Integer-pel: direct copy.
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(at(x_int + i as i32, y_int + j as i32), bit_depth);
            }
        }
    } else if y_frac == 0 {
        // Horizontal-only interpolation.
        let filter = &QPEL_FILTER[x_frac];
        for j in 0..n_pb_h {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 3, ry);
                }
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(((val >> shift1) + offset) >> shift, bit_depth);
            }
        }
    } else if x_frac == 0 {
        // Vertical-only interpolation.
        let filter = &QPEL_FILTER[y_frac];
        for j in 0..n_pb_h {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx, ry + k as i32 - 3);
                }
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(((val >> shift1) + offset) >> shift, bit_depth);
            }
        }
    } else {
        // 2D separable interpolation: horizontal first into i32 intermediate,
        // then vertical on the intermediate.
        let h_filter = &QPEL_FILTER[x_frac];
        let v_filter = &QPEL_FILTER[y_frac];

        // Intermediate buffer needs QPEL_EXTRA_BEFORE extra rows above and
        // QPEL_EXTRA_AFTER extra rows below. Stack-allocated to avoid per-PU
        // malloc/free overhead (this was ~23% of decode time on 1080p).
        // Using i32 for 10-bit safety (8-tap on 1023-max samples can exceed i16 range).
        let tmp_h = n_pb_h + (QPEL_EXTRA_BEFORE + QPEL_EXTRA_AFTER) as usize;
        debug_assert!(tmp.len() >= MC_TMP_LEN);

        // Horizontal pass: produce (n_pb_h + 7) rows of i32 values.
        let y_start = y_int - QPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 3, ry);
                }
                tmp[j * MAX_PB_SIZE + i] = val >> shift1;
            }
        }

        // Vertical pass on the intermediate buffer.
        // Horizontal pass output is at 6 bits of extra precision (coefficients sum to 64).
        // Vertical pass: undo horizontal precision (>> 6), then apply mc_shift/mc_offset.
        // For 8-bit: (val >> 6 + 32) >> 6. For 10-bit: (val >> 6 + 8) >> 4.
        let tmp_off = QPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 3) * MAX_PB_SIZE + i];
                }
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(((val >> 6) + offset) >> shift, bit_depth);
            }
        }
    }
}

/// Luma MC producing i32 intermediates for bi-prediction / weighted prediction
/// (no final clip).
///
/// The output is at "shift-6" precision: for sub-pixel, this is the raw
/// filter output (H/V-only) or `vert_pass >> 6` (2D). For integer-pel,
/// the pixel value is left-shifted by 6 to match. Bi-prediction combines
/// two intermediates as `clip((L0 + L1 + bipred_offset) >> bipred_shift)`.
///
/// Uses i32 output (instead of i16) for 10-bit safety: 8-tap on 1023-max
/// samples left-shifted by 6 can exceed i16 range.
///
/// Generic over `P: Pixel` for multi-bit-depth support.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma_i32<P: Pixel>(
    dst: &mut [i32],
    dst_stride: usize,
    ref_plane: &[P],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0: i32,
    y0: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    mv_x: i16,
    mv_y: i16,
    bit_depth: u8,
    tmp: &mut [i32],
    edge: &mut [P],
) {
    let x_frac = (mv_x & 3) as usize;
    let y_frac = (mv_y & 3) as usize;
    let x_int = x0 + (mv_x >> 2) as i32;
    let y_int = y0 + (mv_y >> 2) as i32;
    // Filter window: taps reach 3 sample(s) before and 4 after on each axis
    // with a fractional MV. Interior blocks read the reference plane
    // directly; edge blocks read a padded copy (see `ref_window`).
    let (bx, ax) = if x_frac != 0 { (3, 4) } else { (0, 0) };
    let (by, ay) = if y_frac != 0 { (3, 4) } else { (0, 0) };
    let (wx0, wy0) = (x_int - bx, y_int - by);
    let (src, src_stride, org) = ref_window(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        wx0,
        wy0,
        n_pb_w + (bx + ax) as usize,
        n_pb_h + (by + ay) as usize,
        edge,
        bit_depth,
    );
    let at = |x: i32, y: i32| -> i32 {
        src[org + (y - wy0) as usize * src_stride + (x - wx0) as usize].to_i32()
    };
    // Intermediate precision shift: 14 - bit_depth (8-bit: 6, 10-bit: 4).
    let shift3 = crate::pixel::mc_shift(bit_depth) as i32;
    // Sub-pel filter output shift: bit_depth - 8 (8-bit: 0, 10-bit: 2).
    let shift1 = (bit_depth as i32 - 8).max(0);

    if x_frac == 0 && y_frac == 0 {
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                dst[j * dst_stride + i] = at(x_int + i as i32, y_int + j as i32) << shift3;
            }
        }
    } else if y_frac == 0 {
        let filter = &QPEL_FILTER[x_frac];
        for j in 0..n_pb_h {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 3, ry);
                }
                dst[j * dst_stride + i] = val >> shift1;
            }
        }
    } else if x_frac == 0 {
        let filter = &QPEL_FILTER[y_frac];
        for j in 0..n_pb_h {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx, ry + k as i32 - 3);
                }
                dst[j * dst_stride + i] = val >> shift1;
            }
        }
    } else {
        let h_filter = &QPEL_FILTER[x_frac];
        let v_filter = &QPEL_FILTER[y_frac];
        let tmp_h = n_pb_h + (QPEL_EXTRA_BEFORE + QPEL_EXTRA_AFTER) as usize;
        debug_assert!(tmp.len() >= MC_TMP_LEN);
        let y_start = y_int - QPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 3, ry);
                }
                tmp[j * MAX_PB_SIZE + i] = val >> shift1;
            }
        }
        let tmp_off = QPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 3) * MAX_PB_SIZE + i];
                }
                dst[j * dst_stride + i] = val >> 6;
            }
        }
    }
}

/// Chroma MC producing i32 intermediates for bi-prediction / weighted prediction.
///
/// Uses i32 output for 10-bit safety. Generic over `P: Pixel`.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma_i32<P: Pixel>(
    dst: &mut [i32],
    dst_stride: usize,
    ref_plane: &[P],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0: i32,
    y0: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    mv_x: i32,
    mv_y: i32,
    bit_depth: u8,
    tmp: &mut [i32],
    edge: &mut [P],
) {
    // Chroma MV in eighth-chroma-sample units (see `chroma_mv`).
    let x_frac = (mv_x & 7) as usize;
    let y_frac = (mv_y & 7) as usize;
    let x_int = x0 + (mv_x >> 3);
    let y_int = y0 + (mv_y >> 3);
    // Filter window: taps reach 1 sample(s) before and 2 after on each axis
    // with a fractional MV. Interior blocks read the reference plane
    // directly; edge blocks read a padded copy (see `ref_window`).
    let (bx, ax) = if x_frac != 0 { (1, 2) } else { (0, 0) };
    let (by, ay) = if y_frac != 0 { (1, 2) } else { (0, 0) };
    let (wx0, wy0) = (x_int - bx, y_int - by);
    let (src, src_stride, org) = ref_window(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        wx0,
        wy0,
        n_pb_w + (bx + ax) as usize,
        n_pb_h + (by + ay) as usize,
        edge,
        bit_depth,
    );
    let at = |x: i32, y: i32| -> i32 {
        src[org + (y - wy0) as usize * src_stride + (x - wx0) as usize].to_i32()
    };
    let shift3 = crate::pixel::mc_shift(bit_depth) as i32;
    let shift1 = (bit_depth as i32 - 8).max(0);

    if x_frac == 0 && y_frac == 0 {
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                dst[j * dst_stride + i] = at(x_int + i as i32, y_int + j as i32) << shift3;
            }
        }
    } else if y_frac == 0 {
        let filter = &EPEL_FILTER[x_frac];
        for j in 0..n_pb_h {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 1, ry);
                }
                dst[j * dst_stride + i] = val >> shift1;
            }
        }
    } else if x_frac == 0 {
        let filter = &EPEL_FILTER[y_frac];
        for j in 0..n_pb_h {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx, ry + k as i32 - 1);
                }
                dst[j * dst_stride + i] = val >> shift1;
            }
        }
    } else {
        let h_filter = &EPEL_FILTER[x_frac];
        let v_filter = &EPEL_FILTER[y_frac];
        let tmp_h = n_pb_h + (EPEL_EXTRA_BEFORE + EPEL_EXTRA_AFTER) as usize;
        debug_assert!(tmp.len() >= MC_TMP_LEN);
        let y_start = y_int - EPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 1, ry);
                }
                tmp[j * MAX_PB_SIZE + i] = val >> shift1;
            }
        }
        let tmp_off = EPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 1) * MAX_PB_SIZE + i];
                }
                dst[j * dst_stride + i] = val >> 6;
            }
        }
    }
}

/// Uni-directional chroma motion compensation (4-tap filter).
///
/// `x0_c, y0_c` are chroma-plane coordinates (i.e. already halved for 4:2:0).
/// `mv_x_c, mv_y_c` are the halved MV components (still in eighth-pel units).
/// `bit_depth` is the chroma bit depth.
///
/// Generic over `P: Pixel` for multi-bit-depth support.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    ref_plane: &[P],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0_c: i32,
    y0_c: i32,
    n_pb_w_c: usize,
    n_pb_h_c: usize,
    mv_x_c: i32,
    mv_y_c: i32,
    bit_depth: u8,
    tmp: &mut [i32],
    edge: &mut [P],
) {
    // Chroma MV in eighth-chroma-sample units (see `chroma_mv`).
    let x_frac = (mv_x_c & 7) as usize;
    let y_frac = (mv_y_c & 7) as usize;
    let x_int = x0_c + (mv_x_c >> 3);
    let y_int = y0_c + (mv_y_c >> 3);
    // Filter window: taps reach 1 sample(s) before and 2 after on each axis
    // with a fractional MV. Interior blocks read the reference plane
    // directly; edge blocks read a padded copy (see `ref_window`).
    let (bx, ax) = if x_frac != 0 { (1, 2) } else { (0, 0) };
    let (by, ay) = if y_frac != 0 { (1, 2) } else { (0, 0) };
    let (wx0, wy0) = (x_int - bx, y_int - by);
    let (src, src_stride, org) = ref_window(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        wx0,
        wy0,
        n_pb_w_c + (bx + ax) as usize,
        n_pb_h_c + (by + ay) as usize,
        edge,
        bit_depth,
    );
    let at = |x: i32, y: i32| -> i32 {
        src[org + (y - wy0) as usize * src_stride + (x - wx0) as usize].to_i32()
    };

    let shift = crate::pixel::mc_shift(bit_depth) as i32;
    let offset = crate::pixel::mc_offset(bit_depth);
    let shift1 = (bit_depth as i32 - 8).max(0);

    if x_frac == 0 && y_frac == 0 {
        for j in 0..n_pb_h_c {
            for i in 0..n_pb_w_c {
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(at(x_int + i as i32, y_int + j as i32), bit_depth);
            }
        }
    } else if y_frac == 0 {
        let filter = &EPEL_FILTER[x_frac];
        for j in 0..n_pb_h_c {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w_c {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 1, ry);
                }
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(((val >> shift1) + offset) >> shift, bit_depth);
            }
        }
    } else if x_frac == 0 {
        let filter = &EPEL_FILTER[y_frac];
        for j in 0..n_pb_h_c {
            let ry = y_int + j as i32;
            for i in 0..n_pb_w_c {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in filter.iter().enumerate() {
                    val += coeff as i32 * at(rx, ry + k as i32 - 1);
                }
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(((val >> shift1) + offset) >> shift, bit_depth);
            }
        }
    } else {
        let h_filter = &EPEL_FILTER[x_frac];
        let v_filter = &EPEL_FILTER[y_frac];

        let tmp_h = n_pb_h_c + (EPEL_EXTRA_BEFORE + EPEL_EXTRA_AFTER) as usize;
        debug_assert!(tmp.len() >= MC_TMP_LEN);

        let y_start = y_int - EPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w_c {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32 * at(rx + k as i32 - 1, ry);
                }
                tmp[j * MAX_PB_SIZE + i] = val >> shift1;
            }
        }

        let tmp_off = EPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h_c {
            for i in 0..n_pb_w_c {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 1) * MAX_PB_SIZE + i];
                }
                dst[j * dst_stride + i] =
                    P::from_i32_clamped(((val >> 6) + offset) >> shift, bit_depth);
            }
        }
    }
}

/// Perform motion compensation for a single PU in an inter CU.
///
/// Reads the MV field from `tab_mvf` at position `(x0, y0)` and writes
/// the prediction into `state.y_plane`, `state.u_plane`, `state.v_plane`.
///
/// For uni-prediction (L0 or L1 only), the prediction is written directly.
/// For bi-prediction (both L0 and L1), the two predictions are averaged:
/// `dst = (predL0 + predL1 + 1) >> 1`.
#[allow(clippy::too_many_arguments)]
pub fn motion_compensation_pu<P: Pixel>(
    state: &mut PictureState<P>,
    scratch: &mut McScratch<P>,
    ref_frames_l0: &[Rc<DecodedPicture>],
    ref_frames_l1: &[Rc<DecodedPicture>],
    x0: u32,
    y0: u32,
    n_pb_w: u32,
    n_pb_h: u32,
    weighted_pred_flag: bool,
    pred_weight_table: &crate::slice::PredWeightTable,
) {
    // Read the MV field from the top-left min-PU of this PU.
    let x_pu = (x0 >> state.log2_min_pu_size) as usize;
    let y_pu = (y0 >> state.log2_min_pu_size) as usize;
    let mvf = state.tab_mvf[y_pu * state.min_pu_width + x_pu];
    let McScratch {
        luma0,
        luma1,
        cb0,
        cr0,
        cb1,
        cr1,
        tmp,
        edge,
    } = scratch;

    if mvf.pred_flag == 0 {
        // Intra PU in an inter slice — MC is not applicable. The intra
        // prediction path has already written the prediction.
        return;
    }

    let pic_w = state.width as i32;
    let pic_h = state.height as i32;
    let y_stride = state.y_stride;
    let uv_stride = state.uv_stride;
    let w = n_pb_w as usize;
    let h = n_pb_h as usize;
    // Chroma geometry: log2(SubWidthC) / log2(SubHeightC). 1 for 4:2:0
    // (half-size chroma blocks), 0 for 4:4:4 (chroma matches luma).
    let csw = state.chroma_shift_w;
    let csh = state.chroma_shift_h;
    let w_c = w >> csw;
    let h_c = h >> csh;
    let x0_c = (x0 >> csw) as i32;
    let y0_c = (y0 >> csh) as i32;
    let c_off = ((y0 >> csh) as usize) * uv_stride + ((x0 >> csw) as usize);

    let is_l0 = mvf.pred_flag & 1 != 0;
    let is_l1 = mvf.pred_flag & 2 != 0;
    let is_bi = is_l0 && is_l1;

    let bit_depth = state.bit_depth;
    let bi_shift = crate::pixel::bipred_shift(bit_depth) as i32;
    let bi_offset = crate::pixel::bipred_offset(bit_depth);

    if is_bi {
        // Bi-prediction: compute L0 and L1 intermediates at i32 precision
        // (before the final clip), then combine as specified by the HEVC spec:
        //   output = clip((L0 + L1 + bipred_offset) >> bipred_shift)
        // For 8-bit: shift = 7, offset = 64. For 10-bit: shift = 5, offset = 16.
        let pred_l0_y = &mut luma0[..];
        let pred_l1_y = &mut luma1[..];
        let pred_l0_u = &mut cb0[..];
        let pred_l0_v = &mut cr0[..];
        let pred_l1_u = &mut cb1[..];
        let pred_l1_v = &mut cr1[..];

        // L0
        let l0_idx = mvf.ref_idx[0] as usize;
        if l0_idx >= ref_frames_l0.len() {
            return;
        }
        let ref_pic_l0 = &ref_frames_l0[l0_idx];
        {
            let rw = ref_pic_l0.width as i32;
            let rh = ref_pic_l0.height as i32;
            let rs = ref_pic_l0.y_stride;
            mc_luma_i32::<P>(
                pred_l0_y,
                w,
                P::extract_slice(&ref_pic_l0.y),
                rs,
                rw,
                rh,
                x0 as i32,
                y0 as i32,
                w,
                h,
                mvf.mv[0].x,
                mvf.mv[0].y,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
            let rw_c = (ref_pic_l0.width >> csw) as i32;
            let rh_c = (ref_pic_l0.height >> csh) as i32;
            let rs_c = ref_pic_l0.uv_stride;
            let mv_x_c = chroma_mv(mvf.mv[0].x, csw);
            let mv_y_c = chroma_mv(mvf.mv[0].y, csh);
            mc_chroma_i32::<P>(
                pred_l0_u,
                w_c,
                P::extract_slice(&ref_pic_l0.u),
                rs_c,
                rw_c,
                rh_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
            mc_chroma_i32::<P>(
                pred_l0_v,
                w_c,
                P::extract_slice(&ref_pic_l0.v),
                rs_c,
                rw_c,
                rh_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
        }

        // L1
        let l1_idx = mvf.ref_idx[1] as usize;
        if l1_idx >= ref_frames_l1.len() {
            return;
        }
        let ref_pic_l1 = &ref_frames_l1[l1_idx];
        {
            let rw = ref_pic_l1.width as i32;
            let rh = ref_pic_l1.height as i32;
            let rs = ref_pic_l1.y_stride;
            mc_luma_i32::<P>(
                pred_l1_y,
                w,
                P::extract_slice(&ref_pic_l1.y),
                rs,
                rw,
                rh,
                x0 as i32,
                y0 as i32,
                w,
                h,
                mvf.mv[1].x,
                mvf.mv[1].y,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
            let rw_c = (ref_pic_l1.width >> csw) as i32;
            let rh_c = (ref_pic_l1.height >> csh) as i32;
            let rs_c = ref_pic_l1.uv_stride;
            let mv_x_c = chroma_mv(mvf.mv[1].x, csw);
            let mv_y_c = chroma_mv(mvf.mv[1].y, csh);
            mc_chroma_i32::<P>(
                pred_l1_u,
                w_c,
                P::extract_slice(&ref_pic_l1.u),
                rs_c,
                rw_c,
                rh_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
            mc_chroma_i32::<P>(
                pred_l1_v,
                w_c,
                P::extract_slice(&ref_pic_l1.v),
                rs_c,
                rw_c,
                rh_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
        }

        // Combine at i32 precision: (L0 + L1 + bipred_offset) >> bipred_shift, then clip.
        let y_off = (y0 as usize) * y_stride + (x0 as usize);
        for j in 0..h {
            for i in 0..w {
                let idx = j * w + i;
                let avg = (pred_l0_y[idx] + pred_l1_y[idx] + bi_offset) >> bi_shift;
                let dst_idx = y_off + j * y_stride + i;
                if dst_idx < state.y_plane.len() {
                    state.y_plane[dst_idx] = P::from_i32_clamped(avg, bit_depth);
                }
            }
        }
        for j in 0..h_c {
            for i in 0..w_c {
                let idx = j * w_c + i;
                let dst_idx = c_off + j * uv_stride + i;
                if dst_idx < state.u_plane.len() {
                    state.u_plane[dst_idx] = P::from_i32_clamped(
                        (pred_l0_u[idx] + pred_l1_u[idx] + bi_offset) >> bi_shift,
                        bit_depth,
                    );
                    state.v_plane[dst_idx] = P::from_i32_clamped(
                        (pred_l0_v[idx] + pred_l1_v[idx] + bi_offset) >> bi_shift,
                        bit_depth,
                    );
                }
            }
        }
    } else {
        // Uni-prediction: write directly into the picture planes.
        let (ref_list, mv, ref_idx, use_l0) = if is_l0 {
            let idx = mvf.ref_idx[0] as usize;
            if idx >= ref_frames_l0.len() {
                return; // malformed stream — ref index out of bounds
            }
            (&ref_frames_l0[idx], mvf.mv[0], idx, true)
        } else {
            let idx = mvf.ref_idx[1] as usize;
            if idx >= ref_frames_l1.len() {
                return;
            }
            (
                &ref_frames_l1[idx],
                mvf.mv[1],
                mvf.ref_idx[1] as usize,
                false,
            )
        };

        if !weighted_pred_flag {
            // Non-weighted uni-prediction: MC directly into the picture planes.
            // Write luma MC result directly to state.y_plane at the PU offset.
            let ref_y = P::extract_slice(&ref_list.y);
            let ref_u = P::extract_slice(&ref_list.u);
            let ref_v = P::extract_slice(&ref_list.v);
            let y_off = (y0 as usize) * y_stride + (x0 as usize);
            mc_luma::<P>(
                &mut state.y_plane[y_off..],
                y_stride,
                ref_y,
                ref_list.y_stride,
                pic_w,
                pic_h,
                x0 as i32,
                y0 as i32,
                w,
                h,
                mv.x,
                mv.y,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );

            let ref_w_c = (ref_list.width >> csw) as i32;
            let ref_h_c = (ref_list.height >> csh) as i32;
            let ref_uv_stride = ref_list.uv_stride;
            let mv_x_c = chroma_mv(mv.x, csw);
            let mv_y_c = chroma_mv(mv.y, csh);

            mc_chroma::<P>(
                &mut state.u_plane[c_off..],
                uv_stride,
                ref_u,
                ref_uv_stride,
                ref_w_c,
                ref_h_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
            mc_chroma::<P>(
                &mut state.v_plane[c_off..],
                uv_stride,
                ref_v,
                ref_uv_stride,
                ref_w_c,
                ref_h_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
        } else {
            // Weighted uni-prediction: compute MC at i16 precision, then apply
            // weight/offset per HEVC spec 8.5.3.3.4.1 / FFmpeg put_hevc_qpel_uni_w.
            let wt = pred_weight_table;
            let (luma_w, luma_o, chroma_w, chroma_o, luma_denom, chroma_denom) = if use_l0 {
                (
                    wt.luma_weight_l0[ref_idx] as i32,
                    wt.luma_offset_l0[ref_idx] as i32,
                    wt.chroma_weight_l0[ref_idx],
                    wt.chroma_offset_l0[ref_idx],
                    wt.luma_log2_weight_denom,
                    wt.chroma_log2_weight_denom,
                )
            } else {
                (
                    wt.luma_weight_l1[ref_idx] as i32,
                    wt.luma_offset_l1[ref_idx] as i32,
                    wt.chroma_weight_l1[ref_idx],
                    wt.chroma_offset_l1[ref_idx],
                    wt.luma_log2_weight_denom,
                    wt.chroma_log2_weight_denom,
                )
            };

            // Luma: MC into i32 intermediate, then apply weight.
            let pred_y = &mut luma0[..];
            mc_luma_i32::<P>(
                pred_y,
                w,
                P::extract_slice(&ref_list.y),
                ref_list.y_stride,
                pic_w,
                pic_h,
                x0 as i32,
                y0 as i32,
                w,
                h,
                mv.x,
                mv.y,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
            // Apply weighted prediction: spec 8.5.3.3.4.1
            // For uni-pred weighted: log2WD = luma_log2_weight_denom + (bit_depth - 8)
            // For 8-bit: log2WD = luma_log2_weight_denom
            // FFmpeg's put_hevc_qpel_uni_w uses:
            //   (((val >> (14 - 8)) * weight + (1 << (log2WD + 14 - 8 - 1))) >> (log2WD + 14 - 8)) + offset
            // The i32 intermediate from mc_luma_i32 is at (14 - bit_depth)
            // bits of extra precision. Weighted pred: clip((pred * weight +
            // round) >> (mc_shift + denom)) + offset.
            let mc_sh = crate::pixel::mc_shift(bit_depth) as i32;
            let shift = mc_sh + luma_denom as i32;
            let round = if shift > 0 { 1i32 << (shift - 1) } else { 0 };
            let y_off = (y0 as usize) * y_stride + (x0 as usize);
            for j in 0..h {
                for i in 0..w {
                    let val = pred_y[j * w + i];
                    let weighted = ((val * luma_w + round) >> shift) + luma_o;
                    let dst_idx = y_off + j * y_stride + i;
                    if dst_idx < state.y_plane.len() {
                        state.y_plane[dst_idx] = P::from_i32_clamped(weighted, bit_depth);
                    }
                }
            }

            // Chroma: similar weighted path.
            let ref_w_c = (ref_list.width >> csw) as i32;
            let ref_h_c = (ref_list.height >> csh) as i32;
            let ref_uv_stride = ref_list.uv_stride;
            let mv_x_c = chroma_mv(mv.x, csw);
            let mv_y_c = chroma_mv(mv.y, csh);
            let c_shift = mc_sh + chroma_denom as i32;
            let c_round = if c_shift > 0 {
                1i32 << (c_shift - 1)
            } else {
                0
            };

            let pred_u = &mut cb0[..];
            let pred_v = &mut cr0[..];
            mc_chroma_i32::<P>(
                pred_u,
                w_c,
                P::extract_slice(&ref_list.u),
                ref_uv_stride,
                ref_w_c,
                ref_h_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );
            mc_chroma_i32::<P>(
                pred_v,
                w_c,
                P::extract_slice(&ref_list.v),
                ref_uv_stride,
                ref_w_c,
                ref_h_c,
                x0_c,
                y0_c,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
                bit_depth,
                &mut tmp[..],
                &mut edge[..],
            );

            for j in 0..h_c {
                for i in 0..w_c {
                    let idx = j * w_c + i;
                    let dst_idx = c_off + j * uv_stride + i;
                    if dst_idx < state.u_plane.len() {
                        let wu = chroma_w[0] as i32;
                        let ou = chroma_o[0] as i32;
                        state.u_plane[dst_idx] = P::from_i32_clamped(
                            ((pred_u[idx] * wu + c_round) >> c_shift) + ou,
                            bit_depth,
                        );
                        let wv = chroma_w[1] as i32;
                        let ov = chroma_o[1] as i32;
                        state.v_plane[dst_idx] = P::from_i32_clamped(
                            ((pred_v[idx] * wv + c_round) >> c_shift) + ov,
                            bit_depth,
                        );
                    }
                }
            }
        }
    }
}

/// Perform motion compensation for an entire inter CU, dispatching to each PU
/// based on the partition mode. This replaces `write_placeholder_prediction`.
#[allow(clippy::too_many_arguments)]
pub fn motion_compensation_cu<P: Pixel>(
    state: &mut PictureState<P>,
    ref_frames_l0: &[Rc<DecodedPicture>],
    ref_frames_l1: &[Rc<DecodedPicture>],
    x0: u32,
    y0: u32,
    cb_size: u32,
    part_mode: crate::cu_tree::PartMode,
    weighted_pred_flag: bool,
    pred_weight_table: &crate::slice::PredWeightTable,
) {
    use crate::cu_tree::PartMode;

    // Lend the picture's scratch buffers to every PU of this CU. The empty
    // placeholder left behind is never used: no other code touches
    // `mc_scratch` while it is out, and it is restored below.
    let mut scratch = std::mem::take(&mut state.mc_scratch);

    // Local helper to reduce repetition when threading weight params.
    macro_rules! mc_pu {
        ($x:expr, $y:expr, $w:expr, $h:expr) => {
            motion_compensation_pu(
                state,
                &mut scratch,
                ref_frames_l0,
                ref_frames_l1,
                $x,
                $y,
                $w,
                $h,
                weighted_pred_flag,
                pred_weight_table,
            )
        };
    }

    match part_mode {
        PartMode::Part2Nx2N => {
            mc_pu!(x0, y0, cb_size, cb_size);
        }
        PartMode::Part2NxN => {
            let half = cb_size / 2;
            mc_pu!(x0, y0, cb_size, half);
            mc_pu!(x0, y0 + half, cb_size, half);
        }
        PartMode::PartNx2N => {
            let half = cb_size / 2;
            mc_pu!(x0, y0, half, cb_size);
            mc_pu!(x0 + half, y0, half, cb_size);
        }
        PartMode::Part2NxnU => {
            let q = cb_size / 4;
            mc_pu!(x0, y0, cb_size, q);
            mc_pu!(x0, y0 + q, cb_size, cb_size - q);
        }
        PartMode::Part2NxnD => {
            let tq = cb_size * 3 / 4;
            mc_pu!(x0, y0, cb_size, tq);
            mc_pu!(x0, y0 + tq, cb_size, cb_size - tq);
        }
        PartMode::PartnLx2N => {
            let q = cb_size / 4;
            mc_pu!(x0, y0, q, cb_size);
            mc_pu!(x0 + q, y0, cb_size - q, cb_size);
        }
        PartMode::PartnRx2N => {
            let tq = cb_size * 3 / 4;
            mc_pu!(x0, y0, tq, cb_size);
            mc_pu!(x0 + tq, y0, cb_size - tq, cb_size);
        }
        PartMode::PartNxN => {
            let half = cb_size / 2;
            for pi in 0..2u32 {
                for pj in 0..2u32 {
                    mc_pu!(x0 + pj * half, y0 + pi * half, half, half);
                }
            }
        }
    }
    state.mc_scratch = scratch;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integer-pel copy should reproduce the reference exactly.
    #[test]
    fn mc_luma_integer_pel_copy() {
        let ref_w = 16usize;
        let ref_h = 16usize;
        let mut ref_plane = vec![0u8; ref_w * ref_h];
        // Fill with a gradient.
        for j in 0..ref_h {
            for i in 0..ref_w {
                ref_plane[j * ref_w + i] = (j * ref_w + i) as u8;
            }
        }

        let w = 4;
        let h = 4;
        let mut dst = vec![0u8; w * h];

        // Zero MV: copy from (2,3).
        mc_luma(
            &mut dst,
            w,
            &ref_plane,
            ref_w,
            ref_w as i32,
            ref_h as i32,
            2,
            3,
            w,
            h,
            0,
            0,
            8,
            &mut vec![0i32; MC_TMP_LEN],
            &mut vec![0u8; MC_EDGE_LEN],
        );

        for j in 0..h {
            for i in 0..w {
                assert_eq!(dst[j * w + i], ref_plane[(3 + j) * ref_w + (2 + i)]);
            }
        }
    }

    /// Test that horizontal-only sub-pel filtering produces reasonable results.
    #[test]
    fn mc_luma_half_pel_horizontal() {
        let ref_w = 16usize;
        let ref_h = 16usize;
        let ref_plane = vec![128u8; ref_w * ref_h];

        let mut dst = vec![0u8; 4 * 4];
        // MV = (2, 0) = half-pel horizontal.
        mc_luma(
            &mut dst,
            4,
            &ref_plane,
            ref_w,
            ref_w as i32,
            ref_h as i32,
            4,
            4,
            4,
            4,
            2,
            0,
            8,
            &mut vec![0i32; MC_TMP_LEN],
            &mut vec![0u8; MC_EDGE_LEN],
        );

        // With uniform reference, all outputs should be 128.
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    /// Test that 2D sub-pel produces reasonable results on a uniform block.
    #[test]
    fn mc_luma_quarter_pel_hv_uniform() {
        let ref_w = 16usize;
        let ref_h = 16usize;
        let ref_plane = vec![200u8; ref_w * ref_h];

        let mut dst = vec![0u8; 4 * 4];
        mc_luma(
            &mut dst,
            4,
            &ref_plane,
            ref_w,
            ref_w as i32,
            ref_h as i32,
            4,
            4,
            4,
            4,
            1,
            3,
            8,
            &mut vec![0i32; MC_TMP_LEN],
            &mut vec![0u8; MC_EDGE_LEN],
        );

        for &v in &dst {
            assert_eq!(v, 200);
        }
    }

    /// Chroma integer-pel copy.
    #[test]
    fn mc_chroma_integer_pel_copy() {
        let ref_w = 8usize;
        let ref_h = 8usize;
        let mut ref_plane = vec![0u8; ref_w * ref_h];
        for j in 0..ref_h {
            for i in 0..ref_w {
                ref_plane[j * ref_w + i] = ((j + i) * 30) as u8;
            }
        }

        let w = 4;
        let h = 4;
        let mut dst = vec![0u8; w * h];

        mc_chroma(
            &mut dst,
            w,
            &ref_plane,
            ref_w,
            ref_w as i32,
            ref_h as i32,
            1,
            1,
            w,
            h,
            0,
            0,
            8,
            &mut vec![0i32; MC_TMP_LEN],
            &mut vec![0u8; MC_EDGE_LEN],
        );

        for j in 0..h {
            for i in 0..w {
                assert_eq!(dst[j * w + i], ref_plane[(1 + j) * ref_w + (1 + i)]);
            }
        }
    }

    /// Chroma sub-pel on uniform block should give the same value.
    #[test]
    fn mc_chroma_sub_pel_uniform() {
        let ref_w = 8usize;
        let ref_h = 8usize;
        let ref_plane = vec![100u8; ref_w * ref_h];

        let mut dst = vec![0u8; 4 * 4];
        mc_chroma(
            &mut dst,
            4,
            &ref_plane,
            ref_w,
            ref_w as i32,
            ref_h as i32,
            2,
            2,
            4,
            4,
            3,
            5,
            8,
            &mut vec![0i32; MC_TMP_LEN],
            &mut vec![0u8; MC_EDGE_LEN],
        );

        for &v in &dst {
            assert_eq!(v, 100);
        }
    }
}
