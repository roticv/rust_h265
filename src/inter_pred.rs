//! Phase 3d-6: Inter prediction — motion compensation with sub-pixel interpolation.
//!
//! Implements HEVC spec 8.5.3.2 (luma and chroma sample interpolation).
//! - Luma uses an 8-tap filter at quarter-pel resolution (4 positions).
//! - Chroma uses a 4-tap filter at eighth-pel resolution (8 positions).
//! - Boundary samples are handled by clamping reference coordinates.
//! - Bi-prediction averages L0 and L1 predictions: `(L0 + L1 + 1) >> 1`.

use std::rc::Rc;

use crate::cu_tree::PictureState;
use crate::dpb::DecodedPicture;

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

/// Fetch a reference luma sample with boundary clamping (spec 8.5.3.2.1).
#[inline]
fn ref_sample(plane: &[u8], stride: usize, x: i32, y: i32, w: i32, h: i32) -> i32 {
    let cx = x.clamp(0, w - 1) as usize;
    let cy = y.clamp(0, h - 1) as usize;
    plane[cy * stride + cx] as i32
}

/// Uni-directional luma motion compensation (8-tap filter).
///
/// Writes `nPbW * nPbH` prediction samples into `dst` at stride `dst_stride`.
/// `x0, y0` are the PU's top-left luma coordinates in the current picture.
/// `mv` is the quarter-pel motion vector.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma(
    dst: &mut [u8],
    dst_stride: usize,
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0: i32,
    y0: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    mv_x: i16,
    mv_y: i16,
) {
    let x_frac = (mv_x & 3) as usize;
    let y_frac = (mv_y & 3) as usize;
    // Integer part of the reference position.
    let x_int = x0 + (mv_x >> 2) as i32;
    let y_int = y0 + (mv_y >> 2) as i32;

    if x_frac == 0 && y_frac == 0 {
        // Integer-pel: direct copy.
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                dst[j * dst_stride + i] = ref_sample(
                    ref_plane,
                    ref_stride,
                    x_int + i as i32,
                    y_int + j as i32,
                    ref_w,
                    ref_h,
                ) as u8;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 3, ry, ref_w, ref_h);
                }
                // shift = 14 - 8 = 6, offset = 32
                dst[j * dst_stride + i] = ((val + 32) >> 6).clamp(0, 255) as u8;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx, ry + k as i32 - 3, ref_w, ref_h);
                }
                dst[j * dst_stride + i] = ((val + 32) >> 6).clamp(0, 255) as u8;
            }
        }
    } else {
        // 2D separable interpolation: horizontal first into i16 intermediate,
        // then vertical on the intermediate.
        let h_filter = &QPEL_FILTER[x_frac];
        let v_filter = &QPEL_FILTER[y_frac];

        // Intermediate buffer needs QPEL_EXTRA_BEFORE extra rows above and
        // QPEL_EXTRA_AFTER extra rows below.
        let tmp_h = n_pb_h + (QPEL_EXTRA_BEFORE + QPEL_EXTRA_AFTER) as usize;
        let mut tmp = vec![0i16; tmp_h * MAX_PB_SIZE];

        // Horizontal pass: produce (n_pb_h + 7) rows of i16 values.
        let y_start = y_int - QPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 3, ry, ref_w, ref_h);
                }
                // Horizontal pass stores without final shift (stays in i16 range).
                // For 8-bit: the result of the 8-tap filter on 8-bit samples
                // fits in i16 since coefficients sum to 64.
                tmp[j * MAX_PB_SIZE + i] = val as i16;
            }
        }

        // Vertical pass on the intermediate buffer.
        let tmp_off = QPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 3) * MAX_PB_SIZE + i] as i32;
                }
                // Two-stage shift: horizontal produced values at <<0 (for 8-bit),
                // vertical divides by 64 (>>6), then final rounding shift >>6.
                // Net: (val >> 6 + 32) >> 6.
                dst[j * dst_stride + i] = (((val >> 6) + 32) >> 6).clamp(0, 255) as u8;
            }
        }
    }
}

/// Luma MC producing i16 intermediates for bi-prediction (no final clip).
///
/// The output is at "shift-6" precision: for sub-pixel, this is the raw
/// filter output (H/V-only) or `vert_pass >> 6` (2D). For integer-pel,
/// the pixel value is left-shifted by 6 to match. Bi-prediction combines
/// two intermediates as `clip((L0 + L1 + 64) >> 7)`.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma_i16(
    dst: &mut [i16],
    dst_stride: usize,
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0: i32,
    y0: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    mv_x: i16,
    mv_y: i16,
) {
    let x_frac = (mv_x & 3) as usize;
    let y_frac = (mv_y & 3) as usize;
    let x_int = x0 + (mv_x >> 2) as i32;
    let y_int = y0 + (mv_y >> 2) as i32;

    if x_frac == 0 && y_frac == 0 {
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                dst[j * dst_stride + i] = (ref_sample(
                    ref_plane,
                    ref_stride,
                    x_int + i as i32,
                    y_int + j as i32,
                    ref_w,
                    ref_h,
                ) << 6) as i16;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 3, ry, ref_w, ref_h);
                }
                dst[j * dst_stride + i] = val as i16;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx, ry + k as i32 - 3, ref_w, ref_h);
                }
                dst[j * dst_stride + i] = val as i16;
            }
        }
    } else {
        let h_filter = &QPEL_FILTER[x_frac];
        let v_filter = &QPEL_FILTER[y_frac];
        let tmp_h = n_pb_h + (QPEL_EXTRA_BEFORE + QPEL_EXTRA_AFTER) as usize;
        let mut tmp = vec![0i16; tmp_h * MAX_PB_SIZE];
        let y_start = y_int - QPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 3, ry, ref_w, ref_h);
                }
                tmp[j * MAX_PB_SIZE + i] = val as i16;
            }
        }
        let tmp_off = QPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 3) * MAX_PB_SIZE + i] as i32;
                }
                // 2D: only the first >> 6 (second >> 6 is left for the combining step)
                dst[j * dst_stride + i] = (val >> 6) as i16;
            }
        }
    }
}

/// Chroma MC producing i16 intermediates for bi-prediction.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma_i16(
    dst: &mut [i16],
    dst_stride: usize,
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0: i32,
    y0: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    mv_x: i16,
    mv_y: i16,
) {
    let x_frac = (mv_x as i32 & 7) as usize;
    let y_frac = (mv_y as i32 & 7) as usize;
    let x_int = x0 + (mv_x as i32 >> 3);
    let y_int = y0 + (mv_y as i32 >> 3);

    if x_frac == 0 && y_frac == 0 {
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                dst[j * dst_stride + i] = (ref_sample(
                    ref_plane,
                    ref_stride,
                    x_int + i as i32,
                    y_int + j as i32,
                    ref_w,
                    ref_h,
                ) << 6) as i16;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 1, ry, ref_w, ref_h);
                }
                dst[j * dst_stride + i] = val as i16;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx, ry + k as i32 - 1, ref_w, ref_h);
                }
                dst[j * dst_stride + i] = val as i16;
            }
        }
    } else {
        let h_filter = &EPEL_FILTER[x_frac];
        let v_filter = &EPEL_FILTER[y_frac];
        let tmp_h = n_pb_h + (EPEL_EXTRA_BEFORE + EPEL_EXTRA_AFTER) as usize;
        let mut tmp = vec![0i16; tmp_h * MAX_PB_SIZE];
        let y_start = y_int - EPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 1, ry, ref_w, ref_h);
                }
                tmp[j * MAX_PB_SIZE + i] = val as i16;
            }
        }
        let tmp_off = EPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h {
            for i in 0..n_pb_w {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 1) * MAX_PB_SIZE + i] as i32;
                }
                dst[j * dst_stride + i] = (val >> 6) as i16;
            }
        }
    }
}

/// Uni-directional chroma motion compensation (4-tap filter).
///
/// `x0_c, y0_c` are chroma-plane coordinates (i.e. already halved for 4:2:0).
/// `mv_x_c, mv_y_c` are the halved MV components (still in eighth-pel units).
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma(
    dst: &mut [u8],
    dst_stride: usize,
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x0_c: i32,
    y0_c: i32,
    n_pb_w_c: usize,
    n_pb_h_c: usize,
    mv_x_c: i16,
    mv_y_c: i16,
) {
    // For 4:2:0, chroma MV has 3 fractional bits (eighth-pel).
    let x_frac = (mv_x_c & 7) as usize;
    let y_frac = (mv_y_c & 7) as usize;
    let x_int = x0_c + (mv_x_c >> 3) as i32;
    let y_int = y0_c + (mv_y_c >> 3) as i32;

    if x_frac == 0 && y_frac == 0 {
        for j in 0..n_pb_h_c {
            for i in 0..n_pb_w_c {
                dst[j * dst_stride + i] = ref_sample(
                    ref_plane,
                    ref_stride,
                    x_int + i as i32,
                    y_int + j as i32,
                    ref_w,
                    ref_h,
                ) as u8;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 1, ry, ref_w, ref_h);
                }
                dst[j * dst_stride + i] = ((val + 32) >> 6).clamp(0, 255) as u8;
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
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx, ry + k as i32 - 1, ref_w, ref_h);
                }
                dst[j * dst_stride + i] = ((val + 32) >> 6).clamp(0, 255) as u8;
            }
        }
    } else {
        let h_filter = &EPEL_FILTER[x_frac];
        let v_filter = &EPEL_FILTER[y_frac];

        let tmp_h = n_pb_h_c + (EPEL_EXTRA_BEFORE + EPEL_EXTRA_AFTER) as usize;
        let mut tmp = vec![0i16; tmp_h * MAX_PB_SIZE];

        let y_start = y_int - EPEL_EXTRA_BEFORE;
        for j in 0..tmp_h {
            let ry = y_start + j as i32;
            for i in 0..n_pb_w_c {
                let rx = x_int + i as i32;
                let mut val: i32 = 0;
                for (k, &coeff) in h_filter.iter().enumerate() {
                    val += coeff as i32
                        * ref_sample(ref_plane, ref_stride, rx + k as i32 - 1, ry, ref_w, ref_h);
                }
                tmp[j * MAX_PB_SIZE + i] = val as i16;
            }
        }

        let tmp_off = EPEL_EXTRA_BEFORE as usize;
        for j in 0..n_pb_h_c {
            for i in 0..n_pb_w_c {
                let mut val: i32 = 0;
                for (k, &coeff) in v_filter.iter().enumerate() {
                    val += coeff as i32 * tmp[(tmp_off + j + k - 1) * MAX_PB_SIZE + i] as i32;
                }
                dst[j * dst_stride + i] = (((val >> 6) + 32) >> 6).clamp(0, 255) as u8;
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
pub fn motion_compensation_pu(
    state: &mut PictureState,
    ref_frames_l0: &[Rc<DecodedPicture>],
    ref_frames_l1: &[Rc<DecodedPicture>],
    x0: u32,
    y0: u32,
    n_pb_w: u32,
    n_pb_h: u32,
) {
    // Read the MV field from the top-left min-PU of this PU.
    let x_pu = (x0 >> state.log2_min_pu_size) as usize;
    let y_pu = (y0 >> state.log2_min_pu_size) as usize;
    let mvf = state.tab_mvf[y_pu * state.min_pu_width + x_pu];

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

    let is_l0 = mvf.pred_flag & 1 != 0;
    let is_l1 = mvf.pred_flag & 2 != 0;
    let is_bi = is_l0 && is_l1;

    if is_bi {
        // Bi-prediction: compute L0 and L1 intermediates at i16 precision
        // (before the final clip), then combine as specified by the HEVC spec:
        //   output = clip((L0 + L1 + offset) >> shift)
        // For 8-bit: shift = 7, offset = 64 (= 1 << 6).
        let mut pred_l0_y = vec![0i16; w * h];
        let mut pred_l1_y = vec![0i16; w * h];
        let w_c = w / 2;
        let h_c = h / 2;
        let mut pred_l0_u = vec![0i16; w_c * h_c];
        let mut pred_l0_v = vec![0i16; w_c * h_c];
        let mut pred_l1_u = vec![0i16; w_c * h_c];
        let mut pred_l1_v = vec![0i16; w_c * h_c];

        // L0
        let ref_pic_l0 = &ref_frames_l0[mvf.ref_idx[0] as usize];
        {
            let rw = ref_pic_l0.width as i32;
            let rh = ref_pic_l0.height as i32;
            let rs = ref_pic_l0.width as usize;
            mc_luma_i16(
                &mut pred_l0_y,
                w,
                &ref_pic_l0.y,
                rs,
                rw,
                rh,
                x0 as i32,
                y0 as i32,
                w,
                h,
                mvf.mv[0].x,
                mvf.mv[0].y,
            );
            let rw_c = (ref_pic_l0.width / 2) as i32;
            let rh_c = (ref_pic_l0.height / 2) as i32;
            let rs_c = (ref_pic_l0.width / 2) as usize;
            let mv_x_c = mvf.mv[0].x;
            let mv_y_c = mvf.mv[0].y;
            mc_chroma_i16(
                &mut pred_l0_u,
                w_c,
                &ref_pic_l0.u,
                rs_c,
                rw_c,
                rh_c,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
            );
            mc_chroma_i16(
                &mut pred_l0_v,
                w_c,
                &ref_pic_l0.v,
                rs_c,
                rw_c,
                rh_c,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
            );
        }

        // L1
        let ref_pic_l1 = &ref_frames_l1[mvf.ref_idx[1] as usize];
        {
            let rw = ref_pic_l1.width as i32;
            let rh = ref_pic_l1.height as i32;
            let rs = ref_pic_l1.width as usize;
            mc_luma_i16(
                &mut pred_l1_y,
                w,
                &ref_pic_l1.y,
                rs,
                rw,
                rh,
                x0 as i32,
                y0 as i32,
                w,
                h,
                mvf.mv[1].x,
                mvf.mv[1].y,
            );
            let rw_c = (ref_pic_l1.width / 2) as i32;
            let rh_c = (ref_pic_l1.height / 2) as i32;
            let rs_c = (ref_pic_l1.width / 2) as usize;
            let mv_x_c = mvf.mv[1].x;
            let mv_y_c = mvf.mv[1].y;
            mc_chroma_i16(
                &mut pred_l1_u,
                w_c,
                &ref_pic_l1.u,
                rs_c,
                rw_c,
                rh_c,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
            );
            mc_chroma_i16(
                &mut pred_l1_v,
                w_c,
                &ref_pic_l1.v,
                rs_c,
                rw_c,
                rh_c,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                w_c,
                h_c,
                mv_x_c,
                mv_y_c,
            );
        }

        // Combine at i16 precision: (L0 + L1 + 64) >> 7, then clip.
        let y_off = (y0 as usize) * y_stride + (x0 as usize);
        for j in 0..h {
            for i in 0..w {
                let idx = j * w + i;
                let avg =
                    ((pred_l0_y[idx] as i32 + pred_l1_y[idx] as i32 + 64) >> 7).clamp(0, 255) as u8;
                let dst_idx = y_off + j * y_stride + i;
                if dst_idx < state.y_plane.len() {
                    state.y_plane[dst_idx] = avg;
                }
            }
        }
        let c_off = (y0 as usize / 2) * uv_stride + (x0 as usize / 2);
        for j in 0..h_c {
            for i in 0..w_c {
                let idx = j * w_c + i;
                let dst_idx = c_off + j * uv_stride + i;
                if dst_idx < state.u_plane.len() {
                    state.u_plane[dst_idx] = ((pred_l0_u[idx] as i32 + pred_l1_u[idx] as i32 + 64)
                        >> 7)
                        .clamp(0, 255) as u8;
                    state.v_plane[dst_idx] = ((pred_l0_v[idx] as i32 + pred_l1_v[idx] as i32 + 64)
                        >> 7)
                        .clamp(0, 255) as u8;
                }
            }
        }
    } else {
        // Uni-prediction: write directly into the picture planes.
        let (ref_list, mv) = if is_l0 {
            (&ref_frames_l0[mvf.ref_idx[0] as usize], mvf.mv[0])
        } else {
            (&ref_frames_l1[mvf.ref_idx[1] as usize], mvf.mv[1])
        };

        // Luma MC.
        let y_off = (y0 as usize) * y_stride + (x0 as usize);
        mc_luma(
            &mut state.y_plane[y_off..],
            y_stride,
            &ref_list.y,
            ref_list.width as usize,
            pic_w,
            pic_h,
            x0 as i32,
            y0 as i32,
            w,
            h,
            mv.x,
            mv.y,
        );

        // Chroma MC (4:2:0).
        let w_c = w / 2;
        let h_c = h / 2;
        let x0_c = (x0 / 2) as i32;
        let y0_c = (y0 / 2) as i32;
        let ref_w_c = (ref_list.width / 2) as i32;
        let ref_h_c = (ref_list.height / 2) as i32;
        let ref_uv_stride = (ref_list.width / 2) as usize;
        // Chroma MV for 4:2:0: the luma MV (in quarter-pel luma units) maps
        // directly to eighth-pel chroma units. The integer part is `mv >> 3`
        // and the 3-bit fractional part is `mv & 7`. We pass the LUMA MV
        // unchanged — `mc_chroma` uses `>> 3` / `& 7` internally.
        // This matches FFmpeg: `mx = av_zero_extend(mv->x, 2 + hshift)`,
        // `x_off = (x0 >> hshift) + (mv->x >> (2 + hshift))`.
        let mv_x_c = mv.x as i32;
        let mv_y_c = mv.y as i32;

        let c_off = (y0 as usize / 2) * uv_stride + (x0 as usize / 2);
        mc_chroma(
            &mut state.u_plane[c_off..],
            uv_stride,
            &ref_list.u,
            ref_uv_stride,
            ref_w_c,
            ref_h_c,
            x0_c,
            y0_c,
            w_c,
            h_c,
            mv_x_c as i16,
            mv_y_c as i16,
        );
        mc_chroma(
            &mut state.v_plane[c_off..],
            uv_stride,
            &ref_list.v,
            ref_uv_stride,
            ref_w_c,
            ref_h_c,
            x0_c,
            y0_c,
            w_c,
            h_c,
            mv_x_c as i16,
            mv_y_c as i16,
        );
    }
}

/// Helper: luma MC from a `DecodedPicture` into a temporary buffer.
#[allow(clippy::too_many_arguments)]
fn mc_luma_from_ref(
    dst: &mut [u8],
    dst_stride: usize,
    ref_pic: &DecodedPicture,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    mv: crate::cu_tree::Mv,
) {
    mc_luma(
        dst,
        dst_stride,
        &ref_pic.y,
        ref_pic.width as usize,
        ref_pic.width as i32,
        ref_pic.height as i32,
        x0,
        y0,
        w,
        h,
        mv.x,
        mv.y,
    );
}

/// Helper: chroma MC (both U and V) from a `DecodedPicture`.
#[allow(clippy::too_many_arguments)]
fn mc_chroma_from_ref_uv(
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    dst_stride_c: usize,
    ref_pic: &DecodedPicture,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    mv: crate::cu_tree::Mv,
) {
    let w_c = w / 2;
    let h_c = h / 2;
    let x0_c = x0 / 2;
    let y0_c = y0 / 2;
    let ref_w_c = (ref_pic.width / 2) as i32;
    let ref_h_c = (ref_pic.height / 2) as i32;
    let ref_uv_stride = (ref_pic.width / 2) as usize;
    // Pass luma MV unchanged — mc_chroma uses >> 3 / & 7 for 1/8-pel chroma.
    let mv_x_c = mv.x;
    let mv_y_c = mv.y;

    mc_chroma(
        dst_u,
        dst_stride_c,
        &ref_pic.u,
        ref_uv_stride,
        ref_w_c,
        ref_h_c,
        x0_c,
        y0_c,
        w_c,
        h_c,
        mv_x_c,
        mv_y_c,
    );
    mc_chroma(
        dst_v,
        dst_stride_c,
        &ref_pic.v,
        ref_uv_stride,
        ref_w_c,
        ref_h_c,
        x0_c,
        y0_c,
        w_c,
        h_c,
        mv_x_c,
        mv_y_c,
    );
}

/// Perform motion compensation for an entire inter CU, dispatching to each PU
/// based on the partition mode. This replaces `write_placeholder_prediction`.
pub fn motion_compensation_cu(
    state: &mut PictureState,
    ref_frames_l0: &[Rc<DecodedPicture>],
    ref_frames_l1: &[Rc<DecodedPicture>],
    x0: u32,
    y0: u32,
    cb_size: u32,
    part_mode: crate::cu_tree::PartMode,
) {
    use crate::cu_tree::PartMode;

    match part_mode {
        PartMode::Part2Nx2N => {
            motion_compensation_pu(
                state,
                ref_frames_l0,
                ref_frames_l1,
                x0,
                y0,
                cb_size,
                cb_size,
            );
        }
        PartMode::Part2NxN => {
            let half = cb_size / 2;
            motion_compensation_pu(state, ref_frames_l0, ref_frames_l1, x0, y0, cb_size, half);
            motion_compensation_pu(
                state,
                ref_frames_l0,
                ref_frames_l1,
                x0,
                y0 + half,
                cb_size,
                half,
            );
        }
        PartMode::PartNx2N => {
            let half = cb_size / 2;
            motion_compensation_pu(state, ref_frames_l0, ref_frames_l1, x0, y0, half, cb_size);
            motion_compensation_pu(
                state,
                ref_frames_l0,
                ref_frames_l1,
                x0 + half,
                y0,
                half,
                cb_size,
            );
        }
        PartMode::Part2NxnU => {
            let q = cb_size / 4;
            motion_compensation_pu(state, ref_frames_l0, ref_frames_l1, x0, y0, cb_size, q);
            motion_compensation_pu(
                state,
                ref_frames_l0,
                ref_frames_l1,
                x0,
                y0 + q,
                cb_size,
                cb_size - q,
            );
        }
        PartMode::Part2NxnD => {
            let tq = cb_size * 3 / 4;
            motion_compensation_pu(state, ref_frames_l0, ref_frames_l1, x0, y0, cb_size, tq);
            motion_compensation_pu(
                state,
                ref_frames_l0,
                ref_frames_l1,
                x0,
                y0 + tq,
                cb_size,
                cb_size - tq,
            );
        }
        PartMode::PartnLx2N => {
            let q = cb_size / 4;
            motion_compensation_pu(state, ref_frames_l0, ref_frames_l1, x0, y0, q, cb_size);
            motion_compensation_pu(
                state,
                ref_frames_l0,
                ref_frames_l1,
                x0 + q,
                y0,
                cb_size - q,
                cb_size,
            );
        }
        PartMode::PartnRx2N => {
            let tq = cb_size * 3 / 4;
            motion_compensation_pu(state, ref_frames_l0, ref_frames_l1, x0, y0, tq, cb_size);
            motion_compensation_pu(
                state,
                ref_frames_l0,
                ref_frames_l1,
                x0 + tq,
                y0,
                cb_size - tq,
                cb_size,
            );
        }
        PartMode::PartNxN => {
            let half = cb_size / 2;
            for pi in 0..2u32 {
                for pj in 0..2u32 {
                    motion_compensation_pu(
                        state,
                        ref_frames_l0,
                        ref_frames_l1,
                        x0 + pj * half,
                        y0 + pi * half,
                        half,
                        half,
                    );
                }
            }
        }
    }
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
        );

        for &v in &dst {
            assert_eq!(v, 100);
        }
    }
}
