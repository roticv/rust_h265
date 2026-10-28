// The Q0/P0 = `0 * xstride` and `(-1) * xstride` patterns are intentional
// for clarity — they mirror the FFmpeg `P3 P2 P1 P0 Q0 Q1 Q2 Q3` macro layout.
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! HEVC deblocking filter (spec 8.7.2).
//!
//! Filters the reconstructed picture in-place to soften block-edge artifacts.
//! For Phase 3b-1 we implement the intra-slice path: every internal TU/CU
//! 8×8 edge inside the picture gets boundary strength `bS = 2` (because both
//! sides are intra), and the strong/normal luma filters and chroma filter
//! are applied per spec 8.7.2.4 / 8.7.2.5.
//!
//! Reference: FFmpeg `libavcodec/hevc/filter.c::deblocking_filter_CTB` and
//! `libavcodec/hevc/dsp_template.c::hevc_loop_filter_*`.

use crate::cu_tree::PictureState;
use crate::pixel::Pixel;
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;

/// HEVC β table indexed by clipped QP, spec table 8-13.
#[rustfmt::skip]
static BETA_TABLE: [u8; 52] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
    22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
    62, 64,
];

/// HEVC tc table indexed by clipped (QP + 2*(bS-1) + tc_offset), spec table 8-14.
#[rustfmt::skip]
static TC_TABLE: [u8; 54] = [
    0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1,  1,  2,  2,  2,  2,  3,  3,  3,  3, 4, 4, 4,
    5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

const MAX_QP: i32 = 51;

/// Compute β for the given QP and offset, scaled for bit depth.
fn beta_lookup(qp: i32, beta_offset: i32, bit_depth: u8) -> i32 {
    let idx = (qp + beta_offset).clamp(0, MAX_QP) as usize;
    (BETA_TABLE[idx] as i32) << (bit_depth - 8)
}

/// Compute tc for the given QP, boundary strength, and offset, scaled for bit depth.
fn tc_lookup(qp: i32, bs: i32, tc_offset: i32, bit_depth: u8) -> i32 {
    // For intra deblocking the offset is 2*(bS - 1) per spec 8.7.2.5.
    let idx = (qp + 2 * (bs - 1) + tc_offset).clamp(0, 53) as usize;
    (TC_TABLE[idx] as i32) << (bit_depth - 8)
}

/// Map a luma QP to the chroma QP via spec table 8-9 (chroma_format_idc=1).
fn chroma_qp_table(qp_i: i32) -> i32 {
    const QP_C: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
    if qp_i < 30 {
        qp_i
    } else if qp_i > 43 {
        qp_i - 6
    } else {
        QP_C[(qp_i - 30) as usize]
    }
}

/// Compute chroma tc for an intra edge (bS == 2), scaled for bit depth.
/// ChromaArrayType 1 (4:2:0) maps qPi via table 8-9; 4:2:2/4:4:4 use
/// `Min(qPi, 51)` (spec 8.6.1 / 8.7.2.5.5).
fn chroma_tc(
    qp_y: i32,
    qp_offset: i32,
    tc_offset: i32,
    bit_depth: u8,
    chroma_array_type: u32,
) -> i32 {
    let qp_i = (qp_y + qp_offset).clamp(0, 57);
    let qp = if chroma_array_type == 1 {
        chroma_qp_table(qp_i)
    } else {
        qp_i.min(51)
    };
    // Intra deblocking: bS - 1 = 1, so tc table index gets +2.
    let idx = (qp + 2 + tc_offset).clamp(0, 53) as usize;
    (TC_TABLE[idx] as i32) << (bit_depth - 8)
}

// ---- Inner luma kernels (per "filter unit" of 4 pixels along the edge) ----

/// One 4-line luma deblocking segment, tap-major: `seg[t][l]` is tap `t`
/// (0 = p3, 1 = p2, 2 = p1, 3 = p0, 4 = q0, 5 = q1, 6 = q2, 7 = q3) on line
/// `l`. Holding a segment as lanes lets the filters below run all four
/// lines at once, which LLVM vectorises.
type LumaSeg = [[i32; 4]; 8];

/// Load the 8-tap × 4-line segment whose q0 on line 0 is `plane[base]`.
/// `xstride` crosses the edge, `ystride` runs along it. Each contiguous run
/// (a line for a vertical edge, a tap for a horizontal one) is read as one
/// slice, so there is one bounds check per run rather than per sample.
#[inline(always)]
fn load_luma_seg<P: Pixel>(plane: &[P], base: usize, xstride: isize, ystride: isize) -> LumaSeg {
    let mut seg = [[0i32; 4]; 8];
    if xstride == 1 {
        // Vertical edge: each line is 8 contiguous samples p3..q3.
        for l in 0..4 {
            let start = (base as isize + l as isize * ystride - 4) as usize;
            let line = &plane[start..start + 8];
            for t in 0..8 {
                seg[t][l] = line[t].to_i32();
            }
        }
    } else {
        // Horizontal edge (ystride == 1): each tap is 4 contiguous samples.
        for (t, tap) in seg.iter_mut().enumerate() {
            let start = (base as isize + (t as isize - 4) * xstride) as usize;
            for (v, &p) in tap.iter_mut().zip(&plane[start..start + 4]) {
                *v = p.to_i32();
            }
        }
    }
    seg
}

/// Store taps `taps` of `seg` back to the plane (inverse of
/// `load_luma_seg`), clipping to the sample range.
#[inline(always)]
fn store_luma_seg<P: Pixel>(
    plane: &mut [P],
    base: usize,
    xstride: isize,
    ystride: isize,
    seg: &LumaSeg,
    taps: std::ops::Range<usize>,
    bit_depth: u8,
) {
    if xstride == 1 {
        for l in 0..4 {
            let start = (base as isize + l as isize * ystride - 4) as usize;
            let line = &mut plane[start..start + 8];
            for t in taps.clone() {
                line[t] = P::from_i32_clamped(seg[t][l], bit_depth);
            }
        }
    } else {
        for t in taps {
            let start = (base as isize + (t as isize - 4) * xstride) as usize;
            for (p, &v) in plane[start..start + 4].iter_mut().zip(&seg[t]) {
                *p = P::from_i32_clamped(v, bit_depth);
            }
        }
    }
}

/// Strong luma filter (spec 8.7.2.5.7, dE == 2) on all four lines of a
/// segment: modifies p2..q2, each change clipped to ±`tc2` (= 2·tC).
#[inline(always)]
fn luma_strong_seg(seg: &mut LumaSeg, tc2: i32) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *seg;
    for l in 0..4 {
        let (p3, p2, p1, p0) = (p3[l], p2[l], p1[l], p0[l]);
        let (q0, q1, q2, q3) = (q0[l], q1[l], q2[l], q3[l]);
        let c = |new: i32, old: i32| old + (new - old).clamp(-tc2, tc2);
        seg[1][l] = c((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3, p2);
        seg[2][l] = c((p2 + p1 + p0 + q0 + 2) >> 2, p1);
        seg[3][l] = c((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3, p0);
        seg[4][l] = c((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3, q0);
        seg[5][l] = c((p0 + q0 + q1 + q2 + 2) >> 2, q1);
        seg[6][l] = c((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3, q2);
    }
}

/// Weak luma filter (spec 8.7.2.5.7, dE == 1) on all four lines of a
/// segment: modifies p0/q0, and p1/q1 when `nd_p` / `nd_q` is 2. A line
/// whose |Δ| reaches 10·tC is left unchanged.
#[inline(always)]
fn luma_weak_seg(seg: &mut LumaSeg, tc: i32, nd_p: i32, nd_q: i32) {
    let tc_2 = tc >> 1;
    let (mod_p1, mod_q1) = (nd_p > 1, nd_q > 1);
    let [_, p2, p1, p0, q0, q1, q2, _] = *seg;
    // Branch-free: compute every line's filtered values, then select per
    // line between filtered and original, so the four lines vectorise.
    for l in 0..4 {
        let (p2, p1, p0, q0, q1, q2) = (p2[l], p1[l], p0[l], q0[l], q1[l], q2[l]);
        let raw = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        let on = raw.abs() < 10 * tc;
        let delta0 = raw.clamp(-tc, tc);
        let np1 = p1 + ((((p2 + p0 + 1) >> 1) - p1 + delta0) >> 1).clamp(-tc_2, tc_2);
        let nq1 = q1 + ((((q2 + q0 + 1) >> 1) - q1 - delta0) >> 1).clamp(-tc_2, tc_2);
        seg[3][l] = if on { p0 + delta0 } else { p0 };
        seg[4][l] = if on { q0 - delta0 } else { q0 };
        seg[2][l] = if on && mod_p1 { np1 } else { p1 };
        seg[5][l] = if on && mod_q1 { nq1 } else { q1 };
    }
}

/// Chroma weak filter (the only chroma filter HEVC has). 4 lines.
fn loop_filter_chroma_weak<P: Pixel>(
    plane: &mut [P],
    base: usize,
    xstride: isize,
    ystride: isize,
    tc: i32,
    bit_depth: u8,
    [no_p, no_q]: [bool; 2],
) {
    for d in 0..4 {
        let line_off = d * ystride;
        let p1 = plane[(base as isize + line_off + (-2) * xstride) as usize].to_i32();
        let p0 = plane[(base as isize + line_off + (-1) * xstride) as usize].to_i32();
        let q0 = plane[(base as isize + line_off + 0 * xstride) as usize].to_i32();
        let q1 = plane[(base as isize + line_off + 1 * xstride) as usize].to_i32();
        let delta0 = ((((q0 - p0) * 4) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        if !no_p {
            plane[(base as isize + line_off + (-1) * xstride) as usize] =
                P::from_i32_clamped(p0 + delta0, bit_depth);
        }
        if !no_q {
            plane[(base as isize + line_off + 0 * xstride) as usize] =
                P::from_i32_clamped(q0 - delta0, bit_depth);
        }
    }
}

// ---- Per-edge dispatch (8 pixels along the edge = 2 filter units of 4) ----

/// Filter one 8-pixel luma edge given strength flags `bs0`, `bs1` for the
/// two halves of the edge. `pix_base` is the index of the first sample
/// of the second-side first row (Q0[0]). `xstride` and `ystride` describe
/// the sample layout (vertical edge: xstride=1, ystride=plane_stride;
/// horizontal edge: xstride=plane_stride, ystride=1).
#[allow(clippy::too_many_arguments)]
fn filter_luma_edge<P: Pixel>(
    plane: &mut [P],
    pix_base: usize,
    xstride: isize,
    ystride: isize,
    qp_avg: i32,
    bs0: i32,
    bs1: i32,
    beta_offset: i32,
    tc_offset: i32,
    bit_depth: u8,
    no_filter: [[bool; 2]; 2],
) {
    let beta = beta_lookup(qp_avg, beta_offset, bit_depth);
    let tcs = [
        if bs0 != 0 {
            tc_lookup(qp_avg, bs0, tc_offset, bit_depth)
        } else {
            0
        },
        if bs1 != 0 {
            tc_lookup(qp_avg, bs1, tc_offset, bit_depth)
        } else {
            0
        },
    ];

    for (j, &tc) in tcs.iter().enumerate() {
        if tc == 0 {
            continue;
        }
        // Each j-half is 4 lines along the edge, starting at line j*4.
        let j_base = (pix_base as isize + (j as isize * 4) * ystride) as usize;
        let mut seg = load_luma_seg(plane, j_base, xstride, ystride);
        let [p3, p2, p1, p0, q0, q1, q2, q3] = seg;

        // Decisions use lines 0 and 3 only (spec 8.7.2.5.3).
        let dp = |l: usize| (p2[l] - 2 * p1[l] + p0[l]).abs();
        let dq = |l: usize| (q2[l] - 2 * q1[l] + q0[l]).abs();
        let (d0_p, d0_q, d3_p, d3_q) = (dp(0), dq(0), dp(3), dq(3));
        let d0 = d0_p + d0_q;
        let d3 = d3_p + d3_q;
        if d0 + d3 >= beta {
            continue;
        }
        let beta_3 = beta >> 3;
        let beta_2 = beta >> 2;
        let tc25 = (tc * 5 + 1) >> 1;
        let strong_line = |l: usize, d: i32| {
            (p3[l] - p0[l]).abs() + (q3[l] - q0[l]).abs() < beta_3
                && (p0[l] - q0[l]).abs() < tc25
                && (d << 1) < beta_2
        };
        if strong_line(0, d0) && strong_line(3, d3) {
            luma_strong_seg(&mut seg, tc << 1);
            // P/Q samples of a lossless or PCM-protected block stay unchanged.
            if !no_filter[j][0] {
                store_luma_seg(plane, j_base, xstride, ystride, &seg, 1..4, bit_depth);
            }
            if !no_filter[j][1] {
                store_luma_seg(plane, j_base, xstride, ystride, &seg, 4..7, bit_depth);
            }
        } else {
            let side = (beta + (beta >> 1)) >> 3;
            let nd_p = if d0_p + d3_p < side { 2 } else { 1 };
            let nd_q = if d0_q + d3_q < side { 2 } else { 1 };
            luma_weak_seg(&mut seg, tc, nd_p, nd_q);
            // P/Q samples of a lossless or PCM-protected block stay unchanged.
            if !no_filter[j][0] {
                store_luma_seg(plane, j_base, xstride, ystride, &seg, 2..4, bit_depth);
            }
            if !no_filter[j][1] {
                store_luma_seg(plane, j_base, xstride, ystride, &seg, 4..6, bit_depth);
            }
        }
    }
}

/// Filter one 8-pixel chroma edge (in chroma samples). `bs0` and `bs1`
/// must both be 2 (otherwise this function isn't called).
#[allow(clippy::too_many_arguments)]
fn filter_chroma_edge<P: Pixel>(
    plane: &mut [P],
    pix_base: usize,
    xstride: isize,
    ystride: isize,
    qp0_avg: i32,
    qp1_avg: i32,
    qp_offset: i32,
    tc_offset: i32,
    bs0: i32,
    bs1: i32,
    bit_depth: u8,
    chroma_array_type: u32,
    no_filter: [[bool; 2]; 2],
) {
    let tcs = [
        if bs0 == 2 {
            chroma_tc(qp0_avg, qp_offset, tc_offset, bit_depth, chroma_array_type)
        } else {
            0
        },
        if bs1 == 2 {
            chroma_tc(qp1_avg, qp_offset, tc_offset, bit_depth, chroma_array_type)
        } else {
            0
        },
    ];
    for j in 0..2 {
        if tcs[j] == 0 {
            continue;
        }
        let j_base = (pix_base as isize + (j as isize * 4) * ystride) as usize;
        loop_filter_chroma_weak(
            plane,
            j_base,
            xstride,
            ystride,
            tcs[j],
            bit_depth,
            no_filter[j],
        );
    }
}

// ---- Top-level dispatch ----

/// Get the per-min-CB QP at luma sample position `(x, y)`.
/// Whether the luma sample at `(x, y)` belongs to a block that the loop
/// filters must not modify (see `PictureState::tab_no_filter`).
#[inline]
fn no_filter_at<P: Pixel>(state: &PictureState<P>, x: usize, y: usize) -> bool {
    state.any_no_filter
        && state
            .tab_no_filter
            .get((y >> 2) * state.min_pu_width + (x >> 2))
            .is_some_and(|&v| v != 0)
}

/// `[segment][side]` no-filter flags for the two 4-line segments of an
/// 8-sample edge: side 0 is P (before the edge), side 1 is Q. `p0` / `q0`
/// are the luma positions of segment 0's P and Q samples; `step` moves to
/// segment 1 along the edge.
#[inline]
fn edge_no_filter<P: Pixel>(
    state: &PictureState<P>,
    p0: (usize, usize),
    q0: (usize, usize),
    step: (usize, usize),
) -> [[bool; 2]; 2] {
    if !state.any_no_filter {
        return [[false; 2]; 2];
    }
    let at = |(x, y): (usize, usize), k: usize| no_filter_at(state, x + k * step.0, y + k * step.1);
    [[at(p0, 0), at(q0, 0)], [at(p0, 1), at(q0, 1)]]
}

fn get_qp_y<P: Pixel>(state: &PictureState<P>, x: i32, y: i32) -> i32 {
    let xc = (x.max(0) >> state.log2_min_cb_size) as usize;
    let yc = (y.max(0) >> state.log2_min_cb_size) as usize;
    state.tab_qp_y[yc * state.min_cb_width + xc] as i32
}

/// Read a boundary strength from `state.bs_vertical` (or `bs_horizontal`),
/// indexed by 4-pixel grid coordinates.
fn read_bs(bs: &[u8], pic_w: usize, x: usize, y: usize) -> i32 {
    bs[(y >> 2) * (pic_w >> 2) + (x >> 2)] as i32
}

/// Check whether a vertical edge at luma x crosses a slice boundary and
/// should be skipped because `slice_loop_filter_across_slices_enabled_flag`
/// is false on either side.
#[inline]
fn skip_vertical_slice_boundary<P: Pixel>(state: &PictureState<P>, x: usize, y: usize) -> bool {
    let log2_ctb = state.log2_ctb_size as usize;
    let left_ctb_col = (x - 1) >> log2_ctb;
    let right_ctb_col = x >> log2_ctb;
    if left_ctb_col == right_ctb_col {
        return false; // not at a CTB column boundary
    }
    let ctb_width = (state.width as usize).div_ceil(1 << log2_ctb);
    let ctb_row = y >> log2_ctb;
    let left_rs = ctb_row * ctb_width + left_ctb_col;
    let right_rs = ctb_row * ctb_width + right_ctb_col;
    if !state.filter_across_tiles && state.tab_tile_id[left_rs] != state.tab_tile_id[right_rs] {
        return true; // tile boundary, filtering across tiles disabled
    }
    if state.tab_slice_addr_rs[left_rs] == state.tab_slice_addr_rs[right_rs] {
        return false; // same slice
    }
    !state.filter_slice_edges[left_rs] || !state.filter_slice_edges[right_rs]
}

/// Check whether a horizontal edge at luma y crosses a slice or tile
/// boundary and should be skipped.
#[inline]
fn skip_horizontal_slice_boundary<P: Pixel>(state: &PictureState<P>, x: usize, y: usize) -> bool {
    let log2_ctb = state.log2_ctb_size as usize;
    let top_ctb_row = (y - 1) >> log2_ctb;
    let bot_ctb_row = y >> log2_ctb;
    if top_ctb_row == bot_ctb_row {
        return false; // not at a CTB row boundary
    }
    let ctb_width = (state.width as usize).div_ceil(1 << log2_ctb);
    let ctb_col = x >> log2_ctb;
    let top_rs = top_ctb_row * ctb_width + ctb_col;
    let bot_rs = bot_ctb_row * ctb_width + ctb_col;
    if !state.filter_across_tiles && state.tab_tile_id[top_rs] != state.tab_tile_id[bot_rs] {
        return true; // tile boundary, filtering across tiles disabled
    }
    if state.tab_slice_addr_rs[top_rs] == state.tab_slice_addr_rs[bot_rs] {
        return false; // same slice
    }
    !state.filter_slice_edges[top_rs] || !state.filter_slice_edges[bot_rs]
}

/// Apply the deblocking filter to the entire reconstructed picture.
///
/// Phase 3b-1 limitation: only the intra-slice path. The boundary strength
/// arrays must already be populated by the CU/TU decode (every internal
/// 8-aligned TU/CU edge inside the picture gets `bS = 2`).
pub fn deblock_picture<P: Pixel>(
    state: &mut PictureState<P>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
) {
    let pic_w = state.width as usize;
    let pic_h = state.height as usize;
    let stride_y = state.y_stride;
    let stride_uv = state.uv_stride;
    let beta_offset = (sh.slice_beta_offset_div2 + pps.pps_beta_offset_div2) * 2;
    let tc_offset = (sh.slice_tc_offset_div2 + pps.pps_tc_offset_div2) * 2;
    let bit_depth_luma = sps.bit_depth_luma;
    let bit_depth_chroma = sps.bit_depth_chroma;

    // ---- Luma vertical edges ----
    let mut y = 0usize;
    while y < pic_h {
        let mut x = 8usize;
        while x < pic_w {
            if skip_vertical_slice_boundary(state, x, y) {
                x += 8;
                continue;
            }
            let bs0 = read_bs(&state.bs_vertical, pic_w, x, y);
            let bs1 = read_bs(&state.bs_vertical, pic_w, x, y + 4);
            if bs0 != 0 || bs1 != 0 {
                let qp_avg = (get_qp_y(state, x as i32 - 1, y as i32)
                    + get_qp_y(state, x as i32, y as i32)
                    + 1)
                    >> 1;
                let pix_base = y * stride_y + x;
                let nf = edge_no_filter(state, (x - 1, y), (x, y), (0, 4));
                filter_luma_edge::<P>(
                    &mut state.y_plane,
                    pix_base,
                    1,
                    stride_y as isize,
                    qp_avg,
                    bs0,
                    bs1,
                    beta_offset,
                    tc_offset,
                    bit_depth_luma,
                    nf,
                );
            }
            x += 8;
        }
        y += 8;
    }

    // ---- Luma horizontal edges ----
    let mut y = 8usize;
    while y < pic_h {
        let mut x = 0usize;
        while x < pic_w {
            if skip_horizontal_slice_boundary(state, x, y) {
                x += 8;
                continue;
            }
            let bs0 = read_bs(&state.bs_horizontal, pic_w, x, y);
            let bs1 = read_bs(&state.bs_horizontal, pic_w, x + 4, y);
            if bs0 != 0 || bs1 != 0 {
                let qp_avg = (get_qp_y(state, x as i32, y as i32 - 1)
                    + get_qp_y(state, x as i32, y as i32)
                    + 1)
                    >> 1;
                let pix_base = y * stride_y + x;
                let nf = edge_no_filter(state, (x, y - 1), (x, y), (4, 0));
                filter_luma_edge::<P>(
                    &mut state.y_plane,
                    pix_base,
                    stride_y as isize,
                    1,
                    qp_avg,
                    bs0,
                    bs1,
                    beta_offset,
                    tc_offset,
                    bit_depth_luma,
                    nf,
                );
            }
            x += 8;
        }
        y += 8;
    }

    // ---- Chroma deblocking ----
    //
    // Chroma edges lie on the 8×8 chroma-sample grid; in luma coordinates that
    // is every `8 << SubWidthC-shift` columns / `8 << SubHeightC-shift` rows,
    // and the two boundary-strength probes along an edge sit `4 << shift` luma
    // samples apart. Chroma pixel coordinates are the luma coordinates shifted
    // right by the subsampling (4:2:0 → the 16-luma grid; 4:4:4 → the 8-luma
    // grid). To keep the borrow checker happy we collect every chroma edge
    // before grabbing &mut state.u_plane / v_plane.
    let chroma_array_type = sps.chroma_array_type();
    if chroma_array_type != 0 {
        let cshift_w = state.chroma_shift_w as usize;
        let cshift_h = state.chroma_shift_h as usize;
        let step_x = 8usize << cshift_w; // vertical-edge column spacing (luma)
        let step_y = 8usize << cshift_h; // horizontal-edge row spacing (luma)
        let bs_dx = 4usize << cshift_w; // 2nd bS probe offset along a horiz edge
        let bs_dy = 4usize << cshift_h; // 2nd bS probe offset along a vert edge

        struct ChromaEdge {
            pix_base: usize,
            xstride: isize,
            ystride: isize,
            qp0_avg: i32,
            qp1_avg: i32,
            bs0: i32,
            bs1: i32,
            no_filter: [[bool; 2]; 2],
        }
        let mut edges: Vec<ChromaEdge> = Vec::new();

        // Vertical edges.
        let mut y_l = 0usize;
        while y_l < pic_h {
            let mut x_l = step_x;
            while x_l < pic_w {
                if skip_vertical_slice_boundary(state, x_l, y_l) {
                    x_l += step_x;
                    continue;
                }
                let bs0 = read_bs(&state.bs_vertical, pic_w, x_l, y_l);
                let bs1 = read_bs(&state.bs_vertical, pic_w, x_l, y_l + bs_dy);
                if bs0 == 2 || bs1 == 2 {
                    let qp0_avg = (get_qp_y(state, x_l as i32 - 1, y_l as i32)
                        + get_qp_y(state, x_l as i32, y_l as i32)
                        + 1)
                        >> 1;
                    let qp1_avg = (get_qp_y(state, x_l as i32 - 1, (y_l + bs_dy) as i32)
                        + get_qp_y(state, x_l as i32, (y_l + bs_dy) as i32)
                        + 1)
                        >> 1;
                    let xc = x_l >> cshift_w;
                    let yc = y_l >> cshift_h;
                    edges.push(ChromaEdge {
                        pix_base: yc * stride_uv + xc,
                        xstride: 1,
                        ystride: stride_uv as isize,
                        qp0_avg,
                        qp1_avg,
                        bs0,
                        bs1,
                        no_filter: edge_no_filter(state, (x_l - 1, y_l), (x_l, y_l), (0, bs_dy)),
                    });
                }
                x_l += step_x;
            }
            y_l += step_y;
        }
        // Horizontal edges.
        let mut y_l = step_y;
        while y_l < pic_h {
            let mut x_l = 0usize;
            while x_l < pic_w {
                if skip_horizontal_slice_boundary(state, x_l, y_l) {
                    x_l += step_x;
                    continue;
                }
                let bs0 = read_bs(&state.bs_horizontal, pic_w, x_l, y_l);
                let bs1 = read_bs(&state.bs_horizontal, pic_w, x_l + bs_dx, y_l);
                if bs0 == 2 || bs1 == 2 {
                    let qp0_avg = (get_qp_y(state, x_l as i32, y_l as i32 - 1)
                        + get_qp_y(state, x_l as i32, y_l as i32)
                        + 1)
                        >> 1;
                    let qp1_avg = (get_qp_y(state, (x_l + bs_dx) as i32, y_l as i32 - 1)
                        + get_qp_y(state, (x_l + bs_dx) as i32, y_l as i32)
                        + 1)
                        >> 1;
                    let xc = x_l >> cshift_w;
                    let yc = y_l >> cshift_h;
                    edges.push(ChromaEdge {
                        pix_base: yc * stride_uv + xc,
                        xstride: stride_uv as isize,
                        ystride: 1,
                        qp0_avg,
                        qp1_avg,
                        bs0,
                        bs1,
                        no_filter: edge_no_filter(state, (x_l, y_l - 1), (x_l, y_l), (bs_dx, 0)),
                    });
                }
                x_l += step_x;
            }
            y_l += step_y;
        }

        // Now apply edges to each chroma plane.
        for c_idx in 1..=2 {
            let qp_offset = if c_idx == 1 {
                pps.pps_cb_qp_offset
            } else {
                pps.pps_cr_qp_offset
            };
            let plane = if c_idx == 1 {
                &mut state.u_plane
            } else {
                &mut state.v_plane
            };
            for e in &edges {
                filter_chroma_edge::<P>(
                    plane,
                    e.pix_base,
                    e.xstride,
                    e.ystride,
                    e.qp0_avg,
                    e.qp1_avg,
                    qp_offset,
                    tc_offset,
                    e.bs0,
                    e.bs1,
                    bit_depth_chroma,
                    chroma_array_type,
                    e.no_filter,
                );
            }
        }
    }
}
