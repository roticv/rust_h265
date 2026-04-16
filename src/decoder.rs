//! Top-level HEVC decoder.
//!
//! Phase 2c-6 scope: a `Decoder` that owns the active VPS/SPS/PPS, accepts
//! NAL units one at a time, parses slice headers + drives the CU tree
//! decode (which in turn calls intra prediction + IDCT + reconstruction),
//! and emits a `Frame` for each completed picture.
//!
//! For Phase 2c-6 we only handle IDR I-slice pictures with one CU per CTU
//! (= what `testdata/tiny_intra.h265` produces). Anything outside that
//! subset is rejected via `Unsupported` from the underlying parsers.
//!
//! Phase 3c-1 extends this to independent multi-slice pictures: a picture
//! can be split across several VCL NAL units where each slice segment
//! carries its own slice header and covers a contiguous range of CTBs
//! starting at `slice_segment_address`. The decoder lazily creates a
//! `PictureState` on the first slice of the picture and finalizes the
//! picture (deblock + SAO) when the CTB count reaches the picture total.

use std::rc::Rc;

use crate::cabac::{CabacContexts, CabacReader};
use crate::cu_tree::{PictureState, decode_coding_quadtree};
use crate::dpb::{
    DecodedPicture, DecodedPictureBuffer, PictureReferenceStatus, ReferencePictureSets,
    apply_ref_pic_list_modification, build_ref_pic_list_temp0, build_ref_pic_list_temp1,
    resolve_ref_pics,
};
use crate::error::DecodeError;
use crate::nal::{NalUnit, NalUnitType};
use crate::pps::{Pps, parse_pps};
use crate::slice::{SliceHeader, SliceType, parse_slice_segment_header};
use crate::sps::{ShortTermRps, Sps, parse_sps};
use crate::vps::{Vps, parse_vps};

/// A reconstructed video frame in YUV420 8-bit planar layout.
///
/// Plane lengths are `width * height` for luma and `(width/2) * (height/2)`
/// for each chroma plane. `pic_order_cnt` is the full (signed) POC
/// computed per spec 8.3.1 — 0 for IDR pictures, and the MSB-extended LSB
/// for non-IDR pictures.
#[derive(Debug, Clone)]
pub struct Frame {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub pic_order_cnt: i32,
}

/// Streaming HEVC decoder.
///
/// Usage:
/// ```ignore
/// let mut dec = rust_h265::decoder::Decoder::new();
/// for nal in nal_units {
///     if let Some(frame) = dec.decode_nal(&nal)? {
///         // ... display, encode, etc.
///     }
/// }
/// if let Some(frame) = dec.flush() {
///     // ... last buffered frame
/// }
/// ```
/// In-flight picture state: the reconstruction buffers plus the bookkeeping
/// needed to stitch multi-slice decode back together.
struct PictureInProgress {
    state: PictureState,
    /// Header of the most recently decoded slice segment. Phase 3c-1 uses
    /// it for deblock/SAO finalization — in the common case all slices in a
    /// picture share the same filter flags, which this approximation
    /// matches.
    last_slice_header: SliceHeader,
    /// Phase 3c-4: header of the most recent *independent* slice segment
    /// in this picture. A dependent slice segment inherits `slice_type`,
    /// `slice_qp_y`, `slice_sao_*`, and the deblock filter fields from
    /// this header; the parser leaves those fields at default values on
    /// dependent slices and the decoder copies them in from here. For a
    /// picture that doesn't use dependent slice segments this is simply a
    /// copy of `last_slice_header` after every slice.
    last_independent_slice_header: SliceHeader,
    /// Phase 3c-4: CABAC contexts snapshot at the end of the previous
    /// slice segment (after the slice's final `end_of_slice_flag = 1`
    /// terminate bin). A dependent slice segment restores its contexts
    /// from this snapshot instead of re-initializing from the slice QP
    /// when the slice does NOT start on a WPP row boundary. `None`
    /// before any slice has finished.
    saved_cabac_state: Option<[u8; crate::cabac_tables::HEVC_CONTEXTS]>,
    /// Phase 3c-4: WPP context state snapshot (state after the 2nd CTB
    /// of a row). For WPP + dependent slice segments (one slice per row
    /// layout), the dependent slice at the start of a new row loads this
    /// state instead of `saved_cabac_state`. Mirrors FFmpeg's
    /// `common_cabac_state->state` save/load used by `load_states`.
    saved_wpp_cabac_state: Option<[u8; crate::cabac_tables::HEVC_CONTEXTS]>,
    /// Number of CTBs already decoded in this picture (sum across all
    /// slice segments seen so far).
    ctbs_decoded: u32,
    /// Total CTBs in the picture = `pic_width_in_ctbs * pic_height_in_ctbs`.
    total_ctbs: u32,
}

/// Tile scan derivation tables for a given (SPS, PPS) pair (spec 6.5.1).
///
/// `ctb_addr_rs_to_ts[rs]` = tile-scan address for a CTB at raster address
/// `rs`. `ctb_addr_ts_to_rs[ts]` is the inverse. `tile_id[rs]` gives the
/// 0-based tile index of the CTB at raster address `rs`.
///
/// Single-tile (`num_tile_columns = 1 && num_tile_rows = 1`) degenerates to
/// identity tables and `tile_id` = 0 everywhere, preserving raster scan.
#[derive(Debug, Clone)]
struct TileScanTables {
    /// Raster → tile-scan. Kept for completeness (we currently only need the
    /// inverse during decode) and for future parallel / out-of-order work.
    #[allow(dead_code)]
    ctb_addr_rs_to_ts: Vec<u32>,
    ctb_addr_ts_to_rs: Vec<u32>,
    tile_id: Vec<u32>,
}

impl TileScanTables {
    /// Derive the scan tables from an SPS and a PPS whose
    /// `resolve_tile_geometry` has already been called.
    fn build(sps: &Sps, pps: &Pps) -> Self {
        let pic_w_ctbs = sps.pic_width_in_ctbs_y() as usize;
        let pic_h_ctbs = sps.pic_height_in_ctbs_y() as usize;
        let total = pic_w_ctbs * pic_h_ctbs;

        let mut rs_to_ts = vec![0u32; total];
        let mut ts_to_rs = vec![0u32; total];
        let mut tile_id = vec![0u32; total];

        // Tile column / row boundaries in CTB coords (cumulative).
        let n_cols = pps.num_tile_columns;
        let n_rows = pps.num_tile_rows;
        let mut col_bd = vec![0u32; n_cols + 1];
        for i in 0..n_cols {
            col_bd[i + 1] = col_bd[i] + pps.column_widths_in_ctbs[i];
        }
        let mut row_bd = vec![0u32; n_rows + 1];
        for i in 0..n_rows {
            row_bd[i + 1] = row_bd[i] + pps.row_heights_in_ctbs[i];
        }

        // Fill `ctb_addr_rs_to_ts` via the formula in FFmpeg's `setup_pps`
        // (spec 6.5.1, HEVC reference decoder). For every raster address we
        // locate its tile (tile_x, tile_y) and count how many CTBs precede
        // it in tile-scan order.
        #[allow(clippy::needless_range_loop)]
        for ctb_addr_rs in 0..total {
            let tb_x = (ctb_addr_rs % pic_w_ctbs) as u32;
            let tb_y = (ctb_addr_rs / pic_w_ctbs) as u32;

            let mut tile_x = 0usize;
            for i in 0..n_cols {
                if tb_x < col_bd[i + 1] {
                    tile_x = i;
                    break;
                }
            }
            let mut tile_y = 0usize;
            for i in 0..n_rows {
                if tb_y < row_bd[i + 1] {
                    tile_y = i;
                    break;
                }
            }

            // Count CTBs in all earlier tiles within the same tile row (tile_y)
            // + all earlier tile rows.
            let mut val: u32 = 0;
            for i in 0..tile_x {
                val += pps.row_heights_in_ctbs[tile_y] * pps.column_widths_in_ctbs[i];
            }
            for i in 0..tile_y {
                val += (pic_w_ctbs as u32) * pps.row_heights_in_ctbs[i];
            }
            val += (tb_y - row_bd[tile_y]) * pps.column_widths_in_ctbs[tile_x]
                + (tb_x - col_bd[tile_x]);

            rs_to_ts[ctb_addr_rs] = val;
            ts_to_rs[val as usize] = ctb_addr_rs as u32;
        }

        // Tile id per CTB raster address. Flattening by tile-row then
        // tile-column gives the standard tile ordering.
        let mut cur_id: u32 = 0;
        for j in 0..n_rows {
            for i in 0..n_cols {
                for y in row_bd[j]..row_bd[j + 1] {
                    for x in col_bd[i]..col_bd[i + 1] {
                        let rs = (y as usize) * pic_w_ctbs + x as usize;
                        tile_id[rs] = cur_id;
                    }
                }
                cur_id += 1;
            }
        }

        Self {
            ctb_addr_rs_to_ts: rs_to_ts,
            ctb_addr_ts_to_rs: ts_to_rs,
            tile_id,
        }
    }
}

#[derive(Default)]
pub struct Decoder {
    vps: Option<Vps>,
    sps: Option<Sps>,
    pps: Option<Pps>,
    /// Phase 3c-2: cached tile-scan tables for the active (SPS, PPS) pair.
    /// Rebuilt lazily on the first slice following a parameter-set change.
    tile_tables: Option<TileScanTables>,
    /// The picture currently being assembled from one or more slice segments.
    /// Phase 3c-1: created on the first slice segment, finalized and
    /// returned as a `Frame` when all CTBs have been decoded.
    current_picture: Option<PictureInProgress>,
    /// Phase 3d-1: decoded picture buffer. Holds recently decoded pictures
    /// for reference lookup. Not yet used for inter decoding (which is the
    /// reason this infrastructure exists) — every picture currently
    /// inserted is an IDR intra picture. Output happens immediately after
    /// decoding; bumping for reorder is a future phase.
    dpb: DecodedPictureBuffer,
    /// Phase 3d-1: POC of the most recent decoded picture with
    /// `temporal_id == 0` (and not a sub-layer non-reference or
    /// RASL/RADL picture). Per spec 8.3.1 this is the "prevTid0Pic"
    /// seed for computing a non-IDR picture's POC from its LSB. We
    /// reset it to 0 at every IDR and update it after every picture
    /// whose NAL type + temporal_id satisfy the "tid0" condition.
    prev_tid0_poc: i32,
    /// Phase 3d-1: most recently derived reference picture sets for the
    /// current picture. Populated by `derive_rps_from_slice_header`; no
    /// consumer yet other than the decoder's own bookkeeping.
    current_rps: ReferencePictureSets,
    /// Phase 3d-2: RefPicList0 for the current P/B slice, built per spec
    /// 8.3.2 after RPS marking. Empty for I slices. Consumed by Phase 3d-5
    /// AMVP to populate `SliceParams::ref_pic_list_pocs`.
    current_ref_list_l0: Vec<Rc<DecodedPicture>>,
    /// Phase 3d-2: RefPicList1 for the current B slice. Empty for P and I
    /// slices. Consumed by Phase 3d-5 AMVP.
    current_ref_list_l1: Vec<Rc<DecodedPicture>>,
}

/// Crop a decoded picture's planes to the conformance window and build a Frame.
/// The pixel planes may have CTU-aligned strides larger than `coded_w`, so we
/// always use `state.y_stride` / `state.uv_stride` for row addressing.
#[allow(clippy::too_many_arguments)]
fn crop_frame(
    state: &crate::cu_tree::PictureState,
    _coded_w: u32,
    _coded_h: u32,
    cropped_w: u32,
    cropped_h: u32,
    left_offset: u32, // in chroma sample units
    top_offset: u32,  // in chroma sample units
    poc: i32,
) -> Frame {
    // For 4:2:0, SubWidthC = SubHeightC = 2.
    let luma_left = (left_offset * 2) as usize;
    let luma_top = (top_offset * 2) as usize;
    let cw = cropped_w as usize;
    let ch = cropped_h as usize;
    let stride_y = state.y_stride;
    let stride_uv = state.uv_stride;

    // Fast path: if the output dimensions match the stride (no padding, no
    // conformance crop), we can do a simple row-copy or even clone.
    let no_crop =
        luma_left == 0 && luma_top == 0 && cw == stride_y && ch == state.y_plane.len() / stride_y;
    if no_crop {
        return Frame {
            y: state.y_plane.clone(),
            u: state.u_plane.clone(),
            v: state.v_plane.clone(),
            width: cropped_w,
            height: cropped_h,
            pic_order_cnt: poc,
        };
    }

    let mut y = Vec::with_capacity(cw * ch);
    for row in 0..ch {
        let src_row = luma_top + row;
        let start = src_row * stride_y + luma_left;
        y.extend_from_slice(&state.y_plane[start..start + cw]);
    }
    let cw_c = cw / 2;
    let ch_c = ch / 2;
    let mut u = Vec::with_capacity(cw_c * ch_c);
    let mut v = Vec::with_capacity(cw_c * ch_c);
    for row in 0..ch_c {
        let src_row = top_offset as usize + row;
        let start = src_row * stride_uv + left_offset as usize;
        u.extend_from_slice(&state.u_plane[start..start + cw_c]);
        v.extend_from_slice(&state.v_plane[start..start + cw_c]);
    }
    Frame {
        y,
        u,
        v,
        width: cropped_w,
        height: cropped_h,
        pic_order_cnt: poc,
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one NAL unit. Returns `Ok(Some(frame))` when a picture has just
    /// finished decoding, `Ok(None)` otherwise (e.g. parameter sets, SEI).
    pub fn decode_nal(&mut self, nal: &NalUnit<'_>) -> Result<Option<Frame>, DecodeError> {
        match nal.nal_unit_type {
            NalUnitType::Vps => {
                self.vps = Some(parse_vps(&nal.rbsp)?);
                Ok(None)
            }
            NalUnitType::Sps => {
                self.sps = Some(parse_sps(&nal.rbsp)?);
                self.tile_tables = None;
                Ok(None)
            }
            NalUnitType::Pps => {
                self.pps = Some(parse_pps(&nal.rbsp)?);
                self.tile_tables = None;
                Ok(None)
            }
            t if t.is_vcl() => self.decode_slice(nal, t),
            _ => Ok(None),
        }
    }

    /// Flush any buffered frame.
    ///
    /// For streams without B-frame reordering (P-only), all frames are
    /// emitted during `decode_nal` and `flush` returns `None`. If a
    /// picture is still in progress (all CTBs decoded but not yet
    /// finalized — shouldn't happen in normal operation), this drains
    /// it as a best-effort measure. Future phases will add proper DPB
    /// bumping for reorder delay.
    pub fn flush(&mut self) -> Option<Frame> {
        // If there's a completed (all CTBs decoded) picture still sitting
        // in current_picture, emit it. This is a defensive fallback — in
        // normal operation the picture is emitted at the end of decode_slice.
        let is_complete = self
            .current_picture
            .as_ref()
            .is_some_and(|pic| pic.ctbs_decoded >= pic.total_ctbs);
        if is_complete {
            let pic = self.current_picture.take().unwrap();
            let sps = self.sps.as_ref()?;
            let pic_width = sps.pic_width_in_luma_samples;
            let pic_height = sps.pic_height_in_luma_samples;
            let cropped_w = sps.cropped_width();
            let cropped_h = sps.cropped_height();
            let poc = pic.last_slice_header.poc;
            return Some(crop_frame(
                &pic.state,
                pic_width,
                pic_height,
                cropped_w,
                cropped_h,
                sps.conf_win_left_offset,
                sps.conf_win_top_offset,
                poc,
            ));
        }
        None
    }

    fn decode_slice(
        &mut self,
        nal: &NalUnit<'_>,
        nut: NalUnitType,
    ) -> Result<Option<Frame>, DecodeError> {
        // Phase 3c-2: resolve PPS tile geometry and cache the tile-scan
        // tables up front. We need `&mut self.pps` for `resolve_tile_geometry`
        // but `&self.sps` for the inputs — take the SPS out of the option
        // temporarily via a clone of the reference.
        {
            let sps = self
                .sps
                .as_ref()
                .ok_or(DecodeError::InvalidSyntax("slice without active SPS"))?;
            let pps = self
                .pps
                .as_mut()
                .ok_or(DecodeError::InvalidSyntax("slice without active PPS"))?;
            if pps.column_widths_in_ctbs.is_empty() {
                pps.resolve_tile_geometry(sps)?;
            }
            if self.tile_tables.is_none() {
                self.tile_tables = Some(TileScanTables::build(sps, pps));
            }
        }

        let sps = self.sps.as_ref().expect("sps present above");
        let pps = self.pps.as_ref().expect("pps present above");
        let tile_tables = self.tile_tables.as_ref().expect("tile tables built above");

        let mut sh = parse_slice_segment_header(&nal.rbsp, nut, sps, pps)?;
        sh.temporal_id = nal.temporal_id;

        // Phase 3c-4: a dependent slice segment inherits the slice-header
        // fields that the parser deliberately left at their defaults —
        // slice_type, slice_qp_y, slice_sao_*, deblock override, ... —
        // from the most recent independent slice segment in this picture.
        // The entry_point_offsets / slice_segment_address / header_size_bits
        // are independently signaled in the dependent slice's own header
        // and must NOT be overwritten.
        if sh.dependent_slice_segment_flag {
            let pic = self
                .current_picture
                .as_ref()
                .ok_or(DecodeError::InvalidSyntax(
                    "dependent slice segment without an active picture",
                ))?;
            let parent = &pic.last_independent_slice_header;
            sh.slice_type = parent.slice_type;
            sh.pic_output_flag = parent.pic_output_flag;
            sh.slice_pic_order_cnt_lsb = parent.slice_pic_order_cnt_lsb;
            sh.slice_sao_luma_flag = parent.slice_sao_luma_flag;
            sh.slice_sao_chroma_flag = parent.slice_sao_chroma_flag;
            sh.slice_qp_delta = parent.slice_qp_delta;
            sh.slice_qp_y = parent.slice_qp_y;
            sh.slice_deblocking_filter_disabled_flag = parent.slice_deblocking_filter_disabled_flag;
            sh.slice_beta_offset_div2 = parent.slice_beta_offset_div2;
            sh.slice_tc_offset_div2 = parent.slice_tc_offset_div2;
            // Inherit inter bitstream fields too, since a dependent slice
            // segment shares the logical slice of its parent.
            sh.short_term_ref_pic_set_sps_flag = parent.short_term_ref_pic_set_sps_flag;
            sh.short_term_ref_pic_set_idx = parent.short_term_ref_pic_set_idx;
            sh.short_term_rps = parent.short_term_rps.clone();
            sh.long_term_rps = parent.long_term_rps.clone();
            sh.slice_temporal_mvp_enabled_flag = parent.slice_temporal_mvp_enabled_flag;
            sh.num_ref_idx_l0_active_minus1 = parent.num_ref_idx_l0_active_minus1;
            sh.num_ref_idx_l1_active_minus1 = parent.num_ref_idx_l1_active_minus1;
            sh.mvd_l1_zero_flag = parent.mvd_l1_zero_flag;
            sh.cabac_init_flag = parent.cabac_init_flag;
            sh.collocated_from_l0_flag = parent.collocated_from_l0_flag;
            sh.collocated_ref_idx = parent.collocated_ref_idx;
            sh.max_num_merge_cand = parent.max_num_merge_cand;
            sh.ref_pic_list_modification = parent.ref_pic_list_modification.clone();
            sh.pred_weight_table = parent.pred_weight_table.clone();
            sh.poc = parent.poc;
        }

        // Phase 3d-1: compute POC from the slice header's pic_order_cnt_lsb.
        // IDR pictures always get POC = 0 (and the spec says prev_tid0_poc
        // is reset too). For non-IDR pictures we use the `compute_poc`
        // helper with the current `prev_tid0_poc` seed. Dependent slice
        // segments inherit the POC from the parent independent segment
        // (populated above) — we only compute it once per picture (on the
        // first, independent slice).
        if sh.first_slice_segment_in_pic_flag {
            if nut.is_idr() {
                sh.poc = 0;
            } else {
                sh.poc = Self::compute_poc(
                    self.prev_tid0_poc,
                    sh.slice_pic_order_cnt_lsb,
                    sps.log2_max_pic_order_cnt_lsb,
                    nut,
                );
            }
        } else if !sh.dependent_slice_segment_flag {
            // A non-first independent slice segment must have the same POC
            // as the rest of the picture; just take it from the current
            // picture in progress.
            if let Some(pic) = self.current_picture.as_ref() {
                sh.poc = pic.last_independent_slice_header.poc;
            }
        }

        // Phase 3c-4: CABAC context setup.
        //
        // - Independent slice segment → fresh `CabacContexts::init` from
        //   the slice QP and slice type, matching FFmpeg's `cabac_init_state`.
        // - Dependent slice segment → restore the contexts as they were at
        //   the end of the previous slice segment (after its last CTB's
        //   `end_of_slice_flag = 1` terminate bin), matching FFmpeg's
        //   behavior in `ff_hevc_cabac_init` which skips `cabac_init_state`
        //   when `dependent_slice_segment_flag == 1`. Exception: if the
        //   dependent slice's first CTB coincides with a WPP row start
        //   and the picture is wider than 1 CTB, use the "end of previous
        //   row's 2nd CTB" WPP save instead (spec 9.3.2.2 / FFmpeg
        //   `load_states` path in `ff_hevc_cabac_init`). Single-column
        //   pictures fall back to a fresh init for the row start, matching
        //   the `ctb_width == 1` branch in FFmpeg.
        //
        // The CABAC byte stream is freshly opened on the new NAL's RBSP
        // either way — the dependent slice has its own bitstream bytes,
        // only the context state is inherited.
        let pic_width_in_ctbs_for_init = sps.pic_width_in_ctbs_y();
        let wpp_for_init = pps.entropy_coding_sync_enabled_flag;
        let tiles_for_init = pps.tiles_enabled_flag;
        let mut contexts = if sh.dependent_slice_segment_flag {
            let pic = self.current_picture.as_ref().expect("checked above");
            // Does the dependent slice's first CTB sit on a WPP row
            // boundary? `slice_segment_address` is a tile-scan address, so
            // for single-tile pictures it equals the raster address and
            // `% pic_width_in_ctbs` gives the column. WPP is incompatible
            // with multi-tile slices in practice so we only special-case
            // the single-tile path.
            let on_row_start = !tiles_for_init
                && sh
                    .slice_segment_address
                    .is_multiple_of(pic_width_in_ctbs_for_init);
            if wpp_for_init && on_row_start && !tiles_for_init {
                if pic_width_in_ctbs_for_init == 1 {
                    CabacContexts::init(sh.slice_qp_y, sh.slice_type, sh.cabac_init_flag)
                } else {
                    let saved = pic.saved_wpp_cabac_state.ok_or(DecodeError::InvalidSyntax(
                        "dependent WPP slice at row start without a saved WPP context state",
                    ))?;
                    CabacContexts { state: saved }
                }
            } else {
                let saved = pic.saved_cabac_state.ok_or(DecodeError::InvalidSyntax(
                    "dependent slice segment without a saved CABAC state",
                ))?;
                CabacContexts { state: saved }
            }
        } else {
            CabacContexts::init(sh.slice_qp_y, sh.slice_type, sh.cabac_init_flag)
        };
        let cabac_byte_offset = sh.header_size_bits / 8;
        let mut cabac = CabacReader::new(&nal.rbsp, cabac_byte_offset);

        let ctb_size = 1u32 << sps.ctb_log2_size_y;
        let pic_width_in_ctbs = sps.pic_width_in_ctbs_y();
        let pic_height_in_ctbs = sps.pic_height_in_ctbs_y();
        let total_ctbs = pic_width_in_ctbs * pic_height_in_ctbs;

        // Phase 3c-1: a first slice segment starts a new picture. Subsequent
        // slice segments (`first_slice_segment_in_pic_flag = 0`) attach to
        // the already-in-flight picture. Phase 3c-2: `slice_segment_address`
        // is a tile-scan address (not raster), and `pic.ctbs_decoded`
        // likewise counts in tile-scan order so the continuity check below
        // still works for tiled pictures.
        if sh.first_slice_segment_in_pic_flag {
            if self.current_picture.is_some() {
                // Starting a new picture while the previous one is still
                // in flight means we missed CTBs. That's a malformed stream
                // for the Phase 3c-1 subset (no WPP / tiles, no dependent
                // slices), so bail loudly rather than silently dropping the
                // previous picture.
                return Err(DecodeError::InvalidSyntax(
                    "new first slice segment arrived while previous picture was incomplete",
                ));
            }

            // Phase 3d-6: build the reference picture lists at picture start
            // (before the CTB loop) so that motion compensation can access
            // the reference frame pixel data. Previously these were built at
            // picture completion (too late for MC).
            self.current_ref_list_l0.clear();
            self.current_ref_list_l1.clear();
            if sh.slice_type != SliceType::I {
                let sps_for_rps = self.sps.as_ref().expect("sps present");
                let sps_st_rps = sps_for_rps.st_ref_pic_sets.clone();
                let log2_max_poc_lsb = sps_for_rps.log2_max_pic_order_cnt_lsb;
                self.dpb.configure_from_sps(sps_for_rps);
                let rps = Self::derive_rps_from_slice_header_parts(&sh, &sps_st_rps);
                Self::apply_rps_marking(&rps, &self.dpb, log2_max_poc_lsb);
                self.current_rps = rps;
                let (l0, l1) = Self::build_ref_pic_lists(&self.current_rps, &self.dpb, &sh)?;
                self.current_ref_list_l0 = l0;
                self.current_ref_list_l1 = l1;
            }

            // Populate `tab_tile_id` on the fresh picture state so intra
            // availability checks can see it.
            let mut ps = PictureState::new(sps);
            let n = ps.tab_tile_id.len();
            ps.tab_tile_id.copy_from_slice(&tile_tables.tile_id[..n]);
            // QP-prediction state (spec 8.6.1 / FFmpeg hevcdec.c:3066-3069):
            // at the first independent segment of a picture, seed `qpy_pred`
            // and `last_qp_y` from `slice_qp` and arm `first_qp_group` so the
            // first QP group predicts from `slice_qp` (fallback when left /
            // above neighbors are unavailable). When `cu_qp_delta_enabled_flag
            // = 0` nothing else updates `last_qp_y`, so it correctly stays at
            // `slice_qp` for every CU.
            ps.qpy_pred = sh.slice_qp_y;
            ps.last_qp_y = sh.slice_qp_y;
            ps.first_qp_group = !sh.dependent_slice_segment_flag;
            self.current_picture = Some(PictureInProgress {
                state: ps,
                last_slice_header: sh.clone(),
                last_independent_slice_header: sh.clone(),
                saved_cabac_state: None,
                saved_wpp_cabac_state: None,
                ctbs_decoded: 0,
                total_ctbs,
            });
        } else {
            let pic = self
                .current_picture
                .as_ref()
                .ok_or(DecodeError::InvalidSyntax(
                    "non-first slice segment without an active picture",
                ))?;
            if pic.total_ctbs != total_ctbs {
                return Err(DecodeError::InvalidSyntax(
                    "slice SPS dimensions changed within picture",
                ));
            }
            if sh.slice_segment_address != pic.ctbs_decoded {
                // Slices must arrive in tile-scan order and cover a
                // contiguous range — gaps / reordering are Phase 3c-4.
                return Err(DecodeError::Unsupported(
                    "non-contiguous slice segment address (tile-scan order)",
                ));
            }
        }

        // Borrow the in-flight picture mutably for the rest of decode.
        let pic = self
            .current_picture
            .as_mut()
            .expect("current_picture set above");
        let state = &mut pic.state;

        // QP-prediction state per slice segment (spec 8.6.1 / FFmpeg
        // hevcdec.c:3066-3069). `first_qp_group` resets at every segment
        // start — to true for independent segments (fallback to slice_qp),
        // to false for dependent ones (inherit parent's `qpy_pred`). When
        // `cu_qp_delta_enabled_flag = 0` nothing else mutates `last_qp_y`,
        // so re-seed it to `slice_qp` for independent segments.
        state.first_qp_group = !sh.dependent_slice_segment_flag;
        if !pps.cu_qp_delta_enabled_flag && !sh.dependent_slice_segment_flag {
            state.last_qp_y = sh.slice_qp_y;
        }

        let wpp = pps.entropy_coding_sync_enabled_flag;
        let tiles_on = pps.tiles_enabled_flag;
        let slice_start_ts = sh.slice_segment_address;

        // Phase 3c-3 (WPP): saved CABAC context state captured after the
        // second CTB of each row, to be loaded at the start of the next row.
        // Not used by tiles — tiles reinit from the slice QP at every tile
        // boundary instead. Phase 3c-4: for dependent slices we seed this
        // from the picture-level `saved_wpp_cabac_state` so that the
        // first inner-row reinit inside the dependent slice can still find
        // a state from the previous slice's row.
        let mut saved_state: Option<[u8; crate::cabac_tables::HEVC_CONTEXTS]> =
            if sh.dependent_slice_segment_flag {
                pic.saved_wpp_cabac_state
            } else {
                None
            };

        // Phase 3d-3/3d-5: construct SliceParams for the CU tree.
        // Include POC information so the AMVP candidate list builder can
        // check same-ref and compute MV scaling.
        let ref_list_l0_pocs: Vec<i32> = self.current_ref_list_l0.iter().map(|p| p.poc).collect();
        let ref_list_l1_pocs: Vec<i32> = self.current_ref_list_l1.iter().map(|p| p.poc).collect();
        // Phase 3e: derive the collocated reference picture for temporal MVP.
        let collocated_ref = if sh.slice_temporal_mvp_enabled_flag && sh.slice_type != SliceType::I
        {
            let col_list = if sh.collocated_from_l0_flag {
                &self.current_ref_list_l0
            } else {
                &self.current_ref_list_l1
            };
            let col_idx = sh.collocated_ref_idx as usize;
            col_list.get(col_idx).cloned()
        } else {
            None
        };
        let slice_params = crate::cu_tree::SliceParams {
            slice_type: sh.slice_type,
            max_num_merge_cand: sh.max_num_merge_cand,
            num_ref_idx_l0_active: sh.num_ref_idx_l0_active_minus1 + 1,
            num_ref_idx_l1_active: if sh.slice_type == SliceType::B {
                sh.num_ref_idx_l1_active_minus1 + 1
            } else {
                0
            },
            mvd_l1_zero_flag: sh.mvd_l1_zero_flag,
            log2_parallel_merge_level: (pps.log2_parallel_merge_level_minus2 + 2) as u8,
            poc: sh.poc,
            ref_pic_list_pocs: [ref_list_l0_pocs, ref_list_l1_pocs],
            ref_frames_l0: self.current_ref_list_l0.clone(),
            ref_frames_l1: self.current_ref_list_l1.clone(),
            collocated_ref,
            slice_temporal_mvp_enabled_flag: sh.slice_temporal_mvp_enabled_flag,
            collocated_from_l0_flag: sh.collocated_from_l0_flag,
            weighted_pred_flag: (pps.weighted_pred_flag && sh.slice_type == SliceType::P)
                || (pps.weighted_bipred_flag && sh.slice_type == SliceType::B),
            pred_weight_table: sh.pred_weight_table.clone(),
        };

        let mut more_data = true;
        // Phase 3c-2: iterate in tile-scan order. For single-tile pictures
        // `ctb_addr_ts_to_rs` is the identity, so the loop visits CTBs in
        // raster order exactly as before.
        let mut ctb_addr_ts: u32 = slice_start_ts;

        // Substream index within the slice — 0 for the first substream
        // (implicit offset 0), 1 for the second (at entry_point_offsets[0]),
        // etc. Bumped every time we cross a tile boundary (tiles) or a row
        // start (WPP).
        let mut substream_idx: u32 = 0;

        while more_data && ctb_addr_ts < total_ctbs {
            let ctb_addr_rs = tile_tables.ctb_addr_ts_to_rs[ctb_addr_ts as usize];
            let col = ctb_addr_rs % pic_width_in_ctbs;
            let is_first_ctb_of_slice = ctb_addr_ts == slice_start_ts;

            // Phase 3c-2: tile boundary reinit. At the start of every tile
            // (other than the first CTB of the slice), re-init the CABAC
            // reader at the tile's entry-point byte offset and reset the
            // context state from the slice QP. The first tile of the slice
            // was already set up above.
            let is_tile_start = if is_first_ctb_of_slice {
                false
            } else {
                let prev_ts = ctb_addr_ts - 1;
                let prev_rs = tile_tables.ctb_addr_ts_to_rs[prev_ts as usize];
                tile_tables.tile_id[ctb_addr_rs as usize] != tile_tables.tile_id[prev_rs as usize]
            };

            // Phase 3c-3 (WPP): row boundary reinit. Mutually exclusive with
            // `is_tile_start` in practice because WPP entry points and tile
            // entry points share the same mechanism. For single-tile
            // pictures with WPP the row start is detected via `col == 0`.
            let is_row_start = col == 0;
            let needs_wpp_reinit = wpp && !tiles_on && is_row_start && !is_first_ctb_of_slice;

            if is_tile_start || needs_wpp_reinit {
                substream_idx += 1;
                let ep_idx = substream_idx as usize;
                if ep_idx == 0 || ep_idx > sh.entry_point_offsets.len() {
                    return Err(DecodeError::InvalidSyntax(
                        "slice missing entry_point_offset for substream",
                    ));
                }
                let byte_offset = cabac_byte_offset + sh.entry_point_offsets[ep_idx - 1] as usize;
                cabac.reinit_at(byte_offset);

                if is_tile_start {
                    // Per-tile CABAC context reinit from the slice QP.
                    contexts =
                        CabacContexts::init(sh.slice_qp_y, sh.slice_type, sh.cabac_init_flag);
                } else {
                    // WPP row start: fresh init on single-column pictures,
                    // otherwise load the state saved after the previous
                    // row's 2nd CTB.
                    if pic_width_in_ctbs == 1 {
                        contexts =
                            CabacContexts::init(sh.slice_qp_y, sh.slice_type, sh.cabac_init_flag);
                    } else if let Some(saved) = saved_state.as_ref() {
                        contexts.state.copy_from_slice(saved);
                    } else {
                        return Err(DecodeError::InvalidSyntax(
                            "WPP row start without a saved context state",
                        ));
                    }
                }
                // Spec 8.6.1 / FFmpeg hls_decode_neighbour:2712-2720: WPP row
                // starts and tile boundaries reset the first-QP-group flag so
                // the next CU with `cu_qp_delta_enabled_flag` predicts from
                // `slice_qp` rather than the previous region's trailing QP.
                state.first_qp_group = true;
            }

            let x_ctb = col * ctb_size;
            let y_ctb = (ctb_addr_rs / pic_width_in_ctbs) * ctb_size;
            // Phase 3b-2: per-CTB SAO parameters decoded BEFORE the coding tree.
            let rx = (x_ctb >> sps.ctb_log2_size_y) as usize;
            let ry = (y_ctb >> sps.ctb_log2_size_y) as usize;
            // Record the slice this CTB belongs to BEFORE decoding, so the
            // intra prediction availability check can see the current CTB's
            // slice address. `tab_slice_addr_rs` is indexed by raster.
            //
            // Phase 3c-4: for a dependent slice segment we store the
            // **independent** parent slice's segment address, not this
            // dependent segment's address. Dependent slice segments are
            // logically part of the same slice as their parent, so
            // cross-segment intra prediction MUST be allowed (spec 3.162:
            // "independent slice segment: [...] the slice segment header
            // information is not inferred from that of a preceding slice
            // segment"; dependent segments inherit and therefore extend
            // the same logical slice). Matches FFmpeg's
            // `tab_slice_address[ctb_addr_rs] = s->sh.slice_addr`, where
            // `sh->slice_addr` is only updated on independent segments
            // (hevcdec.c line 824-826).
            let recorded_slice_addr = if sh.dependent_slice_segment_flag {
                pic.last_independent_slice_header.slice_segment_address as i32
            } else {
                sh.slice_segment_address as i32
            };
            state.tab_slice_addr_rs[ctb_addr_rs as usize] = recorded_slice_addr;
            crate::sao::decode_sao_param(&mut cabac, &mut contexts, state, sps, &sh, rx, ry);
            more_data = decode_coding_quadtree(
                &mut cabac,
                &mut contexts,
                state,
                sps,
                pps,
                sh.slice_qp_y,
                &slice_params,
                x_ctb,
                y_ctb,
                sps.ctb_log2_size_y,
                0,
            )?;
            ctb_addr_ts += 1;

            // Phase 3c-3 (WPP): snapshot the CABAC contexts after the 2nd
            // CTB of each row so the next row can load them. Only active in
            // pure WPP mode — with tiles the per-tile reinit supersedes it.
            if wpp && !tiles_on {
                // For single-tile pictures raster and tile-scan agree, so
                // we can use `ctb_addr_ts` directly as the raster post-index.
                let col_after = ctb_addr_ts % pic_width_in_ctbs;
                let should_save = col_after == 2
                    || (pic_width_in_ctbs == 2 && col_after == 0)
                    || pic_width_in_ctbs == 1;
                if should_save {
                    saved_state = Some(contexts.state);
                }
            }

            // `end_of_slice_flag` (terminate bin) is decoded at the end of
            // every CTB. For non-final rows of a WPP slice / non-final tiles
            // of a tiled slice it's 0 → `more_data` stays true → we fall
            // through to the next substream, which triggers the reinit
            // block above.
        }

        // `more_data == false` means we decoded an `end_of_slice_flag = 1`
        // terminate bin — the slice has finished its CTB range. Any slice
        // must end on a terminate bin or the CABAC state is out of sync.
        if more_data {
            return Err(DecodeError::InvalidSyntax(
                "slice did not end on terminate bin",
            ));
        }

        pic.ctbs_decoded = ctb_addr_ts;
        // Phase 3c-4: snapshot the CABAC contexts at the end of the slice
        // segment so the next dependent slice segment can restore them.
        // We take the snapshot after `more_data` has gone false (i.e. after
        // the final `end_of_slice_flag = 1` terminate bin has been decoded)
        // which places us immediately after the slice's last CTB — exactly
        // the state a dependent slice segment should start from (per spec
        // 9.3.2.3 storage process for context variables).
        //
        // NOTE: `contexts.state` at this point includes the effects of the
        // terminate bin decode, which doesn't mutate the context table
        // (it's a bypass/terminate path), so the snapshot is equivalent to
        // the "after last CTB" state the spec asks for.
        pic.saved_cabac_state = Some(contexts.state);
        // Persist the WPP row-state snapshot so a subsequent dependent
        // slice segment starting on a WPP row boundary can restore it.
        pic.saved_wpp_cabac_state = saved_state;
        if !sh.dependent_slice_segment_flag {
            pic.last_independent_slice_header = sh.clone();
        }
        pic.last_slice_header = sh;

        if pic.ctbs_decoded != total_ctbs {
            // More slice segments still to come for this picture.
            return Ok(None);
        }

        // Picture complete — run in-loop filters and emit the frame.
        let mut pic = self
            .current_picture
            .take()
            .expect("current_picture taken after completion");
        let last_sh = &pic.last_slice_header;

        // Phase 3b-1: in-loop deblocking filter.
        if !last_sh.slice_deblocking_filter_disabled_flag {
            crate::deblock::deblock_picture(&mut pic.state, sps, pps, last_sh);
        }

        // Phase 3b-2: SAO filter (after deblocking).
        crate::sao::apply_sao_picture(&mut pic.state, sps, last_sh);

        // Phase 3d-1: snapshot everything we need from the SPS and the
        // completed slice header so we can release the outstanding
        // `sps` / `pps` / `tile_tables` borrows (they're tied to `self`)
        // and mutably touch `self.dpb` / `self.prev_tid0_poc` /
        // `self.current_rps`.
        let pic_width = sps.pic_width_in_luma_samples;
        let pic_height = sps.pic_height_in_luma_samples;
        let log2_max_poc_lsb = sps.log2_max_pic_order_cnt_lsb;
        let sps_st_ref_pic_sets = sps.st_ref_pic_sets.clone();
        let last_sh_cloned = last_sh.clone();
        let picture_poc = last_sh_cloned.poc;

        // Build the output Frame now (while `sps`/etc. are still in
        // scope) so the `pic` state can be moved into the DPB afterwards.
        // Apply the conformance window crop so the output dimensions match
        // the visible picture (not the CTU-padded coded picture).
        let cropped_w = sps.cropped_width();
        let cropped_h = sps.cropped_height();
        let emitted_frame = crop_frame(
            &pic.state,
            pic_width,
            pic_height,
            cropped_w,
            cropped_h,
            sps.conf_win_left_offset,
            sps.conf_win_top_offset,
            picture_poc,
        );

        // The borrows `sps` / `pps` / `tile_tables` / `last_sh` are no
        // longer used beyond this point — NLL will release them here so
        // we can take mutable borrows of `self.*` below.

        // Phase 3d-1: configure DPB from the active SPS (idempotent) and
        // derive the current picture's reference picture set. The DPB
        // still isn't used for actual decoding at this phase, but we wire
        // the bookkeeping so the scaffolding is in place.
        {
            // Fresh `&Sps` scoped to this block only.
            let sps_for_dpb = self.sps.as_ref().expect("sps still present after decode");
            self.dpb.configure_from_sps(sps_for_dpb);
        }

        self.current_rps =
            Self::derive_rps_from_slice_header_parts(&last_sh_cloned, &sps_st_ref_pic_sets);
        // Mark DPB pictures based on the newly derived RPS.
        Self::apply_rps_marking(&self.current_rps, &self.dpb, log2_max_poc_lsb);

        // Phase 3d-2: build RefPicList0 / RefPicList1 per spec 8.3.2.
        // I slices never reference other pictures → leave both lists empty.
        // P/B slices use the lists for merge/AMVP candidate derivation
        // (Phase 3d-4/3d-5).
        self.current_ref_list_l0.clear();
        self.current_ref_list_l1.clear();
        if last_sh_cloned.slice_type != SliceType::I {
            let (l0, l1) =
                Self::build_ref_pic_lists(&self.current_rps, &self.dpb, &last_sh_cloned)?;
            self.current_ref_list_l0 = l0;
            self.current_ref_list_l1 = l1;
        }

        // Phase 3d-1: update prev_tid0_poc per spec 8.3.1 — done BEFORE
        // the picture is inserted into the DPB, so subsequent non-IDR
        // slice headers will see the current picture's POC as the seed.
        //
        // The spec excludes sub-layer non-reference (N-type) NAL types
        // and RASL/RADL pictures. For Phase 3d-1 we only see IDR pictures
        // in practice, so we take the simple approximation described in
        // the phase plan: update on every IDR or temporal_id == 0.
        let nut_for_tid0 = last_sh_cloned.nal_unit_type;
        let is_tid0 = last_sh_cloned.temporal_id == 0
            && !matches!(
                nut_for_tid0,
                NalUnitType::TrailN
                    | NalUnitType::TsaN
                    | NalUnitType::StsaN
                    | NalUnitType::RadlN
                    | NalUnitType::RadlR
                    | NalUnitType::RaslN
                    | NalUnitType::RaslR
            );
        if nut_for_tid0.is_irap() {
            self.prev_tid0_poc = 0;
        } else if is_tid0 {
            self.prev_tid0_poc = last_sh_cloned.poc;
        }

        // Insert the fresh picture into the DPB so later frames can look
        // it up by POC. Also clean up any unreferenced + already-output
        // pictures (none exist today; this is future-proofing).
        // Phase 3e: store tab_mvf + ref list POCs for temporal MVP.
        let ref_list_l0_pocs: Vec<i32> = self.current_ref_list_l0.iter().map(|p| p.poc).collect();
        let ref_list_l1_pocs: Vec<i32> = self.current_ref_list_l1.iter().map(|p| p.poc).collect();
        let decoded_pic = Rc::new(DecodedPicture::new_with_mvf(
            pic.state.y_plane,
            pic.state.u_plane,
            pic.state.v_plane,
            pic_width,
            pic_height,
            picture_poc,
            pic.state.tab_mvf,
            pic.state.log2_min_pu_size,
            pic.state.min_pu_width,
            pic.state.log2_ctb_size,
            [ref_list_l0_pocs, ref_list_l1_pocs],
        ));
        // Mark as short-term reference initially — a subsequent frame's
        // RPS will flip it to long-term / unused as needed.
        decoded_pic.mark(PictureReferenceStatus::ShortTerm);
        // Mark as output immediately (Phase 3d-1 has no reorder buffer).
        *decoded_pic.output.borrow_mut() = true;
        self.dpb.insert(decoded_pic);
        self.dpb.cleanup_unused();

        Ok(Some(emitted_frame))
    }

    /// Phase 3d-1: compute a picture's POC from `prev_tid0_poc`, the
    /// slice header's `pic_order_cnt_lsb`, and the SPS's
    /// `log2_max_pic_order_cnt_lsb`. Follows HEVC spec 8.3.1 + FFmpeg's
    /// `ff_hevc_compute_poc2` in `ps.c`.
    ///
    /// For BLA pictures the POC MSB is explicitly reset to 0, matching
    /// FFmpeg's behavior.
    fn compute_poc(
        prev_tid0_poc: i32,
        slice_pic_order_cnt_lsb: u32,
        log2_max_pic_order_cnt_lsb: u8,
        nal_unit_type: NalUnitType,
    ) -> i32 {
        let max_poc_lsb = 1i32 << log2_max_pic_order_cnt_lsb;
        let prev_poc_lsb = prev_tid0_poc.rem_euclid(max_poc_lsb);
        let prev_poc_msb = prev_tid0_poc - prev_poc_lsb;
        let cur_lsb = slice_pic_order_cnt_lsb as i32;
        let mut poc_msb = if cur_lsb < prev_poc_lsb && (prev_poc_lsb - cur_lsb) >= max_poc_lsb / 2 {
            prev_poc_msb + max_poc_lsb
        } else if cur_lsb > prev_poc_lsb && (cur_lsb - prev_poc_lsb) > max_poc_lsb / 2 {
            prev_poc_msb - max_poc_lsb
        } else {
            prev_poc_msb
        };
        // BLA pictures: POC MSB forced to 0 per spec 8.3.1.
        if matches!(
            nal_unit_type,
            NalUnitType::BlaWLp | NalUnitType::BlaWRadl | NalUnitType::BlaNLp
        ) {
            poc_msb = 0;
        }
        poc_msb + cur_lsb
    }

    /// Phase 3d-1: derive the reference picture sets for the current
    /// picture from the slice header and the (cloned) SPS ST-RPS array.
    /// This follows spec 8.3.2 but simplified — we don't yet track the
    /// "follow" sets precisely or complain about missing refs, since no
    /// actual decoding uses them yet. Good enough to have the POC lists
    /// plumbed so Phase 3d-2 can pick them up.
    ///
    /// Takes an owned slice of SPS ST-RPSs rather than `&Sps` so the
    /// caller can drop its immutable borrow of `self.sps` and mutably
    /// touch `self.dpb` in the surrounding code. A real
    /// spec-parity derivation would also consume the DPB to walk the
    /// existing picture pool, but Phase 3d-1 only needs POC lists.
    fn derive_rps_from_slice_header_parts(
        sh: &SliceHeader,
        sps_st_ref_pic_sets: &[ShortTermRps],
    ) -> ReferencePictureSets {
        let mut rps = ReferencePictureSets::default();

        // IDR pictures never reference anything from the DPB.
        if sh.nal_unit_type.is_idr() {
            return rps;
        }

        // Resolve the active short-term RPS.
        let st_ref: Option<&ShortTermRps> = if sh.short_term_ref_pic_set_sps_flag {
            sps_st_ref_pic_sets.get(sh.short_term_ref_pic_set_idx as usize)
        } else {
            sh.short_term_rps.as_ref()
        };

        if let Some(st) = st_ref {
            // Short-term "before current" (negative deltas the current
            // picture uses).
            for (i, delta) in st.delta_poc_s0.iter().enumerate() {
                let ref_poc = sh.poc + delta;
                if st.used_by_curr_pic_s0_flag.get(i).copied().unwrap_or(false) {
                    rps.st_curr_before.push(ref_poc);
                } else {
                    rps.st_foll.push(ref_poc);
                }
            }
            // Short-term "after current" (positive deltas the current
            // picture uses).
            for (i, delta) in st.delta_poc_s1.iter().enumerate() {
                let ref_poc = sh.poc + delta;
                if st.used_by_curr_pic_s1_flag.get(i).copied().unwrap_or(false) {
                    rps.st_curr_after.push(ref_poc);
                } else {
                    rps.st_foll.push(ref_poc);
                }
            }
        }

        // Long-term references: lookup by POC LSB. Each entry either goes
        // into `lt_curr` (used by current picture) or `lt_foll` (not
        // used).
        for (i, poc_lsb) in sh.long_term_rps.poc_lsb_lt.iter().enumerate() {
            // For Phase 3d-1 we just record the LSB as the full POC
            // (good enough for set membership; real decoding will redo
            // the MSB computation).
            let ref_poc = *poc_lsb as i32;
            if sh
                .long_term_rps
                .used_by_curr_pic_lt_flag
                .get(i)
                .copied()
                .unwrap_or(false)
            {
                rps.lt_curr.push(ref_poc);
            } else {
                rps.lt_foll.push(ref_poc);
            }
        }

        rps
    }

    /// Phase 3d-2: build `RefPicList0` / `RefPicList1` for a P or B slice
    /// per HEVC spec 8.3.2. Caller must ensure the slice is P or B (I
    /// slices get empty lists).
    ///
    /// Returns `(L0, L1)` where L1 is empty for P slices. Each list has
    /// exactly `num_ref_idx_l{0,1}_active_minus1 + 1` entries and every
    /// entry points to a picture already in the DPB.
    #[allow(clippy::type_complexity)]
    fn build_ref_pic_lists(
        rps: &ReferencePictureSets,
        dpb: &DecodedPictureBuffer,
        sh: &SliceHeader,
    ) -> Result<(Vec<Rc<DecodedPicture>>, Vec<Rc<DecodedPicture>>), DecodeError> {
        let num_poc_total_curr = rps.num_poc_total_curr();
        let active_l0 = sh.num_ref_idx_l0_active_minus1 as usize + 1;
        let num_rps_curr_temp_list0 = active_l0.max(num_poc_total_curr);
        let temp0 = build_ref_pic_list_temp0(rps, num_rps_curr_temp_list0);
        let l0_pocs = apply_ref_pic_list_modification(
            &temp0,
            sh.ref_pic_list_modification
                .ref_pic_list_modification_flag_l0,
            &sh.ref_pic_list_modification.list_entry_l0,
            active_l0,
        )?;
        let l0 = resolve_ref_pics(dpb, &l0_pocs)?;

        let l1 = if sh.slice_type == SliceType::B {
            let active_l1 = sh.num_ref_idx_l1_active_minus1 as usize + 1;
            let num_rps_curr_temp_list1 = active_l1.max(num_poc_total_curr);
            let temp1 = build_ref_pic_list_temp1(rps, num_rps_curr_temp_list1);
            let l1_pocs = apply_ref_pic_list_modification(
                &temp1,
                sh.ref_pic_list_modification
                    .ref_pic_list_modification_flag_l1,
                &sh.ref_pic_list_modification.list_entry_l1,
                active_l1,
            )?;
            resolve_ref_pics(dpb, &l1_pocs)?
        } else {
            Vec::new()
        };

        Ok((l0, l1))
    }

    /// Phase 3d-1: walk the DPB and mark each picture according to the
    /// current RPS. Pictures referenced by the RPS become ShortTerm /
    /// LongTerm; everything else is flipped to UnusedForReference.
    fn apply_rps_marking(
        rps: &ReferencePictureSets,
        dpb: &DecodedPictureBuffer,
        log2_max_pic_order_cnt_lsb: u8,
    ) {
        let max_poc_lsb = 1i32 << log2_max_pic_order_cnt_lsb;
        // Start from "unused" and flip back to ST/LT for any picture
        // actually in the RPS. This is the spec's derivation order.
        dpb.unmark_all_references();
        for pic in dpb.pictures() {
            let poc = pic.poc;
            let poc_lsb = poc.rem_euclid(max_poc_lsb);
            if rps.st_curr_before.contains(&poc)
                || rps.st_curr_after.contains(&poc)
                || rps.st_foll.contains(&poc)
            {
                pic.mark(PictureReferenceStatus::ShortTerm);
            } else if rps.lt_curr.contains(&poc_lsb) || rps.lt_foll.contains(&poc_lsb) {
                pic.mark(PictureReferenceStatus::LongTerm);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::parse_annex_b;

    /// Phase 3d-1: hand-verify the POC computation against the spec
    /// formula for a grab-bag of representative cases. Matches FFmpeg's
    /// `ff_hevc_compute_poc2` (ps.c) line-for-line.
    #[test]
    fn poc_compute_monotonic_lsb_no_wrap() {
        // log2_max_poc_lsb = 4 → max_poc_lsb = 16.
        // prev_tid0_poc = 4 (MSB=0, LSB=4), cur LSB = 5.
        // prev_lsb = 4, diff = 1 → neither branch fires, poc_msb = 0.
        // poc = 0 + 5 = 5.
        assert_eq!(Decoder::compute_poc(4, 5, 4, NalUnitType::TrailR), 5);
    }

    #[test]
    fn poc_compute_wraps_forward_across_msb() {
        // max_poc_lsb = 16; prev_tid0_poc = 15 (MSB=0, LSB=15), cur LSB=1.
        // cur_lsb > prev_lsb but (cur_lsb - prev_lsb) = -14 after signed
        // subtraction... we check the spec condition:
        //   cur_lsb < prev_lsb (1 < 15) && (prev_lsb - cur_lsb) >= max_poc_lsb/2
        //   (15 - 1) = 14 >= 8 → yes, so poc_msb = prev_poc_msb + max_poc_lsb = 16.
        //   poc = 16 + 1 = 17.
        assert_eq!(Decoder::compute_poc(15, 1, 4, NalUnitType::TrailR), 17);
    }

    #[test]
    fn poc_compute_wraps_backward_across_msb() {
        // Classic "LSB went backwards" case (e.g. B-frame references
        // the IDR at POC=16 with prev_tid0_poc=17 and cur LSB=0).
        //
        // max_poc_lsb = 16, prev_tid0_poc = 17 (MSB=16, LSB=1), cur_lsb = 15.
        // Neither "cur_lsb < prev_lsb && (prev - cur) >= 8" nor
        // "cur_lsb > prev_lsb && (cur - prev) > 8" holds:
        //   15 > 1, 15 - 1 = 14 > 8 → backward wrap branch fires.
        //   poc_msb = 16 - 16 = 0.
        //   poc = 0 + 15 = 15.
        assert_eq!(Decoder::compute_poc(17, 15, 4, NalUnitType::TrailR), 15);
    }

    #[test]
    fn poc_compute_bla_forces_msb_zero() {
        // BLA picture: POC MSB is explicitly reset to 0 regardless of
        // `prev_tid0_poc`. With prev = 20 (MSB=16, LSB=4) and cur LSB=5,
        // a non-BLA picture would give poc = 16 + 5 = 21, but a
        // BLA_W_LP explicitly sets poc_msb to 0 → poc = 5.
        assert_eq!(Decoder::compute_poc(20, 5, 4, NalUnitType::BlaWLp), 5);
        // For TrailR the normal path returns 21.
        assert_eq!(Decoder::compute_poc(20, 5, 4, NalUnitType::TrailR), 21);
    }

    #[test]
    fn poc_compute_equal_lsb_passes_through() {
        // prev_tid0_poc = 32 (MSB=32, LSB=0), cur LSB=0 → poc_msb=32.
        assert_eq!(Decoder::compute_poc(32, 0, 4, NalUnitType::TrailR), 32);
    }

    // ----- Phase 3d-2: RefPicList0 / RefPicList1 construction tests -----

    use crate::slice::{LongTermRefPicSet, RefPicListModification};

    /// Build a minimal `SliceHeader` for tests that only exercise the
    /// `build_ref_pic_lists` path. Only the fields the function reads
    /// need to be valid; the rest take default / placeholder values.
    fn test_slice_header_for_ref_lists(
        slice_type: SliceType,
        num_ref_idx_l0_active_minus1: u32,
        num_ref_idx_l1_active_minus1: u32,
        ref_pic_list_modification: RefPicListModification,
    ) -> SliceHeader {
        SliceHeader {
            first_slice_segment_in_pic_flag: true,
            no_output_of_prior_pics_flag: false,
            slice_pic_parameter_set_id: 0,
            dependent_slice_segment_flag: false,
            slice_segment_address: 0,
            slice_type,
            pic_output_flag: true,
            slice_pic_order_cnt_lsb: 0,
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            slice_qp_delta: 0,
            slice_qp_y: 26,
            slice_deblocking_filter_disabled_flag: true,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            entry_point_offsets: Vec::new(),
            header_size_bits: 0,
            short_term_ref_pic_set_sps_flag: false,
            short_term_ref_pic_set_idx: 0,
            short_term_rps: None,
            long_term_rps: LongTermRefPicSet::default(),
            slice_temporal_mvp_enabled_flag: false,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            mvd_l1_zero_flag: false,
            cabac_init_flag: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            max_num_merge_cand: 5,
            ref_pic_list_modification,
            pred_weight_table: crate::slice::PredWeightTable::default(),
            nal_unit_type: NalUnitType::TrailR,
            temporal_id: 0,
            poc: 10,
        }
    }

    fn test_decoded_picture(poc: i32) -> Rc<DecodedPicture> {
        Rc::new(DecodedPicture::new(vec![], vec![], vec![], 16, 16, poc))
    }

    #[test]
    fn build_ref_pic_lists_p_slice_without_modification() {
        // RPS: st_curr_before = [8, 6], st_curr_after = [], lt_curr = [].
        // Current POC = 10, 2 L0 refs → L0 = [pic@8, pic@6].
        let rps = ReferencePictureSets {
            st_curr_before: vec![8, 6],
            st_curr_after: vec![],
            st_foll: vec![],
            lt_curr: vec![],
            lt_foll: vec![],
        };
        let mut dpb = DecodedPictureBuffer::new();
        dpb.insert(test_decoded_picture(8));
        dpb.insert(test_decoded_picture(6));

        let sh = test_slice_header_for_ref_lists(
            SliceType::P,
            1, // num_ref_idx_l0_active_minus1 = 1 → 2 active refs
            0,
            RefPicListModification::default(),
        );

        let (l0, l1) = Decoder::build_ref_pic_lists(&rps, &dpb, &sh).expect("build lists");
        assert_eq!(l0.len(), 2);
        assert_eq!(l0[0].poc, 8);
        assert_eq!(l0[1].poc, 6);
        assert!(l1.is_empty(), "P slice has no L1");
    }

    #[test]
    fn build_ref_pic_lists_b_slice_l1_uses_after_before_order() {
        // RPS: before = [8], after = [12]. For L0 the temp order is
        // [before, after] → [8, 12]; for L1 it's [after, before] → [12, 8].
        let rps = ReferencePictureSets {
            st_curr_before: vec![8],
            st_curr_after: vec![12],
            st_foll: vec![],
            lt_curr: vec![],
            lt_foll: vec![],
        };
        let mut dpb = DecodedPictureBuffer::new();
        dpb.insert(test_decoded_picture(8));
        dpb.insert(test_decoded_picture(12));

        let sh = test_slice_header_for_ref_lists(
            SliceType::B,
            1, // 2 L0 refs
            1, // 2 L1 refs
            RefPicListModification::default(),
        );

        let (l0, l1) = Decoder::build_ref_pic_lists(&rps, &dpb, &sh).expect("build lists");
        assert_eq!(l0.len(), 2);
        assert_eq!(l0[0].poc, 8);
        assert_eq!(l0[1].poc, 12);
        assert_eq!(l1.len(), 2);
        assert_eq!(l1[0].poc, 12);
        assert_eq!(l1[1].poc, 8);
    }

    #[test]
    fn build_ref_pic_lists_applies_modification() {
        let rps = ReferencePictureSets {
            st_curr_before: vec![8, 6],
            st_curr_after: vec![],
            st_foll: vec![],
            lt_curr: vec![],
            lt_foll: vec![],
        };
        let mut dpb = DecodedPictureBuffer::new();
        dpb.insert(test_decoded_picture(8));
        dpb.insert(test_decoded_picture(6));

        // list_entry_l0 = [1, 0] → swap: result = [pic@6, pic@8].
        let modif = RefPicListModification {
            ref_pic_list_modification_flag_l0: true,
            list_entry_l0: vec![1, 0],
            ..Default::default()
        };

        let sh = test_slice_header_for_ref_lists(SliceType::P, 1, 0, modif);
        let (l0, _l1) = Decoder::build_ref_pic_lists(&rps, &dpb, &sh).expect("build lists");
        assert_eq!(l0.len(), 2);
        assert_eq!(l0[0].poc, 6);
        assert_eq!(l0[1].poc, 8);
    }

    #[test]
    fn build_ref_pic_lists_temp_list_wraps_for_high_active_count() {
        // RPS has only 2 pictures but num_ref_idx_l0_active = 5 → the temp
        // list wraps: [8, 6, 8, 6, 8], then first 5 are taken as-is.
        let rps = ReferencePictureSets {
            st_curr_before: vec![8, 6],
            st_curr_after: vec![],
            st_foll: vec![],
            lt_curr: vec![],
            lt_foll: vec![],
        };
        let mut dpb = DecodedPictureBuffer::new();
        dpb.insert(test_decoded_picture(8));
        dpb.insert(test_decoded_picture(6));

        let sh = test_slice_header_for_ref_lists(
            SliceType::P,
            4, // 5 active L0 refs
            0,
            RefPicListModification::default(),
        );
        let (l0, _l1) = Decoder::build_ref_pic_lists(&rps, &dpb, &sh).expect("build lists");
        assert_eq!(l0.len(), 5);
        let pocs: Vec<i32> = l0.iter().map(|p| p.poc).collect();
        assert_eq!(pocs, vec![8, 6, 8, 6, 8]);
    }

    #[test]
    fn build_ref_pic_lists_errors_when_poc_missing_from_dpb() {
        let rps = ReferencePictureSets {
            st_curr_before: vec![8],
            st_curr_after: vec![],
            st_foll: vec![],
            lt_curr: vec![],
            lt_foll: vec![],
        };
        // DPB is empty → the POC=8 lookup fails.
        let dpb = DecodedPictureBuffer::new();
        let sh =
            test_slice_header_for_ref_lists(SliceType::P, 0, 0, RefPicListModification::default());
        assert!(Decoder::build_ref_pic_lists(&rps, &dpb, &sh).is_err());
    }

    /// **Phase 2d byte-exact test**: feed `testdata/tiny_intra.h265` through
    /// `Decoder::decode_nal` and assert the resulting `Frame.y/u/v` matches
    /// `testdata/tiny_intra_ref.yuv` byte-for-byte.
    ///
    /// Reference YUV layout (`ffmpeg -i tiny_intra.h265 -f rawvideo
    /// -pix_fmt yuv420p tiny_intra_ref.yuv`):
    ///
    /// - 256 bytes of luma (16×16) all `0x7E`
    /// - 64 bytes of Cb (8×8) all `0x80`
    /// - 64 bytes of Cr (8×8) all `0x80`
    /// - 384 bytes total
    #[test]
    fn test_decode_tiny_intra_byte_exact() {
        let h265_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tiny_intra.h265");
        let yuv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tiny_intra_ref.yuv");

        let h265 = std::fs::read(h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(yuv_path).expect("read reference yuv");

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();

        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");

        assert_eq!(frame.width, 16);
        assert_eq!(frame.height, 16);
        assert_eq!(frame.y.len(), 256);
        assert_eq!(frame.u.len(), 64);
        assert_eq!(frame.v.len(), 64);

        // Reassemble in the same layout as the reference YUV (Y then U then V).
        let mut decoded = Vec::with_capacity(384);
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );
        assert_eq!(
            decoded, ref_yuv,
            "decoded planes do not match reference YUV byte-for-byte"
        );
    }

    /// **Phase 3b-2 byte-exact test**: 16×16 flat-gray with SAO enabled
    /// (no `--no-sao`). Tests:
    ///
    /// - `sample_adaptive_offset_enabled_flag = 1` SPS path
    /// - `slice_sao_luma_flag` / `slice_sao_chroma_flag` parsing in slice header
    /// - Per-CTB `decode_sao_param` parsing (merge flags, type_idx, offsets,
    ///   eo_class / band_position) at the start of each CTU
    /// - `apply_sao_picture` running over the picture after deblock
    #[test]
    fn test_decode_sao_byte_exact() {
        let h265_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/sao.h265");
        let yuv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/sao_ref.yuv");
        let h265 = std::fs::read(h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(yuv_path).expect("read reference yuv");
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");
        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);
        assert_eq!(
            decoded, ref_yuv,
            "decoded planes do not match reference YUV byte-for-byte"
        );
    }

    /// **Phase 3b-1 byte-exact test**: 32×32 horizontal gradient with
    /// deblocking enabled (no `--no-deblock`). Tests:
    ///
    /// - `slice_deblocking_filter_disabled_flag` plumbing through the slice header
    /// - Per-TU boundary strength marking (intra → bS=2)
    /// - Per-min-CB QP table population
    /// - Luma and chroma deblock filters running over the picture
    #[test]
    fn test_decode_deblock_grad_byte_exact() {
        let h265_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/deblock_grad.h265");
        let yuv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/deblock_grad_ref.yuv");
        let h265 = std::fs::read(h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(yuv_path).expect("read reference yuv");
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");
        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);
        assert_eq!(
            decoded, ref_yuv,
            "decoded planes do not match reference YUV byte-for-byte"
        );
    }

    // Phase 3a-2 4×4 fixture: deferred to Phase 3a-3.
    //
    // x265 with `--max-tu-size 4` on a flat input picks angular intra
    // modes for some 4×4 PUs (visible as gradient patterns in the reference
    // YUV). Validating the 4×4 luma DST end-to-end therefore requires
    // angular intra prediction, which is the next sub-phase. The DST
    // implementation in `inverse_transform::transform_4x4_luma` is correct
    // (it mirrors FFmpeg's `transform_4x4_luma` line-for-line), but
    // exercising it through the full pipeline waits for 3a-3.

    /// **Phase 3a-2 byte-exact test**: 16×16 flat-gray frame with
    /// `--ctu 16 --max-tu-size 8` → 4 CTUs at 16×16, each split into
    /// 4 8×8 luma TUs (transform_tree at log2_trafo=4 has implicit
    /// `split_transform_flag=1` because log2_trafo > max_tb=3). Tests:
    ///
    /// - Recursive `transform_tree` split at log2_trafo > max_tb
    /// - 8×8 inverse DCT (`idct_8x8`)
    /// - 8×8 residual_coding with the 2×2 sub-block scan (`DIAG_SCAN_2X2`)
    /// - `last_significant_coeff_x/y_prefix` for `log2_size = 3`
    /// - `sig_coeff_flag` `scf_offset` for `log2_trafo == 3` (different
    ///   from `log2_trafo == 4` we already covered)
    #[test]
    fn test_decode_tu8_byte_exact() {
        let h265_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tu8.h265");
        let yuv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tu8_ref.yuv");
        let h265 = std::fs::read(h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(yuv_path).expect("read reference yuv");
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");
        let mut decoded = Vec::with_capacity(1536);
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);
        assert_eq!(
            decoded, ref_yuv,
            "decoded planes do not match reference YUV byte-for-byte"
        );
    }

    /// **Phase 3a-2 byte-exact test**: 32×32 flat-gray frame with
    /// `--ctu 32 --max-tu-size 32` → single CTU, single CU, single 32×32
    /// luma TU. Tests:
    ///
    /// - 32×32 inverse DCT (`idct_32x32` / `idct_dc` for the DC fast path)
    /// - 32×32 residual_coding with the 8×8 sub-block scan (`DIAG_SCAN_8X8`)
    /// - `last_significant_coeff_x/y_prefix` context derivation for `log2_size = 5`
    /// - Dequantization with `shift = bit_depth + log2_trafo_size - 5 = 8`
    #[test]
    fn test_decode_tu32_byte_exact() {
        let h265_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tu32.h265");
        let yuv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tu32_ref.yuv");
        let h265 = std::fs::read(h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(yuv_path).expect("read reference yuv");
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");
        assert_eq!(frame.width, 32);
        assert_eq!(frame.height, 32);
        let mut decoded = Vec::with_capacity(1536);
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);
        assert_eq!(
            decoded, ref_yuv,
            "decoded planes do not match reference YUV byte-for-byte"
        );
    }

    /// **Phase 3a-1 byte-exact test**: 32×32 flat-gray frame, `--ctu 16` →
    /// 4 CTUs in raster order. Tests:
    ///
    /// - Multi-CTU loop in `Decoder::decode_slice`
    /// - `end_of_slice_flag` (terminate bin) decoded at each CTB boundary
    /// - `decode_coding_quadtree` returning a `more_data` flag
    /// - Partial-availability reference samples (CTUs 2/3/4 have decoded
    ///   neighbors from the earlier CTUs)
    ///
    /// Reference YUV is 1536 bytes: 1024 luma (all 0x7E) + 256 Cb (0x80) +
    /// 256 Cr (0x80).
    #[test]
    fn test_decode_multi_ctu_byte_exact() {
        let h265_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/multi_ctu.h265");
        let yuv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/multi_ctu_ref.yuv");

        let h265 = std::fs::read(h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(yuv_path).expect("read reference yuv");

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");

        assert_eq!(frame.width, 32);
        assert_eq!(frame.height, 32);
        assert_eq!(frame.y.len(), 1024);
        assert_eq!(frame.u.len(), 256);
        assert_eq!(frame.v.len(), 256);

        let mut decoded = Vec::with_capacity(1536);
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);
        assert_eq!(
            decoded, ref_yuv,
            "decoded planes do not match reference YUV byte-for-byte"
        );
    }

    /// **Phase 3a-3 byte-exact test**: 16x16 flat-gray frame with
    /// `--ctu 16 --max-tu-size 4` to force angular intra prediction modes.
    ///
    /// Fixture: `testdata/angular.h265` + `testdata/angular_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 16x16 flat gray YUV input (luma=0x7E, chroma=128)
    /// x265 --input angular_input.yuv --input-res 16x16 --fps 1 --frames 1 \
    ///   --preset ultrafast --no-wpp --no-signhide --ctu 16 --max-tu-size 4 \
    ///   --no-open-gop --keyint 1 --no-scenecut --no-sao --no-deblock \
    ///   --qp 25 --no-psnr --no-ssim --no-info -o angular.h265
    /// ffmpeg -i angular.h265 -f rawvideo -pix_fmt yuv420p angular_ref.yuv
    /// ```
    #[test]
    fn test_decode_angular_byte_exact() {
        let h265 = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/angular.h265"
        ))
        .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/angular_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 16;
        let h: usize = 16;

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");

        assert_eq!(frame.width as usize, w);
        assert_eq!(frame.height as usize, h);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );

        if decoded != ref_yuv {
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let plane = if i < w * h {
                        "Y"
                    } else if i < w * h + (w / 2) * (h / 2) {
                        "U"
                    } else {
                        "V"
                    };
                    panic!(
                        "mismatch at byte {} (plane {}) ours={} ref={}",
                        i, plane, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3a-3 byte-exact test**: 16x16 vertical-stripe frame with
    /// `--ctu 16 --max-tu-size 4 --qp 30` to force angular intra prediction
    /// modes (modes 2..34). The stripe pattern causes x265 to pick modes like 3
    /// and 34 for many PUs within a single CTU.
    ///
    /// Fixture: `testdata/angular_grad.h265` + `testdata/angular_grad_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 16x16 vertical stripe YUV input (left=40, right=200, chroma=128)
    /// x265 --input angular_grad_input.yuv --input-res 16x16 --fps 1 --frames 1 \
    ///   --preset ultrafast --no-wpp --no-signhide --ctu 16 --max-tu-size 4 \
    ///   --no-open-gop --keyint 1 --no-scenecut --no-sao --no-deblock \
    ///   --qp 30 --no-psnr --no-ssim --no-info -o angular_grad.h265
    /// ffmpeg -i angular_grad.h265 -f rawvideo -pix_fmt yuv420p angular_grad_ref.yuv
    /// ```
    #[test]
    fn test_decode_angular_gradient_byte_exact() {
        let h265 = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/angular_grad.h265"
        ))
        .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/angular_grad_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 16;
        let h: usize = 16;

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");

        assert_eq!(frame.width as usize, w);
        assert_eq!(frame.height as usize, h);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );

        if decoded != ref_yuv {
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let plane = if i < w * h {
                        "Y"
                    } else if i < w * h + (w / 2) * (h / 2) {
                        "U"
                    } else {
                        "V"
                    };
                    panic!(
                        "mismatch at byte {} (plane {}) ours={} ref={}",
                        i, plane, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3a-4 byte-exact test**: 16x16 flat-gray frame with
    /// `--scaling-list default` to enable the HEVC default scaling matrices.
    ///
    /// This exercises:
    /// - `scaling_list_enabled_flag = 1` in SPS (no longer rejected)
    /// - Default scaling list construction (spec tables 7-3..7-6)
    /// - Scaling matrix lookup in `residual_coding` dequantization
    /// - DC scale for 16x16 TUs (`sl_dc`)
    /// - Position downsampling for 16x16: `pos = ((y>>1)<<3) + (x>>1)`
    ///
    /// Fixture: `testdata/scaling_list.h265` + `testdata/scaling_list_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 16x16 flat gray YUV input (luma=0x7E, chroma=128)
    /// x265 --input scaling_list_input.yuv --input-res 16x16 --fps 1 --frames 1 \
    ///   --preset ultrafast --no-wpp --no-signhide --ctu 16 \
    ///   --no-open-gop --keyint 1 --no-scenecut --no-sao --no-deblock \
    ///   --qp 25 --no-psnr --no-ssim --no-info --scaling-list default \
    ///   -o scaling_list.h265
    /// ffmpeg -i scaling_list.h265 -f rawvideo -pix_fmt yuv420p scaling_list_ref.yuv
    /// ```
    #[test]
    fn test_decode_scaling_list_default_byte_exact() {
        let h265 = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/scaling_list.h265"
        ))
        .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/scaling_list_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 16;
        let h: usize = 16;

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");

        assert_eq!(frame.width as usize, w);
        assert_eq!(frame.height as usize, h);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );

        if decoded != ref_yuv {
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let plane = if i < w * h {
                        "Y"
                    } else if i < w * h + (w / 2) * (h / 2) {
                        "U"
                    } else {
                        "V"
                    };
                    panic!(
                        "mismatch at byte {} (plane {}) ours={} ref={}",
                        i, plane, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3a-6 byte-exact test**: 16x16 vertical-stripe frame encoded
    /// with `--signhide`, so `pps_sign_data_hiding_enabled_flag` is set.
    ///
    /// Exercises:
    /// - `sign_data_hiding_enabled_flag = 1` in the PPS (no longer rejected)
    /// - `sign_hidden = (last_nz_pos_in_cg - first_nz_pos_in_cg >= 4)` gate
    /// - Decoding `n_end - 1` sign bits in hidden sub-blocks
    /// - Sum-of-abs parity adjustment on the hidden coefficient
    ///
    /// Fixture: `testdata/signhide.h265` + `testdata/signhide_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 16x16 vertical stripe YUV input (left=40, right=200, chroma=128)
    /// x265 --input signhide_input.yuv --input-res 16x16 --fps 1 --frames 1 \
    ///   --preset ultrafast --no-wpp --signhide --ctu 16 --max-tu-size 4 \
    ///   --no-open-gop --keyint 1 --no-scenecut --no-sao --no-deblock \
    ///   --qp 20 --no-psnr --no-ssim --no-info -o signhide.h265
    /// ffmpeg -i signhide.h265 -f rawvideo -pix_fmt yuv420p signhide_ref.yuv
    /// ```
    #[test]
    fn test_decode_signhide_byte_exact() {
        let h265 = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/signhide.h265"
        ))
        .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/signhide_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 16;
        let h: usize = 16;

        let nals = parse_annex_b(&h265);

        // Sanity-check: the PPS in this fixture must actually have SDH on.
        let pps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Pps)
            .expect("fixture must contain a PPS");
        let pps = parse_pps(&pps_nal.rbsp).expect("parse PPS");
        assert!(
            pps.sign_data_hiding_enabled_flag,
            "signhide fixture must have sign_data_hiding_enabled_flag = 1"
        );

        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");

        assert_eq!(frame.width as usize, w);
        assert_eq!(frame.height as usize, h);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );

        if decoded != ref_yuv {
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let plane = if i < w * h {
                        "Y"
                    } else if i < w * h + (w / 2) * (h / 2) {
                        "U"
                    } else {
                        "V"
                    };
                    panic!(
                        "mismatch at byte {} (plane {}) ours={} ref={}",
                        i, plane, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3a-5 byte-exact test**: 16x16 noise-pattern frame at QP 51.
    ///
    /// The original test attempted `--pcm` but most x265 builds don't
    /// support it. The fixture was generated without `--pcm`, so it tests
    /// the high-QP noise-pattern decode path rather than actual PCM CUs.
    /// The unit tests in `cu_tree.rs` for `decode_pcm_block` and
    /// `CabacReader::pcm_byte_position` cover the PCM path synthetically.
    ///
    /// Fixture: `testdata/pcm.h265` + `testdata/pcm_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 16x16 deterministic xorshift noise YUV input (seed=0xdeadbeef)
    /// x265 --input pcm_input.yuv --input-res 16x16 --fps 1 --frames 1 \
    ///   --preset ultrafast --no-wpp --no-signhide --ctu 16 \
    ///   --no-open-gop --keyint 1 --no-scenecut --no-sao --no-deblock \
    ///   --qp 51 --no-psnr --no-ssim --no-info -o pcm.h265
    /// ffmpeg -i pcm.h265 -f rawvideo -pix_fmt yuv420p pcm_ref.yuv
    /// ```
    #[test]
    fn test_decode_pcm_byte_exact() {
        let h265 = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/pcm.h265"))
            .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/pcm_ref.yuv"))
            .expect("read ref");

        let w: usize = 16;
        let h: usize = 16;

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            match decoder.decode_nal(nal) {
                Ok(Some(f)) => {
                    assert!(frame.is_none(), "fixture has only one frame");
                    frame = Some(f);
                }
                Ok(None) => {}
                Err(e) => panic!("decode_nal error: {e}"),
            }
        }
        let frame = frame.expect("expected one decoded frame");

        assert_eq!(frame.width as usize, w);
        assert_eq!(frame.height as usize, h);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );

        if decoded != ref_yuv {
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let plane = if i < w * h {
                        "Y"
                    } else if i < w * h + (w / 2) * (h / 2) {
                        "U"
                    } else {
                        "V"
                    };
                    panic!(
                        "mismatch at byte {} (plane {}) ours={} ref={}",
                        i, plane, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3c-1 / 3c-3 byte-exact test**: 64×64 flat-gray frame encoded
    /// with `--slices 2`. x265 hard-requires `--wpp` whenever `--slices > 1`,
    /// so the fixture's PPS has `entropy_coding_sync_enabled_flag = 1` and
    /// exercises both the multi-slice infrastructure from 3c-1 AND the WPP
    /// per-row CABAC reinit / state propagation from 3c-3.
    ///
    /// At `--ctu 16` the picture has 16 CTBs laid out 4×4. Each slice
    /// covers 8 CTBs = 2 CTB rows, so each slice carries one
    /// `entry_point_offset` pointing at the start of its second row's
    /// substream.
    ///
    /// Exercises:
    ///
    /// - `first_slice_segment_in_pic_flag = 0` + `slice_segment_address`
    ///   parsing in the slice header
    /// - Multi-slice picture assembly in `Decoder::decode_slice`
    /// - CABAC reinit per slice (fresh contexts from the slice's QP)
    /// - WPP `entropy_coding_sync_enabled_flag = 1` PPS path
    /// - `num_entry_point_offsets` + `entry_point_offset_minus1[]` parsing
    /// - Per-row CABAC reinit + context state save/load across rows
    /// - `end_of_slice_flag` decoded at the end of every row (WPP)
    /// - Deblock + SAO finalize only after the last slice arrives
    /// - Byte-exact match against FFmpeg for a multi-slice WPP bitstream
    #[test]
    fn test_decode_multi_slice_byte_exact() {
        let h265_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/multi_slice.h265");
        let yuv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/multi_slice_ref.yuv");
        let h265 = std::fs::read(h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(yuv_path).expect("read reference yuv");
        let nals = parse_annex_b(&h265);
        let vcl_count = nals.iter().filter(|n| n.nal_unit_type.is_vcl()).count();
        assert_eq!(vcl_count, 2, "fixture must have exactly 2 VCL NAL units");

        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);
        if decoded != ref_yuv {
            let w = frame.width as usize;
            let h = frame.height as usize;
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let (plane, idx) = if i < w * h {
                        ("Y", i)
                    } else if i < w * h + (w / 2) * (h / 2) {
                        ("U", i - w * h)
                    } else {
                        ("V", i - w * h - (w / 2) * (h / 2))
                    };
                    let (px, py) = (idx % w, idx / w);
                    panic!(
                        "first mismatch at byte {} (plane {} x={} y={}) ours={} ref={}",
                        i, plane, px, py, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3c-2 byte-exact test**: 256x256 flat-gray intra frame with
    /// `--tiles 2x2`, encoded by kvazaar. Tests:
    ///
    /// - PPS `tiles_enabled_flag = 1` parsing (no longer rejected)
    /// - `num_tile_columns_minus1` / `num_tile_rows_minus1` / uniform spacing
    /// - `Pps::resolve_tile_geometry` producing `column_widths_in_ctbs`
    /// - `TileScanTables::build` building the raster<->tile-scan mapping
    /// - Per-tile CABAC reinit at the tile boundary (byte offset + state)
    /// - CTB iteration in tile-scan order
    /// - `tab_tile_id` population + intra-availability tile boundary check
    /// - Byte-exact match against FFmpeg
    ///
    /// Fixture: `testdata/tiles.h265` + `testdata/tiles_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 256x256 flat gray YUV input (luma=0x7E, chroma=128)
    /// kvazaar --input tiles_input.yuv --input-res 256x256 --input-fps 1 \
    ///   --frames 1 --preset ultrafast --tiles 2x2 --no-wpp --no-sao \
    ///   --no-deblock --no-signhide --gop 0 --period 1 --qp 25 \
    ///   --output tiles.h265
    /// ffmpeg -i tiles.h265 -f rawvideo -pix_fmt yuv420p tiles_ref.yuv
    /// ```
    #[test]
    fn test_decode_tiles_byte_exact() {
        let h265 = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tiles.h265"))
            .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/tiles_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 256;
        let h: usize = 256;

        let nals = parse_annex_b(&h265);

        // Sanity-check: the PPS really has tiles_enabled_flag = 1.
        let pps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Pps)
            .expect("fixture must contain a PPS");
        let pps = parse_pps(&pps_nal.rbsp).expect("parse tiled PPS");
        assert!(
            pps.tiles_enabled_flag,
            "tiles fixture must have tiles_enabled_flag = 1"
        );
        assert_eq!(pps.num_tile_columns, 2);
        assert_eq!(pps.num_tile_rows, 2);

        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");
        assert_eq!(frame.width as usize, w);
        assert_eq!(frame.height as usize, h);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );
        if decoded != ref_yuv {
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let (plane, idx) = if i < w * h {
                        ("Y", i)
                    } else if i < w * h + (w / 2) * (h / 2) {
                        ("U", i - w * h)
                    } else {
                        ("V", i - w * h - (w / 2) * (h / 2))
                    };
                    let (px, py) = (idx % w, idx / w);
                    panic!(
                        "first mismatch at byte {} (plane {} x={} y={}) ours={} ref={}",
                        i, plane, px, py, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3c-4 byte-exact test**: 128x128 flat-gray intra frame with
    /// `kvazaar --slices wpp --wpp`, which puts each CTB row in its own
    /// slice segment with `dependent_slice_segment_flag = 1`. Tests:
    ///
    /// - PPS `dependent_slice_segments_enabled_flag = 1`
    /// - Slice header parser accepting `dependent_slice_segment_flag = 1`
    /// - `Decoder::decode_slice` copying inherited fields from the
    ///   independent slice header into the dependent slice header
    /// - CABAC save at the end of each slice + WPP row-save restore
    /// - Per-slice `entry_point_offsets` still parsed for dependent slices
    ///
    /// Fixture: `testdata/dep_slices.h265` + `testdata/dep_slices_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 128x128 flat gray YUV input (luma=0x7E, chroma=128)
    /// kvazaar --input dep_slices_input.yuv --input-res 128x128 --input-fps 1 \
    ///   --frames 1 --preset ultrafast --slices wpp --wpp --no-sao \
    ///   --no-deblock --no-signhide --gop 0 --period 1 --qp 25 \
    ///   --output dep_slices.h265
    /// ffmpeg -i dep_slices.h265 -f rawvideo -pix_fmt yuv420p dep_slices_ref.yuv
    /// ```
    #[test]
    fn test_decode_dependent_slices_byte_exact() {
        let h265 = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/dep_slices.h265"
        ))
        .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/dep_slices_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 128;
        let h: usize = 128;

        let nals = parse_annex_b(&h265);

        // Sanity-check: the PPS really enables dependent slice segments.
        let pps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Pps)
            .expect("fixture must contain a PPS");
        let pps = parse_pps(&pps_nal.rbsp).expect("parse dep-slice PPS");
        assert!(
            pps.dependent_slice_segments_enabled_flag,
            "dependent slices fixture must have dependent_slice_segments_enabled_flag = 1"
        );

        let vcl_count = nals.iter().filter(|n| n.nal_unit_type.is_vcl()).count();
        assert!(
            vcl_count >= 2,
            "dependent slices fixture must have at least 2 VCL NAL units (got {})",
            vcl_count
        );

        // Parse each VCL slice header and check that at least one is a
        // dependent slice segment.
        let sps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Sps)
            .expect("fixture must contain an SPS");
        let sps = crate::sps::parse_sps(&sps_nal.rbsp).expect("parse SPS");
        let mut pps_resolved = pps.clone();
        pps_resolved
            .resolve_tile_geometry(&sps)
            .expect("resolve tile geometry");
        let mut dependent_count = 0usize;
        for vcl in nals.iter().filter(|n| n.nal_unit_type.is_vcl()) {
            let sh = crate::slice::parse_slice_segment_header(
                &vcl.rbsp,
                vcl.nal_unit_type,
                &sps,
                &pps_resolved,
            )
            .expect("parse slice header");
            if sh.dependent_slice_segment_flag {
                dependent_count += 1;
            }
        }
        assert!(
            dependent_count >= 1,
            "dependent slices fixture must contain at least one dependent slice segment"
        );

        let mut decoder = Decoder::new();
        let mut frame: Option<Frame> = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).expect("decode_nal") {
                assert!(frame.is_none(), "fixture has only one frame");
                frame = Some(f);
            }
        }
        let frame = frame.expect("expected one decoded frame");
        assert_eq!(frame.width as usize, w);
        assert_eq!(frame.height as usize, h);

        let mut decoded = Vec::with_capacity(ref_yuv.len());
        decoded.extend_from_slice(&frame.y);
        decoded.extend_from_slice(&frame.u);
        decoded.extend_from_slice(&frame.v);

        assert_eq!(
            decoded.len(),
            ref_yuv.len(),
            "size mismatch: {} vs {}",
            decoded.len(),
            ref_yuv.len()
        );
        if decoded != ref_yuv {
            for (i, (a, b)) in decoded.iter().zip(ref_yuv.iter()).enumerate() {
                if a != b {
                    let (plane, idx) = if i < w * h {
                        ("Y", i)
                    } else if i < w * h + (w / 2) * (h / 2) {
                        ("U", i - w * h)
                    } else {
                        ("V", i - w * h - (w / 2) * (h / 2))
                    };
                    let (px, py) = (idx % w, idx / w);
                    panic!(
                        "first mismatch at byte {} (plane {} x={} y={}) ours={} ref={}",
                        i, plane, px, py, a, b
                    );
                }
            }
        }
    }

    /// **Phase 3d-3 inter P-slice byte-exact test**: 2-frame IPP sequence.
    /// - Frame 0 (IDR) decodes byte-exact against FFmpeg reference.
    /// - Frame 1 (P-slice) decodes byte-exact against FFmpeg reference.
    ///
    /// Fixture: `testdata/inter_p.h265` + `testdata/inter_p_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 16x16, 2 identical flat-gray frames (luma=0x7E, chroma=128)
    /// x265 --input inter_p_input.yuv --input-res 16x16 --fps 1 --frames 2 \
    ///   --preset ultrafast --no-wpp --no-signhide --ctu 16 --no-open-gop \
    ///   --keyint 2 --bframes 0 --no-scenecut --no-sao --no-deblock \
    ///   --qp 25 --no-psnr --no-ssim --no-info --no-weightp -o inter_p.h265
    /// ffmpeg -i inter_p.h265 -f rawvideo -pix_fmt yuv420p inter_p_ref.yuv
    /// ```
    #[test]
    fn test_decode_inter_p_slice_no_crash() {
        let h265 = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/inter_p.h265"
        ))
        .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/inter_p_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 16;
        let h: usize = 16;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);
        let frame_size = y_size + 2 * uv_size;

        assert_eq!(
            ref_yuv.len(),
            frame_size * 2,
            "reference YUV should have 2 frames"
        );

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frames: Vec<Frame> = Vec::new();
        for nal in &nals {
            match decoder.decode_nal(nal) {
                Ok(Some(f)) => frames.push(f),
                Ok(None) => {}
                Err(e) => panic!("decode_nal failed: {:?}", e),
            }
        }

        assert!(
            !frames.is_empty(),
            "expected at least 1 decoded frame, got {}",
            frames.len()
        );

        // Frame 0 (IDR): byte-exact against FFmpeg.
        let frame0 = &frames[0];
        assert_eq!(frame0.width as usize, w);
        assert_eq!(frame0.height as usize, h);

        let ref_frame0 = &ref_yuv[..frame_size];
        let mut decoded0 = Vec::with_capacity(frame_size);
        decoded0.extend_from_slice(&frame0.y);
        decoded0.extend_from_slice(&frame0.u);
        decoded0.extend_from_slice(&frame0.v);
        assert_eq!(
            decoded0, ref_frame0,
            "frame 0 (IDR) is not byte-exact against FFmpeg reference"
        );

        // Frame 1 (P-slice): byte-exact against FFmpeg.
        assert!(
            frames.len() >= 2,
            "expected 2 decoded frames, got {}",
            frames.len()
        );
        let frame1 = &frames[1];
        assert_eq!(frame1.width as usize, w);
        assert_eq!(frame1.height as usize, h);

        let ref_frame1 = &ref_yuv[frame_size..frame_size * 2];
        let mut decoded1 = Vec::with_capacity(frame_size);
        decoded1.extend_from_slice(&frame1.y);
        decoded1.extend_from_slice(&frame1.u);
        decoded1.extend_from_slice(&frame1.v);
        assert_eq!(
            decoded1, ref_frame1,
            "frame 1 (P-slice) is not byte-exact against FFmpeg reference"
        );
    }

    /// Phase 3e: B-slice byte-exact test.
    ///
    /// 3-frame sequence (IDR + P + B), all frames byte-exact against FFmpeg.
    ///
    /// Fixture: `testdata/inter_b.h265` + `testdata/inter_b_ref.yuv`
    /// Generated with:
    /// ```text
    /// # 16x16, 3 frames: luma=100, 110, 120; chroma=128
    /// x265 --input inter_b_input.yuv --input-res 16x16 --fps 1 --frames 3 \
    ///   --preset ultrafast --no-wpp --no-signhide --ctu 16 --no-open-gop \
    ///   --keyint 3 --bframes 1 --no-scenecut --no-sao --no-deblock \
    ///   --qp 25 --no-psnr --no-ssim --no-info --no-weightp --no-weightb \
    ///   -o inter_b.h265
    /// ffmpeg -i inter_b.h265 -f rawvideo -pix_fmt yuv420p inter_b_ref.yuv
    /// ```
    #[test]
    fn test_decode_inter_b_slice_byte_exact() {
        let h265 = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/inter_b.h265"
        ))
        .expect("read fixture");
        let ref_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/inter_b_ref.yuv"
        ))
        .expect("read ref");

        let w: usize = 16;
        let h: usize = 16;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);
        let frame_size = y_size + 2 * uv_size;

        assert_eq!(
            ref_yuv.len(),
            frame_size * 3,
            "reference YUV should have 3 frames"
        );

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frames: Vec<Frame> = Vec::new();
        for nal in &nals {
            match decoder.decode_nal(nal) {
                Ok(Some(f)) => frames.push(f),
                Ok(None) => {}
                Err(e) => panic!("decode_nal failed: {:?}", e),
            }
        }

        assert_eq!(
            frames.len(),
            3,
            "expected 3 decoded frames, got {}",
            frames.len()
        );

        // Sort our decoded frames by POC (display order) to match FFmpeg output.
        frames.sort_by_key(|f| f.pic_order_cnt);

        // Compare each frame byte-exact against FFmpeg reference.
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.width as usize, w);
            assert_eq!(frame.height as usize, h);

            let ref_frame = &ref_yuv[i * frame_size..(i + 1) * frame_size];
            let mut decoded = Vec::with_capacity(frame_size);
            decoded.extend_from_slice(&frame.y);
            decoded.extend_from_slice(&frame.u);
            decoded.extend_from_slice(&frame.v);
            assert_eq!(
                decoded, ref_frame,
                "frame {} (POC {}) is not byte-exact against FFmpeg reference",
                i, frame.pic_order_cnt
            );
        }
    }

    /// Phase 3f: 1080p hash test.
    ///
    /// Decodes a 1920x1080 10-frame P-only sequence and verifies all 10
    /// frames decode correctly (byte-exact with FFmpeg).
    ///
    /// Fixture: `testdata/1080p.h265`
    /// Generated with:
    /// ```text
    /// ffmpeg -y -f lavfi -i color=gray:size=1920x1080:rate=30:duration=0.33 \
    ///   -frames:v 10 -pix_fmt yuv420p -f rawvideo input_1080p.yuv
    /// x265 --input input_1080p.yuv --input-res 1920x1080 --fps 30 --frames 10 \
    ///   --preset ultrafast --no-wpp --bframes 0 --ref 1 --keyint 10 --qp 28 \
    ///   --no-open-gop --no-sao --no-deblock --no-signhide \
    ///   --no-psnr --no-ssim --no-info -o 1080p.h265
    /// ```
    #[test]
    fn test_decode_1080p_hash() {
        use sha2::{Digest, Sha256};

        let h265 = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/1080p.h265"))
            .expect("read fixture");

        let num_frames: usize = 10;

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut our_hasher = Sha256::new();
        let mut frame_count = 0usize;

        for (i, nal) in nals.iter().enumerate() {
            match decoder.decode_nal(nal) {
                Ok(Some(frame)) => {
                    our_hasher.update(&frame.y);
                    our_hasher.update(&frame.u);
                    our_hasher.update(&frame.v);
                    frame_count += 1;
                }
                Ok(None) => {}
                Err(e) => panic!(
                    "decode_nal failed on NAL[{}] (frame {}): {:?}",
                    i, frame_count, e
                ),
            }
        }
        // Flush any remaining frames.
        while let Some(frame) = decoder.flush() {
            our_hasher.update(&frame.y);
            our_hasher.update(&frame.u);
            our_hasher.update(&frame.v);
            frame_count += 1;
        }

        assert_eq!(
            frame_count, num_frames,
            "expected {} decoded frames, got {}",
            num_frames, frame_count
        );

        let our_hash = our_hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        // SHA-256 verified against FFmpeg: `ffmpeg -i 1080p.h265 -f rawvideo
        // -pix_fmt yuv420p pipe:1 | shasum -a 256`. The conformance window
        // crop ensures our output matches FFmpeg's 1920×1080 output (not the
        // CTU-padded 1920×1088 coded picture).
        let expected_hash = "ee157a13ccac1b08728ece0719421731e6ce0bbdfabe10c43a5239c3a1b6f810";
        assert_eq!(
            our_hash, expected_hash,
            "1080p SHA-256 mismatch: ours={} expected={}",
            our_hash, expected_hash
        );
    }

    // -----------------------------------------------------------------------
    // Real-world-ish test fixtures
    // -----------------------------------------------------------------------
    //
    // All fixtures below use `testsrc2` or `testsrc` patterns encoded with
    // x265 `--preset ultrafast --ctu 16`. B-frame fixtures require sorting
    // decoded frames by POC (display order) before hashing, because FFmpeg
    // outputs in display order while our decoder outputs in decode order.

    /// Helper: decode a fixture, sort by POC, and return the SHA-256 hash
    /// of all planes concatenated in display order.
    fn decode_and_hash(fixture_name: &str, expected_frames: usize) -> String {
        use sha2::{Digest, Sha256};

        let h265 = std::fs::read(
            concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/").to_string() + fixture_name,
        )
        .unwrap_or_else(|e| panic!("read {fixture_name}: {e}"));

        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frames: Vec<Frame> = Vec::new();

        for (i, nal) in nals.iter().enumerate() {
            match decoder.decode_nal(nal) {
                Ok(Some(frame)) => frames.push(frame),
                Ok(None) => {}
                Err(e) => panic!(
                    "{}: decode_nal failed on NAL[{}] (frame {}): {:?}",
                    fixture_name,
                    i,
                    frames.len(),
                    e
                ),
            }
        }
        // Flush remaining frames (B-frames buffered in DPB).
        while let Some(frame) = decoder.flush() {
            frames.push(frame);
        }

        assert_eq!(
            frames.len(),
            expected_frames,
            "{}: expected {} frames, got {}",
            fixture_name,
            expected_frames,
            frames.len()
        );

        // Sort by POC for display-order hash comparison with FFmpeg.
        frames.sort_by_key(|f| f.pic_order_cnt);

        let mut hasher = Sha256::new();
        for frame in &frames {
            hasher.update(&frame.y);
            hasher.update(&frame.u);
            hasher.update(&frame.v);
        }
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    }

    /// 320x240, 30 frames, bframes=1, ref=4, qp=24 — exercises multi-ref
    /// P/B prediction on varied testsrc2 content. No SAO, no deblock.
    ///
    /// Fixture generated with:
    /// ```text
    /// ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=30:duration=1.0" \
    ///   -frames:v 30 -pix_fmt yuv420p -f rawvideo /tmp/input.yuv
    /// x265 --input /tmp/input.yuv --input-res 320x240 --fps 30 --frames 30 \
    ///   --preset ultrafast --ctu 16 --no-wpp --bframes 1 --ref 4 --qp 24 \
    ///   --keyint 30 --no-open-gop --no-weightp --no-weightb --no-scenecut \
    ///   --no-sao --no-deblock --no-signhide \
    ///   --no-psnr --no-ssim --no-info -o realworld_320x240.h265
    /// ```
    #[test]
    fn test_decode_realworld_320x240_byte_exact() {
        let hash = decode_and_hash("realworld_320x240.h265", 30);
        let expected = "e12a27d0656e2dd3967e11934f32db1f5a03fec48da911429561b8417334690e";
        assert_eq!(
            hash, expected,
            "realworld_320x240 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// 1280x720, 10 frames, bframes=1, ref=2, qp=26 — HD content with
    /// B-frames. 720p is CTU-aligned in width (1280/16=80) but height
    /// (720/16=45) has no padding.
    ///
    /// Fixture generated with:
    /// ```text
    /// ffmpeg -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=0.34" \
    ///   -frames:v 10 -pix_fmt yuv420p -f rawvideo /tmp/input.yuv
    /// x265 --input /tmp/input.yuv --input-res 1280x720 --fps 30 --frames 10 \
    ///   --preset ultrafast --ctu 16 --no-wpp --bframes 1 --ref 2 --qp 26 \
    ///   --keyint 10 --no-open-gop --no-weightp --no-weightb --no-scenecut \
    ///   --no-sao --no-deblock --no-signhide \
    ///   --no-psnr --no-ssim --no-info -o realworld_720p.h265
    /// ```
    #[test]
    fn test_decode_realworld_720p_hash() {
        let hash = decode_and_hash("realworld_720p.h265", 10);
        let expected = "9cbafe78054edc6fc565f80c6339e36a3c536eb58da558f7b4a76523d26ff638";
        assert_eq!(
            hash, expected,
            "realworld_720p hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// 320x240, 20 frames of moving testsrc pattern — exercises non-zero MVs,
    /// sub-pixel interpolation, and temporal prediction with actual motion.
    ///
    /// Fixture generated with:
    /// ```text
    /// ffmpeg -f lavfi -i "testsrc=size=320x240:rate=30:duration=0.67" \
    ///   -frames:v 20 -pix_fmt yuv420p -f rawvideo /tmp/input.yuv
    /// x265 --input /tmp/input.yuv --input-res 320x240 --fps 30 --frames 20 \
    ///   --preset ultrafast --ctu 16 --no-wpp --bframes 1 --ref 2 --qp 22 \
    ///   --keyint 20 --no-open-gop --no-weightp --no-weightb --no-scenecut \
    ///   --no-sao --no-deblock --no-signhide \
    ///   --no-psnr --no-ssim --no-info -o motion_320x240.h265
    /// ```
    #[test]
    fn test_decode_motion_320x240_hash() {
        let hash = decode_and_hash("motion_320x240.h265", 20);
        let expected = "5b7faa6a62ba7932fc643a3b668b1dfc06cae44bfec43651de4b59a1c3aa35fb";
        assert_eq!(
            hash, expected,
            "motion_320x240 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// 320x240, 10 frames with deblocking AND SAO enabled (qp=30 for
    /// visible artifacts that the filters correct). Validates in-loop
    /// filtering on multi-CTU B-frame content.
    ///
    /// Fixture generated with:
    /// ```text
    /// ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=30:duration=0.34" \
    ///   -frames:v 10 -pix_fmt yuv420p -f rawvideo /tmp/input.yuv
    /// x265 --input /tmp/input.yuv --input-res 320x240 --fps 30 --frames 10 \
    ///   --preset ultrafast --ctu 16 --no-wpp --bframes 1 --ref 2 --qp 30 \
    ///   --keyint 10 --no-open-gop --no-weightp --no-weightb --no-scenecut \
    ///   --no-signhide --no-psnr --no-ssim --no-info \
    ///   -o deblock_sao_320x240.h265
    /// ```
    #[test]
    fn test_decode_deblock_sao_320x240_hash() {
        let hash = decode_and_hash("deblock_sao_320x240.h265", 10);
        let expected = "e672d49a06df7798d7c5c1610b5ccfe2e37772bbbf4eee3a2d3877838d12dc82";
        assert_eq!(
            hash, expected,
            "deblock_sao_320x240 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// Same content as deblock_sao_320x240 but with `--no-deblock`.
    /// Verifies that pre-deblock inter prediction is byte-exact.
    ///
    /// Fixture generated with:
    /// ```text
    /// ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=30:duration=0.34" \
    ///   -frames:v 10 -pix_fmt yuv420p -f rawvideo /tmp/input.yuv
    /// x265 --input /tmp/input.yuv --input-res 320x240 --fps 30 --frames 10 \
    ///   --preset ultrafast --ctu 16 --no-wpp --bframes 1 --ref 2 --qp 30 \
    ///   --keyint 10 --no-open-gop --no-weightp --no-weightb --no-scenecut \
    ///   --no-sao --no-deblock --no-psnr --no-ssim --no-info \
    ///   -o deblock_sao_nodeblock.h265
    /// ```
    #[test]
    fn test_decode_deblock_sao_nodeblock_hash() {
        let hash = decode_and_hash("deblock_sao_nodeblock.h265", 10);
        // FFmpeg reference: ffmpeg -i deblock_sao_nodeblock.h265 -f rawvideo -pix_fmt yuv420p pipe:1 | shasum -a 256
        let expected = "609700078af4bf8de52d904a89ae75ced37b1e98e28cb0dc74ebc5bc980faff5";
        assert_eq!(
            hash, expected,
            "deblock_sao_nodeblock hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// Diagnostic test: compare our decoded output of deblock_sao_320x240.h265
    /// against FFmpeg's output, frame by frame, to find which frames mismatch
    /// and where.
    #[test]
    #[ignore]
    fn diag_deblock_sao() {
        use std::process::Command;

        let h265_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/deblock_sao_320x240.h265"
        );
        let h265 = std::fs::read(h265_path).expect("read fixture");

        // Decode with our decoder
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frames: Vec<Frame> = Vec::new();
        for nal in &nals {
            if let Ok(Some(frame)) = decoder.decode_nal(nal) {
                frames.push(frame);
            }
        }
        while let Some(frame) = decoder.flush() {
            frames.push(frame);
        }
        frames.sort_by_key(|f| f.pic_order_cnt);

        // Decode with FFmpeg
        let ffmpeg_out = Command::new("ffmpeg")
            .args([
                "-i", h265_path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
            ])
            .stderr(std::process::Stdio::null())
            .output()
            .expect("ffmpeg");
        let ref_yuv = ffmpeg_out.stdout;

        let w = frames[0].width as usize;
        let h = frames[0].height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);
        let frame_size = y_size + 2 * uv_size;
        assert_eq!(
            ref_yuv.len(),
            frame_size * frames.len(),
            "ref yuv size mismatch"
        );

        let mut first_mismatch_frame = None;
        for (fi, frame) in frames.iter().enumerate() {
            let ref_off = fi * frame_size;
            let ref_y = &ref_yuv[ref_off..ref_off + y_size];
            let ref_u = &ref_yuv[ref_off + y_size..ref_off + y_size + uv_size];
            let ref_v = &ref_yuv[ref_off + y_size + uv_size..ref_off + frame_size];

            let mut mismatch = false;
            let mut y_mismatches = 0;
            // Check Y plane
            for i in 0..y_size {
                if frame.y[i] != ref_y[i] {
                    let px = i % w;
                    let py = i / w;
                    if y_mismatches < 10 {
                        let on_vedge = (px % 8) < 4 || (8 - (px % 8)) < 4;
                        let on_hedge = (py % 8) < 4 || (8 - (py % 8)) < 4;
                        let near_deblock_edge = on_vedge || on_hedge;
                        eprintln!(
                            "  Frame {} (POC {}): Y[{},{}] ours={} ref={} diff={} deblock_edge={}",
                            fi,
                            frame.pic_order_cnt,
                            px,
                            py,
                            frame.y[i],
                            ref_y[i],
                            frame.y[i] as i32 - ref_y[i] as i32,
                            near_deblock_edge
                        );
                    }
                    if first_mismatch_frame.is_none() {
                        first_mismatch_frame = Some(fi);
                    }
                    mismatch = true;
                    y_mismatches += 1;
                }
            }
            if y_mismatches > 0 {
                eprintln!("  Frame {}: total Y mismatches: {}", fi, y_mismatches);
            }
            if !mismatch {
                // Check U plane
                for i in 0..uv_size {
                    if frame.u[i] != ref_u[i] {
                        let px = i % (w / 2);
                        let py = i / (w / 2);
                        eprintln!(
                            "Frame {} (POC {}): U mismatch at ({},{}) ours={} ref={}",
                            fi, frame.pic_order_cnt, px, py, frame.u[i], ref_u[i]
                        );
                        if first_mismatch_frame.is_none() {
                            first_mismatch_frame = Some(fi);
                        }
                        mismatch = true;
                        break;
                    }
                }
            }
            if !mismatch {
                // Check V plane
                for i in 0..uv_size {
                    if frame.v[i] != ref_v[i] {
                        let px = i % (w / 2);
                        let py = i / (w / 2);
                        eprintln!(
                            "Frame {} (POC {}): V mismatch at ({},{}) ours={} ref={}",
                            fi, frame.pic_order_cnt, px, py, frame.v[i], ref_v[i]
                        );
                        if first_mismatch_frame.is_none() {
                            first_mismatch_frame = Some(fi);
                        }
                        mismatch = true;
                        break;
                    }
                }
            }
            if !mismatch {
                eprintln!("Frame {} (POC {}): MATCH", fi, frame.pic_order_cnt);
            }
        }
        if let Some(f) = first_mismatch_frame {
            panic!("First mismatch at frame {}", f);
        }
    }

    /// 320x240, 10 frames with sign data hiding AND default scaling lists
    /// enabled. Tests the coefficient coding path with sign hiding and
    /// the dequantization path with non-flat scaling matrices.
    ///
    /// Fixture generated with:
    /// ```text
    /// ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=30:duration=0.34" \
    ///   -frames:v 10 -pix_fmt yuv420p -f rawvideo /tmp/input.yuv
    /// x265 --input /tmp/input.yuv --input-res 320x240 --fps 30 --frames 10 \
    ///   --preset ultrafast --ctu 16 --no-wpp --signhide --scaling-list default \
    ///   --bframes 1 --ref 2 --qp 26 --keyint 10 --no-open-gop \
    ///   --no-weightp --no-weightb --no-scenecut --no-sao --no-deblock \
    ///   --no-psnr --no-ssim --no-info \
    ///   -o signhide_scaling_320x240.h265
    /// ```
    #[test]
    fn test_decode_signhide_scaling_320x240_hash() {
        let hash = decode_and_hash("signhide_scaling_320x240.h265", 10);
        let expected = "ff3e179ade08f6b3111c5b21d576605f5ee4d22b3ad5747f15c2d78dcbd2e512";
        assert_eq!(
            hash, expected,
            "signhide_scaling_320x240 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// Single 64x64 I-frame with `--aq-mode 1 --qg-size 32`: exercises the
    /// HEVC spec 8.6.1 QP prediction path. Each 32x32 QP group carries its
    /// own `cu_qp_delta`, requiring `get_qPy_pred` / `set_qPy` to average
    /// the left + above QPs from `tab_qp_y` (with `qpy_pred` as fallback
    /// when neighbors are unavailable).
    ///
    /// Fixture generated with:
    /// ```text
    /// ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=30:duration=0.1" \
    ///   -frames:v 1 -pix_fmt yuv420p -f rawvideo /tmp/in.yuv
    /// x265 --input /tmp/in.yuv --input-res 64x64 --fps 30 --frames 1 \
    ///   --preset ultrafast --ctu 64 --keyint 30 --no-open-gop \
    ///   --aq-mode 1 --aq-strength 1.0 --no-cutree --qg-size 32 \
    ///   --no-sao --no-deblock --no-info --no-psnr --no-ssim --no-wpp \
    ///   -o aq_intra_64.h265
    /// ```
    #[test]
    fn test_decode_aq_intra_64_hash() {
        let hash = decode_and_hash("aq_intra_64.h265", 1);
        let expected = "8ba0a950e270204375060527f592b5a0bfc3ad49d51919269d91239d7f09fbcf";
        assert_eq!(
            hash, expected,
            "aq_intra_64 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// 320x240, 5 frames (I+B+P+B+P) with CTU=64, no SAO, no deblock,
    /// constant QP (cu_qp_delta_enabled_flag=0).
    /// Tests the critical CTU=64 fixes: set_ct_depth for leaf CUs only
    /// (not split nodes) and more_data propagation at quad-tree boundaries.
    ///
    /// Fixture generated with:
    /// ```text
    /// x265 --input input.yuv --input-res 320x240 --fps 30 --frames 5 \
    ///   --ctu 64 --keyint 30 --no-open-gop --bframes 1 --no-sao --no-deblock \
    ///   --qp 26 --no-cutree --no-aq --no-psnr --no-ssim --no-info \
    ///   -o ctu64_noqp_nosao_320x240.h265
    /// ```
    #[test]
    fn test_decode_ctu64_wpp_hash() {
        let hash = decode_and_hash("ctu64_wpp.h265", 5);
        let expected = "de7a1ac668d67e19fb052beb7cd5577d3f40bb2b7244dd82b421d98ead1f702c";
        assert_eq!(
            hash, expected,
            "ctu64_wpp hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    #[test]
    fn test_decode_ctu64_320x240_hash() {
        let hash = decode_and_hash("ctu64_noqp_nosao_320x240.h265", 5);
        let expected = "98341166e6c5235b88fe3e9dcc5532084462f704969d608a949d6cc015c09621";
        assert_eq!(
            hash, expected,
            "ctu64_noqp_nosao_320x240 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// 64x64, 1 I-frame, CTU=64, flat grey content — byte-exact baseline.
    #[test]
    fn test_decode_ctu64_flat64_hash() {
        let hash = decode_and_hash("flat64.h265", 1);
        let expected = "cb11e05cb5da949c0e0f5b5a7cb310df35a96a22c45d1ada70d950859fe697d1";
        assert_eq!(
            hash, expected,
            "flat64 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    #[test]
    fn test_decode_ctu64_128x128_hash() {
        let hash = decode_and_hash("ctu64_128x128.h265", 2);
        let expected = "ed0c46cb86b57c9b605971288c21aa1b9016e800ecc99c544e27e12bda9622ee";
        assert_eq!(
            hash, expected,
            "ctu64_128x128 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// 32x32, CTU=32, I+P, gradient. Tests 32x32 TU residual at CTU=32
    /// and weighted prediction (luma_offset=17 applied via pred_weight_table).
    #[test]
    fn test_decode_tu32_inter_hash() {
        let hash = decode_and_hash("tu32_test.h265", 2);
        let expected = "e13bfc4fdfe0cbd5c3d763d8461ceb939fff088a1fb8d1fe3975c99d0d52a802";
        assert_eq!(
            hash, expected,
            "tu32_test hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    #[test]
    #[ignore = "diagnostic — requires /tmp/grad64_ffmpeg.yuv from FFmpeg"]
    fn diag_grad64_pixel() {
        let ref_yuv = std::fs::read("/tmp/grad64_ffmpeg.yuv").expect("read ref yuv");
        let h265 =
            std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/grad64.h265")).unwrap();
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frame = None;
        for nal in &nals {
            if let Ok(Some(f)) = decoder.decode_nal(nal) {
                frame = Some(f);
            }
        }
        if frame.is_none() {
            if let Some(f) = decoder.flush() {
                frame = Some(f);
            }
        }
        let frame = frame.expect("no frame decoded");
        let w = 64usize;
        let h = 64usize;
        // Compare Y plane
        let mut first_y: Option<(usize, usize, u8, u8)> = None;
        let mut y_count = 0u32;
        for j in 0..h {
            for i in 0..w {
                let ours = frame.y[j * w + i];
                let theirs = ref_yuv[j * w + i];
                if ours != theirs {
                    y_count += 1;
                    if first_y.is_none() {
                        first_y = Some((i, j, ours, theirs));
                    }
                }
            }
        }
        eprintln!("Y mismatches: {y_count}");
        if let Some((x, y, ours, theirs)) = first_y {
            eprintln!(
                "First Y mismatch at ({x},{y}): ours={ours} ffmpeg={theirs} diff={}",
                ours as i32 - theirs as i32
            );
            // Dump 8x8 region around first mismatch
            let rx = (x / 8) * 8;
            let ry = (y / 8) * 8;
            eprintln!("8x8 block ({},{})-({},{}):", rx, ry, rx + 7, ry + 7);
            for jj in ry..ry + 8 {
                let mut line = format!("  y={jj:2}: ");
                for ii in rx..(rx + 16).min(w) {
                    let o = frame.y[jj * w + ii];
                    let r = ref_yuv[jj * w + ii];
                    if o == r {
                        line += &format!("{o:4}");
                    } else {
                        line += &format!("{:+4}", o as i32 - r as i32);
                    }
                }
                eprintln!("{line}");
            }
        }
        if y_count > 0 {
            // Show per-8x8 block mismatch counts
            eprintln!("\nPer-8x8 block Y mismatch counts:");
            for by in 0..h / 8 {
                for bx in 0..w / 8 {
                    let mut cnt = 0u32;
                    for jj in 0..8 {
                        for ii in 0..8 {
                            let y = by * 8 + jj;
                            let x = bx * 8 + ii;
                            if frame.y[y * w + x] != ref_yuv[y * w + x] {
                                cnt += 1;
                            }
                        }
                    }
                    if cnt > 0 {
                        eprint!("  ({},{}): {cnt}", bx * 8, by * 8);
                    }
                }
            }
            eprintln!();
        }
        assert_eq!(y_count, 0, "{y_count} Y mismatches");
    }

    /// 64x64, 1 I-frame, CTU=64, 2D gradient content — byte-exact.
    /// Tests PART_NxN per-sub-CU intra mode lookup via tab_ipm.
    #[test]
    fn test_decode_ctu64_grad64_hash() {
        let hash = decode_and_hash("grad64.h265", 1);
        let expected = "9ca645aeecb7492ec294734478e5e8683ed8e872228977e35dd42ca054b9025d";
        assert_eq!(
            hash, expected,
            "grad64 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// 64x64, 1 I-frame, CTU=64, horizontal ramp — byte-exact.
    #[test]
    fn test_decode_ctu64_ramp64_hash() {
        let hash = decode_and_hash("ramp64.h265", 1);
        let expected = "2177e962f4fff5e9e506a60ab415b9fe4080fdd769716a51150fdd43ebba7aae";
        assert_eq!(
            hash, expected,
            "ramp64 hash mismatch:\n  got: {hash}\n  exp: {expected}"
        );
    }

    /// Diagnostic test: compare against FFmpeg per-frame to find mismatches.
    /// Requires ffmpeg on PATH. Run with `cargo test diag_realworld -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic only, requires ffmpeg"]
    fn diag_realworld_320x240_per_frame() {
        use std::process::Command;

        let fixture = "realworld_320x240.h265";
        let fixture_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/").to_string() + fixture;
        let width: usize = 320;
        let height: usize = 240;
        let frame_size = width * height * 3 / 2; // YUV420

        // Decode with FFmpeg to get reference YUV in display order.
        let ffmpeg_out = Command::new("ffmpeg")
            .args([
                "-i",
                &fixture_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "pipe:1",
            ])
            .output()
            .expect("ffmpeg not found");
        assert!(
            ffmpeg_out.status.success(),
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&ffmpeg_out.stderr)
        );
        let ref_yuv = ffmpeg_out.stdout;
        let num_ref_frames = ref_yuv.len() / frame_size;
        eprintln!(
            "FFmpeg produced {} bytes = {} frames",
            ref_yuv.len(),
            num_ref_frames
        );

        // Decode with our decoder.
        let h265 = std::fs::read(&fixture_path).unwrap();
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frames: Vec<Frame> = Vec::new();

        for nal in &nals {
            if let Ok(Some(frame)) = decoder.decode_nal(nal) {
                frames.push(frame);
            }
        }
        while let Some(frame) = decoder.flush() {
            frames.push(frame);
        }

        // Sort by POC for display-order comparison with FFmpeg.
        frames.sort_by_key(|f| f.pic_order_cnt);

        eprintln!("Our decoder produced {} frames", frames.len());
        assert_eq!(frames.len(), num_ref_frames, "frame count mismatch");

        let y_plane_size = width * height;
        let uv_plane_size = (width / 2) * (height / 2);

        let mut first_bad_frame: Option<usize> = None;

        for (idx, frame) in frames.iter().enumerate() {
            let ref_start = idx * frame_size;
            let ref_y = &ref_yuv[ref_start..ref_start + y_plane_size];
            let ref_u =
                &ref_yuv[ref_start + y_plane_size..ref_start + y_plane_size + uv_plane_size];
            let ref_v = &ref_yuv[ref_start + y_plane_size + uv_plane_size..ref_start + frame_size];

            let mut y_mismatches = 0u64;
            let mut u_mismatches = 0u64;
            let mut v_mismatches = 0u64;
            let mut first_y: Option<(usize, usize, u8, u8)> = None;
            let mut first_u: Option<(usize, usize, u8, u8)> = None;
            let mut first_v: Option<(usize, usize, u8, u8)> = None;
            let mut max_y_diff: i32 = 0;
            let mut max_u_diff: i32 = 0;
            let mut max_v_diff: i32 = 0;

            // Compare Y plane.
            for j in 0..height {
                for i in 0..width {
                    let ours = frame.y[j * width + i];
                    let theirs = ref_y[j * width + i];
                    if ours != theirs {
                        y_mismatches += 1;
                        let diff = (ours as i32) - (theirs as i32);
                        if diff.abs() > max_y_diff.abs() {
                            max_y_diff = diff;
                        }
                        if first_y.is_none() {
                            first_y = Some((i, j, ours, theirs));
                        }
                    }
                }
            }

            // Compare U plane.
            let cw = width / 2;
            let ch = height / 2;
            for j in 0..ch {
                for i in 0..cw {
                    let ours = frame.u[j * cw + i];
                    let theirs = ref_u[j * cw + i];
                    if ours != theirs {
                        u_mismatches += 1;
                        let diff = (ours as i32) - (theirs as i32);
                        if diff.abs() > max_u_diff.abs() {
                            max_u_diff = diff;
                        }
                        if first_u.is_none() {
                            first_u = Some((i, j, ours, theirs));
                        }
                    }
                }
            }

            // Compare V plane.
            for j in 0..ch {
                for i in 0..cw {
                    let ours = frame.v[j * cw + i];
                    let theirs = ref_v[j * cw + i];
                    if ours != theirs {
                        v_mismatches += 1;
                        let diff = (ours as i32) - (theirs as i32);
                        if diff.abs() > max_v_diff.abs() {
                            max_v_diff = diff;
                        }
                        if first_v.is_none() {
                            first_v = Some((i, j, ours, theirs));
                        }
                    }
                }
            }

            let total = y_mismatches + u_mismatches + v_mismatches;
            if total > 0 {
                eprintln!(
                    "Frame {} (POC {}): {} mismatches (Y={} U={} V={}), max_diff Y={} U={} V={}",
                    idx,
                    frame.pic_order_cnt,
                    total,
                    y_mismatches,
                    u_mismatches,
                    v_mismatches,
                    max_y_diff,
                    max_u_diff,
                    max_v_diff
                );
                if let Some((x, y, ours, theirs)) = first_y {
                    eprintln!(
                        "  First Y mismatch: ({}, {}): ours={} ffmpeg={} diff={}",
                        x,
                        y,
                        ours,
                        theirs,
                        ours as i32 - theirs as i32
                    );
                }
                if let Some((x, y, ours, theirs)) = first_u {
                    eprintln!(
                        "  First U mismatch: ({}, {}): ours={} ffmpeg={} diff={}",
                        x,
                        y,
                        ours,
                        theirs,
                        ours as i32 - theirs as i32
                    );
                }
                if let Some((x, y, ours, theirs)) = first_v {
                    eprintln!(
                        "  First V mismatch: ({}, {}): ours={} ffmpeg={} diff={}",
                        x,
                        y,
                        ours,
                        theirs,
                        ours as i32 - theirs as i32
                    );
                }
                if first_bad_frame.is_none() {
                    first_bad_frame = Some(idx);
                }
            } else {
                eprintln!("Frame {} (POC {}) matches FFmpeg", idx, frame.pic_order_cnt);
            }
        }

        // Dump a detailed map of mismatches in the first bad frame (frame 0).
        if let Some(bad) = first_bad_frame {
            let frame = &frames[bad];
            let ref_start = bad * frame_size;
            let ref_y = &ref_yuv[ref_start..ref_start + y_plane_size];

            // Show the region around (1, 32) in detail.
            eprintln!(
                "\n=== Mismatch map for frame {} Y plane, rows 28..40, cols 0..20 ===",
                bad
            );
            for j in 28usize..40.min(height) {
                let mut line = format!("  y={:3}: ", j);
                for i in 0usize..20.min(width) {
                    let ours = frame.y[j * width + i];
                    let theirs = ref_y[j * width + i];
                    if ours == theirs {
                        line += &format!("{:4}", ours);
                    } else {
                        let diff = ours as i32 - theirs as i32;
                        line += &format!(" {:+3}", diff);
                    }
                }
                eprintln!("{}", line);
            }

            // Check if the mismatch pattern aligns with block boundaries.
            eprintln!("\n=== Row-by-row mismatch count (first 48 rows) ===");
            for j in 0..48.min(height) {
                let mut count = 0u32;
                for i in 0..width {
                    if frame.y[j * width + i] != ref_y[j * width + i] {
                        count += 1;
                    }
                }
                if count > 0 {
                    eprintln!("  y={:3}: {} mismatches", j, count);
                }
            }

            panic!("First mismatched frame: {} (see stderr for details)", bad);
        }
        eprintln!("All {} frames match FFmpeg byte-exactly!", frames.len());
    }

    /// Diagnostic test: compare ctu64_noqp_nosao_320x240.h265 against FFmpeg per-frame.
    /// Reports which frame(s) mismatch, first mismatch position, and an 8x8 block dump.
    /// Requires ffmpeg on PATH.
    /// Run with: cargo test diag_ctu64_pixel_mismatch -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic only, requires ffmpeg"]
    fn diag_ctu64_pixel_mismatch() {
        use std::process::Command;

        let fixture = "ctu64_noqp_nosao_320x240.h265";
        let fixture_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/").to_string() + fixture;
        let width: usize = 320;
        let height: usize = 240;
        let frame_size = width * height * 3 / 2; // YUV420

        // --- Decode with FFmpeg to get reference YUV in display order ---
        let ffmpeg_out = Command::new("ffmpeg")
            .args([
                "-i",
                &fixture_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "pipe:1",
            ])
            .output()
            .expect("ffmpeg not found on PATH");
        assert!(
            ffmpeg_out.status.success(),
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&ffmpeg_out.stderr)
        );
        let ref_yuv = ffmpeg_out.stdout;
        let num_ref_frames = ref_yuv.len() / frame_size;
        eprintln!(
            "FFmpeg produced {} bytes = {} frames",
            ref_yuv.len(),
            num_ref_frames
        );

        // --- Decode with our decoder ---
        let h265 = std::fs::read(&fixture_path).unwrap();
        let nals = parse_annex_b(&h265);
        let mut decoder = Decoder::new();
        let mut frames: Vec<Frame> = Vec::new();

        for nal in &nals {
            if let Ok(Some(frame)) = decoder.decode_nal(nal) {
                frames.push(frame);
            }
        }
        while let Some(frame) = decoder.flush() {
            frames.push(frame);
        }

        // Sort by POC for display-order comparison with FFmpeg.
        frames.sort_by_key(|f| f.pic_order_cnt);

        eprintln!("Our decoder produced {} frames", frames.len());
        eprintln!(
            "Frame POCs: {:?}",
            frames.iter().map(|f| f.pic_order_cnt).collect::<Vec<_>>()
        );
        assert_eq!(frames.len(), num_ref_frames, "frame count mismatch");

        let y_plane_size = width * height;
        let uv_plane_size = (width / 2) * (height / 2);

        let mut first_bad_frame: Option<usize> = None;
        let mut first_bad_pos: Option<(usize, usize)> = None; // (x, y) in luma

        for (idx, frame) in frames.iter().enumerate() {
            let ref_start = idx * frame_size;
            let ref_y = &ref_yuv[ref_start..ref_start + y_plane_size];
            let ref_u =
                &ref_yuv[ref_start + y_plane_size..ref_start + y_plane_size + uv_plane_size];
            let ref_v = &ref_yuv[ref_start + y_plane_size + uv_plane_size..ref_start + frame_size];

            let mut y_mismatches = 0u64;
            let mut u_mismatches = 0u64;
            let mut v_mismatches = 0u64;
            let mut first_y: Option<(usize, usize, u8, u8)> = None;
            let mut first_u: Option<(usize, usize, u8, u8)> = None;
            let mut first_v: Option<(usize, usize, u8, u8)> = None;
            let mut max_y_diff: i32 = 0;
            let mut max_u_diff: i32 = 0;
            let mut max_v_diff: i32 = 0;

            for j in 0..height {
                for i in 0..width {
                    let ours = frame.y[j * width + i];
                    let theirs = ref_y[j * width + i];
                    if ours != theirs {
                        y_mismatches += 1;
                        let diff = (ours as i32) - (theirs as i32);
                        if diff.abs() > max_y_diff.abs() {
                            max_y_diff = diff;
                        }
                        if first_y.is_none() {
                            first_y = Some((i, j, ours, theirs));
                        }
                    }
                }
            }

            let cw = width / 2;
            let ch = height / 2;
            for j in 0..ch {
                for i in 0..cw {
                    let ours = frame.u[j * cw + i];
                    let theirs = ref_u[j * cw + i];
                    if ours != theirs {
                        u_mismatches += 1;
                        let diff = (ours as i32) - (theirs as i32);
                        if diff.abs() > max_u_diff.abs() {
                            max_u_diff = diff;
                        }
                        if first_u.is_none() {
                            first_u = Some((i, j, ours, theirs));
                        }
                    }
                }
            }

            for j in 0..ch {
                for i in 0..cw {
                    let ours = frame.v[j * cw + i];
                    let theirs = ref_v[j * cw + i];
                    if ours != theirs {
                        v_mismatches += 1;
                        let diff = (ours as i32) - (theirs as i32);
                        if diff.abs() > max_v_diff.abs() {
                            max_v_diff = diff;
                        }
                        if first_v.is_none() {
                            first_v = Some((i, j, ours, theirs));
                        }
                    }
                }
            }

            let total = y_mismatches + u_mismatches + v_mismatches;
            if total > 0 {
                eprintln!(
                    "\nFrame {} (POC {}): {} total mismatches (Y={} U={} V={})",
                    idx, frame.pic_order_cnt, total, y_mismatches, u_mismatches, v_mismatches
                );
                eprintln!(
                    "  Max diffs: Y={} U={} V={}",
                    max_y_diff, max_u_diff, max_v_diff
                );
                if let Some((x, y, ours, theirs)) = first_y {
                    eprintln!(
                        "  First Y mismatch: ({}, {}): ours={} ffmpeg={} diff={}",
                        x,
                        y,
                        ours,
                        theirs,
                        ours as i32 - theirs as i32
                    );
                    // Identify CTU and CU boundaries.
                    let ctu_x = x / 64;
                    let ctu_y = y / 64;
                    eprintln!(
                        "    CTU=({}, {}), within-CTU offset=({}, {})",
                        ctu_x,
                        ctu_y,
                        x % 64,
                        y % 64
                    );
                    eprintln!(
                        "    32x32 block=({}, {}), 16x16 block=({}, {}), 8x8 block=({}, {})",
                        x / 32,
                        y / 32,
                        x / 16,
                        y / 16,
                        x / 8,
                        y / 8
                    );
                }
                if let Some((x, y, ours, theirs)) = first_u {
                    eprintln!(
                        "  First U mismatch: ({}, {}): ours={} ffmpeg={} diff={}",
                        x,
                        y,
                        ours,
                        theirs,
                        ours as i32 - theirs as i32
                    );
                }
                if let Some((x, y, ours, theirs)) = first_v {
                    eprintln!(
                        "  First V mismatch: ({}, {}): ours={} ffmpeg={} diff={}",
                        x,
                        y,
                        ours,
                        theirs,
                        ours as i32 - theirs as i32
                    );
                }

                if first_bad_frame.is_none() {
                    first_bad_frame = Some(idx);
                    if let Some((x, y, _, _)) = first_y {
                        first_bad_pos = Some((x, y));
                    } else if let Some((x, y, _, _)) = first_u {
                        // Map chroma coords to luma for block dump.
                        first_bad_pos = Some((x * 2, y * 2));
                    } else if let Some((x, y, _, _)) = first_v {
                        first_bad_pos = Some((x * 2, y * 2));
                    }
                }
            } else {
                eprintln!("Frame {} (POC {}) matches FFmpeg", idx, frame.pic_order_cnt);
            }
        }

        // --- Detailed dump for the first mismatching frame ---
        if let Some(bad) = first_bad_frame {
            let frame = &frames[bad];
            let ref_start = bad * frame_size;
            let ref_y = &ref_yuv[ref_start..ref_start + y_plane_size];

            if let Some((fx, fy)) = first_bad_pos {
                // Align to 8x8 block boundary, then show a region around it.
                let block_x = (fx / 8) * 8;
                let block_y = (fy / 8) * 8;
                // Show 3 blocks in each direction (24 pixels) for context.
                let x_start = block_x.saturating_sub(8);
                let x_end = (block_x + 16).min(width);
                let y_start = block_y.saturating_sub(8);
                let y_end = (block_y + 16).min(height);

                eprintln!(
                    "\n=== 8x8 block region around first mismatch ({}, {}) in frame {} (POC {}) ===",
                    fx, fy, bad, frame.pic_order_cnt
                );
                eprintln!(
                    "    Region: x={}..{}, y={}..{} (8x8 block at ({}, {}))",
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    block_x / 8,
                    block_y / 8
                );

                // Show ours vs FFmpeg side by side.
                eprintln!("\n--- Our Y values ---");
                eprint!("       ");
                for i in x_start..x_end {
                    eprint!("{:4}", i);
                    if i % 8 == 7 && i + 1 < x_end {
                        eprint!(" |");
                    }
                }
                eprintln!();
                for j in y_start..y_end {
                    eprint!("  y={:3}:", j);
                    for i in x_start..x_end {
                        eprint!("{:4}", frame.y[j * width + i]);
                        if i % 8 == 7 && i + 1 < x_end {
                            eprint!(" |");
                        }
                    }
                    eprintln!();
                    if j % 8 == 7 && j + 1 < y_end {
                        eprint!("       ");
                        for i in x_start..x_end {
                            eprint!("----");
                            if i % 8 == 7 && i + 1 < x_end {
                                eprint!("--");
                            }
                        }
                        eprintln!();
                    }
                }

                eprintln!("\n--- FFmpeg Y values ---");
                eprint!("       ");
                for i in x_start..x_end {
                    eprint!("{:4}", i);
                    if i % 8 == 7 && i + 1 < x_end {
                        eprint!(" |");
                    }
                }
                eprintln!();
                for j in y_start..y_end {
                    eprint!("  y={:3}:", j);
                    for i in x_start..x_end {
                        eprint!("{:4}", ref_y[j * width + i]);
                        if i % 8 == 7 && i + 1 < x_end {
                            eprint!(" |");
                        }
                    }
                    eprintln!();
                    if j % 8 == 7 && j + 1 < y_end {
                        eprint!("       ");
                        for i in x_start..x_end {
                            eprint!("----");
                            if i % 8 == 7 && i + 1 < x_end {
                                eprint!("--");
                            }
                        }
                        eprintln!();
                    }
                }

                eprintln!("\n--- Diff (ours - ffmpeg), only mismatches shown ---");
                eprint!("       ");
                for i in x_start..x_end {
                    eprint!("{:4}", i);
                    if i % 8 == 7 && i + 1 < x_end {
                        eprint!(" |");
                    }
                }
                eprintln!();
                for j in y_start..y_end {
                    eprint!("  y={:3}:", j);
                    for i in x_start..x_end {
                        let ours = frame.y[j * width + i];
                        let theirs = ref_y[j * width + i];
                        if ours == theirs {
                            eprint!("   .");
                        } else {
                            eprint!("{:+4}", ours as i32 - theirs as i32);
                        }
                        if i % 8 == 7 && i + 1 < x_end {
                            eprint!(" |");
                        }
                    }
                    eprintln!();
                    if j % 8 == 7 && j + 1 < y_end {
                        eprint!("       ");
                        for i in x_start..x_end {
                            eprint!("----");
                            if i % 8 == 7 && i + 1 < x_end {
                                eprint!("--");
                            }
                        }
                        eprintln!();
                    }
                }
            }

            // --- Per-CTU mismatch summary for the first bad frame ---
            eprintln!(
                "\n=== Per-CTU (64x64) Y mismatch counts for frame {} ===",
                bad
            );
            let ctu_cols = (width + 63) / 64;
            let ctu_rows = (height + 63) / 64;
            for ctu_r in 0..ctu_rows {
                for ctu_c in 0..ctu_cols {
                    let mut count = 0u32;
                    let y0 = ctu_r * 64;
                    let x0 = ctu_c * 64;
                    for j in y0..(y0 + 64).min(height) {
                        for i in x0..(x0 + 64).min(width) {
                            if frame.y[j * width + i] != ref_y[j * width + i] {
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        eprintln!(
                            "  CTU({},{}) [pixel ({},{})..({},{})]: {} Y mismatches",
                            ctu_c,
                            ctu_r,
                            x0,
                            y0,
                            (x0 + 64).min(width) - 1,
                            (y0 + 64).min(height) - 1,
                            count
                        );
                    }
                }
            }

            // --- Per-8x8 block mismatch summary for first bad CTU ---
            // Find first CTU with mismatches.
            'outer: for ctu_r in 0..ctu_rows {
                for ctu_c in 0..ctu_cols {
                    let y0 = ctu_r * 64;
                    let x0 = ctu_c * 64;
                    let mut ctu_has_mismatch = false;
                    for j in y0..(y0 + 64).min(height) {
                        for i in x0..(x0 + 64).min(width) {
                            if frame.y[j * width + i] != ref_y[j * width + i] {
                                ctu_has_mismatch = true;
                                break;
                            }
                        }
                        if ctu_has_mismatch {
                            break;
                        }
                    }
                    if ctu_has_mismatch {
                        eprintln!(
                            "\n=== Per-8x8 block mismatch in CTU({},{}) ===",
                            ctu_c, ctu_r
                        );
                        for blk_r in 0..8 {
                            for blk_c in 0..8 {
                                let by = y0 + blk_r * 8;
                                let bx = x0 + blk_c * 8;
                                if by >= height || bx >= width {
                                    continue;
                                }
                                let mut count = 0u32;
                                for j in by..(by + 8).min(height) {
                                    for i in bx..(bx + 8).min(width) {
                                        if frame.y[j * width + i] != ref_y[j * width + i] {
                                            count += 1;
                                        }
                                    }
                                }
                                if count > 0 {
                                    eprintln!(
                                        "  8x8 block ({},{}) [pixel ({},{})..({},{})]: {} mismatches",
                                        blk_c,
                                        blk_r,
                                        bx,
                                        by,
                                        (bx + 8).min(width) - 1,
                                        (by + 8).min(height) - 1,
                                        count
                                    );
                                }
                            }
                        }
                        break 'outer;
                    }
                }
            }

            panic!(
                "Pixel mismatch found in frame {} (POC {}). See stderr for detailed diagnostics.",
                bad, frame.pic_order_cnt
            );
        }
        eprintln!("All {} frames match FFmpeg byte-exactly!", frames.len());
    }
}
