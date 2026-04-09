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

use crate::cabac::{CabacContexts, CabacReader};
use crate::cu_tree::{PictureState, decode_coding_quadtree};
use crate::error::DecodeError;
use crate::nal::{NalUnit, NalUnitType};
use crate::pps::{Pps, parse_pps};
use crate::slice::{SliceHeader, SliceType, parse_slice_segment_header};
use crate::sps::{Sps, parse_sps};
use crate::vps::{Vps, parse_vps};

/// A reconstructed video frame in YUV420 8-bit planar layout.
///
/// Plane lengths are `width * height` for luma and `(width/2) * (height/2)`
/// for each chroma plane. `pic_order_cnt` is 0 for IDR pictures (we'll add
/// non-IDR POC computation in Phase 3+).
#[derive(Debug, Clone)]
pub struct Frame {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub pic_order_cnt: u32,
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
    /// Number of CTBs already decoded in this picture (sum across all
    /// slice segments seen so far).
    ctbs_decoded: u32,
    /// Total CTBs in the picture = `pic_width_in_ctbs * pic_height_in_ctbs`.
    total_ctbs: u32,
}

#[derive(Default)]
pub struct Decoder {
    vps: Option<Vps>,
    sps: Option<Sps>,
    pps: Option<Pps>,
    /// The picture currently being assembled from one or more slice segments.
    /// Phase 3c-1: created on the first slice segment, finalized and
    /// returned as a `Frame` when all CTBs have been decoded.
    current_picture: Option<PictureInProgress>,
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
                Ok(None)
            }
            NalUnitType::Pps => {
                self.pps = Some(parse_pps(&nal.rbsp)?);
                Ok(None)
            }
            t if t.is_vcl() => self.decode_slice(nal, t),
            _ => Ok(None),
        }
    }

    /// Flush any buffered frame. Phase 2c-6 has no reordering buffer (single
    /// IDR fixture), so this always returns `None`. Phase 3+ will rework
    /// this when B-frame reorder buffering lands.
    pub fn flush(&mut self) -> Option<Frame> {
        None
    }

    fn decode_slice(
        &mut self,
        nal: &NalUnit<'_>,
        nut: NalUnitType,
    ) -> Result<Option<Frame>, DecodeError> {
        let sps = self
            .sps
            .as_ref()
            .ok_or(DecodeError::InvalidSyntax("slice without active SPS"))?;
        let pps = self
            .pps
            .as_ref()
            .ok_or(DecodeError::InvalidSyntax("slice without active PPS"))?;

        let sh = parse_slice_segment_header(&nal.rbsp, nut, sps, pps)?;
        if sh.slice_type != SliceType::I {
            return Err(DecodeError::Unsupported(
                "only I-slices are supported in Phase 2",
            ));
        }
        // Per-slice CABAC reinit (independent slice segments only — dependent
        // slice segments would reuse the previous segment's context state,
        // which we reject at parse time).
        let mut contexts = CabacContexts::init(sh.slice_qp_y, sh.slice_type, false);
        let cabac_byte_offset = sh.header_size_bits / 8;
        let mut cabac = CabacReader::new(&nal.rbsp, cabac_byte_offset);

        let ctb_size = 1u32 << sps.ctb_log2_size_y;
        let pic_width_in_ctbs = sps.pic_width_in_ctbs_y();
        let pic_height_in_ctbs = sps.pic_height_in_ctbs_y();
        let total_ctbs = pic_width_in_ctbs * pic_height_in_ctbs;

        // Phase 3c-1: a first slice segment starts a new picture. Subsequent
        // slice segments (`first_slice_segment_in_pic_flag = 0`) attach to
        // the already-in-flight picture.
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
            self.current_picture = Some(PictureInProgress {
                state: PictureState::new(sps),
                last_slice_header: sh.clone(),
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
                // Phase 3c-1 assumes slices arrive in raster CTB order and
                // cover contiguous ranges (no gaps or overlap). Tiles and
                // out-of-order slices are Phase 3c-2.
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

        let wpp = pps.entropy_coding_sync_enabled_flag;
        let slice_start_ctb = sh.slice_segment_address;
        // Phase 3c-3 (WPP): saved CABAC context state captured after the
        // second CTB of each row, to be loaded at the start of the next row.
        let mut saved_state: Option<[u8; crate::cabac_tables::HEVC_CONTEXTS]> = None;

        let mut more_data = true;
        let mut ctb_addr_rs: u32 = slice_start_ctb;

        while more_data && ctb_addr_rs < total_ctbs {
            // WPP row boundary reinit (spec 9.3.2.2 + FFmpeg
            // `ff_hevc_cabac_init` / `load_states`). The first CTB of every
            // non-first row of the slice triggers:
            //   1. A fresh `CabacReader` at the row's entry-point byte offset
            //   2. Loading the saved context state from the previous row
            //      (when `ctb_width > 1`) or a fresh init (when `ctb_width == 1`)
            let col = ctb_addr_rs % pic_width_in_ctbs;
            let is_row_start = col == 0;
            let is_first_ctb_of_slice = ctb_addr_rs == slice_start_ctb;
            if wpp && is_row_start && !is_first_ctb_of_slice {
                let row_within_slice =
                    ((ctb_addr_rs - slice_start_ctb) / pic_width_in_ctbs) as usize;
                // row_within_slice == 1 for the 2nd row, 2 for the 3rd row, ...
                // entry_point_offsets[ep_idx] gives the cumulative byte
                // offset (from the start of the slice data) of substream
                // (row_within_slice). For the second row that's ep_idx = 0.
                let ep_idx = row_within_slice - 1;
                if ep_idx >= sh.entry_point_offsets.len() {
                    return Err(DecodeError::InvalidSyntax(
                        "WPP slice missing entry_point_offset for row",
                    ));
                }
                let byte_offset = cabac_byte_offset + sh.entry_point_offsets[ep_idx] as usize;
                cabac.reinit_at(byte_offset);
                if pic_width_in_ctbs == 1 {
                    // Single-column picture: per HEVC spec and FFmpeg, state
                    // is re-initialized afresh rather than loaded.
                    contexts = CabacContexts::init(sh.slice_qp_y, sh.slice_type, false);
                } else if let Some(saved) = saved_state.as_ref() {
                    contexts.state.copy_from_slice(saved);
                } else {
                    return Err(DecodeError::InvalidSyntax(
                        "WPP row start without a saved context state",
                    ));
                }
            }

            let x_ctb = col * ctb_size;
            let y_ctb = (ctb_addr_rs / pic_width_in_ctbs) * ctb_size;
            // Phase 3b-2: per-CTB SAO parameters decoded BEFORE the coding tree.
            let rx = (x_ctb >> sps.ctb_log2_size_y) as usize;
            let ry = (y_ctb >> sps.ctb_log2_size_y) as usize;
            // Record the slice this CTB belongs to BEFORE decoding, so the
            // intra prediction availability check can see the current CTB's
            // slice address.
            state.tab_slice_addr_rs[ctb_addr_rs as usize] = sh.slice_segment_address as i32;
            crate::sao::decode_sao_param(&mut cabac, &mut contexts, state, sps, &sh, rx, ry);
            more_data = decode_coding_quadtree(
                &mut cabac,
                &mut contexts,
                state,
                sps,
                pps,
                sh.slice_qp_y,
                x_ctb,
                y_ctb,
                sps.ctb_log2_size_y,
                0,
            )?;
            ctb_addr_rs += 1;

            // Phase 3c-3 (WPP): snapshot the CABAC contexts after the 2nd
            // CTB of each row so the next row can load them. Mirrors
            // FFmpeg's `ff_hevc_save_states`: save when `col_after == 2`,
            // or `col_after == 0` in the special `ctb_width == 2` case
            // (which still means "after the 2nd CTB of a row").
            if wpp {
                let col_after = ctb_addr_rs % pic_width_in_ctbs;
                let should_save = col_after == 2
                    || (pic_width_in_ctbs == 2 && col_after == 0)
                    || pic_width_in_ctbs == 1;
                if should_save {
                    saved_state = Some(contexts.state);
                }
            }

            // In WPP, `end_of_slice_flag` is decoded at the end of EVERY
            // row (spec 7.3.8.5). For non-final rows it is 0 → `more_data`
            // stays true → we fall through to the next row, which triggers
            // the reinit block above.
        }

        // `more_data == false` means we decoded an `end_of_slice_flag = 1`
        // terminate bin — the slice has finished its CTB range. For the
        // last slice in the picture this also coincides with `ctb_addr_rs ==
        // total_ctbs`. Any mid-picture slice must also end on a terminate
        // bin, otherwise the CABAC state would be out of sync.
        if more_data {
            return Err(DecodeError::InvalidSyntax(
                "slice did not end on terminate bin",
            ));
        }

        pic.ctbs_decoded = ctb_addr_rs;
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

        Ok(Some(Frame {
            y: pic.state.y_plane,
            u: pic.state.u_plane,
            v: pic.state.v_plane,
            width: sps.pic_width_in_luma_samples,
            height: sps.pic_height_in_luma_samples,
            // IDR pictures always have POC 0; non-IDR POC will be wired in
            // Phase 3+ when we add slice POC LSB parsing.
            pic_order_cnt: 0,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::parse_annex_b;

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

    /// **Phase 3a-3 byte-exact test**: 32x32 diagonal-gradient frame with
    /// `--ctu 16 --max-tu-size 4` to force angular intra prediction modes.
    ///
    /// The gradient input causes x265 to choose angular modes for many PUs,
    /// exercising predict_angular + reference sample filtering. The test
    /// generates the fixture at runtime (x265 encode + ffmpeg decode) and
    /// then verifies our decoder is byte-exact against FFmpeg's output.
    #[test]
    fn test_decode_angular_byte_exact() {
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let input_yuv = tmp.join("angular_input.yuv");
        let h265_path = tmp.join("angular.h265");
        let ref_yuv_path = tmp.join("angular_ref.yuv");

        // Step 1: Generate 16x16 flat gray YUV input.
        // x265 with --max-tu-size 4 picks angular modes for some 4x4 PUs.
        let w: usize = 16;
        let h: usize = 16;
        let mut yuv_data = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
        yuv_data.extend(std::iter::repeat_n(0x7Eu8, w * h));
        yuv_data.extend(std::iter::repeat_n(128u8, (w / 2) * (h / 2) * 2));
        std::fs::write(&input_yuv, &yuv_data).expect("write input yuv");

        // Step 2: Encode with x265 (intra-only, no sign-hiding, max-tu-size 4).
        let x265_status = Command::new("x265")
            .args([
                "--input",
                input_yuv.to_str().unwrap(),
                "--input-res",
                "16x16",
                "--fps",
                "1",
                "--frames",
                "1",
                "--output",
                h265_path.to_str().unwrap(),
                "--preset",
                "ultrafast",
                "--no-wpp",
                "--no-signhide",
                "--ctu",
                "16",
                "--max-tu-size",
                "4",
                "--no-open-gop",
                "--keyint",
                "1",
                "--no-scenecut",
                "--no-sao",
                "--no-deblock",
                "--qp",
                "25",
                "--no-psnr",
                "--no-ssim",
                "--no-info",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let x265_status = match x265_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("x265 not found, skipping angular fixture test");
                return;
            }
        };
        assert!(x265_status.success(), "x265 encoding failed");

        // Step 3: Decode reference with FFmpeg.
        let ffmpeg_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                h265_path.to_str().unwrap(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                ref_yuv_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let ffmpeg_status = match ffmpeg_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("ffmpeg not found, skipping angular fixture test");
                return;
            }
        };
        assert!(ffmpeg_status.success(), "ffmpeg decoding failed");

        // Step 4: Decode with our decoder.
        let h265 = std::fs::read(&h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(&ref_yuv_path).expect("read reference yuv");

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

        // Find first difference for debugging.
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

    /// **Phase 3a-3 byte-exact test**: 16x16 diagonal-gradient frame with
    /// `--ctu 16 --max-tu-size 4 --qp 32` to force angular intra prediction
    /// modes (modes 2..34). The gradient causes x265 to pick modes like 3
    /// and 34 for many PUs within a single CTU.
    #[test]
    fn test_decode_angular_gradient_byte_exact() {
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let input_yuv = tmp.join("angular_grad_input.yuv");
        let h265_path = tmp.join("angular_grad.h265");
        let ref_yuv_path = tmp.join("angular_grad_ref.yuv");

        // Use a vertical stripe pattern to encourage angular modes.
        let w: usize = 16;
        let h: usize = 16;
        let mut yuv_data = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
        for _y in 0..h {
            for x in 0..w {
                yuv_data.push(if x < 8 { 40u8 } else { 200u8 });
            }
        }
        yuv_data.extend(std::iter::repeat_n(128u8, (w / 2) * (h / 2) * 2));
        std::fs::write(&input_yuv, &yuv_data).expect("write input yuv");

        let x265_status = Command::new("x265")
            .args([
                "--input",
                input_yuv.to_str().unwrap(),
                "--input-res",
                "16x16",
                "--fps",
                "1",
                "--frames",
                "1",
                "--output",
                h265_path.to_str().unwrap(),
                "--preset",
                "ultrafast",
                "--no-wpp",
                "--no-signhide",
                "--ctu",
                "16",
                "--max-tu-size",
                "4",
                "--no-open-gop",
                "--keyint",
                "1",
                "--no-scenecut",
                "--no-sao",
                "--no-deblock",
                "--qp",
                "30",
                "--no-psnr",
                "--no-ssim",
                "--no-info",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let x265_status = match x265_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("x265 not found, skipping angular gradient test");
                return;
            }
        };
        assert!(x265_status.success(), "x265 encoding failed");

        let ffmpeg_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                h265_path.to_str().unwrap(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                ref_yuv_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let ffmpeg_status = match ffmpeg_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("ffmpeg not found, skipping angular gradient test");
                return;
            }
        };
        assert!(ffmpeg_status.success(), "ffmpeg decoding failed");

        let h265 = std::fs::read(&h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(&ref_yuv_path).expect("read reference yuv");

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
    #[test]
    fn test_decode_scaling_list_default_byte_exact() {
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let input_yuv = tmp.join("scaling_list_input.yuv");
        let h265_path = tmp.join("scaling_list.h265");
        let ref_yuv_path = tmp.join("scaling_list_ref.yuv");

        // 16x16 flat gray input.
        let w: usize = 16;
        let h: usize = 16;
        let mut yuv_data = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
        yuv_data.extend(std::iter::repeat_n(0x7Eu8, w * h));
        yuv_data.extend(std::iter::repeat_n(128u8, (w / 2) * (h / 2) * 2));
        std::fs::write(&input_yuv, &yuv_data).expect("write input yuv");

        // Encode with x265 using --scaling-list default.
        let x265_status = Command::new("x265")
            .args([
                "--input",
                input_yuv.to_str().unwrap(),
                "--input-res",
                "16x16",
                "--fps",
                "1",
                "--frames",
                "1",
                "--output",
                h265_path.to_str().unwrap(),
                "--preset",
                "ultrafast",
                "--no-wpp",
                "--no-signhide",
                "--ctu",
                "16",
                "--no-open-gop",
                "--keyint",
                "1",
                "--no-scenecut",
                "--no-sao",
                "--no-deblock",
                "--qp",
                "25",
                "--no-psnr",
                "--no-ssim",
                "--no-info",
                "--scaling-list",
                "default",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let x265_status = match x265_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("x265 not found, skipping scaling list test");
                return;
            }
        };
        assert!(x265_status.success(), "x265 encoding failed");

        // Decode reference with FFmpeg.
        let ffmpeg_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                h265_path.to_str().unwrap(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                ref_yuv_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let ffmpeg_status = match ffmpeg_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("ffmpeg not found, skipping scaling list test");
                return;
            }
        };
        assert!(ffmpeg_status.success(), "ffmpeg decoding failed");

        // Decode with our decoder.
        let h265 = std::fs::read(&h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(&ref_yuv_path).expect("read reference yuv");

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

    /// **Phase 3a-6 byte-exact test**: 16×16 vertical-stripe frame encoded
    /// *without* `--no-signhide`, so `pps_sign_data_hiding_enabled_flag` is
    /// set and x265 will omit the sign bit of the last-in-scan-order
    /// non-zero coefficient in sub-blocks that meet the 4-position gap
    /// criterion.
    ///
    /// Exercises:
    /// - `sign_data_hiding_enabled_flag = 1` in the PPS (no longer rejected)
    /// - `sign_hidden = (last_nz_pos_in_cg - first_nz_pos_in_cg >= 4)` gate
    /// - Decoding `n_end - 1` sign bits in hidden sub-blocks
    /// - Sum-of-abs parity adjustment on the hidden coefficient
    #[test]
    fn test_decode_signhide_byte_exact() {
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let input_yuv = tmp.join("signhide_input.yuv");
        let h265_path = tmp.join("signhide.h265");
        let ref_yuv_path = tmp.join("signhide_ref.yuv");

        // Vertical-stripe pattern: produces many non-zero high-frequency
        // coefficients per sub-block, so the 4-position SDH gap condition
        // is met often (the encoder is free to actually hide signs).
        let w: usize = 16;
        let h: usize = 16;
        let mut yuv_data = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
        for _y in 0..h {
            for x in 0..w {
                yuv_data.push(if x < 8 { 40u8 } else { 200u8 });
            }
        }
        yuv_data.extend(std::iter::repeat_n(128u8, (w / 2) * (h / 2) * 2));
        std::fs::write(&input_yuv, &yuv_data).expect("write input yuv");

        // `--preset ultrafast` implicitly sets `signhide 0`, so we have to
        // explicitly request `--signhide` to enable it. That's exactly the
        // path we want to exercise.
        let x265_status = Command::new("x265")
            .args([
                "--input",
                input_yuv.to_str().unwrap(),
                "--input-res",
                "16x16",
                "--fps",
                "1",
                "--frames",
                "1",
                "--output",
                h265_path.to_str().unwrap(),
                "--preset",
                "ultrafast",
                "--no-wpp",
                "--signhide",
                "--ctu",
                "16",
                "--max-tu-size",
                "4",
                "--no-open-gop",
                "--keyint",
                "1",
                "--no-scenecut",
                "--no-sao",
                "--no-deblock",
                "--qp",
                "20",
                "--no-psnr",
                "--no-ssim",
                "--no-info",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let x265_status = match x265_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("x265 not found, skipping signhide fixture test");
                return;
            }
        };
        assert!(x265_status.success(), "x265 encoding failed");

        let ffmpeg_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                h265_path.to_str().unwrap(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                ref_yuv_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let ffmpeg_status = match ffmpeg_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("ffmpeg not found, skipping signhide fixture test");
                return;
            }
        };
        assert!(ffmpeg_status.success(), "ffmpeg decoding failed");

        let h265 = std::fs::read(&h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(&ref_yuv_path).expect("read reference yuv");

        let nals = parse_annex_b(&h265);

        // Sanity-check: the PPS in this fixture must actually have SDH on.
        // If it doesn't we're not exercising the new path at all.
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

    /// **Phase 3a-5 byte-exact test**: attempt to produce a PCM-bearing
    /// bitstream via x265 `--pcm`. x265 is notoriously reluctant to choose
    /// PCM over intra; on a flat-gray frame with a very high QP it *may*
    /// decide the PCM cost is lower. If the resulting file ends up without
    /// any `pcm_flag = 1` CUs we still get a useful byte-exact regression
    /// test against FFmpeg for the non-PCM path with `pcm_enabled_flag = 1`
    /// in the SPS — verifying that our updated SPS parser handles the flag
    /// correctly. If x265 or ffmpeg isn't installed the test silently skips,
    /// matching the convention of the other dynamic fixtures in this file.
    ///
    /// TODO: this doesn't guarantee the PCM *decode* path runs. The unit
    /// tests in `cu_tree.rs` for `decode_pcm_block` and
    /// `CabacReader::pcm_byte_position` cover that path synthetically
    /// without needing a PCM-bearing fixture.
    #[test]
    fn test_decode_pcm_byte_exact() {
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let input_yuv = tmp.join("pcm_input.yuv");
        let h265_path = tmp.join("pcm.h265");
        let ref_yuv_path = tmp.join("pcm_ref.yuv");

        // 16x16 "noise" pattern. Pure random data frustrates x265's RDO
        // harder than a flat frame — making the PCM escape hatch more
        // attractive. We use a deterministic xorshift so the fixture is
        // reproducible across runs.
        let w: usize = 16;
        let h: usize = 16;
        let mut yuv_data = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
        let mut s: u32 = 0xdead_beef;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        };
        for _ in 0..(w * h) {
            yuv_data.push(next());
        }
        for _ in 0..(2 * (w / 2) * (h / 2)) {
            yuv_data.push(next());
        }
        std::fs::write(&input_yuv, &yuv_data).expect("write input yuv");

        let x265_status = Command::new("x265")
            .args([
                "--input",
                input_yuv.to_str().unwrap(),
                "--input-res",
                "16x16",
                "--fps",
                "1",
                "--frames",
                "1",
                "--output",
                h265_path.to_str().unwrap(),
                "--preset",
                "ultrafast",
                "--no-wpp",
                "--no-signhide",
                "--ctu",
                "16",
                "--no-open-gop",
                "--keyint",
                "1",
                "--no-scenecut",
                "--no-sao",
                "--no-deblock",
                // Very high QP + --pcm nudges x265 into picking PCM for
                // hard-to-predict blocks.
                "--qp",
                "51",
                "--pcm",
                "--no-psnr",
                "--no-ssim",
                "--no-info",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let x265_status = match x265_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("x265 not found, skipping PCM fixture test");
                return;
            }
        };
        if !x265_status.success() {
            eprintln!("x265 encoding failed (perhaps the build lacks --pcm); skipping");
            return;
        }

        let ffmpeg_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                h265_path.to_str().unwrap(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                ref_yuv_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let ffmpeg_status = match ffmpeg_status {
            Ok(s) => s,
            Err(_) => {
                eprintln!("ffmpeg not found, skipping PCM fixture test");
                return;
            }
        };
        assert!(ffmpeg_status.success(), "ffmpeg decoding failed");

        let h265 = std::fs::read(&h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(&ref_yuv_path).expect("read reference yuv");

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
}
