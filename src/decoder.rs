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
            // Populate `tab_tile_id` on the fresh picture state so intra
            // availability checks can see it.
            let mut ps = PictureState::new(sps);
            let n = ps.tab_tile_id.len();
            ps.tab_tile_id.copy_from_slice(&tile_tables.tile_id[..n]);
            self.current_picture = Some(PictureInProgress {
                state: ps,
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

        let wpp = pps.entropy_coding_sync_enabled_flag;
        let tiles_on = pps.tiles_enabled_flag;
        let slice_start_ts = sh.slice_segment_address;

        // Phase 3c-3 (WPP): saved CABAC context state captured after the
        // second CTB of each row, to be loaded at the start of the next row.
        // Not used by tiles — tiles reinit from the slice QP at every tile
        // boundary instead.
        let mut saved_state: Option<[u8; crate::cabac_tables::HEVC_CONTEXTS]> = None;

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
                    contexts = CabacContexts::init(sh.slice_qp_y, sh.slice_type, false);
                } else {
                    // WPP row start: fresh init on single-column pictures,
                    // otherwise load the state saved after the previous
                    // row's 2nd CTB.
                    if pic_width_in_ctbs == 1 {
                        contexts = CabacContexts::init(sh.slice_qp_y, sh.slice_type, false);
                    } else if let Some(saved) = saved_state.as_ref() {
                        contexts.state.copy_from_slice(saved);
                    } else {
                        return Err(DecodeError::InvalidSyntax(
                            "WPP row start without a saved context state",
                        ));
                    }
                }
            }

            let x_ctb = col * ctb_size;
            let y_ctb = (ctb_addr_rs / pic_width_in_ctbs) * ctb_size;
            // Phase 3b-2: per-CTB SAO parameters decoded BEFORE the coding tree.
            let rx = (x_ctb >> sps.ctb_log2_size_y) as usize;
            let ry = (y_ctb >> sps.ctb_log2_size_y) as usize;
            // Record the slice this CTB belongs to BEFORE decoding, so the
            // intra prediction availability check can see the current CTB's
            // slice address. `tab_slice_addr_rs` is indexed by raster.
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

    /// **Phase 3c-2 byte-exact test**: 128×64 flat-gray intra frame with
    /// `--tiles 2x1`, encoded by kvazaar (x265 has no tile support). Tests:
    ///
    /// - PPS `tiles_enabled_flag = 1` parsing (no longer rejected)
    /// - `num_tile_columns_minus1` / `num_tile_rows_minus1` / uniform spacing
    /// - `Pps::resolve_tile_geometry` producing `column_widths_in_ctbs`
    /// - `TileScanTables::build` building the raster↔tile-scan mapping
    /// - Per-tile CABAC reinit at the tile boundary (byte offset + state)
    /// - CTB iteration in tile-scan order
    /// - `tab_tile_id` population + intra-availability tile boundary check
    /// - Byte-exact match against FFmpeg
    ///
    /// The fixture is generated at runtime by kvazaar. If kvazaar or ffmpeg
    /// are not on the `PATH` the test silently skips, matching the
    /// convention of the other dynamic fixtures above.
    #[test]
    fn test_decode_tiles_byte_exact() {
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let input_yuv = tmp.join("tiles_input.yuv");
        let h265_path = tmp.join("tiles.h265");
        let ref_yuv_path = tmp.join("tiles_ref.yuv");

        // 256×256 flat gray input. At kvazaar's default CTU=64 this is
        // 4×4 CTBs. With a 2×2 tile layout each tile is 2×2 CTBs, which
        // gives genuine reordering between raster and tile scan (e.g.
        // raster CTB 2 has tile-scan index 4), exercising the real reorder
        // path in `decode_slice` (not just the identity mapping of a
        // 1 CTB per tile picture).
        //
        // We keep luma flat gray (not striped) because the upstream decoder
        // does not yet support chroma residual coding; a striped pattern
        // would generate non-zero chroma residuals that x265/kvazaar will
        // happily encode but our decoder can't dequantize yet.
        let w: usize = 256;
        let h: usize = 256;
        let mut yuv_data = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
        yuv_data.extend(std::iter::repeat_n(0x7Eu8, w * h));
        yuv_data.extend(std::iter::repeat_n(128u8, (w / 2) * (h / 2) * 2));
        std::fs::write(&input_yuv, &yuv_data).expect("write input yuv");

        // kvazaar encode. `--slices tiles` puts each tile in its own slice
        // segment, but even without it kvazaar produces a single slice with
        // one entry point per tile boundary. We go with the single-slice
        // form here because our multi-slice handling is already covered by
        // `test_decode_multi_slice_byte_exact`.
        //
        // Notes:
        // - `--gop 0` → all-intra (no B/P frames).
        // - `--period 1` → every frame is a key frame.
        // - Loop filters disabled to keep the fixture scope minimal.
        let kvz = Command::new("/opt/homebrew/bin/kvazaar")
            .args([
                "--input",
                input_yuv.to_str().unwrap(),
                "--input-res",
                "256x256",
                "--input-fps",
                "1",
                "--frames",
                "1",
                "--output",
                h265_path.to_str().unwrap(),
                "--preset",
                "ultrafast",
                "--tiles",
                "2x2",
                "--no-wpp",
                "--no-sao",
                "--no-deblock",
                "--no-signhide",
                "--gop",
                "0",
                "--period",
                "1",
                "--qp",
                "25",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let kvz = match kvz {
            Ok(s) => s,
            Err(_) => {
                eprintln!("kvazaar not found, skipping tiles fixture test");
                return;
            }
        };
        assert!(kvz.success(), "kvazaar encoding failed");

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
                eprintln!("ffmpeg not found, skipping tiles fixture test");
                return;
            }
        };
        assert!(ffmpeg_status.success(), "ffmpeg decoding failed");

        let h265 = std::fs::read(&h265_path).expect("read h265 fixture");
        let ref_yuv = std::fs::read(&ref_yuv_path).expect("read reference yuv");
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
}
