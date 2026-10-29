//! Pure Rust H.265 / HEVC video decoder.
//!
//! Decodes Main and Main 10 profile HEVC bitstreams, and 4:2:0 and 4:4:4
//! range-extensions streams up to 12 bits with every range-extension coding
//! tool (cross-component prediction, RDPCM, transform-skip rotation and
//! context, extended precision, persistent Rice adaptation, CABAC bypass
//! alignment, chroma QP offset lists, ...). 4:4:4 covers HEIF/HEIC images.
//! Streams the decoder cannot decode correctly are rejected with
//! [`DecodeError::Unsupported`] instead of being decoded wrongly: 4:2:2,
//! monochrome, bit depths above 12, and screen content coding. Input can be
//! Annex B (start-code delimited,
//! `.h265` files) via [`parse_annex_b`], length-prefixed NAL units from
//! MP4/MKV containers via [`parse_hvcc`], or single NAL units via
//! [`parse_nal`].
//!
//! The decoder passes all 147 vectors of the JCT-VC HEVC_v1 conformance
//! suite and the 34 JCT-VC-RExt vectors in its formats (4:2:0 and 4:4:4, up
//! to 12 bits), and is byte-exact against FFmpeg on its test fixtures,
//! including real 1080p Big Buck Bunny encodes in 8-bit and 10-bit.
//!
//! # Quick start
//!
//! [`Decoder::with_display_order`] returns frames in display order, as the
//! spec's output process does. It is the recommended mode; decoding order
//! remains the default only for compatibility (see below). Take frames with
//! [`next_output`](Decoder::next_output) after each NAL unit, and call
//! [`flush`](Decoder::flush) until it returns `None` at the end of the
//! stream:
//!
//! ```no_run
//! use rust_h265::{Decoder, PixelData, parse_annex_b};
//!
//! let data = std::fs::read("input.h265")?;
//! let mut decoder = Decoder::new().with_display_order();
//! let mut frames = Vec::new();
//!
//! for nal in parse_annex_b(&data) {
//!     decoder.decode_nal(&nal)?;
//!     while let Some(frame) = decoder.next_output() {
//!         frames.push(frame);
//!     }
//! }
//! while let Some(frame) = decoder.flush() {
//!     frames.push(frame);
//! }
//!
//! for frame in &frames {
//!     match &frame.y {
//!         PixelData::U8(luma) => { /* 8-bit samples */ }
//!         PixelData::U16(luma) => { /* 10-bit samples */ }
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Decoding order
//!
//! Prefer display order. A decoder made with [`Decoder::new`] alone
//! returns each picture from [`decode_nal`](Decoder::decode_nal) as soon as
//! it is decoded, in decoding order, and returns every decoded picture,
//! including those the spec does not output. This suits callers that pair
//! each frame with the container packet it came from. To display such frames, sort them by
//! [`pic_order_cnt`](Frame::pic_order_cnt) within each coded video
//! sequence: POC restarts at every IDR or BLA picture, and at a CRA
//! picture that follows an end-of-sequence NAL unit.
//!
//! # Frames
//!
//! A [`Frame`] holds cropped `y`, `u` and `v` planes as [`PixelData`]:
//! `U8` for 8-bit streams, `U16` for deeper ones. `chroma_format_idc` is 1
//! for 4:2:0, where `u` and `v` have half the width and height, and 3 for
//! 4:4:4, where they are full size.
//!
//! # Resource limits
//!
//! [`Decoder::new`] rejects any SPS whose picture exceeds the Level 6.2
//! maximum of 35,651,584 luma samples or an estimated 512 MB of
//! per-picture state, with [`DecodeError::ResourceLimit`]. Use
//! [`Decoder::with_limits`] to change either bound.

// --- Internal modules (not part of the public API) ---
// allow(dead_code): internal modules contain utility functions, struct fields,
// and table constants kept for spec completeness / future use.
#[allow(dead_code)]
mod bitstream;
#[allow(dead_code)]
mod cabac;
#[allow(dead_code)]
mod cabac_tables;
#[allow(dead_code)]
mod cu_tree;
#[allow(dead_code)]
mod deblock;
#[allow(dead_code)]
mod inter_pred;
#[allow(dead_code)]
mod intra_pred;
#[allow(dead_code)]
mod inverse_transform;
mod output;
#[allow(dead_code)]
pub(crate) mod pixel;
#[allow(dead_code)]
mod pps;
#[allow(dead_code)]
mod profile_tier_level;
#[allow(dead_code)]
mod residual_coding;
#[allow(dead_code)]
mod sao;
#[allow(dead_code)]
mod scaling_list;
#[allow(dead_code)]
mod slice;
#[allow(dead_code)]
mod sps;
#[allow(dead_code)]
mod vps;

// --- Internal but needs pub(crate) access from decoder ---
#[allow(dead_code)]
pub(crate) mod dpb;

// --- Public modules ---
pub mod decoder;
pub mod error;
pub mod nal;

// --- Top-level re-exports for convenience ---
pub use decoder::{DecodeLimits, Decoder, Frame};
pub use error::DecodeError;
pub use nal::{NalUnit, NalUnitType, parse_annex_b, parse_hvcc, parse_nal};
pub use pixel::{Pixel, PixelData};

#[cfg(test)]
mod parameter_set_tests {
    use crate::nal::{NalUnitType, parse_annex_b};

    /// Decode VPS+SPS+PPS from a real x265-encoded fixture and assert the
    /// fields a downstream slice/CTU loop will need.
    ///
    /// Fixture command (run from the crate root):
    /// ```text
    /// ffmpeg -hide_banner -loglevel error \
    ///   -f lavfi -i color=gray:size=16x16:rate=30:duration=0.04 \
    ///   -frames:v 1 -pix_fmt yuv420p -f rawvideo testdata/tiny_input.yuv
    /// x265 --input-res 16x16 --fps 30 --frames 1 --keyint 1 --bframes 0 \
    ///   --ctu 16 --input testdata/tiny_input.yuv -o testdata/tiny_intra.h265
    /// ```
    #[test]
    fn parse_tiny_intra_parameter_sets() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tiny_intra.h265");
        let data = std::fs::read(path).expect("read fixture");
        let nals = parse_annex_b(&data);

        // Find the parameter set NALs. The fixture also contains a PrefixSei
        // (the x265 build banner) and the IDR slice NAL — we don't need them.
        let vps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Vps)
            .expect("VPS NAL");
        let sps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Sps)
            .expect("SPS NAL");
        let pps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Pps)
            .expect("PPS NAL");

        // ---- VPS ----
        let vps = crate::vps::parse_vps(&vps_nal.rbsp).expect("parse VPS");
        assert_eq!(vps.vps_video_parameter_set_id, 0);
        assert_eq!(vps.vps_max_layers_minus1, 0);
        assert_eq!(vps.vps_max_sub_layers_minus1, 0);
        assert!(vps.vps_temporal_id_nesting_flag);
        // Main Still Picture profile, level 1.0
        assert_eq!(vps.profile_tier_level.general_profile_idc, 3);
        assert_eq!(vps.profile_tier_level.general_level_idc, 30);
        assert!(!vps.profile_tier_level.general_tier_flag);

        // ---- SPS ----
        let sps = crate::sps::parse_sps(&sps_nal.rbsp).expect("parse SPS");
        assert_eq!(sps.sps_video_parameter_set_id, 0);
        assert_eq!(sps.sps_seq_parameter_set_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 16);
        assert_eq!(sps.pic_height_in_luma_samples, 16);
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
        // log2_max_pic_order_cnt_lsb_minus4 = 4 -> MaxPicOrderCntLsb = 256
        assert_eq!(sps.log2_max_pic_order_cnt_lsb, 8);

        // CTU derivation: log2_min_cb_minus3=0, log2_diff=1
        // -> MinCbLog2SizeY=3, CtbLog2SizeY=4, CtbSizeY=16.
        assert_eq!(sps.min_cb_log2_size_y, 3);
        assert_eq!(sps.ctb_log2_size_y, 4);
        assert_eq!(sps.ctb_size_y, 16);
        assert_eq!(sps.pic_width_in_ctbs_y(), 1);
        assert_eq!(sps.pic_height_in_ctbs_y(), 1);

        // TU derivation: min_tb_minus2=0, diff=2 -> 2..=4 (4x4..16x16)
        assert_eq!(sps.min_tb_log2_size_y, 2);
        assert_eq!(sps.max_tb_log2_size_y, 4);

        assert!(!sps.scaling_list_enabled_flag);
        assert!(!sps.amp_enabled_flag);
        assert!(!sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.pcm_enabled_flag);
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(!sps.long_term_ref_pics_present_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(!sps.strong_intra_smoothing_enabled_flag);

        // ---- PPS ----
        let pps = crate::pps::parse_pps(&pps_nal.rbsp).expect("parse PPS");
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert!(!pps.dependent_slice_segments_enabled_flag);
        assert!(!pps.output_flag_present_flag);
        assert_eq!(pps.num_extra_slice_header_bits, 0);
        assert!(!pps.sign_data_hiding_enabled_flag);
        assert!(!pps.cabac_init_present_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        assert_eq!(pps.init_qp, 26);
        assert!(pps.cu_qp_delta_enabled_flag);
        assert_eq!(pps.diff_cu_qp_delta_depth, 0);
        assert_eq!(pps.pps_cb_qp_offset, 0);
        assert_eq!(pps.pps_cr_qp_offset, 0);
        assert!(!pps.weighted_pred_flag);
        assert!(!pps.weighted_bipred_flag);
        assert!(!pps.tiles_enabled_flag);
        assert!(!pps.entropy_coding_sync_enabled_flag);
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(!pps.deblocking_filter_override_enabled_flag);
        assert!(pps.pps_deblocking_filter_disabled_flag);
        assert!(!pps.lists_modification_present_flag);
        assert_eq!(pps.log2_parallel_merge_level_minus2, 0);
        assert!(!pps.slice_segment_header_extension_present_flag);
    }
}
