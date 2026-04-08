//! Pure Rust H.265 / HEVC video decoder.
//!
//! See `CLAUDE.md` and `TODO.md` for the implementation plan. As of Phase 1
//! the bitstream reader, NAL parser, and parameter set parsers (VPS / SPS /
//! PPS) are implemented; slice decoding has not started.

#[allow(dead_code)]
mod bitstream;
#[allow(dead_code)]
mod cabac;
#[allow(dead_code)]
mod cabac_tables;
#[allow(dead_code)]
mod cu_tree;
pub mod error;
pub mod nal;
#[allow(dead_code)]
mod pps;
#[allow(dead_code)]
mod profile_tier_level;
#[allow(dead_code)]
mod slice;
#[allow(dead_code)]
mod sps;
#[allow(dead_code)]
mod vps;

#[cfg(test)]
mod parameter_set_tests {
    use super::*;
    use crate::nal::{parse_annex_b, NalUnitType};

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
        let vps = vps::parse_vps(&vps_nal.rbsp).expect("parse VPS");
        assert_eq!(vps.vps_video_parameter_set_id, 0);
        assert_eq!(vps.vps_max_layers_minus1, 0);
        assert_eq!(vps.vps_max_sub_layers_minus1, 0);
        assert!(vps.vps_temporal_id_nesting_flag);
        // Main Still Picture profile, level 1.0
        assert_eq!(vps.profile_tier_level.general_profile_idc, 3);
        assert_eq!(vps.profile_tier_level.general_level_idc, 30);
        assert!(!vps.profile_tier_level.general_tier_flag);

        // ---- SPS ----
        let sps = sps::parse_sps(&sps_nal.rbsp).expect("parse SPS");
        assert_eq!(sps.sps_video_parameter_set_id, 0);
        assert_eq!(sps.sps_seq_parameter_set_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 16);
        assert_eq!(sps.pic_height_in_luma_samples, 16);
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
        // log2_max_pic_order_cnt_lsb_minus4 = 4 → MaxPicOrderCntLsb = 256
        assert_eq!(sps.log2_max_pic_order_cnt_lsb, 8);

        // CTU derivation: log2_min_cb_minus3=0, log2_diff=1
        // → MinCbLog2SizeY=3, CtbLog2SizeY=4, CtbSizeY=16.
        assert_eq!(sps.min_cb_log2_size_y, 3);
        assert_eq!(sps.ctb_log2_size_y, 4);
        assert_eq!(sps.ctb_size_y, 16);
        assert_eq!(sps.pic_width_in_ctbs_y(), 1);
        assert_eq!(sps.pic_height_in_ctbs_y(), 1);

        // TU derivation: min_tb_minus2=0, diff=2 → 2..=4 (4×4..16×16)
        assert_eq!(sps.min_tb_log2_size_y, 2);
        assert_eq!(sps.max_tb_log2_size_y, 4);

        assert!(!sps.scaling_list_enabled_flag);
        assert!(!sps.amp_enabled_flag);
        // Fixture is encoded with --no-sao --no-strong-intra-smoothing.
        assert!(!sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.pcm_enabled_flag);
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(!sps.long_term_ref_pics_present_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(!sps.strong_intra_smoothing_enabled_flag);

        // ---- PPS ----
        let pps = pps::parse_pps(&pps_nal.rbsp).expect("parse PPS");
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert!(!pps.dependent_slice_segments_enabled_flag);
        assert!(!pps.output_flag_present_flag);
        assert_eq!(pps.num_extra_slice_header_bits, 0);
        // Fixture is encoded with --no-signhide.
        assert!(!pps.sign_data_hiding_enabled_flag);
        assert!(!pps.cabac_init_present_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        // init_qp_minus26 = 0 → init_qp = 26
        assert_eq!(pps.init_qp, 26);
        assert!(!pps.constrained_intra_pred_flag);
        assert!(!pps.transform_skip_enabled_flag);
        assert!(pps.cu_qp_delta_enabled_flag);
        assert_eq!(pps.diff_cu_qp_delta_depth, 0);
        assert_eq!(pps.pps_cb_qp_offset, 0);
        assert_eq!(pps.pps_cr_qp_offset, 0);
        assert!(!pps.weighted_pred_flag);
        assert!(!pps.weighted_bipred_flag);
        assert!(!pps.transquant_bypass_enabled_flag);
        assert!(!pps.tiles_enabled_flag);
        assert!(!pps.entropy_coding_sync_enabled_flag);
        assert!(pps.pps_loop_filter_across_slices_enabled_flag);
        // x265 with --no-deblock sets deblocking_filter_control_present and
        // pps_deblocking_filter_disabled.
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(!pps.deblocking_filter_override_enabled_flag);
        assert!(pps.pps_deblocking_filter_disabled_flag);
        assert!(!pps.lists_modification_present_flag);
        assert_eq!(pps.log2_parallel_merge_level_minus2, 0);
        assert!(!pps.slice_segment_header_extension_present_flag);
    }
}
