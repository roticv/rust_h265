//! HEVC Picture Parameter Set parsing (spec 7.3.2.3, semantics 7.4.3.3).
//!
//! For Phase 1 we extract the PPS fields needed by future slice-header parsing
//! and reject the more complex features (tiles, WPP, scaling lists, extensions)
//! with a clear `Unsupported` error.

use crate::bitstream::BitstreamReader;
use crate::error::DecodeError;
use crate::scaling_list::{ScalingList, parse_scaling_list_data};

#[derive(Debug, Clone)]
pub struct Pps {
    pub pps_pic_parameter_set_id: u32,
    pub pps_seq_parameter_set_id: u32,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    /// `init_qp = init_qp_minus26 + 26` (spec 7.4.3.3.1).
    pub init_qp: i32,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_beta_offset_div2: i32,
    pub pps_tc_offset_div2: i32,
    pub pps_scaling_list_data_present_flag: bool,
    /// PPS-level scaling list override. When present, takes priority over SPS scaling list.
    pub scaling_list: Option<ScalingList>,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level_minus2: u32,
    pub slice_segment_header_extension_present_flag: bool,
}

pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, DecodeError> {
    let mut r = BitstreamReader::new(rbsp);

    let pps_pic_parameter_set_id = r.read_ue()?;
    let pps_seq_parameter_set_id = r.read_ue()?;
    let dependent_slice_segments_enabled_flag = r.read_bit()? == 1;
    let output_flag_present_flag = r.read_bit()? == 1;
    let num_extra_slice_header_bits = r.read_bits(3)? as u8;
    let sign_data_hiding_enabled_flag = r.read_bit()? == 1;
    let cabac_init_present_flag = r.read_bit()? == 1;
    let num_ref_idx_l0_default_active_minus1 = r.read_ue()?;
    let num_ref_idx_l1_default_active_minus1 = r.read_ue()?;
    let init_qp_minus26 = r.read_se()?;
    let init_qp = init_qp_minus26 + 26;
    let constrained_intra_pred_flag = r.read_bit()? == 1;
    let transform_skip_enabled_flag = r.read_bit()? == 1;

    let cu_qp_delta_enabled_flag = r.read_bit()? == 1;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
        r.read_ue()?
    } else {
        0
    };
    let pps_cb_qp_offset = r.read_se()?;
    let pps_cr_qp_offset = r.read_se()?;
    let pps_slice_chroma_qp_offsets_present_flag = r.read_bit()? == 1;
    let weighted_pred_flag = r.read_bit()? == 1;
    let weighted_bipred_flag = r.read_bit()? == 1;
    let transquant_bypass_enabled_flag = r.read_bit()? == 1;

    let tiles_enabled_flag = r.read_bit()? == 1;
    let entropy_coding_sync_enabled_flag = r.read_bit()? == 1;
    if tiles_enabled_flag {
        return Err(DecodeError::Unsupported("tiles_enabled_flag not supported"));
    }
    // `entropy_coding_sync_enabled_flag = 1` (WPP) is accepted and wired up in
    // `Decoder::decode_slice` as of Phase 3c-3. Sequential decode only — we
    // still decode rows one after another, but with the spec's per-row CABAC
    // reinit + state propagation.

    let pps_loop_filter_across_slices_enabled_flag = r.read_bit()? == 1;

    let deblocking_filter_control_present_flag = r.read_bit()? == 1;
    let mut deblocking_filter_override_enabled_flag = false;
    let mut pps_deblocking_filter_disabled_flag = false;
    let mut pps_beta_offset_div2 = 0;
    let mut pps_tc_offset_div2 = 0;
    if deblocking_filter_control_present_flag {
        deblocking_filter_override_enabled_flag = r.read_bit()? == 1;
        pps_deblocking_filter_disabled_flag = r.read_bit()? == 1;
        if !pps_deblocking_filter_disabled_flag {
            pps_beta_offset_div2 = r.read_se()?;
            pps_tc_offset_div2 = r.read_se()?;
        }
    }

    let pps_scaling_list_data_present_flag = r.read_bit()? == 1;
    let pps_scaling_list = if pps_scaling_list_data_present_flag {
        let mut sl = ScalingList::default_scaling_list();
        parse_scaling_list_data(&mut r, &mut sl)?;
        Some(sl)
    } else {
        None
    };

    let lists_modification_present_flag = r.read_bit()? == 1;
    let log2_parallel_merge_level_minus2 = r.read_ue()?;
    let slice_segment_header_extension_present_flag = r.read_bit()? == 1;

    // pps_extension_present_flag and any extensions follow — Phase 1 stops
    // here. The next NAL start code bounds the PPS.
    let _pps_extension_present_flag = r.read_bit()?;

    Ok(Pps {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        dependent_slice_segments_enabled_flag,
        output_flag_present_flag,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled_flag,
        cabac_init_present_flag,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        init_qp,
        constrained_intra_pred_flag,
        transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag,
        weighted_pred_flag,
        weighted_bipred_flag,
        transquant_bypass_enabled_flag,
        tiles_enabled_flag,
        entropy_coding_sync_enabled_flag,
        pps_loop_filter_across_slices_enabled_flag,
        deblocking_filter_control_present_flag,
        deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        pps_beta_offset_div2,
        pps_tc_offset_div2,
        pps_scaling_list_data_present_flag,
        scaling_list: pps_scaling_list,
        lists_modification_present_flag,
        log2_parallel_merge_level_minus2,
        slice_segment_header_extension_present_flag,
    })
}
