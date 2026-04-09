//! HEVC Sequence Parameter Set parsing (spec 7.3.2.2, semantics 7.4.3.2).
//!
//! For Phase 1 we extract the fields needed by the rest of the decoder:
//! picture dimensions, CTU/CU/TU size derivation, bit depth, POC LSB width,
//! and the in-loop filter / RPS / temporal MVP enable flags. Anything more
//! exotic (scaling lists, ST-RPS entries, VUI parameters, extensions) is
//! either rejected or skipped.

use crate::bitstream::BitstreamReader;
use crate::error::DecodeError;
use crate::profile_tier_level::{ProfileTierLevel, parse_profile_tier_level};
use crate::scaling_list::{ScalingList, parse_scaling_list_data};

#[derive(Debug, Clone)]
pub struct Sps {
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: bool,
    pub profile_tier_level: ProfileTierLevel,
    pub sps_seq_parameter_set_id: u32,
    pub chroma_format_idc: u32,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub log2_max_pic_order_cnt_lsb: u8,

    // CU/CTU derivation per spec 7.4.3.2.1
    pub min_cb_log2_size_y: u8,
    pub ctb_log2_size_y: u8,
    pub ctb_size_y: u32,
    pub min_tb_log2_size_y: u8,
    pub max_tb_log2_size_y: u8,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,

    pub scaling_list_enabled_flag: bool,
    /// The active scaling list. Present when `scaling_list_enabled_flag` is true.
    /// Contains default values when `sps_scaling_list_data_present_flag` is false,
    /// or explicitly parsed values when true.
    pub scaling_list: Option<ScalingList>,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    /// Bit depth of luma PCM samples (`pcm_sample_bit_depth_luma_minus1 + 1`).
    /// Only meaningful when `pcm_enabled_flag` is true; defaults to
    /// `bit_depth_luma` otherwise.
    pub pcm_sample_bit_depth_luma: u8,
    /// Bit depth of chroma PCM samples.
    pub pcm_sample_bit_depth_chroma: u8,
    /// `Log2MinIpcmCbSizeY` (spec eq. 7-35).
    pub log2_min_pcm_cb_size: u8,
    /// `Log2MaxIpcmCbSizeY` = `log2_min_pcm_cb_size + log2_diff_max_min_pcm_luma_coding_block_size`.
    pub log2_max_pcm_cb_size: u8,
    /// When set, deblocking is disabled across the boundaries of PCM blocks
    /// in this SPS. We don't have deblocking yet so we just store this.
    pub pcm_loop_filter_disabled_flag: bool,
    pub num_short_term_ref_pic_sets: u32,
    pub long_term_ref_pics_present_flag: bool,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
}

impl Sps {
    /// `PicWidthInCtbsY` (spec eq. 7-15).
    pub fn pic_width_in_ctbs_y(&self) -> u32 {
        self.pic_width_in_luma_samples.div_ceil(self.ctb_size_y)
    }

    /// `PicHeightInCtbsY` (spec eq. 7-17).
    pub fn pic_height_in_ctbs_y(&self) -> u32 {
        self.pic_height_in_luma_samples.div_ceil(self.ctb_size_y)
    }
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, DecodeError> {
    let mut r = BitstreamReader::new(rbsp);

    let sps_video_parameter_set_id = r.read_bits(4)? as u8;
    let sps_max_sub_layers_minus1 = r.read_bits(3)? as u8;
    if sps_max_sub_layers_minus1 > 6 {
        return Err(DecodeError::InvalidSyntax(
            "sps_max_sub_layers_minus1 out of range",
        ));
    }
    let sps_temporal_id_nesting_flag = r.read_bit()? == 1;
    let ptl = parse_profile_tier_level(&mut r, sps_max_sub_layers_minus1)?;

    let sps_seq_parameter_set_id = r.read_ue()?;
    let chroma_format_idc = r.read_ue()?;
    if chroma_format_idc != 1 {
        return Err(DecodeError::Unsupported(
            "only 4:2:0 (chroma_format_idc=1) supported",
        ));
    }
    // chroma_format_idc == 3 would have a separate_colour_plane_flag here.

    let pic_width_in_luma_samples = r.read_ue()?;
    let pic_height_in_luma_samples = r.read_ue()?;

    let conformance_window_flag = r.read_bit()? == 1;
    if conformance_window_flag {
        let _conf_win_left_offset = r.read_ue()?;
        let _conf_win_right_offset = r.read_ue()?;
        let _conf_win_top_offset = r.read_ue()?;
        let _conf_win_bottom_offset = r.read_ue()?;
    }

    let bit_depth_luma_minus8 = r.read_ue()?;
    let bit_depth_chroma_minus8 = r.read_ue()?;
    if bit_depth_luma_minus8 != 0 || bit_depth_chroma_minus8 != 0 {
        return Err(DecodeError::Unsupported("only 8-bit supported"));
    }
    let bit_depth_luma = 8u8;
    let bit_depth_chroma = 8u8;

    let log2_max_pic_order_cnt_lsb_minus4 = r.read_ue()?;
    if log2_max_pic_order_cnt_lsb_minus4 > 12 {
        return Err(DecodeError::InvalidSyntax(
            "log2_max_pic_order_cnt_lsb_minus4 out of range",
        ));
    }
    let log2_max_pic_order_cnt_lsb = (log2_max_pic_order_cnt_lsb_minus4 + 4) as u8;

    let sps_sub_layer_ordering_info_present_flag = r.read_bit()? == 1;
    let i_start = if sps_sub_layer_ordering_info_present_flag {
        0
    } else {
        sps_max_sub_layers_minus1 as usize
    };
    for _ in i_start..=sps_max_sub_layers_minus1 as usize {
        let _max_dec_pic_buffering_minus1 = r.read_ue()?;
        let _max_num_reorder_pics = r.read_ue()?;
        let _max_latency_increase_plus1 = r.read_ue()?;
    }

    let log2_min_luma_coding_block_size_minus3 = r.read_ue()?;
    let log2_diff_max_min_luma_coding_block_size = r.read_ue()?;
    let min_cb_log2_size_y = (log2_min_luma_coding_block_size_minus3 + 3) as u8;
    let ctb_log2_size_y = min_cb_log2_size_y + log2_diff_max_min_luma_coding_block_size as u8;
    if !(4..=6).contains(&ctb_log2_size_y) {
        // Spec restricts CtbLog2SizeY to 4..=6 (CTU size 16/32/64).
        return Err(DecodeError::InvalidSyntax("CtbLog2SizeY out of range"));
    }
    let ctb_size_y: u32 = 1 << ctb_log2_size_y;

    let log2_min_luma_transform_block_size_minus2 = r.read_ue()?;
    let log2_diff_max_min_luma_transform_block_size = r.read_ue()?;
    let min_tb_log2_size_y = (log2_min_luma_transform_block_size_minus2 + 2) as u8;
    let max_tb_log2_size_y = min_tb_log2_size_y + log2_diff_max_min_luma_transform_block_size as u8;

    let max_transform_hierarchy_depth_inter = r.read_ue()?;
    let max_transform_hierarchy_depth_intra = r.read_ue()?;

    let scaling_list_enabled_flag = r.read_bit()? == 1;
    let scaling_list = if scaling_list_enabled_flag {
        let mut sl = ScalingList::default_scaling_list();
        let sps_scaling_list_data_present_flag = r.read_bit()? == 1;
        if sps_scaling_list_data_present_flag {
            parse_scaling_list_data(&mut r, &mut sl)?;
        }
        Some(sl)
    } else {
        None
    };

    let amp_enabled_flag = r.read_bit()? == 1;
    let sample_adaptive_offset_enabled_flag = r.read_bit()? == 1;

    let pcm_enabled_flag = r.read_bit()? == 1;
    let (
        pcm_sample_bit_depth_luma,
        pcm_sample_bit_depth_chroma,
        log2_min_pcm_cb_size,
        log2_max_pcm_cb_size,
        pcm_loop_filter_disabled_flag,
    ) = if pcm_enabled_flag {
        // Spec 7.3.2.2 + 7.4.3.2.1.
        let pcm_bd_luma = (r.read_bits(4)? + 1) as u8;
        let pcm_bd_chroma = (r.read_bits(4)? + 1) as u8;
        if pcm_bd_luma > bit_depth_luma || pcm_bd_chroma > bit_depth_chroma {
            return Err(DecodeError::InvalidSyntax(
                "pcm_sample_bit_depth exceeds bit depth",
            ));
        }
        let log2_min_pcm_cb_size = (r.read_ue()? + 3) as u8;
        let log2_diff_max_min_pcm = r.read_ue()? as u8;
        let log2_max_pcm_cb_size = log2_min_pcm_cb_size + log2_diff_max_min_pcm;
        let pcm_loop_filter_disabled = r.read_bit()? == 1;
        (
            pcm_bd_luma,
            pcm_bd_chroma,
            log2_min_pcm_cb_size,
            log2_max_pcm_cb_size,
            pcm_loop_filter_disabled,
        )
    } else {
        // Defaults when PCM is disabled. `log2_min_pcm_cb_size > log2_max_pcm_cb_size`
        // ensures the "in range" test in `decode_coding_unit` never fires.
        (bit_depth_luma, bit_depth_chroma, 8u8, 0u8, false)
    };

    let num_short_term_ref_pic_sets = r.read_ue()?;
    if num_short_term_ref_pic_sets > 0 {
        // st_ref_pic_set() parsing is required for Phase 2; for Phase 1 we
        // gate it. The Phase 1 fixture (single intra frame) has zero ST RPSs.
        return Err(DecodeError::Unsupported(
            "st_ref_pic_set parsing not yet implemented",
        ));
    }

    let long_term_ref_pics_present_flag = r.read_bit()? == 1;
    if long_term_ref_pics_present_flag {
        return Err(DecodeError::Unsupported(
            "long-term reference pictures not yet supported",
        ));
    }

    let sps_temporal_mvp_enabled_flag = r.read_bit()? == 1;
    let strong_intra_smoothing_enabled_flag = r.read_bit()? == 1;

    // VUI and SPS extensions follow but we deliberately stop here — we don't
    // need them for Phase 1, and parsing them in full is a separate task.

    Ok(Sps {
        sps_video_parameter_set_id,
        sps_max_sub_layers_minus1,
        sps_temporal_id_nesting_flag,
        profile_tier_level: ptl,
        sps_seq_parameter_set_id,
        chroma_format_idc,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_pic_order_cnt_lsb,
        min_cb_log2_size_y,
        ctb_log2_size_y,
        ctb_size_y,
        min_tb_log2_size_y,
        max_tb_log2_size_y,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled_flag,
        scaling_list,
        amp_enabled_flag,
        sample_adaptive_offset_enabled_flag,
        pcm_enabled_flag,
        pcm_sample_bit_depth_luma,
        pcm_sample_bit_depth_chroma,
        log2_min_pcm_cb_size,
        log2_max_pcm_cb_size,
        pcm_loop_filter_disabled_flag,
        num_short_term_ref_pic_sets,
        long_term_ref_pics_present_flag,
        sps_temporal_mvp_enabled_flag,
        strong_intra_smoothing_enabled_flag,
    })
}
