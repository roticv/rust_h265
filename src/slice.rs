//! HEVC slice segment header parsing (spec 7.3.6.1, semantics 7.4.7.1).
//!
//! Phase 2 scope: enough of the slice header to decode an IDR I-slice from
//! the Phase 1 fixture (`testdata/tiny_intra.h265`). Anything not exercised
//! by that fixture is gated as `Unsupported` so we never silently advance
//! the bitstream past data we can't interpret.
//!
//! Phase 3c-1 extends this to independent multi-slice pictures: non-first
//! slice segments are allowed, but dependent slice segments remain
//! `Unsupported` (deferred to Phase 3c-4).

use crate::bitstream::BitstreamReader;
use crate::error::DecodeError;
use crate::nal::NalUnitType;
use crate::pps::Pps;
use crate::sps::Sps;

/// HEVC slice type (spec table 7-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    B = 0,
    P = 1,
    I = 2,
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_slice_segment_in_pic_flag: bool,
    pub no_output_of_prior_pics_flag: bool,
    pub slice_pic_parameter_set_id: u32,
    pub dependent_slice_segment_flag: bool,
    pub slice_segment_address: u32,
    pub slice_type: SliceType,
    pub pic_output_flag: bool,
    /// Picture order count LSB. 0 for IDR pictures (not coded in the stream).
    pub slice_pic_order_cnt_lsb: u32,
    pub slice_sao_luma_flag: bool,
    pub slice_sao_chroma_flag: bool,
    pub slice_qp_delta: i32,
    /// Effective slice QP: `pps.init_qp + slice_qp_delta`. Spec eq. 7-54.
    pub slice_qp_y: i32,
    /// Slice-level deblocking disable flag. Inherits from PPS unless an
    /// override is signaled (Phase 3b-1 only supports the inherited path).
    pub slice_deblocking_filter_disabled_flag: bool,
    /// Slice-level β offset (per spec, in `2 *` units when applied).
    pub slice_beta_offset_div2: i32,
    /// Slice-level tc offset (per spec, in `2 *` units when applied).
    pub slice_tc_offset_div2: i32,
    /// Phase 3c-3 (WPP): cumulative byte offsets (relative to the start of
    /// the slice data, i.e. the first byte after the slice header's byte
    /// alignment) where each WPP substream / tile substream starts. The
    /// i-th entry is the byte offset of substream `i + 1`; substream 0
    /// always starts at offset 0.
    ///
    /// Empty for non-WPP / non-tiles slices. For WPP, the length equals
    /// `num_entry_point_offsets` which (for a single slice) equals
    /// `num_ctb_rows_in_slice - 1` when WPP is enabled.
    pub entry_point_offsets: Vec<u32>,
    /// Number of bits consumed by the slice header so we know where the
    /// slice data (CABAC bytestream) begins.
    pub header_size_bits: usize,
}

/// Parse the slice segment header for a single VCL NAL unit.
///
/// `nal_unit_type` is needed because IDR slices skip POC and RPS parsing.
/// `sps` and `pps` provide the gating fields (SAO, deblock, tiles, ...).
pub fn parse_slice_segment_header(
    rbsp: &[u8],
    nal_unit_type: NalUnitType,
    sps: &Sps,
    pps: &Pps,
) -> Result<SliceHeader, DecodeError> {
    let mut r = BitstreamReader::new(rbsp);

    let first_slice_segment_in_pic_flag = r.read_bit()? == 1;

    let mut no_output_of_prior_pics_flag = false;
    if nal_unit_type.is_irap() {
        no_output_of_prior_pics_flag = r.read_bit()? == 1;
    }

    let slice_pic_parameter_set_id = r.read_ue()?;
    if slice_pic_parameter_set_id != pps.pps_pic_parameter_set_id {
        return Err(DecodeError::InvalidSyntax(
            "slice_pic_parameter_set_id does not match active PPS",
        ));
    }

    // spec 7.3.6.1: dependent_slice_segment_flag and slice_segment_address
    // are only present when first_slice_segment_in_pic_flag == 0. Phase 3c-1
    // supports independent slice segments only — dependent slices are
    // Phase 3c-4.
    let mut dependent_slice_segment_flag = false;
    let mut slice_segment_address = 0u32;
    if !first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            dependent_slice_segment_flag = r.read_bit()? == 1;
            if dependent_slice_segment_flag {
                return Err(DecodeError::Unsupported(
                    "dependent slice segments not supported (Phase 3c-4)",
                ));
            }
        }
        // slice_segment_address: ceil(log2(NumCtbsInPic)) bits. Spec eq. 7-78.
        let num_ctbs_in_pic = sps.pic_width_in_ctbs_y() * sps.pic_height_in_ctbs_y();
        let slice_address_length = ceil_log2(num_ctbs_in_pic) as u8;
        slice_segment_address = if slice_address_length > 0 {
            r.read_bits(slice_address_length)?
        } else {
            0
        };
        if slice_segment_address >= num_ctbs_in_pic {
            return Err(DecodeError::InvalidSyntax(
                "slice_segment_address out of range",
            ));
        }
    } else if pps.dependent_slice_segments_enabled_flag {
        // PPS enables dependent slices but this is the first slice segment;
        // the flag is implicitly 0 for the first segment and no bit is coded.
        // We still reject with Unsupported to keep the surface tight — any
        // subsequent slice in this picture could be dependent, which we
        // cannot handle.
        return Err(DecodeError::Unsupported(
            "dependent_slice_segments_enabled_flag=1 not supported (Phase 3c-4)",
        ));
    }

    // Reserved slice header bits (none for our PPS).
    for _ in 0..pps.num_extra_slice_header_bits {
        let _ = r.read_bit()?;
    }

    let slice_type_raw = r.read_ue()?;
    let slice_type = match slice_type_raw {
        0 => SliceType::B,
        1 => SliceType::P,
        2 => SliceType::I,
        _ => {
            return Err(DecodeError::InvalidSyntax("invalid slice_type"));
        }
    };

    if slice_type != SliceType::I {
        // Phase 2 only handles I-slices. P/B slices need ref list + MV decode.
        return Err(DecodeError::Unsupported(
            "only I-slices are supported in Phase 2",
        ));
    }

    let mut pic_output_flag = true;
    if pps.output_flag_present_flag {
        pic_output_flag = r.read_bit()? == 1;
    }

    // separate_colour_plane_flag is gated by chroma_format_idc==3, which we
    // already rejected in SPS parsing.

    if !nal_unit_type.is_idr() {
        // POC + RPS section. IDR pictures skip this entirely (POC = 0).
        // We don't yet handle non-IDR slices, so reject before consuming any
        // bits — that way nothing here can mis-advance the bitstream.
        return Err(DecodeError::Unsupported(
            "non-IDR slice header parsing not yet implemented",
        ));
    }
    let slice_pic_order_cnt_lsb = 0u32;

    let mut slice_sao_luma_flag = false;
    let mut slice_sao_chroma_flag = false;
    if sps.sample_adaptive_offset_enabled_flag {
        slice_sao_luma_flag = r.read_bit()? == 1;
        // For 4:2:0 there's also a chroma SAO flag.
        slice_sao_chroma_flag = r.read_bit()? == 1;
    }

    // The big P/B section is skipped because slice_type == I above.

    let slice_qp_delta = r.read_se()?;
    let slice_qp_y = pps.init_qp + slice_qp_delta;
    if !(-(6 * sps.bit_depth_luma as i32 - 6)..=51).contains(&slice_qp_y) {
        return Err(DecodeError::InvalidSyntax("SliceQpY out of range"));
    }

    if pps.pps_slice_chroma_qp_offsets_present_flag {
        let _slice_cb_qp_offset = r.read_se()?;
        let _slice_cr_qp_offset = r.read_se()?;
    }

    let mut slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
    let mut slice_beta_offset_div2 = 0i32;
    let mut slice_tc_offset_div2 = 0i32;
    if pps.deblocking_filter_override_enabled_flag {
        let deblocking_filter_override_flag = r.read_bit()? == 1;
        if deblocking_filter_override_flag {
            slice_deblocking_filter_disabled_flag = r.read_bit()? == 1;
            if !slice_deblocking_filter_disabled_flag {
                slice_beta_offset_div2 = r.read_se()?;
                slice_tc_offset_div2 = r.read_se()?;
            }
        }
    }

    if pps.pps_loop_filter_across_slices_enabled_flag
        && (slice_sao_luma_flag || slice_sao_chroma_flag || !slice_deblocking_filter_disabled_flag)
    {
        let _slice_loop_filter_across_slices_enabled_flag = r.read_bit()?;
    }

    // Phase 3c-3: WPP entry point parsing. `tiles_enabled_flag` is still
    // rejected at PPS parse time, so this branch currently only fires for
    // `entropy_coding_sync_enabled_flag = 1`.
    let mut entry_point_offsets: Vec<u32> = Vec::new();
    if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
        let num_entry_point_offsets = r.read_ue()?;
        if num_entry_point_offsets > 0 {
            let offset_len_minus1 = r.read_ue()?;
            if offset_len_minus1 >= 32 {
                return Err(DecodeError::InvalidSyntax(
                    "offset_len_minus1 out of range [0, 31]",
                ));
            }
            let offset_len = (offset_len_minus1 + 1) as u8;
            entry_point_offsets.reserve(num_entry_point_offsets as usize);
            let mut cumulative: u32 = 0;
            for _ in 0..num_entry_point_offsets {
                let v = r.read_bits(offset_len)?;
                // Each entry is the substream byte length minus one; the
                // stored "entry point" is the cumulative byte offset of the
                // next substream from the start of the slice data. So
                // substream 0 starts at 0, substream 1 starts at
                // (entry_point_offset_minus1[0] + 1), substream 2 starts at
                // (entry_point_offset_minus1[0] + 1) +
                // (entry_point_offset_minus1[1] + 1), etc.
                cumulative = cumulative
                    .checked_add(v + 1)
                    .ok_or(DecodeError::InvalidSyntax("entry_point_offset overflow"))?;
                entry_point_offsets.push(cumulative);
            }
        }
    }

    if pps.slice_segment_header_extension_present_flag {
        return Err(DecodeError::Unsupported(
            "slice_segment_header_extension not supported",
        ));
    }

    // byte_alignment(): one '1' bit followed by zero bits up to a byte boundary.
    // Spec 7.3.2.11.
    let one_bit = r.read_bit()?;
    if one_bit != 1 {
        return Err(DecodeError::InvalidSyntax("byte_alignment: expected 1 bit"));
    }
    while !at_byte_boundary(&r) {
        let zero_bit = r.read_bit()?;
        if zero_bit != 0 {
            return Err(DecodeError::InvalidSyntax("byte_alignment: expected 0 bit"));
        }
    }

    let (byte_offset, bit_offset) = r.position();
    debug_assert_eq!(bit_offset, 0, "byte_alignment should be byte-aligned");
    let header_size_bits = byte_offset * 8;

    Ok(SliceHeader {
        first_slice_segment_in_pic_flag,
        no_output_of_prior_pics_flag,
        slice_pic_parameter_set_id,
        dependent_slice_segment_flag,
        slice_segment_address,
        slice_type,
        pic_output_flag,
        slice_pic_order_cnt_lsb,
        slice_sao_luma_flag,
        slice_sao_chroma_flag,
        slice_qp_delta,
        slice_qp_y,
        slice_deblocking_filter_disabled_flag,
        slice_beta_offset_div2,
        slice_tc_offset_div2,
        entry_point_offsets,
        header_size_bits,
    })
}

/// Helper for `byte_alignment()` since the bitstream reader does not expose
/// "is at byte boundary" directly.
fn at_byte_boundary(r: &BitstreamReader) -> bool {
    let (_, bit_offset) = r.position();
    bit_offset == 0
}

/// `ceil(log2(n))` per spec convention (spec 5.2). Returns 0 for n <= 1.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built minimal SPS/PPS that matches the Phase 1 fixture's gating
    /// flags, used to keep these tests independent of the parameter set
    /// parsers (which we already test elsewhere).
    fn minimal_pps() -> Pps {
        Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp: 26,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
            cu_qp_delta_enabled_flag: true,
            diff_cu_qp_delta_depth: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_slice_chroma_qp_offsets_present_flag: false,
            weighted_pred_flag: false,
            weighted_bipred_flag: false,
            transquant_bypass_enabled_flag: false,
            tiles_enabled_flag: false,
            entropy_coding_sync_enabled_flag: false,
            pps_loop_filter_across_slices_enabled_flag: true,
            deblocking_filter_control_present_flag: true,
            deblocking_filter_override_enabled_flag: false,
            pps_deblocking_filter_disabled_flag: true,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,
            pps_scaling_list_data_present_flag: false,
            scaling_list: None,
            lists_modification_present_flag: false,
            log2_parallel_merge_level_minus2: 0,
            slice_segment_header_extension_present_flag: false,
        }
    }

    fn minimal_sps() -> Sps {
        use crate::profile_tier_level::ProfileTierLevel;
        Sps {
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: true,
            profile_tier_level: ProfileTierLevel {
                general_profile_space: 0,
                general_tier_flag: false,
                general_profile_idc: 3,
                general_level_idc: 30,
            },
            sps_seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            pic_width_in_luma_samples: 16,
            pic_height_in_luma_samples: 16,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_max_pic_order_cnt_lsb: 8,
            min_cb_log2_size_y: 3,
            ctb_log2_size_y: 4,
            ctb_size_y: 16,
            min_tb_log2_size_y: 2,
            max_tb_log2_size_y: 4,
            max_transform_hierarchy_depth_inter: 0,
            max_transform_hierarchy_depth_intra: 0,
            scaling_list_enabled_flag: false,
            scaling_list: None,
            amp_enabled_flag: false,
            sample_adaptive_offset_enabled_flag: false,
            pcm_enabled_flag: false,
            pcm_sample_bit_depth_luma: 8,
            pcm_sample_bit_depth_chroma: 8,
            log2_min_pcm_cb_size: 8,
            log2_max_pcm_cb_size: 0,
            pcm_loop_filter_disabled_flag: false,
            num_short_term_ref_pic_sets: 0,
            long_term_ref_pics_present_flag: false,
            sps_temporal_mvp_enabled_flag: true,
            strong_intra_smoothing_enabled_flag: false,
        }
    }

    #[test]
    fn test_parse_tiny_intra_slice_header() {
        // Slice RBSP from testdata/tiny_intra.h265 — 0xAD 0xC0 is the full
        // slice header (16 bits), the rest is the CABAC byte stream.
        let rbsp = [0xAD, 0xC0, 0xCE, 0x1F, 0xBF, 0x0B, 0x80];
        let sh =
            parse_slice_segment_header(&rbsp, NalUnitType::IdrNLp, &minimal_sps(), &minimal_pps())
                .expect("parse slice header");

        assert!(sh.first_slice_segment_in_pic_flag);
        assert!(!sh.no_output_of_prior_pics_flag);
        assert_eq!(sh.slice_pic_parameter_set_id, 0);
        assert_eq!(sh.slice_type, SliceType::I);
        assert_eq!(sh.slice_qp_delta, -1);
        assert_eq!(sh.slice_qp_y, 25); // init_qp 26 + (-1)
        assert!(!sh.slice_sao_luma_flag);
        assert!(!sh.slice_sao_chroma_flag);

        // Header is exactly 16 bits → CABAC stream begins at RBSP byte 2.
        assert_eq!(sh.header_size_bits, 16);
    }
}
