//! Support for the fuzz targets in `fuzz/`. Not part of the public API: it
//! may change in any release.

use crate::nal::NalUnit;
use crate::pps::{Pps, parse_pps};
use crate::slice::parse_slice_segment_header;
use crate::sps::{Sps, parse_sps};

/// For each NAL unit, the RBSP byte offset at which its `slice_segment_data()`
/// starts, or `None` for non-VCL NAL units and slices whose header does not
/// parse. Parameter sets are tracked in stream order, as the decoder does.
///
/// The `slice_data` fuzz target mutates only the bytes from that offset on,
/// so every input gets past the parameter sets and slice headers and reaches
/// the CABAC syntax decoders.
pub fn slice_data_offsets(nals: &[NalUnit<'_>]) -> Vec<Option<usize>> {
    let mut sps_table: Vec<Option<Sps>> = vec![None; 16];
    let mut pps_table: Vec<Option<Pps>> = vec![None; 64];
    nals.iter()
        .map(|nal| {
            let nut = nal.nal_unit_type;
            if nut == crate::NalUnitType::Sps {
                if let Ok(sps) = parse_sps(&nal.rbsp)
                    && let Some(slot) = sps_table.get_mut(sps.sps_seq_parameter_set_id as usize)
                {
                    *slot = Some(sps);
                }
                return None;
            }
            if nut == crate::NalUnitType::Pps {
                if let Ok(pps) = parse_pps(&nal.rbsp)
                    && let Some(slot) = pps_table.get_mut(pps.pps_pic_parameter_set_id as usize)
                {
                    *slot = Some(pps);
                }
                return None;
            }
            if !nut.is_vcl() || nal.nuh_layer_id != 0 {
                return None;
            }
            // Slice header prefix: first_slice_segment_in_pic_flag,
            // [no_output_of_prior_pics_flag], slice_pic_parameter_set_id.
            let mut r = crate::bitstream::BitstreamReader::new(&nal.rbsp);
            r.read_bit().ok()?;
            if nut.is_irap() {
                r.read_bit().ok()?;
            }
            let pps_id = r.read_ue().ok()?;
            let pps = pps_table.get_mut(pps_id as usize)?.as_mut()?;
            let sps = sps_table
                .get(pps.pps_seq_parameter_set_id as usize)?
                .as_ref()?;
            if pps.column_widths_in_ctbs.is_empty() {
                pps.resolve_tile_geometry(sps).ok()?;
            }
            let sh = parse_slice_segment_header(&nal.rbsp, nut, sps, pps).ok()?;
            Some(sh.header_size_bits / 8)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    /// Every slice of a valid stream gets an offset, inside its NAL, so the
    /// `slice_data` fuzz target has bytes to corrupt in every fixture.
    #[test]
    fn slice_data_offsets_cover_every_slice() {
        for name in [
            "inter_b.h265",
            "dep_slices.h265",
            "tiles_hm_lf_across_832x256.h265",
            "ctu64_wpp.h265",
            "seek_to_cra.h265",
            "rext_lossless_rot_rdpcm_hm_444_96x64.h265",
        ] {
            let path = format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"));
            let data = std::fs::read(path).expect("fixture");
            let nals = crate::parse_annex_b(&data);
            let offsets = super::slice_data_offsets(&nals);
            for (nal, offset) in nals.iter().zip(&offsets) {
                if nal.nal_unit_type.is_vcl() {
                    let offset = offset.unwrap_or_else(|| panic!("{name}: slice header"));
                    assert!(offset > 0 && offset < nal.rbsp.len(), "{name}: {offset}");
                } else {
                    assert_eq!(*offset, None);
                }
            }
        }
    }
}
