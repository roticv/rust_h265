//! HEVC intra prediction (spec 8.4.4).
//!
//! Phase 2c-5 scope: reference sample derivation/substitution + PLANAR (mode
//! 0) + DC (mode 1) prediction. Mirrors FFmpeg `pred_template.c`'s
//! `intra_pred`, `pred_planar`, and `pred_dc` for the simplified case our
//! fixture exercises (no neighbors → reference samples filled with the
//! default value, no constrained_intra_pred). Angular modes (2..34) and
//! reference sample smoothing for non-trivial neighbor configurations are
//! deferred until later phases need them.
//!
//! Layout convention: `top` and `left` are slices of length `2 * size + 1`,
//! where index 0 corresponds to the corner sample `p[-1][-1]` (top-left) and
//! indices 1..=size cover the immediate neighbors. The "extended" right and
//! bottom samples occupy indices size+1..=2*size.

/// Default reference sample value when nothing is available
/// (`1 << (BitDepth - 1)` per HEVC spec 8.4.4.2.2).
pub fn default_ref_sample(bit_depth: u8) -> u8 {
    1u8 << (bit_depth - 1)
}

/// Reference sample availability flags for a TU. For our Phase 2c-5 fixture
/// (single CTU, no neighbors), all of these are `false`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReferenceAvailability {
    pub up_left: bool,
    pub up: bool,
    pub up_right: bool,
    pub left: bool,
    pub bottom_left: bool,
}

/// Build the `top` and `left` reference sample arrays for a TU at `(x0, y0)`
/// of size `size = 1 << log2_size`. Returns `(top, left)` each of length
/// `2 * size + 1`, where index 0 is the corner.
///
/// For our Phase 2c-5 fixture all neighbors are unavailable, so the
/// substitution rule fills both arrays with `default_ref_sample(bit_depth)`.
/// More elaborate substitution (partial neighbor availability, constrained
/// intra) is gated for later phases.
pub fn build_reference_samples(
    avail: ReferenceAvailability,
    log2_size: u8,
    bit_depth: u8,
) -> (Vec<u8>, Vec<u8>) {
    let size = 1usize << log2_size;
    let len = 2 * size + 1;

    if !avail.up_left
        && !avail.up
        && !avail.up_right
        && !avail.left
        && !avail.bottom_left
    {
        let fill = default_ref_sample(bit_depth);
        return (vec![fill; len], vec![fill; len]);
    }

    // Partial-availability substitution is per spec 8.4.4.2.2 / FFmpeg's
    // `intra_pred` body — not yet needed for our fixture, so be loud about it.
    panic!(
        "build_reference_samples: partial reference availability not yet implemented \
         (avail = {:?})",
        avail
    );
}

/// PLANAR intra prediction (spec 8.4.4.2.5 / FFmpeg `pred_planar`).
///
/// `top` is indexed `[0..=2*size]` with `top[0] = p[-1][-1]` (corner) and
/// `top[i+1] = p[i][-1]` for `i = 0..2*size-1`. Same for `left` along the
/// vertical axis.
///
/// `dst` is the destination buffer of `size * size` samples in row-major
/// order with row stride `dst_stride`.
pub fn predict_planar(
    dst: &mut [u8],
    dst_stride: usize,
    top: &[u8],
    left: &[u8],
    log2_size: u8,
) {
    let size = 1usize << log2_size;
    // Spec eq 8-26:
    //   predSamples[x][y] = ((nT - 1 - x) * p[-1][y] + (x+1) * p[nT][-1]
    //                      + (nT - 1 - y) * p[x][-1] + (y+1) * p[-1][nT] + nT)
    //                      >> (Log2(nT) + 1)
    //
    // The reference layout offsets indices by 1 (corner at index 0), so:
    //   p[-1][y]    = left[y + 1]
    //   p[x][-1]    = top[x + 1]
    //   p[nT][-1]   = top[nT + 1] is INCORRECT; FFmpeg uses top[size]
    // Hmm — looking again at FFmpeg's pred_planar:
    //   top[0..size-1] = p[0..size-1][-1]
    //   top[size]     = p[size-1+1][-1] = p[nT][-1] ← FFmpeg uses index `size`
    //   left[0..size-1] = p[-1][0..size-1]
    //   left[size]    = p[-1][nT]
    // FFmpeg's `top` and `left` pointers are indexed starting from 0 (not -1).
    // We adopt the same convention here, but the underlying buffer has the
    // corner at byte offset 0, so the "FFmpeg `top` pointer" lives at
    // `&buffer[1..]`. For clarity we expose helpers that index that way.
    //
    // To match FFmpeg's pred_planar exactly, we treat the slices `top_p` and
    // `left_p` as starting at index 0 = first non-corner sample.
    let top_p = &top[1..]; // top_p[i] = p[i][-1] for i=0..2*size-1
    let left_p = &left[1..]; // left_p[i] = p[-1][i]

    let shift = log2_size as u32 + 1;
    for y in 0..size {
        for x in 0..size {
            let pred = (size - 1 - x) as i32 * left_p[y] as i32
                + (x + 1) as i32 * top_p[size] as i32
                + (size - 1 - y) as i32 * top_p[x] as i32
                + (y + 1) as i32 * left_p[size] as i32
                + size as i32;
            dst[y * dst_stride + x] = (pred >> shift) as u8;
        }
    }
}

/// DC intra prediction (spec 8.4.4.2.4 / FFmpeg `pred_dc`).
///
/// `apply_luma_filter` is true for luma TUs with `size < 32`, in which case
/// the top row, left column, and top-left corner get a simple smoothing
/// filter applied (spec eq. 8-23).
pub fn predict_dc(
    dst: &mut [u8],
    dst_stride: usize,
    top: &[u8],
    left: &[u8],
    log2_size: u8,
    apply_luma_filter: bool,
) {
    let size = 1usize << log2_size;
    let top_p = &top[1..];
    let left_p = &left[1..];
    let mut dc_sum: i32 = size as i32;
    for i in 0..size {
        dc_sum += left_p[i] as i32 + top_p[i] as i32;
    }
    let dc = (dc_sum >> (log2_size as u32 + 1)) as u8;

    for y in 0..size {
        for x in 0..size {
            dst[y * dst_stride + x] = dc;
        }
    }

    if apply_luma_filter && size < 32 {
        // Top-left corner: (left[0] + 2*dc + top[0] + 2) >> 2
        dst[0] = (((left_p[0] as i32) + 2 * (dc as i32) + (top_p[0] as i32) + 2) >> 2) as u8;
        // Top row x = 1..size: (top[x] + 3*dc + 2) >> 2
        for x in 1..size {
            dst[x] = (((top_p[x] as i32) + 3 * (dc as i32) + 2) >> 2) as u8;
        }
        // Left column y = 1..size: (left[y] + 3*dc + 2) >> 2
        for y in 1..size {
            dst[y * dst_stride] = (((left_p[y] as i32) + 3 * (dc as i32) + 2) >> 2) as u8;
        }
    }
}

/// Add a residual block to a prediction in place, clipping to `[0, 255]`.
/// `residual` and `dst` are the same shape (`size * size`); `dst_stride` is
/// the row stride of `dst`. Used by callers to combine the intra prediction
/// with the inverse-transformed residual.
pub fn add_residual(dst: &mut [u8], dst_stride: usize, residual: &[i16], log2_size: u8) {
    let size = 1usize << log2_size;
    for y in 0..size {
        for x in 0..size {
            let pixel = dst[y * dst_stride + x] as i32 + residual[y * size + x] as i32;
            dst[y * dst_stride + x] = pixel.clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no available neighbors, reference samples must be filled with
    /// the bit-depth midpoint (128 for 8-bit).
    #[test]
    fn test_no_neighbors_fills_with_midpoint() {
        let avail = ReferenceAvailability::default();
        let (top, left) = build_reference_samples(avail, 4, 8);
        assert_eq!(top.len(), 33); // 2 * 16 + 1
        assert_eq!(left.len(), 33);
        assert!(top.iter().all(|&p| p == 128));
        assert!(left.iter().all(|&p| p == 128));
    }

    /// PLANAR with all-128 reference samples must produce all-128 prediction.
    #[test]
    fn test_planar_uniform_neighbors() {
        let top = vec![128u8; 33];
        let left = vec![128u8; 33];
        let mut dst = vec![0u8; 256];
        predict_planar(&mut dst, 16, &top, &left, 4);
        assert!(dst.iter().all(|&p| p == 128), "first row: {:?}", &dst[..16]);
    }

    /// DC with all-128 reference samples must produce all-128 prediction —
    /// the corner-and-edge filter on a uniform input is also a no-op.
    #[test]
    fn test_dc_uniform_neighbors_with_filter() {
        let top = vec![128u8; 33];
        let left = vec![128u8; 33];
        let mut dst = vec![0u8; 256];
        predict_dc(&mut dst, 16, &top, &left, 4, true);
        assert!(dst.iter().all(|&p| p == 128));
    }

    /// `add_residual` clamps to [0, 255] and adds element-by-element.
    #[test]
    fn test_add_residual_basic() {
        let mut dst = vec![128u8; 16];
        let mut residual = vec![0i16; 16];
        residual[0] = 10;
        residual[1] = -200;
        residual[2] = 200;
        add_residual(&mut dst, 4, &residual, 2);
        assert_eq!(dst[0], 138);
        assert_eq!(dst[1], 0); // clamped from -72
        assert_eq!(dst[2], 255); // clamped from 328
        assert_eq!(dst[3], 128);
    }

    /// **End-to-end check for the fixture's luma block**: PLANAR with no
    /// neighbors gives all-128, plus the IDCT residual of all -2, equals
    /// all 0x7E (the reference YUV).
    #[test]
    fn test_fixture_luma_reconstruction() {
        let avail = ReferenceAvailability::default();
        let (top, left) = build_reference_samples(avail, 4, 8);
        let mut block = vec![0u8; 256];
        predict_planar(&mut block, 16, &top, &left, 4);
        let residual = vec![-2i16; 256];
        add_residual(&mut block, 16, &residual, 4);
        assert!(
            block.iter().all(|&p| p == 0x7E),
            "expected all 0x7E (= 126), got first row: {:?}",
            &block[..16]
        );
    }

    /// **End-to-end check for the fixture's chroma blocks**: PLANAR 8×8 with
    /// no neighbors gives all-128, no chroma residual → all 0x80.
    #[test]
    fn test_fixture_chroma_reconstruction() {
        let avail = ReferenceAvailability::default();
        let (top, left) = build_reference_samples(avail, 3, 8);
        let mut block = vec![0u8; 64];
        predict_planar(&mut block, 8, &top, &left, 3);
        // No chroma residual in our fixture (cbf_cb = cbf_cr = 0).
        assert!(
            block.iter().all(|&p| p == 0x80),
            "expected all 0x80 (= 128), got first row: {:?}",
            &block[..8]
        );
    }
}
