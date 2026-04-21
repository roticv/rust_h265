//! Generic pixel type abstraction for multi-bit-depth HEVC decoding.
//!
//! The `Pixel` trait abstracts over `u8` (8-bit) and `u16` (10/12-bit) sample
//! storage. Functions that operate on pixel planes are generic over `P: Pixel`,
//! avoiding the need for runtime enum dispatch or always-u16 memory overhead.
//!
//! Bit depth (8, 10, 12) is a *runtime* parameter passed alongside `P` — the
//! trait only handles the *storage type*. A `u16` pixel could represent 10-bit
//! or 12-bit data depending on `bit_depth`.

/// Trait for pixel sample types. Implemented for `u8` and `u16`.
pub trait Pixel: Copy + Clone + Default + Send + Sync + Sized + 'static {
    /// Convert from a signed 32-bit computation result, clamping to the
    /// valid range `[0, max_val]` where `max_val = (1 << bit_depth) - 1`.
    fn from_i32_clamped(val: i32, bit_depth: u8) -> Self;

    /// Widen to i32 for arithmetic.
    fn to_i32(self) -> i32;

    /// Create a zero value.
    fn zero() -> Self;
}

impl Pixel for u8 {
    #[inline(always)]
    fn from_i32_clamped(val: i32, _bit_depth: u8) -> Self {
        val.clamp(0, 255) as u8
    }

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }
}

impl Pixel for u16 {
    #[inline(always)]
    fn from_i32_clamped(val: i32, bit_depth: u8) -> Self {
        let max = (1i32 << bit_depth) - 1;
        val.clamp(0, max) as u16
    }

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }

    #[inline(always)]
    fn zero() -> Self {
        0
    }
}

/// Maximum pixel value for a given bit depth: `(1 << bit_depth) - 1`.
#[inline(always)]
pub fn max_pixel_val(bit_depth: u8) -> i32 {
    (1i32 << bit_depth) - 1
}

/// Default (mid-grey) reference sample for intra prediction: `1 << (bit_depth - 1)`.
#[inline(always)]
pub fn default_ref_sample(bit_depth: u8) -> i32 {
    1i32 << (bit_depth - 1)
}

/// MC luma filter shift: `14 - bit_depth` for HEVC (spec 8.5.3.2.2.1).
/// 8-bit: 6, 10-bit: 4, 12-bit: 2.
#[inline(always)]
pub fn mc_shift(bit_depth: u8) -> u8 {
    14 - bit_depth
}

/// MC luma filter rounding offset: `1 << (mc_shift - 1)`.
#[inline(always)]
pub fn mc_offset(bit_depth: u8) -> i32 {
    1i32 << (mc_shift(bit_depth) - 1)
}

/// Bi-prediction combining shift: `14 - bit_depth + 1`.
/// 8-bit: 7, 10-bit: 5, 12-bit: 3.
#[inline(always)]
pub fn bipred_shift(bit_depth: u8) -> u8 {
    14 - bit_depth + 1
}

/// Bi-prediction rounding offset: `1 << (bipred_shift - 1)`.
#[inline(always)]
pub fn bipred_offset(bit_depth: u8) -> i32 {
    1i32 << (bipred_shift(bit_depth) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_pixel_clamp() {
        assert_eq!(u8::from_i32_clamped(128, 8), 128u8);
        assert_eq!(u8::from_i32_clamped(-5, 8), 0u8);
        assert_eq!(u8::from_i32_clamped(300, 8), 255u8);
    }

    #[test]
    fn u16_pixel_clamp_10bit() {
        assert_eq!(u16::from_i32_clamped(512, 10), 512u16);
        assert_eq!(u16::from_i32_clamped(-5, 10), 0u16);
        assert_eq!(u16::from_i32_clamped(2000, 10), 1023u16);
    }

    #[test]
    fn u16_pixel_clamp_12bit() {
        assert_eq!(u16::from_i32_clamped(2048, 12), 2048u16);
        assert_eq!(u16::from_i32_clamped(5000, 12), 4095u16);
    }

    #[test]
    fn mc_shift_values() {
        assert_eq!(mc_shift(8), 6);
        assert_eq!(mc_shift(10), 4);
        assert_eq!(mc_shift(12), 2);
    }

    #[test]
    fn mc_offset_values() {
        assert_eq!(mc_offset(8), 32);
        assert_eq!(mc_offset(10), 8);
        assert_eq!(mc_offset(12), 2);
    }

    #[test]
    fn bipred_shift_values() {
        assert_eq!(bipred_shift(8), 7);
        assert_eq!(bipred_shift(10), 5);
        assert_eq!(bipred_shift(12), 3);
    }

    #[test]
    fn default_ref_sample_values() {
        assert_eq!(default_ref_sample(8), 128);
        assert_eq!(default_ref_sample(10), 512);
        assert_eq!(default_ref_sample(12), 2048);
    }

    #[test]
    fn max_pixel_values() {
        assert_eq!(max_pixel_val(8), 255);
        assert_eq!(max_pixel_val(10), 1023);
        assert_eq!(max_pixel_val(12), 4095);
    }
}
