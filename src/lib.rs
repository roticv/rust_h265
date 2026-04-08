//! Pure Rust H.265 / HEVC video decoder.
//!
//! See `CLAUDE.md` and `TODO.md` for the implementation plan. As of Phase 0
//! only the bitstream reader, NAL parser, and error type are implemented.

#[allow(dead_code)]
mod bitstream;
pub mod error;
pub mod nal;
