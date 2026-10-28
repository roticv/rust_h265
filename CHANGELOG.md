# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-28

Adds 4:4:4 decoding, fixes open-GOP streams, hardens the decoder against
malformed input, and decodes real-world 1080p content 2.1–2.4× faster than
0.1.0. All in-tree fixtures and the real-world Big Buck Bunny fixtures
remain byte-exact against FFmpeg.

### Breaking changes

- `Frame` has a new public field, `chroma_format_idc`, and is now
  `#[non_exhaustive]`. Code that builds a `Frame` literally or destructures
  it without `..` must be updated. Future fields will not be breaking.
- `DecodeError` has a new variant, `ResourceLimit`, and is now
  `#[non_exhaustive]`. Exhaustive `match`es need a wildcard arm. Future
  variants will not be breaking.
- The decoder now enforces resource limits by default. An SPS whose
  picture exceeds 35,651,584 luma samples (the Level 6.2 maximum) or an
  estimated 512 MB of per-picture state is rejected with
  `DecodeError::ResourceLimit`. Use `Decoder::with_limits` to change either
  bound.
- A conformance window larger than the coded picture is rejected at SPS
  parse with `DecodeError::InvalidSyntax`. 0.1.0 produced empty frames.
- When decoding starts at a CRA picture (the first picture of the
  bitstream, or the first after an end-of-sequence NAL), its RASL pictures
  are now skipped, as the spec requires and as FFmpeg does. 0.1.0 failed on
  them.

### Added

- **4:4:4 decoding** (`ChromaArrayType` 3), 8-bit and 10-bit, intra and
  inter, including PCM blocks, as used by HEIF/HEIC still images. Intra
  support was contributed by Jens Walter
  ([#1](https://github.com/roticv/rust_h265/pull/1)). 4:2:2 and monochrome
  are still rejected with `DecodeError::Unsupported`.
- `Frame::chroma_format_idc` (1 = 4:2:0, 3 = 4:4:4) for sizing the chroma
  planes.
- `parse_hvcc` for length-prefixed NAL units from MP4/MKV containers, and
  `parse_nal` for a single raw NAL unit.
- `DecodeLimits` and `Decoder::with_limits`, contributed by Stephen
  Judkins ([#2](https://github.com/roticv/rust_h265/pull/2)).
- `NalUnitType::is_cra`.
- End-of-sequence NAL handling: the next picture starts a new coded video
  sequence.

### Fixed

- Open-GOP streams with a mid-stream CRA picture, which x265 produces by
  default for long encodes, failed with "reference picture POC not present
  in DPB". prevTid0Pic was reset to 0 after every IRAP picture instead of
  taking the IRAP picture's POC (spec 8.3.1).
- A CRA that starts decoding now gets POC MSB 0 (NoRaslOutputFlag, spec
  8.1.3), so decoding from a CRA with a large POC LSB, for example after a
  seek, no longer derives a negative POC.
- Inter prediction was wrong for picture widths that are not a multiple of
  the CTB size: reference planes were read with the picture width as their
  stride instead of the CTU-aligned stride.
- About 35 crashes, panics, integer overflows and out-of-memory conditions
  found by fuzzing, across SPS/PPS parameter validation, slice header entry
  points, tile scan tables, scaling lists, prediction weight tables, CABAC
  re-initialisation, coefficient and MVD decoding, motion compensation
  reference reads, dequantisation, SAO and the inverse transform. Each has
  a regression test.
- Conformance window cropping uses checked arithmetic and bounds-checked
  slicing (#2).

### Performance

Single-threaded, real-world 1080p content, Apple M4:

| Fixture | 0.1.0 | 0.2.0 | Speedup |
|---|---:|---:|---:|
| 1080p `safe` preset | 113 fps | 253 fps | 2.24× |
| 720p `safe` preset | 216 fps | 450 fps | 2.09× |
| 1080p `medium` preset | 80 fps | 189 fps | 2.36× |
| 1080p `slow` preset | 82 fps | 187 fps | 2.27× |
| 1080p 10-bit `safe` | 99 fps | 208 fps | 2.10× |
| 1080p 10-bit `medium` | 76 fps | 162 fps | 2.14× |

- Motion compensation filters are written so that the compiler vectorises
  them (NEON on aarch64, SSE/AVX on x86), with no `unsafe`. Interior blocks
  read reference samples directly instead of through a clamping helper, and
  8-bit filter passes accumulate in 16 bits.
- Motion compensation scratch buffers are allocated once per picture
  instead of being zero-filled per block.
- The min-TB Z-scan table is built once per picture geometry instead of
  per picture.
- Deblocking boundary strength skips the full comparison between identical
  motion fields.

See `BENCHMARK.md` for comparisons with FFmpeg and libde265.

## [0.1.0] - 2026-04-21

Initial release. Feature-complete HEVC decoder for Main and Main 10 profile.

### Added

- **Full Main Profile (8-bit 4:2:0) decoding** — byte-exact against FFmpeg on
  all test fixtures and real-world 1080p Big Buck Bunny content at x265 presets
  `ultrafast`, `medium`, and `slow`.
- **Full Main 10 Profile (10-bit 4:2:0) decoding** — byte-exact against FFmpeg.
  Generic `Pixel` trait abstracts over `u8`/`u16` storage; runtime dispatch via
  `PixelData` enum.
- **Streaming API:** `Decoder::decode_nal()` / `flush()` with `Frame` output
  containing `PixelData` planes, `width`, `height`, `bit_depth`, `pic_order_cnt`.
- **Annex B parser** with emulation-prevention-byte removal and zero-copy fast
  path (`Cow::Borrowed`).
- **Parameter set parsing:** VPS, SPS, PPS with full Main/Main 10 syntax.
- **Slice types:** I, P, B slices including hierarchical B-frames (`--bframes 3+`).
- **Block structure:** Quad-tree CTU/CU/PU/TU with CTU sizes 16, 32, 64.
  Asymmetric motion partitions (AMP).
- **Intra prediction:** All 35 modes (planar, DC, 33 angular) with reference
  sample filtering and strong intra smoothing. Constrained intra prediction.
- **Inter prediction:** AMVP + merge mode, temporal MVP, 7/8-tap luma and 4-tap
  chroma sub-pel filters, bi-prediction, weighted prediction (explicit P and B).
- **Transforms:** 4x4 DST (intra luma), 4/8/16/32 IDCT, DC fast path.
  Transform skip and transquant bypass (lossless) modes.
- **Entropy coding:** CABAC with all Main/Main 10 context models. Sign data
  hiding.
- **Residual coding:** Full coefficient decode with scaling lists (SPS and PPS),
  `cu_qp_delta`, `cu_chroma_qp_offset`, slice-level chroma QP offsets.
- **In-loop filters:** Deblocking (8x8 grid, strong/weak luma, chroma) and SAO
  (band offset + edge offset). Correct slice/tile boundary handling with
  `slice_loop_filter_across_slices_enabled_flag`.
- **Multi-slice / tiles / WPP:** Independent and dependent slice segments, tile
  geometry with scan-order tables, WPP with per-row CABAC state save/restore.
  Entry-point-offset EPB compensation for WPP/tile reinit.
- **DPB management:** Reference picture set derivation (short-term and long-term
  with MSB delta cycle resolution), picture marking (ST/LT/unused), ref list
  construction with `ref_pic_list_modification`.
- **PCM blocks:** Raw sample coding with CABAC reinit.
- **10-bit support:** All pixel-processing functions generic over `Pixel` trait.
  Dequant `qp_bd_offset`, MC filter shifts (`14 - bit_depth`), deblock beta/tc
  scaling, SAO band shift parameterized by bit depth. i32 intermediate buffers
  for MC (safe for 10-bit filter output range).
- **Examples:**
  - `play` — minifb window player with POC reorder, `--fps`, `--loop`
  - `dump_frames` — raw YUV output (8-bit yuv420p, 10-bit yuv420p10le)
  - `bench_decode` — single-file throughput with `--warmup` and `--repeat`
  - `bench_realworld` — Big Buck Bunny matrix (4x 8-bit + 2x 10-bit fixtures)
- **127 unit tests** covering the full feature matrix, all byte-exact against
  FFmpeg. Includes fixtures from x265, kvazaar, and the HM reference encoder.

### Performance

On Apple M4, 1080p Big Buck Bunny (single-threaded, no SIMD):

| Content | ours | FFmpeg (1 thread) | Gap |
|---|---|---|---|
| 8-bit safe preset | 223 Mpx/s | 1163 Mpx/s | 5.2x |
| 8-bit medium preset | 153 Mpx/s | 902 Mpx/s | 5.9x |
| 10-bit safe preset | 199 Mpx/s | 511 Mpx/s | 2.6x |
| 10-bit medium preset | 146 Mpx/s | 357 Mpx/s | 2.4x |

### Known limitations

- No SIMD — pure scalar safe Rust.
- No threading — single-threaded decode only.
- 4:2:0 chroma only (no 4:2:2 or 4:4:4).
- No SEI parsing (HDR metadata, timecodes silently skipped).
- No Range Extension features beyond transform skip / transquant bypass.
- No Screen Content Coding (palette, IBC).
- HVCC parameter set extraction (parsing the `HEVCDecoderConfigurationRecord`
  box itself) is left to the caller's container demuxer — `parse_hvcc()` handles
  the per-packet length-prefixed NAL splitting.
