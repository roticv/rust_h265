# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
