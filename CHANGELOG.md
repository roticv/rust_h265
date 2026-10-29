# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Multithreaded decoding:** `Decoder::with_threads(n)` (default 1).
  With 4 threads, 2.5-3.8x the single-threaded speed on the Big Buck
  Bunny fixtures (1080p `safe` 387 -> 1157 fps, `medium` 298 -> 769,
  10-bit `medium` 238 -> 586); about 0.8x FFmpeg's speed with 4 threads
  on 8-bit, 1.3x on 10-bit.
  - Frame threading: each picture decodes on its own thread, up to `n` at
    a time. Pictures that reference one still decoding start right away:
    its loop filters run row by row as its CTB rows finish, and motion
    compensation and temporal MV prediction wait only for the rows they
    read. Works for every stream.
  - Wavefront parallel processing (`entropy_coding_sync_enabled_flag`,
    x265's default): a picture with no other picture to overlap with (the
    first one, or a still image) decodes its CTB rows on separate
    threads, each row two CTBs behind the one above.
  - The loop filters of a picture decoded on the calling thread run over
    bands of picture rows.
  - Pictures run on a persistent pool of `n + 1` worker threads, created
    with the first picture.
  - With more than one thread, frames and slice-data errors come out of
    later calls than with one; `Decoder::next_error()` returns errors
    found after the last `decode_nal`.
  - The output is identical for every thread count, including for
    damaged streams: tested on every fixture with 1-7 threads, on
    corrupted WPP streams, and on both conformance suites.
- `Decoder` is now `Send`: reference pictures are shared with `Arc` and
  atomic marking instead of `Rc` / `RefCell`.
- `dump_frames` and `bench_decode` take `--threads N`.

### Changed

- The crate has `unsafe` code for the first time, confined to one module
  (`src/shared.rs`): multithreaded decoding has one thread reading parts
  of picture tables another thread is writing, which safe Rust cannot
  express. The module documents the rules that make it sound; in test
  builds every wavefront table access is checked against them, and
  motion compensation asserts it reads only reference rows it waited for.
- SAO is enabled per CTB, by the CTB's own slice (spec 8.7.3), instead of
  by the flags of the picture's last slice header.
- The DPB is also emptied when the chroma format changes (as for a bit
  depth change).

### Fixed

- Uni-prediction clamped reference sample positions to the current
  picture's size instead of the reference's (only differs for a
  reference of another size, in a broken stream).
- The published crate no longer includes the test fixtures (`testdata/`),
  the conformance tooling, CI configuration or `CLAUDE.md`: 256 KB
  instead of 1.06 MB. The tests run from a repository checkout.

## [0.3.0] - 2026-10-29

### Added

- **Range extensions (HEVC v2).** Every range-extension coding tool is
  implemented, in 4:2:0 and 4:4:4 up to 12 bits: cross-component
  prediction, implicit and explicit RDPCM, transform-skip rotation and
  contexts, transform skip up to 32x32, extended precision processing,
  persistent Rice adaptation, CABAC bypass alignment, chroma QP offset
  lists, SAO offset scaling, disabled intra smoothing and high-precision
  weighted-prediction offsets. WPP combined with tiles (allowed by the
  high-throughput profiles) is supported. All 34 JCT-VC-RExt vectors in
  these formats pass (FFmpeg passes 21; it lacks extended precision and
  bypass alignment). 4:2:2, monochrome, 16-bit and screen content coding
  are still rejected with `DecodeError::Unsupported`.
- `DecodeError` now implements `Clone`, `PartialEq` and `Eq`.
- **Display-order output:** `Decoder::new().with_display_order()` runs the
  spec C.5.2 output process ("bumping") inside the decoder. Frames are
  released in display order through the new `Decoder::next_output()` as
  soon as the stream's reorder and DPB limits allow, and `flush()` returns
  the remaining ones one per call. Pictures the spec does not output are
  dropped, including those still waiting when an IRAP picture with
  `NoOutputOfPriorPicsFlag = 1` starts a new sequence. The default mode is
  unchanged (decoding order), but display order is the recommended mode
  for new code: it is what a player or transcoder needs, and it handles
  reordering, `pic_output_flag` and sequence boundaries that a caller
  sorting by POC would otherwise have to get right. `dump_frames` and
  `play` now use display order; `dump_frames --decode-order` keeps the old
  path.

### Fixed

- **Range-extension streams decoded wrongly without an error.** The SPS
  and PPS extensions were never parsed, so a stream that enabled, for
  example, cross-component prediction decoded as if the tool were off. The
  extensions are now parsed (the SPS VUI and HRD parameters are skipped to
  reach them) and the tools implemented; screen content coding is rejected
  with `DecodeError::Unsupported`. A rejected SPS or PPS replaces an earlier
  one with the same ID, so later slices do not decode with a stale
  parameter set.
- **Chroma QP offset groups:** `cu_chroma_qp_offset_flag` was allowed once
  per CU instead of once per chroma QP offset group, and the offsets were
  reset per CU instead of per slice segment.
- **Bit depths above 12 are rejected** with `DecodeError::Unsupported`.
  14- and 16-bit streams were accepted, but motion compensation shifts
  underflow above 13 bits (found by fuzzing).
- **One bad slice no longer stops decoding.** A picture whose slices
  failed or were lost stayed "in progress", and every later picture was
  rejected with "new first slice segment arrived while previous picture
  was incomplete". Such a picture is now finished when the next one starts:
  CTBs no slice decoded are filled mid-grey, the loop filters run, and it
  is output and kept as a reference. A reference picture missing from the
  DPB is replaced by a generated mid-grey picture (spec 8.3.3) instead of
  failing with "reference picture POC not present in DPB". `flush()`
  finishes a picture left incomplete at the end of the stream. Damaging
  any single slice of the B-frame test streams now costs at most that
  picture.
- **JCT-VC HEVC_v1 conformance: all 147 vectors now decode bit-exact**
  (95 at the first run; the 4 output-order vectors need display-order mode). New harness in `tools/conformance/`. Fixes found
  with it:
  - SPS and PPS are stored by ID and activated per picture, so streams
    that switch between several parameter sets decode.
  - Reference picture lists are built per slice; temporal MV prediction
    and deblocking read each slice's own lists; long-term references
    matched by POC LSB resolve; long-term MV prediction rules apply.
  - RASL pictures after a BLA are skipped; pictures with
    `pic_output_flag = 0` are not returned; reserved VCL NAL types and
    slice header extensions are skipped.
  - Weighted bi-prediction, 10-bit weighted-prediction offsets and the
    10-bit QP range are correct.
  - `cu_qp_delta` takes its QP prediction from the CU origin. Inside CUs
    larger than the quantization group the QP was wrong; this also broke
    wavefront streams with 64x64 CTBs.
  - Loop filters: lossless and PCM samples are protected, PCM CU edges are
    deblocked, per-slice deblocking offsets and disable are honoured, the
    spec slice-boundary rules apply when slices disagree on filtering
    across slices, and SAO checks diagonal CTBs at CTB corners.
  - Raster slice addresses with tiles, context and QP resets at tile
    starts in dependent slice segments.
  - Constrained intra prediction marks individual reference samples
    unavailable, not whole sides.
  - With a parallel merge level above 2, 8x8 CUs share one merge list
    derived at the CU origin.
  - The motion vector sum wraps modulo 2^16 instead of saturating.
  - Luma and chroma may have different bit depths.
- **Crash on malformed SPS transform sizes** (found by fuzzing; present in
  0.1.0 and 0.2.0). An SPS whose minimum transform block was not smaller
  than its minimum coding block, as in 32-sample transforms inside 16-sample
  CTBs, made a per-transform-block table empty for small pictures, and
  deblocking then panicked on an out-of-bounds index. The SPS parser now
  enforces the spec 7.4.3.2.1 limits that FFmpeg also enforces:
  MinTbLog2SizeY < MinCbLog2SizeY, MaxTbLog2SizeY <= Min(CtbLog2SizeY, 5),
  and both `max_transform_hierarchy_depth` values within
  CtbLog2SizeY - MinTbLog2SizeY.
- **Tiled streams with real content decoded incorrectly.** Several
  derivations that must treat a neighbour in another tile as unavailable
  (spec 6.4.1) checked only the slice, or nothing: the `split_cu_flag` and
  `cu_skip_flag` CABAC contexts, the intra most-probable-mode candidates and
  the SAO merge flags. The CABAC desynchronised at the first tile boundary.
  The existing tiled fixture is flat gray, which hid this. The skip-flag
  context and MPM candidates also now respect slice boundaries.
- `loop_filter_across_tiles_enabled_flag` was parsed but ignored, so
  deblocking and SAO always filtered across tile boundaries.
- A new SPS now clears the active PPS's resolved tile geometry, so a stream
  that changes resolution while keeping its PPS uses the right tile layout.
- A predicted short-term reference picture set is rejected when it holds
  more than 16 pictures in either direction, the same limit as the explicit
  syntax. It was unbounded. This is the input pattern behind libde265
  GHSA-g2rg-wj66-w594; here it caused no memory-safety issue.

### Performance

Single-threaded, real-world content, Apple M4 (`bench_decode`, best of
three interleaved rounds of each release on the same machine):

| Fixture | 0.2.0 | 0.3.0 | Speedup |
|---|---:|---:|---:|
| 1080p `safe` preset | 245 fps | 379 fps | 1.55× |
| 720p `safe` preset | 440 fps | 685 fps | 1.56× |
| 1080p `medium` preset | 182 fps | 296 fps | 1.63× |
| 1080p `slow` preset | 181 fps | 291 fps | 1.61× |
| 1080p 10-bit `safe` | 207 fps | 297 fps | 1.44× |
| 1080p 10-bit `medium` | 157 fps | 235 fps | 1.50× |

Single-threaded FFmpeg 9.0.2 is now 1.5-1.6× faster on 8-bit (2.5× for
0.2.0), and `rust_h265` is 1.3-1.5× faster on 10-bit (about parity for
0.2.0). See `BENCHMARK.md`. The individual changes:

- The decode-time regression since 0.2.0 is gone. The per-sample
  neighbour check added to SAO edge offset (for CTB corners) cost 3-4% on
  SAO-heavy content; interior samples now take a per-row fast path. Hot
  CTB-address helpers no longer divide by the CTB size at run time.
  CPU cycles against 0.2.0 on the Big Buck Bunny 1080p benchmark
  fixtures: `bbb_1080p_5s_safe` unchanged, `bbb_1080p_5s_medium` 2%
  fewer.
- Per-picture buffers are pooled: a new picture reuses the tables and
  motion-compensation scratch of the previous one and the sample planes
  and motion field of a picture that left the DPB, instead of allocating
  and zero-filling them. About 1-2% fewer cycles; with the changes above,
  the 1080p fixtures now take 1-3% fewer cycles than 0.2.0.
- Residual decoding no longer allocates a coefficient block for every
  transform block, and the luma inverse transform runs in place instead
  of on a clone: one buffer is lent to each block in turn. About 1% fewer
  instructions and cycles.
- CABAC decision bins do one table lookup instead of two dependent ones
  (rangeTabLps and both state transitions fused into one table) and
  renormalise with a count-leading-zeros instead of a table. About 0.5%
  fewer instructions and cycles.
- Motion compensation: the interpolation filters accumulate tap by tap in
  fixed 8-sample chunks held in registers, instead of zero-filling a
  64-entry accumulator array for every row, and the bi-prediction /
  weighted-prediction intermediates are 16-bit (as in HM and FFmpeg)
  instead of 32-bit. 11-16% fewer cycles on the Big Buck Bunny 1080p and
  720p fixtures, 8-bit and 10-bit.
- SAO edge offset chooses the offset with comparisons and selects instead
  of a table lookup, so the interior loop vectorises, and works from a
  reused snapshot buffer instead of cloning three planes per picture.
  Intra prediction builds its reference samples in fixed-size stack arrays
  instead of up to nine heap vectors per transform block. 1-4% fewer
  cycles (up to 8.7% fewer instructions on SAO-heavy content).
- SAO edge offset runs in 16-bit lanes (twice the samples per vector
  instruction) and folds a row's first and last sample into the same
  vectorised run whenever their neighbours are available, so a full CTB
  row has no scalar tail. 6-7% fewer cycles on SAO-heavy 8-bit content
  (x265 `medium` / `slow` fixtures), 3% on 10-bit.
- The default bi-prediction average runs in 16-bit lanes with saturating
  adds (exact after the clip, as FFmpeg's NEON code relies on too) instead
  of 32-bit. About 1-2% fewer cycles on B-frame-heavy 8-bit content.
- The 8x8, 16x16 and 32x32 inverse DCTs skip the all-zero rows and columns
  outside the coded coefficient groups (like FFmpeg's `col_limit`) and run
  as even/odd matrix products in fixed-width lanes that vectorise, instead
  of one scalar butterfly per column and row. Inverse transform time fell
  from about 9.6% to 2% of decode time: 14-18% fewer cycles on the x265
  `safe` fixtures, 5-9% on `medium` / `slow`.
- Deblocking boundary strengths skip the internal-edge pass for coding
  units with a single prediction unit (skip and 2Nx2N, where motion is
  uniform and every internal edge has bS 0), and the luma edge loops skip
  64-sample runs of zero strengths with one 16-byte check. Boundary
  strength time fell from about 6% to 2% of decode time: 2-5% fewer
  cycles.

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
