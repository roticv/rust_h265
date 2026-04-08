# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Greenfield.** As of 2026-04-08 the repository contains only `.git` and this file. There is no `Cargo.toml`, no source, and no tests yet. The first task is to scaffold the crate (`cargo init --lib`).

## Project Overview

A pure Rust H.265 / HEVC video decoder library. Goal: a standalone, portable software HEVC decoder, mirroring the philosophy of the sibling `rust_h264` project at `/Users/roticv/Documents/code/rust_h264`. Most devices ship hardware HEVC decoders, but a portable software fallback is the motivation.

## Reference Project: rust_h264

`/Users/roticv/Documents/code/rust_h264` is the canonical reference for *project layout, build/test conventions, public API shape, and decoder structuring patterns* — **not** for codec internals. HEVC is a substantively different codec, not an extension of H.264, so very little algorithm code transfers. Use rust_h264 as a guide for *how to organise a video decoder in this style*; rewrite the codec logic from the HEVC spec (ITU-T H.265 / ISO/IEC 23008-2).

### What carries over from rust_h264 (~10–15%)
- Crate layout and module-per-concern split (`bitstream`, `nal`, `sps`, `pps`, `slice`, `dpb`, `decoder`, `error`, `intra_pred`, `inter_pred`, `residual`, `deblock`, entropy modules, etc.).
- Build/test/example tooling conventions (see below).
- The Annex B parser shape — but note NAL header layout differs: HEVC has a 2-byte NAL header with a 6-bit `nal_unit_type` (vs. 5 bits in H.264), plus `nuh_layer_id` and `nuh_temporal_id_plus1`.
- The CABAC arithmetic engine math (`get_cabac` / `get_cabac_bypass` / `get_cabac_terminate` core renormalization) — but **all context models, init tables, and bin strings are different**.
- General concepts: streaming `decode_nal` API, DPB with `Rc<DecodedPicture>` sharing, POC management, deblocking dispatch after slice decode, multi-slice picture accumulation pattern (`PictureState`).
- The bitstream reader (`read_bit`, `read_bits`, `read_ue`, `read_se`) is reusable nearly verbatim.

### What must be written from scratch (~85%)
1. **Block structure.** H.264's 16×16 macroblocks are gone. HEVC uses Coding Tree Units (CTU, typically 64×64) with quad-tree partitioning into CUs down to 8×8, each split into PUs (prediction) and TUs (transform, recursive 4×4 to 32×32). All MB-based scaffolding from rust_h264 (`MbInfo`, neighbor lookups, `BLOCK_INDEX_TO_OFFSET`, etc.) does not apply — design quad-tree CU/PU/TU traversal from the start.
2. **Intra prediction.** 35 modes (planar + DC + 33 angular) instead of H.264's 9 (4×4) + 4 (16×16). Reference samples are filtered before prediction in HEVC.
3. **Transform.** Adds 16×16 and 32×32 integer DCT, plus 4×4 DST for intra luma. Different basis matrices.
4. **Inter prediction.** AMVP + merge mode for MV prediction; PUs are 2N×2N, 2N×N, N×2N, plus AMP shapes. Reference list construction differs.
5. **Motion-compensation filters.** **Different tap counts and coefficients.** HEVC uses 7-tap and 8-tap luma filters and 4-tap chroma filters (vs. 6-tap luma / bilinear chroma in H.264). Any SIMD half-pel kernels in rust_h264 are not reusable.
6. **Entropy coding.** CABAC contexts and bin strings are completely different — hundreds of new context models. Initial-state tables come from the HEVC spec, not the H.264 ones.
7. **Deblocking.** Different boundary-strength derivation, different filter strengths, applies on an 8×8 grid (not 4×4 like H.264).
8. **New features absent in H.264:** SAO (Sample Adaptive Offset), tiles, wavefront parallel processing (WPP), slice segments (independent/dependent), VPS (Video Parameter Set, in addition to SPS/PPS).

When implementing a feature, read the HEVC spec section directly. Do not assume an H.264 algorithm generalizes — confirm against the spec text every time.

## Build Commands (once `cargo init --lib` is run)

- **Build:** `cargo build`
- **Build release:** `cargo build --release`
- **Test all:** `cargo test`
- **Run single test:** `cargo test <test_name>` (substring match)
- **Run a single test in release mode** (needed for large fixtures like 1080p): `cargo test --release <test_name>`
- **Lint:** `cargo clippy`
- **Format:** `cargo fmt`
- **Check:** `cargo check`

## Design Decisions (carry over from rust_h264)

- **Input format:** Annex B bytestream (start-code delimited `00 00 00 01` / `00 00 01`). HVCC (length-prefixed, used in MP4) is **not** supported — callers must convert. Mirrors the rust_h264 decision and keeps the parser self-contained.
- **Streaming API:** `Decoder::decode_nal(&[u8]) -> Result<Option<Frame>, DecodeError>` plus `flush()`. Callers feed NAL units incrementally; frames are emitted in **decode order** (not display order). Callers re-sort by POC for display, with the same IDR-count caveat documented in the rust_h264 README — increment the IDR counter *after* `decode_nal`, because the call returns the *previous* picture when it sees a new picture header.
- **Performance:** Target is parity (or close) with FFmpeg's software HEVC decoder. Prefer efficient algorithms, minimize allocations, avoid copies. Benchmarks live in `BENCHMARK.md` once measurable.
- **No `unsafe` unless justified.** rust_h264 stays safe outside hot SIMD paths; aim for the same here.
- **`testdata/` is committed in-tree** (matching rust_h264). Soft rule: commit fixtures under ~1MB so `cargo test` works on a fresh clone with no external tools. For anything larger, store a SHA-256 hash of the decoded output instead of the raw `.yuv` (rust_h264's 1080p tests use this pattern).

## Code Structure (target layout, modeled on rust_h264)

Suggested module split for a fresh start. Adapt as needed; HEVC's CU quad-tree may justify additional modules (e.g. a dedicated `cu_tree.rs`).

| File | Purpose |
|------|---------|
| `src/lib.rs` | Public re-exports (`decoder`, `nal`, `error`) and module declarations. Keep internals private; use a `dev-internals` feature to expose `vps`/`sps`/`pps`/`slice` for diagnostic examples (mirrors rust_h264). |
| `src/bitstream.rs` | MSB-first bit reader: `read_bit`, `read_bits`, `read_ue`, `read_se`. Largely portable from rust_h264. |
| `src/nal.rs` | Annex B parsing, emulation-prevention-byte removal (zero-copy fast path with `Cow::Borrowed`), 2-byte HEVC NAL header parsing (`nal_unit_type` 6 bits, `nuh_layer_id` 6 bits, `nuh_temporal_id_plus1` 3 bits). |
| `src/vps.rs` | Video Parameter Set (HEVC-specific, no H.264 analogue). |
| `src/sps.rs` | Sequence Parameter Set — HEVC profile/tier/level, CTU size, scaling lists, RPS structures. |
| `src/pps.rs` | Picture Parameter Set — tile config, dependent slice flag, etc. |
| `src/slice.rs` | Slice segment header parsing (independent + dependent slice segments). |
| `src/cu_tree.rs` | Quad-tree CU/PU/TU traversal: `coding_quadtree`, `coding_unit`, `prediction_unit`, `transform_tree`. The single biggest structural difference from H.264. |
| `src/cabac.rs` | CABAC arithmetic engine (renorm, bypass, terminate). Math reusable from rust_h264; init/state tables are HEVC-specific. |
| `src/cabac_tables.rs` | HEVC context model init values, rangeTabLPS, transIdxLPS/MPS — all from spec section 9.3. |
| `src/intra_pred.rs` | Planar, DC, 33 angular modes; reference sample filtering. |
| `src/inter_pred.rs` | 7/8-tap luma, 4-tap chroma interpolation; bi-pred averaging; weighted prediction. |
| `src/mv_pred.rs` | AMVP candidate list, merge candidate list, MV scaling. |
| `src/residual.rs` | 4×4 DST, 4×4/8×8/16×16/32×32 DCT inverses, dequantization with HEVC scaling lists. |
| `src/deblock.rs` | HEVC deblocking on 8×8 grid, bS derivation per spec 8.7.2. |
| `src/sao.rs` | Sample Adaptive Offset (edge offset + band offset). New in HEVC. |
| `src/dpb.rs` | Reference Picture Set (RPS) — HEVC's RPS replaces H.264's sliding-window/MMCO. Short-term and long-term refs derived from slice header each frame. |
| `src/decoder.rs` | Top-level `Decoder`, `decode_nal` dispatch, `PictureState` accumulation across slice segments, deblocking + SAO + DPB insert. |
| `src/error.rs` | `DecodeError { UnexpectedEof, InvalidSyntax, Unsupported }`. |

Prefer the rust_h264 pattern of bundling per-CTU/per-CU mutable state in a `*Context` struct passed to decode methods, with read-only slice-level params in a separate `*Params` struct.

## Testing Strategy (mirror rust_h264)

The H.264 project's test suite is the *single most valuable* thing to copy structurally. Each test:
1. Reads a `.h265` Annex B fixture from `testdata/`.
2. Decodes it through the public API.
3. Compares the output planes against a reference `.yuv` produced by FFmpeg byte-exact (or SHA-256 for very large fixtures like 1080p where storing raw YUV is impractical).

Generate fixtures with `ffmpeg`/`x265`, e.g.:
```
x265 --input-res 64x64 --fps 30 --frames 8 --preset medium --input input.yuv -o test.h265
ffmpeg -i test.h265 -f rawvideo -pix_fmt yuv420p test.yuv
```
Build the test corpus incrementally, starting with the smallest possible cases and adding features one at a time. Suggested progression:
1. Single I-slice, intra-only, smallest CTU, no SAO, no deblock.
2. Multi-CTU intra frame.
3. Each intra angular mode group.
4. P-slices: skip → uni-pred 2N×2N → AMP shapes → merge mode → AMVP.
5. B-slices.
6. SAO on, deblock on.
7. Tiles, then WPP, then dependent slice segments.
8. 1080p hash-only smoke test.

Commit fixtures in-tree (matching rust_h264) so tests run on a fresh clone without ffmpeg/x265 installed. Document the regeneration command in a comment at the top of the corresponding test. For very large reference outputs (>1MB or so), prefer SHA-256 hashing the decoded planes over committing the raw `.yuv`.

## Examples Directory

Once a basic decoder works, mirror rust_h264's example tooling. Useful first targets:
- `examples/play.rs` — minifb window playback (depends on `dev-dependency = "minifb"`), reorders by POC, supports `--fps`, `--loop`. Copyable from rust_h264 with the import path swapped.
- `examples/dump_frames.rs` — decode to raw YUV420 file in display order.
- `examples/bench_decode.rs` — measure decode throughput vs. wall clock for benchmarking against FFmpeg.

Diagnostic examples (`check_pps`, `compare_tests`, `debug_cavlc` in rust_h264) gate on a `dev-internals` feature so the public API stays minimal.

## Important Pitfalls

- **NAL header size:** HEVC NAL header is **2 bytes**, not 1. Don't copy rust_h264's `nal.rs` byte-for-byte.
- **NAL type values:** HEVC `nal_unit_type` ranges 0–63 (6 bits). VCL types are 0–31; non-VCL are 32–63 (VPS=32, SPS=33, PPS=34, AUD=35, EOS=36, EOB=37, FD=38, PREFIX_SEI=39, SUFFIX_SEI=40). IDR pictures are types 19–20 (IDR_W_RADL, IDR_N_LP), not type 5.
- **First slice in picture:** Detected by `first_slice_segment_in_pic_flag` in the slice header, not by comparing `frame_num`/POC like H.264. Use it to flush the previous picture from `PictureState`.
- **RPS, not MMCO:** HEVC rebuilds the reference picture set from the slice header each frame. Don't carry over rust_h264's sliding-window + MMCO state machine.
- **CTU size is variable.** Don't hard-code 64×64 — read `log2_min_luma_coding_block_size_minus3` and `log2_diff_max_min_luma_coding_block_size` from SPS.
- **8×8 deblocking grid.** HEVC deblocks on an 8×8 grid; rust_h264's per-4×4 boundary loop is the wrong granularity.
- **`pic_order_cnt_lsb` wraparound** uses `MaxPicOrderCntLsb` from SPS; the derivation is similar in spirit to H.264 POC type 0 but the field names differ.
