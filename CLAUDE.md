# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Feature-complete for Main and Main 10 profile (8-bit and 10-bit 4:2:0).**
127 tests pass. Byte-exact against FFmpeg on all in-tree fixtures plus 6
real-world Big Buck Bunny 1080p transcodes (8-bit and 10-bit). No SIMD yet.

## Build Commands

- **Build:** `cargo build`
- **Build release:** `cargo build --release`
- **Test all:** `cargo test --release`
- **Run single test:** `cargo test --release <test_name>`
- **Lint:** `cargo clippy`
- **Format:** `cargo fmt`
- **Play a file:** `cargo run --release --example play -- input.h265 [--fps 30] [--loop]`
- **Dump YUV:** `cargo run --release --example dump_frames -- input.h265 out.yuv`
- **Benchmark:** `cargo run --release --example bench_realworld`

## Architecture

### Public API

```rust
use rust_h265::{Decoder, Frame, DecodeError, parse_annex_b, NalUnitType, Pixel, PixelData};
```

- `Decoder::new()` → `decode_nal(&NalUnit) -> Result<Option<Frame>>` → `flush() -> Option<Frame>`
- `Frame` has `y: PixelData`, `u: PixelData`, `v: PixelData`, `width`, `height`, `bit_depth`, `pic_order_cnt`
- `PixelData::U8(Vec<u8>)` for 8-bit, `PixelData::U16(Vec<u16>)` for 10-bit
- Frames emitted in **decode order** — callers sort by POC for display

### Module layout

| Module | Purpose |
|--------|---------|
| `lib.rs` | Public re-exports, module declarations, `#[allow(dead_code)]` on internals |
| `pixel.rs` | `Pixel` trait (`u8`/`u16`), `PixelData` enum, bit-depth helpers (`mc_shift`, `bipred_shift`, etc.) |
| `decoder.rs` | Top-level `Decoder`, `Frame`, `PictureStateEnum` (u8/u16 dispatch), slice decode orchestration |
| `cu_tree.rs` | `PictureState<P>`, quad-tree CU/PU/TU decode, all CABAC syntax parsing, `SliceParams` |
| `inter_pred.rs` | MC luma 7/8-tap, chroma 4-tap (generic `<P: Pixel>`), bi-pred, weighted pred |
| `intra_pred.rs` | Planar, DC, 33 angular modes, reference sample filtering (generic `<P: Pixel>`) |
| `inverse_transform.rs` | 4×4 DST, 4/8/16/32 IDCT, DC fast path |
| `residual_coding.rs` | CABAC coefficient decode, dequantization (with `qp_bd_offset`), transform_skip |
| `deblock.rs` | 8×8-grid deblocking, beta/tc scaling by `1 << (bit_depth - 8)` |
| `sao.rs` | Band offset + edge offset, per-CTB SAO params, slice boundary handling |
| `dpb.rs` | `DecodedPicture` (with `PixelData` planes), DPB, RPS, ref list construction |
| `cabac.rs` | CABAC arithmetic engine (renorm, bypass, terminate) |
| `cabac_tables.rs` | LPS range, state transition, context init values |
| `slice.rs` | Slice header parsing, `LongTermRefPicSet`, `PredWeightTable` |
| `sps.rs` / `pps.rs` / `vps.rs` | Parameter set parsing |
| `nal.rs` | Annex B parsing, EPB removal, 2-byte NAL header |

### Key design patterns

- **`Pixel` trait generics:** All pixel-processing functions are `<P: Pixel>`. `PictureState<P>` stores `Vec<P>` planes. The decoder dispatches at runtime via `PictureStateEnum { U8(...), U16(...) }` with a `with_picture_state!` macro.
- **`SliceParams`** bundles per-slice state (ref lists, merge candidates, QP offsets, weight tables) passed to CU decode functions. Separate from `PictureState` which is per-picture mutable state.
- **Bit-depth parameterization:** `pixel::mc_shift(bit_depth)` = `14 - bit_depth`, `pixel::bipred_shift(bit_depth)` = `15 - bit_depth`. Dequant uses `qp + 6*(bit_depth-8)`. Deblock scales beta/tc by `1 << (bit_depth-8)`.
- **Stack-allocated MC scratch buffers:** `[i32; MAX_PB_LUMA]` instead of `Vec`, sized for 64×64 max PU.

## Testing Strategy

- 127 tests, all byte-exact against FFmpeg
- Fixtures in `testdata/` (committed in-tree, < 1 MB each)
- Large outputs (1080p, 10-bit) use SHA-256 hashes
- `decode_and_hash()` helper handles both `PixelData::U8` and `PixelData::U16` (10-bit hashes as LE u16 bytes)
- `bench_realworld` downloads Big Buck Bunny and transcodes 6 fixtures (4 × 8-bit, 2 × 10-bit)
- HM reference encoder used for PCM fixture (x265/kvazaar don't emit PCM blocks)
- kvazaar used for `pps_loop_filter_across_slices_enabled_flag = 0` fixture

## Important Pitfalls

- **10-bit MC shifts:** Sub-pel filter output needs `>> (bit_depth - 8)` before the final `>> mc_shift` in both direct-output (`mc_luma`/`mc_chroma`) and intermediate (`mc_luma_i32`/`mc_chroma_i32`) paths. Missing this produces values ~4× too large on 10-bit.
- **`qp_bd_offset`:** Dequant QP must include `6 * (bit_depth - 8)`. For 10-bit this is 12. Without it, dequantized coefficients are wildly wrong.
- **SAO merge flags at slice boundaries:** `sao_merge_left_flag` / `sao_merge_up_flag` must NOT be decoded when the neighbor CTB is in a different slice. Missing this desyncs CABAC for the rest of that slice.
- **`split_cu_flag` neighbor availability:** Must check `tab_slice_addr_rs` for cross-slice neighbors (not just picture boundaries). Missing this causes wrong CABAC context in multi-slice pictures.
- **Inter 4×4 luma TU:** The 4×4 DST is only for intra; inter 4×4 uses regular DCT. Also, `do_chroma_deferred` (blk_idx == 3) must be handled for both intra AND inter paths.
- **EPB compensation for WPP entry points:** `entry_point_offset` is in NAL-byte space (includes EPBs). Must convert to RBSP-byte offset by subtracting EPB count.
