# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Complete for Main and Main 10, and for the 4:2:0 / 4:4:4 range-extension
profiles up to 12 bits with every RExt coding tool.** Conformance
(`tools/conformance/`): JCT-VC HEVC_v1 147 / 147, JCT-VC-RExt 34 / 34 of
the vectors in supported formats. Byte-exact against FFmpeg on all in-tree
fixtures plus 6 real-world Big Buck Bunny 1080p transcodes (8-bit and
10-bit). Rejected with `DecodeError::Unsupported` rather than decoded
wrongly: 4:2:2, monochrome, bit depth > 12, SCC. Display-order output
(`with_display_order`) and recovery from damaged slices are in place. No
threading or SIMD yet.

## Build Commands

- **Build:** `cargo build`
- **Build release:** `cargo build --release`
- **Test all:** `cargo test --release`
- **Run single test:** `cargo test --release <test_name>`
- **Lint:** `cargo clippy --release --all-targets -- -D warnings` (what CI runs)
- **Format:** `cargo fmt`
- **Play a file:** `cargo run --release --example play -- input.h265 [--fps 30] [--loop]`
- **Dump YUV:** `cargo run --release --example dump_frames -- input.h265 out.yuv`
- **Benchmark:** `cargo run --release --example bench_realworld`

## Architecture

### Public API

```rust
use rust_h265::{Decoder, Frame, DecodeError, parse_annex_b, parse_hvcc, parse_nal, NalUnitType, Pixel, PixelData};
```

- `Decoder::new()` → `decode_nal(&NalUnit) -> Result<Option<Frame>>` → `flush() -> Option<Frame>`
- `parse_annex_b(&[u8])` — split Annex B bytestream (start-code delimited, `.h265` files)
- `parse_hvcc(&[u8], length_size)` — split length-prefixed NALs (MP4/MKV container packets)
- `parse_nal(&[u8])` — parse a single raw NAL (no framing)
- `Frame` has `y: PixelData`, `u: PixelData`, `v: PixelData`, `width`, `height`, `bit_depth`, `pic_order_cnt`
- `PixelData::U8(Vec<u8>)` for 8-bit, `PixelData::U16(Vec<u16>)` for 10-bit
- Frames emitted in **decode order** by default — callers sort by POC within each CVS. `Decoder::new().with_display_order()` runs the spec C.5.2 output process instead (`output.rs`): `decode_nal` returns `Ok(None)`, frames come from `next_output()` and, at the end, `flush()` until `None`. `dump_frames` uses display order (`--decode-order` for the old path)

### Module layout

| Module | Purpose |
|--------|---------|
| `lib.rs` | Public re-exports, module declarations, `#[allow(dead_code)]` on internals |
| `pixel.rs` | `Pixel` trait (`u8`/`u16`), `PixelData` enum, bit-depth helpers (`mc_shift`, `bipred_shift`, etc.) |
| `decoder.rs` | Top-level `Decoder`, `Frame`, `PictureStateEnum` (u8/u16 dispatch), slice decode orchestration |
| `cu_tree.rs` | `PictureState<P>`, quad-tree CU/PU/TU decode, all CABAC syntax parsing, `SliceParams` |
| `inter_pred.rs` | MC luma 7/8-tap, chroma 4-tap (generic `<P: Pixel>`, 4:2:0 and 4:4:4 geometry), bi-pred, weighted pred |
| `intra_pred.rs` | Planar, DC, 33 angular modes, reference sample filtering (generic `<P: Pixel>`) |
| `inverse_transform.rs` | 4×4 DST, 4/8/16/32 IDCT, DC fast path |
| `residual_coding.rs` | CABAC coefficient decode, dequantization (with `qp_bd_offset`), transform_skip |
| `deblock.rs` | 8×8-grid deblocking, beta/tc scaling by `1 << (bit_depth - 8)` |
| `sao.rs` | Band offset + edge offset, per-CTB SAO params, slice boundary handling |
| `output.rs` | Display-order output process (spec C.5.2 bumping, NoOutputOfPriorPicsFlag), used by `with_display_order` |
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
- **MC kernels are written for auto-vectorisation:** `inter_pred::filter_px` / `filter_v_i16` loop tap-major over fixed 8-lane chunks (`&[P; 8]` row windows, `[i16; 8]` / `[i32; 8]` accumulators) with a scalar tail for narrow chroma, which LLVM keeps in registers and turns into NEON/SSE. Do not reintroduce per-row accumulator arrays (`[0; MAX_PB_SIZE]` zero-filled per row), per-sample index closures or per-pixel bounds-checked writes: replacing the per-row arrays with chunks took 11-16% off total decode time. 8-bit first passes accumulate in i16 (bounded by `MAX_FILTER_ABS_SUM * 255`), deeper samples and the 2-D second pass in i32. Bi-prediction / weighted intermediates (`mc_luma_i16` / `mc_chroma_i16`, `McScratch`) are i16, as in HM and FFmpeg.
- **Picture buffer pool:** `decoder::PicturePool` keeps the last finished `PictureState` (its tables are refilled with their initial values by `PictureState::new_recycled`) and the planes / motion field of pictures that leave the DPB. Recycled sample planes are *not* cleared: decoding writes every sample inside the picture and nothing reads the CTU-alignment padding. Do not add code that relies on zero-initialised planes; refill any new table in `new_recycled`.
- **Reused MC scratch buffers:** `inter_pred::McScratch` (owned by `PictureState`, lent to each PU by `motion_compensation_cu`) holds the bi-pred / weighted intermediates and the separable-filter temporary, sized for 64×64 luma and 64×64 chroma (4:4:4). Do not reintroduce per-call zero-initialised stack arrays in the MC path: zero-filling ~100 KB per bi-predicted block cost 7-10% of total decode time on 1080p content.

## Testing Strategy

- All tests byte-exact against FFmpeg
- Fixtures in `testdata/` (committed in-tree, < 1 MB each)
- Large outputs (1080p, 10-bit) use SHA-256 hashes
- `decode_and_hash()` helper handles both `PixelData::U8` and `PixelData::U16` (10-bit hashes as LE u16 bytes)
- `bench_realworld` downloads Big Buck Bunny and transcodes 6 fixtures (4 × 8-bit, 2 × 10-bit)
- HM reference encoder used for PCM fixture (x265/kvazaar don't emit PCM blocks)
- kvazaar used for `pps_loop_filter_across_slices_enabled_flag = 0` fixture
- `DecodeLimits` (`Decoder::with_limits`) caps per-picture pixels (Level 6.2 MaxLumaPs) and estimated memory; SPS exceeding it returns `DecodeError::ResourceLimit`
- 4:4:4 fixtures (x265 `--input-csp i444`): `intra444_96x96`, `intra444_10bit_96x96`, `intra444_100x68` (conformance window in luma units), `inter444_96x96` (P), `inter444_b_96x96` (B + weighted pred), `inter444_10bit_96x96`, `inter444_100x68`
- Tiles with real content: `tiles_inter_kvazaar_384x256.h265` (kvazaar 3x2, IPPP, no filtering across tiles), `tiles_hm_no_lf_across_832x256.h265` / `tiles_hm_lf_across_832x256.h265` (HM, non-uniform 3x2, both loop-filter-across-tiles settings), `tiles_resize.h265` (resolution change keeping the PPS)
- `inter420_100x68.h265`: non-CTB-aligned width with inter prediction (reference plane stride != width)
- `pcm444_hm_64x64.h265` / `pcm444_10bit_hm_64x64.h265`: HM-encoded 4:4:4 PCM (full-size chroma payload), 8-bit and 10-bit
- Building HM on macOS with AppleClang 17 / Homebrew OpenSSL 3: drop `-Werror` from `cmake/CMakeBuild/cmake/modules/BBuildEnv.cmake` and stub the `EVP_PKEY_size` call in `SEIDigitallySignedContent.cpp`
- `open_gop_cra.h265` / `seek_to_cra.h265`: x265 `--open-gop` with a mid-stream CRA (POC 150, past the LSB half-range) and the same stream cut to start at the CRA

## Important Pitfalls

- **10-bit MC shifts:** Sub-pel filter output needs `>> (bit_depth - 8)` before the final `>> mc_shift` in both direct-output (`mc_luma`/`mc_chroma`) and intermediate (`mc_luma_i32`/`mc_chroma_i32`) paths. Missing this produces values ~4× too large on 10-bit.
- **`qp_bd_offset`:** Dequant QP must include `6 * (bit_depth - 8)`. For 10-bit this is 12. Without it, dequantized coefficients are wildly wrong.
- **SAO merge flags at slice boundaries:** `sao_merge_left_flag` / `sao_merge_up_flag` must NOT be decoded when the neighbor CTB is in a different slice. Missing this desyncs CABAC for the rest of that slice.
- **`split_cu_flag` neighbor availability:** Must check `tab_slice_addr_rs` for cross-slice neighbors (not just picture boundaries). Missing this causes wrong CABAC context in multi-slice pictures.
- **Inter 4×4 luma TU:** The 4×4 DST is only for intra; inter 4×4 uses regular DCT. Also, `do_chroma_deferred` (blk_idx == 3) must be handled for both intra AND inter paths.
- **prevTid0Pic and CRA:** Per spec 8.3.1 an IRAP is an ordinary prevTid0Pic candidate: store its real POC, never reset to 0 (only an IDR has POC 0). A CRA that starts decoding (first picture or after EOS) has NoRaslOutputFlag = 1: force POC MSB = 0 and skip its RASL pictures, as FFmpeg does.
- **Chroma geometry is not hard-coded 4:2:0:** `PictureState::chroma_shift_w/h` (log2 SubWidthC/SubHeightC) and `Sps::chroma_array_type()` drive every luma-to-chroma mapping. Conformance-window offsets are in chroma units, so they scale by SubWidthC (2 for 4:2:0, 1 for 4:4:4). 4:4:4 chroma: TBs match luma size (no 4×4 deferral), cbf signalled at every size, per-PU chroma modes for NxN, luma-style reference smoothing, chroma QP = Min(qPi, 51), deblock on the 8-sample grid. Inter chroma MC uses `chroma_mv()` = `mv * 2 >> log2(SubWidthC)` in eighth-chroma-sample units, chroma block = PU size >> shift, and the chroma MC scratch buffers are sized 64×64.
- **Reference planes keep the CTU-aligned stride:** `DecodedPicture::y_stride` / `uv_stride` come from `PictureState`, not from `width`. Motion compensation must read references through those fields; using `width` as the stride silently corrupts every inter picture when the width is not a multiple of the CTB size (e.g. 100×68 with CTU 32).
- **Neighbour availability (spec 6.4.1) means same slice AND same tile:** use `cu_tree::neighbour_available` for CABAC context neighbours (split_cu_flag, cu_skip_flag), MPM candidates and similar; SAO merge and merge/AMVP candidates do their own slice+tile checks. Loop filters must honour `loop_filter_across_tiles_enabled_flag` (`PictureState::filter_across_tiles`). Flat-content fixtures hide these bugs; test tiles with real content (`tiles_inter_kvazaar_384x256.h265`, `tiles_hm_*_832x256.h265`).
- **QP prediction uses the CU origin:** `cu_qp_delta` coded in a TU derives qPY_PRED from the CU origin (`PictureState::cu_origin`), not the TU or transform-tree base. In a CU larger than the quantization group the two differ and the TU version reads stale QPs from inside the CU. Only 64×64 CTBs with `diff_cu_qp_delta_depth > 0` expose it.
- **Luma and chroma bit depths can differ:** use `PictureState::bit_depth_c` / `sps.bit_depth_chroma` for every chroma path (MC shifts, weighted prediction, intra default value, clipping, SAO offset cMax). `Sps::needs_u16_samples()` picks the sample type from both depths.
- **Unsupported features are rejected, not ignored:** the SPS parser skips the VUI/HRD (`skip_vui`) to reach the extensions; SCC returns `Unsupported`. A failed SPS/PPS is stored in the ID table as its error (activation returns it), so it replaces an older set with the same ID.
- **Range extensions:** flags live in `Sps::range` and the PPS range fields. Rotation applies to transform-skip *and* transquant-bypass 4×4 intra blocks (HM; FFmpeg skips bypass). StatCoeff (persistent Rice) lives in `CabacContexts`, so it resets and syncs with the contexts. Extended precision with a coefficient range above 16 bits uses `ResidualBlock::coeffs_wide` and `inverse_transform::inverse_transform_ext`; the 8-bit path is unchanged. With WPP and tiles together, a CTB row of a tile is a substream (`tile_col_start` in `decode_slice`). Verify RExt work with the JCT-VC-RExt suite and the HM-recon fixtures (`test_decode_rext_tools_match_hm`); FFmpeg lacks extended precision and bypass alignment and differs on lossless RDPCM.
- **Conformance:** `tools/conformance/run_suite.py` runs the JCT-VC suites from Fluster's JSON (see its README). Re-run it after changes to parsing, prediction or loop filters.
- **EPB compensation for WPP entry points:** `entry_point_offset` is in NAL-byte space (includes EPBs). Must convert to RBSP-byte offset by subtracting EPB count.
