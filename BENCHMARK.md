# Benchmarks

Throughput comparison of `rust_h265` against FFmpeg's built-in HEVC decoder
(`libavcodec/hevc`). Two benchmark matrices:

1. **Real-world content** — Big Buck Bunny (CC-BY Blender Foundation),
   transcoded from 1080p H.264 source into HEVC at several x265 presets.
   Driven by `examples/bench_realworld.rs`, which fetches the source via
   HTTP range requests and caches the transcoded fixtures in
   `target/bench-fixtures/`.
2. **Synthetic fixtures** — the in-tree `testdata/*.h265` conformance-test
   set (mostly `testsrc2` content, encoded with `--preset ultrafast` and
   most filters disabled to make byte-exact testing tractable). Driven by
   `examples/bench_decode.rs`.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M4 (10-core, 10 logical) |
| OS | macOS (Darwin 25.2.0) |
| Rust | rustc 1.92.0 |
| FFmpeg | 8.1 (Homebrew arm64 with NEON) |

## Methodology

- `ours` = our decoder, wall-clock time on the CPU, best-of-N with a warmup.
- `ff-t1` = `ffmpeg -threads 1`, reading `bench: rtime=` from
  `-benchmark` output. Single-threaded to match `rust_h265`.
- `ff-tN` = `ffmpeg -threads 0` (default), frame-parallel across all M4 cores.
- Measurement target ≥ 0.3 s of decode work per run to stay well above the
  1 ms resolution floor of ffmpeg's `rtime` field.

## Real-world content (Big Buck Bunny)

Source: `https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov`,
5-10 second segments starting at 1:00, transcoded with x265.

Reproduce:

```sh
cargo run --release --example bench_realworld
```

### Results

| Fixture | Resolution | Frames | ours | ff-t1 | ff-tN | gap (ff-t1) |
|---|---:|---:|---:|---:|---:|---:|
| `bbb_1080p_5s_safe` | 1920×1080 | 120 | 242 Mpx/s (117 fps) | 1147 Mpx/s (553 fps) | 3888 Mpx/s | **4.7× slower** |
| `bbb_720p_10s_safe` | 1280×720 | 240 | 207 Mpx/s (225 fps) | 933 Mpx/s | 3749 Mpx/s | **4.5× slower** |
| `bbb_1080p_5s_medium` | 1920×1080 | 120 | 172 Mpx/s (83 fps) | 928 Mpx/s | 2488 Mpx/s | **5.4× slower** |
| `bbb_1080p_5s_slow` | 1920×1080 | 120 | 179 Mpx/s (86 fps) | 915 Mpx/s | 2488 Mpx/s | **5.1× slower** |

### Headline

**At 1080p on real content with our supported settings, `rust_h265` delivers
~240 Mpx/s (116 fps); single-threaded FFmpeg delivers ~1150 Mpx/s. FFmpeg is
≈ 4.8× faster.** With frame-parallel threading (which `rust_h265` does not
support) FFmpeg reaches ~4000 Mpx/s — a 16× gap against `rust_h265` serial.

The gap is **consistent (4.5–4.8×)** between 720p and 1080p real content, and
matches the synthetic-fixture 1080p gap (4.5×). This is strong evidence that
the dominant bottleneck is the same across content types.

### Compatibility

All four fixtures (`safe` / `medium` / `slow` presets + the 720p `safe`
encode) are byte-exact against FFmpeg. The `safe` preset is
`--preset ultrafast --no-sao --no-deblock` plus a few other simplifications
used to make byte-exact comparison tractable with very small fixtures;
`medium` and `slow` are stock x265 presets with SAO + deblock enabled.

Two correctness bugs surfaced during this benchmarking work, both fixed
below: one in WPP entry-point handling (triggered by real 1080p content
with EPBs in slice data), and one pair in the inter 4×4 luma TU path
(triggered by `--tu-inter-depth ≥ 2`, which `preset slow` enables by
default).

### Bug fix: emulation-prevention-byte compensation for WPP entry points

The initial run of `bench_realworld` surfaced a correctness bug that the
in-tree conformance suite had not caught: HEVC spec 7.4.7.1 defines
`entry_point_offset_minus1[i]` in NAL-unit-byte units (i.e. the offsets
count the `0x03` emulation prevention bytes in the raw NAL stream). Our
decoder indexed into the RBSP (post-EPB-removal) using those values
directly, which happened to work on small conformance fixtures because
they had no EPBs in their slice data. Real 1080p content routinely emits
EPBs inside slice data, so every WPP row reinit landed a few bytes off,
corrupting the CABAC engine's initial `low` and desyncing the bitstream.

Fix: `src/nal.rs::NalUnit` now tracks the NAL-space positions where
emulation prevention bytes were removed; `src/decoder.rs` converts each
NAL-space `entry_point_offset` to an RBSP-space offset at WPP/tile reinit
time by subtracting the count of EPBs falling inside each substream's
NAL-space byte range (matching FFmpeg `hevcdec.c:2987-3016`). Covered by
`test_decode_wpp_ctu16_hash` (384×216 CTU=16 WPP with bframes=1, byte-exact
against FFmpeg).

### Bug fix: inter 4×4 luma TU path (`--tu-inter-depth ≥ 2`)

`preset slow` enables `--tu-inter-depth 2`, which lets the encoder split an
8×8 inter luma TU into four 4×4 sub-TUs. Two latent bugs sat on that code
path and only fired together under real 1080p slow-preset content:

1. **Inter chroma residual at `blk_idx == 3` was silently skipped.** When
   luma splits to 4×4, spec 7.3.8.11 moves chroma residual to the parent
   TU's position + size. Our `decode_transform_unit` handled this in the
   intra branch but not the inter branch — so inter 4×4-split luma TUs
   with non-zero inherited `cbf_cb` / `cbf_cr` never called
   `residual_coding()`, leaving the chroma coefficients unread and
   misaligning the CABAC state for the rest of the slice. Symptom:
   `InvalidSyntax("cu_qp_delta out of range")` a few hundred bins later.

2. **4×4 inter luma TUs incorrectly used the intra DST.** Spec 8.6.4.2
   reserves the 4×4 DST ("transform_4x4_luma") for **intra** luma TUs;
   inter 4×4 luma must use the regular 4×4 DCT. `apply_residual_to_luma`
   hard-coded `is_luma_intra_4x4 = log2_size == 2`, so it dispatched to
   DST for every 4×4 luma TU — correct for intra, silently wrong for
   inter. Visible as ±1–3 Y-plane residual errors on every inter 4×4 TU,
   cascading through MC into all dependent frames.

Fixes: extend the inter branch in `decode_transform_unit` with an
`else if do_chroma_deferred` arm mirroring FFmpeg `hls_transform_unit:
1456-1478`, and thread `is_intra` into `apply_residual_to_luma` so DST
dispatch is gated on `log2_size == 2 && is_intra`. Covered by
`test_decode_tu_inter_4x4_hash` (128×128, 6 frames,
`--preset medium --tu-inter-depth 3`, ~1.6 KB). `bbb_1080p_5s_slow` is
now byte-exact against FFmpeg.

### Real vs synthetic throughput

Real-world 1080p content costs roughly **3× more per pixel to decode** than
synthetic testsrc2 at comparable settings:

| Content | ours (Mpx/s) | ff-t1 (Mpx/s) |
|---|---:|---:|
| Synthetic 1080p (`testdata/1080p.h265`) | 770 | 3456 |
| Real-world 1080p safe preset (`bbb_1080p_5s_safe`) | 241 | 1152 |

Both decoders see the same ~3× cost increase, so our decoder's ratio stays
the same but both absolute numbers drop. Real content has dense residual
coefficients (more non-zero bins, more sig_coeff_flag CABAC decisions),
more CU-split variety, and more motion — all of which exercise the per-TU
and per-CU bookkeeping that dominates outside SIMD kernels.

## Synthetic fixtures (conformance test set)

Reproduce:

```sh
cargo build --release --example bench_decode
./target/release/examples/bench_decode testdata/1080p.h265 --warmup 2 --repeat 50
```

| Fixture | Resolution | Frames | ours (Mpx/s) | ff-t1 (Mpx/s) | ff-tN (Mpx/s) | ours vs ff-t1 |
|---|---:|---:|---:|---:|---:|---:|
| `1080p.h265` | 1920×1080 | 10 | 768 | 3456 | 3456 | 4.5× slower |
| `realworld_720p.h265` | 1280×720 | 10 | 279 | 461 | 838 | 1.7× slower |
| `realworld_320x240.h265` | 320×240 | 30 | 192 | 256 | 576 | 1.3× slower |
| `motion_320x240.h265` | 320×240 | 20 | 307 | 768 | 768 | 2.5× slower |
| `deblock_sao_320x240.h265` | 320×240 | 10 | 192 | 256 | 384 | 1.3× slower |
| `signhide_scaling_320x240.h265` | 320×240 | 10 | 192 | 256 | 384 | 1.3× slower |
| `aq_p_320x240.h265` | 320×240 | 5 | 192 | 192 | 192 | noise floor |
| `bframes3_128x128.h265` | 128×128 | 10 | 164 | 164 | 164 | noise floor |

These fixtures exist for conformance testing, not performance. They are short
(5–30 frames) and encoded with filters/features disabled to make byte-exact
comparisons tractable. The 1080p fixture is 10 frames of testsrc2 content
(only 1 KB on disk because testsrc2 compresses to almost nothing). Timings
under ~10 ms should be read as "both finish well under a measurement
quantum" rather than as meaningful ratios.

## Interpretation

The 4.5–4.8× single-threaded gap at 1080p across both content types points at
two factors:

1. **SIMD.** FFmpeg's arm64 build uses NEON for the HEVC hot paths: inverse
   transforms, motion-compensation 7/8-tap luma and 4-tap chroma sub-pel
   filters, deblocking, and SAO. `rust_h265` is pure scalar safe Rust; none
   of these kernels are vectorized. At 1080p those kernels are the bulk of
   decode cost, which matches the observed gap.

2. **Threading.** FFmpeg's `-threads 0` enables frame-parallel decode across
   all cores. `rust_h265` is single-threaded by design. The `ff-t1 → ff-tN`
   delta is where threading contributes — roughly 3–4× on all workloads big
   enough to keep 10 cores busy.

The remainder — branch predictability, cache locality, bounds-check
overhead — is second-order.

## Priorities

All four real-world fixtures decode byte-exact. Remaining focus is
performance:

1. **Profile the 1080p safe-preset fixture.** With a 1 s+ decode time
   (`bbb_1080p_5s_safe` at 120 frames × ~8 ms each), `samply` or
   `cargo flamegraph` should cleanly reveal the pixel-budget ordering —
   likely IDCT → MC filters → deblock → intra angular. That confirms
   which hot path is worth SIMD-ing first.

2. **Cheap scalar wins before SIMD.** Bounds-check removal in proven hot
   loops (`get_unchecked` with documented invariants) and stack-allocated
   fixed-size buffers for IDCT coefficients (≤ 32×32) typically pay back
   a lot before you reach for intrinsics.

3. **SIMD, in order of pixel budget.** IDCT first (2 M coefficients per
   1080p frame), then MC luma filter, then MC chroma filter, then deblock.

## Appendix: tool usage

### `bench_decode` — single-fixture throughput

```sh
cargo run --release --example bench_decode -- <file.h265> [--warmup K] [--repeat N]
```

Repeats the decode N times and prints best + avg wall-clock.

### `bench_realworld` — real-content matrix

```sh
cargo run --release --example bench_realworld [--only NAME] [--source URL_OR_PATH]
                                              [--only-generate] [--no-generate]
```

- On first run, downloads the source via ffmpeg (HTTP range requests) and
  transcodes each configured fixture with `libx265`. Cached in
  `target/bench-fixtures/`.
- `--only-generate`: fetch and transcode fixtures, skip the benchmark.
- `--no-generate`: fail if any fixture is missing instead of downloading.
- `--source`: override the default Big Buck Bunny URL (point at a local
  file, or a different public clip).
- Raw TSV results also written to `target/bench-fixtures/results.tsv` for
  downstream analysis.
