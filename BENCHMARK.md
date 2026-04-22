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

### Results — 8-bit (Main profile)

| Fixture | Resolution | Frames | ours | ff-t1 | ff-tN | gap (ff-t1) |
|---|---:|---:|---:|---:|---:|---:|
| `bbb_1080p_5s_safe` | 1920×1080 | 120 | 223 Mpx/s (108 fps) | 1163 Mpx/s | 4217 Mpx/s | **5.2× slower** |
| `bbb_720p_10s_safe` | 1280×720 | 240 | 191 Mpx/s (207 fps) | 925 Mpx/s | 3686 Mpx/s | **4.8× slower** |
| `bbb_1080p_5s_medium` | 1920×1080 | 120 | 153 Mpx/s (74 fps) | 902 Mpx/s | 2326 Mpx/s | **5.9× slower** |
| `bbb_1080p_5s_slow` | 1920×1080 | 120 | 157 Mpx/s (76 fps) | 886 Mpx/s | 2326 Mpx/s | **5.6× slower** |

### Results — 10-bit (Main 10 profile)

| Fixture | Resolution | Frames | ours | ff-t1 | ff-tN | gap (ff-t1) |
|---|---:|---:|---:|---:|---:|---:|
| `bbb_1080p_5s_10bit_safe` | 1920×1080 | 120 | 199 Mpx/s (96 fps) | 511 Mpx/s | 2326 Mpx/s | **2.6× slower** |
| `bbb_1080p_5s_10bit_medium` | 1920×1080 | 120 | 146 Mpx/s (70 fps) | 357 Mpx/s | 1406 Mpx/s | **2.4× slower** |

### Headline

**8-bit:** At 1080p on real content, `rust_h265` delivers ~220 Mpx/s
(108 fps); single-threaded FFmpeg delivers ~1160 Mpx/s. FFmpeg is ≈ 5×
faster. The gap is consistent (4.8–5.9×) across 720p and 1080p real content.

**10-bit:** The gap is **much smaller** — only 2.4–2.6× vs FFmpeg
single-threaded. Our decoder at 199 Mpx/s (96 fps) vs FFmpeg at 511 Mpx/s.
This is because FFmpeg's 10-bit NEON kernels have less of an advantage: u16
samples halve the SIMD throughput (4 samples per 64-bit register instead of
8 for u8), while our scalar code scales more linearly. Both decoders are
significantly slower on 10-bit than 8-bit content (our safe preset: 223 →
199 Mpx/s, FFmpeg: 1163 → 511 Mpx/s).

All six fixtures are byte-exact against FFmpeg.

### Compatibility

All six fixtures (`safe` / `medium` / `slow` 8-bit presets + 720p `safe` +
two 10-bit presets) are byte-exact against FFmpeg. The `safe` preset is
`--preset ultrafast` with `bframes=1:ref=4:no-wpp=1:no-cutree=1`; `medium`
and `slow` are stock x265 presets with SAO + deblock enabled. 10-bit
fixtures use `--output-depth 10` for Main 10 profile.

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

## Profile: `bbb_1080p_5s_safe` (1920x1080, 120 frames, safe preset)

Collected with macOS `sample` (1 ms sampling, `RUSTFLAGS="-C force-frame-pointers=yes"`).
Best-of-5 wall-clock: **1.027 s** (116.8 fps, 242 Mpx/s).

### Function-level breakdown (inclusive, safe preset)

| Function | Inclusive % | Notes |
|---|---:|---|
| `motion_compensation_pu` | **68.7%** | The dominant bottleneck |
| - `mc_luma_i32` | 27.2% | 7/8-tap luma filter (i32 precision for weighted/bi-pred) |
| - `mc_chroma_i32` | 11.1% | 4-tap chroma filter (i32 for weighted/bi-pred) |
| - `mc_chroma` | 7.6% | 4-tap chroma filter (direct pixel-output path) |
| - self (weighted-pred loop, combining) | ~22.8% | Filter arithmetic inner loops |
| `decode_transform_tree` | 18.6% | Residual decode + IDCT + intra pred |
| - `apply_inverse_transform` | 8.4% | IDCT (tr_32: 3.9%, tr_16: 1.8%) |
| - `residual_coding` (CABAC) | 4.7% | sig_coeff_flag / coeff_abs_level bins |
| - `compute_deblocking_boundary_strengths` | 5.5% | Per-4x4-edge BS derivation |
| `PictureState::new` | 6.3% | Per-picture Y/U/V + bookkeeping allocation |
| `deblock_picture` | 3.9% | filter_luma_edge: 2.8% |
| `decode_prediction_unit` | 2.5% | Merge/AMVP syntax parsing |
| `predict_intra_chroma` | 1.3% | Angular + build_reference_samples |
| `decode_bin` (CABAC engine) | 0.3% | Arithmetic core is not the bottleneck |

### Key findings

1. **Motion compensation is 69% of decode time**, not inverse transforms.
   The earlier BENCHMARK.md prediction ("IDCT first, then MC") was inverted.
   On real 1080p content with dense inter blocks, MC runs on nearly every PU
   while IDCT only runs on non-zero TUs (many inter blocks have cbf_luma=0).

2. **MC self-time is filter arithmetic, not allocation.** After replacing
   per-PU `Vec` allocations with stack arrays (`[i32; MAX_PB_LUMA]`), the
   ~22.8% "self" time in `motion_compensation_pu` remained — it's the
   weighted-pred combining loops and filter inner loops, not malloc/free.

3. **PictureState::new at 6.3%** is pure allocation — Y/U/V planes
   (1920x1080x1.5 = 3.1 MB) plus deblocking/QP/MV bookkeeping arrays.
   A picture-buffer pool would amortize this across frames.

4. **IDCT is only 8.4%** — mostly 32x32 (3.9%) and 16x16 (1.8%). Still
   worth SIMD-ing, but the payoff is ~5x smaller than MC filters.

5. **Deblocking is 9.4% combined** — BS derivation (5.5%) plus the actual
   filter (3.9%). The BS derivation calls `inter_boundary_strength` per
   edge, which does MV comparison and ref-picture lookup; this is
   arithmetic, not memory-bound.

6. **CABAC is not a bottleneck** — `decode_bin` at 0.3% and
   `residual_coding` at 4.7% (dominated by coefficient-level bin reads,
   not the arithmetic engine). No need to optimize the CABAC core.

### Slow preset comparison

Profiling `bbb_1080p_5s_slow` (same content, x265 `--preset slow`) shows
the same pattern amplified: MC rises to **85.9%** inclusive (more refs,
deeper hierarchical-B, denser inter blocks), IDCT drops to 4.3%, and SAO
appears at 6.3% (slow-preset encodes with more SAO usage).

### Optimization attempt: stack-allocated MC scratch buffers

Replaced all per-PU heap allocations in the MC path with fixed-size stack
arrays (`[0i32; MAX_PB_LUMA]`, etc. — widened from i16 to i32 for 10-bit
safety). Result: malloc/free call sites in the profile dropped from 79 to
40, but **wall-clock time was unchanged** (~1.04 s → ~1.06 s, within noise).
The allocation overhead was only a few percent at most — the "~23%
self-time" initially attributed to allocation was actually spent in the
filter arithmetic inner loops. The stack arrays are still worth keeping
(fewer heap allocations, better cache locality), but the gap is squarely in
the per-pixel filter computation.

## Interpretation

### 8-bit gap (5–6×)

The single-threaded gap on 8-bit 1080p content is dominated by:

1. **Scalar filter kernels.** The MC 7/8-tap luma and 4-tap chroma sub-pel
   filters account for ~46% of decode time as pure per-pixel arithmetic.
   FFmpeg's arm64 build uses NEON intrinsics — processing 8 u8 pixels per
   128-bit register vs our 1 pixel at a time.

2. **Threading.** FFmpeg's `-threads 0` enables frame-parallel decode across
   all cores. `rust_h265` is single-threaded by design. The `ff-t1 → ff-tN`
   delta is roughly 3–4× on workloads big enough to keep 10 cores busy.

### 10-bit gap (2.4–2.6×)

The 10-bit gap is notably smaller because:

1. **SIMD advantage halved.** 10-bit samples use u16 (2 bytes), so NEON
   processes 4 samples per 64-bit lane instead of 8. FFmpeg's NEON kernels
   exist for 10-bit but their throughput advantage is ~2× (not ~4×).

2. **Our scalar code scales linearly.** The same `i32` arithmetic runs
   regardless of the storage type; the `Pixel` trait abstraction adds
   negligible overhead. Our 10-bit safe preset runs at 89% of 8-bit speed
   (199/223), while FFmpeg drops to 44% (511/1163).

### Allocation and bounds-check overhead

Second-order effects. `PictureState::new` is ~6% (heap allocation per
picture). MC scratch buffers use stack arrays. Bounds checks are present
throughout (no `unsafe`) but are not the bottleneck per the profile.

## Priorities

All six real-world fixtures decode byte-exact. Remaining focus is
performance, ordered by profile-informed impact:

1. **NEON SIMD for MC filters** (~46% of 8-bit decode time).
   `mc_luma` / `mc_luma_i32` 7/8-tap filter first (27%), then
   `mc_chroma` / `mc_chroma_i32` 4-tap (19%). Needs both u8 and u16
   kernel variants.

2. **NEON SIMD for IDCT** (~8%). tr_32 first (4%), then tr_16 (2%).

3. **Pool picture buffers** (~6%). Reuse `PictureState` Y/U/V planes and
   bookkeeping arrays across frames instead of allocating fresh each time.

4. **Deblocking optimization** (~9%). BS derivation could batch per-8x8
   block instead of per-4x4 edge; filter_luma_edge is a SIMD candidate.

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
