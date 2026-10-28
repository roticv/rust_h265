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
| Rust | rustc 1.99.1 |
| FFmpeg | 9.0.2 (Homebrew arm64 with NEON) |

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
| `bbb_1080p_5s_safe` | 1920×1080 | 120 | 304 Mpx/s (147 fps) | 1232 Mpx/s | 4218 Mpx/s | **4.1× slower** |
| `bbb_720p_10s_safe` | 1280×720 | 240 | 250 Mpx/s (271 fps) | 987 Mpx/s | 3511 Mpx/s | **4.0× slower** |
| `bbb_1080p_5s_medium` | 1920×1080 | 120 | 228 Mpx/s (110 fps) | 943 Mpx/s | 2370 Mpx/s | **4.1× slower** |
| `bbb_1080p_5s_slow` | 1920×1080 | 120 | 233 Mpx/s (112 fps) | 939 Mpx/s | 2370 Mpx/s | **4.0× slower** |

### Results — 10-bit (Main 10 profile)

| Fixture | Resolution | Frames | ours | ff-t1 | ff-tN | gap (ff-t1) |
|---|---:|---:|---:|---:|---:|---:|
| `bbb_1080p_5s_10bit_safe` | 1920×1080 | 120 | 293 Mpx/s (141 fps) | 463 Mpx/s | 1975 Mpx/s | **1.6× slower** |
| `bbb_1080p_5s_10bit_medium` | 1920×1080 | 120 | 241 Mpx/s (116 fps) | 301 Mpx/s | 1020 Mpx/s | **1.25× slower** |

Each figure is the best of two full `bench_realworld --no-generate` runs,
taken per column, on the same cached fixtures as earlier rounds.

### Headline

**8-bit:** At 1080p on real content, `rust_h265` delivers 228–304 Mpx/s
(110–147 fps); single-threaded FFmpeg delivers ~940–1230 Mpx/s. FFmpeg is
≈ 4× faster, consistently across 720p and 1080p and all three presets.

**10-bit:** The gap is much smaller, 1.25–1.6× vs FFmpeg single-threaded.
Our decoder runs at 241–293 Mpx/s (116–141 fps) against FFmpeg's 301–463
Mpx/s. Our 10-bit throughput is within a few percent of our 8-bit
throughput, while FFmpeg's drops to under 40% of its 8-bit speed (see
Interpretation). Note that FFmpeg 9.0.2 measured 10–16% lower on the 10-bit
fixtures than FFmpeg 8.1 did in the previous round, so part of the
narrower 10-bit gap reflects FFmpeg, not this decoder.

All six fixtures are byte-exact against FFmpeg.

### What changed since the previous round

The previous round measured 146–223 Mpx/s for `rust_h265`. Three changes
to motion compensation and picture setup account for a 1.3–1.65× speedup:

| Fixture | previous ours | ours now | speedup |
|---|---:|---:|---:|
| `bbb_1080p_5s_safe` | 223 Mpx/s | 304 Mpx/s | 1.36× |
| `bbb_720p_10s_safe` | 191 Mpx/s | 250 Mpx/s | 1.31× |
| `bbb_1080p_5s_medium` | 153 Mpx/s | 228 Mpx/s | 1.49× |
| `bbb_1080p_5s_slow` | 157 Mpx/s | 233 Mpx/s | 1.48× |
| `bbb_1080p_5s_10bit_safe` | 199 Mpx/s | 293 Mpx/s | 1.47× |
| `bbb_1080p_5s_10bit_medium` | 146 Mpx/s | 241 Mpx/s | 1.65× |

1. **Reference windows read directly** (largest gain, +20–48%). Every MC
   filter tap used to go through a helper that clamps both coordinates and
   bounds-checks the index: 8 clamps per output sample for the 8-tap luma
   filter. `ref_window` now checks once per block whether the filter window
   lies inside the reference picture. Interior blocks read the plane
   directly; edge blocks read a padded copy built with the same clamping
   helper, so results are unchanged.
2. **MC scratch buffers reused** (+11–21% over the version before it). The
   MC path zero-filled its intermediates on every prediction block, about
   100 KB per bi-predicted block once 4:4:4 support had enlarged the
   chroma buffers. They now live in a per-picture `McScratch` that every
   block reuses.
3. **Z-scan table built once** (+2–4%). `PictureState::new` rebuilt the
   min-TB Z-scan table, about 130,000 entries at 1080p, for every picture.
   It depends only on picture geometry, so it is now built once and shared.

### Comparison with libde265

libde265 (the decoder libheif uses for HEIC) was measured separately with
a small harness against its C API, Homebrew build 1.1.3. On this machine
that build has no NEON kernels, so it is scalar C against scalar Rust.
Before the changes above libde265 was 1.3–1.6× faster than `rust_h265`
single-threaded; afterwards `rust_h265` matched it on 8-bit 1080p content
(146 vs 147 fps on `bbb_1080p_5s_safe`, 116 vs 115 fps on `medium`) and was
faster on 10-bit. The harness is not in-tree, so treat these as a spot
check rather than a maintained benchmark.

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

_Figures in this subsection are from the earlier measurement round._

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

_This table is from the earlier measurement round and has not been re-run._

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

## Current profile: `bbb_1080p_5s_safe`

Collected with macOS `sample` on the current code (release build with debug
info), grouped by decoder stage. Self time, so each sample counts once.

| Stage | Self time |
|---|---:|
| Motion compensation | 62.4% |
| Deblocking incl. boundary strength | 10.9% |
| Inverse transform | 10.1% |
| CU/PU/TU syntax + MV derivation | 6.7% |
| Residual coding + CABAC | 4.3% |
| Picture setup + allocation | 3.7% |
| Intra prediction | 1.8% |

Top functions: `motion_compensation_pu` 22.3% (bi-pred combining and the
inlined direct-output luma filter), `mc_luma_i32` 18.1%, `mc_chroma_i32`
13.1%, `mc_chroma` 8.8%, `apply_inverse_transform` 5.8%,
`inter_boundary_strength` 4.0%. Motion compensation is still scalar; the
per-tap clamping and per-block zeroing that used to sit on top of the
arithmetic are gone.

## Historical profile: `bbb_1080p_5s_safe` (before the changes above)

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

_Superseded: the stack arrays were later replaced by reused per-picture
buffers, which did pay off once the buffers grew for 4:4:4 (see "What
changed since the previous round")._

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

### 8-bit gap (about 4×)

The single-threaded gap on 8-bit 1080p content is dominated by:

1. **Scalar filter kernels.** Motion compensation is 62% of decode time,
   nearly all of it per-pixel filter and combining arithmetic.
   FFmpeg's arm64 build uses NEON intrinsics — processing 8 u8 pixels per
   128-bit register vs our 1 pixel at a time.

2. **Threading.** FFmpeg's `-threads 0` enables frame-parallel decode across
   all cores. `rust_h265` is single-threaded by design. The `ff-t1 → ff-tN`
   delta is roughly 3–4× on workloads big enough to keep 10 cores busy.

### 10-bit gap (1.25–1.6×)

The 10-bit gap is notably smaller because:

1. **SIMD advantage halved.** 10-bit samples use u16 (2 bytes), so NEON
   processes 4 samples per 64-bit lane instead of 8. FFmpeg's NEON kernels
   exist for 10-bit but their throughput advantage is ~2× (not ~4×).

2. **Our scalar code scales linearly.** The same `i32` arithmetic runs
   regardless of the storage type; the `Pixel` trait abstraction adds
   negligible overhead. Our 10-bit safe preset runs at 96% of 8-bit speed
   (293/304), while FFmpeg drops to 38% (463/1232).

### Allocation and bounds-check overhead

Second-order effects now. Picture setup and allocation are ~4% (fresh
Y/U/V planes and bookkeeping tables per picture; the Z-scan table is
shared). MC scratch buffers are allocated once per picture and reused.
Bounds checks are present throughout (no `unsafe`) but are not the
bottleneck per the profile.

## Priorities

All six real-world fixtures decode byte-exact. Remaining focus is
performance, ordered by profile-informed impact:

1. **NEON SIMD for MC filters** (62% of 8-bit decode time). `mc_luma_i32`
   and the direct-output luma filter first, then `mc_chroma_i32` /
   `mc_chroma`, then the bi-pred combining loop. Needs both u8 and u16
   kernel variants. Now that interior blocks read the reference plane
   directly, the filter loops are straightforward to vectorise.

2. **Deblocking** (~11%). BS derivation could batch per-8x8 block instead
   of per-4x4 edge; `filter_luma_edge` is a SIMD candidate.

3. **NEON SIMD for IDCT** (~10%). tr_32 first, then tr_16.

4. **Pool picture buffers** (~4%). Reuse Y/U/V planes and bookkeeping
   arrays across frames instead of allocating fresh each time.

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
