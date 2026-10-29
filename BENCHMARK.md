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
| `bbb_1080p_5s_safe` | 1920×1080 | 120 | 795 Mpx/s (383 fps) | 1214 Mpx/s | 4147 Mpx/s | **1.53× slower** |
| `bbb_720p_10s_safe` | 1280×720 | 240 | 640 Mpx/s (695 fps) | 987 Mpx/s | 3511 Mpx/s | **1.54× slower** |
| `bbb_1080p_5s_medium` | 1920×1080 | 120 | 585 Mpx/s (282 fps) | 946 Mpx/s | 2347 Mpx/s | **1.62× slower** |
| `bbb_1080p_5s_slow` | 1920×1080 | 120 | 578 Mpx/s (279 fps) | 918 Mpx/s | 2326 Mpx/s | **1.59× slower** |

### Results — 10-bit (Main 10 profile)

| Fixture | Resolution | Frames | ours | ff-t1 | ff-tN | gap (ff-t1) |
|---|---:|---:|---:|---:|---:|---:|
| `bbb_1080p_5s_10bit_safe` | 1920×1080 | 120 | 607 Mpx/s (293 fps) | 469 Mpx/s | 1914 Mpx/s | **1.29× faster** |
| `bbb_1080p_5s_10bit_medium` | 1920×1080 | 120 | 464 Mpx/s (224 fps) | 305 Mpx/s | 1106 Mpx/s | **1.52× faster** |

Each figure is the best of two full `bench_realworld --no-generate` runs,
taken per column, on the same cached fixtures as earlier rounds. The
machine was not idle (load average 3-5 from other work), which affects
both decoders. Counting CPU cycles instead of wall-clock time (`/usr/bin/time
-l`, minimum of 5 runs) gives a slightly smaller 8-bit gap, 1.37-1.44×,
and 0.63-0.75× on 10-bit.

### Headline

**8-bit:** At 1080p on real content, `rust_h265` delivers 578–795 Mpx/s
(279–383 fps); single-threaded FFmpeg delivers ~920–1210 Mpx/s. FFmpeg is
≈ 1.5–1.6× faster, down from 2.5× in the previous round.

**10-bit:** `rust_h265` is faster than single-threaded FFmpeg: 1.29× on
the `safe` preset and 1.52× on `medium`, at 464–607 Mpx/s (224–293 fps)
against FFmpeg's 305–469 Mpx/s. Our 10-bit throughput is 76% of our 8-bit
throughput, while FFmpeg's drops to 39% (see Interpretation).

All six fixtures are byte-exact against FFmpeg.

### Threads

`Decoder::with_threads(n)` against FFmpeg 9.0.2 with `-threads N`, frames
per second (`bench_decode` best of 3; FFmpeg `-benchmark` rtime best of
3). The machine was loaded during this run (load average 5-5.6, from
other work), so the threaded figures are conservative: an earlier run at
load ~2.4 measured 674 fps for `medium` and 530 for `10bit_medium` at 4
threads.

| Fixture | WPP | ours, 1 | 2 | 4 | 8 | FFmpeg, 1 | 4 | default |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|
| `bbb_1080p_5s_safe` | no | 375 | 387 | 390 | 387 | 563 | 1500 | 2000 |
| `bbb_720p_10s_safe` | no | 684 | 718 | 705 | 679 | 1071 | 2927 | 3810 |
| `bbb_1080p_5s_medium` | yes | 281 | 466 | 530 | 482 | 393 | 822 | 1122 |
| `bbb_1080p_5s_slow` | yes | 271 | 446 | 536 | 497 | 440 | 857 | 1111 |
| `bbb_1080p_5s_10bit_safe` | no | 279 | 302 | 292 | 305 | 226 | 645 | 938 |
| `bbb_1080p_5s_10bit_medium` | yes | 225 | 366 | 430 | 418 | 148 | 404 | 550 |

- **WPP streams** (x265's default; the `safe` fixtures disable it): CTB
  rows decode in parallel, 1.9-2.5x with 4 threads (2.2-2.5x on a quieter
  machine). The machine has 4 performance cores; 8 threads is slower than
  4. 10-bit `medium` with 4 threads is faster than FFmpeg with 4.
- **Streams without WPP** only parallelise the loop filters (a few
  percent). FFmpeg's frame threading parallelises any stream, which is
  why it is far ahead on the `safe` fixtures; frame threading is the next
  step for those (see Priorities).

### What changed in this round

The previous round measured 375–486 Mpx/s on 8-bit and 328–414 Mpx/s on
10-bit. This round is 1.41–1.64× faster:

| Fixture | previous round | ours now | speedup |
|---|---:|---:|---:|
| `bbb_1080p_5s_safe` | 486 Mpx/s | 795 Mpx/s | 1.64× |
| `bbb_720p_10s_safe` | 399 Mpx/s | 640 Mpx/s | 1.60× |
| `bbb_1080p_5s_medium` | 375 Mpx/s | 585 Mpx/s | 1.56× |
| `bbb_1080p_5s_slow` | 377 Mpx/s | 578 Mpx/s | 1.53× |
| `bbb_1080p_5s_10bit_safe` | 414 Mpx/s | 607 Mpx/s | 1.47× |
| `bbb_1080p_5s_10bit_medium` | 328 Mpx/s | 464 Mpx/s | 1.41× |

The changes, in order of landing. Each gain is CPU cycles against the
commit before it, on the 1080p fixtures:

1. **Picture buffer pool** (1–2%). A new picture reuses the tables and
   MC scratch of the previous one and the planes and motion field of a
   picture that left the DPB, instead of allocating and zero-filling them.
2. **No per-block coefficient allocation** (about 1%). One coefficient
   buffer is lent to each transform block in turn, and the luma inverse
   transform runs in place.
3. **CABAC table fusion** (about 0.5%). One table lookup per decision bin
   instead of two dependent ones, and renormalisation with count-leading-
   zeros.
4. **MC kernels in 8-sample chunks, 16-bit intermediates** (11–16%). The
   interpolation filters accumulate tap by tap in fixed 8-sample chunks
   held in registers instead of zero-filling a 64-entry accumulator array
   per row; bi-prediction and weighted-prediction intermediates are 16-bit,
   as in HM and FFmpeg.
5. **SAO edge offset vectorised** (1–4%, then another 6–7% on SAO-heavy
   `medium` / `slow`). The offset is chosen with comparisons and selects
   instead of a table lookup, in 16-bit lanes, and a CTB row's first and
   last samples join the vectorised run. Intra reference samples moved to
   stack arrays.
6. **Bi-prediction average in saturating 16-bit lanes** (1–2% on
   B-frame-heavy 8-bit content).
7. **Zero-skipping inverse DCT** (14–18% on `safe`, 5–9% on `medium` /
   `slow`). The 8–32-point transforms skip the all-zero rows and columns
   outside the coded coefficient groups (like FFmpeg's `col_limit`) and
   run as even/odd matrix products that vectorise. Inverse transform time
   fell from about 10% to 2–4% of decode time.
8. **Deblocking boundary strength** (2–5%). Coding units with a single
   prediction unit skip the internal-edge pass, and the edge loops skip
   runs of zero strengths.

### What changed in the previous round

The round before that measured 146–223 Mpx/s for `rust_h265`. Six changes to
motion compensation (MC) and picture setup account for a 2.1–2.45×
speedup:

| Fixture | round before | previous round | speedup |
|---|---:|---:|---:|
| `bbb_1080p_5s_safe` | 223 Mpx/s | 486 Mpx/s | 2.18× |
| `bbb_720p_10s_safe` | 191 Mpx/s | 399 Mpx/s | 2.09× |
| `bbb_1080p_5s_medium` | 153 Mpx/s | 375 Mpx/s | 2.45× |
| `bbb_1080p_5s_slow` | 157 Mpx/s | 377 Mpx/s | 2.40× |
| `bbb_1080p_5s_10bit_safe` | 199 Mpx/s | 414 Mpx/s | 2.08× |
| `bbb_1080p_5s_10bit_medium` | 146 Mpx/s | 328 Mpx/s | 2.25× |

In order of landing, with each step's gain over the one before it:

1. **MC scratch buffers reused** (+11–21%). The MC path zero-filled its
   intermediates on every prediction block, about 100 KB per bi-predicted
   block once 4:4:4 support had enlarged the chroma buffers. They now live
   in a per-picture `McScratch` that every block reuses.
2. **Reference windows read directly** (+20–48%). Every MC filter tap used
   to go through a helper that clamps both coordinates and bounds-checks
   the index: 8 clamps per output sample for the 8-tap luma filter.
   `ref_window` now checks once per block whether the filter window lies
   inside the reference picture. Interior blocks read the plane directly;
   edge blocks read a padded copy built with the same clamping helper, so
   results are unchanged.
3. **Z-scan table built once** (+2–4%). `PictureState::new` rebuilt the
   min-TB Z-scan table, about 130,000 entries at 1080p, for every picture.
   It depends only on picture geometry, so it is now built once and shared.
4. **Vectorisable MC kernels** (+19–48%). The four filter entry points
   share one core, `mc_block` / `interp_block`, whose inner loops are
   tap-major multiply-accumulates over contiguous row slices. LLVM compiles
   them to NEON (SSE/AVX on x86) with no `unsafe`. The bi-prediction and
   weighted-prediction loops write whole rows instead of indexing the
   plane per pixel.
5. **i16 accumulation for 8-bit** (+2–4% on 8-bit). One filter pass over
   8-bit samples is bounded by 112 × 255, so it fits in i16, doubling the
   lanes per instruction. 10-bit keeps i32.
6. **No row copy for intermediates** (+3–9%). Intermediate rows are written
   straight into their destination instead of through a staging buffer.
7. **Deblocking boundary strength** (+3–6%, measured after the tables above
   were taken). Inside one prediction unit the strength is computed on every
   8×8 grid line between identical motion fields, which always gives 0; that
   case is now an inline struct comparison. The luma edge filter was also
   restructured to load each 4-line segment once and filter the lines as
   lanes, but compiler vectorisation helped it little (see Priorities).

### Comparison with libde265

libde265 (the decoder libheif uses for HEIC) was measured separately with
a small harness against its C API, Homebrew build 1.1.3. On this machine
that build has no NEON kernels, so its MC is scalar C. Before the changes
above libde265 was 1.3–1.6× faster than `rust_h265` single-threaded. With
them, `rust_h265` is about 1.6× faster on `bbb_1080p_5s_safe` (239 vs 147
fps in `bench_decode`) and 1.55× faster on `medium` (182 vs 117 fps). The
harness is not in-tree, so treat these as a spot check rather than a
maintained benchmark. It was taken in the previous round and has not been
re-run since.

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
info), grouped by decoder stage. Self time, so each sample counts once;
memory copies are charged to the function that called them.

| Stage | Self time |
|---|---:|
| Motion compensation | 43.0% |
| CU/PU/TU syntax + MV derivation | 18.1% |
| Deblocking incl. boundary strength | 14.5% |
| Residual coding + CABAC | 9.7% |
| Picture setup, allocation and output copies | 5.3% |
| Intra prediction | 5.2% |
| Inverse transform | 3.9% |

Top functions: `motion_compensation_pu` 17.7% (the inlined direct-output
luma filter and the uni-prediction rounding), `mc_luma_i16` 12.9%,
`decode_residual_coding` 9.1%, `filter_luma_edge` 8.2%, `mc_chroma` 6.9%,
`decode_prediction_unit` 5.2%, `decode_transform_tree` 5.1%. The `safe`
preset has SAO off; on `medium` / `slow`, SAO edge offset is about 6%.
MC, SAO and the inverse transform are compiler-vectorised; the luma
deblocking filter is still mostly scalar decision code.

## Historical profile: `bbb_1080p_5s_safe` (before the round before last)

_The figures below predate both rounds above._

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
changed in the previous round")._

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

### 8-bit gap (about 1.5×)

The single-threaded gap on 8-bit 1080p content is dominated by:

1. **Kernel quality.** Motion compensation is 43% of decode time. Its
   loops are compiler-vectorised, but FFmpeg's hand-written NEON still
   does far more per instruction: it uses the Armv8.6 `usdot` dot-product
   instruction for its filters and a separately unrolled kernel per block
   width. The luma deblocking filter (8%) is mostly scalar here and NEON
   with in-register transposes in FFmpeg.

2. **Serial syntax work.** CU/PU/TU parsing, MV derivation, residual
   coding and CABAC are about 28% of our time. FFmpeg's C is leaner here
   but this part does not vectorise in either decoder.

3. **Threading.** FFmpeg's `-threads 0` enables frame-parallel decode across
   all cores. `rust_h265` is single-threaded. The `ff-t1 → ff-tN` delta is
   roughly 2.5–4× on these fixtures.

### 10-bit: faster than FFmpeg

We are faster on 10-bit because:

1. **SIMD advantage halved.** 10-bit samples use u16 (2 bytes), so a NEON
   register holds half as many samples. FFmpeg's NEON kernels exist for
   10-bit but their throughput advantage over vectorised Rust is smaller.

2. **Our code scales nearly linearly.** 8-bit filters accumulate in i16;
   10-bit MC intermediates are i16 too, and everything outside MC uses the
   same arithmetic for both. Our 10-bit safe preset runs at 76% of 8-bit
   speed (607/795), while FFmpeg drops to 39% (469/1214).

### Allocation and bounds-check overhead

Second-order effects now. Picture setup and allocation are ~4% (fresh
Y/U/V planes and bookkeeping tables per picture; the Z-scan table is
shared). MC scratch buffers are allocated once per picture and reused.
Bounds checks are present throughout (the shared picture tables in
`src/shared.rs`, the only `unsafe` code, check bounds too) but are not the
bottleneck per the profile.

## Priorities

All six real-world fixtures decode byte-exact. Picture-buffer pooling and
inverse-transform vectorisation, the previous priorities 2 and 4, are done.
Remaining focus, ordered by expected return:

1. **Hand-written NEON for MC** (~43%, already vectorised). Closing the
   remaining gap to FFmpeg likely needs explicit intrinsics: the `usdot`
   dot-product filters with runtime CPU detection and a plain-NEON
   fallback, and per-width kernels. This needs `unsafe` beyond the one
   module that has it now (`src/shared.rs`), kept as narrow.

2. **Luma deblocking filter** (~8%). Compiler vectorisation was tried
   twice (4-line lanes, then direction-specialised code) and did not pay
   off: each 4-line segment makes its own filter decisions, and vertical
   edges need a transpose. FFmpeg filters 8 lines at a time with NEON and
   in-register transposes; matching it likely needs intrinsics.

3. **Syntax and CABAC** (~28% combined). Serial work with no single hot
   spot: a 64-bit CABAC refill, batched bypass bins, cheaper neighbour
   availability and leaner per-CU bookkeeping are each worth 1–2%.

4. **Frame threading** (wall-clock). WPP row parallelism is done (see
   Threads); streams without WPP still decode on one thread apart from the
   loop filters. Frame-level parallelism (a picture decodes while its
   references are still finishing, waiting on their row progress) would
   cover every stream and combine with WPP. It builds on the same shared
   tables and progress counters.

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
