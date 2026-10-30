# rust_h265

A pure Rust H.265 / HEVC video decoder.

> **Status:** complete for the Main and Main 10 profiles. Passes all 147
> vectors of the JCT-VC HEVC_v1 conformance suite, and is byte-exact
> against FFmpeg on every in-tree fixture and on real 1080p Big Buck Bunny
> encodes (x265 `ultrafast` / `medium` / `slow`, 8-bit and 10-bit).
>
> - **Formats:** 4:2:0 and 4:4:4 (as used by HEIF/HEIC images), 8-bit and
>   10-bit, plus 12-bit for the range-extension profiles.
> - **Tools:** CTU 16/32/64, I/P/B slices including hierarchical B and
>   open-GOP CRA, WPP, tiles, dependent slice segments, SAO, deblocking,
>   `cu_qp_delta`, scaling lists, sign-data hiding, weighted prediction,
>   transform skip, transquant bypass, constrained intra prediction, PCM.
> - **Output:** display order following the spec's output process
>   (`Decoder::with_display_order`, recommended), or decoding order (the
>   default).
> - **Robustness:** damaged or missing slices cost the affected pictures, not
>   the rest of the stream. Fuzzed.
> - **Range extensions:** every range-extension coding tool, in 4:2:0 and
>   4:4:4 up to 12 bits. Passes the 34 JCT-VC-RExt conformance vectors in
>   those formats, including extended precision and CABAC bypass
>   alignment, which FFmpeg does not implement.
> - **Not supported:** 4:2:2, monochrome, bit depths above 12, and screen
>   content coding. These streams are rejected with
>   `DecodeError::Unsupported` rather than decoded wrongly.
> - **Threads:** `Decoder::with_threads(n)`: frame threading (pictures
>   overlap, each waiting only for the reference rows it reads), plus
>   wavefront row decoding for a picture with nothing to overlap with.
>   2.5–3.8× faster with 4 threads on the 1080p benchmark fixtures; the
>   output is identical for every thread count.
> - **Speed:** no hand-written SIMD yet; single-threaded, about 1.5–1.6× slower
>   than single-threaded FFmpeg on 8-bit content and 1.3–1.5× faster on
>   10-bit. See [`BENCHMARK.md`](BENCHMARK.md).

While working on `rust_media` it became clear that there isn't a sufficiently good open source software HEVC decoder that ships as a standalone library. FFmpeg has one, but it isn't split out. Most devices have hardware HEVC decoders, but a portable software fallback is still useful when you want one binary that runs anywhere.

A pure Rust H.264 decoder ([`rust_h264`](https://github.com/roticv/rust_h264)) already exists in this style. HEVC is a substantively different codec, not an extension of H.264, so this is a fresh implementation rather than a port.

## Design

- **Input:** Both **Annex B** (start-code delimited, `.h265` raw files) and **HVCC** (length-prefixed, MP4/MKV containers) formats via `parse_annex_b()` and `parse_hvcc()`.
- **Streaming:** `Decoder::decode_nal(&NalUnit) -> Result<Option<Frame>, DecodeError>` plus `flush()`. NAL units are fed incrementally. Use **display order**, `Decoder::new().with_display_order()` with `next_output()`, unless you need decoding order, which is the default for compatibility (see [frame ordering](#important-frame-ordering)).
- **Performance:** The decoder aims to be fast, with FFmpeg's software HEVC decoder as the target benchmark. Against single-threaded FFmpeg on real 1080p content: about 1.5–1.6× slower on 8-bit and 1.3–1.5× faster on 10-bit (see [`BENCHMARK.md`](BENCHMARK.md)). Motion compensation, SAO and the inverse transform are compiler-vectorised; there are no hand-written NEON / SSE kernels yet.
- **Multi-bit-depth:** 8-bit and 10-bit (Main / Main 10 profile) via a generic `Pixel` trait. 12-bit decodes (the 12-bit range-extensions conformance vectors pass); deeper streams are rejected with `DecodeError::Unsupported`. Pixel planes are `PixelData::U8(Vec<u8>)` or `PixelData::U16(Vec<u16>)` — check `frame.bit_depth` to determine which.
- **Chroma formats:** 4:2:0 and 4:4:4, intra and inter. Check `frame.chroma_format_idc` (1 = 4:2:0, 3 = 4:4:4) to size the `u`/`v` planes: half width and height for 4:2:0, full size for 4:4:4. 4:2:2 and monochrome streams are rejected with `DecodeError::Unsupported`.
- **Range extensions:** all range-extension coding tools are implemented: cross-component prediction, implicit and explicit RDPCM, transform-skip rotation and contexts, transform skip up to 32×32, extended precision processing, persistent Rice adaptation, CABAC bypass alignment, chroma QP offset lists, SAO offset scaling, disabled intra smoothing and high-precision weighted-prediction offsets, plus WPP combined with tiles. Screen content coding is rejected with `DecodeError::Unsupported`.
- **Resource limits:** `Decoder::new()` rejects any SPS whose picture exceeds 35,651,584 luma samples (the Level 6.2 maximum) or an estimated 512 MB of per-picture state, returning `DecodeError::ResourceLimit`. Use `Decoder::with_limits(DecodeLimits { .. })` to change either bound.
- **Pure Rust, `unsafe` in one module.** Multithreaded decoding needs threads reading parts of picture tables another thread is writing, which safe Rust cannot express; `src/shared.rs` (one small module) implements those shared tables, read by one thread while another writes other parts of them (wavefront rows, and pictures referenced while they decode). Its rules are documented there, every table access is checked against them in test builds, and the rest of the crate is safe Rust. Any future SIMD code will follow the same pattern.

## Usage

The public API mirrors `rust_h264`:

```rust
use rust_h265::{Decoder, parse_annex_b};

let h265_data = std::fs::read("input.h265").unwrap();
// Display order (recommended): frames come out in the order to show them.
// Decoding on all cores gives identical frames, sooner.
let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
let mut decoder = Decoder::new().with_display_order().with_threads(threads);

for nal in parse_annex_b(&h265_data) {
    if let Err(e) = decoder.decode_nal(&nal) {
        eprintln!("decode error: {:?}", e);
    }
    // With several threads a frame can come out a few NAL units after its
    // last slice; `next_output` returns what is ready.
    while let Some(frame) = decoder.next_output() {
        // `frame` is a decoded YUV picture:
        //   frame.y, frame.u, frame.v  — PixelData (U8 or U16)
        //   frame.width, frame.height  — luma dimensions
        //   frame.bit_depth            — 8, 10 or 12
        //   frame.chroma_format_idc    — 1 = 4:2:0, 3 = 4:4:4
        //   frame.pic_order_cnt        — display order index
        //
        // Access pixels:
        //   frame.y.as_u8()  -> Option<&[u8]>   (8-bit)
        //   frame.y.as_u16() -> Option<&[u16]>  (10- and 12-bit)
    }
}
// At the end of the stream, take the pictures still waiting.
while let Some(frame) = decoder.flush() {
    // handle frame
}
// Errors in slice data found after the last NAL unit (with threads).
while let Some(e) = decoder.next_error() {
    eprintln!("decode error: {:?}", e);
}
```

### HVCC (MP4/MKV) input

For container-demuxed content (length-prefixed NALs instead of Annex B start codes):

```rust
use rust_h265::{Decoder, parse_hvcc};

let mut decoder = Decoder::new().with_display_order();

// length_size comes from HEVCDecoderConfigurationRecord (typically 4)
let length_size = 4u8;

// Feed each demuxed packet (already length-prefixed)
for packet in demuxer.video_packets() {
    for nal in parse_hvcc(&packet.data, length_size) {
        decoder.decode_nal(&nal)?;
    }
    while let Some(frame) = decoder.next_output() {
        // display `frame`
    }
}
while let Some(frame) = decoder.flush() {
    // display `frame`
}
```

### Important: frame ordering

**By default `decode_nal` returns frames in decoding order, not display
order.** With B-frames, pictures are decoded before earlier pictures they
are shown after.

**Use display order unless you need decoding order.** Create the decoder
with `with_display_order()`. It runs
the spec's output process (C.5.2, "bumping"): frames are released in display
order as soon as the stream's reorder and DPB limits allow, and pictures the
spec never outputs are dropped. In this mode `decode_nal` always returns
`Ok(None)`; take frames with `next_output()`, and call `flush()` until it
returns `None` at the end of the stream:

```rust
use rust_h265::{Decoder, parse_annex_b};

let mut decoder = Decoder::new().with_display_order();
for nal in parse_annex_b(&h265_data) {
    decoder.decode_nal(&nal)?;
    while let Some(frame) = decoder.next_output() {
        // display `frame`
    }
}
while let Some(frame) = decoder.flush() {
    // display `frame`
}
```

The dropped pictures are those with `pic_output_flag = 0`, skipped RASL
pictures, and pictures still waiting when an IRAP picture with
`NoOutputOfPriorPicsFlag = 1` starts a new coded video sequence. This mode
passes all 147 JCT-VC HEVC_v1 conformance vectors, including the ones that
test output order.

Decoding order stays the default so that existing code keeps working. In
that mode every decoded picture is returned, including ones the spec does
not output, and the caller sorts by `pic_order_cnt` within each coded
video sequence (POC restarts at every IDR or BLA picture, and at a CRA
after an end-of-sequence NAL). `examples/dump_frames.rs --decode-order`
shows how. It suits callers that pair each frame with the container packet
it came from; everyone else should use display order, which gets these
rules right for you.

## Differences from H.264

If you're coming from `rust_h264`, the major changes are not API-level — the public surface is essentially the same — but the codec internals are nearly entirely different. A few user-visible differences worth knowing:

- **NAL header is 2 bytes**, not 1. `nal_unit_type` is 6 bits, with VPS=32, SPS=33, PPS=34. IDR pictures are types 19 (`IDR_W_RADL`) and 20 (`IDR_N_LP`), not type 5.
- **VPS** (Video Parameter Set) exists in addition to SPS and PPS.
- **Block structure** is a quad-tree of Coding Tree Units (typically 64×64) instead of fixed 16×16 macroblocks.
- **More intra modes** (35: planar, DC, 33 angular) and larger transforms (up to 32×32, plus 4×4 DST for intra luma).
- **SAO** (Sample Adaptive Offset) is a new in-loop filter on top of deblocking.
- **Tiles, WPP, and slice segments** add picture-level parallelism options absent in H.264.

These do not change how you call the decoder; they change what the decoder has to implement internally.

## Tools

```sh
# Play an H.265 file in a window (press Escape to quit):
cargo run --release --example play -- input.h265 [--fps 30] [--loop] [--threads N]

# Decode to raw YUV in display order (8-bit: yuv420p, 10-bit: yuv420p10le):
cargo run --release --example dump_frames -- input.h265 out.yuv

# Throughput measurement on a single file:
cargo run --release --example bench_decode -- input.h265 --warmup 2 --repeat 10

# Real-world benchmark matrix (downloads Big Buck Bunny, transcodes with
# x265 at several presets in 8-bit and 10-bit, runs both rust_h265 and
# FFmpeg on each):
cargo run --release --example bench_realworld
```

## Testing

```sh
cargo test --release
```

The test suite covers the full feature matrix — 8-bit and 10-bit, 4:2:0 and
4:4:4, CTU 16/32/64, I/P/B slices, open-GOP CRA streams, WPP, tiles,
dependent slices, SAO, deblocking, transform skip, transquant bypass,
constrained intra, PCM, multi-slice, and more. All
in-tree fixtures are byte-exact against FFmpeg; tests that need >1 MB of
reference output (e.g. 1080p) use a SHA-256 hash of the decoded planes. The
`bench_realworld` example covers real 1080p Big Buck Bunny in both 8-bit
and 10-bit and requires `ffmpeg` and `x265` on `$PATH`.

### Conformance

[`tools/conformance/`](tools/conformance/README.md) runs the official
JCT-VC conformance bitstreams (using the suite definitions from
[Fluster](https://github.com/fluendo/fluster)) and compares the MD5 of the
decoded output with the expected value. JCT-VC HEVC_v1 passes 147 / 147.
JCT-VC-RExt passes the 34 vectors in the supported formats (4:2:0 and
4:4:4, up to 12 bits); the other 15 are 4:2:2, monochrome or 16-bit and are
rejected as unsupported.

### Fuzzing

Three fuzz targets using [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
(requires nightly Rust):

```sh
# Full Annex B decode pipeline (seeded with testdata/ fixtures):
cargo +nightly fuzz run decode_annex_b

# NAL header parsing only (lightweight, fast):
cargo +nightly fuzz run parse_nal_headers

# Single synthetic NAL with controlled type/payload:
cargo +nightly fuzz run decode_single_nal
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
