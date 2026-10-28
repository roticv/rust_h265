# rust_h265

A pure Rust H.265 / HEVC video decoder.

> **Status:** functional. Main and Main 10 profile HEVC (8-bit and 10-bit
> 4:2:0) plus 4:4:4 (8-bit and 10-bit, as used by HEIF/HEIC still images)
> decodes end-to-end with CTU 16/32/64, I/P/B slices (including
> hierarchical B), WPP, tiles, dependent slice segments, SAO, deblocking,
> AQ (`cu_qp_delta`), scaling lists, sign-data hiding, weighted prediction,
> transform skip, transquant bypass, constrained intra prediction, and PCM.
> Byte-exact against FFmpeg on every in-tree fixture plus
> real 1080p Big Buck Bunny at x265 presets `ultrafast` / `medium` / `slow`
> in both 8-bit and 10-bit. No threading or hand-written SIMD yet — ~2.5×
> slower than single-threaded FFmpeg on 8-bit content, roughly at parity
> on 10-bit; see
> [`BENCHMARK.md`](BENCHMARK.md).

While working on `rust_media` it became clear that there isn't a sufficiently good open source software HEVC decoder that ships as a standalone library. FFmpeg has one, but it isn't split out. Most devices have hardware HEVC decoders, but a portable software fallback is still useful when you want one binary that runs anywhere.

A pure Rust H.264 decoder ([`rust_h264`](https://github.com/roticv/rust_h264)) already exists in this style. HEVC is a substantively different codec, not an extension of H.264, so this is a fresh implementation rather than a port.

## Design

- **Input:** Both **Annex B** (start-code delimited, `.h265` raw files) and **HVCC** (length-prefixed, MP4/MKV containers) formats via `parse_annex_b()` and `parse_hvcc()`.
- **Streaming:** `Decoder::decode_nal(&NalUnit) -> Result<Option<Frame>, DecodeError>` plus `flush()`. NAL units are fed incrementally. Frames come out in **decoding order** by default, or in **display order** with `Decoder::new().with_display_order()` and `next_output()` (see [frame ordering](#important-frame-ordering)).
- **Performance:** The decoder aims to be fast, with FFmpeg's software HEVC decoder as the target benchmark. Current gap vs single-threaded FFmpeg: ~2.5× on 8-bit, roughly parity on 10-bit real 1080p content. Motion compensation is compiler-vectorised; there are no hand-written NEON / SSE kernels yet.
- **Multi-bit-depth:** 8-bit and 10-bit (Main / Main 10 profile) via a generic `Pixel` trait. 12-bit infrastructure is in place but untested. Pixel planes are `PixelData::U8(Vec<u8>)` or `PixelData::U16(Vec<u16>)` — check `frame.bit_depth` to determine which.
- **Chroma formats:** 4:2:0 and 4:4:4, intra and inter. Check `frame.chroma_format_idc` (1 = 4:2:0, 3 = 4:4:4) to size the `u`/`v` planes: half width and height for 4:2:0, full size for 4:4:4. 4:2:2 and monochrome streams are rejected with `DecodeError::Unsupported`.
- **Resource limits:** `Decoder::new()` rejects any SPS whose picture exceeds 35,651,584 luma samples (the Level 6.2 maximum) or an estimated 512 MB of per-picture state, returning `DecodeError::ResourceLimit`. Use `Decoder::with_limits(DecodeLimits { .. })` to change either bound.
- **Pure Rust, no `unsafe`** in the current codebase. `unsafe` will be reserved for SIMD paths once they land.

## Usage

The public API mirrors `rust_h264`:

```rust
use rust_h265::{Decoder, parse_annex_b};

let h265_data = std::fs::read("input.h265").unwrap();
let nals = parse_annex_b(&h265_data);
let mut decoder = Decoder::new();

for nal in &nals {
    match decoder.decode_nal(nal) {
        Ok(Some(frame)) => {
            // `frame` is a decoded YUV picture:
            //   frame.y, frame.u, frame.v  — PixelData (U8 or U16)
            //   frame.width, frame.height  — luma dimensions
            //   frame.bit_depth            — 8 or 10
            //   frame.chroma_format_idc    — 1 = 4:2:0, 3 = 4:4:4
            //   frame.pic_order_cnt        — display order index
            //
            // Access pixels:
            //   frame.y.as_u8()  -> Option<&[u8]>   (8-bit)
            //   frame.y.as_u16() -> Option<&[u16]>  (10-bit)
        }
        Ok(None) => {} // NAL consumed, no frame ready yet (e.g. VPS/SPS/PPS)
        Err(e) => eprintln!("decode error: {:?}", e),
    }
}
// Flush the last buffered frame
if let Some(frame) = decoder.flush() {
    // handle final frame
}
```

### HVCC (MP4/MKV) input

For container-demuxed content (length-prefixed NALs instead of Annex B start codes):

```rust
use rust_h265::{Decoder, parse_hvcc};

let mut decoder = Decoder::new();

// length_size comes from HEVCDecoderConfigurationRecord (typically 4)
let length_size = 4u8;

// Feed each demuxed packet (already length-prefixed)
for packet in demuxer.video_packets() {
    for nal in parse_hvcc(&packet.data, length_size) {
        decoder.decode_nal(&nal)?;
    }
}
```

### Important: frame ordering

**By default `decode_nal` returns frames in decoding order, not display
order.** With B-frames, pictures are decoded before earlier pictures they
are shown after.

For display order, create the decoder with `with_display_order()`. It runs
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

In the default decoding-order mode every decoded picture is returned, and
the caller sorts by `pic_order_cnt` within each coded video sequence (POC
restarts at every IDR or BLA picture, and at a CRA after an end-of-sequence
NAL). `examples/dump_frames.rs --decode-order` shows how. That mode suits
callers that pair each frame with the container packet it came from.

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
cargo run --release --example play -- input.h265 [--fps 30] [--loop]

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
