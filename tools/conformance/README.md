# Conformance testing

Runs the official JCT-VC HEVC conformance bitstreams against `rust_h265`
and compares the MD5 of the decoded output with the expected value.

The test vectors and expected MD5s come from the suite definitions of
[Fluster](https://github.com/fluendo/fluster) (`test_suites/h.265/*.json`),
which point at the ITU conformance bitstream archive. Only the JSON is used
as data; Fluster itself is not needed.

## Running

```sh
cargo build --release --example dump_frames
curl -LO https://raw.githubusercontent.com/fluendo/fluster/HEAD/test_suites/h.265/JCT-VC-HEVC_V1.json
python3 -I tools/conformance/run_suite.py JCT-VC-HEVC_V1.json \
    target/release/examples/dump_frames /path/to/workdir [--only NAME ...]
```

The runner downloads each vector's ZIP into `workdir/zips/` (checked against
the suite's checksum), extracts only the named bitstream, decodes it with
`dump_frames` (display order: POC order within each coded video sequence),
and compares the MD5 of the YUV output. Results go to
`workdir/results-<suite>.json`. Outputs of failing vectors are kept in
`workdir/out/` for analysis.

## Status

JCT-VC-HEVC_V1 (147 vectors, Main and Main 10): see the table below, which
is updated as failures are fixed.

| Date | Pass | Notes |
|---|---:|---|
| 2026-10-28 | 95 / 147 | First run. |
| 2026-10-28 | 100 / 147 | Slice header extension skipped; per-slice reference lists; RASL skipped after BLA; reserved VCL NAL types ignored; LSB-only long-term references resolved; long-term MV prediction rules. |
| 2026-10-28 | 121 / 147 | pic_output_flag; weighted bi-prediction and 10-bit offsets; 10-bit QP range; lossless/PCM loop-filter protection and PCM CU edges; per-slice deblocking offsets and disable; raster slice addresses with tiles; tile-start context/QP resets; spec slice-boundary rules for mixed across-slice flags. |
