# Compression benchmark

Measures compression and decompression throughput, plus resulting file sizes, for Vortex
versus Parquet (and optionally Lance) across a range of datasets: NYC taxi data, several
[Public BI](https://github.com/cwida/public_bi_benchmark) tables (Arade, Bimbo,
CMSprovider, Euro2016, Food, HashTags), TPC-H `l_comment` variants, and synthetic nested
data. This is the workload behind the `Compression` PR comment.

See [`src/main.rs`](./src/main.rs) for the dataset list and CLI flags (`--formats`,
`--datasets`, `--ops compress,decompress`).

## Running locally

```bash
cargo run -p compress-bench --profile release_debug
```

## GPU decompression

`--gpu-decompress` is opt-in, requires the `cuda` feature, and restricts the suite to the
GPU dataset list in `src/main.rs`. It measures decompression only, for two backends:

- **Vortex** — the file is written with CUDA-compatible BtrBlocks encodings only
  (`only_cuda_compatible`) and a CUDA flat layout, then decoded on the device all the way to
  canonical arrays.
- **Parquet** — the file is rewritten with GPU-friendly writer settings (see below) and every
  page body is decompressed on the device with nvCOMP's batched Snappy or Zstd entrypoints,
  the same decomposition cuDF's Parquet reader uses: column chunks are staged on the device,
  then all pages go through a single batched launch.

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress

# pick the Parquet page codec (default: snappy)
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --gpu-parquet-codec zstd
```

On Linux both backends read through the same pinned, direct-I/O (`O_DIRECT`) reader, so
repeated iterations measure storage bandwidth rather than page-cache hits.

### What the Parquet GPU number does and does not include

Included: column chunk I/O, the host-to-device transfer, and the batched codec launch.

Not included: page *decoding* — the dictionary, RLE and plain decoders that turn a
decompressed page into an Arrow array — because there is no Rust GPU Parquet page decoder to
call. The Vortex backend it is compared against decodes all the way to canonical arrays, so
the Parquet figure is an upper bound on what a full GPU Parquet reader could reach, and the
`vortex:parquet-<codec> gpu ratio decompress time` metric is biased in Parquet's favour.

Walking the per-page Thrift headers also happens on the host, once per file, outside the
measurement; a real GPU Parquet reader decodes page headers on the device.

### GPU-friendly Parquet writer settings

Set in `src/parquet_pages.rs`:

| Setting | Value | Why |
| --- | --- | --- |
| writer version | `PARQUET_1_0` | v1 pages compress the whole page body, which is the unit nvCOMP decompresses. v2 pages put uncompressed levels ahead of the compressed values in the same body. |
| compression | Snappy (default) or Zstd | The two Parquet codecs nvCOMP implements. Snappy has the higher device throughput and is the Parquet default. |
| dictionary | enabled | Keeps the decompressed payload small; the encoding GPU Parquet readers decode fastest. |
| data page size | 1 MiB | Large enough to amortize per-chunk setup, small enough to keep every SM fed. Matches the page size cuDF targets. |
| data page row limit | 1,000,000 | The 20k-row default caps narrow columns' pages far below 1 MiB. |
| statistics | chunk-level | Page statistics only inflate the headers that have to be walked on the host. |

### Correctness

`--gpu-verify` cross-checks device output against the CPU decoders on every iteration:

- Parquet: each decompressed page is copied back and compared byte-for-byte against the host
  Snappy/Zstd output for the same compressed bytes.
- Vortex: each GPU-decoded field is copied back and compared against the same field decoded
  on the CPU, through Arrow with a pinned target type.

Verification runs inline, so timings from a verifying run are not comparable to a plain one —
run it as its own pass:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --gpu-verify --iterations 1
```

Independently of `--gpu-verify`, every Parquet GPU run checks nvCOMP's per-page status and
output-size arrays after the measurement, because a batched launch reports per-page failures
in device memory rather than by failing the launch.

Page-header scanning is covered by CPU-only unit tests in `src/parquet_pages.rs`, which
assert the located page bodies decompress to exactly the bytes `parquet`'s own page reader
produces.
