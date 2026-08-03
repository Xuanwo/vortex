// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU Parquet decompression backend.
//!
//! Parquet's compressed unit is the page, and every page body is an independent block of
//! Snappy or Zstd. That is precisely the batch shape nvCOMP's device decompressors take, and
//! it is how cuDF's Parquet reader gets pages off the CPU: read the column chunks to the
//! device, then decompress every page in one batched launch.
//!
//! What this measures is the decompression stage of a Parquet read — column chunk I/O,
//! host-to-device transfer, and the batched codec launch. Page *decoding* (the dictionary,
//! RLE and plain decoders that turn a decompressed page into an Arrow array) is not
//! included, because there is no Rust GPU Parquet page decoder to call. The Vortex GPU
//! backend it is compared against decodes all the way to canonical arrays, so the Parquet
//! numbers here are an upper bound on what a full GPU Parquet reader could achieve, and the
//! comparison is favourable to Parquet.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::metadata::ParquetMetaDataReader;
use tempfile::NamedTempFile;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::error::vortex_err;
use vortex::io::VortexReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::CudaSessionExt;
use vortex_cuda::PooledFileReadAt;
use vortex_cuda::PooledFileReadAtOptions;
use vortex_cuda::nvcomp::AlignmentRequirements;
use vortex_cuda::nvcomp::snappy;
use vortex_cuda::nvcomp::sys;
use vortex_cuda::nvcomp::sys::nvcompStatus_t;
use vortex_cuda::nvcomp::zstd as nvcomp_zstd;

use crate::parquet_pages::ColumnChunkPages;
use crate::parquet_pages::GpuCodec;
use crate::parquet_pages::gpu_writer_properties;
use crate::parquet_pages::scan_compressed_pages;

/// Parquet compressor whose decompression measurement runs the page codec on the GPU.
pub struct GpuParquetCompressor {
    codec: GpuCodec,
    verify: bool,
}

impl GpuParquetCompressor {
    /// Create a backend that writes pages with `codec` and decompresses them with nvCOMP.
    ///
    /// When `verify` is set, every decompressed page is copied back and compared against the
    /// host codec's output before the measurement is reported.
    pub fn new(codec: GpuCodec, verify: bool) -> Self {
        Self { codec, verify }
    }

    /// Rewrite the source Parquet file with GPU-friendly writer settings.
    fn write_gpu_parquet(&self, parquet_path: &Path) -> Result<(NamedTempFile, u64)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(parquet_path)?)?;
        let schema = Arc::clone(builder.schema());
        let batches: Vec<RecordBatch> = builder.build()?.collect::<Result<Vec<_>, _>>()?;

        let output = NamedTempFile::new()?;
        let mut writer = ArrowWriter::try_new(
            output.reopen()?,
            schema,
            Some(gpu_writer_properties(self.codec)),
        )?;
        for batch in batches {
            writer.write(&batch)?;
        }
        writer.flush()?;
        let size = writer.bytes_written() as u64;
        writer.close()?;
        Ok((output, size))
    }

    fn check_codec(&self, metadata: &ParquetMetaData) -> Result<()> {
        for row_group in metadata.row_groups() {
            for column in row_group.columns() {
                ensure!(
                    self.codec.matches(column.compression()),
                    "column {} was written with {:?}, expected {}",
                    column.column_path(),
                    column.compression(),
                    self.codec.name()
                );
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Compressor for GpuParquetCompressor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    async fn compress(&self, parquet_path: &Path) -> Result<(u64, Duration)> {
        let start = Instant::now();
        let (_file, size) = self.write_gpu_parquet(parquet_path)?;
        Ok((size, start.elapsed()))
    }

    async fn decompress(&self, parquet_path: &Path) -> Result<Duration> {
        let (gpu_file, _) = self.write_gpu_parquet(parquet_path)?;

        let file = File::open(gpu_file.path())?;
        let metadata = ParquetMetaDataReader::new().parse_and_finish(&file)?;
        self.check_codec(&metadata)?;
        drop(file);

        // The page map is a property of the file, so it is resolved once rather than on every
        // iteration. A GPU Parquet reader decodes page headers on the device as part of the
        // read; this benchmark cannot, so the host walk is kept out of the measurement.
        let file_bytes = std::fs::read(gpu_file.path())?;
        let row_groups = scan_compressed_pages(&file_bytes, &metadata)?;
        let plan = DecompressPlan::build(&row_groups, self.codec)?;
        // Only verification needs the host copy afterwards, to decompress the same bytes.
        let file_bytes = self.verify.then_some(file_bytes);

        let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
        let reader = open_reader(gpu_file.path(), &cuda_ctx)?;

        // Row groups are staged and released one at a time so device residency stays bounded
        // by the largest row group rather than by the whole file, matching how the Vortex
        // backend streams one batch at a time.
        let mut elapsed = Duration::ZERO;
        let mut checks = Vec::with_capacity(plan.row_groups.len());
        for row_group in &plan.row_groups {
            let start = Instant::now();
            let device_chunks = stage_row_group(row_group, &reader).await?;
            let output =
                decompress_pages(row_group, &device_chunks, self.codec, &mut cuda_ctx).await?;
            cuda_ctx.synchronize_stream()?;
            elapsed += start.elapsed();

            let DecompressOutput {
                output,
                actual_sizes,
                statuses,
            } = output;
            drop(device_chunks);

            if self.verify {
                // Verified inline, since holding every row group's decompressed output would
                // defeat the point of streaming them. This is why a verifying run's timings
                // are not comparable.
                let file_bytes = file_bytes
                    .as_deref()
                    .ok_or_else(|| anyhow!("verification requires the host file bytes"))?;
                verify_against_host(row_group, file_bytes, self.codec, output).await?;
            }
            checks.push((actual_sizes, statuses));
        }

        // Checked outside the measurement, but on every iteration: a batched launch reports
        // per-page failures in device memory rather than by failing the launch, so without
        // this a broken run would simply look fast.
        for (row_group, (actual_sizes, statuses)) in plan.row_groups.iter().zip(checks) {
            check_statuses(row_group, actual_sizes, statuses).await?;
        }

        Ok(elapsed)
    }
}

/// Stages a row group's column chunks on the device, bounding concurrent pinned staging.
async fn stage_row_group(
    row_group: &RowGroupPlan,
    reader: &PooledFileReadAt,
) -> Result<Vec<BufferHandle>> {
    let reads = row_group
        .chunks
        .iter()
        .map(|chunk| reader.read_at(chunk.offset, chunk.len, Alignment::of::<u8>()))
        .collect::<Vec<_>>();

    stream::iter(reads)
        .buffered(STAGING_CONCURRENCY)
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| anyhow!("failed to stage Parquet column chunks on the device: {e}"))
}

/// Column chunks staged on the device at once. Matches `vortex-cuda`'s file read concurrency.
const STAGING_CONCURRENCY: usize = 32;

/// Opens the GPU Parquet file through the same pinned, direct-I/O reader the Vortex GPU
/// backend uses, so both formats measure storage bandwidth rather than page-cache hits.
fn open_reader(path: &Path, cuda_ctx: &CudaExecutionCtx) -> Result<PooledFileReadAt> {
    let pool = Arc::clone(SESSION.cuda_session().pinned_buffer_pool());
    let options = PooledFileReadAtOptions::default();
    #[cfg(target_os = "linux")]
    let options = options.with_direct_io();

    Ok(PooledFileReadAt::open_with_options(
        path,
        SESSION.handle(),
        pool,
        cuda_ctx.stream().clone(),
        options,
    )?)
}

/// A page to decompress, addressed relative to its column chunk's device buffer.
struct PlannedPage {
    chunk: usize,
    offset_in_chunk: usize,
    compressed_len: usize,
    uncompressed_len: usize,
    output_offset: usize,
}

/// The byte range of a column chunk to stage on the device.
struct PlannedChunk {
    offset: u64,
    len: usize,
}

/// Everything needed to issue one batched nvCOMP launch over a row group's pages.
struct RowGroupPlan {
    chunks: Vec<PlannedChunk>,
    pages: Vec<PlannedPage>,
    output_len: usize,
    max_uncompressed: usize,
}

/// The per-row-group work for one file.
struct DecompressPlan {
    row_groups: Vec<RowGroupPlan>,
}

impl DecompressPlan {
    fn build(row_groups: &[Vec<ColumnChunkPages>], codec: GpuCodec) -> Result<Self> {
        let alignment = decompress_alignments(codec)?;
        ensure!(
            alignment.output.is_power_of_two(),
            "nvcomp reported a non-power-of-two output alignment of {}",
            alignment.output
        );

        let row_groups = row_groups
            .iter()
            .map(|chunks| RowGroupPlan::build(chunks, codec, alignment))
            .collect::<Result<Vec<_>>>()?;

        ensure!(
            row_groups.iter().any(|plan| !plan.pages.is_empty()),
            "Parquet file contains no compressed pages"
        );

        Ok(Self { row_groups })
    }
}

impl RowGroupPlan {
    fn build(
        chunks: &[ColumnChunkPages],
        codec: GpuCodec,
        alignment: AlignmentRequirements,
    ) -> Result<Self> {
        let mut planned_chunks = Vec::with_capacity(chunks.len());
        let mut pages = Vec::new();
        let mut output_len = 0usize;
        let mut max_uncompressed = 0usize;

        for (index, chunk) in chunks.iter().enumerate() {
            planned_chunks.push(PlannedChunk {
                offset: chunk.offset,
                len: chunk.len,
            });

            for page in &chunk.pages {
                let offset_in_chunk = usize::try_from(
                    u64::try_from(page.offset)?
                        .checked_sub(chunk.offset)
                        .ok_or_else(|| anyhow!("page offset precedes its column chunk"))?,
                )?;
                // Pages are decompressed in place from their column chunk's device buffer.
                // CUDA allocations are at least 256-byte aligned, so a page's device address
                // meets nvcomp's requirement exactly when its chunk-relative offset does.
                ensure!(
                    offset_in_chunk.is_multiple_of(alignment.input),
                    "page at file offset {} sits {offset_in_chunk} bytes into its column chunk, \
                     which does not meet nvcomp's {} byte input alignment for {}; \
                     use --gpu-parquet-codec snappy",
                    page.offset,
                    alignment.input,
                    codec.name()
                );

                let output_offset = output_len.next_multiple_of(alignment.output);
                output_len = output_offset + page.uncompressed_len;
                max_uncompressed = max_uncompressed.max(page.uncompressed_len);

                pages.push(PlannedPage {
                    chunk: index,
                    offset_in_chunk,
                    compressed_len: page.compressed_len,
                    uncompressed_len: page.uncompressed_len,
                    output_offset,
                });
            }
        }

        Ok(Self {
            chunks: planned_chunks,
            pages,
            output_len,
            max_uncompressed,
        })
    }
}

fn decompress_alignments(codec: GpuCodec) -> Result<AlignmentRequirements> {
    match codec {
        GpuCodec::Snappy => {
            snappy::decompress_alignment_requirements(snappy::SnappyDecompressOpts::default())
        }
        GpuCodec::Zstd => nvcomp_zstd::decompress_alignment_requirements(
            nvcomp_zstd::ZstdDecompressOpts::default(),
        ),
    }
    .map_err(|e| anyhow!("nvcomp alignment query failed: {e}"))
}

/// Device-side outputs of a batched decompression launch.
struct DecompressOutput {
    output: CudaSlice<u8>,
    actual_sizes: CudaSlice<usize>,
    statuses: CudaSlice<nvcompStatus_t>,
}

/// Enqueues the batched decompression of every page in `plan` onto the context's stream.
async fn decompress_pages(
    plan: &RowGroupPlan,
    device_chunks: &[BufferHandle],
    codec: GpuCodec,
    ctx: &mut CudaExecutionCtx,
) -> Result<DecompressOutput> {
    let num_pages = plan.pages.len();

    let temp_size = match codec {
        GpuCodec::Snappy => {
            snappy::get_decompress_temp_size(num_pages, plan.max_uncompressed, plan.output_len)
        }
        GpuCodec::Zstd => {
            nvcomp_zstd::get_decompress_temp_size(num_pages, plan.max_uncompressed, plan.output_len)
        }
    }
    .map_err(|e| anyhow!("nvcomp temp size query failed: {e}"))?;

    let chunk_bases = device_chunks
        .iter()
        .map(|handle| handle.cuda_device_ptr())
        .collect::<Result<Vec<_>, _>>()?;

    let mut output = ctx.device_alloc::<u8>(plan.output_len)?;
    // Only the allocation address is needed to build the output pointer table; the device
    // write itself is tracked by the guard taken around the launch below.
    let output_base = {
        let (base, _) = output.device_ptr(ctx.stream());
        base
    };

    let mut compressed_ptrs = Vec::with_capacity(num_pages);
    let mut compressed_sizes = Vec::with_capacity(num_pages);
    let mut uncompressed_sizes = Vec::with_capacity(num_pages);
    let mut output_ptrs = Vec::with_capacity(num_pages);
    for page in &plan.pages {
        compressed_ptrs.push(chunk_bases[page.chunk] + page.offset_in_chunk as u64);
        compressed_sizes.push(page.compressed_len);
        uncompressed_sizes.push(page.uncompressed_len);
        output_ptrs.push(output_base + page.output_offset as u64);
    }

    let (compressed_ptrs, compressed_sizes, uncompressed_sizes, output_ptrs) = futures::try_join!(
        ctx.copy_to_device(compressed_ptrs)?,
        ctx.copy_to_device(compressed_sizes)?,
        ctx.copy_to_device(uncompressed_sizes)?,
        ctx.copy_to_device(output_ptrs)?
    )?;

    let mut actual_sizes: CudaSlice<usize> = ctx.device_alloc(num_pages)?;
    let mut statuses: CudaSlice<nvcompStatus_t> = ctx.device_alloc(num_pages)?;
    let mut temp: CudaSlice<u8> = ctx.device_alloc(temp_size)?;

    let stream = ctx.stream();
    let compressed_ptrs_view = compressed_ptrs.cuda_view::<u64>()?;
    let compressed_sizes_view = compressed_sizes.cuda_view::<usize>()?;
    let uncompressed_sizes_view = uncompressed_sizes.cuda_view::<usize>()?;
    let output_ptrs_view = output_ptrs.cuda_view::<u64>()?;

    let (compressed_ptrs_ptr, record_compressed_ptrs) = compressed_ptrs_view.device_ptr(stream);
    let (compressed_sizes_ptr, record_compressed_sizes) = compressed_sizes_view.device_ptr(stream);
    let (uncompressed_sizes_ptr, record_uncompressed_sizes) =
        uncompressed_sizes_view.device_ptr(stream);
    let (output_ptrs_ptr, record_output_ptrs) = output_ptrs_view.device_ptr(stream);
    let (_output_ptr, record_output) = output.device_ptr_mut(stream);
    let (actual_sizes_ptr, record_actual_sizes) = actual_sizes.device_ptr_mut(stream);
    let (statuses_ptr, record_statuses) = statuses.device_ptr_mut(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);

    ctx.launch_external(plan.output_len, || {
        // SAFETY: every pointer is derived from a live device allocation sized by the plan,
        // and each batch metadata array holds exactly `num_pages` entries.
        unsafe {
            match codec {
                GpuCodec::Snappy => snappy::decompress_async(
                    compressed_ptrs_ptr as _,
                    compressed_sizes_ptr as _,
                    uncompressed_sizes_ptr as _,
                    actual_sizes_ptr as _,
                    num_pages,
                    temp_ptr as _,
                    temp_size,
                    output_ptrs_ptr as _,
                    statuses_ptr as _,
                    stream.cu_stream().cast(),
                ),
                GpuCodec::Zstd => nvcomp_zstd::decompress_async(
                    compressed_ptrs_ptr as _,
                    compressed_sizes_ptr as _,
                    uncompressed_sizes_ptr as _,
                    actual_sizes_ptr as _,
                    num_pages,
                    temp_ptr as _,
                    temp_size,
                    output_ptrs_ptr as _,
                    statuses_ptr as _,
                    stream.cu_stream().cast(),
                ),
            }
            .map_err(|e| vortex_err!("nvcomp decompress_async failed: {}", e))
        }
    })?;

    drop((
        record_compressed_ptrs,
        record_compressed_sizes,
        record_uncompressed_sizes,
        record_output_ptrs,
        record_output,
        record_actual_sizes,
        record_statuses,
        record_temp,
    ));
    // The temporary workspace must outlive the launch, which the stream ordering guarantees
    // only while the allocation is alive.
    drop(temp);

    Ok(DecompressOutput {
        output,
        actual_sizes,
        statuses,
    })
}

/// Copies the per-page status and size arrays back and fails on any mismatch.
async fn check_statuses(
    plan: &RowGroupPlan,
    actual_sizes: CudaSlice<usize>,
    statuses: CudaSlice<nvcompStatus_t>,
) -> Result<()> {
    let statuses = CudaDeviceBuffer::new(statuses)
        .copy_to_host(Alignment::of::<nvcompStatus_t>())?
        .await?;
    let actual_sizes = CudaDeviceBuffer::new(actual_sizes)
        .copy_to_host(Alignment::of::<usize>())?
        .await?;

    let statuses = Buffer::<nvcompStatus_t>::from_byte_buffer(statuses);
    let actual_sizes = Buffer::<usize>::from_byte_buffer(actual_sizes);

    for (index, page) in plan.pages.iter().enumerate() {
        let status = statuses.as_slice()[index];
        ensure!(
            status == sys::nvcompStatus_t_nvcompSuccess,
            "page {index} failed to decompress with nvcomp status {status}"
        );
        let actual = actual_sizes.as_slice()[index];
        ensure!(
            actual == page.uncompressed_len,
            "page {index} decompressed to {actual} bytes, expected {}",
            page.uncompressed_len
        );
    }

    Ok(())
}

/// Compares every decompressed page against the host codec's output for the same bytes.
async fn verify_against_host(
    plan: &RowGroupPlan,
    file_bytes: &[u8],
    codec: GpuCodec,
    output: CudaSlice<u8>,
) -> Result<()> {
    let device_output = CudaDeviceBuffer::new(output)
        .copy_to_host(Alignment::of::<u8>())?
        .await?;
    let device_output = device_output.as_slice();

    for (index, page) in plan.pages.iter().enumerate() {
        let chunk = &plan.chunks[page.chunk];
        let start = usize::try_from(chunk.offset)? + page.offset_in_chunk;
        let compressed = &file_bytes[start..start + page.compressed_len];
        let expected = codec.decompress_host(compressed, page.uncompressed_len)?;
        let actual = &device_output[page.output_offset..page.output_offset + page.uncompressed_len];
        ensure!(
            actual == expected.as_slice(),
            "page {index} decompressed on the GPU differs from the host codec output"
        );
    }

    tracing::info!(
        "verified {} GPU-decompressed {} pages against the host codec",
        plan.pages.len(),
        codec.name()
    );
    Ok(())
}
