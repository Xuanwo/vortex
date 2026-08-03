// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use arrow_schema::Field;
use async_trait::async_trait;
use futures::StreamExt;
use tempfile::NamedTempFile;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex_arrow::ArrowSessionExt;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_cuda::CanonicalCudaExt;
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
#[cfg(target_os = "linux")]
use vortex_cuda::PooledFileReadAtOptions;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
use vortex_cuda::layout::register_cuda_layout;

/// Vortex compressor whose decompression measurement executes CUDA-compatible files on the GPU.
pub struct GpuVortexCompressor {
    verify: bool,
}

impl GpuVortexCompressor {
    /// Create the backend.
    ///
    /// When `verify` is set, each GPU-decoded field is copied back to the host and compared
    /// against the same field decoded on the CPU. Verification runs inline, so timings from a
    /// verifying run are not comparable to a plain one.
    pub fn new(verify: bool) -> Self {
        Self { verify }
    }
}

#[async_trait]
impl Compressor for GpuVortexCompressor {
    fn format(&self) -> Format {
        Format::OnDiskVortex
    }

    async fn compress(&self, _parquet_path: &Path) -> Result<(u64, Duration)> {
        anyhow::bail!("GPU compress-bench only supports decompression measurements")
    }

    async fn decompress(&self, parquet_path: &Path) -> Result<Duration> {
        register_cuda_layout(&SESSION);

        let uncompressed = parquet_to_vortex_chunks(parquet_path.to_path_buf()).await?;
        let gpu_file = NamedTempFile::new()?;
        let mut output = tokio::fs::File::create(gpu_file.path()).await?;
        let strategy = WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().only_cuda_compatible())
            .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
            .build();
        SESSION
            .write_options()
            .with_strategy(strategy)
            .write(&mut output, uncompressed.into_array().to_array_stream())
            .await?;
        output.sync_all().await?;
        drop(output);

        if self.verify {
            return verify_against_host_scan(gpu_file.path()).await;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
        let start = Instant::now();
        let file = open_gpu(gpu_file.path()).await?;
        let mut batches = file.scan()?.into_array_stream()?;

        while let Some(batch) = batches.next().await {
            let record = batch?.execute::<StructArray>(cuda_ctx.execution_ctx())?;
            for field in record.iter_unmasked_fields() {
                black_box(field.clone().execute_cuda(&mut cuda_ctx).await?);
            }
        }
        cuda_ctx.synchronize_stream()?;

        Ok(start.elapsed())
    }
}

/// Opens a Vortex file for CUDA execution.
///
/// On Linux direct IO keeps repeated iterations measuring storage bandwidth rather than
/// page-cache hits.
async fn open_gpu(path: &Path) -> Result<vortex::file::VortexFile> {
    let open_options = SESSION.open_options().with_cuda();
    #[cfg(target_os = "linux")]
    let open_options =
        open_options.with_read_at_options(PooledFileReadAtOptions::default().with_direct_io());
    Ok(open_options.open_path(path).await?)
}

/// Decodes the same file on the GPU and on the CPU and fails on the first difference.
///
/// The CPU reference comes from a second, host-only scan rather than from re-decoding the
/// GPU scan's arrays: a CUDA scan hands back arrays whose buffers live in device memory,
/// which the host decoders cannot read.
///
/// Verification runs inline, so the returned duration is not comparable to a plain run.
async fn verify_against_host_scan(path: &Path) -> Result<Duration> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
    let start = Instant::now();

    let gpu_file = open_gpu(path).await?;
    let mut gpu_batches = gpu_file.scan()?.into_array_stream()?;
    let host_file = SESSION.open_options().open_path(path).await?;
    let mut host_batches = host_file.scan()?.into_array_stream()?;

    let mut fields_checked = 0usize;
    loop {
        let (gpu_batch, host_batch) = (gpu_batches.next().await, host_batches.next().await);
        let (gpu_batch, host_batch) = match (gpu_batch, host_batch) {
            (Some(gpu_batch), Some(host_batch)) => (gpu_batch?, host_batch?),
            (None, None) => break,
            _ => bail!("the GPU and CPU scans of the same file produced different batch counts"),
        };

        let gpu_record = gpu_batch.execute::<StructArray>(cuda_ctx.execution_ctx())?;
        let host_record = host_batch.execute::<StructArray>(cuda_ctx.execution_ctx())?;
        ensure!(
            gpu_record.len() == host_record.len(),
            "batch length differs between the GPU and CPU scans: {} vs {}",
            gpu_record.len(),
            host_record.len()
        );

        let gpu_fields = gpu_record
            .iter_unmasked_fields()
            .cloned()
            .collect::<Vec<_>>();
        let host_fields = host_record
            .iter_unmasked_fields()
            .cloned()
            .collect::<Vec<_>>();
        ensure!(
            gpu_fields.len() == host_fields.len(),
            "field count differs between the GPU and CPU scans"
        );

        for (gpu_field, host_field) in gpu_fields.into_iter().zip(host_fields) {
            let decoded = gpu_field.execute_cuda(&mut cuda_ctx).await?;
            let decoded = decoded.into_host().await?.into_array();
            verify_field(&host_field, decoded, cuda_ctx.execution_ctx())?;
            fields_checked += 1;
        }
    }
    cuda_ctx.synchronize_stream()?;

    tracing::info!("verified {fields_checked} GPU-decoded Vortex fields against the CPU decode");
    Ok(start.elapsed())
}

/// Fails unless a GPU-decoded field matches the same field decoded on the CPU.
fn verify_field(host: &ArrayRef, gpu: ArrayRef, ctx: &mut ExecutionCtx) -> Result<()> {
    let expected = SESSION.arrow().execute_arrow(host.clone(), None, ctx)?;
    // Pin the Arrow target type so the two sides cannot land on different but equivalent
    // encodings of the same logical values.
    let target = Field::new("", expected.data_type().clone(), gpu.dtype().is_nullable());
    let actual = SESSION.arrow().execute_arrow(gpu, Some(&target), ctx)?;

    ensure!(
        expected.to_data() == actual.to_data(),
        "GPU decode of a {} field does not match the CPU decode",
        host.encoding_id()
    );
    Ok(())
}
