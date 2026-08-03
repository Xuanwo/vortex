// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wrappers around nvcomp's batched Snappy decompression API.
//!
//! Snappy is the default Parquet page codec and is the fastest of the codecs nvcomp
//! implements on device, which makes it the codec of choice when feeding Parquet pages
//! to the GPU.

use std::ffi::c_void;

use crate::backend::AlignmentRequirements;
pub use crate::backend::DecompressBackend;
use crate::error::NvcompError;
use crate::error::check_status;
use crate::nvcomp_library;
use crate::sys;

/// The largest compressed chunk the Snappy decompressor accepts, in bytes.
pub const MAX_COMPRESSED_CHUNK_SIZE: usize = (1 << 31) - 1;

/// Options for batched Snappy decompression.
#[derive(Debug, Clone, Copy, Default)]
pub struct SnappyDecompressOpts {
    /// Which nvcomp backend performs the decompression.
    pub backend: DecompressBackend,
    /// Sort chunks by size before submitting them to the hardware decompression engine.
    ///
    /// Only used when `backend` selects the hardware engine.
    pub sort_before_hw_decompress: bool,
}

impl SnappyDecompressOpts {
    fn to_nvcomp(self) -> sys::nvcompBatchedSnappyDecompressOpts_t {
        sys::nvcompBatchedSnappyDecompressOpts_t {
            backend: self.backend.to_nvcomp(),
            sort_before_hw_decompress: i32::from(self.sort_before_hw_decompress),
            reserved: [0; 56],
        }
    }
}

/// Computes required temporary buffer size for batched Snappy decompression.
///
/// # Arguments
///
/// * `num_chunks` - Number of compressed chunks to decompress
/// * `max_uncompressed_chunk_bytes` - Maximum uncompressed size of any single chunk
/// * `max_total_uncompressed_bytes` - Total uncompressed size across all chunks
///
/// # Returns
///
/// The required size in bytes for the temporary buffer.
pub fn get_decompress_temp_size(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    max_total_uncompressed_bytes: usize,
) -> Result<usize, NvcompError> {
    get_decompress_temp_size_with_opts(
        num_chunks,
        max_uncompressed_chunk_bytes,
        max_total_uncompressed_bytes,
        SnappyDecompressOpts::default(),
    )
}

/// Computes required temporary buffer size with custom options.
///
/// # Arguments
///
/// * `num_chunks` - Number of compressed chunks to decompress
/// * `max_uncompressed_chunk_bytes` - Maximum uncompressed size of any single chunk
/// * `max_total_uncompressed_bytes` - Total uncompressed size across all chunks
/// * `opts` - Decompression options
///
/// # Returns
///
/// The required size in bytes for the temporary buffer.
pub fn get_decompress_temp_size_with_opts(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    max_total_uncompressed_bytes: usize,
    opts: SnappyDecompressOpts,
) -> Result<usize, NvcompError> {
    let library = nvcomp_library()?;

    let mut temp_bytes: usize = 0;

    let status = unsafe {
        library.nvcompBatchedSnappyDecompressGetTempSizeAsync(
            num_chunks,
            max_uncompressed_chunk_bytes,
            opts.to_nvcomp(),
            &raw mut temp_bytes,
            max_total_uncompressed_bytes,
        )
    };

    check_status(status)?;
    Ok(temp_bytes)
}

/// Returns the minimum buffer alignments required by batched Snappy decompression.
pub fn decompress_alignment_requirements(
    opts: SnappyDecompressOpts,
) -> Result<AlignmentRequirements, NvcompError> {
    let library = nvcomp_library()?;

    let mut requirements = sys::nvcompAlignmentRequirements_t {
        input: 0,
        output: 0,
        temp: 0,
    };

    let status = unsafe {
        library.nvcompBatchedSnappyDecompressGetRequiredAlignments(
            opts.to_nvcomp(),
            &raw mut requirements,
        )
    };

    check_status(status)?;
    Ok(requirements.into())
}

/// Launches batched Snappy decompression asynchronously on the GPU.
///
/// This function decompresses multiple raw Snappy blocks in parallel on the GPU. All
/// pointer arguments must point to device memory, and the operation is executed
/// asynchronously on the provided CUDA stream.
///
/// # Arguments
///
/// * `device_compressed_ptrs` - Device pointer to array of pointers to compressed chunks
/// * `device_compressed_bytes` - Device pointer to array of compressed chunk sizes
/// * `device_uncompressed_bytes` - Device pointer to array of expected uncompressed sizes
/// * `device_actual_uncompressed_bytes` - Device pointer to array for actual uncompressed sizes (output)
/// * `num_chunks` - Number of chunks to decompress
/// * `device_temp_ptr` - Device pointer to temporary workspace buffer
/// * `temp_bytes` - Size of temporary buffer in bytes
/// * `device_uncompressed_ptrs` - Device pointer to array of pointers to output buffers
/// * `device_statuses` - Device pointer to array for per-chunk status codes (output)
/// * `stream` - CUDA stream to execute on
///
/// # Safety
///
/// - All device pointers must be valid and point to properly allocated device memory
/// - `device_compressed_ptrs` must point to valid device pointers
/// - `device_uncompressed_ptrs` must point to valid device pointers
/// - Each output buffer must have at least the corresponding `device_uncompressed_bytes` size
/// - `device_temp_ptr` must have at least `temp_bytes` allocated
/// - The stream must be valid
#[expect(clippy::too_many_arguments)]
pub unsafe fn decompress_async(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    num_chunks: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_statuses: *mut sys::nvcompStatus_t,
    stream: sys::cudaStream_t,
) -> Result<(), NvcompError> {
    // SAFETY: Caller has to ensure all pointers are valid.
    unsafe {
        decompress_async_with_opts(
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            num_chunks,
            device_temp_ptr,
            temp_bytes,
            device_uncompressed_ptrs,
            device_statuses,
            stream,
            SnappyDecompressOpts::default(),
        )
    }
}

/// Launches batched Snappy decompression asynchronously with custom options.
///
/// # Safety
///
/// Same requirements as [`decompress_async`].
#[expect(clippy::too_many_arguments)]
pub unsafe fn decompress_async_with_opts(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    num_chunks: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_statuses: *mut sys::nvcompStatus_t,
    stream: sys::cudaStream_t,
    opts: SnappyDecompressOpts,
) -> Result<(), NvcompError> {
    let library = nvcomp_library()?;

    let status = unsafe {
        library.nvcompBatchedSnappyDecompressAsync(
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            num_chunks,
            device_temp_ptr,
            temp_bytes,
            device_uncompressed_ptrs,
            opts.to_nvcomp(),
            device_statuses,
            stream,
        )
    };

    check_status(status)
}
