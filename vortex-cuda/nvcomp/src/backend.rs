// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Options and metadata shared by nvcomp's batched decompression APIs.

use crate::sys;

/// Backend selection for nvcomp decompression.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecompressBackend {
    /// Let nvcomp auto-select the best backend for the hardware.
    #[default]
    Default,
    /// Use hardware decompression
    Hardware,
    /// Use CUDA
    Cuda,
}

impl DecompressBackend {
    pub(crate) fn to_nvcomp(self) -> sys::nvcompDecompressBackend_t {
        match self {
            Self::Default => sys::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
            Self::Hardware => sys::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_HARDWARE,
            Self::Cuda => sys::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_CUDA,
        }
    }
}

/// Minimum buffer alignments required by an nvcomp algorithm.
///
/// Buffers passed to the batched decompression entrypoints must satisfy these alignments.
/// Exceeding them (for example 16- or 32-byte alignment) may improve throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentRequirements {
    /// Minimum alignment of each compressed input chunk.
    pub input: usize,
    /// Minimum alignment of each decompressed output chunk.
    pub output: usize,
    /// Minimum alignment of the temporary workspace buffer.
    pub temp: usize,
}

impl From<sys::nvcompAlignmentRequirements_t> for AlignmentRequirements {
    fn from(value: sys::nvcompAlignmentRequirements_t) -> Self {
        Self {
            input: value.input,
            output: value.output,
            temp: value.temp,
        }
    }
}
