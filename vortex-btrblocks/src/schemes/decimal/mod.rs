// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression scheme using byte-part decomposition.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::decimal::narrowed_decimal;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsSlots;
use vortex_decimal_byte_parts::MAX_LOWER_PARTS;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Compression scheme for decimal arrays via byte-part decomposition.
///
/// Narrows the decimal to the smallest integer type, compresses the underlying primitive, and wraps
/// the result in a `DecimalBytePartsArray`.
///
/// With `unstable_encodings`, values that stay wider than 64 bits after narrowing are split
/// into a signed most significant part and 64-bit lower parts — one for `i128`, three for
/// `i256` — each compressed independently. That writes more than one child, which readers
/// predating lower parts cannot open, so without the feature such values are left
/// uncompressed instead.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalScheme;

impl Scheme for DecimalScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.byte_parts"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![DecimalByteParts.id()]
    }

    /// Children: msp=0, and with `unstable_encodings`, lower parts=1..=3.
    fn num_children(&self) -> usize {
        if cfg!(feature = "unstable_encodings") {
            DecimalBytePartsSlots::FIXED_COUNT + MAX_LOWER_PARTS
        } else {
            DecimalBytePartsSlots::FIXED_COUNT
        }
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Decimal compression is almost always beneficial (narrowing + primitive compression).
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let decimal = data.array().clone().execute::<DecimalArray>(exec_ctx)?;
        let decimal = narrowed_decimal(decimal);
        let parts = split_decimal(&decimal)?;

        // Splitting a value too wide for one signed part writes more than one child, which a
        // reader predating lower parts cannot open. Until that is stable, leave those values
        // as the canonical decimal rather than emitting a file such a reader would reject.
        if !parts.lower_parts.is_empty() && !cfg!(feature = "unstable_encodings") {
            return Ok(decimal.into_array());
        }

        let msp = compressor.compress_child(
            &parts.msp,
            &compress_ctx,
            self.id(),
            DecimalBytePartsSlots::MSP,
            exec_ctx,
        )?;
        let lower_parts = parts
            .lower_parts
            .iter()
            .enumerate()
            .map(|(idx, part)| {
                compressor.compress_child(
                    part,
                    &compress_ctx,
                    self.id(),
                    DecimalBytePartsSlots::LOWER_PARTS_OFFSET + idx,
                    exec_ctx,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;

        DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, decimal.decimal_dtype())
            .map(|d| d.into_array())
    }
}

#[cfg(test)]
mod tests;
