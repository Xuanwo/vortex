// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::FixedSizeBinary;
use super::FixedSizeBinaryArray;
use super::FixedSizeBinaryArrayExt;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ValidityVTable;
use crate::arrays::BoolArray;
use crate::arrays::Masked;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::arrays::slice::SliceReduce;
use crate::dtype::DType;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

#[derive(Default, Debug)]
pub(super) struct FixedSizeBinaryMaskedValidityRule;

impl ArrayParentReduceRule<FixedSizeBinary> for FixedSizeBinaryMaskedValidityRule {
    type Parent = Masked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, FixedSizeBinary>,
        parent: ArrayView<'_, Masked>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.and(parent.validity()?)?;
        Ok(Some(
            FixedSizeBinaryArray::try_new_handle(
                array.buffer_handle().clone(),
                array.byte_width(),
                array.len(),
                validity,
            )?
            .into_array(),
        ))
    }
}

impl OperationsVTable<FixedSizeBinary> for FixedSizeBinary {
    fn scalar_at(
        array: ArrayView<'_, FixedSizeBinary>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let byte_width = array.byte_width() as usize;
        let values = array.buffer_handle().to_host_sync();
        let start = index * byte_width;
        Ok(Scalar::fixed_size_binary(
            values.slice(start..start + byte_width),
            array.dtype().nullability(),
        ))
    }
}

impl ValidityVTable<FixedSizeBinary> for FixedSizeBinary {
    fn validity(array: ArrayView<'_, FixedSizeBinary>) -> VortexResult<Validity> {
        fixed_width::validity(array)
    }
}

impl CastReduce for FixedSizeBinary {
    fn cast(
        array: ArrayView<'_, FixedSizeBinary>,
        dtype: &DType,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::FixedSizeBinary(byte_width, nullability) = dtype else {
            return Ok(None);
        };
        if *byte_width != array.byte_width() {
            return Ok(None);
        }
        let Some(validity) = array
            .validity()?
            .trivially_cast_nullability(*nullability, array.len())?
        else {
            return Ok(None);
        };
        Ok(Some(
            FixedSizeBinaryArray::try_new_handle(
                array.buffer_handle().clone(),
                *byte_width,
                array.len(),
                validity,
            )?
            .into_array(),
        ))
    }
}

impl CastKernel for FixedSizeBinary {
    fn cast(
        array: ArrayView<'_, FixedSizeBinary>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::FixedSizeBinary(byte_width, nullability) = dtype else {
            return Ok(None);
        };
        if *byte_width != array.byte_width() {
            return Ok(None);
        }
        let validity = array
            .validity()?
            .cast_nullability(*nullability, array.len(), ctx)?;
        Ok(Some(
            FixedSizeBinaryArray::try_new_handle(
                array.buffer_handle().clone(),
                *byte_width,
                array.len(),
                validity,
            )?
            .into_array(),
        ))
    }
}

impl FillNullKernel for FixedSizeBinary {
    fn fill_null(
        array: ArrayView<'_, FixedSizeBinary>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let result_validity = Validity::from(fill_value.dtype().nullability());
        let mut values = array.buffer_handle().to_host_sync().into_mut();
        let fill = fill_value
            .as_binary()
            .value()
            .vortex_expect("top-level fill_null ensure non-null fill value");
        let is_invalid = match array.validity()? {
            Validity::Array(is_valid) => {
                is_valid.execute::<BoolArray>(ctx)?.into_bit_buffer().not()
            }
            _ => unreachable!("checked in entry point"),
        };
        let byte_width = array.byte_width() as usize;
        is_invalid.for_each_set_index(|invalid_index| {
            let start = invalid_index * byte_width;
            values[start..start + byte_width].copy_from_slice(fill.as_slice());
        });
        Ok(Some(
            FixedSizeBinaryArray::new(
                values.freeze(),
                array.byte_width(),
                array.len(),
                result_validity,
            )
            .into_array(),
        ))
    }
}

impl MaskReduce for FixedSizeBinary {
    fn mask(
        array: ArrayView<'_, FixedSizeBinary>,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            FixedSizeBinaryArray::try_new_handle(
                array.buffer_handle().clone(),
                array.byte_width(),
                array.len(),
                array.validity()?.and(Validity::Array(mask.clone()))?,
            )?
            .into_array(),
        ))
    }
}

impl SliceReduce for FixedSizeBinary {
    fn slice(
        array: ArrayView<'_, FixedSizeBinary>,
        range: std::ops::Range<usize>,
    ) -> VortexResult<Option<ArrayRef>> {
        let byte_width = array.byte_width();
        let width = byte_width as usize;
        let values = array
            .buffer_handle()
            .slice(range.start * width..range.end * width);
        let len = range.len();
        let validity = array.validity()?.slice(range)?;
        Ok(Some(
            FixedSizeBinaryArray::try_new_handle(values, byte_width, len, validity)?.into_array(),
        ))
    }
}

impl FixedWidthArray for FixedSizeBinary {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.byte_width() as usize
    }

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        array.buffer_handle().to_host_sync()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<FixedSizeBinaryArray> {
        FixedSizeBinaryArray::try_new(values, array.byte_width(), len, validity)
    }
}
