// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// A builder for canonical fixed-size binary arrays.
pub struct FixedSizeBinaryBuilder {
    dtype: DType,
    values: BufferMut<u8>,
    nulls: LazyBitBufferBuilder,
}

impl FixedSizeBinaryBuilder {
    /// Creates a builder with capacity for `capacity` values.
    pub fn with_capacity(byte_width: u32, nullability: Nullability, capacity: usize) -> Self {
        Self {
            dtype: DType::FixedSizeBinary(byte_width, nullability),
            values: BufferMut::with_capacity(capacity.saturating_mul(byte_width as usize)),
            nulls: LazyBitBufferBuilder::new(capacity),
        }
    }

    /// Appends one non-null fixed-size value.
    pub fn append_value(&mut self, value: impl AsRef<[u8]>) -> VortexResult<()> {
        let value = value.as_ref();
        vortex_ensure!(
            value.len() == self.byte_width(),
            "FixedSizeBinaryBuilder expected {} bytes, got {}",
            self.byte_width(),
            value.len(),
        );
        self.values.extend_from_slice(value);
        self.nulls.append_non_null();
        Ok(())
    }

    fn byte_width(&self) -> usize {
        self.byte_width_u32() as usize
    }

    fn byte_width_u32(&self) -> u32 {
        let DType::FixedSizeBinary(byte_width, _) = self.dtype else {
            unreachable!()
        };
        byte_width
    }

    pub(crate) fn append_fixed_size_binary_array(
        &mut self,
        array: &FixedSizeBinaryArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == self.dtype(),
            "FixedSizeBinaryBuilder expected array with dtype {}, got {}",
            self.dtype(),
            array.dtype(),
        );
        self.values
            .extend_from_slice(array.buffer_handle().to_host_sync().as_slice());
        self.nulls
            .append_validity_mask(&array.as_ref().validity()?.execute_mask(array.len(), ctx)?);
        Ok(())
    }

    /// Finishes this builder directly into a fixed-size binary array.
    pub fn finish_into_fixed_size_binary(&mut self) -> FixedSizeBinaryArray {
        let len = self.nulls.len();
        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());
        FixedSizeBinaryArray::new(
            std::mem::take(&mut self.values).freeze().into_byte_buffer(),
            self.byte_width_u32(),
            len,
            validity,
        )
    }
}

impl ArrayBuilder for FixedSizeBinaryBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.nulls.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.values.push_n(0, n.saturating_mul(self.byte_width()));
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.values.push_n(0, n.saturating_mul(self.byte_width()));
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "FixedSizeBinaryBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype(),
        );
        match scalar.as_binary().value() {
            Some(value) => self.append_value(value),
            None => {
                self.append_null();
                Ok(())
            }
        }
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.values
            .reserve(additional.saturating_mul(self.byte_width()));
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::from_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_fixed_size_binary().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::FixedSizeBinary(self.finish_into_fixed_size_binary())
    }
}
