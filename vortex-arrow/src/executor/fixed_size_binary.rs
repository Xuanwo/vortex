// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeBinaryArray as ArrowFixedSizeBinaryArray;
use arrow_buffer::NullBuffer;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::FixedSizeBinaryArray;
use vortex_array::arrays::FixedSizeBinaryArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::null_buffer::to_null_buffer;

pub(super) fn to_arrow_fixed_size_binary(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = array.execute::<FixedSizeBinaryArray>(ctx)?;
    let byte_width = i32::try_from(array.byte_width())
        .map_err(|_| vortex_err!("Fixed-size binary width exceeds Arrow i32 range"))?;
    let mut nulls = to_null_buffer(array.as_ref().validity()?.execute_mask(array.len(), ctx)?);
    if byte_width == 0 && nulls.is_none() {
        // Arrow cannot infer the row count of zero-width values from the empty values buffer.
        nulls = Some(NullBuffer::new_valid(array.len()));
    }
    Ok(Arc::new(ArrowFixedSizeBinaryArray::new(
        byte_width,
        array.buffer_handle().to_host_sync().into_arrow_buffer(),
        nulls,
    )))
}

#[cfg(test)]
mod tests {
    use arrow_array::Array as ArrowArray;
    use arrow_array::cast::AsArray;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::FixedSizeBinaryArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::to_arrow_fixed_size_binary;

    #[test]
    fn exports_values_and_validity() -> VortexResult<()> {
        let array = FixedSizeBinaryArray::new(
            buffer![1u8, 2, 3, 4, 5, 6].into_byte_buffer(),
            2,
            3,
            Validity::from_iter([true, false, true]),
        );
        let mut ctx = array_session().create_execution_ctx();
        let arrow = to_arrow_fixed_size_binary(array.into_array(), &mut ctx)?;
        let arrow = arrow.as_fixed_size_binary();

        assert_eq!(arrow.value_length(), 2);
        assert_eq!(arrow.value(0), &[1, 2]);
        assert!(arrow.is_null(1));
        assert_eq!(arrow.value(2), &[5, 6]);
        Ok(())
    }

    #[test]
    fn exports_zero_width_length() -> VortexResult<()> {
        let array = FixedSizeBinaryArray::new(
            vortex_buffer::ByteBuffer::empty(),
            0,
            3,
            Validity::NonNullable,
        );
        let mut ctx = array_session().create_execution_ctx();
        let arrow = to_arrow_fixed_size_binary(array.into_array(), &mut ctx)?;
        let arrow = arrow.as_fixed_size_binary();

        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.value_length(), 0);
        assert_eq!(arrow.null_count(), 0);
        Ok(())
    }
}
