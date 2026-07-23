// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::validity::Validity;

impl FixedWidthArray for Primitive {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.ptype().byte_width()
    }

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        array.buffer_handle().to_host_sync()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<PrimitiveArray> {
        let expected_len = len
            .checked_mul(array.ptype().byte_width())
            .ok_or_else(|| vortex_err!("Primitive values buffer length overflows usize"))?;
        vortex_ensure!(
            values.len() == expected_len,
            "Primitive values buffer length does not match output length"
        );
        Ok(PrimitiveArray::from_byte_buffer(
            values,
            array.ptype(),
            validity,
        ))
    }
}
