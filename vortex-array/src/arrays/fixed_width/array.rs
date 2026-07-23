// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::array::Array;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::validity::Validity;

/// Type-specific access needed by shared fixed-width structural compute.
pub(crate) trait FixedWidthArray: VTable {
    fn byte_width(array: ArrayView<'_, Self>) -> usize;

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer;

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Array<Self>>;
}
