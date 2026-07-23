// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::IsSortedIteratorExt;
use crate::ExecutionCtx;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;

pub(super) fn check_fixed_size_binary_sorted(
    array: &FixedSizeBinaryArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let byte_width = array.byte_width() as usize;
    let values = array.buffer_handle().to_host_sync();
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    let iter = validity.iter().enumerate().map(|(index, is_valid)| {
        is_valid.then(|| {
            let start = index * byte_width;
            &values[start..start + byte_width]
        })
    });

    Ok(if strict {
        iter.is_strict_sorted()
    } else {
        iter.is_sorted()
    })
}
