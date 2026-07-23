// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;

use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;

pub(super) fn check_fixed_size_binary_constant(array: &FixedSizeBinaryArray) -> bool {
    let byte_width = array.byte_width() as usize;
    if byte_width == 0 {
        return true;
    }

    array
        .buffer_handle()
        .to_host_sync()
        .as_slice()
        .chunks_exact(byte_width)
        .all_equal()
}
