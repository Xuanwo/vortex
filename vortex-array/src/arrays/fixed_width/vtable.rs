// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::child_to_validity;
use crate::buffer::BufferHandle;
use crate::dtype::Nullability;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

pub(crate) fn buffer(array_name: &str, values: &BufferHandle, idx: usize) -> BufferHandle {
    match idx {
        0 => values.clone(),
        _ => vortex_panic!("{array_name} buffer index {idx} out of bounds"),
    }
}

pub(crate) fn buffer_name(idx: usize) -> Option<String> {
    match idx {
        0 => Some("values".to_string()),
        _ => None,
    }
}

pub(crate) fn single_buffer(buffers: &[BufferHandle]) -> VortexResult<BufferHandle> {
    vortex_ensure!(
        buffers.len() == 1,
        "Expected 1 buffer, got {}",
        buffers.len()
    );
    Ok(buffers[0].clone())
}

pub(crate) fn deserialize_validity(
    nullability: Nullability,
    len: usize,
    children: &dyn ArrayChildren,
) -> VortexResult<Validity> {
    match children.len() {
        0 => Ok(Validity::from(nullability)),
        1 => Ok(Validity::Array(children.get(0, &Validity::DTYPE, len)?)),
        child_count => vortex_bail!("Expected 0 or 1 child, got {child_count}"),
    }
}

pub(crate) fn validate_layout(
    array_name: &str,
    data_len: usize,
    nullability: Nullability,
    len: usize,
    slots: &[Option<ArrayRef>],
) -> VortexResult<()> {
    vortex_ensure!(slots.len() == 1, "{array_name} expects one validity slot");
    vortex_ensure!(
        data_len == len,
        InvalidArgument:
        "{array_name} length {data_len} does not match outer length {len}"
    );
    let validity = child_to_validity(slots[0].as_ref(), nullability);
    if let Some(validity_len) = validity.maybe_len() {
        vortex_ensure!(
            validity_len == len,
            InvalidArgument:
            "{array_name} validity len {validity_len} does not match outer length {len}"
        );
    }
    Ok(())
}

pub(crate) fn validity<V: VTable>(array: ArrayView<'_, V>) -> VortexResult<Validity> {
    Ok(child_to_validity(
        array.slots()[0].as_ref(),
        array.dtype().nullability(),
    ))
}

pub(crate) fn slot_name(idx: usize) -> String {
    match idx {
        0 => "validity".to_string(),
        _ => vortex_panic!("Fixed-width slot index {idx} out of bounds"),
    }
}
