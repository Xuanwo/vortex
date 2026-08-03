// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`super::FilterArray`].
//!
//! The main entrypoint is [`execute_filter`] which filters any [`Canonical`] array.

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::Filter;
use crate::arrays::NullArray;
use crate::arrays::VariantArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::filter::FilterArraySlotsExt;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::scalar::Scalar;
use crate::validity::Validity;

pub(crate) mod byte_compress;

mod simd_compress;
mod slice;
mod take;

mod bitbuffer;
mod buffer;

mod bool;
mod decimal;
mod fixed_size_list;
mod listview;
mod primitive;
mod struct_;
mod union;
mod varbinview;

/// A helper function that lazily filters a [`Validity`] with selection mask values.
fn filter_validity(validity: Validity, mask: &Arc<MaskValues>) -> Validity {
    validity
        .filter(&Mask::Values(Arc::clone(mask)))
        .vortex_expect("Somehow unable to wrap filter around a validity array")
}

pub(super) fn contiguous_filter_range(mask: &Mask) -> Option<Range<usize>> {
    match mask {
        Mask::AllTrue(len) => (*len > 0).then_some(0..*len),
        Mask::AllFalse(_) => None,
        Mask::Values(values) => contiguous_values_range(values),
    }
}

fn contiguous_values_range(mask: &MaskValues) -> Option<Range<usize>> {
    if let Some(slices) = mask.cached_slices() {
        return match slices {
            [(start, end)] => Some(*start..*end),
            _ => None,
        };
    }

    if let Some(indices) = mask.cached_indices() {
        let start = *indices.first()?;
        let end = indices.last()?.checked_add(1)?;
        return (end - start == indices.len()).then_some(start..end);
    }

    let true_count = mask.true_count();
    let start = mask.bit_buffer().set_indices().next()?;
    let end = start.checked_add(true_count)?;
    if end > mask.len() {
        return None;
    }

    // Bound uncached work by probing the candidate run from the cheaper side.
    let contiguous = if true_count <= mask.len() / 2 {
        mask.bit_buffer().count_range(start, end) == true_count
    } else {
        mask.bit_buffer().last_set_index() == Some(end - 1)
    };
    contiguous.then_some(start..end)
}

pub(super) fn prepare_mask_for_reuse(mask: &MaskValues, consumers: usize) {
    buffer::prepare_mask_for_reuse(mask, consumers);
}

/// Check for some fast-path execution conditions before calling [`execute_filter`].
pub(super) fn execute_filter_fast_paths(
    array: ArrayView<'_, Filter>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let true_count = array.mask.true_count();

    // If the mask selects nothing, the output is empty.
    if true_count == 0 {
        return Ok(Some(Canonical::empty(array.dtype()).into_array()));
    }

    // If the mask selects everything, then we can just fully decompress the whole thing.
    if true_count == array.mask.len() {
        return Ok(Some(array.child().clone()));
    }

    // Filtering by one contiguous range is exactly a slice and can remain zero-copy.
    if let Some(range) = contiguous_filter_range(array.filter_mask()) {
        return array.child().slice(range).map(Some);
    }

    // Also check if the array itself is completely null, in which case we only care about the total
    // number of nulls, not the values.
    let child_arr = array.array();
    if child_arr
        .validity()?
        .execute_mask(child_arr.len(), ctx)?
        .true_count()
        == 0
    {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), true_count).into_array(),
        ));
    }

    Ok(None)
}

/// Filter a canonical array by a mask, returning a new canonical array.
pub(super) fn execute_filter(canonical: Canonical, mask: &Arc<MaskValues>) -> Canonical {
    match canonical {
        Canonical::Null(_) => Canonical::Null(NullArray::new(mask.true_count())),
        Canonical::Bool(a) => Canonical::Bool(bool::filter_bool(&a, mask)),
        Canonical::Primitive(a) => Canonical::Primitive(primitive::filter_primitive(&a, mask)),
        Canonical::Decimal(a) => Canonical::Decimal(decimal::filter_decimal(&a, mask)),
        Canonical::VarBinView(a) => Canonical::VarBinView(varbinview::filter_varbinview(&a, mask)),
        Canonical::List(a) => Canonical::List(listview::filter_listview(&a, mask)),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(fixed_size_list::filter_fixed_size_list(&a, mask))
        }
        Canonical::Struct(a) => Canonical::Struct(struct_::filter_struct(&a, mask)),
        Canonical::Union(a) => Canonical::Union(union::filter_union(&a, mask)),
        Canonical::Extension(a) => {
            let filtered_storage = a
                .storage_array()
                .filter(Mask::Values(Arc::clone(mask)))
                .vortex_expect("ExtensionArray storage type somehow could not be filtered");
            Canonical::Extension(ExtensionArray::new(a.ext_dtype().clone(), filtered_storage))
        }
        Canonical::Variant(a) => {
            let filter_mask = Mask::Values(Arc::clone(mask));
            let filtered_core_storage = a
                .core_storage()
                .filter(filter_mask.clone())
                .vortex_expect("VariantArray core_storage could not be filtered");
            let filtered_shredded = a.shredded().map(|shredded| {
                shredded
                    .filter(filter_mask)
                    .vortex_expect("VariantArray shredded child could not be filtered")
            });
            Canonical::Variant(
                VariantArray::try_new(filtered_core_storage, filtered_shredded)
                    .vortex_expect("filtered VariantArray children are row-aligned"),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn uncached_contiguous_filter_executes_as_zero_copy_slice() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(0i32..128);
        let original = array.to_buffer::<i32>();
        let mask = Mask::from_buffer(BitBuffer::from_iter(
            (0..128).map(|index| (37..91).contains(&index)),
        ));
        assert!(mask.values().is_some_and(|values| {
            values.cached_indices().is_none() && values.cached_slices().is_none()
        }));

        let filtered = array
            .into_array()
            .filter(mask)?
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())?;
        let filtered_values = filtered.to_buffer::<i32>();

        assert_eq!(filtered_values.as_slice(), &(37..91).collect::<Vec<_>>());
        assert_eq!(filtered_values.as_ptr(), original.as_ptr().wrapping_add(37));
        Ok(())
    }

    #[test]
    fn uncached_contiguous_range_handles_sparse_and_dense_masks() {
        let cases = [
            (
                Mask::from_buffer(BitBuffer::from_iter(
                    (0..128).map(|index| (37..41).contains(&index)),
                )),
                Some(37..41),
            ),
            (
                Mask::from_buffer(BitBuffer::from_iter(
                    (0..128).map(|index| (3..125).contains(&index)),
                )),
                Some(3..125),
            ),
            (
                Mask::from_buffer(BitBuffer::from_iter(
                    (0..128).map(|index| matches!(index, 1 | 40 | 90)),
                )),
                None,
            ),
            (
                Mask::from_buffer(BitBuffer::from_iter(
                    (0..128).map(|index| !matches!(index, 17 | 63)),
                )),
                None,
            ),
        ];

        for (mask, expected) in cases {
            assert!(mask.values().is_some_and(|values| {
                values.cached_indices().is_none() && values.cached_slices().is_none()
            }));
            assert_eq!(contiguous_filter_range(&mask), expected);
        }
    }
}
