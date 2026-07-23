// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use super::FixedSizeBinary;
use super::FixedSizeBinaryArrayExt;
use super::FixedSizeBinaryData;
use crate::ArrayContext;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::ArrayParts;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::FixedSizeBinaryArray;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

#[test]
fn scalar_at_and_zero_width() -> VortexResult<()> {
    let values = FixedSizeBinaryArray::new(
        buffer![1u8, 2, 3, 4, 5, 6].into_byte_buffer(),
        2,
        3,
        Validity::NonNullable,
    );
    assert_eq!(values.value(1).as_slice(), &[3, 4]);
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(
        values.execute_scalar(2, &mut ctx)?,
        Scalar::fixed_size_binary(vec![5u8, 6], Nullability::NonNullable)
    );

    let empty_values = FixedSizeBinaryArray::new(ByteBuffer::empty(), 0, 4, Validity::NonNullable);
    assert_eq!(empty_values.len(), 4);
    assert_eq!(
        empty_values.dtype(),
        &DType::FixedSizeBinary(0, Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn slice_filter_and_take() -> VortexResult<()> {
    let values = FixedSizeBinaryArray::new(
        buffer![1u8, 2, 3, 4, 5, 6, 7, 8].into_byte_buffer(),
        2,
        4,
        Validity::from_iter([true, false, true, true]),
    )
    .into_array();
    let mut ctx = array_session().create_execution_ctx();

    let sliced = values.slice(1..4)?;
    assert_eq!(
        sliced.execute_scalar(0, &mut ctx)?,
        Scalar::null(DType::FixedSizeBinary(2, Nullability::Nullable))
    );
    assert_eq!(
        sliced.execute_scalar(2, &mut ctx)?,
        Scalar::fixed_size_binary(vec![7u8, 8], Nullability::Nullable)
    );

    let filtered = values.filter(Mask::from_iter([true, true, false, true]))?;
    assert_eq!(filtered.len(), 3);
    assert_eq!(
        filtered.execute_scalar(2, &mut ctx)?,
        Scalar::fixed_size_binary(vec![7u8, 8], Nullability::Nullable)
    );

    let taken = values.take(buffer![3u32, 0].into_array())?;
    assert_eq!(
        taken.execute_scalar(0, &mut ctx)?,
        Scalar::fixed_size_binary(vec![7u8, 8], Nullability::Nullable)
    );
    assert_eq!(
        taken.execute_scalar(1, &mut ctx)?,
        Scalar::fixed_size_binary(vec![1u8, 2], Nullability::Nullable)
    );
    Ok(())
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(2)]
#[case(3)]
#[case(4)]
#[case(8)]
#[case(16)]
#[case(32)]
fn filter_and_take_runtime_widths(#[case] byte_width: u32) -> VortexResult<()> {
    let byte_width_usize = byte_width as usize;
    let mut values = ByteBufferMut::with_capacity(4 * byte_width_usize);
    for row in 0..4u8 {
        values.extend(std::iter::repeat_n(row, byte_width_usize));
    }
    let array = FixedSizeBinaryArray::new(values.freeze(), byte_width, 4, Validity::NonNullable);
    let mut ctx = array_session().create_execution_ctx();

    let filtered = array
        .clone()
        .into_array()
        .filter(Mask::from_iter([true, false, true, false]))?
        .execute::<FixedSizeBinaryArray>(&mut ctx)?;
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered.value(0).as_slice(), vec![0; byte_width_usize]);
    assert_eq!(filtered.value(1).as_slice(), vec![2; byte_width_usize]);

    let taken = array
        .take(buffer![3u32, 1].into_array())?
        .execute::<FixedSizeBinaryArray>(&mut ctx)?;
    assert_eq!(taken.len(), 2);
    assert_eq!(taken.value(0).as_slice(), vec![3; byte_width_usize]);
    assert_eq!(taken.value(1).as_slice(), vec![1; byte_width_usize]);
    Ok(())
}

#[test]
fn take_null_index_ignores_out_of_bounds_physical_value() -> VortexResult<()> {
    let values = FixedSizeBinaryArray::new(
        buffer![1u8, 2, 3, 4].into_byte_buffer(),
        2,
        2,
        Validity::NonNullable,
    );
    let indices = crate::arrays::PrimitiveArray::new(
        buffer![1u64, 2],
        Validity::Array(BoolArray::from_iter([true, false]).into_array()),
    );
    let taken = values.take(indices.into_array())?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        taken.execute_scalar(0, &mut ctx)?,
        Scalar::fixed_size_binary(vec![3u8, 4], Nullability::Nullable)
    );
    assert_eq!(
        taken.execute_scalar(1, &mut ctx)?,
        Scalar::null(DType::FixedSizeBinary(2, Nullability::Nullable))
    );
    Ok(())
}

#[test]
#[should_panic(expected = "out of bounds")]
fn take_out_of_bounds_index_panics() {
    let values = FixedSizeBinaryArray::new(
        buffer![
            1u8, 2, 3, 4, //
            5, 6, 7, 8, //
            9, 10, 11, 12,
        ]
        .into_byte_buffer(),
        4,
        3,
        Validity::NonNullable,
    );
    let indices = buffer![3u32, 0, 1, 2, 0, 1, 2, 0, 1].into_array();
    let mut ctx = array_session().create_execution_ctx();

    drop(
        values
            .take(indices)
            .unwrap()
            .execute::<FixedSizeBinaryArray>(&mut ctx),
    );
}

#[test]
fn array_parts_reject_mismatched_buffer_length() {
    let len = 2;
    let data = FixedSizeBinaryData {
        byte_width: 2,
        buffer: BufferHandle::new_host(buffer![1u8, 2, 3].into_byte_buffer()),
        len,
    };
    let slots = FixedSizeBinaryData::make_slots(&Validity::NonNullable, len);
    let parts = ArrayParts::new(
        FixedSizeBinary,
        DType::FixedSizeBinary(2, Nullability::NonNullable),
        len,
        data,
    )
    .with_slots(slots);

    assert!(FixedSizeBinaryArray::try_from_parts(parts).is_err());
}

#[test]
fn nullable_serde_roundtrip() -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let array = FixedSizeBinaryArray::new(
        buffer![1u8, 2, 3, 4, 5, 6].into_byte_buffer(),
        2,
        3,
        Validity::from_iter([true, false, true]),
    );
    let dtype = array.dtype().clone();
    let len = array.len();

    let array_ctx = ArrayContext::empty();
    let serialized =
        array
            .clone()
            .into_array()
            .serialize(&array_ctx, &session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buffer in serialized {
        concat.extend_from_slice(buffer.as_ref());
    }
    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;

    assert!(decoded.is::<FixedSizeBinary>());
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}
