// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical contiguous storage for fixed-size binary values.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Not;

use smallvec::smallvec;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::TypedArrayRef;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::Dict;
use crate::arrays::Masked;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::FixedSizeBinaryBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        Cast.id(),
        FixedSizeBinary,
        CastExecuteAdaptor(FixedSizeBinary),
    );
    kernels.register_execute_parent_kernel(
        FillNull.id(),
        FixedSizeBinary,
        FillNullExecuteAdaptor(FixedSizeBinary),
    );
    kernels.register_execute_parent_kernel(
        Dict.id(),
        FixedSizeBinary,
        TakeExecuteAdaptor(FixedSizeBinary),
    );
}

static RULES: ParentRuleSet<FixedSizeBinary> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FixedSizeBinaryMaskedValidityRule),
    ParentRuleSet::lift(&CastReduceAdaptor(FixedSizeBinary)),
    ParentRuleSet::lift(&MaskReduceAdaptor(FixedSizeBinary)),
    ParentRuleSet::lift(&SliceReduceAdaptor(FixedSizeBinary)),
]);

/// A canonical array of fixed-size byte strings.
pub type FixedSizeBinaryArray = Array<FixedSizeBinary>;

/// Physical data for a [`FixedSizeBinaryArray`].
#[derive(Clone, Debug)]
pub struct FixedSizeBinaryData {
    byte_width: u32,
    buffer: BufferHandle,
    len: usize,
}

/// Owned components of a [`FixedSizeBinaryArray`].
pub struct FixedSizeBinaryDataParts {
    /// The width of each logical value in bytes.
    pub byte_width: u32,
    /// The contiguous values buffer.
    pub buffer: BufferHandle,
    /// The number of logical values.
    pub len: usize,
    /// The validity of the logical values.
    pub validity: Validity,
}

impl Display for FixedSizeBinaryData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "byte_width: {}", self.byte_width)
    }
}

impl ArrayHash for FixedSizeBinaryData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.buffer.array_hash(state, accuracy);
        self.byte_width.hash(state);
        self.len.hash(state);
    }
}

impl ArrayEq for FixedSizeBinaryData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.buffer.array_eq(&other.buffer, accuracy)
            && self.byte_width == other.byte_width
            && self.len == other.len
    }
}

/// The canonical VTable for fixed-size binary arrays.
#[derive(Clone, Debug)]
pub struct FixedSizeBinary;

impl VTable for FixedSizeBinary {
    type TypedArrayData = FixedSizeBinaryData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.fixed_size_binary");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        fixed_width::buffer("FixedSizeBinaryArray", array.buffer_handle(), idx)
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        fixed_width::buffer_name(idx)
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let mut data = array.data().clone();
        data.buffer = fixed_width::replacement_buffer(buffers)?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn validate(
        &self,
        data: &FixedSizeBinaryData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::FixedSizeBinary(byte_width, nullability) = dtype else {
            vortex_error::vortex_bail!("Expected fixed-size binary dtype, got {dtype:?}");
        };
        vortex_ensure!(
            data.byte_width() == *byte_width,
            "Fixed-size binary dtype width {byte_width} does not match data width {}",
            data.byte_width(),
        );
        fixed_width::validate_layout("FixedSizeBinaryArray", data.len(), *nullability, len, slots)
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            metadata.is_empty(),
            "FixedSizeBinaryArray expects empty metadata"
        );
        let DType::FixedSizeBinary(byte_width, _) = dtype else {
            vortex_error::vortex_bail!("Expected fixed-size binary dtype, got {dtype:?}");
        };
        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;
        let slots = FixedSizeBinaryData::make_slots(&validity, len);
        let data = FixedSizeBinaryData::try_new_handle(
            fixed_width::replacement_buffer(buffers)?,
            *byte_width,
            len,
        )?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder
            .as_any_mut()
            .downcast_mut::<FixedSizeBinaryBuilder>()
        else {
            vortex_error::vortex_bail!(
                "append_to_builder for FixedSizeBinary requires a FixedSizeBinaryBuilder"
            );
        };
        builder.append_fixed_size_binary_array(&array.into_owned(), ctx)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        fixed_width::slot_name(idx)
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Default, Debug)]
struct FixedSizeBinaryMaskedValidityRule;

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
            Validity::Array(is_valid) => is_valid
                .execute::<crate::arrays::BoolArray>(ctx)?
                .into_bit_buffer()
                .not(),
            _ => unreachable!("checked in entry point"),
        };
        let byte_width = array.byte_width() as usize;
        for invalid_index in is_invalid.set_indices() {
            let start = invalid_index * byte_width;
            values[start..start + byte_width].copy_from_slice(fill.as_slice());
        }
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
        array: ArrayView<'_, FixedSizeBinary>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<FixedSizeBinaryArray> {
        FixedSizeBinaryArray::try_new(values, array.byte_width(), len, validity)
    }
}

/// Typed accessors for a [`FixedSizeBinaryArray`].
pub trait FixedSizeBinaryArrayExt: TypedArrayRef<FixedSizeBinary> {
    /// The number of bytes in every value.
    fn byte_width(&self) -> u32 {
        match self.as_ref().dtype() {
            DType::FixedSizeBinary(byte_width, _) => *byte_width,
            _ => unreachable!("FixedSizeBinaryArrayExt requires a fixed-size binary dtype"),
        }
    }

    /// The handle for the contiguous values buffer.
    fn buffer_handle(&self) -> &BufferHandle {
        &self.buffer
    }

    /// Copies the value at `index` into a standalone byte buffer.
    fn value(&self, index: usize) -> ByteBuffer {
        assert!(index < self.len(), "fixed-size binary index out of bounds");
        let byte_width = self.byte_width() as usize;
        let start = index * byte_width;
        self.buffer_handle()
            .to_host_sync()
            .slice(start..start + byte_width)
    }
}

impl<T: TypedArrayRef<FixedSizeBinary>> FixedSizeBinaryArrayExt for T {}

impl FixedSizeBinaryData {
    fn make_slots(validity: &Validity, len: usize) -> ArraySlots {
        smallvec![validity_to_child(validity, len)]
    }

    /// Creates fixed-size binary data backed by a host buffer.
    pub fn new(buffer: impl Into<ByteBuffer>, byte_width: u32, len: usize) -> Self {
        Self::try_new(buffer.into(), byte_width, len)
            .vortex_expect("FixedSizeBinaryData construction failed")
    }

    /// Tries to create fixed-size binary data backed by a host buffer.
    pub fn try_new(buffer: ByteBuffer, byte_width: u32, len: usize) -> VortexResult<Self> {
        Self::try_new_handle(BufferHandle::new_host(buffer), byte_width, len)
    }

    /// Tries to create fixed-size binary data from a buffer handle.
    pub fn try_new_handle(buffer: BufferHandle, byte_width: u32, len: usize) -> VortexResult<Self> {
        let expected_len = len
            .checked_mul(byte_width as usize)
            .ok_or_else(|| vortex_error::vortex_err!("Fixed-size binary buffer length overflow"))?;
        vortex_ensure!(
            buffer.len() == expected_len,
            InvalidArgument: "Fixed-size binary buffer length {} does not match {len} values of width {byte_width}",
            buffer.len(),
        );
        Ok(Self {
            byte_width,
            buffer,
            len,
        })
    }

    /// Returns the number of logical values represented by this data.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this data contains no logical values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bytes in each logical value.
    pub fn byte_width(&self) -> u32 {
        self.byte_width
    }

    /// Returns the underlying values buffer.
    pub fn buffer_handle(&self) -> &BufferHandle {
        &self.buffer
    }
}

impl Array<FixedSizeBinary> {
    /// Decomposes this array into its physical data and validity.
    pub fn into_data_parts(self) -> FixedSizeBinaryDataParts {
        let validity = child_to_validity(self.slots()[0].as_ref(), self.dtype().nullability());
        let data = self.into_data();
        FixedSizeBinaryDataParts {
            byte_width: data.byte_width,
            buffer: data.buffer,
            len: data.len,
            validity,
        }
    }

    /// Creates a fixed-size binary array from one contiguous values buffer.
    ///
    /// `len` is explicit so that arrays with a zero-byte value width can retain their row count.
    pub fn new(
        buffer: impl Into<ByteBuffer>,
        byte_width: u32,
        len: usize,
        validity: Validity,
    ) -> Self {
        Self::try_new(buffer.into(), byte_width, len, validity)
            .vortex_expect("FixedSizeBinaryArray construction failed")
    }

    /// Tries to create a fixed-size binary array from one contiguous values buffer.
    pub fn try_new(
        buffer: ByteBuffer,
        byte_width: u32,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                InvalidArgument: "Fixed-size binary validity length {validity_len} does not match array length {len}",
            );
        }
        Self::try_new_handle(BufferHandle::new_host(buffer), byte_width, len, validity)
    }

    pub(crate) fn try_new_handle(
        buffer: BufferHandle,
        byte_width: u32,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                InvalidArgument: "Fixed-size binary validity length {validity_len} does not match array length {len}",
            );
        }
        let dtype = DType::FixedSizeBinary(byte_width, validity.nullability());
        let slots = FixedSizeBinaryData::make_slots(&validity, len);
        let data = FixedSizeBinaryData::try_new_handle(buffer, byte_width, len)?;
        Array::try_from_parts(ArrayParts::new(FixedSizeBinary, dtype, len, data).with_slots(slots))
    }

    /// Creates an empty fixed-size binary array.
    pub fn empty(byte_width: u32, nullability: Nullability) -> Self {
        Self::new(
            ByteBuffer::empty(),
            byte_width,
            0,
            Validity::from(nullability),
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::FixedSizeBinaryArray;
    use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
    use crate::assert_arrays_eq;
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

        let empty_values =
            FixedSizeBinaryArray::new(ByteBuffer::empty(), 0, 4, Validity::NonNullable);
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
        let array =
            FixedSizeBinaryArray::new(values.freeze(), byte_width, 4, Validity::NonNullable);
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
        let serialized = array.clone().into_array().serialize(
            &array_ctx,
            &session,
            &SerializeOptions::default(),
        )?;
        let mut concat = ByteBufferMut::empty();
        for buffer in serialized {
            concat.extend_from_slice(buffer.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;

        assert!(decoded.is::<super::FixedSizeBinary>());
        assert_arrays_eq!(decoded, array, &mut ctx);
        Ok(())
    }
}
