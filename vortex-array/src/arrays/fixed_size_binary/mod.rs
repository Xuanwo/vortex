// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical contiguous storage for fixed-size binary values.

mod compute;
#[cfg(test)]
mod tests;

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use smallvec::smallvec;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use self::compute::FixedSizeBinaryMaskedValidityRule;
use crate::ArrayRef;
use crate::ArraySlots;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::TypedArrayRef;
use crate::array::VTable;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::FixedSizeBinaryBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
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
        data.buffer = fixed_width::single_buffer(buffers)?;
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
            vortex_bail!("Expected fixed-size binary dtype, got {dtype:?}");
        };
        vortex_ensure!(
            data.byte_width() == *byte_width,
            "Fixed-size binary dtype width {byte_width} does not match data width {}",
            data.byte_width(),
        );
        data.validate_buffer_len()?;
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
            vortex_bail!("Expected fixed-size binary dtype, got {dtype:?}");
        };
        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;
        let slots = FixedSizeBinaryData::make_slots(&validity, len);
        let data = FixedSizeBinaryData::try_new_handle(
            fixed_width::single_buffer(buffers)?,
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
            vortex_bail!("append_to_builder for FixedSizeBinary requires a FixedSizeBinaryBuilder");
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

    /// Returns a zero-copy slice of the value at `index`.
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
        let data = Self {
            byte_width,
            buffer,
            len,
        };
        data.validate_buffer_len()?;
        Ok(data)
    }

    fn validate_buffer_len(&self) -> VortexResult<()> {
        let expected_len = self
            .len
            .checked_mul(self.byte_width as usize)
            .ok_or_else(|| vortex_err!("Fixed-size binary buffer length overflow"))?;
        vortex_ensure!(
            self.buffer.len() == expected_len,
            InvalidArgument: "Fixed-size binary buffer length {} does not match {} values of width {}",
            self.buffer.len(),
            self.len,
            self.byte_width,
        );
        Ok(())
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
