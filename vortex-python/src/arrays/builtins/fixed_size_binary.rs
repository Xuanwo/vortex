// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::pyclass;
use vortex::array::arrays::FixedSizeBinary;

use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;

/// Concrete class for fixed-size binary arrays with the `vortex.fixed_size_binary` encoding.
#[pyclass(name = "FixedSizeBinaryArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFixedSizeBinaryArray;

impl EncodingSubclass for PyFixedSizeBinaryArray {
    type VTable = FixedSizeBinary;
}
