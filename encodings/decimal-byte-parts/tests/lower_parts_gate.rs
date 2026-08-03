// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The lower-parts write gate, checked from outside the crate.
//!
//! The gate lets the crate's own unit tests through so the multi-part paths stay covered by a
//! default `cargo test`. That bypass keys off `cfg!(test)`, which is false for the library
//! when it is compiled as a dependency of this integration test — so this is the only place
//! the gate's real behaviour can be observed.

#![expect(clippy::tests_outside_test_module)]

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::dtype::DecimalDType;
use vortex_buffer::buffer;
use vortex_decimal_byte_parts::DecimalByteParts;

fn msp() -> ArrayRef {
    buffer![1i64, 2, 3].into_array()
}

fn lower_part() -> ArrayRef {
    buffer![1u64, 2, 3].into_array()
}

/// A single-child array is the stable shape and is always constructible.
#[test]
fn single_child_is_always_allowed() {
    assert!(DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2)).is_ok());
    assert!(
        DecimalByteParts::try_new_with_lower_parts(msp(), vec![], DecimalDType::new(19, 2)).is_ok()
    );
}

#[cfg(not(feature = "unstable_encodings"))]
#[test]
fn lower_parts_rejected_without_the_feature() {
    let result = DecimalByteParts::try_new_with_lower_parts(
        msp(),
        vec![lower_part()],
        DecimalDType::new(38, 2),
    );
    let err = result.expect_err("expected the write gate to reject");
    assert!(
        err.to_string().contains("unstable_encodings"),
        "error should name the feature, got: {err}"
    );
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn lower_parts_allowed_with_the_feature() {
    assert!(
        DecimalByteParts::try_new_with_lower_parts(
            msp(),
            vec![lower_part()],
            DecimalDType::new(38, 2),
        )
        .is_ok()
    );
}
