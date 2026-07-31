// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use vortex_decimal_byte_parts::MAX_LOWER_PARTS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

/// Number of values per array: above the 1024-value sampling threshold, so scheme selection
/// runs on sampled estimates as it does for real file chunks.
const N: usize = 16_384;

fn ten_pow(exp: u32) -> i256 {
    i256::from_i128(10).wrapping_pow(exp)
}

/// Deterministic 24-bit noise, so the low part of each value is neither constant nor a
/// sequence — the realistic shape for a wide decimal column with a large fixed magnitude.
fn noise(seed: u64) -> impl Iterator<Item = i128> {
    let mut state = seed;
    iter::repeat_with(move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        i128::from(state >> 40)
    })
}

/// `i128`-backed values that need more than 64 bits, so the encoding must carry one lower
/// part.
fn wide_i128_array(validity: Validity) -> DecimalArray {
    let base = 10i128.pow(25);
    let values: Buffer<i128> = noise(7).take(N).map(|delta| base + delta).collect();
    DecimalArray::new(values, DecimalDType::new(38, 2), validity)
}

/// `i256`-backed values that need more than 128 bits, so the encoding must carry three
/// lower parts.
fn wide_i256_array(validity: Validity) -> DecimalArray {
    let base = ten_pow(40);
    let values: Buffer<i256> = noise(11)
        .take(N)
        .map(|delta| base + i256::from_i128(delta))
        .collect();
    DecimalArray::new(values, DecimalDType::new(76, 2), validity)
}

fn compress(array: &ArrayRef) -> VortexResult<ArrayRef> {
    BtrBlocksCompressor::default().compress(array, &mut SESSION.create_execution_ctx())
}

fn byte_parts(array: &ArrayRef) -> &ArrayRef {
    assert!(
        array.is::<DecimalByteParts>(),
        "expected DecimalByteParts, got {}",
        array.encoding_id()
    );
    array
}

fn lower_part_count(array: &ArrayRef) -> usize {
    byte_parts(array)
        .as_opt::<DecimalByteParts>()
        .vortex_expect("byte parts array")
        .lower_parts()
        .len()
}

#[rstest]
#[case::non_nullable(Validity::NonNullable)]
#[case::all_valid(Validity::AllValid)]
#[case::nullable(Validity::from_iter((0..N).map(|i| i % 3 != 0)))]
fn test_i128_decimal_splits_into_one_lower_part(#[case] validity: Validity) -> VortexResult<()> {
    let array = wide_i128_array(validity).into_array();
    let compressed = compress(&array)?;

    assert_eq!(lower_part_count(&compressed), 1);
    assert_eq!(compressed.dtype(), array.dtype());
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::non_nullable(Validity::NonNullable)]
#[case::all_valid(Validity::AllValid)]
#[case::nullable(Validity::from_iter((0..N).map(|i| i % 5 != 0)))]
fn test_i256_decimal_splits_into_three_lower_parts(#[case] validity: Validity) -> VortexResult<()> {
    let array = wide_i256_array(validity).into_array();
    let compressed = compress(&array)?;

    assert_eq!(lower_part_count(&compressed), MAX_LOWER_PARTS);
    assert_eq!(compressed.dtype(), array.dtype());
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_i256_decimal_round_trips_extreme_values() -> VortexResult<()> {
    // Every 64-bit window exercised, including the sign boundary of the most significant
    // part. Bounded by the precision so the values are legal `Decimal(76, 0)` scalars.
    let max = ten_pow(76) - i256::ONE;
    let values: Buffer<i256> = (0..N)
        .map(|i| match i % 8 {
            0 => i256::ZERO,
            1 => i256::ONE,
            2 => i256::ZERO - i256::ONE,
            3 => i256::from_parts(u128::MAX, 0),
            4 => i256::from_parts(0, 1),
            5 => i256::from_parts(0, -1),
            6 => max,
            _ => i256::ZERO - max,
        })
        .collect();
    let array =
        DecimalArray::new(values, DecimalDType::new(76, 0), Validity::NonNullable).into_array();

    let compressed = compress(&array)?;
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_narrow_decimal_has_no_lower_parts() -> VortexResult<()> {
    // Values that fit 64 bits are narrowed rather than split, even when the declared
    // precision needs an i256.
    let values: Buffer<i256> = (0..N as i128).map(|i| i256::from_i128(i * 3)).collect();
    let array =
        DecimalArray::new(values, DecimalDType::new(76, 2), Validity::NonNullable).into_array();

    let compressed = compress(&array)?;
    assert_eq!(lower_part_count(&compressed), 0);
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_canonical_of_compressed_wide_decimal_keeps_storage_width() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let array = wide_i128_array(Validity::NonNullable).into_array();
    let canonical = compress(&array)?.execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I128);

    let array = wide_i256_array(Validity::NonNullable).into_array();
    let canonical = compress(&array)?.execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I256);
    Ok(())
}

/// Splitting exists to make wide decimals compressible: the parts that do not vary collapse
/// to constants and the varying part bit-packs. Without splitting these arrays are stored as
/// raw 16- and 32-byte values.
#[rstest]
#[case::i128(wide_i128_array(Validity::NonNullable), 16)]
#[case::i256(wide_i256_array(Validity::NonNullable), 32)]
fn test_wide_decimals_compress(
    #[case] array: DecimalArray,
    #[case] uncompressed_bytes_per_value: usize,
) -> VortexResult<()> {
    let array = array.into_array();
    let uncompressed = u64::try_from(uncompressed_bytes_per_value * N)?;
    let compressed = compress(&array)?.nbytes();

    assert!(
        compressed * 4 < uncompressed,
        "expected at least 4x compression, got {uncompressed} -> {compressed} bytes"
    );
    Ok(())
}
