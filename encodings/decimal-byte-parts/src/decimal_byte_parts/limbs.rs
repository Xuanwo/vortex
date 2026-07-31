// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting decimal values into 64-bit parts, and reassembling them.
//!
//! A `DecimalByteParts` array stores each value as a signed most significant part (MSP)
//! followed by `k` unsigned 64-bit lower parts ordered most significant first. The encoded
//! value is
//!
//! ```text
//! msp * 2^(64k) + Σ_{i<k} lower[i] * 2^(64 * (k - 1 - i))
//! ```
//!
//! which is exactly the two's complement bit pattern of the decimal value cut on 64-bit
//! boundaries: the MSP holds the sign and the leading bits, every lower part holds a raw
//! 64-bit window of the magnitude.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// The maximum number of lower parts an encoded decimal can carry.
///
/// The most significant part is at most 64 bits wide, so three additional 64-bit parts
/// saturate the 256-bit maximum width of a Vortex decimal.
pub const MAX_LOWER_PARTS: usize = 3;

/// Number of bits stored in each lower part.
const LOWER_PART_BITS: usize = 64;

/// The dtype every lower part must have: a non-nullable `u64`.
///
/// Validity is carried by the most significant part alone.
pub const LOWER_PART_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);

/// A decimal array decomposed into byte parts.
pub struct DecimalParts {
    /// The signed most significant part, carrying the validity of the whole array.
    pub msp: ArrayRef,
    /// The unsigned 64-bit lower parts, most significant first.
    pub lower_parts: Vec<ArrayRef>,
}

/// The decimal storage type that reassembling the given parts produces.
///
/// # Errors
///
/// Returns an error if `msp_ptype` is not a signed integer, or if there are more than
/// [`MAX_LOWER_PARTS`] lower parts.
pub fn assembled_values_type(
    msp_ptype: PType,
    lower_part_count: usize,
) -> VortexResult<DecimalType> {
    if lower_part_count > MAX_LOWER_PARTS {
        vortex_bail!("at most {MAX_LOWER_PARTS} lower parts are supported, got {lower_part_count}");
    }
    if lower_part_count == 0 {
        return DecimalType::try_from(msp_ptype);
    }
    let bits = msp_ptype.bit_width() + LOWER_PART_BITS * lower_part_count;
    Ok(if bits <= 128 {
        DecimalType::I128
    } else {
        DecimalType::I256
    })
}

/// Split a canonical decimal array into a signed most significant part and unsigned 64-bit
/// lower parts.
///
/// Values narrower than 128 bits are already a single signed part, so they are returned
/// with no lower parts. `i128` values split into an `i64` MSP and one lower part, `i256`
/// values into an `i64` MSP and three lower parts.
///
/// # Errors
///
/// Returns an error if the array's validity cannot be derived.
pub fn split_decimal(decimal: &DecimalArray) -> VortexResult<DecimalParts> {
    let validity = decimal.validity()?;
    Ok(match decimal.values_type() {
        DecimalType::I8 => DecimalParts::flat(decimal.buffer::<i8>(), validity),
        DecimalType::I16 => DecimalParts::flat(decimal.buffer::<i16>(), validity),
        DecimalType::I32 => DecimalParts::flat(decimal.buffer::<i32>(), validity),
        DecimalType::I64 => DecimalParts::flat(decimal.buffer::<i64>(), validity),
        DecimalType::I128 => {
            let (msp, lower) = split_i128(&decimal.buffer::<i128>());
            DecimalParts::new(msp, [lower], validity)
        }
        DecimalType::I256 => {
            let (msp, lower) = split_i256(&decimal.buffer::<i256>());
            DecimalParts::new(msp, lower, validity)
        }
    })
}

/// Reassemble decimal byte parts into a canonical decimal array.
///
/// The parts must already be canonical primitive arrays: a signed MSP, and `u64` lower
/// parts ordered most significant first.
///
/// # Errors
///
/// Returns an error if the parts do not describe a valid decimal, or if the MSP's validity
/// cannot be derived.
pub fn assemble_decimal(
    msp: &PrimitiveArray,
    lower_parts: &[PrimitiveArray],
    decimal_dtype: DecimalDType,
) -> VortexResult<DecimalArray> {
    let validity = msp.validity()?;
    if lower_parts.is_empty() {
        return Ok(match_each_signed_integer_ptype!(msp.ptype(), |P| {
            // SAFETY: the buffer is typed by the array's own ptype, the decimal dtype is the
            // array's, and the validity is taken from the same array.
            unsafe { DecimalArray::new_unchecked(msp.to_buffer::<P>(), decimal_dtype, validity) }
        }));
    }

    // Slice every part to the MSP's length up front: the assembly loops then index slices the
    // compiler knows are long enough, so the per-row bounds checks fall away.
    let len = msp.len();
    let lower: Vec<&[u64]> = lower_parts
        .iter()
        .map(|part| {
            let part = part.as_slice::<u64>();
            vortex_ensure!(
                part.len() >= len,
                "lower part has len {}, expected at least {len}",
                part.len()
            );
            Ok(&part[..len])
        })
        .collect::<VortexResult<_>>()?;

    // The part count is dispatched to a constant so every 64-bit word lands at a compile-time
    // index. Leaving it dynamic costs 1.8x on the `i256` path — see `benches/decimal_assemble.rs`.
    let values = match assembled_values_type(msp.ptype(), lower.len())? {
        DecimalType::I256 => match lower.as_slice() {
            [first] => assemble_i256(msp, [first]),
            [first, second] => assemble_i256(msp, [first, second]),
            [first, second, third] => assemble_i256(msp, [first, second, third]),
            _ => vortex_bail!("unsupported lower part count {}", lower.len()),
        },
        _ => {
            return Ok(DecimalArray::new(
                assemble_i128(msp, lower[0]),
                decimal_dtype,
                validity,
            ));
        }
    };
    Ok(DecimalArray::new(values, decimal_dtype, validity))
}

/// Combine a single row's parts into an `i128`.
#[inline]
pub(crate) fn combine_i128(msp: i64, lower: impl IntoIterator<Item = u64>) -> i128 {
    lower.into_iter().fold(i128::from(msp), |acc, part| {
        (acc << LOWER_PART_BITS) | i128::from(part)
    })
}

/// Combine a single row's parts into an `i256`.
///
/// The lower parts fill the least significant 64-bit words, the MSP the word above them,
/// and the remaining high words are the MSP's sign extension.
#[inline]
pub(crate) fn combine_i256(msp: i64, lower: impl ExactSizeIterator<Item = u64>) -> i256 {
    let count = lower.len();
    let mut words = [if msp < 0 { u64::MAX } else { 0 }; 4];
    for (i, part) in lower.enumerate() {
        words[count - 1 - i] = part;
    }
    words[count] = msp.cast_unsigned();

    i256::from_parts(
        u128::from(words[0]) | (u128::from(words[1]) << LOWER_PART_BITS),
        (u128::from(words[2]) | (u128::from(words[3]) << LOWER_PART_BITS)).cast_signed(),
    )
}

impl DecimalParts {
    /// Parts for a decimal already stored in a single signed integer.
    fn flat<T: NativePType>(values: Buffer<T>, validity: Validity) -> Self {
        Self {
            msp: PrimitiveArray::new(values, validity).into_array(),
            lower_parts: Vec::new(),
        }
    }

    fn new(
        msp: Buffer<i64>,
        lower_parts: impl IntoIterator<Item = Buffer<u64>>,
        validity: Validity,
    ) -> Self {
        Self {
            msp: PrimitiveArray::new(msp, validity).into_array(),
            lower_parts: lower_parts
                .into_iter()
                .map(|part| PrimitiveArray::new(part, Validity::NonNullable).into_array())
                .collect(),
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "splitting a wide integer into 64-bit windows truncates by construction"
)]
fn split_i128(values: &Buffer<i128>) -> (Buffer<i64>, Buffer<u64>) {
    let mut msp = BufferMut::<i64>::with_capacity(values.len());
    let mut lower = BufferMut::<u64>::with_capacity(values.len());
    for value in values.iter() {
        msp.push((value >> LOWER_PART_BITS) as i64);
        lower.push(*value as u64);
    }
    (msp.freeze(), lower.freeze())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "splitting a wide integer into 64-bit windows truncates by construction"
)]
fn split_i256(values: &Buffer<i256>) -> (Buffer<i64>, [Buffer<u64>; MAX_LOWER_PARTS]) {
    let mut msp = BufferMut::<i64>::with_capacity(values.len());
    // Ordered most significant first: bits 191..128, 127..64, 63..0.
    let mut lower = std::array::from_fn::<_, MAX_LOWER_PARTS, _>(|_| {
        BufferMut::<u64>::with_capacity(values.len())
    });
    for value in values.iter() {
        let (low, high) = value.to_parts();
        msp.push((high >> LOWER_PART_BITS) as i64);
        lower[0].push(high as u64);
        lower[1].push((low >> LOWER_PART_BITS) as u64);
        lower[2].push(low as u64);
    }
    (msp.freeze(), lower.map(BufferMut::freeze))
}

/// Only one lower part can share 128 bits with a signed MSP, so this shape is fixed.
#[expect(
    clippy::useless_conversion,
    reason = "the widening to i64 is a no-op only for the i64 arm of the ptype match"
)]
fn assemble_i128(msp: &PrimitiveArray, lower: &[u64]) -> Buffer<i128> {
    let mut out = BufferMut::<i128>::with_capacity(msp.len());
    match_each_signed_integer_ptype!(msp.ptype(), |P| {
        for (value, part) in msp.as_slice::<P>().iter().zip(lower) {
            out.push((i128::from(i64::from(*value)) << LOWER_PART_BITS) | i128::from(*part));
        }
    });
    out.freeze()
}

/// The lower parts fill the least significant 64-bit words, the MSP the word above them, and
/// the remaining high words are the MSP's sign extension.
///
/// `K` is a constant so the word indices are compile-time constants and the placement loop
/// unrolls; the same loop with a runtime part count is 1.8x slower.
#[expect(
    clippy::useless_conversion,
    reason = "the widening to i64 is a no-op only for the i64 arm of the ptype match"
)]
fn assemble_i256<const K: usize>(msp: &PrimitiveArray, lower: [&[u64]; K]) -> Buffer<i256> {
    let mut out = BufferMut::<i256>::with_capacity(msp.len());
    match_each_signed_integer_ptype!(msp.ptype(), |P| {
        for (row, value) in msp.as_slice::<P>().iter().enumerate() {
            let value = i64::from(*value);
            let mut words = [if value < 0 { u64::MAX } else { 0 }; 4];
            for (i, part) in lower.iter().enumerate() {
                words[K - 1 - i] = part[row];
            }
            words[K] = value.cast_unsigned();
            out.push(i256::from_parts(
                u128::from(words[0]) | (u128::from(words[1]) << LOWER_PART_BITS),
                (u128::from(words[2]) | (u128::from(words[3]) << LOWER_PART_BITS)).cast_signed(),
            ));
        }
    });
    out.freeze()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::i256;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    fn round_trip(decimal: DecimalArray) -> VortexResult<DecimalArray> {
        let mut ctx = array_session().create_execution_ctx();
        let parts = split_decimal(&decimal)?;
        let msp = parts.msp.execute::<PrimitiveArray>(&mut ctx)?;
        let lower = parts
            .lower_parts
            .into_iter()
            .map(|part| part.execute::<PrimitiveArray>(&mut ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        assemble_decimal(&msp, &lower, decimal.decimal_dtype())
    }

    #[rstest]
    #[case::zero(0)]
    #[case::one(1)]
    #[case::minus_one(-1)]
    #[case::limb_boundary(1i128 << 64)]
    #[case::just_below_limb_boundary((1i128 << 64) - 1)]
    #[case::negative_limb_boundary(-(1i128 << 64))]
    #[case::max(i128::MAX)]
    #[case::min(i128::MIN)]
    fn test_split_assemble_i128(#[case] value: i128) -> VortexResult<()> {
        let decimal = DecimalArray::new(
            Buffer::from(vec![value]),
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        );
        let round_tripped = round_trip(decimal)?;
        assert_eq!(round_tripped.buffer::<i128>().as_slice(), &[value]);
        Ok(())
    }

    #[rstest]
    #[case::zero(i256::ZERO)]
    #[case::one(i256::ONE)]
    #[case::minus_one(i256::ZERO - i256::ONE)]
    #[case::max(i256::MAX)]
    #[case::min(i256::MIN)]
    #[case::word_1(i256::from_parts(1u128 << 64, 0))]
    #[case::word_2(i256::from_parts(0, 1))]
    #[case::word_3(i256::from_parts(0, 1i128 << 64))]
    #[case::mixed(i256::from_parts(u128::MAX, -3))]
    fn test_split_assemble_i256(#[case] value: i256) -> VortexResult<()> {
        let decimal = DecimalArray::new(
            Buffer::from(vec![value]),
            DecimalDType::new(76, 2),
            Validity::NonNullable,
        );
        let round_tripped = round_trip(decimal)?;
        assert_eq!(round_tripped.buffer::<i256>().as_slice(), &[value]);
        Ok(())
    }

    #[test]
    fn test_split_narrow_decimal_has_no_lower_parts() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![1i32, 2, 3],
            DecimalDType::new(9, 2),
            Validity::NonNullable,
        );
        let parts = split_decimal(&decimal)?;
        assert!(parts.lower_parts.is_empty());
        assert_eq!(parts.msp.dtype().as_ptype(), PType::I32);
        Ok(())
    }

    #[test]
    fn test_split_i256_part_count_and_types() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            Buffer::from(vec![i256::from_i128(i128::MAX), i256::MIN]),
            DecimalDType::new(76, 0),
            Validity::NonNullable,
        );
        let parts = split_decimal(&decimal)?;
        assert_eq!(parts.lower_parts.len(), MAX_LOWER_PARTS);
        assert_eq!(parts.msp.dtype().as_ptype(), PType::I64);
        for part in &parts.lower_parts {
            assert_eq!(part.dtype(), &LOWER_PART_DTYPE);
        }
        Ok(())
    }

    #[test]
    fn test_assembled_values_type() -> VortexResult<()> {
        assert_eq!(assembled_values_type(PType::I32, 0)?, DecimalType::I32);
        assert_eq!(assembled_values_type(PType::I64, 1)?, DecimalType::I128);
        assert_eq!(assembled_values_type(PType::I8, 1)?, DecimalType::I128);
        assert_eq!(assembled_values_type(PType::I8, 2)?, DecimalType::I256);
        assert_eq!(assembled_values_type(PType::I64, 3)?, DecimalType::I256);
        assert!(assembled_values_type(PType::I64, 4).is_err());
        Ok(())
    }
}
