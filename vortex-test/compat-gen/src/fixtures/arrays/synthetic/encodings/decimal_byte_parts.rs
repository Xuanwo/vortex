// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::DecimalDType;
use vortex::array::dtype::FieldNames;
use vortex::array::dtype::i256;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::decimal_byte_parts::DecimalBytePartsArray;
use vortex::encodings::decimal_byte_parts::split_decimal;
use vortex::error::VortexResult;
use vortex_array::ExecutionCtx;

use super::N;
use crate::fixtures::FlatLayoutFixture;

/// Encode a canonical decimal as byte parts, splitting wide values into lower parts.
fn encode_byte_parts(decimal: &DecimalArray) -> VortexResult<DecimalBytePartsArray> {
    let parts = split_decimal(decimal)?;
    DecimalByteParts::try_new_with_lower_parts(
        parts.msp,
        parts.lower_parts,
        decimal.decimal_dtype(),
    )
}

pub struct DecimalBytePartsFixture;

impl FlatLayoutFixture for DecimalBytePartsFixture {
    fn name(&self) -> &str {
        "decimal_byte_parts.vortex"
    }

    fn description(&self) -> &str {
        "Fixed-precision decimal arrays for DecimalByteParts encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![DecimalByteParts.id()]
    }

    fn build(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let values: PrimitiveArray = (0..N as i64).map(|i| i * 100 + (i % 100)).collect();
        let msp_arr = values.into_array();
        let decimal_arr = DecimalByteParts::try_new(msp_arr, decimal_dtype)?;

        let hi_prec_dtype = DecimalDType::new(18, 6);
        let hi_prec_values: PrimitiveArray = (0..N as i64)
            .map(|i| i * 1_000_000 + (i * 7 % 999_999))
            .collect();
        let hi_prec_msp = hi_prec_values.into_array();
        let hi_prec_arr = DecimalByteParts::try_new(hi_prec_msp, hi_prec_dtype)?;

        let neg_dtype = DecimalDType::new(10, 2);
        let neg_values: PrimitiveArray = (0..N as i64).map(|i| -5000 + (i * 3 % 10000)).collect();
        let neg_msp = neg_values.into_array();
        let neg_arr = DecimalByteParts::try_new(neg_msp, neg_dtype)?;
        let nullable_dtype = DecimalDType::new(12, 4);
        let nullable_values = PrimitiveArray::from_option_iter((0..N as i64).map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some((i - 500) * 10_000)
            }
        }))
        .into_array();
        let nullable_arr = DecimalByteParts::try_new(nullable_values, nullable_dtype)?;
        let zero_dtype = DecimalDType::new(10, 2);
        let zero_arr = DecimalByteParts::try_new(
            std::iter::repeat_n(0i64, N)
                .collect::<PrimitiveArray>()
                .into_array(),
            zero_dtype,
        )?;
        let crossing_dtype = DecimalDType::new(12, 3);
        let crossing_values: PrimitiveArray = (0..N as i64).map(|i| (i % 200) - 100).collect();
        let crossing_arr = DecimalByteParts::try_new(crossing_values.into_array(), crossing_dtype)?;
        let trailing_zero_dtype = DecimalDType::new(18, 4);
        let trailing_zero_values: PrimitiveArray =
            (0..N as i64).map(|i| (i % 1000) * 10_000).collect();
        let trailing_zero_arr =
            DecimalByteParts::try_new(trailing_zero_values.into_array(), trailing_zero_dtype)?;
        let near_limit_dtype = DecimalDType::new(18, 0);
        let near_limit_values: PrimitiveArray =
            (0..N as i64).map(|i| 900_000_000_000_000_000 - i).collect();
        let near_limit_arr =
            DecimalByteParts::try_new(near_limit_values.into_array(), near_limit_dtype)?;

        // Wide decimals, split into an MSP plus 64-bit lower parts.
        let wide_128_dtype = DecimalDType::new(38, 2);
        let wide_128 = DecimalArray::new(
            (0..N as i128)
                .map(|i| 10i128.pow(25) + i * 7)
                .collect::<Buffer<i128>>(),
            wide_128_dtype,
            Validity::NonNullable,
        );
        let wide_128_arr = encode_byte_parts(&wide_128)?;

        let wide_256_dtype = DecimalDType::new(76, 2);
        let base = i256::from_i128(10).wrapping_pow(40);
        let wide_256 = DecimalArray::new(
            (0..N as i128)
                .map(|i| base + i256::from_i128(i * 7))
                .collect::<Buffer<i256>>(),
            wide_256_dtype,
            Validity::from_iter((0..N).map(|i| i % 7 != 0)),
        );
        let wide_256_arr = encode_byte_parts(&wide_256)?;

        let arr = StructArray::try_new(
            FieldNames::from([
                "dec_10_2",
                "dec_18_6",
                "dec_negative",
                "dec_nullable",
                "dec_zero",
                "dec_crossing",
                "dec_trailing_zero",
                "dec_near_limit",
                "dec_wide_128",
                "dec_wide_256_nullable",
            ]),
            vec![
                decimal_arr.into_array(),
                hi_prec_arr.into_array(),
                neg_arr.into_array(),
                nullable_arr.into_array(),
                zero_arr.into_array(),
                crossing_arr.into_array(),
                trailing_zero_arr.into_array(),
                near_limit_arr.into_array(),
                wide_128_arr.into_array(),
                wide_256_arr.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
