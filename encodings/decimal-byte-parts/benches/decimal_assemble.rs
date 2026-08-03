// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassembling `DecimalByteParts` into canonical `i128`/`i256` values.
//!
//! Canonicalizing a wide decimal walks a most significant part plus one (`i128`) or three
//! (`i256`) unsigned 64-bit lower parts, and has to produce one wide value per row. The
//! obvious shapes are:
//!
//! - **row**: one pass, gathering the row's word from each part and combining. Sub-variants
//!   differ in whether the part count is known to the compiler (`_const` vs `_dynamic`) and
//!   in whether the output is pushed into a reserved buffer or stored into a pre-sized one
//!   (`_write`).
//! - **column**: one pass per part over the whole output, writing 64-bit lanes directly.
//!
//! Every candidate is spelled out here rather than called through the crate, so the same
//! comparison can be run from any revision. `canonicalize_byte_parts` goes through the array
//! API instead, and so tracks whichever shape the crate currently ships.
//!
//! At 65,536 rows the row shape wins, but for two different reasons per width, and neither
//! is the row-at-a-time access itself:
//!
//! - `i256` is dominated by the part count being invisible to the compiler. Specializing it
//!   is 1.85x. How the output is written barely matters (`_write` ties `_const`), because at
//!   32 bytes per row the stores dominate either way.
//! - `i128` is dominated by the write. Specializing the part count is worth only ~1.04x,
//!   while storing into a pre-sized buffer instead of pushing is 1.6x — the bounds-checked
//!   `push` is the whole cost at 16 bytes per row.
//!
//! Columnar always loses. For `i256` each lane store is strided by 32 bytes, 2.3x slower than
//! the specialized row loop. For `i128` the two-pass column shape beats the *pushing* row
//! loop but still loses to the single-pass `_write` row loop, so the two passes buy nothing
//! once the push is gone.
//!
//! Two further `i256` columnar variants were measured and then removed rather than left here
//! to rot: cache blocking the lane passes over 1024-row blocks recovered part of the strided
//! stores but was still 1.6x slower than the row loop, and expressing the passes as
//! whole-value `i256` shifts was 11x slower.
//!
//! # Hand-written 64-bit words vs `u128` packing
//!
//! `i256::from_parts` takes a `u128` and an `i128`, so the assembly loops end each row with
//! `u128::from(w0) | (u128::from(w1) << 64)`. The reflex is that this must be worse than
//! storing four `u64`s by hand, because 128-bit integers lower badly. `i256_row_words` is
//! that hand-written version, and it ties `i256_row_const` across runs.
//!
//! Disassembling the release build says why: neither shape emits a single `shld`/`shrd`, and
//! both compile to four plain 64-bit stores per row at offsets 0x0/0x8/0x10/0x18. The `i128`
//! loop is the same — `(i128::from(msp) << 64) | i128::from(part)` becomes two 64-bit stores.
//! A shift by a constant multiple of 64 followed by an or is pure data movement, and LLVM
//! recognizes it. The 128-bit codegen worth avoiding is division/remainder, which call into
//! compiler-rt, and shifts by a runtime amount; neither appears here.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

fn main() {
    divan::main();
}

/// Rows per benchmark: a typical scan chunk, and large enough that the output does not fit
/// in L2, so the extra passes of a columnar shape are paid at their real cost.
const LEN: usize = 65_536;

const WORD_BITS: usize = 64;

/// 64-bit words in the widest decimal value (`i256`).
const MAX_VALUE_WORDS: usize = 4;

/// Deterministic pseudo-random words, so no part is constant or a sequence.
fn words(seed: u64, len: usize) -> Buffer<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random()).collect()
}

fn msp(seed: u64, len: usize) -> Buffer<i64> {
    words(seed, len)
        .iter()
        .map(|w| (w >> 40).cast_signed())
        .collect()
}

// ---------------------------------------------------------------------------------------
// i128: one lower part
// ---------------------------------------------------------------------------------------

/// The row shape with the part count only known at runtime: a fold over an iterator of the
/// row's words.
fn i128_row_dynamic(msp: &[i64], lower: &[&[u64]]) -> Buffer<i128> {
    let mut out = BufferMut::<i128>::with_capacity(msp.len());
    for (row, m) in msp.iter().enumerate() {
        out.push(lower.iter().fold(i128::from(*m), |acc, part| {
            (acc << WORD_BITS) | i128::from(part[row])
        }));
    }
    out.freeze()
}

/// The row shape specialized to exactly one lower part.
fn i128_row_const(msp: &[i64], lower: &[u64]) -> Buffer<i128> {
    let mut out = BufferMut::<i128>::with_capacity(msp.len());
    for (m, l) in msp.iter().zip(lower) {
        out.push((i128::from(*m) << WORD_BITS) | i128::from(*l));
    }
    out.freeze()
}

/// The column shape: one pass writes the most significant half of every value, a second
/// pass ORs in the lower part.
fn i128_column(msp: &[i64], lower: &[u64]) -> Buffer<i128> {
    let mut out = BufferMut::<i128>::zeroed(msp.len());
    for (o, m) in out.as_mut_slice().iter_mut().zip(msp) {
        *o = i128::from(*m) << WORD_BITS;
    }
    for (o, l) in out.as_mut_slice().iter_mut().zip(lower) {
        *o |= i128::from(*l);
    }
    out.freeze()
}

/// The row shape writing into a pre-sized buffer rather than pushing into a reserved one,
/// to separate "one pass vs two" from "bounds-checked push vs direct store".
fn i128_row_write(msp: &[i64], lower: &[u64]) -> Buffer<i128> {
    let mut out = BufferMut::<i128>::zeroed(msp.len());
    for ((o, m), l) in out.as_mut_slice().iter_mut().zip(msp).zip(lower) {
        *o = (i128::from(*m) << WORD_BITS) | i128::from(*l);
    }
    out.freeze()
}

// ---------------------------------------------------------------------------------------
// i256: three lower parts
// ---------------------------------------------------------------------------------------

/// The row shape with a runtime part count: per row, a stack array of words is filled at
/// dynamic indices and then packed.
fn i256_row_dynamic(msp: &[i64], lower: &[&[u64]]) -> Buffer<i256> {
    let count = lower.len();
    let mut out = BufferMut::<i256>::with_capacity(msp.len());
    for (row, m) in msp.iter().enumerate() {
        let mut w = [if *m < 0 { u64::MAX } else { 0 }; 4];
        for (i, part) in lower.iter().enumerate() {
            w[count - 1 - i] = part[row];
        }
        w[count] = m.cast_unsigned();
        out.push(i256::from_parts(
            u128::from(w[0]) | (u128::from(w[1]) << WORD_BITS),
            (u128::from(w[2]) | (u128::from(w[3]) << WORD_BITS)).cast_signed(),
        ));
    }
    out.freeze()
}

/// The row shape specialized to exactly three lower parts: every word index is a constant.
fn i256_row_const(msp: &[i64], lower: [&[u64]; 3]) -> Buffer<i256> {
    let mut out = BufferMut::<i256>::with_capacity(msp.len());
    for row in 0..msp.len() {
        out.push(i256::from_parts(
            u128::from(lower[2][row]) | (u128::from(lower[1][row]) << WORD_BITS),
            (u128::from(lower[0][row]) | (u128::from(msp[row].cast_unsigned()) << WORD_BITS))
                .cast_signed(),
        ));
    }
    out.freeze()
}

/// The column shape as lane writes: build the output as 64-bit words and write one lane per
/// pass. Avoids re-reading the output, but every store is strided by 32 bytes.
fn i256_column_lanes(msp: &[i64], lower: [&[u64]; 3]) -> Buffer<i256> {
    let len = msp.len();
    let mut w = BufferMut::<u64>::zeroed_aligned(len * 4, Alignment::of::<i256>());
    let lanes = w.as_mut_slice();
    for (i, m) in msp.iter().enumerate() {
        lanes[i * 4 + 3] = m.cast_unsigned();
    }
    for (lane, part) in lower.iter().enumerate() {
        for (i, word) in part.iter().enumerate() {
            lanes[i * 4 + 2 - lane] = *word;
        }
    }
    // Word order within an `i256` is ascending significance on a little-endian host.
    assert!(cfg!(target_endian = "little"));
    Buffer::<i256>::from_byte_buffer_aligned(w.freeze().into_byte_buffer(), Alignment::of::<i256>())
}

/// The specialized row shape emitting raw 64-bit words: the output is built as a `u64` lane
/// buffer and reinterpreted as `i256` at the end, so no 128-bit shift or or is ever written.
///
/// This exists to answer "shouldn't we avoid `u128` entirely, since LLVM handles 128-bit
/// types badly?" — it does not, for this pattern. See the module docs.
fn i256_row_words<const K: usize>(msp: &[i64], lower: [&[u64]; K]) -> Buffer<i256> {
    let len = msp.len();
    let mut w = BufferMut::<u64>::zeroed_aligned(len * MAX_VALUE_WORDS, Alignment::of::<i256>());
    let lanes = w.as_mut_slice();
    for (row, m) in msp.iter().enumerate() {
        let mut words = [if *m < 0 { u64::MAX } else { 0 }; MAX_VALUE_WORDS];
        for (i, part) in lower.iter().enumerate() {
            words[K - 1 - i] = part[row];
        }
        words[K] = m.cast_unsigned();
        lanes[row * MAX_VALUE_WORDS..(row + 1) * MAX_VALUE_WORDS].copy_from_slice(&words);
    }
    // Word order within an `i256` is ascending significance on a little-endian host.
    assert!(cfg!(target_endian = "little"));
    Buffer::<i256>::from_byte_buffer_aligned(w.freeze().into_byte_buffer(), Alignment::of::<i256>())
}

/// The specialized row shape for `i256`, writing into a pre-sized buffer.
fn i256_row_write<const K: usize>(msp: &[i64], lower: [&[u64]; K]) -> Buffer<i256> {
    let mut out = BufferMut::<i256>::zeroed(msp.len());
    for (row, (o, m)) in out.as_mut_slice().iter_mut().zip(msp).enumerate() {
        let mut words = [if *m < 0 { u64::MAX } else { 0 }; 4];
        for (i, part) in lower.iter().enumerate() {
            words[K - 1 - i] = part[row];
        }
        words[K] = m.cast_unsigned();
        *o = i256::from_parts(
            u128::from(words[0]) | (u128::from(words[1]) << WORD_BITS),
            (u128::from(words[2]) | (u128::from(words[3]) << WORD_BITS)).cast_signed(),
        );
    }
    out.freeze()
}

// ---------------------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------------------

struct Parts {
    msp: Buffer<i64>,
    lower: Vec<Buffer<u64>>,
}

impl Parts {
    fn new(lower_parts: usize) -> Self {
        Self {
            msp: msp(1, LEN),
            lower: (0..lower_parts).map(|i| words(7 + i as u64, LEN)).collect(),
        }
    }

    fn lower_slices(&self) -> Vec<&[u64]> {
        self.lower.iter().map(|part| part.as_slice()).collect()
    }

    fn arrays(&self) -> (PrimitiveArray, Vec<PrimitiveArray>) {
        (
            PrimitiveArray::new(self.msp.clone(), Validity::NonNullable),
            self.lower
                .iter()
                .map(|part| PrimitiveArray::new(part.clone(), Validity::NonNullable))
                .collect(),
        )
    }
}

#[divan::bench]
fn i128_row_dynamic_parts(bencher: Bencher) {
    let parts = Parts::new(1);
    let lower = parts.lower_slices();
    bencher.bench(|| i128_row_dynamic(black_box(parts.msp.as_slice()), black_box(&lower)));
}

#[divan::bench]
fn i128_row_const_parts(bencher: Bencher) {
    let parts = Parts::new(1);
    let lower = parts.lower_slices();
    bencher.bench(|| i128_row_const(black_box(parts.msp.as_slice()), black_box(lower[0])));
}

#[divan::bench]
fn i128_column_parts(bencher: Bencher) {
    let parts = Parts::new(1);
    let lower = parts.lower_slices();
    bencher.bench(|| i128_column(black_box(parts.msp.as_slice()), black_box(lower[0])));
}

#[divan::bench]
fn i128_row_write_parts(bencher: Bencher) {
    let parts = Parts::new(1);
    let lower = parts.lower_slices();
    bencher.bench(|| i128_row_write(black_box(parts.msp.as_slice()), black_box(lower[0])));
}

#[divan::bench]
fn i256_row_dynamic_parts(bencher: Bencher) {
    let parts = Parts::new(3);
    let lower = parts.lower_slices();
    bencher.bench(|| i256_row_dynamic(black_box(parts.msp.as_slice()), black_box(&lower)));
}

#[divan::bench]
fn i256_row_const_parts(bencher: Bencher) {
    let parts = Parts::new(3);
    let lower = parts.lower_slices();
    let lower = [lower[0], lower[1], lower[2]];
    bencher.bench(|| i256_row_const(black_box(parts.msp.as_slice()), black_box(lower)));
}

#[divan::bench]
fn i256_column_lanes_parts(bencher: Bencher) {
    let parts = Parts::new(3);
    let lower = parts.lower_slices();
    let lower = [lower[0], lower[1], lower[2]];
    bencher.bench(|| i256_column_lanes(black_box(parts.msp.as_slice()), black_box(lower)));
}

/// Canonicalizing through the public array API, so the child execution and validity handling
/// around the assembly loop are included.
#[divan::bench]
fn i256_row_write_parts(bencher: Bencher) {
    let parts = Parts::new(3);
    let lower = parts.lower_slices();
    let lower = [lower[0], lower[1], lower[2]];
    bencher.bench(|| i256_row_write(black_box(parts.msp.as_slice()), black_box(lower)));
}

#[divan::bench]
fn i256_row_words_parts(bencher: Bencher) {
    let parts = Parts::new(3);
    let lower = parts.lower_slices();
    let lower = [lower[0], lower[1], lower[2]];
    bencher.bench(|| i256_row_words(black_box(parts.msp.as_slice()), black_box(lower)));
}

#[divan::bench(args = [1, 3])]
fn canonicalize_byte_parts(bencher: Bencher, lower_parts: usize) {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DecimalArray;
    use vortex_decimal_byte_parts::DecimalByteParts;

    let parts = Parts::new(lower_parts);
    let (msp, lower) = parts.arrays();
    let dtype = if lower_parts == 1 {
        DecimalDType::new(38, 2)
    } else {
        DecimalDType::new(76, 2)
    };
    let array = DecimalByteParts::try_new_with_lower_parts(
        msp.into_array(),
        lower.into_iter().map(IntoArray::into_array).collect(),
        dtype,
    )
    .unwrap()
    .into_array();

    let session = array_session();
    bencher
        .with_inputs(|| session.create_execution_ctx())
        .bench_refs(|ctx| {
            black_box(array.clone())
                .execute::<DecimalArray>(ctx)
                .unwrap()
        });
}
