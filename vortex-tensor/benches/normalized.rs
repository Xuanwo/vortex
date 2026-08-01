// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for decoding the `Normalized` encoding over tensor columns.
//!
//! The arms vary vector width and input nullability. Their names are intended to remain stable
//! across implementation changes so CodSpeed can compare them against `develop`.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_tensor::encodings::normalized::Normalized;
use vortex_tensor::vector::Vector;

// Decoding allocates the output inside the timed region, so use the vendored allocator instead
// of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const ROWS: usize = 16_384;
const WIDTHS: &[usize] = &[2, 32, 256];

fn normalized_vectors(width: usize) -> ArrayRef {
    let value = 1.0 / (width as f64).sqrt();
    let elements: Buffer<f64> = (0..ROWS * width).map(|_| value).collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        ROWS,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn norms() -> ArrayRef {
    let values: Buffer<f64> = (0..ROWS).map(|i| 1.0 + ((i % 13) as f64) / 13.0).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

fn bench_normalized(bencher: Bencher, normalized: ArrayRef) {
    let session = vortex_array::array_session();
    let norms = norms();
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| {
            let mut ctx = session.create_execution_ctx();
            let array = Normalized::try_new(normalized.clone(), norms.clone(), &mut ctx).unwrap();
            (array, ctx)
        })
        .bench_values(|(array, mut ctx)| {
            array
                .into_array()
                .execute::<ExtensionArray>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_normalized(bencher, normalized_vectors(width));
}

#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    let validity = Validity::from_iter((0..ROWS).map(|i| i % 8 != 0));
    let normalized = MaskedArray::try_new(normalized_vectors(width), validity)
        .unwrap()
        .into_array();
    bench_normalized(bencher, normalized);
}
