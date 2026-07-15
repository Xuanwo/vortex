// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

/// A row limit shared by the streams executing one scan's independent partitions.
///
/// The single budget is claimed in completion order, not row order, so the combined output is
/// "any `limit` rows", not "the first `limit` in scan order": under concurrency a later partition
/// can drain the budget and starve an earlier one. Only sound for unordered consumers (a bare
/// `LIMIT n`); order-preserving consumers must use a per-partition `ScanBuilder::with_limit`.
#[derive(Clone)]
pub(crate) struct SharedRowLimit(Arc<AtomicU64>);

impl SharedRowLimit {
    pub(crate) fn new(limit: u64) -> Self {
        Self(Arc::new(AtomicU64::new(limit)))
    }

    fn reserve(&self, requested: u64) -> (u64, bool) {
        let mut remaining = self.0.load(Ordering::Relaxed);
        loop {
            let reserved = remaining.min(requested);
            match self.0.compare_exchange_weak(
                remaining,
                remaining - reserved,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return (reserved, reserved == remaining),
                Err(actual) => remaining = actual,
            }
        }
    }

    fn is_exhausted(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 0
    }
}

/// The remaining rows a [`LimitedStream`] may emit, either privately or shared across streams.
pub(crate) enum RowBudget {
    /// A budget private to a single stream.
    Local(u64),
    /// A budget shared across streams executing independent scan partitions.
    Shared(SharedRowLimit),
}

impl RowBudget {
    /// Reserve up to `requested` rows from the budget.
    ///
    /// Returns `(reserved, exhausted)` where `reserved <= requested` and `exhausted` is true
    /// once the budget has reached zero.
    fn reserve(&mut self, requested: u64) -> (u64, bool) {
        match self {
            RowBudget::Local(remaining) => {
                let reserved = (*remaining).min(requested);
                *remaining -= reserved;
                (reserved, *remaining == 0)
            }
            RowBudget::Shared(shared) => shared.reserve(requested),
        }
    }

    fn is_exhausted(&self) -> bool {
        match self {
            RowBudget::Local(remaining) => *remaining == 0,
            RowBudget::Shared(shared) => shared.is_exhausted(),
        }
    }
}

/// Wraps a stream, emitting chunks until its [`RowBudget`] is exhausted, then terminating.
pub(crate) struct LimitedStream {
    inner: BoxStream<'static, VortexResult<ArrayRef>>,
    budget: RowBudget,
}

impl LimitedStream {
    pub(crate) fn new(
        inner: BoxStream<'static, VortexResult<ArrayRef>>,
        budget: RowBudget,
    ) -> Self {
        Self { inner, budget }
    }

    /// Drop the inner stream so no further work (including spawned split tasks) is polled.
    fn abort_pending(&mut self) {
        self.inner = stream::empty().boxed();
    }
}

impl Stream for LimitedStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Avoid reading a chunk we have no budget for. For a shared budget this also stops a
        // partition whose siblings already consumed the limit.
        if self.budget.is_exhausted() {
            self.abort_pending();
            return Poll::Ready(None);
        }

        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let chunk_len = chunk.len() as u64;
                let (reserved, exhausted) = self.budget.reserve(chunk_len);

                if exhausted {
                    self.abort_pending();
                }

                if reserved == 0 {
                    // Either the budget was already exhausted (stop), or this is an empty chunk
                    // while the budget still has room (pass it through).
                    if exhausted {
                        Poll::Ready(None)
                    } else {
                        Poll::Ready(Some(Ok(chunk)))
                    }
                } else if reserved == chunk_len {
                    Poll::Ready(Some(Ok(chunk)))
                } else {
                    let limit = usize::try_from(reserved)
                        .vortex_expect("reserved rows are bounded by the chunk length");
                    Poll::Ready(Some(chunk.slice(0..limit)))
                }
            }
            other => other,
        }
    }
}
