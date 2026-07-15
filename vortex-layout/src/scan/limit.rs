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
use vortex_error::VortexResult;

pub(crate) fn limit_array_stream<S>(
    stream: S,
    limit: Option<u64>,
) -> BoxStream<'static, VortexResult<ArrayRef>>
where
    S: Stream<Item = VortexResult<ArrayRef>> + Send + 'static,
{
    match limit {
        Some(limit) => RowLimitedStream::new(stream.boxed(), limit).boxed(),
        None => stream.boxed(),
    }
}

/// A row limit shared by streams that execute independent scan partitions.
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
}

pub(crate) fn limit_array_stream_shared<S>(
    stream: S,
    limit: Option<SharedRowLimit>,
) -> BoxStream<'static, VortexResult<ArrayRef>>
where
    S: Stream<Item = VortexResult<ArrayRef>> + Send + 'static,
{
    match limit {
        Some(limit) => SharedRowLimitedStream::new(stream.boxed(), limit).boxed(),
        None => stream.boxed(),
    }
}

struct RowLimitedStream {
    inner: BoxStream<'static, VortexResult<ArrayRef>>,
    remaining: u64,
}

impl RowLimitedStream {
    fn new(inner: BoxStream<'static, VortexResult<ArrayRef>>, remaining: u64) -> Self {
        Self { inner, remaining }
    }

    fn abort_pending(&mut self) {
        let inner = std::mem::replace(&mut self.inner, stream::empty().boxed());
        drop(inner);
    }
}

impl Stream for RowLimitedStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.remaining == 0 {
            return Poll::Ready(None);
        }

        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let chunk_len = chunk.len() as u64;
                if chunk_len <= self.remaining {
                    self.remaining -= chunk_len;
                    if self.remaining == 0 {
                        self.abort_pending();
                    }
                    Poll::Ready(Some(Ok(chunk)))
                } else {
                    let limit = match usize::try_from(self.remaining) {
                        Ok(limit) => limit,
                        Err(_) => unreachable!("remaining rows cannot exceed the current chunk"),
                    };
                    self.remaining = 0;
                    self.abort_pending();
                    Poll::Ready(Some(chunk.slice(0..limit)))
                }
            }
            other => other,
        }
    }
}

struct SharedRowLimitedStream {
    inner: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: SharedRowLimit,
}

impl SharedRowLimitedStream {
    fn new(inner: BoxStream<'static, VortexResult<ArrayRef>>, limit: SharedRowLimit) -> Self {
        Self { inner, limit }
    }

    fn abort_pending(&mut self) {
        let inner = std::mem::replace(&mut self.inner, stream::empty().boxed());
        drop(inner);
    }
}

impl Stream for SharedRowLimitedStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let chunk_len = chunk.len() as u64;
                let (reserved, exhausted) = self.limit.reserve(chunk_len);

                if exhausted {
                    self.abort_pending();
                }

                if reserved == 0 {
                    if exhausted {
                        Poll::Ready(None)
                    } else {
                        Poll::Ready(Some(Ok(chunk)))
                    }
                } else if reserved == chunk_len {
                    Poll::Ready(Some(Ok(chunk)))
                } else {
                    let limit = match usize::try_from(reserved) {
                        Ok(limit) => limit,
                        Err(_) => unreachable!("reserved rows cannot exceed the current chunk"),
                    };
                    self.abort_pending();
                    Poll::Ready(Some(chunk.slice(0..limit)))
                }
            }
            other => other,
        }
    }
}
