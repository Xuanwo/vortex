// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::future::Future;
use std::ops::BitAnd;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::limit::RowLimit;

/// The result of a split task.
///
/// Filter errors happen before a row limit reserves any rows, so callers may report them and
/// continue with later splits. Projection errors after reservation cannot safely release rows back
/// to a concurrent limit, so callers must report them and terminate the limited scan.
pub(crate) enum TaskResult {
    /// A completed projection, or an empty split.
    Array(Option<ArrayRef>),
    /// An error that occurred before a row limit reserved rows.
    Recoverable(VortexError),
    /// An error that occurred after a row limit reserved rows.
    Terminal(VortexError),
}

/// A future that executes one split and classifies any failure by whether it happened before or
/// after a row-limit reservation.
#[must_use = "split tasks must be scheduled or awaited"]
pub(crate) struct TaskFuture {
    inner: BoxFuture<'static, TaskResult>,
}

#[derive(Clone, Copy)]
enum ErrorDisposition {
    Recoverable,
    Terminal,
}

impl ErrorDisposition {
    fn result(self, error: VortexError) -> TaskResult {
        match self {
            Self::Recoverable => TaskResult::Recoverable(error),
            Self::Terminal => TaskResult::Terminal(error),
        }
    }
}

impl TaskFuture {
    pub(crate) fn new(future: impl Future<Output = TaskResult> + Send + 'static) -> Self {
        Self {
            inner: future.boxed(),
        }
    }

    fn ready(result: TaskResult) -> Self {
        Self::new(futures::future::ready(result))
    }

    pub(crate) fn empty() -> Self {
        Self::ready(TaskResult::Array(None))
    }

    pub(crate) fn recoverable(error: VortexError) -> Self {
        Self::ready(TaskResult::Recoverable(error))
    }

    pub(crate) fn terminal(error: VortexError) -> Self {
        Self::ready(TaskResult::Terminal(error))
    }

    fn projection(projection: ArrayFuture, errors: ErrorDisposition) -> Self {
        Self::new(async move {
            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(error) => errors.result(error),
            }
        })
    }

    fn filtered_projection(filter_mask: MaskFuture, projection: ArrayFuture) -> Self {
        Self::new(async move {
            let mask = match filter_mask.await {
                Ok(mask) => mask,
                Err(error) => return TaskResult::Recoverable(error),
            };
            if mask.all_false() {
                return TaskResult::Array(None);
            }

            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(error) => TaskResult::Recoverable(error),
            }
        })
    }

    fn limited_filtered_projection(
        ctx: Arc<TaskContext>,
        row_range: Range<u64>,
        filter_mask: MaskFuture,
        row_limit: RowLimit,
    ) -> Self {
        Self::new(async move {
            let mask = match filter_mask.await {
                Ok(mask) => mask,
                Err(error) => return TaskResult::Recoverable(error),
            };
            // A filter error above returns before reserving any rows. Once filtering has
            // succeeded, reserve only the matching rows and defer projection construction until
            // this task is polled by the runtime.
            let mask = row_limit.limit(mask);
            if mask.all_false() {
                return TaskResult::Array(None);
            }

            Self::deferred_projection(ctx, row_range, mask).await
        })
    }

    fn deferred_projection(ctx: Arc<TaskContext>, row_range: Range<u64>, mask: Mask) -> Self {
        Self::new(async move {
            let projection = match ctx.reader.projection_evaluation(
                &row_range,
                &ctx.projection,
                MaskFuture::ready(mask),
            ) {
                Ok(projection) => projection,
                Err(error) => return TaskResult::Terminal(error),
            };
            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(error) => TaskResult::Terminal(error),
            }
        })
    }
}

impl Future for TaskFuture {
    type Output = TaskResult;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

pub(crate) struct Plan {
    layout_reader: LayoutReaderRef,
    projection: Expression,
    filter: Option<Expression>,
}

impl Plan {
    pub(crate) fn new(
        layout_reader: LayoutReaderRef,
        projection: Expression,
        filter: Option<Expression>,
    ) -> Self {
        Self {
            layout_reader,
            projection,
            filter,
        }
    }

    pub(crate) fn task_context(&self) -> Arc<TaskContext> {
        Arc::new(TaskContext {
            filter: self
                .filter
                .clone()
                .map(|filter| Arc::new(FilterExpr::new(filter))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
        })
    }

    pub(crate) fn has_filter(&self) -> bool {
        self.filter.is_some()
    }
}

/// Logic for executing a single split reading task.
/// N.B. read_mask should be evaluated against all_false() before calling this
/// method to avoid creating an empty TaskFuture.
///
/// # Task execution flow
///
/// First, the task's row range (split) is intersected with the global file row-range requested,
/// if any.
///
/// The intersected row range is then further reduced via expression-based pruning. After pruning
/// has eliminated more blocks, the full filter is executed over the remainder of the split.
///
/// The final mask is limited before it is given to the reader to perform a filtered projection
/// over the split data, yielding the projected array (or `None` when the split selects no rows).
/// Limiting before projection prevents decode work for rows that the scan cannot return.
pub(crate) fn split_exec(
    ctx: Arc<TaskContext>,
    read_mask: RowMask,
    row_limit: Option<RowLimit>,
) -> VortexResult<TaskFuture> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();

    let Some(filter) = ctx.filter.as_ref() else {
        let limited = row_limit.is_some();
        let row_mask = if let Some(limit) = row_limit {
            limit.limit(row_mask)
        } else {
            row_mask
        };
        if row_mask.all_false() {
            return Ok(TaskFuture::empty());
        }

        // With no filter, limit the selection before constructing projection work.
        let projection = match ctx.reader.projection_evaluation(
            &row_range,
            &ctx.projection,
            MaskFuture::ready(row_mask),
        ) {
            Ok(projection) => projection,
            Err(err) if limited => return Ok(TaskFuture::terminal(err)),
            Err(err) => return Err(err),
        };
        let errors = if limited {
            ErrorDisposition::Terminal
        } else {
            ErrorDisposition::Recoverable
        };
        return Ok(TaskFuture::projection(projection, errors));
    };

    let filter_mask = build_filter_mask(&ctx.reader, filter, &row_range, row_mask);

    let Some(row_limit) = row_limit else {
        // Without a limit, retain the existing eager projection setup so readers can prefetch
        // projection work while the filter is being evaluated.
        let projection =
            ctx.reader
                .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;
        return Ok(TaskFuture::filtered_projection(filter_mask, projection));
    };

    Ok(TaskFuture::limited_filtered_projection(
        ctx,
        row_range,
        filter_mask,
        row_limit,
    ))
}

/// Evaluate the filter for a split and return the matching mask together with its row range.
///
/// This is the first stage of the ordered filtered-limit pipeline. It performs the same pruning
/// and filter I/O as [`split_exec`] (registered eagerly, outside the returned future, so the IO
/// system can prefetch while earlier splits are still reserving), but it neither reserves against
/// the limit nor projects. The caller reserves the returned mask in split order and then projects
/// it via [`project_split`], which keeps ordered `LIMIT` semantics without serializing I/O.
pub(crate) fn filter_split(
    ctx: Arc<TaskContext>,
    read_mask: RowMask,
) -> BoxFuture<'static, VortexResult<(Range<u64>, Mask)>> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    let filter = ctx
        .filter
        .as_ref()
        .vortex_expect("filter_split requires a filtered scan");
    let filter_mask = build_filter_mask(&ctx.reader, filter, &row_range, row_mask);

    async move {
        let mask = filter_mask.await?;
        Ok((row_range, mask))
    }
    .boxed()
}

/// Project an already-reserved, non-empty mask for a split.
///
/// This is the second stage of the ordered filtered-limit pipeline, run after the caller has
/// reserved rows against the limit in split order (see [`filter_split`]).
pub(crate) fn project_split(
    ctx: Arc<TaskContext>,
    row_range: Range<u64>,
    mask: Mask,
) -> TaskFuture {
    TaskFuture::deferred_projection(ctx, row_range, mask)
}

/// Build the filtered mask for a split.
///
/// The pruning and filter evaluations are constructed OUTSIDE the returned future on purpose:
/// registering these row ranges eagerly is a hint to the IO system that we want to start
/// prefetching the IO for this split.
fn build_filter_mask(
    reader: &Arc<dyn LayoutReader>,
    filter: &Arc<FilterExpr>,
    row_range: &Range<u64>,
    row_mask: Mask,
) -> MaskFuture {
    let reader = Arc::clone(reader);
    let filter = Arc::clone(filter);
    let filter_row_range = row_range.clone();
    MaskFuture::new(row_mask.len(), async move {
        let mut mask = row_mask;
        let mut dynamic_versions = vec![None; filter.conjuncts().len()];

        // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
        for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
            if mask.all_false() {
                return Ok(mask);
            }

            // Store the latest version of the dynamic expression prior to pruning.
            // We will re-run the pruning later if the version has changed in the meantime.
            dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

            let conjunct_mask = reader
                .pruning_evaluation(&filter_row_range, conjunct, mask.clone())?
                .await?;
            mask = mask.bitand(&conjunct_mask);
        }

        // Now we loop through the conjuncts in the preferred order and evaluate them.
        let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
        while let Some(idx) = filter.next_conjunct(&remaining) {
            remaining.set(idx, false);
            if mask.all_false() {
                return Ok(mask);
            }

            let conjunct = &filter.conjuncts()[idx];

            // If the dynamic expression has changed since pruning, re-run the pruning.
            // Store the dynamic update once to avoid TOCTOU race condition.
            let current_version = filter.dynamic_updates(idx).map(|du| du.version());
            if let Some(dv) = current_version
                && dynamic_versions[idx].is_none_or(|v| v < dv)
            {
                // The dynamic expression has changed, re-run the pruning.
                dynamic_versions[idx] = Some(dv);
                let conjunct_mask = reader
                    .pruning_evaluation(&filter_row_range, conjunct, mask.clone())?
                    .await?;
                mask = mask.bitand(&conjunct_mask);
            }
            if mask.all_false() {
                return Ok(mask);
            }

            let conjunct_mask = reader
                .filter_evaluation(&filter_row_range, conjunct, MaskFuture::ready(mask))?
                .await?;
            filter.report_selectivity(idx, conjunct_mask.density());

            // Filter evaluations return a mask already intersected with the input mask.
            mask = conjunct_mask;
        }

        Ok(mask)
    })
}

/// Information needed to execute a single split task.
///
/// Row selection is evaluated before creating a split task so it's not included
pub(crate) struct TaskContext {
    filter: Option<Arc<FilterExpr>>,
    reader: LayoutReaderRef,
    projection: Expression,
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use parking_lot::Mutex;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::TaskContext;
    use super::TaskResult;
    use super::project_split;
    use crate::ArrayFuture;
    use crate::LayoutReader;
    use crate::RowSplits;
    use crate::SplitRange;

    struct BlockingProjectionReader {
        name: Arc<str>,
        dtype: DType,
        started: mpsc::Sender<()>,
        gate: Arc<Mutex<()>>,
    }

    impl LayoutReader for BlockingProjectionReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            1
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            split_range: &SplitRange,
            splits: &mut RowSplits,
        ) -> VortexResult<()> {
            splits.push(split_range.root_row_range().end);
            Ok(())
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: Mask,
        ) -> VortexResult<MaskFuture> {
            Ok(MaskFuture::ready(mask))
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            Ok(mask)
        }

        fn projection_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            self.started
                .send(())
                .map_err(|_| vortex_err!("test projection-start receiver dropped"))?;
            let _guard = self.gate.lock();
            let array = PrimitiveArray::from_iter([0_i32]).into_array();
            Ok(Box::pin(async move { Ok(array) }))
        }
    }

    #[test]
    fn project_split_defers_projection_construction_until_task_poll() -> VortexResult<()> {
        let gate = Arc::new(Mutex::new(()));
        let guard = gate.lock();
        let (started_send, started_recv) = mpsc::channel();
        let reader: Arc<dyn LayoutReader> = Arc::new(BlockingProjectionReader {
            name: Arc::from("blocking-projection"),
            dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
            started: started_send,
            gate: Arc::clone(&gate),
        });
        let ctx = Arc::new(TaskContext {
            filter: None,
            reader,
            projection: root(),
        });

        let (task_send, task_recv) = mpsc::channel();
        let construction = thread::spawn(move || {
            let task = project_split(ctx, 0..1, Mask::new_true(1));
            drop(task_send.send(task));
        });

        let task = match task_recv.recv_timeout(Duration::from_secs(1)) {
            Ok(task) => task,
            Err(_) => {
                drop(guard);
                drop(construction.join());
                return Err(vortex_err!(
                    "constructing a projection task blocked before it was polled"
                ));
            }
        };
        if construction.join().is_err() {
            return Err(vortex_err!("projection-task construction panicked"));
        }
        match started_recv.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {}
            Ok(()) => {
                return Err(vortex_err!(
                    "projection construction started before the task was polled"
                ));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(vortex_err!(
                    "projection-start sender disconnected unexpectedly"
                ));
            }
        }

        let (result_send, result_recv) = mpsc::channel();
        let execution = thread::spawn(move || {
            let result = futures::executor::block_on(task);
            drop(result_send.send(result));
        });

        if started_recv.recv_timeout(Duration::from_secs(1)).is_err() {
            drop(guard);
            drop(execution.join());
            return Err(vortex_err!(
                "projection construction did not start when polled"
            ));
        }
        drop(guard);
        let result = result_recv
            .recv_timeout(Duration::from_secs(1))
            .map_err(|_| vortex_err!("projection task did not complete"))?;
        if execution.join().is_err() {
            return Err(vortex_err!("projection task panicked"));
        }

        assert!(matches!(result, TaskResult::Array(Some(_))));
        Ok(())
    }
}
