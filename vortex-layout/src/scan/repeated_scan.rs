// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use itertools::Either;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::plan;
use crate::scan::plan::TaskFuture;
use crate::scan::plan::TaskResult;
use crate::scan::plan_v2;
use crate::scan::plan_v2::ScanPlanRef;
use crate::scan::splits::Splits;

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan {
    session: VortexSession,
    execution: ExecutionPlan,
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// The natural splits of the file.
    splits: Splits,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Maximal number of rows to read (after filtering).
    limit: Option<u64>,
    /// An optional row limit shared with sibling external partitions.
    row_limit: Option<RowLimit>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

enum ExecutionPlan {
    Plan(plan::Plan),
    PlanV2(plan_v2::PlanV2),
}

impl ExecutionPlan {
    fn has_filter(&self) -> bool {
        match self {
            Self::Plan(plan) => plan.has_filter(),
            Self::PlanV2(plan) => plan.has_filter(),
        }
    }

    fn task_context(&self) -> ExecutionTaskContext {
        match self {
            Self::Plan(plan) => ExecutionTaskContext::Plan(plan.task_context()),
            Self::PlanV2(plan) => ExecutionTaskContext::PlanV2(plan.task_context()),
        }
    }
}

enum ExecutionTaskContext {
    Plan(Arc<plan::TaskContext>),
    PlanV2(Arc<plan_v2::TaskContext>),
}

impl ExecutionTaskContext {
    fn split_exec(
        &self,
        row_mask: RowMask,
        row_limit: Option<RowLimit>,
    ) -> VortexResult<TaskFuture> {
        match self {
            Self::Plan(ctx) => plan::split_exec(Arc::clone(ctx), row_mask, row_limit),
            Self::PlanV2(ctx) => plan_v2::split_exec(Arc::clone(ctx), row_mask, row_limit),
        }
    }

    fn filter_split(
        &self,
        row_mask: RowMask,
    ) -> BoxFuture<'static, VortexResult<(Range<u64>, Mask)>> {
        match self {
            Self::Plan(ctx) => plan::filter_split(Arc::clone(ctx), row_mask),
            Self::PlanV2(ctx) => plan_v2::filter_split(Arc::clone(ctx), row_mask),
        }
    }

    fn project_split(&self, row_range: Range<u64>, mask: Mask) -> TaskFuture {
        match self {
            Self::Plan(ctx) => plan::project_split(Arc::clone(ctx), row_range, mask),
            Self::PlanV2(ctx) => plan_v2::project_split(Arc::clone(ctx), row_range, mask),
        }
    }
}

/// A source of split tasks that has not yet applied task concurrency or output error handling.
#[must_use = "task streams must be scheduled"]
struct TaskStream {
    inner: BoxStream<'static, Task<TaskResult>>,
}

impl TaskStream {
    fn new(stream: impl Stream<Item = Task<TaskResult>> + Send + 'static) -> Self {
        Self {
            inner: stream.boxed(),
        }
    }

    fn eager(handle: Handle, tasks: Vec<TaskFuture>) -> Self {
        Self::new(futures::stream::iter(tasks).map(move |task| handle.spawn(task)))
    }

    fn unordered_filtered_limit(
        handle: Handle,
        split_ranges: Vec<Range<u64>>,
        selection: Selection,
        ctx: Arc<ExecutionTaskContext>,
        row_limit: RowLimit,
    ) -> Self {
        let take_limit = row_limit.clone();
        Self::new(
            futures::stream::iter(split_ranges)
                .take_while(move |_| future::ready(!take_limit.is_exhausted()))
                .filter_map(move |range| {
                    // Build the row mask and split task synchronously so the I/O system sees the
                    // split's ranges as soon as `buffer_unordered` pulls it, without cloning
                    // `selection`.
                    let row_mask = selection.row_mask(&range);
                    let task = (!row_mask.mask().all_false()).then(|| {
                        ctx.split_exec(row_mask, Some(row_limit.clone()))
                            .unwrap_or_else(TaskFuture::terminal)
                    });
                    future::ready(task.map(|task| handle.spawn(task)))
                }),
        )
    }

    fn ordered_filtered_limit(
        handle: Handle,
        split_ranges: Vec<Range<u64>>,
        selection: Selection,
        ctx: Arc<ExecutionTaskContext>,
        row_limit: RowLimit,
        concurrency: usize,
    ) -> Self {
        // Stage one: evaluate the filter for each split concurrently while preserving split
        // order.
        let filter_tasks = {
            let filter_ctx = Arc::clone(&ctx);
            let filter_handle = handle.clone();
            let take_limit = row_limit.clone();
            futures::stream::iter(split_ranges)
                .take_while(move |_| future::ready(!take_limit.is_exhausted()))
                .filter_map(move |range| {
                    // Build the row mask synchronously so the I/O system sees the split's ranges
                    // as soon as `buffered` pulls it, without cloning `selection`.
                    let row_mask = selection.row_mask(&range);
                    let task = (!row_mask.mask().all_false())
                        .then(|| filter_handle.spawn(filter_ctx.filter_split(row_mask)));
                    future::ready(task)
                })
                .buffered(concurrency)
        };

        // Seam + stage two: reserve in split order (this map runs sequentially as the ordered
        // stage-one stream is consumed), then spawn projection work for the reserved mask. The
        // second gate drops results that were already filtering when an earlier split exhausted
        // the budget, including any later filter errors.
        let stage_two_limit = row_limit.clone();
        Self::new(
            filter_tasks
                .take_while(move |_| future::ready(!stage_two_limit.is_exhausted()))
                .map(move |result| {
                    let task = match result {
                        Ok((row_range, mask)) => {
                            let mask = row_limit.limit(mask);
                            if mask.all_false() {
                                TaskFuture::empty()
                            } else {
                                ctx.project_split(row_range, mask)
                            }
                        }
                        Err(error) => TaskFuture::recoverable(error),
                    };
                    handle.spawn(task)
                }),
        )
    }
}

impl Stream for TaskStream {
    type Item = Task<TaskResult>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// A buffered task stream that exposes arrays and applies the task error policy.
pub(crate) struct ScheduledTaskStream {
    tasks: Option<BoxStream<'static, TaskResult>>,
}

impl ScheduledTaskStream {
    fn new(tasks: TaskStream, ordered: bool, concurrency: usize) -> Self {
        let tasks = if ordered {
            tasks.buffered(concurrency).boxed()
        } else {
            tasks.buffer_unordered(concurrency).boxed()
        };
        Self { tasks: Some(tasks) }
    }
}

impl Stream for ScheduledTaskStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let Some(tasks) = self.tasks.as_mut() else {
                return Poll::Ready(None);
            };
            let result = ready!(tasks.as_mut().poll_next(cx));
            let Some(result) = result else {
                self.tasks = None;
                return Poll::Ready(None);
            };
            match result {
                TaskResult::Array(Some(array)) => return Poll::Ready(Some(Ok(array))),
                TaskResult::Array(None) => {}
                TaskResult::Recoverable(error) => return Poll::Ready(Some(Err(error))),
                TaskResult::Terminal(error) => {
                    // Drop queued and in-flight task handles before yielding the error. Their
                    // `Drop` implementations abort work that can no longer contribute to this
                    // limited scan.
                    self.tasks = None;
                    return Poll::Ready(Some(Err(error)));
                }
            }
        }
    }
}

impl RepeatedScan {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        row_range: Option<Range<u64>>,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        let iter = runtime.block_on_stream(stream);
        Ok(ArrayIteratorAdapter::new(dtype, iter))
    }

    pub fn execute_array_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }

    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub(crate) fn new_plan(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: Expression,
        filter: Option<Expression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        limit: Option<u64>,
        row_limit: Option<RowLimit>,
        dtype: DType,
    ) -> Self {
        Self {
            session,
            execution: ExecutionPlan::Plan(plan::Plan::new(layout_reader, projection, filter)),
            ordered,
            row_range,
            selection,
            splits,
            concurrency,
            limit,
            row_limit,
            dtype,
        }
    }

    /// Construct a repeated scan from a prepared heap-allocated physical plan.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub(crate) fn new_plan_v2(
        session: VortexSession,
        projection: ScanPlanRef,
        predicates: Vec<ScanPlanRef>,
        filter: Option<Expression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        limit: Option<u64>,
        row_limit: Option<RowLimit>,
        dtype: DType,
    ) -> Self {
        Self {
            session,
            execution: ExecutionPlan::PlanV2(plan_v2::PlanV2::new(projection, predicates, filter)),
            ordered,
            row_range,
            selection,
            splits,
            concurrency,
            limit,
            row_limit,
            dtype,
        }
    }

    fn split_ranges(&self, row_range: Option<Range<u64>>) -> Vec<Range<u64>> {
        let selection_range: Option<Range<u64>> = match &self.selection {
            Selection::IncludeByIndex(buf) if !buf.is_empty() => {
                Some(buf[0]..buf[buf.len() - 1] + 1)
            }
            Selection::IncludeRoaring(roaring) if !roaring.is_empty() => {
                Some(roaring.min().vortex_expect("empty")..roaring.max().vortex_expect("empty") + 1)
            }
            _ => None,
        };
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        let row_range = intersect_ranges(row_range.as_ref(), selection_range);

        match &self.splits {
            Splits::Natural(vec) => {
                debug_assert!(vec.is_sorted());
                let splits_iter = match row_range {
                    None => Either::Left(vec.iter().copied()),
                    Some(range) => {
                        if range.is_empty() {
                            return Vec::new();
                        }
                        let lo = vec.partition_point(|&x| x < range.start);
                        let hi = vec.partition_point(|&x| x < range.end);
                        Either::Right(
                            iter::once(range.start)
                                .chain(vec[lo..hi].iter().copied())
                                .chain(iter::once(range.end)),
                        )
                    }
                };

                splits_iter
                    .tuple_windows()
                    .map(|(start, end)| start..end)
                    .collect()
            }
            Splits::Ranges(ranges) => match row_range {
                None => ranges.to_vec(),
                Some(range) => {
                    if range.is_empty() {
                        return Vec::new();
                    }
                    ranges
                        .iter()
                        .filter_map(move |r| {
                            let start = cmp::max(r.start, range.start);
                            let end = cmp::min(r.end, range.end);
                            (start < end).then_some(start..end)
                        })
                        .collect()
                }
            },
        }
    }

    pub(crate) fn execute(
        &self,
        row_range: Option<Range<u64>>,
        row_limit: Option<RowLimit>,
    ) -> VortexResult<Vec<TaskFuture>> {
        let mut tasks = Vec::new();
        let ctx = self.execution.task_context();

        for range in self.split_ranges(row_range) {
            if range.start >= range.end {
                continue;
            }
            if row_limit.as_ref().is_some_and(RowLimit::is_exhausted) {
                break;
            }

            let row_mask = self.selection.row_mask(&range);
            if row_mask.mask().all_false() {
                continue;
            }

            tasks.push(ctx.split_exec(row_mask, row_limit.clone())?);
        }

        Ok(tasks)
    }

    pub(crate) fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<ScheduledTaskStream> {
        let num_workers = get_available_parallelism().unwrap_or(1);
        let row_limit = self
            .row_limit
            .clone()
            .or_else(|| self.limit.map(RowLimit::new));
        let concurrency = self.concurrency * num_workers;
        let handle = self.session.handle();

        // With both a filter and a limit we cannot know each split's output row count ahead of
        // time, so split tasks are built lazily as the stream is polled. Once another task drains
        // the shared budget, `take_while` prevents further task creation; already-prefetched
        // tasks observe the exhausted budget after filtering and return `None` without projection.
        if self.execution.has_filter()
            && let Some(row_limit) = row_limit.clone()
        {
            // An ordered LIMIT would be violated if a later split reserved rows before an earlier
            // one, so ordered scans use a two-stage pipeline that keeps I/O concurrent while
            // reserving strictly in split order.
            if self.ordered {
                return Ok(self.ordered_filtered_limit_stream(row_range, row_limit, concurrency));
            }

            let tasks = TaskStream::unordered_filtered_limit(
                handle,
                self.split_ranges(row_range),
                self.selection.clone(),
                Arc::new(self.execution.task_context()),
                row_limit,
            );
            return Ok(ScheduledTaskStream::new(tasks, false, concurrency));
        }

        // No filter (or no limit): build every task eagerly so the IO system sees all split
        // ranges up front. A no-filter limit is applied to each selection mask inside `execute`.
        let tasks = TaskStream::eager(handle, self.execute(row_range, row_limit)?);

        Ok({
            let ordered = self.ordered;
            ScheduledTaskStream::new(tasks, ordered, concurrency)
        })
    }

    /// Ordered filtered scan with a shared row limit.
    ///
    /// I/O concurrency is decoupled from limit reservation so that an ordered `LIMIT` does not
    /// force serial execution. Filters for a window of splits are evaluated concurrently (stage
    /// one), rows are reserved against the shared limit strictly in split order on the seam
    /// between the stages, and only the reserved masks are projected (stage two, also concurrent).
    /// Because reservation happens in split order, the earliest matching rows always win the
    /// budget; because projection only runs for reserved masks, no rows past the limit are
    /// decoded. `take_while` stops feeding stage one once the limit is exhausted, bounding
    /// speculative filter I/O to the pipeline depth.
    fn ordered_filtered_limit_stream(
        &self,
        row_range: Option<Range<u64>>,
        row_limit: RowLimit,
        concurrency: usize,
    ) -> ScheduledTaskStream {
        let tasks = TaskStream::ordered_filtered_limit(
            self.session.handle(),
            self.split_ranges(row_range),
            self.selection.clone(),
            Arc::new(self.execution.task_context()),
            row_limit,
            concurrency,
        );
        ScheduledTaskStream::new(tasks, true, concurrency)
    }
}

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (None, Some(r)) => Some(r),
        (Some(l), None) => Some(l.clone()),
        (Some(l), Some(r)) => Some(cmp::max(l.start, r.start)..cmp::min(l.end, r.end)),
    }
}
