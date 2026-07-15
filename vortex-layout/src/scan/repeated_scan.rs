// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
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
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::limit::LimitedStream;
use crate::scan::limit::RowBudget;
use crate::scan::plan;
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
        row_mask: vortex_scan::row_mask::RowMask,
        limit: Option<&mut u64>,
    ) -> VortexResult<plan::TaskFuture<Option<ArrayRef>>> {
        match self {
            Self::Plan(ctx) => plan::split_exec(Arc::clone(ctx), row_mask, limit),
            Self::PlanV2(ctx) => plan_v2::split_exec(Arc::clone(ctx), row_mask, limit),
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
    pub fn new_plan(
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
            dtype,
        }
    }

    /// Construct a repeated scan from a prepared heap-allocated physical plan.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub fn new_plan_v2(
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
    ) -> VortexResult<Vec<plan::TaskFuture<Option<ArrayRef>>>> {
        let mut limit = self.limit.filter(|_| !self.execution.has_filter());
        let mut tasks = Vec::new();
        let ctx = self.execution.task_context();

        for range in self.split_ranges(row_range) {
            if range.start >= range.end {
                continue;
            }
            if limit.is_some_and(|l| l == 0) {
                break;
            }

            let row_mask = self.selection.row_mask(&range);
            if row_mask.mask().all_false() {
                continue;
            }

            tasks.push(ctx.split_exec(row_mask, limit.as_mut())?);
        }

        Ok(tasks)
    }

    pub(crate) fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl Stream<Item = VortexResult<ArrayRef>> + Send + 'static> {
        let num_workers = get_available_parallelism().unwrap_or(1);
        let concurrency = self.concurrency * num_workers;
        let handle = self.session.handle();

        // With both a filter and a limit we cannot know each split's output row count ahead of
        // time, so split tasks are built lazily as the stream is polled. `buffered`'s read-ahead
        // (bounded by `concurrency`) registers IO for splits eagerly, but only as far as the
        // limit requires: `LimitedStream` drops the inner stream once the limit is reached,
        // capping over-read at `concurrency` splits.
        if self.execution.has_filter() && self.limit.is_some() {
            let ctx = Arc::new(self.execution.task_context());
            let selection = self.selection.clone();
            let tasks =
                futures::stream::iter(self.split_ranges(row_range)).filter_map(move |range| {
                    // Build the row mask and split task synchronously so the IO system sees the
                    // split's ranges as soon as `buffered` pulls it, without cloning `selection`.
                    let row_mask = selection.row_mask(&range);
                    let spawned = (!row_mask.mask().all_false()).then(|| {
                        let task = ctx
                            .split_exec(row_mask, None)
                            .unwrap_or_else(|err| async move { Err(err) }.boxed());
                        handle.spawn(task)
                    });
                    async move { spawned }
                });

            return Ok(schedule(tasks, self.ordered, concurrency, self.limit));
        }

        // No filter (or no limit): build every task eagerly so the IO system sees all split
        // ranges up front. A no-filter limit is applied exactly per split inside `execute`.
        let tasks =
            futures::stream::iter(self.execute(row_range)?).map(move |task| handle.spawn(task));

        Ok(schedule(tasks, self.ordered, concurrency, self.limit))
    }
}

/// Spawn-buffer a stream of split tasks, transposing empty splits away and applying `limit`.
fn schedule<S>(
    tasks: S,
    ordered: bool,
    concurrency: usize,
    limit: Option<u64>,
) -> BoxStream<'static, VortexResult<ArrayRef>>
where
    S: Stream<Item = Task<VortexResult<Option<ArrayRef>>>> + Send + 'static,
{
    let stream = if ordered {
        tasks.buffered(concurrency).boxed()
    } else {
        tasks.buffer_unordered(concurrency).boxed()
    };
    let stream = stream
        .filter_map(|chunk| async move { chunk.transpose() })
        .boxed();

    match limit {
        Some(limit) => LimitedStream::new(stream, RowBudget::Local(limit)).boxed(),
        None => stream,
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
