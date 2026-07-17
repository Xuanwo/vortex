// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

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

pub(crate) type TaskFuture = BoxFuture<'static, TaskResult>;

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
pub fn split_exec(
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
            return Ok(async { TaskResult::Array(None) }.boxed());
        }

        // With no filter, limit the selection before constructing projection work.
        let projection = match ctx.reader.projection_evaluation(
            &row_range,
            &ctx.projection,
            MaskFuture::ready(row_mask),
        ) {
            Ok(projection) => projection,
            Err(err) if limited => return Ok(async move { TaskResult::Terminal(err) }.boxed()),
            Err(err) => return Err(err),
        };
        return Ok(async move {
            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(err) if limited => TaskResult::Terminal(err),
                Err(err) => TaskResult::Recoverable(err),
            }
        }
        .boxed());
    };

    let filter_mask = build_filter_mask(&ctx.reader, filter, &row_range, row_mask);

    let Some(row_limit) = row_limit else {
        // Without a limit, retain the existing eager projection setup so readers can prefetch
        // projection work while the filter is being evaluated.
        let projection =
            ctx.reader
                .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;
        let array_fut = async move {
            let mask = match filter_mask.await {
                Ok(mask) => mask,
                Err(err) => return TaskResult::Recoverable(err),
            };
            if mask.all_false() {
                return TaskResult::Array(None);
            }

            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(err) => TaskResult::Recoverable(err),
            }
        };
        return Ok(array_fut.boxed());
    };

    let array_fut = async move {
        let mask = match filter_mask.await {
            Ok(mask) => mask,
            Err(err) => return TaskResult::Recoverable(err),
        };
        // A filter error above returns before reserving any rows. Once filtering has succeeded,
        // reserve only the matching rows and construct projection work for that limited mask.
        let mask = row_limit.limit(mask);
        if mask.all_false() {
            return TaskResult::Array(None);
        }

        let projection = match ctx.reader.projection_evaluation(
            &row_range,
            &ctx.projection,
            MaskFuture::ready(mask),
        ) {
            Ok(projection) => projection,
            Err(err) => return TaskResult::Terminal(err),
        };
        match projection.await {
            Ok(array) => TaskResult::Array(Some(array)),
            Err(err) => TaskResult::Terminal(err),
        }
    };

    Ok(array_fut.boxed())
}

/// Evaluate the filter for a split and return the matching mask together with its row range.
///
/// This is the first stage of the ordered filtered-limit pipeline. It performs the same pruning
/// and filter I/O as [`split_exec`] (registered eagerly, outside the returned future, so the IO
/// system can prefetch while earlier splits are still reserving), but it neither reserves against
/// the limit nor projects. The caller reserves the returned mask in split order and then projects
/// it via [`project_split`], which keeps ordered `LIMIT` semantics without serializing I/O.
pub fn filter_split(
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
pub fn project_split(ctx: Arc<TaskContext>, row_range: Range<u64>, mask: Mask) -> TaskFuture {
    async move {
        let projection = match ctx.reader.projection_evaluation(
            &row_range,
            &ctx.projection,
            MaskFuture::ready(mask),
        ) {
            Ok(projection) => projection,
            Err(err) => return TaskResult::Terminal(err),
        };
        match projection.await {
            Ok(array) => TaskResult::Array(Some(array)),
            Err(err) => TaskResult::Terminal(err),
        }
    }
    .boxed()
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
