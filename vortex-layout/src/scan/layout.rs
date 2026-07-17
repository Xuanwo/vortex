// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;
use futures::FutureExt;
use futures::Stream;
use futures::stream;
use futures::stream::StreamExt;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_metrics::MetricsRegistry;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::scan_builder::ScanBuilder;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
    split_max_row_count: u64,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

impl LayoutReaderDataSource {
    /// Creates a new [`LayoutReaderDataSource`].
    ///
    /// By default, the entire scan is returned as a single split. This best preserves V1
    /// `ScanBuilder` behavior where one scan covers the full row range, allowing the internal
    /// I/O pipeline and `SplitBy::Layout` chunking to operate without per-split overhead from
    /// redundant expression resolution and layout tree traversal.
    pub fn new(reader: LayoutReaderRef, session: VortexSession) -> Self {
        Self {
            reader,
            session,
            split_max_row_count: u64::MAX,
            metrics_registry: None,
        }
    }

    /// Sets the maximum number of rows per Scan API split.
    ///
    /// Each split drives a [`ScanBuilder`] over its row range, which internally handles
    /// physical layout alignment and I/O pipelining. This controls the engine-level
    /// parallelism granularity, not the I/O granularity.
    pub fn with_split_max_row_count(mut self, row_count: u64) -> Self {
        self.split_max_row_count = row_count;
        self
    }

    /// Sets the metrics registry for tracking scan performance.
    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics);
        self
    }

    /// Optionally sets the metrics registry for tracking scan performance.
    pub fn with_some_metrics_registry(mut self, metrics: Option<Arc<dyn MetricsRegistry>>) -> Self {
        self.metrics_registry = metrics;
        self
    }
}

#[async_trait]
impl DataSource for LayoutReaderDataSource {
    fn dtype(&self) -> &DType {
        self.reader.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.reader.row_count())
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("LayoutReader splits are not yet serializable");
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let total_rows = self.reader.row_count();
        let row_range = scan_request.row_range.unwrap_or(0..total_rows);

        let dtype = scan_request.projection.return_dtype(self.reader.dtype())?;

        // If the dtype is an empty struct, and there is no filter, we can return a special
        // length-only scan.
        if let DType::Struct(fields, Nullability::NonNullable) = &dtype
            && fields.nfields() == 0
            && scan_request.filter.is_none()
        {
            // FIXME(ngates): extract out maybe?
            let row_count = row_range.end - row_range.start;
            let row_count = scan_request.selection.row_count(row_count);

            // Apply the limit.
            let row_count = if let Some(limit) = scan_request.limit {
                row_count.min(limit)
            } else {
                row_count
            };

            return Ok(Box::new(Empty { dtype, row_count }));
        }

        // Check file-level pruning: if the filter can be proven false for the entire row range
        // using file-level statistics (e.g. via FileStatsLayoutReader), skip the scan entirely.
        if let Some(filter) = &scan_request.filter {
            let mask = Mask::new_true(
                usize::try_from(row_range.end - row_range.start).unwrap_or(usize::MAX),
            );
            let pruning_result = self
                .reader
                .pruning_evaluation(&row_range, filter, mask)?
                .now_or_never();
            if let Some(Ok(result_mask)) = pruning_result
                && result_mask.all_false()
            {
                return Ok(Box::new(Empty {
                    dtype,
                    row_count: 0,
                }));
            }
        }

        // An ordered limit must see earlier filtered rows before later external partitions are
        // allowed to reserve any budget. Emit one partition for that path; its inner scan keeps
        // the limit at mask level. Unordered partitions share a completion-order budget instead.
        let ordered_limit = scan_request.ordered && scan_request.limit.is_some();
        let row_limit = (!scan_request.ordered)
            .then(|| scan_request.limit.map(RowLimit::new))
            .flatten();
        let split_size = if ordered_limit {
            row_range.end - row_range.start
        } else {
            self.split_max_row_count
        };

        Ok(Box::new(LayoutReaderScan {
            reader: Arc::clone(&self.reader),
            session: self.session.clone(),
            dtype,
            projection: scan_request.projection,
            filter: scan_request.filter,
            limit: scan_request.limit,
            row_limit,
            selection: scan_request.selection,
            ordered: scan_request.ordered,
            metrics_registry: self.metrics_registry.clone(),
            next_row: row_range.start,
            end_row: row_range.end,
            split_size,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

struct LayoutReaderScan {
    reader: LayoutReaderRef,
    session: VortexSession,
    dtype: DType,
    projection: Expression,
    filter: Option<Expression>,
    limit: Option<u64>,
    row_limit: Option<RowLimit>,
    ordered: bool,
    selection: Selection,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
    next_row: u64,
    end_row: u64,
    split_size: u64,
}

impl DataSourceScan for LayoutReaderScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        let (lower, upper) = self.size_hint();
        match upper {
            Some(u) if u == lower => Precision::exact(lower),
            Some(u) => Precision::inexact(u),
            None => Precision::inexact(lower),
        }
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        (*self).boxed()
    }
}

impl Stream for LayoutReaderScan {
    type Item = VortexResult<PartitionRef>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.next_row >= this.end_row {
            return Poll::Ready(None);
        }

        if this.limit.is_some_and(|limit| limit == 0) {
            return Poll::Ready(None);
        }
        if this.row_limit.as_ref().is_some_and(RowLimit::is_exhausted) {
            return Poll::Ready(None);
        }

        let split_end = this
            .next_row
            .saturating_add(this.split_size)
            .min(this.end_row);
        let row_range = this.next_row..split_end;
        let split = Box::new(LayoutReaderSplit {
            reader: Arc::clone(&this.reader),
            session: this.session.clone(),
            projection: this.projection.clone(),
            filter: this.filter.clone(),
            limit: this.limit,
            row_limit: this.row_limit.clone(),
            ordered: this.ordered,
            row_range,
            selection: this.selection.clone(),
            metrics_registry: this.metrics_registry.clone(),
        }) as PartitionRef;

        this.next_row = split_end;

        Poll::Ready(Some(Ok(split)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.next_row >= self.end_row
            || self.limit.is_some_and(|limit| limit == 0)
            || self.row_limit.as_ref().is_some_and(RowLimit::is_exhausted)
        {
            return (0, Some(0));
        }
        let remaining_rows = self.end_row - self.next_row;
        let splits = remaining_rows.div_ceil(self.split_size);
        (0, Some(usize::try_from(splits).unwrap_or(usize::MAX)))
    }
}

struct LayoutReaderSplit {
    reader: LayoutReaderRef,
    session: VortexSession,
    projection: Expression,
    filter: Option<Expression>,
    limit: Option<u64>,
    row_limit: Option<RowLimit>,
    ordered: bool,
    row_range: Range<u64>,
    selection: Selection,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

impl Partition for LayoutReaderSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[expect(clippy::cast_possible_truncation)]
    fn index(&self) -> usize {
        // Row range is unique per split
        self.row_range.start as usize
    }

    fn row_count(&self) -> Precision<u64> {
        let row_count = self.row_range.end - self.row_range.start;
        let row_count = self.selection.row_count(row_count);
        let row_count = self.limit.map_or(row_count, |limit| row_count.min(limit));

        if self.filter.is_some() || self.row_limit.is_some() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let builder = ScanBuilder::new(self.session, self.reader)
            .with_row_range(self.row_range)
            .with_selection(self.selection)
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_some_limit(self.limit)
            .with_some_row_limit(self.row_limit)
            .with_some_metrics_registry(self.metrics_registry)
            .with_ordered(self.ordered);

        let dtype = builder.dtype()?;
        // Use into_stream() which creates a LazyScanStream that spawns individual I/O
        // tasks onto the runtime, enabling parallel execution across executor threads.
        let stream = builder.into_stream()?.boxed();

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

/// A scan that produces no data, only empty arrays with the correct row count.
struct Empty {
    dtype: DType,
    row_count: u64,
}

impl DataSourceScan for Empty {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter([Ok(self as _)]).boxed()
    }
}

impl Partition for Empty {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        0
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.row_count)
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::exact(0u64)
    }

    fn execute(mut self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let scalar = Scalar::default_value(&self.dtype);
        let dtype = self.dtype.clone();

        // Create an iterator of arrays with the correct row count, respecting u64::MAX limits.
        let iter = std::iter::from_fn(move || {
            if self.row_count == 0 {
                return None;
            }
            let chunk_size = usize::try_from(self.row_count).unwrap_or(usize::MAX);
            self.row_count -= chunk_size as u64;
            Some(VortexResult::Ok(
                ConstantArray::new(scalar.clone(), chunk_size).into_array(),
            ))
        });

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            stream::iter(iter),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;
    use std::task::Poll;

    use futures::StreamExt;
    use futures::TryStreamExt;
    use parking_lot::Mutex;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_mask::Mask;
    use vortex_scan::DataSource;
    use vortex_scan::ScanRequest;

    use super::LayoutReaderDataSource;
    use crate::ArrayFuture;
    use crate::LayoutReader;
    use crate::RowSplits;
    use crate::SplitRange;
    use crate::scan::test::session_with_handle;

    #[derive(Debug)]
    struct TestLayoutReader {
        name: Arc<str>,
        dtype: DType,
        row_count: u64,
        projection_masks: Option<Arc<Mutex<Vec<usize>>>>,
    }

    impl TestLayoutReader {
        fn new(row_count: u64) -> Self {
            Self {
                name: Arc::from("test"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                row_count,
                projection_masks: None,
            }
        }

        fn with_projection_masks(mut self, projection_masks: Arc<Mutex<Vec<usize>>>) -> Self {
            self.projection_masks = Some(projection_masks);
            self
        }
    }

    impl LayoutReader for TestLayoutReader {
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
            self.row_count
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
            row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            let row_range = row_range.clone();
            let projection_masks = self.projection_masks.clone();

            Ok(Box::pin(async move {
                let mask = mask.await?;
                if let Some(projection_masks) = projection_masks {
                    projection_masks.lock().push(mask.true_count());
                }
                let start = i32::try_from(row_range.start)?;
                let end = i32::try_from(row_range.end)?;
                PrimitiveArray::from_iter(start..end)
                    .into_array()
                    .filter(mask)
            }))
        }
    }

    #[derive(Debug)]
    struct DelayedFirstSplitReader {
        name: Arc<str>,
        dtype: DType,
        projection_ranges: Arc<Mutex<Vec<Range<u64>>>>,
        filter_ranges: Arc<Mutex<Vec<Range<u64>>>>,
    }

    impl DelayedFirstSplitReader {
        fn new(projection_ranges: Arc<Mutex<Vec<Range<u64>>>>) -> Self {
            Self {
                name: Arc::from("delayed-first-split"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                projection_ranges,
                filter_ranges: Arc::new(Mutex::new(Vec::new())),
            }
        }

        /// Row ranges whose filter was actually evaluated (recorded when the split's filter
        /// future runs, not when it is merely scheduled).
        fn filter_ranges(&self) -> Arc<Mutex<Vec<Range<u64>>>> {
            Arc::clone(&self.filter_ranges)
        }
    }

    impl LayoutReader for DelayedFirstSplitReader {
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
            4
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            split_range: &SplitRange,
            splits: &mut RowSplits,
        ) -> VortexResult<()> {
            let row_range = split_range.row_range();
            splits.push(split_range.row_offset() + row_range.start + 2);
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
            row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            self.filter_ranges.lock().push(row_range.clone());
            let delay = row_range.start == 0;
            let len = mask.len();

            Ok(MaskFuture::new(len, async move {
                if delay {
                    let mut yielded = false;
                    futures::future::poll_fn(move |cx| {
                        if yielded {
                            Poll::Ready(())
                        } else {
                            yielded = true;
                            cx.waker().wake_by_ref();
                            Poll::Pending
                        }
                    })
                    .await;
                }
                mask.await
            }))
        }

        fn projection_evaluation(
            &self,
            row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            let row_range = row_range.clone();
            let projection_ranges = Arc::clone(&self.projection_ranges);

            Ok(Box::pin(async move {
                projection_ranges.lock().push(row_range.clone());
                let start = i32::try_from(row_range.start)?;
                let end = i32::try_from(row_range.end)?;
                PrimitiveArray::from_iter(start..end)
                    .into_array()
                    .filter(mask.await?)
            }))
        }
    }

    #[test]
    fn filtered_limit_is_global_across_scan_partitions() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let source = LayoutReaderDataSource::new(Arc::new(TestLayoutReader::new(6)), session)
            .with_split_max_row_count(2);

        let scan = runtime.block_on(source.scan(ScanRequest {
            filter: Some(root()),
            limit: Some(3),
            ordered: true,
            ..Default::default()
        }))?;
        let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
        assert_eq!(partitions.len(), 1);

        let mut ctx = array_session().create_execution_ctx();
        let mut values = Vec::new();
        for partition in partitions {
            for chunk in runtime.block_on_stream(partition.execute()?) {
                let primitive = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
                values.extend(primitive.into_buffer::<i32>());
            }
        }

        assert_eq!(values, [0, 1, 2]);
        Ok(())
    }

    #[test]
    fn ordered_filtered_limit_waits_for_the_earlier_split() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let projection_ranges = Arc::new(Mutex::new(Vec::new()));
        let source = LayoutReaderDataSource::new(
            Arc::new(DelayedFirstSplitReader::new(Arc::clone(&projection_ranges))),
            session,
        )
        .with_split_max_row_count(2);

        let scan = runtime.block_on(source.scan(ScanRequest {
            filter: Some(root()),
            limit: Some(1),
            ordered: true,
            ..Default::default()
        }))?;
        let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
        assert_eq!(partitions.len(), 1);

        let mut ctx = array_session().create_execution_ctx();
        let mut values = Vec::new();
        for partition in partitions {
            for chunk in runtime.block_on_stream(partition.execute()?) {
                let primitive = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
                values.extend(primitive.into_buffer::<i32>());
            }
        }

        assert_eq!(values, [0]);
        let projection_ranges = projection_ranges.lock();
        assert_eq!(projection_ranges.len(), 1);
        assert_eq!(projection_ranges[0], 0..2);
        Ok(())
    }

    #[test]
    fn ordered_filtered_limit_evaluates_later_split_filter_concurrently() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let projection_ranges = Arc::new(Mutex::new(Vec::new()));
        let reader = DelayedFirstSplitReader::new(Arc::clone(&projection_ranges));
        let filter_ranges = reader.filter_ranges();
        let source =
            LayoutReaderDataSource::new(Arc::new(reader), session).with_split_max_row_count(2);

        let scan = runtime.block_on(source.scan(ScanRequest {
            filter: Some(root()),
            limit: Some(1),
            ordered: true,
            ..Default::default()
        }))?;
        let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
        assert_eq!(partitions.len(), 1);

        let mut ctx = array_session().create_execution_ctx();
        let mut values = Vec::new();
        for partition in partitions {
            for chunk in runtime.block_on_stream(partition.execute()?) {
                let primitive = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
                values.extend(primitive.into_buffer::<i32>());
            }
        }

        // Ordered LIMIT is still exact: only the first split's earliest row is projected.
        assert_eq!(values, [0]);
        let projection_ranges = projection_ranges.lock();
        assert_eq!(projection_ranges.len(), 1);
        assert_eq!(projection_ranges[0], 0..2);
        drop(projection_ranges);

        // But the later split's filter still runs while the delayed first split reserves, so
        // prefetch is not disabled (serializing to concurrency=1 would only ever filter 0..2).
        let filter_ranges = filter_ranges.lock();
        assert!(
            filter_ranges.contains(&(0..2)) && filter_ranges.contains(&(2..4)),
            "expected both splits' filters to be evaluated, got {filter_ranges:?}"
        );
        Ok(())
    }

    #[test]
    fn unordered_limit_never_projects_more_than_the_global_budget() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let projection_masks = Arc::new(Mutex::new(Vec::new()));
        let source = LayoutReaderDataSource::new(
            Arc::new(
                TestLayoutReader::new(12).with_projection_masks(Arc::clone(&projection_masks)),
            ),
            session,
        )
        .with_split_max_row_count(2);

        let scan = runtime.block_on(source.scan(ScanRequest {
            filter: Some(root()),
            limit: Some(3),
            ordered: false,
            ..Default::default()
        }))?;
        let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
        let chunks = runtime.block_on(
            futures::stream::iter(partitions)
                .map(|partition| partition.execute())
                .try_flatten_unordered(Some(6))
                .try_collect::<Vec<_>>(),
        )?;

        let mut ctx = array_session().create_execution_ctx();
        let values = chunks
            .into_iter()
            .map(|chunk| {
                chunk
                    .execute::<PrimitiveArray>(&mut ctx)
                    .map(|primitive| primitive.into_buffer::<i32>())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let row_count = values.iter().map(|values| values.len()).sum::<usize>();

        assert_eq!(row_count, 3);
        assert_eq!(projection_masks.lock().iter().sum::<usize>(), 3);
        Ok(())
    }
}
