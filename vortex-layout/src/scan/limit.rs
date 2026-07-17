// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_mask::Mask;

/// A cloneable row limit shared by all work that can contribute rows to one scan.
///
/// Rows are reserved from a selection mask before projection work is constructed. This keeps
/// rows that cannot be returned out of projection evaluation entirely. When a limit is shared by
/// concurrent unordered partitions, reservation order is completion order, so callers may return
/// any matching rows. Ordered limited scans instead serialize their external partitions before
/// sharing a `RowLimit`, preserving the first matching rows in scan order.
#[derive(Clone)]
pub(crate) struct RowLimit(Arc<AtomicU64>);

impl RowLimit {
    pub(crate) fn new(limit: u64) -> Self {
        Self(Arc::new(AtomicU64::new(limit)))
    }

    /// Reserve rows selected by `mask` and retain only the earliest granted rows in that mask.
    pub(crate) fn limit(&self, mask: Mask) -> Mask {
        let requested = u64::try_from(mask.true_count()).unwrap_or(u64::MAX);
        let granted = self.reserve(requested);
        let granted = usize::try_from(granted).unwrap_or(usize::MAX);
        mask.limit(granted)
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 0
    }

    fn reserve(&self, requested: u64) -> u64 {
        let mut remaining = self.0.load(Ordering::Relaxed);
        loop {
            let granted = remaining.min(requested);
            match self.0.compare_exchange_weak(
                remaining,
                remaining - granted,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return granted,
                Err(actual) => remaining = actual,
            }
        }
    }
}
