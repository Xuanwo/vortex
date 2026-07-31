// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::map_parts;

pub(super) const PARENT_RULES: ParentRuleSet<DecimalByteParts> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalBytePartsFilterPushDownRule),
    ParentRuleSet::lift(&CastReduceAdaptor(DecimalByteParts)),
    ParentRuleSet::lift(&FilterReduceAdaptor(DecimalByteParts)),
    ParentRuleSet::lift(&MaskReduceAdaptor(DecimalByteParts)),
    ParentRuleSet::lift(&SliceReduceAdaptor(DecimalByteParts)),
]);

#[derive(Debug)]
struct DecimalBytePartsFilterPushDownRule;

impl ArrayParentReduceRule<DecimalByteParts> for DecimalBytePartsFilterPushDownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, DecimalByteParts>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(ngates): we should benchmark whether to push-down filters with "lower parts",
        //  which filters each part separately rather than the canonical wide buffer once.
        map_parts(child, |part| part.filter(parent.filter_mask().clone()))
            .map(|c| Some(c.into_array()))
    }
}
