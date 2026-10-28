## H679 — Previous-task coefficients mostly release next-month behavior

To distinguish an operation label from generic content gating, this preregistered panel uses next, previous and copying demonstrations for identical queries in months and integers. Each domain has four demonstration inputs at indices0,3,6,9 and eight disjoint query inputs. Months are cyclic; integers span2–13 as inputs. Arrow format and one fixed example order. Both task instructions and demonstration mappings change with the operation. Every receiver is tested with its own, next, previous and copy coefficient while preserving its own value. No examples are selected based on correctness.

Native competence criterion remains6/8. Month and integer copying each reach6/8; next tasks each4/8; previous tasks each0/8. Thus no transformation cell in this panel qualifies as competent. These failures are retained and materially restrict interpretation. The demonstration set differs from H678, whose original month task was7/8 and copy8/8; its stronger competence does not transfer to this new set.

| Domain | Receiver instruction | Donor coefficient | Next top1 /8 | Previous top1 /8 | Copy top1 /8 | Receiver-target ΔCE | Next−previous margin change |
|---|---|---|---:|---:|---:|---:|---:|
|month|next|baseline|4|0|0|+0.0000|+0.0000|
|month|next|previous|4|0|0|-0.0084|-0.0256|
|month|next|copy|4|0|1|-0.0138|-0.4430|
|month|previous|baseline|5|0|0|+0.0000|+0.0000|
|month|previous|next|5|0|0|+0.0317|+0.0305|
|month|previous|copy|6|1|0|-0.4100|-0.4565|
|month|copy|baseline|2|0|6|+0.0000|+0.0000|
|month|copy|next|8|0|0|+1.5040|+0.6046|
|month|copy|previous|7|1|0|+1.3953|+0.5653|
|integer|next|baseline|4|0|0|+0.0000|+0.0000|
|integer|next|previous|4|0|0|-0.0086|+0.0043|
|integer|next|copy|4|0|0|+0.1179|-0.0970|
|integer|previous|baseline|3|0|0|+0.0000|+0.0000|
|integer|previous|next|3|0|0|+0.0050|-0.0049|
|integer|previous|copy|3|0|1|+0.0358|-0.0876|
|integer|copy|baseline|1|0|6|+0.0000|+0.0000|
|integer|copy|next|2|0|5|+0.3278|+0.1084|
|integer|copy|previous|2|0|5|+0.3663|+0.1144|

Rows can sum to fewer than8 because other vocabulary tokens can win. Identity swaps are omitted from the table but retained in JSON; they exactly replay native logits.

On month-copy receivers, next-task coefficients yield8/8 next-month predictions. Previous-task coefficients yield7/8 next and1/8 previous, while neither preserves copying. The mean next-minus-previous margin rises with either donor (+0.6046 next donor,+0.5653 previous donor). Therefore importing a coefficient from a previous-task prompt does not reliably import the previous operation. Mean month coefficients are close for next(+0.3394) and previous(+0.3206), versus copying(+0.0605).

Integer effects are weaker: either task coefficient moves copy receivers from1/8 to2/8 next outputs, with no previous outputs and5/8 still copying. Mean coefficients are+0.2216 next,+0.2528 previous and−0.0857 copy. Thus even a similar gate separation across domains does not guarantee the same behavioral switch; receiver values and surrounding computation matter.

The evidence favors a gate that can reduce copying and expose a context-dependent next-item preference, rather than an intrinsic scalar encoding the requested direction. This is a comparison of explanations, not proof that suppression alone generates succession. Native previous tasks fail, so this cannot rule out different gates or coefficients during genuinely successful previous-item computation. It also does not establish which component carries the sequential preference: value/readout geometry, other residual content, or their interaction.

Next cross coefficient and value identity on the original competent H678 month-copy receivers. Keep prompts fixed, choose a deterministic alternate query item as value donor, and compare native/copy versus task gate with receiver versus donor values. Include full native-write replacement accounting rather than simply adding a donor vector. Track both receiver-related and donor-related next/current tokens and full-vocabulary top1. This tests whether the effective control is content-bound, rather than interpreting a scalar alone as the computational unit. Retain the H679 competence failure as evidence of context sensitivity; do not silently replace it with the earlier successful panel.

Audits:192 whole-model forwards (48×4 conditions). Identity swaps replay exactly; full attention reconstruction maximum relative error4.748e-7. Manifest and runner hashes unchanged, finite logits. Single query-token and single answer-token checks passed; fixed disjoint demonstration/query indices. All prompts, values, coefficients and outputs retained. FineWeb26,112 unique documents unchanged. Prior turn was progress; broad goal active and incomplete.
