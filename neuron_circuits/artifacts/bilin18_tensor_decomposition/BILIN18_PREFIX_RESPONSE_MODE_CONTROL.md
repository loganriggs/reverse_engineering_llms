## H619 — Equal-magnitude prefix interactions have very different effects on answer emission

The previous turn was progress: H618 found large capital-answer suppression outside the copying span, but preserved paired factual discrimination. Unprefixed lowercase factorial corners already favored generic continuations. H619 tests that context confound using two explicit four-token prefixes: "As a reminder," and "For the record," crossed with initial capitalization. Both have identical body positions and token IDs. Three tasks (copying, agreement, capital recall), two templates, and free continuation versus appended "Next word:" give768 native prompts. Native plus full/span/complement removals at doses0.25 and1 give1,344 evaluations, including192 zero-removal baselines. The joint64 copying basis stays frozen.

**Replacing the unprefixed baseline sharply weakens the damaging capital effect.** On free-continuation capital prompts all combined recipients start with correct top1, and full/span/complement removals preserve correct top1 at both doses. Full dose1 native-reference KL is0.00172/0.00190 across templates, versus0.666/0.489 for H618's absent-prefix interaction. Every native free-continuation competence cell passes the predeclared gate, although the second capital template's reminder/lowercase cell has only50% accepted-token top1. Passing a threshold is not perfect response-mode matching.

The new interaction is smaller: mean new/old write norm ratios are0.352/0.386 on the two capital templates, with mean flattened-vector cosines0.385/0.349. The new contrast changes both size and direction. Therefore raw effect reduction cannot by itself establish a response-mode mechanism.

**A matched-magnitude audit rules out size alone.** After observing the smaller norms, rescale each new full interaction to the old interaction's total Frobenius norm across body positions. Test old and new directions, plus each direction's fixed-basis span and complement, on identical free-continuation recipient token sequences. This adds576 forwards across the three tasks. Scaling is per example; it does not match every position's norm or match span/complement norms separately. It is a follow-up diagnostic on familiar data, not independent confirmation.

| Capital template | Direction | Component | Native-reference KL | Correct top1 | Paired capital choice | Correct capital mass | Capital vs generic margin |
|---|---|---|---:|---:|---:|---:|---:|
|0|old|full|0.6658|0.3750|1.0000|0.0789|-1.0533|
|0|old|span|0.0120|1.0000|1.0000|0.3747|1.6049|
|0|old|complement|0.8969|0.1250|1.0000|0.0596|-1.4445|
|0|new_matched|full|0.0131|1.0000|1.0000|0.3965|1.5173|
|0|new_matched|span|0.0100|1.0000|1.0000|0.4008|1.6648|
|0|new_matched|complement|0.0104|1.0000|1.0000|0.3506|1.2246|
|1|old|full|0.4891|0.1250|1.0000|0.0501|-1.6541|
|1|old|span|0.0162|1.0000|1.0000|0.2931|0.8328|
|1|old|complement|0.5866|0.1250|1.0000|0.0431|-1.8383|
|1|new_matched|full|0.0164|1.0000|1.0000|0.2363|0.3301|
|1|new_matched|span|0.0080|1.0000|1.0000|0.2998|0.7888|
|1|new_matched|complement|0.0329|1.0000|1.0000|0.2021|0.1147|

Old full removal gives only37.5%/12.5% correct top1. Equally large new full removal preserves100%/100%, with KL0.0131/0.0164—about51×/30× smaller. The old complement reduces top1 to12.5% in both templates; the new complement preserves100%. Correct-capital versus paired-wrong-capital choice remains perfect throughout the matched audit. Thus the old direction is particularly effective at suppressing direct answer emission on these recipients; a generic large perturbation explanation is insufficient.

“Capital vs generic margin” is log probability mass for the correct capital minus log probability mass for three fixed tokens: "the", "a", "not". These were chosen from the previous completion audit and fixed for H619. Positive means the capital outranks their combined mass. This is a narrow diagnostic, not all nonanswer continuations. The old full direction flips the mean margin to−1.05/−1.65; the matched new direction leaves it+1.52/+0.33. Probability mass and KL concern the whole model distribution or explicit token sets; preserved paired discrimination does not prove all factual knowledge is intact.

Copying anchors retain all answers under matched old/new directions. Agreement remains mostly/all correct and the new direction produces less KL than the old one. These controls show the contrast is not simply catastrophic disruption everywhere, but task wording, length and recipient sensitivity are not independently isolated.

**The requested fixed-answer-format control failed.** Appending "Next word:" produces0% accepted-token top1 in every native task/template/prefix/case cell. Copying and capital paired-choice tests remain perfect; agreement paired-choice rates are only0.125–0.25. Top tokens include "A", "The", quotes, "capital" and country names. This suffix did not reliably induce the intended task in this model. All failed cells remain in the experiment, and their small intervention effects cannot be used to conclude that stabilizing response format removes the effect. No prompt was discarded or rewritten using these outputs.

| Task | Template | Format | Native recipient top1 | Full dose1 top1 | Full dose1 KL |
|---|---:|---|---:|---:|---:|
|sequence_copy|0|Free continuation|1.0000|1.0000|0.000746|
|sequence_copy|0|Next word label|0.0000|0.0000|0.000532|
|sequence_copy|1|Free continuation|1.0000|1.0000|0.000541|
|sequence_copy|1|Next word label|0.0000|0.0000|0.000126|
|agreement|0|Free continuation|1.0000|1.0000|0.001513|
|agreement|0|Next word label|0.0000|0.0000|0.001761|
|agreement|1|Free continuation|0.9375|1.0000|0.001029|
|agreement|1|Next word label|0.0000|0.0000|0.001316|
|capital_recall|0|Free continuation|1.0000|1.0000|0.001716|
|capital_recall|0|Next word label|0.0000|0.0000|0.000923|
|capital_recall|1|Free continuation|1.0000|1.0000|0.001896|
|capital_recall|1|Next word label|0.0000|0.0000|0.000871|

Metric caution: this runner's reference is native, unlike H616/H618 full-removal reference. Its legacy `target_relative_error` field measures downstream shift divided by native factorial-interaction norm; it is not approximation error to full removal and is not compared with earlier target-error tables here. The report uses explicit native-reference KL, answer probabilities and competence. The scalar response metrics in the matched audit provide the strongest comparison on identical recipients.

Current inference: the harmful interaction depends specifically on the prefix-absence/lowercase context and its resulting output direction, not just total perturbation magnitude. A context-sensitive answer-emission mechanism is more plausible than a general factual-memory direction, but is not established as a reusable unit. Prefix identity, position changes in the old contrast, and broad continuation statistics remain alternative explanations. The failed answer-label control leaves response-mode isolation incomplete.

Next test whether the harmful complement generalizes across country identity: cross-donor interventions on identical scaffold/positions, with self and norm-matched donor controls, evaluate correct capital against all capitals as well as generic continuations. Split by unique country, accounting for repeated countries in neighboring pairs. If donor complements suppress generic-versus-capital emission without changing capital identity, a shared emission-related direction becomes more plausible. If effects follow donor-specific capitals, factual content is implicated. This directly distinguishes mechanisms without immediately refitting a basis. Include copying and agreement controls where token alignment permits.

Audits: all768 prompts have four-token prefixes, equal body IDs across prefixes, and exactly one initial-case token change; main frozen hashes pass; native Attention5 replay3.15e-7, interaction replay3.21e-5, identity logits2.29e-5. Matched audit checks recipient token IDs identical to H618, full interaction norms equal within1e-6, and reproduces old top1/KL within1e-5. All source captures unchanged. The targeted audit was added after seeing magnitude differences and is reported as adaptive. FineWeb remains26,112 documents. Broad research goal remains active and incomplete.
