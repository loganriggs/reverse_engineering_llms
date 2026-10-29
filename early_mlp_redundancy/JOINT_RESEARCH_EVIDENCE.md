# M54: joint evidence and the actual task-diversity bottleneck

**No validated computational-unit or new-task-discovery claim follows yet.** Common-target response supervision improves 7/12 matched comparisons, and coactivation grouping improves random grouping but loses scalar-source fidelity. M53 adds a mechanistic warning: matching local directions can fail to predict downstream effects. The next useful expansion must supply competent behaviors the current training panel does not already contain.

## Evidence on a common footing

The audit revalidates result hashes against their saved audits, reaggregates all 18 M37 states, verifies all 704 external native targets across architectures and M39, and checks the five persistent task-family training exclusions. The initial audit incorrectly expected352 directions per seed/model/region; corrected to176, consistent with704 across two seeds/two regions. No data or metric changed.

The table uses 32 natural evaluation documents for local write error and intact-output KL, and56 jointly excluded lexical/template development endpoints for intended-answer full-vocabulary top1. Causal errors pool both external generators and all lexical/template quadrants, separately by prefix/task intervention site. These are **different evaluation populations**; columns are not interchangeable measures of a single quantity. Zero response has causal error1. Local error is mean per-input squared reconstruction error normalized by native write energy. All data are familiar development material.

| Representation | Seed | Objective | Local error | Natural KL | Task top1 fraction | Prefix response error | Task response error |
|---|---:|---|---:|---:|---:|---:|---:|
| scalar | 71 | KL | 0.461 | 0.149 | 0.643 | 1.263 | 0.776 |
| scalar | 71 | KL+response | 0.461 | 0.164 | 0.679 | 1.317 | 0.929 |
| block | 71 | KL | 0.492 | 0.177 | 0.714 | 1.274 | 0.935 |
| block | 71 | KL+response | 0.494 | 0.193 | 0.679 | 1.363 | 0.866 |
| shared_fixed | 71 | KL | 0.512 | 0.240 | 0.679 | 1.681 | 1.441 |
| shared_fixed | 71 | KL+response | 0.512 | 0.269 | 0.661 | 1.496 | 1.027 |
| scalar | 72 | KL | 0.458 | 0.132 | 0.786 | 1.153 | 0.631 |
| scalar | 72 | KL+response | 0.464 | 0.154 | 0.714 | 0.964 | 0.521 |
| block | 72 | KL | 0.492 | 0.188 | 0.696 | 1.547 | 0.894 |
| block | 72 | KL+response | 0.492 | 0.219 | 0.643 | 1.196 | 0.827 |
| shared_fixed | 72 | KL | 0.516 | 0.224 | 0.714 | 2.175 | 1.218 |
| shared_fixed | 72 | KL+response | 0.519 | 0.266 | 0.679 | 3.168 | 1.231 |

All initial states and actual support sizes are retained in `validation/joint_research_evidence.json`. Shared fixed unions average roughly28 coordinates versus32 for scalar/block; equal parameter count is not equal active dimension. M39 learned grouping uses a frozen scalar-trained dictionary, so its comparison also does not isolate architecture alone.

| Question | Evidence | What remains unproven |
|---|---|---|
| Does representation reproduce writes? | Direct reconstruction scores | Semantic or causal units |
| Does it preserve predictions? | Intact-output KL, CE, answer accuracy | Correct intervention responses |
| Does it preserve causal responses? | M37/M39 common external directions; mixed supervision gains | Broad unseen-task fidelity |
| Does architectural sharing mean native redundancy? | M48–M53 distinguish gate protection and shared contributions | Literal native parts removable without changing computation |
| Does it discover tasks? | Authored labels and existing competent task panels | Genuine unlabeled task discovery |
| Does it recover manifolds? | Grouping/coactivation and transfer evidence | Stable, selective, reusable native mechanisms across seeds/tasks |

M53 is deliberately excluded from the architecture-score table: each candidate defines different intervention targets and receives oracle recipient norm/support. Its strong conditional errors cannot be compared directly to M37's common external intervention errors.

## Historical breadth audit: five passing families, zero additions

The existing H1039 bank proposed12 training families plus separate whole-family development/evaluation sets. Recomputing its training gates from recorded endpoints yields five eligible families: antonym, plural_noun, past_tense, country_language, number_successor. **All five already occur in the current ten-family training panel.** Reusing that screen as an expansion would add prompts, not new task families.

The seven nonpassing training families are comparative, animal_sound, object_color, addition, subtraction, boolean_negation, month_successor. Failure may reflect prompt format, tokenization, ambiguity, or genuine competence limits; it is not evidence that the underlying operation is absent. Do not lower the gate or promote held-out families to manufacture diversity. Existing held-outs include sequence_copy, spatial_inverse, weekday_successor, role_binding, tense, plus continent_recall/object_function from the broader bank; maintain all applicable restrictions. H1039 whole-family development/evaluation assignments remain fixed. Past_tense is distinct from the historical tense-operation holdout.

## Bounded next experiment

Run a training-only **format/competence screen** for all seven nonpassing families, rather than new featurizer training now. Before any forwards, save two fixed formats per family (explicit question/answer and few-shot completion), four contrast pairs per format, and separate demonstration/query lexical items wherever feasible. This gives56 pairs/112 endpoints before tokenization exclusions. Do not repair or replace individual failed prompts after outcomes. Keep every attempted example and invalid-answer reason.

Check answer-token boundaries, disjoint answer groups, duplicate prompts, and whole-family exclusions before execution. Require at least four valid distinct pairs per family/format, >=75% both-endpoint answer preference, and >=50% intended full-vocabulary top1, using the existing fixed threshold. Report original H1039 results alongside both new formats to distinguish format dependence. This gate is training-only selection; do not read representation performance or move a held-out family. Report unmatched token lengths rather than equating equal length with semantic alignment. A competence pass only authorizes subsequent causal-target design; it does not establish a semantic component.

If no additional family qualifies, report that result and stop this screen. If some qualify, use all qualifying families, freeze a dataset-size/compute-matched old-family-only control before fitting, and construct native causal targets that are identical across scalar/block/shared candidates. Preserve existing whole-family evaluation assignments and label their prior exposure honestly. The added causal diversity and matched control must be specified before launch. No featurizer or membership optimization is authorized merely by a better competence score.

This is a supervised coverage expansion, **not** the requested eventual bootstrapped new-task discovery. Its purpose is to provide more varied causal constraints for that later test.
