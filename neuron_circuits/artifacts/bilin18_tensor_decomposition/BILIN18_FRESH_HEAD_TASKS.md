## H732 — Fresh task controls reveal a selective agreement head and broader recall-head effects

We locked a new panel before evaluating outputs: 80 counterfactual pairs across agreement, capital recall, entity binding, sentiment and sequence copying, each with two format blocks and a same-answer nuisance donor. Agreement nouns, binding names, copy vocabulary and sentiment adjectives differ from H717; capital facts intentionally recur with new templates and pairings. No prompt exactly matches H717. This is fresh evaluation wording, not a claim that its facts or templates were absent from model training.

Four heads were fixed from prior experiments, without selecting on these results: L11H3 (agreement), L13H0 (capital recall), L14H4 (binding reference), L8H3 (copy reference). Every head was tested on every token-valid pair, including failed-competence strata. Each head swap replaces its final-position 128-dimensional pre-output-projection vector with the donor's; other heads/positions remain native and downstream computation runs normally.

Native-competence gates were declared before evaluation: at least four token-valid pairs, at least 75% of pairs correct on both endpoint answer contrasts, and at least 50% pooled intended-answer-set top-1 across the full vocabulary. No individual examples were removed based on model outputs.

| Family | Format A: pairs / both-endpoint correctness / full-vocabulary top-1 | Format B: pairs / both-endpoint correctness / full-vocabulary top-1 | Gate |
|---|---|---|---|
| Agreement | 8 / 100% / 93.8% | 8 / 100% / 75.0% | Both pass |
| Capital recall | 8 / 100% / 12.5% | 8 / 100% / 100% | B only |
| Entity binding | 8 / 100% / 93.8% | 7 / 85.7% / 71.4% | Both pass |
| Sentiment | 8 / 0% / 0% | 8 / 0% / 0% | Both fail |
| Sequence copying | 6 / 100% / 75.0% | 8 / 100% / 100% | Both pass |

Three pairs were excluded solely because intended answers were not valid single-token continuations: copy items 02 and 05, and binding item 10. All remaining 77 pairs and 231 variants were retained. All same-answer nuisance prompts preferred the intended side of the contrast.

The capital Format A result is a useful warning: its answer contrast is correct even though full-vocabulary completion is usually not. Sentiment also fails contrast competence, so this is not solely an omitted capitalization variant in the full-vocabulary answer sets. Examples include predicting “ Excellent” for both positive and negative feedback. We retain the failure rather than editing answer sets or prompts after inspection. These failures constrain this assay, not the model's ability on all possible sentiment or factual prompts.

Mean donor-directed answer-margin change under exact head swaps (A / B):

| Task family | L11H3 agreement head | L13H0 recall head | L14H4 binding reference | L8H3 copy reference |
|---|---:|---:|---:|---:|
| Agreement | 5.636 / 5.118 | 0.001 / -0.005 | -0.001 / -0.001 | -0.001 / ~0 |
| Capital recall | -0.004 / 0.001 | 5.771 / 6.685 | 0.392 / 2.478 | 0.033 / 0.228 |
| Entity binding | -0.002 / ~0 | 0.953 / 0.725 | 3.665 / 3.412 | 0.228 / 0.293 |
| Sequence copying | 0.001 / ~0 | 1.094 / 1.245 | 0.283 / 1.549 | 1.365 / 2.111 |

Capital Format A remains shown but is competence-gate failing. Sentiment effects are small and not interpreted as meaningful task transfer because native competence fails.

The agreement head remains highly selective on this finite panel. Its task-minus-absolute-nuisance margin change is 5.452 / 4.875 on agreement and approximately zero or negative elsewhere. The recall head has positive specificity on competent capital Format B (6.548), binding (0.829 / 0.700), and copying (0.840 / 1.116). This supports a broader causal role than the informal name “capital head,” but does not yet show that the same internal component is reused across these tasks.

Margin transfer is not successful answer replacement. Agreement-head swaps produce the donor's full-vocabulary answer on 62.5% / 56.3% of task interventions; recall-head swaps on capital Format B do so on only 6.25%, and its binding/copy swaps on 0%. Thus a large positive margin effect can remain insufficient to change the final answer.

Audits: 1,463 full-model example forwards completed, including native capture and four interventions per pair per head (two task directions, nuisance and identity). Maximum identity margin error was 5.72e-6; all source/manifest hashes stayed unchanged and intervention values were finite. No new natural corpus documents or original reserved-test evaluations were used. The two format blocks are evaluation strata, not discovery/selection splits; repeated capital facts and finite templates limit independence and generality.

Next: evaluate frozen scalar and learned-group featurizers at L11H3 and L13H0 across this entire panel, retaining failed strata as explicitly diagnostic. Use absolute effect errors where exact-head effects are near zero rather than unstable retention ratios. The informative cross-task test is whether L13H0's components preserve capital, binding and copying effects selectively, beyond matched membership-degree controls. Then intervene on proposed shared components, not merely whole reconstructed heads, before assigning reusable-operation or manifold labels. A separate preregistered format/content diagnostic is needed for sentiment rather than silently repairing this panel.
