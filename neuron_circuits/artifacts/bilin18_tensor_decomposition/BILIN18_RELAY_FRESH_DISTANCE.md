## H849 — Fresh lexical/distance panel confirms compact sentiment transfer and qualifies agreement portability

**Main result.** The frozen rank1 directions transfer to new lexical pairs without refitting. Sentiment preserves87–95% of the full period-state intervention across all tested gap/format cells. Agreement preserves87–96% when adjacent, but inserted wording weakens both the full near/beside relay effect and, in some variants, rank1 fidelity. This supports compact contextual directions while preserving a live adjacency/route limitation for agreement.

**Prospective design.** Lock96 prompt pairs before model evaluation: eight source pairs per task, two formats, three gap variants. Keep preMLP4, original relay token and allH848 bases fixed. Agreement gaps are empty, `right`, `over there` before near/beside; sentiment gaps empty, `indeed`, `overall` before the period. Some words/templates are reused, but full prompts/source pairs are new relative toH840–848. No claim of pretraining or entire-research disjointness. These gaps change wording and potentially syntax as well as token distance; they do not isolate distance alone.

Eight agreement pairs/cell remain aligned. Three sentiment pairs have unequal token lengths, leaving five/cell;18 prompt pairs are excluded by tokenization/alignment before intervention, with no outcome-based replacements. Thus78 pairs are evaluated, but the same13 lexical pairs recur across formats/gaps and must not be counted as78 independent generalization examples.

**Competence.** Every cell passes the frozen gate: >=4 aligned pairs, >=75% pairs binary-correct on both endpoints, >=50% full-vocabulary intended-group top1. Agreement achieves100% on both measures in all cells. Sentiment has100% paired binary correctness and50–100% intended-group full-vocabulary top1. The weakest cell sits exactly at the50% gate; these are semantic-group scores, not exact-token accuracy. All valid pairs receive interventions regardless of individual correctness.

**Results.** Full relay transfer is relative to the native donor-recipient answer-margin gap. Rank1 preservation is relative to that full relay effect. Individual-effect error retains both donor directions. Ratios are neither task accuracy nor unique mediation fractions.

|Task|Format|Gap variant|Full relay transfers|Rank1 preserves full effect|Effect relative RMSE|
|---|---|---:|---:|---:|---:|
|agreement|A|0|34.0%|96.0%|0.115|
|agreement|A|1|14.2%|71.1%|0.370|
|agreement|A|2|2.0%|84.7%|0.222|
|agreement|B|0|33.6%|86.6%|0.176|
|agreement|B|1|16.9%|62.2%|0.435|
|agreement|B|2|5.2%|55.0%|0.514|
|sentiment|A|0|41.5%|89.8%|0.178|
|sentiment|A|1|31.4%|87.0%|0.203|
|sentiment|A|2|34.9%|91.0%|0.162|
|sentiment|B|0|37.3%|89.5%|0.182|
|sentiment|B|1|26.8%|89.9%|0.195|
|sentiment|B|2|29.8%|94.6%|0.145|

The no-gap agreement result provides lexical generalization, including a second preposition format. With `over there`, full near-state transfer falls from34.0% to2.0% in FormatA and from33.6% to5.2% in FormatB, although the model remains fully competent. A small full effect makes preservation ratios less informative; the model may compute agreement through other positions/routes. We did not move the relay to a more favorable token after seeing this failure.

Sentiment's full period-state effect remains substantial at26.8–41.5% of the native gap, and rank1 individual-effect error remains0.145–0.203. This weakens the strict immediate-adjacency explanation for that direction on these specific adverb variants. It does not establish position-invariant polarity computation or a general semantic manifold. New genres, negation, misleading sentiment cues and larger context changes remain untested.

Controls include the rank1 orthogonal complement, other-task direction, and both signs of a random direction at the same edit norm; all rank1/2/4/8 results are retained, with no rank selected on fresh outcomes. H848 showed cross-task directions are nearly orthogonal and yield little cross-task answer transfer. The present evidence favors distinct scalar contextual directions over a single generic relay feature, while a shared routing mechanism remains possible.

**Interpretation and next work.** We now have a concrete causal benchmark for representation learning: an independently frozen low-dimensional direction with fresh lexical transfer, especially robust for sentiment. The next experiments should separate (a) how upstream computation writes this coordinate, (b) which downstream readers use it, and (c) whether natural-data featurizers recover it without task-pair supervision. Use dose-controlled coefficient interventions and matched complementary directions, then train/evaluate at the exact preMLP4 site rather than reusingL1/L2 dictionaries. For agreement, test the first token after the noun on a new distance panel to discriminate a moving local relay from loss of the represented number signal; that would be a new predeclared experiment, not reinterpretation of this frozen-site result.

**Audits.**1,716 full-model forwards; identity edits reproduce logits exactly; edited token identity and position asserted. Manifest, basis and source hashes unchanged; all effects finite; original reserved test untouched; natural-document pool26,112 unchanged. Paired lexical-item bootstrap intervals are saved within each cell and are descriptive for five/eight pairs. H848–849 together made progress by turning a contextual relay into testable frozen scalar directions, while retaining a clear agreement generalization limit. Broad goal remains active.
