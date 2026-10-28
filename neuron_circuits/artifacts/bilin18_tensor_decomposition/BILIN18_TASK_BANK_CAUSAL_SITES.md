## H963 — Expanded task bank has transferable MLP-write effects at multiple sites

Eight discovery-competent task families; all18 MLP layers scanned using discovery examples only. Swap the final-token MLP output bidirectionally between task-changing pairs. Identity and paired +/- isotropic Gaussian edits matched to donor-delta norm at every site; same-answer discovery donor where a different prompt with matching intended token groups and same final input token exists. Full-vocabulary KL to native recipient reported alongside task-margin change. Select the layer with largest mean signed donor effect on discovery; selection saved and hashed before evaluating lexical/structural/joint cohorts. No per-example correctness filtering.

Final-token alignment permits different absolute positions/lengths when the last input token matches. This makes negation usable here without pretending its unequal-length inputs align at every position. It is a whole-output-write intervention, not a discovered feature or natural input-state swap. Earlier layer4 experiments targeted an earlier relay position, so selected layers here do not contradict those results.

| Family | Selected MLP layer (zero-based) | Discovery donor effect | Random-edit RMS | Same-answer RMS | Lexical candidate donor effect |
|---|---|---|---|---|---|
| agreement | 12 | 1.491 | 0.081 | 0.061 | 1.338 |
| capital_recall | 13 | 2.433 | 0.534 | unavailable | 1.822 |
| entity_binding | 10 | 0.175 | 0.055 | unavailable | 0.172 |
| negation | 12 | 0.946 | 0.073 | 0.116 | 0.903 |
| pronoun_gender | 12 | 0.561 | 0.061 | 0.066 | 0.432 |
| sentiment | 12 | 0.909 | 0.076 | 0.158 | 0.765 |
| sequence_copy | 10 | 0.153 | 0.075 | unavailable | 0.168 |
| spatial_inverse | 8 | 1.817 | 0.171 | 0.212 | 1.719 |

Task effects are changes in group-logit margin, not accuracy percentages or fractions of complete task computation. Donor means and control RMS measure different quantities; comparisons describe effect scale, not a formal significance test. Four discovery pairs per task are too few for strong uncertainty claims.

Agreement/sentiment/negation/pronoun gender concentrate at final-token MLP12; factual recall at13; held-out spatial inversion at8. Lexical-candidate effects remain positive at every selected site. Agreement, capital recall and sentiment also retain sizeable effects in both alternate-format cohorts whose native competence gates pass. Entity binding transfers smaller effects (.128–.137 versus .175 discovery). Its random/same-answer controls are closer in scale in some cohorts, so this is weaker evidence for a strong MLP-mediated mechanism.

Whole-task training holdouts remain intact: sequence_copy and spatial_inverse were localized only as evaluation diagnostics and supply no featurizer training examples. Sequence-copy MLP effects are small (.153 discovery versus .075 random RMS), whereas lateral spatial inversion has a large effect (1.817 versus .171). Vertical-spatial alternate prompts fail native competence and have slightly negative selected-site effects; they cannot be presented as positive task transfer. Negation/pronoun/copy alternate-format failures similarly remain diagnostics, even when patch effects are positive.

Same-answer controls are unavailable for some families/contexts because no distinct discovery prompt has the required intended completion and matching final token. This is disclosed rather than substituting an unrelated semantic label. Norm-matched random controls alone do not prove semantic specificity. KL(native donor||patch) is retained as a context-confounded diagnostic, not an exact causal fidelity target.

Audits:6,456/6,456 whole-model forwards including256 native captures; replayed native logits3.53e-5, identity2.82e-5; random-delta norm mismatch2.17e-7 relative. Discovery-only selection independently recomputed, selection hash frozen across validation, aggregate metrics independently checked, inputs unchanged and outputs finite. No featurizer training or reserved-test access; natural-document pool26,112. Last-token writes at all18 layers saved for subsequent discovery-only targets.

Interpretation: expanding supervision yields several causally supported behavior families, and a small set of candidate sites, rather than requiring every task to fit the original layer4 relay. Shared layer12 involvement could reflect reusable computation, separate computations sharing a layer, or generic task readout. Whole-write patches do not distinguish them.

Next: attention-output site screen for entity binding and sequence copy, whose MLP effects are weak relative to the other tasks. Keep sequence copy out of training and compare attention-only write edits with the existing MLP assay, preserving native attention-bus state. Then lock a multi-task causal-target plan around sites with demonstrated effects, and separately test whether layer12 task responses share directions beyond generic output readout. H961 context-bridge predictor evaluation remains pending; no silent abandonment of that frozen experiment.
